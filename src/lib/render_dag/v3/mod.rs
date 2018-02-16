#[macro_use]
mod macros;
mod surface;
#[cfg(test)]
mod test;
mod util;

use ash::{vk, extensions::{Surface, Swapchain}, version::{DeviceV1_0, InstanceV1_0}};
use futures::{future::{join_all, ok, Shared}, prelude::*};
use futures_cpupool::*;
use petgraph::{prelude::*, visit::{self, Walker}};
use std::{self, fmt, ptr, mem::transmute, path::PathBuf, sync::{Arc, Mutex, RwLock}, u64};
use winit;

use super::super::{device, entry, instance, swapchain};
use self::{surface::*, util::*};

enum EdgeFilter {
    All,
    Propagating,
}

fn edge_filter(f: EdgeFilter, edge: &Edge) -> bool {
    match f {
        EdgeFilter::All => true,
        EdgeFilter::Propagating => *edge == Edge::Propagate,
    }
}

decl_node_runtime! {
    RenderNode {
        Instance {
            make make_instance
            static [window: Arc<winit::Window>,
                    events_loop: Arc<winit::EventsLoop>,
                    instance: Arc<instance::Instance>,
                    entry: Arc<entry::Entry>,
                    surface: vk::SurfaceKHR,
                    window_width: u32,
                    window_height: u32]
            dynamic []
        }
        Device {
            make make_device
            static [device: Arc<device::Device>,
                    physical_device: vk::PhysicalDevice,
                    graphics_queue_family: u32,
                    compute_queue_family: u32,
                    // transfer_queue_family: u32, // TODO
                    graphics_queue: Arc<Mutex<vk::Queue>>,
                    compute_queues: Arc<Vec<Mutex<vk::Queue>>>]
            dynamic []
        }
        Swapchain {
            make make_swapchain
            static [handle: Arc<swapchain::Swapchain>,
                    surface_format: vk::SurfaceFormatKHR]
            dynamic []
        }
        Framebuffer {
            make make_framebuffer
            static [images: Arc<Vec<vk::Image>>,
                    image_views: Arc<Vec<vk::ImageView>>,
                    handles: Arc<Vec<vk::Framebuffer>>]
            dynamic [current_present_index: u32]
        }
        PresentFramebuffer {
            make make_present
            static [_dummy: ()] // macros suck
            dynamic []
        }
        CommandPool {
            make make_command_pool
            static [handle: Arc<Mutex<vk::CommandPool>>] // vulkan says we should sync access to this with most (all?) operations
            dynamic []
        }
        PersistentSemaphore {
            make make_persistent_semaphore
            static [handle: vk::Semaphore]
            dynamic []
        }
        AllocateCommandBuffer {
            make make_allocate_commands
            static [handles: Arc<RwLock<Vec<vk::CommandBuffer>>>]
            dynamic [current_frame: vk::CommandBuffer]
            forward vk
            forward Arc
        }
        SubmitCommandBuffer {
            make make_submit_command_buffer
            static [_dummy: ()]
            dynamic []
        }
        Renderpass {
            make make_renderpass
            static [handle: vk::RenderPass]
            dynamic []
        }
        NextSubpass {
            make make_next_subpass
            static [ix: usize]
            dynamic []
        }
        EndRenderpass {
            make make_end_renderpass
            static [_dummy: ()]
            dynamic []
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Edge {
    Propagate,
    Direct,
}

pub type RuntimeGraph = StableDiGraph<RenderNode, Edge>;

pub struct RenderDAG {
    pub graph: RuntimeGraph,
    cpu_pool: CpuPool,
}

impl RenderDAG {
    pub fn new() -> RenderDAG {
        RenderDAG {
            graph: RuntimeGraph::new(),
            cpu_pool: CpuPool::new(1),
        }
    }

    // single queue
    pub fn new_window(&mut self, window_width: u32, window_height: u32) -> NodeIndex {
        let events_loop = winit::EventsLoop::new();
        let window = winit::WindowBuilder::new()
            .with_title("Renderer v3")
            .with_dimensions(window_width, window_height)
            .build(&events_loop)
            .unwrap();
        let (window_width, window_size) = window.get_inner_size().unwrap();

        let entry = entry::Entry::new().unwrap();
        let instance = instance::Instance::new(&entry).unwrap();
        let surface = unsafe { create_surface(entry.vk(), instance.vk(), &window).unwrap() };

        let node = RenderNode::make_instance(
            &self.cpu_pool,
            Arc::new(window),
            Arc::new(events_loop),
            instance,
            entry,
            surface,
            window_width,
            window_height,
        );
        self.graph.add_node(node)
    }

    // first device present
    pub fn new_device(&mut self, instance_ix: NodeIndex) -> Option<(NodeIndex, u32, u32)> {
        let (entry, instance, surface) = match self.graph[instance_ix] {
            RenderNode::Instance {
                ref entry,
                ref instance,
                surface,
                ..
            } => Some((Arc::clone(entry), Arc::clone(instance), surface)),
            _ => None,
        }?;

        let pdevices = instance
            .enumerate_physical_devices()
            .expect("Physical device error");
        let surface_loader = Surface::new(entry.vk(), instance.vk()).expect("Unable to load the Surface extension");

        let pdevice = pdevices[0];
        let graphics_queue_family = {
            instance
                .get_physical_device_queue_family_properties(pdevice)
                .iter()
                .enumerate()
                .filter_map(|(ix, ref info)| {
                    let supports_graphic_and_surface =
                        info.queue_flags.subset(vk::QUEUE_GRAPHICS_BIT) && surface_loader.get_physical_device_surface_support_khr(pdevice, ix as u32, surface);
                    match supports_graphic_and_surface {
                        true => Some(ix as u32),
                        _ => None,
                    }
                })
                .next()
        }?;
        let (compute_queue_family, compute_queue_len) = {
            instance
                .get_physical_device_queue_family_properties(pdevice)
                .iter()
                .enumerate()
                .filter_map(
                    |(ix, ref info)| match info.queue_flags.subset(vk::QUEUE_COMPUTE_BIT) && !info.queue_flags.subset(vk::QUEUE_GRAPHICS_BIT) {
                        true => Some((ix as u32, info.queue_count)),
                        _ => None,
                    },
                )
                .next()
        }?;
        let device = device::Device::new(
            &instance,
            pdevice,
            &[
                (graphics_queue_family, 1),
                (compute_queue_family, compute_queue_len),
            ],
        ).unwrap();
        let graphics_queue = unsafe { device.vk().get_device_queue(graphics_queue_family, 0) };
        let compute_queues = (0..compute_queue_len)
            .map(|ix| {
                let queue = unsafe { device.vk().get_device_queue(compute_queue_family, ix) };
                Mutex::new(queue)
            })
            .collect::<Vec<_>>();
        let graphics_queue = unsafe { device.vk().get_device_queue(graphics_queue_family, 0) };

        let node = RenderNode::make_device(
            &self.cpu_pool,
            device,
            pdevice,
            graphics_queue_family,
            compute_queue_family,
            Arc::new(Mutex::new(graphics_queue)),
            Arc::new(compute_queues),
        );
        let ix = self.graph.add_node(node);
        self.graph.add_edge(instance_ix, ix, Edge::Propagate);
        Some((ix, graphics_queue_family, compute_queue_family))
    }

    pub fn new_swapchain(&mut self, device_ix: NodeIndex) -> Option<NodeIndex> {
        let (entry, instance, surface, window_width, window_height) = search_deps_exactly_one(&self.graph, device_ix, |node| match node {
            &RenderNode::Instance {
                ref entry,
                ref instance,
                surface,
                window_width,
                window_height,
                ..
            } => Some((
                Arc::clone(entry),
                Arc::clone(instance),
                surface,
                window_width,
                window_height,
            )),
            _ => None,
        })?;
        let (device, pdevice) = match self.graph[device_ix] {
            RenderNode::Device {
                ref device,
                physical_device,
                ..
            } => Some((Arc::clone(device), physical_device)),
            _ => None,
        }?;

        let surface_loader = Surface::new(entry.vk(), instance.vk()).expect("Unable to load the Surface extension");
        let present_mode = vk::PresentModeKHR::Fifo;
        let surface_formats = surface_loader
            .get_physical_device_surface_formats_khr(pdevice, surface)
            .unwrap();
        let surface_format = surface_formats
            .iter()
            .map(|sfmt| match sfmt.format {
                vk::Format::Undefined => vk::SurfaceFormatKHR {
                    format: vk::Format::B8g8r8Unorm,
                    color_space: sfmt.color_space,
                },
                _ => sfmt.clone(),
            })
            .nth(0)
            .expect("Unable to find suitable surface format.");
        let surface_capabilities = surface_loader
            .get_physical_device_surface_capabilities_khr(pdevice, surface)
            .unwrap();
        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0 && desired_image_count > surface_capabilities.max_image_count {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D {
                width: window_width,
                height: window_height,
            },
            _ => surface_capabilities.current_extent,
        };
        let pre_transform = if surface_capabilities
            .supported_transforms
            .subset(vk::SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
        {
            vk::SURFACE_TRANSFORM_IDENTITY_BIT_KHR
        } else {
            surface_capabilities.current_transform
        };

        let swapchain_loader = Swapchain::new(instance.vk(), device.vk()).expect("Unable to load swapchain");
        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SwapchainCreateInfoKhr,
            p_next: ptr::null(),
            flags: Default::default(),
            surface: surface,
            min_image_count: desired_image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: surface_resolution.clone(),
            image_usage: vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            image_sharing_mode: vk::SharingMode::Exclusive,
            pre_transform: pre_transform,
            composite_alpha: vk::COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            present_mode: present_mode,
            clipped: 1,
            old_swapchain: vk::SwapchainKHR::null(),
            image_array_layers: 1,
            p_queue_family_indices: ptr::null(),
            queue_family_index_count: 0,
        };
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain_khr(&swapchain_create_info, None)
                .unwrap()
        };

        device.set_object_name(
            vk::DebugReportObjectTypeEXT::SurfaceKhr,
            unsafe { transmute::<_, u64>(swapchain) },
            "Window surface",
        );

        let swapchain = Arc::new(swapchain::Swapchain::new(swapchain_loader, swapchain));
        let node = RenderNode::make_swapchain(&self.cpu_pool, swapchain, surface_format);
        let ix = self.graph.add_node(node);
        self.graph.add_edge(device_ix, ix, Edge::Propagate);
        Some(ix)
    }

    pub fn new_framebuffer(&mut self, swapchain_ix: NodeIndex, renderpass_ix: NodeIndex) -> Option<(NodeIndex, NodeIndex)> {
        let (swapchain, surface_format) = match self.graph[swapchain_ix] {
            RenderNode::Swapchain {
                ref handle,
                ref surface_format,
                ..
            } => Some((Arc::clone(handle), surface_format.clone())),
            _ => None,
        }?;
        let renderpass = match self.graph[renderpass_ix] {
            RenderNode::Renderpass { ref handle, .. } => Some(handle.clone()),
            _ => None,
        }?;
        let device = search_deps_exactly_one(&self.graph, swapchain_ix, |node| match node {
            &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        })?;
        let (window_width, window_height) = search_deps_exactly_one(&self.graph, swapchain_ix, |node| match node {
            &RenderNode::Instance {
                window_width,
                window_height,
                ..
            } => Some((window_width, window_height)),
            _ => None,
        })?;

        let images = swapchain
            .ext
            .get_swapchain_images_khr(swapchain.swapchain)
            .unwrap();
        let image_views = images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo {
                    s_type: vk::StructureType::ImageViewCreateInfo,
                    p_next: ptr::null(),
                    flags: Default::default(),
                    view_type: vk::ImageViewType::Type2d,
                    format: surface_format.format,
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::IMAGE_ASPECT_COLOR_BIT,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image: image,
                };
                unsafe { device.create_image_view(&create_view_info, None).unwrap() }
            })
            .collect::<Vec<_>>();
        let handles = image_views
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments = [present_image_view];
                let frame_buffer_create_info = vk::FramebufferCreateInfo {
                    s_type: vk::StructureType::FramebufferCreateInfo,
                    p_next: ptr::null(),
                    flags: Default::default(),
                    render_pass: renderpass,
                    attachment_count: framebuffer_attachments.len() as u32,
                    p_attachments: framebuffer_attachments.as_ptr(),
                    width: window_width,
                    height: window_height,
                    layers: 1,
                };
                unsafe {
                    device
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap()
                }
            })
            .collect::<Vec<_>>();

        let framebuffer = self.graph.add_node(RenderNode::make_framebuffer(
            &self.cpu_pool,
            Arc::new(images),
            Arc::new(image_views),
            Arc::new(handles),
            0,
        ));
        let present = self.graph
            .add_node(RenderNode::make_present(&self.cpu_pool, ()));
        self.graph
            .add_edge(swapchain_ix, framebuffer, Edge::Propagate);
        self.graph.add_edge(framebuffer, present, Edge::Propagate);
        Some((framebuffer, present))
    }

    pub fn new_command_pool(&mut self, device_ix: NodeIndex, queue_family: u32, flags: vk::CommandPoolCreateFlags) -> Option<NodeIndex> {
        let device = match self.graph[device_ix] {
            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        }?;

        let pool_create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::CommandPoolCreateInfo,
            p_next: ptr::null(),
            flags: flags,
            queue_family_index: queue_family,
        };
        let pool = unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };

        let node = self.graph.add_node(RenderNode::make_command_pool(
            &self.cpu_pool,
            Arc::new(Mutex::new(pool)),
        ));
        device.set_object_name(
            vk::DebugReportObjectTypeEXT::CommandPool,
            unsafe { transmute(pool) },
            &format!("Command Pool {:?}", node),
        );
        self.graph.add_edge(device_ix, node, Edge::Propagate);
        Some(node)
    }

    pub fn new_allocate_command_buffer(&mut self, command_pool_ix: NodeIndex) -> Option<(NodeIndex, NodeIndex)> {
        let device = search_deps_exactly_one(&self.graph, command_pool_ix, |node| match node {
            &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        })?;

        let node = self.graph
            .add_node(RenderNode::make_allocate_commands(&self.cpu_pool, Arc::new(RwLock::new(vec![])), unsafe { vk::CommandBuffer::null() }));
        let submit = self.graph
            .add_node(RenderNode::make_submit_command_buffer(&self.cpu_pool, ()));
        self.graph.add_edge(command_pool_ix, node, Edge::Propagate);
        self.graph.add_edge(node, submit, Edge::Propagate);

        Some((node, submit))
    }

    pub fn new_persistent_semaphore(&mut self, device_ix: NodeIndex) -> Option<NodeIndex> {
        let device = match self.graph[device_ix] {
            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        }?;

        let create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SemaphoreCreateInfo,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
        };
        let semaphore = unsafe { device.create_semaphore(&create_info, None).unwrap() };

        let node = RenderNode::make_persistent_semaphore(&self.cpu_pool, semaphore);
        let ix = self.graph.add_node(node);
        self.graph.add_edge(device_ix, ix, Edge::Propagate);
        unsafe {
            device.set_object_name(
                vk::DebugReportObjectTypeEXT::Semaphore,
                transmute(semaphore),
                &format!("{:?}", ix),
            );
        }

        Some(ix)
    }

    pub fn new_renderpass(
        &mut self,
        device_ix: NodeIndex,
        attachments: &[vk::AttachmentDescription],
        subpass_descs: &[vk::SubpassDescription],
        subpass_dependencies: &[vk::SubpassDependency],
    ) -> Option<(NodeIndex, NodeIndex, Vec<NodeIndex>)> {
        let device = match self.graph[device_ix] {
            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        }?;

        let renderpass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RenderPassCreateInfo,
            flags: Default::default(),
            p_next: ptr::null(),
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: subpass_descs.len() as u32,
            p_subpasses: subpass_descs.as_ptr(),
            dependency_count: subpass_dependencies.len() as u32,
            p_dependencies: subpass_dependencies.as_ptr(),
        };
        let renderpass = unsafe {
            device
                .create_render_pass(&renderpass_create_info, None)
                .unwrap()
        };

        let start_ix = self.graph
            .add_node(RenderNode::make_renderpass(&self.cpu_pool, renderpass));
        let end_ix = self.graph
            .add_node(RenderNode::make_end_renderpass(&self.cpu_pool, ()));
        let mut subpass_ixes = Vec::new();
        let mut previous_ix = start_ix;
        for (ix, subpass) in subpass_descs.iter().skip(1).enumerate() {
            let this_subpass = self.graph
                .add_node(RenderNode::make_next_subpass(&self.cpu_pool, ix));
            subpass_ixes.push(this_subpass);
            self.graph
                .add_edge(previous_ix, this_subpass, Edge::Propagate);
        }
        self.graph.add_edge(previous_ix, end_ix, Edge::Propagate);
        self.graph.add_edge(device_ix, start_ix, Edge::Propagate);

        Some((start_ix, end_ix, subpass_ixes))
    }

    pub fn new_graphics_pipeline(
        &mut self,
        pipeline_layout_ix: NodeIndex,
        input_attributes: &[vk::VertexInputAttributeDescription],
        input_bindings: &[vk::VertexInputBindingDescription],
        shaders: &[PathBuf],
    ) -> Option<NodeIndex> {
        let device = search_deps_exactly_one(&self.graph, pipeline_layout_ix, |node| match node {
            &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        })?;

        /*
        let node = RenderNode::make_persistent_semaphore(&self.cpu_pool, semaphore);
        let ix = self.graph.add_node(node);
        self.graph.add_edge(device_ix, ix, Edge::Propagate);

        Some(ix)
        */
        None
    }

    pub fn render_frame(&self) {
        use petgraph::visit::Walker;
        for ix in visit::Topo::new(&self.graph).iter(&self.graph) {
            match self.graph[ix] {
                RenderNode::Instance { .. } => (),
                RenderNode::Device { .. } => (),
                RenderNode::Swapchain { .. } => (),
                RenderNode::CommandPool { .. } => (),
                RenderNode::PersistentSemaphore { .. } => (),
                RenderNode::Framebuffer { ref dynamic, .. } => {
                    let swapchain = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Swapchain { ref handle, .. } => Some(Arc::clone(handle)),
                        _ => None,
                    }).expect("No swapchain connected to Framebuffer");
                    let signal_semaphore = search_direct_deps_exactly_one(&self.graph, ix, Direction::Outgoing, |node| match node {
                        &RenderNode::PersistentSemaphore { ref handle, .. } => Some(*handle),
                        _ => None,
                    }).expect("No semaphore connected to Framebuffer - what should we signal?");
                    let image_index = unsafe {
                        swapchain
                            .ext
                            .acquire_next_image_khr(
                                swapchain.swapchain,
                                u64::MAX,
                                signal_semaphore,
                                vk::Fence::null(),
                            )
                            .unwrap()
                    };
                    let mut lock = dynamic
                        .write()
                        .expect("failed to lock framebuffer for writing");
                    *lock = spawn_const(
                        &self.cpu_pool,
                        fields::Framebuffer::Dynamic {
                            current_present_index: image_index,
                        },
                    ).shared();
                }
                RenderNode::PresentFramebuffer { ref dynamic, .. } => {
                    let swapchain = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Swapchain { ref handle, .. } => Some(Arc::clone(handle)),
                        _ => None,
                    }).expect("No swapchain connected to Present");
                    let wait_semaphores = search_direct_deps(&self.graph, ix, Direction::Incoming, |node| match node {
                        &RenderNode::PersistentSemaphore { ref handle, .. } => Some(*handle),
                        _ => None,
                    });
                    let present_index = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Framebuffer { ref dynamic, .. } => Some(Arc::clone(dynamic)),
                        _ => None,
                    }).expect("No framebuffer connected to Present - what image should we present?");
                    let present_index_fut = present_index
                        .read()
                        .expect("Failed to read present index")
                        .clone();
                    let graphics_queue = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device {
                            ref graphics_queue, ..
                        } => Some(Arc::clone(graphics_queue)),
                        _ => None,
                    }).expect("No device connected to Present - where should we submit the request?");
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = self.cpu_pool
                        .spawn(
                            order_fut
                                .join(present_index_fut)
                                .map_err(|_| ())
                                .map(move |(_, present_index)| {
                                    let present_info = vk::PresentInfoKHR {
                                        s_type: vk::StructureType::PresentInfoKhr,
                                        p_next: ptr::null(),
                                        wait_semaphore_count: wait_semaphores.len() as u32,
                                        p_wait_semaphores: wait_semaphores.as_ptr(),
                                        swapchain_count: 1,
                                        p_swapchains: &swapchain.swapchain,
                                        p_image_indices: &present_index.current_present_index,
                                        p_results: ptr::null_mut(),
                                    };
                                    let queue = graphics_queue
                                        .lock()
                                        .expect("Failed to acquire lock on graphics queue");
                                    unsafe {
                                        swapchain
                                            .ext
                                            .queue_present_khr(*queue, &present_info)
                                            .unwrap();
                                    };
                                    fields::PresentFramebuffer::Dynamic {}
                                }),
                        )
                        .shared();
                }
                RenderNode::AllocateCommandBuffer { ref handles, ref dynamic, .. } => {
                    let device = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Device not found in deps of AllocateCommandBuffer");
                    let pool = search_direct_deps_exactly_one(&self.graph, ix, Direction::Incoming, |node| match node {
                        &RenderNode::CommandPool { ref handle, .. } => Some(Arc::clone(handle)),
                        _ => None,
                    }).expect("Command pool not found in direct deps of AllocateCommandBuffer");
                    let (fb_count, fb_lock) = search_direct_deps_exactly_one(&self.graph, ix, Direction::Incoming, |node| match node {
                        &RenderNode::Framebuffer { ref images, ref dynamic, .. } => Some((images.len(), Arc::clone(dynamic))),
                        _ => None,
                    }).expect("Framebuffer not found in direct deps of AllocateCommandBuffer");
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let fb_fut = fb_lock.read().expect("Failed to read framebuffer").clone();
                    let handles = Arc::clone(handles);
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    let previous = (*lock).clone();
                    *lock = self.cpu_pool
                        .spawn(previous.join3(fb_fut, order_fut).map_err(|_| ()).map(move |(previous, fb, _)| {
                            let pool_lock = pool.lock()
                                .expect("failed to lock command pool for allocation");
                            let begin_info = vk::CommandBufferBeginInfo {
                                s_type: vk::StructureType::CommandBufferBeginInfo,
                                p_next: ptr::null(),
                                p_inheritance_info: ptr::null(),
                                flags: vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                            };
                            let mut handles = handles.write().expect("Failed to own AllocateCommandBuffer data");
                            let present_ix = fb.current_present_index;
                            if handles.is_empty() {
                                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                                    s_type: vk::StructureType::CommandBufferAllocateInfo,
                                    p_next: ptr::null(),
                                    command_buffer_count: fb_count as u32,
                                    command_pool: *pool_lock,
                                    level: vk::CommandBufferLevel::Primary,
                                };
                                let command_buffers = unsafe {
                                    device
                                        .allocate_command_buffers(&command_buffer_allocate_info)
                                        .unwrap()
                                };
                                let current_frame = command_buffers[present_ix as usize];
                                *handles = command_buffers;
                                unsafe {
                                    device.begin_command_buffer(current_frame, &begin_info);
                                }
                                fields::AllocateCommandBuffer::Dynamic {
                                    current_frame
                                }
                            } else {
                                let current_frame = handles[present_ix as usize];
                                unsafe {
                                    device.reset_command_buffer(current_frame, vk::CommandBufferResetFlags::empty());
                                    device.begin_command_buffer(current_frame, &begin_info);
                                }
                                fields::AllocateCommandBuffer::Dynamic { current_frame }
                            }
                        }))
                        .shared();
                }
                RenderNode::SubmitCommandBuffer { ref dynamic, .. } => {
                    let (device, graphics_queue) = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device {
                            ref device,
                            ref graphics_queue,
                            ..
                        } => Some((Arc::clone(device), Arc::clone(graphics_queue))),
                        _ => None,
                    }).expect("Device not found in deps of SubmitCommandBuffer");
                    let allocated_lock = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::AllocateCommandBuffer { ref dynamic, .. } => Some(Arc::clone(dynamic)),
                        _ => None,
                    }).expect("Device not found in deps of SubmitCommandBuffer");
                    let wait_semaphores = search_direct_deps(&self.graph, ix, Direction::Incoming, |node| match node {
                        &RenderNode::PersistentSemaphore { handle, .. } => Some(handle),
                        _ => None,
                    });
                    let signal_semaphores = search_direct_deps(&self.graph, ix, Direction::Outgoing, |node| match node {
                        &RenderNode::PersistentSemaphore { handle, .. } => Some(handle),
                        _ => None,
                    });
                    let fb_lock = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Framebuffer { ref dynamic, .. } => Some(Arc::clone(dynamic)),
                        _ => None,
                    }).expect("Framebuffer not found in direct deps of SubmitCommandBuffer");
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let fb_fut = fb_lock.read().expect("Failed to read framebuffer").clone();
                    let allocated = allocated_lock
                        .write()
                        .expect("failed to read command buffer")
                        .clone();
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = self.cpu_pool
                        .spawn(
                            order_fut
                                .join3(allocated, fb_fut)
                                .map_err(|_| ())
                                .map(move |(_, allocated, fb)| unsafe {
                                    let cb = allocated.current_frame;
                                    let fb_ix = (*fb).current_present_index;
                                    device.end_command_buffer(cb).unwrap();
                                    let dst_stage_masks = vec![vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT; wait_semaphores.len()];
                                    let submits = [
                                        vk::SubmitInfo {
                                            s_type: vk::StructureType::SubmitInfo,
                                            p_next: ptr::null(),
                                            wait_semaphore_count: wait_semaphores.len() as u32,
                                            p_wait_semaphores: wait_semaphores.as_ptr(),
                                            p_wait_dst_stage_mask: dst_stage_masks.as_ptr(),
                                            command_buffer_count: 1,
                                            p_command_buffers: &cb,
                                            signal_semaphore_count: signal_semaphores.len() as u32,
                                            p_signal_semaphores: signal_semaphores.as_ptr(),
                                        },
                                    ];
                                    let queue_lock = graphics_queue.lock().expect("can't lock the queue");

                                    let submit_fence = {
                                        let create_info = vk::FenceCreateInfo {
                                            s_type: vk::StructureType::FenceCreateInfo,
                                            p_next: ptr::null(),
                                            flags: vk::FenceCreateFlags::empty(),
                                        };
                                        device
                                        .create_fence(&create_info, None)
                                        .expect("Create fence failed.")
                                    };

                                    device.queue_submit(*queue_lock, &submits, submit_fence);

                                    device
                                        .wait_for_fences(&[submit_fence], true, u64::MAX)
                                        .expect("Wait for fence failed.");
                                    device.destroy_fence(submit_fence, None);

                                    fields::SubmitCommandBuffer::Dynamic {}
                                }),
                        )
                        .shared();
                }
                RenderNode::Renderpass {
                    handle,
                    ref dynamic,
                } => {
                    let device = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Device not found in deps of Renderpass");
                    let (fb_handles, fb_lock) = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Framebuffer {
                            ref handles,
                            ref dynamic,
                            ..
                        } => Some((Arc::clone(handles), Arc::clone(dynamic))),
                        _ => None,
                    }).expect("Framebuffer not found in deps of Renderpass");
                    let command_buffer_locked = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::AllocateCommandBuffer { ref dynamic, .. } => Some(Arc::clone(dynamic)),
                        _ => None,
                    }).expect("Command Buffer not found in deps of Renderpass");
                    let (window_width, window_height) = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Instance {
                            window_width,
                            window_height,
                            ..
                        } => Some((window_width, window_height)),
                        _ => None,
                    }).expect("Instance not found in deps of Renderpass");
                    let command_buffer_fut = command_buffer_locked
                        .read()
                        .expect("Could not read command buffer")
                        .clone();
                    let fb_fut = fb_lock.read().expect("Could not read framebuffer").clone();
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = self.cpu_pool
                        .spawn(
                            command_buffer_fut
                                .join(fb_fut)
                                .map_err(|_| ())
                                .map(move |(cb_dyn, fb)| {
                                    let fb_ix = (*fb).current_present_index;
                                    let cb = cb_dyn.current_frame;
                                    let clear_values = vec![
                                        vk::ClearValue::new_color(vk::ClearColorValue::new_float32([0.0; 4])),
                                    ];
                                    let begin_info = vk::RenderPassBeginInfo {
                                        s_type: vk::StructureType::RenderPassBeginInfo,
                                        p_next: ptr::null(),
                                        render_pass: handle,
                                        framebuffer: fb_handles[fb_ix as usize],
                                        render_area: vk::Rect2D {
                                            offset: vk::Offset2D { x: 0, y: 0 },
                                            extent: vk::Extent2D {
                                                width: window_width,
                                                height: window_height,
                                            },
                                        },
                                        clear_value_count: clear_values.len() as u32,
                                        p_clear_values: clear_values.as_ptr(),
                                    };
                                    unsafe {
                                        device.cmd_begin_render_pass(cb, &begin_info, vk::SubpassContents::Inline);
                                    }
                                    fields::Renderpass::Dynamic {}
                                }),
                        )
                        .shared();
                }
                RenderNode::EndRenderpass { ref dynamic, .. } => {
                    let device = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Device not found in deps of EndRenderpass");
                    let command_buffer_locked = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::AllocateCommandBuffer { ref dynamic, .. } => Some(Arc::clone(dynamic)),
                        _ => None,
                    }).expect("Command Buffer not found in deps of EndRenderpass");
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let command_buffer_fut = command_buffer_locked
                        .read()
                        .expect("Could not read command buffer")
                        .clone();
                    let fb_lock = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Framebuffer { ref dynamic, .. } => Some(Arc::clone(dynamic)),
                        _ => None,
                    }).expect("Framebuffer not found in direct deps of EndRenderPass");
                    let fb_fut = fb_lock.read().expect("Failed to read framebuffer").clone();
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = self.cpu_pool
                        .spawn(
                            order_fut
                                .join3(command_buffer_fut, fb_fut)
                                .map_err(|_| ())
                                .map(move |(_, cb_dyn, fb)| {
                                    let fb_ix = (*fb).current_present_index;
                                    let cb = cb_dyn.current_frame;
                                    unsafe {
                                        device.cmd_end_render_pass(cb);
                                    }
                                    fields::EndRenderpass::Dynamic {}
                                }),
                        )
                        .shared();
                }
                RenderNode::NextSubpass {
                    ix: subpass_ix,
                    ref dynamic,
                } => {
                    let device = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Device not found in deps of NextSubpass");
                    let command_buffer_locked = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::AllocateCommandBuffer { ref dynamic, .. } => Some(Arc::clone(dynamic)),
                        _ => None,
                    }).expect("Command Buffer not found in deps of NextSubpass");
                    let command_buffer_fut = command_buffer_locked
                        .read()
                        .expect("Could not read command buffer")
                        .clone();
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let fb_lock = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Framebuffer { ref dynamic, .. } => Some(Arc::clone(dynamic)),
                        _ => None,
                    }).expect("Framebuffer not found in direct deps of EndRenderPass");
                    let fb_fut = fb_lock.read().expect("Failed to read framebuffer").clone();
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = self.cpu_pool
                        .spawn(
                            order_fut
                                .join3(command_buffer_fut, fb_fut)
                                .map_err(|_| ())
                                .map(move |(_, cb_dyn, fb)| {
                                    let fb_ix = (*fb).current_present_index;
                                    let cb = cb_dyn.current_frame;
                                    unsafe {
                                        device.cmd_next_subpass(cb, vk::SubpassContents::Inline);
                                    }
                                    fields::NextSubpass::Dynamic {}
                                }),
                        )
                        .shared();
                }
            }
        }
    }
}

impl Drop for RenderDAG {
    fn drop(&mut self) {
        use std::boxed::FnBox;
        let mut finalizer_graph: StableDiGraph<Box<FnBox()>, Edge> = self.graph.filter_map(
            |ix, node| match node {
                &RenderNode::Instance { ref instance, .. } => {
                    let i = Arc::clone(instance);
                    Some(Box::new(move || {
                        drop(i);
                    }) as Box<FnBox()>)
                }
                &RenderNode::Device { ref device, .. } => {
                    let d = Arc::clone(device);
                    Some(Box::new(move || {
                        drop(d);
                    }) as Box<FnBox()>)
                }
                &RenderNode::Swapchain { ref handle, .. } => {
                    let parent_device = search_deps(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    });
                    let swapchain = Arc::clone(handle);
                    Some(Box::new(move || {
                        drop(swapchain);
                        drop(parent_device);
                    }))
                }
                &RenderNode::Framebuffer {
                    ref images,
                    ref image_views,
                    ..
                } => {
                    let parent_device = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Expected one parent Instance for Framebuffer");
                    let image_views = image_views.clone();

                    Some(Box::new(move || unsafe {
                        for view in image_views.iter() {
                            parent_device.destroy_image_view(*view, None);
                        }
                    }) as Box<FnBox()>)
                }
                &RenderNode::PresentFramebuffer { .. } => None,
                &RenderNode::CommandPool { ref handle, .. } => {
                    let parent_device = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Expected one parent Instance for CommandPool");
                    let command_pool = Arc::clone(handle);
                    Some(Box::new(move || unsafe {
                        let lock = command_pool.lock().expect("Cannot lock command pool");
                        parent_device.destroy_command_pool(*lock, None);
                    }) as Box<FnBox()>)
                }
                &RenderNode::PersistentSemaphore { ref handle, .. } => {
                    let parent_device = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Expected one parent Device for PersistentSemaphore");
                    let handle = handle.clone();

                    Some(Box::new(move || unsafe {
                        parent_device.destroy_semaphore(handle, None);
                    }))
                }
                &RenderNode::AllocateCommandBuffer { ref handles, ref dynamic, .. } => {
                    let parent_device = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Expected one parent Device for AllocateCommandBuffer");
                    let pool = search_direct_deps_exactly_one(&self.graph, ix, Direction::Incoming, |node| match node {
                        &RenderNode::CommandPool { ref handle, .. } => Some(Arc::clone(handle)),
                        _ => None,
                    }).expect("Command pool not found in direct deps of AllocateCommandBuffer");
                    let dynamic = Arc::clone(dynamic);
                    let handles = Arc::clone(handles);

                    Some(Box::new(move || {
                        let lock = dynamic
                            .write()
                            .expect("Failed locking AllocateCommandBuffer for dtor");
                        let pool_lock = pool.lock()
                            .expect("Failed locking CommandPool for AllocateCommandBuffer dtor");
                        let cb_fut = (*lock).clone();
                        let mut cb_lock = handles.write().expect("Failed locking AllocateCB for dtor");
                        unsafe {
                            parent_device.free_command_buffers(*pool_lock, &cb_lock);
                        }
                        (*cb_lock).clear();
                    }))
                }
                &RenderNode::SubmitCommandBuffer { .. } => None,
                &RenderNode::Renderpass { ref handle, .. } => {
                    let parent_device = search_deps_exactly_one(&self.graph, ix, |node| match node {
                        &RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Expected one parent Device for AllocateCommandBuffer");
                    let handle = handle.clone();
                    Some(Box::new(move || unsafe {
                        parent_device.destroy_render_pass(handle, None);
                    }))
                }
                &RenderNode::NextSubpass { .. } => None,
                &RenderNode::EndRenderpass { .. } => None,
            },
            |_, edge| Some(edge.clone()),
        );
        for ix in self.graph.node_indices() {
            match self.graph[ix] {
                RenderNode::Device { ref device, .. } => unsafe {
                    device.device_wait_idle().unwrap();
                },
                _ => (),
            }
        }
        self.graph.clear();
        let order = {
            let reversed = visit::Reversed(&finalizer_graph);
            use petgraph::visit::Walker;
            visit::Topo::new(reversed)
                .iter(&reversed)
                .collect::<Vec<_>>()
        };
        for ix in order {
            let f = finalizer_graph.remove_node(ix).unwrap();
            f.call_box(());
        }
    }
}
