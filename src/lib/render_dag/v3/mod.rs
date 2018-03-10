pub mod alloc;
mod dot_formatter;
mod expanded;
// #[macro_use]
// mod macros;
mod surface;
pub mod util;

use ash::{vk, extensions::{Surface, Swapchain}, version::{DeviceV1_0, InstanceV1_0}};
use futures::{executor::ThreadPool, future::lazy, prelude::*};
use petgraph::{visit, prelude::*};
use std::{self, ptr, ffi::CString, fs::File, io::Read, mem::transmute, path::PathBuf,
          sync::{Arc, Mutex, RwLock}, u64};
use specs;
use winit;

use super::super::{device, entry, instance, swapchain};
use self::{expanded::*, surface::*, util::*};

pub use self::dot_formatter::dot;

pub use self::expanded::RenderNode;

#[derive(Clone, PartialEq, Debug)]
pub enum Edge {
    Propagate,
    Direct,
}

pub type RuntimeGraph = StableDiGraph<RenderNode, Edge>;

pub struct RenderDAG {
    pub graph: RuntimeGraph,
    cpu_pool: ThreadPool,
    frame_number: u32,
}

impl Default for RenderDAG {
    fn default() -> RenderDAG {
        RenderDAG::new()
    }
}

impl RenderDAG {
    pub fn new() -> RenderDAG {
        RenderDAG {
            graph: RuntimeGraph::new(),
            cpu_pool: ThreadPool::new(),
            frame_number: 0,
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
        let (window_width, window_height) = window.get_inner_size().unwrap();

        let entry = entry::Entry::new().unwrap();
        let instance = instance::Instance::new(&entry).unwrap();
        let surface = unsafe { create_surface(entry.vk(), instance.vk(), &window).unwrap() };

        let node = RenderNode::Instance {
            dynamic: dyn(&self.cpu_pool, ()),
            window: Arc::new(window),
            events_loop: Arc::new(events_loop),
            instance: instance,
            entry: entry,
            surface: surface,
            window_width: window_width,
            window_height: window_height,
        };
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
        let surface_loader =
            Surface::new(entry.vk(), instance.vk()).expect("Unable to load the Surface extension");

        let pdevice = pdevices[0];
        let graphics_queue_family = {
            instance
                .get_physical_device_queue_family_properties(pdevice)
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| {
                    let supports_graphic_and_surface = info.queue_flags
                        .subset(vk::QUEUE_GRAPHICS_BIT)
                        && surface_loader.get_physical_device_surface_support_khr(
                            pdevice,
                            ix as u32,
                            surface,
                        );
                    if supports_graphic_and_surface {
                        Some(ix as u32)
                    } else {
                        None
                    }
                })
                .next()
        }?;
        let (compute_queue_family, compute_queue_len) = {
            instance
                .get_physical_device_queue_family_properties(pdevice)
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| {
                    if info.queue_flags.subset(vk::QUEUE_COMPUTE_BIT)
                        && !info.queue_flags.subset(vk::QUEUE_GRAPHICS_BIT)
                    {
                        Some((ix as u32, info.queue_count))
                    } else {
                        None
                    }
                })
                .next()
        }.unwrap_or((graphics_queue_family, 1));
        let device = device::Device::new(
            &instance,
            pdevice,
            &[
                (graphics_queue_family, 1),
                (compute_queue_family, compute_queue_len),
            ],
        ).unwrap();
        let allocator = alloc::create(device.vk().handle(), pdevice).unwrap();
        let graphics_queue = unsafe { device.vk().get_device_queue(graphics_queue_family, 0) };
        let compute_queues = (0..compute_queue_len)
            .map(|ix| {
                let queue = unsafe { device.vk().get_device_queue(compute_queue_family, ix) };
                Mutex::new(queue)
            })
            .collect::<Vec<_>>();

        let node = RenderNode::Device {
            dynamic: dyn(&self.cpu_pool, ()),
            device,
            physical_device: pdevice,
            allocator,
            graphics_queue_family,
            compute_queue_family,
            graphics_queue: Arc::new(Mutex::new(graphics_queue)),
            compute_queues: Arc::new(compute_queues),
        };
        let ix = self.graph.add_node(node);
        self.graph.add_edge(instance_ix, ix, Edge::Propagate);
        Some((ix, graphics_queue_family, compute_queue_family))
    }

    pub fn new_swapchain(&mut self, device_ix: NodeIndex) -> Option<NodeIndex> {
        let (entry, instance, surface, window_width, window_height) =
            search_deps_exactly_one(&self.graph, device_ix, |node| match *node {
                RenderNode::Instance {
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

        let surface_loader =
            Surface::new(entry.vk(), instance.vk()).expect("Unable to load the Surface extension");
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
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
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

        let swapchain_loader =
            Swapchain::new(instance.vk(), device.vk()).expect("Unable to load swapchain");
        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SwapchainCreateInfoKhr,
            p_next: ptr::null(),
            flags: Default::default(),
            surface: surface,
            min_image_count: desired_image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: surface_resolution,
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
        let node = RenderNode::Swapchain {
            dynamic: dyn(&self.cpu_pool, ()),
            handle: swapchain,
            surface_format,
        };
        let ix = self.graph.add_node(node);
        self.graph.add_edge(device_ix, ix, Edge::Propagate);
        Some(ix)
    }

    pub fn new_framebuffer(
        &mut self,
        swapchain_ix: NodeIndex,
        renderpass_ix: NodeIndex,
    ) -> Option<(NodeIndex, NodeIndex)> {
        let (swapchain, surface_format) = match self.graph[swapchain_ix] {
            RenderNode::Swapchain {
                ref handle,
                ref surface_format,
                ..
            } => Some((Arc::clone(handle), surface_format.clone())),
            _ => None,
        }?;
        let renderpass = match self.graph[renderpass_ix] {
            RenderNode::Renderpass { ref handle, .. } => Some(*handle),
            _ => None,
        }?;
        let device = search_deps_exactly_one(&self.graph, swapchain_ix, |node| match *node {
            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        })?;
        let (window_width, window_height) =
            search_deps_exactly_one(&self.graph, swapchain_ix, |node| match *node {
                RenderNode::Instance {
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

        let framebuffer = self.graph.add_node(RenderNode::Framebuffer {
            dynamic: dyn(&self.cpu_pool, fields::Framebuffer::Dynamic::new(0)),
            images: Arc::new(images),
            image_views: Arc::new(image_views),
            handles: Arc::new(handles),
        });
        let present = self.graph.add_node(RenderNode::PresentFramebuffer {
            dynamic: dyn(&self.cpu_pool, ()),
        });
        self.graph
            .add_edge(swapchain_ix, framebuffer, Edge::Propagate);
        self.graph.add_edge(framebuffer, present, Edge::Propagate);
        Some((framebuffer, present))
    }

    pub fn new_command_pool(
        &mut self,
        device_ix: NodeIndex,
        queue_family: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> Option<NodeIndex> {
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

        let node = self.graph.add_node(RenderNode::CommandPool {
            dynamic: dyn(&self.cpu_pool, ()),
            handle: Arc::new(Mutex::new(pool)),
        });
        device.set_object_name(
            vk::DebugReportObjectTypeEXT::CommandPool,
            unsafe { transmute(pool) },
            &format!("Command Pool {:?}", node),
        );
        self.graph.add_edge(device_ix, node, Edge::Propagate);
        Some(node)
    }

    pub fn new_allocate_command_buffer(
        &mut self,
        command_pool_ix: NodeIndex,
    ) -> Option<(NodeIndex, NodeIndex)> {
        let node = self.graph.add_node(RenderNode::make_allocate_commands(
            &self.cpu_pool,
            Arc::new(RwLock::new(vec![])),
            unsafe { vk::CommandBuffer::null() },
        ));
        let submit = self.graph.add_node(RenderNode::SubmitCommandBuffer {
            dynamic: dyn(&self.cpu_pool, ()),
        });
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

        let node = RenderNode::PersistentSemaphore {
            dynamic: dyn(&self.cpu_pool, ()),
            handle: semaphore,
        };
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

        let start_ix = self.graph.add_node(RenderNode::Renderpass {
            dynamic: dyn(&self.cpu_pool, ()),
            handle: renderpass,
        });
        let end_ix = self.graph.add_node(RenderNode::EndRenderpass {
            dynamic: dyn(&self.cpu_pool, ()),
        });
        let previous_ix = start_ix;
        let subpass_ixes = (1..subpass_descs.len())
            .map(|ix| {
                let this_subpass = self.graph.add_node(RenderNode::NextSubpass {
                    dynamic: dyn(&self.cpu_pool, ()),
                    ix,
                });
                self.graph
                    .add_edge(previous_ix, this_subpass, Edge::Propagate);
                this_subpass
            })
            .collect::<Vec<_>>();
        self.graph.add_edge(previous_ix, end_ix, Edge::Propagate);
        self.graph.add_edge(device_ix, start_ix, Edge::Propagate);

        Some((start_ix, end_ix, subpass_ixes))
    }

    pub fn new_descriptor_set_layout(
        &mut self,
        device_ix: NodeIndex,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Option<NodeIndex> {
        let device = match self.graph[device_ix] {
            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        }?;

        let create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DescriptorSetLayoutCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let handle = unsafe {
            device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        };
        let node = self.graph.add_node(RenderNode::DescriptorSetLayout {
            dynamic: dyn(&self.cpu_pool, ()),
            handle,
        });
        self.graph.add_edge(device_ix, node, Edge::Propagate);

        Some(node)
    }

    pub fn new_descriptor_pool(
        &mut self,
        device_ix: NodeIndex,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Option<NodeIndex> {
        let device = match self.graph[device_ix] {
            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        }?;

        let create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DescriptorPoolCreateInfo,
            p_next: ptr::null(),
            flags: vk::DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            max_sets,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
        };

        let handle = unsafe { device.create_descriptor_pool(&create_info, None).unwrap() };

        let node = self.graph.add_node(RenderNode::DescriptorPool {
            dynamic: dyn(&self.cpu_pool, ()),
            handle,
        });
        self.graph.add_edge(device_ix, node, Edge::Propagate);
        Some(node)
    }

    pub fn new_descriptor_set(
        &mut self,
        device_ix: NodeIndex,
        descriptor_pool_ix: NodeIndex,
        layout_ix: NodeIndex,
    ) -> Option<NodeIndex> {
        let device = match self.graph[device_ix] {
            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        }?;
        let pool = match self.graph[descriptor_pool_ix] {
            RenderNode::DescriptorPool { handle, .. } => Some(handle),
            _ => None,
        }?;
        let layout = match self.graph[layout_ix] {
            RenderNode::DescriptorSetLayout { handle, .. } => Some(handle),
            _ => None,
        }?;
        let layouts = &[layout];
        let desc_alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DescriptorSetAllocateInfo,
            p_next: ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
        };
        let mut new_descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&desc_alloc_info).unwrap() };
        let handle = new_descriptor_sets.remove(0);
        let node = self.graph.add_node(RenderNode::DescriptorSet {
            dynamic: dyn(&self.cpu_pool, ()),
            handle,
        });
        self.graph.add_edge(device_ix, node, Edge::Propagate);
        self.graph
            .add_edge(descriptor_pool_ix, node, Edge::Propagate);
        self.graph.add_edge(layout_ix, node, Edge::Propagate);
        Some(node)
    }

    pub fn new_pipeline_layout(
        &mut self,
        device_ix: NodeIndex,
        descriptor_set_ixes: &[NodeIndex],
        push_constant_ranges: Vec<vk::PushConstantRange>,
    ) -> Option<NodeIndex> {
        let device = match self.graph[device_ix] {
            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        }?;
        let descriptor_sets = descriptor_set_ixes
            .iter()
            .map(|ix| {
                if let RenderNode::DescriptorSetLayout { handle, .. } = self.graph[*ix] {
                    handle
                } else {
                    panic!("descriptor set ix invalid")
                }
            })
            .collect::<Vec<_>>();

        let create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PipelineLayoutCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            set_layout_count: descriptor_sets.len() as u32,
            p_set_layouts: descriptor_sets.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as u32,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
        };

        let pipeline_layout = unsafe { device.create_pipeline_layout(&create_info, None).unwrap() };
        let node = self.graph.add_node(RenderNode::make_pipeline_layout(
            &self.cpu_pool,
            Arc::new(push_constant_ranges),
            pipeline_layout,
        ));
        self.graph.add_edge(device_ix, node, Edge::Propagate);

        Some(node)
    }

    pub fn new_graphics_pipeline(
        &mut self,
        pipeline_layout_ix: NodeIndex,
        renderpass_ix: NodeIndex,
        input_attributes: &[vk::VertexInputAttributeDescription],
        input_bindings: &[vk::VertexInputBindingDescription],
        shaders: &[(vk::ShaderStageFlags, PathBuf)],
    ) -> Option<NodeIndex> {
        let device =
            search_deps_exactly_one(&self.graph, pipeline_layout_ix, |node| match *node {
                RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                _ => None,
            })?;
        let (window_width, window_height) =
            search_deps_exactly_one(&self.graph, pipeline_layout_ix, |node| match *node {
                RenderNode::Instance {
                    window_width,
                    window_height,
                    ..
                } => Some((window_width, window_height)),
                _ => None,
            })?;
        let pipeline_layout = match self.graph[pipeline_layout_ix] {
            RenderNode::PipelineLayout { handle, .. } => Some(handle),
            _ => None,
        }?;
        let renderpass = match self.graph[renderpass_ix] {
            RenderNode::Renderpass { handle, .. } => Some(handle),
            _ => None,
        }?;
        let shader_modules = shaders
            .iter()
            .map(|&(stage, ref path)| {
                let file = File::open(path).expect("Could not find shader.");
                let bytes: Vec<u8> = file.bytes().filter_map(|byte| byte.ok()).collect();
                let shader_info = vk::ShaderModuleCreateInfo {
                    s_type: vk::StructureType::ShaderModuleCreateInfo,
                    p_next: ptr::null(),
                    flags: Default::default(),
                    code_size: bytes.len(),
                    p_code: bytes.as_ptr() as *const u32,
                };
                let shader_module = unsafe {
                    device
                        .create_shader_module(&shader_info, None)
                        .expect("Vertex shader module error")
                };
                (shader_module, stage)
            })
            .collect::<Vec<_>>();
        let shader_entry_name = CString::new("main").unwrap();
        let shader_stage_create_infos = shader_modules
            .iter()
            .map(|&(module, stage)| vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: ptr::null(),
                flags: Default::default(),
                module: module,
                p_name: shader_entry_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: stage,
            })
            .collect::<Vec<_>>();
        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PipelineVertexInputStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            vertex_attribute_description_count: input_attributes.len() as u32,
            p_vertex_attribute_descriptions: input_attributes.as_ptr(),
            vertex_binding_description_count: input_bindings.len() as u32,
            p_vertex_binding_descriptions: input_bindings.as_ptr(),
        };
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PipelineInputAssemblyStateCreateInfo,
            flags: Default::default(),
            p_next: ptr::null(),
            primitive_restart_enable: 0,
            topology: vk::PrimitiveTopology::TriangleList,
        };
        let viewports = [
            vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: window_width as f32,
                height: window_height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        ];
        let scissors = [
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: window_width,
                    height: window_height,
                },
            },
        ];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PipelineViewportStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
        };
        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PipelineRasterizationStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            cull_mode: vk::CULL_MODE_BACK_BIT,
            depth_bias_clamp: 0.0,
            depth_bias_constant_factor: 0.0,
            depth_bias_enable: 0,
            depth_bias_slope_factor: 0.0,
            depth_clamp_enable: 0,
            front_face: vk::FrontFace::Clockwise,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::Fill,
            rasterizer_discard_enable: 0,
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PipelineMultisampleStateCreateInfo,
            flags: Default::default(),
            p_next: ptr::null(),
            rasterization_samples: vk::SAMPLE_COUNT_1_BIT,
            sample_shading_enable: 0,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_one_enable: 0,
            alpha_to_coverage_enable: 0,
        };
        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::Keep,
            pass_op: vk::StencilOp::Keep,
            depth_fail_op: vk::StencilOp::Keep,
            compare_op: vk::CompareOp::Always,
            compare_mask: 0,
            write_mask: 0,
            reference: 0,
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PipelineDepthStencilStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::LessOrEqual,
            depth_bounds_test_enable: 0,
            stencil_test_enable: 0,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            min_depth_bounds: 0.0,
        };
        let color_blend_attachment_states = [
            vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SrcColor,
                dst_color_blend_factor: vk::BlendFactor::OneMinusDstColor,
                color_blend_op: vk::BlendOp::Add,
                src_alpha_blend_factor: vk::BlendFactor::Zero,
                dst_alpha_blend_factor: vk::BlendFactor::Zero,
                alpha_blend_op: vk::BlendOp::Add,
                color_write_mask: vk::ColorComponentFlags::all(),
            },
        ];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PipelineColorBlendStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            logic_op_enable: 0,
            logic_op: vk::LogicOp::Clear,
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        /*
        let dynamic_state = [vk::DynamicState::Viewport, vk::DynamicState::Scissor];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PipelineDynamicStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            dynamic_state_count: dynamic_state.len() as u32,
            p_dynamic_states: dynamic_state.as_ptr(),
        };
        */
        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GraphicsPipelineCreateInfo,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stage_create_infos.len() as u32,
            p_stages: shader_stage_create_infos.as_ptr(),
            p_vertex_input_state: &vertex_input_state_info,
            p_input_assembly_state: &vertex_input_assembly_state_info,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state_info,
            p_rasterization_state: &rasterization_info,
            p_multisample_state: &multisample_state_info,
            p_depth_stencil_state: &depth_state_info,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: ptr::null(), // &dynamic_state_info,
            layout: pipeline_layout,
            render_pass: renderpass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
        };
        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphic_pipeline_info],
                    None,
                )
                .expect("Unable to create graphics pipeline")
        };
        for (shader_module, _stage) in shader_modules {
            unsafe {
                device.destroy_shader_module(shader_module, None);
            }
        }

        let node = RenderNode::make_graphics_pipeline(&self.cpu_pool, graphics_pipelines[0]);
        let ix = self.graph.add_node(node);
        self.graph.add_edge(pipeline_layout_ix, ix, Edge::Direct);
        self.graph.add_edge(renderpass_ix, ix, Edge::Propagate);

        Some(ix)
    }

    pub fn new_buffer(
        &mut self,
        device_ix: NodeIndex,
        buffer_usage: vk::BufferUsageFlags,
        allocation_flags: alloc::VmaAllocationCreateFlags,
        allocation_usage: alloc::VmaMemoryUsage,
        size: vk::DeviceSize,
    ) -> Option<NodeIndex> {
        let (allocator, graphics_queue_family, compute_queue_family) =
            search_deps_exactly_one(&self.graph, device_ix, |node| match *node {
                RenderNode::Device {
                    allocator,
                    graphics_queue_family,
                    compute_queue_family,
                    ..
                } => Some((allocator, graphics_queue_family, compute_queue_family)),
                _ => None,
            })?;

        let queue_families = [graphics_queue_family, compute_queue_family];
        let buffer_create_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BufferCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            size,
            usage: buffer_usage,
            sharing_mode: vk::SharingMode::Exclusive,
            queue_family_index_count: queue_families.len() as u32,
            p_queue_family_indices: &queue_families as *const _,
        };

        let allocation_create_info = alloc::VmaAllocationCreateInfo {
            flags: allocation_flags,
            memoryTypeBits: 0,
            pUserData: ptr::null_mut(),
            pool: ptr::null_mut(),
            preferredFlags: 0,
            requiredFlags: 0,
            usage: allocation_usage,
        };

        let (handle, allocation, allocation_info) =
            alloc::create_buffer(allocator, &buffer_create_info, &allocation_create_info).unwrap();

        let node = RenderNode::Buffer {
            dynamic: dyn(&self.cpu_pool, ()),
            handle,
            allocation,
            allocation_info,
        };
        let ix = self.graph.add_node(node);
        self.graph.add_edge(device_ix, ix, Edge::Direct);

        Some(ix)
    }

    pub fn new_draw_calls(
        &mut self,
        f: Arc<Fn(NodeIndex, &RuntimeGraph, &ThreadPool, &specs::World, &Dynamic<()>)>,
    ) -> NodeIndex {
        let node = RenderNode::DrawCalls {
            f,
            dynamic: dyn(&self.cpu_pool, ()),
        };
        self.graph.add_node(node)
    }

    pub fn render_frame(&mut self, world: &specs::World) {
        self.frame_number += 1;
        use petgraph::visit::Walker;
        for ix in visit::Topo::new(&self.graph).iter(&self.graph) {
            match self.graph[ix] {
                RenderNode::Instance { .. }
                | RenderNode::Swapchain { .. }
                | RenderNode::CommandPool { .. }
                | RenderNode::DescriptorSetLayout { .. }
                | RenderNode::DescriptorPool { .. }
                | RenderNode::DescriptorSet { .. }
                | RenderNode::PipelineLayout { .. }
                | RenderNode::Buffer { .. }
                | RenderNode::PersistentSemaphore { .. } => (),
                RenderNode::Device {
                    allocator,
                    ref dynamic,
                    ..
                } => {
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    let frame_index = self.frame_number;
                    *lock = (Box::new(order_fut.map_err(|_| ()).map(move |_| {
                        alloc::set_current_frame_index(allocator, frame_index);

                        ()
                    })) as Box<Future<Item = (), Error = ()>>)
                        .shared() as DynamicInner<()>;
                }
                RenderNode::Framebuffer { ref dynamic, .. } => {
                    let swapchain = search_deps_exactly_one(&self.graph, ix, |node| match *node {
                        RenderNode::Swapchain { ref handle, .. } => Some(Arc::clone(handle)),
                        _ => None,
                    }).expect("No swapchain connected to Framebuffer");
                    let signal_semaphore =
                        search_direct_deps_exactly_one(
                            &self.graph,
                            ix,
                            Direction::Outgoing,
                            |node| match *node {
                                RenderNode::PersistentSemaphore { ref handle, .. } => Some(*handle),
                                _ => None,
                            },
                        ).expect("No semaphore connected to Framebuffer - what should we signal?");
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
                    let swapchain = search_deps_exactly_one(&self.graph, ix, |node| match *node {
                        RenderNode::Swapchain { ref handle, .. } => Some(Arc::clone(handle)),
                        _ => None,
                    }).expect("No swapchain connected to Present");
                    let wait_semaphores = search_direct_deps(
                        &self.graph,
                        ix,
                        Direction::Incoming,
                        |node| match *node {
                            RenderNode::PersistentSemaphore { ref handle, .. } => Some(*handle),
                            _ => None,
                        },
                    );
                    let present_index =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Framebuffer { ref dynamic, .. } => {
                                Some(Arc::clone(dynamic))
                            }
                            _ => None,
                        }).expect(
                            "No framebuffer connected to Present - what image should we present?",
                        );
                    let present_index_fut = present_index
                        .read()
                        .expect("Failed to read present index")
                        .clone();
                    let graphics_queue =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device {
                                ref graphics_queue, ..
                            } => Some(Arc::clone(graphics_queue)),
                            _ => None,
                        }).expect(
                            "No device connected to Present - where should we submit the request?",
                        );
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = (Box::new(order_fut.join(present_index_fut).map_err(|_| ()).map(
                        move |(_, present_index)| {
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
                            ()
                        },
                    )) as Box<Future<Item = (), Error = ()>>)
                        .shared() as DynamicInner<()>;
                }
                RenderNode::AllocateCommandBuffer {
                    ref handles,
                    ref dynamic,
                    ..
                } => {
                    let device = search_deps_exactly_one(&self.graph, ix, |node| match *node {
                        RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Device not found in deps of AllocateCommandBuffer");
                    let pool = search_direct_deps_exactly_one(
                        &self.graph,
                        ix,
                        Direction::Incoming,
                        |node| match *node {
                            RenderNode::CommandPool { ref handle, .. } => Some(Arc::clone(handle)),
                            _ => None,
                        },
                    ).expect(
                        "Command pool not found in direct deps of AllocateCommandBuffer",
                    );
                    let (fb_count, fb_lock) =
                        search_direct_deps_exactly_one(
                            &self.graph,
                            ix,
                            Direction::Incoming,
                            |node| match *node {
                                RenderNode::Framebuffer {
                                    ref images,
                                    ref dynamic,
                                    ..
                                } => Some((images.len(), Arc::clone(dynamic))),
                                _ => None,
                            },
                        ).expect("Framebuffer not found in direct deps of AllocateCommandBuffer");
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let fb_fut = fb_lock.read().expect("Failed to read framebuffer").clone();
                    let handles = Arc::clone(handles);
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = (Box::new(order_fut.join(fb_fut).map_err(|_| ()).map(move |(_, fb)| {
                        let pool_lock = pool.lock()
                            .expect("failed to lock command pool for allocation");
                        let begin_info = vk::CommandBufferBeginInfo {
                            s_type: vk::StructureType::CommandBufferBeginInfo,
                            p_next: ptr::null(),
                            p_inheritance_info: ptr::null(),
                            flags: vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                        };
                        let mut handles = handles
                            .write()
                            .expect("Failed to own AllocateCommandBuffer data");
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
                                device
                                    .begin_command_buffer(current_frame, &begin_info)
                                    .unwrap();
                            }
                            fields::AllocateCommandBuffer::Dynamic { current_frame }
                        } else {
                            let current_frame = handles[present_ix as usize];
                            unsafe {
                                device
                                    .reset_command_buffer(
                                        current_frame,
                                        vk::CommandBufferResetFlags::empty(),
                                    )
                                    .unwrap();
                                device
                                    .begin_command_buffer(current_frame, &begin_info)
                                    .unwrap();
                            }
                            fields::AllocateCommandBuffer::Dynamic { current_frame }
                        }
                    })) as Box<Future<Item = _, Error = ()>>)
                        .shared();
                }
                RenderNode::SubmitCommandBuffer { ref dynamic, .. } => {
                    let (device, graphics_queue) =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device {
                                ref device,
                                ref graphics_queue,
                                ..
                            } => Some((Arc::clone(device), Arc::clone(graphics_queue))),
                            _ => None,
                        }).expect("Device not found in deps of SubmitCommandBuffer");
                    let allocated_lock =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::AllocateCommandBuffer { ref dynamic, .. } => {
                                Some(Arc::clone(dynamic))
                            }
                            _ => None,
                        }).expect("Device not found in deps of SubmitCommandBuffer");
                    let wait_semaphores = search_direct_deps(
                        &self.graph,
                        ix,
                        Direction::Incoming,
                        |node| match *node {
                            RenderNode::PersistentSemaphore { handle, .. } => Some(handle),
                            _ => None,
                        },
                    );
                    let signal_semaphores = search_direct_deps(
                        &self.graph,
                        ix,
                        Direction::Outgoing,
                        |node| match *node {
                            RenderNode::PersistentSemaphore { handle, .. } => Some(handle),
                            _ => None,
                        },
                    );
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let allocated = allocated_lock
                        .write()
                        .expect("failed to read command buffer")
                        .clone();
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = (Box::new(order_fut.join(allocated).map_err(|_| ()).map(
                        move |(_, allocated)| unsafe {
                            let cb = allocated.current_frame;
                            device.end_command_buffer(cb).unwrap();
                            let dst_stage_masks =
                                vec![vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT; wait_semaphores.len()];
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

                            device
                                .queue_submit(*queue_lock, &submits, submit_fence)
                                .unwrap();

                            device
                                .wait_for_fences(&[submit_fence], true, u64::MAX)
                                .expect("Wait for fence failed.");
                            device.destroy_fence(submit_fence, None);

                            ()
                        },
                    )) as Box<Future<Item = (), Error = ()>>)
                        .shared() as DynamicInner<()>;
                }
                RenderNode::Renderpass {
                    handle,
                    ref dynamic,
                } => {
                    let device = search_deps_exactly_one(&self.graph, ix, |node| match *node {
                        RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Device not found in deps of Renderpass");
                    let (fb_handles, fb_lock) =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Framebuffer {
                                ref handles,
                                ref dynamic,
                                ..
                            } => Some((Arc::clone(handles), Arc::clone(dynamic))),
                            _ => None,
                        }).expect("Framebuffer not found in deps of Renderpass");
                    let command_buffer_locked =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::AllocateCommandBuffer { ref dynamic, .. } => {
                                Some(Arc::clone(dynamic))
                            }
                            _ => None,
                        }).expect("Command Buffer not found in deps of Renderpass");
                    let (window_width, window_height) =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Instance {
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
                    *lock = (Box::new(command_buffer_fut.join(fb_fut).map_err(|_| ()).map(
                        move |(cb_dyn, fb)| {
                            let fb_ix = (*fb).current_present_index;
                            let cb = cb_dyn.current_frame;
                            let clear_values = vec![
                                vk::ClearValue {
                                    color: vk::ClearColorValue { float32: [0.0; 4] },
                                },
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
                                device.cmd_begin_render_pass(
                                    cb,
                                    &begin_info,
                                    vk::SubpassContents::Inline,
                                );
                            }
                            ()
                        },
                    )) as Box<Future<Item = (), Error = ()>>)
                        .shared() as DynamicInner<()>;
                }
                RenderNode::EndRenderpass { ref dynamic, .. } => {
                    let device = search_deps_exactly_one(&self.graph, ix, |node| match *node {
                        RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Device not found in deps of EndRenderpass");
                    let command_buffer_locked =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::AllocateCommandBuffer { ref dynamic, .. } => {
                                Some(Arc::clone(dynamic))
                            }
                            _ => None,
                        }).expect("Command Buffer not found in deps of EndRenderpass");
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let command_buffer_fut = command_buffer_locked
                        .read()
                        .expect("Could not read command buffer")
                        .clone();
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = (Box::new(order_fut.join(command_buffer_fut).map_err(|_| ()).map(
                        move |(_, cb_dyn)| {
                            let cb = cb_dyn.current_frame;
                            unsafe {
                                device.cmd_end_render_pass(cb);
                            }
                            ()
                        },
                    )) as Box<Future<Item = (), Error = ()>>)
                        .shared() as DynamicInner<()>;
                }
                RenderNode::NextSubpass { ref dynamic, .. } => {
                    let device = search_deps_exactly_one(&self.graph, ix, |node| match *node {
                        RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Device not found in deps of NextSubpass");
                    let command_buffer_locked =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::AllocateCommandBuffer { ref dynamic, .. } => {
                                Some(Arc::clone(dynamic))
                            }
                            _ => None,
                        }).expect("Command Buffer not found in deps of NextSubpass");
                    let command_buffer_fut = command_buffer_locked
                        .read()
                        .expect("Could not read command buffer")
                        .clone();
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = (Box::new(order_fut.join(command_buffer_fut).map_err(|_| ()).map(
                        move |(_, cb_dyn)| {
                            let cb = cb_dyn.current_frame;
                            unsafe {
                                device.cmd_next_subpass(cb, vk::SubpassContents::Inline);
                            }
                            ()
                        },
                    )) as Box<Future<Item = (), Error = ()>>)
                        .shared() as DynamicInner<()>;
                }
                RenderNode::GraphicsPipeline {
                    handle,
                    ref dynamic,
                } => {
                    let device = search_deps_exactly_one(&self.graph, ix, |node| match *node {
                        RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    }).expect("Device not found in deps of NextSubpass");
                    let command_buffer_locked =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::AllocateCommandBuffer { ref dynamic, .. } => {
                                Some(Arc::clone(dynamic))
                            }
                            _ => None,
                        }).expect("Command Buffer not found in deps of NextSubpass");
                    let command_buffer_fut = command_buffer_locked
                        .read()
                        .expect("Could not read command buffer")
                        .clone();
                    let order_fut = wait_on_direct_deps(&self.cpu_pool, &self.graph, ix);
                    let mut lock = dynamic.write().expect("failed to lock present for writing");
                    *lock = (Box::new(order_fut.join(command_buffer_fut).map_err(|_| ()).map(
                        move |(_, cb_dyn)| {
                            let cb = cb_dyn.current_frame;
                            unsafe {
                                device.cmd_bind_pipeline(
                                    cb,
                                    vk::PipelineBindPoint::Graphics,
                                    handle,
                                );
                            }
                            ()
                        },
                    )) as Box<Future<Item = (), Error = ()>>)
                        .shared();
                }
                RenderNode::DrawCalls { ref f, ref dynamic } => {
                    f(ix, &self.graph, &self.cpu_pool, world, dynamic);
                }
            }
        }
    }
}

impl Drop for RenderDAG {
    fn drop(&mut self) {
        use std::boxed::FnBox;
        let mut finalizer_graph: StableDiGraph<Box<FnBox()>, Edge> = self.graph.filter_map(
            |ix, node| match *node {
                RenderNode::PresentFramebuffer { .. }
                | RenderNode::SubmitCommandBuffer { .. }
                | RenderNode::NextSubpass { .. }
                | RenderNode::EndRenderpass { .. } => None,
                |
                RenderNode::DrawCalls { .. } => None,
                RenderNode::Instance { ref instance, .. } => {
                    let i = Arc::clone(instance);
                    Some(Box::new(move || {
                        drop(i);
                    }) as Box<FnBox()>)
                }
                RenderNode::Device {
                    ref device,
                    allocator,
                    ..
                } => {
                    let d = Arc::clone(device);
                    Some(Box::new(move || {
                        alloc::destroy(allocator);
                        drop(d);
                    }) as Box<FnBox()>)
                }
                RenderNode::Swapchain { ref handle, .. } => {
                    let parent_device = search_deps(&self.graph, ix, |node| match *node {
                        RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                        _ => None,
                    });
                    let swapchain = Arc::clone(handle);
                    Some(Box::new(move || {
                        drop(swapchain);
                        drop(parent_device);
                    }))
                }
                RenderNode::Framebuffer {
                    ref image_views,
                    ref handles,
                    ..
                } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Instance for Framebuffer");
                    let image_views = Arc::clone(image_views);
                    let handles = Arc::clone(handles);

                    Some(Box::new(move || unsafe {
                        for view in image_views.iter() {
                            parent_device.destroy_image_view(*view, None);
                        }
                        for handle in handles.iter() {
                            parent_device.destroy_framebuffer(*handle, None);
                        }
                    }) as Box<FnBox()>)
                }
                RenderNode::CommandPool { ref handle, .. } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Instance for CommandPool");
                    let command_pool = Arc::clone(handle);
                    Some(Box::new(move || unsafe {
                        let lock = command_pool.lock().expect("Cannot lock command pool");
                        parent_device.destroy_command_pool(*lock, None);
                    }) as Box<FnBox()>)
                }
                RenderNode::PersistentSemaphore { ref handle, .. } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Device for PersistentSemaphore");
                    let handle = *handle;

                    Some(Box::new(move || unsafe {
                        parent_device.destroy_semaphore(handle, None);
                    }))
                }
                RenderNode::AllocateCommandBuffer { ref handles, .. } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Device for AllocateCommandBuffer");
                    let pool = search_direct_deps_exactly_one(
                        &self.graph,
                        ix,
                        Direction::Incoming,
                        |node| match *node {
                            RenderNode::CommandPool { ref handle, .. } => Some(Arc::clone(handle)),
                            _ => None,
                        },
                    ).expect(
                        "Command pool not found in direct deps of AllocateCommandBuffer",
                    );
                    let handles = Arc::clone(handles);

                    Some(Box::new(move || {
                        let pool_lock = pool.lock()
                            .expect("Failed locking CommandPool for AllocateCommandBuffer dtor");
                        let mut cb_lock =
                            handles.write().expect("Failed locking AllocateCB for dtor");
                        unsafe {
                            parent_device.free_command_buffers(*pool_lock, &cb_lock);
                        }
                        (*cb_lock).clear();
                    }))
                }
                RenderNode::Renderpass { ref handle, .. } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Device for AllocateCommandBuffer");
                    let handle = *handle;
                    Some(Box::new(move || unsafe {
                        parent_device.destroy_render_pass(handle, None);
                    }))
                }
                RenderNode::DescriptorSetLayout { handle, .. } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Device for DescriptorSetLayout");
                    Some(Box::new(move || unsafe {
                        parent_device.destroy_descriptor_set_layout(handle, None);
                    }))
                }
                RenderNode::DescriptorPool { handle, .. } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Instance for DescriptorPool");
                    Some(Box::new(move || unsafe {
                        parent_device.destroy_descriptor_pool(handle, None);
                    }) as Box<FnBox()>)
                }
                RenderNode::DescriptorSet { handle, .. } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Instance for DescriptorSet");
                    let pool = search_direct_deps_exactly_one(
                        &self.graph,
                        ix,
                        Direction::Incoming,
                        |node| match *node {
                            RenderNode::DescriptorPool { handle, .. } => Some(handle),
                            _ => None,
                        },
                    ).expect("Expected one parent Instance for DescriptorSet");
                    Some(Box::new(move || unsafe {
                        parent_device.free_descriptor_sets(pool, &[handle]);
                    }) as Box<FnBox()>)
                }
                RenderNode::PipelineLayout { ref handle, .. } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Device for PipelineLayout");
                    let handle = *handle;
                    Some(Box::new(move || unsafe {
                        parent_device.destroy_pipeline_layout(handle, None);
                    }))
                }
                RenderNode::GraphicsPipeline { handle, .. } => {
                    let parent_device =
                        search_deps_exactly_one(&self.graph, ix, |node| match *node {
                            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
                            _ => None,
                        }).expect("Expected one parent Device for PipelineLayout");
                    Some(Box::new(move || unsafe {
                        parent_device.destroy_pipeline(handle, None);
                    }))
                }
                RenderNode::Buffer {
                    handle, allocation, ..
                } => {
                    let allocator = search_deps_exactly_one(&self.graph, ix, |node| match *node {
                        RenderNode::Device { allocator, .. } => Some(allocator),
                        _ => None,
                    }).expect("Expected one parent Device for PipelineLayout");
                    Some(Box::new(move || {
                        alloc::destroy_buffer(allocator, handle, allocation)
                    }))
                }
            },
            |_, edge| Some(edge.clone()),
        );
        for ix in self.graph.node_indices() {
            if let RenderNode::Device { ref device, .. } = self.graph[ix] {
                device.device_wait_idle().unwrap();
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
