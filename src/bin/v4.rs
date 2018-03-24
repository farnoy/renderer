#![feature(conservative_impl_trait)]

extern crate ash;
extern crate cgmath;
extern crate forward_renderer;
extern crate futures;
extern crate specs;
extern crate winit;

use ash::{extensions, vk, version::{DeviceV1_0, InstanceV1_0}};
use cgmath::Rotation3;
use forward_renderer::{alloc, create_surface, device, ecs::*, entry, instance, swapchain};
use futures::{channel::oneshot, executor::{block_on, spawn, ThreadPool},
              future::{join_all, lazy, ok, FutureResult, Lazy}, prelude::*};
use std::{ptr, borrow::Cow, default::Default, ffi::CString, fs::File, io::Read,
          mem::{size_of, transmute}, os::raw::c_void, path::PathBuf, sync::{Arc, Mutex}, u64};

struct Instance {
    window: Arc<winit::Window>,
    events_loop: Arc<winit::EventsLoop>,
    instance: Arc<instance::Instance>,
    entry: Arc<entry::Entry>,
    surface: vk::SurfaceKHR,
    window_width: u32,
    window_height: u32,
}

struct Device {
    device: Arc<device::Device>,
    physical_device: vk::PhysicalDevice,
    allocator: alloc::VmaAllocator,
    graphics_queue_family: u32,
    compute_queue_family: u32,
    graphics_queue: Arc<Mutex<vk::Queue>>,
    compute_queues: Arc<Vec<Mutex<vk::Queue>>>,
    transfer_queue: Arc<Mutex<vk::Queue>>,
}

struct Swapchain {
    handle: swapchain::Swapchain,
    surface_format: vk::SurfaceFormatKHR,
}

struct RenderPass {
    handle: vk::RenderPass,
    device: Arc<Device>,
}

struct CommandPool {
    handle: Mutex<vk::CommandPool>,
    device: Arc<Device>,
}

struct Framebuffer {
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    handles: Vec<vk::Framebuffer>,
    device: Arc<Device>,
}

struct Semaphore {
    handle: vk::Semaphore,
    device: Arc<Device>,
}

struct Fence {
    handle: vk::Fence,
    device: Arc<Device>,
}

struct CommandBuffer {
    handle: vk::CommandBuffer,
    pool: Arc<CommandPool>,
    device: Arc<Device>,
}

struct Buffer {
    handle: vk::Buffer,
    allocation: alloc::VmaAllocation,
    allocation_info: alloc::VmaAllocationInfo,
    device: Arc<Device>,
}

struct DescriptorPool {
    handle: vk::DescriptorPool,
    device: Arc<Device>,
}

struct DescriptorSetLayout {
    handle: vk::DescriptorSetLayout,
    device: Arc<Device>,
}

struct DescriptorSet {
    handle: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
    device: Arc<Device>,
}

struct PipelineLayout {
    handle: vk::PipelineLayout,
    device: Arc<Device>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
}

struct GraphicsPipeline {
    handle: vk::Pipeline,
    device: Arc<Device>,
}

impl Drop for Device {
    fn drop(&mut self) {
        alloc::destroy(self.allocator);
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_render_pass(self.handle, None) }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_command_pool(*self.handle.lock().unwrap(), None)
        }
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.device_wait_idle().unwrap();
            for view in self.image_views.iter() {
                self.device.device.destroy_image_view(*view, None);
            }
            for handle in self.handles.iter() {
                self.device.device.destroy_framebuffer(*handle, None);
            }
        }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_semaphore(self.handle, None);
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_fence(self.handle, None);
        }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            let pool_lock = self.pool.handle.lock().unwrap();
            self.device
                .device
                .free_command_buffers(*pool_lock, &[self.handle]);
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { alloc::destroy_buffer(self.device.allocator, self.handle, self.allocation) }
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_descriptor_set_layout(self.handle, None)
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_descriptor_pool(self.handle, None)
        }
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .free_descriptor_sets(self.pool.handle, &[self.handle])
        }
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_pipeline_layout(self.handle, None)
        }
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_pipeline(self.handle, None)
        }
    }
}

fn main() {
    let mut world = specs::World::new();
    world.register::<Position>();
    world.register::<Rotation>();
    world.register::<Scale>();
    world.register::<Matrices>();
    let mut threadpool = ThreadPool::builder().pool_size(4).create();
    let kuk: Lazy<Result<u8, ()>, _> = lazy(|_| Ok(5u8));
    threadpool.run(lazy(|_| {
        println!("kuk");
        fork(kuk).then(|res| {
            println!("res {:?}", res);
            Ok(()) as Result<(), ()>
        });
        Ok(()) as Result<(), ()>
    }));
    let instance = new_window(800, 600);
    let device = new_device(Arc::clone(&instance));
    let swapchain = new_swapchain(Arc::clone(&instance), Arc::clone(&device));
    let present_semaphore = new_semaphore(Arc::clone(&device));
    let rendering_complete_semaphore = new_semaphore(Arc::clone(&device));
    let graphics_command_pool = new_command_pool(
        Arc::clone(&device),
        device.graphics_queue_family,
        vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    );
    let main_renderpass = setup_renderpass(Arc::clone(&device), Arc::clone(&swapchain));
    let framebuffer = new_framebuffer(
        Arc::clone(&instance),
        Arc::clone(&device),
        Arc::clone(&swapchain),
        Arc::clone(&main_renderpass),
    );

    let descriptor_set_layout = new_descriptor_set_layout(
        Arc::clone(&device),
        &[
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_VERTEX_BIT,
                p_immutable_samplers: ptr::null(),
            },
        ],
    );

    let pipeline_layout = new_pipeline_layout(
        Arc::clone(&device),
        &[descriptor_set_layout.handle],
        vec![
            vk::PushConstantRange {
                stage_flags: vk::SHADER_STAGE_VERTEX_BIT,
                offset: 0,
                size: (size_of::<f32>() * 2 * 3) as u32,
            },
        ],
    );
    let triangle_pipeline = new_graphics_pipeline(
        Arc::clone(&instance),
        Arc::clone(&device),
        Arc::clone(&pipeline_layout),
        Arc::clone(&main_renderpass),
        &[],
        &[],
        &[
            (
                vk::SHADER_STAGE_VERTEX_BIT,
                PathBuf::from(env!("OUT_DIR")).join("triangle.vert.spv"),
            ),
            (
                vk::SHADER_STAGE_FRAGMENT_BIT,
                PathBuf::from(env!("OUT_DIR")).join("triangle.frag.spv"),
            ),
        ],
    );
    let descriptor_pool = new_descriptor_pool(
        Arc::clone(&device),
        1,
        &[
            vk::DescriptorPoolSize {
                typ: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1024,
            },
        ],
    );
    let ubo_set = new_descriptor_set(
        Arc::clone(&device),
        Arc::clone(&descriptor_pool),
        Arc::clone(&descriptor_set_layout),
    );
    let ubo_buffer = new_buffer(
        Arc::clone(&device),
        vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        alloc::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_MAPPED_BIT.0 as u32,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        4 * 4 * 4 * 1024,
    );
    {
        let buffer_updates = &[
            vk::DescriptorBufferInfo {
                buffer: ubo_buffer.handle,
                offset: 0,
                range: 1024 * size_of::<cgmath::Matrix4<f32>>() as vk::DeviceSize,
            },
        ];
        unsafe {
            device.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WriteDescriptorSet,
                        p_next: ptr::null(),
                        dst_set: ubo_set.handle,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UniformBuffer,
                        p_image_info: ptr::null(),
                        p_buffer_info: buffer_updates.as_ptr(),
                        p_texel_buffer_view: ptr::null(),
                    },
                ],
                &[],
            );
        }
    }

    let projection = cgmath::perspective(
        cgmath::Deg(60.0),
        instance.window_width as f32 / instance.window_height as f32,
        0.1,
        100.0,
    );
    let view = cgmath::Matrix4::look_at(
        cgmath::Point3::new(0.0, 0.0, -5.0),
        cgmath::Point3::new(0.0, 0.0, 0.0),
        cgmath::vec3(0.0, 1.0, 0.0),
    );
    world
        .create_entity()
        .with::<Position>(Position(cgmath::Vector3::new(0.0, 0.0, 0.0)))
        .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_x(cgmath::Deg(0.0))))
        .with::<Scale>(Scale(1.0))
        .with::<Matrices>(Matrices::one())
        .build();
    let ubo_mapped =
        unsafe { transmute::<*mut c_void, *mut cgmath::Matrix4<f32>>(ubo_buffer.allocation_info.pMappedData) };
    let mut dispatcher = specs::DispatcherBuilder::new()
        .add(SteadyRotation, "steady_rotation", &[])
        .add(
            MVPCalculation { projection, view },
            "mvp",
            &["steady_rotation"],
        )
        .add(MVPUpload { dst: ubo_mapped }, "mvp_upload", &["mvp"])
        .build();

    for _i in 1..400 {
        dispatcher.dispatch(&world.res);
        world.maintain();
        let image_index = unsafe {
            swapchain
                .handle
                .ext
                .acquire_next_image_khr(
                    swapchain.handle.swapchain,
                    u64::MAX,
                    present_semaphore.handle,
                    vk::Fence::null(),
                )
                .unwrap()
        };
        println!("image_index {}", image_index);
        let command_buffer =
            allocate_command_buffer(Arc::clone(&device), Arc::clone(&graphics_command_pool));
        unsafe {
            {
                let dummy_buffer = new_buffer(
                    Arc::clone(&device),
                    vk::BUFFER_USAGE_TRANSFER_DST_BIT,
                    0,
                    alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                    1024 * 256,
                );
                {
                    let begin_info = vk::CommandBufferBeginInfo {
                        s_type: vk::StructureType::CommandBufferBeginInfo,
                        p_next: ptr::null(),
                        p_inheritance_info: ptr::null(),
                        flags: vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                    };
                    device
                        .device
                        .begin_command_buffer(command_buffer.handle, &begin_info);
                }
                {
                    let clear_values = &[
                        vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
                        },
                    ];
                    let begin_info = vk::RenderPassBeginInfo {
                        s_type: vk::StructureType::RenderPassBeginInfo,
                        p_next: ptr::null(),
                        render_pass: main_renderpass.handle,
                        framebuffer: framebuffer.handles[image_index as usize],
                        render_area: vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: instance.window_width,
                                height: instance.window_height,
                            },
                        },
                        clear_value_count: clear_values.len() as u32,
                        p_clear_values: clear_values.as_ptr(),
                    };

println!("marker start");
                    device.device.debug_marker_start(command_buffer.handle, "main renderpass", [0.0, 0.0, 1.0, 1.0]);
println!("marker end");
                    device.device.cmd_begin_render_pass(
                        command_buffer.handle,
                        &begin_info,
                        vk::SubpassContents::Inline,
                    );
                    device.device.cmd_bind_pipeline(
                        command_buffer.handle,
                        vk::PipelineBindPoint::Graphics,
                        triangle_pipeline.handle
                    );
                    device.device.cmd_bind_descriptor_sets(
                        command_buffer.handle,
                        vk::PipelineBindPoint::Graphics,
                        pipeline_layout.handle,
                        0,
                        &[ubo_set.handle],
                        &[]
                    );
                    let constants = [1.0f32, 1.0, -1.0, 1.0, 1.0, -1.0];
                    use std::mem::transmute;
                    use std::slice::from_raw_parts;

                    let casted: &[u8] = {
                        from_raw_parts(
                            transmute::<*const f32, *const u8>(constants.as_ptr()),
                            2 * 3 * 4,
                        )
                    };
                    device.device.cmd_push_constants(
                        command_buffer.handle,
                        pipeline_layout.handle,
                        vk::SHADER_STAGE_VERTEX_BIT,
                        0,
                        casted,
                    );
                    device.device.cmd_draw(command_buffer.handle, 3, 1, 0, 0);
                    device.device.cmd_end_render_pass(command_buffer.handle);
                    device.device.debug_marker_end(command_buffer.handle);
                }
                device.device.cmd_fill_buffer(
                    command_buffer.handle,
                    dummy_buffer.handle,
                    0,
                    1024 * 256,
                    1234,
                );
                device
                    .device
                    .end_command_buffer(command_buffer.handle)
                    .unwrap();
                {
                    let wait_semaphores = &[present_semaphore.handle];
                    let signal_semaphores = &[rendering_complete_semaphore.handle];
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
                            p_command_buffers: &command_buffer.handle,
                            signal_semaphore_count: signal_semaphores.len() as u32,
                            p_signal_semaphores: signal_semaphores.as_ptr(),
                        },
                    ];
                    let queue_lock = device
                        .graphics_queue
                        .lock()
                        .expect("can't lock the submit queue");

                    let submit_fence = new_fence(Arc::clone(&device));

                    device
                        .device
                        .queue_submit(*queue_lock, &submits, submit_fence.handle)
                        .unwrap();

                    {
                        let device = Arc::clone(&device);
                        threadpool.run(lazy(move |_| {
                            spawn(lazy(move |_| {
                                println!("dtor previous frame");
                                device
                                    .device
                                    .wait_for_fences(&[submit_fence.handle], true, u64::MAX)
                                    .expect("Wait for fence failed.");
                                drop(command_buffer);
                                drop(dummy_buffer);
                                drop(submit_fence);
                                Ok(())
                            }))
                        }));
                    }
                }
            }

            {
                let wait_semaphores = &[rendering_complete_semaphore.handle];
                let present_info = vk::PresentInfoKHR {
                    s_type: vk::StructureType::PresentInfoKhr,
                    p_next: ptr::null(),
                    wait_semaphore_count: wait_semaphores.len() as u32,
                    p_wait_semaphores: wait_semaphores.as_ptr(),
                    swapchain_count: 1,
                    p_swapchains: &swapchain.handle.swapchain,
                    p_image_indices: &image_index,
                    p_results: ptr::null_mut(),
                };
                let queue = device
                    .graphics_queue
                    .lock()
                    .expect("Failed to acquire lock on graphics queue");

                swapchain
                    .handle
                    .ext
                    .queue_present_khr(*queue, &present_info)
                    .unwrap();
            }
        }
    }
}

fn fork<T: 'static + Send, E: Send, F: 'static + Send + Sync>(
    f: F,
) -> impl Future<Item = T, Error = oneshot::Canceled>
where
    F: Future<Item = T, Error = E>,
{
    let (sender, receiver) = oneshot::channel::<T>();
    spawn(f.map(|res| {
        sender.send(res);
        ()
    }).then(|_| Ok(())))
        .then(|_| receiver)
}

fn setup_renderpass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
    let attachment_descriptions = [
        vk::AttachmentDescription {
            format: swapchain.surface_format.format,
            flags: vk::AttachmentDescriptionFlags::empty(),
            samples: vk::SAMPLE_COUNT_1_BIT,
            load_op: vk::AttachmentLoadOp::Clear,
            store_op: vk::AttachmentStoreOp::Store,
            stencil_load_op: vk::AttachmentLoadOp::DontCare,
            stencil_store_op: vk::AttachmentStoreOp::DontCare,
            initial_layout: vk::ImageLayout::Undefined,
            final_layout: vk::ImageLayout::PresentSrcKhr,
        },
    ];
    let color_attachment_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::ColorAttachmentOptimal,
    };
    let subpass_descs = [
        vk::SubpassDescription {
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: ptr::null(),
            flags: Default::default(),
            pipeline_bind_point: vk::PipelineBindPoint::Graphics,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            p_resolve_attachments: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        },
    ];
    let subpass_dependencies = [
        vk::SubpassDependency {
            dependency_flags: Default::default(),
            src_subpass: vk::VK_SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            src_access_mask: Default::default(),
            dst_access_mask: vk::ACCESS_COLOR_ATTACHMENT_READ_BIT
                | vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dst_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        },
    ];
    new_renderpass(
        device,
        &attachment_descriptions,
        &subpass_descs,
        &subpass_dependencies,
    )
}

fn new_window(window_width: u32, window_height: u32) -> Arc<Instance> {
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

    Arc::new(Instance {
        window: Arc::new(window),
        events_loop: Arc::new(events_loop),
        instance: instance,
        entry: entry,
        surface: surface,
        window_width: window_width,
        window_height: window_height,
    })
}

fn new_device(instance: Arc<Instance>) -> Arc<Device> {
    let Instance {
        ref entry,
        ref instance,
        surface,
        ..
    } = *instance;

    let pdevices = instance
        .enumerate_physical_devices()
        .expect("Physical device error");
    let surface_loader = extensions::Surface::new(entry.vk(), instance.vk())
        .expect("Unable to load the Surface extension");

    let pdevice = pdevices[0];
    let graphics_queue_family = {
        instance
            .get_physical_device_queue_family_properties(pdevice)
            .iter()
            .enumerate()
            .filter_map(|(ix, info)| {
                let supports_graphic_and_surface = info.queue_flags.subset(vk::QUEUE_GRAPHICS_BIT)
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
            .unwrap()
    };
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
    let transfer_queue_family = if cfg!(feature = "validation") {
        compute_queue_family
    } else {
        instance
            .get_physical_device_queue_family_properties(pdevice)
            .iter()
            .enumerate()
            .filter_map(|(ix, info)| {
                if info.queue_flags.subset(vk::QUEUE_TRANSFER_BIT)
                    && !info.queue_flags.subset(vk::QUEUE_GRAPHICS_BIT)
                    && !info.queue_flags.subset(vk::QUEUE_COMPUTE_BIT)
                {
                    Some(ix as u32)
                } else {
                    None
                }
            })
            .next()
            .unwrap_or(compute_queue_family)
    };
    // TODO: this needs to be reworked in a DAG way, right now vk::Queue handles
    // are copied so there is no thread safety
    let queue_decl = if graphics_queue_family == compute_queue_family
        && graphics_queue_family == transfer_queue_family
    {
        // Renderdoc
        vec![(graphics_queue_family, 1)]
    } else if cfg!(feature = "validation") || compute_queue_family == transfer_queue_family {
        vec![
            (graphics_queue_family, 1),
            (compute_queue_family, compute_queue_len),
        ]
    } else {
        vec![
            (graphics_queue_family, 1),
            (compute_queue_family, compute_queue_len),
            (transfer_queue_family, 1),
        ]
    };
    let device = device::Device::new(&instance, pdevice, &queue_decl).unwrap();
    let allocator = alloc::create(device.vk().handle(), pdevice).unwrap();
    let graphics_queue = unsafe { device.vk().get_device_queue(graphics_queue_family, 0) };
    let compute_queues = (0..compute_queue_len)
        .map(|ix| unsafe { device.vk().get_device_queue(compute_queue_family, ix) })
        .collect::<Vec<_>>();

    let transfer_queue = unsafe { device.vk().get_device_queue(transfer_queue_family, 0) };

    Arc::new(Device {
        device,
        physical_device: pdevice,
        allocator,
        graphics_queue_family,
        compute_queue_family,
        graphics_queue: Arc::new(Mutex::new(graphics_queue)),
        compute_queues: Arc::new(
            compute_queues
                .iter()
                .cloned()
                .map(|a| Mutex::new(a))
                .collect(),
        ),
        transfer_queue: Arc::new(Mutex::new(transfer_queue)),
    })
}

fn new_swapchain(instance: Arc<Instance>, device: Arc<Device>) -> Arc<Swapchain> {
    let Instance {
        ref entry,
        ref instance,
        surface,
        window_width,
        window_height,
        ..
    } = *instance;
    let Device {
        ref device,
        physical_device,
        ..
    } = *device;

    let surface_loader = extensions::Surface::new(entry.vk(), instance.vk())
        .expect("Unable to load the Surface extension");
    let present_mode = vk::PresentModeKHR::Fifo;
    let surface_formats = surface_loader
        .get_physical_device_surface_formats_khr(physical_device, surface)
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
        .get_physical_device_surface_capabilities_khr(physical_device, surface)
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
        extensions::Swapchain::new(instance.vk(), device.vk()).expect("Unable to load swapchain");
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

    let swapchain = swapchain::Swapchain::new(swapchain_loader, swapchain);
    Arc::new(Swapchain {
        handle: swapchain,
        surface_format,
    })
}

fn new_renderpass(
    device: Arc<Device>,
    attachments: &[vk::AttachmentDescription],
    subpass_descs: &[vk::SubpassDescription],
    subpass_dependencies: &[vk::SubpassDependency],
) -> Arc<RenderPass> {
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
            .device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap()
    };

    Arc::new(RenderPass {
        handle: renderpass,
        device,
    })
}

fn new_command_pool(
    device: Arc<Device>,
    queue_family: u32,
    flags: vk::CommandPoolCreateFlags,
) -> Arc<CommandPool> {
    let pool_create_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::CommandPoolCreateInfo,
        p_next: ptr::null(),
        flags: flags,
        queue_family_index: queue_family,
    };
    let pool = unsafe {
        device
            .device
            .create_command_pool(&pool_create_info, None)
            .unwrap()
    };

    let cp = CommandPool {
        handle: Mutex::new(pool),
        device,
    };
    Arc::new(cp)
}

fn new_framebuffer(
    instance: Arc<Instance>,
    device: Arc<Device>,
    swapchain: Arc<Swapchain>,
    renderpass: Arc<RenderPass>,
) -> Arc<Framebuffer> {
    let Swapchain {
        handle: ref swapchain,
        ref surface_format,
        ..
    } = *swapchain;
    let RenderPass { ref handle, .. } = *renderpass;
    let Instance {
        window_width,
        window_height,
        ..
    } = *instance;

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
            unsafe {
                device
                    .device
                    .create_image_view(&create_view_info, None)
                    .unwrap()
            }
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
                render_pass: renderpass.handle,
                attachment_count: framebuffer_attachments.len() as u32,
                p_attachments: framebuffer_attachments.as_ptr(),
                width: window_width,
                height: window_height,
                layers: 1,
            };
            unsafe {
                device
                    .device
                    .create_framebuffer(&frame_buffer_create_info, None)
                    .unwrap()
            }
        })
        .collect::<Vec<_>>();

    let framebuffer = Framebuffer {
        images: images,
        image_views: image_views,
        handles: handles,
        device,
    };

    Arc::new(framebuffer)
}

fn new_semaphore(device: Arc<Device>) -> Arc<Semaphore> {
    let create_info = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SemaphoreCreateInfo,
        p_next: ptr::null(),
        flags: vk::SemaphoreCreateFlags::empty(),
    };
    let semaphore = unsafe { device.device.create_semaphore(&create_info, None).unwrap() };

    Arc::new(Semaphore {
        handle: semaphore,
        device,
    })
}

fn allocate_command_buffer(device: Arc<Device>, pool: Arc<CommandPool>) -> Arc<CommandBuffer> {
    let command_buffers = unsafe {
        let pool_lock = pool.handle.lock().unwrap();
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::CommandBufferAllocateInfo,
            p_next: ptr::null(),
            command_buffer_count: 1,
            command_pool: *pool_lock,
            level: vk::CommandBufferLevel::Primary,
        };
        device
            .device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()
    };

    Arc::new(CommandBuffer {
        handle: command_buffers[0],
        pool,
        device,
    })
}

fn new_fence(device: Arc<Device>) -> Arc<Fence> {
    let create_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FenceCreateInfo,
        p_next: ptr::null(),
        flags: vk::FenceCreateFlags::empty(),
    };
    let fence = unsafe {
        device
            .device
            .create_fence(&create_info, None)
            .expect("Create fence failed.")
    };
    Arc::new(Fence {
        device,
        handle: fence,
    })
}

fn new_buffer(
    device: Arc<Device>,
    buffer_usage: vk::BufferUsageFlags,
    allocation_flags: alloc::VmaAllocationCreateFlags,
    allocation_usage: alloc::VmaMemoryUsage,
    size: vk::DeviceSize,
) -> Arc<Buffer> {
    let queue_families = [device.graphics_queue_family, device.compute_queue_family];
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

    let (handle, allocation, allocation_info) = alloc::create_buffer(
        device.allocator,
        &buffer_create_info,
        &allocation_create_info,
    ).unwrap();

    Arc::new(Buffer {
        handle,
        allocation,
        allocation_info,
        device,
    })
}

fn new_descriptor_set_layout(
    device: Arc<Device>,
    bindings: &[vk::DescriptorSetLayoutBinding],
) -> Arc<DescriptorSetLayout> {
    let create_info = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DescriptorSetLayoutCreateInfo,
        p_next: ptr::null(),
        flags: Default::default(),
        binding_count: bindings.len() as u32,
        p_bindings: bindings.as_ptr(),
    };
    let handle = unsafe {
        device
            .device
            .create_descriptor_set_layout(&create_info, None)
            .unwrap()
    };

    Arc::new(DescriptorSetLayout {
        handle: handle,
        device,
    })
}

fn new_descriptor_pool(
    device: Arc<Device>,
    max_sets: u32,
    pool_sizes: &[vk::DescriptorPoolSize],
) -> Arc<DescriptorPool> {
    let create_info = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DescriptorPoolCreateInfo,
        p_next: ptr::null(),
        flags: vk::DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        max_sets,
        pool_size_count: pool_sizes.len() as u32,
        p_pool_sizes: pool_sizes.as_ptr(),
    };

    let handle = unsafe {
        device
            .device
            .create_descriptor_pool(&create_info, None)
            .unwrap()
    };

    Arc::new(DescriptorPool { handle, device })
}

fn new_descriptor_set(
    device: Arc<Device>,
    pool: Arc<DescriptorPool>,
    layout: Arc<DescriptorSetLayout>,
) -> Arc<DescriptorSet> {
    let layouts = &[layout.handle];
    let desc_alloc_info = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DescriptorSetAllocateInfo,
        p_next: ptr::null(),
        descriptor_pool: pool.handle,
        descriptor_set_count: layouts.len() as u32,
        p_set_layouts: layouts.as_ptr(),
    };
    let mut new_descriptor_sets = unsafe {
        device
            .device
            .allocate_descriptor_sets(&desc_alloc_info)
            .unwrap()
    };
    let handle = new_descriptor_sets.remove(0);

    Arc::new(DescriptorSet {
        handle,
        pool,
        device,
    })
}

fn new_pipeline_layout(
    device: Arc<Device>,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    push_constant_ranges: Vec<vk::PushConstantRange>,
) -> Arc<PipelineLayout> {
    let create_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PipelineLayoutCreateInfo,
        p_next: ptr::null(),
        flags: Default::default(),
        set_layout_count: descriptor_set_layouts.len() as u32,
        p_set_layouts: descriptor_set_layouts.as_ptr(),
        push_constant_range_count: push_constant_ranges.len() as u32,
        p_push_constant_ranges: push_constant_ranges.as_ptr(),
    };

    let pipeline_layout = unsafe {
        device
            .device
            .create_pipeline_layout(&create_info, None)
            .unwrap()
    };

    Arc::new(PipelineLayout {
        handle: pipeline_layout,
        device,
        push_constant_ranges,
    })
}

fn new_graphics_pipeline(
    instance: Arc<Instance>,
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
    renderpass: Arc<RenderPass>,
    input_attributes: &[vk::VertexInputAttributeDescription],
    input_bindings: &[vk::VertexInputBindingDescription],
    shaders: &[(vk::ShaderStageFlags, PathBuf)],
) -> Arc<GraphicsPipeline> {
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
                    .device
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
            width: instance.window_width as f32,
            height: instance.window_height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        },
    ];
    let scissors = [
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: instance.window_width,
                height: instance.window_height,
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
        layout: pipeline_layout.handle,
        render_pass: renderpass.handle,
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: 0,
    };
    let graphics_pipelines = unsafe {
        device
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[graphic_pipeline_info], None)
            .expect("Unable to create graphics pipeline")
    };
    for (shader_module, _stage) in shader_modules {
        unsafe {
            device.device.destroy_shader_module(shader_module, None);
        }
    }

    Arc::new(GraphicsPipeline {
        handle: graphics_pipelines[0],
        device,
    })
}
