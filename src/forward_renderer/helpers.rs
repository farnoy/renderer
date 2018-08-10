use super::{alloc, create_surface, device, entry, instance, swapchain};
use ash::{
    extensions,
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use futures::{future, prelude::*};
use std::{
    default::Default,
    ffi::CString,
    fs::File,
    io::Read,
    mem::transmute,
    path::PathBuf,
    ptr,
    sync::{Arc, Mutex},
    u32, u64,
};
use winit;

pub struct Instance {
    pub _window: Arc<winit::Window>,
    pub instance: Arc<instance::Instance>,
    pub entry: Arc<entry::Entry>,
    pub surface: vk::SurfaceKHR,
    pub window_width: u32,
    pub window_height: u32,
}

pub struct Device {
    pub device: Arc<device::Device>,
    pub physical_device: vk::PhysicalDevice,
    pub allocator: alloc::VmaAllocator,
    pub graphics_queue_family: u32,
    pub compute_queue_family: u32,
    pub graphics_queue: Arc<Mutex<vk::Queue>>,
    pub _compute_queues: Arc<Vec<Mutex<vk::Queue>>>,
    pub _transfer_queue: Arc<Mutex<vk::Queue>>,
}

pub struct Swapchain {
    pub handle: swapchain::Swapchain,
    pub surface_format: vk::SurfaceFormatKHR,
}

pub struct RenderPass {
    pub handle: vk::RenderPass,
    pub device: Arc<Device>,
}

pub struct CommandPool {
    pub handle: Mutex<vk::CommandPool>,
    pub device: Arc<Device>,
}

pub struct Framebuffer {
    pub _images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub depth_images: Vec<(vk::Image, alloc::VmaAllocation, alloc::VmaAllocationInfo)>,
    pub depth_image_views: Vec<vk::ImageView>,
    pub handles: Vec<vk::Framebuffer>,
    pub device: Arc<Device>,
}

pub struct Semaphore {
    pub handle: vk::Semaphore,
    pub device: Arc<Device>,
}

pub struct Fence {
    pub handle: vk::Fence,
    pub device: Arc<Device>,
}

pub struct CommandBuffer {
    pub handle: vk::CommandBuffer,
    pub pool: Arc<CommandPool>,
    pub device: Arc<Device>,
}

pub struct Buffer {
    pub handle: vk::Buffer,
    pub allocation: alloc::VmaAllocation,
    pub allocation_info: alloc::VmaAllocationInfo,
    pub device: Arc<Device>,
}

pub struct DescriptorPool {
    pub handle: vk::DescriptorPool,
    pub device: Arc<Device>,
}

pub struct DescriptorSetLayout {
    pub handle: vk::DescriptorSetLayout,
    pub device: Arc<Device>,
}

pub struct DescriptorSet {
    pub handle: vk::DescriptorSet,
    pub pool: Arc<DescriptorPool>,
    pub device: Arc<Device>,
}

pub struct PipelineLayout {
    pub handle: vk::PipelineLayout,
    pub device: Arc<Device>,
}

pub struct Pipeline {
    pub handle: vk::Pipeline,
    pub device: Arc<Device>,
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
            for view in self.image_views.iter().chain(self.depth_image_views.iter()) {
                self.device.device.destroy_image_view(*view, None);
            }
            for image in &self.depth_images {
                self.device.device.destroy_image(image.0, None);
            }
            for handle in &self.handles {
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
        alloc::destroy_buffer(self.device.allocator, self.handle, self.allocation)
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

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_pipeline(self.handle, None) }
    }
}

pub fn new_window(window_width: u32, window_height: u32) -> (Arc<Instance>, winit::EventsLoop) {
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

    (
        Arc::new(Instance {
            _window: Arc::new(window),
            instance,
            entry,
            surface,
            window_width,
            window_height,
        }),
        events_loop,
    )
}

pub fn new_device(instance: &Instance) -> Arc<Device> {
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
                let supports_graphic_and_surface =
                    info.queue_flags.subset(vk::QueueFlags::GRAPHICS) && surface_loader
                        .get_physical_device_surface_support_khr(pdevice, ix as u32, surface);
                if supports_graphic_and_surface {
                    Some(ix as u32)
                } else {
                    None
                }
            }).next()
            .unwrap()
    };
    let (compute_queue_family, compute_queue_len) = {
        instance
            .get_physical_device_queue_family_properties(pdevice)
            .iter()
            .enumerate()
            .filter_map(|(ix, info)| {
                if info.queue_flags.subset(vk::QueueFlags::COMPUTE)
                    && !info.queue_flags.subset(vk::QueueFlags::GRAPHICS)
                {
                    Some((ix as u32, info.queue_count))
                } else {
                    None
                }
            }).next()
    }.unwrap_or((graphics_queue_family, 1));
    let transfer_queue_family = if cfg!(feature = "validation") {
        compute_queue_family
    } else {
        instance
            .get_physical_device_queue_family_properties(pdevice)
            .iter()
            .enumerate()
            .filter_map(|(ix, info)| {
                if info.queue_flags.subset(vk::QueueFlags::TRANSFER)
                    && !info.queue_flags.subset(vk::QueueFlags::GRAPHICS)
                    && !info.queue_flags.subset(vk::QueueFlags::COMPUTE)
                {
                    Some(ix as u32)
                } else {
                    None
                }
            }).next()
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
        _compute_queues: Arc::new(compute_queues.iter().cloned().map(Mutex::new).collect()),
        _transfer_queue: Arc::new(Mutex::new(transfer_queue)),
    })
}

pub fn new_swapchain(instance: &Instance, device: &Device) -> Arc<Swapchain> {
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
    let present_mode = vk::PresentModeKHR::FIFO;
    let surface_formats = surface_loader
        .get_physical_device_surface_formats_khr(physical_device, surface)
        .unwrap();
    let surface_format = surface_formats
        .iter()
        .map(|sfmt| match sfmt.format {
            vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8_UNORM,
                color_space: sfmt.color_space,
            },
            _ => *sfmt,
        }).nth(0)
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
        u32::MAX => vk::Extent2D {
            width: window_width,
            height: window_height,
        },
        _ => surface_capabilities.current_extent,
    };
    let pre_transform = if surface_capabilities
        .supported_transforms
        .subset(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };

    let swapchain_loader =
        extensions::Swapchain::new(instance.vk(), device.vk()).expect("Unable to load swapchain");
    let swapchain_create_info = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        surface,
        min_image_count: desired_image_count,
        image_color_space: surface_format.color_space,
        image_format: surface_format.format,
        image_extent: surface_resolution,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        pre_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
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
        vk::ObjectType::SURFACE_KHR,
        unsafe { transmute::<_, u64>(swapchain) },
        "Window surface",
    );

    let swapchain = swapchain::Swapchain::new(swapchain_loader, swapchain);
    Arc::new(Swapchain {
        handle: swapchain,
        surface_format,
    })
}

pub fn new_renderpass(
    device: Arc<Device>,
    attachments: &[vk::AttachmentDescription],
    subpass_descs: &[vk::SubpassDescription],
    subpass_dependencies: &[vk::SubpassDependency],
) -> Arc<RenderPass> {
    let renderpass_create_info = vk::RenderPassCreateInfo {
        s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
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

pub fn new_command_pool(
    device: Arc<Device>,
    queue_family: u32,
    flags: vk::CommandPoolCreateFlags,
) -> Arc<CommandPool> {
    let pool_create_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags,
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

pub fn setup_framebuffer(
    instance: &Instance,
    device: Arc<Device>,
    swapchain: &Swapchain,
    renderpass: &RenderPass,
) -> Arc<Framebuffer> {
    let Swapchain {
        handle: ref swapchain,
        ref surface_format,
        ..
    } = *swapchain;
    let Instance {
        window_width,
        window_height,
        ..
    } = *instance;

    let images = swapchain
        .ext
        .get_swapchain_images_khr(swapchain.swapchain)
        .unwrap();
    println!("swapchain images len {}", images.len());
    let depth_images = (0..images.len())
        .map(|_| {
            alloc::create_image(
                device.allocator,
                &vk::ImageCreateInfo {
                    s_type: vk::StructureType::IMAGE_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: Default::default(),
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::D16_UNORM,
                    extent: vk::Extent3D {
                        width: instance.window_width,
                        height: instance.window_height,
                        depth: 1,
                    },
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    queue_family_index_count: 1,
                    p_queue_family_indices: &device.graphics_queue_family,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                },
                &alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: ptr::null_mut(),
                    pool: ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
            ).unwrap()
        }).collect::<Vec<_>>();
    let image_views = images
        .iter()
        .map(|&image| {
            let create_view_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: Default::default(),
                view_type: vk::ImageViewType::TYPE_2D,
                format: surface_format.format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image,
            };
            unsafe {
                device
                    .device
                    .create_image_view(&create_view_info, None)
                    .unwrap()
            }
        }).collect::<Vec<_>>();
    let depth_image_views = depth_images
        .iter()
        .map(|ref image| {
            let create_view_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: Default::default(),
                view_type: vk::ImageViewType::TYPE_2D,
                format: vk::Format::D16_UNORM,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image: image.0,
            };
            unsafe {
                device
                    .device
                    .create_image_view(&create_view_info, None)
                    .unwrap()
            }
        }).collect::<Vec<_>>();
    let handles = image_views
        .iter()
        .zip(depth_image_views.iter())
        .map(|(&present_image_view, &depth_image_view)| {
            let framebuffer_attachments = [present_image_view, depth_image_view];
            let frame_buffer_create_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
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
        }).collect::<Vec<_>>();

    let framebuffer = Framebuffer {
        _images: images,
        image_views,
        depth_images,
        depth_image_views,
        handles,
        device,
    };

    Arc::new(framebuffer)
}

pub fn new_semaphore(device: Arc<Device>) -> Arc<Semaphore> {
    let create_info = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SemaphoreCreateFlags::empty(),
    };
    let semaphore = unsafe { device.device.create_semaphore(&create_info, None).unwrap() };

    Arc::new(Semaphore {
        handle: semaphore,
        device,
    })
}

pub fn _allocate_command_buffer(pool: Arc<CommandPool>) -> Arc<CommandBuffer> {
    let command_buffers = unsafe {
        let pool_lock = pool.handle.lock().unwrap();
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: 1,
            command_pool: *pool_lock,
            level: vk::CommandBufferLevel::PRIMARY,
        };
        pool.device
            .device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()
    };
    let device = Arc::clone(&pool.device);

    Arc::new(CommandBuffer {
        handle: command_buffers[0],
        pool,
        device,
    })
}

pub fn new_fence(device: Arc<Device>) -> Arc<Fence> {
    let create_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
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

pub fn new_buffer(
    device: Arc<Device>,
    buffer_usage: vk::BufferUsageFlags,
    allocation_flags: alloc::VmaAllocationCreateFlagBits,
    allocation_usage: alloc::VmaMemoryUsage,
    size: vk::DeviceSize,
) -> Arc<Buffer> {
    let queue_families = [device.graphics_queue_family, device.compute_queue_family];
    let buffer_create_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        size,
        usage: buffer_usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
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

pub fn new_descriptor_set_layout(
    device: Arc<Device>,
    bindings: &[vk::DescriptorSetLayoutBinding],
) -> Arc<DescriptorSetLayout> {
    let create_info = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
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

    Arc::new(DescriptorSetLayout { handle, device })
}

pub fn new_descriptor_pool(
    device: Arc<Device>,
    max_sets: u32,
    pool_sizes: &[vk::DescriptorPoolSize],
) -> Arc<DescriptorPool> {
    let create_info = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
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

pub fn new_descriptor_set(
    device: Arc<Device>,
    pool: Arc<DescriptorPool>,
    layout: &DescriptorSetLayout,
) -> Arc<DescriptorSet> {
    let layouts = &[layout.handle];
    let desc_alloc_info = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
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

pub fn new_pipeline_layout(
    device: Arc<Device>,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    push_constant_ranges: &[vk::PushConstantRange],
) -> Arc<PipelineLayout> {
    let create_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
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
    })
}

pub fn new_graphics_pipeline(
    instance: &Instance,
    device: Arc<Device>,
    pipeline_layout: &PipelineLayout,
    renderpass: &RenderPass,
    input_attributes: &[vk::VertexInputAttributeDescription],
    input_bindings: &[vk::VertexInputBindingDescription],
    shaders: &[(vk::ShaderStageFlags, PathBuf)],
    subpass: u32,
    rasterizer_discard: bool,
    depth_write: bool,
) -> Arc<Pipeline> {
    let shader_modules = shaders
        .iter()
        .map(|&(stage, ref path)| {
            let file = File::open(path).expect("Could not find shader.");
            let bytes: Vec<u8> = file.bytes().filter_map(|byte| byte.ok()).collect();
            let shader_info = vk::ShaderModuleCreateInfo {
                s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
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
        }).collect::<Vec<_>>();
    let shader_entry_name = CString::new("main").unwrap();
    let shader_stage_create_infos = shader_modules
        .iter()
        .map(|&(module, stage)| vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            module,
            p_name: shader_entry_name.as_ptr(),
            p_specialization_info: ptr::null(),
            stage,
        }).collect::<Vec<_>>();
    let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        vertex_attribute_description_count: input_attributes.len() as u32,
        p_vertex_attribute_descriptions: input_attributes.as_ptr(),
        vertex_binding_description_count: input_bindings.len() as u32,
        p_vertex_binding_descriptions: input_bindings.as_ptr(),
    };
    let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        flags: Default::default(),
        p_next: ptr::null(),
        primitive_restart_enable: 0,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
    };
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: instance.window_width as f32,
        height: instance.window_height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: instance.window_width,
            height: instance.window_height,
        },
    }];
    let viewport_state_info = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        scissor_count: scissors.len() as u32,
        p_scissors: scissors.as_ptr(),
        viewport_count: viewports.len() as u32,
        p_viewports: viewports.as_ptr(),
    };
    /*
    let raster_order_amd = vk::PipelineRasterizationStateRasterizationOrderAMD {
        s_type: vk::StructureType::PipelineRasterizationStateRasterizationOrderAMD,
        p_next: ptr::null(),
        rasterization_order: vk::RasterizationOrderAMD::Relaxed,
    };
    */
    let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: ptr::null(), // unsafe { transmute(&raster_order_amd) },
        flags: Default::default(),
        cull_mode: vk::CullModeFlags::BACK,
        depth_bias_clamp: 0.0,
        depth_bias_constant_factor: 0.0,
        depth_bias_enable: 0,
        depth_bias_slope_factor: 0.0,
        depth_clamp_enable: 0,
        front_face: vk::FrontFace::CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        rasterizer_discard_enable: if rasterizer_discard { 1 } else { 0 },
    };
    let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        flags: Default::default(),
        p_next: ptr::null(),
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: 0,
        min_sample_shading: 0.0,
        p_sample_mask: ptr::null(),
        alpha_to_one_enable: 0,
        alpha_to_coverage_enable: 0,
    };
    let noop_stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        compare_mask: 0,
        write_mask: 0,
        reference: 0,
    };
    let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        depth_test_enable: 1,
        depth_write_enable: if depth_write { 1 } else { 0 },
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        depth_bounds_test_enable: 1,
        stencil_test_enable: 0,
        front: noop_stencil_state,
        back: noop_stencil_state,
        max_depth_bounds: 1.0,
        min_depth_bounds: 0.0,
    };
    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
        blend_enable: 0,
        src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ZERO,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::all(),
    }];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        logic_op_enable: 0,
        logic_op: vk::LogicOp::CLEAR,
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
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
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
        subpass,
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

    Arc::new(Pipeline {
        handle: graphics_pipelines[0],
        device,
    })
}

pub fn new_compute_pipeline(
    device: Arc<Device>,
    pipeline_layout: &PipelineLayout,
    shader: &PathBuf,
) -> Arc<Pipeline> {
    let shader_module = {
        let file = File::open(shader).expect("Could not find shader.");
        let bytes: Vec<u8> = file.bytes().filter_map(|byte| byte.ok()).collect();
        let shader_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            code_size: bytes.len(),
            p_code: bytes.as_ptr() as *const u32,
        };
        unsafe {
            device
                .device
                .create_shader_module(&shader_info, None)
                .expect("Vertex shader module error")
        }
    };
    let shader_entry_name = CString::new("main").unwrap();
    let shader_stage = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        module: shader_module,
        p_name: shader_entry_name.as_ptr(),
        p_specialization_info: ptr::null(),
        stage: vk::ShaderStageFlags::COMPUTE,
    };
    let create_info = vk::ComputePipelineCreateInfo {
        s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineCreateFlags::empty(),
        stage: shader_stage,
        layout: pipeline_layout.handle,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: 0,
    };

    let pipelines = unsafe {
        device
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .unwrap()
    };

    unsafe {
        device.device.destroy_shader_module(shader_module, None);
    }

    Arc::new(Pipeline {
        handle: pipelines[0],
        device,
    })
}

pub fn record_one_time_cb<F: FnOnce(vk::CommandBuffer)>(
    command_pool: Arc<CommandPool>,
    f: F,
) -> impl Future<Item = CommandBuffer, Error = Never> {
    future::lazy(move |_| unsafe {
        let command_buffer = {
            let pool_lock = command_pool.handle.lock().unwrap();
            let command_buffer = {
                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                    s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                    p_next: ptr::null(),
                    command_buffer_count: 1,
                    command_pool: *pool_lock,
                    level: vk::CommandBufferLevel::PRIMARY,
                };
                command_pool
                    .device
                    .device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .unwrap()[0]
            };

            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                p_inheritance_info: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            };
            command_pool
                .device
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();

            f(command_buffer);

            command_pool
                .device
                .device
                .end_command_buffer(command_buffer)
                .unwrap();

            command_buffer
        };

        Ok(CommandBuffer {
            device: Arc::clone(&command_pool.device),
            pool: command_pool,
            handle: command_buffer,
        })
    })
}

pub fn one_time_submit_cb<F: FnOnce(vk::CommandBuffer)>(
    command_pool: Arc<CommandPool>,
    queue: Arc<Mutex<vk::Queue>>,
    f: F,
) -> impl Future<Item = (), Error = Never> {
    record_one_time_cb(Arc::clone(&command_pool), f).and_then(move |command_buffer| {
        let submit_fence = new_fence(Arc::clone(&command_pool.device));

        unsafe {
            let submits = [vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &command_buffer.handle,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
            }];

            let queue_lock = queue.lock().unwrap();

            command_pool
                .device
                .device
                .queue_submit(*queue_lock, &submits, submit_fence.handle)
                .unwrap();
        }

        unsafe {
            command_pool
                .device
                .device
                .wait_for_fences(&[submit_fence.handle], true, u64::MAX)
                .expect("Wait for fence failed.");
        }
        Ok(())
    })
}
