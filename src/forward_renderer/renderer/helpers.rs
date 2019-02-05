use super::{alloc, swapchain};
#[cfg(windows)]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android")))]
use ash::extensions::khr::XlibSurface;
use ash::{
    extensions,
    version::{DeviceV1_0, EntryV1_0},
    vk, Instance as AshInstance,
};
use std::{ffi::CString, fs::File, io::Read, path::PathBuf, ptr, sync::Arc, u32};
#[cfg(windows)]
use winapi;
use winit;

use super::{device::Device, instance::Instance};

pub struct Swapchain {
    pub handle: swapchain::Swapchain,
    pub surface_format: vk::SurfaceFormatKHR,
}

pub struct RenderPass {
    pub handle: vk::RenderPass,
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

pub struct Buffer {
    pub handle: vk::Buffer,
    pub allocation: alloc::VmaAllocation,
    pub allocation_info: alloc::VmaAllocationInfo,
    pub device: Arc<Device>,
}

pub struct Image {
    pub handle: vk::Image,
    pub allocation: alloc::VmaAllocation,
    pub allocation_info: alloc::VmaAllocationInfo,
    pub device: Arc<Device>,
}

pub struct ImageView {
    pub handle: vk::ImageView,
    pub device: Arc<Device>,
}

pub struct Sampler {
    pub handle: vk::Sampler,
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

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_render_pass(self.handle, None) }
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

impl Drop for Buffer {
    fn drop(&mut self) {
        alloc::destroy_buffer(self.device.allocator, self.handle, self.allocation)
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        alloc::destroy_image(self.device.allocator, self.handle, self.allocation)
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.handle, None);
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_sampler(self.handle, None);
        }
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

pub fn new_swapchain(instance: &Instance, device: &Device) -> Swapchain {
    let Instance {
        ref entry,
        surface,
        window_width,
        window_height,
        ..
    } = *instance;
    let Device {
        physical_device, ..
    } = *device;

    let surface_loader = extensions::khr::Surface::new(entry.vk(), instance.vk());
    let present_mode = vk::PresentModeKHR::FIFO;
    let surface_formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .unwrap()
    };
    let surface_format = surface_formats
        .iter()
        .map(|sfmt| match sfmt.format {
            vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8_UNORM,
                color_space: sfmt.color_space,
            },
            _ => *sfmt,
        })
        .nth(0)
        .expect("Unable to find suitable surface format.");
    let surface_capabilities = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .unwrap()
    };
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
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };

    let swapchain_loader = extensions::khr::Swapchain::new(instance.vk(), device.vk());
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(desired_image_count)
        .image_color_space(surface_format.color_space)
        .image_format(surface_format.format)
        .image_extent(surface_resolution)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1);
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .unwrap()
    };

    device.set_object_name(swapchain, "Window surface");

    let swapchain = swapchain::Swapchain::new(swapchain_loader, swapchain);
    Swapchain {
        handle: swapchain,
        surface_format,
    }
}

pub fn setup_framebuffer(
    instance: &Instance,
    device: Arc<Device>,
    swapchain: &Swapchain,
    renderpass: &RenderPass,
) -> Framebuffer {
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

    let images = unsafe {
        swapchain
            .ext
            .get_swapchain_images(swapchain.swapchain)
            .unwrap()
    };
    println!("swapchain images len {}", images.len());
    let depth_images = (0..images.len())
        .map(|_| {
            alloc::create_image(
                device.allocator,
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::D16_UNORM)
                    .extent(vk::Extent3D {
                        width: instance.window_width,
                        height: instance.window_height,
                        depth: 1,
                    })
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .queue_family_indices(&[device.graphics_queue_family])
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
                &alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: ptr::null_mut(),
                    pool: ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let image_views = images
        .iter()
        .map(|&image| {
            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image);
            unsafe {
                device
                    .device
                    .create_image_view(&create_view_info, None)
                    .unwrap()
            }
        })
        .collect::<Vec<_>>();
    let depth_image_views = depth_images
        .iter()
        .map(|ref image| {
            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image.0);
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
        .zip(depth_image_views.iter())
        .map(|(&present_image_view, &depth_image_view)| {
            let framebuffer_attachments = [present_image_view, depth_image_view];
            let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass.handle)
                .attachments(&framebuffer_attachments)
                .width(window_width)
                .height(window_height)
                .layers(1);
            unsafe {
                device
                    .device
                    .create_framebuffer(&frame_buffer_create_info, None)
                    .unwrap()
            }
        })
        .collect::<Vec<_>>();

    Framebuffer {
        _images: images,
        image_views,
        depth_images,
        depth_image_views,
        handles,
        device,
    }
}

pub fn new_semaphore(device: Arc<Device>) -> Semaphore {
    let create_info = vk::SemaphoreCreateInfo::builder();
    let semaphore = unsafe { device.device.create_semaphore(&create_info, None).unwrap() };

    Semaphore {
        handle: semaphore,
        device,
    }
}

pub fn new_fence(device: Arc<Device>) -> Fence {
    let create_info = vk::FenceCreateInfo::builder();
    let fence = unsafe {
        device
            .device
            .create_fence(&create_info, None)
            .expect("Create fence failed.")
    };
    Fence {
        device,
        handle: fence,
    }
}

pub fn new_buffer(
    device: Arc<Device>,
    buffer_usage: vk::BufferUsageFlags,
    allocation_flags: alloc::VmaAllocationCreateFlagBits,
    allocation_usage: alloc::VmaMemoryUsage,
    size: vk::DeviceSize,
) -> Buffer {
    let (queue_family_indices, sharing_mode) =
        if device.compute_queue_family != device.graphics_queue_family {
            (
                vec![device.graphics_queue_family, device.compute_queue_family],
                vk::SharingMode::CONCURRENT,
            )
        } else {
            (
                vec![device.graphics_queue_family],
                vk::SharingMode::EXCLUSIVE,
            )
        };
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(buffer_usage)
        .sharing_mode(sharing_mode)
        .queue_family_indices(&queue_family_indices);

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
    )
    .unwrap();

    Buffer {
        handle,
        allocation,
        allocation_info,
        device,
    }
}

pub fn new_image(
    device: Arc<Device>,
    format: vk::Format,
    extent: vk::Extent3D,
    samples: vk::SampleCountFlags,
    usage: vk::ImageUsageFlags,
    allocation_flags: alloc::VmaAllocationCreateFlagBits,
    allocation_usage: alloc::VmaMemoryUsage,
) -> Image {
    let (queue_family_indices, sharing_mode) =
        if device.compute_queue_family != device.graphics_queue_family {
            (
                vec![device.graphics_queue_family, device.compute_queue_family],
                vk::SharingMode::CONCURRENT,
            )
        } else {
            (
                vec![device.graphics_queue_family],
                vk::SharingMode::EXCLUSIVE,
            )
        };
    let image_create_info = vk::ImageCreateInfo::builder()
        .format(format)
        .extent(extent)
        .samples(samples)
        .usage(usage)
        .mip_levels(1)
        .array_layers(1)
        .image_type(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::LINEAR)
        .initial_layout(vk::ImageLayout::PREINITIALIZED)
        .sharing_mode(sharing_mode)
        .queue_family_indices(&queue_family_indices);

    let allocation_create_info = alloc::VmaAllocationCreateInfo {
        flags: allocation_flags,
        memoryTypeBits: 0,
        pUserData: ptr::null_mut(),
        pool: ptr::null_mut(),
        preferredFlags: 0,
        requiredFlags: 0,
        usage: allocation_usage,
    };

    let (handle, allocation, allocation_info) = alloc::create_image(
        device.allocator,
        &image_create_info,
        &allocation_create_info,
    )
    .unwrap();

    Image {
        handle,
        allocation,
        allocation_info,
        device,
    }
}

pub fn new_image_view(device: Arc<Device>, create_info: &vk::ImageViewCreateInfo) -> ImageView {
    let handle = unsafe { device.create_image_view(&create_info, None).unwrap() };

    ImageView { handle, device }
}

pub fn new_sampler(device: Arc<Device>, info: &vk::SamplerCreateInfoBuilder<'_>) -> Sampler {
    let sampler = unsafe {
        device
            .create_sampler(info, None)
            .expect("Failed to create sampler")
    };

    Sampler {
        handle: sampler,
        device,
    }
}

pub fn new_descriptor_set_layout(
    device: Arc<Device>,
    bindings: &[vk::DescriptorSetLayoutBinding],
) -> DescriptorSetLayout {
    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

    let handle = unsafe {
        device
            .device
            .create_descriptor_set_layout(&create_info, None)
            .unwrap()
    };

    DescriptorSetLayout { handle, device }
}

pub fn new_descriptor_pool(
    device: Arc<Device>,
    max_sets: u32,
    pool_sizes: &[vk::DescriptorPoolSize],
) -> DescriptorPool {
    let create_info = vk::DescriptorPoolCreateInfo::builder()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(max_sets)
        .pool_sizes(pool_sizes);

    let handle = unsafe {
        device
            .device
            .create_descriptor_pool(&create_info, None)
            .unwrap()
    };

    DescriptorPool { handle, device }
}

pub fn new_descriptor_set(
    device: Arc<Device>,
    pool: Arc<DescriptorPool>,
    layout: &DescriptorSetLayout,
) -> DescriptorSet {
    let layouts = &[layout.handle];
    let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool.handle)
        .set_layouts(layouts);

    let mut new_descriptor_sets = unsafe {
        device
            .device
            .allocate_descriptor_sets(&desc_alloc_info)
            .unwrap()
    };
    let handle = new_descriptor_sets.remove(0);

    DescriptorSet {
        handle,
        pool,
        device,
    }
}

pub fn new_pipeline_layout(
    device: Arc<Device>,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    push_constant_ranges: &[vk::PushConstantRange],
) -> PipelineLayout {
    let create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(descriptor_set_layouts)
        .push_constant_ranges(push_constant_ranges);

    let pipeline_layout = unsafe {
        device
            .device
            .create_pipeline_layout(&create_info, None)
            .unwrap()
    };

    PipelineLayout {
        handle: pipeline_layout,
        device,
    }
}

pub fn new_graphics_pipeline2(
    device: Arc<Device>,
    shaders: &[(vk::ShaderStageFlags, PathBuf)],
    mut create_info: vk::GraphicsPipelineCreateInfo,
) -> Pipeline {
    let shader_modules = shaders
        .iter()
        .map(|&(stage, ref path)| {
            let file = File::open(path).expect("Could not find shader.");
            let bytes: Vec<u8> = file.bytes().filter_map(|byte| byte.ok()).collect();
            let (l, aligned, r) = unsafe { bytes.as_slice().align_to() };
            assert!(l.is_empty() && r.is_empty(), "failed to realign code");
            let shader_info = vk::ShaderModuleCreateInfo::builder().code(&aligned);
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
        .map(|&(module, stage)| {
            vk::PipelineShaderStageCreateInfo::builder()
                .module(module)
                .name(&shader_entry_name)
                .stage(stage)
                .build()
        })
        .collect::<Vec<_>>();
    create_info.stage_count = shader_stage_create_infos.len() as u32;
    create_info.p_stages = shader_stage_create_infos.as_ptr();
    let graphics_pipelines = unsafe {
        device
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .expect("Unable to create graphics pipeline")
    };
    for (shader_module, _stage) in shader_modules {
        unsafe {
            device.device.destroy_shader_module(shader_module, None);
        }
    }

    Pipeline {
        handle: graphics_pipelines[0],
        device,
    }
}

pub fn new_compute_pipeline(
    device: Arc<Device>,
    pipeline_layout: &PipelineLayout,
    shader: &PathBuf,
) -> Pipeline {
    let shader_module = {
        let file = File::open(shader).expect("Could not find shader.");
        let bytes: Vec<u8> = file.bytes().filter_map(|byte| byte.ok()).collect();
        let (l, aligned, r) = unsafe { bytes.as_slice().align_to() };
        assert!(l.is_empty() && r.is_empty(), "failed to realign code");
        let shader_info = vk::ShaderModuleCreateInfo::builder().code(&aligned);
        unsafe {
            device
                .device
                .create_shader_module(&shader_info, None)
                .expect("Vertex shader module error")
        }
    };
    let shader_entry_name = CString::new("main").unwrap();
    let shader_stage = vk::PipelineShaderStageCreateInfo::builder()
        .module(shader_module)
        .name(&shader_entry_name)
        .stage(vk::ShaderStageFlags::COMPUTE)
        .build();
    let create_info = vk::ComputePipelineCreateInfo::builder()
        .stage(shader_stage)
        .layout(pipeline_layout.handle)
        .build();

    let pipelines = unsafe {
        device
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .unwrap()
    };

    unsafe {
        device.device.destroy_shader_module(shader_module, None);
    }

    Pipeline {
        handle: pipelines[0],
        device,
    }
}

#[cfg(all(unix, not(target_os = "android")))]
pub unsafe fn create_surface<E: EntryV1_0>(
    entry: &E,
    instance: &AshInstance,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use winit::os::unix::WindowExt;
    let x11_display = window.get_xlib_display().unwrap();
    let x11_window = window.get_xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .window(x11_window as vk::Window)
        .dpy(x11_display as *mut vk::Display);
    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

#[cfg(windows)]
pub unsafe fn create_surface<E: EntryV1_0>(
    entry: &E,
    instance: &AshInstance,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::ffi::c_void;
    use winit::os::windows::WindowExt;
    let hwnd = window.get_hwnd() as *mut winapi::shared::windef::HWND__;
    let hinstance = winapi::um::winuser::GetWindow(hwnd, 0) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
        .hinstance(hinstance)
        .hwnd(hwnd as *const c_void);
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}
