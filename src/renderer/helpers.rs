use super::{device::DescriptorSetLayout, swapchain};
#[cfg(windows)]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android")))]
use ash::extensions::khr::XlibSurface;
use ash::{
    extensions,
    version::{DeviceV1_0, EntryV1_0},
    vk, Instance as AshInstance,
};
use std::{ffi::CString, fs::File, io::Read, path::PathBuf, sync::Arc, u32};
#[cfg(windows)]
use winapi;
use winit;

use super::{device::Device, instance::Instance};

pub struct Swapchain {
    pub handle: swapchain::Swapchain,
    pub surface_format: vk::SurfaceFormatKHR,
}

pub struct SwapchainImage {
    pub handle: vk::Image,
}

pub struct Framebuffer {
    pub handle: vk::Framebuffer,
    pub device: Arc<Device>,
}

pub struct ImageView {
    pub handle: vk::ImageView,
    pub device: Arc<Device>,
}

pub struct Sampler {
    pub handle: vk::Sampler,
    device: Arc<Device>,
}

pub struct PipelineLayout {
    pub handle: vk::PipelineLayout,
    device: Arc<Device>,
}

pub struct Pipeline {
    pub handle: vk::Pipeline,
    device: Arc<Device>,
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_framebuffer(self.handle, None);
        }
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

pub fn new_pipeline_layout(
    device: Arc<Device>,
    descriptor_set_layouts: &[&DescriptorSetLayout],
    push_constant_ranges: &[vk::PushConstantRange],
) -> PipelineLayout {
    let descriptor_set_layout_handles = descriptor_set_layouts
        .iter()
        .map(|l| l.handle)
        .collect::<Vec<_>>();
    let create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&descriptor_set_layout_handles)
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
            let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
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
        let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
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

pub fn pick_lod<T>(lods: &[T], camera_pos: na::Point3<f32>, mesh_pos: na::Point3<f32>) -> &T {
    let distance_from_camera = (camera_pos - mesh_pos).magnitude();
    // TODO: fine-tune this later
    if distance_from_camera > 10.0 {
        lods.last().expect("empty index buffer LODs")
    } else {
        lods.first().expect("empty index buffer LODs")
    }
}
