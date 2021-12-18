use std::{sync::Arc, u32};

#[cfg(windows)]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android")))]
use ash::extensions::khr::XlibSurface;
use ash::{self, extensions, vk};

use super::{Device, Instance};

pub(crate) struct Surface {
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) ext: extensions::khr::Surface,
    pub(crate) surface_format: vk::SurfaceFormatKHR,
    _instance: Arc<Instance>, // destructor ordering
}

impl Surface {
    pub(crate) fn new(instance: &Arc<Instance>) -> Surface {
        let surface = unsafe { create_surface(instance.entry.vk(), instance.vk(), &instance.window).unwrap() };

        let pdevices = unsafe { instance.enumerate_physical_devices().expect("Physical device error") };
        let physical_device = pdevices[0];

        let surface_loader = extensions::khr::Surface::new(instance.entry.vk(), instance.vk());
        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };
        let desired_format = vk::SurfaceFormatKHR {
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            format: vk::Format::R8G8B8A8_UNORM,
        };
        let surface_format = surface_formats
            .iter()
            .cloned()
            .find(|&sfmt| sfmt == desired_format)
            .unwrap_or_else(|| {
                surface_formats
                    .first()
                    .cloned()
                    .expect("Unable to find suitable surface format.")
            });

        Surface {
            surface,
            ext: surface_loader,
            surface_format,
            _instance: Arc::clone(instance),
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.ext.destroy_surface(self.surface, None) }
    }
}

pub(crate) struct Swapchain {
    pub(crate) ext: ash::extensions::khr::Swapchain,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) surface: Surface,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) desired_image_count: u32,
    pub(crate) supports_present_from_compute: bool,
}

impl Swapchain {
    pub(crate) fn new(instance: &Arc<Instance>, device: &Device, surface: Surface) -> Swapchain {
        let supports_present_from_compute = unsafe {
            surface
                .ext
                .get_physical_device_surface_support(
                    device.physical_device,
                    device.compute_queue_family,
                    surface.surface,
                )
                .unwrap()
        };
        let surface_capabilities = unsafe {
            surface
                .ext
                .get_physical_device_surface_capabilities(device.physical_device, surface.surface)
                .unwrap()
        };
        println!("new swapchain surface capabilities {:?}", surface_capabilities);
        let desired_image_count = na::clamp(
            if cfg!(feature = "uncapped") {
                999
            } else {
                // TODO: This is likely also affected by https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/3590
                // It's because of setup_submissions() that pulls signals forward for inactive stages, crossing the
                // queue boundary potentially
                1
            },
            surface_capabilities.min_image_count,
            surface_capabilities.max_image_count,
        );
        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => panic!("not expecting u32::MAX for surface capabilities current_extent"),
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
            .surface(surface.surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface.surface_format.color_space)
            .image_format(surface.surface_format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(if cfg!(feature = "uncapped") {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            })
            .clipped(true)
            .image_array_layers(1);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None).unwrap() };

        device.set_object_name(swapchain, "Window surface");

        Swapchain {
            swapchain,
            ext: swapchain_loader,
            surface,
            width: surface_resolution.width,
            height: surface_resolution.height,
            desired_image_count,
            supports_present_from_compute,
        }
    }

    pub(crate) fn resize_to_fit(&mut self, device: &Device) {
        let surface_capabilities = unsafe {
            self.surface
                .ext
                .get_physical_device_surface_capabilities(device.physical_device, self.surface.surface)
                .unwrap()
        };
        println!("resizing surface capabilities {:?}", surface_capabilities);
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface.surface)
            .min_image_count(self.desired_image_count)
            .image_color_space(self.surface.surface_format.color_space)
            .image_format(self.surface.surface_format.format)
            .image_extent(surface_capabilities.current_extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(if cfg!(feature = "uncapped") {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            })
            .clipped(true)
            .old_swapchain(self.swapchain)
            .image_array_layers(1);
        unsafe {
            let new_swapchain = self.ext.create_swapchain(&swapchain_create_info, None).unwrap();
            self.ext.destroy_swapchain(self.swapchain, None);
            self.swapchain = new_swapchain;
        };

        self.width = surface_capabilities.current_extent.width;
        self.height = surface_capabilities.current_extent.height;
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.ext.destroy_swapchain(self.swapchain, None) }
    }
}

#[cfg(all(unix, not(target_os = "android")))]
unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use winit::platform::unix::WindowExtUnix;
    let x11_display = window.xlib_display().unwrap();
    let x11_window = window.xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .window(x11_window as vk::Window)
        .dpy(x11_display as *mut vk::Display);
    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

#[cfg(windows)]
unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::ffi::c_void;

    use winit::platform::windows::WindowExtWindows;
    let hwnd = window.hwnd() as *mut winapi::shared::windef::HWND__;
    let hinstance = winapi::um::winuser::GetWindow(hwnd, 0) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
        .hinstance(hinstance)
        .hwnd(hwnd as *const c_void);
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}
