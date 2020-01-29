use super::{Device, Instance};
#[cfg(windows)]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android")))]
use ash::extensions::khr::XlibSurface;
use ash::{
    self, extensions,
    version::{EntryV1_0, InstanceV1_0},
    vk,
};
use std::{sync::Arc, u32};

pub struct Surface {
    pub surface: vk::SurfaceKHR,
    pub ext: extensions::khr::Surface,
    pub surface_format: vk::SurfaceFormatKHR,
    _instance: Arc<Instance>, // destructor ordering
}

impl Surface {
    pub fn new(instance: &Arc<Instance>) -> Surface {
        let surface = unsafe {
            create_surface(instance.entry.vk(), &instance.vk(), &instance.window).unwrap()
        };

        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Physical device error")
        };
        let physical_device = pdevices[0];

        let surface_loader = extensions::khr::Surface::new(instance.entry.vk(), instance.vk());
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

        Surface {
            surface,
            ext: surface_loader,
            surface_format,
            _instance: Arc::clone(&instance),
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.ext.destroy_surface(self.surface, None) }
    }
}

pub struct Swapchain {
    pub ext: ash::extensions::khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub surface: Surface,
    pub width: u32,
    pub height: u32,
    device: Arc<Device>, // destructor ordering
}

impl Swapchain {
    pub fn new(instance: &Arc<Instance>, device: &Arc<Device>, surface: Surface) -> Swapchain {
        let surface_capabilities = unsafe {
            surface
                .ext
                .get_physical_device_surface_capabilities(device.physical_device, surface.surface)
                .unwrap()
        };
        let desired_image_count =
            na::clamp(2, surface_capabilities.min_image_count, surface_capabilities.max_image_count);
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
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .image_array_layers(1);
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap()
        };

        device.set_object_name(swapchain, "Window surface");

        Swapchain {
            swapchain,
            ext: swapchain_loader,
            surface,
            device: Arc::clone(&device),
            width: surface_resolution.width,
            height: surface_resolution.height,
        }
    }

    pub fn resize_to_fit(&mut self) {
        let surface_capabilities = unsafe {
            self.surface
                .ext
                .get_physical_device_surface_capabilities(
                    self.device.physical_device,
                    self.surface.surface,
                )
                .unwrap()
        };
        println!("resizing surface capabilities {:?}", surface_capabilities);
        let desired_image_count =
            na::clamp(2, surface_capabilities.min_image_count, surface_capabilities.max_image_count);
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
            .min_image_count(desired_image_count)
            .image_color_space(self.surface.surface_format.color_space)
            .image_format(self.surface.surface_format.format)
            .image_extent(surface_capabilities.current_extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(self.swapchain)
            .image_array_layers(1);
        unsafe {
            let new_swapchain = self
                .ext
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();
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
unsafe fn create_surface<E: EntryV1_0>(
    entry: &E,
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
unsafe fn create_surface<E: EntryV1_0>(
    entry: &E,
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
