use ash;
use ash::extensions::{DebugReport, Surface};
#[cfg(windows)]
use ash::extensions::Win32Surface;
#[cfg(all(unix, not(target_os = "android")))]
use ash::extensions::XlibSurface;
use ash::vk;
use ash::version;
use ash::version::EntryV1_0;
use std::ffi::CString;
use std::ops;
use std::ptr;

// just a handle to implement Drop
#[derive(Clone)]
pub struct Swapchain {
    pub ext: ash::extensions::Swapchain,
    pub swapchain: vk::SwapchainKHR,
}

impl Swapchain {
    pub fn new(ext: ash::extensions::Swapchain, swapchain: vk::SwapchainKHR) -> Swapchain {
        Swapchain {
            ext: ext,
            swapchain: swapchain,
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.ext.destroy_swapchain_khr(self.swapchain, None);
        }
    }
}
