use ash;
use ash::vk;

// just a handle to implement Drop
pub struct Swapchain {
    pub ext: ash::extensions::Swapchain,
    pub swapchain: vk::SwapchainKHR,
}

impl Swapchain {
    pub fn new(ext: ash::extensions::Swapchain, swapchain: vk::SwapchainKHR) -> Swapchain {
        Swapchain { ext, swapchain }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.ext.destroy_swapchain_khr(self.swapchain, None);
        }
    }
}
