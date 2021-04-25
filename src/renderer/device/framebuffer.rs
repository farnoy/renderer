use ash::{version::DeviceV1_0, vk};

use super::Device;

pub(crate) struct Framebuffer {
    pub(crate) handle: vk::Framebuffer,
}

impl Framebuffer {
    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_framebuffer(self.handle, None);
        }
        self.handle = vk::Framebuffer::null();
    }
}

#[cfg(debug_assertions)]
impl Drop for Framebuffer {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::Framebuffer::null(),
            "Framebuffer not destroyed before Drop"
        );
    }
}
