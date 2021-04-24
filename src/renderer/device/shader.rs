use ash::{version::DeviceV1_0, vk};

use super::Device;

pub(crate) struct Shader {
    handle: vk::ShaderModule,
}

impl Shader {
    pub(super) fn new(device: &Device, bytes: &[u8]) -> Shader {
        let (l, aligned, r) = unsafe { bytes.align_to() };
        assert!(l.is_empty() && r.is_empty(), "failed to realign code");
        let shader_info = vk::ShaderModuleCreateInfo::builder().code(&aligned);
        let shader_module = unsafe {
            device
                .device
                .create_shader_module(&shader_info, None)
                .expect("shader module creation error")
        };

        Shader { handle: shader_module }
    }

    pub(crate) fn vk(&self) -> vk::ShaderModule {
        self.handle
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.handle, None);
        }
        self.handle = vk::ShaderModule::null();
    }
}

#[cfg(debug_assertions)]
impl Drop for Shader {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::ShaderModule::null(),
            "Shader not destroyed before Drop"
        );
    }
}
