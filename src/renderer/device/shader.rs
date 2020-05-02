use ash::{version::DeviceV1_0, vk};
use std::{fs::File, io::Read, path::PathBuf, sync::Arc};

use super::Device;

pub struct Shader {
    handle: vk::ShaderModule,
    device: Arc<Device>,
}

impl Shader {
    pub(super) fn new_verify<F>(device: &Arc<Device>, path: &PathBuf, verify: F) -> Shader
    where
        F: Fn(&[u8]) -> bool,
    {
        let file = File::open(path).expect("Could not find shader.");
        let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
        assert!(verify(&bytes));
        let (l, aligned, r) = unsafe { bytes.as_slice().align_to() };
        assert!(l.is_empty() && r.is_empty(), "failed to realign code");
        let shader_info = vk::ShaderModuleCreateInfo::builder().code(&aligned);
        let shader_module = unsafe {
            device
                .device
                .create_shader_module(&shader_info, None)
                .expect("shader module creation error")
        };

        Shader {
            handle: shader_module,
            device: Arc::clone(device),
        }
    }

    pub fn vk(&self) -> vk::ShaderModule {
        self.handle
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.handle, None);
        }
    }
}
