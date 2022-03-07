use ash::vk;

use super::Device;

pub(crate) struct Sampler {
    pub(crate) handle: vk::Sampler,
}

impl Sampler {
    pub(super) fn new(device: &Device, info: &vk::SamplerCreateInfo) -> Sampler {
        let sampler = unsafe {
            device
                .create_sampler(info, device.instance.allocation_callbacks())
                .expect("Failed to create sampler")
        };

        Sampler { handle: sampler }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_sampler(self.handle, device.instance.allocation_callbacks());
        }
        self.handle = vk::Sampler::null();
    }
}

#[cfg(debug_assertions)]
impl Drop for Sampler {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::Sampler::null(),
            "Sampler not destroyed before dropping"
        );
    }
}
