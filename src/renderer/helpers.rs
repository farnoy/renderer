use super::device::Device;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub(crate) struct SwapchainImage {
    #[allow(unused)]
    pub(crate) handle: vk::Image,
}

pub(crate) struct Framebuffer {
    pub(crate) handle: vk::Framebuffer,
    pub(crate) device: Arc<Device>,
}

pub(crate) struct ImageView {
    pub(crate) handle: vk::ImageView,
    pub(crate) device: Arc<Device>,
}

pub(crate) struct Sampler {
    pub(crate) handle: vk::Sampler,
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

pub(crate) fn new_image_view(
    device: Arc<Device>,
    create_info: &vk::ImageViewCreateInfo,
) -> ImageView {
    let handle = unsafe { device.create_image_view(&create_info, None).unwrap() };

    ImageView { handle, device }
}

pub(crate) fn new_sampler(device: Arc<Device>, info: &vk::SamplerCreateInfoBuilder<'_>) -> Sampler {
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

pub(crate) fn pick_lod<T>(
    lods: &[T],
    camera_pos: na::Point3<f32>,
    mesh_pos: na::Point3<f32>,
) -> &T {
    let distance_from_camera = (camera_pos - mesh_pos).magnitude();
    // TODO: fine-tune this later
    if distance_from_camera > 10.0 {
        lods.last().expect("empty index buffer LODs")
    } else {
        lods.first().expect("empty index buffer LODs")
    }
}
