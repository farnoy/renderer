use ash::{version::DeviceV1_0, vk};

use super::device::Device;

pub(crate) const MP_INDIAN_RED: u32 = 0xcd5c5c;

pub(crate) struct SwapchainImage {
    #[allow(unused)]
    pub(crate) handle: vk::Image,
    pub(crate) format: vk::Format,
}

pub(crate) struct Framebuffer {
    pub(crate) handle: vk::Framebuffer,
}

#[derive(Debug)]
pub(crate) struct ImageView {
    pub(crate) handle: vk::ImageView,
}

pub(crate) struct Sampler {
    pub(crate) handle: vk::Sampler,
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

impl ImageView {
    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_image_view(self.handle, None);
        }
        self.handle = vk::ImageView::null();
    }
}

#[cfg(debug_assertions)]
impl Drop for ImageView {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::ImageView::null(),
            "ImageView not destroyed before dropping"
        );
    }
}

impl Sampler {
    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_sampler(self.handle, None);
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

pub(crate) fn new_image_view(device: &Device, create_info: &vk::ImageViewCreateInfo) -> ImageView {
    let handle = unsafe { device.create_image_view(&create_info, None).unwrap() };

    ImageView { handle }
}

pub(crate) fn new_sampler(device: &Device, info: &vk::SamplerCreateInfoBuilder<'_>) -> Sampler {
    let sampler = unsafe { device.create_sampler(info, None).expect("Failed to create sampler") };

    Sampler { handle: sampler }
}

pub(crate) fn pick_lod<T>(lods: &[T], camera_pos: na::Point3<f32>, mesh_pos: na::Point3<f32>) -> &T {
    let distance_from_camera = (camera_pos - mesh_pos).magnitude();
    // TODO: fine-tune this later
    if distance_from_camera > 10.0 {
        lods.last().expect("empty index buffer LODs")
    } else {
        lods.first().expect("empty index buffer LODs")
    }
}
