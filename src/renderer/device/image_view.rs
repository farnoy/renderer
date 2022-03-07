use ash::vk;

use super::Device;

pub(crate) struct ImageView {
    pub(crate) handle: vk::ImageView,
}
impl ImageView {
    pub(super) fn new(device: &Device, create_info: &vk::ImageViewCreateInfo) -> ImageView {
        let handle = unsafe {
            device
                .create_image_view(create_info, device.instance.allocation_callbacks())
                .unwrap()
        };

        ImageView { handle }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_image_view(self.handle, device.instance.allocation_callbacks());
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
