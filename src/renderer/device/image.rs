use std::ptr;

use ash::vk;

use super::{alloc, mapping::MappedBuffer, Device};

pub(crate) struct Image {
    pub(crate) handle: vk::Image,
    allocation: alloc::VmaAllocation,
    allocation_info: alloc::VmaAllocationInfo,
}

impl Image {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &Device,
        format: vk::Format,
        extent: vk::Extent3D,
        samples: vk::SampleCountFlags,
        tiling: vk::ImageTiling,
        initial_layout: vk::ImageLayout,
        usage: vk::ImageUsageFlags,
        allocation_usage: alloc::VmaMemoryUsage,
    ) -> Image {
        let mut queue_family_indices = vec![
            device.graphics_queue_family,
            device.compute_queue_family,
            device.transfer_queue_family,
        ];
        queue_family_indices.sort();
        queue_family_indices.dedup();
        let sharing_mode = if queue_family_indices.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        let image_create_info = vk::ImageCreateInfo::builder()
            .format(format)
            .extent(extent)
            .samples(samples)
            .usage(usage)
            .mip_levels(1)
            .array_layers(1)
            .image_type(vk::ImageType::TYPE_2D)
            .tiling(tiling)
            .initial_layout(initial_layout)
            .sharing_mode(sharing_mode)
            .queue_family_indices(&queue_family_indices);

        let allocation_create_info = alloc::VmaAllocationCreateInfo {
            flags: alloc::VmaAllocationCreateFlagBits(0),
            memoryTypeBits: 0,
            pUserData: ptr::null_mut(),
            pool: ptr::null_mut(),
            preferredFlags: 0,
            requiredFlags: 0,
            usage: allocation_usage,
        };

        let (handle, allocation, allocation_info) =
            alloc::create_image(device.allocator, &image_create_info, &allocation_create_info).unwrap();

        Image {
            handle,
            allocation,
            allocation_info,
        }
    }

    pub(crate) fn map<'a, T>(&'a self, device: &Device) -> ash::prelude::VkResult<MappedBuffer<'a, T>> {
        MappedBuffer::import(device.allocator, self.allocation, &self.allocation_info)
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        alloc::destroy_image(device.allocator, self.handle, self.allocation);
        self.handle = vk::Image::null();
        // TODO: zero out allocation
    }
}

#[cfg(debug_assertions)]
impl Drop for Image {
    fn drop(&mut self) {
        debug_assert_eq!(self.handle, vk::Image::null(), "Image not destroyed before Drop");
    }
}
