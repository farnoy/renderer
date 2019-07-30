use ash::vk;
use std::{ptr, sync::Arc};

use super::{super::alloc, mapping::MappedBuffer, Device};

pub struct Image {
    pub handle: vk::Image,
    allocation: alloc::VmaAllocation,
    allocation_info: alloc::VmaAllocationInfo,
    device: Arc<Device>,
}

impl Image {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &Arc<Device>,
        format: vk::Format,
        extent: vk::Extent3D,
        samples: vk::SampleCountFlags,
        tiling: vk::ImageTiling,
        initial_layout: vk::ImageLayout,
        usage: vk::ImageUsageFlags,
        allocation_usage: alloc::VmaMemoryUsage,
    ) -> Image {
        let (queue_family_indices, sharing_mode) =
            if device.compute_queue_family != device.graphics_queue_family {
                (
                    vec![device.graphics_queue_family, device.compute_queue_family],
                    vk::SharingMode::CONCURRENT,
                )
            } else {
                (
                    vec![device.graphics_queue_family],
                    vk::SharingMode::EXCLUSIVE,
                )
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

        let (handle, allocation, allocation_info) = alloc::create_image(
            device.allocator,
            &image_create_info,
            &allocation_create_info,
        )
        .unwrap();

        Image {
            handle,
            allocation,
            allocation_info,
            device: Arc::clone(device),
        }
    }

    pub fn map<'a, T>(&'a self) -> ash::prelude::VkResult<MappedBuffer<'a, T>> {
        MappedBuffer::import(
            self.device.allocator,
            self.allocation,
            &self.allocation_info,
        )
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        alloc::destroy_image(self.device.allocator, self.handle, self.allocation)
    }
}
