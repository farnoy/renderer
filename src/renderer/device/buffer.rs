use ash::vk;
use std::{ptr, sync::Arc};

use super::{super::alloc, mapping::MappedBuffer, Device};

pub(crate) struct Buffer {
    pub(crate) handle: vk::Buffer,
    allocation: alloc::VmaAllocation,
    pub(crate) allocation_info: alloc::VmaAllocationInfo,
    device: Arc<Device>,
}

impl Buffer {
    pub(super) fn new(
        device: &Arc<Device>,
        buffer_usage: vk::BufferUsageFlags,
        allocation_usage: alloc::VmaMemoryUsage,
        size: vk::DeviceSize,
    ) -> Buffer {
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
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(buffer_usage)
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

        let (handle, allocation, allocation_info) = alloc::create_buffer(
            device.allocator,
            &buffer_create_info,
            &allocation_create_info,
        )
        .unwrap();

        Buffer {
            handle,
            allocation,
            allocation_info,
            device: Arc::clone(device),
        }
    }

    pub(crate) fn map<'a, T>(&'a self) -> ash::prelude::VkResult<MappedBuffer<'a, T>> {
        MappedBuffer::import(
            self.device.allocator,
            self.allocation,
            &self.allocation_info,
        )
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        alloc::destroy_buffer(self.device.allocator, self.handle, self.allocation)
    }
}
