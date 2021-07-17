use std::{marker::PhantomData, mem::size_of, ptr};

use ash::vk;

use super::{
    alloc,
    mapping::{MappedBuffer, MappedStaticBuffer},
    Device,
};

pub(crate) struct Buffer {
    pub(crate) handle: vk::Buffer,
    allocation: alloc::VmaAllocation,
}

pub(crate) struct StaticBuffer<T: Sized> {
    pub(crate) buffer: Buffer,
    _marker: PhantomData<T>,
}

impl Buffer {
    pub(super) fn new(
        device: &Device,
        buffer_usage: vk::BufferUsageFlags,
        allocation_usage: alloc::VmaMemoryUsage,
        size: vk::DeviceSize,
    ) -> Buffer {
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

        let (handle, allocation, _) =
            alloc::create_buffer(device.allocator, &buffer_create_info, &allocation_create_info).unwrap();

        Buffer { handle, allocation }
    }

    pub(super) fn new_exclusive(
        device: &Device,
        buffer_usage: vk::BufferUsageFlags,
        allocation_usage: alloc::VmaMemoryUsage,
        size: vk::DeviceSize,
    ) -> Buffer {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_create_info = alloc::VmaAllocationCreateInfo {
            flags: alloc::VmaAllocationCreateFlagBits(0),
            memoryTypeBits: 0,
            pUserData: ptr::null_mut(),
            pool: ptr::null_mut(),
            preferredFlags: 0,
            requiredFlags: 0,
            usage: allocation_usage,
        };

        let (handle, allocation, _) =
            alloc::create_buffer(device.allocator, &buffer_create_info, &allocation_create_info).unwrap();

        Buffer { handle, allocation }
    }

    pub(crate) fn map<'a, T>(&'a self, device: &Device) -> ash::prelude::VkResult<MappedBuffer<'a, T>> {
        MappedBuffer::import(
            device.allocator,
            self.allocation,
            &alloc::get_allocation_info(device.allocator, self.allocation),
        )
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        alloc::destroy_buffer(device.allocator, self.handle, self.allocation);
        self.handle = vk::Buffer::null();
    }
}

impl<T: Sized> StaticBuffer<T> {
    pub(super) fn new(
        device: &Device,
        buffer_usage: vk::BufferUsageFlags,
        allocation_usage: alloc::VmaMemoryUsage,
    ) -> Self {
        let buffer = Buffer::new(device, buffer_usage, allocation_usage, size_of::<T>() as vk::DeviceSize);
        StaticBuffer {
            buffer,
            _marker: PhantomData,
        }
    }

    pub(super) fn new_exclusive(
        device: &Device,
        buffer_usage: vk::BufferUsageFlags,
        allocation_usage: alloc::VmaMemoryUsage,
    ) -> Self {
        let buffer = Buffer::new_exclusive(device, buffer_usage, allocation_usage, size_of::<T>() as vk::DeviceSize);
        StaticBuffer {
            buffer,
            _marker: PhantomData,
        }
    }

    pub(crate) fn map<'a>(&'a self, device: &Device) -> ash::prelude::VkResult<MappedStaticBuffer<'a, T>> {
        MappedStaticBuffer::import(device.allocator, self.buffer.allocation)
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.buffer.destroy(device);
    }
}

#[cfg(debug_assertions)]
impl Drop for Buffer {
    fn drop(&mut self) {
        debug_assert_eq!(self.handle, vk::Buffer::null(), "Buffer not destroyed before drop");
    }
}
