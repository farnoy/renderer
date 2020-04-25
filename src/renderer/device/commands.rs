use ash::{version::DeviceV1_0, vk};
use parking_lot::{Mutex, MutexGuard};
use std::{ops::Deref, sync::Arc};

use super::{sync::Fence, Device};

pub struct CommandPool {
    handle: Mutex<vk::CommandPool>,
    pub device: Arc<Device>,
}

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    pool: Arc<CommandPool>,
}

pub struct RecordingCommandBuffer<'a> {
    handle: vk::CommandBuffer,
    pub(super) pool: Arc<CommandPool>,
    #[allow(unused)]
    pool_lock: MutexGuard<'a, vk::CommandPool>,
}

impl CommandPool {
    pub(super) fn new(
        device: &Arc<Device>,
        queue_family: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> CommandPool {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(flags)
            .queue_family_index(queue_family);
        let pool = unsafe {
            device
                .device
                .create_command_pool(&pool_create_info, None)
                .unwrap()
        };

        CommandPool {
            handle: Mutex::new(pool),
            device: Arc::clone(device),
        }
    }

    fn allocate_command_buffers(
        &self,
        count: u32,
        lock: &mut vk::CommandPool,
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(count)
            .command_pool(*lock)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            self.device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()
        }
    }

    pub fn record_one_time<'a>(
        self: &'a Arc<CommandPool>,
        name: &str,
    ) -> RecordingCommandBuffer<'a> {
        let mut pool_lock = self.handle.lock();
        let command_buffer = self.allocate_command_buffers(1, &mut *pool_lock).remove(0);
        self.device.set_object_name(command_buffer, name);

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        RecordingCommandBuffer {
            pool: Arc::clone(self),
            pool_lock,
            handle: command_buffer,
        }
    }
}

impl CommandBuffer {
    pub fn submit_once(&self, queue: &mut vk::Queue, fence_name: &str) -> Fence {
        let submit_fence = self.pool.device.new_fence();
        self.pool
            .device
            .set_object_name(submit_fence.handle, fence_name);

        unsafe {
            let command_buffers = &[self.handle];
            let submits = [vk::SubmitInfo::builder()
                .command_buffers(command_buffers)
                .build()];

            self.pool
                .device
                .queue_submit(*queue, &submits, submit_fence.handle)
                .unwrap();
        }

        submit_fence
    }
}

impl RecordingCommandBuffer<'_> {
    pub fn end(self) -> CommandBuffer {
        unsafe {
            self.pool.device.end_command_buffer(self.handle).unwrap();
        }

        CommandBuffer {
            pool: Arc::clone(&self.pool),
            handle: self.handle,
        }
    }
}

impl Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &vk::CommandBuffer {
        &self.handle
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            let lock = self.handle.lock();
            self.device.destroy_command_pool(*lock, None)
        }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            let pool_lock = self.pool.handle.lock();
            self.pool
                .device
                .free_command_buffers(*pool_lock, &[self.handle]);
        }
    }
}

impl Deref for RecordingCommandBuffer<'_> {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &vk::CommandBuffer {
        &self.handle
    }
}
