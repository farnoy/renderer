use ash::{version::DeviceV1_0, vk};
use parking_lot::Mutex;
use std::{ops::Deref, sync::Arc};

use super::{
    super::helpers::{new_fence, Fence},
    Device,
};

pub struct CommandPool {
    handle: Mutex<vk::CommandPool>,
    pub device: Arc<Device>,
}

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    pool: Arc<CommandPool>,
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

    pub fn record_one_time<F: FnOnce(vk::CommandBuffer)>(
        self: &Arc<CommandPool>,
        f: F,
    ) -> CommandBuffer {
        let mut pool_lock = self.handle.lock();
        let command_buffer = self.allocate_command_buffers(1, &mut *pool_lock).remove(0);

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        f(command_buffer);

        unsafe {
            self.device.end_command_buffer(command_buffer).unwrap();
        }

        drop(pool_lock);

        CommandBuffer {
            pool: Arc::clone(self),
            handle: command_buffer,
        }
    }
}

impl CommandBuffer {
    pub fn submit_once(&self, queue: &mut vk::Queue, fence_name: &str) -> Fence {
        let submit_fence = new_fence(Arc::clone(&self.pool.device));
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
