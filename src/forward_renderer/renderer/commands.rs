use ash::{version::DeviceV1_0, vk};
use parking_lot::Mutex;
use std::{mem::transmute, ops::Deref, ptr, sync::Arc};

use super::{
    device::Device,
    helpers::{new_fence, Fence},
};

pub struct CommandPool {
    handle: Mutex<vk::CommandPool>,
    pub device: Arc<Device>,
}

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    pool: Arc<CommandPool>,
}

pub fn record_one_time<F: FnOnce(vk::CommandBuffer)>(
    command_pool: Arc<CommandPool>,
    f: F,
) -> CommandBuffer {
    let command_buffer = {
        let mut pool_lock = command_pool.handle.lock();
        let command_buffer = command_pool
            .allocate_command_buffers(1, &mut *pool_lock)
            .remove(0);

        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            p_inheritance_info: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        };
        unsafe {
            command_pool
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        f(command_buffer);

        unsafe {
            command_pool
                .device
                .end_command_buffer(command_buffer)
                .unwrap();
        }

        command_buffer
    };

    CommandBuffer {
        pool: command_pool,
        handle: command_buffer,
    }
}

impl CommandPool {
    pub fn new(
        device: Arc<Device>,
        queue_family: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> Arc<CommandPool> {
        let pool_create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags,
            queue_family_index: queue_family,
        };
        let pool = unsafe {
            device
                .device
                .create_command_pool(&pool_create_info, None)
                .unwrap()
        };

        let cp = CommandPool {
            handle: Mutex::new(pool),
            device,
        };
        Arc::new(cp)
    }

    fn allocate_command_buffers(
        &self,
        count: u32,
        lock: &mut vk::CommandPool,
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: count,
            command_pool: *lock,
            level: vk::CommandBufferLevel::PRIMARY,
        };
        unsafe {
            self.device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()
        }
    }
}

impl CommandBuffer {
    pub fn submit_once(&self, queue: &mut vk::Queue, fence_name: &str) -> Arc<Fence> {
        let submit_fence = new_fence(Arc::clone(&self.pool.device));
        self.pool.device.set_object_name(
            vk::ObjectType::FENCE,
            unsafe { transmute::<_, u64>(submit_fence.handle) },
            fence_name,
        );

        unsafe {
            let submits = [vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &self.handle,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
            }];

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
