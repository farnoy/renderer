use ash::{version::DeviceV1_0, vk};
use parking_lot::{Mutex, MutexGuard};
use std::{marker::PhantomData, ops::Deref, sync::Arc};

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

pub struct StrictCommandPool {
    device: Arc<Device>,
    handle: vk::CommandPool,
}

pub struct StrictCommandPoolSession<'p>(&'p mut StrictCommandPool);

pub struct StrictRecordingCommandBuffer<'c, 'p> {
    pool: &'c mut StrictCommandPool,
    handle: vk::CommandBuffer,
    phantom: PhantomData<&'p StrictCommandPool>,
}

pub struct StrictDebugMarkerGuard<'a, 'c, 'p> {
    #[allow(unused)]
    recorder: &'a StrictRecordingCommandBuffer<'c, 'p>,
}

pub struct StrictCommandBuffer<'p> {
    handle: vk::CommandBuffer,
    phantom: PhantomData<&'p StrictCommandPool>,
}

impl StrictCommandPool {
    pub fn new(device: &Arc<Device>, queue_family: u32, name: &str) -> StrictCommandPool {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::empty())
            .queue_family_index(queue_family);
        let pool = unsafe {
            device
                .device
                .create_command_pool(&pool_create_info, None)
                .unwrap()
        };
        device.set_object_name(pool, name);

        StrictCommandPool {
            handle: pool,
            device: Arc::clone(device),
        }
    }

    pub fn session(&mut self) -> StrictCommandPoolSession {
        StrictCommandPoolSession(self)
    }

    pub unsafe fn reset(&mut self) {
        self.device
            .reset_command_pool(self.handle, vk::CommandPoolResetFlags::empty())
            .unwrap();
    }
}

impl Drop for StrictCommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.handle, None);
        }
    }
}

impl<'p> StrictCommandPoolSession<'p> {
    pub fn record_one_time<'c>(&'c mut self, name: &str) -> StrictRecordingCommandBuffer<'c, 'p> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(self.0.handle)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers = unsafe {
            self.0
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()
        };
        let command_buffer = command_buffers[0];
        self.0.device.set_object_name(command_buffer, name);

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.0
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        StrictRecordingCommandBuffer {
            pool: self.0,
            handle: command_buffer,
            phantom: PhantomData,
        }
    }
}

impl<'c, 'p> StrictRecordingCommandBuffer<'c, 'p> {
    #[cfg(feature = "vk_names")]
    pub fn debug_marker_around<'a>(
        &'a self,
        name: &str,
        color: [f32; 4],
    ) -> StrictDebugMarkerGuard<'a, 'c, 'p> {
        unsafe {
            use std::ffi::CString;

            let name = CString::new(name).unwrap();
            {
                self.pool
                    .device
                    .instance
                    .debug_utils()
                    .cmd_begin_debug_utils_label(
                        self.handle,
                        &vk::DebugUtilsLabelEXT::builder()
                            .label_name(&name)
                            .color(color),
                    );
            }
            StrictDebugMarkerGuard { recorder: self }
        }
    }

    #[cfg(not(feature = "vk_names"))]
    pub fn debug_marker_around<'a>(
        &'a self,
        _name: &str,
        _color: [f32; 4],
    ) -> StrictDebugMarkerGuard<'a, 'c, 'p> {
        StrictDebugMarkerGuard { recorder: self }
    }

    pub fn end(self) -> StrictCommandBuffer<'p> {
        unsafe {
            self.pool.device.end_command_buffer(self.handle).unwrap();
        }

        StrictCommandBuffer {
            handle: self.handle,
            phantom: PhantomData,
        }
    }
}

impl<'c, 'p> Deref for StrictRecordingCommandBuffer<'c, 'p> {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl<'a, 'c, 'p> Drop for StrictDebugMarkerGuard<'a, 'c, 'p> {
    fn drop(&mut self) {
        #[cfg(feature = "vk_names")]
        unsafe {
            self.recorder
                .pool
                .device
                .instance
                .debug_utils()
                .cmd_end_debug_utils_label(**self.recorder);
        }
    }
}

impl<'p> Deref for StrictCommandBuffer<'p> {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
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
