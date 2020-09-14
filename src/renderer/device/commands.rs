use super::{sync::Fence, Device};
use ash::{version::DeviceV1_0, vk};
use std::{marker::PhantomData, mem::swap, ops::Deref, sync::Arc};

pub struct StrictCommandPool {
    device: Arc<Device>,
    handle: vk::CommandPool,
    queue_family: u32, // only needed for recreate(), remove later
    name: String,      // only needed for recreate(), remove later
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
            queue_family,
            name: name.into(),
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

    // To workaround a performance cliff in AMDVLK
    pub unsafe fn recreate(&mut self) {
        let mut new_command_pool =
            StrictCommandPool::new(&self.device, self.queue_family, &self.name);

        swap(self, &mut new_command_pool);
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

    pub fn submit_once(&self, queue: &mut vk::Queue, fence_name: &str) -> Fence {
        let submit_fence = self.pool.device.new_fence();
        self.pool
            .device
            .set_object_name(submit_fence.handle, fence_name);

        unsafe {
            self.pool.device.end_command_buffer(self.handle).unwrap();
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
