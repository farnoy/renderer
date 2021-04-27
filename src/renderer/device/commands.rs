use std::{marker::PhantomData, ops::Deref};

use ash::{version::DeviceV1_0, vk};
#[cfg(debug_assertions)]
use hashbrown::HashSet;
use microprofile::scope;

use super::{sync::Fence, Device};

pub(crate) struct StrictCommandPool {
    handle: vk::CommandPool,
    #[cfg(debug_assertions)]
    allocated_command_buffers: HashSet<vk::CommandBuffer>,
}

pub(crate) struct StrictCommandPoolSession<'p> {
    pool: &'p mut StrictCommandPool,
    device: &'p Device,
}

pub(crate) struct StrictRecordingCommandBuffer<'c, 'p> {
    session: &'c mut StrictCommandPoolSession<'p>,
    handle: vk::CommandBuffer,
}

pub(crate) struct StrictDebugMarkerGuard<'a, 'c, 'p> {
    #[allow(unused)]
    recorder: &'a StrictRecordingCommandBuffer<'c, 'p>,
}

pub(crate) struct StrictCommandBuffer<'p> {
    handle: vk::CommandBuffer,
    phantom: PhantomData<&'p StrictCommandPool>,
}

impl StrictCommandPool {
    pub(crate) fn new(device: &Device, queue_family: u32, name: &str) -> StrictCommandPool {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            // .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family);
        let pool = unsafe { device.device.create_command_pool(&pool_create_info, None).unwrap() };
        device.set_object_name(pool, name);

        StrictCommandPool {
            handle: pool,
            #[cfg(debug_assertions)]
            allocated_command_buffers: HashSet::new(),
        }
    }

    pub(crate) fn session<'a>(&'a mut self, device: &'a Device) -> StrictCommandPoolSession<'a> {
        StrictCommandPoolSession { pool: self, device }
    }

    pub(crate) fn reset(&mut self, device: &Device) {
        scope!("vk", "vkResetCommandPool");

        unsafe {
            device
                .reset_command_pool(self.handle, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
    }

    pub(crate) fn allocate(&mut self, name: &str, device: &Device) -> vk::CommandBuffer {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(self.handle)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers = unsafe {
            scope!("vk", "vkAllocateCommandBuffers");
            device.allocate_command_buffers(&command_buffer_allocate_info).unwrap()
        };
        let command_buffer = command_buffers[0];
        device.set_object_name(command_buffer, name);
        #[cfg(debug_assertions)]
        self.allocated_command_buffers.insert(command_buffer);
        command_buffer
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_command_pool(self.handle, None);
        }
        self.handle = vk::CommandPool::null();
    }
}

#[cfg(debug_assertions)]
impl Drop for StrictCommandPool {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::CommandPool::null(),
            "StrictCommandPool not destroyed before Drop"
        );
    }
}

impl<'p> StrictCommandPoolSession<'p> {
    pub(crate) fn record_to_specific<'c>(
        &'c mut self,
        handle: vk::CommandBuffer,
    ) -> StrictRecordingCommandBuffer<'c, 'p> {
        scope!("helpers", "record_to_command_buffer");

        debug_assert!(handle != vk::CommandBuffer::null());
        #[cfg(debug_assertions)]
        debug_assert!(self.pool.allocated_command_buffers.contains(&handle));

        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            scope!("vk", "vkBeginCommandBuffer");
            self.device.begin_command_buffer(handle, &begin_info).unwrap();
        }

        StrictRecordingCommandBuffer { session: self, handle }
    }

    pub(crate) fn record_one_time<'c>(&'c mut self, name: &str) -> StrictRecordingCommandBuffer<'c, 'p> {
        scope!("helpers", "record_one_time");

        let command_buffer = self.pool.allocate(name, self.device);
        self.device.set_object_name(command_buffer, name);

        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            scope!("vk", "vkBeginCommandBuffer");
            self.device.begin_command_buffer(command_buffer, &begin_info).unwrap();
        }

        StrictRecordingCommandBuffer {
            session: self,
            handle: command_buffer,
        }
    }
}

impl<'c, 'p> StrictRecordingCommandBuffer<'c, 'p> {
    #[cfg(feature = "vk_names")]
    pub(crate) fn debug_marker_around<'a>(&'a self, name: &str, color: [f32; 4]) -> StrictDebugMarkerGuard<'a, 'c, 'p> {
        unsafe {
            use std::ffi::CString;

            let name = CString::new(name).unwrap();
            {
                self.session.device.instance.debug_utils().cmd_begin_debug_utils_label(
                    self.handle,
                    &vk::DebugUtilsLabelEXT::builder().label_name(&name).color(color),
                );
            }
            StrictDebugMarkerGuard { recorder: self }
        }
    }

    #[cfg(not(feature = "vk_names"))]
    pub(crate) fn debug_marker_around<'a>(
        &'a self,
        _name: &str,
        _color: [f32; 4],
    ) -> StrictDebugMarkerGuard<'a, 'c, 'p> {
        StrictDebugMarkerGuard { recorder: self }
    }

    pub(crate) fn end(self) -> StrictCommandBuffer<'p> {
        scope!("vk", "vkEndCommandBuffer");
        unsafe {
            self.session.device.end_command_buffer(self.handle).unwrap();
        }

        StrictCommandBuffer {
            handle: self.handle,
            phantom: PhantomData,
        }
    }

    pub(crate) fn submit_once(&self, queue: &mut vk::Queue, fence_name: &str) -> Fence {
        let submit_fence = self.session.device.new_fence();
        self.session.device.set_object_name(submit_fence.handle, fence_name);

        unsafe {
            self.session.device.end_command_buffer(self.handle).unwrap();
            let command_buffers = &[self.handle];
            let submits = [vk::SubmitInfo::builder().command_buffers(command_buffers).build()];

            self.session
                .device
                .queue_submit(*queue, &submits, submit_fence.handle)
                .unwrap();
        }

        submit_fence
    }
}

impl Deref for StrictRecordingCommandBuffer<'_, '_> {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for StrictDebugMarkerGuard<'_, '_, '_> {
    fn drop(&mut self) {
        #[cfg(feature = "vk_names")]
        unsafe {
            self.recorder
                .session
                .device
                .instance
                .debug_utils()
                .cmd_end_debug_utils_label(**self.recorder);
        }
    }
}

impl Deref for StrictCommandBuffer<'_> {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}
