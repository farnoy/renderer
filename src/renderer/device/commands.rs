use std::{marker::PhantomData, ops::Deref};

use ash::vk;
#[cfg(debug_assertions)]
use hashbrown::HashSet;
use profiling::scope;

use super::{sync::Fence, Device};

pub(crate) struct StrictCommandPool {
    handle: vk::CommandPool,
    #[cfg(debug_assertions)]
    allocated_command_buffers: HashSet<vk::CommandBuffer>,
}

pub(crate) struct StrictRecordingCommandBuffer<'p> {
    phantom: PhantomData<&'p mut StrictCommandPool>,
    device: &'p Device,
    handle: vk::CommandBuffer,
    /* #[cfg(debug_assertions)]
     * commands: Vec<Command>, */
}

pub(crate) struct StrictDebugMarkerGuard<'a, 'p> {
    #[allow(unused)]
    recorder: &'a StrictRecordingCommandBuffer<'p>,
}

pub(crate) struct StrictCommandBuffer {
    handle: vk::CommandBuffer,
    /* #[cfg(debug_assertions)]
     * commands: Vec<Command>, */
}

// pub(crate) enum Command {
//     CopyBuffer {
//         src: vk::Buffer,
//         dst: vk::Buffer,
//         // TODO: regions
//     },
// }

impl StrictCommandPool {
    pub(crate) fn new(device: &Device, queue_family: u32, name: &str) -> StrictCommandPool {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            // .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family);
        let pool = unsafe {
            device
                .device
                .create_command_pool(&pool_create_info, device.instance.allocation_callbacks())
                .unwrap()
        };
        device.set_object_name(pool, name);

        StrictCommandPool {
            handle: pool,
            #[cfg(debug_assertions)]
            allocated_command_buffers: HashSet::new(),
        }
    }

    pub(crate) fn reset(&mut self, device: &Device) {
        scope!("vk::ResetCommandPool");

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
            scope!("vk::AllocateCommandBuffers");
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
            device.destroy_command_pool(self.handle, device.instance.allocation_callbacks());
        }
        self.handle = vk::CommandPool::null();
    }

    pub(crate) fn record_to_specific<'c>(
        &'c mut self,
        device: &'c Device,
        handle: vk::CommandBuffer,
    ) -> StrictRecordingCommandBuffer<'c> {
        scope!("helpers::record_to_command_buffer");

        debug_assert!(handle != vk::CommandBuffer::null());
        #[cfg(debug_assertions)]
        debug_assert!(self.allocated_command_buffers.contains(&handle));

        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            scope!("vk::BeginCommandBuffer");
            device.begin_command_buffer(handle, &begin_info).unwrap();
        }

        StrictRecordingCommandBuffer {
            phantom: PhantomData,
            device,
            handle,
            // commands: vec![],
        }
    }

    pub(crate) fn record_one_time<'c>(
        &'c mut self,
        device: &'c Device,
        name: &str,
    ) -> StrictRecordingCommandBuffer<'c> {
        scope!("helpers::record_one_time");

        let command_buffer = self.allocate(name, device);
        device.set_object_name(command_buffer, name);

        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            scope!("vk::BeginCommandBuffer");
            device.begin_command_buffer(command_buffer, &begin_info).unwrap();
        }

        StrictRecordingCommandBuffer {
            phantom: PhantomData,
            device,
            handle: command_buffer,
            // commands: vec![],
        }
    }
}

// impl StrictCommandBuffer {
//     #[cfg(debug_assertions)]
//     pub(crate) fn commands(&self) -> &[Command] {
//         &self.commands
//     }
// }

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

impl<'p> StrictRecordingCommandBuffer<'p> {
    #[cfg(feature = "vk_names")]
    pub(crate) fn debug_marker_around<'a>(&'a self, name: &str, color: [f32; 4]) -> StrictDebugMarkerGuard<'a, 'p> {
        unsafe {
            use std::ffi::CString;

            let name = CString::new(name).unwrap();
            {
                self.device.instance.debug_utils().cmd_begin_debug_utils_label(
                    self.handle,
                    &vk::DebugUtilsLabelEXT::builder().label_name(&name).color(color),
                );
            }
            StrictDebugMarkerGuard { recorder: self }
        }
    }

    #[cfg(not(feature = "vk_names"))]
    pub(crate) fn debug_marker_around<'a>(&'a self, _name: &str, _color: [f32; 4]) -> StrictDebugMarkerGuard<'a, 'p> {
        StrictDebugMarkerGuard { recorder: self }
    }

    // #[cfg(feature = "vk_names")]
    // pub(crate) fn debug_marker_around2<'a>(
    //     &'a mut self,
    //     name: &str,
    //     color: [f32; 4],
    // ) -> StrictDebugMarkerGuard2<'a, 'p> {
    //     unsafe {
    //         use std::ffi::CString;

    //         let name = CString::new(name).unwrap();
    //         {
    //             self.device.instance.debug_utils().cmd_begin_debug_utils_label(
    //                 self.handle,
    //                 &vk::DebugUtilsLabelEXT::builder().label_name(&name).color(color),
    //             );
    //         }
    //         StrictDebugMarkerGuard2 { recorder: self }
    //     }
    // }

    // #[cfg(not(feature = "vk_names"))]
    // pub(crate) fn debug_marker_around2<'a>(
    //     &'a self,
    //     _name: &str,
    //     _color: [f32; 4],
    // ) -> StrictDebugMarkerGuard2<'a, 'c, 'p> {
    //     StrictDebugMarkerGuard2 { recorder: self }
    // }

    pub(crate) fn end(self) -> StrictCommandBuffer {
        scope!("vk::EndCommandBuffer");
        unsafe {
            self.device.end_command_buffer(self.handle).unwrap();
        }

        StrictCommandBuffer {
            handle: self.handle,
            // commands: self.commands,
        }
    }

    pub(crate) fn submit_once(&self, queue: &mut vk::Queue, fence_name: &str) -> Fence {
        let submit_fence = self.device.new_fence();
        self.device.set_object_name(submit_fence.handle, fence_name);

        unsafe {
            self.device.end_command_buffer(self.handle).unwrap();
            let command_buffers = &[self.handle];
            let submits = [vk::SubmitInfo::builder().command_buffers(command_buffers).build()];

            self.device.queue_submit(*queue, &submits, submit_fence.handle).unwrap();
        }

        submit_fence
    }
}

// impl StrictDebugMarkerGuard2<'_, '_> {
//     pub(crate) fn cmd_copy_buffer(&mut self, src: vk::Buffer, dst: vk::Buffer, regions:
// &[vk::BufferCopy]) {         self.recorder.commands.push(Command::CopyBuffer { src, dst });
//         unsafe {
//             self.recorder
//                 .device
//                 .cmd_copy_buffer(self.recorder.handle, src, dst, regions)
//         }
//     }
// }

impl Deref for StrictRecordingCommandBuffer<'_> {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for StrictDebugMarkerGuard<'_, '_> {
    fn drop(&mut self) {
        #[cfg(feature = "vk_names")]
        unsafe {
            self.recorder
                .device
                .instance
                .debug_utils()
                .cmd_end_debug_utils_label(**self.recorder);
        }
    }
}

impl Deref for StrictCommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}
