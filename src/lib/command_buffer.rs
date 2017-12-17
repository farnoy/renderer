use ash::version::DeviceV1_0;
use ash::vk;
use std::default::Default;
use std::iter::Iterator;
use std::ptr;
use std::u64;

use super::ExampleBase;

/// Allocates a one time, temporary command buffer, submits it and waits until it's finished.
pub fn one_time_submit_and_wait<F: FnOnce(vk::CommandBuffer)>(base: &ExampleBase, f: F) {
    unsafe {
        let cb = {
            let info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::CommandBufferAllocateInfo,
                p_next: ptr::null(),
                command_pool: base.pool,
                level: vk::CommandBufferLevel::Primary,
                command_buffer_count: 1,
            };
            base.device
                .allocate_command_buffers(&info)
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
        };

        {
            let info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::CommandBufferBeginInfo,
                p_next: ptr::null(),
                flags: vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                p_inheritance_info: ptr::null(),
            };
            base.device.begin_command_buffer(cb, &info).unwrap();
        }

        f(cb);

        base.device.end_command_buffer(cb).unwrap();

        let fence = {
            let info = vk::FenceCreateInfo {
                s_type: vk::StructureType::FenceCreateInfo,
                p_next: ptr::null(),
                flags: Default::default(),
            };

            base.device.create_fence(&info, None).unwrap()
        };

        {
            let info = vk::SubmitInfo {
                s_type: vk::StructureType::SubmitInfo,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &cb,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
            };

            base.device
                .queue_submit(base.present_queue, &[info], fence)
                .unwrap();
        }

        base.device
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();

        base.device.destroy_fence(fence, None);
    }
}
