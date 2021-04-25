use std::{convert::TryInto, mem::size_of};

use ash::vk;
use bevy_ecs::prelude::*;

use super::super::{Buffer, Device, DoubleBuffered, ImageIndex, RenderFrame, VmaMemoryUsage};

pub(crate) type CrashStages = [u32; 2];

pub(crate) struct CrashBuffer(DoubleBuffered<Buffer>);

impl FromWorld for CrashBuffer {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        CrashBuffer(renderer.new_buffered(|ix| {
            let buf = renderer.device.new_buffer(
                vk::BufferUsageFlags::TRANSFER_DST,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_ONLY,
                size_of::<CrashStages>() as vk::DeviceSize,
            );
            renderer
                .device
                .set_object_name(buf.handle, &format!("CrashBuffer[{}]", ix));
            buf
        }))
    }
}

impl CrashBuffer {
    pub(crate) fn record(
        &self,
        renderer: &RenderFrame,
        command_buffer: vk::CommandBuffer,
        pipeline_stage: vk::PipelineStageFlags,
        image_index: &ImageIndex,
        step: u8,
    ) {
        unsafe {
            renderer.device.buffer_marker_fn.cmd_write_buffer_marker_amd(
                command_buffer,
                pipeline_stage,
                self.0.current(image_index.0).handle,
                (step as vk::DeviceSize) * size_of::<u32>() as vk::DeviceSize,
                (renderer.frame_number % (u32::MAX as u64)).try_into().unwrap(),
            );
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.0.into_iter().for_each(|buf| buf.destroy(device));
    }
}
