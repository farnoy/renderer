#[cfg(feature = "crash_debugging")]
use super::super::{alloc::VmaMemoryUsage, Buffer, DoubleBuffered};
use super::super::{Device, ImageIndex, RenderFrame};
use ash::vk;
use bevy_ecs::prelude::*;
#[cfg(feature = "crash_debugging")]
use std::{convert::TryInto, mem::size_of};

pub(crate) type CrashStages = [u32; 2];

#[allow(unused)]
const ZEROED_CRASH: CrashStages = [0; 2];

#[cfg(feature = "crash_debugging")]
pub(crate) struct CrashBuffer(DoubleBuffered<Buffer>);
#[cfg(not(feature = "crash_debugging"))]
pub(crate) struct CrashBuffer;

impl FromWorld for CrashBuffer {
    fn from_world(#[allow(unused)] world: &mut World) -> Self {
        #[cfg(feature = "crash_debugging")]
        {
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
        #[cfg(not(feature = "crash_debugging"))]
        CrashBuffer
    }
}

impl CrashBuffer {
    #[cfg(feature = "crash_debugging")]
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

    #[cfg(not(feature = "crash_debugging"))]
    pub(crate) fn record(
        &self,
        _renderer: &RenderFrame,
        _command_buffer: vk::CommandBuffer,
        _pipeline_stage: vk::PipelineStageFlags,
        _image_index: &ImageIndex,
        _step: u8,
    ) {
    }

    #[allow(unused_variables)]
    pub(crate) fn destroy(self, device: &Device) {
        #[cfg(feature = "crash_debugging")]
        self.0.into_iter().for_each(|buf| buf.destroy(device));
    }
}
