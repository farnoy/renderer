#[cfg(feature = "crash_debugging")]
use super::super::{alloc::VmaMemoryUsage, Buffer, DoubleBuffered};
use super::super::{ImageIndex, RenderFrame};
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

impl FromResources for CrashBuffer {
    fn from_resources(#[allow(unused)] resources: &Resources) -> Self {
        #[cfg(feature = "crash_debugging")]
        {
            let renderer = resources.query::<Res<RenderFrame>>().unwrap();
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
            renderer
                .device
                .buffer_marker_fn
                .cmd_write_buffer_marker_amd(
                    command_buffer,
                    pipeline_stage,
                    self.0.current(image_index.0).handle,
                    (step as vk::DeviceSize) * size_of::<u32>() as vk::DeviceSize,
                    (renderer.frame_number % (u32::MAX as u64))
                        .try_into()
                        .unwrap(),
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
}

#[cfg(feature = "crash_debugging")]
impl Drop for CrashBuffer {
    fn drop(&mut self) {
        for (ix, buffer) in self.0.iter().enumerate() {
            let mapped = buffer.map::<CrashStages>().unwrap();
            dbg!(ix, mapped[0]);
        }
    }
}
