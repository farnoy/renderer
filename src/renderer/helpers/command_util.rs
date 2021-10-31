use ash::vk;

use crate::renderer::{
    device::{Device, DoubleBuffered, StrictCommandPool, StrictRecordingCommandBuffer},
    ImageIndex, RenderFrame,
};

pub(crate) struct CommandUtil {
    command_pools: DoubleBuffered<StrictCommandPool>,
    command_buffers: DoubleBuffered<vk::CommandBuffer>,
}

impl CommandUtil {
    pub(crate) fn new(renderer: &RenderFrame, queue_family: u32) -> Self {
        let mut command_pools = renderer.new_buffered(|ix| {
            StrictCommandPool::new(
                &renderer.device,
                queue_family,
                &format!("CommandUtil Command Pool[{}]", ix),
            )
        });
        let command_buffers = renderer.new_buffered(|ix| {
            command_pools
                .current_mut(ix)
                .allocate(&format!("CommandUtil CB[{}]", ix), &renderer.device)
        });
        CommandUtil {
            command_pools,
            command_buffers,
        }
    }
}

impl CommandUtil {
    pub(crate) fn reset_and_record<'s>(
        &'s mut self,
        renderer: &'s RenderFrame,
        image_index: &ImageIndex,
    ) -> StrictRecordingCommandBuffer<'s> {
        let command_pool = self.command_pools.current_mut(image_index.0);

        command_pool.reset(&renderer.device);

        command_pool.record_to_specific(&renderer.device, *self.command_buffers.current(image_index.0))
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.command_pools.into_iter().for_each(|p| p.destroy(device));
        // command buffers are freed with the pools
    }
}
