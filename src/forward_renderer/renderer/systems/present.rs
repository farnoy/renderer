use super::super::{
    device::{CommandBuffer, Fence},
    RenderFrame,
};
use ash::{version::DeviceV1_0, vk};
use specs::prelude::*;
use std::u64;

// Shared resource
pub struct PresentData {
    pub(in super::super) render_command_buffer: Option<CommandBuffer>,
    pub(in super::super) render_complete_fence: Option<Fence>,
}

// Acquire swapchain image and store the index
pub struct AcquireFramebuffer;

pub struct PresentFramebuffer;

impl<'a> System<'a> for AcquireFramebuffer {
    type SystemData = (WriteExpect<'a, RenderFrame>, Read<'a, PresentData>);

    fn run(&mut self, (mut renderer, present_data): Self::SystemData) {
        if let Some(ref fence) = present_data.render_complete_fence {
            unsafe {
                renderer
                    .device
                    .wait_for_fences(&[fence.handle], true, u64::MAX)
                    .expect("Wait for fence failed.");
            }
        }
        renderer.image_index = unsafe {
            renderer
                .swapchain
                .handle
                .ext
                .acquire_next_image(
                    renderer.swapchain.handle.swapchain,
                    u64::MAX,
                    renderer.present_semaphore.handle,
                    vk::Fence::null(),
                )
                .unwrap()
                .0 // TODO: 2nd argument is boolean describing surface optimality
        };
    }
}

impl Default for PresentData {
    fn default() -> PresentData {
        PresentData {
            render_command_buffer: None,
            render_complete_fence: None,
        }
    }
}

impl<'a> System<'a> for PresentFramebuffer {
    type SystemData = ReadExpect<'a, RenderFrame>;

    fn run(&mut self, renderer: Self::SystemData) {
        {
            let wait_semaphores = &[renderer.rendering_complete_semaphore.handle];
            let swapchains = &[renderer.swapchain.handle.swapchain];
            let image_indices = &[renderer.image_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(wait_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indices);

            let queue = renderer.device.graphics_queue.lock();
            unsafe {
                renderer
                    .swapchain
                    .handle
                    .ext
                    .queue_present(*queue, &present_info)
                    .unwrap();
            }
        }
    }
}
