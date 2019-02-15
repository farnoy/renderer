use super::super::{
    device::{CommandBuffer, Fence, Semaphore},
    RenderFrame,
};
use ash::{version::DeviceV1_0, vk};
use specs::prelude::*;
use std::u64;

// Shared resource
pub struct PresentData {
    pub(in super::super) render_complete_semaphore: Semaphore,
    pub(in super::super) present_semaphore: Semaphore,
    pub(in super::super) render_command_buffer: Option<CommandBuffer>,
    pub(in super::super) render_complete_fence: Fence,
    pub(in super::super) image_index: u32,
}

// Acquire swapchain image and store the index
pub struct AcquireFramebuffer;

pub struct PresentFramebuffer;

pub struct PresentDataSetupHandler;

impl shred::SetupHandler<PresentData> for PresentDataSetupHandler {
    fn setup(res: &mut Resources) {
        let renderer = res.fetch::<RenderFrame>();
        let render_complete_semaphore = renderer.device.new_semaphore();
        renderer.device.set_object_name(
            render_complete_semaphore.handle,
            "Render complete semaphore",
        );

        let present_semaphore = renderer.device.new_semaphore();
        renderer
            .device
            .set_object_name(present_semaphore.handle, "Present semaphore");

        let render_complete_fence = renderer.device.new_fence();
        renderer
            .device
            .set_object_name(render_complete_fence.handle, "Render complete fence");

        drop(renderer);

        res.insert(PresentData {
            render_complete_semaphore,
            present_semaphore,
            render_command_buffer: None,
            render_complete_fence,
            image_index: 0,
        });
    }
}

impl<'a> System<'a> for AcquireFramebuffer {
    type SystemData = (
        ReadExpect<'a, RenderFrame>,
        Write<'a, PresentData, PresentDataSetupHandler>,
    );

    fn run(&mut self, (renderer, mut present_data): Self::SystemData) {
        if present_data.render_command_buffer.is_some() {
            unsafe {
                renderer
                    .device
                    .wait_for_fences(&[present_data.render_complete_fence.handle], true, u64::MAX)
                    .expect("Wait for fence failed.");
            }
        }
        unsafe {
            renderer
                .device
                .reset_fences(&[present_data.render_complete_fence.handle])
                .expect("failed to reset render complete fence");
        }
        present_data.image_index = unsafe {
            renderer
                .swapchain
                .handle
                .ext
                .acquire_next_image(
                    renderer.swapchain.handle.swapchain,
                    u64::MAX,
                    present_data.present_semaphore.handle,
                    vk::Fence::null(),
                )
                .unwrap()
                .0 // TODO: 2nd argument is boolean describing surface optimality
        };
    }
}

impl<'a> System<'a> for PresentFramebuffer {
    type SystemData = (ReadExpect<'a, RenderFrame>, ReadExpect<'a, PresentData>);

    fn run(&mut self, (renderer, present_data): Self::SystemData) {
        {
            let wait_semaphores = &[present_data.render_complete_semaphore.handle];
            let swapchains = &[renderer.swapchain.handle.swapchain];
            let image_indices = &[present_data.image_index];
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
