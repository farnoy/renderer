use super::super::{
    device::{CommandBuffer, DoubleBuffered, Fence, Semaphore},
    RenderFrame,
};
use ash::{version::DeviceV1_0, vk};
#[cfg(feature = "microprofile")]
use microprofile::scope;
use specs::prelude::*;
use std::u64;

pub struct PresentData {
    pub(in super::super) render_complete_semaphore: Semaphore,
    pub(in super::super) present_semaphore: Semaphore,
    pub(in super::super) render_command_buffer: DoubleBuffered<Option<CommandBuffer>>,
    pub(in super::super) render_complete_fence: DoubleBuffered<Fence>,
}

pub struct ImageIndex(pub u32);

impl Default for ImageIndex {
    fn default() -> Self {
        ImageIndex(0)
    }
}

// Acquire swapchain image and store the index
pub struct AcquireFramebuffer;

pub struct PresentFramebuffer;

impl specs::shred::SetupHandler<PresentData> for PresentData {
    fn setup(world: &mut World) {
        if world.has_value::<PresentData>() {
            return;
        }

        let renderer = world.fetch::<RenderFrame>();
        let render_complete_semaphore = renderer.device.new_semaphore();
        renderer.device.set_object_name(
            render_complete_semaphore.handle,
            "Render complete semaphore",
        );

        let present_semaphore = renderer.device.new_semaphore();
        renderer
            .device
            .set_object_name(present_semaphore.handle, "Present semaphore");

        let render_complete_fence = renderer.new_buffered(|ix| {
            let f = renderer.device.new_fence();
            renderer
                .device
                .set_object_name(f.handle, &format!("Render complete fence - {}", ix));
            f
        });

        let render_command_buffer = renderer.new_buffered(|_| None);

        drop(renderer);

        world.insert(PresentData {
            render_complete_semaphore,
            present_semaphore,
            render_command_buffer,
            render_complete_fence,
        });
    }
}

impl<'a> System<'a> for AcquireFramebuffer {
    type SystemData = (
        ReadExpect<'a, RenderFrame>,
        Read<'a, PresentData, PresentData>,
        Write<'a, ImageIndex>,
    );

    fn run(&mut self, (renderer, present_data, mut image_index): Self::SystemData) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "present");
        if present_data
            .render_command_buffer
            .current(image_index.0 + 1)
            .is_some()
        {
            unsafe {
                renderer
                    .device
                    .wait_for_fences(
                        &[present_data
                            .render_complete_fence
                            .current(image_index.0 + 1)
                            .handle],
                        true,
                        u64::MAX,
                    )
                    .expect("Wait for fence failed.");
            }
        }
        unsafe {
            renderer
                .device
                .reset_fences(&[present_data
                    .render_complete_fence
                    .current(image_index.0 + 1)
                    .handle])
                .expect("failed to reset render complete fence");
        }
        image_index.0 = unsafe {
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
    type SystemData = (
        ReadExpect<'a, RenderFrame>,
        Read<'a, PresentData, PresentData>,
        Read<'a, ImageIndex>,
    );

    fn run(&mut self, (renderer, present_data, image_index): Self::SystemData) {
        {
            let wait_semaphores = &[present_data.render_complete_semaphore.handle];
            let swapchains = &[renderer.swapchain.handle.swapchain];
            let image_indices = &[image_index.0];
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
