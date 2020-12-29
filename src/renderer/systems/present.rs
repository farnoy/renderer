use super::super::{device::Semaphore, graphics as graphics_sync, RenderFrame, Swapchain};
use crate::renderer::{
    timeline_value, timeline_value_last, MainAttachments, MainFramebuffer, Resized,
};
use ash::{version::DeviceV1_0, vk};
use bevy_ecs::prelude::*;
#[cfg(feature = "microprofile")]
use microprofile::scope;
use std::u64;

pub(crate) struct PresentData {
    framebuffer_acquire_semaphore: Semaphore,
    render_complete_semaphore: Semaphore,
}

#[derive(Debug)]
pub(crate) struct ImageIndex(pub(crate) u32);

impl Default for ImageIndex {
    fn default() -> Self {
        ImageIndex(0)
    }
}

// Acquire swapchain image and store the index
pub(crate) struct AcquireFramebuffer;

pub(crate) struct PresentFramebuffer;

impl PresentData {
    pub(crate) fn new(renderer: &RenderFrame) -> PresentData {
        let framebuffer_acquire_semaphore = renderer.device.new_semaphore();
        renderer.device.set_object_name(
            framebuffer_acquire_semaphore.handle,
            "Framebuffer acquire semaphore",
        );
        let render_complete_semaphore = renderer.device.new_semaphore();
        renderer.device.set_object_name(
            render_complete_semaphore.handle,
            "Render complete semaphore",
        );

        PresentData {
            framebuffer_acquire_semaphore,
            render_complete_semaphore,
        }
    }
}

pub(crate) fn acquire_framebuffer(
    renderer: Res<RenderFrame>,
    mut swapchain: ResMut<Swapchain>,
    mut main_attachments: ResMut<MainAttachments>,
    mut main_framebuffer: ResMut<MainFramebuffer>,
    mut present_data: ResMut<PresentData>,
    mut image_index: ResMut<ImageIndex>,
    resized: Res<Resized>,
) {
    #[cfg(feature = "profiling")]
    microprofile::scope!("ecs", "AcquireFramebuffer");

    if resized.0 {
        unsafe {
            renderer.device.device_wait_idle().unwrap();
        }
        swapchain.resize_to_fit();
        *main_attachments = MainAttachments::new(&renderer, &swapchain);
        *main_framebuffer = MainFramebuffer::new(&renderer, &main_attachments, &swapchain);
        *present_data = PresentData::new(&renderer);
    }

    let swapchain_needs_recreating =
        AcquireFramebuffer::exec(&renderer, &swapchain, &mut *present_data, &mut *image_index);

    if swapchain_needs_recreating {
        unsafe {
            renderer.device.device_wait_idle().unwrap();
        }
        swapchain.resize_to_fit();
        *main_attachments = MainAttachments::new(&renderer, &swapchain);
        *main_framebuffer = MainFramebuffer::new(&renderer, &main_attachments, &swapchain);
        *present_data = PresentData::new(&renderer);
        AcquireFramebuffer::exec(&renderer, &swapchain, &mut *present_data, &mut *image_index);
    }
}

impl AcquireFramebuffer {
    /// Returns true if framebuffer and swapchain need to be recreated
    pub(crate) fn exec(
        renderer: &RenderFrame,
        swapchain: &Swapchain,
        present_data: &mut PresentData,
        image_index: &mut ImageIndex,
    ) -> bool {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "AcquireFramebuffer::exec");
        let result = unsafe {
            swapchain.ext.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                present_data.framebuffer_acquire_semaphore.handle,
                vk::Fence::null(),
            )
        };

        let device = renderer.device.clone();

        match result {
            Ok((ix, false)) => image_index.0 = ix,
            Ok((ix, true)) => {
                image_index.0 = ix;
                println!("AcquireFramebuffer image suboptimal");
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                println!("out of date in AcquireFramebuffer");
                return true;
            }
            _ => panic!("unknown condition in AcquireFramebuffer"),
        }

        let counter = renderer.graphics_timeline_semaphore.value().unwrap();
        debug_assert!(
            counter < timeline_value::<_, graphics_sync::Start>(renderer.frame_number),
            "AcquireFramebuffer assumption incorrect"
        );
        let wait_semaphore_values = &[
            timeline_value_last::<_, graphics_sync::GuiDraw>(renderer.frame_number),
            0,
        ];
        let signal_semaphore_values = &[timeline_value::<_, graphics_sync::Start>(
            renderer.frame_number,
        )];
        let mut wait_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(wait_semaphore_values)
            .signal_semaphore_values(signal_semaphore_values);

        let wait_semaphores = &[
            renderer.graphics_timeline_semaphore.handle,
            present_data.framebuffer_acquire_semaphore.handle,
        ];
        let queue = renderer.device.graphics_queue.lock();
        let signal_semaphores = &[renderer.graphics_timeline_semaphore.handle];
        let dst_stage_masks = &[
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
        ];
        let submit = vk::SubmitInfo::builder()
            .push_next(&mut wait_timeline)
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(dst_stage_masks)
            .signal_semaphores(signal_semaphores)
            .build();

        unsafe {
            renderer
                .device
                .queue_submit(*queue, &[submit], vk::Fence::null())
                .unwrap();
        }

        false
    }
}

impl PresentFramebuffer {
    pub(crate) fn exec(
        renderer: &mut RenderFrame,
        present_data: &PresentData,
        swapchain: &Swapchain,
        image_index: &ImageIndex,
    ) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "PresentFramebuffer");
        {
            let wait_values = &[timeline_value::<_, graphics_sync::GuiDraw>(
                renderer.frame_number,
            )];
            let mut wait_timeline =
                vk::TimelineSemaphoreSubmitInfo::builder().wait_semaphore_values(wait_values);

            let wait_semaphores = &[renderer.graphics_timeline_semaphore.handle];
            let queue = renderer.device.graphics_queue.lock();
            let signal_semaphores = &[present_data.render_complete_semaphore.handle];
            let dst_stage_masks = vec![vk::PipelineStageFlags::TOP_OF_PIPE];
            let submit = vk::SubmitInfo::builder()
                .push_next(&mut wait_timeline)
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(&dst_stage_masks)
                .signal_semaphores(signal_semaphores)
                .build();

            unsafe {
                renderer
                    .device
                    .queue_submit(*queue, &[submit], vk::Fence::null())
                    .unwrap();
            }
        }

        let wait_semaphores = &[present_data.render_complete_semaphore.handle];
        let swapchains = &[swapchain.swapchain];
        let image_indices = &[image_index.0];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(wait_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let queue = renderer.device.graphics_queue.lock();
        let result = unsafe { swapchain.ext.queue_present(*queue, &present_info) };
        match result {
            Ok(false) => (),
            Ok(true) => println!("PresentFramebuffer image suboptimal"),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => println!("out of date in PresentFramebuffer"),
            Err(vk::Result::ERROR_DEVICE_LOST) => panic!("device lost in PresentFramebuffer"),
            _ => panic!("unknown condition in PresentFramebuffer"),
        }
        renderer.previous_frame_number_for_swapchain_index[image_index.0 as usize] =
            renderer.frame_number;
    }
}
