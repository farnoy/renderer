use super::super::{device::Semaphore, graphics as graphics_sync, RenderFrame, Swapchain};
use crate::renderer::{timeline_value, timeline_value_last};
use ash::{version::DeviceV1_0, vk};
#[cfg(feature = "microprofile")]
use microprofile::scope;
use std::u64;

pub struct PresentData {
    render_complete_semaphore: Semaphore,
}

#[derive(Debug)]
pub struct ImageIndex(pub u32);

impl Default for ImageIndex {
    fn default() -> Self {
        ImageIndex(0)
    }
}

// Acquire swapchain image and store the index
pub struct AcquireFramebuffer;

pub struct PresentFramebuffer;

impl PresentData {
    pub fn new(renderer: &RenderFrame) -> PresentData {
        let render_complete_semaphore = renderer.device.new_semaphore();
        renderer.device.set_object_name(
            render_complete_semaphore.handle,
            "Render complete semaphore",
        );

        PresentData {
            render_complete_semaphore,
        }
    }
}

impl AcquireFramebuffer {
    /// Returns true if framebuffer and swapchain need to be recreated
    pub fn exec(
        renderer: &RenderFrame,
        swapchain: &Swapchain,
        image_index: &mut ImageIndex,
    ) -> bool {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "AcquireFramebuffer::exec");
        let image_acquired_semaphore = renderer.device.new_semaphore();
        renderer.device.set_object_name(
            image_acquired_semaphore.handle,
            "Image acquired semaphore temp",
        );
        let x = renderer.device.new_fence();
        let result = unsafe {
            swapchain.ext.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                image_acquired_semaphore.handle,
                x.handle,
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
            image_acquired_semaphore.handle,
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

        rayon::spawn(move || {
            unsafe {
                device.wait_for_fences(&[x.handle], true, u64::MAX).unwrap();
            }
            drop(image_acquired_semaphore);
        });

        false
    }
}

impl PresentFramebuffer {
    pub fn exec(
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
