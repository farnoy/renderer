use std::u64;

use ash::vk;
use bevy_ecs::prelude::*;
use profiling::scope;

use super::super::{device::Semaphore, RenderFrame, Swapchain};
#[cfg(feature = "crash_debugging")]
use crate::renderer::CrashBuffer;
use crate::renderer::{
    as_of, as_of_last, frame_graph, Device, RenderStage, Resized, Submissions, SwapchainIndexToFrameNumber,
};

pub(crate) struct PresentData {
    framebuffer_acquire_semaphore: Semaphore,
    render_complete_semaphore: Semaphore,
}

#[derive(Default, Clone, Debug, PartialEq)]
pub(crate) struct ImageIndex(pub(crate) u32);

// Acquire swapchain image and store the index
pub(crate) struct AcquireFramebuffer;

pub(crate) struct PresentFramebuffer;

impl PresentData {
    pub(crate) fn new(renderer: &RenderFrame) -> PresentData {
        let framebuffer_acquire_semaphore = renderer.device.new_semaphore();
        renderer
            .device
            .set_object_name(framebuffer_acquire_semaphore.handle, "Framebuffer acquire semaphore");
        let render_complete_semaphore = renderer.device.new_semaphore();
        renderer
            .device
            .set_object_name(render_complete_semaphore.handle, "Render complete semaphore");

        PresentData {
            framebuffer_acquire_semaphore,
            render_complete_semaphore,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.framebuffer_acquire_semaphore.destroy(device);
        self.render_complete_semaphore.destroy(device);
    }
}

pub(crate) fn acquire_framebuffer(
    renderer: Res<RenderFrame>,
    swapchain: Res<Swapchain>,
    mut present_data: ResMut<PresentData>,
    mut image_index: ResMut<ImageIndex>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
) {
    scope!("ecs::AcquireFramebuffer");

    AcquireFramebuffer::exec(
        &renderer,
        &swapchain,
        &mut *present_data,
        &mut *image_index,
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
    );
}

impl AcquireFramebuffer {
    /// Returns true if framebuffer and swapchain need to be recreated
    fn exec(
        renderer: &RenderFrame,
        swapchain: &Swapchain,
        present_data: &mut PresentData,
        image_index: &mut ImageIndex,
        #[cfg(feature = "crash_debugging")] crash_buffer: &CrashBuffer,
    ) {
        scope!("presentation::acquire_framebuffer");
        let result = unsafe {
            scope!("vk::AcquireNextImageKHR");
            swapchain.ext.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                present_data.framebuffer_acquire_semaphore.handle,
                vk::Fence::null(),
            )
        };

        match result {
            Ok((ix, false)) => image_index.0 = ix,
            Ok((ix, true)) => {
                image_index.0 = ix;
                println!("AcquireFramebuffer image suboptimal");
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                panic!("out of date in AcquireFramebuffer");
            }
            _ => panic!("unknown condition in AcquireFramebuffer"),
        }

        #[cfg(debug_assertions)]
        {
            let counter = renderer.auto_semaphores.0
                [<frame_graph::PresentationAcquire::Stage as RenderStage>::SIGNAL_AUTO_SEMAPHORE_IX]
                .value(&renderer.device)
                .unwrap();

            let actual = as_of::<<frame_graph::PresentationAcquire::Stage as RenderStage>::SignalTimelineStage>(
                renderer.frame_number,
            );

            if counter >= actual {
                println!(
                    "AcquireFramebuffer assumption incorrect, counter at {} but expected < {}",
                    counter, actual
                );
            }
        }

        let wait_semaphore_values = &[
            as_of_last::<<frame_graph::Main::Stage as RenderStage>::SignalTimelineStage>(renderer.frame_number),
            0,
        ];
        let signal_semaphore_values = &[as_of::<
            <frame_graph::PresentationAcquire::Stage as RenderStage>::SignalTimelineStage,
        >(renderer.frame_number)];
        let mut wait_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(wait_semaphore_values)
            .signal_semaphore_values(signal_semaphore_values);

        let wait_semaphores = &[
            renderer.auto_semaphores.0[frame_graph::Main::Stage::SIGNAL_AUTO_SEMAPHORE_IX].handle,
            present_data.framebuffer_acquire_semaphore.handle,
        ];
        let queue = renderer.device.graphics_queue().lock();
        let signal_semaphores =
            &[renderer.auto_semaphores.0[frame_graph::PresentationAcquire::Stage::SIGNAL_AUTO_SEMAPHORE_IX].handle];
        let dst_stage_masks = &[vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TOP_OF_PIPE];
        let submit = vk::SubmitInfo::builder()
            .push_next(&mut wait_timeline)
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(dst_stage_masks)
            .signal_semaphores(signal_semaphores)
            .build();

        let result = unsafe { renderer.device.queue_submit(*queue, &[submit], vk::Fence::null()) };

        match result {
            Ok(()) => {}
            Err(res) => {
                #[cfg(feature = "crash_debugging")]
                crash_buffer.dump(&renderer.device);
                panic!("Submit failed, frame={}, error={:?}", renderer.frame_number, res);
            }
        };
    }
}

impl PresentFramebuffer {
    pub(crate) fn exec(
        renderer: Res<RenderFrame>,
        present_data: Res<PresentData>,
        mut swapchain: ResMut<Swapchain>,
        image_index: Res<ImageIndex>,
        mut resized: ResMut<Resized>,
        mut swapchain_index_map: ResMut<SwapchainIndexToFrameNumber>,
        #[cfg(debug_assertions)] mut submissions: ResMut<Submissions>,
    ) {
        scope!("ecs::PresentFramebuffer");

        #[cfg(debug_assertions)]
        {
            let graph = submissions.remaining.get_mut();
            debug_assert_eq!(graph.node_count(), 0);
            debug_assert_eq!(graph.edge_count(), 0);
        }

        let queue = if swapchain.supports_present_from_compute {
            renderer.device.compute_queue_balanced()
        } else {
            renderer.device.graphics_queue().lock()
        };

        {
            scope!("present::first_submit");
            let wait_semaphores = &[vk::SemaphoreSubmitInfoKHR::builder()
                .semaphore(renderer.auto_semaphores.0[frame_graph::Main::Stage::SIGNAL_AUTO_SEMAPHORE_IX].handle)
                .value(as_of::<<frame_graph::Main::Stage as RenderStage>::SignalTimelineStage>(
                    renderer.frame_number,
                ))
                .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                .build()];
            let signal_semaphores = &[vk::SemaphoreSubmitInfoKHR::builder()
                .semaphore(present_data.render_complete_semaphore.handle)
                .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                .build()];
            let submit = vk::SubmitInfo2KHR::builder()
                .wait_semaphore_infos(wait_semaphores)
                .signal_semaphore_infos(signal_semaphores)
                .build();

            unsafe {
                scope!("vk::QueueSubmit");

                renderer
                    .device
                    .synchronization2
                    .queue_submit2(*queue, &[submit], vk::Fence::null())
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

        let result = {
            scope!("vk::QueuePresentKHR");
            unsafe { swapchain.ext.queue_present(*queue, &present_info) }
        };
        drop(queue);
        match result {
            Ok(false) => {
                resized.0 = false;
            }
            Ok(true) => {
                println!("PresentFramebuffer image suboptimal");
                scope!("ecs::resized_wait");
                unsafe {
                    renderer.device.device_wait_idle().unwrap();
                }
                swapchain.resize_to_fit(&renderer.device);
                resized.0 = true;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => panic!("out of date in PresentFramebuffer"),
            Err(vk::Result::ERROR_DEVICE_LOST) => panic!("device lost in PresentFramebuffer"),
            _ => panic!("unknown condition in PresentFramebuffer"),
        }

        swapchain_index_map.map[image_index.0 as usize] = renderer.frame_number;
    }
}
