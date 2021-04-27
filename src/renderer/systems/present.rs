use std::{mem::swap, u64};

use ash::{version::DeviceV1_0, vk};
use bevy_ecs::prelude::*;
use microprofile::scope;

use super::super::{device::Semaphore, GraphicsTimeline, RenderFrame, Swapchain};
use crate::renderer::{Device, MainAttachments, MainFramebuffer, MainRenderpass, Resized, SwapchainIndexToFrameNumber};

pub(crate) struct PresentData {
    framebuffer_acquire_semaphore: Semaphore,
    render_complete_semaphore: Semaphore,
}

#[derive(Clone, Debug, PartialEq)]
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
    main_renderpass: Res<MainRenderpass>,
    mut swapchain: ResMut<Swapchain>,
    mut main_attachments: ResMut<MainAttachments>,
    mut main_framebuffer: ResMut<MainFramebuffer>,
    mut present_data: ResMut<PresentData>,
    mut image_index: ResMut<ImageIndex>,
    resized: Res<Resized>,
) {
    scope!("ecs", "AcquireFramebuffer");

    if resized.0 {
        scope!("ecs", "recreate framebuffer from resize");
        unsafe {
            renderer.device.device_wait_idle().unwrap();
        }
        swapchain.resize_to_fit(&renderer.device);
        let mut new_attachments = MainAttachments::new(&renderer, &swapchain);
        let mut new_framebuffer = MainFramebuffer::new(&renderer, &main_renderpass, &new_attachments, &swapchain);
        swap(&mut *main_attachments, &mut new_attachments);
        swap(&mut *main_framebuffer, &mut new_framebuffer);
        new_attachments.destroy(&renderer.device);
        new_framebuffer.destroy(&renderer.device);
    }

    let swapchain_needs_recreating =
        AcquireFramebuffer::exec(&renderer, &swapchain, &mut *present_data, &mut *image_index);

    if swapchain_needs_recreating {
        scope!("ecs", "recreate framebuffer from out of date");
        unsafe {
            renderer.device.device_wait_idle().unwrap();
        }
        swapchain.resize_to_fit(&renderer.device);
        let mut new_attachments = MainAttachments::new(&renderer, &swapchain);
        let mut new_framebuffer = MainFramebuffer::new(&renderer, &main_renderpass, &new_attachments, &swapchain);
        swap(&mut *main_attachments, &mut new_attachments);
        swap(&mut *main_framebuffer, &mut new_framebuffer);
        new_attachments.destroy(&renderer.device);
        new_framebuffer.destroy(&renderer.device);
        AcquireFramebuffer::exec(&renderer, &swapchain, &mut *present_data, &mut *image_index);
    }
}

impl AcquireFramebuffer {
    /// Returns true if framebuffer and swapchain need to be recreated
    fn exec(
        renderer: &RenderFrame,
        swapchain: &Swapchain,
        present_data: &mut PresentData,
        image_index: &mut ImageIndex,
    ) -> bool {
        scope!("presentation", "acquire framebuffer");
        let result = unsafe {
            scope!("presentation", "vkAcquireNextImageKHR");
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
                println!("out of date in AcquireFramebuffer");
                return true;
            }
            _ => panic!("unknown condition in AcquireFramebuffer"),
        }

        debug_assert!(
            {
                let counter = renderer.graphics_timeline_semaphore.value(&renderer.device).unwrap();

                counter < GraphicsTimeline::Start.as_of(renderer.frame_number)
            },
            "AcquireFramebuffer assumption incorrect"
        );
        let wait_semaphore_values = &[GraphicsTimeline::SceneDraw.as_of_last(renderer.frame_number), 0];
        let signal_semaphore_values = &[GraphicsTimeline::Start.as_of(renderer.frame_number)];
        let mut wait_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(wait_semaphore_values)
            .signal_semaphore_values(signal_semaphore_values);

        let wait_semaphores = &[
            renderer.graphics_timeline_semaphore.handle,
            present_data.framebuffer_acquire_semaphore.handle,
        ];
        let queue = renderer.device.graphics_queue().lock();
        let signal_semaphores = &[renderer.graphics_timeline_semaphore.handle];
        let dst_stage_masks = &[vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TOP_OF_PIPE];
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
        renderer: Res<RenderFrame>,
        present_data: Res<PresentData>,
        swapchain: Res<Swapchain>,
        image_index: Res<ImageIndex>,
        mut swapchain_index_map: ResMut<SwapchainIndexToFrameNumber>,
    ) {
        scope!("ecs", "PresentFramebuffer");

        {
            let wait_values = &[GraphicsTimeline::SceneDraw.as_of(renderer.frame_number)];

            // TODO: only needed to avoid segfault in vk validation layers
            //       https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2662
            let signal_values = &[GraphicsTimeline::SceneDraw.as_of(0)];
            let mut wait_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
                .wait_semaphore_values(wait_values)
                .signal_semaphore_values(signal_values);

            let wait_semaphores = &[renderer.graphics_timeline_semaphore.handle];
            let queue = renderer.device.graphics_queue().lock();
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

        let queue = renderer.device.graphics_queue().lock();
        let result = unsafe { swapchain.ext.queue_present(*queue, &present_info) };
        match result {
            Ok(false) => (),
            Ok(true) => println!("PresentFramebuffer image suboptimal"),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => println!("out of date in PresentFramebuffer"),
            Err(vk::Result::ERROR_DEVICE_LOST) => panic!("device lost in PresentFramebuffer"),
            _ => panic!("unknown condition in PresentFramebuffer"),
        }
        drop(queue);

        swapchain_index_map.map[image_index.0 as usize] = renderer.frame_number;
    }
}
