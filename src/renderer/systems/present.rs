use super::super::{
    device::{CommandBuffer, DoubleBuffered, Fence, Semaphore},
    RenderFrame, Swapchain,
};
use ash::{version::DeviceV1_0, vk};
#[cfg(feature = "microprofile")]
use microprofile::scope;
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

impl PresentData {
    pub fn new(renderer: &RenderFrame) -> PresentData {
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

        PresentData {
            render_complete_semaphore,
            present_semaphore,
            render_command_buffer,
            render_complete_fence,
        }
    }
}

impl AcquireFramebuffer {
    /// Returns true if framebuffer and swapchain need to be recreated
    pub fn exec(
        renderer: &RenderFrame,
        present_data: &PresentData,
        swapchain: &Swapchain,
        image_index: &mut ImageIndex,
    ) -> bool {
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
        let result = unsafe {
            swapchain.ext.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                present_data.present_semaphore.handle,
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

        false
    }
}

impl PresentFramebuffer {
    pub fn exec(
        renderer: &RenderFrame,
        present_data: &PresentData,
        swapchain: &Swapchain,
        image_index: &ImageIndex,
    ) {
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
    }
}
