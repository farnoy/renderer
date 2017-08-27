extern crate ash;
extern crate cgmath;
#[macro_use]
extern crate forward_renderer;
extern crate specs;

use forward_renderer::*;
use ecs::*;
use mesh;

use ash::vk;
use ash::version::*;
use cgmath::One;
use std::default::Default;
use std::ptr;
use std::ffi::CString;
use std::mem;
use std::path::Path;
use std::fs::File;
use std::io::Read;
use ash::util::*;
use std::mem::align_of;

#[derive(Clone, Debug, Copy)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

fn main() {
    let base = ExampleBase::new(1920, 1080);
    unsafe {
        use renderpass::RenderPass;
        let renderpass = renderpass::simple_color::SimpleColor::setup(&base);
        let framebuffers: Vec<vk::Framebuffer> = base.present_image_views
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments = [present_image_view, base.depth_image_view];
                let frame_buffer_create_info = vk::FramebufferCreateInfo {
                    s_type: vk::StructureType::FramebufferCreateInfo,
                    p_next: ptr::null(),
                    flags: Default::default(),
                    render_pass: renderpass.vk(),
                    attachment_count: framebuffer_attachments.len() as u32,
                    p_attachments: framebuffer_attachments.as_ptr(),
                    width: base.surface_resolution.width,
                    height: base.surface_resolution.height,
                    layers: 1,
                };
                base.device
                    .create_framebuffer(&frame_buffer_create_info, None)
                    .unwrap()
            })
            .collect();

        use graphics_pipeline::GraphicsPipeline;
        let graphic_pipeline =
            graphics_pipeline::textured_mesh::TexturedMeshPipeline::new(&base, &renderpass);

        let mut world = World::new(&base.device);

        world
            .create_entity()
            .with::<Position>(Position(cgmath::Vector3::new(0.0, 0.0, 0.0)))
            .with::<Rotation>(Rotation(cgmath::Quaternion::one()))
            .with::<Scale>(Scale(1.0))
            .with::<SimpleColorMesh>(SimpleColorMesh(
                mesh::Mesh::from_gltf(
                    &base,
                    "glTF-Sample-Models/2.0/BoxTextured/glTF/BoxTextured.gltf",
                ).unwrap(),
            ));

        base.render_loop(|| {
            let present_index = base.swapchain_loader
                .acquire_next_image_khr(
                    base.swapchain,
                    std::u64::MAX,
                    base.present_complete_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();
            let framebuffer = framebuffers[present_index as usize];
            record_submit_commandbuffer(base.device.vk(),
                                        base.draw_command_buffer,
                                        base.present_queue,
                                        &[vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
                                        &[base.present_complete_semaphore],
                                        &[base.rendering_complete_semaphore],
                                        |device, draw_command_buffer| {
                renderpass
                    .record_commands(&base,
                    framebuffer,
                 draw_command_buffer,
                 vk::SubpassContents::Inline,
                 |device, command_buffer| {
                     graphic_pipeline.record_commands(&base, command_buffer);
                    // device.cmd_set_viewport(draw_command_buffer, &viewports);
                    // device.cmd_set_scissor(draw_command_buffer, &scissors);

                });
            });
            //let mut present_info_err = mem::uninitialized();
            let present_info = vk::PresentInfoKHR {
                s_type: vk::StructureType::PresentInfoKhr,
                p_next: ptr::null(),
                wait_semaphore_count: 1,
                p_wait_semaphores: &base.rendering_complete_semaphore,
                swapchain_count: 1,
                p_swapchains: &base.swapchain,
                p_image_indices: &present_index,
                p_results: ptr::null_mut(),
            };
            base.swapchain_loader
                .queue_present_khr(base.present_queue, &present_info)
                .unwrap();
        });

        base.device.device_wait_idle().unwrap();
        for framebuffer in framebuffers {
            base.device.destroy_framebuffer(framebuffer, None);
        }
        base.device.destroy_render_pass(renderpass.vk(), None);
    }
}
