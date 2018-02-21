#![allow(unused_variables)]

extern crate ash;
extern crate cgmath;
extern crate forward_renderer;
extern crate futures;
extern crate futures_cpupool;
extern crate gltf;
extern crate petgraph;
extern crate specs;

use forward_renderer::*;
use render_dag::v3::*;

use ash::vk;
use std::default::Default;
use std::ptr;
use std::mem::size_of;
use std::path::PathBuf;

fn main() {
    let mut dag = RenderDAG::new();
    let window_ix = dag.new_window(800, 600);
    let (device_ix, graphics_family, compute_family) = dag.new_device(window_ix).unwrap();
    let swapchain_ix = dag.new_swapchain(device_ix).unwrap();
    let surface_format = match dag.graph[swapchain_ix] {
        RenderNode::Swapchain {
            ref surface_format, ..
        } => Some(surface_format.clone()),
        _ => None,
    }.expect("could not locate surface format");
    let (renderpass_ix, end_renderpass_ix, subpass_ixes) = {
        let attachment_descriptions = [
            vk::AttachmentDescription {
                format: surface_format.format,
                flags: vk::AttachmentDescriptionFlags::empty(),
                samples: vk::SAMPLE_COUNT_1_BIT,
                load_op: vk::AttachmentLoadOp::Clear,
                store_op: vk::AttachmentStoreOp::Store,
                stencil_load_op: vk::AttachmentLoadOp::DontCare,
                stencil_store_op: vk::AttachmentStoreOp::DontCare,
                initial_layout: vk::ImageLayout::Undefined,
                final_layout: vk::ImageLayout::PresentSrcKhr,
            },
        ];
        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::ColorAttachmentOptimal,
        };
        let subpass_descs = [
            vk::SubpassDescription {
                color_attachment_count: 1,
                p_color_attachments: &color_attachment_ref,
                p_depth_stencil_attachment: ptr::null(),
                flags: Default::default(),
                pipeline_bind_point: vk::PipelineBindPoint::Graphics,
                input_attachment_count: 0,
                p_input_attachments: ptr::null(),
                p_resolve_attachments: ptr::null(),
                preserve_attachment_count: 0,
                p_preserve_attachments: ptr::null(),
            },
        ];
        let subpass_dependencies = [
            vk::SubpassDependency {
                dependency_flags: Default::default(),
                src_subpass: vk::VK_SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                src_access_mask: Default::default(),
                dst_access_mask: vk::ACCESS_COLOR_ATTACHMENT_READ_BIT | vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            },
        ];
        dag.new_renderpass(
            device_ix,
            &attachment_descriptions,
            &subpass_descs,
            &subpass_dependencies,
        ).unwrap()
    };
    let (framebuffer_ix, present_ix) = dag.new_framebuffer(swapchain_ix, renderpass_ix).unwrap();
    dag.graph
        .add_edge(renderpass_ix, present_ix, Edge::Propagate);
    let graphics_command_pool_ix = dag.new_command_pool(
        device_ix,
        graphics_family,
        vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    ).unwrap();
    let compute_command_pool_ix = dag.new_command_pool(
        device_ix,
        compute_family,
        vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    ).unwrap();
    let (command_buffer_ix, submit_commands_ix) = dag.new_allocate_command_buffer(graphics_command_pool_ix)
        .unwrap();
    dag.graph
        .add_edge(submit_commands_ix, present_ix, Edge::Propagate);
    dag.graph
        .add_edge(framebuffer_ix, command_buffer_ix, Edge::Propagate);
    dag.graph
        .add_edge(command_buffer_ix, renderpass_ix, Edge::Propagate);
    dag.graph
        .add_edge(end_renderpass_ix, submit_commands_ix, Edge::Direct);

    // Semaphores
    let present_semaphore_ix = dag.new_persistent_semaphore(device_ix).unwrap();
    dag.graph
        .add_edge(framebuffer_ix, present_semaphore_ix, Edge::Direct);
    dag.graph
        .add_edge(present_semaphore_ix, submit_commands_ix, Edge::Direct);
    let rendering_complete_semaphore_ix = dag.new_persistent_semaphore(device_ix).unwrap();
    dag.graph.add_edge(
        submit_commands_ix,
        rendering_complete_semaphore_ix,
        Edge::Direct,
    );
    dag.graph
        .add_edge(rendering_complete_semaphore_ix, present_ix, Edge::Direct);
    let pipeline_layout = dag.new_pipeline_layout(
        device_ix,
        vec![
            vk::PushConstantRange {
                stage_flags: vk::SHADER_STAGE_VERTEX_BIT,
                offset: 0,
                size: (size_of::<f32>() * 2 * 3) as u32,
            },
        ],
    ).unwrap();
    let triangle_pipeline = dag.new_graphics_pipeline(
        pipeline_layout,
        renderpass_ix,
        &[],
        &[],
        &[
            (
                vk::SHADER_STAGE_VERTEX_BIT,
                PathBuf::from(env!("OUT_DIR")).join("triangle.vert.spv"),
            ),
            (
                vk::SHADER_STAGE_FRAGMENT_BIT,
                PathBuf::from(env!("OUT_DIR")).join("triangle.frag.spv"),
            ),
        ],
    ).unwrap();
    dag.graph
        .add_edge(triangle_pipeline, end_renderpass_ix, Edge::Propagate);
    println!("{}", dot(&dag.graph).unwrap());
    for _i in 1..5 {
        dag.render_frame();
        if let RenderNode::PresentFramebuffer { ref dynamic, .. } = dag.graph[present_ix] {
            use futures::Future;
            let lock = dynamic.read().unwrap();
            println!("{:?}", (*lock).clone().wait());
        } else {
            panic!("present framebuffer does not exist?")
        }
    }
}
