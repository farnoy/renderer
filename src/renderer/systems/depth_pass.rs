use std::mem::size_of;

use ash::vk;
use bevy_ecs::prelude::*;
use microprofile::scope;

use crate::{
    ecs::systems::RuntimeConfiguration,
    renderer::{
        binding_size, device::Device, frame_graph, helpers::command_util::CommandUtil, CameraMatrices,
        ConsolidatedMeshBuffers, CopiedResource, CullPassData, ImageIndex, MainAttachments, MainRenderpass, ModelData,
        Pipeline, PipelineLayout, RenderFrame, RenderStage, Submissions, Swapchain,
    },
};
pub(crate) struct DepthPassData {
    depth_pipeline: frame_graph::depth_pipe::Pipeline,
    depth_pipeline_layout: frame_graph::depth_pipe::PipelineLayout,
    // TODO: present should not have visibility into this stuff
    pub(super) renderpass: frame_graph::DepthOnly::RenderPass,
    command_util: CommandUtil,
    pub(super) framebuffer: frame_graph::DepthOnly::Framebuffer,
}

impl FromWorld for DepthPassData {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let model_data = world.get_resource::<ModelData>().unwrap();
        let camera_matrices = world.get_resource::<CameraMatrices>().unwrap();
        let swapchain = world.get_resource::<Swapchain>().unwrap();
        let device = &renderer.device;

        let renderpass = frame_graph::DepthOnly::RenderPass::new(renderer, ());

        let depth_pipeline_layout = frame_graph::depth_pipe::PipelineLayout::new(
            &device,
            (&model_data.model_set_layout, &camera_matrices.set_layout),
        );
        let depth_pipeline = frame_graph::depth_pipe::Pipeline::new(
            &device,
            &depth_pipeline_layout,
            frame_graph::depth_pipe::Specialization {},
            (renderpass.renderpass.handle, 0, vk::SampleCountFlags::TYPE_4),
        );

        let framebuffer = frame_graph::DepthOnly::Framebuffer::new(
            renderer,
            &renderpass,
            &[vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT],
            (),
            (swapchain.width, swapchain.height),
        );

        let command_util = CommandUtil::from_world(world);

        DepthPassData {
            depth_pipeline,
            depth_pipeline_layout,
            renderpass,
            command_util,
            framebuffer,
        }
    }
}

impl DepthPassData {
    pub(crate) fn destroy(self, device: &Device) {
        self.depth_pipeline.destroy(device);
        self.depth_pipeline_layout.destroy(device);
        self.renderpass.destroy(device);
        self.command_util.destroy(device);
        self.framebuffer.destroy(device);
    }
}

pub(crate) fn depth_only_pass(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    mut depth_pass: ResMut<DepthPassData>,
    swapchain: Res<Swapchain>,
    main_attachments: Res<MainAttachments>,
    cull_pass_data: Res<CullPassData>,
    consolidated_mesh_buffers: Res<ConsolidatedMeshBuffers>,
    model_data: Res<ModelData>,
    runtime_config: Res<CopiedResource<RuntimeConfiguration>>,
    camera_matrices: Res<CameraMatrices>,
    submissions: Res<Submissions>,
) {
    scope!("rendering", "depth_only_pass");

    let DepthPassData {
        ref renderpass,
        ref framebuffer,
        ref mut command_util,
        ref depth_pipeline,
        ref depth_pipeline_layout,
        ..
    } = *depth_pass;

    let command_buffer = command_util.reset_and_record(&renderer, &image_index);

    let marker = command_buffer.debug_marker_around("depth prepass", [0.3, 0.3, 0.3, 1.0]);

    cull_pass_data
        .culled_commands_buffer
        .acquire_draw_depth(&renderer, *command_buffer);
    cull_pass_data
        .culled_commands_count_buffer
        .acquire_draw_depth(&renderer, *command_buffer);

    renderpass.begin(
        &renderer,
        &framebuffer,
        *command_buffer,
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: swapchain.width,
                height: swapchain.height,
            },
        },
        &[main_attachments.depth_image_view.handle],
        &[vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
        }],
    );

    unsafe {
        if !runtime_config.debug_aabbs {
            renderer.device.cmd_set_viewport(*command_buffer, 0, &[vk::Viewport {
                x: 0.0,
                y: swapchain.height as f32,
                width: swapchain.width as f32,
                height: -(swapchain.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            }]);
            renderer.device.cmd_set_scissor(*command_buffer, 0, &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.width,
                    height: swapchain.height,
                },
            }]);
            renderer.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *depth_pipeline.pipeline,
            );
            depth_pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                (
                    &model_data.model_set.current(image_index.0),
                    &camera_matrices.set.current(image_index.0),
                ),
            );
            renderer.device.cmd_bind_index_buffer(
                *command_buffer,
                cull_pass_data.culled_index_buffer.buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            renderer.device.cmd_bind_vertex_buffers(
                *command_buffer,
                0,
                &[consolidated_mesh_buffers.position_buffer.buffer.handle],
                &[0],
            );
            renderer.device.cmd_draw_indexed_indirect_count(
                *command_buffer,
                cull_pass_data.culled_commands_buffer.buffer.handle,
                0,
                cull_pass_data.culled_commands_count_buffer.buffer.handle,
                0,
                binding_size::<frame_graph::cull_set::bindings::indirect_commands>() as u32
                    / size_of::<frame_graph::VkDrawIndexedIndirectCommand>() as u32,
                size_of::<frame_graph::VkDrawIndexedIndirectCommand>() as u32,
            );
        }

        renderer.device.cmd_end_render_pass(*command_buffer);
    }
    drop(marker);

    let command_buffer = command_buffer.end();

    submissions.submit(
        &renderer,
        &image_index,
        frame_graph::DepthOnly::INDEX,
        Some(*command_buffer),
    );
}
