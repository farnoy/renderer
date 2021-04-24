use crate::{
    renderer::device::{Device, Pipeline},
    shaders, CameraMatrices, MainRenderpass, RenderFrame,
};
use ash::vk;
use bevy_ecs::prelude::*;

pub(crate) struct DebugAABBPassData {
    pub(crate) pipeline_layout: shaders::debug_aabb::PipelineLayout,
    pub(crate) pipeline: Pipeline,
}

impl FromWorld for DebugAABBPassData {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let main_renderpass = world.get_resource::<MainRenderpass>().unwrap();
        let camera_matrices = world.get_resource::<CameraMatrices>().unwrap();
        let device = &renderer.device;

        let pipeline_layout = shaders::debug_aabb::PipelineLayout::new(&device, &camera_matrices.set_layout);
        let pipeline = renderer.device.new_graphics_pipeline(
            &[
                (vk::ShaderStageFlags::VERTEX, shaders::debug_aabb::VERTEX, None),
                (vk::ShaderStageFlags::FRAGMENT, shaders::debug_aabb::FRAGMENT, None),
            ],
            vk::GraphicsPipelineCreateInfo::builder()
                .vertex_input_state(&shaders::debug_aabb::vertex_input_state())
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                )
                .dynamic_state(
                    &vk::PipelineDynamicStateCreateInfo::builder()
                        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
                )
                .viewport_state(
                    &vk::PipelineViewportStateCreateInfo::builder()
                        .viewport_count(1)
                        .scissor_count(1)
                        .build(),
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::builder()
                        .cull_mode(vk::CullModeFlags::NONE)
                        .line_width(1.0)
                        .polygon_mode(vk::PolygonMode::LINE)
                        .build(),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                        .build(),
                )
                .depth_stencil_state(&vk::PipelineDepthStencilStateCreateInfo::builder().build())
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::builder()
                        .attachments(&[vk::PipelineColorBlendAttachmentState::builder()
                            .blend_enable(true)
                            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .color_blend_op(vk::BlendOp::ADD)
                            .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                            .alpha_blend_op(vk::BlendOp::ADD)
                            .color_write_mask(vk::ColorComponentFlags::all())
                            .build()])
                        .build(),
                )
                .layout(*pipeline_layout.layout)
                .render_pass(main_renderpass.renderpass.renderpass.handle)
                .subpass(1) // FIXME
                .build(),
        );

        DebugAABBPassData {
            pipeline_layout,
            pipeline,
        }
    }
}

impl DebugAABBPassData {
    pub(crate) fn destroy(self, device: &Device) {
        self.pipeline.destroy(device);
        self.pipeline_layout.destroy(device);
    }
}
