use crate::{
    renderer::helpers::{self, Pipeline},
    shaders, CameraMatrices, RenderFrame,
};
use ash::vk;
use std::{path::PathBuf, sync::Arc};

pub(crate) struct DebugAABBPassData {
    pub(crate) pipeline_layout: shaders::debug_aabb::PipelineLayout,
    pub(crate) pipeline: Pipeline,
}

impl DebugAABBPassData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        camera_matrices: &CameraMatrices,
    ) -> DebugAABBPassData {
        let device = &renderer.device;

        let pipeline_layout =
            shaders::debug_aabb::PipelineLayout::new(&device, &camera_matrices.set_layout);
        use std::io::Read;
        let path = std::path::PathBuf::from(env!("OUT_DIR")).join("debug_aabb.vert.spv");
        let file = std::fs::File::open(path).expect("Could not find shader.");
        let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
        debug_assert!(shaders::debug_aabb::load_and_verify_spirv(&bytes));
        let pipeline = helpers::new_graphics_pipeline2(
            Arc::clone(&renderer.device),
            &[
                (
                    vk::ShaderStageFlags::VERTEX,
                    PathBuf::from(env!("OUT_DIR")).join("debug_aabb.vert.spv"),
                ),
                (
                    vk::ShaderStageFlags::FRAGMENT,
                    PathBuf::from(env!("OUT_DIR")).join("debug_aabb.frag.spv"),
                ),
            ],
            vk::GraphicsPipelineCreateInfo::builder()
                .vertex_input_state(&shaders::debug_aabb::vertex_input_state())
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
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
                .layout(pipeline_layout.layout.handle)
                .render_pass(renderer.renderpass.handle)
                .subpass(0)
                .build(),
        );

        DebugAABBPassData {
            pipeline,
            pipeline_layout,
        }
    }
}
