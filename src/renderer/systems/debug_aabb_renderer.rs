use super::super::{
    helpers::{self, Pipeline},
    shaders, CameraMatrices, RenderFrame,
};
use ash::vk;
#[cfg(feature = "microprofile")]
use microprofile::scope;
use std::{path::PathBuf, sync::Arc};

// Render AABB outlines
pub struct DebugAABBPass;

pub struct DebugAABBPassData {
    pub pipeline_layout: shaders::debug_aabb::PipelineLayout,
    pub pipeline: Pipeline,
}

impl DebugAABBPassData {
    pub fn new(renderer: &RenderFrame, camera_matrices: &CameraMatrices) -> DebugAABBPassData {
        let device = &renderer.device;

        let pipeline_layout =
            shaders::debug_aabb::PipelineLayout::new(&device, &camera_matrices.set_layout);
        use std::io::Read;
        let path = std::path::PathBuf::from(env!("OUT_DIR")).join("debug_aabb.vert.spv");
        let file = std::fs::File::open(path).expect("Could not find shader.");
        let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
        let module = spirv_reflect::create_shader_module(&bytes).unwrap();
        debug_assert!(shaders::debug_aabb::verify_spirv(&module));
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
                        .cull_mode(vk::CullModeFlags::BACK)
                        .front_face(vk::FrontFace::CLOCKWISE)
                        .line_width(1.0)
                        .polygon_mode(vk::PolygonMode::LINE)
                        .build(),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                        .build(),
                )
                .depth_stencil_state(
                    &vk::PipelineDepthStencilStateCreateInfo::builder()
                        .depth_test_enable(true)
                        .depth_write_enable(true)
                        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                        .depth_bounds_test_enable(false)
                        .max_depth_bounds(1.0)
                        .min_depth_bounds(0.0)
                        .build(),
                )
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

/*
impl DebugAABBPass {
    #[allow(clippy::too_many_arguments)]
    pub fn exec(
        entities: &EntitiesStorage,
        renderer: &RenderFrame,
        command_buffer: vk::CommandBuffer,
        debug_aabb_pass_data: &DebugAABBPassData,
        aabbs: &ComponentStorage<ncollide3d::bounding_volume::AABB<f32>>,
        image_index: &ImageIndex,
        camera_matrices: &CameraMatrices,
    ) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "debug aabb pass");

        renderer.device.debug_marker_around(
            command_buffer,
            "aabb debug",
            [1.0, 0.0, 0.0, 1.0],
            || {
                unsafe {
                    renderer.device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        debug_aabb_pass_data.pipeline.handle,
                    );
                }
                debug_aabb_pass_data.pipeline_layout.bind_descriptor_sets(
                    &renderer.device,
                    command_buffer,
                    &camera_matrices.set.current(image_index.0),
                );

                for entity_id in (entities.mask() & aabbs.mask()).iter() {
                    let aabb = aabbs.get(entity_id).unwrap();
                    debug_aabb_pass_data.pipeline_layout.push_constants(
                        &renderer.device,
                        command_buffer,
                        &shaders::DebugAABBPushConstants {
                            center: aabb.center().to_homogeneous(),
                            half_extent: aabb.half_extents().push(1.0),
                        },
                    );
                    unsafe {
                        renderer.device.cmd_draw(command_buffer, 36, 1, 0, 0);
                    }
                }
            },
        );
    }
}
*/
