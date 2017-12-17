use ash::version::DeviceV1_0;
use ash::vk;
use std::ffi::CString;
use std::ptr;
use std::mem;
use std::path::Path;
use std::fs::File;
use std::io::Read;

use super::super::ExampleBase;
use super::super::renderpass::RenderPass;
use super::GraphicsPipeline;

#[derive(Clone, Debug, Copy)]
struct Vertex {
    pos: [f32; 4],
    // color: [f32; 4],
}

pub struct TexturedMeshPipeline {
    handle: vk::Pipeline,
}

macro_rules! offset_of{
    ($base: path, $field: ident) => {
        {
            #[allow(unused_unsafe)]
            unsafe{
                let b: $base = mem::uninitialized();
                (&b.$field as *const _ as isize) - (&b as *const _ as isize)
            }
        }
    }
}

impl GraphicsPipeline for TexturedMeshPipeline {
    fn new<R: RenderPass>(base: &ExampleBase, renderpass: &R) -> TexturedMeshPipeline {
        let vertex_spv_file = File::open(Path::new(env!("OUT_DIR")).join("simple_color.vert.spv")).expect("Could not find vertex shader.");
        let frag_spv_file = File::open(Path::new(env!("OUT_DIR")).join("simple_color.frag.spv")).expect("Could not find fragment shader.");

        let vertex_bytes: Vec<u8> = vertex_spv_file
            .bytes()
            .filter_map(|byte| byte.ok())
            .collect();
        let vertex_shader_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::ShaderModuleCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            code_size: vertex_bytes.len(),
            p_code: vertex_bytes.as_ptr() as *const u32,
        };
        let frag_bytes: Vec<u8> = frag_spv_file.bytes().filter_map(|byte| byte.ok()).collect();
        let frag_shader_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::ShaderModuleCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            code_size: frag_bytes.len(),
            p_code: frag_bytes.as_ptr() as *const u32,
        };
        let vertex_shader_module = unsafe {
            base.device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error")
        };

        let fragment_shader_module = unsafe {
            base.device
                .create_shader_module(&frag_shader_info, None)
                .expect("Fragment shader module error")
        };

        let layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PipelineLayoutCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            set_layout_count: 0,
            p_set_layouts: ptr::null(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
        };

        let pipeline_layout = unsafe {
            base.device
                .create_pipeline_layout(&layout_create_info, None)
                .unwrap()
        };

        let shader_entry_name = CString::new("main").unwrap();
        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: ptr::null(),
                flags: Default::default(),
                module: vertex_shader_module,
                p_name: shader_entry_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: vk::SHADER_STAGE_VERTEX_BIT,
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                p_next: ptr::null(),
                flags: Default::default(),
                module: fragment_shader_module,
                p_name: shader_entry_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: vk::SHADER_STAGE_FRAGMENT_BIT,
            },
        ];
        let vertex_input_binding_descriptions = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::Vertex,
            },
        ];
        let vertex_input_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32g32b32a32Sfloat,
                offset: offset_of!(Vertex, pos) as u32,
            }, /*
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32g32b32a32Sfloat,
                offset: offset_of!(Vertex, color) as u32,
            },
            */
        ];
        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PipelineVertexInputStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            vertex_attribute_description_count: vertex_input_attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: vertex_input_attribute_descriptions.as_ptr(),
            vertex_binding_description_count: vertex_input_binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: vertex_input_binding_descriptions.as_ptr(),
        };
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PipelineInputAssemblyStateCreateInfo,
            flags: Default::default(),
            p_next: ptr::null(),
            primitive_restart_enable: 0,
            topology: vk::PrimitiveTopology::TriangleList,
        };
        let viewports = [
            vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: base.surface_resolution.width as f32,
                height: base.surface_resolution.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        ];
        let scissors = [
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: base.surface_resolution.clone(),
            },
        ];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PipelineViewportStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
        };
        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PipelineRasterizationStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            cull_mode: vk::CULL_MODE_NONE,
            depth_bias_clamp: 0.0,
            depth_bias_constant_factor: 0.0,
            depth_bias_enable: 0,
            depth_bias_slope_factor: 0.0,
            depth_clamp_enable: 0,
            front_face: vk::FrontFace::CounterClockwise,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::Fill,
            rasterizer_discard_enable: 0,
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PipelineMultisampleStateCreateInfo,
            flags: Default::default(),
            p_next: ptr::null(),
            rasterization_samples: vk::SAMPLE_COUNT_1_BIT,
            sample_shading_enable: 0,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_one_enable: 0,
            alpha_to_coverage_enable: 0,
        };
        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::Keep,
            pass_op: vk::StencilOp::Keep,
            depth_fail_op: vk::StencilOp::Keep,
            compare_op: vk::CompareOp::Always,
            compare_mask: 0,
            write_mask: 0,
            reference: 0,
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PipelineDepthStencilStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::LessOrEqual,
            depth_bounds_test_enable: 0,
            stencil_test_enable: 0,
            front: noop_stencil_state.clone(),
            back: noop_stencil_state.clone(),
            max_depth_bounds: 1.0,
            min_depth_bounds: 0.0,
        };
        let color_blend_attachment_states = [
            vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SrcColor,
                dst_color_blend_factor: vk::BlendFactor::OneMinusDstColor,
                color_blend_op: vk::BlendOp::Add,
                src_alpha_blend_factor: vk::BlendFactor::Zero,
                dst_alpha_blend_factor: vk::BlendFactor::Zero,
                alpha_blend_op: vk::BlendOp::Add,
                color_write_mask: vk::ColorComponentFlags::all(),
            },
        ];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PipelineColorBlendStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            logic_op_enable: 0,
            logic_op: vk::LogicOp::Clear,
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };
        let dynamic_state = [vk::DynamicState::Viewport, vk::DynamicState::Scissor];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PipelineDynamicStateCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            dynamic_state_count: dynamic_state.len() as u32,
            p_dynamic_states: dynamic_state.as_ptr(),
        };
        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GraphicsPipelineCreateInfo,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stage_create_infos.len() as u32,
            p_stages: shader_stage_create_infos.as_ptr(),
            p_vertex_input_state: &vertex_input_state_info,
            p_input_assembly_state: &vertex_input_assembly_state_info,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state_info,
            p_rasterization_state: &rasterization_info,
            p_multisample_state: &multisample_state_info,
            p_depth_stencil_state: &depth_state_info,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state_info,
            layout: pipeline_layout,
            render_pass: renderpass.vk(),
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
        };
        let graphics_pipelines = unsafe {
            base.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[graphic_pipeline_info], None)
                .expect("Unable to create graphics pipeline")
        };

        TexturedMeshPipeline {
            handle: graphics_pipelines[0],
        }
    }

    fn record_commands(&self, base: &ExampleBase, command_buffer: vk::CommandBuffer) {
        unsafe {
            base.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::Graphics, self.handle);
        }
    }

    fn vk(&self) -> vk::Pipeline {
        self.handle
    }
}
