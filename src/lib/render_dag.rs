use ash::version::DeviceV1_0;
use ash::vk;
// use futures::future::Shared;
use petgraph;
use std::collections::HashMap;
use std::fmt;
use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;

use ecs::World;
use super::ExampleBase;

#[derive(Clone)]
pub enum Node {
    AcquirePresentImage,
    SwapchainAttachment(u8),
    DepthAttachment(u8),
    Subpass(u8),
    RenderPass,
    VertexShader(PathBuf),
    FragmentShader(PathBuf),
    /// Binding, type, stage, count
    DescriptorBinding(u32, vk::DescriptorType, vk::ShaderStageFlags, u32),
    /// Count
    DescriptorSet(u32),
    PipelineLayout,
    /// Binding, stride, rate
    VertexInputBinding(u32, u32, vk::VertexInputRate),
    /// Binding, location, format, offset
    VertexInputAttribute(u32, u32, vk::Format, u32),
    GraphicsPipeline,
    DrawCommands(Arc<Fn(&ExampleBase, &World, &RenderDAG, vk::CommandBuffer)>),
    PresentImage,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Node")
    }
}

// type NodeResult<T> = Arc<Shared<CpuFuture<T, ()>>>;

#[derive(Clone)]
pub enum NodeRuntime {
    BeginRenderPass(vk::RenderPass),
    BeginSubPass(u8),
    BindPipeline(vk::Pipeline, Option<vk::Rect2D>, Option<vk::Viewport>),
    DrawCommands(Arc<Fn(&ExampleBase, &World, &RenderDAG, vk::CommandBuffer)>),
    EndSubPass(u8), // TODO: we should only have BeginSubPass if possible, to model vulkan
    EndRenderPass(vk::RenderPass),
}

impl fmt::Debug for NodeRuntime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NodeRuntime")
    }
}

type BuilderGraph = petgraph::Graph<(&'static str, Node), ()>;
type RuntimeGraph = petgraph::Graph<(&'static str, NodeRuntime), ()>;

pub struct RenderDAG {
    pub graph: RuntimeGraph,
    pub pipeline_layouts: HashMap<&'static str, vk::PipelineLayout>,
    pub descriptor_sets: HashMap<&'static str, Vec<vk::DescriptorSet>>,
    pub renderpasses: HashMap<&'static str, vk::RenderPass>,
}

impl RenderDAG {
    pub fn run(&self, base: &ExampleBase, world: &World, framebuffer: vk::Framebuffer, command_buffer: vk::CommandBuffer) {
        /*
        let pool = CpuPool::new_num_cpus();

        let mut g: petgraph::Graph<Option<Shared<CpuFuture<ExecResult, ()>>>, _> =
            self.graph.map(|_ix, _node| None, |_ix, edge| edge);
            */
        for node in petgraph::algo::toposort(&self.graph, None)
            .expect("RenderDAG has cycles")
            .iter()
            .cloned()
        {
            let inputs = self.graph
                .neighbors_directed(node, petgraph::EdgeDirection::Incoming)
                .map(|ix| self.graph[ix].clone())
                .collect::<Vec<_>>();
            match &self.graph[node].1 {
                &NodeRuntime::BeginRenderPass(renderpass) => base.device.debug_marker_around(
                    command_buffer,
                    &format!("{} -> BeginRenderPass", self.graph[node].0),
                    [0.0; 4],
                    || {
                        let clear_values = [
                            vk::ClearValue::new_color(vk::ClearColorValue::new_float32([0.0, 0.0, 0.0, 0.0])),
                            vk::ClearValue::new_depth_stencil(vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            }),
                        ];

                        let render_pass_begin_info = vk::RenderPassBeginInfo {
                            s_type: vk::StructureType::RenderPassBeginInfo,
                            p_next: ptr::null(),
                            render_pass: renderpass,
                            framebuffer: framebuffer,
                            render_area: vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: base.surface_resolution.clone(),
                            },
                            clear_value_count: clear_values.len() as u32,
                            p_clear_values: clear_values.as_ptr(),
                        };

                        unsafe {
                            base.device.cmd_begin_render_pass(
                                command_buffer,
                                &render_pass_begin_info,
                                vk::SubpassContents::Inline,
                            );
                        }
                    },
                ),
                &NodeRuntime::BeginSubPass(_ix) => {
                    base.device.debug_marker_around(
                        command_buffer,
                        &format!("{} -> BeginSubPass", self.graph[node].0),
                        [0.0; 4],
                        || {
                            let previous_subpass = inputs.iter().find(|i| match i.1 {
                                NodeRuntime::EndSubPass(_) => true,
                                _ => false,
                            });

                            if previous_subpass.is_some() {
                                unsafe {
                                    base.device
                                        .cmd_next_subpass(command_buffer, vk::SubpassContents::Inline);
                                }
                            }
                        },
                    );
                }
                &NodeRuntime::BindPipeline(pipeline, ref scissors_opt, ref viewport_opt) => unsafe {
                    base.device.debug_marker_around(
                        command_buffer,
                        &format!("{} -> BindPipeline", self.graph[node].0),
                        [0.0; 4],
                        || {
                            base.device
                                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::Graphics, pipeline);

                            if let &Some(ref viewport) = viewport_opt {
                                base.device
                                    .cmd_set_viewport(command_buffer, &[viewport.clone()]);
                            }
                            if let &Some(ref scissors) = scissors_opt {
                                base.device
                                    .cmd_set_scissor(command_buffer, &[scissors.clone()]);
                            }
                        },
                    );
                },
                &NodeRuntime::DrawCommands(ref f) => base.device.debug_marker_around(
                    command_buffer,
                    &format!("{} -> DrawCommands", self.graph[node].0),
                    [0.0; 4],
                    || f(base, world, &self, command_buffer),
                ),
                &NodeRuntime::EndSubPass(_ix) => (),
                &NodeRuntime::EndRenderPass(_renderpass) => base.device.debug_marker_around(
                    command_buffer,
                    &format!("{} -> EndRenderPass", self.graph[node].0),
                    [0.0; 4],
                    || unsafe { base.device.cmd_end_render_pass(command_buffer) },
                ),
            }

            /*
            let future = {
                let inputs = g.neighbors_directed(node, petgraph::EdgeDirection::Incoming)
                    .map(|ix| {
                        g[ix]
                            .clone()
                            .expect("previous computation not available")
                            .clone()
                    })
                    .map(|fut| fut.wait().expect("Future failed"))
                    .map(|shared_item| *shared_item)
                    .collect();
                pool.spawn_fn::<_, Result<ExecResult, ()>>(move || Ok(f(inputs)))
                    .shared()
            };
            if self.graph[node].0 == "end" {
                return *future.wait().expect("computation failed");
            } else {
                g[node] = Some(future);
            }
            */
        }
        // panic!("no end node!");
    }
}

pub struct RenderDAGBuilder {
    pub graph: BuilderGraph,
    name_mapping: HashMap<&'static str, petgraph::graph::NodeIndex>,
}

impl RenderDAGBuilder {
    pub fn new() -> RenderDAGBuilder {
        RenderDAGBuilder {
            graph: BuilderGraph::new(),
            name_mapping: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, name: &'static str, value: Node) {
        let ix = self.graph.add_node((name, value));
        assert!(self.name_mapping.insert(name, ix).is_none());
    }

    pub fn add_edge(&mut self, from: &'static str, to: &'static str) {
        let from_ix = self.name_mapping.get(from).unwrap();
        let to_ix = self.name_mapping.get(to).unwrap();
        self.graph.add_edge(from_ix.clone(), to_ix.clone(), ());
    }

    pub fn build(self, base: &ExampleBase) -> RenderDAG {
        let mut output_graph = RuntimeGraph::new();
        let mut renderpasses: HashMap<
            &str,
            (
                petgraph::graph::NodeIndex,
                petgraph::graph::NodeIndex,
                vk::RenderPass,
            ),
        > = HashMap::new();
        let mut subpasses: HashMap<&str, (petgraph::graph::NodeIndex, petgraph::graph::NodeIndex, u8)> = HashMap::new();
        let mut pipeline_layouts: HashMap<&str, vk::PipelineLayout> = HashMap::new();
        let mut pipelines: HashMap<&str, (petgraph::graph::NodeIndex, vk::Pipeline)> = HashMap::new();
        let mut descriptor_set_layouts: HashMap<&str, vk::DescriptorSetLayout> = HashMap::new();
        let mut descriptor_sets: HashMap<&str, Vec<vk::DescriptorSet>> = HashMap::new();
        for node in petgraph::algo::toposort(&self.graph, None)
            .expect("RenderDAGBuilder has cycles")
            .iter()
            .cloned()
        {
            println!("{:?}", self.graph[node]);
            let inputs = self.graph
                .neighbors_directed(node, petgraph::EdgeDirection::Incoming)
                .map(|ix| self.graph[ix].clone())
                .collect::<Vec<_>>();
            match self.graph[node].1 {
                Node::RenderPass => {
                    let mut attachments = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            Node::SwapchainAttachment(ix) => Some((
                                ix,
                                vk::AttachmentDescription {
                                    format: base.surface_format.format,
                                    flags: vk::AttachmentDescriptionFlags::empty(),
                                    samples: vk::SAMPLE_COUNT_1_BIT,
                                    load_op: vk::AttachmentLoadOp::Clear,
                                    store_op: vk::AttachmentStoreOp::Store,
                                    stencil_load_op: vk::AttachmentLoadOp::DontCare,
                                    stencil_store_op: vk::AttachmentStoreOp::DontCare,
                                    initial_layout: vk::ImageLayout::Undefined,
                                    final_layout: vk::ImageLayout::PresentSrcKhr,
                                },
                            )),
                            Node::DepthAttachment(ix) => Some((
                                ix,
                                vk::AttachmentDescription {
                                    format: vk::Format::D16Unorm,
                                    flags: vk::AttachmentDescriptionFlags::empty(),
                                    samples: vk::SAMPLE_COUNT_1_BIT,
                                    load_op: vk::AttachmentLoadOp::Clear,
                                    store_op: vk::AttachmentStoreOp::DontCare,
                                    stencil_load_op: vk::AttachmentLoadOp::DontCare,
                                    stencil_store_op: vk::AttachmentStoreOp::DontCare,
                                    initial_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
                                    final_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
                                },
                            )),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    attachments.sort_by(|&(lhs, _), &(rhs, _)| lhs.cmp(&rhs));
                    let attachments = attachments
                        .iter()
                        .map(|&(_, ref desc)| desc)
                        .cloned()
                        .collect::<Vec<_>>();
                    let color_attachment_ref = vk::AttachmentReference {
                        attachment: 0,
                        layout: vk::ImageLayout::ColorAttachmentOptimal,
                    };
                    let depth_attachment_ref = vk::AttachmentReference {
                        attachment: 1,
                        layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
                    };
                    let mut subpass_descs = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            Node::Subpass(ix) => Some((
                                ix,
                                vk::SubpassDescription {
                                    color_attachment_count: 1,
                                    p_color_attachments: &color_attachment_ref,
                                    p_depth_stencil_attachment: &depth_attachment_ref,
                                    flags: Default::default(),
                                    pipeline_bind_point: vk::PipelineBindPoint::Graphics,
                                    input_attachment_count: 0,
                                    p_input_attachments: ptr::null(),
                                    p_resolve_attachments: ptr::null(),
                                    preserve_attachment_count: 0,
                                    p_preserve_attachments: ptr::null(),
                                },
                            )),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    subpass_descs.sort_by(|&(lhs, _), &(rhs, _)| lhs.cmp(&rhs));
                    let subpass_descs = subpass_descs
                        .iter()
                        .map(|&(_, ref desc)| desc)
                        .cloned()
                        .collect::<Vec<_>>();
                    let mut dependencies = vec![
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
                    // TODO: enhance with graphs
                    for (ix, _subpass) in subpass_descs.iter().enumerate().skip(1) {
                        dependencies.push(vk::SubpassDependency {
                            dependency_flags: Default::default(),
                            src_subpass: ix as u32 - 1,
                            dst_subpass: ix as u32,
                            src_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            src_access_mask: Default::default(),
                            dst_access_mask: vk::ACCESS_COLOR_ATTACHMENT_READ_BIT | vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            dst_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        });
                    }
                    let renderpass_create_info = vk::RenderPassCreateInfo {
                        s_type: vk::StructureType::RenderPassCreateInfo,
                        flags: Default::default(),
                        p_next: ptr::null(),
                        attachment_count: attachments.len() as u32,
                        p_attachments: attachments.as_ptr(),
                        subpass_count: subpass_descs.len() as u32,
                        p_subpasses: subpass_descs.as_ptr(),
                        dependency_count: dependencies.len() as u32,
                        p_dependencies: dependencies.as_ptr(),
                    };
                    let renderpass = unsafe {
                        base.device
                            .vk()
                            .create_render_pass(&renderpass_create_info, None)
                            .unwrap()
                    };

                    let subpasses = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref _name, Node::Subpass(_)) => subpasses.get(node.0),
                            _ => None,
                        })
                        .collect::<Vec<_>>();

                    let start = output_graph.add_node((
                        "begin_render_pass",
                        NodeRuntime::BeginRenderPass(renderpass),
                    ));
                    let end = output_graph.add_node(("end_render_pass", NodeRuntime::EndRenderPass(renderpass)));
                    for &(start_subpass, end_subpass, _) in subpasses {
                        output_graph.add_edge(start, start_subpass, ());
                        output_graph.add_edge(end_subpass, end, ());
                    }
                    output_graph.add_edge(start, end, ());

                    renderpasses.insert(self.graph[node].0, (start, end, renderpass));
                }
                Node::Subpass(ix) => {
                    let previous_subpasses = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref _name, Node::Subpass(_)) => Some(subpasses.get(node.0).unwrap()),
                            _ => None,
                        })
                        .cloned()
                        .collect::<Vec<_>>();

                    let start = output_graph.add_node(("begin_subpass", NodeRuntime::BeginSubPass(ix)));
                    let end = output_graph.add_node(("end_subpass", NodeRuntime::EndSubPass(ix)));
                    output_graph.add_edge(start, end, ());
                    for (_, end_subpass, _) in previous_subpasses {
                        output_graph.add_edge(end_subpass, start, ());
                    }

                    subpasses.insert(self.graph[node].0, (start, end, ix));
                }
                Node::PipelineLayout => {
                    let set_layouts = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            Node::DescriptorSet(_) => Some(descriptor_set_layouts.get(node.0).unwrap().clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();

                    let create_info = vk::PipelineLayoutCreateInfo {
                        s_type: vk::StructureType::PipelineLayoutCreateInfo,
                        p_next: ptr::null(),
                        flags: Default::default(),
                        set_layout_count: set_layouts.len() as u32,
                        p_set_layouts: set_layouts.as_ptr(),
                        push_constant_range_count: 0,
                        p_push_constant_ranges: ptr::null(),
                    };

                    let pipeline_layout = unsafe {
                        base.device
                            .create_pipeline_layout(&create_info, None)
                            .unwrap()
                    };

                    pipeline_layouts.insert(self.graph[node].0, pipeline_layout);
                }
                Node::GraphicsPipeline => {
                    let vertex_attributes = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            Node::VertexInputAttribute(binding, location, format, offset) => Some(vk::VertexInputAttributeDescription {
                                location: location,
                                binding: binding,
                                format: format,
                                offset: offset,
                            }),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let vertex_bindings = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            Node::VertexInputBinding(binding, stride, rate) => Some(vk::VertexInputBindingDescription {
                                binding: binding,
                                stride: stride,
                                input_rate: rate,
                            }),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let shader_modules = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            Node::VertexShader(ref path) => {
                                let file = File::open(path).expect("Could not find shader.");
                                let bytes: Vec<u8> = file.bytes().filter_map(|byte| byte.ok()).collect();
                                let shader_info = vk::ShaderModuleCreateInfo {
                                    s_type: vk::StructureType::ShaderModuleCreateInfo,
                                    p_next: ptr::null(),
                                    flags: Default::default(),
                                    code_size: bytes.len(),
                                    p_code: bytes.as_ptr() as *const u32,
                                };
                                let shader_module = unsafe {
                                    base.device
                                        .create_shader_module(&shader_info, None)
                                        .expect("Vertex shader module error")
                                };

                                Some((shader_module, vk::SHADER_STAGE_VERTEX_BIT))
                            }
                            Node::FragmentShader(ref path) => {
                                let file = File::open(path).expect("Could not find shader.");
                                let bytes: Vec<u8> = file.bytes().filter_map(|byte| byte.ok()).collect();
                                let shader_info = vk::ShaderModuleCreateInfo {
                                    s_type: vk::StructureType::ShaderModuleCreateInfo,
                                    p_next: ptr::null(),
                                    flags: Default::default(),
                                    code_size: bytes.len(),
                                    p_code: bytes.as_ptr() as *const u32,
                                };
                                let shader_module = unsafe {
                                    base.device
                                        .create_shader_module(&shader_info, None)
                                        .expect("Vertex shader module error")
                                };

                                Some((shader_module, vk::SHADER_STAGE_FRAGMENT_BIT))
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let pipeline_layout = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref name, Node::PipelineLayout) => pipeline_layouts.get(name),
                            _ => None,
                        })
                        .next()
                        .expect("no pipeline layout specified for graphics pipeline");
                    let &(_begin_renderpass, _end_renderpass, renderpass) = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref name, Node::RenderPass) => renderpasses.get(name),
                            _ => None,
                        })
                        .next()
                        .expect("no renderpass specified for graphics pipeline");
                    let shader_entry_name = CString::new("main").unwrap();
                    let shader_stage_create_infos = shader_modules
                        .iter()
                        .map(|&(module, stage)| vk::PipelineShaderStageCreateInfo {
                            s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                            p_next: ptr::null(),
                            flags: Default::default(),
                            module: module,
                            p_name: shader_entry_name.as_ptr(),
                            p_specialization_info: ptr::null(),
                            stage: stage,
                        })
                        .collect::<Vec<_>>();
                    let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
                        s_type: vk::StructureType::PipelineVertexInputStateCreateInfo,
                        p_next: ptr::null(),
                        flags: Default::default(),
                        vertex_attribute_description_count: vertex_attributes.len() as u32,
                        p_vertex_attribute_descriptions: vertex_attributes.as_ptr(),
                        vertex_binding_description_count: vertex_bindings.len() as u32,
                        p_vertex_binding_descriptions: vertex_bindings.as_ptr(),
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
                    let subpass_ix = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref _name, Node::Subpass(ix)) => Some(ix),
                            _ => None,
                        })
                        .next()
                        .expect("No subpass bound to this pipeline");
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
                        layout: pipeline_layout.clone(),
                        render_pass: renderpass,
                        subpass: subpass_ix as u32,
                        base_pipeline_handle: vk::Pipeline::null(),
                        base_pipeline_index: 0,
                    };
                    let graphics_pipelines = unsafe {
                        base.device
                            .create_graphics_pipelines(vk::PipelineCache::null(), &[graphic_pipeline_info], None)
                            .expect("Unable to create graphics pipeline")
                    };

                    let graphics_pipeline = graphics_pipelines[0];

                    let bind = output_graph.add_node((
                        "bind_pipeline",
                        NodeRuntime::BindPipeline(
                            graphics_pipeline,
                            Some(scissors[0].clone()),
                            Some(viewports[0].clone()),
                        ),
                    ));
                    let &(subpass_start, subpass_end, _) = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref _name, Node::Subpass(_)) => subpasses.get(node.0),
                            _ => None,
                        })
                        .next()
                        .expect("No subpass specified for DrawCommands");

                    output_graph.add_edge(subpass_start, bind, ());
                    output_graph.add_edge(bind, subpass_end, ());

                    pipelines.insert(self.graph[node].0, (bind, graphics_pipeline));
                }
                Node::DrawCommands(ref f) => {
                    let &(pipeline_bind, _) = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref _name, Node::GraphicsPipeline) => pipelines.get(node.0),
                            _ => None,
                        })
                        .next()
                        .expect("No pipeline specified for DrawCommands");
                    let &(_subpass_start, subpass_end, _) = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref _name, Node::Subpass(_)) => subpasses.get(node.0),
                            _ => None,
                        })
                        .next()
                        .expect("No subpass specified for DrawCommands");

                    let draw = output_graph.add_node(("draw_commands", NodeRuntime::DrawCommands(f.clone())));
                    output_graph.add_edge(pipeline_bind, draw, ());
                    output_graph.add_edge(draw, subpass_end, ());
                }
                Node::DescriptorSet(size) => {
                    let descriptor_sizes = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref _name, Node::DescriptorBinding(_binding, typ, _stage, count)) => Some(vk::DescriptorPoolSize {
                                typ,
                                descriptor_count: count * size,
                            }),

                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
                        s_type: vk::StructureType::DescriptorPoolCreateInfo,
                        p_next: ptr::null(),
                        flags: Default::default(),
                        pool_size_count: descriptor_sizes.len() as u32,
                        p_pool_sizes: descriptor_sizes.as_ptr(),
                        max_sets: size,
                    };
                    let descriptor_pool = unsafe {
                        base.device
                            .create_descriptor_pool(&descriptor_pool_info, None)
                            .unwrap()
                    };

                    let bindings = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref _name, Node::DescriptorBinding(binding, typ, stage, count)) => Some(vk::DescriptorSetLayoutBinding {
                                binding: binding,
                                descriptor_type: typ,
                                descriptor_count: count,
                                stage_flags: stage,
                                p_immutable_samplers: ptr::null(),
                            }),

                            _ => None,
                        })
                        .collect::<Vec<_>>();

                    let layouts = (0..size)
                        .map(|_n| {
                            let descriptor_info = vk::DescriptorSetLayoutCreateInfo {
                                s_type: vk::StructureType::DescriptorSetLayoutCreateInfo,
                                p_next: ptr::null(),
                                flags: Default::default(),
                                binding_count: bindings.len() as u32,
                                p_bindings: bindings.as_ptr(),
                            };

                            unsafe {
                                base.device
                                    .create_descriptor_set_layout(&descriptor_info, None)
                                    .unwrap()
                            }
                        })
                        .collect::<Vec<_>>();

                    let desc_alloc_info = vk::DescriptorSetAllocateInfo {
                        s_type: vk::StructureType::DescriptorSetAllocateInfo,
                        p_next: ptr::null(),
                        descriptor_pool: descriptor_pool,
                        descriptor_set_count: layouts.len() as u32,
                        p_set_layouts: layouts.as_ptr(),
                    };
                    let new_descriptor_sets = unsafe {
                        base.device
                            .allocate_descriptor_sets(&desc_alloc_info)
                            .unwrap()
                    };

                    descriptor_set_layouts.insert(self.graph[node].0, layouts[0]);
                    descriptor_sets.insert(self.graph[node].0, new_descriptor_sets);
                }
                _ => println!("* not doing anything"),
            }
        }
        use std::iter::FromIterator;
        RenderDAG {
            graph: output_graph,
            pipeline_layouts,
            descriptor_sets: descriptor_sets,
            renderpasses: HashMap::from_iter(renderpasses.iter().map(|(k, v)| (*k, v.2))),
        }
    }
}
