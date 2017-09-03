use ash::version::DeviceV1_0;
use ash::vk;
use futures;
use futures::Future;
use futures::future::Shared;
use futures::sync::oneshot;
use futures_cpupool::{CpuPool, CpuFuture};
use petgraph;
use std::collections::HashMap;
use std::fmt;
use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;

use super::World;
use super::super::ExampleBase;
use super::super::ecs::SimpleColorMesh;

#[derive(Clone)]
pub enum Node {
    AcquirePresentImage,
    SwapchainAttachment(u8),
    DepthAttachment(u8),
    Subpass(u8),
    RenderPass,
    VertexShader(PathBuf),
    FragmentShader(PathBuf),
    PipelineLayout,
    VertexInputBinding(u32 /*, stride, rate */),
    /// Binding, location, format, offset
    VertexInputAttribute(u32, u32, vk::Format, u32),
    GraphicsPipeline,
    DrawCommands(Arc<Fn(&ExampleBase, &World, vk::CommandBuffer)>),
    PresentImage,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Node")
    }
}

type NodeResult<T> = Arc<Shared<CpuFuture<T, ()>>>;

#[derive(Clone)]
enum NodeRuntime {
    BeginRenderPass(vk::RenderPass),
    BeginSubPass(u8),
    BindPipeline(vk::Pipeline, Option<vk::Rect2D>, Option<vk::Viewport>),
    DrawCommands(Arc<Fn(&ExampleBase, &World, vk::CommandBuffer)>),
    EndSubPass(u8), // TODO: we should only have BeginSubPass if possible, to model vulkan
    EndRenderPass(vk::RenderPass),
}

impl fmt::Debug for NodeRuntime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NodeRuntime")
    }
}

type BuilderGraph = petgraph::Graph<(String, Node), ()>;
type RuntimeGraph = petgraph::Graph<(String, NodeRuntime), ()>;

pub struct RenderDAG {
    graph: RuntimeGraph,
}

impl RenderDAG {
    pub fn run(&self, base: &ExampleBase, world: &World, framebuffer: vk::Framebuffer, command_buffer: vk::CommandBuffer) {
        let pool = CpuPool::new_num_cpus();

        /*
        let mut g: petgraph::Graph<Option<Shared<CpuFuture<ExecResult, ()>>>, _> =
            self.graph.map(|_ix, _node| None, |_ix, edge| edge);
            */
        for node in petgraph::algo::toposort(&self.graph) {
            println!("arrived at {}", (self.graph[node].0));

            match &self.graph[node].1 {
                &NodeRuntime::BeginRenderPass(renderpass) => {
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
                }
                &NodeRuntime::BeginSubPass(ix) => println!("subpass {}", ix),
                &NodeRuntime::BindPipeline(pipeline, ref scissors_opt, ref viewport_opt) => unsafe {
                    base.device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::Graphics,
                        pipeline,
                    );

                    if let &Some(ref viewport) = viewport_opt {
                        base.device.cmd_set_viewport(
                            command_buffer,
                            &[viewport.clone()],
                        );
                    }
                    if let &Some(ref scissors) = scissors_opt {
                        base.device.cmd_set_scissor(command_buffer, &[scissors.clone()]);
                    }

                },
                &NodeRuntime::DrawCommands(ref f) => f(base, world, command_buffer),
                &NodeRuntime::EndSubPass(ix) => println!("end subpass {}", ix),
                &NodeRuntime::EndRenderPass(_renderpass) => unsafe { base.device.cmd_end_render_pass(command_buffer) },
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
    graph: BuilderGraph,
}

impl RenderDAGBuilder {
    pub fn new() -> RenderDAGBuilder {
        let mut graph = BuilderGraph::new();
        let start = graph.add_node((String::from("acquire_image"), Node::AcquirePresentImage));
        let swapchain_attachment = graph.add_node((
            String::from("swapchain_attachment"),
            Node::SwapchainAttachment(0),
        ));
        let depth_attachment = graph.add_node((String::from("depth_attachment"), Node::DepthAttachment(1)));
        let subpass = graph.add_node((String::from("subpass"), Node::Subpass(0)));
        let renderpass = graph.add_node((String::from("renderpass"), Node::RenderPass));
        let vs = graph.add_node((
            String::from("vertex_shader"),
            Node::VertexShader(
                PathBuf::from(env!("OUT_DIR")).join("simple_color.vert.spv"),
            ),
        ));
        let fs = graph.add_node((
            String::from("fragment_shader"),
            Node::FragmentShader(
                PathBuf::from(env!("OUT_DIR")).join("simple_color.frag.spv"),
            ),
        ));
        let pipeline_layout = graph.add_node((String::from("pipeline_layout"), Node::PipelineLayout));
        let vertex_binding = graph.add_node((
            String::from("vertex_binding"),
            Node::VertexInputBinding(0),
        ));
        let vertex_attribute = graph.add_node((
            String::from("vertex_attribute"),
            Node::VertexInputAttribute(
                0,
                0,
                vk::Format::R32g32b32a32Sfloat,
                0,
            ),
        ));
        let graphics_pipeline = graph.add_node((String::from("graphics_pipeline"), Node::GraphicsPipeline));
        let draw_commands = graph.add_node((
            String::from("draw_commands"),
            Node::DrawCommands(Arc::new(|base, world, command_buffer| {
                use specs::Join;
                println!("drawing SimpleColorMesh");
                for mesh in world.read::<SimpleColorMesh>().join() {
                    unsafe {
                        base.device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[mesh.0.vertex_buffer.buffer()],
                            &[0],
                        );
                        base.device.cmd_bind_index_buffer(
                            command_buffer,
                            mesh.0.index_buffer.buffer(),
                            0,
                            mesh.0.index_type,
                        );
                        base.device.cmd_draw_indexed(
                            command_buffer,
                            mesh.0.index_count,
                            1,
                            0,
                            0,
                            0,
                        );
                    }
                }
            })),
        ));
        let end = graph.add_node((String::from("present_image"), Node::PresentImage));
        graph.add_edge(start, end, ());
        graph.add_edge(start, swapchain_attachment, ());
        graph.add_edge(start, depth_attachment, ());
        graph.add_edge(subpass, renderpass, ());
        graph.add_edge(subpass, graphics_pipeline, ());
        graph.add_edge(subpass, draw_commands, ());
        graph.add_edge(swapchain_attachment, renderpass, ());
        graph.add_edge(depth_attachment, renderpass, ());
        graph.add_edge(vs, graphics_pipeline, ());
        graph.add_edge(fs, graphics_pipeline, ());
        graph.add_edge(renderpass, graphics_pipeline, ());
        graph.add_edge(vertex_binding, graphics_pipeline, ());
        graph.add_edge(vertex_attribute, graphics_pipeline, ());
        graph.add_edge(pipeline_layout, graphics_pipeline, ());
        graph.add_edge(graphics_pipeline, draw_commands, ());
        graph.add_edge(draw_commands, end, ());
        RenderDAGBuilder { graph: graph }
    }

    pub fn build(self, base: &ExampleBase) -> RenderDAG {
        let mut output_graph = RuntimeGraph::new();
        let mut renderpasses: HashMap<&str, (petgraph::graph::NodeIndex, petgraph::graph::NodeIndex, vk::RenderPass)> = HashMap::new();
        let mut subpasses: HashMap<&str, (petgraph::graph::NodeIndex, petgraph::graph::NodeIndex, u8)> = HashMap::new();
        let mut pipeline_layouts: HashMap<&str, vk::PipelineLayout> = HashMap::new();
        let mut pipelines: HashMap<&str, (petgraph::graph::NodeIndex, vk::Pipeline)> = HashMap::new();
        for node in petgraph::algo::toposort(&self.graph) {
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
                    let dependency = vk::SubpassDependency {
                        dependency_flags: Default::default(),
                        src_subpass: vk::VK_SUBPASS_EXTERNAL,
                        dst_subpass: Default::default(),
                        src_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        src_access_mask: Default::default(),
                        dst_access_mask: vk::ACCESS_COLOR_ATTACHMENT_READ_BIT | vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        dst_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    };
                    let subpass = vk::SubpassDescription {
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
                    };
                    let renderpass_create_info = vk::RenderPassCreateInfo {
                        s_type: vk::StructureType::RenderPassCreateInfo,
                        flags: Default::default(),
                        p_next: ptr::null(),
                        attachment_count: attachments.len() as u32,
                        p_attachments: attachments.as_ptr(),
                        subpass_count: 1,
                        p_subpasses: &subpass,
                        dependency_count: 1,
                        p_dependencies: &dependency,
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
                            &(ref name, Node::Subpass(_)) => subpasses.get(node.0.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();

                    let start = output_graph.add_node((
                        String::from("begin_render_pass"),
                        NodeRuntime::BeginRenderPass(renderpass),
                    ));
                    let end = output_graph.add_node((
                        String::from("end_render_pass"),
                        NodeRuntime::EndRenderPass(renderpass),
                    ));
                    for &(start_subpass, end_subpass, _) in subpasses {
                        output_graph.add_edge(start, start_subpass, ());
                        output_graph.add_edge(end_subpass, end, ());
                    }
                    output_graph.add_edge(start, end, ());

                    renderpasses.insert(self.graph[node].0.as_str(), (start, end, renderpass));
                }
                Node::Subpass(ix) => {
                    let start = output_graph.add_node((
                        String::from("begin_subpass"),
                        NodeRuntime::BeginSubPass(ix),
                    ));
                    let end = output_graph.add_node((String::from("end_subpass"), NodeRuntime::EndSubPass(ix)));
                    output_graph.add_edge(start, end, ());

                    subpasses.insert(self.graph[node].0.as_str(), (start, end, ix));
                }
                Node::PipelineLayout => {
                    let create_info = vk::PipelineLayoutCreateInfo {
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
                            .create_pipeline_layout(&create_info, None)
                            .unwrap()
                    };

                    pipeline_layouts.insert(self.graph[node].0.as_str(), pipeline_layout);
                }
                Node::GraphicsPipeline => {
                    let vertex_attributes = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            Node::VertexInputAttribute(location, binding, format, offset) => {
                                Some(vk::VertexInputAttributeDescription {
                                    location: location,
                                    binding: binding,
                                    format: format,
                                    offset: offset,
                                })
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let vertex_bindings = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            Node::VertexInputBinding(binding) => Some(vk::VertexInputBindingDescription {
                                binding: binding,
                                stride: 0,
                                input_rate: vk::VertexInputRate::Vertex,
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
                    println!("Shader modules are {:?}", shader_modules);
                    let pipeline_layout = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref name, Node::PipelineLayout) => pipeline_layouts.get(name.as_str()),
                            _ => None,
                        })
                        .next()
                        .expect("no pipeline layout specified for graphics pipeline");
                    println!(
                        "* creating graphics pipeline from dependencies, {:?}",
                        inputs
                    );
                    let &(begin_renderpass, end_renderpass, renderpass) = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref name, Node::RenderPass) => renderpasses.get(name.as_str()),
                            _ => None,
                        })
                        .next()
                        .expect("no renderpass specified for graphics pipeline");
                    let shader_entry_name = CString::new("main").unwrap();
                    let shader_stage_create_infos = shader_modules
                        .iter()
                        .map(|&(module, stage)| {
                            vk::PipelineShaderStageCreateInfo {
                                s_type: vk::StructureType::PipelineShaderStageCreateInfo,
                                p_next: ptr::null(),
                                flags: Default::default(),
                                module: module,
                                p_name: shader_entry_name.as_ptr(),
                                p_specialization_info: ptr::null(),
                                stage: stage,
                            }
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
                        subpass: 0,
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
                        String::from("bind_pipeline"),
                        NodeRuntime::BindPipeline(graphics_pipeline, Some(scissors[0].clone()), Some(viewports[0].clone())),
                    ));
                    let &(subpass_start, subpass_end, _) = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref name, Node::Subpass(_)) => subpasses.get(node.0.as_str()),
                            _ => None,
                        })
                        .next()
                        .expect("No subpass specified for DrawCommands");

                    output_graph.add_edge(subpass_start, bind, ());
                    output_graph.add_edge(bind, subpass_end, ());

                    pipelines.insert(self.graph[node].0.as_str(), (bind, graphics_pipeline));
                }
                Node::DrawCommands(ref f) => {
                    let &(pipeline_bind, _) = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref name, Node::GraphicsPipeline) => pipelines.get(node.0.as_str()),
                            _ => None,
                        })
                        .next()
                        .expect("No pipeline specified for DrawCommands");
                    let &(subpass_start, subpass_end, _) = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref name, Node::Subpass(_)) => subpasses.get(node.0.as_str()),
                            _ => None,
                        })
                        .next()
                        .expect("No subpass specified for DrawCommands");

                    let draw = output_graph.add_node((
                        String::from("draw_commands"),
                        NodeRuntime::DrawCommands(f.clone()),
                    ));
                    output_graph.add_edge(pipeline_bind, draw, ());
                    output_graph.add_edge(draw, subpass_end, ());
                }
                _ => println!("* not doing anything"),
            }
        }
        RenderDAG { graph: output_graph }
    }
}

pub fn test_renderdag(base: &ExampleBase) {
    let builder = RenderDAGBuilder::new();
    {
        let dot = petgraph::dot::Dot::new(&builder.graph);
        println!("{:?}", dot);
    }
    let built = builder.build(base);
    let dot = petgraph::dot::Dot::new(&built.graph);
    println!("{:?}", dot);
    println!("and its final toposort");
    for node in petgraph::algo::toposort(&built.graph) {
        println!("arrived at {}", (built.graph[node].0));
    }
}
