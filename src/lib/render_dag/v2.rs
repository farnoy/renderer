use ash::version::DeviceV1_0;
use ash::vk;
use futures::prelude::*;
use futures_cpupool::*;
use futures::future::{join_all, Shared};
use petgraph;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::marker::{Send, Sync};
use std::sync::RwLock;
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;
use std::u64;

use ecs::World;
use super::super::device::Device;
use super::super::ExampleBase;

#[allow(unknown_lints, type_complexity)]
#[derive(Clone)]
pub enum NodeBuilder {
    AcquirePresentImage,
    BeginCommandBuffer,
    EndCommandBuffer,
    SwapchainAttachment(u8),
    DepthAttachment(u8),
    BeginSubpass(u8),
    BeginRenderPass,
    EndRenderPass,
    EndSubpass(u8),
    VertexShader(PathBuf),
    GeometryShader(PathBuf),
    FragmentShader(PathBuf),
    Framebuffer,
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
    DrawCommands(Arc<Fn(&Device, &Arc<RwLock<World>>, &RenderDAG, vk::CommandBuffer) + Send + Sync>),
    PresentImage,
}

impl fmt::Debug for NodeBuilder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NodeBuilder")
    }
}

type NodeResult<R, Dyn> = (R, Arc<RwLock<Shared<CpuFuture<Option<Dyn>, ()>>>>);

#[allow(unknown_lints, type_complexity)]
#[derive(Clone)]
pub enum NodeRuntime {
    AcquirePresentImage,
    BeginCommandBuffer(vk::CommandBuffer),
    BeginRenderPass(vk::RenderPass),
    BeginSubPass(u8),
    BindPipeline(vk::Pipeline, Option<vk::Rect2D>, Option<vk::Viewport>),
    DrawCommands(Arc<Fn(&Device, &Arc<RwLock<World>>, &RenderDAG, vk::CommandBuffer) + Send + Sync>),
    EndCommandBuffer,
    EndSubPass(u8), // TODO: we should only have BeginSubPass if possible, to model vulkan
    EndRenderPass,
    Framebuffer(Vec<vk::Framebuffer>),
    PresentImage,
}

impl fmt::Debug for NodeRuntime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let output = match *self {
            NodeRuntime::AcquirePresentImage => "AcquirePresentImage",
            NodeRuntime::BeginCommandBuffer(_) => "BeginCommandBuffer",
            NodeRuntime::BeginRenderPass(_) => "BeginRenderPass",
            _ => "other",
        };
        write!(f, "NodeRuntime::{}", output)
    }
}

#[derive(Clone, Debug)]
pub enum NodeDynamic {
    AcquirePresentImage(u32), // present index
}

type BuilderGraph = petgraph::Graph<(Cow<'static, str>, NodeBuilder), ()>;
type RuntimeGraph = petgraph::Graph<(Cow<'static, str>, NodeResult<NodeRuntime, NodeDynamic>), ()>;

#[derive(Clone)]
pub struct RenderDAG {
    pub graph: RuntimeGraph,
    pub pipeline_layouts: HashMap<Cow<'static, str>, vk::PipelineLayout>,
    pub descriptor_sets: HashMap<Cow<'static, str>, Vec<vk::DescriptorSet>>,
    pool: CpuPool,
}

impl RenderDAG {
    pub fn run(&self, base: &ExampleBase, world: &Arc<RwLock<World>>) -> Result<(), String> {
        let mut last_node = Err("No last node, is the graph valid?".to_string());
        for node in petgraph::algo::toposort(&self.graph, None)
            .expect("RenderDAG has cycles")
            .iter()
            .cloned()
        {
            last_node = Ok(node);
            let inputs: Vec<NodeResult<NodeRuntime, NodeDynamic>> = self.graph
                .neighbors_directed(node, petgraph::EdgeDirection::Incoming)
                .map(|ix| self.graph[ix].1.clone())
                .collect();
            fn select_inputs<Selector>(
                pool: &CpuPool,
                inputs: &[NodeResult<NodeRuntime, NodeDynamic>],
                selector: Selector,
            ) -> impl Future<Item = Vec<(NodeRuntime, Option<NodeDynamic>)>, Error = ()>
            where
                Selector: Fn(&NodeRuntime) -> bool,
            {
                let futures = inputs
                    .iter()
                    .filter_map(|&(ref runtime, ref dynamic)| {
                        if selector(runtime) {
                            let runtime = runtime.clone();
                            Some(
                                dynamic
                                    .read()
                                    .unwrap()
                                    .clone()
                                    .map(move |dyn| (runtime, dyn)),
                            )
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let fut = join_all(futures).map_err(|_| ()).map(|ref shared_fut| {
                    shared_fut
                        .iter()
                        .map(|&(ref runtime, ref shared)| (runtime.clone(), (**shared).clone()))
                        .collect()
                });
                pool.spawn(fut)
            }
            let wait_all = select_inputs(&self.pool, &inputs, |_| true);
            /*
                join_all(input_futures)
                    .map_err(|_| ())
                    .map(|ref shared_fut| shared_fut.iter().map(|shared| (**shared).clone()).collect()),
                    */
            let mut this_dynamic = self.graph[node]
                .1
                 .1
                .write()
                .expect("Failed to acquire ownership lock");
            let this_node: &NodeRuntime = &(self.graph[node].1).0;
            let this_name = self.graph[node].0.clone();
            // println!("Executing {}", this_name.clone());
            match *this_node {
                NodeRuntime::AcquirePresentImage => {
                    let swapchain_loader = base.swapchain_loader.clone();
                    let swapchain = base.swapchain;
                    let present_complete_semaphore = base.present_complete_semaphore;
                    *this_dynamic = self.pool
                        .spawn(wait_all.map(move |_inputs| {
                            let present_index = unsafe {
                                swapchain_loader
                                    .acquire_next_image_khr(
                                        swapchain,
                                        u64::MAX,
                                        present_complete_semaphore,
                                        vk::Fence::null(),
                                    )
                                    .unwrap()
                            };
                            println!("Acquired present index {}", present_index);
                            Some(NodeDynamic::AcquirePresentImage(present_index))
                        }))
                        .shared();
                }
                NodeRuntime::PresentImage => unsafe {
                    let swapchain_loader = base.swapchain_loader.clone();
                    let present_queue = base.present_queue;
                    let rendering_complete_semaphore = base.rendering_complete_semaphore;
                    let swapchain = base.swapchain;
                    *this_dynamic = self.pool
                        .spawn(
                            wait_all
                                .map(move |inputs| {
                                    let present_index = inputs
                                        .iter()
                                        .cloned()
                                        .filter_map(|i| match i {
                                            (NodeRuntime::AcquirePresentImage, Some(NodeDynamic::AcquirePresentImage(present_index))) => Some(present_index),
                                            _ => None,
                                        })
                                        .next()
                                        .unwrap();
                                    let present_info = vk::PresentInfoKHR {
                                        s_type: vk::StructureType::PresentInfoKhr,
                                        p_next: ptr::null(),
                                        wait_semaphore_count: 1,
                                        p_wait_semaphores: &rendering_complete_semaphore,
                                        swapchain_count: 1,
                                        p_swapchains: &swapchain,
                                        p_image_indices: &present_index,
                                        p_results: ptr::null_mut(),
                                    };
                                    println!("presenting index {}", present_index);
                                    swapchain_loader
                                        .queue_present_khr(present_queue, &present_info)
                                        .unwrap();
                                    None
                                })
                                .map_err(|_| ()),
                        )
                        .shared();
                },
                NodeRuntime::BeginCommandBuffer(command_buffer) => {
                    let device = base.device.clone();
                    *this_dynamic = self.pool
                        .spawn(
                            wait_all
                                .map(move |_inputs| unsafe {
                                    device
                                        .reset_command_buffer(
                                            command_buffer,
                                            vk::COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT,
                                        )
                                        .expect("Reset command buffer failed.");
                                    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                                        s_type: vk::StructureType::CommandBufferBeginInfo,
                                        p_next: ptr::null(),
                                        p_inheritance_info: ptr::null(),
                                        flags: vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                    };
                                    device
                                        .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                                        .expect("Begin commandbuffer");
                                    device.debug_marker_start(
                                        command_buffer,
                                        &format!("{} -> CommandBuffer", this_name),
                                        [0.0; 4],
                                    );

                                    None
                                })
                                .map_err(|_| ()),
                        )
                        .shared();
                }
                NodeRuntime::EndCommandBuffer => {
                    let device = base.device.clone();
                    // TODO: represent semaphores in the DAG
                    let present_complete_semaphore = base.present_complete_semaphore;
                    let rendering_complete_semaphore = base.rendering_complete_semaphore;
                    // TODO: represent queues in the DAG
                    let submit_queue = base.present_queue;
                    *this_dynamic = self.pool
                        .spawn(
                            wait_all
                                .map(move |inputs| unsafe {
                                    let command_buffer = inputs
                                        .iter()
                                        .cloned()
                                        .filter_map(|i| match i.0 {
                                            NodeRuntime::BeginCommandBuffer(cb) => Some(cb),
                                            _ => None,
                                        })
                                        .next()
                                        .expect(&format!(
                                            "BeginCommandBuffer not attached to EndCommandBuffer {}",
                                            this_name
                                        ));
                                    device.debug_marker_end(command_buffer);
                                    device
                                        .end_command_buffer(command_buffer)
                                        .expect("End commandbuffer");
                                    let fence_create_info = vk::FenceCreateInfo {
                                        s_type: vk::StructureType::FenceCreateInfo,
                                        p_next: ptr::null(),
                                        flags: vk::FenceCreateFlags::empty(),
                                    };
                                    let submit_fence = device
                                        .create_fence(&fence_create_info, None)
                                        .expect("Create fence failed.");
                                    let wait_semaphores = [present_complete_semaphore];
                                    let signal_semaphores = [rendering_complete_semaphore];
                                    // TODO: represent wait mask in the DAG/Builder node ??
                                    let wait_mask = [vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT];
                                    let submit_info = vk::SubmitInfo {
                                        s_type: vk::StructureType::SubmitInfo,
                                        p_next: ptr::null(),
                                        wait_semaphore_count: wait_semaphores.len() as u32,
                                        p_wait_semaphores: wait_semaphores.as_ptr(),
                                        p_wait_dst_stage_mask: wait_mask.as_ptr(),
                                        // TODO: extract submitting into another node to enable multiple CB submission
                                        command_buffer_count: 1,
                                        p_command_buffers: &command_buffer,
                                        signal_semaphore_count: signal_semaphores.len() as u32,
                                        p_signal_semaphores: signal_semaphores.as_ptr(),
                                    };
                                    println!("EndCommandBuffer, submitting to queue");
                                    device
                                        .queue_submit(submit_queue, &[submit_info], submit_fence)
                                        .expect("queue submit failed.");
                                    use std::u64;
                                    device
                                        .wait_for_fences(&[submit_fence], true, u64::MAX)
                                        .expect("Wait for fence failed.");
                                    device.destroy_fence(submit_fence, None);
                                    None
                                })
                                .map_err(|_| ()),
                        )
                        .shared();
                }
                NodeRuntime::BeginRenderPass(renderpass) => {
                    let device = base.device.clone();
                    let surface_resolution = base.surface_resolution;
                    let renderpass = renderpass;
                    let name = self.graph[node].0.clone();
                    *this_dynamic = self.pool
                        .spawn(wait_all.map(move |inputs| {
                            let command_buffer = inputs
                                .iter()
                                .cloned()
                                .filter_map(|i| match i.0 {
                                    NodeRuntime::BeginCommandBuffer(cb) => Some(cb),
                                    _ => None,
                                })
                                .next()
                                .expect(&format!(
                                    "BeginCommandBuffer not attached to BeginRenderPass {}",
                                    name
                                ));
                            let framebuffer = inputs
                                .iter()
                                .cloned()
                                .filter_map(|i| match i.0 {
                                    NodeRuntime::Framebuffer(fb) => Some(fb),
                                    _ => None,
                                })
                                .next()
                                .expect(&format!(
                                    "Framebuffer not attached to BeginRenderPass {}",
                                    name
                                ));
                            let present_index = inputs
                                .iter()
                                .cloned()
                                .filter_map(|i| match i {
                                    (NodeRuntime::AcquirePresentImage, Some(NodeDynamic::AcquirePresentImage(present_index))) => Some(present_index),
                                    _ => None,
                                })
                                .next()
                                .expect(&format!(
                                    "AcquirePresentImage not attached to BeginRenderPass {}",
                                    name
                                ));
                            device.debug_marker_start(
                                command_buffer,
                                &format!("{} -> RenderPass", this_name),
                                [0.0; 4],
                            );
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
                                framebuffer: framebuffer[present_index as usize],
                                render_area: vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: surface_resolution,
                                },
                                clear_value_count: clear_values.len() as u32,
                                p_clear_values: clear_values.as_ptr(),
                            };

                            unsafe {
                                device.cmd_begin_render_pass(
                                    command_buffer,
                                    &render_pass_begin_info,
                                    vk::SubpassContents::Inline,
                                );
                            }
                            None
                        }))
                        .shared();
                }
                NodeRuntime::BeginSubPass(_ix) => {
                    let device = base.device.clone();
                    let reversed = petgraph::visit::Reversed(&self.graph);
                    let bfs = petgraph::visit::Bfs::new(&reversed, node);
                    use petgraph::visit::Walker;
                    let command_buffer = bfs.iter(&reversed)
                        .filter_map(|parent| match (self.graph[parent].1).0 {
                            NodeRuntime::BeginCommandBuffer(cb) => Some(cb),
                            _ => None,
                        })
                        .next()
                        .expect(&format!(
                            "BFS search couldn't find BeginCommandBuffer for {}",
                            this_name.clone()
                        ));
                    *this_dynamic = self.pool
                        .spawn(wait_all.map(move |inputs| {
                            let previous_subpass = inputs.iter().find(|i| match i.0 {
                                NodeRuntime::EndSubPass(_) => true,
                                _ => false,
                            });

                            if previous_subpass.is_some() {
                                unsafe {
                                    device.cmd_next_subpass(command_buffer, vk::SubpassContents::Inline);
                                }
                            }

                            device.debug_marker_start(
                                command_buffer,
                                &format!("{} -> SubPass", this_name),
                                [0.0; 4],
                            );

                            None
                        }))
                        .shared();
                }
                NodeRuntime::BindPipeline(pipeline, scissors_opt, viewport_opt) => unsafe {
                    let device = base.device.clone();
                    let name = self.graph[node].0.clone();
                    let pipeline = pipeline;
                    let scissors_opt = scissors_opt;
                    let viewport_opt = viewport_opt;
                    let reversed = petgraph::visit::Reversed(&self.graph);
                    let bfs = petgraph::visit::Bfs::new(&reversed, node);
                    use petgraph::visit::Walker;
                    let command_buffer = bfs.iter(&reversed)
                        .filter_map(|parent| match (self.graph[parent].1).0 {
                            NodeRuntime::BeginCommandBuffer(cb) => Some(cb),
                            _ => None,
                        })
                        .next()
                        .expect(&format!(
                            "BFS search couldn't find BeginCommandBuffer for {}",
                            this_name.clone()
                        ));
                    *this_dynamic = self.pool
                        .spawn(wait_all.map(move |_inputs| {
                            device.debug_marker_around(
                                command_buffer,
                                &format!("{} -> BindPipeline", name),
                                [0.0; 4],
                                || {
                                    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::Graphics, pipeline);

                                    if let Some(viewport) = viewport_opt {
                                        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                                    }
                                    if let Some(scissors) = scissors_opt {
                                        device.cmd_set_scissor(command_buffer, &[scissors]);
                                    }
                                },
                            );
                            None
                        }))
                        .shared();
                },
                NodeRuntime::DrawCommands(ref f) => {
                    let device = base.device.clone();
                    let f = f.clone();
                    let render_dag = self.clone();
                    let world = world.clone();
                    let reversed = petgraph::visit::Reversed(&self.graph);
                    let bfs = petgraph::visit::Bfs::new(&reversed, node);
                    use petgraph::visit::Walker;
                    let command_buffer = bfs.iter(&reversed)
                        .filter_map(|parent| match (self.graph[parent].1).0 {
                            NodeRuntime::BeginCommandBuffer(cb) => Some(cb),
                            _ => None,
                        })
                        .next()
                        .expect(&format!(
                            "BFS search couldn't find BeginCommandBuffer for {}",
                            this_name.clone()
                        ));
                    *this_dynamic = self.pool
                        .spawn(wait_all.map(move |_inputs| {
                            let new_device = device.clone();
                            let new_f = f.clone();
                            device.debug_marker_around(
                                command_buffer,
                                &format!("{} -> DrawCommands", this_name),
                                [0.0; 4],
                                move || new_f(&new_device, &world, &render_dag, command_buffer),
                            );
                            None
                        }))
                        .shared();
                }
                NodeRuntime::EndSubPass(_ix) => {
                    let device = base.device.clone();
                    let reversed = petgraph::visit::Reversed(&self.graph);
                    let bfs = petgraph::visit::Bfs::new(&reversed, node);
                    use petgraph::visit::Walker;
                    let command_buffer = bfs.iter(&reversed)
                        .filter_map(|parent| match (self.graph[parent].1).0 {
                            NodeRuntime::BeginCommandBuffer(cb) => Some(cb),
                            _ => None,
                        })
                        .next()
                        .expect(&format!(
                            "BFS search couldn't find BeginCommandBuffer for {}",
                            this_name.clone()
                        ));
                    *this_dynamic = self.pool
                        .spawn(wait_all.map(move |_inputs| {
                            device.debug_marker_end(command_buffer);
                            None
                        }))
                        .shared();
                }
                NodeRuntime::EndRenderPass => {
                    let device = base.device.clone();
                    *this_dynamic = self.pool
                        .spawn(wait_all.map(move |inputs| {
                            let command_buffer = inputs
                                .iter()
                                .cloned()
                                .filter_map(|i| match i.0 {
                                    NodeRuntime::BeginCommandBuffer(cb) => Some(cb),
                                    _ => None,
                                })
                                .next()
                                .expect(&format!(
                                    "BeginCommandBuffer not attached to EndRenderPass {}",
                                    this_name
                                ));
                            let _renderpass = inputs
                                .iter()
                                .cloned()
                                .filter_map(|i| match i.0 {
                                    NodeRuntime::BeginRenderPass(rp) => Some(rp),
                                    _ => None,
                                })
                                .next()
                                .expect(&format!(
                                    "BeginRenderPass not attached to EndRenderPass {}",
                                    this_name
                                ));
                            unsafe { device.cmd_end_render_pass(command_buffer) }
                            device.debug_marker_end(command_buffer);
                            println!("Ended render pass");
                            None
                        }))
                        .shared();
                }
                NodeRuntime::Framebuffer(_) => (),
            }
        }
        last_node.and_then(|last_node| {
            self.pool
                .spawn(
                    self.graph[last_node]
                        .1
                         .1
                        .read()
                        .unwrap()
                        .clone()
                        .map(|_| ())
                        .map_err(|_| "Something went wrong".to_string()),
                )
                .wait()
        })
    }

    pub fn destroy(&mut self, base: &ExampleBase) {
        for node in petgraph::algo::toposort(&self.graph, None)
            .expect("RenderDAGBuilder has cycles")
            .iter()
            .cloned()
        {
            match (self.graph[node].1).0 {
                NodeRuntime::BeginRenderPass(rp) => unsafe { base.device.destroy_render_pass(rp, None) },
                NodeRuntime::BindPipeline(pipeline, _, _) => unsafe { base.device.destroy_pipeline(pipeline, None) },
                NodeRuntime::Framebuffer(ref fbs) => for fb in fbs.iter().cloned() {
                    unsafe { base.device.destroy_framebuffer(fb, None) }
                },
                _ => (),
            }
        }

        for pipeline_layout in self.pipeline_layouts.values() {
            unsafe {
                base.device.destroy_pipeline_layout(*pipeline_layout, None);
            }
        }
    }
}

#[derive(Clone)]
pub struct BuilderNode {
    name: Cow<'static, str>,
}

pub struct RenderDAGBuilder {
    pub graph: BuilderGraph,
    name_mapping: HashMap<Cow<'static, str>, petgraph::graph::NodeIndex>,
}

pub struct CommandBufferBuilder {
    begin: BuilderNode,
    end: BuilderNode,
}

impl CommandBufferBuilder {
    pub fn add_node<T>(&self, builder: &mut RenderDAGBuilder, name: T, value: NodeBuilder) -> BuilderNode
    where
        T: Into<Cow<'static, str>>,
    {
        let node = builder.add_node(name, value);
        builder.add_edge(&self.begin, &node);
        node
    }

    pub fn start_before(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(&self.begin, node);
    }

    pub fn start_after(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(node, &self.begin);
    }

    pub fn end_before(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(&self.end, node);
    }

    pub fn end_after(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(node, &self.end);
    }

    pub fn with_renderpass<T>(&self, builder: &mut RenderDAGBuilder, name: T) -> RenderPassBuilder
    where
        T: Into<Cow<'static, str>>,
    {
        let name = name.into();
        let start_name = name.clone() + "-start";
        let end_name = name.clone() + "-end";
        let (start, end) = (
            self.add_node(builder, start_name, NodeBuilder::BeginRenderPass),
            self.add_node(builder, end_name, NodeBuilder::EndRenderPass),
        );
        self.start_before(builder, &start);
        builder.add_edge(&start, &end);
        self.end_after(builder, &end);
        RenderPassBuilder {
            begin: start,
            end: end,
            command_buffer_builder: self,
        }
    }
}

pub struct RenderPassBuilder<'a> {
    pub begin: BuilderNode,
    pub end: BuilderNode,
    pub command_buffer_builder: &'a CommandBufferBuilder,
}

impl<'a> RenderPassBuilder<'a> {
    pub fn add_node<T>(&self, builder: &mut RenderDAGBuilder, name: T, value: NodeBuilder) -> BuilderNode
    where
        T: Into<Cow<'static, str>>,
    {
        let node = builder.add_node(name, value);
        builder.add_edge(&self.begin, &node);
        node
    }

    pub fn start_before(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(&self.begin, node);
    }

    pub fn start_after(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(node, &self.begin);
    }

    pub fn end_before(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(&self.end, node);
    }

    pub fn end_after(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(node, &self.end);
    }

    pub fn with_subpass<T>(&self, builder: &mut RenderDAGBuilder, name: T, index: u8) -> SubPassBuilder
    where
        T: Into<Cow<'static, str>>,
    {
        let name = name.into();
        let start_name = name.clone() + "-start";
        let end_name = name.clone() + "-end";
        let (start, end) = (
            self.add_node(builder, start_name, NodeBuilder::BeginSubpass(index)),
            // self.add_node(builder, end_name, NodeBuilder::EndSubpass(index)),
            builder.add_node(end_name, NodeBuilder::EndSubpass(index)),
        );
        builder.add_edge(&start, &end);
        self.end_after(builder, &end);
        SubPassBuilder {
            begin: start,
            end: end,
            renderpass_builder: self,
        }
    }
}

pub struct SubPassBuilder<'a> {
    pub begin: BuilderNode,
    pub end: BuilderNode,
    pub renderpass_builder: &'a RenderPassBuilder<'a>,
}

impl<'a> SubPassBuilder<'a> {
    pub fn add_node<T>(&self, builder: &mut RenderDAGBuilder, name: T, value: NodeBuilder) -> BuilderNode
    where
        T: Into<Cow<'static, str>>,
    {
        let node = builder.add_node(name, value);
        builder.add_edge(&self.begin, &node);
        node
    }

    pub fn start_before(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(&self.begin, node);
    }

    pub fn start_after(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(node, &self.begin);
    }

    pub fn end_before(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(&self.end, node);
    }

    pub fn end_after(&self, builder: &mut RenderDAGBuilder, node: &BuilderNode) {
        builder.add_edge(node, &self.end);
    }
}

impl Default for RenderDAGBuilder {
    fn default() -> RenderDAGBuilder {
        RenderDAGBuilder::new()
    }
}

impl RenderDAGBuilder {
    pub fn new() -> RenderDAGBuilder {
        RenderDAGBuilder {
            graph: BuilderGraph::new(),
            name_mapping: HashMap::new(),
        }
    }

    pub fn with_command_buffer<T>(&mut self, name: T) -> CommandBufferBuilder
    where
        T: Into<Cow<'static, str>>,
    {
        let name = name.into();
        let start_name = name.clone() + "-start";
        let end_name = name.clone() + "-end";
        let (start, end) = (
            self.add_node(start_name, NodeBuilder::BeginCommandBuffer),
            self.add_node(end_name, NodeBuilder::EndCommandBuffer),
        );
        self.add_edge(&start, &end);
        CommandBufferBuilder {
            begin: start,
            end: end,
        }
    }

    pub fn add_node<T>(&mut self, name: T, value: NodeBuilder) -> BuilderNode
    where
        T: Into<Cow<'static, str>>,
    {
        let name = name.into();
        let ix = self.graph.add_node((name.clone(), value));
        assert!(self.name_mapping.insert(name.clone(), ix).is_none());
        BuilderNode { name: name }
    }

    pub fn add_edge(&mut self, from: &BuilderNode, to: &BuilderNode) {
        let from_ix = self.name_mapping[&from.name];
        let to_ix = self.name_mapping[&to.name];
        self.graph.update_edge(from_ix, to_ix, ());
    }

    pub fn build(self, base: &ExampleBase) -> RenderDAG {
        let mut output_graph = RuntimeGraph::new();
        let mut renderpasses: HashMap<Cow<'static, str>, vk::RenderPass> = HashMap::new();
        let mut pipeline_layouts: HashMap<Cow<'static, str>, vk::PipelineLayout> = HashMap::new();
        let mut pipelines: HashMap<Cow<'static, str>, (petgraph::graph::NodeIndex, vk::Pipeline)> = HashMap::new();
        let mut descriptor_set_layouts: HashMap<Cow<'static, str>, vk::DescriptorSetLayout> = HashMap::new();
        let mut descriptor_sets: HashMap<Cow<'static, str>, Vec<vk::DescriptorSet>> = HashMap::new();
        let mut name_mapping: HashMap<Cow<'static, str>, petgraph::graph::NodeIndex> = HashMap::new();
        let pool = CpuPool::new_num_cpus();

        for node in petgraph::algo::toposort(&self.graph, None)
            .expect("RenderDAGBuilder has cycles")
            .iter()
            .cloned()
        {
            let inputs = self.graph
                .neighbors_directed(node, petgraph::EdgeDirection::Incoming)
                .map(|ix| self.graph[ix].clone())
                .collect::<Vec<_>>();
            let outputs = self.graph
                .neighbors_directed(node, petgraph::EdgeDirection::Outgoing)
                .map(|ix| self.graph[ix].clone())
                .collect::<Vec<_>>();
            let this_name = self.graph[node].0.clone();
            println!("Building node: {}", this_name.clone());
            match self.graph[node].1 {
                NodeBuilder::BeginRenderPass => {
                    let mut attachments = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            NodeBuilder::SwapchainAttachment(ix) => Some((
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
                            NodeBuilder::DepthAttachment(ix) => Some((
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
                    let mut subpass_descs = outputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            NodeBuilder::BeginSubpass(ix) => Some((
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

                    let start = {
                        let renderpass = renderpass;
                        output_graph.add_node((
                            Cow::from("begin_render_pass"),
                            (
                                NodeRuntime::BeginRenderPass(renderpass),
                                Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                            ),
                        ))
                    };
                    /*
                    let end = {
                        let renderpass = renderpass.clone();
                        output_graph.add_node((
                            Cow::from("end_render_pass"),
                            Arc::new(RwLock::new(pool.spawn_fn(move || {
                                Ok(NodeRuntime::EndRenderPass)
                            }).shared())),
                        ))
                    };
                    */
                    renderpasses.insert(this_name.clone(), renderpass);
                    name_mapping.insert(this_name.clone(), start);

                    if let Some(acquire_image) = inputs
                        .iter()
                        .filter_map(|node| match *node {
                            (ref name, NodeBuilder::AcquirePresentImage) => name_mapping.get(name),
                            _ => None,
                        })
                        .next()
                    {
                        /*
                        let acquire_image = output_graph.add_node((
                            Cow::from("acquire_present_image"),
                            Arc::new(RwLock::new(pool.spawn_fn(|| {
                                Ok(NodeRuntime::AcquirePresentImage(987))
                            }).shared())),
                        ));
                        name_mapping.insert(Cow::from("acquire_present_image"), acquire_image);
                        */
                        output_graph.add_edge(*acquire_image, start, ());
                    }

                    let command_buffer = inputs
                        .iter()
                        .filter_map(|node| match *node {
                            (ref name, NodeBuilder::BeginCommandBuffer) => name_mapping.get(name),
                            _ => None,
                        })
                        .next()
                        .expect(&format!(
                            "No command buffer connected to RenderPass {}",
                            this_name
                        ));
                    output_graph.add_edge(*command_buffer, start, ());
                    // output_graph.add_edge(*command_buffer, end, ());
                }
                NodeBuilder::EndRenderPass => {
                    let node = {
                        output_graph.add_node((
                            this_name.clone(),
                            (
                                NodeRuntime::EndRenderPass,
                                Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                            ),
                        ))
                    };
                    name_mapping.insert(this_name.clone(), node);

                    let dependencies = inputs.iter().map(|node| node.0.clone());
                    for dependency in dependencies {
                        let mapping = *name_mapping.get(&dependency).expect(&format!(
                            "Dependency of EndRenderPass {}, {} does not have a mapping to output graph",
                            this_name, dependency
                        ));
                        output_graph.add_edge(mapping, node, ());
                    }
                }
                NodeBuilder::PresentImage => {
                    let node = {
                        output_graph.add_node((
                            this_name.clone(),
                            (
                                NodeRuntime::PresentImage,
                                Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                            ),
                        ))
                    };

                    let dependencies = inputs.iter().map(|node| node.0.clone());
                    for dependency in dependencies {
                        let mapping = *name_mapping.get(&dependency).expect(&format!(
                            "Dependency of PresentImage {}, {} does not have a mapping to output graph",
                            this_name, dependency
                        ));
                        output_graph.add_edge(mapping, node, ());
                    }
                }
                NodeBuilder::BeginSubpass(ix) => {
                    /*
                    let previous_subpasses = inputs
                        .iter()
                        .filter_map(|node| match node {
                            &(ref name, NodeBuilder::EndSubpass(_)) => name_mapping.get(name),
                            _ => None,
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                        */

                    let start = {
                        let ix = ix;
                        output_graph.add_node((
                            this_name.clone(),
                            (
                                NodeRuntime::BeginSubPass(ix),
                                Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                            ),
                        ))
                    };
                    name_mapping.insert(this_name.clone(), start);
                    /*
                    output_graph.add_edge(command_buffer, start, ());
                    for end_subpass in previous_subpasses {
                        output_graph.add_edge(end_subpass, start, ());
                    }
                    */
                    // let dependencies = inputs.iter().filter(|node| node.1.0 != "main_command_buffer-start").map(|node| node.0.clone());
                    let dependencies = inputs.iter().map(|node| node.0.clone());
                    for dependency in dependencies {
                        let mapping = *name_mapping.get(&dependency).expect(&format!(
                            "Dependency of BeginSubpass {}, {} does not have a mapping to output graph",
                            this_name, dependency
                        ));
                        output_graph.add_edge(mapping, start, ());
                    }
                }
                NodeBuilder::EndSubpass(ix) => {
                    let start = inputs
                        .iter()
                        .filter_map(|i| match *i {
                            (ref name, NodeBuilder::BeginSubpass(_)) => name_mapping.get(name),
                            _ => None,
                        })
                        .cloned()
                        .next()
                        .expect(&format!(
                            "BeginSubpass not attached to NodeBuilder::EndSubpass {}",
                            this_name
                        ));
                    let end = {
                        let ix = ix;
                        output_graph.add_node((
                            this_name.clone(),
                            (
                                NodeRuntime::EndSubPass(ix),
                                Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                            ),
                        ))
                    };
                    output_graph.update_edge(start, end, ());
                    name_mapping.insert(this_name.clone(), end);
                    let dependencies = inputs.iter().map(|node| node.0.clone());
                    for dependency in dependencies {
                        let mapping = *name_mapping.get(&dependency).expect(&format!(
                            "Dependency of EndSubpass {}, {} does not have a mapping to output graph",
                            this_name, dependency
                        ));
                        output_graph.update_edge(mapping, end, ());
                    }
                }
                NodeBuilder::PipelineLayout => {
                    let set_layouts = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            NodeBuilder::DescriptorSet(_) => Some(descriptor_set_layouts[&node.0]),
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

                    pipeline_layouts.insert(self.graph[node].0.clone(), pipeline_layout);
                }
                NodeBuilder::GraphicsPipeline => {
                    let vertex_attributes = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            NodeBuilder::VertexInputAttribute(binding, location, format, offset) => Some(vk::VertexInputAttributeDescription {
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
                            NodeBuilder::VertexInputBinding(binding, stride, rate) => Some(vk::VertexInputBindingDescription {
                                binding: binding,
                                stride: stride,
                                input_rate: rate,
                            }),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    println!("vertex bindings {:?}", vertex_bindings);
                    let shader_modules = inputs
                        .iter()
                        .filter_map(|node| match node.1 {
                            NodeBuilder::VertexShader(ref path) => {
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
                            NodeBuilder::GeometryShader(ref path) => {
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
                                        .expect("Geometry shader module error")
                                };

                                Some((shader_module, vk::SHADER_STAGE_GEOMETRY_BIT))
                            }
                            NodeBuilder::FragmentShader(ref path) => {
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
                        .filter_map(|node| match *node {
                            (ref name, NodeBuilder::PipelineLayout) => pipeline_layouts.get(name),
                            _ => None,
                        })
                        .next()
                        .expect("no pipeline layout specified for graphics pipeline");
                    let reversed = petgraph::visit::Reversed(&self.graph);
                    let bfs = petgraph::visit::Bfs::new(&reversed, node);
                    use petgraph::visit::Walker;
                    let renderpass_node = bfs.iter(&reversed)
                        .filter_map(|parent| match self.graph[parent] {
                            (ref name, NodeBuilder::BeginRenderPass) => name_mapping.get(name),
                            _ => None,
                        })
                        .cloned()
                        .next()
                        .expect(&format!(
                            "BFS search couldn't find BeginRenderPass for {}",
                            this_name.clone()
                        ));
                    let renderpass = match (output_graph[renderpass_node].1).0 {
                        NodeRuntime::BeginRenderPass(rp) => rp,
                        _ => panic!("renderpass does not exist"),
                    };
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
                            extent: base.surface_resolution,
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
                        cull_mode: vk::CULL_MODE_BACK_BIT,
                        depth_bias_clamp: 0.0,
                        depth_bias_constant_factor: 0.0,
                        depth_bias_enable: 0,
                        depth_bias_slope_factor: 0.0,
                        depth_clamp_enable: 0,
                        front_face: vk::FrontFace::Clockwise,
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
                        front: noop_stencil_state,
                        back: noop_stencil_state,
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
                        .filter_map(|node| match *node {
                            (ref _name, NodeBuilder::BeginSubpass(ix)) => Some(ix),
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
                        layout: *pipeline_layout,
                        render_pass: renderpass,
                        subpass: u32::from(subpass_ix),
                        base_pipeline_handle: vk::Pipeline::null(),
                        base_pipeline_index: 0,
                    };
                    let graphics_pipelines = unsafe {
                        base.device
                            .create_graphics_pipelines(vk::PipelineCache::null(), &[graphic_pipeline_info], None)
                            .expect("Unable to create graphics pipeline")
                    };

                    let graphics_pipeline = graphics_pipelines[0];

                    let bind = {
                        let pipeline = graphics_pipeline;
                        let scissors = scissors[0];
                        let viewports = viewports[0];
                        output_graph.add_node((
                            this_name.clone(),
                            (
                                NodeRuntime::BindPipeline(pipeline, Some(scissors), Some(viewports)),
                                Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                            ),
                        ))
                    };
                    name_mapping.insert(this_name.clone(), bind);
                    let subpass = inputs
                        .iter()
                        .filter_map(|i| match *i {
                            (ref name, NodeBuilder::BeginSubpass(_)) => name_mapping.get(name),
                            _ => None,
                        })
                        .cloned()
                        .next()
                        .expect(&format!(
                            "BeginSubpass not attached to NodeBuilder::GraphicsPipeline {}",
                            this_name
                        ));
                    output_graph.add_edge(subpass, bind, ());

                    pipelines.insert(this_name.clone(), (bind, graphics_pipeline));
                }
                NodeBuilder::DrawCommands(ref f) => {
                    let &(pipeline_bind, _) = inputs
                        .iter()
                        .filter_map(|node| match *node {
                            (ref _name, NodeBuilder::GraphicsPipeline) => pipelines.get(&node.0),
                            _ => None,
                        })
                        .next()
                        .expect("No pipeline specified for DrawCommands");
                    let draw = {
                        let f = f.clone();
                        output_graph.add_node((
                            this_name.clone(),
                            (
                                NodeRuntime::DrawCommands(f),
                                Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                            ),
                        ))
                    };
                    name_mapping.insert(this_name.clone(), draw);
                    output_graph.add_edge(pipeline_bind, draw, ());
                }
                NodeBuilder::DescriptorSet(size) => {
                    let descriptor_sizes = inputs
                        .iter()
                        .filter_map(|node| match *node {
                            (ref _name, NodeBuilder::DescriptorBinding(_binding, typ, _stage, count)) => Some(vk::DescriptorPoolSize {
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
                        .filter_map(|node| match *node {
                            (ref _name, NodeBuilder::DescriptorBinding(binding, typ, stage, count)) => Some(vk::DescriptorSetLayoutBinding {
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

                    descriptor_set_layouts.insert(self.graph[node].0.clone(), layouts[0]);
                    descriptor_sets.insert(self.graph[node].0.clone(), new_descriptor_sets);
                }
                NodeBuilder::Framebuffer => {
                    let renderpass = inputs
                        .iter()
                        .filter_map(|node| match *node {
                            (ref name, NodeBuilder::BeginRenderPass) => renderpasses.get(name),
                            _ => None,
                        })
                        .next()
                        .expect(&format!(
                            "No renderpass specified for Framebuffer {}",
                            this_name
                        ));
                    let renderpass_node = inputs
                        .iter()
                        .filter_map(|node| match *node {
                            (ref name, NodeBuilder::BeginRenderPass) => name_mapping.get(name),
                            _ => None,
                        })
                        .next()
                        .expect(&format!(
                            "No renderpass specified for Framebuffer {} or name mapping missing",
                            this_name
                        ));
                    let v: Vec<vk::Framebuffer> = base.present_image_views
                        .iter()
                        .map(|&present_image_view| {
                            let framebuffer_attachments = [present_image_view, base.depth_image_view];
                            let frame_buffer_create_info = vk::FramebufferCreateInfo {
                                s_type: vk::StructureType::FramebufferCreateInfo,
                                p_next: ptr::null(),
                                flags: Default::default(),
                                render_pass: *renderpass,
                                attachment_count: framebuffer_attachments.len() as u32,
                                p_attachments: framebuffer_attachments.as_ptr(),
                                width: base.surface_resolution.width,
                                height: base.surface_resolution.height,
                                layers: 1,
                            };
                            unsafe {
                                base.device
                                    .create_framebuffer(&frame_buffer_create_info, None)
                                    .unwrap()
                            }
                        })
                        .collect();
                    let fb = output_graph.add_node((
                        Cow::from("framebuffer"),
                        (
                            NodeRuntime::Framebuffer(v),
                            Arc::new(RwLock::new(pool.spawn_fn(|| Ok(None)).shared())),
                        ),
                    ));
                    output_graph.add_edge(fb, *renderpass_node, ());
                }
                NodeBuilder::BeginCommandBuffer => unsafe {
                    let pool_create_info = vk::CommandPoolCreateInfo {
                        s_type: vk::StructureType::CommandPoolCreateInfo,
                        p_next: ptr::null(),
                        flags: vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                        queue_family_index: base.queue_family_index,
                    };
                    let command_pool = base.device
                        .create_command_pool(&pool_create_info, None)
                        .unwrap();
                    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                        s_type: vk::StructureType::CommandBufferAllocateInfo,
                        p_next: ptr::null(),
                        command_buffer_count: 1,
                        command_pool: command_pool,
                        level: vk::CommandBufferLevel::Primary,
                    };
                    let command_buffers = base.device
                        .allocate_command_buffers(&command_buffer_allocate_info)
                        .unwrap();
                    let command_buffer = command_buffers[0];
                    name_mapping.insert(
                        this_name.clone(),
                        output_graph.add_node((
                            this_name.clone(),
                            (
                                NodeRuntime::BeginCommandBuffer(command_buffer),
                                Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                            ),
                        )),
                    );
                },
                NodeBuilder::EndCommandBuffer => {
                    let dependencies = inputs.iter().map(|node| node.0.clone());
                    let node = output_graph.add_node((
                        this_name.clone(),
                        (
                            NodeRuntime::EndCommandBuffer,
                            Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                        ),
                    ));
                    name_mapping.insert(this_name.clone(), node);
                    for dependency in dependencies {
                        let mapping = *name_mapping.get(&dependency).expect(&format!(
                            "Dependency of EndCommandBuffer {}, {} does not have a mapping to output graph",
                            this_name, dependency
                        ));
                        output_graph.add_edge(mapping, node, ());
                    }
                }
                NodeBuilder::AcquirePresentImage => {
                    let node = output_graph.add_node((
                        this_name.clone(),
                        (
                            NodeRuntime::AcquirePresentImage,
                            Arc::new(RwLock::new(pool.spawn(Ok(None).into_future()).shared())),
                        ),
                    ));
                    name_mapping.insert(this_name.clone(), node);
                }
                _ => (),
            }
        }
        RenderDAG {
            graph: output_graph,
            pipeline_layouts,
            descriptor_sets,
            pool,
        }
    }
}
