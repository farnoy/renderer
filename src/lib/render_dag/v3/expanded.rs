use futures::prelude::*;
use futures_cpupool::*;
use super::super::super::{device, entry, instance, swapchain};
use winit;
use super::{alloc, util::*};
use std::{fmt, sync::{Arc, Mutex, RwLock}};
use ash::vk;
use petgraph;
use specs;

#[allow(non_snake_case)]
pub mod fields {
    pub mod Framebuffer {
        #[derive(Debug)]
        pub struct Dynamic {
            pub current_present_index: u32,
        }
        impl Dynamic {
            pub fn new(current_present_index: u32) -> Dynamic {
                Dynamic {
                    current_present_index: current_present_index,
                }
            }
        }
    }
    #[allow(non_snake_case)]
    pub mod AllocateCommandBuffer {
        use super::super::vk;
        #[derive(Debug)]
        pub struct Dynamic {
            pub current_frame: vk::CommandBuffer,
        }
        impl Dynamic {
            pub fn new(current_frame: vk::CommandBuffer) -> Dynamic {
                Dynamic {
                    current_frame: current_frame,
                }
            }
        }
    }
}

#[derive(Clone)]
pub enum RenderNode {
    Instance {
        window: Arc<winit::Window>,
        events_loop: Arc<winit::EventsLoop>,
        instance: Arc<instance::Instance>,
        entry: Arc<entry::Entry>,
        surface: vk::SurfaceKHR,
        window_width: u32,
        window_height: u32,
        dynamic: Dynamic<()>,
    },
    Device {
        device: Arc<device::Device>,
        physical_device: vk::PhysicalDevice,
        allocator: alloc::VmaAllocator,
        graphics_queue_family: u32,
        compute_queue_family: u32,
        graphics_queue: Arc<Mutex<vk::Queue>>,
        compute_queues: Arc<Vec<Mutex<vk::Queue>>>,
        dynamic: Dynamic<()>,
    },
    Swapchain {
        handle: Arc<swapchain::Swapchain>,
        surface_format: vk::SurfaceFormatKHR,
        dynamic: Dynamic<()>,
    },
    Framebuffer {
        images: Arc<Vec<vk::Image>>,
        image_views: Arc<Vec<vk::ImageView>>,
        handles: Arc<Vec<vk::Framebuffer>>,
        dynamic: Dynamic<fields::Framebuffer::Dynamic>,
    },
    PresentFramebuffer {
        dynamic: Dynamic<()>,
    },
    CommandPool {
        handle: Arc<Mutex<vk::CommandPool>>,
        dynamic: Dynamic<()>,
    },
    PersistentSemaphore {
        handle: vk::Semaphore,
        dynamic: Dynamic<()>,
    },
    AllocateCommandBuffer {
        handles: Arc<RwLock<Vec<vk::CommandBuffer>>>,
        dynamic: Dynamic<fields::AllocateCommandBuffer::Dynamic>,
    },
    SubmitCommandBuffer {
        dynamic: Dynamic<()>,
    },
    Renderpass {
        handle: vk::RenderPass,
        dynamic: Dynamic<()>,
    },
    NextSubpass {
        ix: usize,
        dynamic: Dynamic<()>,
    },
    EndRenderpass {
        dynamic: Dynamic<()>,
    },
    DescriptorSetLayout {
        handle: vk::DescriptorSetLayout,
        dynamic: Dynamic<()>,
    },
    DescriptorPool {
        handle: vk::DescriptorPool,
        dynamic: Dynamic<()>,
    },
    DescriptorSet {
        handle: vk::DescriptorSet,
        dynamic: Dynamic<()>,
    },
    PipelineLayout {
        push_constant_ranges: Arc<Vec<vk::PushConstantRange>>,
        handle: vk::PipelineLayout,
        dynamic: Dynamic<()>,
    },
    GraphicsPipeline {
        handle: vk::Pipeline,
        dynamic: Dynamic<()>,
    },
    Buffer {
        handle: vk::Buffer,
        allocation: alloc::VmaAllocation,
        allocation_info: alloc::VmaAllocationInfo,
        dynamic: Dynamic<()>,
    },
    DrawCalls {
        f: Arc<
            Fn(
                petgraph::prelude::NodeIndex,
                &super::RuntimeGraph,
                &CpuPool,
                &specs::World,
                &Dynamic<()>,
            ),
        >,
        dynamic: Dynamic<()>,
    },
}
#[allow(unknown_lints)]
#[allow(too_many_arguments)]
impl RenderNode {
    pub fn make_allocate_commands(
        pool: &CpuPool,
        handles: Arc<RwLock<Vec<vk::CommandBuffer>>>,
        current_frame: vk::CommandBuffer,
    ) -> RenderNode {
        RenderNode::AllocateCommandBuffer {
            dynamic: dyn(
                pool,
                fields::AllocateCommandBuffer::Dynamic::new(current_frame),
            ),
            handles: handles,
        }
    }
    pub fn make_pipeline_layout(
        pool: &CpuPool,
        push_constant_ranges: Arc<Vec<vk::PushConstantRange>>,
        handle: vk::PipelineLayout,
    ) -> RenderNode {
        RenderNode::PipelineLayout {
            dynamic: dyn(pool, ()),
            push_constant_ranges: push_constant_ranges,
            handle: handle,
        }
    }
    pub fn make_graphics_pipeline(pool: &CpuPool, handle: vk::Pipeline) -> RenderNode {
        RenderNode::GraphicsPipeline {
            dynamic: dyn(pool, ()),
            handle: handle,
        }
    }
}
impl WaitOn for RenderNode {
    fn waitable(&self, pool: &CpuPool) -> CpuFuture<(), ()> {
        match *self {
            RenderNode::Instance { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::Device { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::Swapchain { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::Framebuffer { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::PresentFramebuffer { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::CommandPool { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::PersistentSemaphore { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::AllocateCommandBuffer { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::SubmitCommandBuffer { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::Renderpass { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::NextSubpass { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::EndRenderpass { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::DescriptorSetLayout { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::DescriptorPool { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::DescriptorSet { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::PipelineLayout { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::GraphicsPipeline { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::Buffer { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
            RenderNode::DrawCalls { ref dynamic, .. } => {
                let fut = dynamic
                    .read()
                    .expect("can't read the waitable dynamic")
                    .clone();
                pool.spawn(fut.map_err(|_| ()).map(|_| ()))
            }
        }
    }
}
impl fmt::Debug for RenderNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let output = match *self {
            RenderNode::Instance { .. } => stringify!(Instance),
            RenderNode::Device { .. } => stringify!(Device),
            RenderNode::Swapchain { .. } => stringify!(Swapchain),
            RenderNode::Framebuffer { .. } => stringify!(Framebuffer),
            RenderNode::PresentFramebuffer { .. } => stringify!(PresentFramebuffer),
            RenderNode::CommandPool { .. } => stringify!(CommandPool),
            RenderNode::PersistentSemaphore { .. } => stringify!(PersistentSemaphore),
            RenderNode::AllocateCommandBuffer { .. } => stringify!(AllocateCommandBuffer),
            RenderNode::SubmitCommandBuffer { .. } => stringify!(SubmitCommandBuffer),
            RenderNode::Renderpass { .. } => stringify!(Renderpass),
            RenderNode::NextSubpass { .. } => stringify!(NextSubpass),
            RenderNode::EndRenderpass { .. } => stringify!(EndRenderpass),
            RenderNode::DescriptorSetLayout { .. } => stringify!(DescriptorSetLayout),
            RenderNode::DescriptorPool { .. } => stringify!(DescriptorPool),
            RenderNode::DescriptorSet { .. } => stringify!(DescriptorSet),
            RenderNode::PipelineLayout { .. } => stringify!(PipelineLayout),
            RenderNode::GraphicsPipeline { .. } => stringify!(GraphicsPipeline),
            RenderNode::Buffer { .. } => stringify!(Buffer),
            RenderNode::DrawCalls { .. } => stringify!(DrawCalls),
        };
        write!(f, "{}", output)
    }
}
