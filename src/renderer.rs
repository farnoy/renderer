#![allow(clippy::too_many_arguments)]

// TODO: pub(crate) should disappear?
mod alloc;
mod device;
mod entry;
mod gltf_mesh;
mod helpers;
mod instance;
pub(crate) mod shaders;
mod swapchain;
mod systems;

pub(crate) use self::{
    device::*,
    gltf_mesh::{load as load_gltf, LoadedMesh},
    helpers::*,
    instance::Instance,
    swapchain::*,
    systems::{
        consolidate_mesh_buffers::*, crash_debugging::*, cull_pipeline::*, debug_aabb_renderer::*, present::*,
        shadow_mapping::*, textures::*,
    },
};
use crate::ecs::{
    components::{ModelMatrix, AABB},
    resources::Camera,
    systems::*,
};
use ash::{
    version::{DeviceV1_0, DeviceV1_2},
    vk,
};
use bevy_ecs::{component::Component, prelude::*};
use microprofile::scope;
use std::{convert::TryInto, mem::size_of, os::raw::c_uchar, sync::Arc};

pub(crate) fn up_vector() -> na::Unit<na::Vector3<f32>> {
    na::Unit::new_unchecked(na::Vector3::y())
}
pub(crate) fn forward_vector() -> na::Unit<na::Vector3<f32>> {
    na::Unit::new_unchecked(na::Vector3::z())
}
pub(crate) fn right_vector() -> na::Unit<na::Vector3<f32>> {
    na::Unit::new_unchecked(na::Vector3::x())
}

#[derive(Clone)]
pub(crate) struct GltfMesh {
    pub(crate) vertex_buffer: Arc<Buffer>,
    pub(crate) normal_buffer: Arc<Buffer>,
    pub(crate) uv_buffer: Arc<Buffer>,
    pub(crate) index_buffers: Arc<Vec<(Buffer, u64)>>,
    pub(crate) vertex_len: u64,
    pub(crate) aabb: ncollide3d::bounding_volume::AABB<f32>,
}

impl GltfMesh {
    pub(crate) fn destroy(self, device: &Device) {
        Arc::try_unwrap(self.vertex_buffer)
            .into_iter()
            .for_each(|b| b.destroy(device));
        Arc::try_unwrap(self.normal_buffer)
            .into_iter()
            .for_each(|b| b.destroy(device));
        Arc::try_unwrap(self.uv_buffer)
            .into_iter()
            .for_each(|b| b.destroy(device));
        Arc::try_unwrap(self.index_buffers)
            .into_iter()
            .for_each(|bs| bs.into_iter().for_each(|(b, _)| b.destroy(device)));
    }
}

#[derive(Debug, Default)]
pub(crate) struct DrawIndex(pub(crate) u32);

pub(crate) struct Position(pub(crate) na::Point3<f32>);
pub(crate) struct Rotation(pub(crate) na::UnitQuaternion<f32>);
pub(crate) struct Scale(pub(crate) f32);

// TODO: rename
pub(crate) struct RenderFrame {
    pub(crate) instance: Arc<Instance>,
    pub(crate) device: Device,
    pub(crate) graphics_timeline_semaphore: TimelineSemaphore,
    pub(crate) compute_timeline_semaphore: TimelineSemaphore,
    pub(crate) shadow_mapping_timeline_semaphore: TimelineSemaphore,
    pub(crate) consolidate_timeline_semaphore: TimelineSemaphore,
    pub(crate) transfer_timeline_semaphore: TimelineSemaphore,
    pub(crate) frame_number: u64,
    pub(crate) buffer_count: usize,
}

#[derive(Clone)]
pub(crate) struct SwapchainIndexToFrameNumber {
    pub(crate) map: DoubleBuffered<u64>,
}

impl FromWorld for SwapchainIndexToFrameNumber {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        SwapchainIndexToFrameNumber {
            map: renderer.new_buffered(|_| 0),
        }
    }
}

renderer_macros::define_timeline!(pub(crate) GraphicsTimeline [Start, SceneDraw]);
renderer_macros::define_timeline!(pub(crate) TransferTimeline [Perform]);
renderer_macros::define_timeline!(pub(crate) ShadowMappingTimeline [Prepare]);

renderer_macros::define_frame! {
    pub(crate) frame_graph {
        attachments {
            Color,
            Depth,
            ShadowMapAtlas
        }
        formats {
            dyn,
            D16_UNORM,
            D16_UNORM
        }
        passes {
            ShadowMapping {
                depth_stencil ShadowMapAtlas
                layouts {
                    ShadowMapAtlas load DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL => DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL store
                }
                subpasses {
                    ShadowMappingMain {
                        depth_stencil { ShadowMapAtlas => DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
                    }
                }
            },
            Main {
                color [Color]
                depth_stencil Depth
                layouts {
                    Depth clear UNDEFINED => DEPTH_STENCIL_READ_ONLY_OPTIMAL discard,
                    Color clear UNDEFINED => PRESENT_SRC_KHR store
                }
                subpasses {
                    DepthPrePass {
                        depth_stencil { Depth => DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
                    },
                    GltfPass {
                        color [Color => COLOR_ATTACHMENT_OPTIMAL]
                        depth_stencil { Depth => DEPTH_STENCIL_READ_ONLY_OPTIMAL }
                    },
                    GuiPass {
                        color [Color => COLOR_ATTACHMENT_OPTIMAL]
                    }
                }
                dependencies {
                    DepthPrePass => GltfPass
                        LATE_FRAGMENT_TESTS => EARLY_FRAGMENT_TESTS
                        DEPTH_STENCIL_ATTACHMENT_WRITE => DEPTH_STENCIL_ATTACHMENT_READ,
                    GltfPass => GuiPass
                        COLOR_ATTACHMENT_OUTPUT => COLOR_ATTACHMENT_OUTPUT
                        COLOR_ATTACHMENT_WRITE => COLOR_ATTACHMENT_READ
                }
            }
        }
        async_passes {
            // purely virtual, just to have a synchronization point at the start of the frame
            PresentationAcquire,
            ComputeCull,
            TransferCull,
            ConsolidateMeshBuffers,
        }
        dependencies {
            ShadowMapping => Main,
            // DepthOnly => Main,
            // Main => Gui,

            // PresentationAcquire => DepthOnly,
            PresentationAcquire => ShadowMapping,
            ShadowMapping => ShadowMapping [last_frame], // because the depth RT is not double buffered
            TransferCull => Main,
            ComputeCull => Main,
            ConsolidateMeshBuffers => Main,
            ConsolidateMeshBuffers => ConsolidateMeshBuffers [last_frame], // because the buffers are not double buffered
            // Gui => Gui [last_frame], // because the resources are not buffered
        }
        // TODO: validate so that if two passes signal the same timeline,
        //       there must be a proper dependency between them
        sync {
            PresentationAcquire => GraphicsTimeline::Start,
            Main => GraphicsTimeline::SceneDraw,
            TransferCull => TransferTimeline::Perform,
            ComputeCull => ComputeTimeline::Perform,
            ShadowMapping => ShadowMappingTimeline::Prepare,
            ConsolidateMeshBuffers => ConsolidateTimeline::Perform,
        }
    }
}

pub(crate) trait RenderStage {
    fn prepare_signal(render_frame: &RenderFrame, semaphores: &mut Vec<vk::Semaphore>, values: &mut Vec<u64>);

    fn prepare_wait(
        image_index: &ImageIndex,
        render_frame: &RenderFrame,
        semaphores: &mut Vec<vk::Semaphore>,
        values: &mut Vec<u64>,
    );

    fn host_signal(render_frame: &RenderFrame) -> ash::prelude::VkResult<()>;

    fn queue_submit(
        image_index: &ImageIndex,
        render_frame: &RenderFrame,
        queue: vk::Queue,
        command_buffers: &[vk::CommandBuffer],
    ) -> ash::prelude::VkResult<()> {
        scope!("vk", "vkQueueSubmit");

        Self::submit_info(image_index, render_frame, command_buffers, |submit_info| unsafe {
            render_frame
                .device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
        })
    }

    fn submit_info<T, F: FnOnce(vk::SubmitInfo) -> T>(
        image_index: &ImageIndex,
        render_frame: &RenderFrame,
        command_buffers: &[vk::CommandBuffer],
        f: F,
    ) -> T {
        let mut signal_semaphores = vec![];
        let mut signal_semaphore_values = vec![];
        Self::prepare_signal(&render_frame, &mut signal_semaphores, &mut signal_semaphore_values);
        let mut wait_semaphores = vec![];
        let mut wait_semaphore_values = vec![];
        Self::prepare_wait(
            &image_index,
            &render_frame,
            &mut wait_semaphores,
            &mut wait_semaphore_values,
        );
        let dst_stage_masks = vec![vk::PipelineStageFlags::TOP_OF_PIPE; wait_semaphores.len()];
        let mut wait_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(&wait_semaphore_values)
            .signal_semaphore_values(&signal_semaphore_values);
        let submit = vk::SubmitInfo::builder()
            .push_next(&mut wait_timeline)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&dst_stage_masks)
            .command_buffers(command_buffers)
            .signal_semaphores(&signal_semaphores)
            .build();

        f(submit)
    }
}

pub(crate) struct MainRenderpass {
    pub(crate) renderpass: frame_graph::Main::RenderPass,
}

impl MainRenderpass {
    pub(crate) fn new(renderer: &RenderFrame, attachments: &MainAttachments) -> Self {
        MainRenderpass {
            renderpass: frame_graph::Main::RenderPass::new(renderer, (attachments.swapchain_images[0].format,)),
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.renderpass.destroy(device);
    }
}

// define_timeline!(compute Perform);
renderer_macros::define_timeline!(pub(crate) ComputeTimeline [Perform]);

impl RenderFrame {
    pub(crate) fn new() -> (RenderFrame, Swapchain, winit::event_loop::EventLoop<()>) {
        let (instance, events_loop) = Instance::new().expect("Failed to create instance");
        let instance = Arc::new(instance);
        let surface = Surface::new(&instance);
        let device = Device::new(&instance, &surface).expect("Failed to create device");
        device.set_object_name(device.handle(), "Device");
        let swapchain = Swapchain::new(&instance, &device, surface);

        // Stat frame number at 1 and semaphores at 16, because validation layers assert
        // wait_semaphore_values at > 0
        let frame_number = 1;
        let graphics_timeline_semaphore =
            device.new_semaphore_timeline(GraphicsTimeline::SceneDraw.as_of_last(frame_number));
        device.set_object_name(graphics_timeline_semaphore.handle, "Graphics timeline semaphore");
        let compute_timeline_semaphore =
            device.new_semaphore_timeline(ComputeTimeline::Perform.as_of_last(frame_number));
        device.set_object_name(compute_timeline_semaphore.handle, "Compute timeline semaphore");
        let shadow_mapping_timeline_semaphore =
            device.new_semaphore_timeline(ShadowMappingTimeline::Prepare.as_of_last(frame_number));
        device.set_object_name(
            shadow_mapping_timeline_semaphore.handle,
            "Shadow mapping timeline semaphore",
        );

        let consolidate_timeline_semaphore =
            device.new_semaphore_timeline(ConsolidateTimeline::Perform.as_of_last(frame_number));
        device.set_object_name(
            consolidate_timeline_semaphore.handle,
            "Consolidate mesh buffers timeline semaphore",
        );

        let transfer_timeline_semaphore =
            device.new_semaphore_timeline(TransferTimeline::Perform.as_of_last(frame_number));
        device.set_object_name(transfer_timeline_semaphore.handle, "Transfer timeline semaphore");

        let buffer_count = unsafe { swapchain.ext.get_swapchain_images(swapchain.swapchain).unwrap().len() };

        (
            RenderFrame {
                instance: Arc::clone(&instance),
                device,
                graphics_timeline_semaphore,
                compute_timeline_semaphore,
                shadow_mapping_timeline_semaphore,
                consolidate_timeline_semaphore,
                transfer_timeline_semaphore,
                frame_number,
                buffer_count,
            },
            swapchain,
            events_loop,
        )
    }

    pub(crate) fn new_buffered<T, F: FnMut(u32) -> T>(&self, creator: F) -> DoubleBuffered<T> {
        DoubleBuffered::new(self.buffer_count, creator)
    }

    pub(crate) fn destroy(self) {
        self.graphics_timeline_semaphore.destroy(&self.device);
        self.compute_timeline_semaphore.destroy(&self.device);
        self.shadow_mapping_timeline_semaphore.destroy(&self.device);
        self.consolidate_timeline_semaphore.destroy(&self.device);
        self.transfer_timeline_semaphore.destroy(&self.device);
    }
}

pub(crate) struct MainDescriptorPool(pub(crate) DescriptorPool);

impl MainDescriptorPool {
    pub(crate) fn new(renderer: &RenderFrame) -> MainDescriptorPool {
        let descriptor_pool = renderer.device.new_descriptor_pool(
            3_000,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 4096_00,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 16384_00,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 4096_00,
                },
            ],
        );
        renderer
            .device
            .set_object_name(descriptor_pool.handle, "Main Descriptor Pool");

        MainDescriptorPool(descriptor_pool)
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.0.destroy(device);
    }
}

pub(crate) struct MainAttachments {
    #[allow(unused)]
    swapchain_images: Vec<SwapchainImage>,
    swapchain_image_views: Vec<ImageView>,
    #[allow(unused)]
    depth_images: Vec<Image>,
    depth_image_views: Vec<ImageView>,
}

impl MainAttachments {
    pub(crate) fn new(renderer: &RenderFrame, swapchain: &Swapchain) -> MainAttachments {
        let images = unsafe { swapchain.ext.get_swapchain_images(swapchain.swapchain).unwrap() };
        println!("swapchain images len {}", images.len());
        let depth_images = (0..images.len())
            .map(|ix| {
                let im = renderer.device.new_image(
                    vk::Format::D16_UNORM,
                    vk::Extent3D {
                        width: swapchain.width,
                        height: swapchain.height,
                        depth: 1,
                    },
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                );
                renderer
                    .device
                    .set_object_name(im.handle, &format!("Depth Target[{}]", ix));
                im
            })
            .collect::<Vec<_>>();
        let image_views = images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain.surface.surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                let handle = unsafe { renderer.device.create_image_view(&create_view_info, None).unwrap() };

                ImageView { handle }
            })
            .collect::<Vec<_>>();
        let depth_image_views = depth_images
            .iter()
            .map(|ref image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D16_UNORM)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image.handle);
                let handle = unsafe { renderer.device.create_image_view(&create_view_info, None).unwrap() };
                ImageView { handle }
            })
            .collect::<Vec<_>>();

        MainAttachments {
            swapchain_images: images
                .iter()
                .cloned()
                .map(|handle| SwapchainImage {
                    handle,
                    format: swapchain.surface.surface_format.format,
                })
                .collect(),
            swapchain_image_views: image_views,
            depth_images,
            depth_image_views,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        for view in self.swapchain_image_views.into_iter() {
            view.destroy(device);
        }
        for view in self.depth_image_views.into_iter() {
            view.destroy(device);
        }
        for depth in self.depth_images.into_iter() {
            depth.destroy(device);
        }
    }
}

pub(crate) struct MainFramebuffer {
    pub(crate) handles: DoubleBuffered<frame_graph::Main::Framebuffer>,
}

impl FromWorld for MainFramebuffer {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let main_attachments = world.get_resource::<MainAttachments>().unwrap();
        let main_renderpass = world.get_resource::<MainRenderpass>().unwrap();
        let swapchain = world.get_resource::<Swapchain>().unwrap();
        Self::new(&renderer, &main_renderpass, &main_attachments, &swapchain)
    }
}

impl MainFramebuffer {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_renderpass: &MainRenderpass,
        main_attachments: &MainAttachments,
        swapchain: &Swapchain,
    ) -> MainFramebuffer {
        let handles = main_attachments
            .swapchain_image_views
            .iter()
            .zip(main_attachments.depth_image_views.iter())
            .enumerate()
            .map(|(ix, (present_image_view, depth_image_view))| {
                let framebuffer_attachments = [present_image_view.handle, depth_image_view.handle];
                frame_graph::Main::Framebuffer::new(
                    renderer,
                    &main_renderpass.renderpass,
                    &framebuffer_attachments,
                    (swapchain.width, swapchain.height),
                    ix as u32,
                )
            })
            .collect::<Vec<_>>();

        MainFramebuffer {
            handles: DoubleBuffered::import(handles),
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.handles.into_iter().for_each(|f| f.destroy(device));
    }
}

pub(crate) struct LocalGraphicsCommandPool<const NAME: usize> {
    pub(crate) pools: DoubleBuffered<StrictCommandPool>,
}

impl<const NAME: usize> FromWorld for LocalGraphicsCommandPool<NAME> {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        LocalGraphicsCommandPool {
            pools: renderer.new_buffered(|ix| {
                StrictCommandPool::new(
                    &renderer.device,
                    renderer.device.graphics_queue_family,
                    &format!("Local[{}] Graphics Command Pool[{}]", NAME, ix),
                )
            }),
        }
    }
}

impl<const NAME: usize> LocalTransferCommandPool<NAME> {
    pub(crate) fn destroy(self, device: &Device) {
        self.pools.into_iter().for_each(|p| p.destroy(device));
    }
}

pub(crate) struct LocalTransferCommandPool<const NAME: usize> {
    pub(crate) pools: DoubleBuffered<StrictCommandPool>,
}

impl<const NAME: usize> FromWorld for LocalTransferCommandPool<NAME> {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        LocalTransferCommandPool {
            pools: renderer.new_buffered(|ix| {
                StrictCommandPool::new(
                    &renderer.device,
                    renderer.device.transfer_queue_family,
                    &format!("Local[{}] Transfer Command Pool[{}]", NAME, ix),
                )
            }),
        }
    }
}

impl<const NAME: usize> LocalGraphicsCommandPool<NAME> {
    pub(crate) fn destroy(self, device: &Device) {
        self.pools.into_iter().for_each(|p| p.destroy(device));
    }
}

pub(crate) struct CameraMatrices {
    pub(crate) set_layout: shaders::camera_set::Layout,
    buffer: DoubleBuffered<shaders::camera_set::bindings::matrices::Buffer>,
    set: DoubleBuffered<shaders::camera_set::Set>,
}

impl CameraMatrices {
    pub(crate) fn new(renderer: &RenderFrame, main_descriptor_pool: &MainDescriptorPool) -> CameraMatrices {
        let buffer = renderer.new_buffered(|ix| {
            let b = renderer.device.new_static_buffer(
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );
            renderer
                .device
                .set_object_name(b.buffer.handle, &format!("Camera matrices Buffer - ix={}", ix));
            b
        });
        let set_layout = shaders::camera_set::Layout::new(&renderer.device);
        let set = renderer.new_buffered(|ix| {
            let mut s = shaders::camera_set::Set::new(&renderer.device, &main_descriptor_pool, &set_layout, ix);

            shaders::camera_set::bindings::matrices::update_whole_buffer(&renderer.device, &mut s, &buffer.current(ix));

            s
        });

        CameraMatrices {
            set_layout,
            buffer,
            set,
        }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.set_layout.destroy(device);
        self.buffer.into_iter().for_each(|b| b.destroy(device));
        self.set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
    }
}

pub(crate) struct ModelData {
    pub(crate) model_set_layout: shaders::model_set::Layout,
    pub(crate) model_set: DoubleBuffered<shaders::model_set::Set>,
    pub(crate) model_buffer: DoubleBuffered<shaders::model_set::bindings::model::Buffer>,
}

impl ModelData {
    pub(crate) fn new(renderer: &RenderFrame, main_descriptor_pool: &MainDescriptorPool) -> ModelData {
        let device = &renderer.device;

        let model_set_layout = shaders::model_set::Layout::new(&device);

        let model_buffer = renderer.new_buffered(|ix| {
            let b = device.new_static_buffer(
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );
            device.set_object_name(b.buffer.handle, &format!("Model Buffer - {}", ix));
            b
        });
        let model_set = renderer.new_buffered(|ix| {
            let mut s = shaders::model_set::Set::new(&renderer.device, &main_descriptor_pool, &model_set_layout, ix);
            shaders::model_set::bindings::model::update_whole_buffer(
                &renderer.device,
                &mut s,
                &model_buffer.current(ix),
            );
            s
        });

        ModelData {
            model_set_layout,
            model_set,
            model_buffer,
        }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.model_set_layout.destroy(device);
        self.model_buffer.into_iter().for_each(|b| b.destroy(device));
        self.model_set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
    }
}

pub(crate) struct DepthPassData {
    pub(crate) depth_pipeline: Pipeline,
    pub(crate) depth_pipeline_layout: shaders::depth_pipe::PipelineLayout,
}

impl DepthPassData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        model_data: &ModelData,
        camera_matrices: &CameraMatrices,
        main_renderpass: &MainRenderpass,
    ) -> DepthPassData {
        let device = &renderer.device;

        let depth_pipeline_layout = shaders::depth_pipe::PipelineLayout::new(
            &device,
            &model_data.model_set_layout,
            &camera_matrices.set_layout,
        );
        let depth_pipeline = device.new_graphics_pipeline(
            &[(vk::ShaderStageFlags::VERTEX, shaders::depth_pipe::VERTEX, None)],
            vk::GraphicsPipelineCreateInfo::builder()
                .vertex_input_state(&shaders::depth_pipe::vertex_input_state())
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                        .build(),
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
                        .polygon_mode(vk::PolygonMode::FILL)
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
                .layout(*depth_pipeline_layout.layout)
                .render_pass(main_renderpass.renderpass.renderpass.handle)
                .subpass(0) // FIXME
                .build(),
        );

        device.set_object_name(*depth_pipeline, "Depth Pipeline");

        DepthPassData {
            depth_pipeline,
            depth_pipeline_layout,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.depth_pipeline.destroy(device);
        self.depth_pipeline_layout.destroy(device);
    }
}

pub(crate) struct Resized(pub(crate) bool);

pub(crate) struct GltfPassData {
    pub(crate) gltf_pipeline: Pipeline,
    pub(crate) gltf_pipeline_layout: shaders::gltf_mesh::PipelineLayout,
}

impl GltfPassData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_renderpass: &MainRenderpass,
        model_data: &ModelData,
        base_color: &BaseColorDescriptorSet,
        shadow_mapping: &ShadowMappingData,
        camera_matrices: &CameraMatrices,
    ) -> GltfPassData {
        let device = &renderer.device;

        /*
        let queue_family_indices = vec![device.graphics_queue_family];
        let image_create_info = vk::ImageCreateInfo::builder()
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: 20 * 16384,
                height: 20 * 16384,
                depth: 1,
            })
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .mip_levels(1)
            .array_layers(1)
            .image_type(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .flags(vk::ImageCreateFlags::SPARSE_BINDING | vk::ImageCreateFlags::SPARSE_RESIDENCY)
            .queue_family_indices(&queue_family_indices);

        let image = unsafe { device.create_image(&image_create_info, None).unwrap() };

        unsafe {
            let mut requirements = device.get_image_memory_requirements(image);
            requirements.size = requirements.alignment;
            let requirements = vec![requirements; 5];
            let create_infos = vec![
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
            ];
            use std::mem::MaybeUninit;
            let mut allocations: [MaybeUninit<alloc::VmaAllocation>; 5] =
                MaybeUninit::uninit_array();
            let mut allocation_infos: [MaybeUninit<alloc::VmaAllocationInfo>; 5] =
                MaybeUninit::uninit_array();
            alloc::vmaAllocateMemoryPages(
                device.allocator,
                requirements.as_ptr(),
                create_infos.as_ptr(),
                5,
                MaybeUninit::first_ptr_mut(&mut allocations),
                MaybeUninit::first_ptr_mut(&mut allocation_infos),
            );

            let allocations = MaybeUninit::slice_get_ref(&allocations);
            let allocation_infos = MaybeUninit::slice_get_ref(&allocation_infos);

            let mut image_binds = vec![];

            for a in allocation_infos.iter() {
                image_binds.push(
                    vk::SparseImageMemoryBind::builder()
                        .memory(a.deviceMemory)
                        .extent(vk::Extent3D {
                            width: 128,
                            height: 128,
                            depth: 1,
                        })
                        .memory_offset(a.offset)
                        .subresource(
                            vk::ImageSubresource::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(0)
                                .build(),
                        )
                        .build(),
                );
            }

            let mut fence = device.new_fence();

            let image_info = vk::SparseImageMemoryBindInfo::builder()
                .binds(&image_binds)
                .image(image)
                .build();

            device.fp_v1_0().queue_bind_sparse(
                *device.graphics_queue.lock(),
                1,
                &*vk::BindSparseInfo::builder().image_binds(&[image_info])
                    as *const vk::BindSparseInfo,
                fence.handle,
            );
            fence.wait();
        }
        */

        let gltf_pipeline_layout = shaders::gltf_mesh::PipelineLayout::new(
            &renderer.device,
            &model_data.model_set_layout,
            &camera_matrices.set_layout,
            &shadow_mapping.user_set_layout,
            &base_color.layout,
        );
        use systems::shadow_mapping::DIM as SHADOW_MAP_DIM;
        let spec = shaders::gltf_mesh::Specialization {
            shadow_map_dim: SHADOW_MAP_DIM,
            shadow_map_dim_squared: SHADOW_MAP_DIM * SHADOW_MAP_DIM,
        };
        let spec_info = spec.get_spec_info();
        let gltf_pipeline = device.new_graphics_pipeline(
            &[
                (
                    vk::ShaderStageFlags::VERTEX,
                    shaders::gltf_mesh::VERTEX,
                    Some(&spec_info),
                ),
                (
                    vk::ShaderStageFlags::FRAGMENT,
                    shaders::gltf_mesh::FRAGMENT,
                    Some(&spec_info),
                ),
            ],
            vk::GraphicsPipelineCreateInfo::builder()
                .vertex_input_state(&shaders::gltf_mesh::vertex_input_state())
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                        .build(),
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
                        .polygon_mode(vk::PolygonMode::FILL)
                        // magic
                        // .depth_bias_enable(false)
                        // .depth_bias_constant_factor(-0.07)
                        // .depth_bias_slope_factor(-1.0)
                        .build(),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                        .build(),
                )
                .depth_stencil_state(
                    &vk::PipelineDepthStencilStateCreateInfo::builder()
                        .depth_write_enable(false)
                        .depth_test_enable(true)
                        .depth_compare_op(vk::CompareOp::EQUAL)
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
                .layout(*gltf_pipeline_layout.layout)
                .render_pass(main_renderpass.renderpass.renderpass.handle)
                .subpass(1) // FIXME
                .build(),
        );
        device.set_object_name(*gltf_pipeline, "GLTF Pipeline");

        GltfPassData {
            gltf_pipeline,
            gltf_pipeline_layout,
        }
    }
}

pub(crate) struct MainPassCommandBuffer(vk::CommandBuffer);

impl Default for MainPassCommandBuffer {
    fn default() -> Self {
        MainPassCommandBuffer(vk::CommandBuffer::null())
    }
}

pub(crate) fn render_frame(
    renderer: Res<RenderFrame>,
    main_renderpass: Res<MainRenderpass>,
    (image_index, model_data): (Res<ImageIndex>, Res<ModelData>),
    (runtime_config, mut runtime_config_gui): (Res<RuntimeConfiguration>, ResMut<GuiCopy<RuntimeConfiguration>>),
    camera_matrices: Res<CameraMatrices>,
    swapchain: Res<Swapchain>,
    consolidated_mesh_buffers: Res<ConsolidatedMeshBuffers>,
    mut local_graphics_command_pool: ResMut<LocalGraphicsCommandPool<2>>,
    debug_aabb_pass_data: Res<DebugAABBPassData>,
    shadow_mapping_data: Res<ShadowMappingData>,
    base_color_descriptor_set: Res<BaseColorDescriptorSet>,
    cull_pass_data: Res<CullPassData>,
    main_framebuffer: Res<MainFramebuffer>,
    (
        mut main_pass_cb,
        depth_pass_data,
        gltf_pass,
        mut gui_render_data,
        camera,
        mut camera_gui,
        mut input_handler,
        mut gui,
    ): (
        ResMut<MainPassCommandBuffer>,
        Res<DepthPassData>,
        Res<GltfPassData>,
        ResMut<GuiRenderData>,
        Res<Camera>,
        ResMut<GuiCopy<Camera>>,
        NonSendMut<InputHandler>,
        NonSendMut<Gui>,
    ),
    crash_buffer: Res<CrashBuffer>,
    query: Query<&AABB>,
) {
    microprofile::scope!("ecs", "render_frame");

    // TODO: count this? pack and defragment draw calls?
    let total = shaders::cull_set::bindings::indirect_commands::SIZE as u32
        / size_of::<shaders::VkDrawIndexedIndirectCommand>() as u32;

    let command_pool = local_graphics_command_pool.pools.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let mut command_session = command_pool.session(&renderer.device);

    let command_buffer = command_session.record_one_time("Main Render CommandBuffer");
    unsafe {
        let _main_renderpass_marker = command_buffer.debug_marker_around("main renderpass", [0.0, 0.0, 1.0, 1.0]);
        main_renderpass.renderpass.begin(
            &renderer,
            main_framebuffer.handles.current(image_index.0),
            *command_buffer,
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.width,
                    height: swapchain.height,
                },
            },
            &[
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                },
            ],
        );
        depth_only_pass(
            &renderer,
            &image_index,
            &depth_pass_data,
            &cull_pass_data,
            &consolidated_mesh_buffers,
            &model_data,
            &runtime_config,
            &camera_matrices,
            &swapchain,
            &command_buffer,
        );
        renderer
            .device
            .cmd_next_subpass(*command_buffer, vk::SubpassContents::INLINE);
        renderer.device.cmd_set_viewport(
            *command_buffer,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: swapchain.height as f32,
                width: swapchain.width as f32,
                height: -(swapchain.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        renderer.device.cmd_set_scissor(
            *command_buffer,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.width,
                    height: swapchain.height,
                },
            }],
        );
        if runtime_config.debug_aabbs {
            microprofile::scope!("ecs", "debug aabb pass");

            let _aabb_marker = command_buffer.debug_marker_around("aabb debug", [1.0, 0.0, 0.0, 1.0]);
            renderer.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *debug_aabb_pass_data.pipeline,
            );
            debug_aabb_pass_data.pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                &camera_matrices.set.current(image_index.0),
            );

            for aabb in &mut query.iter() {
                debug_aabb_pass_data.pipeline_layout.push_constants(
                    &renderer.device,
                    *command_buffer,
                    &shaders::debug_aabb::PushConstants {
                        center: aabb.0.center().coords,
                        half_extent: aabb.0.half_extents(),
                    },
                );
                renderer.device.cmd_draw(*command_buffer, 36, 1, 0, 0);
            }
        } else {
            let _gltf_meshes_marker = command_buffer.debug_marker_around("gltf meshes", [1.0, 0.0, 0.0, 1.0]);
            /*
            renderer.device.cmd_clear_attachments(
                *command_buffer,
                &[vk::ClearAttachment::builder()
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 1,
                        },
                    })
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .build()],
                &[vk::ClearRect::builder()
                    .rect(vk::Rect2D {
                        offset: vk::Offset2D {
                            x: 0,
                            y: 0,
                        },
                        extent: vk::Extent2D {
                            width: swapchain.width,
                            height: swapchain.height,
                        },
                    })
                    .layer_count(1)
                    .base_array_layer(0)
                    .build()],
            );
            */
            // gltf mesh
            renderer.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *gltf_pass.gltf_pipeline,
            );
            gltf_pass.gltf_pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                &model_data.model_set.current(image_index.0),
                &camera_matrices.set.current(image_index.0),
                &shadow_mapping_data.user_set.current(image_index.0),
                &base_color_descriptor_set.set.current(image_index.0),
            );
            renderer.device.cmd_bind_index_buffer(
                *command_buffer,
                cull_pass_data.culled_index_buffer.current(image_index.0).buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            renderer.device.cmd_bind_vertex_buffers(
                *command_buffer,
                0,
                &[
                    consolidated_mesh_buffers.position_buffer.buffer.handle,
                    consolidated_mesh_buffers.normal_buffer.buffer.handle,
                    consolidated_mesh_buffers.uv_buffer.handle,
                ],
                &[0, 0, 0],
            );
            renderer.device.cmd_draw_indexed_indirect_count(
                *command_buffer,
                cull_pass_data
                    .culled_commands_buffer
                    .current(image_index.0)
                    .buffer
                    .handle,
                0,
                cull_pass_data
                    .culled_commands_count_buffer
                    .current(image_index.0)
                    .buffer
                    .handle,
                0,
                total,
                size_of::<shaders::VkDrawIndexedIndirectCommand>() as u32,
            );
            crash_buffer.record(
                &renderer,
                *command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                &image_index,
                0,
            );
        }
        renderer
            .device
            .cmd_next_subpass(*command_buffer, vk::SubpassContents::INLINE);
        render_gui(
            &renderer,
            &mut gui_render_data,
            &runtime_config,
            &mut runtime_config_gui,
            &swapchain,
            &camera,
            &mut camera_gui,
            &mut input_handler,
            &mut gui,
            &command_buffer,
        );
        renderer.device.cmd_end_render_pass(*command_buffer);
    }

    let command_buffer = *command_buffer.end();

    main_pass_cb.0 = command_buffer;
}

fn submit_main_pass(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    main_pass_cb: Res<MainPassCommandBuffer>,
) {
    scope!("ecs", "submit_main_pass");
    debug_assert_ne!(main_pass_cb.0, vk::CommandBuffer::null());

    let queue = renderer.device.graphics_queue().lock();

    frame_graph::Main::Stage::queue_submit(&image_index, &renderer, *queue, &[main_pass_cb.0]).unwrap();
}

pub(crate) struct GuiRenderData {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    texture: Image,
    #[allow(unused)]
    texture_view: ImageView,
    #[allow(unused)]
    sampler: Sampler,
    #[allow(unused)]
    descriptor_set_layout: shaders::imgui_set::Layout,
    descriptor_set: shaders::imgui_set::Set,
    pipeline_layout: shaders::imgui_pipe::PipelineLayout,
    pipeline: Pipeline,
    command_pool: DoubleBuffered<StrictCommandPool>,
}

impl FromWorld for GuiRenderData {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let main_descriptor_pool = world.get_resource::<MainDescriptorPool>().unwrap();
        let main_renderpass = world.get_resource::<MainRenderpass>().unwrap();
        let mut gui = unsafe { world.get_non_send_resource_unchecked_mut::<Gui>().unwrap() };
        Self::new(&renderer, &main_descriptor_pool, &mut gui, &main_renderpass)
    }
}

impl GuiRenderData {
    fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &MainDescriptorPool,
        gui: &mut Gui,
        main_renderpass: &MainRenderpass,
    ) -> GuiRenderData {
        let imgui = &mut gui.imgui;
        imgui
            .io_mut()
            .backend_flags
            .insert(imgui::BackendFlags::RENDERER_HAS_VTX_OFFSET);
        let vertex_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            1024 * 1024 * size_of::<imgui::DrawVert>() as vk::DeviceSize,
        );
        renderer
            .device
            .set_object_name(vertex_buffer.handle, "GUI Vertex Buffer");
        let index_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::INDEX_BUFFER,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            1024 * 1024 * size_of::<imgui::DrawIdx>() as vk::DeviceSize,
        );
        renderer.device.set_object_name(index_buffer.handle, "GUI Index Buffer");
        let texture = {
            let mut fonts = imgui.fonts();
            let imgui_texture = fonts.build_rgba32_texture();
            let texture = renderer.device.new_image(
                vk::Format::R8G8B8A8_UNORM,
                vk::Extent3D {
                    width: imgui_texture.width,
                    height: imgui_texture.height,
                    depth: 1,
                },
                vk::SampleCountFlags::TYPE_1,
                vk::ImageTiling::LINEAR, // todo use optimal?
                vk::ImageLayout::PREINITIALIZED,
                vk::ImageUsageFlags::SAMPLED,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );
            {
                let mut texture_data = texture
                    .map::<c_uchar>(&renderer.device)
                    .expect("failed to map imgui texture");
                texture_data[0..imgui_texture.data.len()].copy_from_slice(imgui_texture.data);
            }
            texture
        };
        let sampler = new_sampler(
            &renderer.device,
            &vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
        );

        let descriptor_set_layout = shaders::imgui_set::Layout::new(&renderer.device);

        let descriptor_set = shaders::imgui_set::Set::new(
            &renderer.device,
            &main_descriptor_pool,
            &descriptor_set_layout,
            0, // FIXME
        );

        let pipeline_layout = shaders::imgui_pipe::PipelineLayout::new(&renderer.device, &descriptor_set_layout);

        let pipeline = renderer.device.new_graphics_pipeline(
            &[
                (vk::ShaderStageFlags::VERTEX, shaders::imgui_pipe::VERTEX, None),
                (vk::ShaderStageFlags::FRAGMENT, shaders::imgui_pipe::FRAGMENT, None),
            ],
            vk::GraphicsPipelineCreateInfo::builder()
                .vertex_input_state(
                    &vk::PipelineVertexInputStateCreateInfo::builder()
                        .vertex_attribute_descriptions(&[
                            vk::VertexInputAttributeDescription {
                                location: 0,
                                binding: 0,
                                format: vk::Format::R32G32_SFLOAT,
                                offset: 0,
                            },
                            vk::VertexInputAttributeDescription {
                                location: 1,
                                binding: 0,
                                format: vk::Format::R32G32_SFLOAT,
                                offset: size_of::<f32>() as u32 * 2,
                            },
                            vk::VertexInputAttributeDescription {
                                location: 2,
                                binding: 0,
                                format: vk::Format::R8G8B8A8_UNORM,
                                offset: size_of::<f32>() as u32 * 4,
                            },
                        ])
                        .vertex_binding_descriptions(&[vk::VertexInputBindingDescription {
                            binding: 0,
                            stride: size_of::<imgui::DrawVert>() as u32,
                            input_rate: vk::VertexInputRate::VERTEX,
                        }])
                        .build(),
                )
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                        .build(),
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
                        .polygon_mode(vk::PolygonMode::FILL)
                        .build(),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
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
                .layout(*pipeline_layout.layout)
                .render_pass(main_renderpass.renderpass.renderpass.handle)
                .subpass(2) // FIXME
                .build(),
        );
        renderer.device.set_object_name(*pipeline, "GUI Pipeline");

        let texture_view = new_image_view(
            &renderer.device,
            &vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(texture.handle),
        );

        unsafe {
            renderer.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set.set.handle)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&[vk::DescriptorImageInfo::builder()
                        .sampler(sampler.handle)
                        .image_view(texture_view.handle)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .build()])
                    .build()],
                &[],
            );
        }

        let mut command_pool = renderer.new_buffered(|ix| {
            StrictCommandPool::new(
                &renderer.device,
                renderer.device.graphics_queue_family,
                &format!("GuiRender Command Pool[{}]", ix),
            )
        });

        let mut session = command_pool.current_mut(0).session(&renderer.device);
        let cb = session.record_one_time("prepare gui texture");
        unsafe {
            renderer.device.cmd_pipeline_barrier(
                *cb,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::HOST,
                Default::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .image(texture.handle)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .old_layout(vk::ImageLayout::PREINITIALIZED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .build()],
            );
            let cb = cb.end();
            let fence = renderer.device.new_fence();
            let queue = renderer.device.graphics_queue().lock();
            renderer
                .device
                .queue_submit(
                    *queue,
                    &[vk::SubmitInfo::builder().command_buffers(&[*cb]).build()],
                    fence.handle,
                )
                .unwrap();
            renderer
                .device
                .wait_for_fences(&[fence.handle], true, u64::MAX)
                .unwrap();
            fence.destroy(&renderer.device);
        }

        GuiRenderData {
            vertex_buffer,
            index_buffer,
            texture,
            texture_view,
            sampler,
            descriptor_set_layout,
            descriptor_set,
            pipeline_layout,
            pipeline,
            command_pool,
        }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.pipeline.destroy(device);
        self.pipeline_layout.destroy(device);
        self.sampler.destroy(device);
        self.texture_view.destroy(device);
        self.texture.destroy(device);
        self.descriptor_set.destroy(&main_descriptor_pool.0, device);
        self.descriptor_set_layout.destroy(device);
        self.command_pool.into_iter().for_each(|p| p.destroy(device));
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}

pub(crate) struct GuiCopy<T>(pub(crate) T);

impl<T: FromWorld> FromWorld for GuiCopy<T> {
    fn from_world(world: &mut World) -> Self {
        GuiCopy(<T as FromWorld>::from_world(world))
    }
}

fn render_gui(
    renderer: &RenderFrame,
    gui_render_data: &mut GuiRenderData,
    runtime_config: &RuntimeConfiguration,
    runtime_config_copy: &mut GuiCopy<RuntimeConfiguration>,
    swapchain: &Swapchain,
    camera: &Camera,
    camera_copy: &mut GuiCopy<Camera>,
    input_handler: &mut InputHandler,
    gui: &mut Gui,
    command_buffer: &StrictRecordingCommandBuffer,
) {
    scope!("rendering", "render_gui");

    let GuiRenderData {
        ref vertex_buffer,
        ref index_buffer,
        ref pipeline_layout,
        ref pipeline,
        ref descriptor_set,
        ..
    } = *gui_render_data;
    let gui_draw_data = gui.update(
        &renderer,
        input_handler,
        &swapchain,
        camera,
        camera_copy,
        runtime_config,
        runtime_config_copy,
    );

    let _gui_debug_marker = command_buffer.debug_marker_around("GUI", [1.0, 1.0, 0.0, 1.0]);
    unsafe {
        renderer.device.cmd_set_viewport(
            **command_buffer,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: swapchain.height as f32,
                width: swapchain.width as f32,
                height: -(swapchain.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        renderer.device.cmd_set_scissor(
            **command_buffer,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.width,
                    height: swapchain.height,
                },
            }],
        );

        pipeline_layout.bind_descriptor_sets(&renderer.device, **command_buffer, &descriptor_set);
        renderer
            .device
            .cmd_bind_pipeline(**command_buffer, vk::PipelineBindPoint::GRAPHICS, **pipeline);
        renderer
            .device
            .cmd_bind_vertex_buffers(**command_buffer, 0, &[vertex_buffer.handle], &[0]);
        renderer
            .device
            .cmd_bind_index_buffer(**command_buffer, index_buffer.handle, 0, vk::IndexType::UINT16);
        let [x, y] = gui_draw_data.display_size;
        {
            pipeline_layout.push_constants(
                &renderer.device,
                **command_buffer,
                &shaders::imgui_pipe::PushConstants {
                    scale: glm::vec2(2.0 / x, 2.0 / y),
                    translate: glm::vec2(-1.0, -1.0),
                },
            );
        }
        {
            let mut vertex_offset_coarse: usize = 0;
            let mut index_offset_coarse: usize = 0;
            let mut vertex_slice = vertex_buffer
                .map::<imgui::DrawVert>(&renderer.device)
                .expect("Failed to map gui vertex buffer");
            let mut index_slice = index_buffer
                .map::<imgui::DrawIdx>(&renderer.device)
                .expect("Failed to map gui index buffer");
            for draw_list in gui_draw_data.draw_lists() {
                let index_len = draw_list.idx_buffer().len();
                index_slice[index_offset_coarse..index_offset_coarse + index_len]
                    .copy_from_slice(draw_list.idx_buffer());
                let vertex_len = draw_list.vtx_buffer().len();
                vertex_slice[vertex_offset_coarse..vertex_offset_coarse + vertex_len]
                    .copy_from_slice(draw_list.vtx_buffer());
                for draw_cmd in draw_list.commands() {
                    match draw_cmd {
                        imgui::DrawCmd::Elements { count, cmd_params } => {
                            renderer.device.cmd_set_scissor(
                                **command_buffer,
                                0,
                                &[vk::Rect2D {
                                    offset: vk::Offset2D {
                                        x: cmd_params.clip_rect[0] as i32,
                                        y: cmd_params.clip_rect[1] as i32,
                                    },
                                    extent: vk::Extent2D {
                                        width: (cmd_params.clip_rect[2] - cmd_params.clip_rect[0]) as u32,
                                        height: (cmd_params.clip_rect[3] - cmd_params.clip_rect[1]) as u32,
                                    },
                                }],
                            );
                            renderer.device.cmd_draw_indexed(
                                **command_buffer,
                                count as u32,
                                1,
                                (index_offset_coarse + cmd_params.idx_offset) as u32,
                                (vertex_offset_coarse + cmd_params.vtx_offset) as i32,
                                0,
                            );
                        }
                        _ => panic!("le wtf"),
                    }
                }
                index_offset_coarse += index_len;
                vertex_offset_coarse += vertex_len;
            }
        }
    }
}

pub(crate) fn model_matrices_upload(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    mut model_data: ResMut<ModelData>,
    query: Query<(&DrawIndex, &ModelMatrix)>,
) {
    scope!("ecs", "ModelMatricesUpload");
    let mut model_mapped = model_data
        .model_buffer
        .current_mut(image_index.0)
        .map(&renderer.device)
        .expect("failed to map Model buffer");
    query.for_each(|(draw_index, model_matrix)| {
        model_mapped.model[draw_index.0 as usize] = model_matrix.0;
    });
}

pub(crate) fn camera_matrices_upload(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    camera: Res<Camera>,
    mut camera_matrices: ResMut<CameraMatrices>,
) {
    microprofile::scope!("ecs", "CameraMatricesUpload");
    let mut model_mapped = camera_matrices
        .buffer
        .current_mut(image_index.0)
        .map(&renderer.device)
        .expect("failed to map camera matrix buffer");
    *model_mapped = shaders::camera_set::bindings::matrices::T {
        projection: camera.projection,
        view: camera.view,
        position: camera.position.coords.push(1.0),
        pv: camera.projection * camera.view,
    };
}

fn depth_only_pass(
    renderer: &RenderFrame,
    image_index: &ImageIndex,
    depth_pass: &DepthPassData,
    cull_pass_data: &CullPassData,
    consolidated_mesh_buffers: &ConsolidatedMeshBuffers,
    model_data: &ModelData,
    runtime_config: &RuntimeConfiguration,
    camera_matrices: &CameraMatrices,
    swapchain: &Swapchain,
    command_buffer: &StrictRecordingCommandBuffer, // vk::CommandBuffer
) {
    scope!("rendering", "depth_only_pass");

    unsafe {
        scope!("depth only", "render commands");
        let _marker = command_buffer.debug_marker_around("depth prepass", [0.3, 0.3, 0.3, 1.0]);

        if runtime_config.debug_aabbs {
            return;
        }
        renderer.device.cmd_set_viewport(
            **command_buffer,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: swapchain.height as f32,
                width: swapchain.width as f32,
                height: -(swapchain.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        renderer.device.cmd_set_scissor(
            **command_buffer,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.width,
                    height: swapchain.height,
                },
            }],
        );
        renderer.device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *depth_pass.depth_pipeline,
        );
        depth_pass.depth_pipeline_layout.bind_descriptor_sets(
            &renderer.device,
            **command_buffer,
            &model_data.model_set.current(image_index.0),
            &camera_matrices.set.current(image_index.0),
        );
        renderer.device.cmd_bind_index_buffer(
            **command_buffer,
            cull_pass_data.culled_index_buffer.current(image_index.0).buffer.handle,
            0,
            vk::IndexType::UINT32,
        );
        renderer.device.cmd_bind_vertex_buffers(
            **command_buffer,
            0,
            &[consolidated_mesh_buffers.position_buffer.buffer.handle],
            &[0],
        );
        renderer.device.cmd_draw_indexed_indirect_count(
            **command_buffer,
            cull_pass_data
                .culled_commands_buffer
                .current(image_index.0)
                .buffer
                .handle,
            0,
            cull_pass_data
                .culled_commands_count_buffer
                .current(image_index.0)
                .buffer
                .handle,
            0,
            shaders::cull_set::bindings::indirect_commands::SIZE as u32
                / size_of::<shaders::VkDrawIndexedIndirectCommand>() as u32,
            size_of::<shaders::VkDrawIndexedIndirectCommand>() as u32,
        );
    }
}

fn gui_writeback<T: Component + Clone>(mut to: ResMut<T>, from: Res<GuiCopy<T>>) {
    let scope_name = format!("gui_writeback<{}>", std::any::type_name::<T>());
    scope!("ecs", scope_name);

    to.clone_from(&from.0);
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub(crate) enum GraphicsPhases {
    ShadowMapping,
    MainPass,
    SubmitMainPass,
    Present,
    CullPass,
    CullPassBypass,
    GuiWritebackConfig,
    GuiWritebackCamera,
    GuiWritebackSwapchainMap,
}

pub(crate) fn graphics_stage() -> SystemStage {
    use GraphicsPhases::*;
    let stage = SystemStage::parallel();

    #[cfg(not(feature = "no_profiling"))]
    let stage = {
        let token = microprofile::get_token("ecs".to_string(), "graphics_stage".to_string(), 0);

        stage
            .with_system(
                (move || {
                    microprofile::enter(token);
                })
                .exclusive_system()
                .at_start(),
            )
            .with_system(
                (|| {
                    microprofile::leave();
                })
                .exclusive_system()
                .at_end(),
            )
    };

    stage
        .with_system(update_base_color_descriptors.system().before(MainPass))
        // should not need to be before SubmitMainPass or even Present, but crashes on amdvlk eventually
        .with_system(cull_pass_bypass.system().label(CullPassBypass).before(SubmitMainPass))
        // should not need to be before SubmitMainPass or even Present, but crashes on amdvlk eventually
        .with_system(cull_pass.system().label(CullPass).before(SubmitMainPass))
        .with_system(prepare_shadow_maps.system().label(ShadowMapping))
        .with_system(render_frame.system().label(MainPass))
        .with_system(
            submit_main_pass
                .system()
                .label(SubmitMainPass)
                .after(ShadowMapping)
                .after(MainPass),
        )
        .with_system(PresentFramebuffer::exec.system().label(Present).after(SubmitMainPass))
        .with_system(
            gui_writeback::<RuntimeConfiguration>
                .system()
                .label(GuiWritebackConfig)
                .after(MainPass)
                .after(CullPass)
                .after(CullPassBypass),
        )
        .with_system(
            gui_writeback::<SwapchainIndexToFrameNumber>
                .system()
                .label(GuiWritebackSwapchainMap)
                .after(Present)
                .after(CullPass)
                .after(CullPassBypass),
        )
        .with_system(
            gui_writeback::<Camera>
                .system()
                .label(GuiWritebackCamera)
                .after(MainPass)
                .after(CullPass),
        )
}
