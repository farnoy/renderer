#![allow(clippy::too_many_arguments)]

// TODO: pub(crate) should disappear?
pub(crate) mod device;
mod entry;
mod gltf_mesh;
mod helpers;
mod instance;
pub(crate) mod shaders;
mod swapchain;
mod systems;

use std::{cmp::max, mem::size_of, os::raw::c_uchar, sync::Arc};

use ash::{
    version::{DeviceV1_0, DeviceV1_2},
    vk,
};
use bevy_ecs::{component::Component, prelude::*};
use microprofile::scope;
use num_traits::ToPrimitive;
use static_assertions::const_assert_eq;

use self::device::{
    Buffer, DescriptorPool, Device, DoubleBuffered, Framebuffer, Image, ImageView, RenderPass, Sampler, Shader,
    StaticBuffer, StrictCommandPool, StrictRecordingCommandBuffer, TimelineSemaphore, VmaMemoryUsage,
};
#[cfg(not(feature = "no_profiling"))]
pub(crate) use self::helpers::MP_INDIAN_RED;
#[cfg(feature = "crash_debugging")]
pub(crate) use self::systems::crash_debugging::CrashBuffer;
#[cfg(feature = "shader_reload")]
pub(crate) use self::systems::shader_reload::{reload_shaders, ReloadedShaders, ShaderReload};
pub(crate) use self::{
    gltf_mesh::{load as load_gltf, LoadedMesh},
    helpers::pick_lod,
    instance::Instance,
    swapchain::{Surface, Swapchain},
    systems::{
        consolidate_mesh_buffers::{consolidate_mesh_buffers, ConsolidateTimeline, ConsolidatedMeshBuffers},
        cull_pipeline::{
            coarse_culling, cull_pass, cull_pass_bypass, CoarseCulled, CullPassData, CullPassDataPrivate,
            INITIAL_WORKGROUP_SIZE,
        },
        debug_aabb_renderer::DebugAABBPassData,
        present::{acquire_framebuffer, ImageIndex, PresentData, PresentFramebuffer},
        shadow_mapping::{
            prepare_shadow_maps, shadow_mapping_mvp_calculation, update_shadow_map_descriptors, ShadowMappingData,
            ShadowMappingDataInternal, ShadowMappingLightMatrices,
        },
        textures::{
            cleanup_base_color_markers, synchronize_base_color_textures_visit, update_base_color_descriptors,
            BaseColorDescriptorSet, BaseColorVisitedMarker, GltfMeshBaseColorTexture,
        },
    },
};
use crate::ecs::{
    components::{ModelMatrix, AABB},
    resources::Camera,
    systems::*,
};

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
            PresentSurface,
            Depth,
            ShadowMapAtlas
        }
        formats {
            dyn,
            dyn,
            D16_UNORM,
            D16_UNORM
        }
        samples {
            4,
            1,
            4,
            1
        }
        passes {
            ShadowMapping {
                attachments [ShadowMapAtlas]
                layouts {
                    ShadowMapAtlas load DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
                                => store DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
                }
                subpasses {
                    ShadowMappingMain {
                        depth_stencil { ShadowMapAtlas => DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
                    }
                }
            },
            Main {
                attachments [Color, Depth, PresentSurface]
                layouts {
                    Depth clear UNDEFINED => discard DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    Color clear UNDEFINED => discard COLOR_ATTACHMENT_OPTIMAL,
                    PresentSurface clear UNDEFINED => store PRESENT_SRC_KHR
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
                        resolve [PresentSurface => COLOR_ATTACHMENT_OPTIMAL]
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
            // These passes have resources that are not double buffered, so they can't start processing until the
            // previous frame frame finished using them. This should perhaps become implicit, there's no use case
            // so far for overlapping computation across frame boundaries
            PresentationAcquire => ShadowMapping,
            PresentationAcquire => TransferCull,
            PresentationAcquire => ComputeCull,
            PresentationAcquire => ConsolidateMeshBuffers,

            TransferCull => Main,
            ComputeCull => Main,
            ConsolidateMeshBuffers => Main,
            ShadowMapping => Main,
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
            renderpass: frame_graph::Main::RenderPass::new(
                renderer,
                (attachments.swapchain_format, attachments.swapchain_format),
            ),
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

        let buffer_count = swapchain.desired_image_count.to_usize().unwrap();

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
        let descriptor_pool = renderer.device.new_descriptor_pool(3_000, &[
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 16384,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 4096,
            },
        ]);
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
    swapchain_image_views: Vec<ImageView>,
    swapchain_format: vk::Format,
    #[allow(unused)]
    depth_images: Vec<Image>,
    depth_image_views: Vec<ImageView>,
    #[allow(unused)]
    color_images: Vec<Image>,
    color_image_views: Vec<ImageView>,
}

impl MainAttachments {
    pub(crate) fn new(renderer: &RenderFrame, swapchain: &Swapchain) -> MainAttachments {
        let images = unsafe { swapchain.ext.get_swapchain_images(swapchain.swapchain).unwrap() };
        assert!(images.len().to_u32().unwrap() >= swapchain.desired_image_count);
        println!("swapchain images len {}", images.len());
        let depth_images = (0..swapchain.desired_image_count)
            .map(|ix| {
                let im = renderer.device.new_image(
                    vk::Format::D16_UNORM,
                    vk::Extent3D {
                        width: swapchain.width,
                        height: swapchain.height,
                        depth: 1,
                    },
                    vk::SampleCountFlags::TYPE_4,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                );
                renderer.device.set_object_name(im.handle, &format!("Depth RT[{}]", ix));
                im
            })
            .collect::<Vec<_>>();
        let color_images = (0..swapchain.desired_image_count)
            .map(|ix| {
                let im = renderer.device.new_image(
                    swapchain.surface.surface_format.format,
                    vk::Extent3D {
                        width: swapchain.width,
                        height: swapchain.height,
                        depth: 1,
                    },
                    vk::SampleCountFlags::TYPE_4,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                );
                renderer.device.set_object_name(im.handle, &format!("Color RT[{}]", ix));
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
        let color_image_views = color_images
            .iter()
            .map(|ref image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain.surface.surface_format.format)
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
                    .image(image.handle);
                renderer.device.new_image_view(&create_view_info)
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
                renderer.device.new_image_view(&create_view_info)
            })
            .collect::<Vec<_>>();

        MainAttachments {
            swapchain_image_views: image_views,
            swapchain_format: swapchain.surface.surface_format.format,
            depth_images,
            depth_image_views,
            color_images,
            color_image_views,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        for view in self.swapchain_image_views.into_iter() {
            view.destroy(device);
        }
        for view in self.depth_image_views.into_iter() {
            view.destroy(device);
        }
        for view in self.color_image_views.into_iter() {
            view.destroy(device);
        }
        for depth in self.depth_images.into_iter() {
            depth.destroy(device);
        }
        for color in self.color_images.into_iter() {
            color.destroy(device);
        }
    }
}

pub(crate) struct MainFramebuffer {
    pub(crate) framebuffer: frame_graph::Main::Framebuffer,
}

impl FromWorld for MainFramebuffer {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let main_renderpass = world.get_resource::<MainRenderpass>().unwrap();
        let swapchain = world.get_resource::<Swapchain>().unwrap();
        Self::new(&renderer, &main_renderpass, &swapchain)
    }
}

impl MainFramebuffer {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_renderpass: &MainRenderpass,
        swapchain: &Swapchain,
    ) -> MainFramebuffer {
        let framebuffer = frame_graph::Main::Framebuffer::new(
            renderer,
            &main_renderpass.renderpass,
            &[
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ],
            (
                swapchain.surface.surface_format.format,
                swapchain.surface.surface_format.format,
            ),
            (swapchain.width, swapchain.height),
        );

        MainFramebuffer { framebuffer }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.framebuffer.destroy(device);
    }
}

pub(crate) struct LocalTransferCommandPool<const NAME: usize> {
    pub(crate) pools: DoubleBuffered<StrictCommandPool>,
}

impl<const NAME: usize> LocalTransferCommandPool<NAME> {
    pub(crate) fn new(renderer: &RenderFrame) -> Self {
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

impl<const NAME: usize> LocalTransferCommandPool<NAME> {
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
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
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
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
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
    pub(crate) depth_pipeline: shaders::depth_pipe::Pipeline,
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
        let depth_pipeline = shaders::depth_pipe::Pipeline::new(
            &device,
            &depth_pipeline_layout,
            shaders::depth_pipe::Specialization {},
            [None],
            &main_renderpass.renderpass.renderpass,
            0,
            vk::SampleCountFlags::TYPE_4,
        );

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
    pub(crate) gltf_pipeline: shaders::gltf_mesh::Pipeline,
    #[cfg(feature = "shader_reload")]
    pub(crate) previous_gltf_pipeline: DoubleBuffered<Option<shaders::gltf_mesh::Pipeline>>,
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
        let gltf_pipeline = shaders::gltf_mesh::Pipeline::new(
            &renderer.device,
            &gltf_pipeline_layout,
            spec,
            [None, None],
            &main_renderpass.renderpass.renderpass,
            1, // FIXME
        );

        GltfPassData {
            gltf_pipeline,
            #[cfg(feature = "shader_reload")]
            previous_gltf_pipeline: renderer.new_buffered(|_| None),
            gltf_pipeline_layout,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.gltf_pipeline.destroy(device);
        #[cfg(feature = "shader_reload")]
        self.previous_gltf_pipeline
            .into_iter()
            .for_each(|p| p.into_iter().for_each(|p| p.destroy(device)));
        self.gltf_pipeline_layout.destroy(device);
    }
}

pub(crate) struct MainPassCommandBuffer {
    command_pools: DoubleBuffered<StrictCommandPool>,
    command_buffers: DoubleBuffered<vk::CommandBuffer>,
}

impl FromWorld for MainPassCommandBuffer {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let mut command_pools = renderer.new_buffered(|ix| {
            StrictCommandPool::new(
                &renderer.device,
                renderer.device.graphics_queue_family,
                &format!("Main Pass Command Pool[{}]", ix),
            )
        });
        let command_buffers = renderer.new_buffered(|ix| {
            command_pools
                .current_mut(ix)
                .allocate(&format!("Main Pass CB[{}]", ix), &renderer.device)
        });
        MainPassCommandBuffer {
            command_pools,
            command_buffers,
        }
    }
}

impl MainPassCommandBuffer {
    pub(crate) fn destroy(self, device: &Device) {
        self.command_pools.into_iter().for_each(|p| p.destroy(device));
    }
}

pub(crate) fn render_frame(
    renderer: Res<RenderFrame>,
    (main_renderpass, main_attachments): (Res<MainRenderpass>, Res<MainAttachments>),
    (image_index, model_data): (Res<ImageIndex>, Res<ModelData>),
    mut runtime_config: ResMut<RuntimeConfiguration>,
    camera_matrices: Res<CameraMatrices>,
    swapchain: Res<Swapchain>,
    consolidated_mesh_buffers: Res<ConsolidatedMeshBuffers>,
    (debug_aabb_pass_data, shadow_mapping_data): (Res<DebugAABBPassData>, Res<ShadowMappingData>),
    base_color_descriptor_set: Res<BaseColorDescriptorSet>,
    cull_pass_data: Res<CullPassData>,
    main_framebuffer: Res<MainFramebuffer>,
    (mut main_pass_cb, depth_pass_data, mut gltf_pass, mut gui_render_data, mut camera, mut input_handler, mut gui): (
        ResMut<MainPassCommandBuffer>,
        Res<DepthPassData>,
        ResMut<GltfPassData>,
        ResMut<GuiRenderData>,
        ResMut<Camera>,
        NonSendMut<InputHandler>,
        NonSendMut<Gui>,
    ),
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
    #[cfg(feature = "shader_reload")] reloaded_shaders: Res<ReloadedShaders>,
    query: Query<&AABB>,
) {
    scope!("ecs", "render_frame");

    let GltfPassData {
        ref mut gltf_pipeline,
        ref gltf_pipeline_layout,
        #[cfg(feature = "shader_reload")]
        ref mut previous_gltf_pipeline,
    } = &mut *gltf_pass;

    // TODO: count this? pack and defragment draw calls?
    let total = shaders::cull_set::bindings::indirect_commands::SIZE as u32
        / size_of::<shaders::VkDrawIndexedIndirectCommand>() as u32;

    #[cfg(feature = "shader_reload")]
    {
        // clean up the old pipeline that was used N frames ago
        if let Some(previous) = previous_gltf_pipeline.current_mut(image_index.0).take() {
            previous.destroy(&renderer.device);
        }

        use systems::shadow_mapping::DIM as SHADOW_MAP_DIM;
        *previous_gltf_pipeline.current_mut(image_index.0) = gltf_pipeline.specialize(
            &renderer.device,
            &gltf_pipeline_layout,
            &shaders::gltf_mesh::Specialization {
                shadow_map_dim: SHADOW_MAP_DIM,
                shadow_map_dim_squared: SHADOW_MAP_DIM * SHADOW_MAP_DIM,
            },
            [None, None],
            &main_renderpass.renderpass.renderpass,
            1, // FIXME
            #[cfg(feature = "shader_reload")]
            &reloaded_shaders,
        );
    }

    let MainPassCommandBuffer {
        ref mut command_pools,
        ref command_buffers,
    } = *main_pass_cb;

    let command_pool = command_pools.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let mut command_session = command_pool.session(&renderer.device);

    let command_buffer = command_session.record_to_specific(*command_buffers.current(image_index.0));
    unsafe {
        let _main_renderpass_marker = command_buffer.debug_marker_around("main renderpass", [0.0, 0.0, 1.0, 1.0]);
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
        main_renderpass.renderpass.begin(
            &renderer,
            &main_framebuffer.framebuffer,
            *command_buffer,
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.width,
                    height: swapchain.height,
                },
            },
            &[
                main_attachments.color_image_views[image_index.0 as usize].handle,
                main_attachments.depth_image_views[image_index.0 as usize].handle,
                main_attachments.swapchain_image_views[image_index.0 as usize].handle,
            ],
            &[
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
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
            &command_buffer,
        );
        renderer
            .device
            .cmd_next_subpass(*command_buffer, vk::SubpassContents::INLINE);
        if runtime_config.debug_aabbs {
            scope!("ecs", "debug aabb pass");

            let _aabb_marker = command_buffer.debug_marker_around("aabb debug", [1.0, 0.0, 0.0, 1.0]);
            renderer.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *debug_aabb_pass_data.pipeline.pipeline,
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
            // gltf mesh
            renderer.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *gltf_pipeline.pipeline,
            );
            gltf_pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                &model_data.model_set.current(image_index.0),
                &camera_matrices.set.current(image_index.0),
                &shadow_mapping_data.user_set.current(image_index.0),
                &base_color_descriptor_set.set,
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
                &[
                    consolidated_mesh_buffers.position_buffer.buffer.handle,
                    consolidated_mesh_buffers.normal_buffer.buffer.handle,
                    consolidated_mesh_buffers.uv_buffer.handle,
                ],
                &[0, 0, 0],
            );
            renderer.device.cmd_draw_indexed_indirect_count(
                *command_buffer,
                cull_pass_data.culled_commands_buffer.buffer.handle,
                0,
                cull_pass_data.culled_commands_count_buffer.buffer.handle,
                0,
                total,
                size_of::<shaders::VkDrawIndexedIndirectCommand>() as u32,
            );
            #[cfg(feature = "crash_debugging")]
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
            &mut runtime_config,
            &swapchain,
            &mut camera,
            &mut input_handler,
            &mut gui,
            &command_buffer,
            #[cfg(feature = "shader_reload")]
            &reloaded_shaders,
        );
        renderer.device.cmd_end_render_pass(*command_buffer);
    }

    let command_buffer = *command_buffer.end();

    *main_pass_cb.command_buffers.current_mut(image_index.0) = command_buffer;
}

fn submit_main_pass(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    main_pass_cb: Res<MainPassCommandBuffer>,
) {
    scope!("ecs", "submit_main_pass");
    let command_buffer = *main_pass_cb.command_buffers.current(image_index.0);
    debug_assert_ne!(command_buffer, vk::CommandBuffer::null());

    let queue = renderer.device.graphics_queue().lock();

    frame_graph::Main::Stage::queue_submit(&image_index, &renderer, *queue, &[command_buffer]).unwrap();
}

pub(crate) struct GuiRenderData {
    pos_buffer: StaticBuffer<[glm::Vec2; 1024 * 1024]>,
    uv_buffer: StaticBuffer<[glm::Vec2; 1024 * 1024]>,
    col_buffer: StaticBuffer<[glm::Vec4; 1024 * 1024]>,
    index_buffer: StaticBuffer<[imgui::DrawIdx; 1024 * 1024]>,
    texture: Image,
    #[allow(unused)]
    texture_view: ImageView,
    #[allow(unused)]
    sampler: Sampler,
    #[allow(unused)]
    descriptor_set_layout: shaders::imgui_set::Layout,
    descriptor_set: shaders::imgui_set::Set,
    pipeline_layout: shaders::imgui_pipe::PipelineLayout,
    pipeline: shaders::imgui_pipe::Pipeline,
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
        let pos_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        );
        let uv_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        );
        let col_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        );
        renderer
            .device
            .set_object_name(pos_buffer.buffer.handle, "GUI Vertex Buffer - pos");
        renderer
            .device
            .set_object_name(uv_buffer.buffer.handle, "GUI Vertex Buffer - uv");
        renderer
            .device
            .set_object_name(col_buffer.buffer.handle, "GUI Vertex Buffer - col");
        let index_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::INDEX_BUFFER,
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        );
        renderer
            .device
            .set_object_name(index_buffer.buffer.handle, "GUI Index Buffer");
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
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );
            {
                let mut texture_data = texture
                    .map::<c_uchar>(&renderer.device)
                    .expect("failed to map imgui texture");
                texture_data[0..imgui_texture.data.len()].copy_from_slice(imgui_texture.data);
            }
            texture
        };
        let sampler = renderer.device.new_sampler(
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

        let pipeline = shaders::imgui_pipe::Pipeline::new(
            &renderer.device,
            &pipeline_layout,
            shaders::imgui_pipe::Specialization {},
            [None, None],
            &main_renderpass.renderpass.renderpass,
            2,
        );

        let texture_view = renderer.device.new_image_view(
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

        let mut command_pool = StrictCommandPool::new(
            &renderer.device,
            renderer.device.graphics_queue_family,
            &"GuiRender Initialization Command Pool",
        );

        let mut session = command_pool.session(&renderer.device);
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

        command_pool.destroy(&renderer.device);

        GuiRenderData {
            pos_buffer,
            uv_buffer,
            col_buffer,
            index_buffer,
            texture,
            texture_view,
            sampler,
            descriptor_set_layout,
            descriptor_set,
            pipeline_layout,
            pipeline,
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
        self.pos_buffer.destroy(device);
        self.uv_buffer.destroy(device);
        self.col_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}

/// This wrapper is used to break dependency chains and let otherwise conflicting systems operate on
/// the same resources as long as the writer can afford a delay in making the mutation available to
/// other users. See [copy_resource] and [writeback_resource] for the complete toolset.
/// To avoid exclusive access to the [World], this wrapper must be initialized ahead of time.
///
/// It can be used the opposite way, with the readers being redirected to the wrapper (with stale
/// data) while the writes are made available immedietaly. There is no need to write back any data
/// in this mode.
pub(crate) struct CopiedResource<T>(pub(crate) T);

impl<T> std::ops::Deref for CopiedResource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: FromWorld> FromWorld for CopiedResource<T> {
    fn from_world(world: &mut World) -> Self {
        CopiedResource(<T as FromWorld>::from_world(world))
    }
}

/// This system can be used to copy data from a resource and place it in the [CopiedResource]
/// wrapper. This requires the wrapper to already be present on the [World].
/// Remember to use [writeback_resource] if the mutations are made to the wrapper.
pub(crate) fn copy_resource<T: Component + Clone>(from: Res<T>, mut to: ResMut<CopiedResource<T>>) {
    #[allow(unused_variables)]
    let scope_name = format!("copy_resource<{}>", std::any::type_name::<T>());
    scope!("ecs", scope_name);

    to.0.clone_from(&from);
}

/// This system writes back the changes from the coppied wrapper to the original resource.
/// The mode of operation where writes are redirected to [CopiedResource] are probably less
/// efficient as they require a writeback
#[allow(dead_code)]
pub(crate) fn writeback_resource<T: Component + Clone>(from: Res<CopiedResource<T>>, mut to: ResMut<T>) {
    #[allow(unused_variables)]
    let scope_name = format!("writeback_resource<{}>", std::any::type_name::<T>());
    scope!("ecs", scope_name);

    to.clone_from(&from.0);
}

fn render_gui(
    renderer: &RenderFrame,
    gui_render_data: &mut GuiRenderData,
    runtime_config: &mut RuntimeConfiguration,
    swapchain: &Swapchain,
    camera: &mut Camera,
    input_handler: &mut InputHandler,
    gui: &mut Gui,
    command_buffer: &StrictRecordingCommandBuffer,
    #[cfg(feature = "shader_reload")] reloaded_shaders: &ReloadedShaders,
) {
    scope!("rendering", "render_gui");

    let GuiRenderData {
        ref pos_buffer,
        ref uv_buffer,
        ref col_buffer,
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
        runtime_config,
        #[cfg(feature = "shader_reload")]
        reloaded_shaders,
    );

    let _gui_debug_marker = command_buffer.debug_marker_around("GUI", [1.0, 1.0, 0.0, 1.0]);
    unsafe {
        pipeline_layout.bind_descriptor_sets(&renderer.device, **command_buffer, &descriptor_set);
        renderer
            .device
            .cmd_bind_pipeline(**command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline.pipeline);
        renderer.device.cmd_bind_vertex_buffers(
            **command_buffer,
            0,
            &[
                pos_buffer.buffer.handle,
                uv_buffer.buffer.handle,
                col_buffer.buffer.handle,
            ],
            &[0, 0, 0],
        );
        renderer
            .device
            .cmd_bind_index_buffer(**command_buffer, index_buffer.buffer.handle, 0, vk::IndexType::UINT16);
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
            let mut pos_slice = pos_buffer
                .map(&renderer.device)
                .expect("Failed to map gui vertex buffer - pos");
            let mut uv_slice = uv_buffer
                .map(&renderer.device)
                .expect("Failed to map gui vertex buffer - uv");
            let mut col_slice = col_buffer
                .map(&renderer.device)
                .expect("Failed to map gui vertex buffer - col");
            let mut index_slice = index_buffer
                .map(&renderer.device)
                .expect("Failed to map gui index buffer");
            for draw_list in gui_draw_data.draw_lists() {
                scope!("rendering", "gui draw list");

                let index_len = draw_list.idx_buffer().len();
                index_slice[index_offset_coarse..index_offset_coarse + index_len]
                    .copy_from_slice(draw_list.idx_buffer());
                let vertex_len = draw_list.vtx_buffer().len();
                for (ix, vertex) in draw_list.vtx_buffer().iter().enumerate() {
                    pos_slice[ix] = glm::Vec2::from(vertex.pos);
                    uv_slice[ix] = glm::Vec2::from(vertex.uv);
                    // TODO: this conversion sucks, need to add customizable vertex attribute formats
                    let byte_color = na::SVector::<u8, 4>::from(vertex.col);
                    col_slice[ix] = byte_color.map(|byte| byte as f32 / 255.0);
                }
                for draw_cmd in draw_list.commands() {
                    match draw_cmd {
                        imgui::DrawCmd::Elements { count, cmd_params } => {
                            renderer.device.cmd_set_scissor(**command_buffer, 0, &[vk::Rect2D {
                                offset: vk::Offset2D {
                                    x: cmd_params.clip_rect[0] as i32,
                                    y: cmd_params.clip_rect[1] as i32,
                                },
                                extent: vk::Extent2D {
                                    width: (cmd_params.clip_rect[2] - cmd_params.clip_rect[0]) as u32,
                                    height: (cmd_params.clip_rect[3] - cmd_params.clip_rect[1]) as u32,
                                },
                            }]);
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

    let mut max_accessed = 0;
    query.for_each(|(draw_index, model_matrix)| {
        model_mapped.model[draw_index.0 as usize] = model_matrix.0;
        max_accessed = max(max_accessed, draw_index.0);
    });

    // sanity check for the following flush calculation
    const_assert_eq!(size_of::<glm::Mat4>(), size_of::<shaders::ModelMatrices>() / 4096,);
    model_mapped.unmap_used_range(0..(max_accessed as vk::DeviceSize * size_of::<glm::Mat4>() as vk::DeviceSize));
}

pub(crate) fn camera_matrices_upload(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    camera: Res<Camera>,
    mut camera_matrices: ResMut<CameraMatrices>,
) {
    scope!("ecs", "CameraMatricesUpload");
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
    command_buffer: &StrictRecordingCommandBuffer, // vk::CommandBuffer
) {
    scope!("rendering", "depth_only_pass");

    unsafe {
        scope!("depth only", "render commands");
        let _marker = command_buffer.debug_marker_around("depth prepass", [0.3, 0.3, 0.3, 1.0]);

        if runtime_config.debug_aabbs {
            return;
        }
        renderer.device.cmd_bind_pipeline(
            **command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *depth_pass.depth_pipeline.pipeline,
        );
        depth_pass.depth_pipeline_layout.bind_descriptor_sets(
            &renderer.device,
            **command_buffer,
            &model_data.model_set.current(image_index.0),
            &camera_matrices.set.current(image_index.0),
        );
        renderer.device.cmd_bind_index_buffer(
            **command_buffer,
            cull_pass_data.culled_index_buffer.buffer.handle,
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
            cull_pass_data.culled_commands_buffer.buffer.handle,
            0,
            cull_pass_data.culled_commands_count_buffer.buffer.handle,
            0,
            shaders::cull_set::bindings::indirect_commands::SIZE as u32
                / size_of::<shaders::VkDrawIndexedIndirectCommand>() as u32,
            size_of::<shaders::VkDrawIndexedIndirectCommand>() as u32,
        );
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub(crate) enum GraphicsPhases {
    ShadowMapping,
    MainPass,
    SubmitMainPass,
    Present,
    CullPass,
    CullPassBypass,
    CopyResource(u8),
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
        // uses update_after_bind descriptors so it needs to finish before submitting
        .with_system(update_base_color_descriptors.system().before(SubmitMainPass))
        // should not need to be before SubmitMainPass or even Present, but crashes on amdvlk eventually
        .with_system(cull_pass_bypass.system().label(CullPassBypass).before(SubmitMainPass))
        // should not need to be before SubmitMainPass or even Present, but crashes on amdvlk eventually
        .with_system(cull_pass.system().label(CullPass).before(SubmitMainPass))
        .with_system(prepare_shadow_maps.system().label(ShadowMapping))
        .with_system(
            copy_resource::<RuntimeConfiguration>
                .system()
                .label(CopyResource(1))
                .before(MainPass)
                .before(CullPass)
                .before(CullPassBypass),
        )
        .with_system(
            copy_resource::<Camera>
                .system()
                .label(CopyResource(2))
                .before(MainPass)
                .before(CullPass),
        )
        .with_system(
            copy_resource::<SwapchainIndexToFrameNumber>
                .system()
                .label(CopyResource(2))
                .before(Present)
                .before(CullPass)
                .before(CullPassBypass),
        )
        .with_system(render_frame.system().label(MainPass))
        .with_system(
            submit_main_pass
                .system()
                .label(SubmitMainPass)
                .after(ShadowMapping)
                .after(MainPass),
        )
        .with_system(PresentFramebuffer::exec.system().label(Present).after(SubmitMainPass))
}
