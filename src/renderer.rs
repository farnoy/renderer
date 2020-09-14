// TODO: pub(crate) should disappear?
mod alloc;
mod device;
mod entry;
mod gltf_mesh;
mod helpers;
mod instance;
pub(crate) mod shaders;
mod swapchain;
mod systems {
    pub(crate) mod consolidate_mesh_buffers;
    pub(crate) mod crash_debugging;
    pub(crate) mod cull_pipeline;
    pub(crate) mod debug_aabb_renderer;
    pub(crate) mod present;
    pub(crate) mod shadow_mapping;
    pub(crate) mod textures;
}

pub(crate) use self::{
    device::*,
    gltf_mesh::{load as load_gltf, LoadedMesh},
    helpers::*,
    instance::Instance,
    swapchain::*,
    systems::{
        consolidate_mesh_buffers::*, crash_debugging::*, cull_pipeline::*, debug_aabb_renderer::*,
        present::*, shadow_mapping::*, textures::*,
    },
};
use crate::{
    define_timeline,
    ecs::{
        components::{ModelMatrix, AABB},
        resources::Camera,
        systems::*,
    },
};
use ash::{version::DeviceV1_0, vk};
use bevy_ecs::prelude::*;
#[cfg(feature = "microprofile")]
use microprofile::scope;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    cell::RefCell, convert::TryInto, mem::size_of, os::raw::c_uchar, path::PathBuf, rc::Rc,
    sync::Arc,
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

#[derive(Debug, Default)]
pub(crate) struct DrawIndex(pub(crate) u32);

pub(crate) struct Position(pub(crate) na::Point3<f32>);
pub(crate) struct Rotation(pub(crate) na::UnitQuaternion<f32>);
pub(crate) struct Scale(pub(crate) f32);

// TODO: rename
pub(crate) struct RenderFrame {
    pub(crate) instance: Arc<Instance>,
    pub(crate) device: Arc<Device>,
    pub(crate) renderpass: RenderPass,
    pub(crate) graphics_timeline_semaphore: TimelineSemaphore,
    pub(crate) compute_timeline_semaphore: TimelineSemaphore,
    pub(crate) frame_number: u64,
    pub(crate) previous_frame_number_for_swapchain_index: SmallVec<[u64; 3]>,
    pub(crate) buffer_count: usize,
}

// TODO: make a separate module for these high level definitions?
define_timeline!(graphics Start, ShadowMapping, DepthPass, SceneDraw, GuiDraw);

trait ImageLayout {
    const VK_FLAG: vk::ImageLayout;
}

#[derive(Debug)]
struct Undefined;
#[derive(Debug)]
struct ColorAttachmentOptimal;
#[derive(Debug)]
struct DepthAttachmentOptimal;
#[derive(Debug)]
struct DepthReadOnlyOptimal;
#[derive(Debug)]
struct PresentSrc;

impl ImageLayout for Undefined {
    const VK_FLAG: vk::ImageLayout = vk::ImageLayout::UNDEFINED;
}

impl ImageLayout for ColorAttachmentOptimal {
    const VK_FLAG: vk::ImageLayout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
}

impl ImageLayout for DepthAttachmentOptimal {
    const VK_FLAG: vk::ImageLayout = vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;
}

impl ImageLayout for DepthReadOnlyOptimal {
    const VK_FLAG: vk::ImageLayout = vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL;
}

impl ImageLayout for PresentSrc {
    const VK_FLAG: vk::ImageLayout = vk::ImageLayout::PRESENT_SRC_KHR;
}

trait TransitionsRenderTarget<const I: u8> {
    type Layout: ImageLayout;
}

impl TransitionsRenderTarget<0> for graphics::Start {
    type Layout = Undefined;
}

impl TransitionsRenderTarget<1> for graphics::Start {
    type Layout = Undefined;
}

impl TransitionsRenderTarget<0> for graphics::DepthPass {
    type Layout = ColorAttachmentOptimal;
}

impl TransitionsRenderTarget<1> for graphics::DepthPass {
    type Layout = DepthAttachmentOptimal;
}

impl TransitionsRenderTarget<0> for graphics::SceneDraw {
    type Layout = ColorAttachmentOptimal;
}

impl TransitionsRenderTarget<1> for graphics::SceneDraw {
    type Layout = DepthReadOnlyOptimal;
}

impl TransitionsRenderTarget<0> for graphics::GuiDraw {
    type Layout = PresentSrc;
}

impl TransitionsRenderTarget<1> for graphics::GuiDraw {
    type Layout = DepthReadOnlyOptimal;
}

define_timeline!(compute Perform);

impl RenderFrame {
    pub(crate) fn new() -> (RenderFrame, Swapchain, winit::event_loop::EventLoop<()>) {
        let (instance, events_loop) = Instance::new().expect("Failed to create instance");
        let instance = Arc::new(instance);
        let surface = Surface::new(&instance);
        let device = Arc::new(Device::new(&instance, &surface).expect("Failed to create device"));
        device.set_object_name(device.handle(), "Device");
        let swapchain = Swapchain::new(&instance, &device, surface);
        let main_renderpass = {
            let color_attachment = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };
            let depth_attachment = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            };

            device.new_renderpass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(unsafe {
                        &*(&[
                            vk::AttachmentDescription::builder()
                                .format(swapchain.surface.surface_format.format)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(<<graphics::SceneDraw as SuccessorStage<_>>::Previous as TransitionsRenderTarget<0>>::Layout::VK_FLAG)
                                .final_layout(<graphics::SceneDraw as TransitionsRenderTarget<0>>::Layout::VK_FLAG),
                            vk::AttachmentDescription::builder()
                                .format(vk::Format::D16_UNORM)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(<<graphics::SceneDraw as SuccessorStage<_>>::Previous as TransitionsRenderTarget<1>>::Layout::VK_FLAG)
                                .final_layout(<graphics::SceneDraw as TransitionsRenderTarget<1>>::Layout::VK_FLAG),
                        ]
                            as *const [vk::AttachmentDescriptionBuilder<'_>; 2]
                            as *const [vk::AttachmentDescription; 2])
                    })
                    .subpasses(unsafe {
                        &*(&[vk::SubpassDescription::builder()
                            .color_attachments(&[color_attachment])
                            .depth_stencil_attachment(&depth_attachment)
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)]
                            as *const [vk::SubpassDescriptionBuilder<'_>; 1]
                            as *const [vk::SubpassDescription; 1])
                    })
                    .dependencies(unsafe {
                        &*(&[vk::SubpassDependency::builder()
                            .src_subpass(vk::SUBPASS_EXTERNAL)
                            .dst_subpass(0)
                            .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)]
                            as *const [vk::SubpassDependencyBuilder<'_>; 1]
                            as *const [vk::SubpassDependency; 1])
                    }),
            )
        };
        device.set_object_name(main_renderpass.handle, "Main pass RenderPass");

        // Stat frame number at 1 and semaphores at 16, because validation layers assert
        // wait_semaphore_values at > 0
        let frame_number = 1;
        let graphics_timeline_semaphore = device
            .new_semaphore_timeline(
                timeline_value_last::<graphics::Timeline, graphics::GuiDraw>(frame_number),
            );
        device.set_object_name(
            graphics_timeline_semaphore.handle,
            "Graphics timeline semaphore",
        );
        let compute_timeline_semaphore = device
            .new_semaphore_timeline(timeline_value_last::<compute::Timeline, compute::Perform>(
                frame_number,
            ));
        device.set_object_name(
            compute_timeline_semaphore.handle,
            "Compute timeline semaphore",
        );

        let buffer_count = unsafe {
            swapchain
                .ext
                .get_swapchain_images(swapchain.swapchain)
                .unwrap()
                .len()
        };

        (
            RenderFrame {
                instance: Arc::clone(&instance),
                device: Arc::clone(&device),
                renderpass: main_renderpass,
                graphics_timeline_semaphore,
                compute_timeline_semaphore,
                frame_number,
                previous_frame_number_for_swapchain_index: SmallVec::from_elem(0, buffer_count),
                buffer_count,
            },
            swapchain,
            events_loop,
        )
    }

    pub(crate) fn new_buffered<T, F: FnMut(u32) -> T>(&self, creator: F) -> DoubleBuffered<T> {
        DoubleBuffered::new(self.buffer_count, creator)
    }
}

#[cfg(feature = "crash_debugging")]
impl Drop for RenderFrame {
    fn drop(&mut self) {
        dbg!(self.frame_number);
    }
}

pub(crate) struct MainDescriptorPool(pub(crate) Arc<DescriptorPool>);

impl MainDescriptorPool {
    pub(crate) fn new(renderer: &RenderFrame) -> MainDescriptorPool {
        let descriptor_pool = Arc::new(renderer.device.new_descriptor_pool(
            3_000,
            &[
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
            ],
        ));
        renderer
            .device
            .set_object_name(descriptor_pool.handle, "Main Descriptor Pool");

        MainDescriptorPool(descriptor_pool)
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
        let images = unsafe {
            swapchain
                .ext
                .get_swapchain_images(swapchain.swapchain)
                .unwrap()
        };
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
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
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
                let handle = unsafe {
                    renderer
                        .device
                        .create_image_view(&create_view_info, None)
                        .unwrap()
                };

                ImageView {
                    handle,
                    device: Arc::clone(&renderer.device),
                }
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
                let handle = unsafe {
                    renderer
                        .device
                        .create_image_view(&create_view_info, None)
                        .unwrap()
                };
                ImageView {
                    handle,
                    device: Arc::clone(&renderer.device),
                }
            })
            .collect::<Vec<_>>();

        MainAttachments {
            swapchain_images: images
                .iter()
                .cloned()
                .map(|handle| SwapchainImage { handle })
                .collect(),
            swapchain_image_views: image_views,
            depth_images,
            depth_image_views,
        }
    }
}

pub(crate) struct MainFramebuffer {
    pub(crate) handles: Vec<Framebuffer>,
}

impl MainFramebuffer {
    pub(crate) fn new(
        renderer: &RenderFrame,
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
                let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(renderer.renderpass.handle)
                    .attachments(&framebuffer_attachments)
                    .width(swapchain.width)
                    .height(swapchain.height)
                    .layers(1);
                let handle = unsafe {
                    renderer
                        .device
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap()
                };
                renderer
                    .device
                    .set_object_name(handle, &format!("Main Framebuffer - {}", ix));
                Framebuffer {
                    handle,
                    device: Arc::clone(&renderer.device),
                }
            })
            .collect::<Vec<_>>();

        MainFramebuffer { handles }
    }
}

pub(crate) struct LocalGraphicsCommandPool {
    pools: DoubleBuffered<StrictCommandPool>,
}

impl LocalGraphicsCommandPool {
    fn new(renderer: &RenderFrame) -> LocalGraphicsCommandPool {
        LocalGraphicsCommandPool {
            pools: renderer.new_buffered(|ix| {
                StrictCommandPool::new(
                    &renderer.device,
                    renderer.device.graphics_queue_family,
                    &format!("Render Command Pool[{}]", ix),
                )
            }),
        }
    }
}

impl FromResources for LocalGraphicsCommandPool {
    fn from_resources(resources: &Resources) -> Self {
        let renderer = resources.query::<Res<RenderFrame>>().unwrap();
        Self::new(&renderer)
    }
}

pub(crate) struct CameraMatrices {
    pub(crate) set_layout: shaders::camera_set::DescriptorSetLayout,
    buffer: DoubleBuffered<Buffer>,
    set: DoubleBuffered<shaders::camera_set::DescriptorSet>,
}

impl CameraMatrices {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &MainDescriptorPool,
    ) -> CameraMatrices {
        let buffer = renderer.new_buffered(|ix| {
            let b = renderer.device.new_buffer(
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                shaders::camera_set::bindings::matrices::SIZE,
            );
            renderer
                .device
                .set_object_name(b.handle, &format!("Camera matrices Buffer - ix={}", ix));
            b
        });
        let set_layout = shaders::camera_set::DescriptorSetLayout::new(&renderer.device);
        renderer
            .device
            .set_object_name(set_layout.layout.handle, "Camera matrices set layout");
        let set = renderer.new_buffered(|ix| {
            let s = shaders::camera_set::DescriptorSet::new(&main_descriptor_pool, &set_layout);
            renderer
                .device
                .set_object_name(s.set.handle, &format!("Camera matrices Set - ix={}", ix));

            s.update_whole_buffer(&renderer, 0, buffer.current(ix));

            s
        });

        CameraMatrices {
            set_layout,
            set,
            buffer,
        }
    }
}

pub(crate) struct ModelData {
    pub(crate) model_set_layout: shaders::model_set::DescriptorSetLayout,
    pub(crate) model_set: DoubleBuffered<shaders::model_set::DescriptorSet>,
    pub(crate) model_buffer: DoubleBuffered<Buffer>,
}

impl ModelData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &MainDescriptorPool,
    ) -> ModelData {
        let device = &renderer.device;

        let model_set_layout = shaders::model_set::DescriptorSetLayout::new(&device);
        device.set_object_name(model_set_layout.layout.handle, "Model Set Layout");

        let model_buffer = renderer.new_buffered(|ix| {
            let b = device.new_buffer(
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                shaders::model_set::bindings::model::SIZE,
            );
            device.set_object_name(b.handle, &format!("Model Buffer - {}", ix));
            b
        });
        let model_set = renderer.new_buffered(|ix| {
            let s =
                shaders::model_set::DescriptorSet::new(&main_descriptor_pool, &model_set_layout);
            device.set_object_name(s.set.handle, &format!("Model Set - {}", ix));
            s.update_whole_buffer(&renderer, 0, &model_buffer.current(ix));
            s
        });

        ModelData {
            model_set_layout,
            model_set,
            model_buffer,
        }
    }
}

pub(crate) struct DepthPassData {
    pub(crate) depth_pipeline: Pipeline,
    pub(crate) depth_pipeline_layout: shaders::depth_pipe::PipelineLayout,
    pub(crate) renderpass: RenderPass,
}

impl DepthPassData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        model_data: &ModelData,
        swapchain: &Swapchain,
        camera_matrices: &CameraMatrices,
    ) -> DepthPassData {
        let device = &renderer.device;

        let color_attachment = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let depth_attachment = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        };

        let renderpass = device.new_renderpass(
            &vk::RenderPassCreateInfo::builder()
                .attachments(unsafe {
                    &*(&[
                        vk::AttachmentDescription::builder()
                            .format(swapchain.surface.surface_format.format)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
                            .final_layout(<graphics::DepthPass as TransitionsRenderTarget<0>>::Layout::VK_FLAG),
                        vk::AttachmentDescription::builder()
                            .format(vk::Format::D16_UNORM)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
                            .final_layout(<graphics::DepthPass as TransitionsRenderTarget<1>>::Layout::VK_FLAG),
                    ] as *const [vk::AttachmentDescriptionBuilder<'_>; 2]
                        as *const [vk::AttachmentDescription; 2])
                })
                .subpasses(unsafe {
                    &*(&[vk::SubpassDescription::builder()
                        .color_attachments(&[color_attachment])
                        .depth_stencil_attachment(&depth_attachment)
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)]
                        as *const [vk::SubpassDescriptionBuilder<'_>; 1]
                        as *const [vk::SubpassDescription; 1])
                })
                .dependencies(unsafe {
                    &*(&[vk::SubpassDependency::builder()
                        .src_subpass(vk::SUBPASS_EXTERNAL)
                        .dst_subpass(0)
                        .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
                        .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                        .dst_access_mask(
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )] as *const [vk::SubpassDependencyBuilder<'_>; 1]
                        as *const [vk::SubpassDependency; 1])
                }),
        );

        renderer
            .device
            .set_object_name(renderpass.handle, "Depth prepass renderpass");

        let depth_pipeline_layout = shaders::depth_pipe::PipelineLayout::new(
            &device,
            &model_data.model_set_layout,
            &camera_matrices.set_layout,
        );
        use std::io::Read;
        let path = std::path::PathBuf::from(env!("OUT_DIR")).join("depth_prepass.vert.spv");
        let file = std::fs::File::open(path).expect("Could not find shader.");
        let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
        debug_assert!(shaders::depth_pipe::load_and_verify_spirv(&bytes));
        let depth_pipeline = device.new_graphics_pipeline(
            &[(
                vk::ShaderStageFlags::VERTEX,
                PathBuf::from(env!("OUT_DIR")).join("depth_prepass.vert.spv"),
            )],
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
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::builder()
                        .attachments(&[vk::PipelineColorBlendAttachmentState {
                            blend_enable: 0,
                            ..vk::PipelineColorBlendAttachmentState::default()
                        }])
                        .build(),
                )
                .layout(*depth_pipeline_layout.layout)
                .render_pass(renderpass.handle)
                .subpass(0)
                .build(),
        );

        device.set_object_name(*depth_pipeline, "Depth Pipeline");

        DepthPassData {
            depth_pipeline_layout,
            depth_pipeline,
            renderpass,
        }
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
        device.set_object_name(*gltf_pipeline_layout.layout, "GLTF Pipeline Layout");
        use std::io::Read;
        let path = std::path::PathBuf::from(env!("OUT_DIR")).join("gltf_mesh.vert.spv");
        let file = std::fs::File::open(path).expect("Could not find shader.");
        let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
        debug_assert!(shaders::gltf_mesh::load_and_verify_spirv(&bytes));
        let gltf_pipeline = device.new_graphics_pipeline(
            &[
                (
                    vk::ShaderStageFlags::VERTEX,
                    PathBuf::from(env!("OUT_DIR")).join("gltf_mesh.vert.spv"),
                ),
                (
                    vk::ShaderStageFlags::FRAGMENT,
                    PathBuf::from(env!("OUT_DIR")).join("gltf_mesh.frag.spv"),
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
                        .depth_bias_enable(false)
                        .depth_bias_constant_factor(-0.07)
                        .depth_bias_slope_factor(-1.0)
                        .build(),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::builder()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                        .build(),
                )
                .depth_stencil_state(
                    &vk::PipelineDepthStencilStateCreateInfo::builder()
                        // .depth_write_enable(true)
                        .depth_test_enable(true)
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
                .layout(*gltf_pipeline_layout.layout)
                .render_pass(renderer.renderpass.handle)
                .subpass(0)
                .build(),
        );
        device.set_object_name(*gltf_pipeline, "GLTF Pipeline");

        GltfPassData {
            gltf_pipeline_layout,
            gltf_pipeline,
        }
    }
}

pub(crate) struct MainRenderCommandPool(LocalGraphicsCommandPool);

impl FromResources for MainRenderCommandPool {
    fn from_resources(resources: &Resources) -> Self {
        MainRenderCommandPool(LocalGraphicsCommandPool::from_resources(resources))
    }
}

pub(crate) fn render_frame(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    model_data: Res<ModelData>,
    runtime_config: Res<RuntimeConfiguration>,
    camera_matrices: Res<CameraMatrices>,
    swapchain: Res<Swapchain>,
    consolidated_mesh_buffers: Res<ConsolidatedMeshBuffers>,
    mut local_graphics_command_pool: Local<MainRenderCommandPool>,
    debug_aabb_pass_data: Res<DebugAABBPassData>,
    shadow_mapping_data: Res<ShadowMappingData>,
    base_color_descriptor_set: Res<BaseColorDescriptorSet>,
    cull_pass_data: Res<CullPassData>,
    main_framebuffer: Res<MainFramebuffer>,
    gltf_pass: Res<GltfPassData>,
    graphics_submissions: Res<GraphicsSubmissions>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
    mut query: Query<&AABB>,
) {
    #[cfg(feature = "profiling")]
    microprofile::scope!("ecs", "Renderer");
    // TODO: count this? pack and defragment draw calls?
    let total = shaders::cull_set::bindings::indirect_commands::SIZE as u32
        / size_of::<vk::DrawIndexedIndirectCommand>() as u32;

    let command_pool = local_graphics_command_pool
        .0
        .pools
        .current_mut(image_index.0);

    unsafe {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("main render", "CP reset");
        command_pool.reset();
    }

    let mut command_session = command_pool.session();

    let command_buffer = command_session.record_one_time("Main Render CommandBuffer");
    unsafe {
        let clear_values = &[
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(renderer.renderpass.handle)
            .framebuffer(main_framebuffer.handles[image_index.0 as usize].handle)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.width,
                    height: swapchain.height,
                },
            })
            .clear_values(clear_values);

        let _main_renderpass_marker =
            command_buffer.debug_marker_around("main renderpass", [0.0, 0.0, 1.0, 1.0]);
        renderer.device.cmd_begin_render_pass(
            *command_buffer,
            &begin_info,
            vk::SubpassContents::INLINE,
        );
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
            #[cfg(feature = "profiling")]
            microprofile::scope!("ecs", "debug aabb pass");

            let _aabb_marker =
                command_buffer.debug_marker_around("aabb debug", [1.0, 0.0, 0.0, 1.0]);
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
                    &shaders::DebugAABBPushConstants {
                        center: aabb.0.center().coords,
                        half_extent: aabb.0.half_extents(),
                    },
                );
                renderer.device.cmd_draw(*command_buffer, 36, 1, 0, 0);
            }
        } else {
            let _gltf_meshes_marker =
                command_buffer.debug_marker_around("gltf meshes", [1.0, 0.0, 0.0, 1.0]);
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
                shadow_mapping_data.user_set.current(image_index.0),
                base_color_descriptor_set.set.current(image_index.0),
            );
            renderer.device.cmd_bind_index_buffer(
                *command_buffer,
                cull_pass_data
                    .culled_index_buffer
                    .current(image_index.0)
                    .handle,
                0,
                vk::IndexType::UINT32,
            );
            renderer.device.cmd_bind_vertex_buffers(
                *command_buffer,
                0,
                &[
                    consolidated_mesh_buffers.position_buffer.handle,
                    consolidated_mesh_buffers.normal_buffer.handle,
                    consolidated_mesh_buffers.uv_buffer.handle,
                ],
                &[0, 0, 0],
            );
            renderer.device.cmd_draw_indexed_indirect(
                *command_buffer,
                cull_pass_data
                    .culled_commands_buffer
                    .current(image_index.0)
                    .handle,
                0,
                total,
                size_of::<vk::DrawIndexedIndirectCommand>() as u32,
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
        renderer.device.cmd_end_render_pass(*command_buffer);
    }

    let command_buffer = command_buffer.end();
    *graphics_submissions.main_render.lock() = *command_buffer;
}

pub(crate) struct GraphicsSubmissions {
    transition_shadow_mapping: Mutex<Option<vk::CommandBuffer>>,
    shadow_mapping: Mutex<vk::CommandBuffer>,
    depth_pass: Mutex<vk::CommandBuffer>,
    main_render: Mutex<vk::CommandBuffer>,
}

impl Default for GraphicsSubmissions {
    fn default() -> Self {
        GraphicsSubmissions {
            transition_shadow_mapping: Mutex::new(None),
            shadow_mapping: Mutex::new(vk::CommandBuffer::null()),
            depth_pass: Mutex::new(vk::CommandBuffer::null()),
            main_render: Mutex::new(vk::CommandBuffer::null()),
        }
    }
}

pub(crate) fn submit_graphics_commands(
    renderer: Res<RenderFrame>,
    consolidated_mesh_buffers: Res<ConsolidatedMeshBuffers>,
    mut graphics_submissions: ResMut<GraphicsSubmissions>,
) {
    let mut command_buffers = vec![
        *graphics_submissions.shadow_mapping.get_mut(),
        *graphics_submissions.depth_pass.get_mut(),
    ];
    if let Some(command_buffer) = graphics_submissions.transition_shadow_mapping.get_mut() {
        command_buffers.insert(0, *command_buffer);
    }

    let wait_semaphores = &[renderer.graphics_timeline_semaphore.handle];
    let wait_dst_stage_mask = &[vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS];
    let wait_semaphore_values = &[timeline_value::<_, graphics::Start>(renderer.frame_number)];
    let signal_semaphore_values = &[timeline_value::<_, graphics::DepthPass>(
        renderer.frame_number,
    )];
    let mut signal_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
        .wait_semaphore_values(wait_semaphore_values)
        .signal_semaphore_values(signal_semaphore_values);
    let signal_semaphores = &[renderer.graphics_timeline_semaphore.handle];
    let first_batch = vk::SubmitInfo::builder()
        .command_buffers(&command_buffers)
        .push_next(&mut signal_timeline)
        .wait_semaphores(wait_semaphores)
        .wait_dst_stage_mask(wait_dst_stage_mask)
        .signal_semaphores(signal_semaphores)
        .build();

    let wait_semaphores = &[
        renderer.graphics_timeline_semaphore.handle,
        renderer.compute_timeline_semaphore.handle,
        consolidated_mesh_buffers.sync_timeline.handle,
    ];
    use systems::consolidate_mesh_buffers::sync as consolidate_mesh_buffers;
    let wait_semaphore_values = &[
        timeline_value::<_, graphics::DepthPass>(renderer.frame_number),
        timeline_value::<_, compute::Perform>(renderer.frame_number),
        timeline_value::<_, consolidate_mesh_buffers::Consolidate>(renderer.frame_number),
    ];
    let dst_stage_masks = &[
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    ];
    let signal_semaphores = &[renderer.graphics_timeline_semaphore.handle];
    let command_buffers = &[*graphics_submissions.main_render.get_mut()];
    let signal_semaphore_values = &[timeline_value::<_, graphics::SceneDraw>(
        renderer.frame_number,
    )];
    let mut signal_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
        .wait_semaphore_values(wait_semaphore_values)
        .signal_semaphore_values(signal_semaphore_values);
    let main_render_submit = vk::SubmitInfo::builder()
        .wait_semaphores(wait_semaphores)
        .push_next(&mut signal_timeline)
        .wait_dst_stage_mask(dst_stage_masks)
        .command_buffers(command_buffers)
        .signal_semaphores(signal_semaphores)
        .build();

    let submits = &[first_batch, main_render_submit];

    let queue = renderer.device.graphics_queue.lock();

    unsafe {
        renderer
            .device
            .queue_submit(*queue, submits, vk::Fence::null())
            .unwrap();
    }
}

pub(crate) struct GuiRender {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    texture: Image,
    #[allow(unused)]
    texture_view: ImageView,
    #[allow(unused)]
    sampler: Sampler,
    #[allow(unused)]
    descriptor_set_layout: shaders::imgui_set::DescriptorSetLayout,
    descriptor_set: shaders::imgui_set::DescriptorSet,
    pipeline_layout: shaders::imgui_pipe::PipelineLayout,
    renderpass: RenderPass,
    pipeline: Pipeline,
    transitioned: bool,
    command_pool: DoubleBuffered<StrictCommandPool>,
}

impl GuiRender {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &MainDescriptorPool,
        swapchain: &Swapchain,
        gui: &mut Gui,
    ) -> GuiRender {
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
        let index_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::INDEX_BUFFER,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            1024 * 1024 * size_of::<imgui::DrawIdx>() as vk::DeviceSize,
        );
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
                    .map::<c_uchar>()
                    .expect("failed to map imgui texture");
                texture_data[0..imgui_texture.data.len()].copy_from_slice(imgui_texture.data);
            }
            texture
        };
        let sampler = new_sampler(
            renderer.device.clone(),
            &vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
        );

        let renderpass = {
            let color_attachment = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };
            let depth_attachment = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            };

            dbg!(swapchain.surface.surface_format.format);

            renderer.device.new_renderpass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(unsafe {
                        &*(&[
                            vk::AttachmentDescription::builder()
                                .format(swapchain.surface.surface_format.format)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(
                                    <<graphics::GuiDraw as SuccessorStage<_>>::Previous as TransitionsRenderTarget<0>>::Layout::VK_FLAG,
                                )
                                .final_layout(
                                    <graphics::GuiDraw as TransitionsRenderTarget<0>>::Layout::VK_FLAG,
                                ),
                            vk::AttachmentDescription::builder()
                                .format(vk::Format::D16_UNORM)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(
                                    <<graphics::GuiDraw as SuccessorStage<_>>::Previous as TransitionsRenderTarget<1>>::Layout::VK_FLAG,
                                )
                                .final_layout(
                                    <graphics::GuiDraw as TransitionsRenderTarget<1>>::Layout::VK_FLAG,
                                ),
                        ]
                            as *const [vk::AttachmentDescriptionBuilder<'_>; 2]
                            as *const [vk::AttachmentDescription; 2])
                    })
                    .subpasses(unsafe {
                        &*(&[vk::SubpassDescription::builder()
                            .color_attachments(&[color_attachment])
                            .depth_stencil_attachment(&depth_attachment)
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)]
                            as *const [vk::SubpassDescriptionBuilder<'_>; 1]
                            as *const [vk::SubpassDescription; 1])
                    })
                    .dependencies(unsafe {
                        &*(&[vk::SubpassDependency::builder()
                            .src_subpass(vk::SUBPASS_EXTERNAL)
                            .dst_subpass(0)
                            .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
                            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)]
                            as *const [vk::SubpassDependencyBuilder<'_>; 1]
                            as *const [vk::SubpassDependency; 1])
                    }),
            )
        };
        renderer
            .device
            .set_object_name(renderpass.handle, "GUI RenderPass");

        let descriptor_set_layout = shaders::imgui_set::DescriptorSetLayout::new(&renderer.device);

        let descriptor_set =
            shaders::imgui_set::DescriptorSet::new(&main_descriptor_pool, &descriptor_set_layout);

        let pipeline_layout =
            shaders::imgui_pipe::PipelineLayout::new(&renderer.device, &descriptor_set_layout);

        debug_assert!(shaders::imgui_pipe::load_and_verify_spirv_file(
            "gui.vert.spv"
        ));
        debug_assert!(shaders::imgui_pipe::load_and_verify_spirv_file(
            "gui.frag.spv"
        ));
        let pipeline = renderer.device.new_graphics_pipeline(
            &[
                (
                    vk::ShaderStageFlags::VERTEX,
                    PathBuf::from(env!("OUT_DIR")).join("gui.vert.spv"),
                ),
                (
                    vk::ShaderStageFlags::FRAGMENT,
                    PathBuf::from(env!("OUT_DIR")).join("gui.frag.spv"),
                ),
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
                .depth_stencil_state(&vk::PipelineDepthStencilStateCreateInfo::default())
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
                .render_pass(renderpass.handle)
                .subpass(0)
                .build(),
        );
        renderer.device.set_object_name(*pipeline, "GUI Pipeline");

        let texture_view = new_image_view(
            Arc::clone(&renderer.device),
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

        let command_pool = renderer.new_buffered(|ix| {
            StrictCommandPool::new(
                &renderer.device,
                renderer.device.graphics_queue_family,
                &format!("GuiRender Command Pool[{}]", ix),
            )
        });

        GuiRender {
            vertex_buffer,
            index_buffer,
            texture,
            texture_view,
            sampler,
            descriptor_set_layout,
            descriptor_set,
            renderpass,
            pipeline_layout,
            pipeline,
            transitioned: false,
            command_pool,
        }
    }

    pub(crate) fn render(
        &mut self,
        gui: Rc<RefCell<Gui>>,
        input_handler: Rc<RefCell<InputHandler>>,
        resources: &mut Resources,
    ) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "GuiRender");
        let (
            renderer,
            image_index,
            mut runtime_config,
            mut cull_pass_data,
            main_framebuffer,
            swapchain,
            mut camera,
        ) = resources
            .query::<(
                Res<RenderFrame>,
                Res<ImageIndex>,
                ResMut<RuntimeConfiguration>,
                ResMut<CullPassData>,
                Res<MainFramebuffer>,
                Res<Swapchain>,
                ResMut<Camera>,
            )>()
            .unwrap();
        let mut gui = gui.borrow_mut();
        let gui_draw_data = gui.update(
            &renderer,
            &input_handler.borrow(),
            &swapchain,
            &mut *camera,
            &mut *runtime_config,
            &mut *cull_pass_data,
        );

        let command_pool = self.command_pool.current_mut(image_index.0);

        unsafe {
            #[cfg(feature = "microprofile")]
            microprofile::scope!("gui render", "CP reset");
            command_pool.reset();
        }

        let mut command_session = command_pool.session();

        let command_buffer = command_session.record_one_time("GuiRender CommandBuffer");

        unsafe {
            let _gui_debug_marker = command_buffer.debug_marker_around("GUI", [1.0, 1.0, 0.0, 1.0]);

            if !self.transitioned {
                renderer.device.cmd_pipeline_barrier(
                    *command_buffer,
                    vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::HOST,
                    Default::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .image(self.texture.handle)
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
            }
            self.transitioned = true;

            renderer.device.cmd_begin_render_pass(
                *command_buffer,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.renderpass.handle)
                    .framebuffer(main_framebuffer.handles[image_index.0 as usize].handle)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: swapchain.width,
                            height: swapchain.height,
                        },
                    })
                    .clear_values(&[]),
                vk::SubpassContents::INLINE,
            );
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

            // Split lifetimes
            let GuiRender {
                ref vertex_buffer,
                ref index_buffer,
                ref pipeline_layout,
                ref pipeline,
                ref descriptor_set,
                ..
            } = self;
            pipeline_layout.bind_descriptor_sets(&renderer.device, *command_buffer, descriptor_set);
            renderer.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                **pipeline,
            );
            renderer.device.cmd_bind_vertex_buffers(
                *command_buffer,
                0,
                &[vertex_buffer.handle],
                &[0],
            );
            renderer.device.cmd_bind_index_buffer(
                *command_buffer,
                index_buffer.handle,
                0,
                vk::IndexType::UINT16,
            );
            let [x, y] = gui_draw_data.display_size;
            {
                pipeline_layout.push_constants(
                    &renderer.device,
                    *command_buffer,
                    &shaders::ImguiPushConstants {
                        scale: glm::vec2(2.0 / x, 2.0 / y),
                        translate: glm::vec2(-1.0, -1.0),
                    },
                );
            }
            {
                let mut vertex_offset_coarse: usize = 0;
                let mut index_offset_coarse: usize = 0;
                let mut vertex_slice = vertex_buffer
                    .map::<imgui::DrawVert>()
                    .expect("Failed to map gui vertex buffer");
                let mut index_slice = index_buffer
                    .map::<imgui::DrawIdx>()
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
                                    *command_buffer,
                                    0,
                                    &[vk::Rect2D {
                                        offset: vk::Offset2D {
                                            x: cmd_params.clip_rect[0] as i32,
                                            y: cmd_params.clip_rect[1] as i32,
                                        },
                                        extent: vk::Extent2D {
                                            width: (cmd_params.clip_rect[2]
                                                - cmd_params.clip_rect[0])
                                                as u32,
                                            height: (cmd_params.clip_rect[3]
                                                - cmd_params.clip_rect[1])
                                                as u32,
                                        },
                                    }],
                                );
                                renderer.device.cmd_draw_indexed(
                                    *command_buffer,
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
            renderer.device.cmd_end_render_pass(*command_buffer);
        }
        let command_buffer = command_buffer.end();
        let wait_semaphores = &[renderer.graphics_timeline_semaphore.handle];
        let wait_semaphore_values = &[timeline_value::<_, graphics::SceneDraw>(
            renderer.frame_number,
        )];
        let dst_stage_masks = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = &[renderer.graphics_timeline_semaphore.handle];
        let command_buffers = &[*command_buffer];
        let signal_semaphore_values = &[timeline_value::<_, graphics::GuiDraw>(
            renderer.frame_number,
        )];
        let mut signal_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(wait_semaphore_values)
            .signal_semaphore_values(signal_semaphore_values)
            .build();
        let submit = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .push_next(&mut signal_timeline)
            .wait_dst_stage_mask(dst_stage_masks)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores)
            .build();
        let queue = renderer.device.graphics_queue.lock();

        unsafe {
            renderer
                .device
                .queue_submit(*queue, &[submit], vk::Fence::null())
                .unwrap();
        }
    }
}

pub(crate) fn model_matrices_upload(
    image_index: Res<ImageIndex>,
    mut model_data: ResMut<ModelData>,
    mut query: Query<(&DrawIndex, &mut ModelMatrix)>,
) {
    #[cfg(feature = "profiling")]
    microprofile::scope!("ecs", "ModelMatricesUpload");
    let mut model_mapped = model_data
        .model_buffer
        .current_mut(image_index.0)
        .map::<glm::Mat4>()
        .expect("failed to map Model buffer");
    for (draw_index, model_matrix) in &mut query.iter() {
        model_mapped[draw_index.0 as usize] = model_matrix.0;
    }
}

pub(crate) fn camera_matrices_upload(
    image_index: Res<ImageIndex>,
    camera: Res<Camera>,
    mut camera_matrices: ResMut<CameraMatrices>,
) {
    #[cfg(feature = "profiling")]
    microprofile::scope!("ecs", "CameraMatricesUpload");
    let mut model_mapped = camera_matrices
        .buffer
        .current_mut(image_index.0)
        .map::<shaders::camera_set::bindings::matrices::T>()
        .expect("failed to map camera matrix buffer");
    model_mapped[0] = shaders::camera_set::bindings::matrices::T {
        projection: camera.projection,
        view: camera.view,
        position: camera.position.coords.push(1.0),
        pv: camera.projection * camera.view,
    };
}

pub(crate) struct DepthPassCommandPool(LocalGraphicsCommandPool);

impl FromResources for DepthPassCommandPool {
    fn from_resources(resources: &Resources) -> Self {
        DepthPassCommandPool(LocalGraphicsCommandPool::from_resources(resources))
    }
}

pub(crate) fn depth_only_pass(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    depth_pass: Res<DepthPassData>,
    mut local_graphics_command_pool: Local<DepthPassCommandPool>,
    main_framebuffer: Res<MainFramebuffer>,
    model_data: Res<ModelData>,
    runtime_config: Res<RuntimeConfiguration>,
    camera: Res<Camera>,
    camera_matrices: Res<CameraMatrices>,
    swapchain: Res<Swapchain>,
    graphics_submissions: Res<GraphicsSubmissions>,
    mut query: Query<(&Position, &DrawIndex, &GltfMesh)>,
) {
    #[cfg(feature = "profiling")]
    microprofile::scope!("ecs", "DepthOnlyPass");

    let command_pool = local_graphics_command_pool
        .0
        .pools
        .current_mut(image_index.0);

    unsafe {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("depth only", "CP reset");
        command_pool.reset();
    }

    let mut command_session = command_pool.session();

    let command_buffer = command_session.record_one_time("Depth Only CommandBuffer");

    unsafe {
        let _marker = command_buffer.debug_marker_around("depth prepass", [0.3, 0.3, 0.3, 1.0]);
        renderer.device.cmd_begin_render_pass(
            *command_buffer,
            &vk::RenderPassBeginInfo::builder()
                .render_pass(depth_pass.renderpass.handle)
                .framebuffer(main_framebuffer.handles[image_index.0 as usize].handle)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: swapchain.width,
                        height: swapchain.height,
                    },
                })
                .clear_values(&[
                    vk::ClearValue::default(),
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ]),
            vk::SubpassContents::INLINE,
        );
        if !runtime_config.debug_aabbs {
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
            renderer.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *depth_pass.depth_pipeline,
            );
            depth_pass.depth_pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                &model_data.model_set.current(image_index.0),
                &camera_matrices.set.current(image_index.0),
            );
            for (mesh_position, draw_index, mesh) in &mut query.iter() {
                let (index_buffer, index_count) =
                    pick_lod(&mesh.index_buffers, camera.position, mesh_position.0);
                renderer.device.cmd_bind_index_buffer(
                    *command_buffer,
                    index_buffer.handle,
                    0,
                    vk::IndexType::UINT32,
                );
                renderer.device.cmd_bind_vertex_buffers(
                    *command_buffer,
                    0,
                    &[mesh.vertex_buffer.handle],
                    &[0],
                );
                renderer.device.cmd_draw_indexed(
                    *command_buffer,
                    (*index_count).try_into().unwrap(),
                    1,
                    0,
                    0,
                    draw_index.0,
                );
            }
        }
        renderer.device.cmd_end_render_pass(*command_buffer);
    }
    let command_buffer = command_buffer.end();
    *graphics_submissions.depth_pass.lock() = *command_buffer;
}
