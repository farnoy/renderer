// TODO: pub(crate) should disappear?
mod alloc;
mod device;
mod entry;
mod gltf_mesh;
mod helpers;
mod instance;
mod swapchain;
mod systems {
    pub mod consolidate_mesh_buffers;
    pub mod cull_pipeline;
    pub mod present;
    pub mod shadow_mapping;
    pub mod textures;
}

use crate::ecs::{
    components::{Matrices, Position},
    systems::Camera,
};
use ash::{version::DeviceV1_0, vk};
use cgmath;
use imgui::{self, im_str};
use microprofile::scope;
use specs::*;
use std::{
    convert::TryInto, mem::size_of, os::raw::c_uchar, path::PathBuf, ptr, slice::from_raw_parts,
    sync::Arc,
};
use winit;

use self::{
    device::{
        Buffer, CommandBuffer, CommandPool, DescriptorPool, DescriptorSet, DescriptorSetLayout,
        Device, DoubleBuffered, Image, RenderPass, Semaphore,
    },
    helpers::*,
    instance::Instance,
};

pub use self::{
    gltf_mesh::{load as load_gltf, LoadedMesh},
    systems::{
        consolidate_mesh_buffers::*, cull_pipeline::*, present::*, shadow_mapping::*, textures::*,
    },
};

#[derive(Clone, Component)]
#[storage(VecStorage)]
pub struct GltfMesh {
    pub vertex_buffer: Arc<Buffer>,
    pub normal_buffer: Arc<Buffer>,
    pub uv_buffer: Arc<Buffer>,
    pub index_buffers: Arc<Vec<(Buffer, u64)>>,
    pub vertex_len: u64,
    pub aabb_c: cgmath::Vector3<f32>,
    pub aabb_h: cgmath::Vector3<f32>,
}

// TODO: rename
pub struct RenderFrame {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub swapchain: Swapchain,
    pub compute_command_pool: Arc<CommandPool>,
    pub renderpass: RenderPass,
}

impl RenderFrame {
    pub fn new() -> (RenderFrame, winit::EventsLoop) {
        let (instance, events_loop) = Instance::new(1920, 1080).expect("Failed to create instance");
        let instance = Arc::new(instance);
        let device = Arc::new(Device::new(&instance).expect("Failed to create device"));
        device.set_object_name(device.handle(), "Device");
        let swapchain = new_swapchain(&instance, &device);
        let compute_command_pool = device.new_command_pool(
            device.compute_queue_family,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        );
        let main_renderpass = {
            let color_attachment = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };
            let depth_attachment = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
            };

            device.new_renderpass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(unsafe {
                        &*(&[
                            vk::AttachmentDescription::builder()
                                .format(swapchain.surface_format.format)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(vk::ImageLayout::UNDEFINED)
                                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                            vk::AttachmentDescription::builder()
                                .format(vk::Format::D32_SFLOAT)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(
                                    vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
                                )
                                .final_layout(
                                    vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
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
        device.set_object_name(main_renderpass.handle, "Main pass RenderPass");

        (
            RenderFrame {
                instance: Arc::clone(&instance),
                device: Arc::clone(&device),
                compute_command_pool: Arc::new(compute_command_pool),
                renderpass: main_renderpass,
                swapchain,
            },
            events_loop,
        )
    }

    pub fn new_buffered<T, F: FnMut(u32) -> T>(&self, creator: F) -> DoubleBuffered<T> {
        DoubleBuffered::new(creator)
    }
}

pub struct MainDescriptorPool(pub Arc<DescriptorPool>);

impl specs::shred::SetupHandler<MainDescriptorPool> for MainDescriptorPool {
    fn setup(world: &mut World) {
        if world.has_value::<MainDescriptorPool>() {
            return;
        }

        let renderer = world.fetch::<RenderFrame>();

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
        drop(renderer);

        world.insert(MainDescriptorPool(descriptor_pool));
    }
}

pub struct MainAttachments {
    pub swapchain_images: Vec<Arc<SwapchainImage>>,
    pub swapchain_image_views: Vec<Arc<ImageView>>,
    pub depth_images: Vec<Arc<Image>>,
    pub depth_image_views: Vec<Arc<ImageView>>,
    pub device: Arc<Device>,
}

impl specs::shred::SetupHandler<MainAttachments> for MainAttachments {
    fn setup(world: &mut World) {
        if world.has_value::<MainAttachments>() {
            return;
        }

        let renderer = world.fetch::<RenderFrame>();
        let Swapchain {
            handle: ref swapchain,
            ref surface_format,
            ..
        } = renderer.swapchain;
        let Instance {
            window_width,
            window_height,
            ..
        } = *renderer.instance;

        let images = unsafe {
            swapchain
                .ext
                .get_swapchain_images(swapchain.swapchain)
                .unwrap()
        };
        println!("swapchain images len {}", images.len());
        let depth_images = (0..images.len())
            .map(|_| {
                renderer.device.new_image(
                    vk::Format::D32_SFLOAT,
                    vk::Extent3D {
                        width: window_width,
                        height: window_height,
                        depth: 1,
                    },
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                )
            })
            .collect::<Vec<_>>();
        let image_views = images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
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
                    .format(vk::Format::D32_SFLOAT)
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

        let attachments = MainAttachments {
            swapchain_images: images
                .iter()
                .cloned()
                .map(|handle| SwapchainImage { handle })
                .map(Arc::new)
                .collect(),
            swapchain_image_views: image_views.into_iter().map(Arc::new).collect(),
            depth_images: depth_images.into_iter().map(Arc::new).collect(),
            depth_image_views: depth_image_views.into_iter().map(Arc::new).collect(),
            device: Arc::clone(&renderer.device),
        };
        drop(renderer);

        world.insert(attachments);
    }
}

pub struct MainFramebuffer {
    pub handles: Vec<Framebuffer>,
}

impl specs::shred::SetupHandler<MainFramebuffer> for MainFramebuffer {
    fn setup(world: &mut World) {
        if world.has_value::<MainFramebuffer>() {
            return;
        }

        let result = world.exec(
            |(renderer, main_attachments): (
                ReadExpect<RenderFrame>,
                Read<MainAttachments, MainAttachments>,
            )| {
                let Instance {
                    window_width,
                    window_height,
                    ..
                } = *renderer.instance;

                let handles = main_attachments
                    .swapchain_image_views
                    .iter()
                    .cloned()
                    .zip(main_attachments.depth_image_views.iter().cloned())
                    .enumerate()
                    .map(|(ix, (present_image_view, depth_image_view))| {
                        let framebuffer_attachments =
                            [present_image_view.handle, depth_image_view.handle];
                        let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                            .render_pass(renderer.renderpass.handle)
                            .attachments(&framebuffer_attachments)
                            .width(window_width)
                            .height(window_height)
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
            },
        );

        world.insert(result);
    }
}

pub struct GraphicsCommandPool(pub Arc<CommandPool>);

impl specs::shred::SetupHandler<GraphicsCommandPool> for GraphicsCommandPool {
    fn setup(world: &mut World) {
        if world.has_value::<GraphicsCommandPool>() {
            return;
        }

        let renderer = world.fetch::<RenderFrame>();
        let graphics_command_pool = renderer.device.new_command_pool(
            renderer.device.graphics_queue_family,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        );
        drop(renderer);

        world.insert(GraphicsCommandPool(Arc::new(graphics_command_pool)));
    }
}

pub struct MVPData {
    pub mvp_set_layout: DescriptorSetLayout,
    pub mvp_set: DoubleBuffered<DescriptorSet>,
    pub mvp_buffer: DoubleBuffered<Buffer>,
}

impl specs::shred::SetupHandler<MVPData> for MVPData {
    fn setup(world: &mut World) {
        if world.has_value::<MVPData>() {
            return;
        }

        let result = world.exec(
            |(renderer, main_descriptor_pool): (
                ReadExpect<RenderFrame>,
                Read<MainDescriptorPool, MainDescriptorPool>,
            )| {
                let device = &renderer.device;

                let mvp_set_layout =
                    device.new_descriptor_set_layout(&[vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
                        p_immutable_samplers: ptr::null(),
                    }]);
                device.set_object_name(mvp_set_layout.handle, "MVP Set Layout");

                let mvp_buffer = DoubleBuffered::new(|ix| {
                    let b = device.new_buffer(
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                        4 * 4 * 4 * 4096,
                    );
                    device.set_object_name(b.handle, &format!("MVP Buffer - {}", ix));
                    b
                });
                let mvp_set = DoubleBuffered::new(|ix| {
                    let s = main_descriptor_pool.0.allocate_set(&mvp_set_layout);
                    device.set_object_name(s.handle, &format!("MVP Set - {}", ix));

                    {
                        let mvp_updates = &[vk::DescriptorBufferInfo {
                            buffer: mvp_buffer.current(ix).handle,
                            offset: 0,
                            range: 4096 * size_of::<cgmath::Matrix4<f32>>() as vk::DeviceSize,
                        }];
                        unsafe {
                            device.update_descriptor_sets(
                                &[vk::WriteDescriptorSet::builder()
                                    .dst_set(s.handle)
                                    .dst_binding(0)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .buffer_info(mvp_updates)
                                    .build()],
                                &[],
                            );
                        }
                    }

                    s
                });

                MVPData {
                    mvp_set_layout,
                    mvp_set,
                    mvp_buffer,
                }
            },
        );

        world.insert(result);
    }
}

pub struct DepthPassData {
    pub depth_pipeline: Pipeline,
    pub depth_pipeline_layout: PipelineLayout,
    pub renderpass: RenderPass,
    pub framebuffer: Vec<Framebuffer>,
    pub complete_semaphore: DoubleBuffered<Semaphore>,
    pub previous_command_buffer: DoubleBuffered<Option<CommandBuffer>>,
}

impl specs::shred::SetupHandler<DepthPassData> for DepthPassData {
    fn setup(world: &mut World) {
        if world.has_value::<DepthPassData>() {
            return;
        }

        let result = world.exec(
            |(renderer, mvp_data, main_attachments): (
                ReadExpect<RenderFrame>,
                Read<MVPData, MVPData>,
                Read<MainAttachments, MainAttachments>,
            )| {
                let device = &renderer.device;
                let instance = &renderer.instance;

                let renderpass = device.new_renderpass(
                    &vk::RenderPassCreateInfo::builder()
                        .attachments(unsafe {
                            &*(&[vk::AttachmentDescription::builder()
                                .format(vk::Format::D32_SFLOAT)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(vk::ImageLayout::UNDEFINED)
                                .final_layout(
                                    vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
                                )]
                                as *const [vk::AttachmentDescriptionBuilder<'_>; 1]
                                as *const [vk::AttachmentDescription; 1])
                        })
                        .subpasses(unsafe {
                            &*(&[vk::SubpassDescription::builder()
                                .depth_stencil_attachment(&vk::AttachmentReference {
                                    attachment: 0,
                                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                })
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
                                )]
                                as *const [vk::SubpassDependencyBuilder<'_>; 1]
                                as *const [vk::SubpassDependency; 1])
                        }),
                );

                renderer
                    .device
                    .set_object_name(renderpass.handle, "Depth prepass renderpass");

                let depth_pipeline_layout =
                    new_pipeline_layout(Arc::clone(&device), &[&mvp_data.mvp_set_layout], &[]);
                let depth_pipeline = new_graphics_pipeline2(
                    Arc::clone(&device),
                    &[(
                        vk::ShaderStageFlags::VERTEX,
                        PathBuf::from(env!("OUT_DIR")).join("depth_prepass.vert.spv"),
                    )],
                    vk::GraphicsPipelineCreateInfo::builder()
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::builder()
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription {
                                        location: 0,
                                        binding: 0,
                                        format: vk::Format::R32G32B32_SFLOAT,
                                        offset: 0,
                                    },
                                ])
                                .vertex_binding_descriptions(&[vk::VertexInputBindingDescription {
                                    binding: 0,
                                    stride: size_of::<f32>() as u32 * 3,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                }])
                                .build(),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::builder()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                .build(),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::builder()
                                .scissors(&[vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: vk::Extent2D {
                                        width: instance.window_width,
                                        height: instance.window_height,
                                    },
                                }])
                                .viewports(&[vk::Viewport {
                                    x: 0.0,
                                    y: (instance.window_height as f32),
                                    width: instance.window_width as f32,
                                    height: -(instance.window_height as f32),
                                    min_depth: 0.0,
                                    max_depth: 1.0,
                                }])
                                .build(),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::builder()
                                .cull_mode(vk::CullModeFlags::BACK)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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
                                    blend_enable: 1,
                                    src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                                    dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                                    color_blend_op: vk::BlendOp::ADD,
                                    src_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                                    dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                                    alpha_blend_op: vk::BlendOp::ADD,
                                    color_write_mask: vk::ColorComponentFlags::all(),
                                }])
                                .build(),
                        )
                        .layout(depth_pipeline_layout.handle)
                        .render_pass(renderpass.handle)
                        .subpass(0)
                        .build(),
                );

                device.set_object_name(depth_pipeline.handle, "Depth Pipeline");

                let framebuffer = main_attachments
                    .depth_image_views
                    .iter()
                    .enumerate()
                    .map(|(ix, depth_image_view)| {
                        let framebuffer_attachments = [depth_image_view.handle];
                        let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                            .render_pass(renderpass.handle)
                            .attachments(&framebuffer_attachments)
                            .width(renderer.instance.window_width)
                            .height(renderer.instance.window_height)
                            .layers(1);
                        let handle = unsafe {
                            renderer
                                .device
                                .create_framebuffer(&frame_buffer_create_info, None)
                                .unwrap()
                        };
                        renderer
                            .device
                            .set_object_name(handle, &format!("Depth only Framebuffer - {}", ix));
                        Framebuffer {
                            handle,
                            device: Arc::clone(&renderer.device),
                        }
                    })
                    .collect();

                let complete_semaphore = renderer.new_buffered(|ix| {
                    let s = renderer.device.new_semaphore();
                    renderer
                        .device
                        .set_object_name(s.handle, &format!("Depth complete semaphore - {}", ix));
                    s
                });

                let previous_command_buffer = renderer.new_buffered(|_| None);

                DepthPassData {
                    depth_pipeline_layout,
                    depth_pipeline,
                    renderpass,
                    framebuffer,
                    complete_semaphore,
                    previous_command_buffer,
                }
            },
        );

        world.insert(result);
    }
}

pub struct GltfPassData {
    pub gltf_pipeline: Pipeline,
    pub gltf_pipeline_layout: PipelineLayout,
}

impl specs::shred::SetupHandler<GltfPassData> for GltfPassData {
    fn setup(world: &mut World) {
        if world.has_value::<GltfPassData>() {
            return;
        }

        let result = world.exec(
            |(renderer, mvp_data, base_color): (
                ReadExpect<RenderFrame>,
                Read<MVPData, MVPData>,
                Read<BaseColorDescriptorSet, BaseColorDescriptorSet>,
            )| {
                let device = &renderer.device;
                let instance = &renderer.instance;

                let gltf_pipeline_layout = new_pipeline_layout(
                    Arc::clone(&device),
                    { &[&mvp_data.mvp_set_layout, &base_color.layout] },
                    &[],
                );
                device.set_object_name(gltf_pipeline_layout.handle, "GLTF Pipeline Layout");
                let gltf_pipeline = new_graphics_pipeline2(
                    Arc::clone(&device),
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
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::builder()
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription {
                                        location: 0,
                                        binding: 0,
                                        format: vk::Format::R32G32B32_SFLOAT,
                                        offset: 0,
                                    },
                                    vk::VertexInputAttributeDescription {
                                        location: 1,
                                        binding: 1,
                                        format: vk::Format::R32G32B32_SFLOAT,
                                        offset: 0,
                                    },
                                    vk::VertexInputAttributeDescription {
                                        location: 2,
                                        binding: 2,
                                        format: vk::Format::R32G32_SFLOAT,
                                        offset: 0,
                                    },
                                ])
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription {
                                        binding: 0,
                                        stride: size_of::<f32>() as u32 * 3,
                                        input_rate: vk::VertexInputRate::VERTEX,
                                    },
                                    vk::VertexInputBindingDescription {
                                        binding: 1,
                                        stride: size_of::<f32>() as u32 * 3,
                                        input_rate: vk::VertexInputRate::VERTEX,
                                    },
                                    vk::VertexInputBindingDescription {
                                        binding: 2,
                                        stride: size_of::<f32>() as u32 * 2,
                                        input_rate: vk::VertexInputRate::VERTEX,
                                    },
                                ])
                                .build(),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::builder()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                .build(),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::builder()
                                .scissors(&[vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: vk::Extent2D {
                                        width: instance.window_width,
                                        height: instance.window_height,
                                    },
                                }])
                                .viewports(&[vk::Viewport {
                                    x: 0.0,
                                    y: (instance.window_height as f32),
                                    width: instance.window_width as f32,
                                    height: -(instance.window_height as f32),
                                    min_depth: 0.0,
                                    max_depth: 1.0,
                                }])
                                .build(),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::builder()
                                .cull_mode(vk::CullModeFlags::BACK)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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
                        .layout(gltf_pipeline_layout.handle)
                        .render_pass(renderer.renderpass.handle)
                        .subpass(0)
                        .build(),
                );
                device.set_object_name(gltf_pipeline.handle, "GLTF Pipeline");

                GltfPassData {
                    gltf_pipeline_layout,
                    gltf_pipeline,
                }
            },
        );

        world.insert(result);
    }
}

pub struct Renderer;

impl<'a> System<'a> for Renderer {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        ReadExpect<'a, RenderFrame>,
        Read<'a, MainAttachments, MainAttachments>,
        Read<'a, MainFramebuffer, MainFramebuffer>,
        Write<'a, Gui, Gui>,
        ReadStorage<'a, GltfMeshBufferIndex>,
        Read<'a, BaseColorDescriptorSet, BaseColorDescriptorSet>,
        ReadExpect<'a, ConsolidatedMeshBuffers>,
        ReadExpect<'a, CullPassData>,
        Write<'a, PresentData, PresentData>,
        Read<'a, ImageIndex>,
        ReadStorage<'a, GltfMesh>,
        ReadStorage<'a, Position>,
        Read<'a, Camera>,
        Read<'a, DepthPassData, DepthPassData>,
        Read<'a, MVPData, MVPData>,
        Read<'a, GltfPassData, GltfPassData>,
        Write<'a, GraphicsCommandPool, GraphicsCommandPool>,
    );

    fn run(
        &mut self,
        (
            renderer,
            main_attachments,
            main_framebuffer,
            mut gui,
            mesh_buffer_indices,
            base_color_descriptor_set,
            consolidated_mesh_buffers,
            cull_pass_data,
            mut present_data,
            image_index,
            meshes,
            positions,
            camera,
            depth_pass,
            mvp_data,
            gltf_pass,
            graphics_command_pool,
        ): Self::SystemData,
    ) {
        microprofile::scope!("ecs", "renderer");
        let total = mesh_buffer_indices.join().count() as u32;
        let command_buffer = graphics_command_pool.0.record_one_time({
            let renderer = &renderer;
            let consolidated_mesh_buffers = &consolidated_mesh_buffers;
            let image_index = &image_index;
            let cull_pass_data = &cull_pass_data;
            let depth_pass = &depth_pass;
            move |command_buffer| unsafe {
                if !gui.transitioned {
                    renderer.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::HOST,
                        vk::PipelineStageFlags::HOST,
                        Default::default(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::builder()
                            .image(gui.texture.handle)
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
                gui.transitioned = true;
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
                            width: renderer.instance.window_width,
                            height: renderer.instance.window_height,
                        },
                    })
                    .clear_values(clear_values);

                renderer.device.debug_marker_around(
                    command_buffer,
                    "main renderpass",
                    [0.0, 0.0, 1.0, 1.0],
                    || {
                        renderer.device.cmd_begin_render_pass(
                            command_buffer,
                            &begin_info,
                            vk::SubpassContents::INLINE,
                        );
                        renderer.device.debug_marker_around(
                            command_buffer,
                            "gltf meshes",
                            [1.0, 0.0, 0.0, 1.0],
                            || {
                                // gltf mesh
                                renderer.device.cmd_bind_pipeline(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    gltf_pass.gltf_pipeline.handle,
                                );
                                renderer.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    gltf_pass.gltf_pipeline_layout.handle,
                                    0,
                                    &[
                                        mvp_data.mvp_set.current(image_index.0).handle,
                                        base_color_descriptor_set.set.current(image_index.0).handle,
                                    ],
                                    &[],
                                );
                                renderer.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    cull_pass_data
                                        .culled_index_buffer
                                        .current(image_index.0)
                                        .handle,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                renderer.device.cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &[
                                        consolidated_mesh_buffers.position_buffer.handle,
                                        consolidated_mesh_buffers.normal_buffer.handle,
                                        consolidated_mesh_buffers.uv_buffer.handle,
                                    ],
                                    &[0, 0, 0],
                                );
                                renderer.device.cmd_draw_indexed_indirect(
                                    command_buffer,
                                    cull_pass_data
                                        .culled_commands_buffer
                                        .current(image_index.0)
                                        .handle,
                                    0,
                                    total,
                                    size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                                );
                            },
                        );
                        renderer.device.debug_marker_around(
                            command_buffer,
                            "GUI",
                            [1.0, 1.0, 0.0, 1.0],
                            || {
                                let pipeline_layout = gui.pipeline_layout.handle;
                                renderer.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    pipeline_layout,
                                    0,
                                    &[gui.descriptor_set.handle],
                                    &[],
                                );
                                renderer.device.cmd_bind_pipeline(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    gui.pipeline.handle,
                                );
                                renderer.device.cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &[gui.vertex_buffer.handle],
                                    &[0],
                                );
                                renderer.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    gui.index_buffer.handle,
                                    0,
                                    vk::IndexType::UINT16,
                                );
                                // Split t
                                let Gui {
                                    ref mut imgui,
                                    ref vertex_buffer,
                                    ref index_buffer,
                                    ..
                                } = *gui;
                                let ui = imgui.frame(
                                    imgui::FrameSize {
                                        logical_size: (
                                            f64::from(renderer.instance.window_width),
                                            f64::from(renderer.instance.window_height),
                                        ),
                                        hidpi_factor: 1.0,
                                    },
                                    1.0,
                                );
                                let alloc_stats = alloc::stats(renderer.device.allocator);
                                let s = format!("Alloc stats {:?}", alloc_stats.total);
                                ui.window(im_str!("Renderer"))
                                    .size((500.0, 300.0), imgui::ImGuiCond::Always)
                                    .build(|| {
                                        ui.text_wrapped(im_str!("{}", s));
                                    });
                                ui.render(|ui, draw_data| {
                                    let (x, y) = ui.imgui().display_size();
                                    let constants = [2.0 / x, 2.0 / y, -1.0, -1.0];

                                    let casted: &[u8] = {
                                        from_raw_parts(
                                            constants.as_ptr() as *const u8,
                                            constants.len() * 4,
                                        )
                                    };
                                    renderer.device.cmd_push_constants(
                                        command_buffer,
                                        pipeline_layout,
                                        vk::ShaderStageFlags::VERTEX,
                                        0,
                                        casted,
                                    );
                                    let mut vertex_offset = 0;
                                    let mut index_offset = 0;
                                    {
                                        let mut vertex_slice = vertex_buffer
                                            .map::<imgui::ImDrawVert>()
                                            .expect("Failed to map gui vertex buffer");
                                        let mut index_slice = index_buffer
                                            .map::<imgui::ImDrawIdx>()
                                            .expect("Failed to map gui index buffer");
                                        for draw_list in draw_data.into_iter() {
                                            index_slice[0..draw_list.idx_buffer.len()]
                                                .copy_from_slice(draw_list.idx_buffer);
                                            vertex_slice[0..draw_list.vtx_buffer.len()]
                                                .copy_from_slice(draw_list.vtx_buffer);
                                            for draw_cmd in draw_list.cmd_buffer {
                                                renderer.device.cmd_draw_indexed(
                                                    command_buffer,
                                                    draw_cmd.elem_count,
                                                    1,
                                                    index_offset,
                                                    vertex_offset,
                                                    0,
                                                );
                                                index_offset += draw_cmd.elem_count as u32;
                                            }
                                            vertex_offset += draw_list.vtx_buffer.len() as i32;
                                        }
                                    }
                                    if false {
                                        return Err(3i8);
                                    }
                                    Ok(())
                                })
                                .expect("failed rendering ui");
                            },
                        );
                        renderer.device.cmd_end_render_pass(command_buffer);
                    },
                );
            }
        });
        let mut wait_semaphores = vec![
            depth_pass.complete_semaphore.current(image_index.0).handle,
            cull_pass_data
                .cull_complete_semaphore
                .current(image_index.0)
                .handle,
        ];
        let mut dst_stage_masks = vec![
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ];
        if let Some(ref semaphore) = consolidated_mesh_buffers.sync_point.current(image_index.0) {
            wait_semaphores.push(semaphore.handle);
            dst_stage_masks.push(vk::PipelineStageFlags::COMPUTE_SHADER);
        }
        let signal_semaphores = &[present_data.render_complete_semaphore.handle];
        let command_buffers = &[*command_buffer];
        let submit = vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&dst_stage_masks)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores)
            .build();
        let queue = renderer.device.graphics_queue.lock();

        unsafe {
            renderer
                .device
                .queue_submit(
                    *queue,
                    &[submit],
                    present_data
                        .render_complete_fence
                        .current(image_index.0)
                        .handle,
                )
                .unwrap();
        }

        *present_data
            .render_command_buffer
            .current_mut(image_index.0) = Some(command_buffer);
    }
}

pub struct Gui {
    pub imgui: imgui::ImGui,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub texture: Image,
    pub texture_view: ImageView,
    pub sampler: Sampler,
    pub descriptor_set_layout: DescriptorSetLayout,
    pub descriptor_set: DescriptorSet,
    pub pipeline_layout: PipelineLayout,
    pub pipeline: Pipeline,
    pub transitioned: bool,
}

impl specs::shred::SetupHandler<Gui> for Gui {
    fn setup(world: &mut World) {
        if world.has_value::<Gui>() {
            return;
        }

        let result = world.exec(
            |(renderer, main_descriptor_pool): (
                ReadExpect<RenderFrame>,
                Read<MainDescriptorPool, MainDescriptorPool>,
            )| {
                let mut imgui = imgui::ImGui::init();
                let vertex_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    4096 * size_of::<imgui::ImDrawVert>() as vk::DeviceSize,
                );
                let index_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    4096 * size_of::<imgui::ImDrawIdx>() as vk::DeviceSize,
                );
                let texture = imgui.prepare_texture(|handle| {
                    let texture = renderer.device.new_image(
                        vk::Format::R8G8B8A8_UNORM,
                        vk::Extent3D {
                            width: handle.width,
                            height: handle.height,
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
                        texture_data[0..handle.pixels.len()].copy_from_slice(handle.pixels);
                    }
                    texture
                });
                let sampler = new_sampler(
                    renderer.device.clone(),
                    &vk::SamplerCreateInfo::builder()
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                );

                let descriptor_set_layout =
                    renderer
                        .device
                        .new_descriptor_set_layout(&[vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            p_immutable_samplers: ptr::null(),
                        }]);

                let descriptor_set = main_descriptor_pool.0.allocate_set(&descriptor_set_layout);

                let pipeline_layout = new_pipeline_layout(
                    renderer.device.clone(),
                    &[&descriptor_set_layout],
                    &[vk::PushConstantRange {
                        offset: 0,
                        size: 4 * size_of::<f32>() as u32,
                        stage_flags: vk::ShaderStageFlags::VERTEX,
                    }],
                );

                let pipeline = new_graphics_pipeline2(
                    renderer.device.clone(),
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
                                    stride: size_of::<imgui::ImDrawVert>() as u32,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                }])
                                .build(),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::builder()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                .build(),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::builder()
                                .scissors(&[vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: vk::Extent2D {
                                        width: renderer.instance.window_width,
                                        height: renderer.instance.window_height,
                                    },
                                }])
                                .viewports(&[vk::Viewport {
                                    x: 0.0,
                                    y: (renderer.instance.window_height as f32),
                                    width: renderer.instance.window_width as f32,
                                    height: -(renderer.instance.window_height as f32),
                                    min_depth: 0.0,
                                    max_depth: 1.0,
                                }])
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
                        .layout(pipeline_layout.handle)
                        .render_pass(renderer.renderpass.handle)
                        .subpass(0)
                        .build(),
                );
                renderer
                    .device
                    .set_object_name(pipeline.handle, "GUI Pipeline");

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
                            .dst_set(descriptor_set.handle)
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

                Gui {
                    imgui,
                    vertex_buffer,
                    index_buffer,
                    texture,
                    texture_view,
                    sampler,
                    descriptor_set_layout,
                    descriptor_set,
                    pipeline_layout,
                    pipeline,
                    transitioned: false,
                }
            },
        );

        world.insert(result);
    }
}

pub struct MVPUpload;

impl<'a> System<'a> for MVPUpload {
    type SystemData = (
        ReadStorage<'a, Matrices>,
        ReadStorage<'a, GltfMeshBufferIndex>,
        Read<'a, ImageIndex>,
        Write<'a, MVPData, MVPData>,
    );

    fn run(&mut self, (matrices, indices, image_index, mut mvp_data): Self::SystemData) {
        microprofile::scope!("ecs", "mvp upload");
        let mut mvp_mapped = mvp_data
            .mvp_buffer
            .current_mut(image_index.0)
            .map::<cgmath::Matrix4<f32>>()
            .expect("failed to map MVP buffer");
        for (index, matrices) in (&indices, &matrices).join() {
            mvp_mapped[index.0 as usize] = matrices.mvp;
        }
    }
}

pub fn setup_ecs(world: &mut World) {
    world.register::<GltfMeshBufferIndex>();
    world.register::<GltfMeshBaseColorTexture>();
    world.register::<BaseColorVisitedMarker>();
    world.register::<CoarseCulled>();
}

pub struct DepthOnlyPass;

impl<'a> System<'a> for DepthOnlyPass {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        ReadExpect<'a, RenderFrame>,
        ReadStorage<'a, GltfMeshBufferIndex>,
        Read<'a, PresentData, PresentData>,
        Read<'a, ImageIndex>,
        ReadStorage<'a, GltfMesh>,
        ReadStorage<'a, Position>,
        Read<'a, Camera>,
        Write<'a, DepthPassData, DepthPassData>,
        Read<'a, MVPData, MVPData>,
        Write<'a, GraphicsCommandPool, GraphicsCommandPool>,
        Read<'a, ShadowMappingData, ShadowMappingData>,
    );

    fn run(
        &mut self,
        (
            renderer,
            mesh_buffer_indices,
            present_data,
            image_index,
            meshes,
            positions,
            camera,
            mut depth_pass,
            mvp_data,
            graphics_command_pool,
            shadow_mapping_data,
        ): Self::SystemData,
    ) {
        microprofile::scope!("ecs", "depth-pass");
        let command_buffer = graphics_command_pool.0.record_one_time({
            let renderer = &renderer;
            let image_index = &image_index;
            let depth_pass = &depth_pass;
            move |command_buffer| unsafe {
                renderer.device.debug_marker_around(
                    command_buffer,
                    "depth prepass",
                    [0.3, 0.3, 0.3, 1.0],
                    || {
                        renderer.device.cmd_begin_render_pass(
                            command_buffer,
                            &vk::RenderPassBeginInfo::builder()
                                .render_pass(depth_pass.renderpass.handle)
                                .framebuffer(depth_pass.framebuffer[image_index.0 as usize].handle)
                                .render_area(vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: vk::Extent2D {
                                        width: renderer.instance.window_width,
                                        height: renderer.instance.window_height,
                                    },
                                })
                                .clear_values(&[vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue {
                                        depth: 1.0,
                                        stencil: 0,
                                    },
                                }]),
                            vk::SubpassContents::INLINE,
                        );
                        renderer.device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            depth_pass.depth_pipeline.handle,
                        );
                        renderer.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            depth_pass.depth_pipeline_layout.handle,
                            0,
                            &[mvp_data.mvp_set.current(image_index.0).handle],
                            &[],
                        );
                        for (mesh, index, mesh_position) in
                            (&meshes, &mesh_buffer_indices, &positions).join()
                        {
                            let (index_buffer, index_count) =
                                pick_lod(&mesh.index_buffers, camera.position, mesh_position.0);
                            renderer.device.cmd_bind_index_buffer(
                                command_buffer,
                                index_buffer.handle,
                                0,
                                vk::IndexType::UINT32,
                            );
                            renderer.device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[mesh.vertex_buffer.handle],
                                &[0],
                            );
                            renderer.device.cmd_draw_indexed(
                                command_buffer,
                                (*index_count).try_into().unwrap(),
                                1,
                                0,
                                0,
                                index.0,
                            );
                        }
                        renderer.device.cmd_end_render_pass(command_buffer);
                    },
                );
            }
        });
        let wait_semaphores = &[shadow_mapping_data
            .complete_semaphore
            .current(image_index.0)
            .handle];
        let dst_stage_masks = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = &[depth_pass.complete_semaphore.current(image_index.0).handle];
        let command_buffers = &[*command_buffer];
        let submit = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
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

        *depth_pass
            .previous_command_buffer
            .current_mut(image_index.0) = Some(command_buffer);
    }
}
