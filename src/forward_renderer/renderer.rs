// TODO: pub(crate) should disappear?
pub mod alloc;
pub mod device;
mod entry;
mod gltf_mesh;
mod helpers;
mod instance;
mod swapchain;
mod systems;

use super::ecs::components::Matrices;
use ash::{prelude::*, version::DeviceV1_0, vk};
use cgmath;
use imgui::{self, im_str};
use specs::prelude::*;
use std::{mem::size_of, os::raw::c_uchar, path::PathBuf, ptr, slice::from_raw_parts, sync::Arc};
use winit;

use self::{
    device::{
        Buffer, CommandPool, DescriptorPool, DescriptorSet, DescriptorSetLayout, Device,
        DoubleBuffered, Image,
    },
    helpers::*,
    instance::Instance,
    systems::{
        consolidate_mesh_buffers::ConsolidatedMeshBuffers, cull_pipeline::CullPassData,
        textures::BaseColorDescriptorSet,
    },
};

pub use self::{
    gltf_mesh::{load as load_gltf, LoadedMesh},
    systems::{
        consolidate_mesh_buffers::ConsolidateMeshBuffers,
        cull_pipeline::{
            AssignBufferIndex, CoarseCulled, CoarseCulling, CullPass, GltfMeshBufferIndex,
        },
        present::{AcquireFramebuffer, PresentData, PresentFramebuffer},
        textures::{GltfMeshBaseColorTexture, SynchronizeBaseColorTextures},
    },
};

#[cfg(not(feature = "renderdoc"))]
pub use self::systems::textures::VisitedMarker as BaseColorVisitedMarker;

// TODO: rename
pub struct RenderFrame {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub swapchain: Swapchain,
    pub framebuffer: Framebuffer,
    pub graphics_command_pool: Arc<CommandPool>,
    pub compute_command_pool: Arc<CommandPool>,
    pub descriptor_pool: Arc<DescriptorPool>,
    pub renderpass: RenderPass,
    pub depth_pipeline: Pipeline,
    pub depth_pipeline_layout: PipelineLayout,
    pub gltf_pipeline: Pipeline,
    pub gltf_pipeline_layout: PipelineLayout,
    pub mvp_set_layout: DescriptorSetLayout,
    pub mvp_set: DoubleBuffered<DescriptorSet>,
    pub mvp_buffer: DoubleBuffered<Buffer>,
    pub mesh_assembly_set_layout: DescriptorSetLayout,
    pub base_color_descriptor_set_layout: DescriptorSetLayout,
}

impl RenderFrame {
    pub fn new() -> (RenderFrame, winit::EventsLoop) {
        let (instance, events_loop) = Instance::new(1920, 1080).expect("Failed to create instance");
        let instance = Arc::new(instance);
        let device = Arc::new(Device::new(&instance).expect("Failed to create device"));
        device.set_object_name(device.handle(), "Device");
        let swapchain = new_swapchain(&instance, &device);
        let graphics_command_pool = device.new_command_pool(
            device.graphics_queue_family,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        );
        let compute_command_pool = device.new_command_pool(
            device.compute_queue_family,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        );
        let main_renderpass = RenderFrame::setup_renderpass(Arc::clone(&device), &swapchain)
            .expect("Failed to create renderpass");
        let framebuffer =
            setup_framebuffer(&instance, Arc::clone(&device), &swapchain, &main_renderpass);

        let descriptor_pool = Arc::new(device.new_descriptor_pool(
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

        let mvp_set_layout = device.new_descriptor_set_layout(&[vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: ptr::null(),
        }]);
        device.set_object_name(mvp_set_layout.handle, "MVP Set Layout");

        let mesh_assembly_set_layout = device.new_descriptor_set_layout(&[
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
            },
        ]);
        device.set_object_name(mesh_assembly_set_layout.handle, "Mesh Assembly Layout");

        let base_color_descriptor_set_layout = {
            let binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
                .binding_flags(&[#[cfg(not(feature = "renderdoc"))]
                vk::DescriptorBindingFlagsEXT::PARTIALLY_BOUND]);
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&[vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 3072,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    p_immutable_samplers: ptr::null(),
                }])
                .next(&*binding_flags);

            device.new_descriptor_set_layout2(&create_info)
        };

        device.set_object_name(
            base_color_descriptor_set_layout.handle,
            "Base Color Consolidated Descriptor Set Layout",
        );

        let gltf_pipeline_layout = new_pipeline_layout(
            Arc::clone(&device),
            #[cfg(feature = "renderdoc")]
            {
                &[&mvp_set_layout]
            },
            #[cfg(not(feature = "renderdoc"))]
            {
                &[&mvp_set_layout, &base_color_descriptor_set_layout]
            },
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
                        .depth_bounds_test_enable(true)
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
                .render_pass(main_renderpass.handle)
                .subpass(1)
                .build(),
        );
        device.set_object_name(gltf_pipeline.handle, "GLTF Pipeline");
        let depth_pipeline_layout =
            new_pipeline_layout(Arc::clone(&device), &[&mvp_set_layout], &[]);
        let depth_pipeline = new_graphics_pipeline2(
            Arc::clone(&device),
            &[(
                vk::ShaderStageFlags::VERTEX,
                PathBuf::from(env!("OUT_DIR")).join("depth_prepass.vert.spv"),
            )],
            vk::GraphicsPipelineCreateInfo::builder()
                .vertex_input_state(
                    &vk::PipelineVertexInputStateCreateInfo::builder()
                        .vertex_attribute_descriptions(&[vk::VertexInputAttributeDescription {
                            location: 0,
                            binding: 0,
                            format: vk::Format::R32G32B32_SFLOAT,
                            offset: 0,
                        }])
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
                        .depth_bounds_test_enable(true)
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
                .render_pass(main_renderpass.handle)
                .subpass(0)
                .build(),
        );

        device.set_object_name(depth_pipeline.handle, "Depth Pipeline");

        let mvp_buffer = DoubleBuffered::new(
            |ix| {
                let b = device.new_buffer(
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    4 * 4 * 4 * 4096,
                );
                device.set_object_name(b.handle, &format!("MVP Buffer - {}", ix));
                b
            },
            framebuffer.image_views.len() as u8,
        );
        let mvp_set = DoubleBuffered::new(
            |ix| {
                let s = descriptor_pool.allocate_set(&mvp_set_layout);
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
            },
            framebuffer.image_views.len() as u8,
        );

        (
            RenderFrame {
                instance: Arc::clone(&instance),
                device: Arc::clone(&device),
                framebuffer,
                depth_pipeline,
                depth_pipeline_layout,
                gltf_pipeline,
                gltf_pipeline_layout,
                graphics_command_pool: Arc::new(graphics_command_pool),
                compute_command_pool: Arc::new(compute_command_pool),
                descriptor_pool,
                mvp_set_layout,
                mvp_set,
                mvp_buffer,
                renderpass: main_renderpass,
                swapchain,
                mesh_assembly_set_layout,
                base_color_descriptor_set_layout,
            },
            events_loop,
        )
    }

    fn setup_renderpass(device: Arc<Device>, swapchain: &Swapchain) -> VkResult<RenderPass> {
        let attachment_descriptions = [
            vk::AttachmentDescription {
                format: swapchain.surface_format.format,
                flags: vk::AttachmentDescriptionFlags::empty(),
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            },
            vk::AttachmentDescription {
                format: vk::Format::D16_UNORM,
                flags: vk::AttachmentDescriptionFlags::empty(),
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
        ];
        let color_attachment = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let depth_attachment = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let subpass_descs = [
            vk::SubpassDescription {
                color_attachment_count: 0,
                p_color_attachments: ptr::null(),
                p_depth_stencil_attachment: &depth_attachment,
                flags: Default::default(),
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                input_attachment_count: 0,
                p_input_attachments: ptr::null(),
                p_resolve_attachments: ptr::null(),
                preserve_attachment_count: 0,
                p_preserve_attachments: ptr::null(),
            },
            vk::SubpassDescription {
                color_attachment_count: 1,
                p_color_attachments: &color_attachment,
                p_depth_stencil_attachment: &depth_attachment,
                flags: Default::default(),
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                input_attachment_count: 0,
                p_input_attachments: ptr::null(),
                p_resolve_attachments: ptr::null(),
                preserve_attachment_count: 0,
                p_preserve_attachments: ptr::null(),
            },
        ];
        let subpass_dependencies = [
            vk::SubpassDependency {
                dependency_flags: Default::default(),
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                src_access_mask: Default::default(),
                dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            },
            vk::SubpassDependency {
                dependency_flags: Default::default(),
                src_subpass: 0,
                dst_subpass: 1,
                src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            },
        ];

        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descriptions)
            .subpasses(&subpass_descs)
            .dependencies(&subpass_dependencies);
        let renderpass = unsafe {
            device
                .device
                .create_render_pass(&renderpass_create_info, None)
        };

        renderpass.map(|handle| RenderPass { handle, device })
    }

    pub fn new_buffered<T, F: FnMut(u32) -> T>(&self, creator: F) -> DoubleBuffered<T> {
        DoubleBuffered::new(creator, self.framebuffer.image_views.len() as u8)
    }
}

pub struct Renderer;

impl<'a> System<'a> for Renderer {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        ReadExpect<'a, RenderFrame>,
        WriteExpect<'a, Gui>,
        ReadStorage<'a, GltfMeshBufferIndex>,
        ReadExpect<'a, BaseColorDescriptorSet>,
        ReadExpect<'a, ConsolidatedMeshBuffers>,
        ReadExpect<'a, CullPassData>,
        WriteExpect<'a, PresentData>,
    );

    fn run(
        &mut self,
        (
            renderer,
            mut gui,
            coarse_culled,
            base_color_descriptor_set,
            consolidated_mesh_buffers,
            cull_pass_data,
            mut present_data,
        ): Self::SystemData,
    ) {
        let total = coarse_culled.join().count() as u32;
        let command_buffer = renderer.graphics_command_pool.record_one_time({
            let renderer = &renderer;
            let consolidated_mesh_buffers = &consolidated_mesh_buffers;
            let present_data = &present_data;
            let cull_pass_data = &cull_pass_data;
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
                    .framebuffer(renderer.framebuffer.handles[present_data.image_index as usize])
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
                            "depth prepass",
                            [0.3, 0.3, 0.3, 1.0],
                            || {
                                renderer.device.cmd_bind_pipeline(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    renderer.depth_pipeline.handle,
                                );
                                renderer.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    renderer.depth_pipeline_layout.handle,
                                    0,
                                    &[renderer.mvp_set.current(present_data.image_index).handle],
                                    &[],
                                );
                                renderer.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    cull_pass_data
                                        .culled_index_buffer
                                        .current(present_data.image_index)
                                        .handle,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                renderer.device.cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &[consolidated_mesh_buffers.position_buffer.handle],
                                    &[0],
                                );
                                renderer.device.cmd_draw_indexed_indirect(
                                    command_buffer,
                                    cull_pass_data
                                        .culled_commands_buffer
                                        .current(present_data.image_index)
                                        .handle,
                                    0,
                                    total,
                                    size_of::<u32>() as u32 * 5,
                                );
                                renderer
                                    .device
                                    .cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);
                            },
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
                                    renderer.gltf_pipeline.handle,
                                );
                                renderer.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    renderer.gltf_pipeline_layout.handle,
                                    0,
                                    &[
                                        renderer.mvp_set.current(present_data.image_index).handle,
                                        #[cfg(not(feature = "renderdoc"))]
                                        base_color_descriptor_set
                                            .set
                                            .current(present_data.image_index)
                                            .handle,
                                    ],
                                    &[],
                                );
                                renderer.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    cull_pass_data
                                        .culled_index_buffer
                                        .current(present_data.image_index)
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
                                        .current(present_data.image_index)
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
            present_data.present_semaphore.handle,
            cull_pass_data
                .cull_complete_semaphore
                .current(present_data.image_index)
                .handle,
        ];
        if let Some(ref semaphore) = consolidated_mesh_buffers.sync_point {
            wait_semaphores.push(semaphore.handle);
        }
        let signal_semaphores = &[present_data.render_complete_semaphore.handle];
        let dst_stage_masks = &[
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ];
        let command_buffers = &[*command_buffer];
        let submit = vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(dst_stage_masks)
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
                        .current(present_data.image_index)
                        .handle,
                )
                .unwrap();
        }

        let ix = present_data.image_index;
        *present_data.render_command_buffer.current_mut(ix) = Some(command_buffer);
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

impl Gui {
    pub fn new(renderer: &RenderFrame) -> Gui {
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

        let descriptor_set = renderer
            .descriptor_pool
            .allocate_set(&descriptor_set_layout);

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
                .subpass(1)
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
    }
}

pub struct MVPUpload;

impl<'a> System<'a> for MVPUpload {
    type SystemData = (
        ReadStorage<'a, Matrices>,
        ReadStorage<'a, GltfMeshBufferIndex>,
        ReadExpect<'a, PresentData>,
        WriteExpect<'a, RenderFrame>,
    );

    fn run(&mut self, (matrices, indices, present_data, mut renderer): Self::SystemData) {
        let mut mvp_mapped = renderer
            .mvp_buffer
            .current_mut(present_data.image_index)
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
    #[cfg(not(feature = "renderdoc"))]
    world.register::<BaseColorVisitedMarker>();
    world.register::<CoarseCulled>();
}
