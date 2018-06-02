use super::alloc;
use super::components::*;
use ash::{version::DeviceV1_0, vk};
use cgmath;
use futures::{
    executor::{self, block_on, spawn, ThreadPool}, future::lazy,
};
use specs::prelude::*;
use std::{cmp::min, mem::size_of, path::PathBuf, ptr, slice::from_raw_parts, sync::Arc, u64};
use winit;

use super::helpers::*;

pub struct RenderFrame {
    pub threadpool: executor::ThreadPool,
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub swapchain: Arc<Swapchain>,
    pub framebuffer: Arc<Framebuffer>,
    pub present_semaphore: Arc<Semaphore>,
    pub rendering_complete_semaphore: Arc<Semaphore>,
    pub graphics_command_pool: Arc<CommandPool>,
    pub renderpass: Arc<RenderPass>,
    pub depth_pipeline: Arc<Pipeline>,
    pub depth_pipeline_layout: Arc<PipelineLayout>,
    pub gltf_pipeline: Arc<Pipeline>,
    pub gltf_pipeline_layout: Arc<PipelineLayout>,
    pub ubo_set: Arc<DescriptorSet>,
    pub ubo_buffer: Arc<Buffer>,
    pub model_set: Arc<DescriptorSet>,
    pub model_buffer: Arc<Buffer>,
    pub culled_commands_buffer: Arc<Buffer>,
    pub culled_index_buffer: Option<Arc<Buffer>>,
    pub cull_pipeline: Arc<Pipeline>,
    pub cull_pipeline_layout: Arc<PipelineLayout>,
    pub cull_set: Arc<DescriptorSet>,
}

impl RenderFrame {
    pub fn new() -> (RenderFrame, winit::EventsLoop) {
        let (instance, events_loop) = new_window(1920, 1080);
        let device = new_device(&instance);
        let swapchain = new_swapchain(&instance, &device);
        let present_semaphore = new_semaphore(Arc::clone(&device));
        let rendering_complete_semaphore = new_semaphore(Arc::clone(&device));
        let graphics_command_pool = new_command_pool(
            Arc::clone(&device),
            device.graphics_queue_family,
            vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        );
        let main_renderpass = RenderFrame::setup_renderpass(Arc::clone(&device), &swapchain);
        let framebuffer =
            setup_framebuffer(&instance, Arc::clone(&device), &swapchain, &main_renderpass);

        let descriptor_pool = new_descriptor_pool(
            Arc::clone(&device),
            30,
            &[
                vk::DescriptorPoolSize {
                    typ: vk::DescriptorType::UniformBuffer,
                    descriptor_count: 1024,
                },
                vk::DescriptorPoolSize {
                    typ: vk::DescriptorType::StorageBuffer,
                    descriptor_count: 1024,
                },
            ],
        );

        let command_generation_descriptor_set_layout = new_descriptor_set_layout(
            Arc::clone(&device),
            &[
                vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::StorageBuffer,
                    descriptor_count: 1,
                    stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                    p_immutable_samplers: ptr::null(),
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::StorageBuffer,
                    descriptor_count: 1,
                    stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                    p_immutable_samplers: ptr::null(),
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 2,
                    descriptor_type: vk::DescriptorType::StorageBuffer,
                    descriptor_count: 1,
                    stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                    p_immutable_samplers: ptr::null(),
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 3,
                    descriptor_type: vk::DescriptorType::StorageBuffer,
                    descriptor_count: 1,
                    stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                    p_immutable_samplers: ptr::null(),
                },
            ],
        );
        let descriptor_set_layout = new_descriptor_set_layout(
            Arc::clone(&device),
            &[vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_VERTEX_BIT | vk::SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            }],
        );
        let model_view_set_layout = new_descriptor_set_layout(
            Arc::clone(&device),
            &[vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            }],
        );

        let model_set_layout = new_descriptor_set_layout(
            Arc::clone(&device),
            &[vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_VERTEX_BIT,
                p_immutable_samplers: ptr::null(),
            }],
        );

        #[repr(C)]
        struct MeshData {
            entity_id: u32,
            gltf_id: u32,
            index_count: u32,
            index_offset: u32,
            vertex_offset: u32,
        }

        let command_generation_pipeline_layout = new_pipeline_layout(
            Arc::clone(&device),
            &[
                descriptor_set_layout.handle,
                command_generation_descriptor_set_layout.handle,
            ],
            &[vk::PushConstantRange {
                stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                offset: 0,
                size: size_of::<MeshData>() as u32,
            }],
        );
        let command_generation_pipeline = new_compute_pipeline(
            Arc::clone(&device),
            &command_generation_pipeline_layout,
            &PathBuf::from(env!("OUT_DIR")).join("generate_work.comp.spv"),
        );

        let command_generation_descriptor_set = new_descriptor_set(
            Arc::clone(&device),
            Arc::clone(&descriptor_pool),
            &command_generation_descriptor_set_layout,
        );
        let command_generation_buffer = new_buffer(
            Arc::clone(&device),
            vk::BUFFER_USAGE_INDIRECT_BUFFER_BIT | vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
            alloc::VmaAllocationCreateFlagBits(0),
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            size_of::<u32>() as vk::DeviceSize * 5 * 900,
        );

        let gltf_pipeline_layout = new_pipeline_layout(
            Arc::clone(&device),
            &[descriptor_set_layout.handle, model_set_layout.handle],
            &[],
        );
        let gltf_pipeline = new_graphics_pipeline(
            &instance,
            Arc::clone(&device),
            &gltf_pipeline_layout,
            &main_renderpass,
            &[
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32g32b32Sfloat,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 1,
                    format: vk::Format::R32g32b32Sfloat,
                    offset: 0,
                },
            ],
            &[
                vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: size_of::<f32>() as u32 * 3,
                    input_rate: vk::VertexInputRate::Vertex,
                },
                vk::VertexInputBindingDescription {
                    binding: 1,
                    stride: size_of::<f32>() as u32 * 3,
                    input_rate: vk::VertexInputRate::Vertex,
                },
            ],
            &[
                (
                    vk::SHADER_STAGE_VERTEX_BIT,
                    PathBuf::from(env!("OUT_DIR")).join("gltf_mesh.vert.spv"),
                ),
                (
                    vk::SHADER_STAGE_FRAGMENT_BIT,
                    PathBuf::from(env!("OUT_DIR")).join("gltf_mesh.frag.spv"),
                ),
            ],
            1,
            false,
            false,
        );
        let depth_pipeline_layout =
            new_pipeline_layout(Arc::clone(&device), &[descriptor_set_layout.handle], &[]);
        let depth_pipeline = new_graphics_pipeline(
            &instance,
            Arc::clone(&device),
            &depth_pipeline_layout,
            &main_renderpass,
            &[vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32g32b32Sfloat,
                offset: 0,
            }],
            &[vk::VertexInputBindingDescription {
                binding: 0,
                stride: size_of::<f32>() as u32 * 3,
                input_rate: vk::VertexInputRate::Vertex,
            }],
            &[(
                vk::SHADER_STAGE_VERTEX_BIT,
                PathBuf::from(env!("OUT_DIR")).join("depth_prepass.vert.spv"),
            )],
            0,
            true,
            true,
        );
        let ubo_set = new_descriptor_set(
            Arc::clone(&device),
            Arc::clone(&descriptor_pool),
            &descriptor_set_layout,
        );
        let ubo_buffer = new_buffer(
            Arc::clone(&device),
            vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            4 * 4 * 4 * 1024,
        );
        {
            let buffer_updates = &[vk::DescriptorBufferInfo {
                buffer: ubo_buffer.handle,
                offset: 0,
                range: 1024 * size_of::<cgmath::Matrix4<f32>>() as vk::DeviceSize,
            }];
            unsafe {
                device.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WriteDescriptorSet,
                        p_next: ptr::null(),
                        dst_set: ubo_set.handle,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UniformBuffer,
                        p_image_info: ptr::null(),
                        p_buffer_info: buffer_updates.as_ptr(),
                        p_texel_buffer_view: ptr::null(),
                    }],
                    &[],
                );
            }
        }
        let model_view_set = new_descriptor_set(
            Arc::clone(&device),
            Arc::clone(&descriptor_pool),
            &model_view_set_layout,
        );
        let model_view_buffer = new_buffer(
            Arc::clone(&device),
            vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            4 * 4 * 4 * 1024,
        );
        {
            let buffer_updates = &[vk::DescriptorBufferInfo {
                buffer: model_view_buffer.handle,
                offset: 0,
                range: 1024 * size_of::<cgmath::Matrix4<f32>>() as vk::DeviceSize,
            }];
            unsafe {
                device.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WriteDescriptorSet,
                        p_next: ptr::null(),
                        dst_set: model_view_set.handle,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UniformBuffer,
                        p_image_info: ptr::null(),
                        p_buffer_info: buffer_updates.as_ptr(),
                        p_texel_buffer_view: ptr::null(),
                    }],
                    &[],
                );
            }
        }

        let model_set = new_descriptor_set(
            Arc::clone(&device),
            Arc::clone(&descriptor_pool),
            &model_view_set_layout,
        );
        let model_buffer = new_buffer(
            Arc::clone(&device),
            vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            4 * 4 * 4 * 1024,
        );
        {
            let buffer_updates = &[vk::DescriptorBufferInfo {
                buffer: model_buffer.handle,
                offset: 0,
                range: 1024 * size_of::<cgmath::Matrix4<f32>>() as vk::DeviceSize,
            }];
            unsafe {
                device.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WriteDescriptorSet,
                        p_next: ptr::null(),
                        dst_set: model_set.handle,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UniformBuffer,
                        p_image_info: ptr::null(),
                        p_buffer_info: buffer_updates.as_ptr(),
                        p_texel_buffer_view: ptr::null(),
                    }],
                    &[],
                );
            }
        }

        let threadpool = ThreadPool::builder().pool_size(4).create().unwrap();
        (
            RenderFrame {
                threadpool: threadpool,
                instance: Arc::clone(&instance),
                device: Arc::clone(&device),
                framebuffer: Arc::clone(&framebuffer),
                depth_pipeline: Arc::clone(&depth_pipeline),
                depth_pipeline_layout: Arc::clone(&depth_pipeline_layout),
                gltf_pipeline: Arc::clone(&gltf_pipeline),
                gltf_pipeline_layout: Arc::clone(&gltf_pipeline_layout),
                graphics_command_pool: Arc::clone(&graphics_command_pool),
                ubo_set: Arc::clone(&ubo_set),
                ubo_buffer,
                model_set: Arc::clone(&model_set),
                model_buffer,
                present_semaphore: Arc::clone(&present_semaphore),
                rendering_complete_semaphore: Arc::clone(&rendering_complete_semaphore),
                renderpass: Arc::clone(&main_renderpass),
                swapchain: Arc::clone(&swapchain),
                culled_commands_buffer: Arc::clone(&command_generation_buffer),
                culled_index_buffer: None,
                cull_pipeline: Arc::clone(&command_generation_pipeline),
                cull_pipeline_layout: Arc::clone(&command_generation_pipeline_layout),
                cull_set: Arc::clone(&command_generation_descriptor_set),
            },
            events_loop,
        )
    }

    fn setup_renderpass(device: Arc<Device>, swapchain: &Swapchain) -> Arc<RenderPass> {
        let attachment_descriptions = [
            vk::AttachmentDescription {
                format: swapchain.surface_format.format,
                flags: vk::AttachmentDescriptionFlags::empty(),
                samples: vk::SAMPLE_COUNT_1_BIT,
                load_op: vk::AttachmentLoadOp::Clear,
                store_op: vk::AttachmentStoreOp::Store,
                stencil_load_op: vk::AttachmentLoadOp::DontCare,
                stencil_store_op: vk::AttachmentStoreOp::DontCare,
                initial_layout: vk::ImageLayout::Undefined,
                final_layout: vk::ImageLayout::PresentSrcKhr,
            },
            vk::AttachmentDescription {
                format: vk::Format::D16Unorm,
                flags: vk::AttachmentDescriptionFlags::empty(),
                samples: vk::SAMPLE_COUNT_1_BIT,
                load_op: vk::AttachmentLoadOp::Clear,
                store_op: vk::AttachmentStoreOp::Store,
                stencil_load_op: vk::AttachmentLoadOp::DontCare,
                stencil_store_op: vk::AttachmentStoreOp::DontCare,
                initial_layout: vk::ImageLayout::Undefined,
                final_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
            },
        ];
        let color_attachment = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::ColorAttachmentOptimal,
        };
        let depth_attachment = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
        };
        let subpass_descs = [
            vk::SubpassDescription {
                color_attachment_count: 0,
                p_color_attachments: ptr::null(),
                p_depth_stencil_attachment: &depth_attachment,
                flags: Default::default(),
                pipeline_bind_point: vk::PipelineBindPoint::Graphics,
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
                pipeline_bind_point: vk::PipelineBindPoint::Graphics,
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
                src_subpass: vk::VK_SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                src_access_mask: Default::default(),
                dst_access_mask: vk::ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                    | vk::ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                dst_stage_mask: vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            },
            vk::SubpassDependency {
                dependency_flags: Default::default(),
                src_subpass: 0,
                dst_subpass: 1,
                src_stage_mask: vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                src_access_mask: Default::default(),
                dst_access_mask: vk::ACCESS_COLOR_ATTACHMENT_READ_BIT
                    | vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            },
        ];
        new_renderpass(
            device,
            &attachment_descriptions,
            &subpass_descs,
            &subpass_dependencies,
        )
    }
}

impl<'a> System<'a> for RenderFrame {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, GltfMesh>,
        ReadStorage<'a, GltfMeshBufferIndex>,
    );

    fn run(&mut self, (entities, meshes, mesh_indices): Self::SystemData) {
        let image_index = unsafe {
            self.swapchain
                .handle
                .ext
                .acquire_next_image_khr(
                    self.swapchain.handle.swapchain,
                    u64::MAX,
                    self.present_semaphore.handle,
                    vk::Fence::null(),
                )
                .unwrap()
        };
        let command_buffer_future = record_one_time_cb(Arc::clone(&self.graphics_command_pool), {
            let main_renderpass = Arc::clone(&self.renderpass);
            let framebuffer = Arc::clone(&self.framebuffer);
            let instance = Arc::clone(&self.instance);
            let device = Arc::clone(&self.device);
            let ubo_set = Arc::clone(&self.ubo_set);
            let model_set = Arc::clone(&self.model_set);
            let depth_pipeline = Arc::clone(&self.depth_pipeline);
            let depth_pipeline_layout = Arc::clone(&self.depth_pipeline_layout);
            let gltf_pipeline = Arc::clone(&self.gltf_pipeline);
            let gltf_pipeline_layout = Arc::clone(&self.gltf_pipeline_layout);
            let cull_set = Arc::clone(&self.cull_set);
            let culled_index_buffer = Arc::clone(self.culled_index_buffer.as_ref().unwrap());
            let culled_commands_buffer = Arc::clone(&self.culled_commands_buffer);
            let cull_pipeline = Arc::clone(&self.cull_pipeline);
            let cull_pipeline_layout = Arc::clone(&self.cull_pipeline_layout);
            move |command_buffer| unsafe {
                device.device.debug_marker_around(
                    command_buffer,
                    "cull pass",
                    [0.0, 1.0, 0.0, 1.0],
                    || {
                        device.device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::Compute,
                            cull_pipeline.handle,
                        );
                        device.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::Compute,
                            cull_pipeline_layout.handle,
                            0,
                            &[ubo_set.handle, cull_set.handle],
                            &[],
                        );
                        device.device.cmd_fill_buffer(
                            command_buffer,
                            culled_commands_buffer.handle,
                            0,
                            size_of::<u32>() as vk::DeviceSize
                                + size_of::<u32>() as vk::DeviceSize * 5 * 900,
                            0,
                        );
                        device.device.cmd_pipeline_barrier(
                            command_buffer,
                            vk::PIPELINE_STAGE_TRANSFER_BIT,
                            vk::PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk::DependencyFlags::empty(),
                            &[vk::MemoryBarrier {
                                s_type: vk::StructureType::MemoryBarrier,
                                p_next: ptr::null(),
                                src_access_mask: vk::ACCESS_TRANSFER_WRITE_BIT,
                                dst_access_mask: vk::ACCESS_SHADER_READ_BIT
                                    | vk::ACCESS_SHADER_WRITE_BIT,
                            }],
                            &[],
                            &[],
                        );
                        let mut index_offset = 0;
                        for (entity, mesh, mesh_index) in
                            (&*entities, &meshes, &mesh_indices).join()
                        {
                            let constants = [
                                entity.id() as u32,
                                mesh_index.0,
                                mesh.index_len as u32,
                                index_offset,
                                0,
                            ];
                            index_offset += mesh.index_len as u32;

                            let casted: &[u8] = {
                                from_raw_parts(constants.as_ptr() as *const u8, constants.len() * 4)
                            };
                            device.device.cmd_push_constants(
                                command_buffer,
                                cull_pipeline_layout.handle,
                                vk::SHADER_STAGE_COMPUTE_BIT,
                                0,
                                casted,
                            );
                            let index_len = mesh.index_len as u32;
                            let workgroup_size = 64; // TODO: make a specialization constant, not hardcoded
                            let workgroup_count = index_len / 3 / workgroup_size
                                + min(1, index_len / 3 % workgroup_size);
                            device
                                .device
                                .cmd_dispatch(command_buffer, workgroup_count, 1, 1);
                        }
                        device.device.cmd_pipeline_barrier(
                            command_buffer,
                            vk::PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk::PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                            vk::DependencyFlags::empty(),
                            &[vk::MemoryBarrier {
                                s_type: vk::StructureType::MemoryBarrier,
                                p_next: ptr::null(),
                                src_access_mask: vk::ACCESS_SHADER_WRITE_BIT,
                                dst_access_mask: vk::ACCESS_INDIRECT_COMMAND_READ_BIT,
                            }],
                            &[],
                            &[],
                        );
                    },
                );

                let clear_values = &[
                    vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    },
                    vk::ClearValue {
                        depth: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ];
                let begin_info = vk::RenderPassBeginInfo {
                    s_type: vk::StructureType::RenderPassBeginInfo,
                    p_next: ptr::null(),
                    render_pass: main_renderpass.handle,
                    framebuffer: framebuffer.handles[image_index as usize],
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: instance.window_width,
                            height: instance.window_height,
                        },
                    },
                    clear_value_count: clear_values.len() as u32,
                    p_clear_values: clear_values.as_ptr(),
                };

                device.device.debug_marker_around(
                    command_buffer,
                    "main renderpass",
                    [0.0, 0.0, 1.0, 1.0],
                    || {
                        device.device.cmd_begin_render_pass(
                            command_buffer,
                            &begin_info,
                            vk::SubpassContents::Inline,
                        );
                        device.device.debug_marker_around(
                            command_buffer,
                            "depth prepass",
                            [0.3, 0.3, 0.3, 1.0],
                            || {
                                device.device.cmd_bind_pipeline(
                                    command_buffer,
                                    vk::PipelineBindPoint::Graphics,
                                    depth_pipeline.handle,
                                );
                                device.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::Graphics,
                                    depth_pipeline_layout.handle,
                                    0,
                                    &[ubo_set.handle],
                                    &[],
                                );
                                device.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    culled_index_buffer.handle,
                                    0,
                                    vk::IndexType::Uint32,
                                );
                                let first_entity = (&*entities).join().next().unwrap();
                                let mesh = meshes.get(first_entity).unwrap();
                                device.device.cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &[mesh.vertex_buffer.handle],
                                    &[0],
                                );
                                device.device.cmd_draw_indexed_indirect(
                                    command_buffer,
                                    culled_commands_buffer.handle,
                                    0,
                                    900, // TODO: find max of GltfMeshBufferIndex
                                    size_of::<u32>() as u32 * 5,
                                );
                                device
                                    .device
                                    .cmd_next_subpass(command_buffer, vk::SubpassContents::Inline);
                            },
                        );
                        device.device.debug_marker_around(
                            command_buffer,
                            "gltf meshes",
                            [1.0, 0.0, 0.0, 1.0],
                            || {
                                // gltf mesh
                                device.device.cmd_bind_pipeline(
                                    command_buffer,
                                    vk::PipelineBindPoint::Graphics,
                                    gltf_pipeline.handle,
                                );
                                device.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::Graphics,
                                    gltf_pipeline_layout.handle,
                                    0,
                                    &[ubo_set.handle, model_set.handle],
                                    &[],
                                );
                                device.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    culled_index_buffer.handle,
                                    0,
                                    vk::IndexType::Uint32,
                                );
                                let first_entity = (&*entities).join().next().unwrap();
                                let mesh = meshes.get(first_entity).unwrap();
                                device.device.cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &[mesh.vertex_buffer.handle, mesh.normal_buffer.handle],
                                    &[0, 0],
                                );
                                device.device.cmd_draw_indexed_indirect(
                                    command_buffer,
                                    culled_commands_buffer.handle,
                                    0,
                                    900, // TODO: find max of GltfMeshBufferIndex
                                    size_of::<u32>() as u32 * 5,
                                );
                            },
                        );
                        device.device.cmd_end_render_pass(command_buffer);
                    },
                );
                device.device.debug_marker_end(command_buffer);
            }
        });
        let command_buffer = block_on(command_buffer_future).unwrap();
        unsafe {
            let wait_semaphores = &[self.present_semaphore.handle];
            let signal_semaphores = &[self.rendering_complete_semaphore.handle];
            let dst_stage_masks = vec![vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT; wait_semaphores.len()];
            let submits = [vk::SubmitInfo {
                s_type: vk::StructureType::SubmitInfo,
                p_next: ptr::null(),
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: dst_stage_masks.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &command_buffer.handle,
                signal_semaphore_count: signal_semaphores.len() as u32,
                p_signal_semaphores: signal_semaphores.as_ptr(),
            }];
            let queue = self
                .device
                .graphics_queue
                .lock()
                .expect("can't lock the submit queue");

            let submit_fence = new_fence(Arc::clone(&self.device));

            self.device
                .device
                .queue_submit(*queue, &submits, submit_fence.handle)
                .unwrap();

            {
                let device = Arc::clone(&self.device);
                self.threadpool
                    .run(lazy(move |_| {
                        spawn(lazy(move |_| {
                            // println!("dtor previous frame");
                            device
                                .device
                                .wait_for_fences(&[submit_fence.handle], true, u64::MAX)
                                .expect("Wait for fence failed.");
                            drop(command_buffer);
                            drop(submit_fence);
                            Ok(())
                        }))
                    }))
                    .unwrap();
            }

            {
                let wait_semaphores = &[self.rendering_complete_semaphore.handle];
                let present_info = vk::PresentInfoKHR {
                    s_type: vk::StructureType::PresentInfoKhr,
                    p_next: ptr::null(),
                    wait_semaphore_count: wait_semaphores.len() as u32,
                    p_wait_semaphores: wait_semaphores.as_ptr(),
                    swapchain_count: 1,
                    p_swapchains: &self.swapchain.handle.swapchain,
                    p_image_indices: &image_index,
                    p_results: ptr::null_mut(),
                };

                self.swapchain
                    .handle
                    .ext
                    .queue_present_khr(*queue, &present_info)
                    .unwrap();
            }
        }
    }
}
