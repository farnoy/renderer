use std::{convert::TryInto, mem::size_of};

use ash::{version::DeviceV1_0, vk};
use bevy_ecs::prelude::*;
use microprofile::scope;
use static_assertions::const_assert_eq;

use crate::{
    ecs::components::{Light, Position, Rotation},
    renderer::{
        device::VmaMemoryUsage, frame_graph, pick_lod, shaders, shaders::LightMatrices, CameraMatrices, DepthPassData,
        Device, DoubleBuffered, DrawIndex, GltfMesh, GraphicsTimeline, Image, ImageIndex, ImageView,
        LocalGraphicsCommandPool, MainDescriptorPool, ModelData, Pipeline, RenderFrame, RenderStage, Sampler,
        ShadowMappingTimeline, StrictCommandPool, SwapchainIndexToFrameNumber,
    },
};

pub(crate) const MAP_SIZE: u32 = 4096;
// dimensions of the square texture, 4x4 slots = 16 in total
pub(crate) const DIM: u32 = 4;

pub(crate) struct ShadowMappingData {
    depth_pipeline: Pipeline,
    renderpass: frame_graph::ShadowMapping::RenderPass,
    depth_image: Image,
    depth_image_view: ImageView,
    framebuffer: frame_graph::ShadowMapping::Framebuffer,
    pub(crate) user_set_layout: shaders::shadow_map_set::Layout,
    pub(crate) user_set: DoubleBuffered<shaders::shadow_map_set::Set>,
    user_sampler: Sampler,
}

impl ShadowMappingData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        depth_pass_data: &DepthPassData,
        main_descriptor_pool: &mut MainDescriptorPool,
    ) -> ShadowMappingData {
        let renderpass = frame_graph::ShadowMapping::RenderPass::new(renderer, ());

        let depth_pipeline = renderer.device.new_graphics_pipeline(
            &[(vk::ShaderStageFlags::VERTEX, shaders::depth_pipe::VERTEX, None)],
            vk::GraphicsPipelineCreateInfo::builder()
                .vertex_input_state(&shaders::depth_pipe::vertex_input_state())
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST),
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
                .layout(*depth_pass_data.depth_pipeline_layout.layout)
                .render_pass(renderpass.renderpass.handle)
                .subpass(0)
                .build(),
        );
        renderer
            .device
            .set_object_name(*depth_pipeline, "Shadow mapping depth Pipeline");

        let depth_image = renderer.device.new_image(
            vk::Format::D16_UNORM,
            // 16 slots in total
            vk::Extent3D {
                height: MAP_SIZE * DIM,
                width: MAP_SIZE * DIM,
                depth: 1,
            },
            vk::SampleCountFlags::TYPE_1,
            vk::ImageTiling::OPTIMAL,
            vk::ImageLayout::PREINITIALIZED,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        );
        renderer
            .device
            .set_object_name(depth_image.handle, "Shadow mapping depth image");

        let depth_image_view = unsafe {
            let handle = renderer
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::D16_UNORM)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(depth_image.handle),
                    None,
                )
                .unwrap();

            ImageView { handle }
        };

        let mut command_pool = StrictCommandPool::new(
            &renderer.device,
            renderer.device.graphics_queue_family,
            "Quick command pool for ShadowMapping constructor",
        );

        let mut session = command_pool.session(&renderer.device);
        let cb = session.record_one_time("transition shadow mapping texture");
        unsafe {
            renderer.device.cmd_pipeline_barrier(
                *cb,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::HOST,
                Default::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .image(depth_image.handle)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .old_layout(vk::ImageLayout::PREINITIALIZED)
                    .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)
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

        let framebuffer = frame_graph::ShadowMapping::Framebuffer::new(
            renderer,
            &renderpass,
            &[depth_image_view.handle],
            (MAP_SIZE * DIM, MAP_SIZE * DIM),
            0,
        );

        let user_set_layout = shaders::shadow_map_set::Layout::new(&renderer.device);

        let user_sampler = renderer.device.new_sampler(
            &vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .compare_enable(true)
                .compare_op(vk::CompareOp::LESS_OR_EQUAL),
        );

        let user_set = renderer.new_buffered(|ix| {
            let s = shaders::shadow_map_set::Set::new(&renderer.device, &main_descriptor_pool, &user_set_layout, ix);

            {
                let sampler_updates = &[vk::DescriptorImageInfo::builder()
                    .image_view(depth_image_view.handle)
                    .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)
                    .sampler(user_sampler.handle)
                    .build()];
                unsafe {
                    renderer.device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::builder()
                            .dst_set(s.set.handle)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(sampler_updates)
                            .build()],
                        &[],
                    );
                }
            }

            s
        });

        ShadowMappingData {
            renderpass,
            depth_image,
            depth_image_view,
            depth_pipeline,
            framebuffer,
            user_set_layout,
            user_set,
            user_sampler,
        }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.depth_pipeline.destroy(device);
        self.renderpass.destroy(device);
        self.user_sampler.destroy(device);
        self.depth_image_view.destroy(device);
        self.depth_image.destroy(device);
        self.user_set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
        self.user_set_layout.destroy(device);
        self.framebuffer.destroy(device);
    }
}

/// Holds Projection and view matrices for each light.
pub(crate) struct ShadowMappingLightMatrices {
    matrices_set: DoubleBuffered<shaders::camera_set::Set>,
    matrices_buffer: DoubleBuffered<shaders::camera_set::bindings::matrices::Buffer>,
}

impl ShadowMappingLightMatrices {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &MainDescriptorPool,
        camera_matrices: &CameraMatrices,
        entity_ix: u32,
    ) -> ShadowMappingLightMatrices {
        let matrices_buffer = renderer.new_buffered(|ix| {
            let b = renderer.device.new_static_buffer(
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );
            renderer.device.set_object_name(
                b.buffer.handle,
                &format!(
                    "Shadow Mapping Light matrices Buffer - entity={:?} ix={}",
                    entity_ix, ix
                ),
            );
            b
        });
        let matrices_set = renderer.new_buffered(|ix| {
            let mut s =
                shaders::camera_set::Set::new(&renderer.device, &main_descriptor_pool, &camera_matrices.set_layout, ix);
            renderer.device.set_object_name(
                s.set.handle,
                &format!("camera_set Set [{}] - entity={:?}", ix, entity_ix),
            );
            shaders::camera_set::bindings::matrices::update_whole_buffer(
                &renderer.device,
                &mut s,
                &matrices_buffer.current(ix),
            );
            s
        });

        ShadowMappingLightMatrices {
            matrices_set,
            matrices_buffer,
        }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.matrices_buffer.into_iter().for_each(|b| b.destroy(device));
        self.matrices_set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
    }
}

pub(crate) fn shadow_mapping_mvp_calculation(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    query: Query<(&Position, &Rotation, &mut ShadowMappingLightMatrices)>,
) {
    const_assert_eq!(size_of::<LightMatrices>(), 208);

    scope!("ecs", "shadow mapping light matrices calculation");

    query.for_each_mut(|(light_position, light_rotation, mut light_matrix)| {
        let near = 10.0;
        let far = 400.0;
        let projection = glm::perspective_lh_zo(1.0, glm::radians(&glm::vec1(70.0)).x, near, far);

        let view =
            glm::translation(&(light_rotation.0 * (-light_position.0.coords))) * light_rotation.0.to_homogeneous();
        let mut matrices_mapped = light_matrix
            .matrices_buffer
            .current_mut(image_index.0)
            .map(&renderer.device)
            .expect("failed to map Light matrices buffer");
        *matrices_mapped = shaders::CameraMatrices {
            projection,
            view,
            position: light_position.0.coords.push(1.0),
            pv: projection * view,
        };
    });
}

/// Identifies stale shadow maps in the atlas and refreshes them
pub(crate) fn prepare_shadow_maps(
    renderer: Res<RenderFrame>,
    depth_pass: Res<DepthPassData>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<SwapchainIndexToFrameNumber>,
    shadow_mapping: Res<ShadowMappingData>,
    model_data: Res<ModelData>,
    mut local_graphics_command_pool: ResMut<LocalGraphicsCommandPool<1>>,
    mesh_query: Query<(&DrawIndex, &Position, &GltfMesh)>,
    shadow_query: Query<(&Position, &ShadowMappingLightMatrices), With<Light>>,
) {
    microprofile::scope!("ecs", "shadow_mapping");

    renderer
        .shadow_mapping_timeline_semaphore
        .wait(
            &renderer.device,
            ShadowMappingTimeline::Prepare.as_of_previous(&image_index, &swapchain_index_map),
        )
        .unwrap();

    let command_pool = local_graphics_command_pool.pools.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let mut command_session = command_pool.session(&renderer.device);

    let command_buffer = command_session.record_one_time("Shadow Mapping CommandBuffer");

    unsafe {
        let _shadow_mapping_marker = command_buffer.debug_marker_around("shadow mapping", [0.8, 0.1, 0.1, 1.0]);

        shadow_mapping.renderpass.begin(
            &renderer,
            &shadow_mapping.framebuffer,
            *command_buffer,
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: DIM * MAP_SIZE,
                    height: DIM * MAP_SIZE,
                },
            },
            &[],
        );
        renderer.device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *shadow_mapping.depth_pipeline,
        );

        for (ix, (light_position, shadow_mvp)) in shadow_query.iter().enumerate() {
            depth_pass.depth_pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                &model_data.model_set.current(image_index.0),
                &shadow_mvp.matrices_set.current(image_index.0),
            );

            let row = ix as u32 / DIM;
            let column = ix as u32 % DIM;
            let x = MAP_SIZE * column;
            let y = MAP_SIZE * row;
            renderer.device.cmd_set_viewport(*command_buffer, 0, &[vk::Viewport {
                x: x as f32,
                y: (MAP_SIZE * (row + 1)) as f32,
                width: MAP_SIZE as f32,
                height: -(MAP_SIZE as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            }]);
            renderer.device.cmd_set_scissor(*command_buffer, 0, &[vk::Rect2D {
                offset: vk::Offset2D {
                    x: x as i32,
                    y: y as i32,
                },
                extent: vk::Extent2D {
                    width: MAP_SIZE,
                    height: MAP_SIZE,
                },
            }]);
            renderer.device.cmd_clear_attachments(
                *command_buffer,
                &[vk::ClearAttachment::builder()
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 1 },
                    })
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .build()],
                &[vk::ClearRect::builder()
                    .rect(vk::Rect2D {
                        offset: vk::Offset2D {
                            x: x as i32,
                            y: y as i32,
                        },
                        extent: vk::Extent2D {
                            width: MAP_SIZE,
                            height: MAP_SIZE,
                        },
                    })
                    .layer_count(1)
                    .base_array_layer(0)
                    .build()],
            );

            for (draw_index, mesh_position, mesh) in &mut mesh_query.iter() {
                let (index_buffer, index_count) = pick_lod(&mesh.index_buffers, light_position.0, mesh_position.0);
                renderer
                    .device
                    .cmd_bind_index_buffer(*command_buffer, index_buffer.handle, 0, vk::IndexType::UINT32);
                renderer
                    .device
                    .cmd_bind_vertex_buffers(*command_buffer, 0, &[mesh.vertex_buffer.handle], &[0]);
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
    let queue = renderer.device.graphics_queue().lock();

    frame_graph::ShadowMapping::Stage::queue_submit(&image_index, &renderer, *queue, &[*command_buffer]).unwrap();
}

pub(crate) fn update_shadow_map_descriptors(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<SwapchainIndexToFrameNumber>,
    shadow_mapping: Res<ShadowMappingData>,
    shadow_query: Query<&ShadowMappingLightMatrices, With<Light>>,
) {
    renderer
        .graphics_timeline_semaphore
        .wait(
            &renderer.device,
            GraphicsTimeline::SceneDraw.as_of_previous(&image_index, &swapchain_index_map),
        )
        .unwrap();

    // Update descriptor sets so that users of lights have the latest info
    // preallocate all required memory so as not to invalidate references during iteration
    let mut mvp_updates = vec![[vk::DescriptorBufferInfo::default(); 1]; DIM as usize * DIM as usize];
    let mut write_descriptors = Vec::with_capacity(DIM as usize * DIM as usize);
    for (ix, shadow_mvp) in shadow_query.iter().enumerate() {
        mvp_updates[ix] = [vk::DescriptorBufferInfo {
            buffer: shadow_mvp.matrices_buffer.current(image_index.0).buffer.handle,
            offset: 0,
            range: size_of::<LightMatrices>() as vk::DeviceSize,
        }];
        write_descriptors.push(
            vk::WriteDescriptorSet::builder()
                .dst_set(shadow_mapping.user_set.current(image_index.0).set.handle)
                .dst_binding(0)
                .dst_array_element(ix as u32)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&mvp_updates[ix])
                .build(),
        );
    }
    {
        microprofile::scope!("shadow mapping", "vkUpdateDescriptorSets");

        unsafe {
            renderer.device.update_descriptor_sets(&write_descriptors, &[]);
        }
    }
}
