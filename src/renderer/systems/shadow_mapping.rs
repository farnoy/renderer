use std::{convert::TryInto, mem::size_of};

use ash::vk;
use bevy_ecs::prelude::*;
use petgraph::graph::NodeIndex;
use profiling::scope;
use static_assertions::const_assert_eq;

#[cfg(feature = "crash_debugging")]
use crate::renderer::CrashBuffer;
use crate::{
    ecs::components::{Light, Position, Rotation},
    renderer::{
        camera_set, device::VmaMemoryUsage, frame_graph, frame_graph::LightMatrices, pick_lod,
        systems::depth_pass::depth_pipe, update_whole_buffer, BufferType, CameraMatrices, Device, DoubleBuffered,
        DrawIndex, GltfMesh, ImageIndex, ImageView, MainDescriptorPool, ModelData, RenderFrame, RenderStage, Sampler,
        SmartPipeline, SmartPipelineLayout, SmartSet, SmartSetLayout, StrictCommandPool, Submissions,
        SwapchainIndexToFrameNumber,
    },
};

pub(crate) const MAP_SIZE: u32 = 4096;
// dimensions of the square texture, 4x4 slots = 16 in total
pub(crate) const DIM: u32 = 4;

renderer_macros::define_resource! { ShadowMapAtlas = Image DEPTH D16_UNORM }

renderer_macros::define_set! {
    shadow_map_set {
        light_data 16 of STORAGE_BUFFER partially bound from [VERTEX, FRAGMENT],
        shadow_maps COMBINED_IMAGE_SAMPLER from [FRAGMENT]
    }
}

renderer_macros::define_pass! { ShadowMapping on graphics }
renderer_macros::define_renderpass! {
    ShadowMappingRP {
        depth_stencil { ShadowMapAtlas DEPTH_STENCIL_ATTACHMENT_OPTIMAL load => store }
    }
}

pub(crate) struct ShadowMappingData {
    depth_pipeline_layout: SmartPipelineLayout<depth_pipe::PipelineLayout>,
    depth_pipeline: SmartPipeline<depth_pipe::Pipeline>,
    pub(crate) depth_image: ShadowMapAtlas,
    depth_image_view: ImageView,
    pub(crate) user_set_layout: SmartSetLayout<shadow_map_set::Layout>,
    pub(crate) user_set: DoubleBuffered<SmartSet<shadow_map_set::Set>>,
    user_sampler: Sampler,
}

pub(crate) struct ShadowMappingDataInternal {
    command_pools: DoubleBuffered<StrictCommandPool>,
    command_buffers: DoubleBuffered<vk::CommandBuffer>,
}

impl ShadowMappingData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        model_data: &ModelData,
        camera_matrices: &CameraMatrices,
        main_descriptor_pool: &mut MainDescriptorPool,
    ) -> ShadowMappingData {
        let depth_pipeline_layout = SmartPipelineLayout::new(
            &renderer.device,
            (&model_data.model_set_layout, &camera_matrices.set_layout),
        );
        let depth_pipeline = SmartPipeline::new(
            &renderer.device,
            &depth_pipeline_layout,
            depth_pipe::Specialization {},
            vk::SampleCountFlags::TYPE_1,
        );

        let depth_image = ShadowMapAtlas::import(
            &renderer.device,
            renderer.device.new_image_exclusive(
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
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            ),
        );

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
                    renderer.device.allocation_callbacks(),
                )
                .unwrap();

            ImageView { handle }
        };

        let mut command_pool = StrictCommandPool::new(
            &renderer.device,
            renderer.device.graphics_queue_family,
            "Quick command pool for ShadowMapping constructor",
        );

        let cb = command_pool.record_one_time(&renderer.device, "transition shadow mapping texture");
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

        let user_set_layout = SmartSetLayout::new(&renderer.device);

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
            let s = SmartSet::new(&renderer.device, main_descriptor_pool, &user_set_layout, ix);

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
            depth_image,
            depth_image_view,
            depth_pipeline_layout,
            depth_pipeline,
            user_set_layout,
            user_set,
            user_sampler,
        }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.depth_pipeline.destroy(device);
        self.depth_pipeline_layout.destroy(device);
        self.user_sampler.destroy(device);
        self.depth_image_view.destroy(device);
        self.depth_image.destroy(device);
        self.user_set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
        self.user_set_layout.destroy(device);
    }
}

impl FromWorld for ShadowMappingDataInternal {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();

        let mut command_pools = renderer.new_buffered(|ix| {
            StrictCommandPool::new(
                &renderer.device,
                renderer.device.graphics_queue_family,
                &format!("Shadow Mapping Command Pool[{}]", ix),
            )
        });
        let command_buffers = renderer.new_buffered(|ix| {
            command_pools
                .current_mut(ix)
                .allocate(&format!("Shadow Mapping CB[{}]", ix), &renderer.device)
        });

        Self {
            command_pools,
            command_buffers,
        }
    }
}

impl ShadowMappingDataInternal {
    pub(crate) fn destroy(self, device: &Device) {
        self.command_pools.into_iter().for_each(|p| p.destroy(device));
    }
}

/// Holds Projection and view matrices for each light.
#[derive(Component)]
pub(crate) struct ShadowMappingLightMatrices {
    matrices_set: DoubleBuffered<SmartSet<camera_set::Set>>,
    matrices_buffer: DoubleBuffered<BufferType<camera_set::bindings::matrices>>,
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
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER,
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
            let mut s = SmartSet::new(&renderer.device, main_descriptor_pool, &camera_matrices.set_layout, ix);
            renderer.device.set_object_name(
                s.set.handle,
                &format!("camera_set Set [{}] - entity={:?}", ix, entity_ix),
            );
            update_whole_buffer::<camera_set::bindings::matrices>(
                &renderer.device,
                &mut s,
                matrices_buffer.current(ix),
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
    mut query: Query<(&Position, &Rotation, &mut ShadowMappingLightMatrices)>,
) {
    const_assert_eq!(size_of::<LightMatrices>(), 208);

    scope!("ecs::shadow_mapping_light_matrices_calculation");

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
        *matrices_mapped = frame_graph::CameraMatrices {
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
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<SwapchainIndexToFrameNumber>,
    shadow_mapping: Res<ShadowMappingData>,
    mut shadow_mapping_internal: ResMut<ShadowMappingDataInternal>,
    model_data: Res<ModelData>,
    submissions: Res<Submissions>,
    mesh_query: Query<(&DrawIndex, &Position, &GltfMesh)>,
    shadow_query: Query<(&Position, &ShadowMappingLightMatrices), With<Light>>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
) {
    scope!("ecs::shadow_mapping");

    if !submissions
        .active_graph
        .contains_node(NodeIndex::from(frame_graph::ShadowMapping::INDEX))
    {
        return;
    }

    frame_graph::ShadowMapping::Stage::wait_previous(&renderer, &image_index, &swapchain_index_map);

    let ShadowMappingDataInternal {
        ref mut command_pools,
        ref command_buffers,
    } = *shadow_mapping_internal;

    let command_pool = command_pools.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let command_buffer = command_pool.record_to_specific(&renderer.device, *command_buffers.current(image_index.0));

    unsafe {
        let _shadow_mapping_marker = command_buffer.debug_marker_around("shadow mapping", [0.8, 0.1, 0.1, 1.0]);
        let _guard = renderer_macros::barrier!(
            command_buffer,
            ShadowMapAtlas.prepare rw in ShadowMapping attachment in ShadowMappingRP; &shadow_mapping.depth_image
        );

        ShadowMappingRP::begin(
            &renderer,
            *command_buffer,
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: DIM * MAP_SIZE,
                    height: DIM * MAP_SIZE,
                },
            },
            &[shadow_mapping.depth_image_view.handle],
            &[],
        );
        renderer.device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            shadow_mapping.depth_pipeline.vk(),
        );

        for (ix, (light_position, shadow_mvp)) in shadow_query.iter().enumerate() {
            shadow_mapping.depth_pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                (
                    model_data.model_set.current(image_index.0),
                    shadow_mvp.matrices_set.current(image_index.0),
                ),
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

        renderer.device.dynamic_rendering.cmd_end_rendering(*command_buffer);
    }
    let command_buffer = command_buffer.end();

    submissions.submit(
        &renderer,
        frame_graph::ShadowMapping::INDEX,
        *command_buffer,
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
    );
}

pub(crate) fn update_shadow_map_descriptors(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<SwapchainIndexToFrameNumber>,
    shadow_mapping: Res<ShadowMappingData>,
    shadow_query: Query<&ShadowMappingLightMatrices, With<Light>>,
) {
    scope!("ecs::update_shadow_map_descriptors");

    frame_graph::Main::Stage::wait_previous(&renderer, &image_index, &swapchain_index_map);

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
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mvp_updates[ix])
                .build(),
        );
    }

    unsafe {
        renderer.device.update_descriptor_sets(&write_descriptors, &[]);
    }
}
