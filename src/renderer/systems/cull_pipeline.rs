use std::cmp::min;

use ash::vk::{self, Handle};
use bevy_ecs::prelude::*;
use microprofile::scope;
use num_traits::ToPrimitive;

#[cfg(not(feature = "no_profiling"))]
use crate::renderer::helpers::MP_INDIAN_RED;
#[cfg(feature = "shader_reload")]
use crate::renderer::ReloadedShaders;
use crate::{
    ecs::{
        components::{Position, AABB},
        resources::Camera,
        systems::RuntimeConfiguration,
    },
    renderer::{
        device::{Device, DoubleBuffered, StrictCommandPool, VmaMemoryUsage},
        frame_graph::{self, compact_draw_stream, cull_commands_count_set, cull_set, generate_work},
        helpers::pick_lod,
        systems::{consolidate_mesh_buffers::ConsolidatedMeshBuffers, present::ImageIndex},
        CameraMatrices, ComputeTimeline, CopiedResource, DrawIndex, GltfMesh, LocalTransferCommandPool,
        MainDescriptorPool, ModelData, RenderFrame, RenderStage, Shader, Submissions, SwapchainIndexToFrameNumber,
        TransferTimeline,
    },
};

// Should this entity be discarded when rendering
// Coarse and based on AABB being fully out of the frustum
pub(crate) struct CoarseCulled(pub(crate) bool);

pub(crate) struct CullPassData {
    pub(crate) culled_commands_buffer: frame_graph::resources::indirect_commands_buffer,
    pub(crate) culled_commands_count_buffer: frame_graph::resources::indirect_commands_count,
    pub(crate) culled_index_buffer: cull_set::bindings::out_index_buffer::Buffer,
    compact_draw_stream_layout: compact_draw_stream::PipelineLayout,
    compact_draw_stream_pipeline: compact_draw_stream::Pipeline,
    cull_set_layout: cull_set::Layout,
    cull_count_set_layout: cull_commands_count_set::Layout,
    cull_set: cull_set::Set,
    cull_count_set: cull_commands_count_set::Set,
    // TODO: reorganize this
    bypass_command_buffers: DoubleBuffered<vk::CommandBuffer>,
}

pub(crate) struct CullPassDataPrivate {
    cull_pipeline_layout: generate_work::PipelineLayout,
    cull_pipeline: generate_work::Pipeline,
    command_pools: DoubleBuffered<StrictCommandPool>,
    command_buffers: DoubleBuffered<vk::CommandBuffer>,
    previous_cull_pipeline: DoubleBuffered<Option<generate_work::Pipeline>>,
    cull_shader: Shader,
}

pub(crate) const INITIAL_WORKGROUP_SIZE: u32 = 384;

pub(crate) fn coarse_culling(
    task_pool: Res<bevy_tasks::ComputeTaskPool>,
    camera: Res<Camera>,
    mut query: Query<(&AABB, &mut CoarseCulled)>,
) {
    scope!("ecs", "coarse_culling");

    query.par_for_each_mut(&task_pool, 2, |(aabb, mut coarse_culled)| {
        scope!("parallel", "coarse_culling", MP_INDIAN_RED);
        let mut outside = false;
        'per_plane: for plane in camera.frustum_planes.iter() {
            let e = aabb.0.half_extents().dot(&plane.xyz().abs());

            let s = plane.dot(&aabb.0.center().to_homogeneous());
            if s - e > 0.0 {
                outside = true;
                break 'per_plane;
            }
        }
        coarse_culled.0 = outside;
    });
}

impl CullPassDataPrivate {
    pub(crate) fn new(
        renderer: &RenderFrame,
        cull_pass_data: &CullPassData,
        model_data: &ModelData,
        camera_matrices: &CameraMatrices,
    ) -> CullPassDataPrivate {
        let cull_specialization = generate_work::Specialization {
            local_workgroup_size: INITIAL_WORKGROUP_SIZE,
        };
        let cull_shader = renderer.device.new_shader(generate_work::COMPUTE);
        let cull_pipeline_layout = generate_work::PipelineLayout::new(
            &renderer.device,
            &model_data.model_set_layout,
            &camera_matrices.set_layout,
            &cull_pass_data.cull_set_layout,
        );
        let cull_pipeline =
            generate_work::Pipeline::new(&renderer.device, &cull_pipeline_layout, cull_specialization, [Some(
                &cull_shader,
            )]);
        let mut command_pools = renderer.new_buffered(|ix| {
            StrictCommandPool::new(
                &renderer.device,
                renderer.device.compute_queue_family,
                &format!("Cull pass Command Pool[{}]", ix),
            )
        });
        let command_buffers = renderer.new_buffered(|ix| {
            command_pools
                .current_mut(ix)
                .allocate(&format!("Cull pass CB[{}]", ix), &renderer.device)
        });

        CullPassDataPrivate {
            cull_pipeline_layout,
            cull_pipeline,
            command_pools,
            command_buffers,
            previous_cull_pipeline: renderer.new_buffered(|_| None),
            cull_shader,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.cull_pipeline_layout.destroy(device);
        self.cull_pipeline.destroy(device);
        for pipe in self.previous_cull_pipeline.into_iter().flatten() {
            pipe.destroy(device);
        }
        self.command_pools.into_iter().for_each(|pool| pool.destroy(device));
        self.cull_shader.destroy(device);
    }
}

impl CullPassData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &mut MainDescriptorPool,
        consolidated_mesh_buffers: &ConsolidatedMeshBuffers,
        transfer_command_pool: &mut LocalTransferCommandPool<0>,
    ) -> CullPassData {
        let device = &renderer.device;

        let cull_set_layout = cull_set::Layout::new(&renderer.device);
        let cull_count_set_layout = cull_commands_count_set::Layout::new(&renderer.device);

        let compact_draw_stream_layout =
            compact_draw_stream::PipelineLayout::new(&device, &cull_set_layout, &cull_count_set_layout);

        let compact_draw_stream_pipeline = compact_draw_stream::Pipeline::new(
            &renderer.device,
            &compact_draw_stream_layout,
            compact_draw_stream::Specialization {
                local_workgroup_size: 1024,
                draw_calls_to_compact: 2400, // FIXME
            },
            [None],
        );

        let culled_index_buffer = {
            let b = device.new_static_buffer(
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            );
            device.set_object_name(b.buffer.handle, "Global culled index buffer");
            b
        };

        // let culled_commands_buffer = {
        //     let b = device.new_static_buffer(
        //         vk::BufferUsageFlags::INDIRECT_BUFFER
        //             | vk::BufferUsageFlags::STORAGE_BUFFER
        //             | vk::BufferUsageFlags::TRANSFER_SRC
        //             | vk::BufferUsageFlags::TRANSFER_DST,
        //         VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        //     );
        //     device.set_object_name(b.buffer.handle, "indirect draw commands buffer");
        //     b
        // };
        let culled_commands_buffer = frame_graph::resources::indirect_commands_buffer::new(device);

        let cull_set = {
            let mut s = cull_set::Set::new(&renderer.device, &main_descriptor_pool, &cull_set_layout, 0);
            cull_set::bindings::indirect_commands::update_whole_buffer(
                &renderer.device,
                &mut s,
                &culled_commands_buffer.0,
            );
            cull_set::bindings::out_index_buffer::update_whole_buffer(&renderer.device, &mut s, &culled_index_buffer);
            cull_set::bindings::vertex_buffer::update_whole_buffer(
                &renderer.device,
                &mut s,
                &consolidated_mesh_buffers.position_buffer,
            );
            cull_set::bindings::index_buffer::update_whole_buffer(
                &renderer.device,
                &mut s,
                &consolidated_mesh_buffers.index_buffer,
            );

            s
        };

        let culled_commands_count_buffer = frame_graph::resources::indirect_commands_count::new(&device);

        let cull_count_set = {
            let mut s =
                cull_commands_count_set::Set::new(&renderer.device, &main_descriptor_pool, &cull_count_set_layout, 0);
            cull_commands_count_set::bindings::indirect_commands_count::update_whole_buffer(
                &renderer.device,
                &mut s,
                &culled_commands_count_buffer.0,
            );
            s
        };
        let bypass_command_buffers = renderer.new_buffered(|ix| {
            transfer_command_pool
                .pools
                .current_mut(ix)
                .allocate(&format!("Cull Pass Bypass CB[{}]", ix), &renderer.device)
        });

        CullPassData {
            culled_commands_buffer,
            culled_commands_count_buffer,
            culled_index_buffer,
            compact_draw_stream_layout,
            compact_draw_stream_pipeline,
            cull_set_layout,
            cull_count_set_layout,
            cull_set,
            cull_count_set,
            bypass_command_buffers,
        }
    }

    pub(crate) fn configure_pipeline(
        renderer: &RenderFrame,
        cull_pass_data_private: &mut CullPassDataPrivate,
        image_index: &ImageIndex,
        runtime_config: &RuntimeConfiguration,
        #[cfg(feature = "shader_reload")] reloaded_shaders: &ReloadedShaders,
    ) {
        // clean up the old pipeline that was used N frames ago
        if let Some(previous) = cull_pass_data_private
            .previous_cull_pipeline
            .current_mut(image_index.0)
            .take()
        {
            previous.destroy(&renderer.device);
        }

        let spec = generate_work::Specialization {
            local_workgroup_size: runtime_config.compute_cull_workgroup_size,
        };

        *cull_pass_data_private.previous_cull_pipeline.current_mut(image_index.0) =
            cull_pass_data_private.cull_pipeline.specialize(
                &renderer.device,
                &cull_pass_data_private.cull_pipeline_layout,
                &spec,
                [Some(&cull_pass_data_private.cull_shader)],
                #[cfg(feature = "shader_reload")]
                reloaded_shaders,
            );
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.culled_commands_buffer.destroy(device);
        self.culled_commands_count_buffer.destroy(device);
        self.culled_index_buffer.destroy(device);
        self.compact_draw_stream_pipeline.destroy(device);
        self.compact_draw_stream_layout.destroy(device);
        self.cull_set.destroy(&main_descriptor_pool.0, device);
        self.cull_count_set.destroy(&main_descriptor_pool.0, device);
        self.cull_set_layout.destroy(device);
        self.cull_count_set_layout.destroy(device);
    }
}

pub(crate) fn cull_pass_bypass(
    renderer: Res<RenderFrame>,
    cull_pass_data: Res<CullPassData>,
    runtime_config: Res<CopiedResource<RuntimeConfiguration>>,
    mut transfer_command_pool: ResMut<LocalTransferCommandPool<0>>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<CopiedResource<SwapchainIndexToFrameNumber>>,
    submissions: Res<Submissions>,
) {
    scope!("ecs", "cull pass bypass");

    if runtime_config.debug_aabbs {
        submissions.submit(&renderer, &image_index, frame_graph::TransferCull::INDEX, None);
        return;
    }

    renderer
        .transfer_timeline_semaphore
        .wait(
            &renderer.device,
            TransferTimeline::Perform.as_of_previous(&image_index, &swapchain_index_map),
        )
        .unwrap();

    let command_pool = transfer_command_pool.pools.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let cull_cb = command_pool.record_to_specific(
        &renderer.device,
        *cull_pass_data.bypass_command_buffers.current(image_index.0),
    );
    unsafe {
        scope!("cull pass", "cb recording");
        let _copy_over_marker = cull_cb.debug_marker_around("copy over cull data", [0.0, 1.0, 0.0, 1.0]);
        cull_pass_data
            .culled_commands_buffer
            .acquire_copy_frozen(&renderer, *cull_cb);
        renderer.device.cmd_copy_buffer(
            *cull_cb,
            cull_pass_data.culled_commands_buffer.buffer.handle,
            cull_pass_data.culled_commands_buffer.buffer.handle,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: cull_set::bindings::indirect_commands::SIZE,
            }],
        );
        // renderer.device.cmd_copy_buffer(
        //     *cull_cb,
        //     cull_pass_data
        //         .culled_commands_count_buffer
        //         .current(previous_ix.0)
        //         .buffer
        //         .handle,
        //     cull_pass_data
        //         .culled_commands_count_buffer
        //         .current(image_index.0)
        //         .buffer
        //         .handle,
        //     &[vk::BufferCopy {
        //         src_offset: 0,
        //         dst_offset: 0,
        //         size: cull_commands_count_set::bindings::indirect_commands_count::SIZE,
        //     }],
        // );
        // renderer.device.cmd_copy_buffer(
        //     *cull_cb,
        //     cull_pass_data
        //         .culled_index_buffer
        //         .current(previous_ix.0)
        //         .buffer
        //         .handle,
        //     cull_pass_data
        //         .culled_index_buffer
        //         .current(image_index.0)
        //         .buffer
        //         .handle,
        //     &[vk::BufferCopy {
        //         src_offset: 0,
        //         dst_offset: 0,
        //         size: cull_set::bindings::out_index_buffer::SIZE,
        //     }],
        // );
        cull_pass_data
            .culled_commands_buffer
            .release_copy_frozen(&renderer, *cull_cb);
    }
    let cull_cb = cull_cb.end();

    submissions.submit(
        &renderer,
        &image_index,
        frame_graph::TransferCull::INDEX,
        Some(*cull_cb),
    );
}

pub(crate) fn cull_pass(
    renderer: Res<RenderFrame>,
    cull_pass_data: Res<CullPassData>,
    mut cull_pass_data_private: ResMut<CullPassDataPrivate>,
    runtime_config: Res<CopiedResource<RuntimeConfiguration>>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<CopiedResource<SwapchainIndexToFrameNumber>>,
    consolidated_mesh_buffers: Res<ConsolidatedMeshBuffers>,
    camera: Res<CopiedResource<Camera>>,
    model_data: Res<ModelData>,
    camera_matrices: Res<CameraMatrices>,
    submissions: Res<Submissions>,
    query: Query<(&DrawIndex, &Position, &GltfMesh, &CoarseCulled)>,
    #[cfg(feature = "shader_reload")] reloaded_shaders: Res<ReloadedShaders>,
) {
    scope!("ecs", "cull pass");

    if runtime_config.debug_aabbs {
        submissions.submit(&renderer, &image_index, frame_graph::ComputeCull::INDEX, None);

        return;
    }

    renderer
        .compute_timeline_semaphore
        .wait(
            &renderer.device,
            ComputeTimeline::Perform.as_of_previous(&image_index, &swapchain_index_map),
        )
        .unwrap();

    CullPassData::configure_pipeline(
        &renderer,
        &mut cull_pass_data_private,
        &image_index,
        &runtime_config,
        #[cfg(feature = "shader_reload")]
        &reloaded_shaders,
    );

    let CullPassDataPrivate {
        command_pools: ref mut command_pool,
        ref command_buffers,
        ref cull_pipeline_layout,
        ref cull_pipeline,
        ..
    } = *cull_pass_data_private;

    let command_pool = command_pool.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let cull_cb = command_pool.record_to_specific(&renderer.device, *command_buffers.current(image_index.0));
    unsafe {
        scope!("cull pass", "cb recording");

        let commands_buffer = &cull_pass_data.culled_commands_buffer;
        commands_buffer.acquire_reset(&renderer, *cull_cb);
        cull_pass_data
            .culled_commands_count_buffer
            .acquire_copy_frozen(&renderer, *cull_cb);

        if runtime_config.freeze_culling {
            let _copy_over_marker = cull_cb.debug_marker_around("copy over cull data", [0.0, 1.0, 0.0, 1.0]);
            // renderer.device.cmd_copy_buffer(
            //     *cull_cb,
            //     cull_pass_data
            //         .culled_commands_buffer
            //         .current(previous_ix.0)
            //         .buffer
            //         .handle,
            //     cull_pass_data
            //         .culled_commands_buffer
            //         .current(image_index.0)
            //         .buffer
            //         .handle,
            //     &[vk::BufferCopy {
            //         src_offset: 0,
            //         dst_offset: 0,
            //         size: cull_set::bindings::indirect_commands::SIZE,
            //     }],
            // );
            renderer.device.cmd_copy_buffer(
                *cull_cb,
                cull_pass_data.culled_commands_count_buffer.buffer.handle,
                cull_pass_data.culled_commands_count_buffer.buffer.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: cull_commands_count_set::bindings::indirect_commands_count::SIZE,
                }],
            );
            renderer.device.cmd_copy_buffer(
                *cull_cb,
                cull_pass_data.culled_index_buffer.buffer.handle,
                cull_pass_data.culled_index_buffer.buffer.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: cull_set::bindings::out_index_buffer::SIZE,
                }],
            );
        } else {
            // Clear the command buffer before using
            renderer.device.cmd_fill_buffer(
                *cull_cb,
                commands_buffer.buffer.handle,
                0,
                cull_set::bindings::indirect_commands::SIZE,
                0,
            );
            commands_buffer.barrier_reset_cull(&renderer, *cull_cb);
            let cull_pass_marker = cull_cb.debug_marker_around("cull pass", [0.0, 1.0, 0.0, 1.0]);
            renderer
                .device
                .cmd_bind_pipeline(*cull_cb, vk::PipelineBindPoint::COMPUTE, *cull_pipeline.pipeline);
            cull_pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *cull_cb,
                &model_data.model_set.current(image_index.0),
                &camera_matrices.set.current(image_index.0),
                &cull_pass_data.cull_set,
            );

            let mut index_offset_in_output = 0;

            for (draw_index, mesh_position, mesh, coarse_culled) in &mut query.iter() {
                if coarse_culled.0 {
                    continue;
                }
                let vertex_offset = consolidated_mesh_buffers
                    .vertex_offsets
                    .get(&mesh.vertex_buffer.handle.as_raw())
                    .expect("Vertex buffer not consolidated");
                let (index_buffer, index_len) = pick_lod(&mesh.index_buffers, camera.position, mesh_position.0);
                let index_offset = consolidated_mesh_buffers
                    .index_offsets
                    .get(&index_buffer.handle.as_raw())
                    .expect("Index buffer not consolidated");

                let push_constants = frame_graph::generate_work::PushConstants {
                    gltf_index: draw_index.0,
                    index_count: index_len.to_u32().unwrap(),
                    index_offset: index_offset.to_u32().unwrap(),
                    index_offset_in_output,
                    vertex_offset: vertex_offset.to_i32().unwrap(),
                };

                index_offset_in_output += index_len.to_u32().unwrap();

                cull_pipeline_layout.push_constants(&renderer.device, *cull_cb, &push_constants);
                let index_len = *index_len as u32;
                let workgroup_size = cull_pipeline.spec().local_workgroup_size;
                let workgroup_count = index_len / 3 / workgroup_size + min(1, index_len / 3 % workgroup_size);
                debug_assert!(
                    renderer.device.limits.max_compute_work_group_count[0] >= workgroup_count,
                    "max_compute_work_group_count[0] violated"
                );
                debug_assert!(
                    renderer.device.limits.max_compute_work_group_invocations >= workgroup_size,
                    "max_compute_work_group_invocations violated"
                );
                debug_assert!(
                    renderer.device.limits.max_compute_work_group_size[0] >= workgroup_size,
                    "max_compute_work_group_size[0] violated"
                );
                renderer.device.cmd_dispatch(*cull_cb, workgroup_count, 1, 1);
            }

            drop(cull_pass_marker);

            commands_buffer.barrier_cull_compact(&renderer, *cull_cb);
            let _compact_marker = cull_cb.debug_marker_around("compact draw stream", [0.0, 1.0, 1.0, 1.0]);

            renderer.device.cmd_bind_pipeline(
                *cull_cb,
                vk::PipelineBindPoint::COMPUTE,
                *cull_pass_data.compact_draw_stream_pipeline.pipeline,
            );
            cull_pass_data.compact_draw_stream_layout.bind_descriptor_sets(
                &renderer.device,
                *cull_cb,
                &cull_pass_data.cull_set,
                &cull_pass_data.cull_count_set,
            );
            renderer.device.cmd_dispatch(*cull_cb, 1, 1, 1);
        }
        commands_buffer.release_compact(&renderer, *cull_cb);
        cull_pass_data
            .culled_commands_count_buffer
            .release_compute(&renderer, *cull_cb);
    }
    let cull_cb = cull_cb.end();

    submissions.submit(&renderer, &image_index, frame_graph::ComputeCull::INDEX, Some(*cull_cb));
}
