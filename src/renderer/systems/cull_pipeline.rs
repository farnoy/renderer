use std::{cmp::min, mem::take};

use ash::{
    version::DeviceV1_0,
    vk::{self, Handle},
};
use bevy_ecs::prelude::*;
use microprofile::scope;
use num_traits::ToPrimitive;

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
        frame_graph,
        helpers::{pick_lod, MP_INDIAN_RED},
        shaders::{self, compact_draw_stream, cull_commands_count_set, cull_set, generate_work},
        systems::{consolidate_mesh_buffers::ConsolidatedMeshBuffers, present::ImageIndex},
        CameraMatrices, ComputeTimeline, CopiedResource, DrawIndex, GltfMesh, LocalTransferCommandPool,
        MainDescriptorPool, ModelData, RenderFrame, RenderStage, Shader, SwapchainIndexToFrameNumber, TransferTimeline,
    },
};

// Should this entity be discarded when rendering
// Coarse and based on AABB being fully out of the frustum
pub(crate) struct CoarseCulled(pub(crate) bool);

pub(crate) struct CullPassData {
    pub(crate) culled_commands_buffer: DoubleBuffered<cull_set::bindings::indirect_commands::Buffer>,
    pub(crate) culled_commands_count_buffer:
        DoubleBuffered<cull_commands_count_set::bindings::indirect_commands_count::Buffer>,
    pub(crate) culled_index_buffer: DoubleBuffered<cull_set::bindings::out_index_buffer::Buffer>,
    compact_draw_stream_layout: compact_draw_stream::PipelineLayout,
    compact_draw_stream_pipeline: compact_draw_stream::Pipeline,
    cull_set_layout: cull_set::Layout,
    cull_count_set_layout: cull_commands_count_set::Layout,
    cull_set: DoubleBuffered<cull_set::Set>,
    cull_count_set: DoubleBuffered<cull_commands_count_set::Set>,
}

pub(crate) struct CullPassDataPrivate {
    cull_pipeline_layout: generate_work::PipelineLayout,
    cull_pipeline: generate_work::Pipeline,
    command_pool: DoubleBuffered<StrictCommandPool>,
    previous_cull_pipeline: DoubleBuffered<Option<generate_work::Pipeline>>,
    cull_shader: Shader,
}

pub(crate) const INITIAL_WORKGROUP_SIZE: u32 = 512;

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
        let cull_pipeline = generate_work::Pipeline::new(
            &renderer.device,
            &cull_pipeline_layout,
            cull_specialization,
            Some(&cull_shader),
        );
        CullPassDataPrivate {
            cull_pipeline_layout,
            cull_pipeline,
            command_pool: renderer.new_buffered(|ix| {
                StrictCommandPool::new(
                    &renderer.device,
                    renderer.device.compute_queue_family,
                    &format!("Cull pass Command Pool[{}]", ix),
                )
            }),
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
        self.command_pool.into_iter().for_each(|pool| pool.destroy(device));
        self.cull_shader.destroy(device);
    }
}

impl CullPassData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &mut MainDescriptorPool,
        consolidated_mesh_buffers: &ConsolidatedMeshBuffers,
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
            None,
        );

        let culled_index_buffer = renderer.new_buffered(|ix| {
            let b = device.new_static_buffer(
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            );
            device.set_object_name(b.buffer.handle, &format!("Global culled index buffer - {}", ix));
            b
        });

        let culled_commands_buffer = renderer.new_buffered(|ix| {
            let b = device.new_static_buffer(
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            );
            device.set_object_name(b.buffer.handle, &format!("indirect draw commands buffer - {}", ix));
            b
        });

        let cull_set = renderer.new_buffered(|ix| {
            let mut s = cull_set::Set::new(&renderer.device, &main_descriptor_pool, &cull_set_layout, ix);
            cull_set::bindings::indirect_commands::update_whole_buffer(
                &renderer.device,
                &mut s,
                &culled_commands_buffer.current(ix),
            );
            cull_set::bindings::out_index_buffer::update_whole_buffer(
                &renderer.device,
                &mut s,
                &culled_index_buffer.current(ix),
            );
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
        });

        let culled_commands_count_buffer = renderer.new_buffered(|ix| {
            let b = device.new_static_buffer(
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            );
            device.set_object_name(
                b.buffer.handle,
                &format!("indirect draw commands count buffer - {}", ix),
            );
            b
        });

        let cull_count_set = renderer.new_buffered(|ix| {
            let mut s =
                cull_commands_count_set::Set::new(&renderer.device, &main_descriptor_pool, &cull_count_set_layout, ix);
            cull_commands_count_set::bindings::indirect_commands_count::update_whole_buffer(
                &renderer.device,
                &mut s,
                &culled_commands_count_buffer.current(ix),
            );
            s
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
        if let Some(previous) = take(cull_pass_data_private.previous_cull_pipeline.current_mut(image_index.0)) {
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
                Some(&cull_pass_data_private.cull_shader),
                #[cfg(feature = "shader_reload")]
                reloaded_shaders,
            );
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.culled_commands_buffer.into_iter().for_each(|b| b.destroy(device));
        self.culled_commands_count_buffer
            .into_iter()
            .for_each(|b| b.destroy(device));
        self.culled_index_buffer.into_iter().for_each(|b| b.destroy(device));
        self.compact_draw_stream_pipeline.destroy(device);
        self.compact_draw_stream_layout.destroy(device);
        self.cull_set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
        self.cull_count_set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
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
) {
    scope!("ecs", "cull pass bypass");

    if runtime_config.debug_aabbs {
        let queue = renderer.device.compute_queue(0).lock();
        frame_graph::TransferCull::Stage::queue_submit(&image_index, &renderer, *queue, &[]).unwrap();
        return;
    }

    renderer
        .transfer_timeline_semaphore
        .wait(
            &renderer.device,
            TransferTimeline::Perform.as_of_previous(&image_index, &swapchain_index_map),
        )
        .unwrap();

    let previous_ix = ImageIndex(
        swapchain_index_map
            .map
            .iter()
            .position(|ix| *ix == renderer.frame_number - 1)
            .expect("could not find previous ImageIndex")
            .to_u32()
            .unwrap(),
    );

    let command_pool = transfer_command_pool.pools.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let mut command_session = command_pool.session(&renderer.device);

    let cull_cb = command_session.record_one_time(&format!("Cull Pass Bypass CB[{}]", image_index.0));
    unsafe {
        scope!("cull pass", "cb recording");
        let _copy_over_marker = cull_cb.debug_marker_around("copy over cull data", [0.0, 1.0, 0.0, 1.0]);
        renderer.device.cmd_copy_buffer(
            *cull_cb,
            cull_pass_data
                .culled_commands_buffer
                .current(previous_ix.0)
                .buffer
                .handle,
            cull_pass_data
                .culled_commands_buffer
                .current(image_index.0)
                .buffer
                .handle,
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
    }
    let cull_cb = cull_cb.end();

    let queue = renderer.device.transfer_queue().lock();
    frame_graph::TransferCull::Stage::queue_submit(&image_index, &renderer, *queue, &[*cull_cb]).unwrap();
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
    query: Query<(&DrawIndex, &Position, &GltfMesh, &CoarseCulled)>,
    #[cfg(feature = "shader_reload")] reloaded_shaders: Res<ReloadedShaders>,
) {
    microprofile::scope!("ecs", "cull pass");

    if runtime_config.debug_aabbs {
        let queue = renderer.device.compute_queue(0).lock();
        frame_graph::ComputeCull::Stage::queue_submit(&image_index, &renderer, *queue, &[]).unwrap();

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

    let previous_ix = ImageIndex(
        swapchain_index_map
            .map
            .iter()
            .position(|ix| *ix == renderer.frame_number - 1)
            .expect("could not find previous ImageIndex")
            .to_u32()
            .unwrap(),
    );

    let CullPassDataPrivate {
        ref mut command_pool,
        ref cull_pipeline_layout,
        ref cull_pipeline,
        ..
    } = *cull_pass_data_private;

    let command_pool = command_pool.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let mut command_session = command_pool.session(&renderer.device);

    let cull_cb = command_session.record_one_time(&format!("Cull Pass CB[{}]", image_index.0));
    unsafe {
        scope!("cull pass", "cb recording");

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
                cull_pass_data
                    .culled_commands_count_buffer
                    .current(previous_ix.0)
                    .buffer
                    .handle,
                cull_pass_data
                    .culled_commands_count_buffer
                    .current(image_index.0)
                    .buffer
                    .handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: cull_commands_count_set::bindings::indirect_commands_count::SIZE,
                }],
            );
            renderer.device.cmd_copy_buffer(
                *cull_cb,
                cull_pass_data.culled_index_buffer.current(previous_ix.0).buffer.handle,
                cull_pass_data.culled_index_buffer.current(image_index.0).buffer.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: cull_set::bindings::out_index_buffer::SIZE,
                }],
            );
        } else {
            // Clear the command buffer before using
            let commands_buffer = cull_pass_data.culled_commands_buffer.current(image_index.0);
            renderer.device.cmd_fill_buffer(
                *cull_cb,
                commands_buffer.buffer.handle,
                0,
                cull_set::bindings::indirect_commands::SIZE,
                0,
            );
            renderer.device.cmd_pipeline_barrier(
                *cull_cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                Default::default(),
                &[],
                &[vk::BufferMemoryBarrier::builder()
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .buffer(commands_buffer.buffer.handle)
                    .size(vk::WHOLE_SIZE)
                    .build()],
                &[],
            );

            let cull_pass_marker = cull_cb.debug_marker_around("cull pass", [0.0, 1.0, 0.0, 1.0]);
            renderer
                .device
                .cmd_bind_pipeline(*cull_cb, vk::PipelineBindPoint::COMPUTE, *cull_pipeline.pipeline);
            cull_pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *cull_cb,
                &model_data.model_set.current(image_index.0),
                &camera_matrices.set.current(image_index.0),
                &cull_pass_data.cull_set.current(image_index.0),
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

                let push_constants = shaders::generate_work::PushConstants {
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

            renderer.device.cmd_pipeline_barrier(
                *cull_cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                Default::default(),
                &[],
                &[vk::BufferMemoryBarrier::builder()
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .buffer(
                        cull_pass_data
                            .culled_commands_buffer
                            .current(image_index.0)
                            .buffer
                            .handle,
                    )
                    .size(vk::WHOLE_SIZE)
                    .build()],
                &[],
            );

            let _compact_marker = cull_cb.debug_marker_around("compact draw stream", [0.0, 1.0, 1.0, 1.0]);

            renderer.device.cmd_bind_pipeline(
                *cull_cb,
                vk::PipelineBindPoint::COMPUTE,
                *cull_pass_data.compact_draw_stream_pipeline.pipeline,
            );
            cull_pass_data.compact_draw_stream_layout.bind_descriptor_sets(
                &renderer.device,
                *cull_cb,
                &cull_pass_data.cull_set.current(image_index.0),
                &cull_pass_data.cull_count_set.current(image_index.0),
            );
            renderer.device.cmd_dispatch(*cull_cb, 1, 1, 1);
        }
    }
    let cull_cb = cull_cb.end();

    let queue = renderer.device.compute_queue(0).lock();
    frame_graph::ComputeCull::Stage::queue_submit(&image_index, &renderer, *queue, &[*cull_cb]).unwrap();
}
