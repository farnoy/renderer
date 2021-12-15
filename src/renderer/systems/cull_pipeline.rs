use ash::vk::{self, Handle};
use bevy_ecs::prelude::*;
use num_traits::ToPrimitive;
use petgraph::graph::NodeIndex;
use profiling::scope;

#[cfg(feature = "crash_debugging")]
use crate::renderer::CrashBuffer;
#[cfg(feature = "shader_reload")]
use crate::renderer::ReloadedShaders;
use crate::{
    ecs::{
        components::{Position, AABB},
        resources::Camera,
        systems::RuntimeConfiguration,
    },
    renderer::{
        binding_size, camera_set,
        device::{Device, DoubleBuffered, VmaMemoryUsage},
        frame_graph,
        helpers::{command_util::CommandUtil, pick_lod},
        model_set,
        systems::{consolidate_mesh_buffers::ConsolidatedMeshBuffers, present::ImageIndex},
        update_whole_buffer, CameraMatrices, CopiedResource, DrawIndex, GltfMesh, MainDescriptorPool, ModelData,
        RenderFrame, RenderStage, SmartPipeline, SmartPipelineLayout, SmartSet, SmartSetLayout, Submissions,
        SwapchainIndexToFrameNumber,
    },
};

renderer_macros::define_set! {
    cull_set {
        indirect_commands STORAGE_BUFFER from [COMPUTE],
        out_index_buffer STORAGE_BUFFER from [COMPUTE],
        vertex_buffer STORAGE_BUFFER from [COMPUTE],
        index_buffer STORAGE_BUFFER from [COMPUTE]
    }
}
renderer_macros::define_set! {
    cull_commands_count_set {
        indirect_commands_count STORAGE_BUFFER from [COMPUTE]
    }
}
renderer_macros::define_pipe! {
    generate_work {
        descriptors [model_set, camera_set, cull_set]
        specialization_constants [1 => local_workgroup_size: u32]
        compute
    }
}
renderer_macros::define_pipe! {
    compact_draw_stream {
        descriptors [cull_set, cull_commands_count_set]
        specialization_constants [
            1 => local_workgroup_size: u32,
            2 => draw_calls_to_compact: u32
        ]
        varying subgroup size
        compute
    }
}

renderer_macros::define_pass!(ComputeCull on compute);
renderer_macros::define_pass!(TransferCull on transfer);

// Should this entity be discarded when rendering
// Coarse and based on AABB being fully out of the frustum
pub(crate) struct CoarseCulled(pub(crate) bool);

renderer_macros::define_resource! { IndirectCommandsBuffer = StaticBuffer<crate::renderer::frame_graph::IndirectCommands> }
renderer_macros::define_resource! { IndirectCommandsCount = StaticBuffer<crate::renderer::frame_graph::IndirectCommandsCount> }
renderer_macros::define_resource! { CulledIndexBuffer = StaticBuffer<crate::renderer::frame_graph::OutIndexBuffer> }

pub(crate) struct CullPassData {
    pub(crate) culled_commands_buffer: IndirectCommandsBuffer,
    pub(crate) culled_commands_count_buffer: IndirectCommandsCount,
    pub(crate) culled_index_buffer: CulledIndexBuffer,
    compact_draw_stream_layout: SmartPipelineLayout<compact_draw_stream::PipelineLayout>,
    compact_draw_stream_pipeline: SmartPipeline<compact_draw_stream::Pipeline>,
    cull_set_layout: SmartSetLayout<cull_set::Layout>,
    cull_count_set_layout: SmartSetLayout<cull_commands_count_set::Layout>,
    cull_set: SmartSet<cull_set::Set>,
    cull_count_set: SmartSet<cull_commands_count_set::Set>,
}

pub(crate) struct CullPassDataPrivate {
    cull_pipeline_layout: SmartPipelineLayout<generate_work::PipelineLayout>,
    cull_pipeline: SmartPipeline<generate_work::Pipeline>,
    compute_command_util: CommandUtil,
    previous_cull_pipeline: DoubleBuffered<Option<SmartPipeline<generate_work::Pipeline>>>,
}

pub(crate) struct TransferCullPrivate {
    transfer_command_util: CommandUtil,
}

pub(crate) const INITIAL_WORKGROUP_SIZE: u32 = 384;

pub(crate) fn coarse_culling(
    task_pool: Res<bevy_tasks::ComputeTaskPool>,
    camera: Res<Camera>,
    mut query: Query<(&AABB, &mut CoarseCulled)>,
) {
    scope!("ecs::coarse_culling");

    query.par_for_each_mut(&task_pool, 2, |(aabb, mut coarse_culled)| {
        scope!("parallel::coarse_culling");
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
    fn new(
        renderer: &RenderFrame,
        cull_pass_data: &CullPassData,
        model_data: &ModelData,
        camera_matrices: &CameraMatrices,
    ) -> CullPassDataPrivate {
        let cull_specialization = generate_work::Specialization {
            local_workgroup_size: INITIAL_WORKGROUP_SIZE,
        };
        let cull_pipeline_layout = SmartPipelineLayout::new(
            &renderer.device,
            (
                &model_data.model_set_layout,
                &camera_matrices.set_layout,
                &cull_pass_data.cull_set_layout,
            ),
        );
        let cull_pipeline = SmartPipeline::new(&renderer.device, &cull_pipeline_layout, cull_specialization, ());

        CullPassDataPrivate {
            cull_pipeline_layout,
            cull_pipeline,
            compute_command_util: CommandUtil::new(renderer, renderer.device.compute_queue_family),
            previous_cull_pipeline: renderer.new_buffered(|_| None),
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.cull_pipeline_layout.destroy(device);
        self.cull_pipeline.destroy(device);
        for pipe in self.previous_cull_pipeline.into_iter().flatten() {
            pipe.destroy(device);
        }
        self.compute_command_util.destroy(device);
    }
}

impl CullPassData {
    fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &mut MainDescriptorPool,
        consolidated_mesh_buffers: &ConsolidatedMeshBuffers,
    ) -> CullPassData {
        let device = &renderer.device;

        let cull_set_layout = SmartSetLayout::new(&renderer.device);
        let cull_count_set_layout = SmartSetLayout::new(&renderer.device);

        let compact_draw_stream_layout = SmartPipelineLayout::new(device, (&cull_set_layout, &cull_count_set_layout));

        let compact_draw_stream_pipeline = SmartPipeline::new(
            &renderer.device,
            &compact_draw_stream_layout,
            compact_draw_stream::Specialization {
                local_workgroup_size: 1024,
                draw_calls_to_compact: 2400, // FIXME
            },
            (),
        );

        let culled_index_buffer = CulledIndexBuffer::new(device);

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
        let culled_commands_buffer = IndirectCommandsBuffer::new(device);

        let cull_set = {
            let mut s = SmartSet::new(&renderer.device, main_descriptor_pool, &cull_set_layout, 0);
            update_whole_buffer::<cull_set::bindings::indirect_commands>(
                &renderer.device,
                &mut s,
                &culled_commands_buffer.0,
            );
            update_whole_buffer::<cull_set::bindings::out_index_buffer>(&renderer.device, &mut s, &culled_index_buffer);
            update_whole_buffer::<cull_set::bindings::vertex_buffer>(
                &renderer.device,
                &mut s,
                &consolidated_mesh_buffers.position_buffer,
            );
            update_whole_buffer::<cull_set::bindings::index_buffer>(
                &renderer.device,
                &mut s,
                &consolidated_mesh_buffers.index_buffer,
            );

            s
        };

        let culled_commands_count_buffer = IndirectCommandsCount::new(device);

        let cull_count_set = {
            let mut s = SmartSet::new(&renderer.device, main_descriptor_pool, &cull_count_set_layout, 0);
            update_whole_buffer::<cull_commands_count_set::bindings::indirect_commands_count>(
                &renderer.device,
                &mut s,
                &culled_commands_count_buffer.0,
            );
            s
        };

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
                (),
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

impl FromWorld for CullPassData {
    fn from_world(world: &mut World) -> Self {
        world.resource_scope(|world, mut main_descriptor_pool: Mut<MainDescriptorPool>| {
            let renderer = world.get_resource().unwrap();
            let consolidated_mesh_buffers = world.get_resource().unwrap();

            CullPassData::new(renderer, &mut main_descriptor_pool, consolidated_mesh_buffers)
        })
    }
}

impl FromWorld for CullPassDataPrivate {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource().unwrap();
        let cull_pass_data = world.get_resource().unwrap();
        let model_data = world.get_resource().unwrap();
        let camera_matrices = world.get_resource().unwrap();

        CullPassDataPrivate::new(renderer, cull_pass_data, model_data, camera_matrices)
    }
}

impl TransferCullPrivate {
    fn new(renderer: &RenderFrame) -> TransferCullPrivate {
        TransferCullPrivate {
            transfer_command_util: CommandUtil::new(renderer, renderer.device.transfer_queue_family),
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.transfer_command_util.destroy(device);
    }
}

impl FromWorld for TransferCullPrivate {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource().unwrap();

        Self::new(renderer)
    }
}

pub(crate) fn cull_pass_bypass(
    renderer: Res<RenderFrame>,
    cull_pass_data: Res<CullPassData>,
    mut transfer_cull_private: ResMut<TransferCullPrivate>,
    runtime_config: Res<CopiedResource<RuntimeConfiguration>>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<CopiedResource<SwapchainIndexToFrameNumber>>,
    submissions: Res<Submissions>,
    renderer_input: Res<renderer_macro_lib::RendererInput>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
) {
    scope!("ecs::cull_pass_bypass");

    if !submissions
        .active_graph
        .contains_node(NodeIndex::from(frame_graph::TransferCull::INDEX))
    {
        return;
    }

    if runtime_config.debug_aabbs {
        submissions.submit(
            &renderer,
            frame_graph::TransferCull::INDEX,
            None,
            &renderer_input,
            #[cfg(feature = "crash_debugging")]
            &crash_buffer,
        );
        return;
    }

    frame_graph::TransferCull::Stage::wait_previous(&renderer, &image_index, &swapchain_index_map);

    debug_assert!(
        runtime_config.freeze_culling,
        "freeze_culling must be on in cull_pass_bypass"
    );

    let cull_cb = transfer_cull_private
        .transfer_command_util
        .reset_and_record(&renderer, &image_index);
    unsafe {
        scope!("cull_pass::cb_recording");
        let _copy_over_marker = cull_cb.debug_marker_around("copy over cull data", [0.0, 1.0, 0.0, 1.0]);

        let _guard = renderer_macros::barrier!(
            *cull_cb,
            IndirectCommandsBuffer.copy_frozen rw in TransferCull transfer copy if [FREEZE_CULLING],
            IndirectCommandsCount.copy_frozen rw in TransferCull transfer copy if [FREEZE_CULLING],
            CulledIndexBuffer.copy_frozen rw in TransferCull transfer copy if [FREEZE_CULLING]
        );
        renderer.device.cmd_copy_buffer(
            *cull_cb,
            cull_pass_data.culled_commands_buffer.buffer.handle,
            cull_pass_data.culled_commands_buffer.buffer.handle,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: binding_size::<cull_set::bindings::indirect_commands>(),
            }],
        );
        renderer.device.cmd_copy_buffer(
            *cull_cb,
            cull_pass_data.culled_commands_count_buffer.buffer.handle,
            cull_pass_data.culled_commands_count_buffer.buffer.handle,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: binding_size::<cull_commands_count_set::bindings::indirect_commands_count>(),
            }],
        );
        renderer.device.cmd_copy_buffer(
            *cull_cb,
            cull_pass_data.culled_index_buffer.buffer.handle,
            cull_pass_data.culled_index_buffer.buffer.handle,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: binding_size::<cull_set::bindings::out_index_buffer>(),
            }],
        );
    }
    let cull_cb = cull_cb.end();

    submissions.submit(
        &renderer,
        frame_graph::TransferCull::INDEX,
        Some(*cull_cb),
        &renderer_input,
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
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
    renderer_input: Res<renderer_macro_lib::RendererInput>,
    query: Query<(&DrawIndex, &Position, &GltfMesh, &CoarseCulled)>,
    #[cfg(feature = "shader_reload")] reloaded_shaders: Res<ReloadedShaders>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
) {
    scope!("ecs::cull_pass");

    if !submissions
        .active_graph
        .contains_node(NodeIndex::from(frame_graph::ComputeCull::INDEX))
    {
        return;
    }

    if runtime_config.debug_aabbs {
        submissions.submit(
            &renderer,
            frame_graph::ComputeCull::INDEX,
            None,
            &renderer_input,
            #[cfg(feature = "crash_debugging")]
            &crash_buffer,
        );

        return;
    }

    frame_graph::ComputeCull::Stage::wait_previous(&renderer, &image_index, &swapchain_index_map);

    debug_assert!(
        !runtime_config.freeze_culling,
        "freeze_culling must be off in cull_pass"
    );

    CullPassData::configure_pipeline(
        &renderer,
        &mut cull_pass_data_private,
        &image_index,
        &runtime_config,
        #[cfg(feature = "shader_reload")]
        &reloaded_shaders,
    );

    let CullPassDataPrivate {
        ref mut compute_command_util,
        ref cull_pipeline_layout,
        ref cull_pipeline,
        ..
    } = *cull_pass_data_private;

    let cull_cb = compute_command_util.reset_and_record(&renderer, &image_index);
    let indirect_commands_buffer = &cull_pass_data.culled_commands_buffer;
    let indirect_commands_count = &cull_pass_data.culled_commands_count_buffer;
    unsafe {
        scope!("cull pass::cb_recording");

        let commands_buffer = &cull_pass_data.culled_commands_buffer;

        let cull_pass_marker = cull_cb.debug_marker_around("cull pass", [0.0, 1.0, 0.0, 1.0]);
        // Clear the command buffer before using
        let guard = renderer_macros::barrier!(
            *cull_cb, [FREEZE_CULLING => false],
            IndirectCommandsBuffer.reset w in ComputeCull transfer clear if [!FREEZE_CULLING],
            IndirectCommandsCount.reset w in ComputeCull transfer clear if [!FREEZE_CULLING]
        );
        renderer.device.cmd_fill_buffer(
            *cull_cb,
            commands_buffer.buffer.handle,
            0,
            binding_size::<cull_set::bindings::indirect_commands>(),
            0,
        );
        renderer.device.cmd_fill_buffer(
            *cull_cb,
            indirect_commands_count.buffer.handle,
            0,
            binding_size::<cull_commands_count_set::bindings::indirect_commands_count>(),
            0,
        );
        drop(guard);
        renderer
            .device
            .cmd_bind_pipeline(*cull_cb, vk::PipelineBindPoint::COMPUTE, cull_pipeline.vk());
        cull_pipeline_layout.bind_descriptor_sets(
            &renderer.device,
            *cull_cb,
            (
                model_data.model_set.current(image_index.0),
                camera_matrices.set.current(image_index.0),
                &cull_pass_data.cull_set,
            ),
        );

        let guard = renderer_macros::barrier!(
            *cull_cb, [FREEZE_CULLING => false],
            IndirectCommandsBuffer.cull rw in ComputeCull descriptor generate_work.cull_set.indirect_commands after [reset] if [!FREEZE_CULLING]; indirect_commands_buffer,
            ConsolidatedPositionBuffer.in_cull r in ComputeCull descriptor generate_work.cull_set.indirect_commands after [consolidate] if [!FREEZE_CULLING],
            ConsolidatedIndexBuffer.cull_from r in ComputeCull descriptor generate_work.cull_set.index_buffer after [consolidate] if [!FREEZE_CULLING],
            CulledIndexBuffer.cull w in ComputeCull descriptor generate_work.cull_set.out_index_buffer if [!FREEZE_CULLING]
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

            let push_constants = generate_work::PushConstants {
                gltf_index: draw_index.0,
                index_count: index_len.to_u32().unwrap(),
                index_offset: index_offset.to_u32().unwrap(),
                index_offset_in_output,
                vertex_offset: vertex_offset.to_i32().unwrap(),
            };

            index_offset_in_output += index_len.to_u32().unwrap();

            cull_pipeline_layout.push_constants(&renderer.device, *cull_cb, &push_constants);
            let index_len = *index_len as u32;
            let workgroup_size = cull_pipeline.specialization.local_workgroup_size;
            let workgroup_count = index_len / 3 / workgroup_size + u32::from(index_len / 3 % workgroup_size > 0);
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

        drop(guard);
        drop(cull_pass_marker);

        let _compact_marker = cull_cb.debug_marker_around("compact draw stream", [0.0, 1.0, 1.0, 1.0]);
        let _guard = renderer_macros::barrier!(
            *cull_cb, [FREEZE_CULLING => false],
            IndirectCommandsBuffer.compact rw in ComputeCull descriptor compact_draw_stream.cull_set.indirect_commands after [cull] if [!FREEZE_CULLING]; indirect_commands_buffer,
            IndirectCommandsCount.compute rw in ComputeCull descriptor compact_draw_stream.cull_commands_count_set.indirect_commands_count after [reset] if [!FREEZE_CULLING]; indirect_commands_count,
        );

        renderer.device.cmd_bind_pipeline(
            *cull_cb,
            vk::PipelineBindPoint::COMPUTE,
            cull_pass_data.compact_draw_stream_pipeline.vk(),
        );
        cull_pass_data.compact_draw_stream_layout.bind_descriptor_sets(
            &renderer.device,
            *cull_cb,
            (&cull_pass_data.cull_set, &cull_pass_data.cull_count_set),
        );
        let calls_to_compact = query.iter().count() as u32;
        let workgroup_size = cull_pass_data
            .compact_draw_stream_pipeline
            .specialization
            .local_workgroup_size;
        let workgroup_count = calls_to_compact / workgroup_size + u32::from(calls_to_compact % workgroup_size > 0);
        renderer.device.cmd_dispatch(*cull_cb, 1, workgroup_count, 1);
    }
    let cull_cb = cull_cb.end();

    submissions.submit(
        &renderer,
        frame_graph::ComputeCull::INDEX,
        Some(*cull_cb),
        &renderer_input,
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
    );
}
