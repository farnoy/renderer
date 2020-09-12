use crate::{
    ecs::{components::AABB, resources::Camera, systems::RuntimeConfiguration},
    renderer::{
        alloc, compute,
        device::{Buffer, DoubleBuffered, Event, StrictCommandPool},
        helpers::{self, pick_lod, Pipeline},
        shaders::{self, cull_set, generate_work},
        systems::{consolidate_mesh_buffers::ConsolidatedMeshBuffers, present::ImageIndex},
        timeline_value, CameraMatrices, DrawIndex, GltfMesh, MainDescriptorPool, ModelData,
        Position, RenderFrame, Shader,
    },
};
use ash::{
    version::DeviceV1_0,
    vk::{self, Handle},
};
use bevy_ecs::prelude::*;
use helpers::PipelineLayout;
#[cfg(feature = "microprofile")]
use microprofile::scope;
use num_traits::ToPrimitive;
use parking_lot::Mutex;
use std::{cmp::min, ffi::CStr, sync::Arc};

// Cull geometry in compute pass
pub struct CullPass;

// Should this entity be discarded when rendering
// Coarse and based on AABB being fully out of the frustum
pub struct CoarseCulled(pub bool);

pub struct CullPassData {
    pub culled_commands_buffer: DoubleBuffered<Buffer>,
    pub culled_index_buffer: DoubleBuffered<Buffer>,
    pub cull_pipeline_layout: generate_work::PipelineLayout,
    pub cull_pipeline: Arc<Pipeline>,
    pub cull_set_layout: cull_set::DescriptorSetLayout,
    pub cull_set: DoubleBuffered<cull_set::DescriptorSet>,
    pub specialization: generate_work::Specialization,
    shader: Shader,
}

// Internal storage for cleanup purposes
pub struct CullPassDataPrivate {
    command_pool: DoubleBuffered<StrictCommandPool>,
    previous_cull_pipeline: DoubleBuffered<Option<Arc<Pipeline>>>,
    previous_specialization: generate_work::Specialization,
}

const INITIAL_WORKGROUP_SIZE: u32 = 512;

pub struct CullPassEvent {
    pub cull_complete_event: DoubleBuffered<Mutex<Event>>,
}

pub fn coarse_culling(camera: Res<Camera>, mut query: Query<(&AABB, &mut CoarseCulled)>) {
    for (aabb, mut coarse_culled) in &mut query.iter() {
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
    }
}

impl CullPassDataPrivate {
    pub fn new(renderer: &RenderFrame) -> CullPassDataPrivate {
        CullPassDataPrivate {
            command_pool: renderer.new_buffered(|ix| {
                StrictCommandPool::new(
                    &renderer.device,
                    renderer.device.compute_queue_family,
                    &format!("Cull pass Command Pool[{}]", ix),
                )
            }),
            previous_cull_pipeline: renderer.new_buffered(|_| None),
            previous_specialization: generate_work::Specialization {
                local_workgroup_size: INITIAL_WORKGROUP_SIZE,
            },
        }
    }
}

impl CullPassData {
    pub fn new(
        renderer: &RenderFrame,
        model_data: &ModelData,
        main_descriptor_pool: &mut MainDescriptorPool,
        camera_matrices: &CameraMatrices,
    ) -> CullPassData {
        let device = &renderer.device;

        let cull_set_layout = cull_set::DescriptorSetLayout::new(&renderer.device);
        device.set_object_name(cull_set_layout.layout.handle, "Cull Descriptor Set Layout");

        let cull_pipeline_layout = generate_work::PipelineLayout::new(
            &device,
            &model_data.model_set_layout,
            &camera_matrices.set_layout,
            &cull_set_layout,
        );

        let path = std::path::PathBuf::from(env!("OUT_DIR")).join("generate_work.comp.spv");
        let shader = renderer.device.new_shader(
            &path,
            |_| true, // TODO: spirq fails to load spec const used to define dimensions
                      // generate_work::load_and_verify_spirv,
        );

        let specialization = generate_work::Specialization {
            local_workgroup_size: INITIAL_WORKGROUP_SIZE,
        };

        let cull_pipeline = CullPassData::create_pipeline(
            renderer,
            &shader,
            &specialization,
            &cull_pipeline_layout.layout,
            None,
        );

        let culled_index_buffer = renderer.new_buffered(|ix| {
            let b = device.new_buffer(
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                cull_set::bindings::out_index_buffer::SIZE,
            );
            device.set_object_name(b.handle, &format!("Global culled index buffer - {}", ix));
            b
        });

        let culled_commands_buffer = renderer.new_buffered(|ix| {
            let b = device.new_buffer(
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                cull_set::bindings::indirect_commands::SIZE,
            );
            device.set_object_name(b.handle, &format!("indirect draw commands buffer - {}", ix));
            b
        });

        let cull_set = renderer.new_buffered(|ix| {
            let s = cull_set::DescriptorSet::new(&main_descriptor_pool, &cull_set_layout);
            device.set_object_name(s.set.handle, &format!("Cull Descriptor Set - {}", ix));
            s.update_whole_buffer(&renderer, 0, &culled_commands_buffer.current(ix));
            s.update_whole_buffer(&renderer, 1, &culled_index_buffer.current(ix));
            s
        });

        CullPassData {
            culled_commands_buffer,
            culled_index_buffer,
            cull_pipeline: Arc::new(cull_pipeline),
            cull_pipeline_layout,
            cull_set_layout,
            cull_set,
            specialization,
            shader,
        }
    }

    fn configure_pipeline(
        renderer: &RenderFrame,
        cull_pass_data: &mut CullPassData,
        cull_pass_data_private: &mut CullPassDataPrivate,
        image_index: &ImageIndex,
    ) {
        if cull_pass_data.specialization != cull_pass_data_private.previous_specialization {
            let cull_pipeline = CullPassData::create_pipeline(
                renderer,
                &cull_pass_data.shader,
                &cull_pass_data.specialization,
                &cull_pass_data.cull_pipeline_layout.layout,
                Some(&cull_pass_data.cull_pipeline),
            );

            *cull_pass_data_private
                .previous_cull_pipeline
                .current_mut(image_index.0) = Some(Arc::clone(&cull_pass_data.cull_pipeline));
            cull_pass_data.cull_pipeline = Arc::new(cull_pipeline);
            cull_pass_data_private.previous_specialization = cull_pass_data.specialization.clone();
        } else {
            // potentially destroy unused pipeline
            *cull_pass_data_private
                .previous_cull_pipeline
                .current_mut(image_index.0) = None;
        }
    }

    fn create_pipeline(
        renderer: &RenderFrame,
        shader: &Shader,
        spec: &generate_work::Specialization,
        layout: &PipelineLayout,
        base_pipeline: Option<&Pipeline>,
    ) -> Pipeline {
        let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let spec_info = spec.get_spec_info();
        let pipe = helpers::new_compute_pipelines(
            Arc::clone(&renderer.device),
            &[vk::ComputePipelineCreateInfo::builder()
                .stage(
                    vk::PipelineShaderStageCreateInfo::builder()
                        .module(shader.vk())
                        .name(&shader_entry_name)
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .specialization_info(&spec_info)
                        .build(),
                )
                .layout(layout.handle)
                .flags(vk::PipelineCreateFlags::ALLOW_DERIVATIVES)
                .base_pipeline_handle(
                    base_pipeline
                        .map(|pipe| pipe.handle)
                        .unwrap_or_else(vk::Pipeline::null),
                )],
        )
        .into_iter()
        .next()
        .unwrap();

        renderer
            .device
            .set_object_name(pipe.handle, "Cull Descriptor Pipeline");

        pipe
    }
}

pub fn cull_pass(
    renderer: Res<RenderFrame>,
    mut cull_pass_data: ResMut<CullPassData>,
    mut cull_pass_data_private: ResMut<CullPassDataPrivate>,
    runtime_config: Res<RuntimeConfiguration>,
    image_index: Res<ImageIndex>,
    consolidated_mesh_buffers: Res<ConsolidatedMeshBuffers>,
    camera: Res<Camera>,
    model_data: Res<ModelData>,
    camera_matrices: Res<CameraMatrices>,
    mut query: Query<(&DrawIndex, &Position, &GltfMesh)>,
) {
    #[cfg(feature = "microprofile")]
    microprofile::scope!("ecs", "cull pass");

    {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("cull pass", "configure pipeline");
        CullPassData::configure_pipeline(
            &renderer,
            &mut cull_pass_data,
            &mut cull_pass_data_private,
            &image_index,
        );
    }

    if runtime_config.debug_aabbs {
        let wait_semaphores = &[renderer.compute_timeline_semaphore.handle];
        let wait_semaphore_values =
            &[timeline_value!(compute @ last renderer.frame_number => PERFORM)];
        let signal_semaphores = &[renderer.compute_timeline_semaphore.handle];
        let signal_semaphore_values =
            &[timeline_value!(compute @ renderer.frame_number => PERFORM)];
        let dst_stage_masks = vec![vk::PipelineStageFlags::TOP_OF_PIPE; wait_semaphores.len()];
        let mut wait_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(wait_semaphore_values)
            .signal_semaphore_values(signal_semaphore_values);
        let submit = vk::SubmitInfo::builder()
            .push_next(&mut wait_timeline)
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&dst_stage_masks)
            .signal_semaphores(signal_semaphores)
            .build();

        let queue = renderer.device.compute_queues[0].lock();

        unsafe {
            renderer
                .device
                .queue_submit(*queue, &[submit], vk::Fence::null())
                .unwrap();
        }
        return;
    }

    {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("cull pass", "wait previous");
        renderer
            .compute_timeline_semaphore
            .wait(timeline_value!(compute @ previous image_index; of renderer => PERFORM))
            .unwrap();
    }

    {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("cull pass", "update descriptors");

        cull_pass_data
            .cull_set
            .current(image_index.0)
            .update_whole_buffer(&renderer, 2, &consolidated_mesh_buffers.position_buffer);
        cull_pass_data
            .cull_set
            .current(image_index.0)
            .update_whole_buffer(&renderer, 3, &consolidated_mesh_buffers.index_buffer);
    }

    let command_pool = cull_pass_data_private
        .command_pool
        .current_mut(image_index.0);

    unsafe {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("cull pass", "CP reset");
        command_pool.recreate();
    }

    let mut command_session = command_pool.session();

    let cull_cb = command_session.record_one_time(&format!("Cull Pass CB[{}]", image_index.0));
    unsafe {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("cull pass", "cb recording");
        let _debug_marker = cull_cb.debug_marker_around("cull pass", [0.0, 1.0, 0.0, 1.0]);
        // Clear the command buffer before using
        {
            let commands_buffer = cull_pass_data.culled_commands_buffer.current(image_index.0);
            renderer.device.cmd_fill_buffer(
                *cull_cb,
                commands_buffer.handle,
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
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .buffer(commands_buffer.handle)
                    .size(vk::WHOLE_SIZE)
                    .build()],
                &[],
            );
        }
        renderer.device.cmd_bind_pipeline(
            *cull_cb,
            vk::PipelineBindPoint::COMPUTE,
            cull_pass_data.cull_pipeline.handle,
        );
        cull_pass_data.cull_pipeline_layout.bind_descriptor_sets(
            &renderer.device,
            *cull_cb,
            &model_data.model_set.current(image_index.0),
            &camera_matrices.set.current(image_index.0),
            &cull_pass_data.cull_set.current(image_index.0),
        );

        let mut index_offset_in_output = 0i32;

        for (draw_index, mesh_position, mesh) in &mut query.iter() {
            let vertex_offset = consolidated_mesh_buffers
                .vertex_offsets
                .get(&mesh.vertex_buffer.handle.as_raw())
                .expect("Vertex buffer not consolidated");
            let (index_buffer, index_len) =
                pick_lod(&mesh.index_buffers, camera.position, mesh_position.0);
            let index_offset = consolidated_mesh_buffers
                .index_offsets
                .get(&index_buffer.handle.as_raw())
                .expect("Index buffer not consolidated");

            let push_constants = shaders::GenerateWorkPushConstants {
                gltf_index: draw_index.0,
                index_count: index_len.to_u32().unwrap(),
                index_offset: index_offset.to_u32().unwrap(),
                index_offset_in_output,
                vertex_offset: vertex_offset.to_i32().unwrap(),
            };

            index_offset_in_output += index_len.to_i32().unwrap();

            cull_pass_data.cull_pipeline_layout.push_constants(
                &renderer.device,
                *cull_cb,
                &push_constants,
            );
            let index_len = *index_len as u32;
            let workgroup_size = cull_pass_data.specialization.local_workgroup_size;
            let workgroup_count =
                index_len / 3 / workgroup_size + min(1, index_len / 3 % workgroup_size);
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
            renderer
                .device
                .cmd_dispatch(*cull_cb, workgroup_count, 1, 1);

            // To silence synchronization validation warnings here - doesn't fix anything
            #[cfg(feature = "sync_validation")]
            renderer.device.cmd_pipeline_barrier(
                *cull_cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[
                    vk::BufferMemoryBarrier::builder()
                        .buffer(
                            cull_pass_data
                                .culled_commands_buffer
                                .current(image_index.0)
                                .handle,
                        )
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .size(vk::WHOLE_SIZE)
                        .build(),
                    vk::BufferMemoryBarrier::builder()
                        .buffer(
                            cull_pass_data
                                .culled_index_buffer
                                .current(image_index.0)
                                .handle,
                        )
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .size(vk::WHOLE_SIZE)
                        .build(),
                ],
                &[],
            );
        }
    }
    let cull_cb = cull_cb.end();
    let wait_semaphores = &[renderer.compute_timeline_semaphore.handle];
    let wait_semaphore_values = &[timeline_value!(compute @ last renderer.frame_number => PERFORM)];
    let signal_semaphores = &[renderer.compute_timeline_semaphore.handle];
    let signal_semaphore_values = &[timeline_value!(compute @ renderer.frame_number => PERFORM)];
    let dst_stage_masks = vec![vk::PipelineStageFlags::TOP_OF_PIPE; wait_semaphores.len()];
    let command_buffers = &[*cull_cb];
    let mut wait_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
        .wait_semaphore_values(wait_semaphore_values)
        .signal_semaphore_values(signal_semaphore_values);
    let submit = vk::SubmitInfo::builder()
        .push_next(&mut wait_timeline)
        .wait_semaphores(wait_semaphores)
        .wait_dst_stage_mask(&dst_stage_masks)
        .command_buffers(command_buffers)
        .signal_semaphores(signal_semaphores)
        .build();

    {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("cull pass", "submit");
        let queue = renderer.device.compute_queues[0].lock();

        unsafe {
            renderer
                .device
                .queue_submit(*queue, &[submit], vk::Fence::null())
                .unwrap();
        }
    }
}
