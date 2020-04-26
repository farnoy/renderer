use crate::{
    ecs::{components::AABB, resources::Camera, systems::RuntimeConfiguration},
    renderer::{
        alloc, compute,
        device::{Buffer, CommandBuffer, DoubleBuffered, Event},
        helpers::{self, pick_lod, Pipeline},
        systems::{consolidate_mesh_buffers::ConsolidatedMeshBuffers, present::ImageIndex},
        timeline_value, CameraMatrices, DrawIndex, GltfMesh, MainDescriptorPool, ModelData,
        Position, RenderFrame,
    },
};
use ash::{
    version::DeviceV1_0,
    vk::{self, Handle},
};
use legion::prelude::*;
#[cfg(feature = "microprofile")]
use microprofile::scope;
use num_traits::ToPrimitive;
use parking_lot::Mutex;
use std::{cmp::min, path::PathBuf, sync::Arc};

// Cull geometry in compute pass
pub struct CullPass;

// Should this entity be discarded when rendering
// Coarse and based on AABB being fully out of the frustum
pub struct CoarseCulled(pub bool);

pub struct CoarseCulling;

pub struct CullPassData {
    pub culled_commands_buffer: DoubleBuffered<Buffer>,
    pub culled_index_buffer: DoubleBuffered<Buffer>,
    pub cull_pipeline_layout: super::super::shaders::generate_work::PipelineLayout,
    pub cull_pipeline: Pipeline,
    pub cull_set_layout: super::super::shaders::cull_set::DescriptorSetLayout,
    pub cull_set: DoubleBuffered<super::super::shaders::cull_set::DescriptorSet>,
}

// Internal storage for cleanup purposes
pub struct CullPassDataPrivate {
    previous_run_command_buffer: DoubleBuffered<Option<CommandBuffer>>, // to clean it up
}

pub struct CullPassEvent {
    pub cull_complete_event: DoubleBuffered<Mutex<Event>>,
}

impl CoarseCulling {
    pub fn exec_system() -> Box<(dyn Schedulable + 'static)> {
        SystemBuilder::<()>::new("CoarseCulling")
            .read_resource::<Camera>()
            .with_query(<(Read<AABB>, Write<CoarseCulled>)>::query())
            .build(move |_commands, mut world, ref camera, query| {
                for (ref aabb, ref mut coarse_culled) in query.iter_mut(&mut world) {
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
            })
    }
}

impl CullPassDataPrivate {
    pub fn new(renderer: &RenderFrame) -> CullPassDataPrivate {
        CullPassDataPrivate {
            previous_run_command_buffer: renderer.new_buffered(|_| None),
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

        let cull_set_layout =
            super::super::shaders::cull_set::DescriptorSetLayout::new(&renderer.device);
        device.set_object_name(cull_set_layout.layout.handle, "Cull Descriptor Set Layout");

        let cull_pipeline_layout = super::super::shaders::generate_work::PipelineLayout::new(
            &device,
            &model_data.model_set_layout,
            &camera_matrices.set_layout,
            &cull_set_layout,
        );
        use std::io::Read;
        let path = std::path::PathBuf::from(env!("OUT_DIR")).join("generate_work.comp.spv");
        let file = std::fs::File::open(path).expect("Could not find shader.");
        let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
        let module = spirv_reflect::create_shader_module(&bytes).unwrap();
        debug_assert!(super::super::shaders::generate_work::verify_spirv(&module));
        let cull_pipeline = helpers::new_compute_pipeline(
            Arc::clone(&device),
            &cull_pipeline_layout.layout,
            &PathBuf::from(env!("OUT_DIR")).join("generate_work.comp.spv"),
        );

        let culled_index_buffer = renderer.new_buffered(|ix| {
            let b = device.new_buffer(
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                super::super::shaders::cull_set::bindings::out_index_buffer::SIZE,
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
                super::super::shaders::cull_set::bindings::indirect_commands::SIZE,
            );
            device.set_object_name(b.handle, &format!("indirect draw commands buffer - {}", ix));
            b
        });

        let cull_set = renderer.new_buffered(|ix| {
            let s = super::super::shaders::cull_set::DescriptorSet::new(
                &main_descriptor_pool,
                &cull_set_layout,
            );
            device.set_object_name(s.set.handle, &format!("Cull Descriptor Set - {}", ix));
            s.update_whole_buffer(&renderer, 0, &culled_commands_buffer.current(ix));
            s.update_whole_buffer(&renderer, 1, &culled_index_buffer.current(ix));
            s
        });

        CullPassData {
            culled_commands_buffer,
            culled_index_buffer,
            cull_pipeline,
            cull_pipeline_layout,
            cull_set_layout,
            cull_set,
        }
    }
}

impl CullPass {
    pub fn exec_system() -> Box<(dyn legion::systems::schedule::Schedulable + 'static)> {
        SystemBuilder::<()>::new("CullPass")
            .read_resource::<RenderFrame>()
            .read_resource::<CullPassData>()
            .write_resource::<CullPassDataPrivate>()
            .read_resource::<RuntimeConfiguration>()
            .read_resource::<ImageIndex>()
            .read_resource::<ConsolidatedMeshBuffers>()
            .read_resource::<Camera>()
            .read_resource::<ModelData>()
            .read_resource::<CameraMatrices>()
            .with_query(<(Read<DrawIndex>, Read<Position>, Read<GltfMesh>)>::query())
            .build(move |_commands, world, resources, query| {
                let (
                    ref renderer,
                    ref cull_pass_data,
                    ref mut cull_pass_data_private,
                    ref runtime_config,
                    ref image_index,
                    ref consolidate_mesh_buffers,
                    ref camera,
                    ref model_data,
                    ref camera_matrices,
                ) = resources;
                #[cfg(feature = "microprofile")]
                microprofile::scope!("ecs", "cull pass");

                if runtime_config.debug_aabbs {
                    renderer
                        .compute_timeline_semaphore
                        .signal(timeline_value!(compute @ renderer.frame_number => PERFORM))
                        .expect("Failed to bypass culling & signal compute semaphore");
                    return;
                }

                if cull_pass_data_private
                    .previous_run_command_buffer
                    .current(image_index.0)
                    .is_some()
                {
                    renderer
                        .compute_timeline_semaphore
                        .wait(timeline_value!(compute @ renderer.frame_number - 2 => PERFORM))
                        .unwrap();
                }

                cull_pass_data
                    .cull_set
                    .current(image_index.0)
                    .update_whole_buffer(&renderer, 2, &consolidate_mesh_buffers.position_buffer);
                cull_pass_data
                    .cull_set
                    .current(image_index.0)
                    .update_whole_buffer(&renderer, 3, &consolidate_mesh_buffers.index_buffer);

                let mut index_offset_in_output = 0i32;

                let cull_cb = renderer
                    .compute_command_pool
                    .record_one_time("cull pass cb");
                unsafe {
                    let _debug_marker = renderer.device.debug_marker_around2(
                        &cull_cb,
                        "cull pass",
                        [0.0, 1.0, 0.0, 1.0],
                    );
                    // Clear the command buffer before using
                    {
                        use crate::renderer::shaders::cull_set;
                        let commands_buffer =
                            cull_pass_data.culled_commands_buffer.current(image_index.0);
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
                    for (draw_index, mesh_position, mesh) in query.iter(&world) {
                        let vertex_offset = consolidate_mesh_buffers
                            .vertex_offsets
                            .get(&mesh.vertex_buffer.handle.as_raw())
                            .expect("Vertex buffer not consolidated");
                        let (index_buffer, index_len) =
                            pick_lod(&mesh.index_buffers, camera.position, mesh_position.0);
                        let index_offset = consolidate_mesh_buffers
                            .index_offsets
                            .get(&index_buffer.handle.as_raw())
                            .expect("Index buffer not consolidated");

                        let push_constants = super::super::shaders::GenerateWorkPushConstants {
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
                        let workgroup_size = 512; // TODO: make a specialization constant, not hardcoded
                        let workgroup_count =
                            index_len / 3 / workgroup_size + min(1, index_len / 3 % workgroup_size);
                        renderer
                            .device
                            .cmd_dispatch(*cull_cb, workgroup_count, 1, 1);
                    }
                }
                let cull_cb = cull_cb.end();
                let wait_semaphores = &[renderer.compute_timeline_semaphore.handle];
                let wait_semaphore_values =
                    &[timeline_value!(compute @ last renderer.frame_number => PERFORM)];
                let signal_semaphores = &[renderer.compute_timeline_semaphore.handle];
                let signal_semaphore_values =
                    &[timeline_value!(compute @ renderer.frame_number => PERFORM)];
                let dst_stage_masks =
                    vec![vk::PipelineStageFlags::TOP_OF_PIPE; wait_semaphores.len()];
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

                let queue = renderer.device.compute_queues[0].lock();

                unsafe {
                    renderer
                        .device
                        .queue_submit(*queue, &[submit], vk::Fence::null())
                        .unwrap();
                }

                let ix = image_index.0;
                *cull_pass_data_private
                    .previous_run_command_buffer
                    .current_mut(ix) = Some(cull_cb);
                // also destroys the previous one
            })
    }
}
