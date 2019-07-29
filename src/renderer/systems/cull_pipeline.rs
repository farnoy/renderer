use super::{
    super::{
        alloc,
        device::{Buffer, CommandBuffer, DoubleBuffered, Event, Fence, Semaphore},
        helpers::{self, pick_lod, Pipeline},
        CameraMatrices, GltfMesh, MainDescriptorPool, ModelData, RenderFrame,
    },
    consolidate_mesh_buffers::ConsolidatedMeshBuffers,
    present::ImageIndex,
};
use crate::ecs::{
    components::{Position, AABB},
    systems::Camera,
};
use ash::{
    version::DeviceV1_0,
    vk::{self, Handle},
};
#[cfg(feature = "microprofile")]
use microprofile::scope;
use num_traits::ToPrimitive;
use parking_lot::Mutex;
use specs::*;
use std::{cmp::min, path::PathBuf, sync::Arc, u64};

// Cull geometry in compute pass
pub struct CullPass;

// Should this entity be discarded when rendering
// Coarse and based on AABB being fully out of the frustum
#[derive(Clone, Component, Debug)]
#[storage(VecStorage)]
pub struct CoarseCulled(pub bool);

pub struct CoarseCulling;

pub struct CullPassData {
    pub culled_commands_buffer: DoubleBuffered<Buffer>,
    pub culled_index_buffer: DoubleBuffered<Buffer>,
    pub cull_pipeline_layout: super::super::shaders::generate_work::PipelineLayout,
    pub cull_pipeline: Pipeline,
    pub cull_set_layout: super::super::shaders::cull_set::DescriptorSetLayout,
    pub cull_set: DoubleBuffered<super::super::shaders::cull_set::DescriptorSet>,
    pub cull_complete_semaphore: DoubleBuffered<Semaphore>,
    pub cull_complete_fence: DoubleBuffered<Fence>,
}

// Internal storage for cleanup purposes
pub struct CullPassDataPrivate {
    previous_run_command_buffer: DoubleBuffered<Option<CommandBuffer>>, // to clean it up
}

pub struct CullPassEvent {
    pub cull_complete_event: DoubleBuffered<Mutex<Event>>,
}

impl<'a> System<'a> for CoarseCulling {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, AABB>,
        Read<'a, Camera>,
        WriteStorage<'a, CoarseCulled>,
    );

    fn run(&mut self, (entities, aabb, camera, mut culled): Self::SystemData) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "coarse culling");
        for (entity_id, aabb) in (&*entities, &aabb).join() {
            #[allow(unused)]
            let mut outside = false;
            'per_plane: for plane in camera.frustum_planes.iter() {
                let e = aabb.h.dot(&plane.xyz().abs());

                let s = plane.dot(&aabb.c.push(1.0));
                if s - e > 0.0 {
                    outside = true;
                    break 'per_plane;
                }
            }
            culled
                .insert(entity_id, CoarseCulled(outside))
                .expect("failed to update coarse culled");
        }
    }
}

impl specs::shred::SetupHandler<CullPassDataPrivate> for CullPassDataPrivate {
    fn setup(world: &mut World) {
        if world.has_value::<CullPassDataPrivate>() {
            return;
        }

        let renderer = world.fetch::<RenderFrame>();

        let previous_run_command_buffer = renderer.new_buffered(|_| None);

        drop(renderer);

        world.insert(CullPassDataPrivate {
            previous_run_command_buffer,
        });
    }
}

impl specs::shred::SetupHandler<CullPassData> for CullPassData {
    fn setup(world: &mut World) {
        if world.has_value::<CullPassData>() {
            return;
        }

        let result = world.exec(
            #[allow(clippy::type_complexity)]
            |(renderer, model_data, main_descriptor_pool, camera_matrices): (
                ReadExpect<RenderFrame>,
                Read<ModelData, ModelData>,
                Write<MainDescriptorPool, MainDescriptorPool>,
                Read<CameraMatrices, CameraMatrices>,
            )| {
                let device = &renderer.device;

                let cull_set_layout =
                    super::super::shaders::cull_set::DescriptorSetLayout::new(&renderer.device);
                device.set_object_name(cull_set_layout.layout.handle, "Cull Descriptor Set Layout");

                let cull_pipeline_layout =
                    super::super::shaders::generate_work::PipelineLayout::new(
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
                    device
                        .set_object_name(b.handle, &format!("Global culled index buffer - {}", ix));
                    b
                });

                let cull_complete_semaphore = renderer.new_buffered(|ix| {
                    let s = renderer.device.new_semaphore();
                    renderer.device.set_object_name(
                        s.handle,
                        &format!("Cull pass complete semaphore - {}", ix),
                    );
                    s
                });

                let culled_commands_buffer = renderer.new_buffered(|ix| {
                    let b = device.new_buffer(
                        vk::BufferUsageFlags::INDIRECT_BUFFER
                            | vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST,
                        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                        super::super::shaders::cull_set::bindings::indirect_commands::SIZE,
                    );
                    device.set_object_name(
                        b.handle,
                        &format!("indirect draw commands buffer - {}", ix),
                    );
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

                let cull_complete_fence = renderer.new_buffered(|ix| {
                    let f = renderer.device.new_fence();
                    renderer
                        .device
                        .set_object_name(f.handle, &format!("Cull complete fence - {}", ix));
                    f
                });

                drop(renderer);

                CullPassData {
                    culled_commands_buffer,
                    culled_index_buffer,
                    cull_pipeline,
                    cull_pipeline_layout,
                    cull_set_layout,
                    cull_set,
                    cull_complete_semaphore,
                    cull_complete_fence,
                }
            },
        );

        world.insert(result);
    }
}

impl specs::shred::SetupHandler<CullPassEvent> for CullPassEvent {
    fn setup(world: &mut World) {
        if world.has_value::<CullPassEvent>() {
            return;
        }

        let renderer = world.fetch::<RenderFrame>();
        let cull_complete_event = renderer.new_buffered(|ix| {
            let e = renderer.device.new_event();
            renderer
                .device
                .set_object_name(e.handle, &format!("Cull complete event - {}", ix));
            Mutex::new(e)
        });

        drop(renderer);

        world.insert(CullPassEvent {
            cull_complete_event,
        });
    }
}

impl<'a> System<'a> for CullPass {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        Entities<'a>,
        ReadExpect<'a, RenderFrame>,
        Read<'a, CullPassData, CullPassData>,
        Write<'a, CullPassDataPrivate, CullPassDataPrivate>,
        ReadStorage<'a, GltfMesh>,
        Read<'a, ImageIndex>,
        ReadExpect<'a, ConsolidatedMeshBuffers>,
        ReadStorage<'a, Position>,
        Read<'a, Camera>,
        Read<'a, ModelData, ModelData>,
        Read<'a, CameraMatrices, CameraMatrices>,
    );

    fn run(
        &mut self,
        (
            entities,
            renderer,
            cull_pass_data,
            mut cull_pass_data_private,
            meshes,
            image_index,
            consolidate_mesh_buffers,
            positions,
            camera,
            model_data,
            camera_matrices,
        ): Self::SystemData,
    ) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "cull pass");
        if cull_pass_data_private
            .previous_run_command_buffer
            .current(image_index.0)
            .is_some()
        {
            unsafe {
                renderer
                    .device
                    .wait_for_fences(
                        &[cull_pass_data
                            .cull_complete_fence
                            .current(image_index.0)
                            .handle],
                        true,
                        u64::MAX,
                    )
                    .expect("Wait for fence failed.");
            }
        }
        unsafe {
            renderer
                .device
                .reset_fences(&[cull_pass_data
                    .cull_complete_fence
                    .current(image_index.0)
                    .handle])
                .expect("failed to reset cull complete fence");
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
            .record_one_time(|command_buffer| unsafe {
                renderer.device.debug_marker_around(
                    command_buffer,
                    "cull pass",
                    [0.0, 1.0, 0.0, 1.0],
                    || {
                        // Clear the command buffer before using
                        {
                            let commands_buffer =
                                cull_pass_data.culled_commands_buffer.current(image_index.0);
                            renderer.device.cmd_fill_buffer(
                                command_buffer,
                                commands_buffer.handle,
                                0,
                                super::super::shaders::cull_set::bindings::indirect_commands::SIZE,
                                0,
                            );
                            renderer.device.cmd_pipeline_barrier(
                                command_buffer,
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
                                    .build()],
                                &[],
                            );
                        }
                        renderer.device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            cull_pass_data.cull_pipeline.handle,
                        );
                        cull_pass_data.cull_pipeline_layout.bind_descriptor_sets(
                            &renderer.device,
                            command_buffer,
                            &model_data.model_set.current(image_index.0),
                            &camera_matrices.set.current(image_index.0),
                            &cull_pass_data.cull_set.current(image_index.0),
                        );
                        for (entity, mesh, mesh_position) in
                            (&*entities, &meshes, &positions).join()
                        {
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
                                gltf_index: entity.id(),
                                index_count: index_len.to_u32().unwrap(),
                                index_offset: index_offset.to_u32().unwrap(),
                                index_offset_in_output,
                                vertex_offset: vertex_offset.to_i32().unwrap(),
                            };

                            index_offset_in_output += index_len.to_i32().unwrap();

                            cull_pass_data.cull_pipeline_layout.push_constants(
                                &renderer.device,
                                command_buffer,
                                &push_constants,
                            );
                            let index_len = *index_len as u32;
                            let workgroup_size = 512; // TODO: make a specialization constant, not hardcoded
                            let workgroup_count = index_len / 3 / workgroup_size
                                + min(1, index_len / 3 % workgroup_size);
                            renderer
                                .device
                                .cmd_dispatch(command_buffer, workgroup_count, 1, 1);
                        }
                        // }
                    },
                );
            });
        let wait_semaphores = &[];
        let signal_semaphores = [cull_pass_data
            .cull_complete_semaphore
            .current(image_index.0)
            .handle];
        let dst_stage_masks = vec![vk::PipelineStageFlags::TOP_OF_PIPE; wait_semaphores.len()];
        let command_buffers = &[*cull_cb];
        let submit = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&dst_stage_masks)
            .command_buffers(command_buffers)
            .signal_semaphores(&signal_semaphores)
            .build();

        let queue = renderer.device.compute_queues[0].lock();

        unsafe {
            renderer
                .device
                .queue_submit(
                    *queue,
                    &[submit],
                    cull_pass_data
                        .cull_complete_fence
                        .current(image_index.0)
                        .handle,
                )
                .unwrap();
        }

        let ix = image_index.0;
        *cull_pass_data_private
            .previous_run_command_buffer
            .current_mut(ix) = Some(cull_cb); // also destroys the previous one
    }
}
