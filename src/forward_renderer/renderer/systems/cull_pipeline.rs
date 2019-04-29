use super::{
    super::{
        super::ecs::{
            components::{GltfMesh, AABB},
            systems::Camera,
        },
        alloc,
        device::{
            Buffer, CommandBuffer, DescriptorSet, DescriptorSetLayout, DoubleBuffered, Fence,
            Semaphore,
        },
        helpers::{self, Pipeline, PipelineLayout},
        RenderFrame,
    },
    consolidate_mesh_buffers::ConsolidatedMeshBuffers,
    present::PresentData,
};
use ash::{
    version::DeviceV1_0,
    vk::{self, Handle},
};
use microprofile::scope;
use num_traits::ToPrimitive;
use specs::prelude::*;
use specs_derive::Component;
use std::{cmp::min, mem::size_of, path::PathBuf, ptr, slice::from_raw_parts, sync::Arc, u64};

// Cull geometry in compute pass
pub struct CullPass;

// Should this entity be discarded when rendering
// Coarse and based on AABB being fully out of the frustum
#[derive(Clone, Component, Debug)]
#[storage(VecStorage)]
pub struct CoarseCulled(pub bool);

// Index in device generated indirect commands
// Can be absent if culled
#[derive(Clone, Component)]
#[storage(VecStorage)]
pub struct GltfMeshBufferIndex(pub u32);

pub struct AssignBufferIndex;

pub struct CoarseCulling;

pub struct CullPassData {
    pub culled_commands_buffer: DoubleBuffered<Buffer>,
    pub culled_index_buffer: DoubleBuffered<Buffer>,
    pub cull_pipeline_layout: PipelineLayout,
    pub cull_pipeline: Pipeline,
    pub cull_set_layout: DescriptorSetLayout,
    pub cull_set: DoubleBuffered<DescriptorSet>,
    pub cull_complete_semaphore: DoubleBuffered<Semaphore>,
    pub previous_run_command_buffer: DoubleBuffered<Option<CommandBuffer>>, // to clean it up
    pub cull_complete_fence: DoubleBuffered<Fence>,
}

pub struct CullPassDataSetupHandler;

impl<'a> System<'a> for AssignBufferIndex {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, GltfMesh>,
        ReadStorage<'a, CoarseCulled>,
        WriteStorage<'a, GltfMeshBufferIndex>,
    );

    fn run(&mut self, (entities, meshes, coarse_culled, mut indices): Self::SystemData) {
        microprofile::scope!("ecs", "assign buffer index");
        let mut ix = 0;
        for (entity, _mesh, coarse_culled) in (&*entities, &meshes, &coarse_culled).join() {
            if coarse_culled.0 {
                indices.remove(entity);
            } else {
                drop(indices.insert(entity, GltfMeshBufferIndex(ix as u32)));
                ix += 1;
            }
        }
    }
}

impl<'a> System<'a> for CoarseCulling {
    type SystemData = (
        ReadStorage<'a, AABB>,
        Read<'a, Camera>,
        WriteStorage<'a, CoarseCulled>,
    );

    fn run(&mut self, (aabb, camera, mut culled): Self::SystemData) {
        microprofile::scope!("ecs", "coarse culling");
        for (aabb, culled) in (&aabb, &mut culled).join() {
            let mut outside = false;
            'per_plane: for plane in camera.frustum_planes.iter() {
                let e =
                    aabb.h.x * plane.x.abs() + aabb.h.y * plane.y.abs() + aabb.h.z * plane.z.abs();

                let s = cgmath::dot(aabb.c.extend(1.0), *plane);
                if s - e > 0.0 {
                    outside = true;
                    break 'per_plane;
                }
            }
            culled.0 = outside;
        }
    }
}

impl shred::SetupHandler<CullPassData> for CullPassDataSetupHandler {
    fn setup(world: &mut World) {
        let renderer = world.fetch::<RenderFrame>();
        let device = &renderer.device;

        let cull_set_layout = device.new_descriptor_set_layout(&[
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
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: ptr::null(),
            },
        ]);
        device.set_object_name(cull_set_layout.handle, "Cull Descriptor Set Layout");

        #[repr(C)]
        struct MeshData {
            gltf_id: u32,
            index_count: u32,
            index_offset: u32,
            index_offset_in_output: u32,
            vertex_offset: i32,
        }

        let cull_pipeline_layout = helpers::new_pipeline_layout(
            Arc::clone(&device),
            &[&renderer.mvp_set_layout, &cull_set_layout],
            &[vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: size_of::<MeshData>() as u32,
            }],
        );
        let cull_pipeline = helpers::new_compute_pipeline(
            Arc::clone(&device),
            &cull_pipeline_layout,
            &PathBuf::from(env!("OUT_DIR")).join("generate_work.comp.spv"),
        );

        let culled_index_buffer = renderer.new_buffered(|ix| {
            let b = device.new_buffer(
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                size_of::<u32>() as vk::DeviceSize * 300_000_000,
            );
            device.set_object_name(b.handle, &format!("Global culled index buffer - {}", ix));
            b
        });

        let cull_complete_semaphore = renderer.new_buffered(|ix| {
            let s = renderer.device.new_semaphore();
            renderer
                .device
                .set_object_name(s.handle, &format!("Cull pass complete semaphore - {}", ix));
            s
        });

        let culled_commands_buffer = renderer.new_buffered(|ix| {
            let b = device.new_buffer(
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                size_of::<vk::DrawIndexedIndirectCommand>() as vk::DeviceSize * 2400,
            );
            device.set_object_name(b.handle, &format!("indirect draw commands buffer - {}", ix));
            b
        });

        let cull_set = renderer.new_buffered(|ix| {
            let s = renderer.descriptor_pool.allocate_set(&cull_set_layout);
            device.set_object_name(s.handle, &format!("Cull Descriptor Set - {}", ix));

            {
                let cull_updates = &[
                    vk::DescriptorBufferInfo {
                        buffer: culled_commands_buffer.current(ix).handle,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    },
                    vk::DescriptorBufferInfo {
                        buffer: culled_index_buffer.current(ix).handle,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    },
                ];
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::builder()
                            .dst_set(s.handle)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(cull_updates)
                            .build()],
                        &[],
                    );
                }
            }

            s
        });

        let cull_complete_fence = renderer.new_buffered(|ix| {
            let f = renderer.device.new_fence();
            renderer
                .device
                .set_object_name(f.handle, &format!("Cull complete fence - {}", ix));
            f
        });

        let previous_run_command_buffer = renderer.new_buffered(|_| None);

        drop(renderer);

        world.insert(CullPassData {
            culled_commands_buffer,
            culled_index_buffer,
            cull_pipeline,
            cull_pipeline_layout,
            cull_set_layout,
            cull_set,
            cull_complete_semaphore,
            previous_run_command_buffer,
            cull_complete_fence,
        });
    }
}

impl<'a> System<'a> for CullPass {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        ReadExpect<'a, RenderFrame>,
        Write<'a, CullPassData, CullPassDataSetupHandler>,
        ReadStorage<'a, GltfMesh>,
        ReadStorage<'a, GltfMeshBufferIndex>,
        ReadExpect<'a, PresentData>,
        ReadExpect<'a, ConsolidatedMeshBuffers>,
    );

    fn run(
        &mut self,
        (
            renderer,
            mut cull_pass_data,
            meshes,
            mesh_indices,
            present_data,
            consolidate_mesh_buffers,
        ): Self::SystemData,
    ) {
        microprofile::scope!("ecs", "cull pass");
        if cull_pass_data
            .previous_run_command_buffer
            .current(present_data.image_index)
            .is_some()
        {
            unsafe {
                renderer
                    .device
                    .wait_for_fences(
                        &[cull_pass_data
                            .cull_complete_fence
                            .current(present_data.image_index)
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
                    .current(present_data.image_index)
                    .handle])
                .expect("failed to reset cull complete fence");
        }

        {
            let cull_updates = &[
                vk::DescriptorBufferInfo {
                    buffer: consolidate_mesh_buffers.position_buffer.handle,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: consolidate_mesh_buffers.index_buffer.handle,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
            ];
            unsafe {
                renderer.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::builder()
                        .dst_set(
                            cull_pass_data
                                .cull_set
                                .current(present_data.image_index)
                                .handle,
                        )
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(cull_updates)
                        .build()],
                    &[],
                );
            }
        }

        let mut index_offset_in_output = 0;

        let cull_cb = renderer
            .compute_command_pool
            .record_one_time(|command_buffer| unsafe {
                renderer.device.debug_marker_around(
                    command_buffer,
                    "cull pass",
                    [0.0, 1.0, 0.0, 1.0],
                    || {
                        renderer.device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            cull_pass_data.cull_pipeline.handle,
                        );
                        renderer.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            cull_pass_data.cull_pipeline_layout.handle,
                            0,
                            &[
                                renderer.mvp_set.current(present_data.image_index).handle,
                                cull_pass_data
                                    .cull_set
                                    .current(present_data.image_index)
                                    .handle,
                            ],
                            &[],
                        );
                        for (mesh, mesh_index) in (&meshes, &mesh_indices).join() {
                            let vertex_offset = consolidate_mesh_buffers
                                .vertex_offsets
                                .get(&mesh.vertex_buffer.handle.as_raw())
                                .expect("Vertex buffer not consolidated");
                            let index_offset = consolidate_mesh_buffers
                                .index_offsets
                                .get(&mesh.index_buffer.handle.as_raw())
                                .expect("Index buffer not consolidated");
                            let constants = [
                                mesh_index.0,
                                mesh.index_len.to_u32().unwrap(),
                                index_offset.to_u32().unwrap(),
                                index_offset_in_output,
                                vertex_offset.to_i32().unwrap() as u32,
                            ];

                            index_offset_in_output += mesh.index_len.to_u32().unwrap();

                            let casted: &[u8] = {
                                from_raw_parts(constants.as_ptr() as *const u8, constants.len() * 4)
                            };
                            renderer.device.cmd_push_constants(
                                command_buffer,
                                cull_pass_data.cull_pipeline_layout.handle,
                                vk::ShaderStageFlags::COMPUTE,
                                0,
                                casted,
                            );
                            let index_len = mesh.index_len as u32;
                            let workgroup_size = 512; // TODO: make a specialization constant, not hardcoded
                            let workgroup_count = index_len / 3 / workgroup_size
                                + min(1, index_len / 3 % workgroup_size);
                            renderer
                                .device
                                .cmd_dispatch(command_buffer, workgroup_count, 1, 1);
                        }
                    },
                );
            });
        let wait_semaphores = &[];
        let signal_semaphores = [cull_pass_data
            .cull_complete_semaphore
            .current(present_data.image_index)
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
                        .current(present_data.image_index)
                        .handle,
                )
                .unwrap();
        }

        let ix = present_data.image_index;
        *cull_pass_data.previous_run_command_buffer.current_mut(ix) = Some(cull_cb); // also destroys the previous one
    }
}
