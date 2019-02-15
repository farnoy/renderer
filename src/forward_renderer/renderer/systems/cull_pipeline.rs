use super::{
    super::{
        super::ecs::components::{GltfMesh, GltfMeshBufferIndex, GltfMeshCullDescriptorSet},
        alloc,
        device::{
            Buffer, CommandBuffer, DescriptorSet, DescriptorSetLayout, DoubleBuffered, Fence,
            Semaphore,
        },
        helpers::{self, Pipeline, PipelineLayout},
        RenderFrame,
    },
    consolidate_vertex_buffers::ConsolidatedVertexBuffers,
    present::PresentData,
};
use ash::{
    version::DeviceV1_0,
    vk::{self, Handle},
};
use num_traits::ToPrimitive;
use specs::prelude::*;
use std::{cmp::min, mem::size_of, path::PathBuf, ptr, slice::from_raw_parts, sync::Arc, u64};

// Cull geometry in compute pass
pub struct CullPass;

// Updates the descriptor set that holds vertex & index buffers for compute shader to read
pub struct UpdateCullDescriptorsForMeshes;

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

impl shred::SetupHandler<CullPassData> for CullPassDataSetupHandler {
    fn setup(res: &mut Resources) {
        let renderer = res.fetch::<RenderFrame>();
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
        ]);
        device.set_object_name(cull_set_layout.handle, "Cull Descriptor Set Layout");

        #[repr(C)]
        struct MeshData {
            gltf_id: u32,
            index_count: u32,
            index_offset: u32,
            vertex_offset: i32,
        }

        let cull_pipeline_layout = helpers::new_pipeline_layout(
            Arc::clone(&device),
            &[
                &renderer.mvp_set_layout,
                &cull_set_layout,
                &renderer.mesh_assembly_set_layout,
            ],
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

        let culled_index_buffer = DoubleBuffered::new(|ix| {
            let b = device.new_buffer(
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                size_of::<u32>() as vk::DeviceSize * 300_000_000,
            );
            device.set_object_name(b.handle, &format!("Global culled index buffer - {}", ix));
            b
        });

        let cull_complete_semaphore = DoubleBuffered::new(|ix| {
            let s = renderer.device.new_semaphore();
            renderer
                .device
                .set_object_name(s.handle, &format!("Cull pass complete semaphore - {}", ix));
            s
        });

        let culled_commands_buffer = DoubleBuffered::new(|ix| {
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

        let cull_set = DoubleBuffered::new(|ix| {
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

        let cull_complete_fence = DoubleBuffered::new(|ix| {
            let f = renderer.device.new_fence();
            renderer
                .device
                .set_object_name(f.handle, &format!("Cull complete fence - {}", ix));
            f
        });

        drop(renderer);

        res.insert(CullPassData {
            culled_commands_buffer,
            culled_index_buffer,
            cull_pipeline,
            cull_pipeline_layout,
            cull_set_layout,
            cull_set,
            cull_complete_semaphore,
            previous_run_command_buffer: DoubleBuffered::new(|_| None),
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
        ReadStorage<'a, GltfMeshCullDescriptorSet>,
        ReadExpect<'a, PresentData>,
        ReadExpect<'a, ConsolidatedVertexBuffers>,
    );

    fn run(
        &mut self,
        (
            renderer,
            mut cull_pass_data,
            meshes,
            mesh_indices,
            mesh_assembly_sets,
            present_data,
            consolidated_vertex_buffers,
        ): Self::SystemData,
    ) {
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

        let mut index_offset = 0;

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
                        for (mesh, mesh_index, mesh_assembly_set) in
                            (&meshes, &mesh_indices, &mesh_assembly_sets).join()
                        {
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
                                    mesh_assembly_set.0.handle,
                                ],
                                &[],
                            );
                            let vertex_offset = consolidated_vertex_buffers
                                .offsets
                                .get(&mesh.vertex_buffer.handle.as_raw())
                                .expect("Vertex buffer not consolidated");
                            let constants = [
                                mesh_index.0,
                                mesh.index_len as u32,
                                index_offset,
                                vertex_offset.to_u32().unwrap(),
                            ];
                            index_offset += mesh.index_len.to_u32().unwrap();

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

impl<'a> System<'a> for UpdateCullDescriptorsForMeshes {
    type SystemData = (
        Entities<'a>,
        ReadExpect<'a, RenderFrame>,
        ReadStorage<'a, GltfMesh>,
        WriteStorage<'a, GltfMeshCullDescriptorSet>,
    );

    fn run(&mut self, (entities, renderer, meshes, mut mesh_descriptor_sets): Self::SystemData) {
        let mut entities_to_update = specs::BitSet::new();
        for (entity, _, ()) in (&*entities, &meshes, !&mesh_descriptor_sets).join() {
            entities_to_update.add(entity.id());
        }

        for (entity, _, mesh) in (&*entities, &entities_to_update, &meshes).join() {
            let descriptor_set = renderer
                .descriptor_pool
                .allocate_set(&renderer.mesh_assembly_set_layout);
            let updates = &[
                vk::DescriptorBufferInfo::builder()
                    .buffer(mesh.index_buffer.handle)
                    .range(vk::WHOLE_SIZE)
                    .build(),
                vk::DescriptorBufferInfo::builder()
                    .buffer(mesh.vertex_buffer.handle)
                    .range(vk::WHOLE_SIZE)
                    .build(),
            ];
            unsafe {
                renderer.device.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set.handle)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(updates)
                        .build()],
                    &[],
                );
            }
            let res = mesh_descriptor_sets
                .insert(entity, GltfMeshCullDescriptorSet(descriptor_set))
                .expect("Failed to insert GltfMeshCullDescriptorSet");
            assert!(res.is_none()); // double check that there was nothing there
        }
    }
}
