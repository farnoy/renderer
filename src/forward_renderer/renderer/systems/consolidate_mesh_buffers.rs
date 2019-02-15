use super::super::{
    super::ecs::components::GltfMesh,
    alloc,
    device::{Buffer, CommandBuffer, Fence, Semaphore},
    RenderFrame,
};
use ash::{
    version::DeviceV1_0,
    vk::{self, Handle},
};
use hashbrown::{hash_map::Entry, HashMap};
use specs::prelude::*;
use std::{mem::size_of, u64};

/// Describes layout of gltf mesh vertex data in a shared buffer
pub struct ConsolidatedMeshBuffers {
    /// Maps from vertex buffer handle to offset within the consolidated buffer
    /// Hopefully this is distinct enough
    pub vertex_offsets: HashMap<u64, vk::DeviceSize>,
    /// Next free vertex offset in the buffer that can be used for a new mesh
    next_vertex_offset: vk::DeviceSize,
    /// Maps from index buffer handle to offset within the consolidated buffer
    pub index_offsets: HashMap<u64, vk::DeviceSize>,
    /// Next free index offset in the buffer that can be used for a new mesh
    next_index_offset: vk::DeviceSize,
    /// Stores position data for each mesh
    pub position_buffer: Buffer,
    /// Stores normal data for each mesh
    pub normal_buffer: Buffer,
    /// Stores uv data for each mesh
    pub uv_buffer: Buffer,
    /// Stores index data for each mesh
    pub index_buffer: Buffer,
    /// If this semaphore is present, a modification to the consolidated buffer has happened
    /// and the user must synchronize with it
    pub sync_point: Option<Semaphore>,
    /// Holds the command buffer executed in the previous frame, to clean it up safely in the following frame
    previous_run_command_buffer: Option<CommandBuffer>,
    /// Holds the fence used to synchronize the transfer that occured in previous frame.
    sync_point_fence: Fence,
}

/// Identifies distinct GLTF meshes in components and copies them to a shared buffer
pub struct ConsolidateMeshBuffers;

// TODO: dynamic unloading of meshes
// TODO: use actual transfer queue for the transfers

impl shred::SetupHandler<ConsolidatedMeshBuffers> for ConsolidatedMeshBuffers {
    fn setup(res: &mut Resources) {
        let renderer = res.fetch::<RenderFrame>();
        let vertex_offsets = HashMap::new();
        let index_offsets = HashMap::new();

        let position_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            size_of::<[f32; 3]>() as vk::DeviceSize * 10 /* distinct meshes */ * 30_000 /* unique vertices */
        );
        let normal_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            size_of::<[f32; 3]>() as vk::DeviceSize * 10 /* distinct meshes */ * 30_000 /* unique vertices */
        );
        let uv_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            size_of::<[f32; 2]>() as vk::DeviceSize * 10 /* distinct meshes */ * 30_000 /* unique vertices */
        );
        let index_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            size_of::<u32>() as vk::DeviceSize * 3 /* indices per triangle */ * 10 /* distinct meshes */ * 30_000 /* unique triangles */
        );
        let sync_point_fence = renderer.device.new_fence();
        renderer.device.set_object_name(
            sync_point_fence.handle,
            "Consolidate vertex buffers sync point fence",
        );
        drop(renderer);

        res.insert(ConsolidatedMeshBuffers {
            vertex_offsets,
            next_vertex_offset: 0,
            index_offsets,
            next_index_offset: 0,
            position_buffer,
            normal_buffer,
            uv_buffer,
            index_buffer,
            sync_point: None,
            previous_run_command_buffer: None,
            sync_point_fence,
        });
    }
}

impl<'a> System<'a> for ConsolidateMeshBuffers {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        WriteExpect<'a, RenderFrame>,
        ReadStorage<'a, GltfMesh>,
        Write<'a, ConsolidatedMeshBuffers, ConsolidatedMeshBuffers>,
    );

    fn run(&mut self, (renderer, meshes, mut consolidate_mesh_buffers): Self::SystemData) {
        if consolidate_mesh_buffers
            .previous_run_command_buffer
            .is_some()
        {
            unsafe {
                renderer
                    .device
                    .wait_for_fences(
                        &[consolidate_mesh_buffers.sync_point_fence.handle],
                        true,
                        u64::MAX,
                    )
                    .expect("Wait for fence failed.");
            }
        }
        unsafe {
            renderer
                .device
                .reset_fences(&[consolidate_mesh_buffers.sync_point_fence.handle])
                .expect("failed to reset consolidate vertex buffers sync point fence");
        }

        let mut needs_transfer = false;
        let command_buffer = renderer
            .graphics_command_pool
            .record_one_time(|command_buffer| {
                for mesh in (&meshes).join() {
                    let ConsolidatedMeshBuffers {
                        ref mut next_vertex_offset,
                        ref mut next_index_offset,
                        ref position_buffer,
                        ref normal_buffer,
                        ref uv_buffer,
                        ref index_buffer,
                        ref mut vertex_offsets,
                        ref mut index_offsets,
                        ..
                    } = *consolidate_mesh_buffers;

                    if let Entry::Vacant(v) =
                        vertex_offsets.entry(mesh.vertex_buffer.handle.as_raw())
                    {
                        v.insert(*next_vertex_offset);
                        let size_3 = mesh.vertex_len * size_of::<[f32; 3]>() as vk::DeviceSize;
                        let size_2 = mesh.vertex_len * size_of::<[f32; 2]>() as vk::DeviceSize;
                        let offset_3 =
                            *next_vertex_offset * size_of::<[f32; 3]>() as vk::DeviceSize;
                        let offset_2 =
                            *next_vertex_offset * size_of::<[f32; 2]>() as vk::DeviceSize;

                        unsafe {
                            // vertex
                            renderer.device.cmd_copy_buffer(
                                command_buffer,
                                mesh.vertex_buffer.handle,
                                position_buffer.handle,
                                &[vk::BufferCopy::builder()
                                    .size(size_3)
                                    .dst_offset(offset_3)
                                    .build()],
                            );
                            // normal
                            renderer.device.cmd_copy_buffer(
                                command_buffer,
                                mesh.normal_buffer.handle,
                                normal_buffer.handle,
                                &[vk::BufferCopy::builder()
                                    .size(size_3)
                                    .dst_offset(offset_3)
                                    .build()],
                            );
                            // uv
                            renderer.device.cmd_copy_buffer(
                                command_buffer,
                                mesh.uv_buffer.handle,
                                uv_buffer.handle,
                                &[vk::BufferCopy::builder()
                                    .size(size_2)
                                    .dst_offset(offset_2)
                                    .build()],
                            );
                        }
                        *next_vertex_offset += mesh.vertex_len;
                        needs_transfer = true;
                    }

                    if let Entry::Vacant(v) = index_offsets.entry(mesh.index_buffer.handle.as_raw())
                    {
                        v.insert(*next_index_offset);

                        unsafe {
                            renderer.device.cmd_copy_buffer(
                                command_buffer,
                                mesh.index_buffer.handle,
                                index_buffer.handle,
                                &[vk::BufferCopy::builder()
                                    .size(mesh.index_len * size_of::<u32>() as vk::DeviceSize)
                                    .dst_offset(
                                        *next_index_offset * size_of::<u32>() as vk::DeviceSize,
                                    )
                                    .build()],
                            );
                        }
                        *next_index_offset += mesh.index_len;
                        needs_transfer = true;
                    }
                }
            });

        if needs_transfer {
            let semaphore = renderer.device.new_semaphore();
            renderer.device.set_object_name(
                semaphore.handle,
                "Consolidate Mesh buffers sync point semaphore",
            );
            let command_buffers = &[*command_buffer];
            let signal_semaphores = &[semaphore.handle];
            let submit = vk::SubmitInfo::builder()
                .command_buffers(command_buffers)
                .signal_semaphores(signal_semaphores)
                .build();

            consolidate_mesh_buffers.sync_point = Some(semaphore);
            consolidate_mesh_buffers.previous_run_command_buffer = Some(command_buffer); // potentially destroys the previous one

            let queue = renderer.device.graphics_queue.lock();

            unsafe {
                renderer
                    .device
                    .queue_submit(
                        *queue,
                        &[submit],
                        consolidate_mesh_buffers.sync_point_fence.handle,
                    )
                    .unwrap();
            }
        } else {
            consolidate_mesh_buffers.sync_point = None;
            consolidate_mesh_buffers.previous_run_command_buffer = None; // potentially destroys the previous one
        }
    }
}
