use crate::{
    define_timeline,
    renderer::{
        alloc,
        device::{Buffer, TimelineSemaphore},
        timeline_value, timeline_value_last, timeline_value_previous, GltfMesh, ImageIndex,
        LocalGraphicsCommandPool, RenderFrame,
    },
};
use ash::{
    version::DeviceV1_0,
    vk::{self, Handle},
};
use bevy_ecs::prelude::*;
use hashbrown::{hash_map::Entry, HashMap};
#[cfg(feature = "microprofile")]
use microprofile::scope;
use std::{mem::size_of, u64};

/// Describes layout of gltf mesh vertex data in a shared buffer
pub(crate) struct ConsolidatedMeshBuffers {
    /// Maps from vertex buffer handle to offset within the consolidated buffer
    /// Hopefully this is distinct enough
    pub(crate) vertex_offsets: HashMap<u64, vk::DeviceSize>,
    /// Next free vertex offset in the buffer that can be used for a new mesh
    next_vertex_offset: vk::DeviceSize,
    /// Maps from index buffer handle to offset within the consolidated buffer
    pub(crate) index_offsets: HashMap<u64, vk::DeviceSize>,
    /// Next free index offset in the buffer that can be used for a new mesh
    next_index_offset: vk::DeviceSize,
    /// Stores position data for each mesh
    pub(crate) position_buffer: Buffer,
    /// Stores normal data for each mesh
    pub(crate) normal_buffer: Buffer,
    /// Stores uv data for each mesh
    pub(crate) uv_buffer: Buffer,
    /// Stores index data for each mesh
    pub(crate) index_buffer: Buffer,
    /// If this semaphore is present, a modification to the consolidated buffer has happened
    /// and the user must synchronize with it
    pub(crate) sync_timeline: TimelineSemaphore,
    /// Pool for commands recorded as part of this pass
    command_pool: LocalGraphicsCommandPool,
}

define_timeline!(sync Consolidate);

/// Identifies distinct GLTF meshes in components and copies them to a shared buffer
pub(crate) fn consolidate_mesh_buffers(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    mut consolidated_mesh_buffers: ResMut<ConsolidatedMeshBuffers>,
    query: Query<&GltfMesh>,
) {
    #[cfg(feature = "profiling")]
    microprofile::scope!("ecs", "consolidate mesh buffers");

    let ConsolidatedMeshBuffers {
        ref mut next_vertex_offset,
        ref mut next_index_offset,
        ref position_buffer,
        ref normal_buffer,
        ref uv_buffer,
        ref index_buffer,
        ref mut vertex_offsets,
        ref mut index_offsets,
        ref mut command_pool,
        ref sync_timeline,
        ..
    } = *consolidated_mesh_buffers;

    {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("consolidate mesh buffers", "wait previous");
        sync_timeline
            .wait(timeline_value_previous::<_, sync::Consolidate>(
                &image_index,
                &renderer,
            ))
            .unwrap();
    }

    let command_pool = command_pool.pools.current_mut(image_index.0);

    unsafe {
        #[cfg(feature = "microprofile")]
        microprofile::scope!("consolidate mesh buffers", "CP reset");
        command_pool.reset();
    }

    let mut command_session = command_pool.session();

    let mut needs_transfer = false;
    let command_buffer = command_session.record_one_time("consolidate mesh buffers cb");
    for mesh in &mut query.iter() {
        if let Entry::Vacant(v) = vertex_offsets.entry(mesh.vertex_buffer.handle.as_raw()) {
            v.insert(*next_vertex_offset);
            let size_3 = mesh.vertex_len * size_of::<[f32; 3]>() as vk::DeviceSize;
            let size_2 = mesh.vertex_len * size_of::<[f32; 2]>() as vk::DeviceSize;
            let offset_3 = *next_vertex_offset * size_of::<[f32; 3]>() as vk::DeviceSize;
            let offset_2 = *next_vertex_offset * size_of::<[f32; 2]>() as vk::DeviceSize;

            unsafe {
                // vertex
                renderer.device.cmd_copy_buffer(
                    *command_buffer,
                    mesh.vertex_buffer.handle,
                    position_buffer.handle,
                    &[vk::BufferCopy::builder()
                        .size(size_3)
                        .dst_offset(offset_3)
                        .build()],
                );
                // normal
                renderer.device.cmd_copy_buffer(
                    *command_buffer,
                    mesh.normal_buffer.handle,
                    normal_buffer.handle,
                    &[vk::BufferCopy::builder()
                        .size(size_3)
                        .dst_offset(offset_3)
                        .build()],
                );
                // uv
                renderer.device.cmd_copy_buffer(
                    *command_buffer,
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

        for (lod_index_buffer, index_len) in mesh.index_buffers.iter() {
            if let Entry::Vacant(v) = index_offsets.entry(lod_index_buffer.handle.as_raw()) {
                v.insert(*next_index_offset);

                unsafe {
                    renderer.device.cmd_copy_buffer(
                        *command_buffer,
                        lod_index_buffer.handle,
                        index_buffer.handle,
                        &[vk::BufferCopy::builder()
                            .size(index_len * size_of::<u32>() as vk::DeviceSize)
                            .dst_offset(*next_index_offset * size_of::<u32>() as vk::DeviceSize)
                            .build()],
                    );
                }
                *next_index_offset += index_len;
                needs_transfer = true;
            }
        }
    }
    let command_buffer = command_buffer.end();

    if needs_transfer {
        let command_buffers = &[*command_buffer];
        let signal_semaphores = &[sync_timeline.handle];
        let signal_semaphore_values = &[timeline_value::<_, sync::Consolidate>(
            renderer.frame_number,
        )];
        let mut wait_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
            .signal_semaphore_values(signal_semaphore_values);
        let submit = vk::SubmitInfo::builder()
            .push_next(&mut wait_timeline)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores)
            .build();

        let queue = renderer.device.graphics_queue.lock();

        unsafe {
            renderer
                .device
                .queue_submit(*queue, &[submit], vk::Fence::null())
                .unwrap();
        }
    } else {
        sync_timeline
            .signal(timeline_value::<_, sync::Consolidate>(
                renderer.frame_number,
            ))
            .unwrap();
    }
}

// TODO: dynamic unloading of meshes
// TODO: use actual transfer queue for the transfers

impl ConsolidatedMeshBuffers {
    pub(crate) fn new(renderer: &RenderFrame) -> ConsolidatedMeshBuffers {
        let vertex_offsets = HashMap::new();
        let index_offsets = HashMap::new();

        let position_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            super::super::shaders::cull_set::bindings::vertex_buffer::SIZE,
        );
        let normal_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            super::super::shaders::cull_set::bindings::vertex_buffer::SIZE,
        );
        let uv_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            size_of::<super::super::shaders::UVBuffer>() as vk::DeviceSize,
        );
        let index_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            super::super::shaders::cull_set::bindings::index_buffer::SIZE,
        );
        let sync_timeline = renderer
            .device
            .new_semaphore_timeline(timeline_value_last::<_, sync::Consolidate>(
                renderer.frame_number,
            ));
        renderer.device.set_object_name(
            sync_timeline.handle,
            "Consolidate mesh buffers sync timeline",
        );
        let sync_point_fence = renderer.device.new_fence();
        renderer.device.set_object_name(
            sync_point_fence.handle,
            "Consolidate vertex buffers sync point fence",
        );

        let command_pool = LocalGraphicsCommandPool::new(&renderer);

        ConsolidatedMeshBuffers {
            vertex_offsets,
            next_vertex_offset: 0,
            index_offsets,
            next_index_offset: 0,
            position_buffer,
            normal_buffer,
            uv_buffer,
            index_buffer,
            sync_timeline,
            command_pool,
        }
    }
}
