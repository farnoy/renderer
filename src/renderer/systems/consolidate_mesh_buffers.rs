use std::{mem::size_of, u64};

use ash::{
    version::DeviceV1_0,
    vk::{self, Handle},
};
use bevy_ecs::prelude::*;
use hashbrown::{hash_map::Entry, HashMap};
use microprofile::scope;

use crate::renderer::{
    device::{Buffer, Device, DoubleBuffered, StrictCommandPool, VmaMemoryUsage},
    frame_graph,
    shaders::cull_set,
    GltfMesh, ImageIndex, RenderFrame, RenderStage, SwapchainIndexToFrameNumber,
};

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
    pub(crate) position_buffer: cull_set::bindings::vertex_buffer::Buffer,
    /// Stores normal data for each mesh
    pub(crate) normal_buffer: cull_set::bindings::vertex_buffer::Buffer,
    /// Stores uv data for each mesh
    pub(crate) uv_buffer: Buffer,
    /// Stores index data for each mesh
    pub(crate) index_buffer: cull_set::bindings::index_buffer::Buffer,
    command_pools: DoubleBuffered<StrictCommandPool>,
    command_buffers: DoubleBuffered<vk::CommandBuffer>,
}

renderer_macros::define_timeline!(pub(crate) ConsolidateTimeline [Perform]);

/// Identifies distinct GLTF meshes in components and copies them to a shared buffer
pub(crate) fn consolidate_mesh_buffers(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<SwapchainIndexToFrameNumber>,
    mut consolidated_mesh_buffers: ResMut<ConsolidatedMeshBuffers>,
    query: Query<&GltfMesh>,
) {
    scope!("ecs", "consolidate mesh buffers");

    let ConsolidatedMeshBuffers {
        ref mut next_vertex_offset,
        ref mut next_index_offset,
        ref position_buffer,
        ref normal_buffer,
        ref uv_buffer,
        ref index_buffer,
        ref mut vertex_offsets,
        ref mut index_offsets,
        ref mut command_pools,
        ref command_buffers,
    } = *consolidated_mesh_buffers;

    // Wait until we can reuse the command pool
    renderer
        .consolidate_timeline_semaphore
        .wait(
            &renderer.device,
            ConsolidateTimeline::Perform.as_of_previous(&image_index, &swapchain_index_map),
        )
        .unwrap();

    let command_pool = command_pools.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let mut command_session = command_pool.session(&renderer.device);

    let mut needs_transfer = false;
    let command_buffer = command_session.record_to_specific(*command_buffers.current(image_index.0));
    for mesh in &mut query.iter() {
        if let Entry::Vacant(v) = vertex_offsets.entry(mesh.vertex_buffer.handle.as_raw()) {
            debug_assert!(
                *next_vertex_offset + mesh.vertex_len
                    < (size_of::<cull_set::bindings::vertex_buffer::T>() / size_of::<[f32; 3]>()) as u64,
            );
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
                    position_buffer.buffer.handle,
                    &[vk::BufferCopy::builder().size(size_3).dst_offset(offset_3).build()],
                );
                // normal
                renderer.device.cmd_copy_buffer(
                    *command_buffer,
                    mesh.normal_buffer.handle,
                    normal_buffer.buffer.handle,
                    &[vk::BufferCopy::builder().size(size_3).dst_offset(offset_3).build()],
                );
                // uv
                renderer
                    .device
                    .cmd_copy_buffer(*command_buffer, mesh.uv_buffer.handle, uv_buffer.handle, &[
                        vk::BufferCopy::builder().size(size_2).dst_offset(offset_2).build(),
                    ]);
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
                        index_buffer.buffer.handle,
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
    let command_buffers = if needs_transfer { vec![*command_buffer] } else { vec![] };

    let queue = renderer.device.graphics_queue().lock();

    frame_graph::ConsolidateMeshBuffers::Stage::queue_submit(&image_index, &renderer, *queue, &command_buffers)
        .unwrap();

    // TODO: host_signal is too quick in optimized builds, surpassing last frame's async signal
    // operation       need to do an empty submit to wait for last frame first, then signal
    // renderpass3::ConsolidateMeshBuffers::host_signal(&renderer).unwrap();
}

// TODO: dynamic unloading of meshes
// TODO: use actual transfer queue for the transfers

impl ConsolidatedMeshBuffers {
    pub(crate) fn new(renderer: &RenderFrame) -> ConsolidatedMeshBuffers {
        let vertex_offsets = HashMap::new();
        let index_offsets = HashMap::new();

        let position_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER, // storage_buffer not needed?
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        );
        renderer
            .device
            .set_object_name(position_buffer.buffer.handle, "Consolidated position buffer");
        let normal_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        );
        renderer
            .device
            .set_object_name(normal_buffer.buffer.handle, "Consolidated normals buffer");
        let uv_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            size_of::<super::super::shaders::UVBuffer>() as vk::DeviceSize,
        );
        renderer
            .device
            .set_object_name(uv_buffer.handle, "Consolidated UV buffer");
        let index_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        );
        renderer
            .device
            .set_object_name(index_buffer.buffer.handle, "Consolidated Index buffer");
        let mut command_pools = renderer.new_buffered(|ix| {
            StrictCommandPool::new(
                &renderer.device,
                renderer.device.graphics_queue_family,
                &format!("Consolidate Mesh Buffers Command Pool[{}]", ix),
            )
        });
        let command_buffers = renderer.new_buffered(|ix| {
            command_pools
                .current_mut(ix)
                .allocate(&format!("Consolidate Mesh Buffers CB[{}]", ix), &renderer.device)
        });

        ConsolidatedMeshBuffers {
            vertex_offsets,
            next_vertex_offset: 0,
            index_offsets,
            next_index_offset: 0,
            position_buffer,
            normal_buffer,
            uv_buffer,
            index_buffer,
            command_pools,
            command_buffers,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.position_buffer.destroy(device);
        self.normal_buffer.destroy(device);
        self.uv_buffer.destroy(device);
        self.index_buffer.destroy(device);
        self.command_pools.into_iter().for_each(|p| p.destroy(device));
    }
}
