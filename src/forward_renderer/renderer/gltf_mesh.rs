use ash::{version::DeviceV1_0, vk};
use cgmath;
use gltf;
use meshopt;
use std::{mem::size_of, sync::Arc, u64};

use super::{alloc, new_buffer, Buffer, RenderFrame};

pub struct LoadedMesh {
    pub vertex_buffer: Buffer,
    pub normal_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_len: u64,
    pub aabb_c: cgmath::Vector3<f32>,
    pub aabb_h: cgmath::Vector3<f32>,
}

#[derive(Clone, Default)]
struct Pos(pub [f32; 3]);

impl meshopt::DecodePosition for Pos {
    fn decode_position(&self) -> [f32; 3] {
        self.0
    }
}

pub fn load(renderer: &RenderFrame, path: &str) -> LoadedMesh {
    let (loaded, buffers, _images) = gltf::import(path).expect("Failed loading mesh");
    let mesh = loaded.meshes().next().unwrap();
    let primitive = mesh.primitives().next().unwrap();
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let positions = reader
        .read_positions()
        .unwrap()
        .map(Pos)
        .collect::<Vec<_>>();
    let bounding_box = primitive.bounding_box();
    let aabb_c =
        (cgmath::Vector3::from(bounding_box.max) + cgmath::Vector3::from(bounding_box.min)) / 2.0;
    let aabb_h =
        (cgmath::Vector3::from(bounding_box.max) - cgmath::Vector3::from(bounding_box.min)) / 2.0;
    let normals = reader.read_normals().unwrap().collect::<Vec<_>>();
    let mut indices = reader
        .read_indices()
        .unwrap()
        .into_u32()
        .collect::<Vec<_>>();
    meshopt::optimize_vertex_cache_in_place(&mut indices, positions.len());
    meshopt::optimize_overdraw_in_place(&mut indices, &positions, 1.05);
    let remap = meshopt::optimize_vertex_fetch_remap(&indices, positions.len());
    let indices_new = meshopt::remap_index_buffer(Some(&indices), positions.len(), &remap);
    let positions_new = meshopt::remap_vertex_buffer(&positions, positions.len(), &remap);
    let normals_new = meshopt::remap_vertex_buffer(&normals, positions.len(), &remap);

    // shadow with optimized buffers
    let indices = indices_new;
    let normals = normals_new;
    let positions = positions_new;

    let vertex_len = positions.len() as u64;
    let vertex_size = size_of::<f32>() as u64 * 3 * vertex_len;
    let normals_size = size_of::<f32>() as u64 * 3 * vertex_len;
    let vertex_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        alloc::VmaAllocationCreateFlagBits(0),
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        vertex_size,
    );
    renderer
        .device
        .set_object_name(vertex_buffer.handle, "Gltf mesh Vertex buffer");
    let vertex_upload_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::TRANSFER_SRC,
        alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        vertex_size,
    );
    renderer.device.set_object_name(
        vertex_upload_buffer.handle,
        "Gltf mesh Vertex upload buffer",
    );
    unsafe {
        let p = vertex_upload_buffer.allocation_info.pMappedData as *mut [f32; 3];
        for (ix, data) in positions.iter().enumerate() {
            *p.add(ix) = data.0;
        }
    }
    let normal_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        alloc::VmaAllocationCreateFlagBits(0),
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        normals_size,
    );
    renderer
        .device
        .set_object_name(normal_buffer.handle, "Gltf mesh Normal buffer");
    let normal_upload_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::TRANSFER_SRC,
        alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        normals_size,
    );
    renderer.device.set_object_name(
        normal_upload_buffer.handle,
        "Gltf mesh Normal upload buffer",
    );
    unsafe {
        let p = normal_upload_buffer.allocation_info.pMappedData as *mut [f32; 3];
        for (ix, data) in normals.iter().enumerate() {
            *p.add(ix) = *data;
        }
    }
    let index_len = indices.len() as u64;
    let index_size = size_of::<u32>() as u64 * index_len;
    let index_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        alloc::VmaAllocationCreateFlagBits(0),
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        index_size,
    );
    renderer
        .device
        .set_object_name(index_buffer.handle, "Gltf mesh index buffer");
    let index_upload_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::TRANSFER_SRC,
        alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        index_size,
    );
    renderer
        .device
        .set_object_name(index_upload_buffer.handle, "Gltf mesh index upload buffer");
    unsafe {
        let p = index_upload_buffer.allocation_info.pMappedData as *mut u32;
        for (ix, data) in indices.iter().enumerate() {
            *p.add(ix) = *data;
        }
    }
    let upload = renderer.graphics_command_pool.record_one_time({
        let vertex_buffer = &vertex_buffer;
        let vertex_upload_buffer = &vertex_upload_buffer;
        let normal_buffer = &normal_buffer;
        let normal_upload_buffer = &normal_upload_buffer;
        let index_buffer = &index_buffer;
        let index_upload_buffer = &index_upload_buffer;
        let device = &renderer.device;
        move |command_buffer| unsafe {
            device.device.cmd_copy_buffer(
                command_buffer,
                vertex_upload_buffer.handle,
                vertex_buffer.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: vertex_buffer.allocation_info.size,
                }],
            );
            device.device.cmd_copy_buffer(
                command_buffer,
                normal_upload_buffer.handle,
                normal_buffer.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: normal_buffer.allocation_info.size,
                }],
            );
            device.device.cmd_copy_buffer(
                command_buffer,
                index_upload_buffer.handle,
                index_buffer.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: index_buffer.allocation_info.size,
                }],
            );
        }
    });
    let mut graphics_queue = renderer.device.graphics_queue.lock();
    let upload_fence = upload.submit_once(&mut *graphics_queue, "upload gltf mesh commands");
    unsafe {
        renderer
            .device
            .wait_for_fences(&[upload_fence.handle], true, u64::MAX)
            .expect("Wait for fence failed.");
    }

    LoadedMesh {
        vertex_buffer,
        normal_buffer,
        index_buffer,
        index_len,
        aabb_c,
        aabb_h,
    }
}
