use ash::{version::DeviceV1_0, vk};
use futures::executor::block_on;
use gltf_importer;
use gltf_utils::PrimitiveIterators;
use std::{mem::{size_of, transmute}, sync::Arc};

use super::{alloc, new_buffer, one_time_submit_cb, Buffer, RenderFrame};

pub fn load(renderer: &RenderFrame, path: &str) -> (Arc<Buffer>, Arc<Buffer>, Arc<Buffer>, u64) {
    let importer = gltf_importer::import(path);
    let (loaded, buffers) = importer.unwrap();
    // let scene = loaded.scenes().next().unwrap();
    // let node = scene.nodes().next().unwrap();
    let mesh = loaded.meshes().next().unwrap();
    let primitive = mesh.primitives().next().unwrap();
    let positions = primitive.positions(&buffers).unwrap();
    let normals = primitive.normals(&buffers).unwrap();
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
    renderer.device.set_object_name(
        vk::ObjectType::BUFFER,
        unsafe { transmute::<_, u64>(vertex_buffer.handle) },
        "Gltf mesh Vertex buffer",
    );
    let vertex_upload_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::TRANSFER_SRC,
        alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        vertex_size,
    );
    renderer.device.set_object_name(
        vk::ObjectType::BUFFER,
        unsafe { transmute::<_, u64>(vertex_upload_buffer.handle) },
        "Gltf mesh Vertex upload buffer",
    );
    unsafe {
        let p = vertex_upload_buffer.allocation_info.pMappedData as *mut [f32; 3];
        for (ix, data) in positions.enumerate() {
            *p.offset(ix as isize) = data;
        }
    }
    let normal_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        alloc::VmaAllocationCreateFlagBits(0),
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        normals_size,
    );
    renderer.device.set_object_name(
        vk::ObjectType::BUFFER,
        unsafe { transmute::<_, u64>(normal_buffer.handle) },
        "Gltf mesh Normal buffer",
    );
    let normal_upload_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::TRANSFER_SRC,
        alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        normals_size,
    );
    renderer.device.set_object_name(
        vk::ObjectType::BUFFER,
        unsafe { transmute::<_, u64>(normal_upload_buffer.handle) },
        "Gltf mesh Normal upload buffer",
    );
    unsafe {
        let p = normal_upload_buffer.allocation_info.pMappedData as *mut [f32; 3];
        for (ix, data) in normals.enumerate() {
            *p.offset(ix as isize) = data;
        }
    }
    let indices = PrimitiveIterators::indices(&primitive, &buffers)
        .unwrap()
        .into_u32();
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
    renderer.device.set_object_name(
        vk::ObjectType::BUFFER,
        unsafe { transmute::<_, u64>(index_buffer.handle) },
        "Gltf mesh index buffer",
    );
    let index_upload_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::TRANSFER_SRC,
        alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        index_size,
    );
    renderer.device.set_object_name(
        vk::ObjectType::BUFFER,
        unsafe { transmute::<_, u64>(index_upload_buffer.handle) },
        "Gltf mesh index upload buffer",
    );
    unsafe {
        let p = index_upload_buffer.allocation_info.pMappedData as *mut u32;
        for (ix, data) in indices.enumerate() {
            *p.offset(ix as isize) = data;
        }
    }
    let upload = one_time_submit_cb(
        Arc::clone(&renderer.graphics_command_pool),
        Arc::clone(&renderer.device.graphics_queue),
        {
            let vertex_buffer = Arc::clone(&vertex_buffer);
            let vertex_upload_buffer = Arc::clone(&vertex_upload_buffer);
            let normal_buffer = Arc::clone(&normal_buffer);
            let normal_upload_buffer = Arc::clone(&normal_upload_buffer);
            let index_buffer = Arc::clone(&index_buffer);
            let index_upload_buffer = Arc::clone(&index_upload_buffer);
            let device = Arc::clone(&renderer.device);
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
        },
    );

    block_on(upload).unwrap();

    (vertex_buffer, normal_buffer, index_buffer, index_len)
}
