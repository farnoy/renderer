#[macro_use]
extern crate ash;
extern crate cgmath;
extern crate futures;
extern crate gltf;
extern crate gltf_importer;
extern crate gltf_utils;
pub extern crate internal_alloc;
extern crate rayon;
extern crate specs;
#[macro_use]
extern crate specs_derive;
extern crate time;
#[cfg(windows)]
extern crate winapi;
extern crate winit;

mod forward_renderer;

use ash::{version::DeviceV1_0, vk};
use cgmath::Rotation3;
use forward_renderer::{alloc, components::*, helpers::*, renderer::*, systems::*};
use futures::executor::block_on;
use gltf_utils::PrimitiveIterators;
use specs::Builder;
use std::{mem::size_of, ptr, sync::Arc, u64};
use winit::{Event, KeyboardInput, WindowEvent};

fn main() {
    let mut world = specs::World::new();
    world.register::<Position>();
    world.register::<Rotation>();
    world.register::<Scale>();
    world.register::<Matrices>();
    world.register::<GltfMesh>();
    world.register::<GltfMeshBufferIndex>();
    let rayon_threadpool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap(),
    );
    /*
    multithreading example
    threadpool.run(lazy(move |_| {
        println!("start parent");
        let lhs =
            spawn_with_handle(lazy(move |_| {
                println!("start lhs");
                std::thread::sleep_ms(5000);
                println!("end rhs");
                future::ok::<u32, Never>(10)
            }));
        let rhs =
            spawn_with_handle(lazy(move |_| {
                println!("start lhs");
                std::thread::sleep_ms(5000);
                println!("end rhs");
                future::ok::<u32, Never>(10)
            }));
        lhs.join(rhs).and_then(|(lhs, rhs)| {
            lhs.join(rhs).and_then(|(l, r)| {
                println!("end parent, results are ({}, {})", l, r);
                future::ok::<u32, Never>(5)
            })
        })
    })).unwrap();
    */

    let (mut renderer, mut events_loop) = RenderFrame::new();

    let (vertex_buffer, normal_buffer, index_buffer, index_len) = {
        // Mesh load
        let path = "glTF-Sample-Models/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf";
        // let path = "glTF-Sample-Models/2.0/Box/glTF/Box.gltf";
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
            vk::BUFFER_USAGE_VERTEX_BUFFER_BIT
                | vk::BUFFER_USAGE_TRANSFER_DST_BIT
                | vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
            alloc::VmaAllocationCreateFlagBits(0),
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            vertex_size,
        );
        let vertex_upload_buffer = new_buffer(
            Arc::clone(&renderer.device),
            vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            vertex_size,
        );
        unsafe {
            let p = vertex_upload_buffer.allocation_info.pMappedData as *mut [f32; 3];
            for (ix, data) in positions.enumerate() {
                *p.offset(ix as isize) = data;
            }
        }
        let normal_buffer = new_buffer(
            Arc::clone(&renderer.device),
            vk::BUFFER_USAGE_VERTEX_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            alloc::VmaAllocationCreateFlagBits(0),
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            normals_size,
        );
        let normal_upload_buffer = new_buffer(
            Arc::clone(&renderer.device),
            vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            normals_size,
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
            vk::BUFFER_USAGE_INDEX_BUFFER_BIT
                | vk::BUFFER_USAGE_TRANSFER_DST_BIT
                | vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
            alloc::VmaAllocationCreateFlagBits(0),
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            index_size,
        );
        let index_upload_buffer = new_buffer(
            Arc::clone(&renderer.device),
            vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
            alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            index_size,
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
    };

    let culled_index_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BUFFER_USAGE_INDEX_BUFFER_BIT | vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
        alloc::VmaAllocationCreateFlagBits(0),
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        size_of::<u32>() as vk::DeviceSize * index_len * 900,
    );

    {
        let buffer_updates = &[
            vk::DescriptorBufferInfo {
                buffer: renderer.culled_commands_buffer.handle,
                offset: 0,
                range: vk::VK_WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: index_buffer.handle,
                offset: 0,
                range: vk::VK_WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vertex_buffer.handle,
                offset: 0,
                range: vk::VK_WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: culled_index_buffer.handle,
                offset: 0,
                range: vk::VK_WHOLE_SIZE,
            },
        ];
        unsafe {
            renderer.device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WriteDescriptorSet,
                    p_next: ptr::null(),
                    dst_set: renderer.cull_set.handle,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: buffer_updates.len() as u32,
                    descriptor_type: vk::DescriptorType::StorageBuffer,
                    p_image_info: ptr::null(),
                    p_buffer_info: buffer_updates.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                }],
                &[],
            );
        }
    }

    renderer.culled_index_buffer = Some(culled_index_buffer);

    let projection = cgmath::perspective(
        cgmath::Deg(60.0),
        renderer.instance.window_width as f32 / renderer.instance.window_height as f32,
        0.1,
        100.0,
    );
    let view = cgmath::Matrix4::look_at(
        cgmath::Point3::new(0.0, 1.0, -2.0),
        cgmath::Point3::new(0.0, 0.0, 0.0),
        cgmath::vec3(0.0, -1.0, 0.0),
    );
    for ix in 0..900 {
        world
            .create_entity()
            .with::<Position>(Position(cgmath::Vector3::new(
                ((ix % 3) * 2 - 2) as f32,
                0.0,
                2.0 + ix as f32,
            )))
            .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_x(cgmath::Deg(0.0))))
            .with::<Scale>(Scale(0.6))
            .with::<Matrices>(Matrices::one())
            .with::<GltfMesh>(GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                index_buffer: Arc::clone(&index_buffer),
                index_len,
            })
            .with::<GltfMeshBufferIndex>(GltfMeshBufferIndex(0))
            .build();
    }
    let mut dispatcher = specs::DispatcherBuilder::new()
        .with_pool(Arc::clone(&rayon_threadpool))
        .with(SteadyRotation, "steady_rotation", &[])
        .with(AssignBufferIndex, "assign_buffer_index", &[])
        .with(
            MVPCalculation { projection, view },
            "mvp",
            &["steady_rotation"],
        )
        .with(
            MVPUpload {
                dst_mvp: Arc::clone(&renderer.ubo_buffer),
                dst_model: Arc::clone(&renderer.model_buffer),
            },
            "mvp_upload",
            &["mvp"],
        )
        .with(
            renderer,
            "render_frame",
            &["assign_buffer_index", "mvp_upload"],
        )
        .build();

    'frame: loop {
        let mut quit = false;
        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(w, h),
                ..
            } => {
                println!("The window was resized to {}x{}", w, h);
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            }
            | Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(winit::VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                quit = true;
            }
            _ => (),
        });
        if quit {
            break 'frame;
        }
        dispatcher.dispatch(&world.res);
        world.maintain();
    }
}
