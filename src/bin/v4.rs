extern crate ash;
extern crate cgmath;
extern crate forward_renderer;
extern crate futures;
extern crate gltf;
extern crate gltf_importer;
extern crate gltf_utils;
extern crate specs;
extern crate winit;

use ash::{version::DeviceV1_0, vk};
use cgmath::Rotation3;
use forward_renderer::{alloc, ecs::*, helpers::*};
use futures::{
    executor::{block_on, spawn, spawn_with_handle, ThreadPool}, future::{self, lazy}, prelude::*,
};
use gltf_utils::PrimitiveIterators;
use std::{default::Default, mem::size_of, path::PathBuf, ptr, sync::Arc, u64};
use winit::{Event, KeyboardInput, WindowEvent};

fn main() {
    let mut world = specs::World::new();
    world.register::<Position>();
    world.register::<Rotation>();
    world.register::<Scale>();
    world.register::<Matrices>();
    let mut threadpool = ThreadPool::builder().pool_size(4).create().unwrap();
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
    let (instance, mut events_loop) = new_window(1920, 1080);
    let device = new_device(&instance);
    let swapchain = new_swapchain(&instance, &device);
    let present_semaphore = new_semaphore(Arc::clone(&device));
    let rendering_complete_semaphore = new_semaphore(Arc::clone(&device));
    let graphics_command_pool = new_command_pool(
        Arc::clone(&device),
        device.graphics_queue_family,
        vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    );
    let main_renderpass = setup_renderpass(Arc::clone(&device), &swapchain);
    let framebuffer =
        setup_framebuffer(&instance, Arc::clone(&device), &swapchain, &main_renderpass);

    let descriptor_pool = new_descriptor_pool(
        Arc::clone(&device),
        30,
        &[
            vk::DescriptorPoolSize {
                typ: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1024,
            },
            vk::DescriptorPoolSize {
                typ: vk::DescriptorType::StorageBuffer,
                descriptor_count: 1024,
            },
        ],
    );

    let command_generation_descriptor_set_layout = new_descriptor_set_layout(
        Arc::clone(&device),
        &[
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::StorageBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::StorageBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::StorageBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::StorageBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 4,
                descriptor_type: vk::DescriptorType::StorageBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_COMPUTE_BIT | vk::SHADER_STAGE_VERTEX_BIT,
                p_immutable_samplers: ptr::null(),
            },
        ],
    );
    let descriptor_set_layout = new_descriptor_set_layout(
        Arc::clone(&device),
        &[vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UniformBuffer,
            descriptor_count: 1,
            stage_flags: vk::SHADER_STAGE_VERTEX_BIT | vk::SHADER_STAGE_COMPUTE_BIT,
            p_immutable_samplers: ptr::null(),
        }],
    );
    let model_view_set_layout = new_descriptor_set_layout(
        Arc::clone(&device),
        &[vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UniformBuffer,
            descriptor_count: 1,
            stage_flags: vk::SHADER_STAGE_COMPUTE_BIT,
            p_immutable_samplers: ptr::null(),
        }],
    );

    let model_set_layout = new_descriptor_set_layout(
        Arc::clone(&device),
        &[vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UniformBuffer,
            descriptor_count: 1,
            stage_flags: vk::SHADER_STAGE_VERTEX_BIT,
            p_immutable_samplers: ptr::null(),
        }],
    );

    let command_generation_pipeline_layout = new_pipeline_layout(
        Arc::clone(&device),
        &[
            descriptor_set_layout.handle,
            model_view_set_layout.handle,
            command_generation_descriptor_set_layout.handle,
        ],
        &[],
    );
    let command_generation_pipeline = new_compute_pipeline(
        Arc::clone(&device),
        &command_generation_pipeline_layout,
        &PathBuf::from(env!("OUT_DIR")).join("generate_work.comp.spv"),
    );

    let command_generation_descriptor_set = new_descriptor_set(
        Arc::clone(&device),
        Arc::clone(&descriptor_pool),
        &command_generation_descriptor_set_layout,
    );
    let command_generation_buffer = new_buffer(
        Arc::clone(&device),
        vk::BUFFER_USAGE_INDIRECT_BUFFER_BIT | vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
        alloc::VmaAllocationCreateFlagBits(0),
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        size_of::<u32>() as vk::DeviceSize * 5 * 64,
    );

    let triangle_pipeline_layout = new_pipeline_layout(
        Arc::clone(&device),
        &[descriptor_set_layout.handle],
        &[vk::PushConstantRange {
            stage_flags: vk::SHADER_STAGE_VERTEX_BIT,
            offset: 0,
            size: (size_of::<f32>() * 2 * 3) as u32,
        }],
    );
    let triangle_pipeline = new_graphics_pipeline(
        &instance,
        Arc::clone(&device),
        &triangle_pipeline_layout,
        &main_renderpass,
        &[],
        &[],
        &[
            (
                vk::SHADER_STAGE_VERTEX_BIT,
                PathBuf::from(env!("OUT_DIR")).join("triangle.vert.spv"),
            ),
            (
                vk::SHADER_STAGE_FRAGMENT_BIT,
                PathBuf::from(env!("OUT_DIR")).join("triangle.frag.spv"),
            ),
        ],
    );
    let gltf_pipeline_layout = new_pipeline_layout(
        Arc::clone(&device),
        &[descriptor_set_layout.handle, model_set_layout.handle],
        &[vk::PushConstantRange {
            stage_flags: vk::SHADER_STAGE_VERTEX_BIT,
            offset: 0,
            size: size_of::<u32>() as u32,
        }],
    );
    let gltf_pipeline = new_graphics_pipeline(
        &instance,
        Arc::clone(&device),
        &gltf_pipeline_layout,
        &main_renderpass,
        &[
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32g32b32Sfloat,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 1,
                format: vk::Format::R32g32b32Sfloat,
                offset: 0,
            },
        ],
        &[
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: size_of::<f32>() as u32 * 3,
                input_rate: vk::VertexInputRate::Vertex,
            },
            vk::VertexInputBindingDescription {
                binding: 1,
                stride: size_of::<f32>() as u32 * 3,
                input_rate: vk::VertexInputRate::Vertex,
            },
        ],
        &[
            (
                vk::SHADER_STAGE_VERTEX_BIT,
                PathBuf::from(env!("OUT_DIR")).join("gltf_mesh.vert.spv"),
            ),
            (
                vk::SHADER_STAGE_FRAGMENT_BIT,
                PathBuf::from(env!("OUT_DIR")).join("gltf_mesh.frag.spv"),
            ),
        ],
    );
    let ubo_set = new_descriptor_set(
        Arc::clone(&device),
        Arc::clone(&descriptor_pool),
        &descriptor_set_layout,
    );
    let ubo_buffer = new_buffer(
        Arc::clone(&device),
        vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        4 * 4 * 4 * 1024,
    );
    {
        let buffer_updates = &[vk::DescriptorBufferInfo {
            buffer: ubo_buffer.handle,
            offset: 0,
            range: 1024 * size_of::<cgmath::Matrix4<f32>>() as vk::DeviceSize,
        }];
        unsafe {
            device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WriteDescriptorSet,
                    p_next: ptr::null(),
                    dst_set: ubo_set.handle,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UniformBuffer,
                    p_image_info: ptr::null(),
                    p_buffer_info: buffer_updates.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                }],
                &[],
            );
        }
    }
    let model_view_set = new_descriptor_set(
        Arc::clone(&device),
        Arc::clone(&descriptor_pool),
        &model_view_set_layout,
    );
    let model_view_buffer = new_buffer(
        Arc::clone(&device),
        vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        4 * 4 * 4 * 1024,
    );
    {
        let buffer_updates = &[vk::DescriptorBufferInfo {
            buffer: model_view_buffer.handle,
            offset: 0,
            range: 1024 * size_of::<cgmath::Matrix4<f32>>() as vk::DeviceSize,
        }];
        unsafe {
            device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WriteDescriptorSet,
                    p_next: ptr::null(),
                    dst_set: model_view_set.handle,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UniformBuffer,
                    p_image_info: ptr::null(),
                    p_buffer_info: buffer_updates.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                }],
                &[],
            );
        }
    }

    let model_set = new_descriptor_set(
        Arc::clone(&device),
        Arc::clone(&descriptor_pool),
        &model_view_set_layout,
    );
    let model_buffer = new_buffer(
        Arc::clone(&device),
        vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        alloc::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_MAPPED_BIT,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        4 * 4 * 4 * 1024,
    );
    {
        let buffer_updates = &[vk::DescriptorBufferInfo {
            buffer: model_buffer.handle,
            offset: 0,
            range: 1024 * size_of::<cgmath::Matrix4<f32>>() as vk::DeviceSize,
        }];
        unsafe {
            device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WriteDescriptorSet,
                    p_next: ptr::null(),
                    dst_set: model_set.handle,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UniformBuffer,
                    p_image_info: ptr::null(),
                    p_buffer_info: buffer_updates.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                }],
                &[],
            );
        }
    }

    let (vertex_buffer, normal_buffer, vertex_len, index_buffer, index_len) = {
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
            Arc::clone(&device),
            vk::BUFFER_USAGE_VERTEX_BUFFER_BIT
                | vk::BUFFER_USAGE_TRANSFER_DST_BIT
                | vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
            alloc::VmaAllocationCreateFlagBits(0),
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            vertex_size,
        );
        let vertex_upload_buffer = new_buffer(
            Arc::clone(&device),
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
            Arc::clone(&device),
            vk::BUFFER_USAGE_VERTEX_BUFFER_BIT | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            alloc::VmaAllocationCreateFlagBits(0),
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            normals_size,
        );
        let normal_upload_buffer = new_buffer(
            Arc::clone(&device),
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
            Arc::clone(&device),
            vk::BUFFER_USAGE_INDEX_BUFFER_BIT
                | vk::BUFFER_USAGE_TRANSFER_DST_BIT
                | vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
            alloc::VmaAllocationCreateFlagBits(0),
            alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            index_size,
        );
        let index_upload_buffer = new_buffer(
            Arc::clone(&device),
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
            Arc::clone(&graphics_command_pool),
            Arc::clone(&device.graphics_queue),
            {
                let vertex_buffer = Arc::clone(&vertex_buffer);
                let vertex_upload_buffer = Arc::clone(&vertex_upload_buffer);
                let normal_buffer = Arc::clone(&normal_buffer);
                let normal_upload_buffer = Arc::clone(&normal_upload_buffer);
                let index_buffer = Arc::clone(&index_buffer);
                let index_upload_buffer = Arc::clone(&index_upload_buffer);
                let device = Arc::clone(&device);
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

        (
            vertex_buffer,
            normal_buffer,
            vertex_len,
            index_buffer,
            index_len,
        )
    };

    let culled_index_buffer = new_buffer(
        Arc::clone(&device),
        vk::BUFFER_USAGE_INDEX_BUFFER_BIT | vk::BUFFER_USAGE_STORAGE_BUFFER_BIT,
        alloc::VmaAllocationCreateFlagBits(0),
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        size_of::<u32>() as vk::DeviceSize * index_len,
    );
    let normal_debug_buffer = new_buffer(
        Arc::clone(&device),
        vk::BUFFER_USAGE_INDEX_BUFFER_BIT
            | vk::BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        alloc::VmaAllocationCreateFlagBits(0),
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        size_of::<f32>() as vk::DeviceSize * 4 * 5 * 1000,
    );

    {
        let buffer_updates = &[
            vk::DescriptorBufferInfo {
                buffer: command_generation_buffer.handle,
                offset: 0,
                range: size_of::<u32>() as vk::DeviceSize * 5 * 64,
            },
            vk::DescriptorBufferInfo {
                buffer: index_buffer.handle,
                offset: 0,
                range: size_of::<u32>() as vk::DeviceSize * index_len,
            },
            vk::DescriptorBufferInfo {
                buffer: vertex_buffer.handle,
                offset: 0,
                range: size_of::<f32>() as vk::DeviceSize * 3 * vertex_len,
            },
            vk::DescriptorBufferInfo {
                buffer: culled_index_buffer.handle,
                offset: 0,
                range: size_of::<u32>() as vk::DeviceSize * index_len,
            },
        ];
        let buffer_updates2 = &[vk::DescriptorBufferInfo {
            buffer: normal_debug_buffer.handle,
            offset: 0,
            range: size_of::<f32>() as vk::DeviceSize * 4 * 5 * 1000,
        }];
        unsafe {
            device.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WriteDescriptorSet,
                        p_next: ptr::null(),
                        dst_set: command_generation_descriptor_set.handle,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: buffer_updates.len() as u32,
                        descriptor_type: vk::DescriptorType::StorageBuffer,
                        p_image_info: ptr::null(),
                        p_buffer_info: buffer_updates.as_ptr(),
                        p_texel_buffer_view: ptr::null(),
                    },
                    vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WriteDescriptorSet,
                        p_next: ptr::null(),
                        dst_set: command_generation_descriptor_set.handle,
                        dst_binding: 4,
                        dst_array_element: 0,
                        descriptor_count: buffer_updates2.len() as u32,
                        descriptor_type: vk::DescriptorType::StorageBuffer,
                        p_image_info: ptr::null(),
                        p_buffer_info: buffer_updates2.as_ptr(),
                        p_texel_buffer_view: ptr::null(),
                    },
                ],
                &[],
            );
        }
    }

    let projection = cgmath::perspective(
        cgmath::Deg(60.0),
        instance.window_width as f32 / instance.window_height as f32,
        0.1,
        100.0,
    );
    let view = cgmath::Matrix4::look_at(
        cgmath::Point3::new(0.0, 1.0, -2.0),
        cgmath::Point3::new(0.0, 0.0, 0.0),
        cgmath::vec3(0.0, -1.0, 0.0),
    );
    for depth in 0..300 {
        world
            .create_entity()
            .with::<Position>(Position(cgmath::Vector3::new(2.0, 0.0, 2.0 + depth as f32)))
            .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_x(cgmath::Deg(0.0))))
            .with::<Scale>(Scale(0.6))
            .with::<Matrices>(Matrices::one())
            .build();
        world
            .create_entity()
            .with::<Position>(Position(cgmath::Vector3::new(0.0, 0.0, 2.0 + depth as f32)))
            .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_x(cgmath::Deg(0.0))))
            .with::<Scale>(Scale(0.6))
            .with::<Matrices>(Matrices::one())
            .build();
        world
            .create_entity()
            .with::<Position>(Position(cgmath::Vector3::new(
                -2.0,
                0.0,
                2.0 + depth as f32,
            )))
            .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_x(cgmath::Deg(0.0))))
            .with::<Scale>(Scale(0.6))
            .with::<Matrices>(Matrices::one())
            .build();
    }
    let ubo_mapped = ubo_buffer.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>;
    let model_view_mapped =
        model_view_buffer.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>;
    let model_mapped = model_buffer.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>;
    let mut dispatcher = specs::DispatcherBuilder::new()
        .with(SteadyRotation, "steady_rotation", &[])
        .with(
            MVPCalculation { projection, view },
            "mvp",
            &["steady_rotation"],
        )
        .with(
            MVPUpload {
                dst_mvp: ubo_mapped,
                dst_mv: model_view_mapped,
                dst_model: model_mapped,
            },
            "mvp_upload",
            &["mvp"],
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
        let image_index = unsafe {
            swapchain
                .handle
                .ext
                .acquire_next_image_khr(
                    swapchain.handle.swapchain,
                    u64::MAX,
                    present_semaphore.handle,
                    vk::Fence::null(),
                )
                .unwrap()
        };
        let command_buffer_future = record_one_time_cb(
            Arc::clone(&graphics_command_pool),
            {
                let main_renderpass = Arc::clone(&main_renderpass);
                let framebuffer = Arc::clone(&framebuffer);
                let instance = Arc::clone(&instance);
                let device = Arc::clone(&device);
                let command_generation_buffer = Arc::clone(&command_generation_buffer);
                let command_generation_descriptor_set = Arc::clone(&command_generation_descriptor_set);
                let command_generation_pipeline = Arc::clone(&command_generation_pipeline);
                let command_generation_pipeline_layout = Arc::clone(&command_generation_pipeline_layout);
                let ubo_set = Arc::clone(&ubo_set);
                let model_set = Arc::clone(&model_set);
                let model_view_set = Arc::clone(&model_view_set);
                let gltf_pipeline = Arc::clone(&gltf_pipeline);
                let gltf_pipeline_layout = Arc::clone(&gltf_pipeline_layout);
                let triangle_pipeline = Arc::clone(&triangle_pipeline);
                let triangle_pipeline_layout = Arc::clone(&triangle_pipeline_layout);
                let vertex_buffer = Arc::clone(&vertex_buffer);
                let normal_buffer = Arc::clone(&normal_buffer);
                let index_buffer = Arc::clone(&index_buffer);
                let culled_index_buffer = Arc::clone(&culled_index_buffer);
                move |command_buffer| unsafe {
                    let clear_values = &[
                        vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
                        },
                        vk::ClearValue {
                            depth: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ];
                    let begin_info = vk::RenderPassBeginInfo {
                        s_type: vk::StructureType::RenderPassBeginInfo,
                        p_next: ptr::null(),
                        render_pass: main_renderpass.handle,
                        framebuffer: framebuffer.handles[image_index as usize],
                        render_area: vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: instance.window_width,
                                height: instance.window_height,
                            },
                        },
                        clear_value_count: clear_values.len() as u32,
                        p_clear_values: clear_values.as_ptr(),
                    };

                    device.device.debug_marker_around(
                        command_buffer,
                        "generate commands",
                        [0.0, 1.0, 0.0, 1.0],
                        || {
                            device.device.cmd_bind_pipeline(
                                command_buffer,
                                vk::PipelineBindPoint::Compute,
                                command_generation_pipeline.handle,
                            );
                            device.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::Compute,
                                command_generation_pipeline_layout.handle,
                                0,
                                &[
                                    ubo_set.handle,
                                    model_view_set.handle,
                                    command_generation_descriptor_set.handle,
                                ],
                                &[],
                            );
                            device.device.cmd_fill_buffer(
                                command_buffer,
                                command_generation_buffer.handle,
                                0,
                                size_of::<u32>() as vk::DeviceSize * 5 * 64,
                                0,
                            );
                            device.device.cmd_dispatch(
                                command_buffer,
                                index_len as u32 / 3,
                                1,
                                1,
                            );
                        },
                    );

                    device.device.debug_marker_around(
                        command_buffer,
                        "main renderpass",
                        [0.0, 0.0, 1.0, 1.0],
                        || {
                            device.device.cmd_begin_render_pass(
                                command_buffer,
                                &begin_info,
                                vk::SubpassContents::Inline,
                            );
                            device.device.cmd_bind_pipeline(
                                command_buffer,
                                vk::PipelineBindPoint::Graphics,
                                triangle_pipeline.handle,
                            );
                            device.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::Graphics,
                                triangle_pipeline_layout.handle,
                                0,
                                &[ubo_set.handle],
                                &[],
                            );
                            let constants = [1.0f32, 1.0, -1.0, 1.0, 1.0, -1.0];
                            use std::slice::from_raw_parts;

                            let casted: &[u8] =
                                { from_raw_parts(constants.as_ptr() as *const u8, 2 * 3 * 4) };
                            device.device.cmd_push_constants(
                                command_buffer,
                                triangle_pipeline_layout.handle,
                                vk::SHADER_STAGE_VERTEX_BIT,
                                0,
                                casted,
                            );
                            device.device.cmd_draw(command_buffer, 3, 1, 0, 0);

                            {
                                // gltf mesh
                                device.device.cmd_bind_pipeline(
                                    command_buffer,
                                    vk::PipelineBindPoint::Graphics,
                                    gltf_pipeline.handle,
                                );
                                device.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::Graphics,
                                    gltf_pipeline_layout.handle,
                                    0,
                                    &[ubo_set.handle, model_set.handle],
                                    &[],
                                );
                                device.device.cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &[vertex_buffer.handle, normal_buffer.handle],
                                    &[0, 0],
                                );
                                device.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    // index_buffer.handle,
                                    culled_index_buffer.handle,
                                    0,
                                    vk::IndexType::Uint32,
                                );
                                for ix in 0..2 {
                                    let constants = [ix as u32];
                                    use std::slice::from_raw_parts;

                                    let casted: &[u8] =
                                        { from_raw_parts(constants.as_ptr() as *const u8, 4) };
                                    device.device.cmd_push_constants(
                                        command_buffer,
                                        gltf_pipeline_layout.handle,
                                        vk::SHADER_STAGE_VERTEX_BIT,
                                        0,
                                        casted,
                                    );
                                    if ix == 0 {
                                        device.device.cmd_draw_indexed_indirect(
                                            command_buffer,
                                            command_generation_buffer.handle,
                                            0,
                                            1,
                                            size_of::<u32>() as u32 * 5,
                                        );
                                    } else {
                                        device.device.cmd_bind_index_buffer(
                                            command_buffer,
                                            index_buffer.handle,
                                            0,
                                            vk::IndexType::Uint32,
                                        );
                                        device.device.cmd_draw_indexed(
                                            command_buffer,
                                            index_len as u32,
                                            1,
                                            0,
                                            0,
                                            ix,
                                        );
                                    }
                                }
                            }
                            device.device.cmd_end_render_pass(command_buffer);
                        },
                    );
                    device.device.debug_marker_end(command_buffer);
                }
            },
        );
        let command_buffer = block_on(command_buffer_future).unwrap();
        unsafe {
            let wait_semaphores = &[present_semaphore.handle];
            let signal_semaphores = &[rendering_complete_semaphore.handle];
            let dst_stage_masks = vec![vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT; wait_semaphores.len()];
            let submits = [vk::SubmitInfo {
                s_type: vk::StructureType::SubmitInfo,
                p_next: ptr::null(),
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: dst_stage_masks.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &command_buffer.handle,
                signal_semaphore_count: signal_semaphores.len() as u32,
                p_signal_semaphores: signal_semaphores.as_ptr(),
            }];
            let queue_lock = device
                .graphics_queue
                .lock()
                .expect("can't lock the submit queue");

            let submit_fence = new_fence(Arc::clone(&device));

            device
                .device
                .queue_submit(*queue_lock, &submits, submit_fence.handle)
                .unwrap();

            {
                let device = Arc::clone(&device);
                threadpool
                    .run(lazy(move |_| {
                        spawn(lazy(move |_| {
                            // println!("dtor previous frame");
                            device
                                .device
                                .wait_for_fences(&[submit_fence.handle], true, u64::MAX)
                                .expect("Wait for fence failed.");
                            drop(command_buffer);
                            drop(submit_fence);
                            Ok(())
                        }))
                    }))
                    .unwrap();
            }
        }
        unsafe {
            let wait_semaphores = &[rendering_complete_semaphore.handle];
            let present_info = vk::PresentInfoKHR {
                s_type: vk::StructureType::PresentInfoKhr,
                p_next: ptr::null(),
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: &swapchain.handle.swapchain,
                p_image_indices: &image_index,
                p_results: ptr::null_mut(),
            };
            let queue = device
                .graphics_queue
                .lock()
                .expect("Failed to acquire lock on graphics queue");

            swapchain
                .handle
                .ext
                .queue_present_khr(*queue, &present_info)
                .unwrap();
        }
    }
}

fn setup_renderpass(device: Arc<Device>, swapchain: &Swapchain) -> Arc<RenderPass> {
    let attachment_descriptions = [
        vk::AttachmentDescription {
            format: swapchain.surface_format.format,
            flags: vk::AttachmentDescriptionFlags::empty(),
            samples: vk::SAMPLE_COUNT_1_BIT,
            load_op: vk::AttachmentLoadOp::Clear,
            store_op: vk::AttachmentStoreOp::Store,
            stencil_load_op: vk::AttachmentLoadOp::DontCare,
            stencil_store_op: vk::AttachmentStoreOp::DontCare,
            initial_layout: vk::ImageLayout::Undefined,
            final_layout: vk::ImageLayout::PresentSrcKhr,
        },
        vk::AttachmentDescription {
            format: vk::Format::D16Unorm,
            flags: vk::AttachmentDescriptionFlags::empty(),
            samples: vk::SAMPLE_COUNT_1_BIT,
            load_op: vk::AttachmentLoadOp::Clear,
            store_op: vk::AttachmentStoreOp::Store,
            stencil_load_op: vk::AttachmentLoadOp::DontCare,
            stencil_store_op: vk::AttachmentStoreOp::DontCare,
            initial_layout: vk::ImageLayout::Undefined,
            final_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
        },
    ];
    let color_attachment = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::ColorAttachmentOptimal,
    };
    let depth_attachment = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
    };
    let subpass_descs = [vk::SubpassDescription {
        color_attachment_count: 1,
        p_color_attachments: &color_attachment,
        p_depth_stencil_attachment: &depth_attachment,
        flags: Default::default(),
        pipeline_bind_point: vk::PipelineBindPoint::Graphics,
        input_attachment_count: 0,
        p_input_attachments: ptr::null(),
        p_resolve_attachments: ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: ptr::null(),
    }];
    let subpass_dependencies = [vk::SubpassDependency {
        dependency_flags: Default::default(),
        src_subpass: vk::VK_SUBPASS_EXTERNAL,
        dst_subpass: 0,
        src_stage_mask: vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        src_access_mask: Default::default(),
        dst_access_mask: vk::ACCESS_COLOR_ATTACHMENT_READ_BIT
            | vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        dst_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    }];
    new_renderpass(
        device,
        &attachment_descriptions,
        &subpass_descs,
        &subpass_dependencies,
    )
}
