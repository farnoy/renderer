#![feature(non_modrs_mods)]
#![feature(futures_api)]
#![feature(await_macro)]
#![feature(async_await)]

#[macro_use]
extern crate ash;
extern crate cgmath;
extern crate futures;
extern crate gltf;
extern crate gltf_importer;
extern crate gltf_utils;
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
use forward_renderer::{
    ecs::{components::*, systems::*, Bundle},
    renderer::{alloc, load_gltf, new_buffer, RenderFrame, Renderer},
};
use specs::Builder;
use std::{mem::size_of, ptr, sync::Arc};
use winit::{dpi::LogicalSize, Event, KeyboardInput, WindowEvent};

fn main() {
    let mut world = specs::World::new();
    world.add_bundle(Bundle);
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

    let (vertex_buffer, normal_buffer, index_buffer, index_len) = load_gltf(
        &renderer,
        "glTF-Sample-Models/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf",
    );

    let culled_index_buffer = new_buffer(
        Arc::clone(&renderer.device),
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
        alloc::VmaAllocationCreateFlagBits(0),
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        size_of::<u32>() as vk::DeviceSize * index_len * 100,
    );

    {
        let buffer_updates = &[
            vk::DescriptorBufferInfo {
                buffer: renderer.culled_commands_buffer.handle,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: index_buffer.handle,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vertex_buffer.handle,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: culled_index_buffer.handle,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        unsafe {
            renderer.device.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: renderer.cull_set.handle,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: buffer_updates.len() as u32,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: buffer_updates.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                }],
                &[],
            );
        }
    }

    renderer.culled_index_buffer = Some(culled_index_buffer);

    for ix in 0..100 {
        world
            .create_entity()
            .with::<Position>(Position(cgmath::Vector3::new(
                ((ix % 3) * 2 - 2) as f32,
                0.0,
                2.0 + ix as f32,
            ))).with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(
                (ix * 10) as f32,
            )))).with::<Scale>(Scale(0.6))
            .with::<Matrices>(Matrices::one())
            .with::<GltfMesh>(GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                index_buffer: Arc::clone(&index_buffer),
                index_len,
            }).with::<GltfMeshBufferIndex>(GltfMeshBufferIndex(0))
            .build();
    }

    let dst_mvp = Arc::clone(&renderer.ubo_buffer);
    let dst_model = Arc::clone(&renderer.model_buffer);

    world.add_resource(renderer);

    let mut dispatcher = specs::DispatcherBuilder::new()
        .with_pool(Arc::clone(&rayon_threadpool))
        .with(SteadyRotation, "steady_rotation", &[])
        .with(AssignBufferIndex, "assign_buffer_index", &[])
        .with(
            MVPCalculation { projection, view },
            "mvp",
            &["steady_rotation"],
        ).with(MVPUpload { dst_mvp, dst_model }, "mvp_upload", &["mvp"])
        .with(
            Renderer,
            "render_frame",
            &["assign_buffer_index", "mvp_upload"],
        ).build();

    'frame: loop {
        let mut quit = false;
        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(LogicalSize { width, height }),
                ..
            } => {
                println!("The window was resized to {}x{}", width, height);
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
            world
                .read_resource::<RenderFrame>()
                .device
                .device_wait_idle()
                .unwrap();
            break 'frame;
        }
        dispatcher.dispatch(&world.res);
        world.maintain();
    }
}
