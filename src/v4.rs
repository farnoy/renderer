#![feature(arbitrary_self_types)]

extern crate ash;
extern crate cgmath;
extern crate gltf;
extern crate image;
extern crate imgui;
extern crate meshopt;
extern crate parking_lot;
extern crate rayon;
extern crate specs;
extern crate specs_derive;
#[cfg(windows)]
extern crate winapi;
extern crate winit;

mod forward_renderer;

use crate::forward_renderer::{
    ecs::{components::*, setup, systems::*},
    renderer::{
        alloc, load_gltf, AcquireFramebuffer, CullGeometry, Gui, LoadedMesh, PresentFramebuffer,
        RenderFrame, Renderer,
    },
};
use ash::{version::DeviceV1_0, vk};
use cgmath::{Rotation3, Zero};
use parking_lot::Mutex;
use specs::Builder;
use std::{mem::size_of, sync::Arc};

fn main() {
    let mut world = specs::World::new();
    setup(&mut world);
    let rayon_threadpool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap(),
    );

    let (mut renderer, events_loop) = RenderFrame::new();

    let LoadedMesh {
        vertex_buffer,
        normal_buffer,
        index_buffer,
        index_len,
        aabb_c,
        aabb_h,
        base_color: _base_color,
    } = load_gltf(
        &renderer,
        "vendor/glTF-Sample-Models/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf",
    );

    let vertex_buffer = Arc::new(vertex_buffer);
    let normal_buffer = Arc::new(normal_buffer);
    let index_buffer = Arc::new(index_buffer);

    let culled_index_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        size_of::<u32>() as vk::DeviceSize * index_len * 2400,
    );
    renderer
        .device
        .set_object_name(culled_index_buffer.handle, "culled index buffer");

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
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(renderer.cull_set.handle)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(buffer_updates)
                    .build()],
                &[],
            );
        }
    }

    renderer.culled_index_buffer = Some(culled_index_buffer);

    world
        .create_entity()
        .with::<Position>(Position(cgmath::Vector3::new(0.0, 0.0, 0.0)))
        .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0))))
        .with::<Scale>(Scale(1.0))
        .with::<Matrices>(Matrices::one())
        .with::<CoarseCulled>(CoarseCulled(false))
        .with::<AABB>(AABB {
            c: cgmath::Vector3::zero(),
            h: cgmath::Vector3::zero(),
        })
        .with::<GltfMesh>(GltfMesh {
            vertex_buffer: Arc::clone(&vertex_buffer),
            normal_buffer: Arc::clone(&normal_buffer),
            index_buffer: Arc::clone(&index_buffer),
            index_len,
            aabb_c,
            aabb_h,
        })
        .with::<GltfMeshBufferIndex>(GltfMeshBufferIndex(0))
        .build();

    world
        .create_entity()
        .with::<Position>(Position(cgmath::Vector3::new(-5.0, 3.0, 2.0)))
        .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0))))
        .with::<Scale>(Scale(1.0))
        .with::<Matrices>(Matrices::one())
        .with::<CoarseCulled>(CoarseCulled(false))
        .with::<AABB>(AABB {
            c: cgmath::Vector3::zero(),
            h: cgmath::Vector3::zero(),
        })
        .with::<GltfMesh>(GltfMesh {
            vertex_buffer: Arc::clone(&vertex_buffer),
            normal_buffer: Arc::clone(&normal_buffer),
            index_buffer: Arc::clone(&index_buffer),
            index_len,
            aabb_c,
            aabb_h,
        })
        .with::<GltfMeshBufferIndex>(GltfMeshBufferIndex(0))
        .build();

    for ix in 0..2398 {
        let rot = cgmath::Quaternion::from_angle_y(cgmath::Deg((ix * 20) as f32));
        let pos = {
            use cgmath::Rotation;
            rot.rotate_vector(cgmath::vec3(0.0, -2.0, 5.0 + (ix / 10) as f32))
        };
        world
            .create_entity()
            .with::<Position>(Position(pos))
            .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(
                (ix * 20) as f32,
            ))))
            .with::<Scale>(Scale(0.6))
            .with::<Matrices>(Matrices::one())
            .with::<CoarseCulled>(CoarseCulled(false))
            .with::<AABB>(AABB {
                c: cgmath::Vector3::zero(),
                h: cgmath::Vector3::zero(),
            })
            .with::<GltfMesh>(GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                index_buffer: Arc::clone(&index_buffer),
                index_len,
                aabb_c,
                aabb_h,
            })
            .with::<GltfMeshBufferIndex>(GltfMeshBufferIndex(0))
            .build();
    }

    let gui = Gui::new(&renderer);

    world.add_resource(renderer);
    world.add_resource(gui);

    let quit_handle = Arc::new(Mutex::new(false));

    let dispatcher_builder = specs::DispatcherBuilder::new()
        .with_pool(Arc::clone(&rayon_threadpool))
        .with_thread_local(InputHandler {
            events_loop,
            quit_handle: quit_handle.clone(),
            move_mouse: true,
        })
        .with(CalculateFrameTiming, "calculate_frame_timing", &[])
        .with_barrier()
        .with(SteadyRotation, "steady_rotation", &[])
        .with(FlyCamera::default(), "fly_camera", &[])
        .with(ProjectCamera, "project_camera", &["fly_camera"])
        .with(
            MVPCalculation,
            "mvp",
            &["steady_rotation", "project_camera"],
        )
        .with(AABBCalculation, "aabb_calc", &["mvp"])
        .with(
            CoarseCulling,
            "coarse_culling",
            &["aabb_calc", "project_camera"],
        )
        .with(
            AssignBufferIndex,
            "assign_buffer_index",
            &["coarse_culling"],
        )
        .with(MVPUpload, "mvp_upload", &["mvp"])
        .with(AcquireFramebuffer, "acquire_framebuffer", &[])
        .with(
            CullGeometry::new(&world.read_resource::<RenderFrame>().device),
            "cull_geometry",
            &[
                "acquire_framebuffer",
                "assign_buffer_index",
                "mvp_upload",
                "coarse_culling",
            ],
        )
        .with(Renderer, "render_frame", &["cull_geometry"])
        .with(PresentFramebuffer, "present_framebuffer", &["render_frame"]);

    // print stages of execution
    // println!("{:?}", dispatcher_builder);

    let mut dispatcher = dispatcher_builder.build();

    dispatcher.setup(&mut world.res);

    'frame: loop {
        dispatcher.dispatch_thread_local(&world.res);
        dispatcher.dispatch_par(&world.res);
        world.maintain();
        if *quit_handle.lock() {
            unsafe {
                world
                    .read_resource::<RenderFrame>()
                    .device
                    .device_wait_idle()
                    .unwrap();
            }
            break 'frame;
        }
    }
}
