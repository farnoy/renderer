#![feature(arbitrary_self_types)]
#![feature(stmt_expr_attributes)]

extern crate ash;
extern crate cgmath;
extern crate gltf;
extern crate hashbrown;
extern crate image;
extern crate imgui;
extern crate meshopt;
extern crate microprofile;
extern crate num_traits;
extern crate parking_lot;
extern crate specs;
#[cfg(windows)]
extern crate winapi;
extern crate winit;

pub mod ecs {
    pub mod components;
    pub mod systems;
}
pub mod renderer;

use crate::renderer::{
    load_gltf, setup_ecs as renderer_setup_ecs, AcquireFramebuffer, AssignBufferIndex,
    CoarseCulling, ConsolidateMeshBuffers, CullPass, DepthOnlyPass, GltfMesh,
    GltfMeshBaseColorTexture, LoadedMesh, MVPUpload, PresentFramebuffer, RenderFrame, Renderer,
    SynchronizeBaseColorTextures,
};
use ash::version::DeviceV1_0;
use cgmath::{EuclideanSpace, Rotation3};
use ecs::{components::*, systems::*};
use microprofile::scope;
use parking_lot::Mutex;
use specs::{
    rayon,
    shred::{par, seq, ParSeq},
    Builder, WorldExt,
};
use std::sync::Arc;

fn main() {
    microprofile::init!();
    let mut world = specs::World::new();
    world.register::<Position>();
    world.register::<Rotation>();
    world.register::<Scale>();
    world.register::<Matrices>();
    world.register::<AABB>();
    world.register::<GltfMesh>();
    renderer_setup_ecs(&mut world);
    let rayon_threadpool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap(),
    );

    let (renderer, events_loop) = RenderFrame::new();

    world.insert(renderer);

    let quit_handle = Arc::new(Mutex::new(false));

    let mut dispatcher = ParSeq::new(
        seq![
            AcquireFramebuffer,
            CalculateFrameTiming,
            InputHandler {
                events_loop,
                quit_handle: quit_handle.clone(),
                move_mouse: true,
            },
            par![FlyCamera::default(), ConsolidateMeshBuffers,],
            ProjectCamera,
            MVPCalculation,
            AABBCalculation,
            CoarseCulling,
            AssignBufferIndex,
            SynchronizeBaseColorTextures,
            MVPUpload,
            par![DepthOnlyPass, CullPass,],
            Renderer,
            PresentFramebuffer,
        ],
        rayon_threadpool,
    );

    dispatcher.setup(&mut world);

    let LoadedMesh {
        vertex_buffer,
        normal_buffer,
        uv_buffer,
        index_buffers,
        vertex_len,
        aabb_c,
        aabb_h,
        base_color,
    } = load_gltf(
        &mut world,
        "vendor/glTF-Sample-Models/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf",
    );

    let vertex_buffer = Arc::new(vertex_buffer);
    let normal_buffer = Arc::new(normal_buffer);
    let uv_buffer = Arc::new(uv_buffer);
    let index_buffers = Arc::new(index_buffers);
    let base_color = Arc::new(base_color);

    world
        .create_entity()
        .with::<Position>(Position(cgmath::Point3::new(0.0, 5.0, 0.0)))
        .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0))))
        .with::<Scale>(Scale(1.0))
        .with::<GltfMesh>(GltfMesh {
            vertex_buffer: Arc::clone(&vertex_buffer),
            normal_buffer: Arc::clone(&normal_buffer),
            uv_buffer: Arc::clone(&uv_buffer),
            index_buffers: Arc::clone(&index_buffers),
            vertex_len,
            aabb_c,
            aabb_h,
        })
        .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
        .build();

    world
        .create_entity()
        .with::<Position>(Position(cgmath::Point3::new(0.0, 5.0, 5.0)))
        .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(
            90.0,
        ))))
        .with::<Scale>(Scale(1.0))
        .with::<GltfMesh>(GltfMesh {
            vertex_buffer: Arc::clone(&vertex_buffer),
            normal_buffer: Arc::clone(&normal_buffer),
            uv_buffer: Arc::clone(&uv_buffer),
            index_buffers: Arc::clone(&index_buffers),
            vertex_len,
            aabb_c,
            aabb_h,
        })
        .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
        .build();

    world
        .create_entity()
        .with::<Position>(Position(cgmath::Point3::new(-5.0, 5.0, 0.0)))
        .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_z(cgmath::Deg(
            60.0,
        ))))
        .with::<Scale>(Scale(1.0))
        .with::<GltfMesh>(GltfMesh {
            vertex_buffer: Arc::clone(&vertex_buffer),
            normal_buffer: Arc::clone(&normal_buffer),
            uv_buffer: Arc::clone(&uv_buffer),
            index_buffers: Arc::clone(&index_buffers),
            vertex_len,
            aabb_c,
            aabb_h,
        })
        .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
        .build();

    {
        let LoadedMesh {
            vertex_buffer,
            normal_buffer,
            uv_buffer,
            index_buffers,
            vertex_len,
            aabb_c,
            aabb_h,
            base_color,
        } = load_gltf(
            &mut world,
            "vendor/glTF-Sample-Models/2.0/TwoSidedPlane/glTF/TwoSidedPlane.gltf",
        );

        let vertex_buffer = Arc::new(vertex_buffer);
        let normal_buffer = Arc::new(normal_buffer);
        let uv_buffer = Arc::new(uv_buffer);
        let index_buffers = Arc::new(index_buffers);
        let base_color = Arc::new(base_color);

        world
            .create_entity()
            .with::<Position>(Position(cgmath::Point3::new(0.0, 0.0, 0.0)))
            .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0))))
            .with::<Scale>(Scale(50.0))
            .with::<GltfMesh>(GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffers: Arc::clone(&index_buffers),
                vertex_len,
                aabb_c,
                aabb_h,
            })
            .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
            .build();
    }

    {
        let LoadedMesh {
            vertex_buffer,
            normal_buffer,
            uv_buffer,
            index_buffers,
            vertex_len,
            aabb_c,
            aabb_h,
            base_color,
        } = load_gltf(
            &mut world,
            "vendor/glTF-Sample-Models/2.0/BoxTextured/glTF/BoxTextured.gltf",
        );

        let vertex_buffer = Arc::new(vertex_buffer);
        let normal_buffer = Arc::new(normal_buffer);
        let uv_buffer = Arc::new(uv_buffer);
        let index_buffers = Arc::new(index_buffers);
        let base_color = Arc::new(base_color);

        world
            .create_entity()
            .with::<Position>(Position(cgmath::Point3::new(5.0, 3.0, 2.0)))
            .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0))))
            .with::<Scale>(Scale(1.0))
            .with::<GltfMesh>(GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffers: Arc::clone(&index_buffers),
                vertex_len,
                aabb_c,
                aabb_h,
            })
            .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
            .build();
    }

    for ix in 0..200 {
        let rot = cgmath::Quaternion::from_angle_y(cgmath::Deg((ix * 20) as f32));
        let pos = {
            use cgmath::Rotation;
            cgmath::Point3::from_vec(rot.rotate_vector(cgmath::vec3(
                0.0,
                2.0,
                5.0 + (ix / 10) as f32,
            )))
        };
        world
            .create_entity()
            .with::<Position>(Position(pos))
            .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(
                (ix * 20) as f32,
            ))))
            .with::<Scale>(Scale(0.6))
            .with::<GltfMesh>(GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffers: Arc::clone(&index_buffers),
                vertex_len,
                aabb_c,
                aabb_h,
            })
            .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
            .build();
    }

    'frame: loop {
        microprofile::flip!();
        microprofile::scope!("game-loop", "all");
        {
            microprofile::scope!("game-loop", "dispatch");
            dispatcher.dispatch(&world);
        }
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
    microprofile::shutdown!();
}
