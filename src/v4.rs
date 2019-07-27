#![feature(arbitrary_self_types)]

extern crate ash;
extern crate gltf;
extern crate hashbrown;
extern crate image;
extern crate imgui;
extern crate meshopt;
#[cfg(feature = "microprofile")]
extern crate microprofile;
extern crate nalgebra as na;
extern crate nalgebra_glm as glm;
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
    load_gltf, right_vector, setup_ecs as renderer_setup_ecs, up_vector, AcquireFramebuffer,
    AssignBufferIndex, CoarseCulling, ConsolidateMeshBuffers, CullPass, DepthOnlyPass, GltfMesh,
    GltfMeshBaseColorTexture, LoadedMesh, MVPUpload, PrepareShadowMaps, PresentFramebuffer,
    RenderFrame, Renderer, ShadowMappingMVPCalculation, SynchronizeBaseColorTextures,
};
use ash::version::DeviceV1_0;
use ecs::{components::*, systems::*};
#[cfg(feature = "microprofile")]
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
use specs::{
    rayon,
    shred::{par, seq, ParSeq},
    Builder, WorldExt,
};
use std::sync::Arc;

fn main() {
    #[cfg(feature = "profiling")]
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
            par![
                seq![FlyCamera::default(), ProjectCamera,],
                ConsolidateMeshBuffers,
            ],
            par![
                seq![MVPCalculation, AABBCalculation,],
                ShadowMappingMVPCalculation,
            ],
            CoarseCulling,
            AssignBufferIndex,
            SynchronizeBaseColorTextures,
            MVPUpload,
            par![CullPass, seq![PrepareShadowMaps, DepthOnlyPass,],],
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

    let light = world
        .create_entity()
        .with::<Position>(Position(na::Point3::new(-6.0, 50.0, 0.0)))
        .with::<Rotation>(Rotation(na::UnitQuaternion::look_at_lh(
            &(na::Point3::new(0.0, 0.0, 0.0) - &na::Point3::new(-6.0, 50.0, 0.0)),
            &up_vector(),
        )))
        .with::<Light>(Light { strength: 1.0 })
        .build();

    world
        .create_entity()
        .with::<Position>(Position(na::Point3::new(-6.0, 50.0, 8.0)))
        .with::<Rotation>(Rotation(na::UnitQuaternion::look_at_lh(
            &(na::Point3::new(0.0, 0.0, 0.0) - &na::Point3::new(-6.0, 50.0, 8.0)),
            &up_vector(),
        )))
        .with::<Light>(Light { strength: 0.7 })
        .build();

    world
        .create_entity()
        .with::<Position>(Position(na::Point3::new(0.0, 5.0, 0.0)))
        .with::<Rotation>(Rotation(na::UnitQuaternion::identity()))
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
        .with::<Position>(Position(na::Point3::new(0.0, 5.0, 5.0)))
        .with::<Rotation>(Rotation(na::UnitQuaternion::from_axis_angle(
            &up_vector(),
            f32::pi() / 2.0,
        )))
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
        .with::<Position>(Position(na::Point3::new(-5.0, 5.0, 0.0)))
        .with::<Rotation>(Rotation(na::UnitQuaternion::from_axis_angle(
            &right_vector(),
            f32::pi() / 3.0,
        )))
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
            .with::<Position>(Position(na::Point3::new(0.0, 0.0, 0.0)))
            .with::<Rotation>(Rotation(na::UnitQuaternion::identity()))
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
            .with::<Position>(Position(na::Point3::new(5.0, 3.0, 2.0)))
            .with::<Rotation>(Rotation(na::UnitQuaternion::identity()))
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
        let angle = f32::pi() * (ix as f32 * 20.0) / 180.0;
        let rot = na::Rotation3::from_axis_angle(&na::Unit::new_normalize(na::Vector3::y()), angle);
        let pos = rot.transform_point(&na::Point3::new(0.0, 2.0, 5.0 + (ix / 10) as f32));
        world
            .create_entity()
            .with::<Position>(Position(pos))
            .with::<Rotation>(Rotation(na::UnitQuaternion::from_axis_angle(
                &na::Unit::new_normalize(na::Vector3::y()),
                angle,
            )))
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

    let mut f = 0.01f32;
    'frame: loop {
        #[cfg(feature = "profiling")]
        microprofile::flip!();
        #[cfg(feature = "profiling")]
        microprofile::scope!("game-loop", "all");
        {
            let mut positions = world.write_storage::<Position>();
            let pos = &mut positions.get_mut(light).unwrap().0;
            let mut rotations = world.write_storage::<Rotation>();
            let rot = &mut rotations.get_mut(light).unwrap().0;
            pos.x = f.sin() * 4.0 + -6.0;
            *rot = na::UnitQuaternion::look_at_lh(
                &(na::Point3::new(0.0, 0.0, 0.0) - *pos),
                &up_vector(),
            );
            f += 0.01;
        }
        {
            #[cfg(feature = "profiling")]
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
    #[cfg(feature = "profiling")]
    microprofile::shutdown!();
}
