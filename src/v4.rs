#![feature(arbitrary_self_types)]

extern crate ash;
extern crate cgmath;
extern crate gltf;
extern crate hashbrown;
extern crate image;
extern crate imgui;
extern crate meshopt;
extern crate num_traits;
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
        load_gltf, AcquireFramebuffer, ConsolidateVertexBuffers, CullPass, Gui, LoadedMesh,
        PresentFramebuffer, RenderFrame, Renderer, SynchronizeBaseColorTextures,
        UpdateCullDescriptorsForMeshes,
    },
};
use ash::version::DeviceV1_0;
use cgmath::{Rotation3, Zero};
use parking_lot::Mutex;
use specs::Builder;
use std::sync::Arc;

fn main() {
    let mut world = specs::World::new();
    setup(&mut world);
    let rayon_threadpool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap(),
    );

    let (renderer, events_loop) = RenderFrame::new();

    let LoadedMesh {
        vertex_buffer,
        normal_buffer,
        uv_buffer,
        index_buffer,
        vertex_len,
        index_len,
        aabb_c,
        aabb_h,
        base_color,
    } = load_gltf(
        &renderer,
        "vendor/glTF-Sample-Models/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf",
    );

    let vertex_buffer = Arc::new(vertex_buffer);
    let normal_buffer = Arc::new(normal_buffer);
    let uv_buffer = Arc::new(uv_buffer);
    let index_buffer = Arc::new(index_buffer);
    let base_color = Arc::new(base_color);

    world
        .create_entity()
        .with::<Position>(Position(cgmath::Vector3::new(0.0, 1.0, 0.0)))
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
            uv_buffer: Arc::clone(&uv_buffer),
            index_buffer: Arc::clone(&index_buffer),
            vertex_len,
            index_len,
            aabb_c,
            aabb_h,
        })
        .with::<GltfMeshBufferIndex>(GltfMeshBufferIndex(0))
        .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
        .build();

    {
        let LoadedMesh {
            vertex_buffer,
            normal_buffer,
            uv_buffer,
            index_buffer,
            vertex_len,
            index_len,
            aabb_c,
            aabb_h,
            base_color,
        } = load_gltf(
            &renderer,
            "vendor/glTF-Sample-Models/2.0/TwoSidedPlane/glTF/TwoSidedPlane.gltf",
        );

        let vertex_buffer = Arc::new(vertex_buffer);
        let normal_buffer = Arc::new(normal_buffer);
        let uv_buffer = Arc::new(uv_buffer);
        let index_buffer = Arc::new(index_buffer);
        let base_color = Arc::new(base_color);

        world
            .create_entity()
            .with::<Position>(Position(cgmath::Vector3::new(0.0, 0.0, 0.0)))
            .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_y(cgmath::Deg(0.0))))
            .with::<Scale>(Scale(50.0))
            .with::<Matrices>(Matrices::one())
            .with::<CoarseCulled>(CoarseCulled(false))
            .with::<AABB>(AABB {
                c: cgmath::Vector3::zero(),
                h: cgmath::Vector3::zero(),
            })
            .with::<GltfMesh>(GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffer: Arc::clone(&index_buffer),
                vertex_len,
                index_len,
                aabb_c,
                aabb_h,
            })
            .with::<GltfMeshBufferIndex>(GltfMeshBufferIndex(0))
            .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
            .build();
    }

    {
        let LoadedMesh {
            vertex_buffer,
            normal_buffer,
            uv_buffer,
            index_buffer,
            vertex_len,
            index_len,
            aabb_c,
            aabb_h,
            base_color,
        } = load_gltf(
            &renderer,
            "vendor/glTF-Sample-Models/2.0/BoxTextured/glTF/BoxTextured.gltf",
        );

        let vertex_buffer = Arc::new(vertex_buffer);
        let normal_buffer = Arc::new(normal_buffer);
        let uv_buffer = Arc::new(uv_buffer);
        let index_buffer = Arc::new(index_buffer);
        let base_color = Arc::new(base_color);

        world
            .create_entity()
            .with::<Position>(Position(cgmath::Vector3::new(5.0, 3.0, 2.0)))
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
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffer: Arc::clone(&index_buffer),
                vertex_len,
                index_len,
                aabb_c,
                aabb_h,
            })
            .with::<GltfMeshBufferIndex>(GltfMeshBufferIndex(0))
            .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
            .build();
    }

    for ix in 0..2398 {
        let rot = cgmath::Quaternion::from_angle_y(cgmath::Deg((ix * 20) as f32));
        let pos = {
            use cgmath::Rotation;
            rot.rotate_vector(cgmath::vec3(0.0, 2.0, 5.0 + (ix / 10) as f32))
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
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffer: Arc::clone(&index_buffer),
                vertex_len,
                index_len,
                aabb_c,
                aabb_h,
            })
            .with::<GltfMeshBaseColorTexture>(GltfMeshBaseColorTexture(Arc::clone(&base_color)))
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
        .with(FlyCamera::default(), "fly_camera", &[])
        .with(ProjectCamera, "project_camera", &["fly_camera"])
        .with(MVPCalculation, "mvp", &["project_camera"])
        .with(AABBCalculation, "aabb_calc", &["mvp"])
        .with(ConsolidateVertexBuffers, "consolidate_vertex_buffers", &[])
        .with(
            CoarseCulling,
            "coarse_culling",
            &["aabb_calc", "project_camera"],
        )
        .with(
            UpdateCullDescriptorsForMeshes,
            "update_cull_descriptors",
            &[],
        )
        .with(
            AssignBufferIndex,
            "assign_buffer_index",
            &["coarse_culling"],
        )
        .with(
            SynchronizeBaseColorTextures,
            "synchronize_base_color_textures",
            &["assign_buffer_index"],
        )
        .with(MVPUpload, "mvp_upload", &["mvp", "assign_buffer_index"])
        .with(AcquireFramebuffer, "acquire_framebuffer", &[])
        .with(
            CullPass,
            "cull_pass",
            &[
                "acquire_framebuffer",
                "assign_buffer_index",
                "mvp_upload",
                "coarse_culling",
                "update_cull_descriptors",
                "consolidate_vertex_buffers",
            ],
        )
        .with(
            Renderer,
            "render_frame",
            &["cull_pass", "synchronize_base_color_textures"],
        )
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
