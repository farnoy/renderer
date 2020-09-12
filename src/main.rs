#![feature(arbitrary_self_types)]
#![feature(backtrace)]
#![feature(vec_remove_item)]
#![feature(const_int_pow)]
#![allow(clippy::new_without_default)]
#![feature(maybe_uninit_uninit_array, maybe_uninit_slice)]

#[macro_use]
extern crate lazy_static;

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub mod ecs;
pub mod renderer;

use ash::version::DeviceV1_0;
use bevy_ecs::*;
use ecs::{components::*, resources::*, systems::*};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
#[cfg(feature = "microprofile")]
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
use renderer::{DrawIndex, *};
use std::{cell::RefCell, convert::TryInto, rc::Rc, sync::Arc};

fn main() {
    #[cfg(feature = "profiling")]
    microprofile::init!();
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    let (renderer, swapchain, events_loop) = RenderFrame::new();

    let mut world = World::new();
    let mut resources = Resources::default();

    let quit_handle = Arc::new(Mutex::new(false));

    let present_data = PresentData::new(&renderer);
    let image_index = ImageIndex::default();
    let consolidated_mesh_buffers = ConsolidatedMeshBuffers::new(&renderer);
    let graphics_command_pool = GraphicsCommandPool::new(&renderer);

    let mut main_descriptor_pool = MainDescriptorPool::new(&renderer);
    let camera_matrices = CameraMatrices::new(&renderer, &main_descriptor_pool);

    let base_color_descriptor_set =
        BaseColorDescriptorSet::new(&renderer, &mut main_descriptor_pool);
    let model_data = ModelData::new(&renderer, &main_descriptor_pool);

    let cull_pass_data = CullPassData::new(
        &renderer,
        &model_data,
        &mut main_descriptor_pool,
        &camera_matrices,
    );
    let cull_pass_data_private = CullPassDataPrivate::new(&renderer);
    let main_attachments = MainAttachments::new(&renderer, &swapchain);
    let depth_pass_data = DepthPassData::new(
        &renderer,
        &model_data,
        &main_attachments,
        &swapchain,
        &camera_matrices,
    );
    let shadow_mapping_data =
        ShadowMappingData::new(&renderer, &depth_pass_data, &mut main_descriptor_pool);

    let gui = Rc::new(RefCell::new(Gui::new()));
    let mut gui_render = GuiRender::new(
        &renderer,
        &main_descriptor_pool,
        &swapchain,
        &mut gui.borrow_mut(),
    );
    let mut imgui_platform = WinitPlatform::init(&mut gui.borrow_mut().imgui);
    imgui_platform.attach_window(
        gui.borrow_mut().imgui.io_mut(),
        &renderer.instance.window,
        HiDpiMode::Locked(1.0), // TODO: Revert this to Default if we can make the app obey winit DPI
    );

    let input_handler = Rc::new(RefCell::new(InputHandler {
        events_loop,
        quit_handle: quit_handle.clone(),
        imgui_platform,
    }));

    let gltf_pass = GltfPassData::new(
        &renderer,
        &model_data,
        &base_color_descriptor_set,
        &shadow_mapping_data,
        &camera_matrices,
    );

    let debug_aabb_pass_data = DebugAABBPassData::new(&renderer, &camera_matrices);

    let main_framebuffer = MainFramebuffer::new(&renderer, &main_attachments, &swapchain);

    let LoadedMesh {
        vertex_buffer,
        normal_buffer,
        uv_buffer,
        index_buffers,
        vertex_len,
        aabb,
        base_color,
    } = load_gltf(
        &renderer,
        &graphics_command_pool,
        "vendor/glTF-Sample-Models/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf",
    );

    let vertex_buffer = Arc::new(vertex_buffer);
    let normal_buffer = Arc::new(normal_buffer);
    let uv_buffer = Arc::new(uv_buffer);
    let index_buffers = Arc::new(index_buffers);
    let base_color = Arc::new(base_color);

    let max_entities = 30;
    debug_assert!(max_entities > 7); // 7 static ones

    // lights
    world.spawn_batch(vec![
        (
            Light { strength: 1.0 },
            Position(na::Point3::new(30.0, 20.0, -40.1)),
            Rotation(na::UnitQuaternion::look_at_lh(
                &(na::Point3::new(0.0, 0.0, 0.0) - na::Point3::new(30.0, 20.0, -40.1)),
                &up_vector(),
            )),
            ShadowMappingLightMatrices::new(&renderer, &main_descriptor_pool, &camera_matrices, 0),
        ),
        (
            Light { strength: 0.7 },
            Position(na::Point3::new(0.1, 17.0, 0.1)),
            Rotation(na::UnitQuaternion::look_at_lh(
                &(na::Point3::new(0.0, 0.0, 0.0) - na::Point3::new(0.1, 17.0, 0.1)),
                &up_vector(),
            )),
            ShadowMappingLightMatrices::new(&renderer, &main_descriptor_pool, &camera_matrices, 1),
        ),
    ]);

    let (
        box_vertex_buffer,
        box_normal_buffer,
        box_uv_buffer,
        box_index_buffers,
        box_base_color,
        box_vertex_len,
        box_aabb,
    ) = {
        let LoadedMesh {
            vertex_buffer,
            normal_buffer,
            uv_buffer,
            index_buffers,
            vertex_len,
            aabb,
            base_color,
        } = {
            load_gltf(
                &renderer,
                &graphics_command_pool,
                "vendor/glTF-Sample-Models/2.0/BoxTextured/glTF/BoxTextured.gltf",
            )
        };

        (
            Arc::new(vertex_buffer),
            Arc::new(normal_buffer),
            Arc::new(uv_buffer),
            Arc::new(index_buffers),
            Arc::new(base_color),
            vertex_len,
            aabb,
        )
    };

    let mesh_library = MeshLibrary {
        projectile: GltfMesh {
            vertex_buffer: Arc::clone(&box_vertex_buffer),
            normal_buffer: Arc::clone(&box_normal_buffer),
            uv_buffer: Arc::clone(&box_uv_buffer),
            index_buffers: Arc::clone(&box_index_buffers),
            vertex_len: box_vertex_len,
            aabb: box_aabb,
        },
        projectile_texture: Arc::clone(&box_base_color),
    };

    // objects
    world.spawn_batch(vec![
        (
            Position(na::Point3::new(0.0, 5.0, 0.0)),
            Rotation(na::UnitQuaternion::identity()),
            Scale(1.0),
            ModelMatrix::default(),
            GltfMeshBaseColorTexture(Arc::clone(&base_color)),
            GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffers: Arc::clone(&index_buffers),
                vertex_len,
                aabb,
            },
            AABB::default(),
            CoarseCulled(false),
            DrawIndex::default(),
        ),
        (
            Position(na::Point3::new(0.0, 5.0, 5.0)),
            Rotation(na::UnitQuaternion::from_axis_angle(
                &up_vector(),
                f32::pi() / 2.0,
            )),
            Scale(1.0),
            ModelMatrix::default(),
            GltfMeshBaseColorTexture(Arc::clone(&base_color)),
            GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffers: Arc::clone(&index_buffers),
                vertex_len,
                aabb,
            },
            AABB::default(),
            CoarseCulled(false),
            DrawIndex::default(),
        ),
        (
            Position(na::Point3::new(-5.0, 5.0, 0.0)),
            Rotation(na::UnitQuaternion::from_axis_angle(
                &up_vector(),
                f32::pi() / 3.0,
            )),
            Scale(1.0),
            ModelMatrix::default(),
            GltfMeshBaseColorTexture(Arc::clone(&base_color)),
            GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffers: Arc::clone(&index_buffers),
                vertex_len,
                aabb,
            },
            AABB::default(),
            CoarseCulled(false),
            DrawIndex::default(),
        ),
        (
            Position(na::Point3::new(5.0, 3.0, 2.0)),
            Rotation(na::UnitQuaternion::identity()),
            Scale(1.0),
            ModelMatrix::default(),
            GltfMeshBaseColorTexture(Arc::clone(&box_base_color)),
            GltfMesh {
                vertex_buffer: Arc::clone(&box_vertex_buffer),
                normal_buffer: Arc::clone(&box_normal_buffer),
                uv_buffer: Arc::clone(&box_uv_buffer),
                index_buffers: Arc::clone(&box_index_buffers),
                vertex_len: box_vertex_len,
                aabb: box_aabb,
            },
            AABB::default(),
            CoarseCulled(false),
            DrawIndex::default(),
        ),
        (
            Position(na::Point3::new(0.0, -29.0, 0.0)),
            Rotation(na::UnitQuaternion::identity()),
            Scale(50.0),
            ModelMatrix::default(),
            GltfMeshBaseColorTexture(Arc::clone(&box_base_color)),
            GltfMesh {
                vertex_buffer: Arc::clone(&box_vertex_buffer),
                normal_buffer: Arc::clone(&box_normal_buffer),
                uv_buffer: Arc::clone(&box_uv_buffer),
                index_buffers: Arc::clone(&box_index_buffers),
                vertex_len: box_vertex_len,
                aabb: box_aabb,
            },
            AABB::default(),
            CoarseCulled(false),
            DrawIndex::default(),
        ),
    ]);

    world.spawn_batch((7..max_entities).map(|ix| {
        let angle = f32::pi() * (ix as f32 * 20.0) / 180.0;
        let rot = na::Rotation3::from_axis_angle(&na::Unit::new_normalize(na::Vector3::y()), angle);
        let pos = rot.transform_point(&na::Point3::new(
            0.0,
            (ix as f32 * -0.01) + 2.0,
            5.0 + (ix / 10) as f32,
        ));

        (
            Position(pos),
            Rotation(na::UnitQuaternion::from_axis_angle(
                &na::Unit::new_normalize(na::Vector3::y()),
                angle,
            )),
            Scale(0.6),
            ModelMatrix::default(),
            GltfMeshBaseColorTexture(Arc::clone(&base_color)),
            GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffers: Arc::clone(&index_buffers),
                vertex_len,
                aabb,
            },
            AABB::default(),
            CoarseCulled(false),
            DrawIndex::default(),
        )
    }));

    resources.insert(swapchain);
    resources.insert(mesh_library);
    resources.insert(Resized(false));
    resources.insert(FrameTiming::default());
    resources.insert(InputState::default());
    resources.insert(InputActions::default());
    resources.insert(Camera::default());
    resources.insert(RuntimeConfiguration::new());
    resources.insert(renderer);
    resources.insert(graphics_command_pool);
    resources.insert(image_index);
    resources.insert(consolidated_mesh_buffers);
    resources.insert(main_descriptor_pool);
    resources.insert(camera_matrices);
    resources.insert(base_color_descriptor_set);
    resources.insert(model_data);
    resources.insert(cull_pass_data);
    resources.insert(cull_pass_data_private);
    resources.insert(shadow_mapping_data);
    resources.insert(depth_pass_data);
    resources.insert(present_data);
    resources.insert(main_attachments);
    resources.insert(main_framebuffer);
    resources.insert(gltf_pass);
    resources.insert(debug_aabb_pass_data);

    let mut schedule = Schedule::default();
    schedule.add_stage("acquire_framebuffer");
    fn acquire_framebuffer(
        renderer: Res<RenderFrame>,
        mut swapchain: ResMut<Swapchain>,
        mut main_attachments: ResMut<MainAttachments>,
        mut main_framebuffer: ResMut<MainFramebuffer>,
        mut present_data: ResMut<PresentData>,
        model_data: Res<ModelData>,
        camera_matrices: Res<CameraMatrices>,
        mut depth_pass_data: ResMut<DepthPassData>,
        mut image_index: ResMut<ImageIndex>,
        resized: Res<Resized>,
    ) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "AcquireFramebuffer");

        if resized.0 {
            unsafe {
                renderer.device.device_wait_idle().unwrap();
            }
            swapchain.resize_to_fit();
            *main_attachments = MainAttachments::new(&renderer, &swapchain);
            *depth_pass_data = DepthPassData::new(
                &renderer,
                &model_data,
                &main_attachments,
                &swapchain,
                &camera_matrices,
            );
            *main_framebuffer = MainFramebuffer::new(&renderer, &main_attachments, &swapchain);
            *present_data = PresentData::new(&renderer);
        }

        let swapchain_needs_recreating =
            AcquireFramebuffer::exec(&renderer, &swapchain, &mut *image_index);
        if swapchain_needs_recreating {
            unsafe {
                renderer.device.device_wait_idle().unwrap();
            }
            swapchain.resize_to_fit();
            *main_attachments = MainAttachments::new(&renderer, &swapchain);
            *depth_pass_data = DepthPassData::new(
                &renderer,
                &model_data,
                &main_attachments,
                &swapchain,
                &camera_matrices,
            );
            *main_framebuffer = MainFramebuffer::new(&renderer, &main_attachments, &swapchain);
            *present_data = PresentData::new(&renderer);
            AcquireFramebuffer::exec(&renderer, &swapchain, &mut *image_index);
        }
    }
    schedule.add_system_to_stage("acquire_framebuffer", acquire_framebuffer.system());
    schedule.add_system_to_stage(
        "acquire_framebuffer",
        (|mut frame_timing: ResMut<FrameTiming>| {
            #[cfg(feature = "profiling")]
            microprofile::scope!("ecs", "CalculateFrameTiming");
            CalculateFrameTiming::exec(&mut frame_timing);
        })
        .system(),
    );
    schedule.add_system_to_stage(
        "acquire_framebuffer",
        (|mut query: Query<
            With<GltfMeshBaseColorTexture, With<Position, With<GltfMesh, &mut DrawIndex>>>,
        >| {
            #[cfg(feature = "profiling")]
            microprofile::scope!("ecs", "AssignDrawIndex");
            for (counter, ref mut draw_idx) in query.iter().into_iter().enumerate() {
                draw_idx.0 = counter
                    .try_into()
                    .expect("failed to downcast draw_idx to u32");
            }
        })
        .system(),
    );
    schedule.add_system_to_stage("acquire_framebuffer", camera_controller.system());
    schedule.add_system_to_stage(
        "acquire_framebuffer",
        (|swapchain: Res<Swapchain>, mut camera: ResMut<Camera>| {
            #[cfg(feature = "profiling")]
            microprofile::scope!("ecs", "ProjectCamera");
            ProjectCamera::exec(&*swapchain, &mut *camera);
        })
        .system(),
    );
    schedule.add_system_to_stage("acquire_framebuffer", camera_controller.system());
    schedule.add_system_to_stage("acquire_framebuffer", launch_projectiles_test.system());
    schedule.add_system_to_stage("acquire_framebuffer", update_projectiles.system());
    schedule.add_stage("render setup");
    schedule.add_system_to_stage("render setup", consolidate_mesh_buffers.system());
    schedule.add_system_to_stage("render setup", model_matrix_calculation.system());
    schedule.add_system_to_stage("render setup", aabb_calculation.system());
    schedule.add_system_to_stage("render setup", shadow_mapping_mvp_calculation.system());
    schedule.add_system_to_stage("render setup", coarse_culling.system());
    schedule.add_system_to_stage(
        "render setup",
        synchronize_base_color_textures_visit.system(),
    );
    schedule.add_system_to_stage("render setup", camera_matrices_upload.system());
    schedule.add_system_to_stage("render setup", model_matrices_upload.system());
    schedule.add_stage("consolidate textures");
    schedule.add_system_to_stage(
        "consolidate textures",
        synchronize_base_color_textures_consolidate.system(),
    );
    // TODO: remove to unlock more parallelism
    schedule.add_stage("cull pass");
    schedule.add_system_to_stage("cull pass", cull_pass.system());
    // TODO: remove to unlock more parallelism
    schedule.add_stage("shadow mapping");
    schedule.add_system_to_stage("shadow mapping", prepare_shadow_maps.system());
    // TODO: remove to unlock more parallelism
    schedule.add_stage("depth only");
    schedule.add_system_to_stage("depth only", depth_only_pass.system());
    // TODO: remove to unlock more parallelism
    schedule.add_stage("main render");
    schedule.add_system_to_stage("main render", render_frame.system());
    // TODO: add the final stage for submitting all work to the graphics queue

    schedule.initialize(&mut resources);

    resources.insert(bevy_tasks::ComputeTaskPool(bevy_tasks::TaskPool::new()));

    let mut executor = ParallelExecutor::default();

    'frame: loop {
        #[cfg(feature = "profiling")]
        microprofile::flip!();
        #[cfg(feature = "profiling")]
        microprofile::scope!("game-loop", "all");
        {
            {
                #[cfg(feature = "profiling")]
                microprofile::scope!("game-loop", "input");
                let input_handler = Rc::clone(&input_handler);
                let gui = Rc::clone(&gui);
                InputHandler::run(input_handler, gui, &mut resources)
            }

            {
                #[cfg(feature = "profiling")]
                microprofile::scope!("game-loop", "ecs");
                executor.run(&mut schedule, &mut world, &mut resources);
                // schedule.run(&mut world, &mut resources);
            }
            {
                #[cfg(feature = "profiling")]
                microprofile::scope!("game-loop", "gui");
                gui_render.render(Rc::clone(&gui), Rc::clone(&input_handler), &mut resources);
            }
            let (mut renderer, present_data, swapchain, image_index) = resources
                .query::<(
                    ResMut<RenderFrame>,
                    Res<PresentData>,
                    Res<Swapchain>,
                    Res<ImageIndex>,
                )>()
                .unwrap();

            {
                #[cfg(feature = "profiling")]
                microprofile::scope!("game-loop", "present");
                PresentFramebuffer::exec(&mut renderer, &present_data, &swapchain, &image_index);
            }
            renderer.frame_number += 1;
            // world.defrag(None);
        }
        if *quit_handle.lock() {
            unsafe {
                resources
                    .get::<RenderFrame>()
                    .unwrap()
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
