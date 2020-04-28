#![feature(arbitrary_self_types)]
#![feature(backtrace)]
#![feature(vec_remove_item)]
#![feature(const_int_pow)]
#![allow(clippy::new_without_default)]

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub mod ecs;
pub mod renderer;

use ash::version::DeviceV1_0;
use ecs::{components::*, resources::*, systems::*};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use legion::prelude::*;
#[cfg(feature = "microprofile")]
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
use renderer::{DrawIndex, *};
use std::{cell::RefCell, rc::Rc, sync::Arc};

fn main() {
    #[cfg(feature = "profiling")]
    microprofile::init!();
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    let (renderer, swapchain, events_loop) = RenderFrame::new();

    let universe = Universe::new();
    let mut world = universe.create_world();
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
    let gui_render = GuiRender::new(
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
    world.insert(
        (),
        vec![
            (
                Light { strength: 1.0 },
                Position(na::Point3::new(30.0, 20.0, -40.1)),
                Rotation(na::UnitQuaternion::look_at_lh(
                    &(na::Point3::new(0.0, 0.0, 0.0) - na::Point3::new(30.0, 20.0, -40.1)),
                    &up_vector(),
                )),
                ShadowMappingLightMatrices::new(
                    &renderer,
                    &main_descriptor_pool,
                    &camera_matrices,
                    0,
                ),
            ),
            (
                Light { strength: 0.7 },
                Position(na::Point3::new(0.1, 17.0, 0.1)),
                Rotation(na::UnitQuaternion::look_at_lh(
                    &(na::Point3::new(0.0, 0.0, 0.0) - na::Point3::new(0.1, 17.0, 0.1)),
                    &up_vector(),
                )),
                ShadowMappingLightMatrices::new(
                    &renderer,
                    &main_descriptor_pool,
                    &camera_matrices,
                    1,
                ),
            ),
        ],
    );

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
            aabb: box_aabb.clone(),
        },
        projectile_texture: Arc::clone(&box_base_color),
    };

    // objects
    world.insert(
        (),
        vec![
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
                    aabb: aabb.clone(),
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
                    aabb: aabb.clone(),
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
                    aabb: aabb.clone(),
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
                    aabb: box_aabb.clone(),
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
        ],
    );

    world.insert(
        (),
        (7..max_entities).map(|ix| {
            let angle = f32::pi() * (ix as f32 * 20.0) / 180.0;
            let rot =
                na::Rotation3::from_axis_angle(&na::Unit::new_normalize(na::Vector3::y()), angle);
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
                    aabb: aabb.clone(),
                },
                AABB::default(),
                CoarseCulled(false),
                DrawIndex::default(),
            )
        }),
    );

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

    let mut schedule = Schedule::builder()
        .add_thread_local({
            let input_handler = Rc::clone(&input_handler);
            let gui = Rc::clone(&gui);
            InputHandler::exec_system(input_handler, gui)
        })
        .add_thread_local({
            SystemBuilder::<()>::new("AcquireFramebuffer")
                .read_resource::<RenderFrame>()
                .write_resource::<Swapchain>()
                .write_resource::<MainAttachments>()
                .write_resource::<MainFramebuffer>()
                .write_resource::<PresentData>()
                .read_resource::<ModelData>()
                .read_resource::<CameraMatrices>()
                .write_resource::<DepthPassData>()
                .write_resource::<ImageIndex>()
                .read_resource::<Resized>()
                .build_thread_local(move |_commands, _world, resources, _queries| {
                    let (
                        ref renderer,
                        ref mut swapchain,
                        ref mut main_attachments,
                        ref mut main_framebuffer,
                        ref mut present_data,
                        ref model_data,
                        ref camera_matrices,
                        ref mut depth_pass_data,
                        ref mut image_index,
                        ref resized,
                    ) = resources;

                    if resized.0 {
                        unsafe {
                            renderer.device.device_wait_idle().unwrap();
                        }
                        swapchain.resize_to_fit();
                        **main_attachments = MainAttachments::new(&renderer, &swapchain);
                        **depth_pass_data = DepthPassData::new(
                            &renderer,
                            &model_data,
                            &main_attachments,
                            &swapchain,
                            &camera_matrices,
                        );
                        **main_framebuffer =
                            MainFramebuffer::new(&renderer, &main_attachments, &swapchain);
                        **present_data = PresentData::new(&renderer);
                    }

                    let swapchain_needs_recreating =
                        AcquireFramebuffer::exec(&renderer, &swapchain, &mut *image_index);
                    if swapchain_needs_recreating {
                        unsafe {
                            renderer.device.device_wait_idle().unwrap();
                        }
                        swapchain.resize_to_fit();
                        **main_attachments = MainAttachments::new(&renderer, &swapchain);
                        **depth_pass_data = DepthPassData::new(
                            &renderer,
                            &model_data,
                            &main_attachments,
                            &swapchain,
                            &camera_matrices,
                        );
                        **main_framebuffer =
                            MainFramebuffer::new(&renderer, &main_attachments, &swapchain);
                        **present_data = PresentData::new(&renderer);
                        AcquireFramebuffer::exec(&renderer, &swapchain, &mut *image_index);
                    }
                })
        })
        .add_system(
            SystemBuilder::<()>::new("CalculateFrameTiming")
                .write_resource::<FrameTiming>()
                .build(move |_commands, _world, mut frame_timing, _queries| {
                    CalculateFrameTiming::exec(&mut frame_timing);
                }),
        )
        .add_system(
            SystemBuilder::<()>::new("AssignDrawIndex")
                .with_query(<Write<DrawIndex>>::query().filter(
                    component::<GltfMeshBaseColorTexture>()
                        & component::<Position>()
                        & component::<GltfMesh>(),
                ))
                .build(move |_commands, mut world, _, query| {
                    for (counter, ref mut draw_idx) in query.iter_mut(&mut world).enumerate() {
                        draw_idx.0 = counter as u32;
                    }
                }),
        )
        .add_system(CameraController::exec_system())
        .add_system(
            SystemBuilder::<()>::new("ProjectCamera")
                .read_resource::<Swapchain>()
                .write_resource::<Camera>()
                .build(move |_commands, _world, resources, _queries| {
                    let (swapchain, ref mut camera) = resources;
                    ProjectCamera::exec(swapchain, &mut *camera);
                }),
        )
        .add_system(LaunchProjectileTest::exec_system())
        .add_system(UpdateProjectiles::exec_system())
        .flush()
        .add_system(ConsolidateMeshBuffers::exec_system())
        .add_system(ModelMatrixCalculation::exec_system())
        .add_system(AABBCalculation::exec_system())
        .add_system(ShadowMappingMVPCalculation::exec_system())
        .add_system(CoarseCulling::exec_system())
        .add_system(SynchronizeBaseColorTextures::visit_system())
        .flush() // TODO: remove later
        .add_system(SynchronizeBaseColorTextures::consolidate_system())
        .add_system(CameraMatricesUpload::exec_system())
        .add_system(ModelMatricesUpload::exec_system())
        .add_system(CullPass::exec_system())
        .add_system(PrepareShadowMaps::exec_system())
        .add_system(DepthOnlyPass::exec_system())
        .add_system(Renderer::exec_system())
        .flush()
        .add_thread_local(GuiRender::exec_system(
            Rc::clone(&gui),
            Rc::clone(&input_handler),
            gui_render,
        ))
        .flush()
        .add_system(
            SystemBuilder::<()>::new("PresentFramebuffer")
                .read_resource::<RenderFrame>()
                .read_resource::<ImageIndex>()
                .read_resource::<Swapchain>()
                .write_resource::<PresentData>()
                .build(move |_commands, _world, resources, _queries| {
                    let (ref renderer, ref image_index, ref swapchain, ref mut present_data) =
                        resources;
                    PresentFramebuffer::exec(&renderer, &present_data, &swapchain, &image_index);
                }),
        )
        .add_system(
            SystemBuilder::<()>::new("End frame")
                .write_resource::<RenderFrame>()
                .build(move |_commands, _world, renderer, _queries| {
                    renderer.frame_number += 1;
                }),
        )
        .build();

    'frame: loop {
        #[cfg(feature = "profiling")]
        microprofile::flip!();
        #[cfg(feature = "profiling")]
        microprofile::scope!("game-loop", "all");
        {
            #[cfg(feature = "profiling")]
            microprofile::scope!("game-loop", "ecs");
            schedule.execute(&mut world, &mut resources);
            world.defrag(None);
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
