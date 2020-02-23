#![feature(arbitrary_self_types)]
#![feature(backtrace)]
#![allow(clippy::new_without_default)]

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub mod ecs {
    pub mod components;
    pub mod custom;
    pub mod resources;
    pub mod systems;
}
pub mod renderer;

use ash::version::DeviceV1_0;
use ecs::{components::*, custom::*, resources::*, systems::*};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use legion::prelude::*;
#[cfg(feature = "microprofile")]
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
use rayon;
use renderer::*;
use std::{cell::RefCell, rc::Rc, sync::Arc};

fn main() {
    #[cfg(feature = "profiling")]
    microprofile::init!();
    let mut position_storage = ComponentStorage::<na::Point3<f32>>::new();
    let mut rotation_storage = ComponentStorage::<na::UnitQuaternion<f32>>::new();
    let mut scale_storage = ComponentStorage::<f32>::new();
    let model_matrices_storage = ComponentStorage::<glm::Mat4>::new();
    let aabb_storage = ComponentStorage::<ncollide3d::bounding_volume::AABB<f32>>::new();
    let mut meshes_storage = ComponentStorage::<GltfMesh>::new();
    let mut light_storage = ComponentStorage::<Light>::new();
    let projectile_velocities_storage = ComponentStorage::<f32>::new();
    let projectile_target_storage = ComponentStorage::<na::Point3<f32>>::new();
    let mut base_color_texture_storage = ComponentStorage::<GltfMeshBaseColorTexture>::new();
    let base_color_visited_storage = ComponentStorage::<BaseColorVisitedMarker>::new();
    let coarse_culled_storage = ComponentStorage::<CoarseCulled>::new();
    let shadow_mapping_light_matrices_storage =
        ComponentStorage::<ShadowMappingLightMatrices>::new();
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    let (renderer, swapchain, events_loop) = RenderFrame::new();

    let universe = Universe::new();
    let mut world = universe.create_world();
    world.resources.insert(swapchain);

    struct PositionStorage(ComponentStorage<na::Point3<f32>>);
    struct RotationStorage(ComponentStorage<na::UnitQuaternion<f32>>);
    struct ScaleStorage(ComponentStorage<f32>);
    struct ProjectileTargetStorage(ComponentStorage<na::Point3<f32>>);
    struct ProjectileVelocityStorage(ComponentStorage<f32>);
    struct ModelMatrixStorage(ComponentStorage<glm::Mat4>);

    let quit_handle = Arc::new(Mutex::new(false));

    let present_data = PresentData::new(&renderer);
    let image_index = ImageIndex::default();
    world.resources.insert(FrameTiming::default());
    world.resources.insert(InputState::default());
    world.resources.insert(Camera::default());
    world.resources.insert(FlyCamera::default());
    let consolidated_mesh_buffers = ConsolidatedMeshBuffers::new(&renderer);
    let graphics_command_pool = GraphicsCommandPool::new(&renderer);

    let mut entities = EntitiesStorage::new();

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
    let main_attachments = MainAttachments::new(&renderer, &world.resources.get().unwrap());
    let depth_pass_data = DepthPassData::new(
        &renderer,
        &model_data,
        &main_attachments,
        &world.resources.get().unwrap(),
        &camera_matrices,
    );
    let shadow_mapping_data =
        ShadowMappingData::new(&renderer, &depth_pass_data, &mut main_descriptor_pool);

    world.resources.insert(RuntimeConfiguration::new());

    let gui = Rc::new(RefCell::new(Gui::new()));
    let mut gui_render = GuiRender::new(&renderer, &main_descriptor_pool, &mut gui.borrow_mut());
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

    let main_framebuffer = MainFramebuffer::new(
        &renderer,
        &main_attachments,
        &world.resources.get().unwrap(),
    );

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

    let ixes = entities.allocate_mask(max_entities);
    debug_assert_eq!(ixes.to_vec(), (0..max_entities).collect::<Vec<_>>());

    position_storage.replace_mask(&ixes);
    rotation_storage.replace_mask(&ixes);

    let mut light_only = croaring::Bitmap::create();
    light_only.add_many(&[0, 1]);
    light_storage.replace_mask(&light_only);
    let rest = ixes - light_only;
    scale_storage.replace_mask(&rest);
    meshes_storage.replace_mask(&rest);
    base_color_texture_storage.replace_mask(&rest);

    position_storage.insert(0, na::Point3::new(30.0, 20.0, -40.1));
    rotation_storage.insert(
        0,
        na::UnitQuaternion::look_at_lh(
            &(na::Point3::new(0.0, 0.0, 0.0) - na::Point3::new(30.0, 20.0, -40.1)),
            &up_vector(),
        ),
    );
    light_storage.insert(0, Light { strength: 1.0 });

    position_storage.insert(1, na::Point3::new(0.1, 17.0, 0.1));
    rotation_storage.insert(
        1,
        na::UnitQuaternion::look_at_lh(
            &(na::Point3::new(0.0, 0.0, 0.0) - na::Point3::new(0.1, 17.0, 0.1)),
            &up_vector(),
        ),
    );
    light_storage.insert(1, Light { strength: 0.7 });

    position_storage.insert(2, na::Point3::new(0.0, 5.0, 0.0));
    rotation_storage.insert(2, na::UnitQuaternion::identity());
    scale_storage.insert(2, 1.0);
    meshes_storage.insert(
        2,
        GltfMesh {
            vertex_buffer: Arc::clone(&vertex_buffer),
            normal_buffer: Arc::clone(&normal_buffer),
            uv_buffer: Arc::clone(&uv_buffer),
            index_buffers: Arc::clone(&index_buffers),
            vertex_len,
            aabb: aabb.clone(),
        },
    );
    base_color_texture_storage.insert(2, GltfMeshBaseColorTexture(Arc::clone(&base_color)));

    position_storage.insert(3, na::Point3::new(0.0, 5.0, 5.0));
    rotation_storage.insert(
        3,
        na::UnitQuaternion::from_axis_angle(&up_vector(), f32::pi() / 2.0),
    );
    scale_storage.insert(3, 1.0);
    meshes_storage.insert(
        3,
        GltfMesh {
            vertex_buffer: Arc::clone(&vertex_buffer),
            normal_buffer: Arc::clone(&normal_buffer),
            uv_buffer: Arc::clone(&uv_buffer),
            index_buffers: Arc::clone(&index_buffers),
            vertex_len,
            aabb: aabb.clone(),
        },
    );
    base_color_texture_storage.insert(3, GltfMeshBaseColorTexture(Arc::clone(&base_color)));

    position_storage.insert(4, na::Point3::new(-5.0, 5.0, 0.0));
    rotation_storage.insert(
        4,
        na::UnitQuaternion::from_axis_angle(&up_vector(), f32::pi() / 3.0),
    );
    scale_storage.insert(4, 1.0);
    meshes_storage.insert(
        4,
        GltfMesh {
            vertex_buffer: Arc::clone(&vertex_buffer),
            normal_buffer: Arc::clone(&normal_buffer),
            uv_buffer: Arc::clone(&uv_buffer),
            index_buffers: Arc::clone(&index_buffers),
            vertex_len,
            aabb: aabb.clone(),
        },
    );
    base_color_texture_storage.insert(4, GltfMeshBaseColorTexture(Arc::clone(&base_color)));

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
    world.resources.insert(mesh_library);

    position_storage.insert(5, na::Point3::new(5.0, 3.0, 2.0));
    rotation_storage.insert(5, na::UnitQuaternion::identity());
    scale_storage.insert(5, 1.0);
    meshes_storage.insert(
        5,
        GltfMesh {
            vertex_buffer: Arc::clone(&box_vertex_buffer),
            normal_buffer: Arc::clone(&box_normal_buffer),
            uv_buffer: Arc::clone(&box_uv_buffer),
            index_buffers: Arc::clone(&box_index_buffers),
            vertex_len: box_vertex_len,
            aabb: box_aabb.clone(),
        },
    );
    base_color_texture_storage.insert(5, GltfMeshBaseColorTexture(Arc::clone(&box_base_color)));

    position_storage.insert(6, na::Point3::new(0.0, -29.0, 0.0));
    rotation_storage.insert(6, na::UnitQuaternion::identity());
    scale_storage.insert(6, 50.0);
    meshes_storage.insert(
        6,
        GltfMesh {
            vertex_buffer: Arc::clone(&box_vertex_buffer),
            normal_buffer: Arc::clone(&box_normal_buffer),
            uv_buffer: Arc::clone(&box_uv_buffer),
            index_buffers: Arc::clone(&box_index_buffers),
            vertex_len: box_vertex_len,
            aabb: box_aabb.clone(),
        },
    );
    base_color_texture_storage.insert(6, GltfMeshBaseColorTexture(Arc::clone(&box_base_color)));

    for ix in 7..max_entities {
        let angle = f32::pi() * (ix as f32 * 20.0) / 180.0;
        let rot = na::Rotation3::from_axis_angle(&na::Unit::new_normalize(na::Vector3::y()), angle);
        let pos = rot.transform_point(&na::Point3::new(
            0.0,
            (ix as f32 * -0.01) + 2.0,
            5.0 + (ix / 10) as f32,
        ));

        position_storage.insert(ix, pos);
        rotation_storage.insert(
            ix,
            na::UnitQuaternion::from_axis_angle(&na::Unit::new_normalize(na::Vector3::y()), angle),
        );
        scale_storage.insert(ix, 0.6);
        meshes_storage.insert(
            ix,
            GltfMesh {
                vertex_buffer: Arc::clone(&vertex_buffer),
                normal_buffer: Arc::clone(&normal_buffer),
                uv_buffer: Arc::clone(&uv_buffer),
                index_buffers: Arc::clone(&index_buffers),
                vertex_len,
                aabb: aabb.clone(),
            },
        );
        base_color_texture_storage.insert(ix, GltfMeshBaseColorTexture(Arc::clone(&base_color)));
    }

    world.resources.insert(entities);
    world.resources.insert(PositionStorage(position_storage));
    world.resources.insert(RotationStorage(rotation_storage));
    world.resources.insert(ScaleStorage(scale_storage));
    world
        .resources
        .insert(ProjectileTargetStorage(projectile_target_storage));
    world
        .resources
        .insert(ProjectileVelocityStorage(projectile_velocities_storage));
    world.resources.insert(meshes_storage);
    world.resources.insert(base_color_texture_storage);
    world.resources.insert(renderer);
    world.resources.insert(graphics_command_pool);
    world.resources.insert(image_index);
    world.resources.insert(consolidated_mesh_buffers);
    world
        .resources
        .insert(ModelMatrixStorage(model_matrices_storage));
    world.resources.insert(aabb_storage);
    world
        .resources
        .insert(shadow_mapping_light_matrices_storage);
    world.resources.insert(light_storage);
    world.resources.insert(main_descriptor_pool);
    world.resources.insert(camera_matrices);
    world.resources.insert(coarse_culled_storage);
    world.resources.insert(base_color_visited_storage);
    world.resources.insert(base_color_descriptor_set);
    world.resources.insert(model_data);
    world.resources.insert(cull_pass_data);
    world.resources.insert(cull_pass_data_private);
    world.resources.insert(shadow_mapping_data);
    world.resources.insert(depth_pass_data);
    world.resources.insert(present_data);
    world.resources.insert(main_attachments);
    world.resources.insert(main_framebuffer);
    world.resources.insert(debug_aabb_pass_data);
    world.resources.insert(gltf_pass);

    let mut schedule =
        Schedule::builder()
            .add_thread_local({
                let gui = Rc::clone(&gui);
                let input_handler = Rc::clone(&input_handler);
                SystemBuilder::<()>::new("AcquireFramebuffer")
                    .read_resource::<RenderFrame>()
                    .write_resource::<InputState>()
                    .write_resource::<Camera>()
                    .write_resource::<RuntimeConfiguration>()
                    .write_resource::<Swapchain>()
                    .write_resource::<MainAttachments>()
                    .write_resource::<MainFramebuffer>()
                    .write_resource::<PresentData>()
                    .read_resource::<ModelData>()
                    .read_resource::<CameraMatrices>()
                    .write_resource::<DepthPassData>()
                    .write_resource::<ImageIndex>()
                    .build_thread_local(move |_commands, _world, resources, _queries| {
                        let (
                            ref renderer,
                            ref mut input_state,
                            ref mut camera,
                            ref mut runtime_config,
                            ref mut swapchain,
                            ref mut main_attachments,
                            ref mut main_framebuffer,
                            ref mut present_data,
                            ref model_data,
                            ref camera_matrices,
                            ref mut depth_pass_data,
                            ref mut image_index,
                        ) = resources;
                        let window_resized = input_handler.borrow_mut().exec(
                            &renderer.instance.window,
                            &mut gui.borrow_mut().imgui,
                            &mut *input_state,
                            &mut *camera,
                            &mut *runtime_config,
                        );

                        if window_resized {
                            unsafe {
                                &renderer.device.device_wait_idle().unwrap();
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
                SystemBuilder::<()>::new("FlyCamera")
                    .read_resource::<InputState>()
                    .read_resource::<FrameTiming>()
                    .read_resource::<RuntimeConfiguration>()
                    .write_resource::<Camera>()
                    .write_resource::<FlyCamera>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            input,
                            frame_timing,
                            runtime_config,
                            ref mut camera,
                            ref mut fly_camera,
                        ) = resources;
                        fly_camera.exec(input, frame_timing, runtime_config, &mut *camera);
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("ProjectCamera")
                    .read_resource::<Swapchain>()
                    .write_resource::<Camera>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (swapchain, ref mut camera) = resources;
                        ProjectCamera::exec(swapchain, &mut *camera);
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("ProjectCamera")
                    .write_resource::<EntitiesStorage>()
                    .write_resource::<PositionStorage>()
                    .write_resource::<RotationStorage>()
                    .write_resource::<ScaleStorage>()
                    .write_resource::<ProjectileTargetStorage>()
                    .write_resource::<ProjectileVelocityStorage>()
                    .write_resource::<ComponentStorage<GltfMesh>>()
                    .write_resource::<ComponentStorage<GltfMeshBaseColorTexture>>()
                    .write_resource::<Camera>()
                    .read_resource::<MeshLibrary>()
                    .read_resource::<InputState>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref mut entities,
                            ref mut position_storage,
                            ref mut rotation_storage,
                            ref mut scale_storage,
                            ref mut projectile_target_storage,
                            ref mut projectile_velocity_storage,
                            ref mut meshes_storage,
                            ref mut base_color_texture_storage,
                            ref mut camera,
                            ref mesh_library,
                            ref input_state,
                        ) = resources;
                        LaunchProjectileTest::exec(
                            &mut *entities,
                            &mut position_storage.0,
                            &mut rotation_storage.0,
                            &mut scale_storage.0,
                            &mut *meshes_storage,
                            &mut *base_color_texture_storage,
                            &mut projectile_target_storage.0,
                            &mut projectile_velocity_storage.0,
                            &mut *camera,
                            &mesh_library,
                            &input_state,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("UpdateProjectiles")
                    .write_resource::<EntitiesStorage>()
                    .write_resource::<PositionStorage>()
                    .write_resource::<RotationStorage>()
                    .write_resource::<ProjectileTargetStorage>()
                    .write_resource::<ProjectileVelocityStorage>()
                    .read_resource::<FrameTiming>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref mut entities,
                            ref mut position_storage,
                            ref mut rotation_storage,
                            ref mut projectile_target_storage,
                            ref mut projectile_velocity_storage,
                            ref frame_timing,
                        ) = resources;
                        UpdateProjectiles::exec(
                            &mut *entities,
                            &mut position_storage.0,
                            &mut rotation_storage.0,
                            &mut projectile_target_storage.0,
                            &mut projectile_velocity_storage.0,
                            &frame_timing,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("ConsolidateMeshBuffers")
                    .read_resource::<RenderFrame>()
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<GraphicsCommandPool>()
                    .read_resource::<ComponentStorage<GltfMesh>>()
                    .write_resource::<ConsolidatedMeshBuffers>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref renderer,
                            ref entities,
                            ref graphics_command_pool,
                            ref meshes_storage,
                            ref mut consolidated_mesh_buffers,
                        ) = resources;
                        ConsolidateMeshBuffers::exec(
                            &renderer,
                            &*entities,
                            &*graphics_command_pool,
                            &*meshes_storage,
                            &mut *consolidated_mesh_buffers,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("ModelMatrixCalculation")
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<PositionStorage>()
                    .read_resource::<RotationStorage>()
                    .read_resource::<ScaleStorage>()
                    .write_resource::<ModelMatrixStorage>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref entities,
                            ref positions,
                            ref rotations,
                            ref scales,
                            ref mut model_matrices,
                        ) = resources;
                        ModelMatrixCalculation::exec(
                            &*entities,
                            &positions.0,
                            &rotations.0,
                            &scales.0,
                            &mut model_matrices.0,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("AABBCalculation")
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<ModelMatrixStorage>()
                    .read_resource::<ComponentStorage<GltfMesh>>()
                    .write_resource::<ComponentStorage<ncollide3d::bounding_volume::AABB<f32>>>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (ref entities, ref model_matrices, ref meshes, ref mut aabbs) =
                            resources;
                        AABBCalculation::exec(&entities, &model_matrices.0, &meshes, &mut *aabbs);
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("ShadowMappingMVPCalculation")
                    .read_resource::<RenderFrame>()
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<PositionStorage>()
                    .read_resource::<RotationStorage>()
                    .write_resource::<ComponentStorage<ShadowMappingLightMatrices>>()
                    .read_resource::<ComponentStorage<Light>>()
                    .read_resource::<ImageIndex>()
                    .read_resource::<MainDescriptorPool>()
                    .read_resource::<CameraMatrices>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref renderer,
                            ref entities,
                            ref positions,
                            ref rotations,
                            ref mut light_matrices,
                            ref lights,
                            ref image_index,
                            ref main_descriptor_pool,
                            ref camera_matrices,
                        ) = resources;
                        ShadowMappingMVPCalculation::exec(
                            &renderer,
                            &entities,
                            &positions.0,
                            &rotations.0,
                            &mut *light_matrices,
                            &lights,
                            &image_index,
                            &main_descriptor_pool,
                            &camera_matrices,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("CoarseCulling")
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<ComponentStorage<ncollide3d::bounding_volume::AABB<f32>>>()
                    .read_resource::<Camera>()
                    .write_resource::<ComponentStorage<CoarseCulled>>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (ref entities, ref aabbs, ref camera, ref mut coarse_culled_storage) =
                            resources;
                        CoarseCulling::exec(
                            &entities,
                            &aabbs,
                            &camera,
                            &mut *coarse_culled_storage,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("SynchronizeBaseColorTextures")
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<RenderFrame>()
                    .read_resource::<BaseColorDescriptorSet>()
                    .read_resource::<ComponentStorage<GltfMeshBaseColorTexture>>()
                    .read_resource::<ImageIndex>()
                    .write_resource::<ComponentStorage<BaseColorVisitedMarker>>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref entities,
                            ref renderer,
                            ref base_color_descriptor_set,
                            ref base_color_textures,
                            ref image_index,
                            ref mut visited_markers,
                        ) = resources;
                        SynchronizeBaseColorTextures::exec(
                            &entities,
                            &renderer,
                            &base_color_descriptor_set,
                            &base_color_textures,
                            &image_index,
                            &mut *visited_markers,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("CameraMatricesUpload")
                    .read_resource::<ImageIndex>()
                    .read_resource::<Camera>()
                    .write_resource::<CameraMatrices>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (ref image_index, ref camera, ref mut camera_matrices) = resources;
                        CameraMatricesUpload::exec(&image_index, &camera, &mut *camera_matrices);
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("CameraMatricesUpload")
                    .read_resource::<ModelMatrixStorage>()
                    .read_resource::<ImageIndex>()
                    .write_resource::<ModelData>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (ref model_matrices_storage, ref image_index, ref mut model_data) =
                            resources;
                        ModelMatricesUpload::exec(
                            &model_matrices_storage.0,
                            &image_index,
                            &mut *model_data,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("CullPass")
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<RenderFrame>()
                    .read_resource::<CullPassData>()
                    .write_resource::<CullPassDataPrivate>()
                    .read_resource::<ComponentStorage<GltfMesh>>()
                    .read_resource::<ImageIndex>()
                    .read_resource::<ConsolidatedMeshBuffers>()
                    .read_resource::<PositionStorage>()
                    .read_resource::<Camera>()
                    .read_resource::<ModelData>()
                    .read_resource::<CameraMatrices>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref entities,
                            ref renderer,
                            ref cull_pass_data,
                            ref mut cull_pass_data_private,
                            ref meshes,
                            ref image_index,
                            ref consolidated_mesh_buffers,
                            ref positions,
                            ref camera,
                            ref model_data,
                            ref camera_matrices,
                        ) = resources;
                        CullPass::exec(
                            &*entities,
                            &renderer,
                            &cull_pass_data,
                            &mut *cull_pass_data_private,
                            &meshes,
                            &image_index,
                            &consolidated_mesh_buffers,
                            &positions.0,
                            &camera,
                            &model_data,
                            &camera_matrices,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("PrepareShadowMaps")
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<RenderFrame>()
                    .read_resource::<DepthPassData>()
                    .read_resource::<ImageIndex>()
                    .write_resource::<GraphicsCommandPool>()
                    .write_resource::<ShadowMappingData>()
                    .read_resource::<ComponentStorage<GltfMesh>>()
                    .read_resource::<ComponentStorage<Light>>()
                    .read_resource::<ComponentStorage<ShadowMappingLightMatrices>>()
                    .read_resource::<ModelData>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref entities,
                            ref renderer,
                            ref depth_pass_data,
                            ref image_index,
                            ref mut graphics_command_pool,
                            ref mut shadow_mapping_data,
                            ref meshes,
                            ref lights,
                            ref shadow_matrices,
                            ref model_data,
                        ) = resources;
                        PrepareShadowMaps::exec(
                            &entities,
                            &renderer,
                            &depth_pass_data,
                            &image_index,
                            &mut *graphics_command_pool,
                            &mut *shadow_mapping_data,
                            &meshes,
                            &lights,
                            &shadow_matrices,
                            &model_data,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("DepthOnlyPass")
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<RenderFrame>()
                    .read_resource::<ImageIndex>()
                    .write_resource::<GraphicsCommandPool>()
                    .write_resource::<DepthPassData>()
                    .read_resource::<ComponentStorage<GltfMesh>>()
                    .read_resource::<ModelData>()
                    .read_resource::<RuntimeConfiguration>()
                    .read_resource::<PositionStorage>()
                    .read_resource::<Camera>()
                    .read_resource::<CameraMatrices>()
                    .read_resource::<Swapchain>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref entities,
                            ref renderer,
                            ref image_index,
                            ref mut graphics_command_pool,
                            ref mut depth_pass_data,
                            ref meshes,
                            ref model_data,
                            ref runtime_config,
                            ref positions,
                            ref camera,
                            ref camera_matrices,
                            ref swapchain,
                        ) = resources;
                        DepthOnlyPass::exec(
                            &renderer,
                            &runtime_config,
                            &entities,
                            &image_index,
                            &meshes,
                            &positions.0,
                            &camera,
                            &camera_matrices,
                            &mut *depth_pass_data,
                            &swapchain,
                            &model_data,
                            &mut *graphics_command_pool,
                        );
                    }),
            )
            .add_thread_local({
                let gui = Rc::clone(&gui);
                let input_handler = Rc::clone(&input_handler);
                SystemBuilder::<()>::new("Renderer")
                    .read_resource::<RenderFrame>()
                    .read_resource::<EntitiesStorage>()
                    .read_resource::<ImageIndex>()
                    .write_resource::<GraphicsCommandPool>()
                    .read_resource::<ModelData>()
                    .write_resource::<RuntimeConfiguration>()
                    .read_resource::<CameraMatrices>()
                    .read_resource::<Swapchain>()
                    .read_resource::<ConsolidatedMeshBuffers>()
                    .read_resource::<ComponentStorage<ncollide3d::bounding_volume::AABB<f32>>>()
                    .write_resource::<PresentData>()
                    .read_resource::<ShadowMappingData>()
                    .read_resource::<BaseColorDescriptorSet>()
                    .read_resource::<CullPassData>()
                    .read_resource::<Camera>()
                    .read_resource::<MainFramebuffer>()
                    .read_resource::<DebugAABBPassData>()
                    .read_resource::<GltfPassData>()
                    .build_thread_local(move |_commands, _world, resources, _queries| {
                        let (
                            ref renderer,
                            ref entities,
                            ref image_index,
                            ref mut graphics_command_pool,
                            ref model_data,
                            ref mut runtime_config,
                            ref camera_matrices,
                            ref swapchain,
                            ref consolidated_mesh_buffers,
                            ref aabb_storage,
                            ref mut present_data,
                            ref shadow_mapping_data,
                            ref base_color_descriptor_set,
                            ref cull_pass_data,
                            ref camera,
                            ref main_framebuffer,
                            ref debug_aabb_pass_data,
                            ref gltf_pass,
                        ) = resources;
                        let mut gui = gui.borrow_mut();
                        let gui_draw_data = gui.update(
                            &renderer,
                            &input_handler.borrow(),
                            &swapchain,
                            &camera,
                            &mut *runtime_config,
                        );
                        Renderer::exec(
                            &renderer,
                            &runtime_config,
                            &main_framebuffer,
                            &swapchain,
                            &entities,
                            &debug_aabb_pass_data,
                            &aabb_storage,
                            &mut gui_render,
                            &gui_draw_data,
                            &base_color_descriptor_set,
                            &consolidated_mesh_buffers,
                            &cull_pass_data,
                            &mut *present_data,
                            &image_index,
                            &model_data,
                            &gltf_pass,
                            &mut *graphics_command_pool,
                            &*shadow_mapping_data,
                            &*camera_matrices,
                        );
                    })
            })
            .add_system(
                SystemBuilder::<()>::new("PresentFramebuffer")
                    .read_resource::<RenderFrame>()
                    .read_resource::<ImageIndex>()
                    .read_resource::<Swapchain>()
                    .write_resource::<PresentData>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (ref renderer, ref image_index, ref swapchain, ref mut present_data) =
                            resources;
                        PresentFramebuffer::exec(
                            &renderer,
                            &present_data,
                            &swapchain,
                            &image_index,
                        );
                    }),
            )
            .add_system(
                SystemBuilder::<()>::new("MaintainECS")
                    .write_resource::<RenderFrame>()
                    .write_resource::<EntitiesStorage>()
                    .write_resource::<ComponentStorage<GltfMesh>>()
                    .write_resource::<PositionStorage>()
                    .write_resource::<RotationStorage>()
                    .write_resource::<ScaleStorage>()
                    .build(move |_commands, _world, resources, _queries| {
                        let (
                            ref mut renderer,
                            ref mut entities,
                            ref mut meshes,
                            ref mut positions,
                            ref mut rotations,
                            ref mut scales,
                        ) = resources;
                        {
                            let maintain_mask = entities.maintain();
                            meshes.maintain(&maintain_mask);
                            positions.0.maintain(&maintain_mask);
                            rotations.0.maintain(&maintain_mask);
                            scales.0.maintain(&maintain_mask);
                        }

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
            schedule.execute(&mut world);
            world.defrag(None);
        }
        if *quit_handle.lock() {
            unsafe {
                world
                    .resources
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
