#![feature(arbitrary_self_types)]

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
#[cfg(feature = "microprofile")]
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
use rayon;
use renderer::*;
use std::sync::Arc;

fn main() {
    #[cfg(feature = "profiling")]
    microprofile::init!();
    let mut position_storage = ComponentStorage::<na::Point3<f32>>::new();
    let mut rotation_storage = ComponentStorage::<na::UnitQuaternion<f32>>::new();
    let mut scale_storage = ComponentStorage::<f32>::new();
    let mut model_matrices_storage = ComponentStorage::<glm::Mat4>::new();
    let mut aabb_storage = ComponentStorage::<AABB>::new();
    let mut meshes_storage = ComponentStorage::<GltfMesh>::new();
    let mut light_storage = ComponentStorage::<Light>::new();
    let mut projectile_velocities_storage = ComponentStorage::<f32>::new();
    let mut projectile_target_storage = ComponentStorage::<na::Point3<f32>>::new();
    let mut base_color_texture_storage = ComponentStorage::<GltfMeshBaseColorTexture>::new();
    let mut base_color_visited_storage = ComponentStorage::<BaseColorVisitedMarker>::new();
    let mut coarse_culled_storage = ComponentStorage::<CoarseCulled>::new();
    let mut shadow_mapping_light_matrices_storage =
        ComponentStorage::<ShadowMappingLightMatrices>::new();
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    let (renderer, mut swapchain, events_loop) = RenderFrame::new();

    let quit_handle = Arc::new(Mutex::new(false));

    let mut present_data = PresentData::new(&renderer);
    let mut image_index = ImageIndex::default();
    let mut frame_timing = FrameTiming::default();
    let mut input_state = InputState::default();
    let mut camera = Camera::default();
    let mut fly_camera = FlyCamera::default();
    let mut consolidated_mesh_buffers = ConsolidatedMeshBuffers::new(&renderer);
    let mut graphics_command_pool = GraphicsCommandPool::new(&renderer);

    let mut entities = EntitiesStorage::new();

    let mut main_descriptor_pool = MainDescriptorPool::new(&renderer);
    let mut camera_matrices = CameraMatrices::new(&renderer, &main_descriptor_pool);

    let base_color_descriptor_set =
        BaseColorDescriptorSet::new(&renderer, &mut main_descriptor_pool);
    let mut model_data = ModelData::new(&renderer, &main_descriptor_pool);

    let cull_pass_data = CullPassData::new(
        &renderer,
        &model_data,
        &mut main_descriptor_pool,
        &camera_matrices,
    );
    let mut cull_pass_data_private = CullPassDataPrivate::new(&renderer);
    let mut main_attachments = MainAttachments::new(&renderer, &swapchain);
    let mut depth_pass_data = DepthPassData::new(
        &renderer,
        &model_data,
        &main_attachments,
        &swapchain,
        &camera_matrices,
    );
    let mut shadow_mapping_data =
        ShadowMappingData::new(&renderer, &depth_pass_data, &mut main_descriptor_pool);
    let mut gui = Gui::new(&renderer, &main_descriptor_pool);

    let mut imgui_platform = WinitPlatform::init(&mut gui.imgui);
    imgui_platform.attach_window(
        gui.imgui.io_mut(),
        &renderer.instance.window,
        HiDpiMode::Locked(1.0), // TODO: Revert this to Default if we can make the app obey winit DPI
    );
    let mut input_handler = InputHandler {
        events_loop,
        quit_handle: quit_handle.clone(),
        move_mouse: false,
        imgui_platform,
    };

    let gltf_pass = GltfPassData::new(
        &renderer,
        &model_data,
        &base_color_descriptor_set,
        &shadow_mapping_data,
        &camera_matrices,
    );

    let mut main_framebuffer = MainFramebuffer::new(&renderer, &main_attachments, &swapchain);

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
        &renderer,
        &graphics_command_pool,
        "vendor/glTF-Sample-Models/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf",
    );

    let vertex_buffer = Arc::new(vertex_buffer);
    let normal_buffer = Arc::new(normal_buffer);
    let uv_buffer = Arc::new(uv_buffer);
    let index_buffers = Arc::new(index_buffers);
    let base_color = Arc::new(base_color);

    let ixes = entities.allocate_mask(207);
    debug_assert_eq!(ixes.to_vec(), (0..207).collect::<Vec<_>>());

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
            aabb_c,
            aabb_h,
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
            aabb_c,
            aabb_h,
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
            aabb_c,
            aabb_h,
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
        box_aabb_c,
        box_aabb_h,
    ) = {
        let LoadedMesh {
            vertex_buffer,
            normal_buffer,
            uv_buffer,
            index_buffers,
            vertex_len,
            aabb_c,
            aabb_h,
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
            aabb_c,
            aabb_h,
        )
    };

    let mesh_library = MeshLibrary {
        projectile: GltfMesh {
            vertex_buffer: Arc::clone(&box_vertex_buffer),
            normal_buffer: Arc::clone(&box_normal_buffer),
            uv_buffer: Arc::clone(&box_uv_buffer),
            index_buffers: Arc::clone(&box_index_buffers),
            vertex_len: box_vertex_len,
            aabb_c: box_aabb_c,
            aabb_h: box_aabb_h,
        },
        projectile_texture: Arc::clone(&box_base_color),
    };

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
            aabb_c: box_aabb_c,
            aabb_h: box_aabb_h,
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
            aabb_c: box_aabb_c,
            aabb_h: box_aabb_h,
        },
    );
    base_color_texture_storage.insert(6, GltfMeshBaseColorTexture(Arc::clone(&box_base_color)));

    for ix in 7..207 {
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
                aabb_c,
                aabb_h,
            },
        );
        base_color_texture_storage.insert(ix, GltfMeshBaseColorTexture(Arc::clone(&base_color)));
    }

    'frame: loop {
        #[cfg(feature = "profiling")]
        microprofile::flip!();
        #[cfg(feature = "profiling")]
        microprofile::scope!("game-loop", "all");
        {
            #[cfg(feature = "profiling")]
            microprofile::scope!("game-loop", "ecs");
            let window_resized = input_handler.exec(
                &renderer.instance.window,
                &mut gui.imgui,
                &mut input_state,
                &mut camera,
            );

            if window_resized {
                unsafe {
                    renderer.device.device_wait_idle().unwrap();
                }
                swapchain.resize_to_fit();
                main_attachments = MainAttachments::new(&renderer, &swapchain);
                depth_pass_data = DepthPassData::new(
                    &renderer,
                    &model_data,
                    &main_attachments,
                    &swapchain,
                    &camera_matrices,
                );
                main_framebuffer = MainFramebuffer::new(&renderer, &main_attachments, &swapchain);
                present_data = PresentData::new(&renderer);
            }

            let swapchain_needs_recreating =
                AcquireFramebuffer::exec(&renderer, &present_data, &swapchain, &mut image_index);
            if swapchain_needs_recreating {
                unsafe {
                    renderer.device.device_wait_idle().unwrap();
                }
                swapchain.resize_to_fit();
                main_attachments = MainAttachments::new(&renderer, &swapchain);
                depth_pass_data = DepthPassData::new(
                    &renderer,
                    &model_data,
                    &main_attachments,
                    &swapchain,
                    &camera_matrices,
                );
                main_framebuffer = MainFramebuffer::new(&renderer, &main_attachments, &swapchain);
                present_data = PresentData::new(&renderer);
                AcquireFramebuffer::exec(&renderer, &present_data, &swapchain, &mut image_index);
            }

            CalculateFrameTiming::exec(&mut frame_timing);
            fly_camera.exec(&input_state, &frame_timing, &mut camera);
            ProjectCamera::exec(&swapchain, &mut camera);
            LaunchProjectileTest::exec(
                &mut entities,
                &mut position_storage,
                &mut rotation_storage,
                &mut scale_storage,
                &mut meshes_storage,
                &mut base_color_texture_storage,
                &mut projectile_target_storage,
                &mut projectile_velocities_storage,
                &mut camera,
                &mesh_library,
                &input_state,
            );
            UpdateProjectiles::exec(
                &mut entities,
                &mut position_storage,
                &rotation_storage,
                &projectile_target_storage,
                &mut projectile_velocities_storage,
                &frame_timing,
            );
            rayon::join(
                || {
                    ConsolidateMeshBuffers::exec(
                        &renderer,
                        &entities,
                        &graphics_command_pool,
                        &meshes_storage,
                        &image_index,
                        &mut consolidated_mesh_buffers,
                    )
                },
                || {
                    rayon::join(
                        || {
                            ModelMatrixCalculation::exec(
                                &entities,
                                &position_storage,
                                &rotation_storage,
                                &scale_storage,
                                &mut model_matrices_storage,
                            );
                            AABBCalculation::exec(
                                &entities,
                                &model_matrices_storage,
                                &meshes_storage,
                                &mut aabb_storage,
                            );
                        },
                        || {
                            ShadowMappingMVPCalculation::exec(
                                &renderer,
                                &entities,
                                &position_storage,
                                &rotation_storage,
                                &mut shadow_mapping_light_matrices_storage,
                                &light_storage,
                                &image_index,
                                &main_descriptor_pool,
                                &camera_matrices,
                            );
                        },
                    )
                },
            );
            rayon::join(
                || {
                    rayon::join(
                        || {
                            CoarseCulling::exec(
                                &entities,
                                &aabb_storage,
                                &camera,
                                &mut coarse_culled_storage,
                            )
                        },
                        || {
                            SynchronizeBaseColorTextures::exec(
                                &entities,
                                &renderer,
                                &base_color_descriptor_set,
                                &base_color_texture_storage,
                                &image_index,
                                &mut base_color_visited_storage,
                            )
                        },
                    );
                },
                || {
                    rayon::join(
                        || CameraMatricesUpload::exec(&image_index, &camera, &mut camera_matrices),
                        || {
                            ModelMatricesUpload::exec(
                                &model_matrices_storage,
                                &image_index,
                                &mut model_data,
                            )
                        },
                    );
                },
            );
            rayon::join(
                || {
                    CullPass::exec(
                        &entities,
                        &renderer,
                        &cull_pass_data,
                        &mut cull_pass_data_private,
                        &meshes_storage,
                        &image_index,
                        &consolidated_mesh_buffers,
                        &position_storage,
                        &camera,
                        &model_data,
                        &camera_matrices,
                    )
                },
                || {
                    PrepareShadowMaps::exec(
                        &entities,
                        &renderer,
                        &depth_pass_data,
                        &image_index,
                        &mut graphics_command_pool,
                        &mut shadow_mapping_data,
                        &meshes_storage,
                        &light_storage,
                        &shadow_mapping_light_matrices_storage,
                        &present_data,
                        &model_data,
                    );
                    DepthOnlyPass::exec(
                        &renderer,
                        &entities,
                        &image_index,
                        &meshes_storage,
                        &position_storage,
                        &camera,
                        &camera_matrices,
                        &mut depth_pass_data,
                        &swapchain,
                        &model_data,
                        &mut graphics_command_pool,
                        &shadow_mapping_data,
                    );
                },
            );
            Renderer::exec(
                &renderer,
                &main_framebuffer,
                &swapchain,
                &mut gui,
                &input_handler,
                &base_color_descriptor_set,
                &consolidated_mesh_buffers,
                &cull_pass_data,
                &mut present_data,
                &image_index,
                &depth_pass_data,
                &model_data,
                &gltf_pass,
                &mut graphics_command_pool,
                &shadow_mapping_data,
                &camera_matrices,
            );
            PresentFramebuffer::exec(&renderer, &present_data, &swapchain, &image_index);
            {
                let maintain_mask = entities.maintain();
                meshes_storage.maintain(&maintain_mask);
                position_storage.maintain(&maintain_mask);
                rotation_storage.maintain(&maintain_mask);
                scale_storage.maintain(&maintain_mask);
            }
        }
        if *quit_handle.lock() {
            unsafe {
                renderer.device.device_wait_idle().unwrap();
            }
            break 'frame;
        }
    }
    #[cfg(feature = "profiling")]
    microprofile::shutdown!();
}
