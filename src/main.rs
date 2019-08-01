#![feature(arbitrary_self_types)]

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub mod ecs {
    pub mod components;
    pub mod custom;
    pub mod systems;
}
pub mod renderer;

use ash::version::DeviceV1_0;
use ecs::{components::*, custom::*, systems::*};
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
    let position_storage = &mut ComponentStorage::<na::Point3<f32>>::new();
    let rotation_storage = &mut ComponentStorage::<na::UnitQuaternion<f32>>::new();
    let scale_storage = &mut ComponentStorage::<f32>::new();
    let model_matrices_storage = &mut ComponentStorage::<glm::Mat4>::new();
    let aabb_storage = &mut ComponentStorage::<AABB>::new();
    let meshes_storage = &mut ComponentStorage::<GltfMesh>::new();
    let light_storage = &mut ComponentStorage::<Light>::new();
    let base_color_texture_storage = &mut ComponentStorage::<GltfMeshBaseColorTexture>::new();
    let base_color_visited_storage = &mut ComponentStorage::<BaseColorVisitedMarker>::new();
    let coarse_culled_storage = &mut ComponentStorage::<CoarseCulled>::new();
    let shadow_mapping_light_matrices_storage =
        &mut ComponentStorage::<ShadowMappingLightMatrices>::new();
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    let (mut renderer, events_loop) = RenderFrame::new();
    let renderer = &mut renderer;

    let quit_handle = Arc::new(Mutex::new(false));

    let present_data = &mut PresentData::new(&renderer);
    let image_index = &mut ImageIndex::default();
    let frame_timing = &mut FrameTiming::default();
    let input_handler = &mut InputHandler {
        events_loop,
        quit_handle: quit_handle.clone(),
        move_mouse: true,
    };
    let input_state = &mut InputState::default();
    let camera = &mut Camera::default();
    let fly_camera = &mut FlyCamera::default();
    let consolidated_mesh_buffers = &mut ConsolidatedMeshBuffers::new(&renderer);
    let graphics_command_pool = &mut GraphicsCommandPool::new(&renderer);

    let entities = &mut EntitiesStorage::new();

    let main_descriptor_pool = &mut MainDescriptorPool::new(&renderer);
    let camera_matrices = &mut CameraMatrices::new(&renderer, &main_descriptor_pool);

    let base_color_descriptor_set = &BaseColorDescriptorSet::new(&renderer, main_descriptor_pool);
    let model_data = &mut ModelData::new(&renderer, &main_descriptor_pool);

    let cull_pass_data = &CullPassData::new(
        &renderer,
        &model_data,
        main_descriptor_pool,
        &camera_matrices,
    );
    let cull_pass_data_private = &mut CullPassDataPrivate::new(&renderer);
    let main_attachments = &MainAttachments::new(&renderer);
    let depth_pass_data =
        &mut DepthPassData::new(&renderer, &model_data, &main_attachments, &camera_matrices);
    let shadow_mapping_data =
        &mut ShadowMappingData::new(&renderer, &depth_pass_data, main_descriptor_pool);
    let gui = &mut Gui::new(&renderer, &main_descriptor_pool);
    let gltf_pass = &GltfPassData::new(
        &renderer,
        &model_data,
        &base_color_descriptor_set,
        &shadow_mapping_data,
        &camera_matrices,
    );

    let main_framebuffer = &MainFramebuffer::new(&renderer, &main_attachments);

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

    let idx = &mut 0usize;
    'frame: loop {
        #[cfg(feature = "profiling")]
        microprofile::flip!();
        #[cfg(feature = "profiling")]
        microprofile::scope!("game-loop", "all");
        {
            #[cfg(feature = "profiling")]
            microprofile::scope!("game-loop", "ecs");
            // TODO: par_custom! needs to work with custom rayon threadpool
            {
                let entity_id = entities.allocate();
                position_storage.insert(entity_id, na::Point3::new(0.0, 0.0, *idx as f32 * 0.05));
                rotation_storage.insert(entity_id, na::UnitQuaternion::identity());
                scale_storage.insert(entity_id, 1.0);
                meshes_storage.insert(
                    entity_id,
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
                base_color_texture_storage.insert(
                    entity_id,
                    GltfMeshBaseColorTexture(Arc::clone(&box_base_color)),
                );
                if entity_id >= 400 {
                    entities.remove(std::cmp::max(250, entity_id - 200));
                    entities.remove(std::cmp::min(500, entity_id + 50));
                }
                *idx += 1;
            }
            AcquireFramebuffer::exec(renderer, present_data, image_index);
            CalculateFrameTiming::exec(frame_timing);
            input_handler.exec(input_state, camera);
            par_custom! {
                {
                    fly_camera.exec(input_state, frame_timing, camera);
                    ProjectCamera::exec(renderer, camera);
                }
                {
                    ConsolidateMeshBuffers::exec(renderer, entities, graphics_command_pool, meshes_storage, image_index, consolidated_mesh_buffers);
                }
            }
            par_custom! {
                {
                    ModelMatrixCalculation::exec(entities, position_storage, rotation_storage, scale_storage, model_matrices_storage);
                    AABBCalculation::exec(entities, model_matrices_storage, meshes_storage, aabb_storage);
                }
                {
                    ShadowMappingMVPCalculation::exec(renderer, entities, position_storage, rotation_storage, shadow_mapping_light_matrices_storage, light_storage, image_index, main_descriptor_pool, camera_matrices);
                }
            }
            CoarseCulling::exec(entities, aabb_storage, camera, coarse_culled_storage);
            SynchronizeBaseColorTextures::exec(
                entities,
                renderer,
                base_color_descriptor_set,
                base_color_texture_storage,
                image_index,
                base_color_visited_storage,
            );
            par_custom! {
                {
                    CameraMatricesUpload::exec(image_index, camera, camera_matrices);
                }
                {
                    ModelMatricesUpload::exec(model_matrices_storage, image_index, model_data);
                }
            }
            par_custom! {
                {
                    CullPass::exec(entities, renderer, cull_pass_data, cull_pass_data_private, meshes_storage, image_index, consolidated_mesh_buffers, position_storage, camera, model_data, camera_matrices);
                }
                {
                    PrepareShadowMaps::exec(entities, renderer, depth_pass_data, image_index, graphics_command_pool, shadow_mapping_data, meshes_storage, light_storage, shadow_mapping_light_matrices_storage, present_data, model_data);
                    DepthOnlyPass::exec(renderer, entities, image_index, meshes_storage, position_storage, camera, camera_matrices, depth_pass_data, model_data, graphics_command_pool, shadow_mapping_data);
                }
            }
            Renderer::exec(
                renderer,
                main_framebuffer,
                gui,
                base_color_descriptor_set,
                consolidated_mesh_buffers,
                cull_pass_data,
                present_data,
                image_index,
                depth_pass_data,
                model_data,
                gltf_pass,
                graphics_command_pool,
                shadow_mapping_data,
                camera_matrices,
            );
            PresentFramebuffer::exec(renderer, present_data, image_index);
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
