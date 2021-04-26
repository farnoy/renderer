#![feature(backtrace)]
#![allow(clippy::new_without_default)]
#![warn(clippy::cast_lossless)]
#![warn(
    unreachable_pub,
    single_use_lifetimes,
    unused_qualifications,
    absolute_paths_not_starting_with_crate,
    macro_use_extern_crate
)]

extern crate static_assertions;

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub(crate) mod ecs;
pub(crate) mod renderer;

use std::sync::Arc;

use ash::version::DeviceV1_0;
use bevy_app::*;
use bevy_ecs::{
    component::{ComponentDescriptor, StorageType},
    prelude::*,
};
use ecs::{
    components::{Deleting, Light, ModelMatrix, Position, Rotation, Scale, AABB},
    resources::{Camera, InputActions, MeshLibrary},
    systems::{
        aabb_calculation, assign_draw_index, calculate_frame_timing, camera_controller, cleanup_deleted_entities,
        launch_projectiles_test, model_matrix_calculation, project_camera, update_projectiles, FrameTiming, Gui,
        InputHandler, RuntimeConfiguration,
    },
};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
#[cfg(feature = "crash_debugging")]
use renderer::CrashBuffer;
use renderer::{
    acquire_framebuffer, camera_matrices_upload, cleanup_base_color_markers, coarse_culling, consolidate_mesh_buffers,
    graphics_stage, load_gltf, model_matrices_upload, shadow_mapping_mvp_calculation,
    synchronize_base_color_textures_visit, up_vector, update_shadow_map_descriptors, BaseColorDescriptorSet,
    BaseColorVisitedMarker, CameraMatrices, CoarseCulled, ConsolidatedMeshBuffers, CopiedResource, CullPassData,
    CullPassDataPrivate, DebugAABBPassData, DepthPassData, DrawIndex, GltfMesh, GltfMeshBaseColorTexture, GltfPassData,
    GuiRenderData, ImageIndex, LoadedMesh, LocalGraphicsCommandPool, LocalTransferCommandPool, MainAttachments,
    MainDescriptorPool, MainFramebuffer, MainPassCommandBuffer, MainRenderpass, ModelData, PresentData, RenderFrame,
    Resized, ShadowMappingData, ShadowMappingLightMatrices, SwapchainIndexToFrameNumber,
};
#[cfg(feature = "shader_reload")]
use renderer::{reload_shaders, ReloadedShaders, ShaderReload};

fn main() {
    microprofile::init!();
    microprofile::on_thread_create!("main");

    env_logger::init();

    let (renderer, swapchain, events_loop) = RenderFrame::new();

    let mut app = App::build();

    app.register_component(ComponentDescriptor::new::<BaseColorVisitedMarker>(
        StorageType::SparseSet,
    ));

    app.register_component(ComponentDescriptor::new::<Deleting>(StorageType::SparseSet));

    let quit_handle = Arc::new(Mutex::new(false));

    let present_data = PresentData::new(&renderer);
    let image_index = ImageIndex::default();
    let consolidated_mesh_buffers = ConsolidatedMeshBuffers::new(&renderer);

    let mut main_descriptor_pool = MainDescriptorPool::new(&renderer);
    let camera_matrices = CameraMatrices::new(&renderer, &main_descriptor_pool);

    let base_color_descriptor_set = BaseColorDescriptorSet::new(&renderer, &mut main_descriptor_pool);
    let model_data = ModelData::new(&renderer, &main_descriptor_pool);

    let cull_pass_data = CullPassData::new(&renderer, &mut main_descriptor_pool, &consolidated_mesh_buffers);
    let cull_pass_data_private = CullPassDataPrivate::new(&renderer, &cull_pass_data, &model_data, &camera_matrices);

    let main_attachments = MainAttachments::new(&renderer, &swapchain);
    let main_renderpass = MainRenderpass::new(&renderer, &main_attachments);
    let depth_pass_data = DepthPassData::new(&renderer, &model_data, &camera_matrices, &main_renderpass);
    let shadow_mapping_data = ShadowMappingData::new(&renderer, &depth_pass_data, &mut main_descriptor_pool);

    let mut gui = Gui::new();
    let mut imgui_platform = WinitPlatform::init(&mut gui.imgui);
    imgui_platform.attach_window(
        gui.imgui.io_mut(),
        &renderer.instance.window,
        HiDpiMode::Locked(1.0), // TODO: Revert this to Default if we can make the app obey winit DPI
    );

    let input_handler = InputHandler {
        events_loop,
        quit_handle: quit_handle.clone(),
        imgui_platform,
    };

    let gltf_pass = GltfPassData::new(
        &renderer,
        &main_renderpass,
        &model_data,
        &base_color_descriptor_set,
        &shadow_mapping_data,
        &camera_matrices,
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
    app.app.world.spawn_batch(vec![
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
    app.app.world.spawn_batch(vec![
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
            Rotation(na::UnitQuaternion::from_axis_angle(&up_vector(), f32::pi() / 2.0)),
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
            Rotation(na::UnitQuaternion::from_axis_angle(&up_vector(), f32::pi() / 3.0)),
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

    drop(box_vertex_buffer);
    drop(box_normal_buffer);
    drop(box_uv_buffer);
    drop(box_index_buffers);
    drop(box_base_color);

    app.app.world.spawn_batch((7..max_entities).map(|ix| {
        let angle = f32::pi() * (ix as f32 * 20.0) / 180.0;
        let rot = na::Rotation3::from_axis_angle(&na::Unit::new_normalize(na::Vector3::y()), angle);
        let pos = rot.transform_point(&na::Point3::new(0.0, (ix as f32 * -0.01) + 2.0, 5.0 + (ix / 10) as f32));

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

    drop(vertex_buffer);
    drop(normal_buffer);
    drop(uv_buffer);
    drop(index_buffers);
    drop(base_color);

    app.insert_resource(swapchain);
    app.insert_resource(mesh_library);
    app.insert_resource(Resized(false));
    app.init_resource::<FrameTiming>();
    app.init_resource::<InputActions>();
    app.init_resource::<Camera>();
    app.init_resource::<RuntimeConfiguration>();
    app.init_resource::<CopiedResource<RuntimeConfiguration>>();
    app.init_resource::<CopiedResource<Camera>>();
    app.insert_resource(renderer);
    app.insert_resource(image_index);
    app.insert_resource(consolidated_mesh_buffers);
    app.insert_resource(main_descriptor_pool);
    app.insert_resource(camera_matrices);
    app.insert_resource(base_color_descriptor_set);
    app.insert_resource(model_data);
    app.insert_resource(cull_pass_data);
    app.insert_resource(cull_pass_data_private);
    app.insert_resource(shadow_mapping_data);
    app.insert_resource(depth_pass_data);
    app.insert_resource(present_data);
    app.insert_resource(main_renderpass);
    app.insert_resource(main_attachments);
    app.insert_resource(gltf_pass);
    app.init_resource::<SwapchainIndexToFrameNumber>();
    app.init_resource::<CopiedResource<SwapchainIndexToFrameNumber>>();
    app.init_resource::<DebugAABBPassData>();
    app.init_resource::<MainFramebuffer>();
    app.insert_non_send_resource(input_handler);
    app.insert_non_send_resource(gui);
    app.init_resource::<GuiRenderData>();
    #[cfg(feature = "crash_debugging")]
    app.init_resource::<CrashBuffer>();
    #[cfg(feature = "shader_reload")]
    {
        app.init_non_send_resource::<ShaderReload>();
        app.init_resource::<ReloadedShaders>();
    }
    app.init_resource::<LocalTransferCommandPool<0>>();
    app.init_resource::<LocalGraphicsCommandPool<0>>();
    app.init_resource::<LocalGraphicsCommandPool<1>>();
    app.init_resource::<LocalGraphicsCommandPool<2>>();
    app.init_resource::<MainPassCommandBuffer>();

    app.add_plugin(bevy_log::LogPlugin);

    #[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
    enum UpdatePhase {
        Input,
        AcquireFramebuffer,
        CalculateFrameTiming,
        AssignDrawIndex,
        CameraController,
        RenderSetup,
        ProjectCamera,
        LaunchProjectiles,
        UpdateProjectiles,
        Gameplay,
        #[cfg(feature = "shader_reload")]
        ReloadShaders,
    }

    use UpdatePhase::*;

    app.add_system(InputHandler::run.system().label(Input));
    #[cfg(feature = "shader_reload")]
    app.add_system(reload_shaders.system().label(ReloadShaders));
    app.add_system(acquire_framebuffer.system().label(AcquireFramebuffer).after(Input));
    app.add_system(
        calculate_frame_timing
            .system()
            .label(CalculateFrameTiming)
            .after(AcquireFramebuffer),
    );

    app.add_system_set(
        SystemSet::new()
            .label(Gameplay)
            .with_system(
                camera_controller
                    .system()
                    .label(CameraController)
                    .after(CalculateFrameTiming),
            )
            .with_system(
                project_camera
                    .system()
                    .label(ProjectCamera)
                    .after(CameraController)
                    .after(AcquireFramebuffer),
            )
            .with_system(
                launch_projectiles_test
                    .system()
                    .label(LaunchProjectiles)
                    .after(ProjectCamera)
                    .after(CalculateFrameTiming),
            )
            .with_system(
                update_projectiles
                    .system()
                    .label(UpdateProjectiles)
                    .after(CalculateFrameTiming),
            ),
    );

    #[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
    enum RenderSetup {
        ConsolidateMeshBuffers,
        ModelMatrixCalculation,
        AABBCalculation,
        ShadowMappingMVPCalculation,
        CoarseCulling,
        SynchronizeBaseColorTexturesVisit,
        CameraMatricesUpload,
        ModelMatricesUpload,
        UpdateShadowMapDescriptors,
    }

    app.add_system_set(
        SystemSet::new()
            .label(RenderSetup)
            .after(AcquireFramebuffer)
            .after(Gameplay)
            .with_system(assign_draw_index.system().label(AssignDrawIndex))
            .with_system(
                consolidate_mesh_buffers
                    .system()
                    .label(RenderSetup::ConsolidateMeshBuffers),
            )
            .with_system(
                model_matrix_calculation
                    .system()
                    .label(RenderSetup::ModelMatrixCalculation),
            )
            .with_system(
                aabb_calculation
                    .system()
                    .label(RenderSetup::AABBCalculation)
                    .after(RenderSetup::ModelMatrixCalculation),
            )
            .with_system(
                shadow_mapping_mvp_calculation
                    .system()
                    .label(RenderSetup::ShadowMappingMVPCalculation),
            )
            .with_system(
                coarse_culling
                    .system()
                    .label(RenderSetup::CoarseCulling)
                    .after(RenderSetup::AABBCalculation),
            )
            .with_system(
                synchronize_base_color_textures_visit
                    .system()
                    .label(RenderSetup::SynchronizeBaseColorTexturesVisit),
            )
            .with_system(
                camera_matrices_upload
                    .system()
                    .label(RenderSetup::CameraMatricesUpload)
                    .after(ProjectCamera),
            )
            .with_system(
                model_matrices_upload
                    .system()
                    .label(RenderSetup::ModelMatricesUpload)
                    .after(AssignDrawIndex)
                    .after(RenderSetup::ModelMatrixCalculation),
            )
            .with_system(
                update_shadow_map_descriptors
                    .system()
                    .label(RenderSetup::UpdateShadowMapDescriptors)
                    .after(RenderSetup::ShadowMappingMVPCalculation),
            ),
    );

    app.add_stage_after(CoreStage::Update, "graphics work", graphics_stage());

    #[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
    enum CleanupPhases {
        CleanupBaseColorMarkers,
        CleanupDeletedEntities,
        IncrementFrameNumber,
    }

    use CleanupPhases::*;

    app.add_stage_after(
        CoreStage::PostUpdate,
        "cleanup",
        SystemStage::parallel()
            .with_system(
                cleanup_base_color_markers
                    .exclusive_system()
                    .at_end()
                    .label(CleanupBaseColorMarkers)
                    .before(CleanupDeletedEntities),
            )
            .with_system(
                cleanup_deleted_entities
                    .exclusive_system()
                    .at_end()
                    .label(CleanupDeletedEntities),
            )
            .with_system(
                (|mut renderer: ResMut<RenderFrame>| {
                    renderer.frame_number += 1;
                })
                .exclusive_system()
                .at_end()
                .label(IncrementFrameNumber)
                .after(CleanupDeletedEntities),
            ),
    );

    app.insert_resource(bevy_tasks::ComputeTaskPool(bevy_tasks::TaskPool::new()));

    if cfg!(debug_assertions) {
        app.insert_resource(bevy_ecs::schedule::ReportExecutionOrderAmbiguities);
    }

    'frame: loop {
        microprofile::flip!();

        microprofile::scope!("game-loop", "all");

        app.app.update();

        if *quit_handle.lock() {
            unsafe {
                app.app
                    .world
                    .get_resource::<RenderFrame>()
                    .unwrap()
                    .device
                    .device_wait_idle()
                    .unwrap();
            }
            break 'frame;
        }
    }

    let render_frame = app.app.world.remove_resource::<RenderFrame>().unwrap();

    app.app
        .world
        .remove_resource::<DepthPassData>()
        .unwrap()
        .destroy(&render_frame.device);

    app.app
        .world
        .remove_resource::<GltfPassData>()
        .unwrap()
        .destroy(&render_frame.device);

    app.app
        .world
        .remove_resource::<CullPassDataPrivate>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<ConsolidatedMeshBuffers>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<DebugAABBPassData>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<MainRenderpass>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<MainAttachments>()
        .unwrap()
        .destroy(&render_frame.device);
    let main_descriptor_pool = app.app.world.remove_resource::<MainDescriptorPool>().unwrap();
    app.app
        .world
        .remove_resource::<CullPassData>()
        .unwrap()
        .destroy(&render_frame.device, &main_descriptor_pool);
    app.app
        .world
        .remove_resource::<BaseColorDescriptorSet>()
        .unwrap()
        .destroy(&render_frame.device, &main_descriptor_pool);
    app.app
        .world
        .remove_resource::<ShadowMappingData>()
        .unwrap()
        .destroy(&render_frame.device, &main_descriptor_pool);
    app.app
        .world
        .remove_resource::<GuiRenderData>()
        .unwrap()
        .destroy(&render_frame.device, &main_descriptor_pool);
    app.app
        .world
        .remove_resource::<CameraMatrices>()
        .unwrap()
        .destroy(&render_frame.device, &main_descriptor_pool);
    app.app
        .world
        .remove_resource::<ModelData>()
        .unwrap()
        .destroy(&render_frame.device, &main_descriptor_pool);
    app.app
        .world
        .remove_resource::<MeshLibrary>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<MainFramebuffer>()
        .unwrap()
        .destroy(&render_frame.device);

    let entities = app
        .app
        .world
        .query_filtered::<Entity, With<BaseColorVisitedMarker>>()
        .iter(&app.app.world)
        .collect::<Vec<_>>();
    for entity in entities {
        let marker = app
            .app
            .world
            .entity_mut(entity)
            .remove::<BaseColorVisitedMarker>()
            .unwrap();
        marker.destroy(&render_frame.device);
    }

    let entities = app
        .app
        .world
        .query_filtered::<Entity, With<GltfMeshBaseColorTexture>>()
        .iter(&app.app.world)
        .collect::<Vec<_>>();
    for entity in entities {
        let marker = app
            .app
            .world
            .entity_mut(entity)
            .remove::<GltfMeshBaseColorTexture>()
            .unwrap();
        drop(Arc::try_unwrap(marker.0).map(|im| im.destroy(&render_frame.device)));
    }

    let entities = app
        .app
        .world
        .query_filtered::<Entity, With<ShadowMappingLightMatrices>>()
        .iter(&app.app.world)
        .collect::<Vec<_>>();
    for entity in entities {
        let light_matrices = app
            .app
            .world
            .entity_mut(entity)
            .remove::<ShadowMappingLightMatrices>()
            .unwrap();
        light_matrices.destroy(&render_frame.device, &main_descriptor_pool);
    }

    let entities = app
        .app
        .world
        .query_filtered::<Entity, With<GltfMesh>>()
        .iter(&app.app.world)
        .collect::<Vec<_>>();
    for entity in entities {
        app.app
            .world
            .entity_mut(entity)
            .remove::<GltfMesh>()
            .unwrap()
            .destroy(&render_frame.device);
    }

    #[cfg(feature = "crash_debugging")]
    app.app
        .world
        .remove_resource::<CrashBuffer>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<LocalTransferCommandPool<0>>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<LocalGraphicsCommandPool<0>>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<LocalGraphicsCommandPool<1>>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<LocalGraphicsCommandPool<2>>()
        .unwrap()
        .destroy(&render_frame.device);
    app.app
        .world
        .remove_resource::<PresentData>()
        .unwrap()
        .destroy(&render_frame.device);
    drop(app);

    main_descriptor_pool.destroy(&render_frame.device);
    render_frame.destroy();

    /* debug dependency graph
    let mut last: &dyn StageLabel = &"Start";
    let mut last_ix = 0usize;
    let stage_clusters: Vec<Box<dyn SystemLabel>> =
        vec![Gameplay.dyn_clone(), RenderSetup.dyn_clone()];
    println!("digraph G {{");
    println!("  graph [compound=true];");
    println!("  rankdir=LR;");
    for (ix, (label, stage)) in app.app.schedule.iter_stages().enumerate() {
        println!("  subgraph cluster_{} {{", ix);
        println!("    label = {:?};", label);
        let system_stage = match stage.downcast_ref::<SystemStage>() {
            Some(s) => s,
            None => {
                println!("    // unknown stage\n  }}");
                continue;
            }
        };
        let systems = system_stage.parallel_systems();
        println!(
            "    {:?} [label=single style={}];",
            label,
            if systems.len() < 1 { "filled" } else { "invis" }
        );
        for stage_cluster in stage_clusters.iter() {
            println!(
                "    subgraph cluster_{:?} {{ label = {:?}; }}",
                stage_cluster, stage_cluster
            );
        }
        let casted = unsafe {
            std::mem::transmute::<_, &[bevy_ecs::schedule::ParallelSystemContainer]>(systems)
        };
        let mut names =
            hashbrown::HashMap::<Box<dyn SystemLabel>, Vec<bevy_ecs::system::SystemId>>::with_capacity(
                casted.len(),
            );
        for system in casted {
            use bevy_ecs::schedule::GraphNode;
            for label in system.labels() {
                names
                    .entry(label.clone())
                    .or_default()
                    .push(system.system().id());
            }
        }
        let mut connected_clusters: Vec<(Box<dyn SystemLabel>, Box<dyn SystemLabel>)> = vec![];
        for system in casted {
            use bevy_ecs::schedule::GraphNode;
            let mut in_cluster = None;
            'outer: for stage_cluster in stage_clusters.iter() {
                for label in system.labels() {
                    if stage_cluster == label {
                        in_cluster = Some(stage_cluster.dyn_clone());
                        break 'outer;
                    }
                }
            }
            println!(
                "    {} {} [label={:?}]; {}",
                match in_cluster {
                    Some(ref cluster) => format!("subgraph cluster_{:?} {{", cluster),
                    None => "".to_string(),
                },
                system.system().id().0,
                system
                    .labels()
                    .get(0)
                    .unwrap_or(&SystemLabel::dyn_clone(&system.name())),
                if in_cluster.is_some() { "}" } else { "" },
            );
            for downstream_dep in system.before() {
                let mut downstream_cluster = None;
                for stage_cluster in stage_clusters.iter() {
                    if stage_cluster == downstream_dep {
                        downstream_cluster = Some(stage_cluster.dyn_clone());
                    }
                }
                for downstream_system in names.get(downstream_dep).unwrap() {
                    println!("    {} -> {};", system.system().id().0, downstream_system.0,);
                }
                for downstream_system in names.get(downstream_dep).unwrap() {
                    match (&downstream_cluster, &in_cluster) {
                        (Some(ref downstream_cluster), Some(ref in_cluster)) => {
                            if !connected_clusters
                                .iter()
                                .any(|(ref x, ref y)| (in_cluster, downstream_cluster) == (x, y))
                            {
                                println!(
                                    "    {} -> {} [ltail=cluster_{:?} lhead=cluster_{:?}];",
                                    system.system().id().0,
                                    downstream_system.0,
                                    in_cluster,
                                    downstream_cluster,
                                );
                                connected_clusters
                                    .push((in_cluster.dyn_clone(), downstream_cluster.dyn_clone()));
                            }
                        }
                        _ => println!("    {} -> {};", system.system().id().0, downstream_system.0),
                    }
                }
            }
            for upstream_dep in system.after() {
                let mut upstream_cluster = None;
                for stage_cluster in stage_clusters.iter() {
                    if stage_cluster == upstream_dep {
                        upstream_cluster = Some(stage_cluster.dyn_clone());
                    }
                }
                for upstream_system in names.get(upstream_dep).unwrap() {
                    match (&upstream_cluster, &in_cluster) {
                        (Some(ref upstream_cluster), Some(ref in_cluster)) => {
                            if !connected_clusters
                                .iter()
                                .any(|(ref x, ref y)| (upstream_cluster, in_cluster) == (x, y))
                            {
                                println!(
                                    "    {} -> {} [ltail=cluster_{:?} lhead=cluster_{:?}];",
                                    upstream_system.0,
                                    system.system().id().0,
                                    upstream_cluster,
                                    in_cluster,
                                );
                                connected_clusters
                                    .push((upstream_cluster.dyn_clone(), in_cluster.dyn_clone()));
                            }
                        }
                        _ => println!("    {} -> {};", upstream_system.0, system.system().id().0),
                    }
                }
            }
        }
        println!("  }}");
        println!(
            "  {:?} -> {:?} [lhead=cluster_{} ltail=cluster_{}];",
            last, label, ix, last_ix
        );
        last = label;
        last_ix = ix;
    }

    println!("  {:?} -> End;", last);
    println!("}}");

    */

    microprofile::shutdown!();
}
