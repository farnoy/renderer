mod camera_controller;
mod input;

pub(crate) mod components;
pub(crate) mod resources {
    use std::sync::Arc;

    pub(crate) use super::{camera_controller::Camera, input::InputActions};
    use crate::renderer::{
        device::{Device, Image},
        GltfMesh,
    };

    pub(crate) struct MeshLibrary {
        pub(crate) projectile: GltfMesh,
        pub(crate) projectile_base_color: Arc<Image>,
        pub(crate) projectile_normal_map: Arc<Image>,
    }

    impl MeshLibrary {
        pub(crate) fn destroy(self, device: &Device) {
            self.projectile.destroy(device);
            drop(Arc::try_unwrap(self.projectile_base_color).map(|image| image.destroy(device)));
            drop(Arc::try_unwrap(self.projectile_normal_map).map(|image| image.destroy(device)));
        }
    }
}
pub(crate) mod systems {
    use std::{sync::Arc, time::Instant};

    use bevy_ecs::prelude::*;
    use na::RealField;
    use profiling::scope;
    use winit::{self, event::MouseButton};

    pub(crate) use super::{camera_controller::camera_controller, input::InputHandler};
    #[cfg(feature = "shader_reload")]
    pub(crate) use crate::renderer::ReloadedShaders;
    use crate::{
        ecs::{
            components::{
                Deleting, ModelMatrix, Position, ProjectileTarget, ProjectileVelocity, Rotation, Scale, AABB,
            },
            resources::{Camera, InputActions, MeshLibrary},
        },
        renderer::{
            forward_vector, up_vector, CoarseCulled, DrawIndex, GltfMesh, GltfMeshBaseColorTexture,
            GltfMeshNormalTexture, ImageIndex, RenderFrame, Swapchain, INITIAL_WORKGROUP_SIZE,
        },
    };

    pub(crate) fn model_matrix_calculation(
        task_pool: Res<bevy_tasks::ComputeTaskPool>,
        mut query: Query<(&Position, &Rotation, &Scale, &mut ModelMatrix)>,
    ) {
        scope!("ecs::ModelMatrixCalculation");

        query.par_for_each_mut(&task_pool, 2, |(pos, rot, scale, mut model_matrix)| {
            scope!("parallel::ModelMatrixCalculation");

            model_matrix.0 =
                glm::translation(&pos.0.coords) * rot.0.to_homogeneous() * glm::scaling(&glm::Vec3::repeat(scale.0));
        });
    }

    pub(crate) fn project_camera(swapchain: Res<Swapchain>, mut camera: ResMut<Camera>) {
        scope!("ecs::project_camera");

        let near = 0.1;
        let far = 100.0;
        let aspect = swapchain.width as f32 / swapchain.height as f32;
        let fov_y = glm::radians(&glm::vec1(70.0));

        camera.projection = glm::perspective_lh_zo(aspect, fov_y.x, near, far);

        let dir = camera.rotation.transform_vector(&forward_vector());
        let extended_forward = camera.position + dir;
        let up = camera.rotation.transform_vector(&up_vector());

        camera.view = glm::look_at_lh(&camera.position.coords, &extended_forward.coords, &up);

        let m = camera.projection * camera.view;
        camera.frustum_planes = [
            -(m.row(3) + m.row(0)).transpose(),
            -(m.row(3) - m.row(0)).transpose(),
            -(m.row(3) + m.row(1)).transpose(),
            -(m.row(3) - m.row(1)).transpose(),
            -(m.row(3) + m.row(2)).transpose(),
            -(m.row(3) - m.row(2)).transpose(),
        ];
    }

    pub(crate) struct FrameTiming {
        pub(crate) previous_frame: Instant,
        pub(crate) time_delta: f32,
    }

    impl Default for FrameTiming {
        fn default() -> FrameTiming {
            FrameTiming {
                previous_frame: Instant::now(),
                time_delta: 0.0,
            }
        }
    }

    pub(crate) fn calculate_frame_timing(mut frame_timing: ResMut<FrameTiming>) {
        scope!("ecs::CalculateFrameTiming");

        let now = Instant::now();
        let duration = now - frame_timing.previous_frame;
        frame_timing.time_delta = duration.as_secs() as f32 + (duration.subsec_micros() as f32 / 1e6);
        frame_timing.previous_frame = now;
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn assign_draw_index(
        mut query: Query<
            &mut DrawIndex,
            (
                With<Position>,
                With<GltfMeshBaseColorTexture>,
                With<GltfMesh>,
                Without<Deleting>,
            ),
        >,
    ) {
        scope!("ecs::AssignDrawIndex");

        let mut counter = 0u32;

        query.for_each_mut(|mut draw_idx| {
            draw_idx.0 = counter;
            counter += 1;
        });
    }

    pub(crate) fn aabb_calculation(
        task_pool: Res<bevy_tasks::ComputeTaskPool>,
        mut query: Query<(&ModelMatrix, &GltfMesh, &mut AABB)>,
    ) {
        scope!("ecs::AABBCalculation");

        query.par_for_each_mut(&task_pool, 2, |(model_matrix, mesh, mut aabb)| {
            scope!("parallel::AABBCalculation");

            let min = mesh.aabb.mins;
            let max = mesh.aabb.maxs;
            let (min, max) = [
                // bottom half (min y)
                na::Point3::new(min.x, min.y, min.z),
                na::Point3::new(max.x, min.y, min.z),
                na::Point3::new(min.x, min.y, max.z),
                na::Point3::new(max.x, min.y, max.z),
                // top half (max y)
                na::Point3::new(min.x, max.y, min.z),
                na::Point3::new(max.x, max.y, min.z),
                na::Point3::new(min.x, max.y, max.z),
                na::Point3::new(max.x, max.y, max.z),
            ]
            .iter()
            .map(|vertex| model_matrix.0 * vertex.to_homogeneous())
            .map(|vertex| vertex.xyz() / vertex.w)
            .fold(
                ((f32::MAX, f32::MAX, f32::MAX), (f32::MIN, f32::MIN, f32::MIN)),
                |((min_x, min_y, min_z), (max_x, max_y, max_z)), vertex| {
                    (
                        (min_x.min(vertex.x), min_y.min(vertex.y), min_z.min(vertex.z)),
                        (max_x.max(vertex.x), max_y.max(vertex.y), max_z.max(vertex.z)),
                    )
                },
            );
            let min = na::Vector3::new(min.0, min.1, min.2);
            let max = na::Vector3::new(max.0, max.1, max.2);
            let new = ncollide3d::bounding_volume::AABB::from_half_extents(
                na::Point3::from((max + min) / 2.0),
                (max - min) / 2.0,
            );
            aabb.0 = new;
        });
    }

    pub(crate) fn launch_projectiles_test(
        mut commands: Commands,
        input_actions: Res<InputActions>,
        mesh_library: Res<MeshLibrary>,
        camera: Res<Camera>,
        renderer: Res<RenderFrame>,
        mut last_frame_launched: Local<u64>, // stores frame number so we can debounce the launches
    ) {
        scope!("ecs::LaunchProjectiles");

        if *last_frame_launched + 60 < renderer.frame_number && input_actions.get_mouse_down(MouseButton::Left) {
            let target = camera.position + camera.rotation * (100.0 * (&forward_vector().into_inner()));
            *last_frame_launched = renderer.frame_number;
            commands.spawn().insert_bundle((
                Position(camera.position),
                Rotation(camera.rotation),
                Scale(1.0),
                ModelMatrix::default(),
                GltfMeshBaseColorTexture(Arc::clone(&mesh_library.projectile_base_color)),
                GltfMeshNormalTexture(Arc::clone(&mesh_library.projectile_normal_map)),
                mesh_library.projectile.clone(),
                AABB::default(),
                CoarseCulled(false),
                DrawIndex::default(),
                ProjectileTarget(target),
                ProjectileVelocity(20.0),
            ));
        }
    }

    pub(crate) fn update_projectiles(
        mut commands: Commands,
        frame_timing: Res<FrameTiming>,
        renderer: Res<RenderFrame>,
        image_index: Res<ImageIndex>,
        mut query: Query<(Entity, &mut Position, &Rotation, &ProjectileTarget, &ProjectileVelocity), Without<Deleting>>,
    ) {
        scope!("ecs::UpdateProjectiles");

        for (entity, mut position, rotation, target, velocity) in query.iter_mut() {
            if na::distance(&position.0, &target.0) < 0.1 {
                // remove DrawIndex to prevent from participating in rendering
                // TODO: there should be a better command to remove all data components to "deactivate" an entity,
                //       it's bad to add another archetype when removing just one of many components
                commands.entity(entity).remove::<DrawIndex>().insert(Deleting {
                    frame_number: renderer.frame_number,
                    image_index: image_index.clone(),
                });
                continue;
            }
            let velocity_scaled = velocity.0 * frame_timing.time_delta;
            let increment = velocity_scaled * (rotation.0 * forward_vector().into_inner());
            position.0 += increment;
        }
    }

    /// Grab-bag for renderer and player controller variables for now
    #[derive(Clone)]
    pub(crate) struct RuntimeConfiguration {
        pub(crate) debug_aabbs: bool,
        pub(crate) fly_mode: bool,
        pub(crate) freeze_culling: bool,
        #[cfg(not(feature = "nort"))]
        pub(crate) rt: bool,
        pub(crate) compute_cull_workgroup_size: u32,
    }

    impl Default for RuntimeConfiguration {
        fn default() -> RuntimeConfiguration {
            RuntimeConfiguration {
                debug_aabbs: false,
                fly_mode: false,
                freeze_culling: false,
                #[cfg(not(feature = "nort"))]
                rt: true,
                compute_cull_workgroup_size: INITIAL_WORKGROUP_SIZE,
            }
        }
    }

    impl RuntimeConfiguration {
        pub(crate) fn rt(&self) -> bool {
            #[cfg(feature = "nort")]
            return false;
            #[cfg(not(feature = "nort"))]
            return self.rt;
        }
    }

    pub(crate) struct Gui {
        pub(crate) imgui: imgui::Context,
    }

    impl Gui {
        pub(crate) fn new() -> Gui {
            let mut imgui = imgui::Context::create();

            imgui.style_mut().frame_border_size = 1.0;
            imgui.style_mut().frame_rounding = 4.0;

            Gui { imgui }
        }

        pub(crate) fn update<'a>(
            &'a mut self,
            renderer: &RenderFrame,
            input_handler: &mut InputHandler,
            swapchain: &Swapchain,
            camera: &mut Camera,
            runtime_config: &mut RuntimeConfiguration,
            #[cfg(feature = "shader_reload")] reloaded_shaders: &ReloadedShaders,
        ) -> &'a imgui::DrawData {
            scope!("gui::update");

            let imgui = &mut self.imgui;
            imgui.io_mut().display_size = [swapchain.width as f32, swapchain.height as f32];
            input_handler
                .imgui_platform
                .prepare_frame(imgui.io_mut(), &renderer.instance.window)
                .unwrap();
            let ui = imgui.frame();

            let alloc_stats = renderer.device.allocation_stats();
            let mut position = [camera.position[0], camera.position[1], camera.position[2]];
            let (x, y, z) = camera.rotation.euler_angles();
            let mut rotation = [x * 180.0 / f32::pi(), y * 180.0 / f32::pi(), z * 180.0 / f32::pi()];
            imgui::Window::new("Debug").always_auto_resize(true).build(&ui, || {
                ui.text("Allocation stats:");
                ui.bullet_text(&format!("block count: {}", alloc_stats.total.blockCount));
                ui.bullet_text(&format!("alloc count: {}", alloc_stats.total.allocationCount));

                use humansize::{file_size_opts, FileSize};
                let used = alloc_stats.total.usedBytes.file_size(file_size_opts::BINARY).unwrap();
                ui.bullet_text(&format!("used size: {}", used));
                let unused = alloc_stats.total.unusedBytes.file_size(file_size_opts::BINARY).unwrap();
                ui.bullet_text(&format!("unused size: {}", unused));

                ui.spacing();

                if imgui::CollapsingHeader::new("Shader settings")
                    .default_open(true)
                    .build(&ui)
                {
                    ui.set_next_item_width(100.0);
                    imgui::Slider::new(
                        "Compute cull workgroup size",
                        1,
                        renderer.device.limits.max_compute_work_group_size[0],
                    )
                    .build(&ui, &mut runtime_config.compute_cull_workgroup_size);
                }

                if imgui::CollapsingHeader::new("Camera").default_open(true).build(&ui) {
                    ui.input_float3("position", &mut position).build();
                    ui.input_float3("rotation", &mut rotation).build();
                    ui.checkbox("[G] Fly mode", &mut runtime_config.fly_mode);
                }

                if imgui::CollapsingHeader::new("Debug options")
                    .default_open(true)
                    .build(&ui)
                {
                    ui.checkbox("Debug collision AABBs", &mut runtime_config.debug_aabbs);
                    ui.checkbox("Freeze culling data", &mut runtime_config.freeze_culling);
                    #[cfg(not(feature = "nort"))]
                    ui.checkbox("Use Raytracing", &mut runtime_config.rt);
                }

                #[cfg(feature = "shader_reload")]
                if imgui::CollapsingHeader::new("Shader reloader")
                    .default_open(true)
                    .build(&ui)
                {
                    use std::path::Path;

                    ui.label_text("age", "shader");
                    for (path, (instant, _)) in reloaded_shaders.0.iter() {
                        let parsed = Path::new(path);
                        let relative = parsed
                            .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("shaders"))
                            .unwrap();
                        ui.label_text(
                            &format!("{:?}s", instant.elapsed().as_secs()),
                            relative.to_str().unwrap(),
                        )
                    }
                }
            });

            camera.position = position.into();
            camera.rotation = na::UnitQuaternion::from_euler_angles(
                rotation[0] * f32::pi() / 180.0,
                rotation[1] * f32::pi() / 180.0,
                rotation[2] * f32::pi() / 180.0,
            );

            runtime_config.compute_cull_workgroup_size = na::clamp(
                runtime_config.compute_cull_workgroup_size,
                1,
                renderer.device.limits.max_compute_work_group_size[0],
            );

            input_handler
                .imgui_platform
                .prepare_render(&ui, &renderer.instance.window);

            ui.render()
        }
    }

    pub(crate) fn cleanup_deleted_entities(world: &mut World) {
        scope!("ecs::cleanup_deleted_entities");

        let frame_number = world.get_resource::<RenderFrame>().unwrap().frame_number;
        let swapchain_index = world.get_resource::<ImageIndex>().cloned().unwrap();
        let mut entities = vec![];

        world
            .query::<(Entity, &Deleting)>()
            .for_each(world, |(entity, deleting)| {
                if deleting.frame_number < frame_number && deleting.image_index == swapchain_index {
                    entities.push(entity);
                }
            });

        for entity in entities {
            world.entity_mut(entity).despawn();
        }
    }
}
