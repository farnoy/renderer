mod camera_controller;
mod input;

pub mod components;
pub mod resources {
    pub use super::{
        camera_controller::Camera,
        input::{InputActions, InputHandler, InputState},
    };

    use super::super::renderer::{GltfMesh, Image};
    use std::sync::Arc;

    pub struct MeshLibrary {
        pub projectile: GltfMesh,
        pub projectile_texture: Arc<Image>,
    }
}
pub mod systems {
    pub use super::{camera_controller::CameraController, input::InputHandler};

    use super::super::renderer::*;
    use crate::ecs::{
        components::{ModelMatrix, ProjectileTarget, ProjectileVelocity, AABB},
        resources::{Camera, InputActions, MeshLibrary},
    };
    use imgui::im_str;
    #[cfg(feature = "microprofile")]
    use microprofile::scope;
    use na::RealField;
    use std::{sync::Arc, time::Instant};
    use winit::{self, event::MouseButton};

    use crate::renderer::{forward_vector, up_vector, GltfMesh, Swapchain};

    pub struct ModelMatrixCalculation;

    impl ModelMatrixCalculation {
        pub fn exec_system() -> Box<(dyn legion::systems::schedule::Schedulable + 'static)> {
            use legion::prelude::*;
            SystemBuilder::<()>::new("ModelMatrixCalculation - exec")
                .with_query(<(
                    Read<Position>,
                    Read<Rotation>,
                    Read<Scale>,
                    Write<ModelMatrix>,
                )>::query())
                .build(move |_commands, mut world, _resources, ref mut query| {
                    #[cfg(feature = "profiling")]
                    microprofile::scope!("ecs", "ModelMatrixCalculation");
                    for (pos, rot, scale, mut model_matrix) in query.iter_mut(&mut world) {
                        model_matrix.0 = glm::translation(&pos.0.coords)
                            * rot.0.to_homogeneous()
                            * glm::scaling(&glm::Vec3::repeat(scale.0));
                    }
                })
        }
    }

    pub struct ProjectCamera;

    impl ProjectCamera {
        pub fn exec(swapchain: &Swapchain, camera: &mut Camera) {
            #[cfg(feature = "profiling")]
            microprofile::scope!("ecs", "project camera");
            let near = 0.1;
            let far = 100.0;
            let aspect = swapchain.width as f32 / swapchain.height as f32;
            let fovy = glm::radians(&glm::vec1(70.0));

            camera.projection = glm::perspective_lh_zo(aspect, fovy.x, near, far);

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
    }

    pub struct FrameTiming {
        pub previous_frame: Instant,
        pub time_delta: f32,
    }

    impl Default for FrameTiming {
        fn default() -> FrameTiming {
            FrameTiming {
                previous_frame: Instant::now(),
                time_delta: 0.0,
            }
        }
    }

    pub struct CalculateFrameTiming;

    impl CalculateFrameTiming {
        pub fn exec(frame_timing: &mut FrameTiming) {
            let now = Instant::now();
            let duration = now - frame_timing.previous_frame;
            frame_timing.time_delta =
                duration.as_secs() as f32 + (duration.subsec_micros() as f32 / 1e6);
            frame_timing.previous_frame = now;
        }
    }

    pub struct AABBCalculation;

    impl AABBCalculation {
        pub fn exec_system() -> Box<(dyn legion::systems::schedule::Schedulable + 'static)> {
            use legion::prelude::*;
            SystemBuilder::<()>::new("AABBCalculation - exec")
                .with_query(<(Read<ModelMatrix>, Read<GltfMesh>, Write<AABB>)>::query())
                .build(move |_commands, mut world, _resources, ref mut query| {
                    #[cfg(feature = "profiling")]
                    microprofile::scope!("ecs", "AABBCalculation");
                    use std::f32::{MAX, MIN};
                    for (model_matrix, mesh, mut aabb) in query.iter_mut(&mut world) {
                        let min = mesh.aabb.mins();
                        let max = mesh.aabb.maxs();
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
                            ((MAX, MAX, MAX), (MIN, MIN, MIN)),
                            |((minx, miny, minz), (maxx, maxy, maxz)), vertex| {
                                (
                                    (minx.min(vertex.x), miny.min(vertex.y), minz.min(vertex.z)),
                                    (maxx.max(vertex.x), maxy.max(vertex.y), maxz.max(vertex.z)),
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
                    }
                })
        }
    }

    pub struct LaunchProjectileTest;

    impl LaunchProjectileTest {
        pub fn exec_system() -> Box<(dyn legion::systems::schedule::Schedulable + 'static)> {
            use legion::prelude::*;
            SystemBuilder::<()>::new("LaunchProjectiles")
                .read_resource::<InputActions>()
                .read_resource::<MeshLibrary>()
                .read_resource::<Camera>()
                .build(move |commands, _world, resources, _query| {
                    #[cfg(feature = "profiling")]
                    microprofile::scope!("ecs", "LaunchProjectiles");
                    let (ref input_actions, ref mesh_library, ref camera) = resources;
                    if input_actions.get_mouse_down(MouseButton::Left) {
                        let target = camera.position
                            + camera.rotation * (100.0 * (&forward_vector().into_inner()));
                        commands.insert(
                            (),
                            Some((
                                Position(camera.position),
                                Rotation(camera.rotation),
                                Scale(1.0),
                                ModelMatrix::default(),
                                GltfMeshBaseColorTexture(Arc::clone(
                                    &mesh_library.projectile_texture,
                                )),
                                mesh_library.projectile.clone(),
                                AABB::default(),
                                CoarseCulled(false),
                                DrawIndex::default(),
                                ProjectileTarget(target),
                                ProjectileVelocity(20.0),
                            )),
                        );
                    }
                })
        }
    }
    pub struct UpdateProjectiles;

    impl UpdateProjectiles {
        pub fn exec_system() -> Box<(dyn legion::systems::schedule::Schedulable + 'static)> {
            use legion::prelude::*;
            SystemBuilder::<()>::new("UpdateProjectiles")
                .read_resource::<FrameTiming>()
                .with_query(<(
                    Write<Position>,
                    Read<Rotation>,
                    Read<ProjectileTarget>,
                    Read<ProjectileVelocity>,
                )>::query())
                .build(move |commands, mut world, ref frame_timing, query| {
                    #[cfg(feature = "profiling")]
                    microprofile::scope!("ecs", "UpdateProjectiles");
                    for (entity, (mut position, rotation, target, velocity)) in
                        query.iter_entities_mut(&mut world)
                    {
                        if na::distance(&position.0, &target.0) < 0.1 {
                            commands.delete(entity);
                            continue;
                        }
                        let velocity_scaled = velocity.0 * frame_timing.time_delta;
                        let increment =
                            velocity_scaled * (rotation.0 * forward_vector().into_inner());
                        position.0 += increment;
                    }
                })
        }
    }

    /// Grab-bag for renderer and player controller variables for now
    pub struct RuntimeConfiguration {
        pub debug_aabbs: bool,
        pub fly_mode: bool,
    }

    impl RuntimeConfiguration {
        pub fn new() -> RuntimeConfiguration {
            RuntimeConfiguration {
                debug_aabbs: false,
                fly_mode: false,
            }
        }
    }

    pub struct Gui {
        pub imgui: imgui::Context,
    }

    impl Gui {
        pub fn new() -> Gui {
            let mut imgui = imgui::Context::create();

            imgui.style_mut().frame_border_size = 1.0;
            imgui.style_mut().frame_rounding = 4.0;

            Gui { imgui }
        }

        pub fn update<'a>(
            &'a mut self,
            renderer: &RenderFrame,
            input_handler: &InputHandler,
            swapchain: &Swapchain,
            camera: &Camera,
            runtime_config: &mut RuntimeConfiguration,
        ) -> &'a imgui::DrawData {
            let imgui = &mut self.imgui;
            imgui.io_mut().display_size = [swapchain.width as f32, swapchain.height as f32];
            input_handler
                .imgui_platform
                .prepare_frame(imgui.io_mut(), &renderer.instance.window)
                .unwrap();
            let ui = imgui.frame();

            let alloc_stats = renderer.device.allocation_stats();
            imgui::Window::new(im_str!("Debug"))
                .always_auto_resize(true)
                .build(&ui, || {
                    ui.text(&im_str!("Allocation stats:"));
                    ui.bullet_text(&im_str!("block count: {}", alloc_stats.total.blockCount));
                    ui.bullet_text(&im_str!(
                        "alloc count: {}",
                        alloc_stats.total.allocationCount
                    ));
                    let used = unbytify::bytify(alloc_stats.total.usedBytes);
                    ui.bullet_text(&im_str!("used bytes: {} {}", used.0, used.1,));
                    let unused = unbytify::bytify(alloc_stats.total.unusedBytes);
                    ui.bullet_text(&im_str!("unused bytes: {} {}", unused.0, unused.1,));

                    ui.spacing();

                    if ui
                        .collapsing_header(&im_str!("Camera"))
                        .default_open(true)
                        .build()
                    {
                        ui.text(&im_str!("Camera:"));
                        let x = camera.position.x;
                        let y = camera.position.y;
                        let z = camera.position.z;
                        let s = format!("position: x={:.2} y={:.2} z={:.2}", x, y, z);
                        ui.bullet_text(&im_str!("{}", s));
                        let (x, y, z) = camera.rotation.euler_angles();
                        let (x, y, z) = (
                            x * 180.0 / f32::pi(),
                            y * 180.0 / f32::pi(),
                            z * 180.0 / f32::pi(),
                        );
                        let s = format!("rotation: x={:5.2} y={:5.2} z={:5.2}", x, y, z);
                        ui.bullet_text(&im_str!("{}", s));
                        ui.checkbox(
                            &im_str!("[G] Camera fly mode"),
                            &mut runtime_config.fly_mode,
                        );
                        ui.spacing();
                    }
                    ui.checkbox(
                        &im_str!("Debug collision AABBs"),
                        &mut runtime_config.debug_aabbs,
                    );
                });

            input_handler
                .imgui_platform
                .prepare_render(&ui, &renderer.instance.window);

            ui.render()
        }
    }
}
