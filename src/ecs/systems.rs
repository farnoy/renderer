use super::{super::renderer::*, custom::*, resources::*};
use imgui::im_str;
use imgui_winit_support::WinitPlatform;
#[cfg(feature = "microprofile")]
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
use std::{sync::Arc, time::Instant};
use winit::{
    self,
    dpi::PhysicalSize,
    event::{
        ButtonId, DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent,
    },
    platform::desktop::EventLoopExtDesktop,
};

use crate::renderer::{
    forward_vector, right_vector, up_vector, GltfMesh, GltfMeshBaseColorTexture, Swapchain,
};

pub struct ModelMatrixCalculation;

impl ModelMatrixCalculation {
    pub fn exec(
        entities: &EntitiesStorage,
        positions: &ComponentStorage<na::Point3<f32>>,
        rotations: &ComponentStorage<na::UnitQuaternion<f32>>,
        scales: &ComponentStorage<f32>,
        model_matrices: &mut ComponentStorage<glm::Mat4>,
    ) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "model matrix calculation");
        model_matrices
            .replace_mask(&(entities.mask() & positions.mask() & rotations.mask() & scales.mask()));

        for entity_id in model_matrices.mask().clone().iter() {
            let pos = positions.get(entity_id).unwrap();
            let rot = rotations.get(entity_id).unwrap();
            let scale = scales.get(entity_id).unwrap();
            *model_matrices
                .entry(entity_id)
                .or_insert(glm::Mat4::identity()) = glm::translation(&pos.coords)
                * rot.to_homogeneous()
                * glm::scaling(&glm::Vec3::repeat(*scale));
        }
    }
}

pub struct Camera {
    pub position: na::Point3<f32>,
    pub rotation: na::UnitQuaternion<f32>,
    pub projection: na::Matrix4<f32>,
    pub view: na::Matrix4<f32>,
    // left -> right -> bottom -> top -> near -> far
    pub frustum_planes: [na::Vector4<f32>; 6],
}

impl Default for Camera {
    fn default() -> Camera {
        let position = na::Point3::new(0.0, 1.0, 2.0);
        let projection = na::Matrix4::identity();
        let view = na::Matrix4::identity();
        let rotation = na::UnitQuaternion::identity();
        let zero = na::Vector4::new(0.0, 0.0, 0.0, 0.0);

        Camera {
            position,
            rotation,
            projection,
            view,
            frustum_planes: [zero; 6],
        }
    }
}

pub struct InputHandler {
    pub events_loop: winit::event_loop::EventLoop<()>,
    pub quit_handle: Arc<Mutex<bool>>,
    pub imgui_platform: WinitPlatform,
}

impl InputHandler {
    /// Returns true if resized
    pub fn exec(
        &mut self,
        window: &winit::window::Window,
        gui: &mut imgui::Context,
        input_state: &mut InputState,
        camera: &mut Camera,
        runtime_config: &mut RuntimeConfiguration,
    ) -> bool {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "input handler");
        let quit_handle = Arc::clone(&self.quit_handle);
        input_state.clear();
        let fly_mode = runtime_config.fly_mode;
        let platform = &mut self.imgui_platform;
        let mut toggle_fly_mode = false;
        let mut resized = false;
        // println!("event loop run_return");
        self.events_loop
            .run_return(|event, window_target, control_flow| {
                // dbg!(&event);
                // platform.handle_event(gui.io_mut(), &window, &event);
                // println!("imgui handled event");
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::Resized(PhysicalSize { width, height }),
                        ..
                    } => {
                        println!("The window was resized to {}x{}", width, height);
                        // hangs for now
                        // let logical_size = PhysicalSize { width, height }.to_logical::<f32>(window.scale_factor());
                        // println!("logical {:?}", logical_size);
                        resized = true;
                    }
                    Event::WindowEvent {
                        event: WindowEvent::ScaleFactorChanged { scale_factor, .. },
                        ..
                    } => {
                        println!("Scale factor changed {}", scale_factor);
                    }
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => {
                        *quit_handle.lock() = true;
                    }
                    Event::WindowEvent {
                        event:
                            WindowEvent::KeyboardInput {
                                input:
                                    KeyboardInput {
                                        state,
                                        virtual_keycode,
                                        ..
                                    },
                                ..
                            },
                        ..
                    } => {
                        match state {
                            ElementState::Pressed => input_state.key_presses.push(virtual_keycode),
                            ElementState::Released => {
                                input_state.key_releases.push(virtual_keycode)
                            }
                        }
                        match virtual_keycode {
                            Some(VirtualKeyCode::G) if state == ElementState::Pressed => {
                                toggle_fly_mode = true;
                            }
                            Some(VirtualKeyCode::Escape) => {
                                *quit_handle.lock() = true;
                            }
                            _ => (),
                        }
                    }
                    Event::DeviceEvent {
                        event: DeviceEvent::MouseMotion { delta: (x, y), .. },
                        ..
                    } if fly_mode => {
                        let y_angle = f32::pi() / 180.0 * y as f32;
                        let x_angle = f32::pi() / 180.0 * x as f32;
                        camera.rotation *= na::Rotation3::from_axis_angle(&right_vector(), y_angle);
                        camera.rotation =
                            na::Rotation3::from_axis_angle(&up_vector(), x_angle) * camera.rotation;
                    }
                    Event::DeviceEvent {
                        event:
                            DeviceEvent::Button {
                                button,
                                state: ElementState::Pressed,
                            },
                        ..
                    } => {
                        input_state.button_presses.push(button);
                    }
                    _ => (),
                };
                *control_flow = winit::event_loop::ControlFlow::Exit;
            });
        runtime_config.fly_mode = if toggle_fly_mode { !fly_mode } else { fly_mode };
        platform
            .prepare_frame(gui.io_mut(), &window)
            .expect("Failed to prepare frame");

        resized
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
    previous_frame: Instant,
    time_delta: f32,
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
    pub fn exec(
        entities: &EntitiesStorage,
        model_matrices: &ComponentStorage<glm::Mat4>,
        meshes: &ComponentStorage<GltfMesh>,
        aabb: &mut ComponentStorage<ncollide3d::bounding_volume::AABB<f32>>,
    ) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "aabb calculation");
        use std::f32::{MAX, MIN};
        let desired = entities.mask() & model_matrices.mask() & meshes.mask();
        aabb.replace_mask(&desired);
        for entity_id in desired.iter() {
            let model_matrix = model_matrices.get(entity_id).unwrap();
            let mesh = meshes.get(entity_id).unwrap();
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
            .map(|vertex| model_matrix * vertex.to_homogeneous())
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
            *aabb.entry(entity_id).or_insert(
                ncollide3d::bounding_volume::AABB::from_half_extents(
                    na::Point3::origin(),
                    na::zero(),
                ),
            ) = new;
        }
    }
}

pub struct InputState {
    key_presses: Vec<Option<VirtualKeyCode>>,
    key_releases: Vec<Option<VirtualKeyCode>>,
    button_presses: Vec<ButtonId>,
}

impl InputState {
    fn clear(&mut self) {
        self.key_presses.clear();
        self.key_releases.clear();
        self.button_presses.clear();
    }
}

impl Default for InputState {
    fn default() -> InputState {
        InputState {
            key_presses: vec![],
            key_releases: vec![],
            button_presses: vec![],
        }
    }
}

pub struct FlyCamera {
    forward: bool,
    backward: bool,
    up: bool,
    down: bool,
    right: bool,
    left: bool,
    fast: bool,
}

impl Default for FlyCamera {
    fn default() -> FlyCamera {
        FlyCamera {
            forward: false,
            backward: false,
            up: false,
            down: false,
            right: false,
            left: false,
            fast: false,
        }
    }
}

impl FlyCamera {
    pub fn exec(
        &mut self,
        input: &InputState,
        frame_timing: &FrameTiming,
        runtime_config: &RuntimeConfiguration,
        camera: &mut Camera,
    ) {
        if !runtime_config.fly_mode {
            return;
        }

        for key in &input.key_presses {
            match key {
                Some(VirtualKeyCode::W) => {
                    self.forward = true;
                }
                Some(VirtualKeyCode::S) => {
                    self.backward = true;
                }
                Some(VirtualKeyCode::D) => {
                    self.right = true;
                }
                Some(VirtualKeyCode::A) => {
                    self.left = true;
                }
                Some(VirtualKeyCode::Space) => {
                    self.up = true;
                }
                Some(VirtualKeyCode::LControl) => {
                    self.down = true;
                }
                Some(VirtualKeyCode::LShift) => {
                    self.fast = true;
                }
                _ => (),
            }
        }
        for key in &input.key_releases {
            match key {
                Some(VirtualKeyCode::W) => {
                    self.forward = false;
                }
                Some(VirtualKeyCode::S) => {
                    self.backward = false;
                }
                Some(VirtualKeyCode::D) => {
                    self.right = false;
                }
                Some(VirtualKeyCode::A) => {
                    self.left = false;
                }
                Some(VirtualKeyCode::Space) => {
                    self.up = false;
                }
                Some(VirtualKeyCode::LControl) => {
                    self.down = false;
                }
                Some(VirtualKeyCode::LShift) => {
                    self.fast = false;
                }
                _ => (),
            }
        }
        let mut speed = if self.fast { 10.0 } else { 1.0 };
        speed *= frame_timing.time_delta;
        let mut increment: na::Vector3<f32> = na::zero();
        if self.forward {
            increment += speed * camera.rotation.transform_vector(&forward_vector())
        }
        if self.backward {
            increment -= speed * camera.rotation.transform_vector(&forward_vector());
        }
        if self.up {
            increment += speed * camera.rotation.transform_vector(&up_vector());
        }
        if self.down {
            increment -= speed * camera.rotation.transform_vector(&up_vector());
        }
        if self.right {
            increment += speed * camera.rotation.transform_vector(&right_vector());
        }
        if self.left {
            increment -= speed * camera.rotation.transform_vector(&right_vector());
        }

        camera.position += increment;
    }
}

pub struct LaunchProjectileTest;

impl LaunchProjectileTest {
    #[allow(clippy::too_many_arguments)]
    pub fn exec(
        entities: &mut EntitiesStorage,
        position_storage: &mut ComponentStorage<na::Point3<f32>>,
        rotation_storage: &mut ComponentStorage<na::UnitQuaternion<f32>>,
        scale_storage: &mut ComponentStorage<f32>,
        meshes_storage: &mut ComponentStorage<GltfMesh>,
        textures_storage: &mut ComponentStorage<GltfMeshBaseColorTexture>,
        projectile_target_storage: &mut ComponentStorage<na::Point3<f32>>,
        projectile_velocities_storage: &mut ComponentStorage<f32>,
        camera: &mut Camera,
        mesh_library: &MeshLibrary,
        input_state: &InputState,
    ) {
        if input_state.button_presses.iter().any(|p| *p == 1) {
            let projectile = entities.allocate();
            position_storage.insert(projectile, camera.position);
            rotation_storage.insert(projectile, camera.rotation);
            scale_storage.insert(projectile, 1.0);
            let target =
                camera.position + camera.rotation * (100.0 * (&forward_vector().into_inner()));
            projectile_target_storage.insert(projectile, target);
            projectile_velocities_storage.insert(projectile, 20.0);
            meshes_storage.insert(projectile, mesh_library.projectile.clone());
            textures_storage.insert(
                projectile,
                GltfMeshBaseColorTexture(Arc::clone(&mesh_library.projectile_texture)),
            );
        }
    }
}

pub struct UpdateProjectiles;

impl UpdateProjectiles {
    pub fn exec(
        entities: &mut EntitiesStorage,
        position_storage: &mut ComponentStorage<na::Point3<f32>>,
        rotation_storage: &ComponentStorage<na::UnitQuaternion<f32>>,
        projectile_target_storage: &ComponentStorage<na::Point3<f32>>,
        projectile_velocities_storage: &mut ComponentStorage<f32>,
        frame_timing: &FrameTiming,
    ) {
        let projectiles = entities.mask()
            & rotation_storage.mask()
            & projectile_target_storage.mask()
            & projectile_velocities_storage.mask();
        for projectile in projectiles.iter() {
            let position = position_storage.entry(projectile).assume();
            let target = projectile_target_storage.get(projectile).unwrap();
            if na::distance(position, target) < 0.1 {
                entities.remove(projectile);
                continue;
            }
            let rotation = rotation_storage.get(projectile).unwrap();
            let velocity = projectile_velocities_storage.get(projectile).unwrap();
            let velocity_scaled = velocity * frame_timing.time_delta;
            let increment = velocity_scaled * (rotation * forward_vector().into_inner());
            *position += increment;
        }
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
