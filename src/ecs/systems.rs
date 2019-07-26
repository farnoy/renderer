use super::components::*;
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
use specs::prelude::*;
use std::{sync::Arc, time::Instant};
use winit::{
    self, dpi::LogicalSize, DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode,
    WindowEvent,
};

use crate::renderer::{forward_vector, right_vector, up_vector, GltfMesh, RenderFrame};

pub struct MVPCalculation;

impl<'a> System<'a> for MVPCalculation {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Rotation>,
        ReadStorage<'a, Scale>,
        WriteStorage<'a, Matrices>,
        Read<'a, Camera>,
    );

    fn run(
        &mut self,
        (entities, positions, rotations, scales, mut mvps, camera): Self::SystemData,
    ) {
        microprofile::scope!("ecs", "mvp calculation");
        let mut entities_to_update = vec![];
        for (entity_id, _, _, _, ()) in (&*entities, &positions, &rotations, &scales, !&mvps).join()
        {
            entities_to_update.push(entity_id);
        }
        for entity_id in entities_to_update.into_iter() {
            mvps.insert(entity_id, Matrices::one())
                .expect("failed to insert missing matrices");
        }
        (&positions, &rotations, &scales, &mut mvps)
            .par_join()
            .for_each(|(pos, rot, scale, mvp)| {
                let translation = na::Translation3::from(pos.0.coords);
                let model = na::Similarity3::from_parts(translation, rot.0, scale.0);
                mvp.model = na::Matrix4::<f32>::from(na::Similarity3::from_parts(
                    translation,
                    rot.0,
                    scale.0,
                ));

                mvp.mvp = camera.projection * na::Matrix4::<f32>::from(camera.view * model);
            });
    }
}

pub struct Camera {
    pub position: na::Point3<f32>,
    pub rotation: na::UnitQuaternion<f32>,
    pub projection: na::Matrix4<f32>,
    pub view: na::Isometry3<f32>,
    // left -> right -> bottom -> top -> near -> far
    pub frustum_planes: [na::Vector4<f32>; 6],
}

impl Default for Camera {
    fn default() -> Camera {
        let position = na::Point3::new(0.0, 1.0, 2.0);
        let projection = na::Matrix4::identity();
        let view = na::Isometry3::identity();
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
    pub events_loop: winit::EventsLoop,
    pub quit_handle: Arc<Mutex<bool>>,
    pub move_mouse: bool,
}

impl<'a> System<'a> for InputHandler {
    type SystemData = (Write<'a, InputState>, Write<'a, Camera>);

    fn run(&mut self, (mut input_state, mut camera): Self::SystemData) {
        microprofile::scope!("ecs", "input handler");
        let quit_handle = Arc::clone(&self.quit_handle);
        input_state.clear();
        let move_mouse = self.move_mouse;
        let mut toggle_move_mouse = false;
        self.events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(LogicalSize { width, height }),
                ..
            } => {
                println!("The window was resized to {}x{}", width, height);
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
                    ElementState::Released => input_state.key_releases.push(virtual_keycode),
                }
                match virtual_keycode {
                    Some(VirtualKeyCode::G) if state == ElementState::Pressed => {
                        toggle_move_mouse = true;
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
            } if move_mouse => {
                let y_angle = f32::pi() / 180.0 * y as f32;
                let x_angle = f32::pi() / 180.0 * x as f32;
                camera.rotation =
                    camera.rotation * na::Rotation3::from_axis_angle(&right_vector(), y_angle);
                camera.rotation =
                    na::Rotation3::from_axis_angle(&up_vector(), x_angle) * camera.rotation;
            }
            _ => (),
        });
        self.move_mouse = if toggle_move_mouse {
            !self.move_mouse
        } else {
            self.move_mouse
        };
    }
}

pub struct ProjectCamera;

impl<'a> System<'a> for ProjectCamera {
    type SystemData = (ReadExpect<'a, RenderFrame>, Write<'a, Camera>);

    fn run(&mut self, (renderer, mut camera): Self::SystemData) {
        microprofile::scope!("ecs", "project camera");
        let near = 0.1;
        let far = 100.0;
        let aspect = renderer.instance.window_width as f32 / renderer.instance.window_height as f32;
        let fovy = f32::pi() * 70.0 / 180.0;

        camera.projection = glm::perspective_lh_zo(aspect, fovy, near, far);

        let dir = camera.rotation.transform_vector(&forward_vector());
        let extended_forward = camera.position + dir;
        let up = camera.rotation.transform_vector(&up_vector());

        camera.view = na::Isometry3::look_at_lh(&camera.position, &extended_forward, &up);

        let m = (camera.projection * camera.view.to_homogeneous()).transpose();
        camera.frustum_planes = [
            -(m.column(3) + m.column(0)),
            -(m.column(3) - m.column(0)),
            -(m.column(3) + m.column(1)),
            -(m.column(3) - m.column(1)),
            -(m.column(3) + m.column(2)),
            -(m.column(3) - m.column(2)),
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

impl<'a> System<'a> for CalculateFrameTiming {
    type SystemData = Write<'a, FrameTiming>;

    fn run(&mut self, mut frame_timing: Self::SystemData) {
        let now = Instant::now();
        let duration = now - frame_timing.previous_frame;
        frame_timing.time_delta =
            duration.as_secs() as f32 + (duration.subsec_micros() as f32 / 1e6);
        frame_timing.previous_frame = now;
    }
}

pub struct AABBCalculation;

impl<'a> System<'a> for AABBCalculation {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Matrices>,
        ReadStorage<'a, GltfMesh>,
        WriteStorage<'a, AABB>,
    );

    fn run(&mut self, (entities, matrices, mesh, mut aabb): Self::SystemData) {
        microprofile::scope!("ecs", "aabb calculation");
        use std::f32::{MAX, MIN};
        for (entity_id, matrices, mesh) in (&*entities, &matrices, &mesh).join() {
            let min = mesh.aabb_c - mesh.aabb_h;
            let max = mesh.aabb_c + mesh.aabb_h;
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
            .map(|vertex| matrices.model * vertex.to_homogeneous())
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
            aabb.insert(
                entity_id,
                AABB {
                    c: (max + min) / 2.0,
                    h: (max - min) / 2.0,
                },
            )
            .expect("Failed to insert AABB");
        }
    }
}

pub struct InputState {
    key_presses: Vec<Option<winit::VirtualKeyCode>>,
    key_releases: Vec<Option<winit::VirtualKeyCode>>,
}

impl InputState {
    fn clear(&mut self) {
        self.key_presses.clear();
        self.key_releases.clear();
    }
}

impl Default for InputState {
    fn default() -> InputState {
        InputState {
            key_presses: Vec::new(),
            key_releases: Vec::new(),
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

impl<'a> System<'a> for FlyCamera {
    type SystemData = (
        Read<'a, InputState>,
        Read<'a, FrameTiming>,
        Write<'a, Camera>,
    );

    fn run(&mut self, (input, frame_timing, mut camera): Self::SystemData) {
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
