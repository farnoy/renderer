use super::{components::*, custom::*};
#[cfg(feature = "microprofile")]
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
use std::{sync::Arc, time::Instant};
use winit::{
    self, dpi::LogicalSize, DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode,
    WindowEvent,
};

use crate::renderer::{forward_vector, right_vector, up_vector, GltfMesh, RenderFrame};

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
    pub events_loop: winit::EventsLoop,
    pub quit_handle: Arc<Mutex<bool>>,
    pub move_mouse: bool,
}

impl InputHandler {
    pub fn exec(&mut self, input_state: &mut InputState, camera: &mut Camera) {
        #[cfg(feature = "profiling")]
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
                camera.rotation *= na::Rotation3::from_axis_angle(&right_vector(), y_angle);
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

impl ProjectCamera {
    pub fn exec(renderer: &RenderFrame, camera: &mut Camera) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "project camera");
        let near = 0.1;
        let far = 100.0;
        let aspect = renderer.instance.window_width as f32 / renderer.instance.window_height as f32;
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
        aabb: &mut ComponentStorage<AABB>,
    ) {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "aabb calculation");
        use std::f32::{MAX, MIN};
        let desired = entities.mask() & model_matrices.mask() & meshes.mask();
        aabb.replace_mask(&desired);
        for entity_id in desired.iter() {
            let model_matrix = model_matrices.get(entity_id).unwrap();
            let mesh = meshes.get(entity_id).unwrap();
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
            let new = AABB {
                c: (max + min) / 2.0,
                h: (max - min) / 2.0,
            };
            *aabb.entry(entity_id).or_insert(AABB {
                c: na::zero(),
                h: na::zero(),
            }) = new;
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

impl FlyCamera {
    pub fn exec(&mut self, input: &InputState, frame_timing: &FrameTiming, camera: &mut Camera) {
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
