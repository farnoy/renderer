use super::components::*;
use cgmath::{self, One, Zero};
use parking_lot::Mutex;
use specs::prelude::*;
use std::{slice::from_raw_parts_mut, sync::Arc, time::Instant};
use winit::{
    self, dpi::LogicalSize, DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode,
    WindowEvent,
};

use super::super::renderer::{Buffer, RenderFrame};

pub struct SteadyRotation;

impl<'a> System<'a> for SteadyRotation {
    type SystemData = (WriteStorage<'a, Rotation>);

    fn run(&mut self, mut rotations: Self::SystemData) {
        use cgmath::Rotation3;
        let incremental = cgmath::Quaternion::from_angle_y(cgmath::Deg(1.0));
        for rot in (&mut rotations).join() {
            *rot = Rotation(incremental * rot.0);
        }
    }
}

pub struct MVPCalculation;

impl<'a> System<'a> for MVPCalculation {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        ReadStorage<'a, Position>,
        ReadStorage<'a, Rotation>,
        ReadStorage<'a, Scale>,
        WriteStorage<'a, Matrices>,
        Read<'a, Camera>,
    );

    fn run(&mut self, (positions, rotations, scales, mut mvps, camera): Self::SystemData) {
        for (pos, rot, scale, mvp) in (&positions, &rotations, &scales, &mut mvps).join() {
            mvp.model = cgmath::Matrix4::from_translation(pos.0)
                * cgmath::Matrix4::from(rot.0)
                * cgmath::Matrix4::from_scale(scale.0);

            mvp.mvp = camera.projection * camera.view * mvp.model;
            mvp.mv = camera.view * mvp.model;
        }
    }
}

pub struct MVPUpload {
    pub dst_mvp: Arc<Buffer>,
    pub dst_model: Arc<Buffer>,
}

unsafe impl Send for MVPUpload {}

impl<'a> System<'a> for MVPUpload {
    type SystemData = (Entities<'a>, ReadStorage<'a, Matrices>);

    fn run(&mut self, (entities, matrices): Self::SystemData) {
        (&*entities, &matrices)
            .par_join()
            .for_each(|(entity, matrices)| {
                // println!("Writing at {:?} contents {:?}", entity.id(), matrices.mvp);
                let out_mvp = unsafe {
                    from_raw_parts_mut(
                        self.dst_mvp.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>,
                        4096,
                    )
                };
                let out_model = unsafe {
                    from_raw_parts_mut(
                        self.dst_model.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>,
                        4096,
                    )
                };
                out_mvp[entity.id() as usize] = matrices.mvp;
                out_model[entity.id() as usize] = matrices.model;
            });
    }
}

pub struct AssignBufferIndex;

impl<'a> System<'a> for AssignBufferIndex {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, GltfMesh>,
        ReadStorage<'a, CoarseCulled>,
        WriteStorage<'a, GltfMeshBufferIndex>,
    );

    fn run(&mut self, (entities, meshes, coarse_culled, mut indices): Self::SystemData) {
        for (ix, (entity, _mesh, coarse_culled)) in
            (&*entities, &meshes, &coarse_culled).join().enumerate()
        {
            if coarse_culled.0 {
                indices.remove(entity);
            } else {
                drop(indices.insert(entity, GltfMeshBufferIndex(ix as u32)));
            }
        }
    }
}

pub struct Camera {
    pub position: cgmath::Point3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
    pub projection: cgmath::Matrix4<f32>,
    pub view: cgmath::Matrix4<f32>,
    // left -> right -> bottom -> top -> near -> far
    pub frustum_planes: [cgmath::Vector4<f32>; 6],
}

static UP_VECTOR: cgmath::Vector3<f32> = cgmath::Vector3 {
    x: 0.0,
    y: 1.0,
    z: 0.0,
};
static FORWARD_VECTOR: cgmath::Vector3<f32> = cgmath::Vector3 {
    x: 0.0,
    y: 0.0,
    z: 1.0,
};
static RIGHT_VECTOR: cgmath::Vector3<f32> = cgmath::Vector3 {
    x: 1.0,
    y: 0.0,
    z: 0.0,
};
static CENTER_VECTOR: cgmath::Vector3<f32> = cgmath::Vector3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

impl Default for Camera {
    fn default() -> Camera {
        use cgmath::Zero;
        let position = cgmath::Point3::new(0.0, 1.0, 2.0);
        let projection = cgmath::Matrix4::one();
        let view = cgmath::Matrix4::one();
        let rotation = cgmath::Quaternion::one();
        let zero = cgmath::Vector4::zero();

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
        use cgmath::Rotation3;
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
                camera.rotation =
                    camera.rotation * cgmath::Quaternion::from_angle_x(cgmath::Deg(y as f32));
                camera.rotation =
                    cgmath::Quaternion::from_angle_y(cgmath::Deg(x as f32)) * camera.rotation;
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
        use cgmath::{Angle, Matrix, Rotation};
        // Left handed perspective projection
        let near = 0.1;
        let far = 100.0;
        let aspect = renderer.instance.window_width as f32 / renderer.instance.window_height as f32;
        let f = cgmath::Rad::cot(cgmath::Rad::from(cgmath::Deg(60.0)) / 2.0);

        let c0r0 = f / aspect;
        let c0r1 = 0.0;
        let c0r2 = 0.0;
        let c0r3 = 0.0;

        let c1r0 = 0.0;
        let c1r1 = f;
        let c1r2 = 0.0;
        let c1r3 = 0.0;

        let c2r0 = 0.0;
        let c2r1 = 0.0;
        let c2r2 = (far + near) / (far - near);
        let c2r3 = 1.0;

        let c3r0 = 0.0;
        let c3r1 = 0.0;
        let c3r2 = (2.0 * far * near) / (near - far);
        let c3r3 = 0.0;

        // #[cfg_attr(rustfmt, rustfmt_skip)]
        camera.projection = cgmath::Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1,
            c3r2, c3r3,
        );
        let dir = CENTER_VECTOR - camera.rotation.rotate_vector(FORWARD_VECTOR);
        let up = camera.rotation.rotate_vector(UP_VECTOR);
        camera.view = cgmath::Matrix4::look_at_dir(camera.position, dir, up);
        let m = (camera.projection * camera.view).transpose();
        camera.frustum_planes = [
            -(m.w + m.x),
            -(m.w - m.x),
            -(m.w + m.y),
            -(m.w - m.y),
            -(m.w + m.z),
            -(m.w - m.z),
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
        ReadStorage<'a, Matrices>,
        ReadStorage<'a, GltfMesh>,
        WriteStorage<'a, AABB>,
    );

    fn run(&mut self, (matrices, mesh, mut aabb): Self::SystemData) {
        use std::f32::{MAX, MIN};
        for (matrices, mesh, mut aabb) in (&matrices, &mesh, &mut aabb).join() {
            let min = mesh.aabb_c - mesh.aabb_h;
            let max = mesh.aabb_c + mesh.aabb_h;
            let (min, max) = [
                // bottom half (min y)
                cgmath::vec3(min.x, min.y, min.z),
                cgmath::vec3(max.x, min.y, min.z),
                cgmath::vec3(min.x, min.y, max.z),
                cgmath::vec3(max.x, min.y, max.z),
                // top half (max y)
                cgmath::vec3(min.x, max.y, min.z),
                cgmath::vec3(max.x, max.y, min.z),
                cgmath::vec3(min.x, max.y, max.z),
                cgmath::vec3(max.x, max.y, max.z),
            ]
            .iter()
            .map(|vertex| matrices.model * vertex.extend(1.0))
            .map(|vertex| vertex / vertex.w)
            .fold(
                ((MAX, MAX, MAX), (MIN, MIN, MIN)),
                |((minx, miny, minz), (maxx, maxy, maxz)), vertex| {
                    (
                        (minx.min(vertex.x), miny.min(vertex.y), minz.min(vertex.z)),
                        (maxx.max(vertex.x), maxy.max(vertex.y), maxz.max(vertex.z)),
                    )
                },
            );
            let min = cgmath::Vector3::from(min);
            let max = cgmath::Vector3::from(max);
            aabb.c = (max + min) / 2.0;
            aabb.h = (max - min) / 2.0;
        }
    }
}

pub struct CoarseCulling;

impl<'a> System<'a> for CoarseCulling {
    type SystemData = (
        ReadStorage<'a, AABB>,
        Read<'a, Camera>,
        WriteStorage<'a, CoarseCulled>,
    );

    fn run(&mut self, (aabb, camera, mut culled): Self::SystemData) {
        let mut count = 0;
        for (aabb, culled) in (&aabb, &mut culled).join() {
            let mut outside = false;
            'per_plane: for plane in camera.frustum_planes.iter() {
                let e =
                    aabb.h.x * plane.x.abs() + aabb.h.y * plane.y.abs() + aabb.h.z * plane.z.abs();

                let s = cgmath::dot(aabb.c.extend(1.0), *plane);
                if s - e > 0.0 {
                    outside = true;
                    break 'per_plane;
                }
            }
            culled.0 = outside;
            if cfg!(debug_assertions) && culled.0 {
                count += 1
            }
        }

        if cfg!(debug_assertions) {
            println!("coarse culled {}", count)
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
        use cgmath::Rotation;
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
        let mut increment = cgmath::Vector3::zero();
        if self.forward {
            increment += speed * camera.rotation.rotate_vector(FORWARD_VECTOR)
        }
        if self.backward {
            increment -= speed * camera.rotation.rotate_vector(FORWARD_VECTOR);
        }
        if self.up {
            increment += speed * camera.rotation.rotate_vector(UP_VECTOR);
        }
        if self.down {
            increment -= speed * camera.rotation.rotate_vector(UP_VECTOR);
        }
        if self.right {
            increment += speed * camera.rotation.rotate_vector(RIGHT_VECTOR);
        }
        if self.left {
            increment -= speed * camera.rotation.rotate_vector(RIGHT_VECTOR);
        }

        camera.position += increment;
    }
}
