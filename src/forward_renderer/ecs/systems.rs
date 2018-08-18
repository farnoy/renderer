use super::components::*;
use cgmath;
use parking_lot::Mutex;
use specs::prelude::*;
use std::{slice::from_raw_parts_mut, sync::Arc};
use winit::{self, dpi::LogicalSize, Event, KeyboardInput, WindowEvent};

use super::super::renderer::Buffer;

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

pub struct MVPCalculation {
    pub projection: cgmath::Matrix4<f32>,
    pub view: cgmath::Matrix4<f32>,
}

impl<'a> System<'a> for MVPCalculation {
    type SystemData = (
        ReadStorage<'a, Position>,
        ReadStorage<'a, Rotation>,
        ReadStorage<'a, Scale>,
        WriteStorage<'a, Matrices>,
    );

    fn run(&mut self, (positions, rotations, scales, mut mvps): Self::SystemData) {
        for (pos, rot, scale, mvp) in (&positions, &rotations, &scales, &mut mvps).join() {
            mvp.model = cgmath::Matrix4::from_translation(pos.0)
                * cgmath::Matrix4::from(rot.0)
                * cgmath::Matrix4::from_scale(scale.0);

            mvp.mvp = self.projection * self.view * mvp.model;
            mvp.mv = self.view * mvp.model;
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
        ReadStorage<'a, GltfMesh>,
        WriteStorage<'a, GltfMeshBufferIndex>,
    );

    fn run(&mut self, (meshes, mut indices): Self::SystemData) {
        for (ix, (_mesh, buffer_index)) in (&meshes, &mut indices).join().enumerate() {
            buffer_index.0 = ix as u32;
        }
    }
}

pub struct InputHandler {
    pub events_loop: winit::EventsLoop,
    pub quit_handle: Arc<Mutex<bool>>,
}

impl<'a> System<'a> for InputHandler {
    type SystemData = (Entities<'a>,);

    fn run(&mut self, entities: Self::SystemData) {
        let quit_handle = Arc::clone(&self.quit_handle);
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
            }
            | Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(winit::VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                *quit_handle.lock() = true;
            }
            _ => (),
        });
    }
}
