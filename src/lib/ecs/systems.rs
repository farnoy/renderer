use super::components::*;
use cgmath;
use specs::prelude::*;
use std::sync::Arc;

use super::super::helpers;

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
    pub dst_mvp: Arc<helpers::Buffer>,
    pub dst_mv: Arc<helpers::Buffer>,
    pub dst_model: Arc<helpers::Buffer>,
}

unsafe impl Send for MVPUpload {}

impl<'a> System<'a> for MVPUpload {
    type SystemData = (Entities<'a>, ReadStorage<'a, Matrices>);

    fn run(&mut self, (entities, matrices): Self::SystemData) {
        (&*entities, &matrices)
            .par_join()
            .for_each(|(entity, matrices)| {
                // println!("Writing at {:?} contents {:?}", entity.id(), matrices.mvp);
                use std::slice;
                let out_mvp = unsafe {
                    slice::from_raw_parts_mut(
                        self.dst_mvp.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>,
                        1024,
                    )
                };
                let out_mv = unsafe {
                    slice::from_raw_parts_mut(
                        self.dst_mv.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>,
                        1024,
                    )
                };
                let out_model = unsafe {
                    slice::from_raw_parts_mut(
                        self.dst_model.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>,
                        1024,
                    )
                };
                out_mvp[entity.id() as usize] = matrices.mvp;
                out_mv[entity.id() as usize] = matrices.mv;
                out_model[entity.id() as usize] = matrices.model;
            });
    }
}
