use cgmath;
use specs::*;
use time;
use super::components::*;

pub struct SteadyRotation;

impl<'a> System<'a> for SteadyRotation {
    type SystemData = (WriteStorage<'a, Rotation>);

    fn run(&mut self, mut rotations: Self::SystemData) {
        let rotation = Rotation(
            cgmath::Matrix3::from_angle_y(-cgmath::Deg(time::precise_time_s() as f32 * 60.0)).into(),
        );
        for rot in (&mut rotations).join() {
            *rot = rotation;
        }
    }
}

pub struct MVPCalculation {
    pub projection: cgmath::Matrix4<f32>,
    pub view: cgmath::Matrix4<f32>,
}

impl<'a> System<'a> for MVPCalculation {
    type SystemData = (ReadStorage<'a, Position>, ReadStorage<'a, Rotation>, ReadStorage<'a, Scale>, WriteStorage<'a, MVP>);

    fn run(&mut self, (positions, rotations, scales, mut mvps): Self::SystemData) {
        for (pos, rot, scale, mvp) in (&positions, &rotations, &scales, &mut mvps).join() {
            let model = cgmath::Matrix4::from_translation(pos.0) * cgmath::Matrix4::from(rot.0) * cgmath::Matrix4::from_scale(scale.0);

            mvp.0 = self.projection * self.view * model;
        }
    }
}
