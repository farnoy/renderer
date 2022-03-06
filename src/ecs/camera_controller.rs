use bevy_ecs::prelude::*;
use winit::{self, event::VirtualKeyCode};

use crate::{
    ecs::{resources::InputActions, systems::*},
    renderer::{forward_vector, right_vector, up_vector},
};

#[derive(Clone, Component)]
pub(crate) struct Camera {
    pub(crate) position: na::Point3<f32>,
    pub(crate) rotation: na::UnitQuaternion<f32>,
    pub(crate) projection: na::Matrix4<f32>,
    pub(crate) view: na::Matrix4<f32>,
    // left -> right -> bottom -> top -> near -> far
    pub(crate) frustum_planes: [na::Vector4<f32>; 6],
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

pub(crate) fn camera_controller(
    input_actions: Res<InputActions>,
    frame_timing: Res<FrameTiming>,
    runtime_config: Res<FutureRuntimeConfiguration>,
    mut camera: ResMut<Camera>,
) {
    if !runtime_config.0[0].fly_mode {
        return;
    }

    let forward = input_actions.get_key_down(VirtualKeyCode::W);
    let backward = input_actions.get_key_down(VirtualKeyCode::S);
    let right = input_actions.get_key_down(VirtualKeyCode::D);
    let left = input_actions.get_key_down(VirtualKeyCode::A);
    let up = input_actions.get_key_down(VirtualKeyCode::Space);
    let down = input_actions.get_key_down(VirtualKeyCode::LControl);
    let fast = input_actions.get_key_down(VirtualKeyCode::LShift);

    let speed = if fast { 10.0 } else { 1.0 } * frame_timing.time_delta;
    let mut increment: na::Vector3<f32> = na::zero();
    if forward {
        increment += speed * camera.rotation.transform_vector(&forward_vector())
    }
    if backward {
        increment -= speed * camera.rotation.transform_vector(&forward_vector());
    }
    if up {
        increment += speed * camera.rotation.transform_vector(&up_vector());
    }
    if down {
        increment -= speed * camera.rotation.transform_vector(&up_vector());
    }
    if right {
        increment += speed * camera.rotation.transform_vector(&right_vector());
    }
    if left {
        increment -= speed * camera.rotation.transform_vector(&right_vector());
    }

    camera.position += increment;
}

/*
pub(crate) struct CameraController;

impl CameraController {
    pub(crate) fn exec_system() -> Box<(dyn legion::systems::schedule::Schedulable + 'static)> {
        use legion::prelude::*;
        SystemBuilder::<()>::new("CameraController")
            .read_resource::<InputActions>()
            .read_resource::<FrameTiming>()
            .read_resource::<RuntimeConfiguration>()
            .write_resource::<Camera>()
            .build(move |_commands, _world, resources, _query| {
            })
    }
}
*/
