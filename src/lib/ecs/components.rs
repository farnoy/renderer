use cgmath;
use specs::*;

use super::super::mesh;

#[derive(Clone, Copy, Component)]
#[component(VecStorage)]
pub struct Position(pub cgmath::Vector3<f32>);

#[derive(Clone, Copy, Component)]
#[component(VecStorage)]
pub struct Rotation(pub cgmath::Quaternion<f32>);

#[derive(Clone, Copy, Component)]
#[component(VecStorage)]
pub struct Scale(pub f32);

#[derive(Component)]
#[component(VecStorage)]
pub struct SimpleColorMesh(pub mesh::SimpleColor);

#[derive(Component)]
#[component(VecStorage)]
pub struct TriangleMesh(pub mesh::TriangleMesh);

#[derive(Clone, Copy, Component)]
#[component(VecStorage)]
pub struct Light {
    pub strength: f32,
}

#[derive(Clone, Copy, Component, Debug)]
#[component(VecStorage)]
pub struct Matrices {
    pub mvp: cgmath::Matrix4<f32>,
    pub mv: cgmath::Matrix4<f32>,
}

impl Matrices {
    pub fn one() -> Matrices {
        use cgmath::One;
        Matrices {
            mvp: cgmath::Matrix4::one(),
            mv: cgmath::Matrix4::one(),
        }
    }
}