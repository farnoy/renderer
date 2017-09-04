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

#[derive(Clone, Copy, Component)]
#[component(VecStorage)]
pub struct MVP(pub cgmath::Matrix4<f32>);
