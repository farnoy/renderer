use ash::vk;
use cgmath;
use specs::*;

use super::super::mesh;
use super::super::render_dag::v3::alloc;

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

#[derive(Clone, Copy, Component, Debug)]
#[component(VecStorage)]
pub struct VertexBuffer {
    pub buffer: vk::Buffer,
    pub allocation: alloc::VmaAllocation,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    pub index_type: vk::IndexType,
}

#[derive(Clone, Copy, Component, Debug)]
#[component(BTreeStorage)]
pub struct UploadJob {
    pub src: vk::Buffer,
    pub src_allocation: alloc::VmaAllocation,
    pub dst: vk::Buffer,
    pub size: vk::DeviceSize,
}
