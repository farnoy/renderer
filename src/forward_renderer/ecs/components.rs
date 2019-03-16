use super::super::renderer::device::Buffer;
use cgmath;
use specs::*;
use specs_derive::Component;
use std::sync::Arc;

#[derive(Clone, Copy, Component)]
#[storage(VecStorage)]
pub struct Position(pub cgmath::Vector3<f32>);

#[derive(Clone, Copy, Component)]
#[storage(VecStorage)]
pub struct Rotation(pub cgmath::Quaternion<f32>);

#[derive(Clone, Copy, Component)]
#[storage(VecStorage)]
pub struct Scale(pub f32);

#[derive(Clone, Copy, Component)]
#[storage(VecStorage)]
pub struct Light {
    pub strength: f32,
}

#[derive(Clone, Copy, Component, Debug)]
#[storage(VecStorage)]
pub struct Matrices {
    pub mvp: cgmath::Matrix4<f32>,
    pub model: cgmath::Matrix4<f32>,
}

impl Matrices {
    pub fn one() -> Matrices {
        use cgmath::One;
        Matrices {
            mvp: cgmath::Matrix4::one(),
            model: cgmath::Matrix4::one(),
        }
    }
}

#[derive(Clone, Component)]
#[storage(VecStorage)]
pub struct GltfMesh {
    pub vertex_buffer: Arc<Buffer>,
    pub normal_buffer: Arc<Buffer>,
    pub uv_buffer: Arc<Buffer>,
    pub index_buffer: Arc<Buffer>,
    pub vertex_len: u64,
    pub index_len: u64,
    pub aabb_c: cgmath::Vector3<f32>,
    pub aabb_h: cgmath::Vector3<f32>,
}

// Stores the AABB after translation, rotation, scale
#[derive(Clone, Component, Debug)]
#[storage(VecStorage)]
pub struct AABB {
    pub c: cgmath::Vector3<f32>,
    pub h: cgmath::Vector3<f32>,
}

/*
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
*/
