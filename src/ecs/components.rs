use cgmath;
use specs::{storage::BTreeStorage, Component, *};

#[derive(Clone, Copy, Component)]
#[storage(VecStorage)]
pub struct Position(pub cgmath::Point3<f32>);

#[derive(Clone, Copy, Component)]
#[storage(VecStorage)]
pub struct Rotation(pub cgmath::Quaternion<f32>);

#[derive(Clone, Copy, Component)]
#[storage(VecStorage)]
pub struct Scale(pub f32);

#[derive(Clone, Copy, Component)]
#[storage(BTreeStorage)]
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
