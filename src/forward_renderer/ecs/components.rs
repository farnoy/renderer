use super::super::renderer::Buffer;
use cgmath;
use specs::*;
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
    pub mv: cgmath::Matrix4<f32>,
    pub model: cgmath::Matrix4<f32>,
}

impl Matrices {
    pub fn one() -> Matrices {
        use cgmath::One;
        Matrices {
            mvp: cgmath::Matrix4::one(),
            mv: cgmath::Matrix4::one(),
            model: cgmath::Matrix4::one(),
        }
    }
}

#[derive(Clone, Component)]
#[storage(VecStorage)]
pub struct GltfMesh {
    pub vertex_buffer: Arc<Buffer>,
    pub normal_buffer: Arc<Buffer>,
    pub index_buffer: Arc<Buffer>,
    pub index_len: u64,
    pub bounding_box: (cgmath::Vector3<f32>, cgmath::Vector3<f32>),
}

impl GltfMesh {
    pub fn aabb_vertices(&self) -> [cgmath::Vector3<f32>; 8] {
        let (min, max) = self.bounding_box;
        [
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
    }
}

// Stores AABB in clip space, after mvp transformation
#[derive(Clone, Component)]
#[storage(VecStorage)]
pub struct AABB {
    pub min: cgmath::Vector3<f32>,
    pub max: cgmath::Vector3<f32>,
}

impl AABB {
    pub fn aabb_vertices(&self) -> [cgmath::Vector3<f32>; 8] {
        let (min, max) = (self.min, self.max);
        [
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
    }
}

impl Default for AABB {
    fn default() -> AABB {
        use cgmath::Zero;
        AABB {
            min: cgmath::Vector3::zero(),
            max: cgmath::Vector3::zero(),
        }
    }
}

// Should this entity be discarded when rendering
// Coarse and based on AABB being fully out of the frustum
#[derive(Clone, Component, Debug)]
#[storage(VecStorage)]
pub struct CoarseCulled(pub bool);

// Index in device generated indirect commands
// Can be absent if culled
#[derive(Clone, Component)]
#[storage(VecStorage)]
pub struct GltfMeshBufferIndex(pub u32);

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
