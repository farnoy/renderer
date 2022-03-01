use bevy_ecs::prelude::*;

use crate::renderer::ImageIndex;

#[derive(Component)]
pub(crate) struct Position(pub(crate) na::Point3<f32>);
#[derive(Component)]
pub(crate) struct Rotation(pub(crate) na::UnitQuaternion<f32>);
#[derive(Component)]
pub(crate) struct Scale(pub(crate) f32);

#[derive(Component)]
pub(crate) struct Light {
    #[allow(unused)]
    pub(crate) strength: f32,
}

#[derive(Component)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct AABB(pub(crate) ncollide3d::bounding_volume::AABB<f32>);

#[derive(Component)]
pub(crate) struct ModelMatrix(pub(crate) glm::Mat4);

#[derive(Component)]
pub(crate) struct ProjectileTarget(pub(crate) na::Point3<f32>);
#[derive(Component)]
pub(crate) struct ProjectileVelocity(pub(crate) f32);

/// Used as a marker, will be despawned at the end of the next frame that used the same swapchain
/// index. Systems that allocate dynamic resources for entities should use this as a signal to clean
/// up.
#[derive(Debug, Component)]
#[component(storage = "SparseSet")]
pub(crate) struct Deleting {
    pub(crate) frame_number: u64,
    pub(crate) image_index: ImageIndex,
}

impl Default for AABB {
    fn default() -> AABB {
        AABB(ncollide3d::bounding_volume::AABB::<f32>::from_half_extents(
            na::Point3::origin(),
            na::zero(),
        ))
    }
}

impl Default for ModelMatrix {
    fn default() -> ModelMatrix {
        ModelMatrix(glm::Mat4::identity())
    }
}
