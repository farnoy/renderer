pub(crate) struct Light {
    #[allow(unused)]
    pub(crate) strength: f32,
}

pub(crate) struct AABB(pub(crate) ncollide3d::bounding_volume::AABB<f32>);

pub(crate) struct ModelMatrix(pub(crate) glm::Mat4);

pub(crate) struct ProjectileTarget(pub(crate) na::Point3<f32>);
pub(crate) struct ProjectileVelocity(pub(crate) f32);

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
