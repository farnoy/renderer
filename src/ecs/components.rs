pub struct Light {
    pub strength: f32,
}

pub struct AABB(pub ncollide3d::bounding_volume::AABB<f32>);

pub struct ModelMatrix(pub glm::Mat4);

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
