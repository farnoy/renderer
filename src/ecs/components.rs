pub struct Light {
    pub strength: f32,
}

// Stores the AABB after translation, rotation, scale
pub struct AABB {
    pub c: na::Vector3<f32>,
    pub h: na::Vector3<f32>,
}
