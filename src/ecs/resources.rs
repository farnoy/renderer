use super::super::renderer::{GltfMesh, Image};
use std::sync::Arc;

pub struct MeshLibrary {
    pub projectile: GltfMesh,
    pub projectile_texture: Arc<Image>,
}
