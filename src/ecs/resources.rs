use super::super::renderer::{GltfMesh, Image};
use std::sync::Arc;
use winit::{
    self,
    event::{MouseButton, VirtualKeyCode},
};

pub struct MeshLibrary {
    pub projectile: GltfMesh,
    pub projectile_texture: Arc<Image>,
}
