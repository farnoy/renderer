mod components;
mod systems;

pub use self::systems::*;
pub use self::components::*;

use specs;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use super::device::Device;

pub struct World {
    world: specs::World,
    device: Arc<Device>,
}

impl World {
    pub fn new(device: &Arc<Device>) -> World {
        let mut world = specs::World::new();
        world.register::<Position>();
        world.register::<Rotation>();
        world.register::<Scale>();
        world.register::<SimpleColorMesh>();
        world.register::<TriangleMesh>();
        world.register::<Light>();
        world.register::<Matrices>();

        World {
            world: world,
            device: device.clone(),
        }
    }
}

impl Deref for World {
    type Target = specs::World;

    fn deref(&self) -> &specs::World {
        &self.world
    }
}

impl DerefMut for World {
    fn deref_mut(&mut self) -> &mut specs::World {
        &mut self.world
    }
}

impl Drop for World {
    fn drop(&mut self) {
        use specs::Join;

        let mut mesh_storage = self.world.write::<SimpleColorMesh>();

        for ix in self.world.entities().join() {
            if let Some(detached) = mesh_storage.remove(ix) {
                use super::mesh::Mesh;

                unsafe {
                    detached.0.free(self.device.vk());
                }
            }
        }
    }
}
