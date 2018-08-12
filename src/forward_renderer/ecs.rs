pub mod components;
pub mod systems;

use specs::world;

use self::components::*;

pub struct Bundle;

impl world::Bundle for Bundle {
    fn add_to_world(self, world: &mut world::World) {
        world.register::<Position>();
        world.register::<Rotation>();
        world.register::<Scale>();
        world.register::<Matrices>();
        world.register::<GltfMesh>();
        world.register::<GltfMeshBufferIndex>();
    }
}
