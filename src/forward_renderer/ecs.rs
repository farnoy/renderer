pub mod components;
pub mod systems;

use specs::World;

use self::components::*;
use super::renderer::setup_ecs as renderer_setup;

pub fn setup(world: &mut World) {
    world.register::<Position>();
    world.register::<Rotation>();
    world.register::<Scale>();
    world.register::<Matrices>();
    world.register::<AABB>();
    world.register::<GltfMesh>();

    renderer_setup(world);
}
