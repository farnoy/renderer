pub mod components;
pub mod systems;

use specs::World;

use self::components::*;
use super::renderer::BaseColorVisitedMarker;

pub fn setup(world: &mut World) {
    world.register::<Position>();
    world.register::<Rotation>();
    world.register::<Scale>();
    world.register::<Matrices>();
    world.register::<AABB>();
    world.register::<GltfMesh>();
    world.register::<GltfMeshBufferIndex>();
    world.register::<GltfMeshCullDescriptorSet>();
    world.register::<GltfMeshBaseColorTexture>();
    world.register::<BaseColorVisitedMarker>();
    world.register::<CoarseCulled>();
}
