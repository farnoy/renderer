#[cfg(not(feature = "renderdoc"))]
use super::super::{
    super::ecs::components::{GltfMeshBaseColorTexture, GltfMeshBufferIndex},
    device::DescriptorSet,
    helpers, RenderFrame,
};
#[cfg(not(feature = "renderdoc"))]
use ash::{version::DeviceV1_0, vk};
use specs::prelude::*;
#[cfg(not(feature = "renderdoc"))]
use specs_derive::Component;

// Synchronize base color texture of GLTF meshes into the shared descriptor set for base color textures
pub struct SynchronizeBaseColorTextures;

pub struct BaseColorDescriptorSet {
    #[cfg(not(feature = "renderdoc"))]
    pub(in super::super) set: DescriptorSet,
    #[cfg(not(feature = "renderdoc"))]
    sampler: helpers::Sampler,
}

#[derive(Component)]
#[storage(VecStorage)]
#[cfg(not(feature = "renderdoc"))]
pub struct VisitedMarker {
    image_view: helpers::ImageView,
}

pub struct BaseColorSetupHandler;

impl shred::SetupHandler<BaseColorDescriptorSet> for BaseColorSetupHandler {
    #[cfg(not(feature = "renderdoc"))]
    fn setup(res: &mut Resources) {
        let renderer = res.fetch::<RenderFrame>();
        let set = renderer
            .descriptor_pool
            .allocate_set(&renderer.base_color_descriptor_set_layout);
        renderer
            .device
            .set_object_name(set.handle, "Base Color Consolidated descriptor set");

        let sampler = helpers::new_sampler(
            renderer.device.clone(),
            &vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
        );
        drop(renderer);

        res.insert(BaseColorDescriptorSet { set, sampler });
    }

    #[cfg(feature = "renderdoc")]
    fn setup(res: &mut Resources) {
        res.insert(BaseColorDescriptorSet {});
    }
}

#[cfg(not(feature = "renderdoc"))]
impl<'a> System<'a> for SynchronizeBaseColorTextures {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        Entities<'a>,
        WriteExpect<'a, RenderFrame>,
        Write<'a, BaseColorDescriptorSet, BaseColorSetupHandler>,
        ReadStorage<'a, GltfMeshBaseColorTexture>,
        ReadStorage<'a, GltfMeshBufferIndex>,
        WriteStorage<'a, VisitedMarker>,
    );

    fn run(
        &mut self,
        (
            entities,
            renderer,
            base_color_descriptor_set,
            base_color_textures,
            buffer_indices,
            mut visited_markers,
        ): Self::SystemData,
    ) {
        let mut entities_to_update = BitSet::new();
        for (entity, _, ()) in (&*entities, &base_color_textures, !&visited_markers).join() {
            entities_to_update.add(entity.id());
        }

        for (entity, _, base_color) in
            (&*entities, &entities_to_update, &base_color_textures).join()
        {
            let image_view = helpers::new_image_view(
                renderer.device.clone(),
                &vk::ImageViewCreateInfo::builder()
                    .components(
                        vk::ComponentMapping::builder()
                            .r(vk::ComponentSwizzle::IDENTITY)
                            .g(vk::ComponentSwizzle::IDENTITY)
                            .b(vk::ComponentSwizzle::IDENTITY)
                            .a(vk::ComponentSwizzle::IDENTITY)
                            .build(),
                    )
                    .image(base_color.0.handle)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            );
            let res = visited_markers
                .insert(entity, VisitedMarker { image_view })
                .expect("failed to insert VisitedMarker");
            assert!(res.is_none()); // double check that there was nothing there
        }

        for (buffer_index, marker) in (&buffer_indices, &visited_markers).join() {
            let sampler_updates = &[vk::DescriptorImageInfo::builder()
                .image_view(marker.image_view.handle)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(base_color_descriptor_set.sampler.handle)
                .build()];
            unsafe {
                renderer.device.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::builder()
                        .dst_set(base_color_descriptor_set.set.handle)
                        .dst_binding(0)
                        .dst_array_element(buffer_index.0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(sampler_updates)
                        .build()],
                    &[],
                );
            }
        }
    }
}

#[cfg(feature = "renderdoc")]
impl<'a> System<'a> for SynchronizeBaseColorTextures {
    type SystemData = (Write<'a, BaseColorDescriptorSet, BaseColorSetupHandler>,);

    #[allow(unused_variables)]
    fn run(&mut self, base_color_descriptor_set: Self::SystemData) {}
}
