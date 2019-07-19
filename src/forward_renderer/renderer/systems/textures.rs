use super::super::device::Image;
use super::{
    super::{
        device::{DescriptorSet, DescriptorSetLayout, DoubleBuffered},
        helpers, MainDescriptorPool, RenderFrame,
    },
    cull_pipeline::GltfMeshBufferIndex,
    present::ImageIndex,
};
use ash::{version::DeviceV1_0, vk};
use specs::prelude::*;
use specs::Component;
use std::{ptr, sync::Arc};

// Synchronize base color texture of GLTF meshes into the shared descriptor set for base color textures
pub struct SynchronizeBaseColorTextures;

pub struct BaseColorDescriptorSet {
    pub layout: DescriptorSetLayout,
    pub(in super::super) set: DoubleBuffered<DescriptorSet>,
    sampler: helpers::Sampler,
}

#[derive(Component)]
#[storage(VecStorage)]
pub struct VisitedMarker {
    image_view: helpers::ImageView,
}

// Holds the base color texture that will be mapped into a single,
// shared Descriptor Set
#[derive(Component)]
#[storage(VecStorage)]
pub struct GltfMeshBaseColorTexture(pub Arc<Image>);

impl specs::shred::SetupHandler<BaseColorDescriptorSet> for BaseColorDescriptorSet {
    fn setup(world: &mut World) {
        if world.has_value::<BaseColorDescriptorSet>() {
            return;
        }

        let result = world.exec(
            |(renderer, main_descriptor_pool): (
                ReadExpect<RenderFrame>,
                Write<MainDescriptorPool, MainDescriptorPool>,
            )| {
                let layout = {
                    let mut binding_flags =
                        vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
                            .binding_flags(&[vk::DescriptorBindingFlagsEXT::PARTIALLY_BOUND]);
                    let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&[vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 3072,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            p_immutable_samplers: ptr::null(),
                        }])
                        .push_next(&mut binding_flags);

                    renderer.device.new_descriptor_set_layout2(&create_info)
                };

                renderer.device.set_object_name(
                    layout.handle,
                    "Base Color Consolidated Descriptor Set Layout",
                );

                let set = renderer.new_buffered(|ix| {
                    let s = main_descriptor_pool.0.allocate_set(&layout);
                    renderer.device.set_object_name(
                        s.handle,
                        &format!("Base Color Consolidated descriptor set - {}", ix),
                    );
                    s
                });

                let sampler = helpers::new_sampler(
                    renderer.device.clone(),
                    &vk::SamplerCreateInfo::builder()
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                );

                BaseColorDescriptorSet {
                    layout,
                    set,
                    sampler,
                }
            },
        );

        world.insert(result);
    }
}

impl<'a> System<'a> for SynchronizeBaseColorTextures {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        Entities<'a>,
        ReadExpect<'a, RenderFrame>,
        Write<'a, BaseColorDescriptorSet, BaseColorDescriptorSet>,
        ReadStorage<'a, GltfMeshBaseColorTexture>,
        ReadStorage<'a, GltfMeshBufferIndex>,
        Read<'a, ImageIndex>,
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
            image_index,
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
                        .dst_set(base_color_descriptor_set.set.current(image_index.0).handle)
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
