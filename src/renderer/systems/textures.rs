use super::super::super::ecs::custom::*;
use super::super::device::Image;
use super::{
    super::{device::DoubleBuffered, helpers, MainDescriptorPool, RenderFrame},
    present::ImageIndex,
};
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// Synchronize base color texture of GLTF meshes into the shared descriptor set for base color textures
pub struct SynchronizeBaseColorTextures;

pub struct BaseColorDescriptorSet {
    pub layout: super::super::shaders::base_color_set::DescriptorSetLayout,
    pub(in super::super) set: DoubleBuffered<super::super::shaders::base_color_set::DescriptorSet>,
    sampler: helpers::Sampler,
}

pub struct BaseColorVisitedMarker {
    image_view: helpers::ImageView,
}

// Holds the base color texture that will be mapped into a single,
// shared Descriptor Set
pub struct GltfMeshBaseColorTexture(pub Arc<Image>);

impl BaseColorDescriptorSet {
    pub fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &mut MainDescriptorPool,
    ) -> BaseColorDescriptorSet {
        let layout =
            super::super::shaders::base_color_set::DescriptorSetLayout::new(&renderer.device);

        renderer.device.set_object_name(
            layout.layout.handle,
            "Base Color Consolidated Descriptor Set Layout",
        );

        let set = renderer.new_buffered(|ix| {
            let s = super::super::shaders::base_color_set::DescriptorSet::new(
                &main_descriptor_pool,
                &layout,
            );
            renderer.device.set_object_name(
                s.set.handle,
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
    }
}

impl SynchronizeBaseColorTextures {
    pub fn exec(
        entities: &EntitiesStorage,
        renderer: &RenderFrame,
        base_color_descriptor_set: &BaseColorDescriptorSet,
        base_color_textures: &ComponentStorage<GltfMeshBaseColorTexture>,
        image_index: &ImageIndex,
        visited_markers: &mut ComponentStorage<BaseColorVisitedMarker>,
    ) {
        let to_update = (entities.mask() & base_color_textures.mask()) - visited_markers.mask();

        visited_markers.replace_mask(&(visited_markers.mask() | &to_update));

        for entity_id in to_update.iter() {
            let base_color = base_color_textures.get(entity_id).unwrap();
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
            let res = visited_markers.insert(entity_id, BaseColorVisitedMarker { image_view });
            assert!(res.is_none()); // double check that there was nothing there
        }

        let mut counter: u64 = 0;
        assert_eq!(
            vk::Result::SUCCESS,
            (renderer.device.get_semaphore_counter_value)(
                renderer.device.handle(),
                renderer.timeline_semaphore.handle,
                &mut counter
            ),
            "Get semaphore counter value failed",
        );
        dbg!(counter, renderer.frame_number, renderer.frame_number * 16);

        // wait on last frame completion
        let wait_ix = renderer.frame_number * 16;
        let wait_ixes = &[wait_ix];
        let wait_semaphores = &[renderer.timeline_semaphore.handle];
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(wait_semaphores)
            .values(wait_ixes);
        assert_eq!(
            vk::Result::SUCCESS,
            (renderer.device.wait_semaphores)(renderer.device.handle(), &*wait_info, std::u64::MAX),
            "Wait for ix {} failed.",
            wait_ix
        );

        for entity_id in visited_markers.mask().iter() {
            let marker = visited_markers.get(entity_id).unwrap();
            let sampler_updates = &[vk::DescriptorImageInfo::builder()
                .image_view(marker.image_view.handle)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(base_color_descriptor_set.sampler.handle)
                .build()];
            unsafe {
                renderer.device.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::builder()
                        .dst_set(
                            base_color_descriptor_set
                                .set
                                .current(image_index.0)
                                .set
                                .handle,
                        )
                        .dst_binding(0)
                        .dst_array_element(entity_id)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(sampler_updates)
                        .build()],
                    &[],
                );
            }
        }
    }
}
