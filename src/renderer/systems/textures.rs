use crate::renderer::{
    device::{DoubleBuffered, Image},
    graphics as graphics_sync, helpers,
    systems::present::ImageIndex,
    timeline_value_previous, DrawIndex, MainDescriptorPool, RenderFrame,
};
use ash::{version::DeviceV1_0, vk};
use bevy_ecs::prelude::*;
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

pub fn synchronize_base_color_textures_visit(
    mut commands: Commands,
    renderer: Res<RenderFrame>,
    mut query: Query<Without<BaseColorVisitedMarker, (Entity, &GltfMeshBaseColorTexture)>>,
) {
    for (entity, base_color) in &mut query.iter() {
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
        commands.insert_one(entity, BaseColorVisitedMarker { image_view });
    }
}

#[allow(unused_mut)]
pub fn synchronize_base_color_textures_consolidate(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    mut base_color_descriptor_set: ResMut<BaseColorDescriptorSet>,
    mut query: Query<(&DrawIndex, &BaseColorVisitedMarker)>,
) {
    renderer
        .graphics_timeline_semaphore
        .wait(timeline_value_previous::<_, graphics_sync::GuiDraw>(
            &image_index,
            &renderer,
        ))
        .unwrap();

    for (draw_id, marker) in &mut query.iter() {
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
                    .dst_array_element(draw_id.0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(sampler_updates)
                    .build()],
                &[],
            );
        }
    }
}
