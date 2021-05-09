use std::sync::Arc;

use ash::{version::DeviceV1_0, vk};
use bevy_ecs::prelude::*;
use microprofile::scope;

use crate::{
    ecs::components::Deleting,
    renderer::{
        shaders, systems::present::ImageIndex, Device, DrawIndex, GraphicsTimeline, Image, ImageView,
        MainDescriptorPool, RenderFrame, Sampler, SwapchainIndexToFrameNumber,
    },
};

pub(crate) struct BaseColorDescriptorSet {
    pub(crate) layout: shaders::base_color_set::Layout,
    pub(crate) set: shaders::base_color_set::Set,
    sampler: Sampler,
}

pub(crate) struct BaseColorVisitedMarker {
    image_view: ImageView,
}

// Holds the base color texture that will be mapped into a single,
// shared Descriptor Set
pub(crate) struct GltfMeshBaseColorTexture(pub(crate) Arc<Image>);

impl BaseColorDescriptorSet {
    pub(crate) fn new(renderer: &RenderFrame, main_descriptor_pool: &mut MainDescriptorPool) -> BaseColorDescriptorSet {
        let layout = shaders::base_color_set::Layout::new(&renderer.device);

        let set = shaders::base_color_set::Set::new(&renderer.device, &main_descriptor_pool, &layout, 0);

        let sampler = renderer.device.new_sampler(
            &vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
        );

        BaseColorDescriptorSet { layout, set, sampler }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.sampler.destroy(device);
        self.set.destroy(&main_descriptor_pool.0, device);
        self.layout.destroy(device);
    }
}

impl BaseColorVisitedMarker {
    pub(crate) fn destroy(self, device: &Device) {
        self.image_view.destroy(device);
    }
}

pub(crate) fn synchronize_base_color_textures_visit(
    mut commands: Commands,
    renderer: Res<RenderFrame>,
    query: Query<(Entity, &GltfMeshBaseColorTexture), Without<BaseColorVisitedMarker>>,
) {
    for (entity, base_color) in &mut query.iter() {
        let image_view = renderer.device.new_image_view(
            &vk::ImageViewCreateInfo::builder()
                .components(
                    vk::ComponentMapping::builder()
                        .r(vk::ComponentSwizzle::IDENTITY)
                        .g(vk::ComponentSwizzle::IDENTITY)
                        .b(vk::ComponentSwizzle::IDENTITY)
                        .a(vk::ComponentSwizzle::ONE)
                        .build(),
                )
                .image(base_color.0.handle)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(if cfg!(feature = "compress_textures") {
                    vk::Format::BC7_UNORM_BLOCK
                } else {
                    vk::Format::R8G8B8A8_UNORM
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        );
        commands.entity(entity).insert(BaseColorVisitedMarker { image_view });
    }
}

pub(crate) fn update_base_color_descriptors(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<SwapchainIndexToFrameNumber>,
    base_color_descriptor_set: Res<BaseColorDescriptorSet>,
    query: Query<(&DrawIndex, &BaseColorVisitedMarker)>,
) {
    scope!("ecs", "update_base_color_descriptors");

    renderer
        .graphics_timeline_semaphore
        .wait(
            &renderer.device,
            GraphicsTimeline::SceneDraw.as_of_previous(&image_index, &swapchain_index_map),
        )
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
                    .dst_set(base_color_descriptor_set.set.set.handle)
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

pub(crate) fn cleanup_base_color_markers(world: &mut World) {
    scope!("ecs", "cleanup_base_color_markers");

    let frame_number = world.get_resource::<RenderFrame>().unwrap().frame_number;
    let swapchain_index = world.get_resource::<ImageIndex>().cloned().unwrap();

    let mut entities = vec![];

    world
        .query_filtered::<(Entity, &Deleting), With<BaseColorVisitedMarker>>()
        .for_each(world, |(entity, deleting)| {
            if deleting.frame_number < frame_number && deleting.image_index == swapchain_index {
                entities.push(entity);
            }
        });

    let markers = entities
        .into_iter()
        .map(|entity| world.entity_mut(entity).remove::<BaseColorVisitedMarker>().unwrap())
        .collect::<Vec<_>>();

    let renderer = world.get_resource::<RenderFrame>().unwrap();

    for marker in markers.into_iter() {
        marker.destroy(&renderer.device);
    }

    // the descriptor binding isn't UPDATE_AFTER_BIND (TODO: const assertion would be nice)
    // so we shouldn't need to overwrite it, if the DrawIndex of that entity gets reused, the
    // binding will be updated
}
