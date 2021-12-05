use std::{mem::replace, sync::Arc};

use ash::vk;
use bevy_ecs::prelude::*;
use profiling::scope;

use crate::{
    ecs::components::Deleting,
    renderer::{
        device::DoubleBuffered, frame_graph, systems::present::ImageIndex, textures_set, CopiedResource, Device,
        DrawIndex, Image, ImageView, MainDescriptorPool, RenderFrame, RenderStage, Sampler, SmartSet, SmartSetLayout,
        SwapchainIndexToFrameNumber,
    },
};

pub(crate) struct BaseColorDescriptorSet {
    pub(crate) layout: SmartSetLayout<textures_set::Layout>,
    pub(crate) set: DoubleBuffered<SmartSet<textures_set::Set>>,
    sampler: Sampler,
}

pub(crate) struct BaseColorVisitedMarker {
    image_view: ImageView,
}

pub(crate) struct NormalMapVisitedMarker {
    image_view: ImageView,
}

// Holds the base color texture that will be mapped into a single,
// shared Descriptor Set
#[derive(Clone)]
pub(crate) struct GltfMeshBaseColorTexture(pub(crate) Arc<Image>);

// Holds the normal map texture
#[derive(Clone)]
pub(crate) struct GltfMeshNormalTexture(pub(crate) Arc<Image>);

impl BaseColorDescriptorSet {
    pub(crate) fn new(renderer: &RenderFrame, main_descriptor_pool: &mut MainDescriptorPool) -> BaseColorDescriptorSet {
        let layout = SmartSetLayout::new(&renderer.device);

        let set = renderer.new_buffered(|ix| SmartSet::new(&renderer.device, main_descriptor_pool, &layout, ix));

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
        self.set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
        self.layout.destroy(device);
    }
}

impl BaseColorVisitedMarker {
    pub(crate) fn destroy(self, device: &Device) {
        self.image_view.destroy(device);
    }
}

impl NormalMapVisitedMarker {
    pub(crate) fn destroy(self, device: &Device) {
        self.image_view.destroy(device);
    }
}

pub(crate) fn synchronize_base_color_textures_visit(
    mut commands: Commands,
    renderer: Res<RenderFrame>,
    query: Query<
        (Entity, &GltfMeshBaseColorTexture, &GltfMeshNormalTexture),
        (
            Without<BaseColorVisitedMarker>,
            Without<NormalMapVisitedMarker>,
            Without<Deleting>,
        ),
    >,
) {
    for (entity, base_color, normal_map) in &mut query.iter() {
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
                    vk::Format::R8G8B8A8_SRGB
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        );
        renderer.device.set_object_name(
            image_view.handle,
            &format!("EntityId({:?}).BaseColorVisitedMarker.image_view", entity),
        );

        let normal_map_image_view = renderer.device.new_image_view(
            &vk::ImageViewCreateInfo::builder()
                .components(
                    vk::ComponentMapping::builder()
                        .r(vk::ComponentSwizzle::IDENTITY)
                        .g(vk::ComponentSwizzle::IDENTITY)
                        .b(vk::ComponentSwizzle::IDENTITY)
                        .a(vk::ComponentSwizzle::IDENTITY)
                        .build(),
                )
                .image(normal_map.0.handle)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(if cfg!(feature = "compress_textures") {
                    vk::Format::BC7_UNORM_BLOCK
                } else {
                    vk::Format::R8G8B8A8_SRGB
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        );
        renderer.device.set_object_name(
            normal_map_image_view.handle,
            &format!("EntityId({:?}).NormalMapVisitedMarker.image_view", entity),
        );

        commands
            .entity(entity)
            .insert(BaseColorVisitedMarker { image_view })
            .insert(NormalMapVisitedMarker {
                image_view: normal_map_image_view,
            });
    }
}

pub(crate) fn recreate_base_color_descriptor_set(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<SwapchainIndexToFrameNumber>,
    mut base_color_descriptor_set: ResMut<BaseColorDescriptorSet>,
    main_descriptor_pool: Res<MainDescriptorPool>,
) {
    scope!("ecs::recreate_base_color_descriptor_set");

    frame_graph::Main::Stage::wait_previous(&renderer, &image_index, &swapchain_index_map);

    let new_set = SmartSet::new(
        &renderer.device,
        &main_descriptor_pool,
        &base_color_descriptor_set.layout,
        image_index.0,
    );

    replace(base_color_descriptor_set.set.current_mut(image_index.0), new_set)
        .destroy(&main_descriptor_pool.0, &renderer.device);
}

pub(crate) fn update_base_color_descriptors(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<SwapchainIndexToFrameNumber>,
    base_color_descriptor_set: Res<BaseColorDescriptorSet>,
    query: Query<(&DrawIndex, &BaseColorVisitedMarker, &NormalMapVisitedMarker)>,
) {
    scope!("ecs::update_base_color_descriptors");

    frame_graph::Main::Stage::wait_previous(&renderer, &image_index, &swapchain_index_map);

    for (draw_id, base_color, normal_map) in &mut query.iter() {
        let base_color_update = &[vk::DescriptorImageInfo::builder()
            .image_view(base_color.image_view.handle)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(base_color_descriptor_set.sampler.handle)
            .build()];
        let normal_map_update = &[vk::DescriptorImageInfo::builder()
            .image_view(normal_map.image_view.handle)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(base_color_descriptor_set.sampler.handle)
            .build()];
        unsafe {
            renderer.device.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_set(base_color_descriptor_set.set.current(image_index.0).set.handle)
                        .dst_binding(0)
                        .dst_array_element(draw_id.0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(base_color_update)
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(base_color_descriptor_set.set.current(image_index.0).set.handle)
                        .dst_binding(1)
                        .dst_array_element(draw_id.0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(normal_map_update)
                        .build(),
                ],
                &[],
            );
        }
    }
}

pub(crate) fn cleanup_base_color_markers(world: &mut World) {
    scope!("ecs::cleanup_base_color_markers");

    let renderer = world.get_resource::<RenderFrame>().unwrap();
    let frame_number = world.get_resource::<RenderFrame>().unwrap().frame_number;
    let swapchain_index = world.get_resource::<ImageIndex>().cloned().unwrap();
    let previous_indices = world
        .get_resource::<CopiedResource<SwapchainIndexToFrameNumber>>()
        .unwrap();

    frame_graph::Main::Stage::wait_previous(renderer, &swapchain_index, previous_indices);

    let mut entities = vec![];

    world
        .query_filtered::<(Entity, &Deleting), (With<BaseColorVisitedMarker>, With<NormalMapVisitedMarker>)>()
        .for_each(world, |(entity, deleting)| {
            if deleting.frame_number < frame_number && deleting.image_index == swapchain_index {
                entities.push(entity);
            }
        });

    world.resource_scope(|world, renderer: Mut<RenderFrame>| {
        let markers = entities.into_iter().map(|entity| {
            (
                world.entity_mut(entity).remove::<BaseColorVisitedMarker>().unwrap(),
                world.entity_mut(entity).remove::<NormalMapVisitedMarker>().unwrap(),
            )
        });

        for (base_color, normal_map) in markers {
            base_color.destroy(&renderer.device);
            normal_map.destroy(&renderer.device);
        }
    });

    // the descriptor binding isn't UPDATE_AFTER_BIND (TODO: const assertion would be nice)
    // so we shouldn't need to overwrite it, if the DrawIndex of that entity gets reused, the
    // binding will be updated
}
