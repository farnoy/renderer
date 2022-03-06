use ash::vk;
use bevy_ecs::prelude::*;
use petgraph::prelude::*;
use profiling::scope;
use renderer_vma::VmaMemoryUsage;

#[cfg(feature = "crash_debugging")]
use crate::renderer::CrashBuffer;
use crate::renderer::{
    device::{Device, ImageView},
    frame_graph,
    helpers::command_util::CommandUtil,
    ImageIndex, MainDescriptorPool, RenderFrame, SmartPipeline, SmartPipelineLayout, SmartSet, SmartSetLayout,
    Submissions, Swapchain, SwapchainIndexToFrameNumber,
};

renderer_macros::define_pass!(ReferenceRaytrace on compute);
renderer_macros::define_resource!(ReferenceRaytraceOutput = Image COLOR);

renderer_macros::define_set! {
    rt_output_set {
        output_image STORAGE_IMAGE from [COMPUTE],
    }
}

renderer_macros::define_pipe! {
    reference_rt {
        descriptors [rt_output_set]
        varying subgroup size
        compute
    }
}

pub(crate) fn reference_raytrace(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    swapchain_index_map: Res<SwapchainIndexToFrameNumber>,
    submissions: Res<Submissions>,
    data: Res<ReferenceRTData>,
    mut priv_data: ResMut<ReferenceRTDataPrivate>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
) {
    scope!("ecs::reference_raytrace");

    if !submissions
        .active_graph
        .contains_node(NodeIndex::from(frame_graph::ReferenceRaytrace::INDEX))
    {
        return;
    }

    let ReferenceRTData { ref output_image, .. } = *data;
    let ReferenceRTDataPrivate {
        ref mut command_util,
        ref pipe,
        ref pipe_layout,
        ref set,
        ..
    } = *priv_data;

    let cb = command_util.reset_and_record(&renderer, &image_index);

    let marker = cb.debug_marker_around("reference RT", [1.0, 0.0, 0.0, 1.0]);

    let barrier = renderer_macros::barrier!(
        cb,
        TLAS.in_reference r in ReferenceRaytrace descriptor gltf_mesh.acceleration_set.top_level_as after [build] if [RT, REFERENCE_RT],
        ReferenceRaytraceOutput.generate w in ReferenceRaytrace descriptor gltf_mesh.acceleration_set.top_level_as layout GENERAL if [RT, REFERENCE_RT]; output_image
    );

    unsafe {
        renderer
            .device
            .cmd_bind_pipeline(*cb, vk::PipelineBindPoint::COMPUTE, pipe.vk());

        pipe_layout.bind_descriptor_sets(&renderer.device, *cb, (set,));

        renderer.device.cmd_dispatch(*cb, 50, 400, 1);
    }

    drop(barrier);
    drop(marker);

    let cb = cb.end();

    submissions.submit(
        &renderer,
        frame_graph::ReferenceRaytrace::INDEX,
        Some(*cb),
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
    )
}

pub(crate) struct ReferenceRTData {
    pub(crate) output_image: ReferenceRaytraceOutput,
    pub(crate) output_image_view: ImageView,
}

impl FromWorld for ReferenceRTData {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let swapchain = world.get_resource::<Swapchain>().unwrap();
        let image = renderer.device.new_image_exclusive(
            vk::Format::R8G8B8A8_SNORM,
            vk::Extent3D {
                width: swapchain.width,
                height: swapchain.height,
                depth: 1,
            },
            vk::SampleCountFlags::TYPE_1,
            vk::ImageTiling::OPTIMAL,
            vk::ImageLayout::UNDEFINED,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE,
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        );
        let output_image_view = renderer.device.new_image_view(
            &vk::ImageViewCreateInfo::builder()
                .components(
                    vk::ComponentMapping::builder()
                        .r(vk::ComponentSwizzle::IDENTITY)
                        .g(vk::ComponentSwizzle::IDENTITY)
                        .b(vk::ComponentSwizzle::IDENTITY)
                        .a(vk::ComponentSwizzle::IDENTITY)
                        .build(),
                )
                .image(image.handle)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SNORM)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                ),
        );
        ReferenceRTData {
            output_image: ReferenceRaytraceOutput::import(&renderer.device, image),
            output_image_view,
        }
    }
}

impl ReferenceRTData {
    pub(crate) fn destroy(self, device: &Device) {
        self.output_image.destroy(device);
        self.output_image_view.destroy(device);
    }
}

pub(crate) struct ReferenceRTDataPrivate {
    set_layout: SmartSetLayout<rt_output_set::Layout>,
    set: SmartSet<rt_output_set::Set>,
    pipe_layout: SmartPipelineLayout<reference_rt::PipelineLayout>,
    pipe: SmartPipeline<reference_rt::Pipeline>,
    command_util: CommandUtil,
}

impl FromWorld for ReferenceRTDataPrivate {
    fn from_world(world: &mut World) -> Self {
        let renderer: &RenderFrame = world.get_resource().unwrap();
        let main_descriptor_pool: &MainDescriptorPool = world.get_resource().unwrap();
        let rt_data: &ReferenceRTData = world.get_resource().unwrap();
        let set_layout = SmartSetLayout::new(&renderer.device);
        let set = SmartSet::new(&renderer.device, main_descriptor_pool, &set_layout, 0);
        {
            let update = &[vk::DescriptorImageInfo::builder()
                .image_view(rt_data.output_image_view.handle)
                .image_layout(vk::ImageLayout::GENERAL) // TODO: extract this out of the frame_graph
                .build()];
            unsafe {
                renderer.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::builder()
                        .dst_set(set.vk_handle())
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(update)
                        .build()],
                    &[],
                );
            }
            let pipe_layout = SmartPipelineLayout::new(&renderer.device, (&set_layout,));
            let pipe = SmartPipeline::new(&renderer.device, &pipe_layout, reference_rt::Specialization {}, ());

            ReferenceRTDataPrivate {
                set_layout,
                set,
                pipe_layout,
                pipe,
                command_util: CommandUtil::new(renderer, renderer.device.compute_queue_family),
            }
        }
    }
}

impl ReferenceRTDataPrivate {
    pub(crate) fn destroy(self, device: &Device, pool: &MainDescriptorPool) {
        self.command_util.destroy(device);
        self.pipe.destroy(device);
        self.pipe_layout.destroy(device);
        self.set.destroy(&pool.0, device);
        self.set_layout.destroy(device);
    }
}
