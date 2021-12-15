use ash::vk;
use bevy_ecs::prelude::*;
use petgraph::prelude::*;
use profiling::scope;
use renderer_vma::VmaMemoryUsage;

#[cfg(feature = "crash_debugging")]
use crate::renderer::CrashBuffer;
use crate::renderer::{
    device::Device, frame_graph, helpers::command_util::CommandUtil, ImageIndex, RenderFrame, Submissions, Swapchain,
};

renderer_macros::define_pass!(ReferenceRaytrace on compute);
renderer_macros::define_resource!(ReferenceRaytraceOutput = Image);

pub(crate) fn reference_raytrace(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    submissions: Res<Submissions>,
    data: Res<ReferenceRTData>,
    mut priv_data: ResMut<ReferenceRTDataPrivate>,
    renderer_input: Res<renderer_macro_lib::RendererInput>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
) {
    scope!("ecs::reference_raytrace");

    if !submissions
        .active_graph
        .contains_node(NodeIndex::from(frame_graph::ReferenceRaytrace::INDEX))
    {
        return;
    }

    let ReferenceRTData { ref output_image } = *data;
    let ReferenceRTDataPrivate { ref mut command_util } = *priv_data;

    let cb = command_util.reset_and_record(&renderer, &image_index);

    let marker = cb.debug_marker_around("reference RT", [1.0, 0.0, 0.0, 1.0]);

    let barrier = renderer_macros::barrier!(
        *cb,
        TLAS.in_reference r in ReferenceRaytrace descriptor gltf_mesh.acceleration_set.top_level_as after [build],
        ReferenceRaytraceOutput.generate w in ReferenceRaytrace descriptor gltf_mesh.acceleration_set.top_level_as layout GENERAL; {output_image}
    );

    drop(barrier);
    drop(marker);

    let cb = cb.end();

    submissions.submit(
        &renderer,
        frame_graph::ReferenceRaytrace::INDEX,
        Some(*cb),
        &renderer_input,
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
    )
}

pub(crate) struct ReferenceRTData {
    pub(crate) output_image: ReferenceRaytraceOutput,
}

impl FromWorld for ReferenceRTData {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let swapchain = world.get_resource::<Swapchain>().unwrap();
        let image = renderer.device.new_image_exclusive(
            vk::Format::R8G8B8A8_UINT,
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
        ReferenceRTData {
            output_image: ReferenceRaytraceOutput::import(&renderer.device, image),
        }
    }
}

impl ReferenceRTData {
    pub(crate) fn destroy(self, device: &Device) {
        self.output_image.destroy(device);
    }
}

pub(crate) struct ReferenceRTDataPrivate {
    command_util: CommandUtil,
}

impl FromWorld for ReferenceRTDataPrivate {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource().unwrap();
        ReferenceRTDataPrivate {
            command_util: CommandUtil::new(renderer, renderer.device.compute_queue_family),
        }
    }
}

impl ReferenceRTDataPrivate {
    pub(crate) fn destroy(self, device: &Device) {
        self.command_util.destroy(device);
    }
}
