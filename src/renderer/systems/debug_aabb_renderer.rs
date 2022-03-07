use bevy_ecs::prelude::*;

use crate::{
    renderer::{debug_aabb, device::Device, SmartPipeline, SmartPipelineLayout},
    CameraMatrices, RenderFrame,
};

pub(crate) struct DebugAABBPassData {
    pub(crate) pipeline_layout: SmartPipelineLayout<debug_aabb::PipelineLayout>,
    pub(crate) pipeline: SmartPipeline<debug_aabb::Pipeline>,
}

impl FromWorld for DebugAABBPassData {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let camera_matrices = world.get_resource::<CameraMatrices>().unwrap();
        let device = &renderer.device;

        let pipeline_layout = SmartPipelineLayout::new(device, (&camera_matrices.set_layout,));
        let pipeline = SmartPipeline::new(&renderer.device, &pipeline_layout, debug_aabb::Specialization {}, ());

        DebugAABBPassData {
            pipeline_layout,
            pipeline,
        }
    }
}

impl DebugAABBPassData {
    pub(crate) fn destroy(self, device: &Device) {
        self.pipeline.destroy(device);
        self.pipeline_layout.destroy(device);
    }
}
