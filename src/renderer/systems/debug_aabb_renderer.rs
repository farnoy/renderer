use bevy_ecs::prelude::*;

use crate::{
    renderer::{device::Device, shaders},
    CameraMatrices, MainRenderpass, RenderFrame,
};

pub(crate) struct DebugAABBPassData {
    pub(crate) pipeline_layout: shaders::debug_aabb::PipelineLayout,
    pub(crate) pipeline: shaders::debug_aabb::Pipeline,
}

impl FromWorld for DebugAABBPassData {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let main_renderpass = world.get_resource::<MainRenderpass>().unwrap();
        let camera_matrices = world.get_resource::<CameraMatrices>().unwrap();
        let device = &renderer.device;

        let pipeline_layout = shaders::debug_aabb::PipelineLayout::new(&device, &camera_matrices.set_layout);
        let pipeline = shaders::debug_aabb::Pipeline::new(
            &renderer.device,
            &pipeline_layout,
            shaders::debug_aabb::Specialization {},
            [None, None],
            &main_renderpass.renderpass.renderpass,
            1,
        );

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
