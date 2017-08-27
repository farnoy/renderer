use ash::vk;
use std::ptr;
use std::sync::Arc;

use super::instance::Instance;
use super::ExampleBase;
use super::renderpass::RenderPass;

pub mod textured_mesh;

pub trait GraphicsPipeline {
    fn new<R: RenderPass>(&ExampleBase, &R) -> Self;
    fn record_commands(&self, &ExampleBase, vk::CommandBuffer);
    fn vk(&self) -> vk::Pipeline;
}
