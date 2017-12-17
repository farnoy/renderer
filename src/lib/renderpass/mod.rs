use ash::vk;

pub mod simple_color;

use super::device::AshDevice;
use super::ExampleBase;

pub trait RenderPass {
    fn teardown(&self, &ExampleBase) {}

    fn record_commands<F: FnOnce(&AshDevice, vk::CommandBuffer)>(&self, &ExampleBase, vk::Framebuffer, vk::CommandBuffer, vk::SubpassContents, f: F);

    // TODO: this is not ideal
    fn vk(&self) -> vk::RenderPass;
}
