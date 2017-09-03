use ash;
use ash::version::DeviceV1_0;
use ash::vk;
use specs;
use std::ptr;

use super::super::device::AshDevice;
use super::super::ecs::SimpleColorMesh;
use super::super::ExampleBase;
use super::RenderPass;

pub struct SimpleColor {
    renderpass: vk::RenderPass,
}

/*
impl<'a> specs::System<'a> for SimpleColor {
    type SystemData = (specs::ReadStorage<'a, SimpleColorMesh>);

    fn run(&mut self, data: Self::SystemData) {}
}
*/

impl SimpleColor {
    pub fn setup(base: &ExampleBase) -> SimpleColor {
        let renderpass_attachments = [
            vk::AttachmentDescription {
                format: base.surface_format.format,
                flags: vk::AttachmentDescriptionFlags::empty(),
                samples: vk::SAMPLE_COUNT_1_BIT,
                load_op: vk::AttachmentLoadOp::Clear,
                store_op: vk::AttachmentStoreOp::Store,
                stencil_load_op: vk::AttachmentLoadOp::DontCare,
                stencil_store_op: vk::AttachmentStoreOp::DontCare,
                initial_layout: vk::ImageLayout::Undefined,
                final_layout: vk::ImageLayout::PresentSrcKhr,
            },
            vk::AttachmentDescription {
                format: vk::Format::D16Unorm,
                flags: vk::AttachmentDescriptionFlags::empty(),
                samples: vk::SAMPLE_COUNT_1_BIT,
                load_op: vk::AttachmentLoadOp::Clear,
                store_op: vk::AttachmentStoreOp::DontCare,
                stencil_load_op: vk::AttachmentLoadOp::DontCare,
                stencil_store_op: vk::AttachmentStoreOp::DontCare,
                initial_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
                final_layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
            },
        ];
        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::ColorAttachmentOptimal,
        };
        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DepthStencilAttachmentOptimal,
        };
        let dependency = vk::SubpassDependency {
            dependency_flags: Default::default(),
            src_subpass: vk::VK_SUBPASS_EXTERNAL,
            dst_subpass: Default::default(),
            src_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            src_access_mask: Default::default(),
            dst_access_mask: vk::ACCESS_COLOR_ATTACHMENT_READ_BIT | vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dst_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        };
        let subpass = vk::SubpassDescription {
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            flags: Default::default(),
            pipeline_bind_point: vk::PipelineBindPoint::Graphics,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            p_resolve_attachments: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        };
        let renderpass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RenderPassCreateInfo,
            flags: Default::default(),
            p_next: ptr::null(),
            attachment_count: renderpass_attachments.len() as u32,
            p_attachments: renderpass_attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &dependency,
        };
        let renderpass = unsafe {
            base.device
                .vk()
                .create_render_pass(&renderpass_create_info, None)
                .unwrap()
        };
        SimpleColor { renderpass: renderpass }
    }
}

impl RenderPass for SimpleColor {
    fn vk(&self) -> vk::RenderPass {
        self.renderpass
    }

    fn record_commands<F: FnOnce(&AshDevice, vk::CommandBuffer)>(
        &self,
        base: &ExampleBase,
        framebuffer: vk::Framebuffer,
        command_buffer: vk::CommandBuffer,
        subpass_contents: vk::SubpassContents,
        f: F,
    ) {
        let clear_values = [
            vk::ClearValue::new_color(vk::ClearColorValue::new_float32([0.0, 0.0, 0.0, 0.0])),
            vk::ClearValue::new_depth_stencil(vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            }),
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RenderPassBeginInfo,
            p_next: ptr::null(),
            render_pass: self.renderpass,
            framebuffer: framebuffer,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: base.surface_resolution.clone(),
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
        };

        unsafe {
            base.device.vk().cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                subpass_contents,
            );
            f(base.device.vk(), command_buffer);
            base.device.vk().cmd_end_render_pass(command_buffer);
        }
    }
}
