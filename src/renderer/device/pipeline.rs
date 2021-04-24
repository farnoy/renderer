use super::{DescriptorSetLayout, Device};
use ash::{version::DeviceV1_0, vk};
use std::{ffi::CString, ops::Deref};

pub(crate) struct PipelineLayout {
    handle: vk::PipelineLayout,
}

pub(crate) struct Pipeline {
    handle: vk::Pipeline,
}

impl PipelineLayout {
    pub(super) fn new(
        device: &Device,
        descriptor_set_layouts: &[&DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> PipelineLayout {
        let descriptor_set_layout_handles = descriptor_set_layouts.iter().map(|l| l.handle).collect::<Vec<_>>();
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layout_handles)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe { device.device.create_pipeline_layout(&create_info, None).unwrap() };

        PipelineLayout {
            handle: pipeline_layout,
        }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe { device.destroy_pipeline_layout(self.handle, None) }
        self.handle = vk::PipelineLayout::null();
    }
}

impl Deref for PipelineLayout {
    type Target = vk::PipelineLayout;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

#[cfg(debug_assertions)]
impl Drop for PipelineLayout {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::PipelineLayout::null(),
            "PipelineLayout not destroyed before Drop"
        );
    }
}

impl Pipeline {
    pub(crate) fn new_graphics_pipeline(
        device: &Device,
        shaders: &[(vk::ShaderStageFlags, &[u8], Option<&vk::SpecializationInfo>)],
        mut create_info: vk::GraphicsPipelineCreateInfo,
    ) -> Pipeline {
        let shader_modules = shaders
            .iter()
            .map(|&(stage, ref bytes, spec_info)| {
                let (l, aligned, r) = unsafe { bytes.align_to() };
                assert!(l.is_empty() && r.is_empty(), "failed to realign code");
                let shader_info = vk::ShaderModuleCreateInfo::builder().code(&aligned);
                let shader_module = unsafe {
                    device
                        .device
                        .create_shader_module(&shader_info, None)
                        .expect("Vertex shader module error")
                };
                (shader_module, stage, spec_info)
            })
            .collect::<Vec<_>>();
        let shader_entry_name = CString::new("main").unwrap();
        let shader_stage_create_infos = shader_modules
            .iter()
            .map(|&(module, stage, spec_info)| {
                let create_info = vk::PipelineShaderStageCreateInfo::builder()
                    .module(module)
                    .name(&shader_entry_name)
                    .stage(stage);
                match spec_info {
                    Some(spec_info) => create_info.specialization_info(spec_info).build(),
                    None => create_info.build(),
                }
            })
            .collect::<Vec<_>>();
        create_info.stage_count = shader_stage_create_infos.len() as u32;
        create_info.p_stages = shader_stage_create_infos.as_ptr();
        let graphics_pipelines = unsafe {
            device
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .expect("Unable to create graphics pipeline")
        };
        for (shader_module, _stage, _spec_info) in shader_modules {
            unsafe {
                device.device.destroy_shader_module(shader_module, None);
            }
        }

        Pipeline {
            handle: graphics_pipelines[0],
            // device,
        }
    }

    pub(crate) fn new_compute_pipelines(
        device: &Device,
        create_infos: &[vk::ComputePipelineCreateInfoBuilder<'_>],
    ) -> Vec<Pipeline> {
        let infos = create_infos.iter().map(|builder| **builder).collect::<Vec<_>>();
        unsafe {
            device
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), infos.as_slice(), None)
                .expect("Unable to create compute pipelines")
                .into_iter()
                .map(|handle| Pipeline {
                    handle,
                    // device: Arc::clone(&device),
                })
                .collect()
        }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe { device.destroy_pipeline(self.handle, None) }
        self.handle = vk::Pipeline::null();
    }
}

impl Deref for Pipeline {
    type Target = vk::Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

#[cfg(debug_assertions)]
impl Drop for Pipeline {
    fn drop(&mut self) {
        debug_assert_eq!(self.handle, vk::Pipeline::null(), "Pipeline not destroyed before Drop");
    }
}

/*
#[derive(PartialEq, Eq)]
pub(crate) enum Viewport {
    Dynamic,
    Static(usize, usize),
}

pub(crate) struct StaticPipeline<const VIEWPORT: Viewport> {
    inner: Pipeline,
}

impl<const VIEWPORT: Viewport> StaticPipeline<VIEWPORT> {
    pub(crate) fn bind(device: &Device, command_buffer: vk::CommandBuffer) {
        match VIEWPORT {
            Viewport::Dynamic => {
                unsafe {
                    device.cmd_set_viewport(command_buffer, 0, &[]);
                }
            },
            Viewport::Static(w, h) => {
            }
        }
    }
}
*/
