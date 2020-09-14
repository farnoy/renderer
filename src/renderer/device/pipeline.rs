use super::{DescriptorSetLayout, Device};
use ash::{version::DeviceV1_0, vk};
use std::{ffi::CString, fs::File, io::Read, ops::Deref, path::PathBuf, sync::Arc};

pub(crate) struct PipelineLayout {
    handle: vk::PipelineLayout,
    device: Arc<Device>,
}

pub(crate) struct Pipeline {
    handle: vk::Pipeline,
    device: Arc<Device>,
}

impl PipelineLayout {
    pub(crate) fn new(
        device: &Arc<Device>,
        descriptor_set_layouts: &[&DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> PipelineLayout {
        let descriptor_set_layout_handles = descriptor_set_layouts
            .iter()
            .map(|l| l.handle)
            .collect::<Vec<_>>();
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layout_handles)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe {
            device
                .device
                .create_pipeline_layout(&create_info, None)
                .unwrap()
        };

        PipelineLayout {
            handle: pipeline_layout,
            device: Arc::clone(device),
        }
    }
}

impl Deref for PipelineLayout {
    type Target = vk::PipelineLayout;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_pipeline_layout(self.handle, None)
        }
    }
}

impl Pipeline {
    pub(crate) fn new_graphics_pipeline(
        device: Arc<Device>,
        shaders: &[(vk::ShaderStageFlags, PathBuf)],
        mut create_info: vk::GraphicsPipelineCreateInfo,
    ) -> Pipeline {
        let shader_modules = shaders
            .iter()
            .map(|&(stage, ref path)| {
                let file = File::open(path).expect("Could not find shader.");
                let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
                let (l, aligned, r) = unsafe { bytes.as_slice().align_to() };
                assert!(l.is_empty() && r.is_empty(), "failed to realign code");
                let shader_info = vk::ShaderModuleCreateInfo::builder().code(&aligned);
                let shader_module = unsafe {
                    device
                        .device
                        .create_shader_module(&shader_info, None)
                        .expect("Vertex shader module error")
                };
                (shader_module, stage)
            })
            .collect::<Vec<_>>();
        let shader_entry_name = CString::new("main").unwrap();
        let shader_stage_create_infos = shader_modules
            .iter()
            .map(|&(module, stage)| {
                vk::PipelineShaderStageCreateInfo::builder()
                    .module(module)
                    .name(&shader_entry_name)
                    .stage(stage)
                    .build()
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
        for (shader_module, _stage) in shader_modules {
            unsafe {
                device.device.destroy_shader_module(shader_module, None);
            }
        }

        Pipeline {
            handle: graphics_pipelines[0],
            device,
        }
    }

    pub(crate) fn new_compute_pipelines(
        device: Arc<Device>,
        create_infos: &[vk::ComputePipelineCreateInfoBuilder<'_>],
    ) -> Vec<Pipeline> {
        let infos = create_infos
            .iter()
            .map(|builder| **builder)
            .collect::<Vec<_>>();
        unsafe {
            device
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), infos.as_slice(), None)
                .expect("Unable to create compute pipelines")
                .into_iter()
                .map(|handle| Pipeline {
                    handle,
                    device: Arc::clone(&device),
                })
                .collect()
        }
    }
}

impl Deref for Pipeline {
    type Target = vk::Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_pipeline(self.handle, None) }
    }
}
