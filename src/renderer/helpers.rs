use super::device::DescriptorSetLayout;
use ash::{version::DeviceV1_0, vk};
use std::{ffi::CString, fs::File, io::Read, path::PathBuf, sync::Arc, u32};

use super::device::Device;

pub(crate) struct SwapchainImage {
    #[allow(unused)]
    pub(crate) handle: vk::Image,
}

pub(crate) struct Framebuffer {
    pub(crate) handle: vk::Framebuffer,
    pub(crate) device: Arc<Device>,
}

pub(crate) struct ImageView {
    pub(crate) handle: vk::ImageView,
    pub(crate) device: Arc<Device>,
}

pub(crate) struct Sampler {
    pub(crate) handle: vk::Sampler,
    device: Arc<Device>,
}

pub(crate) struct PipelineLayout {
    pub(crate) handle: vk::PipelineLayout,
    device: Arc<Device>,
}

pub(crate) struct Pipeline {
    pub(crate) handle: vk::Pipeline,
    device: Arc<Device>,
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_framebuffer(self.handle, None);
        }
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.handle, None);
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_sampler(self.handle, None);
        }
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

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_pipeline(self.handle, None) }
    }
}

pub(crate) fn new_image_view(
    device: Arc<Device>,
    create_info: &vk::ImageViewCreateInfo,
) -> ImageView {
    let handle = unsafe { device.create_image_view(&create_info, None).unwrap() };

    ImageView { handle, device }
}

pub(crate) fn new_sampler(device: Arc<Device>, info: &vk::SamplerCreateInfoBuilder<'_>) -> Sampler {
    let sampler = unsafe {
        device
            .create_sampler(info, None)
            .expect("Failed to create sampler")
    };

    Sampler {
        handle: sampler,
        device,
    }
}

pub(crate) fn new_pipeline_layout(
    device: Arc<Device>,
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
        device,
    }
}

pub(crate) fn new_graphics_pipeline2(
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

pub(crate) fn pick_lod<T>(
    lods: &[T],
    camera_pos: na::Point3<f32>,
    mesh_pos: na::Point3<f32>,
) -> &T {
    let distance_from_camera = (camera_pos - mesh_pos).magnitude();
    // TODO: fine-tune this later
    if distance_from_camera > 10.0 {
        lods.last().expect("empty index buffer LODs")
    } else {
        lods.first().expect("empty index buffer LODs")
    }
}
