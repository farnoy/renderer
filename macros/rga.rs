use std::{env, fs::File, path::Path};

use ash::vk;
use serde::{ser::SerializeSeq, Serialize, Serializer};
use serde_json::to_writer_pretty;
use syn::Ident;

use crate::{DescriptorSet, Pipe, SpecificPipe};

#[derive(Serialize)]
pub(crate) struct RgaPipeline {
    #[serde(rename = "VkPipelineLayoutCreateInfo")]
    pipeline_layout_create_info: RgaPipelineLayoutCreateInfo,
    #[serde(rename = "VkDescriptorSetLayoutCreateInfo")]
    descriptor_set_layouts: Vec<RgaDescriptorSetLayoutCreateInfo>,
    version: u8,
    #[serde(rename = "VkComputePipelineCreateInfo", skip_serializing_if = "Option::is_none")]
    compute: Option<RgaComputePipeline>,
}

#[derive(Serialize)]
pub(crate) struct RgaComputePipeline {
    #[serde(rename = "basePipelineIndex")]
    base_pipeline_index: i32,
    flags: u8,
    #[serde(rename = "pNext", serialize_with = "hex")]
    p_next: usize,
    #[serde(rename = "sType", serialize_with = "ser_structure_type")]
    s_type: vk::StructureType,
    stage: RgaComputeStage,
}

#[derive(Serialize)]
pub(crate) struct RgaComputeStage {
    flags: u8,
    #[serde(rename = "pNext", serialize_with = "hex")]
    p_next: usize,
    #[serde(rename = "module", serialize_with = "hex")]
    module: usize,
    #[serde(rename = "sType", serialize_with = "ser_structure_type")]
    s_type: vk::StructureType,
    #[serde(serialize_with = "ser_shader_stage")]
    stage: vk::ShaderStageFlags,
}

#[derive(Serialize)]
pub(crate) struct RgaPipelineLayoutCreateInfo {
    #[serde(rename = "pNext", serialize_with = "hex")]
    p_next: usize,
    #[serde(rename = "pushConstantRangeCount")]
    push_constant_range_count: usize,
    #[serde(rename = "pPushConstantRanges")]
    push_constant_ranges: Vec<RgaPushConstantRange>,
    #[serde(rename = "sType", serialize_with = "ser_structure_type")]
    s_type: vk::StructureType,
    flags: u8,
    #[serde(rename = "setLayoutCount")]
    set_layout_count: usize,
    #[serde(rename = "pSetLayouts", serialize_with = "hex_vec")]
    set_layouts: Vec<usize>,
}

#[derive(Serialize)]
pub(crate) struct RgaPushConstantRange {
    offset: usize,
    size: usize,
    #[serde(rename = "stageFlags", serialize_with = "ser_shader_stage")]
    stage_flags: vk::ShaderStageFlags,
}

#[derive(Serialize)]
pub(crate) struct RgaDescriptorSetLayoutCreateInfo {
    flags: u8,
    #[serde(rename = "bindingCount")]
    binding_count: usize,
    #[serde(rename = "pBindings")]
    bindings: Vec<RgaDescriptorBinding>,
    #[serde(rename = "pNext", serialize_with = "hex")]
    p_next: usize,
    #[serde(rename = "sType", serialize_with = "ser_structure_type")]
    s_type: vk::StructureType,
}

#[derive(Serialize)]
pub(crate) struct RgaDescriptorBinding {
    binding: u32,
    #[serde(rename = "descriptorCount")]
    descriptor_count: usize,
    #[serde(rename = "descriptorType", serialize_with = "ser_descriptor_type")]
    descriptor_type: vk::DescriptorType,
    #[serde(rename = "stageFlags", serialize_with = "ser_shader_stage")]
    stage_flags: vk::ShaderStageFlags,
}

pub(crate) fn dump_rga(sets: &[DescriptorSet], pipe: &Pipe, push_const_ty: Option<&spirq::Type>) {
    let stage_flags = match pipe.specific {
        SpecificPipe::Graphics(_) => vk::ShaderStageFlags::ALL_GRAPHICS,
        SpecificPipe::Compute(_) => vk::ShaderStageFlags::COMPUTE,
    };
    let mut rga_pipe = RgaPipeline {
        pipeline_layout_create_info: RgaPipelineLayoutCreateInfo {
            push_constant_range_count: push_const_ty.map(|_| 1).unwrap_or(0),
            push_constant_ranges: push_const_ty
                .map(|ty| {
                    vec![RgaPushConstantRange {
                        offset: 0,
                        size: ty.nbyte().unwrap(),
                        stage_flags,
                    }]
                })
                .unwrap_or(vec![]),
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 0,
            set_layouts: vec![],
            p_next: 0,
            flags: 0,
        },
        descriptor_set_layouts: vec![],
        compute: None,
        version: 3,
    };

    for set_id in pipe.descriptors.iter() {
        rga_pipe
            .pipeline_layout_create_info
            .set_layouts
            .push(rga_pipe.pipeline_layout_create_info.set_layout_count);
        rga_pipe.pipeline_layout_create_info.set_layout_count += 1;

        let set = sets
            .iter()
            .find(|candidate| candidate.name == *set_id.get_ident().unwrap())
            .unwrap();
        rga_pipe.descriptor_set_layouts.push(RgaDescriptorSetLayoutCreateInfo {
            flags: 0,
            binding_count: set.bindings.len(),
            bindings: set
                .bindings
                .iter()
                .enumerate()
                .map(|(ix, binding)| RgaDescriptorBinding {
                    binding: ix as u32,
                    descriptor_count: binding.count.base10_parse().unwrap(),
                    descriptor_type: ident_to_descriptor_type(&binding.descriptor_type),
                    stage_flags: binding
                        .stages
                        .iter()
                        .map(ident_to_shader_stage)
                        .fold(vk::ShaderStageFlags::empty(), |acc, stage| acc | stage),
                })
                .collect(),
            p_next: 0,
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        });

        match &pipe.specific {
            SpecificPipe::Compute(_) => {
                rga_pipe.compute = Some(RgaComputePipeline {
                    base_pipeline_index: -1,
                    flags: 0,
                    p_next: 0,
                    s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
                    stage: RgaComputeStage {
                        flags: 0,
                        p_next: 0,
                        module: 0,
                        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                        stage: vk::ShaderStageFlags::COMPUTE,
                    },
                });
            }
            SpecificPipe::Graphics(_) => {
                // TODO
            }
        }
    }

    let rga_path = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("rga")
        .join(format!("{}.{}pso", pipe.name.to_string(), match &pipe.specific {
            SpecificPipe::Graphics(_) => 'g',
            SpecificPipe::Compute(_) => 'c',
        }));

    let file = File::create(rga_path).unwrap();
    to_writer_pretty(file, &rga_pipe).unwrap();
}

fn hex<S>(num: &usize, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let formatted = format!("{:#x}", num);
    serializer.serialize_str(&formatted)
}

fn hex_vec<S>(nums: &Vec<usize>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(nums.len()))?;
    for num in nums {
        let formatted = format!("{:#x}", num);
        seq.serialize_element(&formatted)?;
    }
    seq.end()
}

fn ser_structure_type<S>(s_type: &vk::StructureType, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_i32(s_type.as_raw())
}

fn ser_descriptor_type<S>(desc: &vk::DescriptorType, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_i32(desc.as_raw())
}

fn ser_shader_stage<S>(shader_stage: &vk::ShaderStageFlags, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_u32(shader_stage.as_raw())
}

fn ident_to_descriptor_type(id: &Ident) -> vk::DescriptorType {
    if id == "UNIFORM_BUFFER" {
        vk::DescriptorType::UNIFORM_BUFFER
    } else if id == "STORAGE_BUFFER" {
        vk::DescriptorType::STORAGE_BUFFER
    } else if id == "STORAGE_BUFFER_DYNAMIC" {
        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
    } else if id == "COMBINED_IMAGE_SAMPLER" {
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
    } else if id == "ACCELERATION_STRUCTURE_KHR" {
        vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
    } else {
        unimplemented!("ident_to_descriptor_type {}", id)
    }
}

fn ident_to_shader_stage(id: &Ident) -> vk::ShaderStageFlags {
    if id == "VERTEX" {
        vk::ShaderStageFlags::VERTEX
    } else if id == "FRAGMENT" {
        vk::ShaderStageFlags::FRAGMENT
    } else if id == "COMPUTE" {
        vk::ShaderStageFlags::COMPUTE
    } else {
        unimplemented!("ident_to_shader_stage {}", id)
    }
}
