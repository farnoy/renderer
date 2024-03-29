use std::{env, fs::File, path::Path};

use ash::vk;
use hashbrown::HashMap;
use serde::{ser::SerializeSeq, Serialize, Serializer};
use serde_json::to_writer_pretty;

use crate::{DescriptorSet, Pipeline, SpecificPipe};

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
    flags: u32,
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
    descriptor_count: u32,
    #[serde(rename = "descriptorType", serialize_with = "ser_descriptor_type")]
    descriptor_type: vk::DescriptorType,
    #[serde(rename = "stageFlags", serialize_with = "ser_shader_stage")]
    stage_flags: vk::ShaderStageFlags,
}

pub(crate) fn dump_rga(
    sets: &HashMap<String, DescriptorSet>,
    pipe: &Pipeline,
    push_const_ty: Option<&spirq::ty::Type>,
) {
    let stage_flags = match pipe.specific {
        SpecificPipe::Graphics(_) => vk::ShaderStageFlags::ALL_GRAPHICS,
        SpecificPipe::Compute => vk::ShaderStageFlags::COMPUTE,
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
                .unwrap_or_default(),
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

    for (set_id, _conditionals) in pipe.descriptor_sets.iter() {
        rga_pipe
            .pipeline_layout_create_info
            .set_layouts
            .push(rga_pipe.pipeline_layout_create_info.set_layout_count);
        rga_pipe.pipeline_layout_create_info.set_layout_count += 1;

        let set_id = syn::parse_str::<syn::Path>(set_id).unwrap();
        let set_id = set_id.segments.last().unwrap().ident.to_string();
        let set = sets.get(&set_id).unwrap();
        rga_pipe.descriptor_set_layouts.push(RgaDescriptorSetLayoutCreateInfo {
            flags: 0,
            binding_count: set.bindings.len(),
            bindings: set
                .bindings
                .iter()
                .enumerate()
                .map(|(ix, binding)| RgaDescriptorBinding {
                    binding: ix as u32,
                    descriptor_count: binding.count,
                    descriptor_type: str_to_descriptor_type(&binding.descriptor_type),
                    stage_flags: binding
                        .shader_stages
                        .iter()
                        .map(|x| str_to_shader_stage(x))
                        .fold(vk::ShaderStageFlags::empty(), |acc, stage| acc | stage),
                })
                .collect(),
            p_next: 0,
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        });

        match &pipe.specific {
            SpecificPipe::Compute => {
                rga_pipe.compute = Some(RgaComputePipeline {
                    base_pipeline_index: -1,
                    flags: 0,
                    p_next: 0,
                    s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
                    stage: RgaComputeStage {
                        flags: if pipe.varying_subgroup_stages.iter().any(|s| s == "COMPUTE") {
                            vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE_EXT.as_raw()
                        } else {
                            vk::PipelineShaderStageCreateFlags::empty().as_raw()
                        },
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
        .join(format!("{}.{}pso", pipe.name, match &pipe.specific {
            SpecificPipe::Graphics(_) => 'g',
            SpecificPipe::Compute => 'c',
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

fn str_to_descriptor_type(id: &str) -> vk::DescriptorType {
    match id {
        "UNIFORM_BUFFER" => vk::DescriptorType::UNIFORM_BUFFER,
        "STORAGE_BUFFER" => vk::DescriptorType::STORAGE_BUFFER,
        "STORAGE_BUFFER_DYNAMIC" => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
        "COMBINED_IMAGE_SAMPLER" => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        "ACCELERATION_STRUCTURE_KHR" => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        "STORAGE_IMAGE" => vk::DescriptorType::STORAGE_IMAGE,
        _ => unimplemented!("ident_to_descriptor_type {}", id),
    }
}

fn str_to_shader_stage(id: &str) -> vk::ShaderStageFlags {
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
