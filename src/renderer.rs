#![allow(clippy::too_many_arguments)]

// TODO: pub(crate) should disappear?
pub(crate) mod device;
mod entry;
mod gltf_mesh_io;
mod helpers;
mod instance;
mod swapchain;
mod systems;
#[cfg(feature = "shader_reload")]
use std::time::Instant;
use std::{
    cmp::max,
    env,
    fs::{read_dir, remove_file, File},
    hint::unreachable_unchecked,
    io::Write,
    marker::PhantomData,
    mem::{replace, size_of, take},
    path::Path,
    sync::Arc,
};

use ash::vk;
use bevy_ecs::{component::Component, prelude::*};
use bevy_system_graph::*;
use hashbrown::HashMap;
use itertools::Itertools;
use lazy_static::lazy_static;
use num_traits::ToPrimitive;
use parking_lot::{Mutex, MutexGuard};
use petgraph::{
    algo::has_path_connecting,
    prelude::*,
    stable_graph::{NodeIndex, StableDiGraph},
    visit::{IntoEdgeReferences, IntoNodeReferences},
};
use profiling::scope;
use renderer_macro_lib::{
    inputs::DependencyType,
    resource_claims::{ResourceClaim, ResourceClaims, ResourceDefinitionType, ResourceUsageKind},
    Conditional, QueueFamily,
};
#[cfg(feature = "shader_reload")]
use smallvec::smallvec;
use smallvec::SmallVec;
use static_assertions::const_assert_eq;

#[cfg(feature = "crash_debugging")]
pub(crate) use self::systems::crash_debugging::CrashBuffer;
#[cfg(feature = "shader_reload")]
pub(crate) use self::systems::shader_reload::{reload_shaders, ReloadedShaders, ShaderReload};
use self::{
    device::{
        Buffer, DescriptorPool, Device, DoubleBuffered, Image, ImageView, Sampler, Shader, StaticBuffer,
        StrictCommandPool, StrictRecordingCommandBuffer, VmaMemoryUsage,
    },
    helpers::command_util::CommandUtil,
    systems::{
        cull_pipeline::cull_set, depth_pass::depth_only_pass, reference_raytracer::reference_raytrace,
        shadow_mapping::shadow_map_set,
    },
};
pub(crate) use self::{
    gltf_mesh_io::{load as load_gltf, LoadedMesh},
    helpers::pick_lod,
    instance::Instance,
    swapchain::{Surface, Swapchain},
    systems::{
        acceleration_strucures::{
            build_acceleration_structures, AccelerationStructures, AccelerationStructuresInternal,
        },
        consolidate_mesh_buffers::{consolidate_mesh_buffers, ConsolidatedMeshBuffers},
        cull_pipeline::{
            coarse_culling, cull_pass, cull_pass_bypass, CoarseCulled, CullPassData, CullPassDataPrivate,
            TransferCullPrivate, INITIAL_WORKGROUP_SIZE,
        },
        debug_aabb_renderer::DebugAABBPassData,
        depth_pass::DepthPassData,
        present::{acquire_framebuffer, ImageIndex, PresentData, PresentFramebuffer},
        reference_raytracer::{ReferenceRTData, ReferenceRTDataPrivate},
        scene_loader::{
            initiate_scene_loader, traverse_and_decode_scenes, upload_loaded_meshes,
            LoadedMesh as SceneLoaderLoadedMesh, ScenesToLoad, UploadMeshesData,
        },
        shadow_mapping::{
            prepare_shadow_maps, shadow_mapping_mvp_calculation, update_shadow_map_descriptors, ShadowMappingData,
            ShadowMappingDataInternal, ShadowMappingLightMatrices,
        },
        textures::{
            cleanup_base_color_markers, recreate_base_color_descriptor_set, synchronize_base_color_textures_visit,
            update_base_color_descriptors, BaseColorDescriptorSet, BaseColorVisitedMarker, GltfMeshBaseColorTexture,
            GltfMeshNormalTexture, NormalMapVisitedMarker,
        },
    },
};
use crate::ecs::{
    components::{ModelMatrix, AABB},
    resources::Camera,
    systems::*,
};

pub(crate) fn up_vector() -> na::Unit<na::Vector3<f32>> {
    na::Unit::new_unchecked(na::Vector3::y())
}
pub(crate) fn forward_vector() -> na::Unit<na::Vector3<f32>> {
    na::Unit::new_unchecked(na::Vector3::z())
}
pub(crate) fn right_vector() -> na::Unit<na::Vector3<f32>> {
    na::Unit::new_unchecked(na::Vector3::x())
}

#[derive(Clone, Component)]
pub(crate) struct GltfMesh {
    pub(crate) vertex_buffer: Arc<Buffer>,
    pub(crate) normal_buffer: Arc<Buffer>,
    pub(crate) uv_buffer: Arc<Buffer>,
    pub(crate) tangent_buffer: Arc<Buffer>,
    pub(crate) index_buffers: Arc<Vec<(Buffer, u64)>>,
    pub(crate) vertex_len: u64,
    pub(crate) aabb: ncollide3d::bounding_volume::AABB<f32>,
}

impl GltfMesh {
    pub(crate) fn destroy(self, device: &Device) {
        Arc::try_unwrap(self.vertex_buffer)
            .into_iter()
            .for_each(|b| b.destroy(device));
        Arc::try_unwrap(self.normal_buffer)
            .into_iter()
            .for_each(|b| b.destroy(device));
        Arc::try_unwrap(self.tangent_buffer)
            .into_iter()
            .for_each(|b| b.destroy(device));
        Arc::try_unwrap(self.uv_buffer)
            .into_iter()
            .for_each(|b| b.destroy(device));
        Arc::try_unwrap(self.index_buffers)
            .into_iter()
            .for_each(|bs| bs.into_iter().for_each(|(b, _)| b.destroy(device)));
    }
}

#[derive(Debug, Default, Component)]
pub(crate) struct DrawIndex(pub(crate) u32);

// TODO: rename
pub(crate) struct RenderFrame {
    pub(crate) instance: Arc<Instance>,
    pub(crate) device: Device,
    pub(crate) auto_semaphores: AutoSemaphores,
    pub(crate) frame_number: u64,
    pub(crate) buffer_count: usize,
}

#[derive(Clone, Component)]
pub(crate) struct SwapchainIndexToFrameNumber {
    pub(crate) map: DoubleBuffered<u64>,
}

impl FromWorld for SwapchainIndexToFrameNumber {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        SwapchainIndexToFrameNumber {
            map: renderer.new_buffered(|_| 0),
        }
    }
}

pub(crate) type UVBuffer = [[f32; 2]; size_of::<frame_graph::VertexBuffer>() / size_of::<[f32; 3]>()];
pub(crate) type TangentBuffer = [[f32; 4]; size_of::<frame_graph::VertexBuffer>() / size_of::<[f32; 3]>()];

// sanity checks
const_assert_eq!(
    size_of::<frame_graph::VertexBuffer>() / size_of::<[f32; 3]>(),
    10 * 300_000
);
const_assert_eq!(
    size_of::<frame_graph::VkDrawIndexedIndirectCommand>(),
    size_of::<vk::DrawIndexedIndirectCommand>(),
);

// https://users.rust-lang.org/t/can-i-conveniently-compile-bytes-into-a-rust-program-with-a-specific-alignment/24049/2
// #[repr(C)] // guarantee 'bytes' comes after '_align'
// struct AlignedAs<Align, Bytes: ?Sized> {
//     _align: [Align; 0],
//     bytes: Bytes,
// }

// macro_rules! include_bytes_align_as {
//     ($align_ty:ty, $path:literal) => {{
//         // const block expression to encapsulate the static
//         use crate::renderer::AlignedAs;

//         // this assignment is made possible by CoerceUnsized
//         static ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
//             _align: [],
//             bytes: *include_bytes!($path),
//         };

//         &ALIGNED.bytes
//     }};
// }

// pub(crate) use include_bytes_align_as;

lazy_static! {
    pub(crate) static ref RENDERER_INPUT: renderer_macro_lib::RendererInput =
        bincode::deserialize(frame_graph::CROSSBAR).unwrap();
}

renderer_macros::define_timelines! {}

renderer_macros::define_frame!(frame_graph);

renderer_macros::define_renderpass! {
    MainRP {
        color [ Color COLOR_ATTACHMENT_OPTIMAL clear => store ]
        depth_stencil { DepthRT DEPTH_STENCIL_READ_ONLY_OPTIMAL load => discard }
    }
}

// purely virtual, just to have a synchronization point at the start of the frame
renderer_macros::define_pass!(PresentationAcquire on graphics);

renderer_macros::define_pass!(Main on graphics);

renderer_macros::define_set!(
    model_set {
        model STORAGE_BUFFER from [VERTEX, COMPUTE]
    }
);

renderer_macros::define_set! {
    camera_set {
        matrices UNIFORM_BUFFER from [VERTEX, FRAGMENT, COMPUTE]
    }
}
renderer_macros::define_set! {
    textures_set {
        base_color 3072 of COMBINED_IMAGE_SAMPLER partially bound update after bind from [FRAGMENT],
        normal_map 3072 of COMBINED_IMAGE_SAMPLER partially bound update after bind from [FRAGMENT]
    }
}
renderer_macros::define_set! {
    acceleration_set {
        top_level_as ACCELERATION_STRUCTURE_KHR update after bind from [FRAGMENT],
        random_seed UNIFORM_BUFFER from [FRAGMENT]
    }
}
renderer_macros::define_set! {
    imgui_set {
        texture COMBINED_IMAGE_SAMPLER from [FRAGMENT]
    }
}

renderer_macros::define_pipe!(
    gltf_mesh {
        descriptors [model_set, camera_set, shadow_map_set, textures_set, acceleration_set if [RT]]
        specialization_constants [
            10 => shadow_map_dim: u32,
            11 => shadow_map_dim_squared: u32,
        ]
        graphics
        dynamic renderpass MainRP
        samples 4
        vertex_inputs [position: vec3, normal: vec3, uv: vec2, tangent: vec4]
        stages [VERTEX, FRAGMENT]
        cull mode BACK
        depth test true
        depth compare op EQUAL
    }
);
renderer_macros::define_pipe! {
    debug_aabb {
        descriptors [camera_set]
        graphics
        dynamic renderpass MainRP
        samples 4
        stages [VERTEX, FRAGMENT]
        polygon mode LINE
    }
}

renderer_macros::define_renderpass! {
    ImguiRP {
        color [ Color COLOR_ATTACHMENT_OPTIMAL load => store ]
    }
}

renderer_macros::define_pipe! {
    imgui_pipe {
        descriptors [imgui_set]
        graphics
        dynamic renderpass ImguiRP
        samples 4
        vertex_inputs [pos: vec2, uv: vec2, col: vec4]
        stages [VERTEX, FRAGMENT]
    }
}

pub(crate) trait TimelineStage {
    const OFFSET: u64;
    const CYCLE: u64;
}

fn as_of<T: TimelineStage>(frame_number: u64) -> u64 {
    frame_number * T::CYCLE + T::OFFSET
}

fn as_of_last<T: TimelineStage>(frame_number: u64) -> u64 {
    as_of::<T>(frame_number - 1)
}

fn as_of_previous<T: TimelineStage>(image_index: &ImageIndex, indices: &SwapchainIndexToFrameNumber) -> u64 {
    let frame_number = indices.map[image_index.0 as usize];
    as_of::<T>(frame_number)
}

pub(crate) trait RenderStage {
    type SignalTimelineStage: TimelineStage;
    const SIGNAL_AUTO_SEMAPHORE_IX: usize;

    fn wait_previous(
        renderer: &RenderFrame,
        image_index: &ImageIndex,
        swapchain_index_map: &SwapchainIndexToFrameNumber,
    ) {
        renderer.auto_semaphores.0[Self::SIGNAL_AUTO_SEMAPHORE_IX]
            .wait(
                &renderer.device,
                as_of_previous::<Self::SignalTimelineStage>(image_index, swapchain_index_map),
            )
            .unwrap();
    }
}

pub(crate) trait DescriptorSetLayout {
    const DEBUG_NAME: &'static str;
    fn binding_flags() -> SmallVec<[vk::DescriptorBindingFlags; 8]>;
    fn binding_layout() -> SmallVec<[vk::DescriptorSetLayoutBinding; 8]>;
}

pub(crate) trait PipelineLayout {
    type SmartDescriptorSetLayouts: RefTuple;
    type SmartDescriptorSets: RefTuple;
    type PushConstants;

    const IS_GRAPHICS: bool;
    const DEBUG_NAME: &'static str;

    #[allow(clippy::new_ret_no_self)]
    fn new(device: &Device, sets: <Self::SmartDescriptorSetLayouts as RefTuple>::Ref<'_>) -> device::PipelineLayout;

    fn bind_descriptor_sets(
        layout: vk::PipelineLayout,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        sets: <Self::SmartDescriptorSets as RefTuple>::Ref<'_>,
    );
}

pub(crate) trait Pipeline: Sized {
    type Layout: PipelineLayout;
    type DynamicArguments;
    type Specialization: PipelineSpecialization;

    const NAME: &'static str;

    fn default_shader_stages(switches: &HashMap<String, bool>) -> SmallVec<[Vec<u8>; 4]> {
        let input = &RENDERER_INPUT;

        let pipe = input.pipelines.get(Self::NAME).unwrap();

        let mut output = SmallVec::new();

        let conditionals = pipe
            .unique_conditionals()
            .into_iter()
            .filter(|&d| switches.get(d).cloned().unwrap_or(false))
            .collect_vec()
            .join(",");

        for shader_stage in pipe.specific.stages() {
            use std::io::Read;

            let file_extension = match shader_stage.as_str() {
                "VERTEX" => "vert",
                "COMPUTE" => "comp",
                "FRAGMENT" => "frag",
                _ => unimplemented!("Unknown shader stage"),
            };

            let shader_path =
                Path::new(&env!("OUT_DIR")).join(format!("{}.{}.[{}].spv", Self::NAME, file_extension, conditionals));
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .open(&shader_path)
                .expect("Failed to find shader file");
            let mut buf = vec![];
            f.read_to_end(&mut buf).unwrap();
            output.push(buf);
        }

        output
    }

    fn shader_stages() -> SmallVec<[vk::ShaderStageFlags; 4]> {
        let input = &RENDERER_INPUT;

        let pipe = input.pipelines.get(Self::NAME).unwrap();

        pipe.specific
            .stages()
            .into_iter()
            .map(|shader_stage| match shader_stage.as_str() {
                "VERTEX" => vk::ShaderStageFlags::VERTEX,
                "COMPUTE" => vk::ShaderStageFlags::COMPUTE,
                "FRAGMENT" => vk::ShaderStageFlags::FRAGMENT,
                _ => unimplemented!("Unknown shader stage"),
            })
            .collect()
    }

    fn varying_subgroup_stages() -> SmallVec<[bool; 4]> {
        let input = &RENDERER_INPUT;

        let pipe = input.pipelines.get(Self::NAME).unwrap();

        pipe.specific
            .stages()
            .iter()
            .map(|shader_stage| {
                pipe.varying_subgroup_stages
                    .iter()
                    .any(|candidate| candidate == shader_stage)
            })
            .collect()
    }

    fn new_raw(
        device: &Device,
        layout: vk::PipelineLayout,
        stages: &[vk::PipelineShaderStageCreateInfo],
        flags: vk::PipelineCreateFlags,
        base_handle: vk::Pipeline,
        dynamic_arguments: Self::DynamicArguments,
    ) -> device::Pipeline;
}

fn new_pipe_generic<P: Pipeline>(
    device: &Device,
    layout: &SmartPipelineLayout<P::Layout>,
    specialization: &P::Specialization,
    base_pipeline_handle: Option<vk::Pipeline>,
    shaders: Option<SmallVec<[Shader; 4]>>,
    dynamic_arguments: P::DynamicArguments,
    switches: &HashMap<String, bool>,
) -> (SmallVec<[Shader; 4]>, device::Pipeline) {
    use std::ffi::CStr;
    let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
    let shaders: SmallVec<[Shader; 4]> = shaders.unwrap_or_else(|| {
        P::default_shader_stages(switches)
            .iter()
            .map(|x| device.new_shader(x))
            .collect()
    });
    let stages = P::shader_stages();
    let spec_info = specialization.get_spec_info();
    let mut flags = vk::PipelineCreateFlags::ALLOW_DERIVATIVES;
    if base_pipeline_handle.is_some() {
        flags |= vk::PipelineCreateFlags::DERIVATIVE;
    }

    let stages = shaders
        .iter()
        .zip(stages.iter())
        .zip(P::varying_subgroup_stages().iter())
        .map(|((shader, stage), &varying_subgroup)| {
            vk::PipelineShaderStageCreateInfo::builder()
                .module(shader.vk())
                .name(shader_entry_name)
                .stage(*stage)
                .specialization_info(&spec_info)
                .flags(if varying_subgroup {
                    vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE_EXT
                } else {
                    vk::PipelineShaderStageCreateFlags::empty()
                })
                .build()
        })
        .collect::<SmallVec<[_; 4]>>();

    let base_pipeline_handle = base_pipeline_handle.unwrap_or_else(vk::Pipeline::null);

    (
        shaders,
        P::new_raw(
            device,
            layout.vk(),
            &stages,
            flags,
            base_pipeline_handle,
            dynamic_arguments,
        ),
    )
}

pub(crate) struct SmartPipelineLayout<L: PipelineLayout> {
    pub(crate) layout: device::PipelineLayout,
    l: PhantomData<L>,
}

impl<L: PipelineLayout> SmartPipelineLayout<L> {
    pub(crate) fn new(
        device: &Device,
        sets: <<L as PipelineLayout>::SmartDescriptorSetLayouts as RefTuple>::Ref<'_>,
    ) -> Self {
        let layout = L::new(device, sets);
        SmartPipelineLayout { layout, l: PhantomData }
    }

    pub(crate) fn bind_descriptor_sets(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        sets: <L::SmartDescriptorSets as RefTuple>::Ref<'_>,
    ) {
        L::bind_descriptor_sets(*self.layout, device, command_buffer, sets);
    }

    pub(crate) fn push_constants(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        push_constants: &L::PushConstants,
    ) {
        unsafe {
            let casted: &[u8] =
                std::slice::from_raw_parts(push_constants as *const _ as *const u8, size_of::<L::PushConstants>());
            device.cmd_push_constants(
                command_buffer,
                *self.layout,
                // TODO: imprecise and wasting scalar registers, probably
                if L::IS_GRAPHICS {
                    vk::ShaderStageFlags::ALL_GRAPHICS
                } else {
                    vk::ShaderStageFlags::COMPUTE
                },
                0,
                casted,
            );
        }
    }

    pub(crate) fn vk(&self) -> vk::PipelineLayout {
        *self.layout
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.layout.destroy(device);
    }
}

/// Pipeline wrapper that caches the shader modules to allow for fast re-specialization or shader
/// reloading in development
pub(crate) struct SmartPipeline<P: Pipeline> {
    inner: device::Pipeline,
    specialization: P::Specialization,
    shaders: SmallVec<[Shader; 4]>,
    switches: HashMap<String, bool>,
    #[cfg(feature = "shader_reload")]
    last_updates: SmallVec<[Instant; 4]>,
}

pub(crate) trait PipelineSpecialization: PartialEq + Clone {
    fn get_spec_info(&self) -> vk::SpecializationInfo;
}

impl<P: Pipeline> SmartPipeline<P> {
    pub(crate) fn new(
        device: &Device,
        layout: &SmartPipelineLayout<P::Layout>,
        specialization: P::Specialization,
        dynamic_arguments: P::DynamicArguments,
    ) -> Self {
        scope!("pipelines::new");
        Self::new_internal(device, layout, specialization, None, dynamic_arguments, &HashMap::new())
    }

    #[allow(unused)]
    pub(crate) fn new_switched(
        device: &Device,
        layout: &SmartPipelineLayout<P::Layout>,
        specialization: P::Specialization,
        dynamic_arguments: P::DynamicArguments,
        switches: &HashMap<String, bool>,
    ) -> Self {
        scope!("pipelines::new");
        Self::new_internal(device, layout, specialization, None, dynamic_arguments, switches)
    }

    fn new_internal(
        device: &Device,
        layout: &SmartPipelineLayout<P::Layout>,
        specialization: P::Specialization,
        mut base_pipeline: Option<&mut Self>,
        dynamic_arguments: P::DynamicArguments,
        switches: &HashMap<String, bool>,
    ) -> Self {
        #[allow(clippy::needless_option_as_deref)]
        let base_pipeline_handle = base_pipeline.as_ref().map(|pipe| pipe.inner.vk());
        #[allow(clippy::needless_option_as_deref)]
        let shaders = base_pipeline.as_deref_mut().map(|p| take(&mut p.shaders));
        #[cfg(feature = "shader_reload")]
        #[allow(clippy::needless_option_as_deref)]
        let base_last_updates = base_pipeline.as_deref_mut().map(|p| take(&mut p.last_updates));

        let (shaders, pipe) = new_pipe_generic::<P>(
            device,
            layout,
            &specialization,
            base_pipeline_handle,
            shaders,
            dynamic_arguments,
            switches,
        );

        #[cfg(feature = "shader_reload")]
        let last_updates;

        #[cfg(feature = "shader_reload")]
        if let Some(base_last_updates) = base_last_updates {
            last_updates = base_last_updates
        } else {
            let stage_count = P::default_shader_stages(switches).len();
            last_updates = smallvec![Instant::now(); stage_count];
        }

        SmartPipeline {
            inner: pipe,
            specialization,
            shaders,
            switches: switches.clone(),
            #[cfg(feature = "shader_reload")]
            last_updates,
        }
    }

    /// Re-specializes the pipeline if needed and returns the old one that might still be in use.
    pub(crate) fn specialize(
        &mut self,
        device: &Device,
        layout: &SmartPipelineLayout<P::Layout>,
        new_spec: &P::Specialization,
        dynamic_arguments: P::DynamicArguments,
        #[cfg(feature = "shader_reload")] reloaded_shaders: &ReloadedShaders,
    ) -> Option<Self> {
        self.specialize_switched(
            device,
            layout,
            new_spec,
            dynamic_arguments,
            &HashMap::new(),
            #[cfg(feature = "shader_reload")]
            reloaded_shaders,
        )
    }

    /// Re-specializes the pipeline if needed and returns the old one that might still be in use.
    pub(crate) fn specialize_switched(
        &mut self,
        device: &Device,
        layout: &SmartPipelineLayout<P::Layout>,
        new_spec: &P::Specialization,
        dynamic_arguments: P::DynamicArguments,
        switches: &HashMap<String, bool>,
        #[cfg(feature = "shader_reload")] reloaded_shaders: &ReloadedShaders,
    ) -> Option<Self> {
        scope!("pipelines::specialize");
        use std::mem::swap;

        #[cfg(feature = "shader_reload")]
        let new_shaders: SmallVec<[(Instant, Option<Shader>); 4]> = self
            .shaders
            .iter()
            .zip(
                P::shader_stage_paths()
                    .iter()
                    .zip(P::default_shader_stages(switches).iter()),
            )
            .enumerate()
            .map(
                |(stage_ix, (_shader, (&path, default_code)))| match reloaded_shaders.0.get(path) {
                    Some((ts, code)) if *ts > self.last_updates[stage_ix] => {
                        let static_spv = spirq::SpirvBinary::from(default_code.as_slice());
                        let static_entry = static_spv
                            .reflect_vec()
                            .unwrap()
                            .into_iter()
                            .find(|entry| entry.name == "main")
                            .unwrap();
                        let mut static_descs = static_entry.descs().collect::<Vec<_>>();
                        static_descs.sort_by_key(|d| d.desc_bind);
                        let mut static_inputs = static_entry.inputs().collect::<Vec<_>>();
                        static_inputs.sort_by_key(|d| d.location);
                        let mut static_outputs = static_entry.outputs().collect::<Vec<_>>();
                        static_outputs.sort_by_key(|d| d.location);
                        let mut static_spec_consts = static_entry.spec.spec_consts().collect::<Vec<_>>();
                        static_spec_consts.sort_by_key(|d| d.spec_id);

                        let new_spv = spirq::SpirvBinary::from(code.as_slice());
                        if let Ok(entry_points) = new_spv.reflect_vec() {
                            if let Some(entry) = entry_points.into_iter().find(|entry| entry.name == "main") {
                                let mut entry_descs = entry.descs().collect::<Vec<_>>();
                                entry_descs.sort_by_key(|d| d.desc_bind);
                                let mut entry_inputs = entry.inputs().collect::<Vec<_>>();
                                entry_inputs.sort_by_key(|d| d.location);
                                let mut entry_outputs = entry.outputs().collect::<Vec<_>>();
                                entry_outputs.sort_by_key(|d| d.location);
                                let mut entry_spec_consts = entry.spec.spec_consts().collect::<Vec<_>>();
                                entry_spec_consts.sort_by_key(|d| d.spec_id);
                                if static_entry.exec_model == entry.exec_model
                                    && static_descs == entry_descs
                                    && static_inputs == entry_inputs
                                    && static_outputs == entry_outputs
                                    && static_spec_consts == entry_spec_consts
                                {
                                    (*ts, Some(device.new_shader(code)))
                                } else {
                                    eprintln!(
                                        "Failed to validate live reloaded shader interface \
                                                       against the static. Restart the application"
                                    );
                                    (Instant::now(), None)
                                }
                            } else {
                                eprintln!("Failed to find the main entry point in live reloaded spirv");
                                (Instant::now(), None)
                            }
                        } else {
                            eprintln!("Failed to reflect on live reloaded spirv");
                            (Instant::now(), None)
                        }
                    }
                    _ => (Instant::now(), None),
                },
            )
            .collect();

        #[cfg(feature = "shader_reload")]
        let any_new_shaders: bool = new_shaders.iter().any(|(_ts, s)| s.is_some());

        #[cfg(not(feature = "shader_reload"))]
        let any_new_shaders: bool = false;

        let different_switches = self.switches != *switches;

        if self.specialization != *new_spec || any_new_shaders || different_switches {
            #[cfg(feature = "shader_reload")]
            for ((shader, last_update), (t, new)) in self
                .shaders
                .iter_mut()
                .zip(self.last_updates.iter_mut())
                .zip(new_shaders.into_iter())
            {
                if let Some(new) = new {
                    replace(shader, new).destroy(device);
                    *last_update = t;
                }
            }
            let mut replacement = Self::new_internal(
                device,
                layout,
                new_spec.clone(),
                if different_switches { None } else { Some(self) },
                dynamic_arguments,
                switches,
            );
            swap(&mut *self, &mut replacement);
            Some(replacement)
        } else {
            None
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.inner.destroy(device);
        self.shaders.into_iter().for_each(|x| x.destroy(device));
    }

    pub(crate) fn vk(&self) -> vk::Pipeline {
        self.inner.vk()
    }
}

pub(crate) trait RefTuple {
    #[allow(single_use_lifetimes)]
    type Ref<'a>
    where
        Self: 'a;
}

impl RefTuple for () {
    #[allow(single_use_lifetimes)]
    type Ref<'a>
    where
        Self: 'a,
    = ();
}

impl<A> RefTuple for (A,) {
    type Ref<'a>
    where
        Self: 'a,
    = (&'a A,);
}

impl<A, B> RefTuple for (A, B) {
    type Ref<'a>
    where
        Self: 'a,
    = (&'a A, &'a B);
}
impl<A, B, C> RefTuple for (A, B, C) {
    type Ref<'a>
    where
        Self: 'a,
    = (&'a A, &'a B, &'a C);
}

impl<A, B, C, D> RefTuple for (A, B, C, D) {
    type Ref<'a>
    where
        Self: 'a,
    = (&'a A, &'a B, &'a C, &'a D);
}

impl<A, B, C, D, E> RefTuple for (A, B, C, D, E) {
    type Ref<'a>
    where
        Self: 'a,
    = (&'a A, &'a B, &'a C, &'a D, &'a E);
}

pub(crate) struct SmartSetLayout<T: DescriptorSetLayout> {
    pub(crate) layout: device::DescriptorSetLayout,
    t: PhantomData<T>,
}

impl<T: DescriptorSetLayout> SmartSetLayout<T> {
    pub(crate) fn new(device: &Device) -> Self {
        let binding_flags = T::binding_flags();
        let flags = if binding_flags
            .iter()
            .any(|x| x.contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND))
        {
            vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
        } else {
            vk::DescriptorSetLayoutCreateFlags::empty()
        };
        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&binding_flags);
        let bindings = T::binding_layout();
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .flags(flags)
            .push_next(&mut binding_flags);
        let layout = device.new_descriptor_set_layout(&create_info);
        device.set_object_name(layout.handle, T::DEBUG_NAME);

        SmartSetLayout { layout, t: PhantomData }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.layout.destroy(device);
    }
}

pub(crate) struct SmartSet<T: DescriptorSet> {
    pub(crate) set: device::DescriptorSet,
    _t: PhantomData<T>,
}

impl<T: DescriptorSet> SmartSet<T> {
    pub(crate) fn new(
        device: &Device,
        main_descriptor_pool: &MainDescriptorPool,
        layout: &SmartSetLayout<T::Layout>,
        ix: u32,
    ) -> Self {
        let set = main_descriptor_pool.0.allocate_set(device, &layout.layout);
        device.set_object_name(set.handle, &format!("{}[{}", T::DEBUG_NAME, ix));
        SmartSet { set, _t: PhantomData }
    }

    pub(crate) fn vk_handle(&self) -> vk::DescriptorSet {
        self.set.handle
    }

    pub(crate) fn destroy(self, pool: &DescriptorPool, device: &Device) {
        self.set.destroy(pool, device);
    }
}

pub(crate) trait DescriptorSet {
    type Layout: DescriptorSetLayout;

    const DEBUG_NAME: &'static str;

    fn vk_handle(&self) -> vk::DescriptorSet;
}

pub(crate) trait DescriptorBufferBinding {
    type T;
    type Set: DescriptorSet;
    const SIZE: vk::DeviceSize = size_of::<Self::T>() as vk::DeviceSize;
    const INDEX: u32;
    const DESCRIPTOR_TYPE: vk::DescriptorType;
}

fn update_whole_buffer<T: DescriptorBufferBinding>(device: &Device, set: &mut SmartSet<T::Set>, buf: &BufferType<T>) {
    let buffer_updates = &[vk::DescriptorBufferInfo {
        buffer: buf.buffer.handle,
        offset: 0,
        range: T::SIZE,
    }];
    unsafe {
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::builder()
                .dst_set(set.vk_handle())
                .dst_binding(T::INDEX as u32)
                .descriptor_type(T::DESCRIPTOR_TYPE)
                .buffer_info(buffer_updates)
                .build()],
            &[],
        );
    }
}

pub(crate) type BufferType<B> = StaticBuffer<BindingT<B>>;
pub(crate) type BindingT<B> = <B as DescriptorBufferBinding>::T;
pub(crate) fn binding_size<B: DescriptorBufferBinding>() -> vk::DeviceSize {
    B::SIZE
}

impl RenderFrame {
    pub(crate) fn new() -> (RenderFrame, Swapchain, winit::event_loop::EventLoop<()>) {
        let (instance, events_loop) = Instance::new();
        let instance = Arc::new(instance);
        let surface = Surface::new(&instance);
        let device = Device::new(&instance, &surface).expect("Failed to create device");
        device.set_object_name(device.handle(), "Device");
        let swapchain = Swapchain::new(&instance, &device, surface);

        // Start frame number at 1 because something could await for semaphores from frame - 1, so we would
        // underflow
        let frame_number = 1;
        let buffer_count = swapchain.desired_image_count.to_usize().unwrap();
        let auto_semaphores = AutoSemaphores::new(&device);

        (
            RenderFrame {
                instance: Arc::clone(&instance),
                device,
                auto_semaphores,
                frame_number,
                buffer_count,
            },
            swapchain,
            events_loop,
        )
    }

    pub(crate) fn new_buffered<T, F: FnMut(u32) -> T>(&self, creator: F) -> DoubleBuffered<T> {
        DoubleBuffered::new(self.buffer_count, creator)
    }

    pub(crate) fn destroy(self) {
        self.auto_semaphores.destroy(&self.device);
    }
}

pub(crate) struct MainDescriptorPool(pub(crate) DescriptorPool);

impl MainDescriptorPool {
    pub(crate) fn new(renderer: &RenderFrame) -> MainDescriptorPool {
        let descriptor_pool = renderer.device.new_descriptor_pool(3_000, &[
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 16384,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 4096 * 40,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 128,
            },
        ]);
        renderer
            .device
            .set_object_name(descriptor_pool.handle, "Main Descriptor Pool");

        MainDescriptorPool(descriptor_pool)
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.0.destroy(device);
    }
}

renderer_macros::define_resource!(Color = Image COLOR B8G8R8A8_UNORM);
renderer_macros::define_resource!(PresentSurface DoubleBuffered = Image COLOR B8G8R8A8_UNORM);
pub(crate) struct MainAttachments {
    #[allow(unused)]
    swapchain_image_views: Vec<ImageView>,
    #[allow(unused)]
    swapchain_format: vk::Format,
    #[allow(unused)]
    depth_image: Image,
    depth_image_view: ImageView,
    #[allow(unused)]
    color_image: Color,
    color_image_view: ImageView,
}

impl MainAttachments {
    pub(crate) fn new(renderer: &RenderFrame, swapchain: &Swapchain) -> MainAttachments {
        let images = unsafe { swapchain.ext.get_swapchain_images(swapchain.swapchain).unwrap() };
        assert!(images.len().to_u32().unwrap() >= swapchain.desired_image_count);
        println!("swapchain images len {}", images.len());
        for (ix, swapchain_image) in images.iter().enumerate() {
            renderer
                .device
                .set_object_name(*swapchain_image, &format!("Swapchain Image[{}]", ix));
        }
        let depth_image = {
            let im = renderer.device.new_image(
                vk::Format::D16_UNORM,
                vk::Extent3D {
                    width: swapchain.width,
                    height: swapchain.height,
                    depth: 1,
                },
                vk::SampleCountFlags::TYPE_4,
                vk::ImageTiling::OPTIMAL,
                vk::ImageLayout::UNDEFINED,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | if cfg!(feature = "nsight_profiling") {
                        vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
                    } else {
                        vk::ImageUsageFlags::empty()
                    },
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            );
            renderer.device.set_object_name(im.handle, "Depth RT");
            im
        };
        let color_image = {
            let im = renderer.device.new_image_exclusive(
                vk::Format::B8G8R8A8_UNORM,
                vk::Extent3D {
                    width: swapchain.width,
                    height: swapchain.height,
                    depth: 1,
                },
                vk::SampleCountFlags::TYPE_4,
                vk::ImageTiling::OPTIMAL,
                vk::ImageLayout::UNDEFINED,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | if cfg!(feature = "nsight_profiling") {
                        vk::ImageUsageFlags::TRANSFER_DST
                    } else {
                        vk::ImageUsageFlags::empty()
                    },
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            );
            Color::import(&renderer.device, im)
        };
        let image_views = images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain.surface.surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                let handle = unsafe {
                    renderer
                        .device
                        .create_image_view(&create_view_info, renderer.device.allocation_callbacks())
                        .unwrap()
                };

                ImageView { handle }
            })
            .collect::<Vec<_>>();
        let color_image_view = {
            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(color_image.handle);
            renderer.device.new_image_view(&create_view_info)
        };
        let depth_image_view = {
            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(depth_image.handle);
            renderer.device.new_image_view(&create_view_info)
        };

        MainAttachments {
            swapchain_image_views: image_views,
            swapchain_format: swapchain.surface.surface_format.format,
            depth_image,
            depth_image_view,
            color_image,
            color_image_view,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        for view in self.swapchain_image_views.into_iter() {
            view.destroy(device);
        }
        self.depth_image_view.destroy(device);
        self.color_image_view.destroy(device);
        self.depth_image.destroy(device);
        self.color_image.destroy(device);
    }
}

pub(crate) struct CameraMatrices {
    pub(crate) set_layout: SmartSetLayout<camera_set::Layout>,
    buffer: DoubleBuffered<BufferType<camera_set::bindings::matrices>>,
    set: DoubleBuffered<SmartSet<camera_set::Set>>,
}

impl CameraMatrices {
    pub(crate) fn new(renderer: &RenderFrame, main_descriptor_pool: &MainDescriptorPool) -> CameraMatrices {
        let buffer = renderer.new_buffered(|ix| {
            let b = renderer.device.new_static_buffer(
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );
            renderer
                .device
                .set_object_name(b.buffer.handle, &format!("Camera matrices Buffer - ix={}", ix));
            b
        });
        let set_layout = SmartSetLayout::new(&renderer.device);
        let set = renderer.new_buffered(|ix| {
            let mut s = SmartSet::new(&renderer.device, main_descriptor_pool, &set_layout, ix);

            update_whole_buffer::<camera_set::bindings::matrices>(&renderer.device, &mut s, buffer.current(ix));

            s
        });

        CameraMatrices {
            set_layout,
            buffer,
            set,
        }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.set_layout.destroy(device);
        self.buffer.into_iter().for_each(|b| b.destroy(device));
        self.set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
    }
}

pub(crate) struct ModelData {
    pub(crate) model_set_layout: SmartSetLayout<model_set::Layout>,
    pub(crate) model_set: DoubleBuffered<SmartSet<model_set::Set>>,
    pub(crate) model_buffer: DoubleBuffered<BufferType<model_set::bindings::model>>,
}

impl ModelData {
    pub(crate) fn new(renderer: &RenderFrame, main_descriptor_pool: &MainDescriptorPool) -> ModelData {
        let device = &renderer.device;

        let model_set_layout = SmartSetLayout::new(device);

        let model_buffer = renderer.new_buffered(|ix| {
            let b = device.new_static_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );
            device.set_object_name(b.buffer.handle, &format!("Model Buffer - {}", ix));
            b
        });
        let model_set = renderer.new_buffered(|ix| {
            let mut s = SmartSet::new(&renderer.device, main_descriptor_pool, &model_set_layout, ix);
            update_whole_buffer::<model_set::bindings::model>(&renderer.device, &mut s, model_buffer.current(ix));
            s
        });

        ModelData {
            model_set_layout,
            model_set,
            model_buffer,
        }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.model_set_layout.destroy(device);
        self.model_buffer.into_iter().for_each(|b| b.destroy(device));
        self.model_set
            .into_iter()
            .for_each(|s| s.destroy(&main_descriptor_pool.0, device));
    }
}

pub(crate) struct Resized(pub(crate) bool);

pub(crate) struct GltfPassData {
    pub(crate) gltf_pipeline: SmartPipeline<gltf_mesh::Pipeline>,
    pub(crate) previous_gltf_pipeline: DoubleBuffered<Option<SmartPipeline<gltf_mesh::Pipeline>>>,
    pub(crate) gltf_pipeline_layout: SmartPipelineLayout<gltf_mesh::PipelineLayout>,
    command_util: CommandUtil,
}

impl GltfPassData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        model_data: &ModelData,
        base_color: &BaseColorDescriptorSet,
        shadow_mapping: &ShadowMappingData,
        camera_matrices: &CameraMatrices,
        acceleration_structures: &AccelerationStructures,
    ) -> GltfPassData {
        /*
        let queue_family_indices = vec![device.graphics_queue_family];
        let image_create_info = vk::ImageCreateInfo::builder()
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D {
                width: 20 * 16384,
                height: 20 * 16384,
                depth: 1,
            })
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .mip_levels(1)
            .array_layers(1)
            .image_type(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .flags(vk::ImageCreateFlags::SPARSE_BINDING | vk::ImageCreateFlags::SPARSE_RESIDENCY)
            .queue_family_indices(&queue_family_indices);

        let image = unsafe { device.create_image(&image_create_info, None).unwrap() };

        unsafe {
            let mut requirements = device.get_image_memory_requirements(image);
            requirements.size = requirements.alignment;
            let requirements = vec![requirements; 5];
            let create_infos = vec![
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
                alloc::VmaAllocationCreateInfo {
                    flags: alloc::VmaAllocationCreateFlagBits(0),
                    memoryTypeBits: 0,
                    pUserData: std::ptr::null_mut(),
                    pool: std::ptr::null_mut(),
                    preferredFlags: 0,
                    requiredFlags: 0,
                    usage: alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                },
            ];
            use std::mem::MaybeUninit;
            let mut allocations: [MaybeUninit<alloc::VmaAllocation>; 5] =
                MaybeUninit::uninit_array();
            let mut allocation_infos: [MaybeUninit<alloc::VmaAllocationInfo>; 5] =
                MaybeUninit::uninit_array();
            alloc::vmaAllocateMemoryPages(
                device.allocator,
                requirements.as_ptr(),
                create_infos.as_ptr(),
                5,
                MaybeUninit::first_ptr_mut(&mut allocations),
                MaybeUninit::first_ptr_mut(&mut allocation_infos),
            );

            let allocations = MaybeUninit::slice_get_ref(&allocations);
            let allocation_infos = MaybeUninit::slice_get_ref(&allocation_infos);

            let mut image_binds = vec![];

            for a in allocation_infos.iter() {
                image_binds.push(
                    vk::SparseImageMemoryBind::builder()
                        .memory(a.deviceMemory)
                        .extent(vk::Extent3D {
                            width: 128,
                            height: 128,
                            depth: 1,
                        })
                        .memory_offset(a.offset)
                        .subresource(
                            vk::ImageSubresource::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(0)
                                .build(),
                        )
                        .build(),
                );
            }

            let mut fence = device.new_fence();

            let image_info = vk::SparseImageMemoryBindInfo::builder()
                .binds(&image_binds)
                .image(image)
                .build();

            device.fp_v1_0().queue_bind_sparse(
                *device.graphics_queue.lock(),
                1,
                &*vk::BindSparseInfo::builder().image_binds(&[image_info])
                    as *const vk::BindSparseInfo,
                fence.handle,
            );
            fence.wait();
        }
        */

        let gltf_pipeline_layout = SmartPipelineLayout::new(
            &renderer.device,
            (
                &model_data.model_set_layout,
                &camera_matrices.set_layout,
                &shadow_mapping.user_set_layout,
                &base_color.layout,
                &acceleration_structures.set_layout,
            ),
        );
        use systems::shadow_mapping::DIM as SHADOW_MAP_DIM;
        let spec = gltf_mesh::Specialization {
            shadow_map_dim: SHADOW_MAP_DIM,
            shadow_map_dim_squared: SHADOW_MAP_DIM * SHADOW_MAP_DIM,
        };
        let gltf_pipeline = SmartPipeline::new(&renderer.device, &gltf_pipeline_layout, spec, ());

        let command_util = CommandUtil::new(renderer, renderer.device.graphics_queue_family);

        GltfPassData {
            gltf_pipeline,
            previous_gltf_pipeline: renderer.new_buffered(|_| None),
            gltf_pipeline_layout,
            command_util,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.gltf_pipeline.destroy(device);
        self.previous_gltf_pipeline
            .into_iter()
            .for_each(|p| p.into_iter().for_each(|p| p.destroy(device)));
        self.gltf_pipeline_layout.destroy(device);
        self.command_util.destroy(device);
    }
}

pub(crate) fn render_frame(
    renderer: Res<RenderFrame>,
    main_attachments: Res<MainAttachments>,
    (image_index, model_data, swapchain_index_map): (Res<ImageIndex>, Res<ModelData>, Res<SwapchainIndexToFrameNumber>),
    (runtime_config, mut future_configs): (Res<RuntimeConfiguration>, ResMut<FutureRuntimeConfiguration>),
    (camera_matrices, swapchain): (Res<CameraMatrices>, Res<Swapchain>),
    consolidated_mesh_buffers: Res<ConsolidatedMeshBuffers>,
    (debug_aabb_pass_data, shadow_mapping_data): (Res<DebugAABBPassData>, Res<ShadowMappingData>),
    base_color_descriptor_set: Res<BaseColorDescriptorSet>,
    cull_pass_data: Res<CullPassData>,
    (acceleration_structures, reference_rt_data): (Res<AccelerationStructures>, Res<ReferenceRTData>),
    (submissions, mut gltf_pass, mut gui_render_data, mut camera, mut input_handler, mut gui): (
        Res<Submissions>,
        ResMut<GltfPassData>,
        ResMut<GuiRenderData>,
        ResMut<Camera>,
        NonSendMut<InputHandler>,
        NonSendMut<Gui>,
    ),
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
    #[cfg(feature = "shader_reload")] reloaded_shaders: Res<ReloadedShaders>,
    query: Query<&AABB>,
) {
    scope!("ecs::render_frame");

    let GltfPassData {
        ref mut gltf_pipeline,
        ref gltf_pipeline_layout,
        ref mut previous_gltf_pipeline,
        ref mut command_util,
    } = &mut *gltf_pass;

    // TODO: count this? pack and defragment draw calls?
    let total = binding_size::<cull_set::bindings::indirect_commands>() as u32
        / size_of::<frame_graph::VkDrawIndexedIndirectCommand>() as u32;

    {
        // clean up the old pipeline that was used N frames ago
        if let Some(previous) = previous_gltf_pipeline.current_mut(image_index.0).take() {
            previous.destroy(&renderer.device);
        }

        use systems::shadow_mapping::DIM as SHADOW_MAP_DIM;
        *previous_gltf_pipeline.current_mut(image_index.0) = gltf_pipeline.specialize_switched(
            &renderer.device,
            gltf_pipeline_layout,
            &gltf_mesh::Specialization {
                shadow_map_dim: SHADOW_MAP_DIM,
                shadow_map_dim_squared: SHADOW_MAP_DIM * SHADOW_MAP_DIM,
            },
            (),
            &[runtime_config.rt]
                .into_iter()
                .map(|rt| ("RT".to_owned(), rt))
                .collect(),
            #[cfg(feature = "shader_reload")]
            &reloaded_shaders,
        );
    }

    let command_buffer = command_util.reset_and_record(&renderer, &image_index);
    let main_renderpass_marker = command_buffer.debug_marker_around("main renderpass", [0.0, 0.0, 1.0, 1.0]);
    let guard = renderer_macros::barrier!(
        command_buffer,
        IndirectCommandsBuffer.draw_from r in Main indirect buffer after [compact, copy_frozen] if [!DEBUG_AABB],
        IndirectCommandsCount.draw_from r in Main indirect buffer after [draw_depth] if [!DEBUG_AABB],
        TLAS.in_main r in Main descriptor gltf_mesh.acceleration_set.top_level_as after [build] if [!DEBUG_AABB, RT],
        ShadowMapAtlas.apply r in Main descriptor gltf_mesh.shadow_map_set.shadow_maps layout DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL after [prepare] if [!DEBUG_AABB]; &shadow_mapping_data.depth_image,
        Color.render rw in Main attachment in Main layout COLOR_ATTACHMENT_OPTIMAL; &main_attachments.color_image,
        ConsolidatedPositionBuffer.draw_from r in Main vertex buffer after [in_depth] if [!DEBUG_AABB],
        ConsolidatedNormalBuffer.draw_from r in Main vertex buffer after [consolidate] if [!DEBUG_AABB],
        CulledIndexBuffer.draw_from r in Main index buffer after [copy_frozen, cull] if [!DEBUG_AABB],
        DepthRT.in_main r in Main attachment in Main layout DEPTH_STENCIL_READ_ONLY_OPTIMAL after [draw_depth] if [!DEBUG_AABB]; &main_attachments.depth_image,
    );
    let reference_rt = runtime_config.rt && runtime_config.reference_rt;
    unsafe {
        renderer.device.cmd_set_viewport(*command_buffer, 0, &[vk::Viewport {
            x: 0.0,
            y: swapchain.height as f32,
            width: swapchain.width as f32,
            height: -(swapchain.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        }]);
        renderer.device.cmd_set_scissor(*command_buffer, 0, &[vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: swapchain.width,
                height: swapchain.height,
            },
        }]);
        MainRP::begin(
            &renderer,
            *command_buffer,
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.width,
                    height: swapchain.height,
                },
            },
            &[
                main_attachments.color_image_view.handle,
                main_attachments.depth_image_view.handle,
            ],
            &[vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            }],
        );
        if runtime_config.debug_aabbs {
            scope!("ecs::debug_aabb_pass");

            let _aabb_marker = command_buffer.debug_marker_around("aabb debug", [1.0, 0.0, 0.0, 1.0]);
            renderer.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                debug_aabb_pass_data.pipeline.vk(),
            );
            debug_aabb_pass_data.pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                (camera_matrices.set.current(image_index.0),),
            );

            for aabb in &mut query.iter() {
                debug_aabb_pass_data.pipeline_layout.push_constants(
                    &renderer.device,
                    *command_buffer,
                    &debug_aabb::PushConstants {
                        center: aabb.0.center().coords,
                        half_extent: aabb.0.half_extents(),
                    },
                );
                renderer.device.cmd_draw(*command_buffer, 36, 1, 0, 0);
            }
        } else {
            let _gltf_meshes_marker = command_buffer.debug_marker_around("gltf meshes", [1.0, 0.0, 0.0, 1.0]);
            // gltf mesh
            renderer
                .device
                .cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, gltf_pipeline.vk());

            // TODO: make this variable binding nicer
            renderer.device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                gltf_pipeline_layout.vk(),
                0,
                &[
                    model_data.model_set.current(image_index.0).vk_handle(),
                    camera_matrices.set.current(image_index.0).vk_handle(),
                    shadow_mapping_data.user_set.current(image_index.0).vk_handle(),
                    base_color_descriptor_set.set.current(image_index.0).vk_handle(),
                    acceleration_structures.set.current(image_index.0).vk_handle(),
                ][..(if runtime_config.rt { 5 } else { 4 })],
                &[],
            );

            // gltf_pipeline_layout.bind_descriptor_sets(
            //     &renderer.device,
            //     *command_buffer,
            //     (
            //         model_data.model_set.current(image_index.0),
            //         camera_matrices.set.current(image_index.0),
            //         shadow_mapping_data.user_set.current(image_index.0),
            //         base_color_descriptor_set.set.current(image_index.0),
            //         acceleration_structures.set.current(image_index.0),
            //     ),
            // );
            renderer.device.cmd_bind_index_buffer(
                *command_buffer,
                cull_pass_data.culled_index_buffer.buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            renderer.device.cmd_bind_vertex_buffers(
                *command_buffer,
                0,
                &[
                    consolidated_mesh_buffers.position_buffer.buffer.handle,
                    consolidated_mesh_buffers.normal_buffer.buffer.handle,
                    consolidated_mesh_buffers.uv_buffer.handle,
                    consolidated_mesh_buffers.tangent_buffer.handle,
                ],
                &[0, 0, 0, 0],
            );
            #[cfg(feature = "crash_debugging")]
            crash_buffer.record(
                &renderer,
                *command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                &image_index,
                0,
            );
            renderer.device.cmd_draw_indexed_indirect_count(
                *command_buffer,
                cull_pass_data.culled_commands_buffer.buffer.handle,
                0,
                cull_pass_data.culled_commands_count_buffer.buffer.handle,
                0,
                total,
                size_of::<frame_graph::VkDrawIndexedIndirectCommand>() as u32,
            );
            #[cfg(feature = "crash_debugging")]
            crash_buffer.record(
                &renderer,
                *command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                &image_index,
                1,
            );
        }
        renderer.device.dynamic_rendering.cmd_end_rendering(*command_buffer);
    }
    drop(guard);
    drop(main_renderpass_marker);

    ImguiRP::begin(
        &renderer,
        *command_buffer,
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: swapchain.width,
                height: swapchain.height,
            },
        },
        &[main_attachments.color_image_view.handle],
        &[],
    );

    render_gui(
        &renderer,
        &mut gui_render_data,
        &mut future_configs,
        &swapchain,
        &mut camera,
        &mut input_handler,
        &mut gui,
        &command_buffer,
        &submissions,
        #[cfg(feature = "shader_reload")]
        &reloaded_shaders,
    );
    unsafe {
        renderer.device.dynamic_rendering.cmd_end_rendering(*command_buffer);
    }

    let swapchain_images = unsafe { swapchain.ext.get_swapchain_images(swapchain.swapchain).unwrap() };

    struct SwapchainImageWrapper {
        handle: vk::Image,
    }
    let swapchain_image = SwapchainImageWrapper {
        handle: swapchain_images[image_index.0 as usize],
    };

    unsafe {
        let _guard = renderer_macros::barrier!(
            command_buffer,
            Color.resolve r in Main transfer copy layout TRANSFER_SRC_OPTIMAL after [render]; &main_attachments.color_image,
            PresentSurface.resolve_color w clobber in Main transfer copy layout TRANSFER_DST_OPTIMAL; &swapchain_image
        );

        renderer.device.cmd_resolve_image(
            *command_buffer,
            main_attachments.color_image.handle,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            swapchain_images[image_index.0 as usize],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageResolve::builder()
                .src_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .build(),
                )
                .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .dst_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .build(),
                )
                .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .extent(vk::Extent3D {
                    width: swapchain.width,
                    height: swapchain.height,
                    depth: 1,
                })
                .build()],
        );
    }

    unsafe {
        let _rt_copy_marker = command_buffer.debug_marker_around("copy reference RT output", [0.0, 1.0, 1.0, 1.0]);
        let _guard = renderer_macros::barrier!(
            command_buffer,
            ReferenceRaytraceOutput.in_main r in Main transfer copy layout TRANSFER_SRC_OPTIMAL after [generate] if []; &reference_rt_data.output_image,
            PresentSurface.blit_reference_rt rw in Main transfer copy layout TRANSFER_DST_OPTIMAL after [resolve_color] if []; &swapchain_image
        );
        if reference_rt {
            renderer.device.cmd_blit_image(
                *command_buffer,
                reference_rt_data.output_image.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_images[image_index.0 as usize],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::builder()
                    .src_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .build(),
                    )
                    .src_offsets([vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D {
                        x: swapchain.width as i32,
                        y: swapchain.height as i32,
                        z: 1,
                    }])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .build(),
                    )
                    .dst_offsets([vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D {
                        x: swapchain.width as i32,
                        y: swapchain.height as i32,
                        z: 1,
                    }])
                    .build()],
                vk::Filter::LINEAR,
            );
        }
    }

    let command_buffer = *command_buffer.end();

    submissions.submit(
        &renderer,
        frame_graph::Main::INDEX,
        Some(command_buffer),
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
    );
}

pub(crate) struct GuiRenderData {
    pos_buffer: StaticBuffer<[glm::Vec2; 1024 * 1024]>,
    uv_buffer: StaticBuffer<[glm::Vec2; 1024 * 1024]>,
    col_buffer: StaticBuffer<[glm::Vec4; 1024 * 1024]>,
    index_buffer: StaticBuffer<[imgui::DrawIdx; 1024 * 1024]>,
    texture: Image,
    #[allow(unused)]
    texture_view: ImageView,
    #[allow(unused)]
    sampler: Sampler,
    #[allow(unused)]
    descriptor_set_layout: SmartSetLayout<imgui_set::Layout>,
    descriptor_set: SmartSet<imgui_set::Set>,
    pipeline_layout: SmartPipelineLayout<imgui_pipe::PipelineLayout>,
    pipeline: SmartPipeline<imgui_pipe::Pipeline>,
}

impl FromWorld for GuiRenderData {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let main_descriptor_pool = world.get_resource::<MainDescriptorPool>().unwrap();
        let mut gui = unsafe { world.get_non_send_resource_unchecked_mut::<Gui>().unwrap() };
        Self::new(renderer, main_descriptor_pool, &mut gui)
    }
}

impl GuiRenderData {
    fn new(renderer: &RenderFrame, main_descriptor_pool: &MainDescriptorPool, gui: &mut Gui) -> GuiRenderData {
        let imgui = &mut gui.imgui;
        imgui
            .io_mut()
            .backend_flags
            .insert(imgui::BackendFlags::RENDERER_HAS_VTX_OFFSET);
        let pos_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        );
        let uv_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        );
        let col_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        );
        renderer
            .device
            .set_object_name(pos_buffer.buffer.handle, "GUI Vertex Buffer - pos");
        renderer
            .device
            .set_object_name(uv_buffer.buffer.handle, "GUI Vertex Buffer - uv");
        renderer
            .device
            .set_object_name(col_buffer.buffer.handle, "GUI Vertex Buffer - col");
        let index_buffer = renderer.device.new_static_buffer(
            vk::BufferUsageFlags::INDEX_BUFFER,
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        );
        renderer
            .device
            .set_object_name(index_buffer.buffer.handle, "GUI Index Buffer");
        let texture_staging = {
            let mut fonts = imgui.fonts();
            let imgui_texture = fonts.build_alpha8_texture();
            let texture = renderer.device.new_image(
                vk::Format::R8_UNORM,
                vk::Extent3D {
                    width: imgui_texture.width,
                    height: imgui_texture.height,
                    depth: 1,
                },
                vk::SampleCountFlags::TYPE_1,
                vk::ImageTiling::LINEAR,
                vk::ImageLayout::PREINITIALIZED,
                vk::ImageUsageFlags::TRANSFER_SRC,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );
            {
                let mut texture_data = texture
                    .map::<u8>(&renderer.device)
                    .expect("failed to map imgui texture");
                texture_data[0..imgui_texture.data.len()].copy_from_slice(imgui_texture.data);
            }
            texture
        };
        let mut fonts = imgui.fonts();
        let imgui_texture = fonts.build_alpha8_texture();
        let texture = {
            renderer.device.new_image(
                vk::Format::R8_UNORM,
                vk::Extent3D {
                    width: imgui_texture.width,
                    height: imgui_texture.height,
                    depth: 1,
                },
                vk::SampleCountFlags::TYPE_1,
                vk::ImageTiling::OPTIMAL,
                vk::ImageLayout::UNDEFINED,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            )
        };
        let sampler = renderer.device.new_sampler(
            &vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
        );

        let descriptor_set_layout = SmartSetLayout::new(&renderer.device);

        let descriptor_set = SmartSet::new(
            &renderer.device,
            main_descriptor_pool,
            &descriptor_set_layout,
            0, // FIXME
        );

        let pipeline_layout = SmartPipelineLayout::new(&renderer.device, (&descriptor_set_layout,));

        let pipeline = SmartPipeline::new(&renderer.device, &pipeline_layout, imgui_pipe::Specialization {}, ());

        let texture_view = renderer.device.new_image_view(
            &vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8_UNORM)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::ZERO,
                    b: vk::ComponentSwizzle::ZERO,
                    a: vk::ComponentSwizzle::ZERO,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(texture.handle),
        );

        unsafe {
            renderer.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set.set.handle)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&[vk::DescriptorImageInfo::builder()
                        .sampler(sampler.handle)
                        .image_view(texture_view.handle)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .build()])
                    .build()],
                &[],
            );
        }

        let mut command_pool = StrictCommandPool::new(
            &renderer.device,
            renderer.device.graphics_queue_family,
            "GuiRender Initialization Command Pool",
        );

        let cb = command_pool.record_one_time(&renderer.device, "prepare gui texture");
        unsafe {
            renderer.device.cmd_pipeline_barrier(
                *cb,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                Default::default(),
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier::builder()
                        .image(texture_staging.handle)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .old_layout(vk::ImageLayout::PREINITIALIZED)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::HOST_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .build(),
                    vk::ImageMemoryBarrier::builder()
                        .image(texture.handle)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .build(),
                ],
            );
            renderer.device.cmd_copy_image(
                *cb,
                texture_staging.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                texture.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageCopy {
                    src_offset: vk::Offset3D::default(),
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offset: vk::Offset3D::default(),
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    extent: vk::Extent3D {
                        width: imgui_texture.width,
                        height: imgui_texture.height,
                        depth: 1,
                    },
                }],
            );
            renderer.device.cmd_pipeline_barrier(
                *cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                Default::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .image(texture.handle)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .build()],
            );
            let cb = cb.end();
            let fence = renderer.device.new_fence();
            let queue = renderer.device.graphics_queue().lock();
            renderer
                .device
                .queue_submit(
                    *queue,
                    &[vk::SubmitInfo::builder().command_buffers(&[*cb]).build()],
                    fence.handle,
                )
                .unwrap();
            renderer
                .device
                .wait_for_fences(&[fence.handle], true, u64::MAX)
                .unwrap();
            fence.destroy(&renderer.device);
            texture_staging.destroy(&renderer.device);
        }

        command_pool.destroy(&renderer.device);

        GuiRenderData {
            pos_buffer,
            uv_buffer,
            col_buffer,
            index_buffer,
            texture,
            texture_view,
            sampler,
            descriptor_set_layout,
            descriptor_set,
            pipeline_layout,
            pipeline,
        }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.pipeline.destroy(device);
        self.pipeline_layout.destroy(device);
        self.sampler.destroy(device);
        self.texture_view.destroy(device);
        self.texture.destroy(device);
        self.descriptor_set.destroy(&main_descriptor_pool.0, device);
        self.descriptor_set_layout.destroy(device);
        self.pos_buffer.destroy(device);
        self.uv_buffer.destroy(device);
        self.col_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}

/// This wrapper is used to break dependency chains and let otherwise conflicting systems operate on
/// the same resources as long as the writer can afford a delay in making the mutation available to
/// other users. See [copy_resource] and [writeback_resource] for the complete toolset.
/// To avoid exclusive access to the [World], this wrapper must be initialized ahead of time.
///
/// It can be used the opposite way, with the readers being redirected to the wrapper (with stale
/// data) while the writes are made available immedietaly. There is no need to write back any data
/// in this mode.
pub(crate) struct CopiedResource<T>(pub(crate) T);

impl<T> std::ops::Deref for CopiedResource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: FromWorld> FromWorld for CopiedResource<T> {
    fn from_world(world: &mut World) -> Self {
        CopiedResource(<T as FromWorld>::from_world(world))
    }
}

/// This system can be used to copy data from a resource and place it in the [CopiedResource]
/// wrapper. This requires the wrapper to already be present on the [World].
/// Remember to use [writeback_resource] if the mutations are made to the wrapper.
pub(crate) fn copy_resource<T: Component + Clone>(from: Res<T>, mut to: ResMut<CopiedResource<T>>) {
    scope!("ecs::copy_resource");

    to.0.clone_from(&from);
}

fn render_gui(
    renderer: &RenderFrame,
    gui_render_data: &mut GuiRenderData,
    runtime_config: &mut FutureRuntimeConfiguration,
    swapchain: &Swapchain,
    camera: &mut Camera,
    input_handler: &mut InputHandler,
    gui: &mut Gui,
    command_buffer: &StrictRecordingCommandBuffer,
    submissions: &Submissions,
    #[cfg(feature = "shader_reload")] reloaded_shaders: &ReloadedShaders,
) {
    scope!("rendering::render_gui");

    let GuiRenderData {
        ref pos_buffer,
        ref uv_buffer,
        ref col_buffer,
        ref index_buffer,
        ref pipeline_layout,
        ref pipeline,
        ref descriptor_set,
        ..
    } = *gui_render_data;
    let gui_draw_data = gui.update(
        renderer,
        input_handler,
        swapchain,
        camera,
        runtime_config,
        submissions,
        #[cfg(feature = "shader_reload")]
        reloaded_shaders,
    );

    let _gui_debug_marker = command_buffer.debug_marker_around("GUI", [1.0, 1.0, 0.0, 1.0]);
    unsafe {
        pipeline_layout.bind_descriptor_sets(&renderer.device, **command_buffer, (descriptor_set,));
        renderer
            .device
            .cmd_bind_pipeline(**command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.vk());
        renderer.device.cmd_bind_vertex_buffers(
            **command_buffer,
            0,
            &[
                pos_buffer.buffer.handle,
                uv_buffer.buffer.handle,
                col_buffer.buffer.handle,
            ],
            &[0, 0, 0],
        );
        renderer
            .device
            .cmd_bind_index_buffer(**command_buffer, index_buffer.buffer.handle, 0, vk::IndexType::UINT16);
        let [x, y] = gui_draw_data.display_size;
        {
            pipeline_layout.push_constants(&renderer.device, **command_buffer, &imgui_pipe::PushConstants {
                scale: glm::vec2(2.0 / x, 2.0 / y),
                translate: glm::vec2(-1.0, -1.0),
            });
        }
        {
            let mut vertex_offset_coarse: usize = 0;
            let mut index_offset_coarse: usize = 0;
            let mut pos_slice = pos_buffer
                .map(&renderer.device)
                .expect("Failed to map gui vertex buffer - pos");
            let mut uv_slice = uv_buffer
                .map(&renderer.device)
                .expect("Failed to map gui vertex buffer - uv");
            let mut col_slice = col_buffer
                .map(&renderer.device)
                .expect("Failed to map gui vertex buffer - col");
            let mut index_slice = index_buffer
                .map(&renderer.device)
                .expect("Failed to map gui index buffer");
            for draw_list in gui_draw_data.draw_lists() {
                scope!("rendering::gui_draw_list");

                let index_len = draw_list.idx_buffer().len();
                index_slice[index_offset_coarse..index_offset_coarse + index_len]
                    .copy_from_slice(draw_list.idx_buffer());
                let vertex_len = draw_list.vtx_buffer().len();
                for (ix, vertex) in draw_list.vtx_buffer().iter().enumerate() {
                    pos_slice[ix] = glm::Vec2::from(vertex.pos);
                    uv_slice[ix] = glm::Vec2::from(vertex.uv);
                    // TODO: this conversion sucks, need to add customizable vertex attribute formats
                    let byte_color = na::SVector::<u8, 4>::from(vertex.col);
                    col_slice[ix] = byte_color.map(|byte| f32::from(byte) / 255.0);
                }
                for draw_cmd in draw_list.commands() {
                    match draw_cmd {
                        imgui::DrawCmd::Elements { count, cmd_params } => {
                            renderer.device.cmd_set_scissor(**command_buffer, 0, &[vk::Rect2D {
                                offset: vk::Offset2D {
                                    x: cmd_params.clip_rect[0] as i32,
                                    y: cmd_params.clip_rect[1] as i32,
                                },
                                extent: vk::Extent2D {
                                    width: (cmd_params.clip_rect[2] - cmd_params.clip_rect[0]) as u32,
                                    height: (cmd_params.clip_rect[3] - cmd_params.clip_rect[1]) as u32,
                                },
                            }]);
                            renderer.device.cmd_draw_indexed(
                                **command_buffer,
                                count as u32,
                                1,
                                (index_offset_coarse + cmd_params.idx_offset) as u32,
                                (vertex_offset_coarse + cmd_params.vtx_offset) as i32,
                                0,
                            );
                        }
                        _ => panic!("le wtf"),
                    }
                }
                index_offset_coarse += index_len;
                vertex_offset_coarse += vertex_len;
            }
        }
    }
}

pub(crate) fn model_matrices_upload(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    mut model_data: ResMut<ModelData>,
    query: Query<(&DrawIndex, &ModelMatrix)>,
) {
    scope!("ecs::ModelMatricesUpload");
    let mut model_mapped = model_data
        .model_buffer
        .current_mut(image_index.0)
        .map(&renderer.device)
        .expect("failed to map Model buffer");

    let mut max_accessed = 0;
    query.for_each(|(draw_index, model_matrix)| {
        model_mapped.model[draw_index.0 as usize] = model_matrix.0;
        max_accessed = max(max_accessed, draw_index.0);
    });

    // sanity check for the following flush calculation
    const_assert_eq!(size_of::<glm::Mat4>(), size_of::<frame_graph::ModelMatrices>() / 4096,);
    model_mapped.unmap_used_range(0..(u64::from(max_accessed) * size_of::<glm::Mat4>() as vk::DeviceSize));
}

pub(crate) fn camera_matrices_upload(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    camera: Res<Camera>,
    mut camera_matrices: ResMut<CameraMatrices>,
) {
    scope!("ecs::CameraMatricesUpload");
    let mut model_mapped = camera_matrices
        .buffer
        .current_mut(image_index.0)
        .map(&renderer.device)
        .expect("failed to map camera matrix buffer");
    *model_mapped = BindingT::<camera_set::bindings::matrices> {
        projection: camera.projection,
        view: camera.view,
        position: camera.position.coords.push(1.0),
        pv: camera.projection * camera.view,
    };
}

pub(crate) fn recreate_main_framebuffer(
    renderer: Res<RenderFrame>,
    mut attachments: ResMut<MainAttachments>,
    resized: Res<Resized>,
    swapchain: Res<Swapchain>,
) {
    if !resized.0 {
        return;
    }

    replace(&mut *attachments, MainAttachments::new(&renderer, &swapchain)).destroy(&renderer.device);
}

pub(crate) fn graphics_stage() -> SystemStage {
    let stage = SystemStage::parallel();

    let test = SystemGraph::new();

    let copy_runtime_config = test.root(copy_resource::<RuntimeConfiguration>);
    let copy_camera = test.root(copy_resource::<Camera>);
    let copy_indices = test.root(copy_resource::<SwapchainIndexToFrameNumber>);
    let setup_submissions = test.root(setup_submissions);

    let initial = (copy_runtime_config, copy_camera, copy_indices, setup_submissions);

    let recreate_base_color = test.root(recreate_base_color_descriptor_set);
    let recreate_main_framebuffer = test.root(recreate_main_framebuffer);

    let update_base_color = recreate_base_color.then(update_base_color_descriptors);

    let (consolidate_mesh_buffers, upload_loaded_meshes, cull_bypass, build_as, shadow_mapping, reference_rt) = initial
        .join_all((
            consolidate_mesh_buffers.system(),
            upload_loaded_meshes.system(),
            cull_pass_bypass.system(),
            build_acceleration_structures.system(),
            prepare_shadow_maps.system(),
            reference_raytrace.system(),
        ));

    let cull = consolidate_mesh_buffers.then(cull_pass);

    let depth = (recreate_main_framebuffer.clone(), consolidate_mesh_buffers.clone()).join(depth_only_pass);

    let main_pass = (
        recreate_main_framebuffer,
        update_base_color,
        build_as.clone(),
        consolidate_mesh_buffers,
    )
        .join(render_frame);

    let build_as_submit = build_as.then(submit_pending);

    (
        main_pass,
        cull,
        cull_bypass,
        depth,
        shadow_mapping,
        upload_loaded_meshes,
        reference_rt,
        build_as_submit,
    )
        .join(PresentFramebuffer::exec);

    stage.with_system_set(test.into())
}

#[derive(PartialEq, Debug, Clone)]
pub(crate) enum TrackedSubmission {
    Preparing,
    Executable(Option<vk::CommandBuffer>),
    Submitting,
    Submitted,
}

pub(crate) struct Submissions {
    pub(crate) active_graph: StableDiGraph<String, DependencyType>,
    pub(crate) upcoming_graph: StableDiGraph<String, DependencyType>,
    pub(crate) active_resources: HashMap<String, ResourceClaims>,
    /// This stores the resources graph for the next frame. It will be copied to `active_resources`
    /// on the next frame. This allows us to look ahead to how resources will be used in the
    /// next frame, and transition them accurately during configuration changes.
    pub(crate) upcoming_resources: HashMap<String, ResourceClaims>,
    pub(crate) remaining: Mutex<StableDiGraph<TrackedSubmission, ()>>,
    pub(crate) upcoming_remaining: Mutex<StableDiGraph<TrackedSubmission, ()>>,
    /// When the graph is simplified according to conditionals, we pick up inactive nodes and signal
    /// them at the next downstream opportunity. This allows us to increment all the semaphores
    /// appropriately all of the time while toggle freely between conditionals.
    pub(crate) extra_signals: HashMap<NodeIndex, Vec<NodeIndex>>,
    pub(crate) upcoming_extra_signals: HashMap<NodeIndex, Vec<NodeIndex>>,
    // Holds the mapping from a the graph node index of each compute pass to the virtualized index
    // of the compute queue to use. This exposes all the underlying parallelism of compute
    // submissions. The virtual queue index can be modulo'd with the number of queues exposed by
    // the hardware.
    pub(crate) virtual_queue_indices: HashMap<NodeIndex, usize>,
    pub(crate) upcoming_virtual_queue_indices: HashMap<NodeIndex, usize>,
}

impl Submissions {
    pub(crate) fn new() -> Submissions {
        Submissions {
            active_graph: Default::default(),
            upcoming_graph: Default::default(),
            active_resources: Default::default(),
            upcoming_resources: Default::default(),
            remaining: Mutex::default(),
            upcoming_remaining: Mutex::default(),
            extra_signals: Default::default(),
            upcoming_extra_signals: Default::default(),
            virtual_queue_indices: Default::default(),
            upcoming_virtual_queue_indices: Default::default(),
        }
    }

    pub(crate) fn produce_submission(&self, node_ix: u32, cb: Option<vk::CommandBuffer>) {
        scope!("rendering::produce_submission");
        let mut g = self.remaining.lock();
        let weight = g
            .node_weight_mut(NodeIndex::from(node_ix))
            .unwrap_or_else(|| panic!("Node not found while submitting {}", node_ix));
        debug_assert!(*weight == TrackedSubmission::Preparing, "node_ix = {}", node_ix);
        *weight = TrackedSubmission::Executable(cb);
    }

    pub(crate) fn submit(
        &self,
        renderer: &RenderFrame,
        node_ix: u32,
        cb: Option<vk::CommandBuffer>,
        #[cfg(feature = "crash_debugging")] crash_buffer: &CrashBuffer,
    ) {
        scope!("rendering::submit_command_buffer");
        let mut g = self.remaining.lock();
        let weight = g
            .node_weight_mut(NodeIndex::from(node_ix))
            .unwrap_or_else(|| panic!("Node not found while submitting {}", node_ix));
        debug_assert!(*weight == TrackedSubmission::Preparing, "node_ix = {}", node_ix);
        let _span = tracing::info_span!(target: "renderer", "Submitting command buffer", pass_name = self.active_graph.node_weight(NodeIndex::from(node_ix)).unwrap().as_str()).entered();
        *weight = TrackedSubmission::Executable(cb);
        update_submissions(
            renderer,
            self,
            g,
            #[cfg(feature = "crash_debugging")]
            crash_buffer,
        );
    }

    pub(crate) fn dump_live_graphs(&self) -> Result<(), anyhow::Error> {
        let root_dir = env!("CARGO_MANIFEST_DIR");
        let src = Path::new(&root_dir).join("live-diagnostics");
        for x in read_dir(&src)? {
            remove_file(x?.path())?;
        }
        let data = &RENDERER_INPUT;
        for (res_name, claim) in self.active_resources.iter() {
            let src = src.join(&format!("{}.dot", res_name));
            let mut file = File::create(src)?;

            writeln!(file, "digraph {{")?;
            for (ix, claim) in claim.graph.node_references() {
                writeln!(file, "{} [ label = \"{}\" ]", ix.index(), &claim.step_name,)?;
            }
            for edge in claim.graph.edge_references() {
                let source = &claim.graph[edge.source()];
                let target = &claim.graph[edge.target()];
                let source_queue = data.async_passes.get(&source.pass_name).map(|p| p.queue).unwrap();
                let target_queue = data.async_passes.get(&target.pass_name).map(|p| p.queue).unwrap();
                let color = if source.pass_name == target.pass_name {
                    "green"
                } else if source_queue == target_queue {
                    "blue"
                } else {
                    "red"
                };
                writeln!(
                    file,
                    "{} -> {} [ color = {} ]",
                    edge.source().index(),
                    edge.target().index(),
                    color
                )?;
            }
            writeln!(file, "}}")?;
        }

        let src = src.join("dependency_graph.dot");
        let mut file = File::create(src)?;

        let graph = &self.active_graph;

        writeln!(file, "digraph {{")?;
        for (ix, pass) in graph.node_references() {
            let (sem_ix, step_ix) = data.timeline_semaphore_mapping.get(pass).unwrap();
            writeln!(
                file,
                "{} [ label = \"{}\n({}, {})\", shape = \"rectangle\" ]",
                ix.index(),
                pass,
                sem_ix,
                step_ix,
            )?;
        }
        for edge in graph.edge_references() {
            let source = graph[edge.source()].as_str();
            let target = graph[edge.target()].as_str();
            let source_queue = data.async_passes.get(source).map(|p| p.queue).unwrap();
            let target_queue = data.async_passes.get(target).map(|p| p.queue).unwrap();
            let color = if source_queue == target_queue { "blue" } else { "red" };
            writeln!(
                file,
                "{} -> {} [ color = {} ]",
                edge.source().index(),
                edge.target().index(),
                color
            )?;
        }
        writeln!(file, "}}")?;

        Ok(())
    }

    #[allow(clippy::ptr_arg)]
    pub(crate) fn barrier_buffer(
        &self,
        renderer: &RenderFrame,
        data: &renderer_macro_lib::RendererInput,
        _command_buffer: vk::CommandBuffer,
        resource_name: &str,
        step_name: &str,
        buffer: vk::Buffer,
        acquire_buffer_barriers: &mut Vec<vk::BufferMemoryBarrier2KHR>,
        #[allow(unused)] release_buffer_barriers: &mut Vec<vk::BufferMemoryBarrier2KHR>,
    ) {
        let claims = self.active_resources.get(resource_name).unwrap();
        let next_claims = self.upcoming_resources.get(resource_name).unwrap();

        debug_assert!(matches!(claims.ty, ResourceDefinitionType::StaticBuffer { .. }));

        let this_node = match claims.map.get(step_name) {
            Some(&node) => node,
            None => {
                return;
            }
        };
        let this = match claims.graph.node_weight(this_node) {
            Some(node) => node,
            None => {
                return;
            }
        };
        let get_queue_family_index_for_pass = |pass: &str| {
            data.async_passes
                .get(pass)
                .map(|async_pass| async_pass.queue)
                .expect("pass not found")
        };
        let get_runtime_queue_family = |ty: QueueFamily| match ty {
            QueueFamily::Graphics => renderer.device.graphics_queue_family,
            QueueFamily::Compute => renderer.device.compute_queue_family,
            QueueFamily::Transfer => renderer.device.transfer_queue_family,
        };
        let get_layout_from_renderpass = |step: &ResourceClaim| match step.usage {
            ResourceUsageKind::Attachment(ref renderpass) => data
                .renderpasses
                .get(renderpass)
                .and_then(|x| x.depth_stencil.as_ref().map(|d| &d.layout)),
            _ => None,
        };

        // let connected_components = petgraph::algo::connected_components(&claims.graph);
        // if connected_components != 1 {
        //     let msg = "resource claims graph must have one connected component".to_string();
        //     validation_errors.extend(quote!(compile_error!(#msg);));
        // }
        if petgraph::algo::is_cyclic_directed(&claims.graph) {
            panic!("resource claims graph is cyclic");
        }

        // neighbors with wraparound on both ends to make the code aware of dependencies in the prev/next
        // iterations of the graph
        let mut incoming = claims.graph.neighbors_directed(this_node, Incoming).peekable();
        let incoming = if incoming.peek().is_none() {
            incoming.chain(claims.graph.externals(Outgoing)).collect_vec()
        } else {
            incoming.collect_vec()
        };
        let mut outgoing = claims.graph.neighbors_directed(this_node, Outgoing).peekable();
        let _outgoing = if outgoing.peek().is_none() {
            outgoing.chain(next_claims.graph.externals(Incoming)).collect_vec()
        } else {
            outgoing.collect_vec()
        };

        let this_queue = get_queue_family_index_for_pass(&this.pass_name);
        let _this_queue_runtime = get_runtime_queue_family(this_queue);
        let _this_layout_in_renderpass = get_layout_from_renderpass(this);
        let stage_flags = |step: &ResourceClaim, include_reads: bool, include_writes: bool| {
            use vk::{AccessFlags2KHR as A, PipelineStageFlags2KHR as S};
            let mut accesses = A::empty();
            let mut stages = S::empty();
            if step.reads && include_reads {
                let (local_accesses, local_stages) = match &step.usage {
                    ResourceUsageKind::VertexBuffer => (A::VERTEX_ATTRIBUTE_READ, S::VERTEX_ATTRIBUTE_INPUT),
                    ResourceUsageKind::IndexBuffer => (A::INDEX_READ, S::INDEX_INPUT),
                    ResourceUsageKind::Attachment(ref _renderpass) => {
                        // TODO: imprecise
                        (A::MEMORY_READ, S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS)
                    }
                    ResourceUsageKind::IndirectBuffer => (A::INDIRECT_COMMAND_READ, S::DRAW_INDIRECT),
                    ResourceUsageKind::TransferCopy => (A::TRANSFER_READ, S::COPY),
                    // TODO: sync validation is picky when just CLEAR from sync2 is used
                    ResourceUsageKind::TransferClear => (A::TRANSFER_READ, S::ALL_TRANSFER),
                    ResourceUsageKind::Descriptor(set_name, binding_name, pipeline_name) => {
                        let pipeline = data
                            .pipelines
                            .get(pipeline_name)
                            .expect("pipeline not found in barrier");
                        let set = data.descriptor_sets.get(set_name).expect("set not found in barrier");
                        let binding = set
                            .bindings
                            .iter()
                            .find(|candidate| candidate.name == *binding_name)
                            .expect("binding not found in barrier");
                        let pipeline_stages = pipeline.specific.stages();

                        let mut accesses = A::empty();
                        let mut stages = S::empty();

                        binding
                            .shader_stages
                            .iter()
                            .filter(|x| pipeline_stages.contains(*x))
                            .map(|stage| match stage.as_str() {
                                "COMPUTE" => (A::SHADER_READ, S::COMPUTE_SHADER),
                                _ => unimplemented!("barrier! descriptor stage {}", stage),
                            })
                            .for_each(|(x, y)| {
                                accesses |= x;
                                stages |= y;
                            });
                        (accesses, stages)
                    }
                };
                accesses |= local_accesses;
                stages |= local_stages;
            }
            if step.writes && include_writes {
                let (local_accesses, local_stages) = match &step.usage {
                    ResourceUsageKind::VertexBuffer => panic!("can't have a write access to VertexBuffer"),
                    ResourceUsageKind::IndexBuffer => panic!("can't have a write access to IndexBuffer"),
                    ResourceUsageKind::Attachment(ref _renderpass) => {
                        // TODO: imprecise
                        (A::MEMORY_WRITE, S::COLOR_ATTACHMENT_OUTPUT)
                    }
                    ResourceUsageKind::IndirectBuffer => panic!("can't have a write access to IndirectBuffer"),
                    ResourceUsageKind::TransferCopy => (A::TRANSFER_WRITE, S::COPY),
                    // TODO: sync validation is picky when just CLEAR from sync2 is used
                    ResourceUsageKind::TransferClear => (A::TRANSFER_WRITE, S::ALL_TRANSFER),
                    ResourceUsageKind::Descriptor(set_name, binding_name, pipeline_name) => {
                        let pipeline = data
                            .pipelines
                            .get(pipeline_name)
                            .expect("pipeline not found in barrier");
                        let set = data.descriptor_sets.get(set_name).expect("set not found in barrier");
                        let binding = set
                            .bindings
                            .iter()
                            .find(|candidate| candidate.name == *binding_name)
                            .expect("binding not found in barrier");
                        let pipeline_stages = pipeline.specific.stages();

                        let mut accesses = A::empty();
                        let mut stages = S::empty();

                        binding
                            .shader_stages
                            .iter()
                            .filter(|x| pipeline_stages.contains(*x))
                            .map(|stage| match stage.as_str() {
                                "COMPUTE" => (A::SHADER_WRITE, S::COMPUTE_SHADER),
                                _ => unimplemented!("barrier! descriptor stage {}", stage),
                            })
                            .for_each(|(x, y)| {
                                accesses |= x;
                                stages |= y;
                            });
                        (accesses, stages)
                    }
                };
                accesses |= local_accesses;
                stages |= local_stages;
            }
            (accesses, stages)
        };
        let mut emitted_barriers = 0;

        for dep in incoming {
            let _this_external = !claims.graph.neighbors_directed(this_node, Incoming).any(|_| true);
            let prev_step = &claims.graph[dep];
            let prev_queue = get_queue_family_index_for_pass(&prev_step.pass_name);
            let _prev_queue_runtime = get_runtime_queue_family(prev_queue);
            let _prev_layout_in_renderpass = get_layout_from_renderpass(prev_step);

            if prev_step.step_name != this.step_name && prev_step.pass_name == this.pass_name {
                let (src_access, src_stage) = stage_flags(prev_step, false, true);
                let (dst_access, dst_stage) = stage_flags(this, true, true);

                assert!(
                    buffer != vk::Buffer::null(),
                    "Expected barrier call to provide resource expr {} {} {} {}",
                    resource_name,
                    step_name,
                    prev_step.pass_name,
                    this.pass_name
                );
                emitted_barriers += 1;
                acquire_buffer_barriers.push(
                    vk::BufferMemoryBarrier2KHR::builder()
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_stage_mask(src_stage)
                        .dst_stage_mask(dst_stage)
                        .src_access_mask(src_access)
                        .dst_access_mask(dst_access)
                        .buffer(buffer)
                        .size(vk::WHOLE_SIZE)
                        .build(),
                );
            }
        }
        assert!(
            emitted_barriers <= 1,
            "Resource claim {}.{} generated more than one acquire barrier, please disambiguate edges at runtime",
            resource_name,
            step_name
        );
    }

    #[allow(clippy::ptr_arg)]
    pub(crate) fn barrier_acceleration_structure(
        &self,
        renderer: &RenderFrame,
        data: &renderer_macro_lib::RendererInput,
        _command_buffer: vk::CommandBuffer,
        resource_name: &str,
        step_name: &str,
        acquire_memory_barriers: &mut Vec<vk::MemoryBarrier2KHR>,
        #[allow(unused)] release_memory_barriers: &mut Vec<vk::MemoryBarrier2KHR>,
    ) {
        let claims = self.active_resources.get(resource_name).unwrap();

        debug_assert!(matches!(claims.ty, ResourceDefinitionType::AccelerationStructure));

        let this_node = match claims.map.get(step_name) {
            Some(&node) => node,
            None => {
                return;
            }
        };
        let this = match claims.graph.node_weight(this_node) {
            Some(node) => node,
            None => {
                return;
            }
        };
        let get_queue_family_index_for_pass = |pass: &str| {
            data.async_passes
                .get(pass)
                .map(|async_pass| async_pass.queue)
                .expect("pass not found")
        };
        let get_runtime_queue_family = |ty: QueueFamily| match ty {
            QueueFamily::Graphics => renderer.device.graphics_queue_family,
            QueueFamily::Compute => renderer.device.compute_queue_family,
            QueueFamily::Transfer => renderer.device.transfer_queue_family,
        };
        let get_layout_from_renderpass = |step: &ResourceClaim| match step.usage {
            ResourceUsageKind::Attachment(ref renderpass) => data
                .renderpasses
                .get(renderpass)
                .and_then(|x| x.depth_stencil.as_ref().map(|d| &d.layout)),
            _ => None,
        };

        // let connected_components = petgraph::algo::connected_components(&claims.graph);
        // if connected_components != 1 {
        //     let msg = "resource claims graph must have one connected component".to_string();
        //     validation_errors.extend(quote!(compile_error!(#msg);));
        // }
        if petgraph::algo::is_cyclic_directed(&claims.graph) {
            panic!("resource claims graph is cyclic");
        }

        // neighbors with wraparound on both ends to make the code aware of dependencies in the prev/next
        // iterations of the graph
        let mut incoming = claims.graph.neighbors_directed(this_node, Incoming).peekable();
        let incoming = if incoming.peek().is_none() {
            incoming.chain(claims.graph.externals(Outgoing)).collect_vec()
        } else {
            incoming.collect_vec()
        };
        let mut outgoing = claims.graph.neighbors_directed(this_node, Outgoing).peekable();
        let _outgoing = if outgoing.peek().is_none() {
            outgoing.chain(claims.graph.externals(Incoming)).collect_vec()
        } else {
            outgoing.collect_vec()
        };

        let this_queue = get_queue_family_index_for_pass(&this.pass_name);
        let _this_queue_runtime = get_runtime_queue_family(this_queue);
        let _this_layout_in_renderpass = get_layout_from_renderpass(this);
        let _stage_flags = |step: &ResourceClaim, include_reads: bool, include_writes: bool| {
            use vk::{AccessFlags2KHR as A, PipelineStageFlags2KHR as S};
            let mut accesses = A::empty();
            let mut stages = S::empty();
            if step.reads && include_reads {
                let (local_accesses, local_stages) = match &step.usage {
                    ResourceUsageKind::VertexBuffer => (A::VERTEX_ATTRIBUTE_READ, S::VERTEX_ATTRIBUTE_INPUT),
                    ResourceUsageKind::IndexBuffer => (A::INDEX_READ, S::INDEX_INPUT),
                    ResourceUsageKind::Attachment(ref _renderpass) => {
                        // TODO: imprecise
                        (A::MEMORY_READ, S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS)
                    }
                    ResourceUsageKind::IndirectBuffer => (A::INDIRECT_COMMAND_READ, S::DRAW_INDIRECT),
                    ResourceUsageKind::TransferCopy => (A::TRANSFER_READ, S::COPY),
                    // TODO: sync validation is picky when just CLEAR from sync2 is used
                    ResourceUsageKind::TransferClear => (A::TRANSFER_READ, S::ALL_TRANSFER),
                    ResourceUsageKind::Descriptor(set_name, binding_name, pipeline_name) => {
                        let pipeline = data
                            .pipelines
                            .get(pipeline_name)
                            .expect("pipeline not found in barrier");
                        let set = data.descriptor_sets.get(set_name).expect("set not found in barrier");
                        let binding = set
                            .bindings
                            .iter()
                            .find(|candidate| candidate.name == *binding_name)
                            .expect("binding not found in barrier");
                        let pipeline_stages = pipeline.specific.stages();

                        let mut accesses = A::empty();
                        let mut stages = S::empty();

                        binding
                            .shader_stages
                            .iter()
                            .filter(|x| pipeline_stages.contains(*x))
                            .map(|stage| match stage.as_str() {
                                "COMPUTE" => (A::SHADER_READ, S::COMPUTE_SHADER),
                                _ => unimplemented!("barrier! descriptor stage {}", stage),
                            })
                            .for_each(|(x, y)| {
                                accesses |= x;
                                stages |= y;
                            });
                        (accesses, stages)
                    }
                };
                accesses |= local_accesses;
                stages |= local_stages;
            }
            if step.writes && include_writes {
                let (local_accesses, local_stages) = match &step.usage {
                    ResourceUsageKind::VertexBuffer => panic!("can't have a write access to VertexBuffer"),
                    ResourceUsageKind::IndexBuffer => panic!("can't have a write access to IndexBuffer"),
                    ResourceUsageKind::Attachment(ref _renderpass) => {
                        // TODO: imprecise
                        (A::MEMORY_WRITE, S::COLOR_ATTACHMENT_OUTPUT)
                    }
                    ResourceUsageKind::IndirectBuffer => panic!("can't have a write access to IndirectBuffer"),
                    ResourceUsageKind::TransferCopy => (A::TRANSFER_WRITE, S::COPY),
                    // TODO: sync validation is picky when just CLEAR from sync2 is used
                    ResourceUsageKind::TransferClear => (A::TRANSFER_WRITE, S::ALL_TRANSFER),
                    ResourceUsageKind::Descriptor(set_name, binding_name, pipeline_name) => {
                        let pipeline = data
                            .pipelines
                            .get(pipeline_name)
                            .expect("pipeline not found in barrier");
                        let set = data.descriptor_sets.get(set_name).expect("set not found in barrier");
                        let binding = set
                            .bindings
                            .iter()
                            .find(|candidate| candidate.name == *binding_name)
                            .expect("binding not found in barrier");
                        let pipeline_stages = pipeline.specific.stages();

                        let mut accesses = A::empty();
                        let mut stages = S::empty();

                        binding
                            .shader_stages
                            .iter()
                            .filter(|x| pipeline_stages.contains(*x))
                            .map(|stage| match stage.as_str() {
                                "COMPUTE" => (A::SHADER_WRITE, S::COMPUTE_SHADER),
                                _ => unimplemented!("barrier! descriptor stage {}", stage),
                            })
                            .for_each(|(x, y)| {
                                accesses |= x;
                                stages |= y;
                            });
                        (accesses, stages)
                    }
                };
                accesses |= local_accesses;
                stages |= local_stages;
            }
            (accesses, stages)
        };
        let mut emitted_barriers = 0;

        for dep in incoming {
            let _this_external = !claims.graph.neighbors_directed(this_node, Incoming).any(|_| true);
            let prev_step = &claims.graph[dep];
            let prev_queue = get_queue_family_index_for_pass(&prev_step.pass_name);
            let _prev_queue_runtime = get_runtime_queue_family(prev_queue);
            let _prev_layout_in_renderpass = get_layout_from_renderpass(prev_step);

            if prev_step.pass_name == this.pass_name {
                emitted_barriers += 1;
                acquire_memory_barriers.push(
                    vk::MemoryBarrier2KHR::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)
                        .build(),
                );
            }
        }
        assert!(
            emitted_barriers <= 1,
            "Resource claim {}.{} generated more than one acquire barrier, please disambiguate edges at runtime",
            resource_name,
            step_name
        );
    }

    pub(crate) fn barrier_image(
        &self,
        renderer: &RenderFrame,
        swapchain_index_map: &SwapchainIndexToFrameNumber,
        image_index: &ImageIndex,
        data: &renderer_macro_lib::RendererInput,
        _command_buffer: vk::CommandBuffer,
        resource_name: &str,
        step_name: &str,
        image: vk::Image,
        acquire_image_barriers: &mut Vec<vk::ImageMemoryBarrier2KHR>,
        release_image_barriers: &mut Vec<vk::ImageMemoryBarrier2KHR>,
    ) {
        let claims = self.active_resources.get(resource_name).unwrap();
        let next_claims = self.upcoming_resources.get(resource_name).unwrap();
        let aspect = match claims.ty {
            ResourceDefinitionType::Image { ref aspect, .. } => aspect,
            _ => panic!(),
        };

        let this_node = match claims.map.get(step_name) {
            Some(&node) => node,
            None => {
                return;
            }
        };
        let this = match claims.graph.node_weight(this_node) {
            Some(node) => node,
            None => {
                return;
            }
        };
        let get_queue_family_index_for_pass = |pass: &str| {
            data.async_passes
                .get(pass)
                .map(|async_pass| async_pass.queue)
                .expect("pass not found")
        };
        let get_runtime_queue_family = |ty: QueueFamily| match ty {
            QueueFamily::Graphics => renderer.device.graphics_queue_family,
            QueueFamily::Compute => renderer.device.compute_queue_family,
            QueueFamily::Transfer => renderer.device.transfer_queue_family,
        };
        let get_layout_from_renderpass = |step: &ResourceClaim| match step.usage {
            ResourceUsageKind::Attachment(ref renderpass) => data
                .renderpasses
                .get(renderpass)
                .and_then(|x| x.depth_stencil.as_ref().map(|d| &d.layout)),
            _ => None,
        };

        // let connected_components = petgraph::algo::connected_components(&claims.graph);
        // if connected_components != 1 {
        //     let msg = "resource claims graph must have one connected component".to_string();
        //     validation_errors.extend(quote!(compile_error!(#msg);));
        // }
        if petgraph::algo::is_cyclic_directed(&claims.graph) {
            panic!("resource claims graph is cyclic");
        }

        // neighbors with wraparound on both ends to make the code aware of dependencies in the prev/next
        // iterations of the graph
        let mut incoming = claims.graph.neighbors_directed(this_node, Incoming).peekable();
        let incoming = if incoming.peek().is_none() {
            incoming.chain(claims.graph.externals(Outgoing)).collect_vec()
        } else {
            incoming.collect_vec()
        };
        let mut outgoing = claims
            .graph
            .neighbors_directed(this_node, Outgoing)
            .map(|ix| &claims.graph[ix])
            .peekable();
        let outgoing = if outgoing.peek().is_none() {
            outgoing
                .chain(next_claims.graph.externals(Incoming).map(|ix| &next_claims.graph[ix]))
                .collect_vec()
        } else {
            outgoing.collect_vec()
        };

        let this_queue = get_queue_family_index_for_pass(&this.pass_name);
        let this_queue_runtime = get_runtime_queue_family(this_queue);
        let this_layout_in_renderpass = get_layout_from_renderpass(this);
        let _stage_flags = |step: &ResourceClaim, include_reads: bool, include_writes: bool| {
            use vk::{AccessFlags2KHR as A, PipelineStageFlags2KHR as S};
            let mut accesses = A::empty();
            let mut stages = S::empty();
            if step.reads && include_reads {
                let (local_accesses, local_stages) = match &step.usage {
                    ResourceUsageKind::VertexBuffer => (A::VERTEX_ATTRIBUTE_READ, S::VERTEX_ATTRIBUTE_INPUT),
                    ResourceUsageKind::IndexBuffer => (A::INDEX_READ, S::INDEX_INPUT),
                    ResourceUsageKind::Attachment(ref _renderpass) => {
                        // TODO: imprecise
                        (A::MEMORY_READ, S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS)
                    }
                    ResourceUsageKind::IndirectBuffer => (A::INDIRECT_COMMAND_READ, S::DRAW_INDIRECT),
                    ResourceUsageKind::TransferCopy => (A::TRANSFER_READ, S::COPY),
                    // TODO: sync validation is picky when just CLEAR from sync2 is used
                    ResourceUsageKind::TransferClear => (A::TRANSFER_READ, S::ALL_TRANSFER),
                    ResourceUsageKind::Descriptor(set_name, binding_name, pipeline_name) => {
                        let pipeline = data
                            .pipelines
                            .get(pipeline_name)
                            .expect("pipeline not found in barrier");
                        let set = data.descriptor_sets.get(set_name).expect("set not found in barrier");
                        let binding = set
                            .bindings
                            .iter()
                            .find(|candidate| candidate.name == *binding_name)
                            .expect("binding not found in barrier");
                        let pipeline_stages = pipeline.specific.stages();

                        let mut accesses = A::empty();
                        let mut stages = S::empty();

                        binding
                            .shader_stages
                            .iter()
                            .filter(|x| pipeline_stages.contains(*x))
                            .map(|stage| match stage.as_str() {
                                "COMPUTE" => (A::SHADER_READ, S::COMPUTE_SHADER),
                                _ => unimplemented!("barrier! descriptor stage {}", stage),
                            })
                            .for_each(|(x, y)| {
                                accesses |= x;
                                stages |= y;
                            });
                        (accesses, stages)
                    }
                };
                accesses |= local_accesses;
                stages |= local_stages;
            }
            if step.writes && include_writes {
                let (local_accesses, local_stages) = match &step.usage {
                    ResourceUsageKind::VertexBuffer => panic!("can't have a write access to VertexBuffer"),
                    ResourceUsageKind::IndexBuffer => panic!("can't have a write access to IndexBuffer"),
                    ResourceUsageKind::Attachment(ref _renderpass) => {
                        // TODO: imprecise
                        (A::MEMORY_WRITE, S::COLOR_ATTACHMENT_OUTPUT)
                    }
                    ResourceUsageKind::IndirectBuffer => panic!("can't have a write access to IndirectBuffer"),
                    ResourceUsageKind::TransferCopy => (A::TRANSFER_WRITE, S::COPY),
                    // TODO: sync validation is picky when just CLEAR from sync2 is used
                    ResourceUsageKind::TransferClear => (A::TRANSFER_WRITE, S::ALL_TRANSFER),
                    ResourceUsageKind::Descriptor(set_name, binding_name, pipeline_name) => {
                        let pipeline = data
                            .pipelines
                            .get(pipeline_name)
                            .expect("pipeline not found in barrier");
                        let set = data.descriptor_sets.get(set_name).expect("set not found in barrier");
                        let binding = set
                            .bindings
                            .iter()
                            .find(|candidate| candidate.name == *binding_name)
                            .expect("binding not found in barrier");
                        let pipeline_stages = pipeline.specific.stages();

                        let mut accesses = A::empty();
                        let mut stages = S::empty();

                        binding
                            .shader_stages
                            .iter()
                            .filter(|x| pipeline_stages.contains(*x))
                            .map(|stage| match stage.as_str() {
                                "COMPUTE" => (A::SHADER_WRITE, S::COMPUTE_SHADER),
                                _ => unimplemented!("barrier! descriptor stage {}", stage),
                            })
                            .for_each(|(x, y)| {
                                accesses |= x;
                                stages |= y;
                            });
                        (accesses, stages)
                    }
                };
                accesses |= local_accesses;
                stages |= local_stages;
            }
            (accesses, stages)
        };
        let convert_layout = |layout: &str| {
            use vk::ImageLayout as L;
            match layout {
                "COLOR_ATTACHMENT_OPTIMAL" => L::COLOR_ATTACHMENT_OPTIMAL,
                "DEPTH_STENCIL_ATTACHMENT_OPTIMAL" => L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                "DEPTH_STENCIL_READ_ONLY_OPTIMAL" => L::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                "DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL" => L::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
                "GENERAL" => L::GENERAL,
                "TRANSFER_SRC_OPTIMAL" => L::TRANSFER_SRC_OPTIMAL,
                "TRANSFER_DST_OPTIMAL" => L::TRANSFER_DST_OPTIMAL,
                "PRESENT_SRC_KHR" => L::PRESENT_SRC_KHR,
                "UNDEFINED" => L::UNDEFINED,
                _ => todo!("unknown layout {}", layout),
            }
        };
        let mut emitted_barriers = 0;

        for dep in incoming {
            let this_external = !claims.graph.neighbors_directed(this_node, Incoming).any(|_| true);
            let prev_step = &claims.graph[dep];
            let prev_queue = get_queue_family_index_for_pass(&prev_step.pass_name);
            let prev_queue_runtime = get_runtime_queue_family(prev_queue);
            let prev_layout_in_renderpass = get_layout_from_renderpass(prev_step);

            let _same_pass = prev_step.pass_name == this.pass_name;

            assert!(
                image != vk::Image::null(),
                "Expected barrier call to provide resource expr"
            );
            let this_layout = this.layout.as_ref().or(this_layout_in_renderpass).unwrap_or_else(|| {
                panic!(
                    "did not specify desired image layout {}.{}",
                    &this.pass_name, &this.step_name
                )
            });
            let prev_layout = prev_step
                .layout
                .as_ref()
                .or(prev_layout_in_renderpass)
                .unwrap_or_else(|| {
                    panic!(
                        "did not specify desired image layout {}.{}",
                        &prev_step.pass_name, &prev_step.step_name
                    )
                });
            let this_layout = convert_layout(this_layout);
            let prev_layout = if this.clobber {
                vk::ImageLayout::UNDEFINED
            } else {
                convert_layout(prev_layout)
            };
            let aspect = match aspect.as_str() {
                "COLOR" => vk::ImageAspectFlags::COLOR,
                "DEPTH" => vk::ImageAspectFlags::DEPTH,
                _ => todo!("unimplemented Image aspect"),
            };
            let mut barrier = vk::ImageMemoryBarrier2KHR::builder()
                .src_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(aspect)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                )
                .image(image);
            if this_external
                && (renderer.frame_number == 1
                    || claims.double_buffered && swapchain_index_map.map[image_index.0 as usize] == 0)
            {
                // TODO: improve this with explicit clobber?
                tracing::Span::current().record("src_layout", &"UNDEFINED");
                tracing::Span::current().record("dst_layout", &format_args!("{this_layout:?}"));
                tracing::Span::current().record("src_queue_family", &"QUEUE_FAMILY_IGNORED");
                tracing::Span::current().record("dst_queue_family", &"QUEUE_FAMILY_IGNORED");
                barrier = barrier
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(this_layout);
            } else {
                if prev_layout != this_layout {
                    tracing::Span::current().record("src_layout", &format_args!("{prev_layout:?}"));
                    tracing::Span::current().record("dst_layout", &format_args!("{this_layout:?}"));
                }
                if prev_queue_runtime != this_queue_runtime {
                    tracing::Span::current().record("src_queue_family", &prev_queue_runtime);
                    tracing::Span::current().record("dst_queue_family", &this_queue_runtime);
                }
                barrier = barrier
                    .src_queue_family_index(prev_queue_runtime)
                    .dst_queue_family_index(this_queue_runtime)
                    .old_layout(prev_layout)
                    .new_layout(this_layout);
            }
            emitted_barriers += 1;
            tracing::debug!("emitted acquire barrier");
            acquire_image_barriers.push(barrier.build());
        }
        assert!(
            emitted_barriers <= 1,
            "Resource claim {}.{} generated more than one acquire barrier, please disambiguate edges at runtime",
            resource_name,
            step_name
        );
        emitted_barriers = 0;
        for next_step in outgoing {
            let next_queue = get_queue_family_index_for_pass(&next_step.pass_name);
            let next_queue_runtime = get_runtime_queue_family(next_queue);
            let next_layout_in_renderpass = get_layout_from_renderpass(next_step);

            if this_queue == next_queue || this_queue_runtime == next_queue_runtime {
                continue;
            }

            assert!(
                image != vk::Image::null(),
                "Expected barrier call to provide resource expr"
            );
            let this_layout = this.layout.as_ref().or(this_layout_in_renderpass).unwrap_or_else(|| {
                panic!(
                    "did not specify desired image layout {}.{}",
                    &this.pass_name, &this.step_name
                )
            });
            let next_layout = next_step
                .layout
                .as_ref()
                .or(next_layout_in_renderpass)
                .unwrap_or_else(|| {
                    panic!(
                        "did not specify desired image layout {}.{}",
                        &next_step.pass_name, &next_step.step_name
                    )
                });
            let this_layout = convert_layout(this_layout);
            let next_layout = convert_layout(next_layout);
            let aspect = match aspect.as_str() {
                "COLOR" => vk::ImageAspectFlags::COLOR,
                "DEPTH" => vk::ImageAspectFlags::DEPTH,
                _ => todo!("unimplemented Image aspect"),
            };
            emitted_barriers += 1;
            if this_layout != next_layout {
                tracing::Span::current().record("src_layout", &format_args!("{this_layout:?}"));
                tracing::Span::current().record("dst_layout", &format_args!("{next_layout:?}"));
            }
            if this_queue_runtime != next_queue_runtime {
                tracing::Span::current().record("src_queue_family", &this_queue_runtime);
                tracing::Span::current().record("dst_queue_family", &next_queue_runtime);
            }
            tracing::debug!("emitted release barrier");
            release_image_barriers.push(
                vk::ImageMemoryBarrier2KHR::builder()
                    .src_queue_family_index(this_queue_runtime)
                    .dst_queue_family_index(next_queue_runtime)
                    .old_layout(this_layout)
                    .new_layout(next_layout)
                    .src_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(aspect)
                            .level_count(1)
                            .layer_count(1)
                            .build(),
                    )
                    .image(image)
                    .build(),
            );
        }
        assert!(
            emitted_barriers <= 1,
            "Resource claim {}.{} generated more than one release barrier, please disambiguate edges at runtime",
            resource_name,
            step_name
        );
    }
}

pub(crate) fn setup_submissions(
    mut submissions: ResMut<Submissions>,
    current_config: Res<RuntimeConfiguration>,
    future_configs: Res<FutureRuntimeConfiguration>,
) {
    let Submissions {
        ref mut remaining,
        ref mut upcoming_remaining,
        ref mut extra_signals,
        ref mut upcoming_extra_signals,
        ref mut active_graph,
        ref mut upcoming_graph,
        ref mut active_resources,
        ref mut upcoming_resources,
        ref mut virtual_queue_indices,
        ref mut upcoming_virtual_queue_indices,
    } = *submissions;

    if *current_config == future_configs.0[0] {
        let remaining = remaining.get_mut();
        let upcoming_remaining = upcoming_remaining.get_mut();
        assert_eq!(remaining.node_count(), 0);
        assert_eq!(remaining.edge_count(), 0);
        remaining.clone_from(upcoming_remaining);

        extra_signals.clone_from(upcoming_extra_signals);
        active_resources.clone_from(upcoming_resources);
        active_graph.clone_from(upcoming_graph);
        virtual_queue_indices.clone_from(upcoming_virtual_queue_indices);
        return;
    }

    let input = &RENDERER_INPUT;

    let mut graph2 = input.dependency_graph.clone();

    // Use the config for the next frame to prepare the upcoming plan
    let runtime_config = &future_configs.0[0];
    let switches: HashMap<String, bool> = [
        ("FREEZE_CULLING".to_string(), runtime_config.freeze_culling),
        ("DEBUG_AABB".to_string(), runtime_config.debug_aabbs),
        ("RT".to_string(), runtime_config.rt),
        ("REFERENCE_RT".to_string(), runtime_config.reference_rt),
    ]
    .into_iter()
    .collect();

    *extra_signals = take(upcoming_extra_signals);
    *active_resources = take(upcoming_resources);
    *active_graph = take(upcoming_graph);
    *upcoming_resources = input.resources.clone();
    {
        let remaining = remaining.get_mut();
        let upcoming_remaining = upcoming_remaining.get_mut();
        debug_assert!(remaining.node_weights().all(|x| *x == TrackedSubmission::Submitted));
        *remaining = take(upcoming_remaining);
    }

    let eval_cond = |c: &Conditional| {
        c.conditions.iter().all(|cond| {
            let mut eval = switches.get(&cond.switch_name).cloned().unwrap_or(false);
            if cond.neg {
                eval = !eval;
            }
            eval
        })
    };

    // Stage 1: Cull resource graphs by evaluating conditionals
    for claims in upcoming_resources.values_mut() {
        claims
            .map
            // Drain inactive resource steps
            .drain_filter(|_step_name, &mut step_ix| {
                let step = &claims.graph[step_ix];

                !eval_cond(&step.conditional) || !graph2.node_references().any(|(_, name)| **name == step.pass_name)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|(_step_name, step_ix)| {
                // dbg!(step_name);
                claims.graph.remove_node(step_ix).unwrap();
            });
    }

    // Stage 2: Remove resource claims whose results are not read later
    for (resource_name, claims) in upcoming_resources.iter_mut() {
        for u in claims.graph.node_indices().collect::<Vec<_>>() {
            let mut dfs = petgraph::visit::Dfs::new(&claims.graph, u);
            // dfs.next(&claims.graph); // skip self
            let mut read_back = false;
            while let Some(candidate) = dfs.next(&claims.graph) {
                if claims.graph[candidate].reads & !claims.graph[candidate].writes {
                    read_back = true;
                    break;
                }
            }

            if !read_back {
                println!("not read back: {}.{}", resource_name, &claims.graph[u].step_name);
                claims.graph.remove_node(u);
            }
        }
    }
    // Stage 3: Remove passes that don't modify (by writing) any active resource
    for pass_name in input.async_passes.keys() {
        if pass_name != "PresentationAcquire"
            && pass_name != "Present"
            && !upcoming_resources.values().any(|resource| {
                resource
                    .graph
                    .node_weights()
                    .any(|claim| claim.pass_name == *pass_name && claim.writes)
            })
        {
            let to_remove = graph2
                .node_references()
                .find(|(_, name)| *name == pass_name)
                .map(|(ix, _)| ix)
                .unwrap();
            debug_assert!(graph2.remove_node(to_remove).is_some());
        }
    }
    for (resource_name, claims) in upcoming_resources.iter_mut() {
        for u in claims.graph.node_indices().collect::<Vec<_>>() {
            let mut dfs = petgraph::visit::Dfs::new(&claims.graph, u);
            // dfs.next(&claims.graph); // skip self
            let mut read_back = false;
            while let Some(candidate) = dfs.next(&claims.graph) {
                if claims.graph[candidate].reads & !claims.graph[candidate].writes {
                    read_back = true;
                    break;
                }
            }

            if !read_back {
                println!("not read back: {}.{}", resource_name, &claims.graph[u].step_name);
                claims.graph.remove_node(u);
            }
        }
    }
    for pass_name in input.async_passes.keys() {
        if pass_name != "PresentationAcquire"
            && pass_name != "Present"
            && !upcoming_resources.values().any(|resource| {
                resource
                    .graph
                    .node_weights()
                    .any(|claim| claim.pass_name == *pass_name && claim.writes)
            })
        {
            graph2
                .node_references()
                .find(|(_, name)| *name == pass_name)
                .map(|(ix, _)| ix)
                .into_iter()
                .for_each(|ix| drop(graph2.remove_node(ix)));
        }
    }

    // Stage 4: Cull the active graph from stuff that does not lead to Present
    let main_node = NodeIndex::from(frame_graph::Present::INDEX);
    let _presentation_acquire = NodeIndex::from(frame_graph::PresentationAcquire::INDEX);
    for u in graph2.node_indices().collect::<Vec<_>>() {
        if !has_path_connecting(&graph2, u, main_node, None) {
            // dbg!(&graph2[u]);
            graph2.remove_node(u);
        }
    }
    // Stage 5: Remove resource claims again now that even more passes have been disabled
    for claims in upcoming_resources.values_mut() {
        for u in claims.graph.node_indices().collect::<Vec<_>>() {
            let step = &claims.graph[u];
            if !graph2.node_references().any(|(_, name)| **name == step.pass_name) {
                dbg!(&step.pass_name);
                claims.graph.remove_node(u);
                break;
            }
        }
    }
    // Stage 6: make up for the missing semaphore signals by pushing them to the next submission after
    // the original one (that is inactive)
    for node in input.dependency_graph.node_indices() {
        // If the node is missing from the active graph, find the first active downstream node to pick up
        // extra_signals
        if !graph2.contains_node(node) {
            let mut dfs = petgraph::visit::Dfs::new(&input.dependency_graph, node);
            dfs.next(&input.dependency_graph); // skip self
            while let Some(candidate) = dfs.next(&input.dependency_graph) {
                if graph2.contains_node(candidate) {
                    upcoming_extra_signals.entry(candidate).or_insert(vec![]).push(node);
                    break;
                }
            }
        }
    }

    // Stage 7: Perform a transitive reduction to minimize the number of semaphores that passes need to
    // wait on
    renderer_macro_lib::transitive_reduction_stable(&mut graph2);

    // dbg!(petgraph::dot::Dot::with_config(
    //     &graph2.map(|_, node_ident| node_ident.to_string(), |_, _| ""),
    //     &[petgraph::dot::Config::EdgeNoLabel]
    // ));

    let graph = upcoming_remaining.get_mut();

    *graph = graph2.map(|_, _| TrackedSubmission::Preparing, |_, _| ());
    graph.remove_node(NodeIndex::from(frame_graph::PresentationAcquire::INDEX));

    *upcoming_virtual_queue_indices = {
        let mut mapping = HashMap::new();
        let toposort = petgraph::algo::toposort(&*graph, None).unwrap();
        let mut compute_indices = |family| {
            let toposort_compute = toposort.iter().filter(|ix| {
                let name = &graph2[**ix];
                input
                    .async_passes
                    .get(name)
                    .map(|async_pass| async_pass.queue == family)
                    .unwrap()
            });

            for (queue_ix, &node_ix) in toposort_compute.enumerate() {
                assert!(mapping.insert(node_ix, queue_ix).is_none());
            }
        };
        compute_indices(QueueFamily::Graphics);
        compute_indices(QueueFamily::Compute);
        compute_indices(QueueFamily::Transfer);
        mapping
    };

    *upcoming_graph = graph2;
}

fn submit_pending(
    renderer: Res<RenderFrame>,
    submissions: Res<Submissions>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
) {
    let graph = submissions.remaining.lock();
    update_submissions(
        &renderer,
        &submissions,
        graph,
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
    );
}

fn update_submissions(
    renderer: &RenderFrame,
    submissions: &Submissions,
    mut graph: MutexGuard<StableDiGraph<TrackedSubmission, ()>>,
    #[cfg(feature = "crash_debugging")] crash_buffer: &CrashBuffer,
) {
    use petgraph::visit::Reversed;
    scope!("rendering::update_submissions");

    let Submissions {
        ref extra_signals,
        ref active_graph,
        ref virtual_queue_indices,
        ..
    } = submissions;
    let input = &RENDERER_INPUT;

    // helpers that will be used to detect hazards
    let runtime_queue_family = |q| match q {
        QueueFamily::Graphics => renderer.device.graphics_queue_family,
        QueueFamily::Compute => renderer.device.compute_queue_family,
        QueueFamily::Transfer => renderer.device.transfer_queue_family,
    };
    let effective_ix = |q, virt_ix| match q {
        QueueFamily::Graphics => renderer.device.graphics_queue_virtualized_to_effective_ix(virt_ix),
        QueueFamily::Compute => renderer.device.compute_queue_virtualized_to_effective_ix(virt_ix),
        QueueFamily::Transfer => renderer.device.transfer_queue_virtualized_to_effective_ix(virt_ix),
    };

    let mut roots = graph.externals(Outgoing).collect::<Vec<_>>();
    debug_assert!(roots.len() == 1);
    let target = roots.pop().unwrap();

    loop {
        let mut node = None;
        let mut bfs = Bfs::new(Reversed(&*graph), target);

        // Find something we can submit
        while let Some(candidate) = bfs.next(Reversed(&*graph)) {
            match graph[candidate] {
                TrackedSubmission::Executable(_) => {}
                _ => continue,
            }
            let candidate_queue_family = input.async_passes[&active_graph[candidate]].queue;
            let candidate_runtime_queue = runtime_queue_family(candidate_queue_family);
            let candidate_effective_ix = effective_ix(candidate_queue_family, virtual_queue_indices[&candidate]);
            let mut bfs = Bfs::new(Reversed(&*graph), candidate);
            bfs.next(Reversed(&*graph)); // skip self
            let mut compatible_ancestors = true;
            while let Some(ancestor) = bfs.next(Reversed(&*graph)) {
                if graph[ancestor] == TrackedSubmission::Submitted {
                    continue;
                }

                // OOO submits are legal[1] in Vulkan, this code just makes sure that we don't have a circular
                // dependency within the same runtime queue, which would deadlock that queue. The Nvidia driver
                // exposes virtual queues and makes room for OOO submits, but the theoretical
                // benefits are immediately negated in practice and it ends up being slower if we submit work
                // that waits on a timeline semaphore before a matching signal operation is submitted. On top
                // of that, vkQueuePresentKHR() gets 10x slower as well, from ~100us to ~1ms.
                //
                // [1]: 6.6. Queue Forward Progress <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap6.html#commandbuffers-submission-progress>
                if cfg!(feature = "submit_ooo") {
                    let ancestor_queue_family = input.async_passes[&active_graph[ancestor]].queue;
                    // TODO: are there more opportunities?
                    let ancestor_runtime_queue = runtime_queue_family(ancestor_queue_family);
                    let ancestor_effective_ix = effective_ix(ancestor_queue_family, virtual_queue_indices[&ancestor]);
                    if candidate_runtime_queue == ancestor_runtime_queue
                        && candidate_effective_ix == ancestor_effective_ix
                    {
                        compatible_ancestors = false;
                        break;
                    }
                } else {
                    compatible_ancestors = false;
                    break;
                }
            }
            if compatible_ancestors {
                node = Some(candidate);
                break;
            }
        }

        let node = match node {
            Some(node) => node,
            None => break,
        };
        let cb = match graph[node] {
            TrackedSubmission::Executable(ref mut cb) => cb.take(),
            _ => {
                if cfg!(debug_assertions) {
                    unreachable!()
                } else {
                    // SAFETY: should be fine as we tested for Executable while searching for the candidate in the loop
                    // above
                    unsafe { unreachable_unchecked() }
                }
            }
        };

        *graph.node_weight_mut(node).unwrap() = TrackedSubmission::Submitting;
        MutexGuard::unlocked(&mut graph, || {
            scope!("unlocked");
            let pass_name = &active_graph[node];
            let queue_family = input.async_passes[pass_name].queue;
            let virtualized_ix = virtual_queue_indices[&node];
            let queue = match queue_family {
                QueueFamily::Graphics => renderer.device.graphics_queue_virtualized(virtualized_ix),
                QueueFamily::Compute => renderer.device.compute_queue_virtualized(virtualized_ix),
                QueueFamily::Transfer => renderer.device.transfer_queue_virtualized(virtualized_ix),
            };
            let buf = &mut [vk::CommandBuffer::null()];
            let command_buffers: &[vk::CommandBuffer] = match cb {
                Some(cmd) => {
                    buf[0] = cmd;
                    buf
                }
                None => &[],
            };
            let submit_result = {
                let wait_semaphores = active_graph
                    .edges_directed(node, Incoming)
                    .map(|edge| {
                        let incoming_pass_name = &active_graph[edge.source()];
                        let &(semaphore_ix, stage_ix) =
                            input.timeline_semaphore_mapping.get(incoming_pass_name).unwrap();
                        let &cycle_value = input.timeline_semaphore_cycles.get(&semaphore_ix).unwrap();
                        let value = match edge.weight() {
                            DependencyType::SameFrame => renderer.frame_number * cycle_value + stage_ix,
                            DependencyType::LastFrame => (renderer.frame_number - 1) * cycle_value + stage_ix,
                            DependencyType::LastAccess => todo!(),
                        };

                        vk::SemaphoreSubmitInfoKHR::builder()
                            .semaphore(renderer.auto_semaphores.0[semaphore_ix].handle)
                            .value(value)
                            .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                            .build()
                    })
                    .collect::<Vec<_>>();
                let mut signal_semaphores = HashMap::new();
                {
                    // Signal this submission
                    let &(semaphore_ix, stage_ix) = input.timeline_semaphore_mapping.get(pass_name).unwrap();
                    let &cycle_value = input.timeline_semaphore_cycles.get(&semaphore_ix).unwrap();

                    signal_semaphores.insert(semaphore_ix, renderer.frame_number * cycle_value + stage_ix);
                };
                // Signal extra submissions that are inactive in the current configuration
                for &extra in extra_signals.get(&node).unwrap_or(&vec![]) {
                    let extra_pass_name = &input.dependency_graph[extra];
                    let &(semaphore_ix, stage_ix) = input.timeline_semaphore_mapping.get(extra_pass_name).unwrap();
                    let &cycle_value = input.timeline_semaphore_cycles.get(&semaphore_ix).unwrap();

                    let value = renderer.frame_number * cycle_value + stage_ix;
                    signal_semaphores
                        .entry(semaphore_ix)
                        .and_modify(|existing| {
                            *existing = max(*existing, value);
                        })
                        .or_insert(value);
                }
                let signal_semaphore_infos = signal_semaphores
                    .iter()
                    .map(|(&semaphore_ix, &value)| {
                        vk::SemaphoreSubmitInfoKHR::builder()
                            .semaphore(renderer.auto_semaphores.0[semaphore_ix].handle)
                            .value(value)
                            .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                            .build()
                    })
                    .collect::<Vec<_>>();
                let command_buffer_infos = command_buffers
                    .iter()
                    .map(|cb| vk::CommandBufferSubmitInfoKHR::builder().command_buffer(*cb).build())
                    .collect::<Vec<_>>();

                let submit = vk::SubmitInfo2KHR::builder()
                    .wait_semaphore_infos(&wait_semaphores)
                    .signal_semaphore_infos(&signal_semaphore_infos)
                    .command_buffer_infos(&command_buffer_infos)
                    .build();

                let queue = queue.lock();
                scope!("queue locked");

                if tracing::enabled!(target: "renderer", tracing::Level::INFO) {
                    let mut str = format!("Submitting {pass_name} to queue family {queue_family:?}[{virtualized_ix}]");
                    for edge in active_graph.edges_directed(node, Incoming) {
                        let incoming_pass_name = &active_graph[edge.source()];
                        let &(semaphore_ix, stage_ix) =
                            input.timeline_semaphore_mapping.get(incoming_pass_name).unwrap();
                        let &cycle_value = input.timeline_semaphore_cycles.get(&semaphore_ix).unwrap();
                        let value = match edge.weight() {
                            DependencyType::SameFrame => renderer.frame_number * cycle_value + stage_ix,
                            DependencyType::LastFrame => (renderer.frame_number - 1) * cycle_value + stage_ix,
                            DependencyType::LastAccess => todo!(),
                        };

                        str += &format!("\nWaiting: AutoSemaphores[{semaphore_ix}] <- {value}");
                    }

                    for (semaphore_ix, value) in signal_semaphores.iter() {
                        str += &format!("\nSignaling: AutoSemaphores[{semaphore_ix}] <- {value}");
                    }
                    tracing::info!(target: "renderer", "{str}");
                }

                unsafe {
                    scope!("vk::QueueSubmit");
                    renderer
                        .device
                        .synchronization2
                        .queue_submit2(*queue, &[submit], vk::Fence::null())
                }
            };

            match submit_result {
                Ok(()) => {}
                Err(res) => {
                    #[cfg(feature = "crash_debugging")]
                    crash_buffer.dump(&renderer.device);
                    panic!("Submit failed, frame={}, error={:?}", renderer.frame_number, res);
                }
            };
        });
        // we can clean it up now to unlock downstream submissions
        *graph.node_weight_mut(node).unwrap() = TrackedSubmission::Submitted;
    }
}
