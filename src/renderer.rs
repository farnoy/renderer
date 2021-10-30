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
    marker::PhantomData,
    mem::{size_of, take},
    sync::Arc,
};

use ash::vk;
use bevy_ecs::{component::Component, prelude::*};
use microprofile::scope;
use num_traits::ToPrimitive;
use parking_lot::{Mutex, MutexGuard};
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use smallvec::{smallvec, SmallVec};
use static_assertions::const_assert_eq;

#[cfg(not(feature = "no_profiling"))]
pub(crate) use self::helpers::MP_INDIAN_RED;
#[cfg(feature = "crash_debugging")]
pub(crate) use self::systems::crash_debugging::CrashBuffer;
#[cfg(feature = "shader_reload")]
pub(crate) use self::systems::shader_reload::{reload_shaders, ReloadedShaders, ShaderReload};
use self::{
    device::{
        Buffer, DescriptorPool, Device, DoubleBuffered, Framebuffer, Image, ImageView, RenderPass, Sampler, Shader,
        StaticBuffer, StrictCommandPool, StrictRecordingCommandBuffer, VmaMemoryUsage,
    },
    systems::{cull_pipeline::cull_set, depth_pass::depth_only_pass, shadow_mapping::shadow_map_set},
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

#[derive(Clone)]
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

#[derive(Debug, Default)]
pub(crate) struct DrawIndex(pub(crate) u32);

// TODO: rename
pub(crate) struct RenderFrame {
    pub(crate) instance: Arc<Instance>,
    pub(crate) device: Device,
    pub(crate) auto_semaphores: AutoSemaphores,
    pub(crate) frame_number: u64,
    pub(crate) buffer_count: usize,
}

#[derive(Clone)]
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
    10 * 30_000
);
const_assert_eq!(
    size_of::<frame_graph::VkDrawIndexedIndirectCommand>(),
    size_of::<vk::DrawIndexedIndirectCommand>(),
);

// https://users.rust-lang.org/t/can-i-conveniently-compile-bytes-into-a-rust-program-with-a-specific-alignment/24049/2
#[repr(C)] // guarantee 'bytes' comes after '_align'
struct AlignedAs<Align, Bytes: ?Sized> {
    _align: [Align; 0],
    bytes: Bytes,
}

macro_rules! include_bytes_align_as {
    ($align_ty:ty, $path:literal) => {{
        // const block expression to encapsulate the static
        use crate::renderer::AlignedAs;

        // this assignment is made possible by CoerceUnsized
        static ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($path),
        };

        &ALIGNED.bytes
    }};
}

pub(crate) use include_bytes_align_as;

renderer_macros::define_timelines! {}

renderer_macros::define_frame! {
    frame_graph {
        attachments {
            Color,
            PresentSurface,
            Depth,
            ShadowMapAtlas
        }
        formats {
            dyn,
            dyn,
            D16_UNORM,
            D16_UNORM
        }
        samples {
            4,
            1,
            4,
            1
        }
        passes {
            ShadowMapping {
                attachments [ShadowMapAtlas]
                layouts {
                    ShadowMapAtlas load DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
                                => store DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
                }
                subpasses {
                    ShadowMappingMain {
                        depth_stencil { ShadowMapAtlas => DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
                    }
                }
            },
            DepthOnly {
                attachments [Depth]
                layouts {
                    Depth clear UNDEFINED => store DEPTH_STENCIL_READ_ONLY_OPTIMAL
                }
                subpasses {
                    Main {
                        depth_stencil { Depth => DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
                    }
                }
            },
            Main {
                attachments [Color, Depth, PresentSurface]
                layouts {
                    Depth load DEPTH_STENCIL_READ_ONLY_OPTIMAL => discard DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    Color clear UNDEFINED => discard COLOR_ATTACHMENT_OPTIMAL,
                    PresentSurface clear UNDEFINED => store PRESENT_SRC_KHR
                }
                subpasses {
                    GltfPass {
                        color [Color => COLOR_ATTACHMENT_OPTIMAL]
                        depth_stencil { Depth => DEPTH_STENCIL_READ_ONLY_OPTIMAL }
                    },
                    GuiPass {
                        color [Color => COLOR_ATTACHMENT_OPTIMAL]
                        resolve [PresentSurface => COLOR_ATTACHMENT_OPTIMAL]
                    }
                }
                dependencies {
                    GltfPass => GuiPass
                        COLOR_ATTACHMENT_OUTPUT => COLOR_ATTACHMENT_OUTPUT
                        COLOR_ATTACHMENT_WRITE => COLOR_ATTACHMENT_READ
                }
            }
        }
        async_passes {
            // purely virtual, just to have a synchronization point at the start of the frame
            PresentationAcquire,
            ComputeCull on compute,
            BuildAccelerationStructures on compute,
            TransferCull on transfer,
            ConsolidateMeshBuffers on graphics,
        }
    }
}

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
        descriptors [model_set, camera_set, shadow_map_set, textures_set, acceleration_set]
        specialization_constants [
            10 => shadow_map_dim: u32,
            11 => shadow_map_dim_squared: u32,
        ]
        graphics
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
        samples 4
        stages [VERTEX, FRAGMENT]
        polygon mode LINE
    }
}

renderer_macros::define_pipe! {
    imgui_pipe {
        descriptors [imgui_set]
        graphics
        samples 4
        vertex_inputs [pos: vec2, uv: vec2, col: vec4]
        stages [VERTEX, FRAGMENT]
    }
}

pub(crate) type WaitValueAccessor = fn(u64, &ImageIndex) -> u64;
pub(crate) type SemaphoreAccessor = usize;

pub(crate) trait TimelineStage {
    const OFFSET: u64;
    const CYCLE: u64;
}

const fn as_of<T: TimelineStage>(frame_number: u64) -> u64 {
    frame_number * T::CYCLE + T::OFFSET
}

const fn as_of_last<T: TimelineStage>(frame_number: u64) -> u64 {
    as_of::<T>(frame_number - 1)
}

fn as_of_previous<T: TimelineStage>(image_index: &ImageIndex, indices: &SwapchainIndexToFrameNumber) -> u64 {
    let frame_number = indices.map[image_index.0 as usize];
    as_of::<T>(frame_number)
}

pub(crate) trait RenderStage {
    type SignalTimelineStage: TimelineStage;
    const SIGNAL_AUTO_SEMAPHORE_IX: usize;

    fn wait_semaphore_timeline() -> SmallVec<[(WaitValueAccessor, SemaphoreAccessor); 4]>;

    fn queue_submit(
        image_index: &ImageIndex,
        render_frame: &RenderFrame,
        queue: vk::Queue,
        command_buffers: &[vk::CommandBuffer],
    ) -> ash::prelude::VkResult<()> {
        scope!("vk", "vkQueueSubmit");

        let signal_semaphore = vk::SemaphoreSubmitInfoKHR::builder()
            .semaphore(render_frame.auto_semaphores.0[Self::SIGNAL_AUTO_SEMAPHORE_IX].handle)
            .value(as_of::<Self::SignalTimelineStage>(render_frame.frame_number))
            .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
            .build();
        let mut wait_semaphores: SmallVec<[vk::SemaphoreSubmitInfoKHR; 4]> = smallvec![];
        for (value_accessor, semaphore_ix) in Self::wait_semaphore_timeline() {
            wait_semaphores.push(
                vk::SemaphoreSubmitInfoKHR::builder()
                    .semaphore(render_frame.auto_semaphores.0[semaphore_ix].handle)
                    .value(value_accessor(render_frame.frame_number, image_index))
                    .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    .build(),
            );
        }
        let command_buffer_infos = command_buffers
            .iter()
            .map(|cb| vk::CommandBufferSubmitInfoKHR::builder().command_buffer(*cb).build())
            .collect::<Vec<_>>();

        let submit = vk::SubmitInfo2KHR::builder()
            .wait_semaphore_infos(&wait_semaphores)
            .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore))
            .command_buffer_infos(&command_buffer_infos)
            .build();

        unsafe {
            render_frame
                .device
                .synchronization2
                .queue_submit2(queue, &[submit], vk::Fence::null())
        }
    }

    fn wait_previous(
        renderer: &RenderFrame,
        image_index: &ImageIndex,
        swapchain_index_map: &SwapchainIndexToFrameNumber,
    ) {
        renderer.auto_semaphores.0[Self::SIGNAL_AUTO_SEMAPHORE_IX]
            .wait(
                &renderer.device,
                as_of_previous::<Self::SignalTimelineStage>(&image_index, &swapchain_index_map),
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

    fn default_shader_stages() -> SmallVec<[&'static [u8]; 4]>;
    fn shader_stages() -> SmallVec<[vk::ShaderStageFlags; 4]>;
    #[cfg(feature = "shader_reload")]
    fn shader_stage_paths() -> SmallVec<[&'static str; 4]>;
    fn varying_subgroup_stages() -> SmallVec<[bool; 4]>;

    fn vk(&self) -> vk::Pipeline;

    fn new_raw(
        device: &Device,
        layout: vk::PipelineLayout,
        stages: &[vk::PipelineShaderStageCreateInfo],
        flags: vk::PipelineCreateFlags,
        base_handle: vk::Pipeline,
        dynamic_arguments: Self::DynamicArguments,
    ) -> Self;

    fn destroy(self, device: &Device);
}

fn new_pipe_generic<P: Pipeline>(
    device: &Device,
    layout: &SmartPipelineLayout<P::Layout>,
    specialization: &P::Specialization,
    base_pipeline_handle: Option<vk::Pipeline>,
    shaders: Option<SmallVec<[Shader; 4]>>,
    dynamic_arguments: P::DynamicArguments,
) -> (SmallVec<[Shader; 4]>, P) {
    use std::ffi::CStr;
    let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
    let shaders: SmallVec<[Shader; 4]> = shaders.unwrap_or_else(|| {
        P::default_shader_stages()
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
                .name(&shader_entry_name)
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

    let base_pipeline_handle = base_pipeline_handle.clone().unwrap_or_else(vk::Pipeline::null);

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
            return;
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
    inner: P,
    specialization: P::Specialization,
    shaders: SmallVec<[Shader; 4]>,
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
        scope!("pipelines", "new");
        Self::new_internal(device, layout, specialization, None, dynamic_arguments)
    }

    fn new_internal(
        device: &Device,
        layout: &SmartPipelineLayout<P::Layout>,
        specialization: P::Specialization,
        mut base_pipeline: Option<&mut Self>,
        dynamic_arguments: P::DynamicArguments,
    ) -> Self {
        let base_pipeline_handle = base_pipeline.as_ref().map(|pipe| pipe.inner.vk());
        let shaders = base_pipeline.as_deref_mut().map(|p| take(&mut p.shaders));

        let (shaders, pipe) = new_pipe_generic(
            device,
            &layout,
            &specialization,
            base_pipeline_handle,
            shaders,
            dynamic_arguments,
        );

        #[cfg(feature = "shader_reload")]
        let last_updates;

        #[cfg(feature = "shader_reload")]
        if let Some(base_pipe) = base_pipeline {
            last_updates = take(&mut base_pipe.last_updates);
        } else {
            let stage_count = P::default_shader_stages().len();
            last_updates = smallvec![Instant::now(); stage_count];
        }

        SmartPipeline {
            inner: pipe,
            specialization,
            shaders,
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
        scope!("pipelines", "specialize");
        use std::mem::swap;

        #[cfg(feature = "shader_reload")]
        let new_shaders: SmallVec<[(Instant, Option<Shader>); 4]> = self
            .shaders
            .iter()
            .zip(P::shader_stage_paths().iter().zip(P::default_shader_stages().iter()))
            .enumerate()
            .map(
                |(stage_ix, (_shader, (&path, &default_code)))| match reloaded_shaders.0.get(path) {
                    Some((ts, code)) if *ts > self.last_updates[stage_ix] => {
                        let static_spv = spirq::SpirvBinary::from(default_code);
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
                                    (ts.clone(), Some(device.new_shader(&code)))
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

        if self.specialization != *new_spec || any_new_shaders {
            #[cfg(feature = "shader_reload")]
            for ((shader, last_update), (t, new)) in self
                .shaders
                .iter_mut()
                .zip(self.last_updates.iter_mut())
                .zip(new_shaders.into_iter())
            {
                if let Some(new) = new {
                    std::mem::replace(shader, new).destroy(device);
                    *last_update = t;
                }
            }
            let mut replacement = Self::new_internal(device, layout, new_spec.clone(), Some(self), dynamic_arguments);
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
        device.set_object_name(layout.handle, &format!("{}", T::DEBUG_NAME));

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
pub(crate) const fn binding_size<B: DescriptorBufferBinding>() -> vk::DeviceSize {
    B::SIZE
}

pub(crate) struct MainRenderpass {
    pub(crate) renderpass: frame_graph::Main::RenderPass,
}

impl MainRenderpass {
    pub(crate) fn new(renderer: &RenderFrame, attachments: &MainAttachments) -> Self {
        MainRenderpass {
            renderpass: frame_graph::Main::RenderPass::new(
                renderer,
                (attachments.swapchain_format, attachments.swapchain_format),
            ),
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.renderpass.destroy(device);
    }
}

impl RenderFrame {
    pub(crate) fn new() -> (RenderFrame, Swapchain, winit::event_loop::EventLoop<()>) {
        let (instance, events_loop) = Instance::new().expect("Failed to create instance");
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

renderer_macros::define_resource!(Color = Image);
pub(crate) struct MainAttachments {
    #[allow(unused)]
    swapchain_image_views: Vec<ImageView>,
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
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            );
            renderer.device.set_object_name(im.handle, "Depth RT");
            im
        };
        let color_image = {
            let im = renderer.device.new_image_exclusive(
                swapchain.surface.surface_format.format,
                vk::Extent3D {
                    width: swapchain.width,
                    height: swapchain.height,
                    depth: 1,
                },
                vk::SampleCountFlags::TYPE_4,
                vk::ImageTiling::OPTIMAL,
                vk::ImageLayout::UNDEFINED,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
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
                let handle = unsafe { renderer.device.create_image_view(&create_view_info, None).unwrap() };

                ImageView { handle }
            })
            .collect::<Vec<_>>();
        let color_image_view = {
            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain.surface.surface_format.format)
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

pub(crate) struct MainFramebuffer {
    pub(crate) framebuffer: frame_graph::Main::Framebuffer,
}

impl FromWorld for MainFramebuffer {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let main_renderpass = world.get_resource::<MainRenderpass>().unwrap();
        let swapchain = world.get_resource::<Swapchain>().unwrap();
        Self::new(&renderer, &main_renderpass, &swapchain)
    }
}

impl MainFramebuffer {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_renderpass: &MainRenderpass,
        swapchain: &Swapchain,
    ) -> MainFramebuffer {
        let framebuffer = frame_graph::Main::Framebuffer::new(
            renderer,
            &main_renderpass.renderpass,
            &[
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ],
            (
                swapchain.surface.surface_format.format,
                swapchain.surface.surface_format.format,
            ),
            (swapchain.width, swapchain.height),
        );

        MainFramebuffer { framebuffer }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.framebuffer.destroy(device);
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
            let mut s = SmartSet::new(&renderer.device, &main_descriptor_pool, &set_layout, ix);

            update_whole_buffer::<camera_set::bindings::matrices>(&renderer.device, &mut s, &buffer.current(ix));

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

        let model_set_layout = SmartSetLayout::new(&device);

        let model_buffer = renderer.new_buffered(|ix| {
            let b = device.new_static_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );
            device.set_object_name(b.buffer.handle, &format!("Model Buffer - {}", ix));
            b
        });
        let model_set = renderer.new_buffered(|ix| {
            let mut s = SmartSet::new(&renderer.device, &main_descriptor_pool, &model_set_layout, ix);
            update_whole_buffer::<model_set::bindings::model>(&renderer.device, &mut s, &model_buffer.current(ix));
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
    #[cfg(feature = "shader_reload")]
    pub(crate) previous_gltf_pipeline: DoubleBuffered<Option<SmartPipeline<gltf_mesh::Pipeline>>>,
    pub(crate) gltf_pipeline_layout: SmartPipelineLayout<gltf_mesh::PipelineLayout>,
}

impl GltfPassData {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_renderpass: &MainRenderpass,
        model_data: &ModelData,
        base_color: &BaseColorDescriptorSet,
        shadow_mapping: &ShadowMappingData,
        camera_matrices: &CameraMatrices,
        acceleration_structures: &AccelerationStructures,
    ) -> GltfPassData {
        /*
        let queue_family_indices = vec![device.graphics_queue_family];
        let image_create_info = vk::ImageCreateInfo::builder()
            .format(vk::Format::R8G8B8A8_UNORM)
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
        let gltf_pipeline = SmartPipeline::new(
            &renderer.device,
            &gltf_pipeline_layout,
            spec,
            (main_renderpass.renderpass.renderpass.handle, 0), // FIXME
        );

        GltfPassData {
            gltf_pipeline,
            #[cfg(feature = "shader_reload")]
            previous_gltf_pipeline: renderer.new_buffered(|_| None),
            gltf_pipeline_layout,
        }
    }

    pub(crate) fn destroy(self, device: &Device) {
        self.gltf_pipeline.destroy(device);
        #[cfg(feature = "shader_reload")]
        self.previous_gltf_pipeline
            .into_iter()
            .for_each(|p| p.into_iter().for_each(|p| p.destroy(device)));
        self.gltf_pipeline_layout.destroy(device);
    }
}

pub(crate) struct MainPassCommandBuffer {
    command_pools: DoubleBuffered<StrictCommandPool>,
    command_buffers: DoubleBuffered<vk::CommandBuffer>,
}

impl FromWorld for MainPassCommandBuffer {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource::<RenderFrame>().unwrap();
        let mut command_pools = renderer.new_buffered(|ix| {
            StrictCommandPool::new(
                &renderer.device,
                renderer.device.graphics_queue_family,
                &format!("Main Pass Command Pool[{}]", ix),
            )
        });
        let command_buffers = renderer.new_buffered(|ix| {
            command_pools
                .current_mut(ix)
                .allocate(&format!("Main Pass CB[{}]", ix), &renderer.device)
        });
        MainPassCommandBuffer {
            command_pools,
            command_buffers,
        }
    }
}

impl MainPassCommandBuffer {
    pub(crate) fn destroy(self, device: &Device) {
        self.command_pools.into_iter().for_each(|p| p.destroy(device));
    }
}

pub(crate) fn render_frame(
    renderer: Res<RenderFrame>,
    (main_renderpass, main_attachments): (Res<MainRenderpass>, Res<MainAttachments>),
    (image_index, model_data): (Res<ImageIndex>, Res<ModelData>),
    mut runtime_config: ResMut<RuntimeConfiguration>,
    camera_matrices: Res<CameraMatrices>,
    swapchain: Res<Swapchain>,
    consolidated_mesh_buffers: Res<ConsolidatedMeshBuffers>,
    (debug_aabb_pass_data, shadow_mapping_data): (Res<DebugAABBPassData>, Res<ShadowMappingData>),
    base_color_descriptor_set: Res<BaseColorDescriptorSet>,
    cull_pass_data: Res<CullPassData>,
    (main_framebuffer, acceleration_structures): (Res<MainFramebuffer>, Res<AccelerationStructures>),
    (submissions, mut main_pass_cb, mut gltf_pass, mut gui_render_data, mut camera, mut input_handler, mut gui): (
        Res<Submissions>,
        ResMut<MainPassCommandBuffer>,
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
    scope!("ecs", "render_frame");

    let GltfPassData {
        ref mut gltf_pipeline,
        ref gltf_pipeline_layout,
        #[cfg(feature = "shader_reload")]
        ref mut previous_gltf_pipeline,
    } = &mut *gltf_pass;

    // TODO: count this? pack and defragment draw calls?
    let total = binding_size::<cull_set::bindings::indirect_commands>() as u32
        / size_of::<frame_graph::VkDrawIndexedIndirectCommand>() as u32;

    #[cfg(feature = "shader_reload")]
    {
        // clean up the old pipeline that was used N frames ago
        if let Some(previous) = previous_gltf_pipeline.current_mut(image_index.0).take() {
            previous.destroy(&renderer.device);
        }

        use systems::shadow_mapping::DIM as SHADOW_MAP_DIM;
        *previous_gltf_pipeline.current_mut(image_index.0) = gltf_pipeline.specialize(
            &renderer.device,
            &gltf_pipeline_layout,
            &gltf_mesh::Specialization {
                shadow_map_dim: SHADOW_MAP_DIM,
                shadow_map_dim_squared: SHADOW_MAP_DIM * SHADOW_MAP_DIM,
            },
            (main_renderpass.renderpass.renderpass.handle, 0), // FIXME
            #[cfg(feature = "shader_reload")]
            &reloaded_shaders,
        );
    }

    let MainPassCommandBuffer {
        ref mut command_pools,
        ref command_buffers,
    } = *main_pass_cb;

    let command_pool = command_pools.current_mut(image_index.0);

    command_pool.reset(&renderer.device);

    let command_buffer = command_pool.record_to_specific(&renderer.device, *command_buffers.current(image_index.0));
    let main_renderpass_marker = command_buffer.debug_marker_around("main renderpass", [0.0, 0.0, 1.0, 1.0]);
    let guard = renderer_macros::barrier!(
        *command_buffer,
        IndirectCommandsBuffer.draw_from r in Main indirect buffer after [compact, copy_frozen],
        IndirectCommandsCount.draw_from r in Main indirect buffer after [draw_depth],
        ConsolidatedPositionBuffer.in_main r in Main vertex buffer after [in_depth],
        TLAS.in_main r in Main descriptor gltf_mesh.acceleration_set.top_level_as after [build],
        ShadowMapAtlas.apply r in Main descriptor gltf_mesh.shadow_map_set.shadow_maps after [prepare],
        Color.render rw in Main attachment,
        ConsolidatedPositionBuffer.draw_from r in Main vertex buffer after [consolidate],
        ConsolidatedNormalBuffer.draw_from r in Main vertex buffer after [consolidate],
        CulledIndexBuffer.draw_from r in Main index buffer after [copy_frozen, cull]
    );
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
        main_renderpass.renderpass.begin(
            &renderer,
            &main_framebuffer.framebuffer,
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
                main_attachments.swapchain_image_views[image_index.0 as usize].handle,
            ],
            &[
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                },
            ],
        );
        if runtime_config.debug_aabbs {
            scope!("ecs", "debug aabb pass");

            let _aabb_marker = command_buffer.debug_marker_around("aabb debug", [1.0, 0.0, 0.0, 1.0]);
            renderer.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                debug_aabb_pass_data.pipeline.vk(),
            );
            debug_aabb_pass_data.pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                (&camera_matrices.set.current(image_index.0),),
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
            gltf_pipeline_layout.bind_descriptor_sets(
                &renderer.device,
                *command_buffer,
                (
                    &model_data.model_set.current(image_index.0),
                    &camera_matrices.set.current(image_index.0),
                    &shadow_mapping_data.user_set.current(image_index.0),
                    &base_color_descriptor_set.set.current(image_index.0),
                    &acceleration_structures.set,
                ),
            );
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
                0,
            );
        }
        renderer
            .device
            .cmd_next_subpass(*command_buffer, vk::SubpassContents::INLINE);
        render_gui(
            &renderer,
            &mut gui_render_data,
            &mut runtime_config,
            &swapchain,
            &mut camera,
            &mut input_handler,
            &mut gui,
            &command_buffer,
            #[cfg(feature = "shader_reload")]
            &reloaded_shaders,
        );
        renderer.device.cmd_end_render_pass(*command_buffer);
    }
    drop(guard);
    drop(main_renderpass_marker);

    let command_buffer = *command_buffer.end();

    *main_pass_cb.command_buffers.current_mut(image_index.0) = command_buffer;
    submissions.submit(&renderer, &image_index, frame_graph::Main::INDEX, Some(command_buffer));
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
        let main_renderpass = world.get_resource::<MainRenderpass>().unwrap();
        let mut gui = unsafe { world.get_non_send_resource_unchecked_mut::<Gui>().unwrap() };
        Self::new(&renderer, &main_descriptor_pool, &mut gui, &main_renderpass)
    }
}

impl GuiRenderData {
    fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &MainDescriptorPool,
        gui: &mut Gui,
        main_renderpass: &MainRenderpass,
    ) -> GuiRenderData {
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
            &main_descriptor_pool,
            &descriptor_set_layout,
            0, // FIXME
        );

        let pipeline_layout = SmartPipelineLayout::new(&renderer.device, (&descriptor_set_layout,));

        let pipeline = SmartPipeline::new(
            &renderer.device,
            &pipeline_layout,
            imgui_pipe::Specialization {},
            (main_renderpass.renderpass.renderpass.handle, 1),
        );

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
            &"GuiRender Initialization Command Pool",
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
    #[allow(unused_variables)]
    let scope_name = format!("copy_resource<{}>", std::any::type_name::<T>());
    scope!("ecs", scope_name);

    to.0.clone_from(&from);
}

/// This system writes back the changes from the coppied wrapper to the original resource.
/// The mode of operation where writes are redirected to [CopiedResource] are probably less
/// efficient as they require a writeback
#[allow(dead_code)]
pub(crate) fn writeback_resource<T: Component + Clone>(from: Res<CopiedResource<T>>, mut to: ResMut<T>) {
    #[allow(unused_variables)]
    let scope_name = format!("writeback_resource<{}>", std::any::type_name::<T>());
    scope!("ecs", scope_name);

    to.clone_from(&from.0);
}

fn render_gui(
    renderer: &RenderFrame,
    gui_render_data: &mut GuiRenderData,
    runtime_config: &mut RuntimeConfiguration,
    swapchain: &Swapchain,
    camera: &mut Camera,
    input_handler: &mut InputHandler,
    gui: &mut Gui,
    command_buffer: &StrictRecordingCommandBuffer,
    #[cfg(feature = "shader_reload")] reloaded_shaders: &ReloadedShaders,
) {
    scope!("rendering", "render_gui");

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
        &renderer,
        input_handler,
        &swapchain,
        camera,
        runtime_config,
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
                scope!("rendering", "gui draw list");

                let index_len = draw_list.idx_buffer().len();
                index_slice[index_offset_coarse..index_offset_coarse + index_len]
                    .copy_from_slice(draw_list.idx_buffer());
                let vertex_len = draw_list.vtx_buffer().len();
                for (ix, vertex) in draw_list.vtx_buffer().iter().enumerate() {
                    pos_slice[ix] = glm::Vec2::from(vertex.pos);
                    uv_slice[ix] = glm::Vec2::from(vertex.uv);
                    // TODO: this conversion sucks, need to add customizable vertex attribute formats
                    let byte_color = na::SVector::<u8, 4>::from(vertex.col);
                    col_slice[ix] = byte_color.map(|byte| byte as f32 / 255.0);
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
    scope!("ecs", "ModelMatricesUpload");
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
    model_mapped.unmap_used_range(0..(max_accessed as vk::DeviceSize * size_of::<glm::Mat4>() as vk::DeviceSize));
}

pub(crate) fn camera_matrices_upload(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    camera: Res<Camera>,
    mut camera_matrices: ResMut<CameraMatrices>,
) {
    scope!("ecs", "CameraMatricesUpload");
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

pub(crate) fn graphics_stage() -> SystemStage {
    let stage = SystemStage::parallel();

    #[cfg(not(feature = "no_profiling"))]
    let stage = {
        let token = microprofile::get_token("ecs".to_string(), "graphics_stage".to_string(), 0);

        stage
            .with_system(
                (move || {
                    microprofile::enter(token);
                })
                .exclusive_system()
                .at_start(),
            )
            .with_system(
                (|| {
                    microprofile::leave();
                })
                .exclusive_system()
                .at_end(),
            )
    };

    let test = SystemGraph::new();

    let copy_runtime_config = test.root(copy_resource::<RuntimeConfiguration>);
    let copy_camera = test.root(copy_resource::<Camera>);
    let copy_indices = test.root(copy_resource::<SwapchainIndexToFrameNumber>);
    let setup_submissions = test.root(setup_submissions);
    let consolidate_mesh_buffers = setup_submissions.then(consolidate_mesh_buffers);

    let initial = (copy_runtime_config, copy_camera, copy_indices, consolidate_mesh_buffers);
    let (recreate_base_color, cull, cull_bypass, depth, build_as, shadow_mapping) = initial.join_all((
        recreate_base_color_descriptor_set.system(),
        cull_pass.system(),
        cull_pass_bypass.system(),
        depth_only_pass.system(),
        build_acceleration_structures.system(),
        prepare_shadow_maps.system(),
    ));

    let update_base_color = recreate_base_color.then(update_base_color_descriptors);

    // TODO: this only needs to wait before submission, could record in parallel
    let main_pass = update_base_color.then(render_frame);

    (
        main_pass,
        update_base_color,
        cull,
        cull_bypass,
        depth,
        build_as,
        shadow_mapping,
    )
        .join(PresentFramebuffer::exec);

    stage.with_system_set(test)
}

pub(crate) struct Submissions {
    pub(crate) remaining: Mutex<StableDiGraph<Option<Option<vk::CommandBuffer>>, (), u8>>,
}

impl Submissions {
    pub(crate) fn new() -> Submissions {
        Submissions {
            remaining: Mutex::default(),
        }
    }

    pub(crate) fn submit(
        &self,
        renderer: &RenderFrame,
        image_index: &ImageIndex,
        node_ix: u8,
        cb: Option<vk::CommandBuffer>,
    ) {
        scope!("rendering", "submit command buffer");
        let mut g = self.remaining.lock();
        let weight = g
            .node_weight_mut(NodeIndex::from(node_ix))
            .expect(&format!("Node not found while submitting {}", node_ix));
        debug_assert!(weight.is_none(), "node_ix = {}", node_ix);
        *weight = Some(cb);
        update_submissions(renderer, image_index, g);
    }
}

fn setup_submissions(mut submissions: ResMut<Submissions>) {
    scope!("rendering", "setup_submissions");
    let graph = submissions.remaining.get_mut();
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);

    graph.extend_with_edges(frame_graph::DEPENDENCY_GRAPH);
    graph.remove_node(NodeIndex::from(frame_graph::PresentationAcquire::INDEX));
}

fn update_submissions(
    renderer: &RenderFrame,
    image_index: &ImageIndex,
    mut graph: MutexGuard<StableDiGraph<Option<Option<vk::CommandBuffer>>, (), u8>>,
) {
    scope!("rendering", "update_submissions");
    use petgraph::Direction;

    let mut should_continue = true;
    while graph.node_count() > 0 && should_continue {
        should_continue = false;
        let roots = graph.externals(Direction::Incoming).collect::<Vec<_>>();
        'inner: for node in roots {
            match graph.node_weight_mut(node) {
                None => break 'inner, // someone else changed it up while we were unlocked
                Some(ref mut cb @ Some(_)) => {
                    let cb = cb.take();
                    // leave None behind so that others won't try to submit this, but will
                    // continue to see it as a blocking dependency
                    MutexGuard::unlocked_fair(&mut graph, || {
                        frame_graph::submit_stage_by_index(renderer, image_index, node, cb.unwrap());
                    });
                    // we can clean it up now to unlock downstream submissions
                    graph.remove_node(node).expect("remove node failed");
                    should_continue = true;
                }
                _ => {}
            }
        }
    }
}
