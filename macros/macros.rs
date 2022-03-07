#![feature(extend_one)]

use std::env;

use hashbrown::HashMap;
use itertools::Itertools;
use petgraph::visit::IntoNodeReferences;
use proc_macro2::TokenStream;
use proc_macro_error::proc_macro_error;
use quote::{format_ident, quote, ToTokens};
use renderer_macro_lib::{
    extract_optional_dyn, fetch,
    inputs::{to_rust_type, to_vk_format},
    resource_claims::{
        ResourceBarrierInput, ResourceClaimInput, ResourceClaims, ResourceDefinitionInput, ResourceDefinitionType,
        ResourceUsageKind,
    },
    Binding, DescriptorSet, LoadOp, NamedField, Pipeline, RenderPass, RenderPassAttachment, SpecificPipe, StaticOrDyn,
    StoreOp,
};
use syn::{parse_macro_input, parse_quote, parse_str, Ident, LitBool, Path, Type};

#[proc_macro]
pub fn define_resource(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input: ResourceDefinitionInput = syn::parse(input).unwrap();
    let name = format_ident!("{}", &input.resource_name);
    let resource_debug_name = format!("Resource[{}]", &input.resource_name);
    let data = fetch().unwrap();
    let claims = data.resources.get(&input.resource_name).unwrap();
    match input.ty {
        ResourceDefinitionType::StaticBuffer { type_name } => {
            let type_name = parse_str::<Type>(&type_name).unwrap();
            let usage_flags = claims
                .graph
                .node_weights()
                .flat_map(|claim| match &claim.usage {
                    ResourceUsageKind::VertexBuffer => vec![quote!(VERTEX_BUFFER)],
                    ResourceUsageKind::IndexBuffer => vec![quote!(INDEX_BUFFER)],
                    ResourceUsageKind::Attachment(ref _renderpass) => todo!(),
                    ResourceUsageKind::IndirectBuffer => vec![quote!(INDIRECT_BUFFER)],
                    ResourceUsageKind::TransferCopy | ResourceUsageKind::TransferClear => {
                        let mut acu = vec![];
                        if claim.reads {
                            acu.push(quote!(TRANSFER_SRC));
                        }
                        if claim.writes {
                            acu.push(quote!(TRANSFER_DST));
                        }
                        acu
                    }
                    ResourceUsageKind::Descriptor(set_name, binding_name, _pipeline_name) => {
                        let set = data.descriptor_sets.get(set_name).expect("set not found in barrier");
                        let binding = set
                            .bindings
                            .iter()
                            .find(|candidate| candidate.name == *binding_name)
                            .expect("binding not found in barrier");

                        match binding.descriptor_type.as_str() {
                            "STORAGE_BUFFER" => vec![quote!(STORAGE_BUFFER)],
                            _ => unimplemented!("static buffer usage {}", &binding.descriptor_type),
                        }
                    }
                })
                .collect_vec();

            quote! {
                pub(crate) struct #name(pub(crate) crate::renderer::StaticBuffer<#type_name>);

                impl #name {
                    pub(crate) fn new(device: &crate::renderer::device::Device) -> Self {
                        let b = device.new_static_buffer(
                            vk::BufferUsageFlags::empty() #(| vk::BufferUsageFlags::#usage_flags)*,
                            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                        );
                        device.set_object_name(b.buffer.handle, #resource_debug_name);
                        Self(b)
                    }

                    pub(crate) fn destroy(self, device: &crate::renderer::device::Device) {
                        self.0.destroy(device);
                    }
                }

                impl std::ops::Deref for #name {
                    type Target = crate::renderer::StaticBuffer<#type_name>;

                    fn deref(&self) -> &Self::Target {
                        &self.0
                    }
                }
            }
        }
        ResourceDefinitionType::Image { .. } => quote! {
            #[allow(non_camel_case_types)]
            pub(crate) struct #name(pub(crate) crate::renderer::device::Image);

            impl #name {
                pub(crate) fn import(device: &crate::renderer::Device, image: crate::renderer::device::Image) -> Self {
                    device.set_object_name(image.handle, #resource_debug_name);
                    Self(image)
                }

                pub(crate) fn destroy(self, device: &crate::renderer::device::Device) {
                    self.0.destroy(device);
                }
            }

            impl std::ops::Deref for #name {
                type Target = crate::renderer::device::Image;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }
        },
        // TODO: need a better approach probably
        ResourceDefinitionType::AccelerationStructure => quote!(),
    }
    .into()
}

#[proc_macro_error]
#[proc_macro]
pub fn barrier(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as ResourceBarrierInput);
    let data = fetch().unwrap();

    let mut validation_errors = quote!();
    let mut barrier_calls = quote!();
    let command_buffer_expr = input.command_buffer_ident;

    for claim in input.claims {
        let ResourceClaimInput {
            ref resource_name,
            ref step_name,
            ..
        } = claim;
        let claims = data.resources.get(&claim.resource_name).unwrap();

        let this_node = *claims.map.get(&claim.step_name).unwrap();
        let _this = &claims.graph[this_node];

        // let connected_components = petgraph::algo::connected_components(&claims.graph);
        // if connected_components != 1 {
        //     let msg = "resource claims graph must have one connected component".to_string();
        //     validation_errors.extend(quote!(compile_error!(#msg);));
        // }
        if petgraph::algo::is_cyclic_directed(&claims.graph) {
            let msg = "resource claims graph is cyclic".to_string();
            validation_errors.extend(quote!(compile_error!(#msg);));
        }
        match claims.ty {
            ResourceDefinitionType::StaticBuffer { .. } => {
                let resource_expr = claim
                    .resource_ident
                    .map(|x| quote!({#x}.buffer.handle))
                    .unwrap_or(quote!(vk::Buffer::null()));
                barrier_calls.extend_one(quote! {
                    submissions.barrier_buffer(
                        &renderer,
                        &crate::renderer::RENDERER_INPUT,
                        *#command_buffer_expr,
                        #resource_name,
                        #step_name,
                        #resource_expr,
                        &mut acquire_buffer_barriers,
                        &mut release_buffer_barriers,
                    );
                });
            }
            ResourceDefinitionType::AccelerationStructure => {
                barrier_calls.extend_one(quote! {
                    submissions.barrier_acceleration_structure(
                        &renderer,
                        &crate::renderer::RENDERER_INPUT,
                        *#command_buffer_expr,
                        #resource_name,
                        #step_name,
                        &mut acquire_memory_barriers,
                        &mut release_memory_barriers,
                    );
                });
            }
            ResourceDefinitionType::Image { .. } => {
                let resource_expr = claim
                    .resource_ident
                    .map(|x| quote!({#x}.handle))
                    .unwrap_or(quote!(vk::Image::null()));
                barrier_calls.extend_one(quote! {
                    let span = tracing::trace_span!(
                        "image",
                        resource_name = #resource_name, step_name = #step_name,
                        src_layout = tracing::field::Empty, dst_layout = tracing::field::Empty,
                        src_queue_family = tracing::field::Empty, dst_queue_family = tracing::field::Empty,
                    ).entered();
                    submissions.barrier_image(
                        &renderer,
                        &swapchain_index_map,
                        &image_index,
                        &crate::renderer::RENDERER_INPUT,
                        *#command_buffer_expr,
                        #resource_name,
                        #step_name,
                        #resource_expr,
                        &mut acquire_image_barriers,
                        &mut release_image_barriers,
                    );
                    drop(span);
                });
            }
        }
    }

    quote! {
        unsafe {
            let trace_span = tracing::trace_span!("barrier!").entered();
            #validation_errors
            let mut acquire_buffer_barriers = vec![];
            let mut acquire_memory_barriers = vec![];
            let mut acquire_image_barriers = vec![];
            let mut release_buffer_barriers = vec![];
            let mut release_memory_barriers = vec![];
            let mut release_image_barriers = vec![];
            #barrier_calls;
            {
                let marker = #command_buffer_expr.debug_marker_around("barrier![acquire]", [1.0, 0.0, 0.0, 1.0]);
                if !acquire_buffer_barriers.is_empty() || !acquire_memory_barriers.is_empty() || !acquire_image_barriers.is_empty() {
                    renderer.device.synchronization2.cmd_pipeline_barrier2(
                        *#command_buffer_expr,
                        &vk::DependencyInfoKHR::builder()
                            .buffer_memory_barriers(&acquire_buffer_barriers)
                            .memory_barriers(&acquire_memory_barriers)
                            .image_memory_barriers(&acquire_image_barriers),
                    );
                }
            }
            scopeguard::guard((), |()| {
                let release_buffer_barriers = release_buffer_barriers;
                let release_memory_barriers = release_memory_barriers;
                let release_image_barriers = release_image_barriers;
                let marker = #command_buffer_expr.debug_marker_around("barrier![release]", [1.0, 1.0, 0.0, 1.0]);
                if !release_buffer_barriers.is_empty() || !release_memory_barriers.is_empty() || !release_image_barriers.is_empty() {
                    renderer.device.synchronization2.cmd_pipeline_barrier2(
                        *#command_buffer_expr,
                        &vk::DependencyInfoKHR::builder()
                            .buffer_memory_barriers(&release_buffer_barriers)
                            .memory_barriers(&release_memory_barriers)
                            .image_memory_barriers(&release_image_barriers),
                    );
                    drop(trace_span);
                }
            })
        }
    }
    .into()
}

#[proc_macro]
pub fn define_timelines(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let data = fetch().unwrap();

    let mut grouped = HashMap::<_, Vec<_>>::new();
    for (pass_name, (sem_ix, step_ix)) in data.timeline_semaphore_mapping.iter() {
        grouped.entry(*sem_ix).or_default().push((*step_ix, pass_name));
    }

    let auto_count = grouped.len();
    let mut output = quote!();
    let mut maximums = HashMap::new();

    for (sem_ix, stages) in grouped.into_iter() {
        let max = stages.iter().map(|x| x.0).max().unwrap().next_power_of_two();
        assert!(maximums.insert(sem_ix, max).is_none());
        // TODO: reuse prevomputed timeline_semaphore_cycles
        assert_eq!(data.timeline_semaphore_cycles.get(&sem_ix), Some(&max));

        let stage_definitions = stages.into_iter().map(|(stage_ix, pass_name)| {
            let stage_name = format_ident!("{}", pass_name);
            quote! {
                pub(crate) struct #stage_name;
                impl crate::renderer::TimelineStage for #stage_name {
                    const OFFSET: u64 = #stage_ix;
                    const CYCLE: u64 = #max;
                }
            }
        });

        let semaphore_name = format_ident!("AutoSemaphore{}", sem_ix);

        output.extend(quote! {
            pub(crate) mod #semaphore_name {
                #(
                    #stage_definitions
                )*
            }
        });
    }

    let maximums = maximums
        .into_iter()
        .sorted_by_key(|(sem_ix, _)| *sem_ix)
        .map(|(_, max_value)| max_value)
        .collect_vec();

    output.extend(quote! {
        pub(crate) struct AutoSemaphores([crate::renderer::device::TimelineSemaphore; #auto_count]);

        impl AutoSemaphores {
            pub(crate) fn new(device: &crate::renderer::device::Device) -> Self {
                let mut ix = 0;
                let maximums = [
                    #(#maximums),*
                ];
                let inner = [(); #auto_count].map(|_| {
                    let s = device.new_semaphore_timeline(maximums[ix]);
                    device.set_object_name(s.handle, &format!("AutoSemaphore[{}]", ix));
                    ix += 1;
                    s
                });
                AutoSemaphores(inner)
            }

            pub(crate) fn destroy(self, device: &crate::renderer::device::Device) {
                self.0.into_iter().for_each(|sem| sem.destroy(device));
            }
        }
    });

    output.into()
}

#[proc_macro]
pub fn define_frame(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    define_frame2().into()
}

fn define_frame2() -> TokenStream {
    let validation_errors = quote!();

    let new_data = fetch().unwrap();

    let dependency_graph = &new_data.dependency_graph;

    let wait_instance = new_data
        .async_passes
        .keys()
        .map(|pass| {
            let (signal_out_sem, _) = new_data
                .timeline_semaphore_mapping
                .get(pass)
                .expect("no sync for this pass");
            let signal_timeline_ident = format_ident!("AutoSemaphore{}", signal_out_sem);
            let pass_ident = format_ident!("{}", pass);
            let signal_timeline_path: Path = parse_quote!(#signal_timeline_ident::#pass_ident);

            (pass, quote! {
                impl RenderStage for Stage {
                    type SignalTimelineStage = super::super::#signal_timeline_path;
                    const SIGNAL_AUTO_SEMAPHORE_IX: usize = #signal_out_sem;
                }
            })
        })
        .collect::<HashMap<_, _>>();

    let async_pass_definitions = new_data.async_passes.keys().map(|pass_name| {
        let wait_instance = wait_instance.get(pass_name).unwrap();
        let graph_ix = dependency_graph
            .node_references()
            .find(|(_ix, name)| *name == pass_name)
            .map(|x| x.0)
            .unwrap()
            .index();

        let pass_name_ident = format_ident!("{}", pass_name);
        quote! {
            #[allow(non_snake_case)]
            pub(crate) mod #pass_name_ident {
                use super::{vk, RenderStage, RenderFrame, ImageIndex, as_of, as_of_last};

                #[derive(Debug, Clone, Copy)]
                pub(crate) struct Stage;

                pub(crate) const INDEX: u32 = #graph_ix as u32;

                #wait_instance
            }
        }
    });

    let shader_definitions =
        syn::parse_str::<TokenStream>(&new_data.shader_information.shader_type_definitions).unwrap();

    // let queue_manager = prepare_queue_manager(&new_data, &dependency_graph);

    let name = format_ident!("{}", new_data.name);
    let src = env::var("OUT_DIR").unwrap();
    let src = std::path::Path::new(&src).join("crossbar.bincode");
    let src = src.to_string_lossy();
    let expanded = quote! {
        #validation_errors

        pub(crate) mod #name {
            use ash::vk;
            use super::{Device, RenderStage,
                        RenderFrame, ImageIndex, as_of, as_of_last};
            use std::mem::size_of;

            pub(crate) static CROSSBAR: &[u8] = include_bytes!(#src);

            #(
                #async_pass_definitions
            )*

            #shader_definitions
        }
    };

    expanded
}

macro_rules! make_split {
    ($name:ident, [$($letter:ident),+] [$($ix:tt),+]) => {
        #[allow(unused)]
        fn $name<$($letter),+>(it: impl Iterator<Item = ($($letter),+)> + Clone) -> ($(Vec<$letter>),+)
        where
            $($letter: Clone),+
        {
            (
                $(
                    it.clone().map(|tup| tup . $ix).collect()
                ),+
            )
        }
    };
}

make_split!(split2, [A, B] [0, 1]);
make_split!(split3, [A, B, C] [0, 1, 2]);
make_split!(split4, [A, B, C, D] [0, 1, 2, 3]);
make_split!(split5, [A, B, C, D, E] [0, 1, 2, 3, 4]);
make_split!(split6, [A, B, C, D, E, F] [0, 1, 2, 3, 4, 5]);
make_split!(split7, [A, B, C, D, E, F, G] [0, 1, 2, 3, 4, 5, 6]);

#[proc_macro]
pub fn define_set(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let set = parse_macro_input!(input as DescriptorSet);
    let data = fetch().unwrap();

    define_set_old(&data.name, &set, &data.shader_information.set_binding_type_names).into()
}

fn define_set_old(
    renderer_name: &str,
    set: &DescriptorSet,
    set_binding_type_names: &HashMap<(String, String), String>,
) -> TokenStream {
    let DescriptorSet { name, bindings, .. } = set;
    let set_name = format_ident!("{}", name);

    let (binding_ix, desc_type, desc_count, stage, const_binding) =
        split5(bindings.iter().enumerate().map(|(ix, binding)| {
            let Binding {
                descriptor_type,
                partially_bound,
                update_after_bind,
                count,
                shader_stages,
                ..
            } = binding;
            let descriptor_type = format_ident!("{}", descriptor_type);
            (
                ix,
                quote! { vk::DescriptorType::#descriptor_type },
                count,
                shader_stages
                    .iter()
                    .map(|s| { let s = format_ident!("{}", s); quote!(vk::ShaderStageFlags::#s) })
                    .collect::<Vec<_>>(),
                {
                    let partially_bound = partially_bound
                        .then(|| quote!(| vk::DescriptorBindingFlags::PARTIALLY_BOUND.as_raw()));
                    let update_after_bind = update_after_bind
                        .then(|| quote!(| vk::DescriptorBindingFlags::UPDATE_AFTER_BIND.as_raw()));
                    quote! {
                        vk::DescriptorBindingFlags::from_raw(vk::DescriptorBindingFlags::empty().as_raw() #partially_bound #update_after_bind)
                    }
                },
            )
        }));

    let binding_definition = bindings
        .iter()
        .enumerate()
        .map(|(binding_ix, binding)| {
            let Binding {
                name, descriptor_type, ..
            } = binding;
            let name = format_ident!("{}", name);

            if descriptor_type == "UNIFORM_BUFFER" || descriptor_type == "STORAGE_BUFFER" {
                let ty = set_binding_type_names
                    .get(&(set_name.to_string(), name.to_string()))
                    .expect("failed to find set binding type name");
                let ty = format_ident!("{}", ty);
                let binding_ix = binding_ix as u32;

                let descriptor_type = format_ident!("{}", descriptor_type);
                let renderer_name = format_ident!("{}", renderer_name);
                quote! {
                    pub(crate) struct #name;

                    impl crate::renderer::DescriptorBufferBinding for #name {
                        // TODO: root path hardcoded
                        type T = crate::renderer::#renderer_name::#ty;
                        type Set = super::Set;
                        const INDEX: u32 = #binding_ix;
                        const DESCRIPTOR_TYPE: ash::vk::DescriptorType = ash::vk::DescriptorType::#descriptor_type;
                    }
                }
            } else {
                quote!()
            }
        })
        .collect::<Vec<_>>();

    let layout_debug_name = format!("{} Layout", set_name);
    let set_debug_name = set_name.to_string();

    quote! {
        pub(crate) mod #set_name {
            use ash::vk;
            use crate::renderer::{
                device::{DescriptorSetLayout, DescriptorSet, Device, Buffer, DescriptorPool},
                MainDescriptorPool,
            };

            pub(crate) struct Layout {
                pub(crate) layout: DescriptorSetLayout,
            }

            impl crate::renderer::DescriptorSetLayout for Layout {
                const DEBUG_NAME: &'static str = #layout_debug_name;

                fn binding_flags() -> smallvec::SmallVec<[vk::DescriptorBindingFlags; 8]> {
                    smallvec::smallvec![
                        #(#const_binding),*
                    ]
                }
                fn binding_layout() -> smallvec::SmallVec<[vk::DescriptorSetLayoutBinding; 8]> {
                    smallvec::smallvec![
                        #(
                            vk::DescriptorSetLayoutBinding {
                                binding: #binding_ix as u32,
                                descriptor_type: #desc_type,
                                descriptor_count: #desc_count,
                                stage_flags: #(#stage)|*,
                                p_immutable_samplers: std::ptr::null(),
                            }
                        ),*
                    ]
                }
            }

            pub(crate) struct Set {
                pub(crate) set: DescriptorSet,
            }

            impl crate::renderer::DescriptorSet for Set {
                type Layout = Layout;

                const DEBUG_NAME: &'static str = #set_debug_name;

                fn vk_handle(&self) -> vk::DescriptorSet {
                    self.set.handle
                }
            }

            pub(crate) mod bindings {
                #(
                    #binding_definition
                )*
            }
        }
    }
}

#[proc_macro]
pub fn define_pipe(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let pipe = parse_macro_input!(input as Pipeline);
    let data = fetch().unwrap();

    define_pipe_old(
        &pipe,
        &data.resources,
        &data.renderpasses,
        data.shader_information.push_constant_type_definitions.get(&pipe.name),
    )
    .into()
}

fn define_pipe_old(
    pipe: &Pipeline,
    resources: &HashMap<String, ResourceClaims>,
    renderpasses: &HashMap<String, RenderPass>,
    push_constant_type: Option<&String>,
) -> TokenStream {
    let Pipeline {
        name,
        specialization_constants,
        descriptor_sets: descriptors,
        specific,
        ..
    } = pipe;
    let specialization = {
        let (field_id, field_name, field_ty) =
            split3(specialization_constants.iter().map(|(id, NamedField { name, ty })| {
                (
                    id,
                    syn::parse_str::<Ident>(name).unwrap(),
                    syn::parse_str::<Type>(ty).unwrap(),
                )
            }));
        let field_count = specialization_constants.len();
        let zero = quote!(0u32);
        let field_offset = field_ty.iter().fold(vec![zero], |mut offsets, ty| {
            let last = offsets.last().unwrap();
            let this = quote!(size_of::<#ty>() as u32);
            let new = quote!(#last + #this);
            offsets.push(new);
            offsets
        });

        quote! {
            #[repr(C)]
            #[derive(Debug, PartialEq, Clone)]
            pub(crate) struct Specialization {
                #(pub(crate) #field_name : #field_ty),*
            }

            pub(crate) const SPEC_MAP:
                [vk::SpecializationMapEntry; #field_count] = [
                #(
                    vk::SpecializationMapEntry {
                        constant_id: #field_id,
                        offset: #field_offset,
                        size: size_of::<#field_ty>(),
                    }
                ),*
            ];

            impl crate::renderer::PipelineSpecialization for Specialization {
                fn get_spec_info(&self) -> vk::SpecializationInfo {
                    if size_of::<Specialization>() > 0 {
                        let (left, spec_data, right) = unsafe {
                            from_raw_parts(self as *const Specialization, 1)
                            .align_to::<u8>()
                        };
                        assert!(
                            left.is_empty() && right.is_empty(),
                            "spec constant alignment failed"
                        );
                        vk::SpecializationInfo::builder()
                            .map_entries(&SPEC_MAP)
                            .data(spec_data)
                            .build()
                    } else {
                        vk::SpecializationInfo::default()
                    }
                }
            }
        }
    };
    let stage_flags = match specific {
        // TODO: imprecise
        SpecificPipe::Graphics(_) => quote!(vk::ShaderStageFlags::ALL_GRAPHICS),
        SpecificPipe::Compute => quote!(vk::ShaderStageFlags::COMPUTE),
    };
    let pipeline_bind_point = match specific {
        SpecificPipe::Graphics(_) => quote!(vk::PipelineBindPoint::GRAPHICS),
        SpecificPipe::Compute => quote!(vk::PipelineBindPoint::COMPUTE),
    };
    let push_constant_range = match push_constant_type {
        Some(_) => quote! {
            vk::PushConstantRange {
                stage_flags: #stage_flags,
                offset: 0,
                size: size_of::<PushConstants>() as u32,
            }
        },
        None => quote!(),
    };
    let fn_push_constants = match push_constant_type {
        Some(_) => quote! {
            pub(crate) fn push_constants(
                &self,
                device: &device::Device,
                command_buffer: vk::CommandBuffer,
                push_constants: &PushConstants) {
                // #[allow(unused_qualifications)]
                unsafe {
                    let casted: &[u8] = from_raw_parts(
                        push_constants as *const _ as *const u8,
                        size_of::<PushConstants>(),
                    );
                    device.cmd_push_constants(
                        command_buffer,
                        *self.layout,
                        #stage_flags,
                        0,
                        casted,
                    );
                    return;
                }
            }
        },
        None => quote!(),
    };

    let vertex_definitions = match specific {
        SpecificPipe::Compute => quote!(),
        SpecificPipe::Graphics(graphics) => {
            let (binding_ix, binding_format, binding_rust_type) = split3(
                graphics
                    .vertex_inputs
                    .iter()
                    .enumerate()
                    .map(|(binding_ix, NamedField { ty, .. })| {
                        (
                            binding_ix as u32,
                            to_vk_format(syn::parse_str::<Type>(ty).unwrap().to_token_stream()),
                            to_rust_type(syn::parse_str::<Type>(ty).unwrap().to_token_stream()),
                        )
                    }),
            );
            let binding_count = graphics.vertex_inputs.len();

            quote! {
                static ATTRIBUTE_DESCS:
                    [vk::VertexInputAttributeDescription; #binding_count] = [
                    #(
                        vk::VertexInputAttributeDescription {
                            location: #binding_ix,
                            binding: #binding_ix,
                            format: #binding_format,
                            offset: 0,
                        }
                    ),*
                ];

                static BINDING_DESCS:
                    [vk::VertexInputBindingDescription; #binding_count] = [
                    #(
                        vk::VertexInputBindingDescription {
                            binding: #binding_ix,
                            stride: size_of::<#binding_rust_type>() as u32,
                            input_rate: vk::VertexInputRate::VERTEX,
                        }
                    ),*
                ];

                pub(crate) fn vertex_input_state<'a>()
                    -> vk::PipelineVertexInputStateCreateInfoBuilder<'a> {
                    vk::PipelineVertexInputStateCreateInfo::builder()
                        .vertex_attribute_descriptions(&ATTRIBUTE_DESCS)
                        .vertex_binding_descriptions(&BINDING_DESCS)
                }
            }
        }
    };

    let pipe_arguments_new_types = match specific {
        SpecificPipe::Graphics(specific) => {
            let mut types = vec![];
            match specific.dynamic_renderpass {
                Some(_) => {}
                None => {
                    types.push(quote!(vk::RenderPass));
                    types.push(quote!(u32)); // subpass ix
                }
            }
            if specific.samples.is_dyn() {
                types.push(quote!(vk::SampleCountFlags));
            }
            quote! {
                (#(#types),*)
            }
        }
        SpecificPipe::Compute => quote!(()),
    };
    let pipe_argument_short = match specific {
        SpecificPipe::Graphics(specific) => {
            let mut args = vec![];
            match specific.dynamic_renderpass {
                Some(_) => {}
                None => {
                    args.push(quote!(renderpass));
                    args.push(quote!(subpass));
                }
            }
            match extract_optional_dyn(&specific.samples, quote!(dynamic_samples)) {
                Some(arg) => {
                    args.push(arg);
                }
                None => {}
            }
            args
        }
        SpecificPipe::Compute => vec![],
    };

    let pipeline_definition_inner = match specific {
        SpecificPipe::Graphics(specific) => {
            let truthy: &LitBool = &parse_quote!(true);
            let falsy: &LitBool = &parse_quote!(false);
            let polygon_mode: Ident = parse_quote!(FILL);
            let polygon_mode = specific
                .polygon_mode
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(polygon_mode);
            let front_face: Ident = parse_quote!(CLOCKWISE);
            let front_face = specific
                .front_face_mode
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(front_face);
            let front_face_dynamic =
                extract_optional_dyn(&specific.front_face_mode, quote!(vk::DynamicState::FRONT_FACE_EXT,));
            let topology_mode: Ident = parse_quote!(TRIANGLE_LIST);
            let topology_mode = specific
                .topology_mode
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(topology_mode);
            let topology_dynamic = extract_optional_dyn(
                &specific.topology_mode,
                quote!(vk::DynamicState::PRIMITIVE_TOPOLOGY_EXT,),
            );
            let cull_mode: Ident = parse_quote!(NONE);
            let cull_mode = specific
                .cull_mode
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(cull_mode);
            let cull_mode_dynamic = extract_optional_dyn(&specific.cull_mode, quote!(vk::DynamicState::CULL_MODE_EXT,));
            let depth_test_enable = falsy;
            let depth_test_enable = specific
                .depth_test_enable
                .as_ref()
                .map(|p| if *p { truthy } else { falsy })
                .unwrap_or_default(depth_test_enable);
            let depth_test_dynamic = extract_optional_dyn(
                &specific.depth_test_enable,
                quote!(vk::DynamicState::DEPTH_TEST_ENABLE_EXT,),
            );
            let depth_write_enable = falsy;
            let depth_write_enable = specific
                .depth_write_enable
                .as_ref()
                .map(|p| if *p { truthy } else { falsy })
                .unwrap_or_default(depth_write_enable);
            let depth_write_dynamic = extract_optional_dyn(
                &specific.depth_write_enable,
                quote!(vk::DynamicState::DEPTH_WRITE_ENABLE_EXT,),
            );
            let depth_bounds_enable = falsy;
            let depth_bounds_enable = specific
                .depth_bounds_enable
                .as_ref()
                .map(|p| if *p { truthy } else { falsy })
                .unwrap_or_default(depth_bounds_enable);
            let depth_bounds_dynamic = extract_optional_dyn(
                &specific.depth_bounds_enable,
                quote!(vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE_EXT,),
            );
            let depth_compare_op: Ident = parse_quote!(NEVER);
            let depth_compare_op = specific
                .depth_compare_op
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(depth_compare_op);
            let depth_compare_dynamic = extract_optional_dyn(
                &specific.depth_compare_op,
                quote!(vk::DynamicState::DEPTH_COMPARE_OP_EXT,),
            );
            let sample_count = match &specific.samples {
                StaticOrDyn::Static(c) => {
                    let s = format_ident!("TYPE_{}", c);
                    quote!(vk::SampleCountFlags::#s)
                }
                StaticOrDyn::Dyn => quote!(dynamic_samples),
            };
            let dynamic_renderpass_setup = match specific.dynamic_renderpass {
                Some(ref renderpass) => {
                    let renderpass = renderpasses.get(renderpass).expect("Dynamic renderpass not found");
                    let color_attachment_formats = renderpass.color.iter().map(|x| {
                        let format = resources
                            .get(&x.resource_name)
                            .and_then(|x| {
                                if let ResourceDefinitionType::Image { ref format, .. } = x.ty {
                                    Some(format)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_else(|| {
                                panic!("Resource not found or not an image {name}", name = &x.resource_name)
                            });
                        let format = format_ident!("{}", format);
                        quote!(vk::Format::#format)
                    });
                    let depth_stencil_format = renderpass
                        .depth_stencil
                        .as_ref()
                        .map(|x| {
                            let format = resources
                                .get(&x.resource_name)
                                .and_then(|x| {
                                    if let ResourceDefinitionType::Image { ref format, .. } = x.ty {
                                        Some(format)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or_else(|| {
                                    panic!("Resource not found or not an image {name}", name = &x.resource_name)
                                });
                            let format = format_ident!("{}", format);
                            quote!(.depth_attachment_format(vk::Format::#format))
                        })
                        .unwrap_or_default();
                    quote! {
                        let color_attachment_formats = [#(#color_attachment_formats),*];
                        let mut pipeline_rendering = vk::PipelineRenderingCreateInfoKHR::builder()
                            .color_attachment_formats(&color_attachment_formats)
                            #depth_stencil_format
                        ;
                    }
                }
                None => quote!(),
            };

            let renderpass_options = match specific.dynamic_renderpass {
                Some(_) => {
                    quote! {
                        .push_next(&mut pipeline_rendering)
                    }
                }
                None => quote! {
                    .render_pass(renderpass)
                    .subpass(subpass)
                },
            };

            quote! {
                #dynamic_renderpass_setup;
                let pipeline = device.new_graphics_pipeline(
                    vk::GraphicsPipelineCreateInfo::builder()
                        .vertex_input_state(&vertex_input_state())
                        .stages(stages)
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::builder()
                                .topology(vk::PrimitiveTopology::#topology_mode)
                                .build(),
                        )
                        .dynamic_state(&vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&[
                            vk::DynamicState::VIEWPORT,
                            vk::DynamicState::SCISSOR,
                            #front_face_dynamic
                            #topology_dynamic
                            #cull_mode_dynamic
                            #depth_test_dynamic
                            #depth_write_dynamic
                            #depth_compare_dynamic
                            #depth_bounds_dynamic
                        ]))
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::builder()
                                .viewport_count(1)
                                .scissor_count(1)
                                .build(),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::builder()
                                .front_face(vk::FrontFace::#front_face)
                                .cull_mode(vk::CullModeFlags::#cull_mode)
                                .line_width(1.0)
                                .polygon_mode(vk::PolygonMode::#polygon_mode)
                                .build(),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::builder()
                                .rasterization_samples(#sample_count)
                                .build(),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::builder()
                                .depth_test_enable(#depth_test_enable)
                                .depth_write_enable(#depth_write_enable)
                                .depth_compare_op(vk::CompareOp::#depth_compare_op)
                                .depth_bounds_test_enable(#depth_bounds_enable)
                                .max_depth_bounds(1.0)
                                .min_depth_bounds(0.0)
                                .build(),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::builder()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::builder()
                                    .blend_enable(true)
                                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                                    .color_blend_op(vk::BlendOp::ADD)
                                    .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                                    .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                                    .alpha_blend_op(vk::BlendOp::ADD)
                                    .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A)
                                    .build()])
                                .build(),
                        )
                        .layout(layout)
                        #renderpass_options
                        .base_pipeline_handle(base_pipeline_handle)
                        .base_pipeline_index(-1)
                        .build(),
                );
            }
        }
        SpecificPipe::Compute => {
            quote! {
                let pipeline = device.new_compute_pipelines(&[
                    vk::ComputePipelineCreateInfo::builder()
                        .stage(stages[0])
                        .layout(layout)
                        .flags(flags)
                        .base_pipeline_handle(base_pipeline_handle)
                        .base_pipeline_index(-1)
                ]).into_iter().next().unwrap();
            }
        }
    };

    let pipe_debug_name = format!("{}::Pipeline", pipe.name);
    let pipeline_definition2 = quote! {
        pub(crate) struct Pipeline {
            pub(crate) pipeline: crate::renderer::device::Pipeline,
        }

        impl crate::renderer::Pipeline for Pipeline {
            type DynamicArguments = #pipe_arguments_new_types;
            type Layout = PipelineLayout;
            type Specialization = Specialization;

            const NAME: &'static str = #name;

            fn new_raw(
                device: &Device,
                layout: vk::PipelineLayout,
                stages: &[vk::PipelineShaderStageCreateInfo],
                flags: vk::PipelineCreateFlags,
                base_pipeline_handle: vk::Pipeline,
                (#(#pipe_argument_short),*): Self::DynamicArguments,
            ) -> crate::renderer::device::Pipeline {
                #pipeline_definition_inner

                device.set_object_name(*pipeline, #pipe_debug_name);

                pipeline
            }
        }
    };
    let push_constant_type_name = match push_constant_type {
        Some(_) => quote!(PushConstants),
        None => quote!(()),
    };
    let pipeline_debug_name = name;
    let is_graphics = match specific {
        SpecificPipe::Graphics(_) => true,
        SpecificPipe::Compute => false,
    };
    let name = syn::parse_str::<Ident>(name).unwrap();
    let descriptor_path = descriptors
        .iter()
        .map(|(x, _)| syn::parse_str::<Path>(x).unwrap())
        .collect_vec();
    let descriptor_ident = descriptor_path
        .iter()
        .map(|p| p.segments.last().unwrap().clone())
        .collect_vec();

    let push_constant_type = push_constant_type.map(|x| syn::parse_str::<TokenStream>(x).unwrap());

    quote! {
        pub(crate) mod #name {
            use crate::renderer::device::{self, Device};
            use ash::vk;
            use std::{mem::size_of, slice::from_raw_parts};
            use profiling::scope;

            pub(crate) struct PipelineLayout {
                pub(crate) layout: device::PipelineLayout,
            }


            impl crate::renderer::PipelineLayout for PipelineLayout {
                type SmartDescriptorSetLayouts = (
                    #(crate::renderer::SmartSetLayout<super::#descriptor_path::Layout>,)*
                );
                type SmartDescriptorSets = (
                    #(crate::renderer::SmartSet<super::#descriptor_path::Set>,)*
                );
                type PushConstants = #push_constant_type_name;

                const IS_GRAPHICS: bool = #is_graphics;
                const DEBUG_NAME: &'static str = #pipeline_debug_name;

                fn new(device: &Device, (#(#descriptor_ident,)*): <Self::SmartDescriptorSetLayouts as crate::renderer::RefTuple>::Ref<'_>) -> crate::renderer::device::PipelineLayout {
                    #[allow(unused_qualifications)]
                    device.new_pipeline_layout(
                        &[#(&#descriptor_ident.layout),*],
                        &[#push_constant_range],
                    )
                }

                fn bind_descriptor_sets(
                    layout: vk::PipelineLayout,
                    device: &Device,
                    command_buffer: vk::CommandBuffer,
                    (#(#descriptor_ident,)*): <Self::SmartDescriptorSets as crate::renderer::RefTuple>::Ref<'_>,
                ) {
                    unsafe {
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            #pipeline_bind_point,
                            layout,
                            0,
                            &[#(#descriptor_ident.vk_handle()),*],
                            &[],
                        );
                    }
                }
            }

            impl PipelineLayout {
                #fn_push_constants

                pub(crate) fn destroy(self, device: &Device) {
                    self.layout.destroy(device);
                }
            }

            #pipeline_definition2

            #push_constant_type

            #specialization

            #vertex_definitions
        }
    }
}

#[proc_macro]
pub fn define_pass(_input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // let input = parse_macro_input!(input as inputs::AsyncPass);
    // let pass = AsyncPass::from(&input);

    quote!().into()
}

#[proc_macro]
pub fn define_renderpass(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let pass = parse_macro_input!(input as RenderPass);
    let data = fetch().unwrap();

    let passes_that_need_clearing: usize = pass
        .color
        .iter()
        .chain(pass.depth_stencil.iter())
        .flat_map(|attachment| {
            let RenderPassAttachment { load_op, .. } = attachment;
            (load_op == &LoadOp::Clear).then(|| 1)
        })
        .sum();
    let name = syn::parse_str::<Ident>(&pass.name).unwrap();
    let renderpass_begin = format!("{}::{}::begin", data.name, pass.name);
    let attachment_count: usize = pass.color.len() + pass.depth_stencil.as_ref().map(|_| 1).unwrap_or(0);
    let color_attachments = pass
        .color
        .iter()
        .enumerate()
        .map(|(ix, color)| {
            let load_op = match color.load_op {
                LoadOp::Clear => quote!(CLEAR),
                LoadOp::Load => quote!(LOAD),
                LoadOp::DontCare => quote!(DONT_CARE),
            };
            let store_op = match color.store_op {
                StoreOp::Store => quote!(STORE),
                StoreOp::Discard => quote!(DONT_CARE),
            };
            let color_attachment_clear_count = pass
                .color
                .iter()
                .take(ix)
                .filter(|color| color.load_op == LoadOp::Clear)
                .count();
            let clear_value = match color.load_op {
                LoadOp::Clear => quote!(.clear_value(clear_values[#color_attachment_clear_count])),
                _ => quote!(),
            };
            let layout = format_ident!("{}", color.layout);
            quote! {
                vk::RenderingAttachmentInfoKHR::builder()
                    .image_view(attachments[#ix])
                    .image_layout(vk::ImageLayout::#layout)
                    .load_op(vk::AttachmentLoadOp::#load_op)
                    .store_op(vk::AttachmentStoreOp::#store_op)
                    #clear_value
                    .build()
            }
        })
        .collect_vec();

    let (depth_attachment, depth_attachment_apply) = pass
        .depth_stencil
        .as_ref()
        .map(|depth| {
            let load_op = match depth.load_op {
                LoadOp::Clear => quote!(CLEAR),
                LoadOp::Load => quote!(LOAD),
                LoadOp::DontCare => quote!(DONT_CARE),
            };
            let store_op = match depth.store_op {
                StoreOp::Store => quote!(STORE),
                StoreOp::Discard => quote!(DONT_CARE),
            };
            let color_attachment_clear_count = pass.color.iter().filter(|color| color.load_op == LoadOp::Clear).count();
            let clear_value = match depth.load_op {
                LoadOp::Clear => quote!(.clear_value(clear_values[#color_attachment_clear_count])),
                _ => quote!(),
            };
            let layout = format_ident!("{}", depth.layout);
            let color_attachment_count = pass.color.len();
            (
                quote! {
                    let depth_attachment = vk::RenderingAttachmentInfoKHR::builder()
                        .image_view(attachments[#color_attachment_count])
                        .image_layout(vk::ImageLayout::#layout)
                        .load_op(vk::AttachmentLoadOp::#load_op)
                        .store_op(vk::AttachmentStoreOp::#store_op)
                        #clear_value
                        ;
                },
                quote!(.depth_attachment(&depth_attachment)),
            )
        })
        .unwrap_or((quote!(), quote!()));

    quote! {
        pub(crate) struct #name;

        impl #name {
            pub(crate) fn begin(
                renderer: &RenderFrame,
                command_buffer: vk::CommandBuffer,
                render_area: vk::Rect2D,
                attachments: &[vk::ImageView; #attachment_count],
                clear_values: &[vk::ClearValue; #passes_that_need_clearing],
            ) {
                use profiling::scope;
                scope!(#renderpass_begin);

                let color_attachments = [#(#color_attachments),*];
                #depth_attachment

                let rendering_info = vk::RenderingInfoKHR::builder()
                    .render_area(render_area)
                    .layer_count(1)
                    .color_attachments(&color_attachments)
                    #depth_attachment_apply
                ;

                unsafe {
                    scope!("vk::CmdBeginRendering");

                    renderer.device.dynamic_rendering.cmd_begin_rendering(
                        command_buffer,
                        &rendering_info
                    );
                }
            }
        }
    }
    .into()
}
