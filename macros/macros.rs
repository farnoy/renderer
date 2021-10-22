#![feature(extend_one)]

mod inputs;
mod keywords;
mod parsed;
mod rga;

use std::{env, fs::File, io::Read, iter::FromIterator};

use convert_case::{Case, Casing};
use derive_syn_parse::Parse;
use hashbrown::HashMap;
use inputs::*;
use itertools::Itertools;
use petgraph::{
    graph::{DiGraph, NodeIndex},
    stable_graph::StableDiGraph,
    visit::EdgeRef,
    Direction,
};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, ToTokens};
use syn::{parse_macro_input, parse_quote, Expr, Ident, Path, Visibility};

#[proc_macro]
pub fn define_timeline(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    define_timeline2(proc_macro2::TokenStream::from(input)).into()
}

fn define_timeline2(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    #[derive(Parse)]
    struct TimelineInput {
        visibility: Visibility,
        name: Ident,
        stages: Unbracket<UnArray<Ident>>,
    }

    let TimelineInput {
        visibility,
        name,
        stages,
    } = match syn::parse2(input) {
        Ok(i) => i,
        Err(e) => return proc_macro2::TokenStream::from(e.to_compile_error()),
    };

    let max = stages.len().next_power_of_two() as u64;

    let variant2 = stages
        .iter()
        .enumerate()
        .map(|(ix, stage)| {
            let mut ix = ix + 1;
            if ix == stages.len() {
                ix = ix.next_power_of_two();
            }
            let ix = ix as u64;

            quote! {
                #visibility struct #stage;
                impl #stage {
                    #visibility const VALUE: u64 = #ix;

                    #visibility const fn as_of(&self, frame_number: u64) -> u64 {
                        frame_number * #max + #ix
                    }

                    #visibility const fn as_of_last(&self, frame_number: u64) -> u64 {
                        self.as_of(frame_number - 1)
                    }

                    #visibility fn as_of_previous(
                        &self,
                        image_index: &ImageIndex,
                        indices: &SwapchainIndexToFrameNumber
                    ) -> u64 {
                        // can't be const fn because of smallvec indexing
                        let frame_number = indices.map[image_index.0 as usize];
                        self.as_of(frame_number)
                    }
                }
            }
        })
        .collect::<Vec<_>>();

    quote! {
        #[allow(non_snake_case)]
        #visibility mod #name {
            use super::{RenderFrame, ImageIndex, SwapchainIndexToFrameNumber};

            #(
                #variant2
            )*
        }
    }
}

#[proc_macro]
pub fn define_frame(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let frame_input = parse_macro_input!(input as FrameInput);
    let FrameInput {
        name,
        visibility,
        attachments,
        formats,
        samples,
        passes: UnArray(passes),
        async_passes: UnArray(async_passes),
        dependencies,
        sync: UnArray(sync),
        sets,
        pipelines,
        ..
    } = &frame_input;

    let renderer_input = parsed::RendererInput::from(&frame_input);

    let mut validation_errors = quote!();

    let sync: HashMap<String, Expr> = sync
        .into_iter()
        .map(|ArrowPair((ident, exp))| (ident.to_string(), exp.clone()))
        .collect();

    let graph_ixes = passes
        .iter()
        .map(|pass| &pass.name)
        .chain(async_passes.iter().map(|Sequence((ident, _))| ident))
        .enumerate()
        .map(|(ix, a)| (a.clone(), ix as u32))
        .collect::<HashMap<Ident, u32>>();

    let all_passes = passes
        .iter()
        .map(|pass| &pass.name)
        .chain(async_passes.iter().map(|Sequence((ident, _))| ident))
        .collect::<Vec<_>>();

    let mut dependency_graph = DiGraph::<Ident, DependencyType, u32>::new();
    for (ix, ident) in passes
        .iter()
        .map(|pass| &pass.name)
        .chain(async_passes.iter().map(|Sequence((ident, _))| ident))
        .enumerate()
    {
        assert_eq!(ix, dependency_graph.add_node(ident.clone()).index());
    }

    for resource in renderer_input.resources.values() {
        for (ix, usage) in resource.usages.iter().enumerate() {
            let prev_usage = match ix {
                0 => continue,
                ix => &resource.usages[ix - 1],
            };

            if prev_usage.pass == usage.pass {
                continue;
            }

            let prev_ix = graph_ixes.get(&prev_usage.pass).unwrap();
            let this_ix = graph_ixes.get(&usage.pass).unwrap();

            dependency_graph.update_edge(
                NodeIndex::from(*prev_ix),
                NodeIndex::from(*this_ix),
                DependencyType::SameFrame,
            );
        }
    }

    for ArrowPair((from, Sequence((to, dep)))) in dependencies.iter() {
        dependency_graph.update_edge(
            NodeIndex::from(*graph_ixes.get(&from).unwrap()),
            NodeIndex::from(*graph_ixes.get(&to).unwrap()),
            dep.0.unwrap_or(DependencyType::SameFrame),
        );
    }
    for (name, ix) in graph_ixes.iter() {
        *dependency_graph.node_weight_mut(NodeIndex::from(*ix)).unwrap() = name.to_owned();
    }
    let connected_components = petgraph::algo::connected_components(&dependency_graph);
    if connected_components != 1 {
        let msg = format!("sync graph must have one connected component");
        validation_errors.extend(quote!(compile_error!(#msg);));
    }
    if petgraph::algo::is_cyclic_directed(&dependency_graph) {
        let msg = format!("sync graph is cyclic");
        validation_errors.extend(quote!(compile_error!(#msg);));
    }

    /*
    // Visualize the dependency graph
    dbg!(petgraph::dot::Dot::with_config(
        &dependency_graph.map(|_, node_ident| node_ident.to_string(), |_, _| ""),
        &[petgraph::dot::Config::EdgeNoLabel]
    ));
    */

    let dynamic_attachments = attachments
        .iter()
        .zip(formats.iter())
        .enumerate()
        .flat_map(|(ix, (_, format))| if format.is_dyn() { Some(ix) } else { None })
        .collect::<Vec<_>>();

    let wait_instance = passes
        .iter()
        .map(|pass| &pass.name)
        .chain(async_passes.iter().map(|Sequence((ident, _))| ident))
        .map(|pass| {
            let signal_out = sync.get(&pass.to_string()).expect("no sync for this pass");
            let t = syn::parse2::<syn::Path>(signal_out.to_token_stream())
                .expect("sync values must consist of 2 path segments");
            let signal_timeline_member = format_ident!(
                "{}_semaphore",
                t.segments.first().unwrap().ident.to_string().to_case(Case::Snake)
            );

            let wait_inner = dependency_graph
                .edges_directed(NodeIndex::from(*graph_ixes.get(&pass).unwrap()), Direction::Incoming)
                .map(|edge| {
                    let from = all_passes.get(edge.source().index()).unwrap();
                    let signaled = sync.get(&from.to_string()).unwrap();
                    let t = syn::parse2::<Path>(signaled.to_token_stream())
                        .expect("sync values must consist of 2 path segments");
                    let timeline_member = format_ident!(
                        "{}_semaphore",
                        t.segments.first().unwrap().ident.to_string().to_case(Case::Snake)
                    );

                    let as_of = match edge.weight() {
                        &DependencyType::SameFrame => quote!(as_of(render_frame.frame_number)),
                        &DependencyType::LastFrame => quote!(as_of_last(render_frame.frame_number)),
                        &DependencyType::LastAccess => {
                            quote!(as_of_previous(&image_index, &render_frame))
                        }
                    };

                    quote! {
                        semaphores.push(
                            vk::SemaphoreSubmitInfoKHR::builder()
                                .semaphore(render_frame.#timeline_member.handle)
                                .value(super::super::#signaled.#as_of)
                                .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                                .build()
                        );
                    }
                })
                .collect::<Vec<_>>();

            quote! {
                impl RenderStage for Stage {
                    fn prepare_signal(render_frame: &RenderFrame, semaphores: &mut Vec<vk::SemaphoreSubmitInfoKHR>) {
                        semaphores.push(
                            vk::SemaphoreSubmitInfoKHR::builder()
                                .semaphore(render_frame.#signal_timeline_member.handle)
                                .value(super::super::#signal_out.as_of(render_frame.frame_number))
                                .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                                .build()
                        );
                    }

                    fn prepare_wait(image_index: &ImageIndex, render_frame: &RenderFrame,
                                    semaphores: &mut Vec<vk::SemaphoreSubmitInfoKHR>) {
                        #(#wait_inner)*
                    }

                    fn host_signal(render_frame: &RenderFrame) -> ash::prelude::VkResult<()> {
                        render_frame.#signal_timeline_member.signal(
                            &render_frame.device,
                            super::super::#signal_out.as_of(render_frame.frame_number)
                        )
                    }
                }
            }
        });

    let pass_definitions = passes
        .iter()
        .zip(wait_instance.clone())
        .enumerate()
        .map(|(pass_ix, (pass, wait_instance))| {
            let pass_name = &pass.name;
            let format_param_ty = pass
                .attachments
                .iter()
                .filter(|attachment| {
                    let attachment_ix = attachments.iter().position(|at| at == *attachment).unwrap();
                    let format = &formats[attachment_ix];
                    format.is_dyn()
                })
                .map(|_| quote!(vk::Format))
                .collect_vec();
            let attachment_desc = pass
                .attachments
                .iter()
                .map(|attachment| {
                    let attachment_ix = attachments.iter().position(|at| at == attachment).unwrap();
                    let format = &formats[attachment_ix];
                    let Sequence((
                        _,
                        ArrowPair((Sequence((load_op, initial_layout)), Sequence((store_op, final_layout)))),
                    )) = pass
                        .layouts
                        .iter()
                        .find(|Sequence((at, _))| at == attachment)
                        .expect("No layout info for used attachment");
                    let load_op = match load_op {
                        LoadOp::Clear => quote!(CLEAR),
                        LoadOp::Load => quote!(LOAD),
                        LoadOp::DontCare => quote!(DONT_CARE),
                    };
                    let store_op = match store_op {
                        StoreOp::Store => quote!(STORE),
                        StoreOp::Discard => quote!(DONT_CARE),
                    };
                    let format = match format {
                        StaticOrDyn::Dyn => {
                            let dyn_ix = dynamic_attachments.binary_search(&attachment_ix).unwrap();
                            let index = syn::Index::from(dyn_ix);
                            quote!(attachment_formats.#index)
                        }
                        StaticOrDyn::Static(format) => quote!(vk::Format::#format),
                    };
                    let samples = format_ident!("TYPE_{}", samples.0[attachment_ix].base10_parse::<u8>().unwrap());
                    quote! {
                        vk::AttachmentDescription::builder()
                            .format(#format)
                            .samples(vk::SampleCountFlags::#samples)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(vk::ImageLayout::#initial_layout)
                            .final_layout(vk::ImageLayout::#final_layout)
                            .load_op(vk::AttachmentLoadOp::#load_op)
                            .store_op(vk::AttachmentStoreOp::#store_op)
                            .build()
                    }
                })
                .collect_vec();

            let attachment_count = attachment_desc.len();
            let renderpass_definition = define_renderpass(&frame_input, pass);
            let framebuffer_definition = define_framebuffer(&frame_input, pass);

            quote! {
                #[allow(non_snake_case)]
                pub(crate) mod #pass_name {
                    use super::{vk, Device, RenderStage, RenderFrame, ImageIndex,
                        OriginalFramebuffer, OriginalRenderPass};

                    #[derive(Debug, Clone, Copy)]
                    pub(crate) struct Stage;

                    pub(crate) const INDEX: u8 = #pass_ix as u8;

                    #renderpass_definition

                    #framebuffer_definition

                    #wait_instance

                    #visibility fn get_attachment_descriptions(
                        attachment_formats: (#(#format_param_ty,)*)
                    ) -> [vk::AttachmentDescription; #attachment_count] {
                        [
                            #(
                                #attachment_desc
                            ),*
                        ]
                    }
                }
            }
        })
        .collect_vec();

    let async_pass_definitions = async_passes
        .iter()
        .map(|Sequence((ident, _))| ident)
        // .zip(passes.len()..)
        .zip(wait_instance.skip(passes.len()))
        .enumerate()
        .map(|(pass_ix, (pass, wait_instance))| {
            let pass_ix = passes.len() + pass_ix;

            quote! {
                #[allow(non_snake_case)]
                pub(crate) mod #pass {
                    use super::{vk, RenderStage, RenderFrame, ImageIndex};

                    #[derive(Debug, Clone, Copy)]
                    pub(crate) struct Stage;

                    pub(crate) const INDEX: u8 = #pass_ix as u8;

                    #wait_instance
                }
            }
        });

    let shader_definitions = define_renderer(sets, pipelines);
    let resource_definitions = TokenStream::from_iter(
        renderer_input
            .resources
            .values()
            .map(|resource| define_resource(&renderer_input, resource)),
    );

    let queue_manager = prepare_queue_manager(&renderer_input, &graph_ixes, &dependency_graph);

    let expanded = quote! {
        #validation_errors

        pub(crate) mod #name {
            use ash::vk;
            use super::{Device, RenderStage, RenderFrame, ImageIndex,
                        RenderPass as OriginalRenderPass,
                        Framebuffer as OriginalFramebuffer};
            use std::mem::size_of;

            #queue_manager

            #(
                #pass_definitions
            )*

            #(
                #async_pass_definitions
            )*

            #shader_definitions

            pub(crate) mod resources {
                use ash::vk;
                use crate::renderer::{RenderFrame, device::{Device, Image, StaticBuffer, VmaMemoryUsage}};

                #resource_definitions
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

fn define_resource(input: &parsed::RendererInput, resource: &parsed::ResourceInput) -> TokenStream {
    let parsed::ResourceInput { name, kind, .. } = resource;

    let string_name = name.to_string();
    let string_renderer_name = input.name.to_string();
    let resource_debug_name = format!("{}::resources::{}", string_renderer_name, string_name);

    let get_queue_family_index_for_pass = |pass: &Ident| {
        input
            .passes
            .get(&pass)
            .and(Some(parsed::QueueFamily::Graphics))
            .or_else(|| input.async_passes.get(&pass).map(|async_pass| async_pass.queue))
            .expect("pass not found")
    };

    let get_runtime_queue_family = |ty: parsed::QueueFamily| match ty {
        parsed::QueueFamily::Graphics => quote!(renderer.device.graphics_queue_family),
        parsed::QueueFamily::Compute => quote!(renderer.device.compute_queue_family),
        parsed::QueueFamily::Transfer => quote!(renderer.device.transfer_queue_family),
    };

    let per_usage_functions = resource.usages.iter().enumerate().map(|(ix, usage)| {
        let acquire_ident = format_ident!("acquire_{}", usage.name);
        let release_ident = format_ident!("release_{}", usage.name);
        let prev_usage = match ix {
            0 => resource.usages.last().unwrap(),
            ix => &resource.usages[(ix - 1) % resource.usages.len()],
        };
        let next_usage = &resource.usages[(ix + 1) % resource.usages.len()];
        let barrier_ident = format_ident!("barrier_{}_{}", prev_usage.name, usage.name);

        let prev_queue = get_queue_family_index_for_pass(&prev_usage.pass);
        let this_queue = get_queue_family_index_for_pass(&usage.pass);
        let next_queue = get_queue_family_index_for_pass(&next_usage.pass);
        let prev_queue_runtime = get_runtime_queue_family(prev_queue);
        let this_queue_runtime = get_runtime_queue_family(this_queue);
        let next_queue_runtime = get_runtime_queue_family(next_queue);

        let acquire = {
            let bypass = if ix == 0 {
                quote!(if renderer.frame_number == 1 {
                    return;
                })
            } else {
                quote!()
            };

            if prev_queue == this_queue {
                quote!()
            } else {
                quote! {
                    pub(crate) fn #acquire_ident(&self, renderer: &RenderFrame, cb: vk::CommandBuffer) {
                        #bypass

                        unsafe {
                            let buffer_barrier = &[
                                vk::BufferMemoryBarrier2KHR::builder()
                                    // TODO: granularity
                                    .src_queue_family_index(#prev_queue_runtime)
                                    .dst_queue_family_index(#this_queue_runtime)
                                    .src_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                                    .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                                    .src_access_mask(vk::AccessFlags2KHR::MEMORY_READ |
                                                     vk::AccessFlags2KHR::MEMORY_WRITE)
                                    .dst_access_mask(vk::AccessFlags2KHR::MEMORY_READ |
                                                     vk::AccessFlags2KHR::MEMORY_WRITE)
                                    .buffer(self.0.buffer.handle)
                                    .size(vk::WHOLE_SIZE)
                                    .build()
                            ];
                            renderer.device.synchronization2.cmd_pipeline_barrier2(
                                cb,
                                &vk::DependencyInfoKHR::builder().buffer_memory_barriers(buffer_barrier),
                            );
                        }
                    }
                }
            }
        };

        // Barriers are only generated within the same pass and when neither one side of the dependency is a
        // renderpass attachment
        let barrier = if prev_usage.pass == usage.pass
            && prev_usage.usage != parsed::ResourceUsageKind::Attachment
            && usage.usage != parsed::ResourceUsageKind::Attachment
        {
            let buffer_barrier = match kind {
                parsed::ResourceKind::StaticBuffer { .. } => quote!(vk::BufferMemoryBarrier2KHR::builder()
                    // TODO: granularity
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .src_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                    .buffer(self.0.buffer.handle)
                    .size(vk::WHOLE_SIZE)
                    .build()),
                parsed::ResourceKind::Image => quote!(),
            };
            let image_barrier = match kind {
                parsed::ResourceKind::StaticBuffer { .. } => quote!(),
                parsed::ResourceKind::Image => todo!("not needed yet"),
                /* quote!(vk::ImageMemoryBarrier2KHR::builder()
                 *     .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                 *     .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                 *     .src_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                 *     .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                 *     .src_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                 *     .dst_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                 *     .image(self.0.handle)
                 *     .subresource_range(vk::ImageSubresource::builder().)), */
            };
            quote! {
                pub(crate) fn #barrier_ident(&self, renderer: &RenderFrame, cb: vk::CommandBuffer) {
                    unsafe {
                        let buffer_barrier = &[
                            #buffer_barrier
                        ];
                        let image_barrier = &[
                            #image_barrier
                        ];
                        renderer.device.synchronization2.cmd_pipeline_barrier2(
                            cb,
                            &vk::DependencyInfoKHR::builder()
                                .buffer_memory_barriers(buffer_barrier)
                                .image_memory_barriers(image_barrier),
                        );
                    }
                }
            }
        } else {
            quote!()
        };

        let release = if this_queue == next_queue {
            quote!()
        } else {
            quote! {
                pub(crate) fn #release_ident(&self, renderer: &RenderFrame, cb: vk::CommandBuffer) {
                    unsafe {
                        let buffer_barrier = &[
                            vk::BufferMemoryBarrier2KHR::builder()
                                // TODO: granularity
                                .src_queue_family_index(#this_queue_runtime)
                                .dst_queue_family_index(#next_queue_runtime)
                                .src_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                                .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                                .src_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                                .dst_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                                .buffer(self.0.buffer.handle)
                                .size(vk::WHOLE_SIZE)
                                .build()
                        ];
                        renderer.device.synchronization2.cmd_pipeline_barrier2(
                            cb,
                            &vk::DependencyInfoKHR::builder().buffer_memory_barriers(buffer_barrier),
                        );
                    }
                }
            }
        };

        quote! {
            #acquire
            #barrier
            #release
        }
    });

    let mut validation_errors = quote!();

    match kind {
        parsed::ResourceKind::StaticBuffer { type_name } => quote! {
            #[allow(non_camel_case_types)]
            pub(crate) struct #name(pub(crate) StaticBuffer<super::#type_name>);

            impl #name {
                pub(crate) fn new(device: &Device) -> Self {
                    let b = device.new_static_buffer_exclusive(
                        vk::BufferUsageFlags::INDIRECT_BUFFER
                            | vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                    );
                    device.set_object_name(b.buffer.handle, #resource_debug_name);
                    Self(b)
                }

                pub(crate) fn destroy(self, device: &Device) {
                    self.0.destroy(device);
                }

                #(#per_usage_functions)*
            }

            impl std::ops::Deref for #name {
                type Target = StaticBuffer<super::#type_name>;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            #validation_errors
        },
        parsed::ResourceKind::Image => quote! {
            #[allow(non_camel_case_types)]
            pub(crate) struct #name(pub(crate) Image);

            impl #name {
                pub(crate) fn import(device: &Device, image: Image) -> Self {
                    device.set_object_name(image.handle, #resource_debug_name);
                    Self(image)
                }

                pub(crate) fn destroy(self, device: &Device) {
                    self.0.destroy(device);
                }

                #(#per_usage_functions)*
            }

            impl std::ops::Deref for #name {
                type Target = Image;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            #validation_errors
        },
    }
}

fn define_renderpass(frame_input: &FrameInput, pass: &Pass) -> TokenStream {
    let passes_that_need_clearing = pass.attachments.iter().enumerate().map(|(ix, attachment)| {
        let Sequence((_, ArrowPair((Sequence((load_op, _)), _)))) = pass
            .layouts
            .iter()
            .find(|Sequence((at, _))| at == attachment)
            .expect("pass missing attachment in layouts block");
        (ix, load_op == &LoadOp::Clear)
    });
    let passes_that_need_clearing_compact = passes_that_need_clearing
        .clone()
        .filter(|(_, need_clearing)| *need_clearing)
        .map(|(ix, _)| ix)
        .collect_vec();
    let passes_that_need_clearing_fetch = passes_that_need_clearing
        .clone()
        .zip(
            passes_that_need_clearing
                .clone()
                .scan(0usize, |ix, (_, need_clearing)| {
                    let ret = Some(*ix);
                    if need_clearing {
                        *ix += 1;
                    }
                    ret
                }),
        )
        .map(|((_, need_clearing), compact_ix)| {
            if need_clearing {
                quote!(clear_values[#compact_ix])
            } else {
                quote!(vk::ClearValue::default())
            }
        })
        .collect_vec();
    let passes_that_need_clearing_len = passes_that_need_clearing_compact.len();
    let format_param_ty = pass
        .attachments
        .iter()
        .filter(|attachment| {
            let attachment_ix = frame_input.attachments.iter().position(|at| at == *attachment).unwrap();
            let format = &frame_input.formats[attachment_ix];
            format.is_dyn()
        })
        .map(|_| quote!(vk::Format))
        .collect_vec();
    let debug_name = format!("{} - {}", frame_input.name.to_string(), pass.name.to_string());

    let renderpass_begin = format!("{}::{}::begin", frame_input.name.to_string(), pass.name.to_string(),);

    let (subpass_prerequisites, subpass_descriptions) = split2(pass.subpasses.iter().map(|subpass| {
        let (color_attachment_name, color_attachment_layout) = split2(
            subpass
                .color
                .as_ref()
                .unwrap_or(&Unbracket(UnArray(vec![])))
                .iter()
                .map(|ArrowPair(pair)| pair.clone()),
        );
        let (resolve_attachment_name, resolve_attachment_layout) = split2(
            subpass
                .resolve
                .as_ref()
                .unwrap_or(&Unbracket(UnArray(vec![])))
                .iter()
                .map(|ArrowPair(pair)| pair.clone()),
        );
        assert!(
            resolve_attachment_name.is_empty() || (color_attachment_name.len() == resolve_attachment_name.len()),
            "If resolving any attachments, must provide one for\
            each color attachment, ATTACHMENT_UNUSED not supported yet"
        );
        let color_attachment_ix = color_attachment_name.iter().map(|needle| {
            pass.attachments
                .iter()
                .position(|candidate| candidate == needle)
                .expect("subpass color refers to nonexistent attachment")
        });
        let (depth_stencil_attachment_name, depth_stencil_attachment_layout) = split2(
            subpass
                .depth_stencil
                .as_ref()
                .iter()
                .map(|Unbrace(ArrowPair(pair))| pair.clone()),
        );
        let depth_stencil_attachment_ix = depth_stencil_attachment_name.iter().map(|needle| {
            pass.attachments
                .iter()
                .position(|candidate| candidate == needle)
                .expect("subpass depth stencil refers to nonexistent attachment")
        });
        let resolve_attachment_ix = resolve_attachment_name.iter().map(|needle| {
            pass.attachments
                .iter()
                .position(|candidate| candidate == needle)
                .expect("subpass color refers to nonexistent attachment")
        });

        let color_attachment_var = format_ident!(
            "color_attachments_{}_{}",
            pass.name.to_string(),
            subpass.name.to_string()
        );
        let depth_stencil_var = depth_stencil_attachment_name
            .iter()
            .map(|_| {
                format_ident!(
                    "depth_stencil_attachment_{}_{}",
                    pass.name.to_string(),
                    subpass.name.to_string()
                )
            })
            .collect_vec();
        let resolve_attachment_var = resolve_attachment_name
            .iter()
            .map(|_| {
                format_ident!(
                    "resolve_attachments_{}_{}",
                    pass.name.to_string(),
                    subpass.name.to_string()
                )
            })
            .collect_vec();

        let resolve_attachment_references = quote! {
            #(
                vk::AttachmentReference {
                    attachment: #resolve_attachment_ix as u32,
                    layout: vk::ImageLayout::#resolve_attachment_layout,
                }
            ),*
        };

        (
            quote! {
                let #color_attachment_var = &[
                    #(
                        vk::AttachmentReference {
                            attachment: #color_attachment_ix as u32,
                            layout: vk::ImageLayout::#color_attachment_layout,
                        }
                    ),*
                ];
                #(
                    let #depth_stencil_var =
                        vk::AttachmentReference {
                            attachment: #depth_stencil_attachment_ix as u32,
                            layout: vk::ImageLayout::#depth_stencil_attachment_layout,
                        };
                )*
                #(
                    let #resolve_attachment_var = &[
                        #resolve_attachment_references
                    ];
                )*
            },
            quote! {
                vk::SubpassDescription::builder()
                    .color_attachments(#color_attachment_var)
                    #( .depth_stencil_attachment(&#depth_stencil_var) )*
                    #( .resolve_attachments(#resolve_attachment_var) )*
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .build()
            },
        )
    }));

    let zero_dependencies = Unbrace(UnArray(vec![]));
    let subpass_dependency = pass
        .dependencies
        .as_ref()
        .unwrap_or(&zero_dependencies)
        .iter()
        .map(|dep| {
            // (dep.from, dep.to, dep.src_stage, dep.dst_stage, dep.src_access, dep.dst_access);
            let src_subpass = pass
                .subpasses
                .iter()
                .position(|candidate| candidate.name == dep.from)
                .expect("did not find src subpass");
            let dst_subpass = pass
                .subpasses
                .iter()
                .position(|candidate| candidate.name == dep.to)
                .expect("did not find src subpass");
            let SubpassDependency {
                src_stage: src_stage_mask,
                dst_stage: dst_stage_mask,
                src_access: src_access_mask,
                dst_access: dst_access_mask,
                ..
            } = dep;

            quote! {
                vk::SubpassDependency::builder()
                    .src_subpass(#src_subpass as u32)
                    .dst_subpass(#dst_subpass as u32)
                    .src_stage_mask(vk::PipelineStageFlags::#src_stage_mask)
                    .dst_stage_mask(vk::PipelineStageFlags::#dst_stage_mask)
                    .src_access_mask(vk::AccessFlags::#src_access_mask)
                    .dst_access_mask(vk::AccessFlags::#dst_access_mask)
                    .build()
            }
        });

    let attachment_count = pass.attachments.len();

    quote! {
        // TODO: beginning & clearing
        pub(crate) struct RenderPass {
            pub(crate) renderpass: OriginalRenderPass,
        }

        impl RenderPass {
            pub(crate) fn new(
                renderer: &RenderFrame,
                attachment_formats: (#(#format_param_ty,)*)
            ) -> Self {
                #(
                    #subpass_prerequisites
                )*
                let subpasses = &[
                    #( #subpass_descriptions ),*
                ];
                let dependencies = &[
                    #( #subpass_dependency ),*
                ];
                let attachments = get_attachment_descriptions(attachment_formats);
                let renderpass = renderer.device.new_renderpass(
                    &vk::RenderPassCreateInfo::builder()
                        .attachments(&attachments)
                        .subpasses(subpasses)
                        .dependencies(dependencies)
                );
                renderer
                    .device
                    .set_object_name(renderpass.handle, #debug_name);
                RenderPass { renderpass }
            }

            pub(crate) fn begin(
                &self,
                renderer: &RenderFrame,
                framebuffer: &Framebuffer, // TODO: or compatible
                command_buffer: vk::CommandBuffer,
                render_area: vk::Rect2D,
                attachments: &[vk::ImageView; #attachment_count],
                clear_values: &[vk::ClearValue; #passes_that_need_clearing_len],
            ) {
                use microprofile::scope;
                scope!("macros", #renderpass_begin);

                let mut attachment_info = vk::RenderPassAttachmentBeginInfo::builder()
                    .attachments(attachments);
                let clear_values = [
                    #(
                        #passes_that_need_clearing_fetch
                    ),*
                ];
                let begin_info = vk::RenderPassBeginInfo::builder()
                    .push_next(&mut attachment_info)
                    .render_pass(self.renderpass.handle)
                    .framebuffer(framebuffer.framebuffer.handle)
                    .render_area(render_area)
                    .clear_values(&clear_values);

                unsafe {
                    scope!("vk", "vkCmdBeginRenderPass");

                    renderer.device.cmd_begin_render_pass(
                        command_buffer,
                        &begin_info,
                        vk::SubpassContents::INLINE,
                    );
                }
            }

            pub(crate) fn destroy(self, device: &Device) {
                self.renderpass.destroy(device);
            }
        }
    }
}

fn define_framebuffer(frame_input: &FrameInput, pass: &Pass) -> TokenStream {
    let debug_name = format!(
        "{} framebuffer {} [{{}}]",
        frame_input.name.to_string(),
        pass.name.to_string()
    );
    let attachment_count = pass.attachments.len();
    let attachment_ix = pass.attachments.iter().enumerate().map(|(ix, _)| ix).collect_vec();

    let dynamic_attachments = frame_input
        .attachments
        .iter()
        .zip(frame_input.formats.iter())
        .enumerate()
        .flat_map(|(ix, (_, format))| if format.is_dyn() { Some(ix) } else { None })
        .collect::<Vec<_>>();
    let format_param_ty = pass
        .attachments
        .iter()
        .filter(|attachment| {
            let attachment_ix = frame_input.attachments.iter().position(|at| at == *attachment).unwrap();
            let format = &frame_input.formats[attachment_ix];
            format.is_dyn()
        })
        .map(|_| quote!(vk::Format))
        .collect_vec();
    let attachment_formats_expr = pass
        .attachments
        .iter()
        .map(|attachment| {
            let attachment_ix = frame_input.attachments.iter().position(|at| at == attachment).unwrap();
            let format = &frame_input.formats[attachment_ix];
            match format {
                StaticOrDyn::Dyn => {
                    let dyn_ix = dynamic_attachments.binary_search(&attachment_ix).unwrap();
                    let index = syn::Index::from(dyn_ix);
                    quote!(dyn_attachment_formats.#index)
                }
                StaticOrDyn::Static(format) => quote!(vk::Format::#format),
            }
        })
        .collect_vec();

    quote! {
        pub(crate) struct Framebuffer {
            pub(crate) framebuffer: OriginalFramebuffer,
        }

        impl Framebuffer {
            pub(crate) fn new(
                renderer: &RenderFrame,
                renderpass: &RenderPass,
                image_usages: &[vk::ImageUsageFlags; #attachment_count],
                dyn_attachment_formats: (#(#format_param_ty,)*),
                (width, height): (u32, u32),
            ) -> Self {
                let view_formats: [[vk::Format; 1]; #attachment_count] = [
                    #([#attachment_formats_expr]),*
                ];
                let image_infos: [vk::FramebufferAttachmentImageInfo; #attachment_count] = [
                    #(
                        vk::FramebufferAttachmentImageInfo::builder()
                            .view_formats(&view_formats[#attachment_ix])
                            .width(width)
                            .height(height)
                            .layer_count(1)
                            .usage(image_usages[#attachment_ix])
                            .build()
                    ),*
                ];
                let mut attachments_info = vk::FramebufferAttachmentsCreateInfo::builder()
                    .attachment_image_infos(&image_infos);
                let attachments = [vk::ImageView::null(); #attachment_count];
                let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                    .flags(vk::FramebufferCreateFlags::IMAGELESS)
                    .push_next(&mut attachments_info)
                    .render_pass(renderpass.renderpass.handle)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1);
                let handle = unsafe {
                    renderer
                        .device
                        .create_framebuffer(&frame_buffer_create_info, None)
                        .unwrap()
                };
                renderer
                    .device
                    .set_object_name(handle, #debug_name);
                Framebuffer { framebuffer: OriginalFramebuffer { handle } }
            }

            pub(crate) fn destroy(self, device: &Device) {
                self.framebuffer.destroy(device);
            }
        }
    }
}

// it could also coalesce some submits into one submit with multiple command buffers
// but only when no sync points are needed in between
// need to figure out how to "communicate" between proc macros, particularly the dependency tree
// #[proc_macro]
// pub fn submit_coalesced(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
//     let UnArray(stages) = parse_macro_input!(input as UnArray<ArrowPair<Expr, Expr>>);

//     let mut built_up = quote! {
//         unsafe {
//             renderer
//                 .device
//                 .queue_submit(
//                     *queue,
//                     &submits,
//                     vk::Fence::null(),
//                 )
//         }
//     };

//     for ArrowPair((stage, command_buffers)) in stages.iter().rev() {
//         built_up = quote! {
//             #stage::Stage::submit_info(
//                 &image_index,
//                 &renderer,
//                 #command_buffers,
//                 |submit| {
//                     submits.push(submit);
//                     #built_up
//                 }
//             )
//         };
//     }

//     let out = quote! {
//         {
//             let mut submits = vec![];
//             #built_up
//         }
//     };

//     proc_macro::TokenStream::from(out)
// }

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

fn define_set(set: &DescriptorSet, set_binding_type_names: &HashMap<(Ident, Ident), Ident>) -> TokenStream {
    let DescriptorSet {
        name: set_name,
        bindings,
        ..
    } = set;

    let binding_count = bindings.len();
    let (binding_ix, partially_bound, update_after_bind, desc_type, desc_count, stage) =
        split6(bindings.iter().enumerate().map(|(ix, binding)| {
            let Binding {
                descriptor_type,
                partially_bound,
                update_after_bind,
                count,
                stages,
                ..
            } = binding;
            (
                ix,
                partially_bound.is_some(),
                update_after_bind.is_some(),
                quote! { vk::DescriptorType::#descriptor_type },
                count,
                stages
                    .iter()
                    .map(|s| quote!(vk::ShaderStageFlags::#s))
                    .collect::<Vec<_>>(),
            )
        }));

    let binding_definition = bindings
        .iter()
        .enumerate()
        .map(|(binding_ix, binding)| {
            let Binding {
                name, descriptor_type, ..
            } = binding;

            if descriptor_type == "UNIFORM_BUFFER" || descriptor_type == "STORAGE_BUFFER" {
                let ty = set_binding_type_names
                    .get(&(set_name.clone(), name.clone()))
                    .expect("failed to find set binding type name");

                quote! {
                    pub(crate) mod #name {
                        use ash::vk;
                        use crate::renderer::device::StaticBuffer;
                        use super::super::super::#ty;
                        use std::mem::size_of;

                        pub(crate) type T = #ty;
                        pub(crate) const SIZE: vk::DeviceSize = size_of::<T>() as vk::DeviceSize;

                        pub(crate) type Buffer = StaticBuffer<#ty>;

                        // TODO: add batching and return vk::WriteDescriptorSet when lifetimes are improved in ash

                        pub(crate) fn update_whole_buffer(device: &super::super::Device,
                                                          set: &mut super::super::Set,
                                                          buf: &Buffer) {
                            let buffer_updates = &[vk::DescriptorBufferInfo {
                                buffer: buf.buffer.handle,
                                offset: 0,
                                range: SIZE,
                            }];
                            unsafe {
                                device.update_descriptor_sets(
                                    &[vk::WriteDescriptorSet::builder()
                                        .dst_set(set.set.handle)
                                        .dst_binding(#binding_ix as u32)
                                        .descriptor_type(vk::DescriptorType::#descriptor_type)
                                        .buffer_info(buffer_updates)
                                        .build()],
                                    &[],
                                );
                            }
                        }
                    }
                }
            } else {
                quote!()
            }
        })
        .collect::<Vec<_>>();

    let layout_debug_name = format!("{} Layout", set_name.to_string());
    let set_debug_name = format!("{} Set [{{}}]", set_name.to_string());

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

            impl Layout {
                pub(crate) fn new(device: &Device) -> Layout {
                    let binding_flags = &[
                        #(
                            {
                                let _x = #binding_ix;
                                let mut flags = vk::DescriptorBindingFlags::empty();
                                if #partially_bound {
                                    flags |= vk::DescriptorBindingFlags::PARTIALLY_BOUND;
                                }
                                if #update_after_bind {
                                    flags |= vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
                                }
                                flags
                            }
                        ),*
                    ];
                    let flags = if #( #update_after_bind )||* {
                        vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
                    } else {
                        vk::DescriptorSetLayoutCreateFlags::empty()
                    };
                    let mut binding_flags =
                        vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                            .binding_flags(binding_flags);
                    let bindings = Layout::bindings();
                    let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&bindings)
                        .flags(flags)
                        .push_next(&mut binding_flags);
                    let inner = device.new_descriptor_set_layout(&create_info);
                    device.set_object_name(inner.handle, #layout_debug_name);

                    Layout {
                        layout: inner
                    }
                }

                pub(crate) fn destroy(self, device: &Device) {
                    self.layout.destroy(device);
                }

                pub(crate) fn bindings() -> [vk::DescriptorSetLayoutBinding; #binding_count] {
                    [
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

            impl Set {
                pub(crate) fn new(device: &Device,
                                main_descriptor_pool: &MainDescriptorPool,
                                layout: &Layout,
                                ix: u32) -> Set {
                    let set = main_descriptor_pool.0.allocate_set(device, &layout.layout);
                    device.set_object_name(set.handle, &format!(#set_debug_name, ix));
                    Set { set }
                }

                pub(crate) fn destroy(self, pool: &DescriptorPool, device: &Device) {
                    self.set.destroy(pool, device);
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

fn define_pipe(pipe: &Pipe, push_constant_type: Option<TokenStream>) -> TokenStream {
    let Pipe {
        name,
        specialization_constants,
        descriptors,
        specific,
        varying_subgroup_stages,
        ..
    } = pipe;
    let specialization = {
        let empty_vec = vec![];
        let s = match specialization_constants {
            Some(Unbracket(UnArray(s))) => s,
            None => &empty_vec,
        };
        let (field_id, field_name, field_ty) = split3(s.iter().map(|ArrowPair((id, field))| {
            let name = field.ident.as_ref().unwrap();
            let ty = &field.ty;

            (id, name, ty)
        }));
        let field_count = s.len();
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

            impl Specialization {
                pub(crate) fn get_spec_info(&self) -> vk::SpecializationInfo {
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
    let descriptor_ref = descriptors;
    let stage_flags = match specific {
        // TODO: imprecise
        SpecificPipe::Graphics(_) => quote!(vk::ShaderStageFlags::ALL_GRAPHICS),
        SpecificPipe::Compute(_) => quote!(vk::ShaderStageFlags::COMPUTE),
    };
    let pipeline_bind_point = match specific {
        SpecificPipe::Graphics(_) => quote!(vk::PipelineBindPoint::GRAPHICS),
        SpecificPipe::Compute(_) => quote!(vk::PipelineBindPoint::COMPUTE),
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
        SpecificPipe::Compute(_) => quote!(),
        SpecificPipe::Graphics(graphics) => {
            let (binding_ix, binding_format, binding_rust_type) = split3(
                graphics
                    .vertex_inputs
                    .iter()
                    .flat_map(|inputs| inputs.iter())
                    .enumerate()
                    .map(|(binding_ix, field)| {
                        (
                            binding_ix as u32,
                            to_vk_format(field.ty.to_token_stream()),
                            to_rust_type(field.ty.to_token_stream()),
                        )
                    }),
            );
            let binding_count = graphics.vertex_inputs.as_ref().map(|inputs| inputs.len()).unwrap_or(0);

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

    let mut spirv_code = quote!();
    for shader_stage in pipe.specific.stages() {
        let shader_path = std::path::Path::new(&env::var("OUT_DIR").unwrap()).join(format!(
            "{}.{}.spv",
            pipe.name.to_string(),
            shader_stage_to_file_extension(&shader_stage),
        ));
        let shader_path = shader_path.to_str().unwrap();
        let shader_src_path = std::path::Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("src")
            .join("shaders")
            .join(format!(
                "{}.{}",
                pipe.name.to_string(),
                shader_stage_to_file_extension(&shader_stage),
            ));
        let shader_src_path = shader_src_path.to_str().unwrap();
        let shader_stage_path = format_ident!("{}_PATH", &shader_stage);
        spirv_code.extend_one(quote! {
            pub(crate) static #shader_stage: &'static [u8] = include_bytes_align_as!(u32, #shader_path);
            #[cfg(feature = "shader_reload")]
            pub(crate) static #shader_stage_path: &'static str = #shader_src_path;
        });
    }

    let mut errors = quote!();

    let stage_count = specific.stages().len();

    let pipe_arguments = match specific {
        SpecificPipe::Graphics(specific) => {
            let dynamic_samples =
                extract_optional_dyn(&specific.samples, quote!(dynamic_samples: vk::SampleCountFlags,))
                    .unwrap_or(quote!());
            quote! {
                renderpass: &device::RenderPass,
                subpass: u32,
                #dynamic_samples
            }
        }
        SpecificPipe::Compute(_) => quote!(),
    };
    let pipe_argument_short = match specific {
        SpecificPipe::Graphics(specific) => {
            let mut args = vec![quote!(renderpass), quote!(subpass)];
            match extract_optional_dyn(&specific.samples, quote!(dynamic_samples)) {
                Some(arg) => {
                    args.push(arg);
                }
                None => {}
            }
            args
        }
        SpecificPipe::Compute(_) => vec![],
    };
    let shader_stage = specific.stages();
    let stage_ix = shader_stage.iter().enumerate().map(|(ix, _)| ix).collect_vec();
    let shader_stage_path = shader_stage
        .iter()
        .map(|shader_stage| format_ident!("{}_PATH", &shader_stage))
        .collect_vec();
    let allow_varying_snippet = quote!(vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE_EXT);
    let forbid_varying_snippet = quote!(vk::PipelineShaderStageCreateFlags::empty());
    let varying_subgroup_stages_flags = shader_stage
        .iter()
        .map(|stage| {
            match varying_subgroup_stages {
                // varying all stages
                Some(UnOption(None)) => &allow_varying_snippet,
                // varying in this stage
                Some(UnOption(Some(Unbracket(UnArray(stages)))))
                    if stages.iter().any(|candidate| candidate == stage) =>
                {
                    &allow_varying_snippet
                }
                _ => &forbid_varying_snippet,
            }
        })
        .collect_vec();

    let pipeline_definition_inner = match specific {
        SpecificPipe::Graphics(specific) => {
            let polygon_mode = parse_quote!(FILL);
            let polygon_mode = specific
                .polygon_mode
                .as_ref()
                .map(|p| p.as_ref().unwrap_or_warn_redundant(&polygon_mode, &mut errors))
                .unwrap_or(&polygon_mode);
            let front_face = parse_quote!(CLOCKWISE);
            let front_face = specific
                .front_face_mode
                .as_ref()
                .map(|p| p.as_ref().unwrap_or_warn_redundant(&front_face, &mut errors))
                .unwrap_or(&front_face);
            let front_face_dynamic =
                extract_optional_dyn(&specific.front_face_mode, quote!(vk::DynamicState::FRONT_FACE_EXT,));
            let topology_mode = parse_quote!(TRIANGLE_LIST);
            let topology_mode = specific
                .topology_mode
                .as_ref()
                .map(|p| p.as_ref().unwrap_or_warn_redundant(&topology_mode, &mut errors))
                .unwrap_or(&topology_mode);
            let topology_dynamic = extract_optional_dyn(
                &specific.topology_mode,
                quote!(vk::DynamicState::PRIMITIVE_TOPOLOGY_EXT,),
            );
            let cull_mode = parse_quote!(NONE);
            let cull_mode = specific
                .cull_mode
                .as_ref()
                .map(|p| p.as_ref().unwrap_or_warn_redundant(&cull_mode, &mut errors))
                .unwrap_or(&cull_mode);
            let cull_mode_dynamic = extract_optional_dyn(&specific.cull_mode, quote!(vk::DynamicState::CULL_MODE_EXT,));
            let depth_test_enable = parse_quote!(false);
            let depth_test_enable = specific
                .depth_test_enable
                .as_ref()
                .map(|p| p.as_ref().unwrap_or_warn_redundant(&depth_test_enable, &mut errors))
                .unwrap_or(&depth_test_enable);
            let depth_test_dynamic = extract_optional_dyn(
                &specific.depth_test_enable,
                quote!(vk::DynamicState::DEPTH_TEST_ENABLE_EXT,),
            );
            let depth_write_enable = parse_quote!(false);
            let depth_write_enable = specific
                .depth_write_enable
                .as_ref()
                .map(|p| p.as_ref().unwrap_or_warn_redundant(&depth_write_enable, &mut errors))
                .unwrap_or(&depth_write_enable);
            let depth_write_dynamic = extract_optional_dyn(
                &specific.depth_write_enable,
                quote!(vk::DynamicState::DEPTH_WRITE_ENABLE_EXT,),
            );
            let depth_bounds_enable = parse_quote!(false);
            let depth_bounds_enable = specific
                .depth_bounds_enable
                .as_ref()
                .map(|p| p.as_ref().unwrap_or_warn_redundant(&depth_bounds_enable, &mut errors))
                .unwrap_or(&depth_bounds_enable);
            let depth_bounds_dynamic = extract_optional_dyn(
                &specific.depth_bounds_enable,
                quote!(vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE_EXT,),
            );
            let depth_compare_op = parse_quote!(NEVER);
            let depth_compare_op = specific
                .depth_compare_op
                .as_ref()
                .map(|p| p.as_ref().unwrap_or_warn_redundant(&depth_compare_op, &mut errors))
                .unwrap_or(&depth_compare_op);
            let depth_compare_dynamic = extract_optional_dyn(
                &specific.depth_compare_op,
                quote!(vk::DynamicState::DEPTH_COMPARE_OP_EXT,),
            );
            let sample_count = match &specific.samples {
                Some(StaticOrDyn::Static(c)) => {
                    let s = format_ident!("TYPE_{}", c.base10_parse::<u8>().unwrap());
                    quote!(vk::SampleCountFlags::#s)
                }
                Some(StaticOrDyn::Dyn) => quote!(dynamic_samples),
                None => quote!(vk::SampleCountFlags::TYPE_1),
            };

            quote! {
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
                                    .color_write_mask(vk::ColorComponentFlags::all())
                                    .build()])
                                .build(),
                        )
                        .layout(*layout.layout)
                        .render_pass(renderpass.handle)
                        .subpass(subpass)
                        .build(),
                );
            }
        }
        SpecificPipe::Compute(_) => {
            quote! {
                let pipeline = device.new_compute_pipelines(&[
                    vk::ComputePipelineCreateInfo::builder()
                        .stage(stages[0])
                        .layout(*layout.layout)
                        .flags(flags)
                        .base_pipeline_handle(base_pipeline_handle)
                        .base_pipeline_index(-1)
                ]).into_iter().next().unwrap();
            }
        }
    };

    let pipe_debug_name = format!("{}::Pipeline", pipe.name.to_string());
    let specialize_msg = format!("{}::Pipeline::specialize", pipe.name.to_string());
    let new_internal_msg = format!("{}::Pipeline::new_internal", pipe.name.to_string());
    let pipeline_definition2 = quote! {
        use std::time::Instant;
        #[cfg(feature = "shader_reload")]
        use super::super::{device::Shader, ReloadedShaders};

        pub(crate) struct Pipeline {
            pub(crate) pipeline: device::Pipeline,
            specialization: Specialization,
            #[cfg(feature = "shader_reload")]
            last_reloaded_shaders: [Option<Shader>; #stage_count],
            #[cfg(feature = "shader_reload")]
            last_updates: [Instant; #stage_count],
        }

        impl Pipeline {
            pub(crate) fn new(
                device: &Device,
                layout: &PipelineLayout,
                spec: Specialization,
                shader: [Option<&device::Shader>; #stage_count],
                #pipe_arguments
            ) -> Self {
                Self::new_internal(device, layout, spec, None, shader, #(#pipe_argument_short),*)
            }

            fn new_internal(
                device: &Device,
                layout: &PipelineLayout,
                specialization: Specialization,
                base_pipeline: Option<&Self>,
                shaders: [Option<&device::Shader>; #stage_count],
                #pipe_arguments
            ) -> Self {
                scope!("macros", #new_internal_msg);

                use std::ffi::CStr;
                let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
                let base_pipeline_handle = base_pipeline
                    .map(|pipe| *pipe.pipeline)
                    .unwrap_or_else(vk::Pipeline::null);
                let shaders: [Result<&device::Shader, device::Shader>; #stage_count] = [
                    #(
                        shaders[#stage_ix].ok_or_else(|| device.new_shader(#shader_stage))
                    ),*
                ];
                // let subgroup_size_controls = &mut [
                //     #(
                //         vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT::builder()
                //             #subgroup_sizes_calls
                //     ),*
                // ];
                let shader_handles = shaders
                    .iter()
                    .map(|s| s.as_ref().map(|s| s.vk()).unwrap_or_else(|s| s.vk()))
                    .collect::<Vec<_>>();
                let spec_info = specialization.get_spec_info();
                let mut flags = vk::PipelineCreateFlags::ALLOW_DERIVATIVES;
                if base_pipeline.is_some() {
                    flags |= vk::PipelineCreateFlags::DERIVATIVE;
                }

                let stages = &[
                    #(
                        vk::PipelineShaderStageCreateInfo::builder()
                            .module(shader_handles[#stage_ix])
                            .name(&shader_entry_name)
                            .stage(vk::ShaderStageFlags::#shader_stage)
                            .specialization_info(&spec_info)
                            .flags(#varying_subgroup_stages_flags)
                            // .push_next(&mut subgroup_size_controls[#stage_ix])
                            .build()
                    ),*
                ];

                #pipeline_definition_inner;

                device.set_object_name(*pipeline, #pipe_debug_name);

                std::array::IntoIter::new(shaders)
                    .for_each(|s| s.err().into_iter().for_each(|s| s.destroy(device)));

                Pipeline {
                    pipeline,
                    specialization,
                    #[cfg(feature = "shader_reload")]
                    last_reloaded_shaders: [
                        #(
                            {
                                let _utilize = #stage_ix;
                                None
                            }
                        ),*
                    ],
                    #[cfg(feature = "shader_reload")]
                    last_updates: [Instant::now(); #stage_count],
                }
            }

            /// Re-specializes the pipeline if needed and returns the old one that might still be in use.
            pub(crate) fn specialize(
                &mut self,
                device: &Device,
                layout: &PipelineLayout,
                new_spec: &Specialization,
                shaders: [Option<&device::Shader>; #stage_count],
                #pipe_arguments
                #[cfg(feature = "shader_reload")] reloaded_shaders: &ReloadedShaders,
            ) -> Option<Self> {
                scope!("macros", #specialize_msg);
                use std::mem::swap;

                #[cfg(feature = "shader_reload")]
                let mut new_shaders: [(Instant, Option<device::Shader>); #stage_count] = [
                    #(
                        match reloaded_shaders.0.get(#shader_stage_path) {
                            Some((ts, code)) if *ts > self.last_updates[#stage_ix] => {
                                let static_spv = spirq::SpirvBinary::from(#shader_stage);
                                let static_entry = static_spv.reflect_vec().unwrap()
                                    .into_iter().find(|entry| entry.name == "main").unwrap();
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
                                           && static_spec_consts == entry_spec_consts {
                                            (ts.clone(), Some(device.new_shader(&code)))
                                        } else {
                                            eprintln!("Failed to validate live reloaded shader interface \
                                                       against the static. Restart the application");
                                            (Instant::now(), None)
                                        }
                                    }
                                    else {
                                        eprintln!("Failed to find the main entry point in live reloaded spirv");
                                        (Instant::now(), None)
                                    }
                                } else {
                                    eprintln!("Failed to reflect on live reloaded spirv");
                                    (Instant::now(), None)
                                }
                            }
                            _ => (Instant::now(), None)
                        }
                    ),*
                ];

                #[cfg(feature = "shader_reload")]
                let any_new_shaders: bool = new_shaders.iter().any(|(_ts, s)| s.is_some());

                #[cfg(not(feature = "shader_reload"))]
                let any_new_shaders: bool = false;

                if self.specialization != *new_spec || any_new_shaders {
                    let mut replacement = Self::new_internal(
                        device,
                        layout,
                        new_spec.clone(),
                        Some(&self),
                        #[cfg(feature = "shader_reload")]
                        [
                            #(
                                new_shaders[#stage_ix].1.as_ref()
                                    .or(shaders[#stage_ix])
                                    .or(self.last_reloaded_shaders[#stage_ix].as_ref())
                            ),*
                        ],
                        #[cfg(not(feature = "shader_reload"))]
                        shaders,
                        #(#pipe_argument_short),*
                    );
                    #[cfg(feature = "shader_reload")]
                    {
                        replacement.last_updates = [
                            #(
                                new_shaders[#stage_ix].1.as_ref()
                                    .map(|s| new_shaders[#stage_ix].0.clone())
                                    .unwrap_or(self.last_updates[#stage_ix])
                            ),*
                        ];

                        replacement.last_reloaded_shaders = [
                            #(
                                new_shaders[#stage_ix].1.take()
                                    .or_else(|| self.last_reloaded_shaders[#stage_ix].take())
                            ),*
                        ];
                    }
                    swap(&mut *self, &mut replacement);
                    Some(replacement)
                } else {
                    None
                }
            }

            pub(crate) fn spec(&self) -> &Specialization { &self.specialization }

            pub(crate) fn destroy(self, device: &Device) {
                self.pipeline.destroy(device);
                #[cfg(feature = "shader_reload")]
                std::array::IntoIter::new(self.last_reloaded_shaders)
                    .for_each(|s| s.into_iter().for_each(|s| s.destroy(device)));
            }
        }
    };

    quote! {
        pub(crate) mod #name {
            use crate::renderer::device::{self, Device};
            use ash::vk;
            use std::{mem::size_of, slice::from_raw_parts};
            use microprofile::scope;

            pub(crate) struct PipelineLayout {
                pub(crate) layout: device::PipelineLayout,
            }

            impl PipelineLayout {
                pub(crate) fn new(
                    device: &Device,
                    #(#descriptor_ref: &super::#descriptor_ref::Layout,)*
                ) -> PipelineLayout {
                    #[allow(unused_qualifications)]
                    let layout = device.new_pipeline_layout(
                        &[
                            #(
                                &#descriptor_ref.layout
                            ),*
                        ],
                        &[#push_constant_range],
                    );

                    // TOOD: debug name

                    PipelineLayout { layout }
                }

                #fn_push_constants

                pub(crate) fn bind_descriptor_sets(
                    &self,
                    device: &Device,
                    command_buffer: vk::CommandBuffer
                    #(, #descriptor_ref: &super::#descriptor_ref::Set)*) {
                    unsafe {
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            #pipeline_bind_point,
                            *self.layout,
                            0,
                            &[
                                #(
                                    #descriptor_ref.set.handle
                                ),*
                            ],
                            &[],
                        );
                    }
                }

                pub(crate) fn destroy(self, device: &Device) {
                    self.layout.destroy(device);
                }
            }

            #pipeline_definition2

            #push_constant_type

            #specialization

            #vertex_definitions

            #spirv_code

            #errors
        }
    }
}

fn define_renderer(sets: &UnArray<DescriptorSet>, pipelines: &UnArray<Pipe>) -> proc_macro2::TokenStream {
    let mut output = quote!();

    let mut defined_types = HashMap::new();
    // (set_name, binding_name) => type_name
    // only available for buffers and stuff that is defined, image bindings won't have them
    let mut set_binding_type_names: HashMap<(Ident, Ident), Ident> = HashMap::new();

    for pipe in pipelines.iter() {
        let mut push_constant_type = None;
        let mut push_constant_spir_type = None;

        for shader_stage in pipe.specific.stages() {
            let shader_path = std::path::Path::new(&env::var("OUT_DIR").unwrap()).join(format!(
                "{}.{}.spv",
                pipe.name.to_string(),
                shader_stage_to_file_extension(&shader_stage),
            ));
            let shader_file = File::open(&shader_path).unwrap();
            let bytes: Vec<u8> = shader_file.bytes().filter_map(std::result::Result::ok).collect();
            let spv = spirq::SpirvBinary::from(bytes);
            let entry_points = spv.reflect_vec().expect("failed to reflect on spirv");
            let entry = entry_points
                .iter()
                .find(|entry| entry.name == "main")
                .expect("Failed to load entry point");

            for spv in entry.spec.spec_consts() {
                let rusty = pipe
                    .specialization_constants
                    .as_ref()
                    .map(|x| x.iter().find(|p| spv.spec_id == p.0 .0.base10_parse::<u32>().unwrap()))
                    .flatten();

                match rusty {
                    Some(rusty) => {
                        if !compare_types(spv.ty, &rusty.1.ty) {
                            let msg = format!(
                                "shader {} spec constant mismatch for id = {} shader type = {:?}, rusty type = {:?}",
                                shader_path.to_string_lossy(),
                                spv.spec_id,
                                spv.ty,
                                rusty.1.ty,
                            );
                            output.extend(quote!(compile_error!(#msg);));
                            continue;
                        }
                    }
                    None => {
                        let id = spv.spec_id;
                        let msg = format!(
                            "shader {} missing rust side of spec const id = {}",
                            shader_path.to_string_lossy(),
                            id
                        );
                        output.extend(quote! {
                            compile_error!(#msg);
                        });
                    }
                }
            }

            pipe.specialization_constants.as_ref().map(|x| {
                for ArrowPair((rusty_id, _field)) in x.iter() {
                    if entry
                        .spec
                        .spec_consts()
                        .any(|c| c.spec_id == rusty_id.base10_parse::<u32>().unwrap())
                    {
                        continue;
                    }
                    let msg = format!(
                        "shader {} missing shader side of spec const id = {}",
                        shader_path.to_string_lossy(),
                        rusty_id
                    );
                    output.extend(quote! {
                        compile_error!(#msg);
                    });
                }
            });

            for desc in entry.descs() {
                let rusty = sets
                    .0
                    .iter()
                    .find(|set| {
                        // TODO: more validation
                        &set.name
                            == pipe.descriptors[desc.desc_bind.set() as usize]
                                .get_ident()
                                .expect("failed to find a rust-side descriptor for binding")
                    })
                    .expect("failed to find a rust-side descriptor set");
                let rusty_binding = &rusty.bindings[desc.desc_bind.bind() as usize];

                match desc.desc_ty {
                    spirq::ty::DescriptorType::StorageBuffer(n, spirq::Type::Struct(s))
                    | spirq::ty::DescriptorType::UniformBuffer(n, spirq::Type::Struct(s)) => {
                        match (desc.desc_ty, rusty_binding.descriptor_type.to_string().as_str()) {
                            (spirq::ty::DescriptorType::StorageBuffer(..), "STORAGE_BUFFER") => {}
                            (spirq::ty::DescriptorType::UniformBuffer(..), "UNIFORM_BUFFER") => {}
                            (spirq::ty::DescriptorType::Image(..), "COMBINED_IMAGE_SAMPLER") => {}
                            (spir_ty, rusty_ty) => {
                                let msg = format!(
                                    "Incorrect shader binding at set {} binding {}, shader declares {:?}, rusty binding is {}",
                                    rusty.name.to_string(),
                                    rusty_binding.name.to_string(),
                                    spir_ty,
                                    rusty_ty,
                                );
                                output.extend_one(quote! {
                                    compile_error!(#msg);
                                })
                            }
                        }
                        if *n != rusty_binding.count.base10_parse::<u32>().unwrap() {
                            let msg = format!(
                                "Wrong descriptor count for set {} binding {}, shader needs {}",
                                rusty.name.to_string(),
                                rusty_binding.name.to_string(),
                                n
                            );
                            output.extend_one(quote! {
                                compile_error!(#msg);
                            })
                        }
                        let name = Ident::new(s.name().unwrap(), proc_macro2::Span::call_site());
                        let (prerequisites, _name) = spirq_type_to_rust(&spirq::Type::Struct(s.clone()));
                        for (name, spirq_ty, definition) in prerequisites.into_iter() {
                            defined_types
                                .entry(name)
                                .and_modify(|placed| {
                                    if placed != &spirq_ty {
                                        let msg = format!(
                                            "defined_types mismatch:\n\
                                            previous: {:?}\n\
                                            incoming: {:?}",
                                            placed, &spirq_ty
                                        );
                                        output.extend_one(quote! {compile_error!(#msg);});
                                    }
                                })
                                .or_insert_with(|| {
                                    output.extend_one(definition);
                                    spirq_ty.clone()
                                });
                        }

                        set_binding_type_names
                            .entry((rusty.name.clone(), rusty_binding.name.clone()))
                            .and_modify(|placed_ty| {
                                if placed_ty != &name {
                                    let msg = format!(
                                        "set binding type name mismatch:\n\
                                            previous: {:?}\n\
                                            incoming: {:?}",
                                        placed_ty.to_string(),
                                        &name.to_string()
                                    );
                                    output.extend_one(quote! {compile_error!(#msg);});
                                }
                            })
                            .or_insert(name);
                    }
                    _ => {}
                }
            }

            if let Some(push_const) = entry.get_push_const() {
                let (prereqs, _) = spirq_type_to_rust(push_const);
                assert!(
                    push_constant_type.is_none(),
                    "no support for reused/multiple push constant ranges between shader stages"
                );
                push_constant_type = Some(prereqs.into_iter().map(|(_, _, tokens)| tokens).collect());
                push_constant_spir_type = Some(push_const.clone());
            }
        }

        rga::dump_rga(&sets.0, pipe, push_constant_spir_type.as_ref());
        output.extend_one(define_pipe(pipe, push_constant_type))
    }

    output.extend(sets.iter().map(|set| define_set(set, &set_binding_type_names)));

    output
}

fn shader_stage_to_file_extension(id: &Ident) -> &'static str {
    if id == "VERTEX" {
        "vert"
    } else if id == "COMPUTE" {
        "comp"
    } else if id == "FRAGMENT" {
        "frag"
    } else {
        unimplemented!("Unknown shader stage")
    }
}

fn compare_types(spv: &spirq::Type, rust: &syn::Type) -> bool {
    match (spv, rust) {
        (spirq::Type::Scalar(spirq::ty::ScalarType::Signed(4)), syn::Type::Path(p)) => p.path == parse_quote!(i32),
        (spirq::Type::Scalar(spirq::ty::ScalarType::Unsigned(4)), syn::Type::Path(p)) => p.path == parse_quote!(u32),
        (spirq::Type::Scalar(spirq::ty::ScalarType::Unsigned(2)), syn::Type::Path(p)) => p.path == parse_quote!(u16),
        _ => unimplemented!("unimplemented spirq type comparison {:?} and {:?}", spv, rust,),
    }
}

// returns prerequisite + struct field definition
fn spirq_type_to_rust(spv: &spirq::Type) -> (Vec<(String, spirq::Type, TokenStream)>, TokenStream) {
    use spirq::*;
    match spv {
        Type::Matrix(ty::MatrixType {
            vec_ty:
                ty::VectorType {
                    scalar_ty: ty::ScalarType::Float(4),
                    nscalar: 4,
                },
            nvec: 4,
            stride: 16,
            major: ty::MatrixAxisOrder::ColumnMajor,
        }) => (vec![], quote!(glm::Mat4)),
        Type::Scalar(ty::ScalarType::Float(4)) => (vec![], quote!(f32)),
        Type::Scalar(ty::ScalarType::Signed(4)) => (vec![], quote!(i32)),
        Type::Scalar(ty::ScalarType::Unsigned(4)) => (vec![], quote!(u32)),
        Type::Scalar(ty::ScalarType::Unsigned(2)) => (vec![], quote!(u16)),
        Type::Struct(s) => {
            let name = Ident::new(s.name().unwrap(), Span::call_site());
            let (field_name, field_offset, field_ty) = split3((0..s.nmember()).map(|ix| {
                let field = s.get_member(ix).unwrap();
                let field_name = field.name.as_ref().unwrap().to_case(Case::Snake);
                (
                    Ident::new(&field_name, Span::call_site()),
                    field.offset,
                    spirq_type_to_rust(&field.ty),
                )
            }));
            let (prereq, field_ty) = split2(field_ty.into_iter());
            let mut prereq = prereq.into_iter().flatten().collect_vec();

            let zero = quote!(0usize);
            let rust_offset = field_ty.iter().fold(vec![zero], |mut offsets, ty| {
                let last = offsets.last().unwrap();
                let this = quote!(size_of::<#ty>());
                let new = quote!(#last + #this);
                offsets.push(new);
                offsets
            });

            prereq.push((s.name().unwrap().to_string(), spv.clone(), quote! {
                #[repr(C)]
                pub(crate) struct #name {
                    #(pub(crate) #field_name : #field_ty),*
                }

                #(
                    static_assertions::const_assert_eq!(#rust_offset, #field_offset);
                )*
            }));

            (prereq, quote!(#name))
        }
        Type::Vector(ty::VectorType {
            scalar_ty: ty::ScalarType::Float(4),
            nscalar: 4,
        }) => (vec![], quote!(glm::Vec4)),
        Type::Vector(ty::VectorType {
            scalar_ty: ty::ScalarType::Unsigned(4),
            nscalar: 3,
        }) => (vec![], quote!(glm::UVec3)),
        Type::Vector(ty::VectorType {
            scalar_ty: ty::ScalarType::Float(4),
            nscalar: 3,
        }) => (vec![], quote!(glm::Vec3)),
        Type::Vector(ty::VectorType {
            scalar_ty: ty::ScalarType::Float(4),
            nscalar: 2,
        }) => (vec![], quote!(glm::Vec2)),
        Type::Array(arr) if arr.nrepeat().is_some() => {
            let nrepeat = arr.nrepeat().unwrap() as usize;
            let (prereq, inner) = spirq_type_to_rust(arr.proto_ty());
            (prereq, quote!([#inner; #nrepeat]))
        }
        _ => unimplemented!("spirq_type_to_rust {:?}", spv),
    }
}

#[test]
fn test_timeline1() {
    use std::{io::Write, process::Stdio};
    let unformatted = format!("{}", define_timeline2(quote! {pub(crate) TestTimeline [One, Two]}));
    let mut child = std::process::Command::new("rustfmt")
        .args(&["--config", "newline_style=Unix"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(unformatted.as_bytes());
    let output = child.wait_with_output().unwrap();
    let formatted = std::str::from_utf8(&output.stdout).unwrap();
    pretty_assertions::assert_eq!(formatted, indoc::indoc! {"
        #[allow(non_snake_case)]
        pub(crate) mod TestTimeline {
            use super::{ImageIndex, RenderFrame, SwapchainIndexToFrameNumber};
            pub(crate) struct One;
            impl One {
                pub(crate) const VALUE: u64 = 1u64;

                pub(crate) const fn as_of(&self, frame_number: u64) -> u64 {
                    frame_number * 2u64 + 1u64
                }

                pub(crate) const fn as_of_last(&self, frame_number: u64) -> u64 {
                    self.as_of(frame_number - 1)
                }

                pub(crate) fn as_of_previous(&self, image_index: &ImageIndex, indices: &SwapchainIndexToFrameNumber) -> u64 {
                    let frame_number = indices.map[image_index.0 as usize];
                    self.as_of(frame_number)
                }
            }
            pub(crate) struct Two;
            impl Two {
                pub(crate) const VALUE: u64 = 2u64;

                pub(crate) const fn as_of(&self, frame_number: u64) -> u64 {
                    frame_number * 2u64 + 2u64
                }

                pub(crate) const fn as_of_last(&self, frame_number: u64) -> u64 {
                    self.as_of(frame_number - 1)
                }

                pub(crate) fn as_of_previous(&self, image_index: &ImageIndex, indices: &SwapchainIndexToFrameNumber) -> u64 {
                    let frame_number = indices.map[image_index.0 as usize];
                    self.as_of(frame_number)
                }
            }
        }
    "});
}

fn prepare_queue_manager(
    input: &parsed::RendererInput,
    graph_ixes: &HashMap<Ident, u32>,
    dependency_graph: &DiGraph<Ident, DependencyType>,
) -> TokenStream {
    assert!(
        input.passes.len() + input.async_passes.len() < usize::from(u8::MAX),
        "queue_manager can only handle dependency graphs with u8 indices"
    );

    let edges_definitions = dependency_graph
        .edge_references()
        .map(|edge| {
            let source = edge.source().index() as u8;
            let target = edge.target().index() as u8;
            quote! {
                (#source, #target)
            }
        })
        .collect_vec();

    let toposort_compute_virtual_queue_index = {
        let toposort = petgraph::algo::toposort(&dependency_graph, None).unwrap();
        let toposort_compute = toposort
            .iter()
            .filter(|ix| {
                let name = dependency_graph.node_weight(**ix).unwrap();
                input
                    .async_passes
                    .get(name)
                    .map(|async_pass| async_pass.queue == parsed::QueueFamily::Compute)
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();

        let mut toposort_grouped_compute: Vec<Vec<NodeIndex>> = vec![];
        for ix in toposort_compute.iter() {
            match toposort_grouped_compute.last_mut() {
                // if no path bridges from the last stage to ix
                Some(last)
                    if !last.iter().any(|candidate| {
                        petgraph::algo::has_path_connecting(&dependency_graph, *candidate, **ix, None)
                    }) =>
                {
                    last.push(**ix);
                }
                _ => toposort_grouped_compute.push(vec![**ix]),
            }
        }
        let mut mapping = HashMap::new();
        for stage in toposort_grouped_compute {
            for (queue_ix, node_ix) in stage.iter().enumerate() {
                assert!(mapping.insert(node_ix.index(), queue_ix).is_none());
            }
        }
        mapping
    };

    let submit_clauses = input
        .passes
        .values()
        .map(|parsed::Pass { ref name, .. }| (name, parsed::QueueFamily::Graphics))
        .chain(
            input
                .async_passes
                .values()
                .map(|parsed::AsyncPass { name, queue, .. }| (name, *queue)),
        )
        .map(|(name, queue)| {
            let queue_def = match queue {
                parsed::QueueFamily::Graphics => quote!(renderer.device.graphics_queue()),
                parsed::QueueFamily::Compute => {
                    let ix = graph_ixes.get(name).unwrap();
                    let virtual_queue_index = toposort_compute_virtual_queue_index.get(&(*ix as usize)).unwrap();
                    quote!(renderer.device.compute_queues[#virtual_queue_index % renderer.device.compute_queues.len()])
                }
                parsed::QueueFamily::Transfer => quote!(renderer.device.transfer_queue.as_ref().unwrap()),
            };
            quote! {
                ix if ix.index() as u8 == #name::INDEX => {
                    let queue = #queue_def.lock();
                    let buf = &mut [vk::CommandBuffer::null()];
                    let cmds: &[vk::CommandBuffer] = match cb {
                        Some(cmd) => {
                            buf[0] = cmd;
                            buf
                        }
                        None => &[],
                    };
                    #name::Stage::queue_submit(&image_index, &renderer, *queue, cmds).unwrap();
                }
            }
        })
        .collect_vec();
    let update_scope_name = format!("{}::update_submissions", input.name.to_string());

    quote! {
        use bevy_ecs::prelude::*;

        pub(crate) fn setup_submissions(
            renderer: &RenderFrame,
            graph: &mut petgraph::stable_graph::StableDiGraph::<Option<Option<vk::CommandBuffer>>, (), u8>,
        ) {
            assert_eq!(graph.node_count(), 0);
            assert_eq!(graph.edge_count(), 0);

            graph.extend_with_edges(&[#(#edges_definitions),*]);
        }

        pub(crate) fn update_submissions(
            renderer: &RenderFrame,
            image_index: &ImageIndex,
            mut graph: parking_lot::MutexGuard<petgraph::stable_graph::StableDiGraph::<Option<Option<vk::CommandBuffer>>, (), u8>>,
        ) {
            use microprofile::scope;
            scope!("macros", #update_scope_name);
            use petgraph::{Direction, stable_graph::{NodeIndex}};

            let submit = |ix: NodeIndex<u8>, cb: Option<vk::CommandBuffer>| {
                match ix {
                    #(#submit_clauses),*
                    _ => panic!("Invalid pass index in queue_manager submit()"),
                }
            };

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
                            parking_lot::MutexGuard::unlocked_fair(&mut graph, || {
                                submit(node, cb.unwrap());
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
    }
}
