#![feature(extend_one)]

use std::{env, fmt::Debug, fs::File, io::Read};

use convert_case::{Case, Casing};
use derive_more::Deref;
use derive_syn_parse::Parse;
use hashbrown::HashMap;
use itertools::Itertools;
use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::EdgeRef,
    Direction,
};
use proc_macro2::{Span, TokenStream, TokenTree};
use quote::{format_ident, quote, ToTokens};
use syn::{
    braced, bracketed, custom_keyword,
    parse::{Parse, ParseStream, Result},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    token::Brace,
    Expr, Field, Ident, ItemStruct, LitBool, LitInt, Path, Token, Visibility,
};

#[proc_macro]
pub fn define_timeline(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
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
    } = parse_macro_input!(input as TimelineInput);

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

    let expanded = quote! {
        #[allow(non_snake_case)]
        #visibility mod #name {
            use super::{RenderFrame, ImageIndex, SwapchainIndexToFrameNumber};

            #(
                #variant2
            )*
        }
    };

    proc_macro::TokenStream::from(expanded)
}

struct Pair((Ident, TokenTree));

impl Parse for Pair {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        Ok(Pair((name, input.parse()?)))
    }
}

#[derive(Debug, Deref)]
struct ArrowPair<A, B>((A, B));

impl<A: Parse, B: Parse> Parse for ArrowPair<A, B> {
    fn parse(input: ParseStream) -> Result<Self> {
        let a = input.parse()?;
        input.parse::<Token![=>]>()?;
        let b = input.parse()?;
        Ok(ArrowPair((a, b)))
    }
}

#[derive(Clone, Debug, Deref)]
struct Sequence<A, B>((A, B));

impl<A: Parse + Debug, B: Parse + Debug> Parse for Sequence<A, B> {
    fn parse(input: ParseStream) -> Result<Self> {
        let peeking = input.fork();
        let a = peeking.parse::<A>();
        let b = peeking.parse::<B>();

        if a.is_ok() && b.is_ok() {
            Ok(Sequence((input.parse().unwrap(), input.parse().unwrap())))
        } else {
            // This is so that Sequence only consumes anything if it can parse both
            // otherwise it gets confused when Sequence<A, B> is used back to back with Sequence<A, C>
            Err(syn::Error::new(Span::call_site(), "asd"))
        }
    }
}

#[derive(Debug, Deref)]
struct Unbrace<T>(T);

impl<T: Parse> Parse for Unbrace<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        let inner;
        braced!(inner in input);
        let t = inner.parse()?;
        Ok(Unbrace(t))
    }
}

#[derive(Debug, Deref)]
struct Unbracket<T>(T);

impl<T: Parse> Parse for Unbracket<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        let inner;
        bracketed!(inner in input);
        let t = inner.parse()?;
        Ok(Unbracket(t))
    }
}

#[derive(Debug, Deref)]
struct UnArray<T>(Vec<T>);

impl<T: Parse> Parse for UnArray<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(UnArray(
            Punctuated::<T, Token![,]>::parse_terminated(input)?
                .into_iter()
                .collect(),
        ))
    }
}

#[derive(Clone, Debug, Deref)]
struct UnOption<T>(Option<T>);

impl<T: Parse> Parse for UnOption<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<T>().map(|t| UnOption(Some(t))).or(Ok(UnOption(None)))
    }
}

#[derive(Clone, Debug)]
enum StaticOrDyn<T> {
    Static(T),
    Dyn,
}

impl<T: Parse> Parse for StaticOrDyn<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        input
            .parse::<Token![dyn]>()
            .map(|_| StaticOrDyn::Dyn)
            .or_else(|_| input.parse::<T>().map(StaticOrDyn::Static))
    }
}

impl<T> StaticOrDyn<T> {
    fn is_dyn(&self) -> bool {
        match self {
            StaticOrDyn::Static(_) => false,
            StaticOrDyn::Dyn => true,
        }
    }

    #[allow(dead_code)]
    fn unwrap_or(self, default: T) -> T {
        match self {
            StaticOrDyn::Static(s) => s,
            StaticOrDyn::Dyn => default,
        }
    }

    fn as_ref(&self) -> StaticOrDyn<&T> {
        match self {
            StaticOrDyn::Static(s) => StaticOrDyn::Static(&s),
            StaticOrDyn::Dyn => StaticOrDyn::Dyn,
        }
    }
}

impl<T: PartialEq + Debug> StaticOrDyn<T> {
    fn unwrap_or_warn_redundant(self, default: T, error_container: &mut TokenStream) -> T {
        match self {
            StaticOrDyn::Static(s) => {
                if s == default {
                    let msg = format!(
                        "redundant static definition, would default to this anyway {:?}",
                        default,
                    );
                    error_container.extend_one(quote!(compile_error!(#msg)));
                    default
                } else {
                    s
                }
            }
            StaticOrDyn::Dyn => default,
        }
    }
}

fn extract_optional_dyn<T, R>(a: &Option<StaticOrDyn<T>>, when_dyn: R) -> Option<R> {
    match a {
        Some(StaticOrDyn::Dyn) => Some(when_dyn),
        _ => None,
    }
}

#[derive(Parse)]
struct Pass {
    name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[inside(brace)]
    #[allow(dead_code)]
    color_kw: Option<kw::color>,
    #[inside(brace)]
    #[parse_if(color_kw.is_some())]
    color: Option<Unbracket<UnArray<Ident>>>,
    #[inside(brace)]
    #[allow(dead_code)]
    depth_stencil_kw: Option<kw::depth_stencil>,
    #[inside(brace)]
    #[parse_if(depth_stencil_kw.is_some())]
    depth_stencil: Option<Ident>,
    #[prefix(kw::layouts in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    layouts_brace: Brace,
    #[inside(layouts_brace)]
    layouts: UnArray<Sequence<Ident, ArrowPair<Sequence<LoadOp, Expr>, Sequence<StoreOp, Expr>>>>,
    #[inside(brace)]
    #[allow(dead_code)]
    subpasses_kw: kw::subpasses,
    #[inside(brace)]
    subpasses: Unbrace<UnArray<Subpass>>,
    #[inside(brace)]
    #[allow(dead_code)]
    dependencies_kw: Option<kw::dependencies>,
    #[parse_if(dependencies_kw.is_some())]
    #[inside(brace)]
    dependencies: Option<Unbrace<UnArray<SubpassDependency>>>,
}

#[derive(Parse)]
struct Subpass {
    name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[inside(brace)]
    #[allow(dead_code)]
    color_kw: Option<kw::color>,
    #[inside(brace)]
    #[parse_if(color_kw.is_some())]
    color: Option<Unbracket<UnArray<ArrowPair<Ident, Ident>>>>,
    #[inside(brace)]
    #[allow(dead_code)]
    depth_stencil_kw: Option<kw::depth_stencil>,
    #[inside(brace)]
    #[parse_if(depth_stencil_kw.is_some())]
    depth_stencil: Option<Unbrace<ArrowPair<Ident, Ident>>>,
}

#[derive(Parse)]
struct SubpassDependency {
    from: Ident,
    #[allow(dead_code)]
    arrow_1: Token![=>],
    to: Ident,
    src_stage: Expr,
    #[allow(dead_code)]
    arrow_2: Token![=>],
    dst_stage: Expr,
    src_access: Expr,
    #[allow(dead_code)]
    arrow_3: Token![=>],
    dst_access: Expr,
}

#[derive(Debug, PartialEq)]
enum LoadOp {
    Load,
    Clear,
    DontCare,
}

impl Parse for LoadOp {
    fn parse(input: ParseStream) -> Result<Self> {
        input
            .parse::<kw::load>()
            .and(Ok(Self::Load))
            .or_else(|_| input.parse::<kw::clear>().and(Ok(Self::Clear)))
            .or_else(|_| input.parse::<kw::dont_care>().and(Ok(Self::DontCare)))
    }
}

#[derive(Debug)]
enum StoreOp {
    Store,
    Discard,
}

impl Parse for StoreOp {
    fn parse(input: ParseStream) -> Result<Self> {
        input
            .parse::<kw::store>()
            .and(Ok(Self::Store))
            .or_else(|_| input.parse::<kw::discard>().and(Ok(Self::Discard)))
    }
}

impl Pass {
    fn attachments(&self) -> Vec<Ident> {
        let zero = vec![];
        self.color
            .as_ref()
            .map(|Unbracket(UnArray(color_attachments))| color_attachments)
            .unwrap_or(&zero)
            .iter()
            .chain(self.depth_stencil.iter())
            .cloned()
            .collect()
    }
}

#[derive(Parse)]
struct FrameInput {
    visibility: Visibility,
    #[allow(dead_code)]
    name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[prefix(kw::attachments in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    attachments_brace: Brace,
    #[inside(attachments_brace)]
    attachments: UnArray<Ident>,
    #[prefix(kw::formats in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    formats_brace: Brace,
    #[inside(formats_brace)]
    formats: UnArray<StaticOrDyn<Expr>>,
    #[prefix(kw::passes in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    passes_brace: Brace,
    #[inside(passes_brace)]
    passes: UnArray<Pass>,
    #[prefix(kw::async_passes in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    async_passes_brace: Brace,
    #[inside(async_passes_brace)]
    async_passes: UnArray<Ident>,
    #[prefix(kw::dependencies in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    dependencies_brace: Brace,
    #[inside(dependencies_brace)]
    dependencies: UnArray<ArrowPair<Ident, Sequence<Ident, UnOption<DependencyType>>>>,
    #[prefix(kw::sync in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    sync_brace: Brace,
    #[inside(sync_brace)]
    sync: UnArray<ArrowPair<Ident, Expr>>,
}
#[derive(Debug, Clone, Copy)]
enum DependencyType {
    SameFrame,
    LastFrame,
    LastAccess,
}

impl Parse for DependencyType {
    fn parse(input: ParseStream) -> Result<Self> {
        let inner;
        bracketed!(inner in input);
        inner
            .parse::<kw::same_frame>()
            .and(Ok(DependencyType::SameFrame))
            .or(inner.parse::<kw::last_frame>().and(Ok(DependencyType::LastFrame)))
            .or(inner.parse::<kw::last_access>().and(Ok(DependencyType::LastAccess)))
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
        passes: UnArray(passes),
        async_passes: UnArray(async_passes),
        dependencies,
        sync: UnArray(sync),
        ..
    } = &frame_input;

    let sync: HashMap<String, Expr> = sync
        .into_iter()
        .map(|ArrowPair((ident, exp))| (ident.to_string(), exp.clone()))
        .collect();

    let graph_ixes = passes
        .iter()
        .map(|pass| &pass.name)
        .chain(async_passes.iter())
        .enumerate()
        .map(|(ix, a)| (a.to_string(), ix as u32))
        .collect::<HashMap<String, u32>>();

    let all_passes = passes
        .iter()
        .map(|pass| &pass.name)
        .chain(async_passes.iter())
        .collect::<Vec<_>>();

    let dependency_graph = DiGraph::<(), DependencyType, u32>::from_edges(dependencies.iter().map(
        |ArrowPair((from, Sequence((to, dep))))| {
            (
                *graph_ixes.get(&from.to_string()).unwrap(),
                *graph_ixes.get(&to.to_string()).unwrap(),
                dep.0.unwrap_or(DependencyType::SameFrame),
            )
        },
    ));

    let dynamic_attachments = attachments
        .iter()
        .zip(formats.iter())
        .enumerate()
        .flat_map(|(ix, (_, format))| if format.is_dyn() { Some(ix) } else { None })
        .collect::<Vec<_>>();

    let wait_instance = passes
        .iter()
        .map(|pass| &pass.name)
        .chain(async_passes.iter())
        .map(|pass| {
            let signal_out = sync.get(&pass.to_string()).expect("no sync for this pass");
            let t = syn::parse2::<syn::Path>(signal_out.to_token_stream())
                .expect("sync values must consist of 2 path segments");
            let signal_timeline_member = format_ident!(
                "{}_semaphore",
                t.segments.first().unwrap().ident.to_string().to_case(Case::Snake)
            );

            let wait_inner = dependency_graph
                .edges_directed(
                    NodeIndex::from(*graph_ixes.get(&pass.to_string()).unwrap()),
                    Direction::Incoming,
                )
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
                        semaphores.push(render_frame.#timeline_member.handle);
                        values.push(super::super::#signaled.#as_of);
                    }
                })
                .collect::<Vec<_>>();

            quote! {
                impl RenderStage for Stage {
                    fn prepare_signal(render_frame: &RenderFrame, semaphores: &mut Vec<vk::Semaphore>,
                                      values: &mut Vec<u64>) {
                        semaphores.push(render_frame.#signal_timeline_member.handle);
                        values.push(super::super::#signal_out.as_of(render_frame.frame_number));
                    }

                    fn prepare_wait(image_index: &ImageIndex, render_frame: &RenderFrame,
                                    semaphores: &mut Vec<vk::Semaphore>, values: &mut Vec<u64>) {
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
        .map(|(pass, wait_instance)| {
            let pass_name = &pass.name;
            let format_param_ty = pass
                .attachments()
                .iter()
                .filter(|attachment| {
                    let attachment_ix = attachments.iter().position(|at| at == *attachment).unwrap();
                    let format = &formats[attachment_ix];
                    format.is_dyn()
                })
                .map(|_| quote!(vk::Format))
                .collect_vec();
            let attachment_desc = pass
                .attachments()
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
                    quote! {
                        vk::AttachmentDescription::builder()
                            .format(#format)
                            .samples(vk::SampleCountFlags::TYPE_1)
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
                    use super::{vk, DeviceV1_0, Device, RenderStage, RenderFrame, ImageIndex,
                        OriginalFramebuffer, OriginalRenderPass};

                    #[derive(Debug, Clone, Copy)]
                    pub(crate) struct Stage;

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
        // .zip(passes.len()..)
        .zip(wait_instance.skip(passes.len()))
        .map(|(pass, wait_instance)| {
            quote! {
                #[allow(non_snake_case)]
                pub(crate) mod #pass {
                    use super::{vk, DeviceV1_0, RenderStage, RenderFrame, ImageIndex};

                    #[derive(Debug, Clone, Copy)]
                    pub(crate) struct Stage;

                    #wait_instance
                }
            }
        });

    let expanded = quote! {
        pub(crate) mod #name {
            use ash::{vk, version::DeviceV1_0};
            use super::{Device, RenderStage, RenderFrame, ImageIndex,
                        RenderPass as OriginalRenderPass,
                        Framebuffer as OriginalFramebuffer};

            #(
                #pass_definitions
            )*

            #(
                #async_pass_definitions
            )*
        }
    };

    proc_macro::TokenStream::from(expanded)
}

fn define_renderpass(frame_input: &FrameInput, pass: &Pass) -> TokenStream {
    let pass_attachments = pass.attachments();
    let passes_that_need_clearing = pass_attachments.iter().enumerate().map(|(ix, attachment)| {
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
    let format_param_ty = pass_attachments
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
        let color_attachment_ix = color_attachment_name.iter().map(|needle| {
            pass.attachments()
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
            pass.attachments()
                .iter()
                .position(|candidate| candidate == needle)
                .expect("subpass depth stencil refers to nonexistent attachment")
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
            },
            quote! {
                vk::SubpassDescription::builder()
                    .color_attachments(#color_attachment_var)
                    #( .depth_stencil_attachment(&#depth_stencil_var) )*
                    // #( #bind_depth_stencil  )*
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
                clear_values: &[vk::ClearValue; #passes_that_need_clearing_len],
            ) {
                use microprofile::scope;
                scope!("macros", #renderpass_begin);

                let clear_values = [
                    #(
                        #passes_that_need_clearing_fetch
                    ),*
                ];
                let begin_info = vk::RenderPassBeginInfo::builder()
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
    let attachment_count = pass.attachments().len();

    quote! {
        pub(crate) struct Framebuffer {
            pub(crate) framebuffer: OriginalFramebuffer,
        }

        impl Framebuffer {
            pub(crate) fn new(
                renderer: &RenderFrame,
                renderpass: &RenderPass,
                attachments: &[vk::ImageView; #attachment_count],
                (width, height): (u32, u32),
                ix: u32,
            ) -> Self {
                let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(renderpass.renderpass.handle)
                    .attachments(attachments)
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
                    .set_object_name(handle, &format!(#debug_name, ix));
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
#[proc_macro]
pub fn submit_coalesced(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let UnArray(stages) = parse_macro_input!(input as UnArray<ArrowPair<Expr, Expr>>);

    let mut built_up = quote! {
        unsafe {
            renderer
                .device
                .queue_submit(
                    *queue,
                    &submits,
                    vk::Fence::null(),
                )
        }
    };

    for ArrowPair((stage, command_buffers)) in stages.iter().rev() {
        built_up = quote! {
            #stage::Stage::submit_info(
                &image_index,
                &renderer,
                #command_buffers,
                |submit| {
                    submits.push(submit);
                    #built_up
                }
            )
        };
    }

    let out = quote! {
        {
            let mut submits = vec![];
            #built_up
        }
    };

    proc_macro::TokenStream::from(out)
}

#[proc_macro]
pub fn to_vk_format(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    custom_keyword!(vec2);
    custom_keyword!(vec3);
    custom_keyword!(vec4);

    let parser = |input: ParseStream| {
        input
            .parse::<vec2>()
            .and(Ok(quote!(vk::Format::R32G32_SFLOAT)))
            .or_else(|_| input.parse::<vec3>().and(Ok(quote!(vk::Format::R32G32B32_SFLOAT))))
            .or_else(|_| input.parse::<vec4>().and(Ok(quote!(vk::Format::R32G32B32A32_SFLOAT))))
    };

    let out = parse_macro_input!(input with parser);

    proc_macro::TokenStream::from(out)
}

#[proc_macro]
pub fn to_rust_type(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    custom_keyword!(vec2);
    custom_keyword!(vec3);
    custom_keyword!(vec4);

    let parser = |input: ParseStream| {
        input
            .parse::<vec2>()
            .and(Ok(quote!(glm::Vec2)))
            .or_else(|_| input.parse::<vec3>().and(Ok(quote!(glm::Vec3))))
            .or_else(|_| input.parse::<vec4>().and(Ok(quote!(glm::Vec4))))
            .or_else(|_| input.parse::<TokenTree>().map(|t| quote!(#t)))
    };

    let out = parse_macro_input!(input with parser);

    proc_macro::TokenStream::from(out)
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

#[derive(Debug, Deref)]
struct NamedField(Field);
impl Parse for NamedField {
    fn parse(input: ParseStream) -> Result<Self> {
        Field::parse_named(input).map(NamedField)
    }
}

fn define_set(set: &DescriptorSet, set_binding_type_names: &HashMap<(Ident, Ident), Ident>) -> TokenStream {
    let DescriptorSet {
        name: set_name,
        bindings,
        ..
    } = set;

    let binding_count = bindings.len();
    let (binding_ix, partially_bound, desc_type, desc_count, stage) =
        split5(bindings.iter().enumerate().map(|(ix, binding)| {
            let Binding {
                descriptor_type,
                partially_bound,
                count,
                stages,
                ..
            } = binding;
            (
                ix,
                partially_bound.is_some(),
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
                        use ash::{version::DeviceV1_0, vk};
                        use super::super::super::{#ty, StaticBuffer};
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
            use super::{vk, DescriptorSetLayout, DescriptorSet, Device, Buffer, MainDescriptorPool, DescriptorPool};

            pub(crate) struct Layout {
                pub(crate) layout: DescriptorSetLayout,
            }

            impl Layout {
                pub(crate) fn new(device: &Device) -> Layout {
                    let binding_flags = &[
                        #(
                            {
                                let _x = #binding_ix;
                                if #partially_bound {
                                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                } else {
                                    vk::DescriptorBindingFlags::default()
                                }
                            }
                        ),*
                    ];
                    let mut binding_flags =
                        vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                            .binding_flags(binding_flags);
                    let bindings = Layout::bindings();
                    let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&bindings)
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

            #[deprecated(note = "compatibility with old macro")]
            pub(crate) fn bindings() -> [vk::DescriptorSetLayoutBinding; #binding_count] {
                Layout::bindings()
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
                            TokenStream::from(to_vk_format(field.ty.to_token_stream().into())),
                            TokenStream::from(to_rust_type(field.ty.to_token_stream().into())),
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
        spirv_code.extend_one(quote! {
            pub(crate) static #shader_stage: &'static [u8] = include_bytes_align_as!(u32, #shader_path);
        });
    }

    let mut errors = quote!();

    let pipeline_definition = match specific {
        SpecificPipe::Graphics(specific) => {
            let pipe_debug_name = format!("{}::Pipeline", pipe.name.to_string());
            let specialize_msg = format!("{}::Pipeline::specialize", pipe.name.to_string());
            let new_internal_msg = format!("{}::Pipeline::new_internal", pipe.name.to_string());
            let stage_flag = specific.stages.iter().cloned().collect_vec();
            let stage_ix = stage_flag.iter().enumerate().map(|(ix, _)| ix).collect_vec();
            let stage_count = specific.stages.len();

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

            quote! {
                pub(crate) struct Pipeline {
                    pub(crate) pipeline: device::Pipeline,
                    specialization: Specialization
                }

                impl Pipeline {
                    pub(crate) fn new(
                        device: &Device,
                        layout: &PipelineLayout,
                        specialization: Specialization,
                        base_pipeline: Option<&Self>,
                        shaders: Option<&[&device::Shader; #stage_count]>,
                        renderpass: &device::RenderPass,
                        subpass: u32
                    ) -> Self {
                        scope!("macros", #new_internal_msg);

                        use std::ffi::CStr;
                        let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
                        let base_pipeline_handle = base_pipeline
                            .map(|pipe| *pipe.pipeline)
                            .unwrap_or_else(vk::Pipeline::null);
                        let shaders = shaders.ok_or_else(|| {
                            [
                                #( device.new_shader(#stage_flag) ),*
                            ]
                        });
                        let shader_handles: Vec<&device::Shader> = shaders
                            .as_ref()
                            .map(|borrowed| borrowed.iter().map(|b| *b).collect())
                            .unwrap_or_else(|owned| owned.iter().map(|owned| owned).collect());
                        let spec_info = specialization.get_spec_info();
                        let mut flags = vk::PipelineCreateFlags::ALLOW_DERIVATIVES;
                        if base_pipeline.is_some() {
                            flags |= vk::PipelineCreateFlags::DERIVATIVE;
                        }

                        let stage_shaders = [
                            #( (vk::ShaderStageFlags::#stage_flag, shader_handles[#stage_ix], Some(&spec_info)) ),*
                        ];
                        let pipeline = device.new_graphics_pipeline(
                            &stage_shaders,
                            vk::GraphicsPipelineCreateInfo::builder()
                                .vertex_input_state(&vertex_input_state())
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
                                        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
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
                        device.set_object_name(*pipeline, #pipe_debug_name);

                        shaders.err().map(|shaders| std::array::IntoIter::new(shaders).for_each(|s| s.destroy(device)));

                        Pipeline {
                            pipeline,
                            specialization,
                        }
                    }

                    /// Re-specializes the pipeline if needed and returns the old one that might still be in use.
                    pub(crate) fn specialize(
                        &mut self,
                        device: &Device,
                        layout: &PipelineLayout,
                        new_spec: &Specialization,
                        shaders: Option<&[&device::Shader; #stage_count]>,
                        renderpass: &device::RenderPass,
                        subpass: u32
                    ) -> Option<Self> {
                        scope!("macros", #specialize_msg);
                        use std::mem::swap;

                        if self.specialization != *new_spec {
                            let mut replacement = Self::new(
                                device,
                                layout,
                                new_spec.clone(),
                                Some(&self),
                                shaders,
                                renderpass,
                                subpass
                            );
                            swap(&mut *self, &mut replacement);
                            Some(replacement)
                        } else {
                            None
                        }
                    }

                    pub(crate) fn spec(&self) -> &Specialization { &self.specialization }

                    pub(crate) fn destroy(self, device: &Device) { self.pipeline.destroy(device); }
                }
                use std::ops::Deref;
            }
        }
        SpecificPipe::Compute(_) => {
            let pipe_debug_name = format!("{}::Pipeline", pipe.name.to_string());
            let specialize_msg = format!("{}::Pipeline::specialize", pipe.name.to_string());
            let new_internal_msg = format!("{}::Pipeline::new_internal", pipe.name.to_string());

            quote! {
                pub(crate) struct Pipeline {
                    pub(crate) pipeline: device::Pipeline,
                    specialization: Specialization
                }

                impl Pipeline {
                    pub(crate) fn new(
                        device: &Device,
                        layout: &PipelineLayout,
                        spec: Specialization,
                        shader: Option<&device::Shader>,
                    ) -> Self {
                        Self::new_internal(device, layout, spec, None, shader)
                    }

                    fn new_internal(
                        device: &Device,
                        layout: &PipelineLayout,
                        specialization: Specialization,
                        base_pipeline: Option<&Self>,
                        shader: Option<&device::Shader>,
                    ) -> Self {
                        scope!("macros", #new_internal_msg);

                        use std::ffi::CStr;
                        let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
                        let base_pipeline_handle = base_pipeline
                            .map(|pipe| *pipe.pipeline)
                            .unwrap_or_else(vk::Pipeline::null);
                        let shader = shader.ok_or_else(|| device.new_shader(COMPUTE));
                        let shader_handle = shader.as_ref().map(|s| s.vk()).unwrap_or_else(|s| s.vk());
                        let spec_info = specialization.get_spec_info();
                        let mut flags = vk::PipelineCreateFlags::ALLOW_DERIVATIVES;
                        if base_pipeline.is_some() {
                            flags |= vk::PipelineCreateFlags::DERIVATIVE;
                        }
                        let pipeline = device.new_compute_pipelines(&[
                            vk::ComputePipelineCreateInfo::builder()
                                .stage(
                                    vk::PipelineShaderStageCreateInfo::builder()
                                        .module(shader_handle)
                                        .name(&shader_entry_name)
                                        .stage(vk::ShaderStageFlags::COMPUTE)
                                        .specialization_info(&spec_info)
                                        .build(),
                                )
                                .layout(*layout.layout)
                                .flags(flags)
                                .base_pipeline_handle(base_pipeline_handle)
                                .base_pipeline_index(-1)
                        ]).into_iter().next().unwrap();
                        device.set_object_name(*pipeline, #pipe_debug_name);

                        shader.err().map(|s| s.destroy(device));

                        Pipeline {
                            pipeline,
                            specialization,
                        }
                    }

                    /// Re-specializes the pipeline if needed and returns the old one that might still be in use.
                    pub(crate) fn specialize(
                        &mut self,
                        device: &Device,
                        layout: &PipelineLayout,
                        new_spec: &Specialization,
                        shader: Option<&device::Shader>,
                    ) -> Option<Self> {
                        scope!("macros", #specialize_msg);
                        use std::mem::swap;

                        if self.specialization != *new_spec {
                            let mut replacement = Self::new_internal(
                                device,
                                layout,
                                new_spec.clone(),
                                Some(&self),
                                shader
                            );
                            swap(&mut *self, &mut replacement);
                            Some(replacement)
                        } else {
                            None
                        }
                    }

                    pub(crate) fn spec(&self) -> &Specialization { &self.specialization }

                    pub(crate) fn destroy(self, device: &Device) { self.pipeline.destroy(device); }
                }
                use std::ops::Deref;
            }
        }
    };

    quote! {
        pub(crate) mod #name {
            use crate::renderer::device::{self, Device};
            use ash::{vk, version::DeviceV1_0};
            use std::{mem::size_of, slice::from_raw_parts};
            use microprofile::scope;
            use static_assertions::const_assert_eq;

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

            #pipeline_definition

            #push_constant_type

            #specialization

            #vertex_definitions

            #spirv_code

            #errors
        }
    }
}

mod kw {
    use syn::custom_keyword;

    custom_keyword!(compare);
    custom_keyword!(op);
    custom_keyword!(bounds);
    custom_keyword!(cull);
    custom_keyword!(depth);
    custom_keyword!(test);
    custom_keyword!(write);
    custom_keyword!(front);
    custom_keyword!(face);
    custom_keyword!(topology);
    custom_keyword!(polygon);
    custom_keyword!(mode);
    custom_keyword!(subpasses);
    custom_keyword!(load);
    custom_keyword!(dont_care);
    custom_keyword!(store);
    custom_keyword!(clear);
    custom_keyword!(discard);
    custom_keyword!(color);
    custom_keyword!(depth_stencil);
    custom_keyword!(preserve);
    custom_keyword!(input);
    custom_keyword!(attachments);
    custom_keyword!(formats);
    custom_keyword!(passes);
    custom_keyword!(retain);
    custom_keyword!(layouts);
    custom_keyword!(async_passes);
    custom_keyword!(dependencies);
    custom_keyword!(sync);

    custom_keyword!(same_frame);
    custom_keyword!(last_frame);
    custom_keyword!(last_access);
    custom_keyword!(descriptors);
    custom_keyword!(compute);
    custom_keyword!(graphics);
    custom_keyword!(push_constants);
    custom_keyword!(specialization_constants);
    custom_keyword!(vertex_inputs);
    custom_keyword!(count);
    custom_keyword!(stages);
    custom_keyword!(partially);
    custom_keyword!(bound);
    custom_keyword!(pipelines);
    custom_keyword!(sets);
    custom_keyword!(data_types);
}

#[derive(Parse, Debug)]
struct DataType {
    s: ItemStruct,
}
#[derive(Parse, Debug)]
struct DescriptorSet {
    name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[inside(brace)]
    bindings: UnArray<Binding>,
}
#[derive(Parse, Debug)]
#[allow(dead_code)]
struct Binding {
    name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[prefix(Token![type] in brace)]
    #[inside(brace)]
    descriptor_type: Ident,
    #[inside(brace)]
    partially_bound: UnOption<Sequence<kw::partially, kw::bound>>,
    #[prefix(kw::count in brace)]
    #[inside(brace)]
    count: LitInt,
    #[prefix(kw::stages in brace)]
    #[inside(brace)]
    stages: Unbracket<UnArray<Ident>>,
}

#[derive(Parse, Debug)]
struct Pipe {
    name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[prefix(kw::descriptors in brace)]
    #[inside(brace)]
    descriptors: Unbracket<UnArray<Path>>,
    #[inside(brace)]
    spec_const_tok: Option<kw::specialization_constants>,
    #[parse_if(spec_const_tok.is_some())]
    #[inside(brace)]
    specialization_constants: Option<Unbracket<UnArray<ArrowPair<LitInt, NamedField>>>>,
    #[inside(brace)]
    specific: SpecificPipe,
}
#[derive(Parse, Debug)]
struct GraphicsPipe {
    vertex_inputs_kw: Option<kw::vertex_inputs>,
    #[parse_if(vertex_inputs_kw.is_some())]
    vertex_inputs: Option<Unbracket<UnArray<NamedField>>>,
    #[prefix(kw::stages)]
    stages: Unbracket<UnArray<Ident>>,
    polygon_mode_kw: UnOption<Sequence<kw::polygon, kw::mode>>,
    #[parse_if(polygon_mode_kw.0.is_some())]
    polygon_mode: Option<StaticOrDyn<Ident>>,
    topology_kw: Option<kw::topology>,
    #[parse_if(topology_kw.is_some())]
    topology_mode: Option<StaticOrDyn<Ident>>,
    front_face_kw: UnOption<Sequence<kw::front, kw::face>>,
    #[parse_if(front_face_kw.0.is_some())]
    front_face_mode: Option<StaticOrDyn<Ident>>,
    cull_mode_kw: UnOption<Sequence<kw::cull, kw::mode>>,
    #[parse_if(cull_mode_kw.0.is_some())]
    cull_mode: Option<StaticOrDyn<Ident>>,
    depth_test_enable_kw: UnOption<Sequence<kw::depth, kw::test>>,
    #[parse_if(depth_test_enable_kw.0.is_some())]
    depth_test_enable: Option<StaticOrDyn<LitBool>>,
    depth_write_enable_kw: UnOption<Sequence<kw::depth, kw::write>>,
    #[parse_if(depth_write_enable_kw.0.is_some())]
    depth_write_enable: Option<StaticOrDyn<LitBool>>,
    depth_compare_op_kw: UnOption<Sequence<kw::depth, Sequence<kw::compare, kw::op>>>,
    #[parse_if(depth_compare_op_kw.0.is_some())]
    depth_compare_op: Option<StaticOrDyn<Ident>>,
    depth_bounds_enable_kw: UnOption<Sequence<kw::depth, kw::bounds>>,
    #[parse_if(depth_bounds_enable_kw.0.is_some())]
    depth_bounds_enable: Option<StaticOrDyn<LitBool>>,
}
#[derive(Parse, Debug)]
struct ComputePipe {}
#[derive(Debug)]
enum SpecificPipe {
    Graphics(GraphicsPipe),
    Compute(ComputePipe),
}
impl Parse for SpecificPipe {
    fn parse(input: ParseStream) -> Result<Self> {
        input
            .parse::<kw::compute>()
            .and_then(|_| input.parse().map(SpecificPipe::Compute))
            .or_else(|_| {
                input
                    .parse::<kw::graphics>()
                    .and_then(|_| input.parse().map(SpecificPipe::Graphics))
            })
    }
}
impl SpecificPipe {
    fn stages(&self) -> Vec<Ident> {
        match self {
            SpecificPipe::Compute(_) => vec![parse_quote!(COMPUTE)],
            SpecificPipe::Graphics(g) => g.stages.iter().cloned().collect(),
        }
    }
}

#[proc_macro]
pub fn define_renderer(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    #[derive(Parse)]
    struct Input {
        // #[prefix(kw::data_types)]
        // #[brace]
        // #[allow(dead_code)]
        // data_types_brace: Brace,
        // #[inside(data_types_brace)]
        // data_types: UnArray<DataType>,
        #[prefix(kw::sets)]
        #[brace]
        #[allow(dead_code)]
        sets_brace: Brace,
        #[inside(sets_brace)]
        sets: UnArray<DescriptorSet>,
        #[prefix(kw::pipelines)]
        #[brace]
        #[allow(dead_code)]
        pipelines_brace: Brace,
        #[inside(pipelines_brace)]
        pipelines: UnArray<Pipe>,
    }

    let Input { sets, pipelines, .. } = parse_macro_input!(input as Input);

    let mut output = quote!();

    let mut defined_types = HashMap::new();
    // (set_name, binding_name) => type_name
    // only available for buffers and stuff that is defined, image bindings won't have them
    let mut set_binding_type_names: HashMap<(Ident, Ident), Ident> = HashMap::new();

    for pipe in pipelines.iter() {
        let mut push_constant_type = None;

        for shader_stage in pipe.specific.stages() {
            let shader_path = std::path::Path::new(&env::var("OUT_DIR").unwrap()).join(format!(
                "{}.{}.spv",
                pipe.name.to_string(),
                shader_stage_to_file_extension(&shader_stage),
            ));
            let shader_file = File::open(&shader_path).unwrap();
            let bytes: Vec<u8> = shader_file.bytes().filter_map(std::result::Result::ok).collect();
            let spv = spirq::SpirvBinary::from(bytes);
            let entry_points = spv.reflect().expect("failed to reflect on spirv");
            let entry = entry_points
                .iter()
                .find(|entry| entry.name == "main")
                .expect("Failed to load entry point");

            for spec_const in entry.spec.spec_consts().zip_longest(
                pipe.specialization_constants
                    .as_ref()
                    .map(|x| x.iter().map(|a| &a.0).collect_vec())
                    .iter()
                    .flatten(),
            ) {
                match spec_const {
                    itertools::EitherOrBoth::Both(spv, rusty) => {
                        let rust_id = rusty.0.base10_parse().unwrap();
                        if spv.spec_id != rust_id {
                            let msg = format!(
                                "shader {} spec constant mismatch shader id = {}, rusty id = {}",
                                shader_path.to_string_lossy(),
                                spv.spec_id,
                                rust_id,
                            );
                            output.extend(quote!(compile_error!(#msg);));
                            continue;
                        }
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
                    itertools::EitherOrBoth::Left(spv) => {
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
                    itertools::EitherOrBoth::Right(rusty) => {
                        let id = &rusty.0;
                        let msg = format!(
                            "shader {} missing shader side of spec const id = {}",
                            shader_path.to_string_lossy(),
                            id
                        );
                        output.extend(quote! {
                            compile_error!(#msg);
                        });
                    }
                }
            }
            for desc in entry.descs() {
                let rusty = sets
                    .0
                    .iter()
                    .find(|set| {
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
                        if *n != rusty_binding.count.base10_parse().unwrap() {
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
            }
        }

        output.extend_one(define_pipe(pipe, push_constant_type))
    }

    output.extend(sets.iter().map(|set| define_set(set, &set_binding_type_names)));

    proc_macro::TokenStream::from(output)
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
                    const_assert_eq!(#rust_offset, #field_offset);
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
        _ => unimplemented!("{:?}", spv),
    }
}
