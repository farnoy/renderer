use std::fmt::Debug;

use derive_syn_parse::Parse;
use proc_macro2::TokenTree;
use quote::quote;
use serde::{Deserialize, Serialize};
use syn::{
    braced, bracketed,
    parse::{Parse, ParseStream, Parser, Result},
    parse_quote,
    punctuated::Punctuated,
    token::{Brace, Bracket},
    Expr, Field, Ident, LitBool, LitInt, Path, Token, Type,
};

use super::keywords as kw;

#[derive(Parse)]
#[allow(unused)]
pub struct ResourceInput {
    pub name: Ident,
    pub kind: ResourceKind,
    #[bracket]
    usages_bracket: Bracket,
    #[inside(usages_bracket)]
    pub usages: UnArray<Sequence<Ident, Sequence<Token![in], Sequence<Ident, ResourceUsage>>>>,
}

pub enum ResourceKind {
    StaticBuffer(StaticBufferResource),
    Image,
    AccelerationStructure,
}

impl Parse for ResourceKind {
    fn parse(input: ParseStream) -> Result<Self> {
        input
            .parse::<kw::Image>()
            .and(Ok(Self::Image))
            .or_else(|_| {
                input
                    .parse::<kw::AccelerationStructure>()
                    .and(Ok(Self::AccelerationStructure))
            })
            .or_else(|_| input.parse::<StaticBufferResource>().map(Self::StaticBuffer))
    }
}

#[derive(Parse)]
pub struct StaticBufferResource {
    _static_buffer_kw: kw::StaticBuffer,
    _br_start: Token![<],
    pub type_name: Type,
    _br_end: Token![>],
}

#[derive(Clone, Debug)]
pub enum ResourceUsage {
    Attachment,
    VertexBuffer,
    IndexBuffer,
    IndirectBuffer,
    TransferCopy,
    TransferClear,
    /// (set, binding, pipeline_name)
    Descriptor(Ident, Ident, Ident),
}

impl Parse for ResourceUsage {
    fn parse(input: ParseStream) -> Result<Self> {
        input
            .parse::<Sequence<kw::indirect, kw::buffer>>()
            .and(Ok(ResourceUsage::IndirectBuffer))
            .or_else(|_x| input.parse::<kw::attachment>().and(Ok(ResourceUsage::Attachment)))
            .or_else(|_| {
                input
                    .parse::<Sequence<kw::vertex, kw::buffer>>()
                    .and(Ok(ResourceUsage::VertexBuffer))
            })
            .or_else(|_| {
                input
                    .parse::<Sequence<kw::index, kw::buffer>>()
                    .and(Ok(ResourceUsage::IndexBuffer))
            })
            .or_else(|_x| {
                input
                    .parse::<Sequence<kw::transfer, kw::copy>>()
                    .and(Ok(ResourceUsage::TransferCopy))
            })
            .or_else(|_x| {
                input
                    .parse::<Sequence<kw::transfer, kw::clear>>()
                    .and(Ok(ResourceUsage::TransferClear))
            })
            .or_else(|_x| {
                input
                    .parse::<kw::descriptor>()
                    .and(input.parse::<Ident>().and_then(|pipe_name| {
                        input.parse::<Token![.]>().and(input.parse::<Ident>().and_then(|set| {
                            input.parse::<Token![.]>().and(
                                input
                                    .parse::<Ident>()
                                    .and_then(|binding| Ok(ResourceUsage::Descriptor(set, binding, pipe_name))),
                            )
                        }))
                    }))
            })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DependencyType {
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

#[derive(Debug, Clone, Copy)]
pub enum QueueFamily {
    Graphics,
    Compute,
    Transfer,
}

impl Parse for QueueFamily {
    fn parse(input: ParseStream) -> Result<Self> {
        input
            .parse::<kw::graphics>()
            .and(Ok(QueueFamily::Graphics))
            .or(input.parse::<kw::compute>().and(Ok(QueueFamily::Compute)))
            .or(input.parse::<kw::transfer>().and(Ok(QueueFamily::Transfer)))
    }
}

#[derive(Clone)]
pub struct Sequence<A, B>(pub (A, B));

impl<A: Parse, B: Parse> Parse for Sequence<A, B> {
    fn parse(input: ParseStream) -> Result<Self> {
        // Forking here so that Sequence only consumes anything if it can parse everything
        let peeking = input.fork();
        let a = peeking.parse::<A>();
        let b = peeking.parse::<B>();

        if a.is_ok() && b.is_ok() {
            input
                .parse()
                .and_then(|a| input.parse().map(|b| (a, b)))
                .map(|tup| Sequence(tup))
        } else {
            a.and_then(|a| b.map(|b| (a, b))).map(|tup| Sequence(tup))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Unbrace<T>(pub T);

impl<T: Parse> Parse for Unbrace<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        let inner;
        braced!(inner in input);
        let t = inner.parse()?;
        Ok(Unbrace(t))
    }
}

#[derive(Clone, Debug)]
pub struct Unbracket<T>(pub T);

impl<T: Parse> Parse for Unbracket<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        let inner;
        bracketed!(inner in input);
        let t = inner.parse()?;
        Ok(Unbracket(t))
    }
}

#[derive(Debug, Clone)]
pub struct UnArray<T>(pub Vec<T>);

impl<T: Parse> Parse for UnArray<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(UnArray(
            Punctuated::<T, Token![,]>::parse_terminated(input)?
                .into_iter()
                .collect(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct UnOption<T>(pub Option<T>);

impl<T: Parse> Parse for UnOption<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<T>().map(|t| UnOption(Some(t))).or(Ok(UnOption(None)))
    }
}

#[derive(Clone, Debug)]
pub enum StaticOrDyn<T> {
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
    pub fn is_dyn(&self) -> bool {
        match self {
            StaticOrDyn::Static(_) => false,
            StaticOrDyn::Dyn => true,
        }
    }

    #[allow(dead_code)]
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            StaticOrDyn::Static(s) => s,
            StaticOrDyn::Dyn => default,
        }
    }

    pub fn as_ref(&self) -> StaticOrDyn<&T> {
        match self {
            StaticOrDyn::Static(s) => StaticOrDyn::Static(&s),
            StaticOrDyn::Dyn => StaticOrDyn::Dyn,
        }
    }
}

impl<T: PartialEq + Debug> super::StaticOrDyn<T> {
    pub fn unwrap_or_default(self, default: T) -> T {
        match self {
            super::StaticOrDyn::Static(s) => s,
            super::StaticOrDyn::Dyn => default,
        }
    }
}

pub struct Pair(pub (Ident, TokenTree));

impl Parse for Pair {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        Ok(Pair((name, input.parse()?)))
    }
}

#[derive(Clone)]
pub struct ArrowPair<A, B>(pub (A, B));

impl<A: Parse, B: Parse> Parse for ArrowPair<A, B> {
    fn parse(input: ParseStream) -> Result<Self> {
        let a = input.parse()?;
        input.parse::<Token![=>]>()?;
        let b = input.parse()?;
        Ok(ArrowPair((a, b)))
    }
}

pub fn extract_optional_dyn<T, R>(a: &super::StaticOrDyn<T>, when_dyn: R) -> Option<R> {
    match a {
        super::StaticOrDyn::Dyn => Some(when_dyn),
        _ => None,
    }
}

#[derive(Parse)]
pub struct Pass {
    pub name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[prefix(kw::attachments in brace)]
    #[inside(brace)]
    pub attachments: Unbracket<UnArray<Ident>>,
    #[prefix(kw::layouts in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    layouts_brace: Brace,
    #[inside(layouts_brace)]
    pub layouts: UnArray<Sequence<Ident, ArrowPair<Sequence<LoadOp, Expr>, Sequence<StoreOp, Expr>>>>,
    #[inside(brace)]
    #[allow(dead_code)]
    subpasses_kw: kw::subpasses,
    #[inside(brace)]
    pub subpasses: Unbrace<UnArray<Subpass>>,
    #[inside(brace)]
    #[allow(dead_code)]
    dependencies_kw: Option<kw::dependencies>,
    #[parse_if(dependencies_kw.is_some())]
    #[inside(brace)]
    pub dependencies: Option<Unbrace<UnArray<SubpassDependency>>>,
}

#[derive(Parse)]
pub struct Subpass {
    pub name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[inside(brace)]
    #[allow(dead_code)]
    color_kw: Option<kw::color>,
    #[inside(brace)]
    #[parse_if(color_kw.is_some())]
    pub color: Option<Unbracket<UnArray<ArrowPair<Ident, Ident>>>>,
    #[inside(brace)]
    #[allow(dead_code)]
    depth_stencil_kw: Option<kw::depth_stencil>,
    #[inside(brace)]
    #[parse_if(depth_stencil_kw.is_some())]
    pub depth_stencil: Option<Unbrace<ArrowPair<Ident, Ident>>>,
    #[inside(brace)]
    #[allow(dead_code)]
    resolve_kw: Option<kw::resolve>,
    #[inside(brace)]
    #[parse_if(resolve_kw.is_some())]
    pub resolve: Option<Unbracket<UnArray<ArrowPair<Ident, Ident>>>>,
}

#[derive(Parse, Clone)]
pub struct SubpassDependency {
    pub from: Ident,
    #[allow(dead_code)]
    arrow_1: Token![=>],
    pub to: Ident,
    pub src_stage: Expr,
    #[allow(dead_code)]
    arrow_2: Token![=>],
    pub dst_stage: Expr,
    pub src_access: Expr,
    #[allow(dead_code)]
    arrow_3: Token![=>],
    pub dst_access: Expr,
}

#[derive(Debug, PartialEq, Clone)]
pub enum LoadOp {
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

#[derive(Debug, Clone)]
pub enum StoreOp {
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

#[derive(Parse)]
pub struct FrameInput {
    #[allow(dead_code)]
    pub(crate) name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[prefix(kw::attachments in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    attachments_brace: Brace,
    #[inside(attachments_brace)]
    pub(crate) attachments: UnArray<Ident>,
    #[prefix(kw::formats in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    formats_brace: Brace,
    #[inside(formats_brace)]
    pub(crate) formats: UnArray<StaticOrDyn<Expr>>,
    #[prefix(kw::samples in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    samples_brace: Brace,
    #[inside(samples_brace)]
    pub(crate) samples: UnArray<LitInt>,
    #[prefix(kw::passes in brace)]
    #[brace]
    #[inside(brace)]
    #[allow(dead_code)]
    passes_brace: Brace,
    #[inside(passes_brace)]
    pub(crate) passes: UnArray<Pass>,
}

#[derive(Parse)]
// TODO: rename to just pass, rename passes to renderpasses
pub struct AsyncPass {
    pub name: Ident,
    #[allow(dead_code)]
    on: kw::on,
    pub family: QueueFamily,
    #[allow(dead_code)]
    if_cond: Option<Token![if]>,
    #[parse_if(if_cond.is_some())]
    pub conditional: Option<Conditional>,
}

#[derive(Parse)]
pub struct Conditional {
    #[bracket]
    #[allow(dead_code)]
    bracket: Bracket,
    #[inside(bracket)]
    pub conditions: UnArray<Sequence<UnOption<Token![!]>, Ident>>,
}

#[derive(Parse)]
pub struct DescriptorSet {
    pub name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[inside(brace)]
    pub bindings: UnArray<Binding>,
}
#[derive(Parse)]
#[allow(dead_code)]
pub struct Binding {
    pub name: Ident,
    pub count: UnOption<Sequence<LitInt, kw::of>>,
    pub descriptor_type: Ident,
    pub partially_bound: UnOption<Sequence<kw::partially, kw::bound>>,
    pub update_after_bind: UnOption<Sequence<kw::update, Sequence<kw::after, kw::bind>>>,
    #[prefix(kw::from)]
    pub stages: Unbracket<UnArray<Ident>>,
}

#[derive(Parse)]
pub struct Pipe {
    pub name: Ident,
    #[brace]
    #[allow(dead_code)]
    brace: Brace,
    #[prefix(kw::descriptors in brace)]
    #[inside(brace)]
    pub descriptors: Unbracket<UnArray<Path>>,
    #[inside(brace)]
    pub spec_const_tok: Option<kw::specialization_constants>,
    #[parse_if(spec_const_tok.is_some())]
    #[inside(brace)]
    pub specialization_constants: Option<Unbracket<UnArray<ArrowPair<LitInt, NamedField>>>>,
    #[inside(brace)]
    pub subgroup_sizes_kw: UnOption<Sequence<Sequence<kw::varying, kw::subgroup>, kw::size>>,
    #[parse_if(subgroup_sizes_kw.0.is_some())]
    #[inside(brace)]
    pub varying_subgroup_stages: Option<UnOption<Unbracket<UnArray<Ident>>>>,
    #[inside(brace)]
    pub specific: SpecificPipe,
}
#[derive(Parse)]
#[allow(unused)]
pub struct GraphicsPipe {
    samples_kw: Option<kw::samples>,
    #[parse_if(samples_kw.is_some())]
    pub samples: Option<StaticOrDyn<LitInt>>,
    vertex_inputs_kw: Option<kw::vertex_inputs>,
    #[parse_if(vertex_inputs_kw.is_some())]
    pub vertex_inputs: Option<Unbracket<UnArray<NamedField>>>,
    #[prefix(kw::stages)]
    pub stages: Unbracket<UnArray<Ident>>,
    polygon_mode_kw: UnOption<Sequence<kw::polygon, kw::mode>>,
    #[parse_if(polygon_mode_kw.0.is_some())]
    pub polygon_mode: Option<StaticOrDyn<Ident>>,
    topology_kw: Option<kw::topology>,
    #[parse_if(topology_kw.is_some())]
    pub topology_mode: Option<StaticOrDyn<Ident>>,
    front_face_kw: UnOption<Sequence<kw::front, kw::face>>,
    #[parse_if(front_face_kw.0.is_some())]
    pub front_face_mode: Option<StaticOrDyn<Ident>>,
    cull_mode_kw: UnOption<Sequence<kw::cull, kw::mode>>,
    #[parse_if(cull_mode_kw.0.is_some())]
    pub cull_mode: Option<StaticOrDyn<Ident>>,
    depth_test_enable_kw: UnOption<Sequence<kw::depth, kw::test>>,
    #[parse_if(depth_test_enable_kw.0.is_some())]
    pub depth_test_enable: Option<StaticOrDyn<LitBool>>,
    depth_write_enable_kw: UnOption<Sequence<kw::depth, kw::write>>,
    #[parse_if(depth_write_enable_kw.0.is_some())]
    pub depth_write_enable: Option<StaticOrDyn<LitBool>>,
    depth_compare_op_kw: UnOption<Sequence<kw::depth, Sequence<kw::compare, kw::op>>>,
    #[parse_if(depth_compare_op_kw.0.is_some())]
    pub depth_compare_op: Option<StaticOrDyn<Ident>>,
    depth_bounds_enable_kw: UnOption<Sequence<kw::depth, kw::bounds>>,
    #[parse_if(depth_bounds_enable_kw.0.is_some())]
    pub depth_bounds_enable: Option<StaticOrDyn<LitBool>>,
}
#[derive(Parse)]
pub struct ComputePipe {}
pub enum SpecificPipe {
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
    pub fn stages(&self) -> Vec<Ident> {
        match self {
            SpecificPipe::Compute(_) => vec![parse_quote!(COMPUTE)],
            SpecificPipe::Graphics(g) => g.stages.0 .0.iter().cloned().collect(),
        }
    }
}

pub struct NamedField(pub Field);
impl Parse for NamedField {
    fn parse(input: ParseStream) -> Result<Self> {
        Field::parse_named(input).map(NamedField)
    }
}

pub fn to_vk_format(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let parser = |input: ParseStream| {
        input
            .parse::<kw::vec2>()
            .and(Ok(quote!(vk::Format::R32G32_SFLOAT)))
            .or_else(|_| input.parse::<kw::vec3>().and(Ok(quote!(vk::Format::R32G32B32_SFLOAT))))
            .or_else(|_| {
                input
                    .parse::<kw::vec4>()
                    .and(Ok(quote!(vk::Format::R32G32B32A32_SFLOAT)))
            })
    };

    parser.parse2(input).unwrap()
}

pub fn to_rust_type(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let parser = |input: ParseStream| {
        input
            .parse::<kw::vec2>()
            .and(Ok(quote!(glm::Vec2)))
            .or_else(|_| input.parse::<kw::vec3>().and(Ok(quote!(glm::Vec3))))
            .or_else(|_| input.parse::<kw::vec4>().and(Ok(quote!(glm::Vec4))))
            .or_else(|_| input.parse::<TokenTree>().map(|t| quote!(#t)))
    };

    parser.parse2(input).unwrap()
}
