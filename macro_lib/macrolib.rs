#![feature(extend_one)]
use std::{
    env,
    fmt::Debug,
    fs::File,
    io::{Read, Write},
    path::Path,
};

use anyhow::ensure;
use convert_case::{Case, Casing};
use derive_syn_parse::Parse;
use hashbrown::HashMap;
use indexmap::IndexSet;
use itertools::Itertools;
use keywords as kw;
use petgraph::{
    algo::has_path_connecting,
    graph::DiGraph,
    stable_graph::{NodeIndex, StableDiGraph},
    visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences},
    Direction,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use resource_claims::{ResourceClaims, ResourceClaimsBuilder};
use serde::{Deserialize, Serialize};
use syn::{
    parse::Parse,
    parse2, parse_quote,
    visit::{self, Visit},
    Expr, Ident, ItemMacro,
};

use crate::resource_claims::{ResourceBarrierInput, ResourceDefinitionInput};

pub mod inputs;
pub mod keywords;
pub mod resource_claims;
mod rga;

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct RendererInput {
    pub name: String,
    pub passes: HashMap<String, Pass>,
    pub async_passes: HashMap<String, AsyncPass>,
    pub resources: HashMap<String, ResourceClaims>,
    pub descriptor_sets: HashMap<String, DescriptorSet>,
    pub attachments: HashMap<String, Attachment>,
    pub pipelines: HashMap<String, Pipeline>,
    pub shader_information: ShaderInformation,
    /// DAG of Passes
    pub dependency_graph: StableDiGraph<String, inputs::DependencyType>,
    /// Pass -> (SemaphoreIx, StageIx)
    pub timeline_semaphore_mapping: HashMap<String, (usize, u64)>,
    /// SemaphoreIx -> CycleValue
    pub timeline_semaphore_cycles: HashMap<usize, u64>,
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct ShaderInformation {
    pub shader_type_definitions: String,
    pub set_binding_type_names: HashMap<(String, String), String>,
    /// Pipeline name -> PushConstant type
    pub push_constant_type_definitions: HashMap<String, String>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum LoadOp {
    Load,
    Clear,
    DontCare,
}

impl Parse for LoadOp {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        input
            .parse::<kw::load>()
            .and(Ok(Self::Load))
            .or_else(|_| input.parse::<kw::clear>().and(Ok(Self::Clear)))
            .or_else(|_| input.parse::<kw::dont_care>().and(Ok(Self::DontCare)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoreOp {
    Store,
    Discard,
}

impl Parse for StoreOp {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        input
            .parse::<kw::store>()
            .and(Ok(Self::Store))
            .or_else(|_| input.parse::<kw::discard>().and(Ok(Self::Discard)))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Pass {
    pub name: String,
    pub attachments: Vec<String>,
    pub layouts: HashMap<String, PassLayout>,
    pub subpasses: Vec<Subpass>,
    pub dependencies: Vec<SubpassDependency>,
}

impl Parse for Pass {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        pub struct Inner {
            pub name: Ident,
            #[brace]
            #[allow(dead_code)]
            brace: syn::token::Brace,
            #[prefix(kw::attachments in brace)]
            #[inside(brace)]
            pub attachments: inputs::Unbracket<inputs::UnArray<Ident>>,
            #[prefix(kw::layouts in brace)]
            #[brace]
            #[inside(brace)]
            #[allow(dead_code)]
            layouts_brace: syn::token::Brace,
            #[inside(layouts_brace)]
            pub layouts: inputs::UnArray<inputs::Sequence<Ident, PassLayout>>,
            #[inside(brace)]
            #[allow(dead_code)]
            subpasses_kw: kw::subpasses,
            #[inside(brace)]
            pub subpasses: inputs::Unbrace<inputs::UnArray<Subpass>>,
            #[inside(brace)]
            #[allow(dead_code)]
            dependencies_kw: Option<kw::dependencies>,
            #[parse_if(dependencies_kw.is_some())]
            #[inside(brace)]
            pub dependencies: Option<inputs::Unbrace<inputs::UnArray<SubpassDependency>>>,
        }

        let p = Inner::parse(input)?;

        Ok(Pass {
            name: p.name.to_string(),
            attachments: p.attachments.0 .0.iter().map(|x| x.to_string()).collect(),
            layouts: HashMap::from_iter(
                p.layouts
                    .0
                    .iter()
                    .map(|inputs::Sequence((name, layout))| (name.to_string(), layout.clone())),
            ),
            subpasses: p.subpasses.0 .0.clone(),
            dependencies: p
                .dependencies
                .as_ref()
                .cloned()
                .map(|inputs::Unbrace(inputs::UnArray(x))| x)
                .unwrap_or(vec![]),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassLayout {
    pub load_op: LoadOp,
    pub initial_layout: String,
    pub store_op: StoreOp,
    pub final_layout: String,
}

impl Parse for PassLayout {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let inputs::ArrowPair((
            inputs::Sequence((load_op, initial_layout)),
            inputs::Sequence((store_op, final_layout)),
        )) = inputs::ArrowPair::<inputs::Sequence<LoadOp, Expr>, inputs::Sequence<StoreOp, Expr>>::parse(input)?;

        Ok(PassLayout {
            load_op,
            initial_layout: initial_layout.to_token_stream().to_string(),
            store_op,
            final_layout: final_layout.to_token_stream().to_string(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subpass {
    pub name: String,
    pub color_attachments: Vec<(String, String)>,
    pub depth_stencil_attachments: Option<(String, String)>,
    pub resolve_attachments: Vec<(String, String)>,
}

impl Parse for Subpass {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        pub struct Inner {
            pub name: Ident,
            #[brace]
            #[allow(dead_code)]
            brace: syn::token::Brace,
            #[inside(brace)]
            #[allow(dead_code)]
            color_kw: Option<kw::color>,
            #[inside(brace)]
            #[parse_if(color_kw.is_some())]
            pub color: Option<inputs::Unbracket<inputs::UnArray<inputs::ArrowPair<Ident, Ident>>>>,
            #[inside(brace)]
            #[allow(dead_code)]
            depth_stencil_kw: Option<kw::depth_stencil>,
            #[inside(brace)]
            #[parse_if(depth_stencil_kw.is_some())]
            pub depth_stencil: Option<inputs::Unbrace<inputs::ArrowPair<Ident, Ident>>>,
            #[inside(brace)]
            #[allow(dead_code)]
            resolve_kw: Option<kw::resolve>,
            #[inside(brace)]
            #[parse_if(resolve_kw.is_some())]
            pub resolve: Option<inputs::Unbracket<inputs::UnArray<inputs::ArrowPair<Ident, Ident>>>>,
        }
        let Inner {
            name,
            color,
            depth_stencil,
            resolve,
            ..
        } = Inner::parse(input)?;

        Ok(Subpass {
            name: name.to_string(),
            color_attachments: color
                .as_ref()
                .cloned()
                .map(|b| {
                    b.0 .0
                        .into_iter()
                        .map(|x| (x.0 .0.to_string(), x.0 .1.to_string()))
                        .collect()
                })
                .unwrap_or(vec![]),
            depth_stencil_attachments: depth_stencil
                .as_ref()
                .cloned()
                .map(|x| (x.0 .0 .0.to_string(), x.0 .0 .1.to_string())),
            resolve_attachments: resolve
                .as_ref()
                .cloned()
                .map(|b| {
                    b.0 .0
                        .into_iter()
                        .map(|x| (x.0 .0.to_string(), x.0 .1.to_string()))
                        .collect()
                })
                .unwrap_or(vec![]),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubpassDependency {
    pub src: String,
    pub dst: String,
    pub src_stage: String,
    pub dst_stage: String,
    pub src_access: String,
    pub dst_access: String,
}

impl Parse for SubpassDependency {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        pub struct Inner {
            pub from: Ident,
            #[allow(dead_code)]
            arrow_1: syn::Token![=>],
            pub to: Ident,
            pub src_stage: Expr,
            #[allow(dead_code)]
            arrow_2: syn::Token![=>],
            pub dst_stage: Expr,
            pub src_access: Expr,
            #[allow(dead_code)]
            arrow_3: syn::Token![=>],
            pub dst_access: Expr,
        }
        let d = Inner::parse(input)?;
        Ok(SubpassDependency {
            src: d.from.to_string(),
            dst: d.to.to_string(),
            src_stage: d.src_stage.to_token_stream().to_string(),
            dst_stage: d.dst_stage.to_token_stream().to_string(),
            src_access: d.src_access.to_token_stream().to_string(),
            dst_access: d.dst_access.to_token_stream().to_string(),
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum QueueFamily {
    Graphics,
    Compute,
    Transfer,
}

impl Parse for QueueFamily {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        input
            .parse::<kw::graphics>()
            .and(Ok(QueueFamily::Graphics))
            .or(input.parse::<kw::compute>().and(Ok(QueueFamily::Compute)))
            .or(input.parse::<kw::transfer>().and(Ok(QueueFamily::Transfer)))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AsyncPass {
    pub name: String,
    pub queue: QueueFamily,
    pub conditional: Conditional,
}

impl Parse for AsyncPass {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        pub struct Inner {
            pub name: Ident,
            #[allow(dead_code)]
            on: kw::on,
            pub family: QueueFamily,
            pub conditional: Conditional,
        }

        let i = Inner::parse(input)?;

        Ok(AsyncPass {
            name: i.name.to_token_stream().to_string(),
            queue: i.family,
            conditional: i.conditional,
        })
    }
}

#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Conditional {
    pub conditions: Vec<Condition>,
}

impl Parse for Conditional {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        pub struct Inner {
            _if_cond: syn::Token![if],
            #[bracket]
            #[allow(dead_code)]
            bracket: syn::token::Bracket,
            #[inside(bracket)]
            pub conditions: inputs::UnArray<Condition>,
        }
        let i = inputs::UnOption::<Inner>::parse(input)?;
        let c =
            i.0.map(|c| Conditional {
                conditions: c.conditions.0,
            })
            .unwrap_or_default();
        Ok(c)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Condition {
    pub neg: bool,
    pub switch_name: String,
}

impl Parse for Condition {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let inputs::Sequence((neg, switch_name)) =
            inputs::Sequence::<inputs::UnOption<syn::Token![!]>, Ident>::parse(input)?;

        Ok(Condition {
            neg: neg.0.is_some(),
            switch_name: switch_name.to_string(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DescriptorSet {
    pub name: String,
    pub bindings: Vec<Binding>,
}

impl Parse for DescriptorSet {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        pub struct Inner {
            pub name: Ident,
            #[brace]
            #[allow(dead_code)]
            brace: syn::token::Brace,
            #[inside(brace)]
            pub bindings: inputs::UnArray<Binding>,
        }
        let p = Inner::parse(input)?;

        Ok(DescriptorSet {
            name: p.name.to_string(),
            bindings: p.bindings.0,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Binding {
    pub name: String,
    pub count: u32,
    pub descriptor_type: String,
    pub partially_bound: bool,
    pub update_after_bind: bool,
    pub shader_stages: Vec<String>,
}

impl Parse for Binding {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        pub struct Inner {
            pub name: Ident,
            pub count: inputs::UnOption<inputs::Sequence<syn::LitInt, kw::of>>,
            pub descriptor_type: Ident,
            pub partially_bound: inputs::UnOption<inputs::Sequence<kw::partially, kw::bound>>,
            pub update_after_bind:
                inputs::UnOption<inputs::Sequence<kw::update, inputs::Sequence<kw::after, kw::bind>>>,
            #[prefix(kw::from)]
            pub stages: inputs::Unbracket<inputs::UnArray<Ident>>,
        }
        let b = Inner::parse(input)?;
        Ok(Binding {
            name: b.name.to_string(),
            count: b
                .count
                .0
                .as_ref()
                .map(|inputs::Sequence((a, _))| a.clone())
                .unwrap_or(parse_quote!(1))
                .base10_parse::<u32>()
                .unwrap(),
            descriptor_type: b.descriptor_type.to_string(),
            partially_bound: b.partially_bound.0.is_some(),
            update_after_bind: b.update_after_bind.0.is_some(),
            shader_stages: b.stages.0 .0.iter().map(|x| x.to_string()).collect(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Attachment {
    pub name: String,
    pub format: StaticOrDyn<String>,
    pub samples: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedField {
    pub name: String,
    pub ty: String,
}

impl Parse for NamedField {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let f = syn::Field::parse_named(input)?;
        Ok(NamedField {
            name: f.ident.as_ref().unwrap().to_string(),
            ty: f.ty.to_token_stream().to_string(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pipeline {
    pub name: String,
    pub descriptor_sets: Vec<(String, Conditional)>,
    pub specialization_constants: Vec<(u32, NamedField)>,
    pub varying_subgroup_stages: Vec<String>,
    pub specific: SpecificPipe,
}

impl Pipeline {
    pub fn unique_conditionals(&self) -> IndexSet<&str> {
        self.descriptor_sets
            .iter()
            .flat_map(|(_, c)| c.conditions.iter().map(|c| c.switch_name.as_str()))
            .collect()
    }

    pub fn permutations(&self) -> impl Iterator<Item = (&Pipeline, String, IndexSet<&str>)> {
        let mut combinations: Vec<IndexSet<&str>> = vec![IndexSet::new()];
        for conditional in self.unique_conditionals() {
            combinations = combinations
                .into_iter()
                .cartesian_product(std::iter::once(conditional))
                .flat_map(|(d, b)| {
                    let mut clone = d.clone();
                    clone.insert(b);
                    [clone, d]
                })
                .collect();
        }
        combinations
            .into_iter()
            .cartesian_product(self.specific.stages())
            .map(move |(conditionals, stage)| (self, stage, conditionals))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StaticOrDyn<T> {
    Static(T),
    Dyn,
}

impl<T: Parse> Parse for StaticOrDyn<T> {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        input
            .parse::<syn::Token![dyn]>()
            .map(|_| StaticOrDyn::Dyn)
            .or_else(|_| input.parse::<T>().map(StaticOrDyn::Static))
    }
}

impl<T> StaticOrDyn<T> {
    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> StaticOrDyn<U> {
        match self {
            StaticOrDyn::Static(a) => StaticOrDyn::Static(f(a)),
            StaticOrDyn::Dyn => StaticOrDyn::Dyn,
        }
    }

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

impl<T: PartialEq + Debug> StaticOrDyn<T> {
    pub fn unwrap_or_default(self, default: T) -> T {
        match self {
            StaticOrDyn::Static(s) => s,
            StaticOrDyn::Dyn => default,
        }
    }
}

pub fn extract_optional_dyn<T, R>(a: &StaticOrDyn<T>, when_dyn: R) -> Option<R> {
    match a {
        StaticOrDyn::Dyn => Some(when_dyn),
        _ => None,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphicsPipe {
    pub samples: StaticOrDyn<u8>,
    pub vertex_inputs: Vec<NamedField>,
    pub stages: Vec<String>,
    pub polygon_mode: StaticOrDyn<String>,
    pub topology_mode: StaticOrDyn<String>,
    pub front_face_mode: StaticOrDyn<String>,
    pub cull_mode: StaticOrDyn<String>,
    pub depth_test_enable: StaticOrDyn<bool>,
    pub depth_write_enable: StaticOrDyn<bool>,
    pub depth_compare_op: StaticOrDyn<String>,
    pub depth_bounds_enable: StaticOrDyn<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SpecificPipe {
    Graphics(GraphicsPipe),
    Compute,
}

impl SpecificPipe {
    pub fn stages(&self) -> Vec<String> {
        match self {
            SpecificPipe::Graphics(x) => x.stages.clone(),
            SpecificPipe::Compute => vec!["COMPUTE".to_string()],
        }
    }
}

pub fn persist(data: &RendererInput) -> anyhow::Result<()> {
    let src = env::var("OUT_DIR")?;
    let src = Path::new(&src).join("crossbar.bincode");
    let crossbar = File::create(src)?;
    bincode::serialize_into(crossbar, data)?;
    Ok(())
}

pub fn fetch() -> anyhow::Result<RendererInput> {
    let src = env::var("OUT_DIR")?;
    let src = Path::new(&src).join("crossbar.bincode");
    let crossbar = File::open(src)?;
    Ok(bincode::deserialize_from(crossbar)?)
}

pub fn analyze() -> anyhow::Result<RendererInput> {
    let src = env::var("CARGO_MANIFEST_DIR")?;
    let src = Path::new(&src);

    let mut visitor = AnalysisVisitor::default();

    for file_path in glob::glob(&format!("{}/src/**/*.rs", src.display()))? {
        let src = file_path?;
        let mut file_src = File::open(src)?;
        let mut content = String::new();
        file_src.read_to_string(&mut content)?;
        let frame_input = syn::parse_file(&content)?;
        visitor.visit_file(&frame_input);
    }

    let mut data = visitor.data;

    if visitor.had_errors {
        println!("cargo:warning=Build had errors, analysis data is stale!");
        fetch()
    } else {
        data.resources = ResourceClaimsBuilder::convert(visitor.resource_claims)?;

        let dependency_graph = calculate_depencency_graph(&data)?;
        let semaphore_mapping = assign_semaphores_to_stages(&dependency_graph, &data.passes, &data.async_passes);
        data.timeline_semaphore_mapping = semaphore_mapping
            .into_iter()
            .map(|(k, v)| (dependency_graph[k].clone(), v))
            .collect();

        data.timeline_semaphore_cycles =
            data.timeline_semaphore_mapping
                .iter()
                .fold(HashMap::new(), |mut map, (_stage, (sem_ix, value))| {
                    map.entry(*sem_ix)
                        .and_modify(|x| {
                            *x = std::cmp::max(*x, *value);
                        })
                        .or_insert(*value);
                    map
                });
        data.timeline_semaphore_cycles.values_mut().for_each(|x| {
            *x = x.next_power_of_two();
        });

        data.dependency_graph = dependency_graph;

        dump_dependency_graph(&data)?;

        dump_resource_graphs(&data)?;

        Ok(data)
    }
}

fn dump_dependency_graph(data: &RendererInput) -> Result<(), anyhow::Error> {
    let root_dir = env::var("CARGO_MANIFEST_DIR")?;
    let src = Path::new(&root_dir).join("diagnostics").join("dependency_graph.dot");
    let mut file = File::create(src)?;

    let graph = &data.dependency_graph;

    writeln!(file, "digraph {{")?;
    for (ix, pass) in graph.node_references() {
        let (sem_ix, step_ix) = data.timeline_semaphore_mapping.get(pass).unwrap();
        let conditional = data
            .async_passes
            .get(pass)
            .map(|p| &p.conditional)
            .map(|c| {
                c.conditions
                    .iter()
                    .map(|c| format!("{}{}", if c.neg { "!" } else { "" }, &c.switch_name))
                    .collect::<Vec<_>>()
            })
            .filter(|strings| !strings.is_empty())
            .map(|strings| format!("\n[{}] ", strings.join(", ")))
            .unwrap_or("".to_string());
        writeln!(
            file,
            "{} [ label = \"{}\n({}, {}){}\", shape = \"rectangle\" ]",
            ix.index(),
            pass,
            sem_ix,
            step_ix,
            conditional,
        )?;
    }
    for edge in graph.edge_references() {
        let source = graph[edge.source()].as_str();
        let target = graph[edge.target()].as_str();
        let source_queue = data
            .passes
            .get(source)
            .map(|_| QueueFamily::Graphics)
            .or_else(|| data.async_passes.get(source).map(|p| p.queue))
            .unwrap();
        let target_queue = data
            .passes
            .get(target)
            .map(|_| QueueFamily::Graphics)
            .or_else(|| data.async_passes.get(target).map(|p| p.queue))
            .unwrap();
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

fn dump_resource_graphs(data: &RendererInput) -> Result<(), anyhow::Error> {
    let root_dir = env::var("CARGO_MANIFEST_DIR")?;
    for (res_name, claim) in data.resources.iter() {
        let src = Path::new(&root_dir)
            .join("diagnostics/resources")
            .join(&format!("{}.dot", res_name));
        let mut file = File::create(src)?;

        writeln!(file, "digraph {{")?;
        for (ix, claim) in claim.graph.node_references() {
            let conditional = claim
                .conditional
                .conditions
                .iter()
                .map(|c| format!("{}{}", if c.neg { "!" } else { "" }, &c.switch_name))
                .collect_vec();
            let conditional = format!("[{}] ", conditional.join(", "));
            writeln!(
                file,
                "{} [ label = \"{} {}\" ]",
                ix.index(),
                &claim.step_name,
                conditional
            )?;
        }
        for edge in claim.graph.edge_references() {
            let source = &claim.graph[edge.source()];
            let target = &claim.graph[edge.target()];
            let source_queue = data
                .passes
                .get(&source.pass_name)
                .map(|_| QueueFamily::Graphics)
                .or_else(|| data.async_passes.get(&source.pass_name).map(|p| p.queue))
                .unwrap();
            let target_queue = data
                .passes
                .get(&target.pass_name)
                .map(|_| QueueFamily::Graphics)
                .or_else(|| data.async_passes.get(&target.pass_name).map(|p| p.queue))
                .unwrap();
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

    Ok(())
}

/// Returns Rust type definitions corresponding shader types and a map of `(set_name, binding_name)
/// -> type_name`
pub fn analyze_shader_types(
    sets: &HashMap<String, DescriptorSet>,
    pipelines: &HashMap<String, Pipeline>,
) -> Result<ShaderInformation, Vec<String>> {
    let mut output = quote!();
    let mut set_binding_type_names = HashMap::new();
    let mut defined_types = HashMap::new();
    let mut push_constant_type_definitions = HashMap::new();
    let mut errors = vec![];

    for pipe in pipelines.values() {
        let mut push_constant_type: Option<String> = None;
        let mut push_constant_spir_type = None;

        for (pipe, shader_stage, conditionals) in pipe.permutations() {
            let conditionals = conditionals.into_iter().collect_vec();
            let shader_path = std::path::Path::new(&env::var("OUT_DIR").unwrap()).join(format!(
                "{}.{}.[{}].spv",
                pipe.name.to_string(),
                shader_stage_to_file_extension(&shader_stage),
                conditionals.join(",")
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
                let rusty = pipe.specialization_constants.iter().find(|p| spv.spec_id == p.0);

                match rusty {
                    Some((_, rusty)) => {
                        if !compare_types(spv.ty, &rusty.ty) {
                            let msg = format!(
                                "shader {} spec constant mismatch for id = {} shader type = {:?}, rusty type = {:?}",
                                shader_path.to_string_lossy(),
                                spv.spec_id,
                                spv.ty,
                                rusty.ty,
                            );
                            errors.extend_one(msg);
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
                        errors.extend_one(msg);
                    }
                }
            }
            for (rusty_id, _field) in pipe.specialization_constants.iter() {
                if entry.spec.spec_consts().any(|c| c.spec_id == *rusty_id) {
                    continue;
                }
                let msg = format!(
                    "shader {} missing shader side of spec const id = {}",
                    shader_path.to_string_lossy(),
                    rusty_id
                );
                errors.extend_one(msg);
            }

            for desc in entry.descs() {
                let set_name =
                    syn::parse_str::<syn::Path>(&pipe.descriptor_sets[desc.desc_bind.set() as usize].0).unwrap();
                let set_name = set_name.segments.last().unwrap();
                let rusty = sets
                    .get(&set_name.ident.to_string())
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
                                    "Incorrect shader binding at set {} binding {}\
                                    , shader declares {:?}, rusty binding is {}",
                                    rusty.name.to_string(),
                                    rusty_binding.name.to_string(),
                                    spir_ty,
                                    rusty_ty,
                                );
                                errors.extend_one(msg)
                            }
                        }
                        if *n != rusty_binding.count {
                            let msg = format!(
                                "Wrong descriptor count for set {} binding {}, shader needs {}",
                                rusty.name.to_string(),
                                rusty_binding.name.to_string(),
                                n
                            );
                            errors.extend_one(msg)
                        }
                        let name = s.name().unwrap();
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
                                        errors.extend_one(msg);
                                    }
                                })
                                .or_insert_with(|| {
                                    output.extend_one(definition);
                                    spirq_ty.clone()
                                });
                        }

                        set_binding_type_names
                            .entry((rusty.name.clone(), rusty_binding.name.clone()))
                            .and_modify(|placed_ty: &mut String| {
                                if placed_ty != &name {
                                    let msg = format!(
                                        "set binding type name mismatch:\n\
                                            previous: {:?}\n\
                                            incoming: {:?}",
                                        placed_ty.to_string(),
                                        &name
                                    );
                                    errors.extend_one(msg);
                                }
                            })
                            .or_insert(name.to_string());
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
                push_constant_type = Some(
                    prereqs
                        .into_iter()
                        .map(|(_, _, tokens)| tokens)
                        .collect::<TokenStream>()
                        .to_string(),
                );
                push_constant_spir_type = Some(push_const.clone());
            }
        }

        rga::dump_rga(sets, pipe, push_constant_spir_type.as_ref());
        if let Some(definition) = push_constant_type {
            push_constant_type_definitions.insert(pipe.name.clone(), definition);
        }
    }

    if errors.is_empty() {
        Ok(ShaderInformation {
            shader_type_definitions: output.to_string(),
            set_binding_type_names,
            push_constant_type_definitions,
        })
    } else {
        Err(errors)
    }
}

fn shader_stage_to_file_extension(id: &str) -> &'static str {
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

fn compare_types(spv: &spirq::Type, rust: &str) -> bool {
    match (spv, rust) {
        (spirq::Type::Scalar(spirq::ty::ScalarType::Signed(4)), "i32") => true,
        (spirq::Type::Scalar(spirq::ty::ScalarType::Unsigned(4)), "u32") => true,
        (spirq::Type::Scalar(spirq::ty::ScalarType::Unsigned(2)), "u16") => true,
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
            let name = format_ident!("{}", s.name().unwrap());
            let (field_name, field_offset, field_ty) = split3((0..s.nmember()).map(|ix| {
                let field = s.get_member(ix).unwrap();
                let field_name = field.name.as_ref().unwrap().to_case(Case::Snake);
                (
                    format_ident!("{}", field_name),
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

#[derive(Default)]
struct AnalysisVisitor {
    data: RendererInput,
    resource_claims: HashMap<String, ResourceClaimsBuilder>,
    had_errors: bool,
}

impl<'ast> Visit<'ast> for AnalysisVisitor {
    fn visit_item_macro(&mut self, node: &'ast ItemMacro) {
        if node.mac.path == parse_quote!(renderer_macros::define_frame) {
            if let Ok(parsed) = parse2::<RendererInput>(node.mac.tokens.clone()) {
                self.data.name = parsed.name;
                self.data.passes.extend(parsed.passes);
                self.data.async_passes.extend(parsed.async_passes);
                self.data.descriptor_sets.extend(parsed.descriptor_sets);
                self.data.attachments.extend(parsed.attachments);
                self.data.pipelines.extend(parsed.pipelines);
            } else {
                self.had_errors = true;
            }
        } else if node.mac.path == parse_quote!(renderer_macros::define_resource) {
            if let Ok(parsed) = parse2::<ResourceDefinitionInput>(node.mac.tokens.clone()) {
                let claim = self.resource_claims.entry(parsed.resource_name).or_default();
                claim.ty = Some(parsed.ty);
            } else {
                self.had_errors = true;
            }
        } else if node.mac.path == parse_quote!(renderer_macros::define_set) {
            if let Ok(set) = parse2::<DescriptorSet>(node.mac.tokens.clone()) {
                self.data
                    .descriptor_sets
                    .entry(set.name.clone())
                    .and_modify(|_| {
                        println!("cargo:warning=Duplicate descriptor set definition: {}", &set.name);
                        self.had_errors = true;
                    })
                    .or_insert(set);
            } else {
                self.had_errors = true;
            }
        } else if node.mac.path == parse_quote!(renderer_macros::define_pipe) {
            if let Ok(pipe) = parse2::<Pipeline>(node.mac.tokens.clone()) {
                self.data
                    .pipelines
                    .entry(pipe.name.clone())
                    .and_modify(|_| {
                        println!("cargo:warning=Duplicate pipeline definition: {}", &pipe.name);
                        self.had_errors = true;
                    })
                    .or_insert(pipe);
            } else {
                self.had_errors = true;
            }
        } else if node.mac.path == parse_quote!(renderer_macros::define_pass) {
            if let Ok(pass) = parse2::<AsyncPass>(node.mac.tokens.clone()) {
                self.data
                    .async_passes
                    .entry(pass.name.clone())
                    .and_modify(|_| {
                        println!("cargo:warning=Duplicate async pass definition: {}", &pass.name);
                        self.had_errors = true;
                    })
                    .or_insert(pass);
            } else {
                self.had_errors = true;
            }
        }

        visit::visit_item_macro(self, node);
    }

    fn visit_expr_macro(&mut self, node: &'ast syn::ExprMacro) {
        if node.mac.path == parse_quote!(renderer_macros::barrier) {
            if let Ok(input) = parse2::<ResourceBarrierInput>(node.mac.tokens.clone()) {
                for claim in input.claims {
                    self.resource_claims
                        .entry(claim.resource_name.clone())
                        .or_default()
                        .record(claim);
                }
            } else {
                self.had_errors = true;
            }
        }

        visit::visit_expr_macro(self, node);
    }
}

/// https://cs.stackexchange.com/a/29133
pub fn transitive_reduction<A, B>(g: &mut DiGraph<A, B>) {
    for u in g.node_indices() {
        let descendants = g.neighbors_directed(u, Direction::Outgoing).collect_vec();
        for v in descendants {
            let mut dfs = petgraph::visit::Dfs::new(&*g, v);
            dfs.next(&*g); // skip self
            while let Some(v_prime) = dfs.next(&*g) {
                g.find_edge(u, v_prime).map(|edge_ix| g.remove_edge(edge_ix).unwrap());
            }
        }
    }
}

pub fn transitive_reduction_stable<A, B, Ix: petgraph::adj::IndexType>(g: &mut StableDiGraph<A, B, Ix>) {
    for u in g.node_indices().collect::<Vec<_>>() {
        let descendants = g.neighbors_directed(u, Direction::Outgoing).collect_vec();
        for v in descendants {
            let mut dfs = petgraph::visit::Dfs::new(&*g, v);
            dfs.next(&*g); // skip self
            while let Some(v_prime) = dfs.next(&*g) {
                g.find_edge(u, v_prime).map(|edge_ix| g.remove_edge(edge_ix).unwrap());
            }
        }
    }
}

fn calculate_depencency_graph(
    new_data: &RendererInput,
) -> anyhow::Result<StableDiGraph<String, inputs::DependencyType, u32>> {
    let mut dependency_graph = StableDiGraph::<String, inputs::DependencyType, u32>::new();

    for claims in new_data.resources.values() {
        for edge in claims.graph.edge_references() {
            let from_pass = &claims.graph.node_weight(edge.source()).unwrap().pass_name;
            let to_pass = &claims.graph.node_weight(edge.target()).unwrap().pass_name;

            if from_pass == to_pass {
                continue;
            }

            let from_ix = dependency_graph
                .node_indices()
                .find(|&c| dependency_graph[c] == *from_pass)
                .unwrap_or_else(|| dependency_graph.add_node(from_pass.clone()));
            let to_ix = dependency_graph
                .node_indices()
                .find(|&c| dependency_graph[c] == *to_pass)
                .unwrap_or_else(|| dependency_graph.add_node(to_pass.clone()));

            dependency_graph.update_edge(from_ix, to_ix, inputs::DependencyType::SameFrame);
        }
    }

    // FIXME: PresentationAcquire is a virtual stage, should ideally be replaced by LastFrame/LastAccess
    // dependencies across graphs
    let presentation_acquire = dependency_graph
        .node_indices()
        .find(|&c| dependency_graph[c] == "PresentationAcquire")
        .unwrap_or_else(|| dependency_graph.add_node("PresentationAcquire".to_string()));
    for ix in dependency_graph.node_indices().collect_vec() {
        if presentation_acquire != ix {
            dependency_graph.update_edge(presentation_acquire, ix, inputs::DependencyType::SameFrame);
        }
    }

    // dbg!(petgraph::dot::Dot::with_config(
    //     &dependency_graph.map(|_, node_ident| node_ident.to_string(), |_, _| ""),
    //     &[petgraph::dot::Config::EdgeNoLabel]
    // ));
    transitive_reduction_stable(&mut dependency_graph);
    // dbg!(petgraph::dot::Dot::with_config(
    //     &dependency_graph.map(|_, node_ident| node_ident.to_string(), |_, _| ""),
    //     &[petgraph::dot::Config::EdgeNoLabel]
    // ));

    // let connected_components = petgraph::algo::connected_components(&dependency_graph);
    // ensure!(
    //     connected_components == 1,
    //     "sync graph must have one connected component"
    // );
    ensure!(
        !petgraph::algo::is_cyclic_directed(&dependency_graph),
        "sync graph is cyclic"
    );

    Ok(dependency_graph)
}

/// Takes a validated dependency graph and comes up with an assignment of semaphores, returing a
/// mapping from each node to (semaphore_index, stage_within_semaphore_index)
pub fn assign_semaphores_to_stages(
    graph: &StableDiGraph<String, inputs::DependencyType>,
    passes: &HashMap<String, Pass>,
    async_passes: &HashMap<String, AsyncPass>,
) -> HashMap<NodeIndex, (usize, u64)> {
    let mut mapping = HashMap::new();

    // Start from PresentationAcquire
    let presentation_acquire = graph
        .node_indices()
        .find(|&c| graph[c] == "PresentationAcquire")
        .unwrap();
    let mut dfs = petgraph::visit::Dfs::new(graph, presentation_acquire);

    dfs.next(graph); // Skip PresentationAcquire which we set up before the loop
    mapping.insert(presentation_acquire, (0, 1));

    let mut last_semaphore_ix = 0;
    let mut last_stage_ix = 1;
    let mut last_node = presentation_acquire;
    let mut last_queue_family = QueueFamily::Graphics;

    while let Some(this_node) = dfs.next(graph) {
        // TODO: preallocate DfsSpace?
        // TODO: queue_family matching is a workaround for https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/3590
        let queue_family = passes
            .get(&graph[this_node])
            .map(|_| QueueFamily::Graphics)
            .or_else(|| async_passes.get(&graph[this_node]).map(|p| p.queue))
            .unwrap();
        if last_queue_family == queue_family && has_path_connecting(graph, last_node, this_node, None) {
            last_stage_ix += 1;
            assert!(mapping.insert(this_node, (last_semaphore_ix, last_stage_ix)).is_none());
        } else {
            last_semaphore_ix += 1;
            last_stage_ix = 1;
            last_queue_family = queue_family;
            assert!(mapping.insert(this_node, (last_semaphore_ix, last_stage_ix)).is_none());
        }
        last_node = this_node;
    }
    mapping
}

impl Parse for Pipeline {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        struct InnerPipe {
            pub name: Ident,
            #[brace]
            #[allow(dead_code)]
            brace: syn::token::Brace,
            #[prefix(kw::descriptors in brace)]
            #[inside(brace)]
            pub descriptors: inputs::Unbracket<inputs::UnArray<inputs::Sequence<syn::Path, Conditional>>>,
            #[inside(brace)]
            #[allow(unused)]
            pub spec_const_tok: Option<kw::specialization_constants>,
            #[parse_if(spec_const_tok.is_some())]
            #[inside(brace)]
            pub specialization_constants:
                Option<inputs::Unbracket<inputs::UnArray<inputs::ArrowPair<syn::LitInt, NamedField>>>>,
            #[inside(brace)]
            #[allow(unused)]
            pub subgroup_sizes_kw:
                inputs::UnOption<inputs::Sequence<inputs::Sequence<kw::varying, kw::subgroup>, kw::size>>,
            #[parse_if(subgroup_sizes_kw.0.is_some())]
            #[inside(brace)]
            pub varying_subgroup_stages: Option<inputs::UnOption<inputs::Unbracket<inputs::UnArray<Ident>>>>,
            #[inside(brace)]
            pub specific: InnerSpecificPipe,
        }
        #[derive(Parse)]
        #[allow(unused)]
        struct InnerGraphicsPipe {
            samples_kw: Option<kw::samples>,
            #[parse_if(samples_kw.is_some())]
            pub samples: Option<StaticOrDyn<syn::LitInt>>,
            vertex_inputs_kw: Option<kw::vertex_inputs>,
            #[parse_if(vertex_inputs_kw.is_some())]
            pub vertex_inputs: Option<inputs::Unbracket<inputs::UnArray<NamedField>>>,
            #[prefix(kw::stages)]
            pub stages: inputs::Unbracket<inputs::UnArray<Ident>>,
            polygon_mode_kw: inputs::UnOption<inputs::Sequence<kw::polygon, kw::mode>>,
            #[parse_if(polygon_mode_kw.0.is_some())]
            pub polygon_mode: Option<StaticOrDyn<Ident>>,
            topology_kw: Option<kw::topology>,
            #[parse_if(topology_kw.is_some())]
            pub topology_mode: Option<StaticOrDyn<Ident>>,
            front_face_kw: inputs::UnOption<inputs::Sequence<kw::front, kw::face>>,
            #[parse_if(front_face_kw.0.is_some())]
            pub front_face_mode: Option<StaticOrDyn<Ident>>,
            cull_mode_kw: inputs::UnOption<inputs::Sequence<kw::cull, kw::mode>>,
            #[parse_if(cull_mode_kw.0.is_some())]
            pub cull_mode: Option<StaticOrDyn<Ident>>,
            depth_test_enable_kw: inputs::UnOption<inputs::Sequence<kw::depth, kw::test>>,
            #[parse_if(depth_test_enable_kw.0.is_some())]
            pub depth_test_enable: Option<StaticOrDyn<syn::LitBool>>,
            depth_write_enable_kw: inputs::UnOption<inputs::Sequence<kw::depth, kw::write>>,
            #[parse_if(depth_write_enable_kw.0.is_some())]
            pub depth_write_enable: Option<StaticOrDyn<syn::LitBool>>,
            depth_compare_op_kw: inputs::UnOption<inputs::Sequence<kw::depth, inputs::Sequence<kw::compare, kw::op>>>,
            #[parse_if(depth_compare_op_kw.0.is_some())]
            pub depth_compare_op: Option<StaticOrDyn<Ident>>,
            depth_bounds_enable_kw: inputs::UnOption<inputs::Sequence<kw::depth, kw::bounds>>,
            #[parse_if(depth_bounds_enable_kw.0.is_some())]
            pub depth_bounds_enable: Option<StaticOrDyn<syn::LitBool>>,
        }
        #[derive(Parse)]
        struct InnerComputePipe {}
        enum InnerSpecificPipe {
            Graphics(InnerGraphicsPipe),
            Compute(InnerComputePipe),
        }
        impl Parse for InnerSpecificPipe {
            fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
                input
                    .parse::<kw::compute>()
                    .and_then(|_| input.parse().map(InnerSpecificPipe::Compute))
                    .or_else(|_| {
                        input
                            .parse::<kw::graphics>()
                            .and_then(|_| input.parse().map(InnerSpecificPipe::Graphics))
                    })
            }
        }
        let input = InnerPipe::parse(input)?;
        let specific = match input.specific {
            InnerSpecificPipe::Graphics(g) => SpecificPipe::Graphics(GraphicsPipe {
                samples: g
                    .samples
                    .map(|x| x.map(|d| d.base10_parse().unwrap()))
                    .unwrap_or(StaticOrDyn::Static(1)),
                vertex_inputs: g.vertex_inputs.map(|x| x.0 .0).unwrap_or(vec![]),
                stages: g.stages.0 .0.into_iter().map(|x| x.to_string()).collect(),
                polygon_mode: g
                    .polygon_mode
                    .map(|x| x.map(|x| x.to_string()))
                    .unwrap_or(StaticOrDyn::Static("FILL".to_string())),
                topology_mode: g
                    .topology_mode
                    .map(|x| x.map(|x| x.to_string()))
                    .unwrap_or(StaticOrDyn::Static("TRIANGLE_LIST".to_string())),
                front_face_mode: g
                    .front_face_mode
                    .map(|x| x.map(|x| x.to_string()))
                    .unwrap_or(StaticOrDyn::Static("CLOCKWISE".to_string())),
                cull_mode: g
                    .cull_mode
                    .map(|x| x.map(|x| x.to_string()))
                    .unwrap_or(StaticOrDyn::Static("NONE".to_string())),
                depth_test_enable: g
                    .depth_test_enable
                    .map(|x| x.map(|x| x.value))
                    .unwrap_or(StaticOrDyn::Static(false)),
                depth_write_enable: g
                    .depth_write_enable
                    .map(|x| x.map(|x| x.value))
                    .unwrap_or(StaticOrDyn::Static(false)),
                depth_compare_op: g
                    .depth_compare_op
                    .map(|x| x.map(|x| x.to_string()))
                    .unwrap_or(StaticOrDyn::Static("NEVER".to_string())),
                depth_bounds_enable: g
                    .depth_bounds_enable
                    .map(|x| x.map(|x| x.value))
                    .unwrap_or(StaticOrDyn::Static(false)),
            }),
            InnerSpecificPipe::Compute(_) => SpecificPipe::Compute,
        };
        Ok(Pipeline {
            name: input.name.to_string(),
            descriptor_sets: input
                .descriptors
                .0
                 .0
                .into_iter()
                .map(|inputs::Sequence((x, conditional))| (x.to_token_stream().to_string(), conditional))
                .collect(),
            specialization_constants: input
                .specialization_constants
                .into_iter()
                .flat_map(|inputs::Unbracket(inputs::UnArray(x))| {
                    x.into_iter()
                        .map(|inputs::ArrowPair((a, b))| (a.base10_parse().unwrap(), b))
                })
                .collect(),
            varying_subgroup_stages: match input.varying_subgroup_stages {
                Some(inputs::UnOption(Some(inputs::Unbracket(inputs::UnArray(v))))) => {
                    v.into_iter().map(|x| x.to_string()).collect()
                }
                Some(inputs::UnOption(None)) => specific.stages(),
                None => vec![],
            },
            specific,
        })
    }
}

impl Parse for RendererInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        pub struct Inner {
            #[allow(dead_code)]
            pub(crate) name: Ident,
            #[brace]
            #[allow(dead_code)]
            brace: syn::token::Brace,
            #[prefix(kw::attachments in brace)]
            #[brace]
            #[inside(brace)]
            #[allow(dead_code)]
            attachments_brace: syn::token::Brace,
            #[inside(attachments_brace)]
            pub(crate) attachments: inputs::UnArray<Ident>,
            #[prefix(kw::formats in brace)]
            #[brace]
            #[inside(brace)]
            #[allow(dead_code)]
            formats_brace: syn::token::Brace,
            #[inside(formats_brace)]
            pub(crate) formats: inputs::UnArray<StaticOrDyn<Expr>>,
            #[prefix(kw::samples in brace)]
            #[brace]
            #[inside(brace)]
            #[allow(dead_code)]
            samples_brace: syn::token::Brace,
            #[inside(samples_brace)]
            pub(crate) samples: inputs::UnArray<syn::LitInt>,
            #[prefix(kw::passes in brace)]
            #[brace]
            #[inside(brace)]
            #[allow(dead_code)]
            passes_brace: syn::token::Brace,
            #[inside(passes_brace)]
            pub(crate) passes: inputs::UnArray<Pass>,
        }
        let input = Inner::parse(input)?;
        let passes = input.passes.0;
        let attachments = input
            .attachments
            .0
            .into_iter()
            .zip(input.samples.0.into_iter())
            .zip(input.formats.0.into_iter())
            .map(|((name, samples), format)| {
                (name.to_string(), Attachment {
                    name: name.to_string(),
                    format: format.map(|x| x.to_token_stream().to_string()),
                    samples: samples.base10_parse().unwrap(),
                })
            })
            .collect();
        Ok(RendererInput {
            name: input.name.to_string(),
            passes: HashMap::from_iter(passes.into_iter().map(|pass| (pass.name.to_string(), pass))),
            attachments,
            ..Default::default()
        })
    }
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
