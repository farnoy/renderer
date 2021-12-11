use anyhow::ensure;
use derive_syn_parse::Parse;
use hashbrown::HashMap;
use petgraph::{
    stable_graph::{NodeIndex, StableDiGraph},
    visit::IntoNodeReferences,
};
use quote::ToTokens;
use serde::{Deserialize, Serialize};
use syn::{parse::Parse, Expr, Ident, Token};

use super::keywords as kw;
use crate::{
    inputs::{self, ArrowPair, Sequence, UnArray, UnOption, Unbracket},
    Conditional,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceClaims {
    pub map: HashMap<String, NodeIndex>,
    pub graph: StableDiGraph<ResourceClaim, ()>,
    pub ty: ResourceDefinitionType,
}

/// This is the struct used during static analysis
#[derive(Default, Debug)]
pub(crate) struct ResourceClaimsBuilder {
    pub map: HashMap<String, NodeIndex>,
    pub graph: StableDiGraph<Option<ResourceClaim>, ()>,
    pub ty: Option<ResourceDefinitionType>,
}

impl ResourceClaimsBuilder {
    pub fn record(&mut self, claim: ResourceClaimInput) {
        let node = *self
            .map
            .entry(claim.step_name.clone())
            .and_modify(|ix| {
                *self.graph.node_weight_mut(*ix).unwrap() = Some(claim.clone().carve());
            })
            .or_insert_with(|| self.graph.add_node(Some(claim.clone().carve())));
        for dep in claim.after {
            let from = self.map.entry(dep).or_insert_with(|| self.graph.add_node(None));
            self.graph.update_edge(*from, node, ());
        }
    }

    pub fn convert(claims: HashMap<String, ResourceClaimsBuilder>) -> anyhow::Result<HashMap<String, ResourceClaims>> {
        for (resource_name, claims) in &claims {
            for (claim_ix, claim) in claims.graph.node_references() {
                let (step_name, _) = claims
                    .map
                    .iter()
                    .find(|(_, candidate_ix)| claim_ix == **candidate_ix)
                    .expect("Could not find the claim with the hashmap");
                ensure!(
                    claim.is_some(),
                    "resource hasn't been claimed, {}.{}",
                    resource_name,
                    step_name
                );
            }
            ensure!(
                claims.ty.is_some(),
                "missing define_resource! macro call for {}",
                &resource_name
            );
        }

        let res = claims
            .into_iter()
            .map(|(name, builder)| {
                (name, ResourceClaims {
                    map: builder.map,
                    graph: builder.graph.map(|_, node| node.clone().unwrap(), |_, _| ()),
                    ty: builder.ty.unwrap(),
                })
            })
            .collect::<HashMap<_, _>>();

        Ok(res)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceClaim {
    pub resource_name: String,
    pub step_name: String,
    pub pass_name: String,
    pub usage: super::ResourceUsageKind,
    pub reads: bool,
    pub writes: bool,
    pub conditional: Conditional,
}

#[derive(Debug, Clone)]
pub struct ResourceBarrierInput {
    pub command_buffer_ident: Expr,
    pub claims: Vec<ResourceClaimInput>,
    pub conditional_context: HashMap<String, Expr>,
}

#[derive(Debug, Clone)]
pub struct ResourceClaimInput {
    pub resource_name: String,
    pub step_name: String,
    pub pass_name: String,
    pub usage: super::ResourceUsageKind,
    pub reads: bool,
    pub writes: bool,
    pub after: Vec<String>,
    pub conditional: Conditional,
    pub resource_ident: Option<Expr>,
}

#[derive(Clone)]
pub struct ResourceDefinitionInput {
    pub resource_name: String,
    pub ty: ResourceDefinitionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceDefinitionType {
    StaticBuffer { type_name: String },
    Image,
    AccelerationStructure,
}

impl ResourceClaimInput {
    pub fn carve(self) -> ResourceClaim {
        ResourceClaim {
            resource_name: self.resource_name,
            step_name: self.step_name,
            pass_name: self.pass_name,
            usage: self.usage,
            reads: self.reads,
            writes: self.writes,
            conditional: self.conditional,
        }
    }
}

impl Parse for ResourceBarrierInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        struct Inner {
            command_buffer: Expr,
            #[allow(unused)]
            conditionals: UnOption<Sequence<Token![,], Unbracket<UnArray<ArrowPair<Ident, Expr>>>>>,
            _sep: Token![,],
            claims: UnArray<ResourceClaimInput>,
        }

        let s = Inner::parse(input)?;
        Ok(ResourceBarrierInput {
            command_buffer_ident: s.command_buffer,
            claims: s.claims.0,
            conditional_context: s
                .conditionals
                .0
                .map(|Sequence((_, Unbracket(UnArray(c))))| {
                    c.into_iter()
                        .map(|ArrowPair((name, value))| (name.to_string(), value))
                        .collect()
                })
                .unwrap_or_default(),
        })
    }
}

impl Parse for ResourceClaimInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        #[allow(unused)]
        struct Inner {
            resource_name: Ident,
            _dot: Token![.],
            step_name: Ident,
            r: Option<kw::r>,
            w: Option<kw::w>,
            rw: Option<kw::rw>,
            _in: Token![in],
            pass_name: Ident,
            usage: inputs::ResourceUsage,
            _after: Option<kw::after>,
            #[parse_if(_after.is_some())]
            after: Option<Unbracket<UnArray<Ident>>>,
            conditional: UnOption<inputs::Conditional>,
            sep: Option<Token![;]>,
            #[parse_if(sep.is_some())]
            resource_expr: Option<Expr>,
        }

        let s = Inner::parse(input)?;
        let reads = s.r.is_some() || s.rw.is_some();
        let writes = s.w.is_some() || s.rw.is_some();
        Ok(ResourceClaimInput {
            resource_name: s.resource_name.to_string(),
            step_name: s.step_name.to_string(),
            pass_name: s.pass_name.to_string(),
            usage: super::ResourceUsageKind::from(s.usage),
            after: s
                .after
                .map(|Unbracket(UnArray(idents))| idents.into_iter().map(|i| i.to_string()).collect())
                .unwrap_or(vec![]),
            resource_ident: s.resource_expr,
            reads,
            writes,
            conditional: s.conditional.0.as_ref().map(Conditional::from).unwrap_or_default(),
        })
    }
}

impl Parse for ResourceDefinitionInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        #[derive(Parse)]
        struct Inner {
            resource_name: Ident,
            _dot: Token![=],
            kind: inputs::ResourceKind,
        }

        let s = Inner::parse(input)?;
        match s.kind {
            inputs::ResourceKind::StaticBuffer(inputs::StaticBufferResource { type_name, .. }) => {
                Ok(ResourceDefinitionInput {
                    resource_name: s.resource_name.to_string(),
                    ty: ResourceDefinitionType::StaticBuffer {
                        type_name: type_name.to_token_stream().to_string(),
                    },
                })
            }
            inputs::ResourceKind::Image => Ok(ResourceDefinitionInput {
                resource_name: s.resource_name.to_string(),
                ty: ResourceDefinitionType::Image,
            }),
            inputs::ResourceKind::AccelerationStructure => Ok(ResourceDefinitionInput {
                resource_name: s.resource_name.to_string(),
                ty: ResourceDefinitionType::AccelerationStructure,
            }),
        }
    }
}
