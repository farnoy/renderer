use std::iter::FromIterator;

use hashbrown::HashMap;
use syn::{Expr, Ident, Token};

use super::{inputs, keywords};

#[derive(Debug)]
pub(crate) struct RendererInput {
    pub(crate) name: Ident,
    pub(crate) passes: HashMap<Ident, Pass>,
    pub(crate) async_passes: HashMap<Ident, AsyncPass>,
    pub(crate) resources: HashMap<Ident, ResourceInput>,
}

#[derive(Debug)]
pub(crate) struct ResourceInput {
    pub(crate) name: Ident,
    pub(crate) kind: ResourceKind,
    pub(crate) usages: Vec<ResourceUsage>,
}

#[derive(Debug)]
pub(crate) enum ResourceKind {
    StaticBuffer { type_name: Ident },
    Image,
}

#[derive(Debug)]
pub(crate) struct ResourceUsage {
    pub(crate) name: Ident,
    pub(crate) pass: Ident,
    pub(crate) usage: ResourceUsageKind,
}

#[derive(Debug, PartialEq)]
pub(crate) enum ResourceUsageKind {
    Attachment,
    IndirectBuffer,
    TransferCopy,
    TransferClear,
    /// (set, binding)
    Descriptor(String, String),
}

#[derive(Debug)]
pub(crate) struct Pass {
    pub(crate) name: Ident,
    pub(crate) attachments: Vec<Ident>,
    pub(crate) layouts: HashMap<Ident, PassLayout>,
    pub(crate) subpasses: Vec<Subpass>,
    pub(crate) dependencies: Vec<SubpassDependency>,
}

#[derive(Debug)]
pub(crate) struct PassLayout {
    pub(crate) load_op: inputs::LoadOp,
    pub(crate) initial_layout: Expr,
    pub(crate) store_op: inputs::StoreOp,
    pub(crate) final_layout: Expr,
}

#[derive(Debug)]
pub(crate) struct Subpass {
    pub(crate) name: Ident,
    pub(crate) color_attachments: Vec<(Ident, Ident)>,
    pub(crate) depth_stencil_attachments: Option<(Ident, Ident)>,
    pub(crate) resolve_attachments: Vec<(Ident, Ident)>,
}

#[derive(Debug)]
pub(crate) struct SubpassDependency {
    pub(crate) src: Ident,
    pub(crate) dst: Ident,
    pub(crate) src_stage: Expr,
    pub(crate) dst_stage: Expr,
    pub(crate) src_access: Expr,
    pub(crate) dst_access: Expr,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum QueueFamily {
    Graphics,
    Compute,
    Transfer,
}

#[derive(Debug)]
pub(crate) struct AsyncPass {
    pub(crate) name: Ident,
    pub(crate) queue: QueueFamily,
}

impl From<&inputs::FrameInput> for RendererInput {
    fn from(input: &inputs::FrameInput) -> Self {
        let resources = input.resources.0.iter().map(ResourceInput::from);
        let passes = input.passes.0.iter().map(Pass::from);
        let async_passes = input.async_passes.0.iter().map(AsyncPass::from);
        RendererInput {
            name: input.name.clone(),
            resources: HashMap::from_iter(resources.map(|res| (res.name.clone(), res))),
            passes: HashMap::from_iter(passes.map(|pass| (pass.name.clone(), pass))),
            async_passes: HashMap::from_iter(async_passes.map(|async_pass| (async_pass.name.clone(), async_pass))),
        }
    }
}

impl From<&inputs::ResourceInput> for ResourceInput {
    fn from(r: &inputs::ResourceInput) -> Self {
        ResourceInput {
            name: r.name.clone(),
            kind: match r.kind {
                inputs::ResourceKind::StaticBuffer(inputs::StaticBufferResource { ref type_name, .. }) => {
                    ResourceKind::StaticBuffer {
                        type_name: type_name.clone(),
                    }
                }
                inputs::ResourceKind::Image => ResourceKind::Image,
            },
            usages: r.usages.0.iter().cloned().map(ResourceUsage::from).collect(),
        }
    }
}

impl From<inputs::Sequence<syn::Ident, inputs::Sequence<Token![in], inputs::Sequence<Ident, inputs::ResourceUsage>>>>
    for ResourceUsage
{
    fn from(
        inputs::Sequence((usage_name, inputs::Sequence((_in, inputs::Sequence((pass_name, resource_usage)))))): inputs::Sequence<syn::Ident, inputs::Sequence<Token![in], inputs::Sequence<Ident, inputs::ResourceUsage>>>,
    ) -> Self {
        ResourceUsage {
            name: usage_name,
            pass: pass_name,
            usage: resource_usage.into(),
        }
    }
}

impl From<inputs::ResourceUsage> for ResourceUsageKind {
    fn from(u: inputs::ResourceUsage) -> Self {
        match u {
            inputs::ResourceUsage::Attachment => Self::Attachment,
            inputs::ResourceUsage::IndirectBuffer => Self::IndirectBuffer,
            inputs::ResourceUsage::TransferCopy => Self::TransferCopy,
            inputs::ResourceUsage::TransferClear => Self::TransferClear,
            inputs::ResourceUsage::Descriptor(set, binding) => Self::Descriptor(set.to_string(), binding.to_string()),
        }
    }
}

impl From<&inputs::UnOption<inputs::Sequence<keywords::on, inputs::QueueFamily>>> for QueueFamily {
    fn from(i: &inputs::UnOption<inputs::Sequence<keywords::on, inputs::QueueFamily>>) -> Self {
        i.0.as_ref()
            .map(|inputs::Sequence((_kw, queue))| match queue {
                inputs::QueueFamily::Graphics => Self::Graphics,
                inputs::QueueFamily::Compute => Self::Compute,
                inputs::QueueFamily::Transfer => Self::Transfer,
            })
            .unwrap_or(Self::Graphics)
    }
}

impl From<&inputs::Sequence<Ident, inputs::UnOption<inputs::Sequence<keywords::on, inputs::QueueFamily>>>>
    for AsyncPass
{
    fn from(
        i: &inputs::Sequence<Ident, inputs::UnOption<inputs::Sequence<keywords::on, inputs::QueueFamily>>>,
    ) -> Self {
        let inputs::Sequence((name, queue)) = i;
        AsyncPass {
            name: name.clone(),
            queue: QueueFamily::from(queue),
        }
    }
}

impl From<&inputs::ArrowPair<inputs::Sequence<inputs::LoadOp, Expr>, inputs::Sequence<inputs::StoreOp, Expr>>>
    for PassLayout
{
    fn from(
        p: &inputs::ArrowPair<inputs::Sequence<inputs::LoadOp, Expr>, inputs::Sequence<inputs::StoreOp, Expr>>,
    ) -> Self {
        let inputs::ArrowPair((
            inputs::Sequence((load_op, initial_layout)),
            inputs::Sequence((store_op, final_layout)),
        )) = p.clone();

        PassLayout {
            load_op,
            initial_layout,
            store_op,
            final_layout,
        }
    }
}

impl From<&inputs::Subpass> for Subpass {
    fn from(s: &inputs::Subpass) -> Self {
        let inputs::Subpass {
            name,
            color,
            depth_stencil,
            resolve,
            ..
        } = s;

        Subpass {
            name: name.clone(),
            color_attachments: color
                .as_ref()
                .cloned()
                .map(|b| b.0 .0.into_iter().map(|x| (x.0 .0, x.0 .1)).collect())
                .unwrap_or(vec![]),
            depth_stencil_attachments: depth_stencil.as_ref().cloned().map(|x| (x.0 .0 .0, x.0 .0 .1)),
            resolve_attachments: resolve
                .as_ref()
                .cloned()
                .map(|b| b.0 .0.into_iter().map(|x| (x.0 .0, x.0 .1)).collect())
                .unwrap_or(vec![]),
        }
    }
}

impl From<&inputs::SubpassDependency> for SubpassDependency {
    fn from(d: &inputs::SubpassDependency) -> Self {
        let d = d.clone();
        SubpassDependency {
            src: d.from,
            dst: d.to,
            src_stage: d.src_stage,
            dst_stage: d.dst_stage,
            src_access: d.src_access,
            dst_access: d.dst_access,
        }
    }
}

impl Pass {
    pub(crate) fn queue(&self) -> QueueFamily {
        QueueFamily::Graphics
    }
}

impl From<&inputs::Pass> for Pass {
    fn from(p: &inputs::Pass) -> Self {
        Pass {
            name: p.name.clone(),
            attachments: p.attachments.0 .0.clone(),
            layouts: HashMap::from_iter(
                p.layouts
                    .iter()
                    .map(|inputs::Sequence((name, layout))| (name.clone(), PassLayout::from(layout))),
            ),
            subpasses: p.subpasses.iter().map(Subpass::from).collect(),
            dependencies: p
                .dependencies
                .as_ref()
                .map(|inputs::Unbrace(inputs::UnArray(x))| x.iter().map(SubpassDependency::from).collect())
                .unwrap_or(vec![]),
        }
    }
}
