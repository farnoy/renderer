#![feature(extend_one)]

use std::env;

use hashbrown::HashMap;
use itertools::Itertools;
use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::{EdgeRef, GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers, IntoNodeReferences, Visitable},
    Direction,
};
use proc_macro2::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use quote::{format_ident, quote, ToTokens};
use renderer_macro_lib::{
    fetch,
    inputs::{self, extract_optional_dyn, to_rust_type, to_vk_format, DependencyType, FrameInput},
    resource_claims::{ResourceBarrierInput, ResourceClaim, ResourceDefinitionInput, ResourceDefinitionType},
    AsyncPass, Binding, DescriptorSet, LoadOp, NamedField, Pass, PassLayout, Pipeline, QueueFamily, RendererInput,
    ResourceUsageKind, SpecificPipe, StaticOrDyn, StoreOp, SubpassDependency,
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
                    ResourceUsageKind::Attachment => todo!(),
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
                    pub(crate) fn new(device: &Device) -> Self {
                        let b = device.new_static_buffer(
                            vk::BufferUsageFlags::empty() #(| vk::BufferUsageFlags::#usage_flags)*,
                            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                        );
                        device.set_object_name(b.buffer.handle, #resource_debug_name);
                        Self(b)
                    }

                    pub(crate) fn destroy(self, device: &Device) {
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
        ResourceDefinitionType::Image => quote! {
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
            }

            impl std::ops::Deref for #name {
                type Target = Image;

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
    let mut acquire_buffer_barriers: Vec<TokenStream> = vec![];
    let mut acquire_memory_barriers: Vec<TokenStream> = vec![];
    #[allow(unused_mut)]
    let mut acquire_buffer_barriers_with_bypass: Vec<TokenStream> = vec![];
    #[allow(unused_mut)]
    let mut release_buffer_barriers: Vec<TokenStream> = vec![];
    #[allow(unused_mut)]
    let mut release_memory_barriers: Vec<TokenStream> = vec![];
    let command_buffer_expr = input.command_buffer_ident;

    for claim in input.claims {
        let claims = data.resources.get(&claim.resource_name).unwrap();

        let this_node = *claims.map.get(&claim.step_name).unwrap();
        let this = &claims.graph[this_node];
        let get_queue_family_index_for_pass = |pass: &str| {
            data.passes
                .get(pass)
                .and(Some(QueueFamily::Graphics))
                .or_else(|| data.async_passes.get(pass).map(|async_pass| async_pass.queue))
                .expect("pass not found")
        };
        let get_runtime_queue_family = |ty: QueueFamily| match ty {
            QueueFamily::Graphics => quote!(renderer.device.graphics_queue_family),
            QueueFamily::Compute => quote!(renderer.device.compute_queue_family),
            QueueFamily::Transfer => quote!(renderer.device.transfer_queue_family),
        };

        let connected_components = petgraph::algo::connected_components(&claims.graph);
        if connected_components != 1 {
            let msg = "resource claims graph must have one connected component".to_string();
            validation_errors.extend(quote!(compile_error!(#msg);));
        }
        if petgraph::algo::is_cyclic_directed(&claims.graph) {
            let msg = "resource claims graph is cyclic".to_string();
            validation_errors.extend(quote!(compile_error!(#msg);));
            // TODO: early exit here so that toposort doesn't see a cycle
        }
        // dbg!(&input.resource_name, petgraph::dot::Dot::with_config(
        //     &claims.graph.map(
        //         |_, node_ident| node_ident.as_ref().map(|c| c.step_name.clone()),
        //         |_, _| ""
        //     ),
        //     &[petgraph::dot::Config::EdgeNoLabel]
        // ));
        let grouped_topo = grouped_toposort(&claims.graph);
        let in_first_stage = grouped_topo
            .first()
            .map(|stage| stage.iter().any(|&candidate| candidate == this_node))
            .unwrap_or(false);
        let in_last_stage = grouped_topo
            .last()
            .map(|stage| stage.iter().any(|&candidate| candidate == this_node))
            .unwrap_or(false);

        // neighbors with wraparound on both ends to make the code aware of dependencies in the prev/next
        // iterations of the graph
        let incoming = claims.graph.neighbors_directed(this_node, Direction::Incoming).chain(
            in_first_stage
                .then(|| grouped_topo.last())
                .into_iter()
                .flatten()
                .flat_map(|stage| stage.iter().cloned()),
        );
        let outgoing = claims.graph.neighbors_directed(this_node, Direction::Outgoing).chain(
            in_last_stage
                .then(|| grouped_topo.first())
                .into_iter()
                .flatten()
                .flat_map(|stage| stage.iter().cloned()),
        );

        let this_queue = get_queue_family_index_for_pass(&this.pass_name);
        let this_queue_runtime = get_runtime_queue_family(this_queue);
        let mut acquired_already = false;
        for dep in incoming {
            let prev_step = &claims.graph[dep];
            let prev_queue = get_queue_family_index_for_pass(&prev_step.pass_name);
            let prev_queue_runtime = get_runtime_queue_family(prev_queue);

            if prev_step.pass_name == this.pass_name
                && prev_step.usage != ResourceUsageKind::Attachment
                && this.usage != ResourceUsageKind::Attachment
            {
                match claims.ty {
                    ResourceDefinitionType::StaticBuffer { .. } => {
                        let stage_flags = |step: &ResourceClaim, include_reads: bool, include_writes: bool| {
                            let (access_prefixes, stages) = match &step.usage {
                                ResourceUsageKind::VertexBuffer => (vec![quote!(VERTEX_ATTRIBUTE)], vec![quote!(VERTEX_INPUT)]),
                                ResourceUsageKind::IndexBuffer => (vec![quote!(INDEX)], vec![quote!(INDEX_INPUT)]),
                                ResourceUsageKind::Attachment => (vec![quote!(MEMORY)], vec![quote!(
                                    EARLY_FRAGMENT_TESTS), quote!(
                                        LATE_FRAGMENT_TESTS), quote!(
                                        COLOR_ATTACHMENT_OUTPUT)])
                                , // TODO: imprecise
                                ResourceUsageKind::IndirectBuffer => (vec![quote!(INDIRECT_COMMAND)], vec![quote!(DRAW_INDIRECT)]),
                                ResourceUsageKind::TransferCopy => (vec![quote!(TRANSFER)], vec![quote!(COPY)]),
                                // TODO: sync validation is picky when just CLEAR from sync2 is used
                                ResourceUsageKind::TransferClear => (vec![quote!(TRANSFER)], vec![quote!(ALL_TRANSFER)]),
                                ResourceUsageKind::Descriptor(set_name, binding_name, pipeline_name) => {
                                    let pipeline = data.pipelines.get(pipeline_name).expect("pipeline not found in barrier");
                                    let set = data.descriptor_sets.get(set_name).expect("set not found in barrier");
                                    let binding = set.bindings.iter().find(|candidate| candidate.name == *binding_name).expect("binding not found in barrier");
                                    let pipeline_stages = pipeline.specific.stages();

                                    let mut access_prefixes = vec![];
                                    let mut stages = vec![];

                                    binding.shader_stages.iter()
                                    .filter(|x| pipeline_stages.contains(*x))
                                    .map(|stage| match stage.as_str() {
                                        "COMPUTE" => (quote!(SHADER), quote!(COMPUTE_SHADER)),
                                        _ => unimplemented!("barrier! descriptor stage {}", stage)
                                    })
                                    .for_each(|(x, y)| {
                                        access_prefixes.push(x);
                                        stages.push(y);
                                    });
                                    (access_prefixes, stages)
                                }
                            };

                            let access_prefixes = access_prefixes
                                .into_iter()
                                .flat_map(|prefix| {
                                    let mut output = vec![];
                                    if step.reads && include_reads {
                                        output.push(format_ident!("{}_READ", prefix.to_string()).to_token_stream());
                                    }
                                    if step.writes && include_writes {
                                        output.push(format_ident!("{}_WRITE", prefix.to_string()).to_token_stream());
                                    }
                                    output
                                })
                                .collect_vec();

                            (access_prefixes, stages)
                        };
                        let (src_access, src_stage) = stage_flags(&prev_step, false, true);
                        let (dst_access, dst_stage) = stage_flags(&this, true, true);
                        assert!(
                            claim.resource_ident.is_some(),
                            "Expected barrier call to provide resource expr {:?}",
                            &claim
                        );
                        let resource_ident = &claim.resource_ident;

                        acquire_buffer_barriers.push(quote! {
                            vk::BufferMemoryBarrier2KHR::builder()
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .src_stage_mask(#(vk::PipelineStageFlags2KHR::#src_stage)|*)
                            .dst_stage_mask(#(vk::PipelineStageFlags2KHR::#dst_stage)|*)
                            .src_access_mask(#(vk::AccessFlags2KHR::#src_access)|*)
                            .dst_access_mask(#(vk::AccessFlags2KHR::#dst_access)|*)
                            .buffer({#resource_ident}.buffer.handle)
                            .size(vk::WHOLE_SIZE)
                            .build()
                        });
                    }
                    ResourceDefinitionType::AccelerationStructure => {
                        acquire_memory_barriers.push(quote! {
                            vk::MemoryBarrier2KHR::builder()
                                .src_stage_mask(vk::PipelineStageFlags2KHR::ACCELERATION_STRUCTURE_BUILD)
                                .src_access_mask(vk::AccessFlags2KHR::ACCELERATION_STRUCTURE_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                                .dst_access_mask(vk::AccessFlags2KHR::ACCELERATION_STRUCTURE_READ)
                                .build()
                        });
                    }
                    _ => todo!("acquire unknown resource"),
                }
            }

            if prev_queue == this_queue {
                continue;
            }

            // TODO: No exclusive resources to transfer for now, bring it back in the future 
            if acquired_already && false {
                // Visualize the dependency graph
                // dbg!(petgraph::dot::Dot::with_config(
                //     &claims.graph.map(
                //         |_, node_ident| node_ident.as_ref().unwrap().step_name.clone(),
                //         |_, _| ""
                //     ),
                //     &[petgraph::dot::Config::EdgeNoLabel]
                // ));
                eprintln!("Warning: Skipping a duplicate acquire barrier from {} to {} for resource {} because we don't unify these barriers (for now)", &prev_step.step_name, &this.step_name, &claim.resource_name);
                continue;
            }
            acquired_already = true;

            match claims.ty {
                ResourceDefinitionType::StaticBuffer { .. } => {
                    // buffers are never exclusive, so we skip this
                    /*
                    let acquire_buffer_barrier = quote! {
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
                            .buffer(#resource_ident.buffer.handle)
                            .size(vk::WHOLE_SIZE)
                            .build()
                    };
                    if in_first_stage {
                        acquire_buffer_barriers_with_bypass.push(acquire_buffer_barrier);
                    } else {
                        acquire_buffer_barriers.push(acquire_buffer_barrier);
                    }
                    */
                }
                ResourceDefinitionType::AccelerationStructure => {
                    // acquire_memory_barriers.push(quote! {
                    //     vk::MemoryBarrier2KHR::builder()
                    //         .src_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    //         .src_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                    //         .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    //         .dst_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                    //         .build()
                    // });
                }
                _ => todo!("same queue buffer unknown resource"),
            }
        }
        let mut released_already = false;
        for dep in outgoing {
            let next_step = &claims.graph[dep];
            let next_queue = get_queue_family_index_for_pass(&next_step.pass_name);
            let next_queue_runtime = get_runtime_queue_family(next_queue);

            if this_queue == next_queue {
                continue;
            }

            // TODO: No exclusive resources to transfer for now, bring it back in the future 
            if released_already && false {
                eprintln!("Warning: Skipping a duplicate release barrier from {} to {} for resource {} because we don't unify these barriers (for now)", &this.step_name, &next_step.step_name, &claim.resource_name);
                continue;
            }
            released_already = true;

            match claims.ty {
                ResourceDefinitionType::StaticBuffer { .. } => {
                    // buffers are never exclusive, so we skip this
                    /*
                    release_buffer_barriers.push(quote! {
                        vk::BufferMemoryBarrier2KHR::builder()
                            // TODO: granularity
                            .src_queue_family_index(#this_queue_runtime)
                            .dst_queue_family_index(#next_queue_runtime)
                            .src_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                            .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                            .src_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                            .dst_access_mask(vk::AccessFlags2KHR::MEMORY_READ | vk::AccessFlags2KHR::MEMORY_WRITE)
                            .buffer(#resource_ident.buffer.handle)
                            .size(vk::WHOLE_SIZE)
                            .build()
                    });
                    */
                }
                ResourceDefinitionType::AccelerationStructure => {
                    // release_memory_barriers.push(quote! {
                    //     vk::MemoryBarrier2KHR::builder()
                    //         .src_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    //         .src_access_mask(vk::AccessFlags2KHR::MEMORY_READ |
                    // vk::AccessFlags2KHR::MEMORY_WRITE)
                    //         .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    //         .dst_access_mask(vk::AccessFlags2KHR::MEMORY_READ |
                    // vk::AccessFlags2KHR::MEMORY_WRITE)         .build()
                    // });
                }
                _ => todo!("release unknown resource"),
            }
        }
    }

    let acquire_buffer_barriers_len = acquire_buffer_barriers.len();
    let acquire_buffer_barriers_with_bypass_len = acquire_buffer_barriers_with_bypass.len();
    let acquire_block =
        (acquire_buffer_barriers_len + acquire_buffer_barriers_with_bypass_len + acquire_memory_barriers.len() > 0)
            .then(|| {
                quote! {
                    let acquire_buffer_barriers = [
                        #(#acquire_buffer_barriers,)*
                        #(#acquire_buffer_barriers_with_bypass,)*
                    ];
                    let acquire_memory_barriers = [
                        #(#acquire_memory_barriers,)*
                    ];
                    let slice_count = if #acquire_buffer_barriers_with_bypass_len > 0 && renderer.frame_number > 1 {
                        #acquire_buffer_barriers_len + #acquire_buffer_barriers_with_bypass_len
                    } else {
                        #acquire_buffer_barriers_len
                    };
                    renderer.device.synchronization2.cmd_pipeline_barrier2(
                        #command_buffer_expr,
                        &vk::DependencyInfoKHR::builder()
                            .buffer_memory_barriers(&acquire_buffer_barriers[0..slice_count])
                            .memory_barriers(&acquire_memory_barriers),
                    );

                }
            });
    let release_block = (release_buffer_barriers.len() + release_memory_barriers.len() > 0).then(|| {
        quote! {
            let release_buffer_barriers = [
                #(#release_buffer_barriers,)*
            ];
            let release_memory_barriers = [
                #(#release_memory_barriers,)*
            ];
            renderer.device.synchronization2.cmd_pipeline_barrier2(
                #command_buffer_expr,
                &vk::DependencyInfoKHR::builder()
                    .buffer_memory_barriers(&release_buffer_barriers)
                    .memory_barriers(&release_memory_barriers),
            );
        }
    });
    quote! {
        unsafe {
            #validation_errors
            {
                #acquire_block
            }
            scopeguard::guard((), |()| {
                #release_block
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
pub fn define_frame(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let _frame_input = parse_macro_input!(input as FrameInput);

    define_frame2().into()
}

fn define_frame2() -> TokenStream {
    let validation_errors = quote!();

    let new_data = fetch().unwrap();

    let dependency_graph = &new_data.dependency_graph;

    let wait_instance = new_data
        .passes
        .keys()
        .chain(new_data.async_passes.keys())
        .map(|pass| {
            let ix = dependency_graph
                .node_indices()
                .find(|&ix| *pass == dependency_graph[ix])
                .expect("did not find pass in dependency graph");
            let (signal_out_sem, _) = new_data
                .timeline_semaphore_mapping
                .get(pass)
                .expect("no sync for this pass");
            let signal_timeline_ident = format_ident!("AutoSemaphore{}", signal_out_sem);
            let pass_ident = format_ident!("{}", pass);
            let signal_timeline_path: Path = parse_quote!(#signal_timeline_ident::#pass_ident);

            let wait_inner = dependency_graph
                .edges_directed(ix, Direction::Incoming)
                .map(|edge| {
                    let from = &dependency_graph[edge.source()];
                    let (wait_sem, _) = new_data
                        .timeline_semaphore_mapping
                        .get(from)
                        .expect("no sync for this pass");
                    let wait_timeline_ident = format_ident!("AutoSemaphore{}", wait_sem);
                    let from_ident = format_ident!("{}", from);
                    let wait_timeline_path: Path = parse_quote!(#wait_timeline_ident::#from_ident);

                    let as_of = match edge.weight() {
                        &DependencyType::SameFrame => quote!(as_of::<super::super::#wait_timeline_path>(frame_number)),
                        &DependencyType::LastFrame => {
                            quote!(as_of_last::<super::super::#wait_timeline_path>(frame_number))
                        }
                        &DependencyType::LastAccess => {
                            // TODO: broken but no need to fix for now
                            quote!(as_of_previous(&image_index, &render_frame))
                        }
                    };

                    quote! {
                        {
                            fn accessor(frame_number: u64, image_index: &crate::renderer::ImageIndex) -> u64 {
                                #as_of
                            }
                            (accessor as fn(u64, &crate::renderer::ImageIndex) -> u64, #wait_sem)
                        }
                    }
                })
                .collect::<Vec<_>>();

            (pass, quote! {

                impl RenderStage for Stage {
                    type SignalTimelineStage = super::super::#signal_timeline_path;
                    const SIGNAL_AUTO_SEMAPHORE_IX: usize = #signal_out_sem;

                    fn wait_semaphore_timeline() -> smallvec::SmallVec<[(WaitValueAccessor, SemaphoreAccessor); 4]> {
                        smallvec::smallvec![
                            #(#wait_inner),*
                        ]
                    }
                    // fn signal_semaphore_timeline() -> smallvec::SmallVec<[(SignalValueAccessor, SemaphoreAccessor); 4]> {
                    //     fn accessor(frame_number: u64) -> u64 {
                    //         as_of::<super::super::#signal_timeline_path>(frame_number)
                    //     }
                    //     smallvec::smallvec![
                    //         (accessor as fn(u64) -> u64, #signal_out_sem)
                    //     ]
                    // }
                }
            })
        })
        .collect::<HashMap<_, _>>();

    let pass_definitions = new_data
        .passes
        .values()
        .map(|pass| {
            let pass_name = &pass.name;
            let format_param_ty = pass
                .attachments
                .iter()
                .filter(|&attachment| new_data.attachments.get(attachment).unwrap().format.is_dyn())
                .map(|_| quote!(vk::Format))
                .collect_vec();
            let attachment_desc = pass
                .attachments
                .iter()
                .map(|attachment| {
                    let data = new_data.attachments.get(attachment).expect("attachment not found");
                    let format = &data.format;
                    let PassLayout {
                        load_op,
                        initial_layout,
                        store_op,
                        final_layout,
                    } = pass
                        .layouts
                        .iter()
                        .find(|(name, _)| *name == attachment)
                        .map(|(_name, layout)| layout)
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
                    let format = format.clone().map(|s| format_ident!("{}", s));
                    let format = match format {
                        StaticOrDyn::Dyn => {
                            let dyn_ix = pass
                                .attachments
                                .iter()
                                .filter(|&at| new_data.attachments.get(at).unwrap().format.is_dyn())
                                .position(|at| at == attachment)
                                .unwrap();
                            let index = syn::Index::from(dyn_ix);
                            quote!(attachment_formats.#index)
                        }
                        StaticOrDyn::Static(format) => quote!(vk::Format::#format),
                    };
                    let samples = format_ident!("TYPE_{}", data.samples);
                    let initial_layout = format_ident!("{}", initial_layout);
                    let final_layout = format_ident!("{}", final_layout);
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
            let renderpass_definition = define_renderpass(&new_data, pass);
            let framebuffer_definition = define_framebuffer(&new_data, pass);
            let graph_ix = dependency_graph
                .node_references()
                .find(|(_ix, name)| *name == &pass_name.to_string())
                .map(|x| x.0)
                .unwrap()
                .index();
            let wait_instance = wait_instance.get(&pass_name.to_string()).unwrap();

            let pass_name_ident = format_ident!("{}", pass_name);
            quote! {
                #[allow(non_snake_case)]
                pub(crate) mod #pass_name_ident {
                    use super::{vk, Device, RenderStage, WaitValueAccessor,
                        SemaphoreAccessor, RenderFrame, ImageIndex, as_of, as_of_last,
                        OriginalFramebuffer, OriginalRenderPass};

                    #[derive(Debug, Clone, Copy)]
                    pub(crate) struct Stage;

                    pub(crate) const INDEX: u8 = #graph_ix as u8;

                    #renderpass_definition

                    #framebuffer_definition

                    #wait_instance

                    pub(crate) fn get_attachment_descriptions(
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

    let async_pass_definitions = new_data.async_passes.keys()
        .map(|pass_name| {
            let wait_instance = wait_instance.get(pass_name).unwrap();
            let graph_ix = dependency_graph
                .node_references()
                .find(|(_ix, name)| *name == pass_name)
                .map(|x| x.0)
                .unwrap().index();

            let pass_name_ident = format_ident!("{}", pass_name);
            quote! {
                #[allow(non_snake_case)]
                pub(crate) mod #pass_name_ident {
                    use super::{vk, RenderStage, WaitValueAccessor, SemaphoreAccessor, RenderFrame, ImageIndex, as_of, as_of_last};

                    #[derive(Debug, Clone, Copy)]
                    pub(crate) struct Stage;

                    pub(crate) const INDEX: u8 = #graph_ix as u8;

                    #wait_instance
                }
            }
        });

    let shader_definitions =
        syn::parse_str::<TokenStream>(&new_data.shader_information.shader_type_definitions).unwrap();

    let queue_manager = prepare_queue_manager(&new_data, &dependency_graph);

    let name = format_ident!("{}", new_data.name);
    let expanded = quote! {
        #validation_errors

        pub(crate) mod #name {
            use ash::vk;
            use super::{Device, RenderStage, WaitValueAccessor, SemaphoreAccessor,
                        RenderFrame, ImageIndex, as_of, as_of_last,
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
        }
    };

    expanded
}

fn define_renderpass(new_data: &RendererInput, pass: &Pass) -> TokenStream {
    let passes_that_need_clearing = pass.attachments.iter().enumerate().map(|(ix, attachment)| {
        let PassLayout { load_op, .. } = pass
            .layouts
            .iter()
            .find(|(name, _layout)| *name == attachment)
            .map(|(_name, at)| at)
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
            let format = &new_data.attachments.get(*attachment).unwrap().format;
            format.is_dyn()
        })
        .map(|_| quote!(vk::Format))
        .collect_vec();
    let debug_name = format!("{} - {}", new_data.name, pass.name);

    let renderpass_begin = format!("{}::{}::begin", new_data.name, pass.name);

    let (subpass_prerequisites, subpass_descriptions) = split2(pass.subpasses.iter().map(|subpass| {
        let (color_attachment_name, color_attachment_layout) = split2(
            subpass
                .color_attachments
                .iter()
                .map(|(name, layout)| (name, format_ident!("{}", layout))),
        );
        let (resolve_attachment_name, resolve_attachment_layout) = split2(
            subpass
                .resolve_attachments
                .iter()
                .map(|(name, layout)| (name, format_ident!("{}", layout))),
        );
        assert!(
            resolve_attachment_name.is_empty() || (color_attachment_name.len() == resolve_attachment_name.len()),
            "If resolving any attachments, must provide one for\
            each color attachment, ATTACHMENT_UNUSED not supported yet"
        );
        let color_attachment_ix = color_attachment_name.iter().map(|&needle| {
            pass.attachments
                .iter()
                .position(|candidate| candidate == needle)
                .expect("subpass color refers to nonexistent attachment")
        });
        let (depth_stencil_attachment_name, depth_stencil_attachment_layout) = split2(
            subpass
                .depth_stencil_attachments
                .iter()
                .map(|(name, layout)| (name, format_ident!("{}", layout))),
        );
        let depth_stencil_attachment_ix = depth_stencil_attachment_name.iter().map(|&needle| {
            pass.attachments
                .iter()
                .position(|candidate| candidate == needle)
                .expect("subpass depth stencil refers to nonexistent attachment")
        });
        let resolve_attachment_ix = resolve_attachment_name.iter().map(|&needle| {
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

    let subpass_dependency = pass.dependencies.iter().map(|dep| {
        // (dep.from, dep.to, dep.src_stage, dep.dst_stage, dep.src_access, dep.dst_access);
        let src_subpass = pass
            .subpasses
            .iter()
            .position(|candidate| candidate.name == dep.src)
            .expect("did not find src subpass");
        let dst_subpass = pass
            .subpasses
            .iter()
            .position(|candidate| candidate.name == dep.dst)
            .expect("did not find src subpass");
        let SubpassDependency {
            src_stage: src_stage_mask,
            dst_stage: dst_stage_mask,
            src_access: src_access_mask,
            dst_access: dst_access_mask,
            ..
        } = dep;
        let src_stage_mask = format_ident!("{}", src_stage_mask);
        let dst_stage_mask = format_ident!("{}", dst_stage_mask);
        let src_access_mask = format_ident!("{}", src_access_mask);
        let dst_access_mask = format_ident!("{}", dst_access_mask);

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

fn define_framebuffer(new_data: &RendererInput, pass: &Pass) -> TokenStream {
    let debug_name = format!(
        "{} framebuffer {} [{{}}]",
        new_data.name.to_string(),
        pass.name.to_string()
    );
    let attachment_count = pass.attachments.len();
    let attachment_ix = pass.attachments.iter().enumerate().map(|(ix, _)| ix).collect_vec();

    let format_param_ty = pass
        .attachments
        .iter()
        .filter(|attachment| {
            let format = &new_data.attachments.get(*attachment).unwrap().format;
            format.is_dyn()
        })
        .map(|_| quote!(vk::Format))
        .collect_vec();
    let attachment_formats_expr = pass
        .attachments
        .iter()
        .map(|attachment| {
            let format = &new_data.attachments.get(attachment).unwrap().format;
            match format {
                StaticOrDyn::Dyn => {
                    let dyn_ix = pass
                        .attachments
                        .iter()
                        .filter(|&at| new_data.attachments.get(at).unwrap().format.is_dyn())
                        .position(|at| at == attachment)
                        .unwrap();
                    let index = syn::Index::from(dyn_ix);
                    quote!(dyn_attachment_formats.#index)
                }
                StaticOrDyn::Static(format) => {
                    let format = format_ident!("{}", format);
                    quote!(vk::Format::#format)
                }
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

#[proc_macro]
pub fn define_set(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let inputs::DescriptorSet { name, .. } = parse_macro_input!(input as inputs::DescriptorSet);
    let data = fetch().unwrap();
    let set = data.descriptor_sets.get(&name.to_string()).unwrap();

    define_set_old(&data.name, set, &data.shader_information.set_binding_type_names).into()
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

    let layout_debug_name = format!("{} Layout", set_name.to_string());
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
    let input = parse_macro_input!(input as inputs::Pipe);
    let pipe = Pipeline::from(input);
    let data = fetch().unwrap();

    define_pipe_old(
        &pipe,
        data.shader_information.push_constant_type_definitions.get(&pipe.name),
    )
    .into()
}

fn define_pipe_old(pipe: &Pipeline, push_constant_type: Option<&String>) -> TokenStream {
    let Pipeline {
        name,
        specialization_constants,
        descriptor_sets: descriptors,
        specific,
        varying_subgroup_stages,
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
        let shader_stage = syn::parse_str::<Ident>(&shader_stage).unwrap();
        spirv_code.extend_one(quote! {
            pub(crate) static #shader_stage: &'static [u8] = crate::renderer::include_bytes_align_as!(u32, #shader_path);
            #[cfg(feature = "shader_reload")]
            pub(crate) static #shader_stage_path: &'static str = #shader_src_path;
        });
    }

    let pipe_arguments_new_types = match specific {
        SpecificPipe::Graphics(specific) => {
            let dynamic_samples =
                extract_optional_dyn(&specific.samples, quote!(vk::SampleCountFlags)).unwrap_or_default();
            quote! {
                (vk::RenderPass, u32, #dynamic_samples)
            }
        }
        SpecificPipe::Compute => quote!(()),
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
        SpecificPipe::Compute => vec![],
    };
    let shader_stage = specific.stages();
    let shader_stage_path = shader_stage
        .iter()
        .map(|shader_stage| format_ident!("{}_PATH", &shader_stage))
        .collect_vec();
    let allow_varying_snippet = quote!(true);
    let forbid_varying_snippet = quote!(false);
    let varying_subgroup_stages = shader_stage
        .iter()
        .map(|stage| {
            if varying_subgroup_stages.iter().any(|candidate| candidate == stage) {
                &allow_varying_snippet
            } else {
                &forbid_varying_snippet
            }
        })
        .collect_vec();

    let pipeline_definition_inner = match specific {
        SpecificPipe::Graphics(specific) => {
            let truthy: &LitBool = &parse_quote!(true);
            let falsy: &LitBool = &parse_quote!(false);
            let polygon_mode: Ident = parse_quote!(FILL);
            let polygon_mode = specific
                .polygon_mode
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(polygon_mode.clone());
            let front_face: Ident = parse_quote!(CLOCKWISE);
            let front_face = specific
                .front_face_mode
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(front_face.clone());
            let front_face_dynamic =
                extract_optional_dyn(&specific.front_face_mode, quote!(vk::DynamicState::FRONT_FACE_EXT,));
            let topology_mode: Ident = parse_quote!(TRIANGLE_LIST);
            let topology_mode = specific
                .topology_mode
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(topology_mode.clone());
            let topology_dynamic = extract_optional_dyn(
                &specific.topology_mode,
                quote!(vk::DynamicState::PRIMITIVE_TOPOLOGY_EXT,),
            );
            let cull_mode: Ident = parse_quote!(NONE);
            let cull_mode = specific
                .cull_mode
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(cull_mode.clone());
            let cull_mode_dynamic = extract_optional_dyn(&specific.cull_mode, quote!(vk::DynamicState::CULL_MODE_EXT,));
            let depth_test_enable = falsy;
            let depth_test_enable = specific
                .depth_test_enable
                .as_ref()
                .map(|p| if *p { truthy } else { falsy })
                .unwrap_or_default(&depth_test_enable);
            let depth_test_dynamic = extract_optional_dyn(
                &specific.depth_test_enable,
                quote!(vk::DynamicState::DEPTH_TEST_ENABLE_EXT,),
            );
            let depth_write_enable = falsy;
            let depth_write_enable = specific
                .depth_write_enable
                .as_ref()
                .map(|p| if *p { truthy } else { falsy })
                .unwrap_or_default(&depth_write_enable);
            let depth_write_dynamic = extract_optional_dyn(
                &specific.depth_write_enable,
                quote!(vk::DynamicState::DEPTH_WRITE_ENABLE_EXT,),
            );
            let depth_bounds_enable = falsy;
            let depth_bounds_enable = specific
                .depth_bounds_enable
                .as_ref()
                .map(|p| if *p { truthy } else { falsy })
                .unwrap_or_default(&depth_bounds_enable);
            let depth_bounds_dynamic = extract_optional_dyn(
                &specific.depth_bounds_enable,
                quote!(vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE_EXT,),
            );
            let depth_compare_op: Ident = parse_quote!(NEVER);
            let depth_compare_op = specific
                .depth_compare_op
                .as_ref()
                .map(|p| format_ident!("{}", p))
                .unwrap_or_default(depth_compare_op.clone());
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
                        .layout(layout)
                        .render_pass(renderpass)
                        .subpass(subpass)
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

    let pipe_debug_name = format!("{}::Pipeline", pipe.name.to_string());
    let shader_stage = shader_stage
        .iter()
        .map(|x| syn::parse_str::<Ident>(x).unwrap())
        .collect_vec();
    let pipeline_definition2 = quote! {
        pub(crate) struct Pipeline {
            pub(crate) pipeline: crate::renderer::device::Pipeline,
        }

        impl crate::renderer::Pipeline for Pipeline {
            type DynamicArguments = #pipe_arguments_new_types;
            type Layout = PipelineLayout;
            type Specialization = Specialization;

            fn default_shader_stages() -> smallvec::SmallVec<[&'static [u8]; 4]> {
                smallvec::smallvec![#(#shader_stage),*]
            }

            fn shader_stages() -> smallvec::SmallVec<[vk::ShaderStageFlags; 4]> {
                smallvec::smallvec![#(vk::ShaderStageFlags::#shader_stage),*]
            }

            #[cfg(feature = "shader_reload")]
            fn shader_stage_paths() -> smallvec::SmallVec<[&'static str; 4]> {
                smallvec::smallvec![#(#shader_stage_path),*]
            }

            fn varying_subgroup_stages() -> smallvec::SmallVec<[bool; 4]> {
                smallvec::smallvec![#(#varying_subgroup_stages),*]
            }

            fn vk(&self) -> vk::Pipeline {
                *self.pipeline
            }

            fn new_raw(
                device: &Device,
                layout: vk::PipelineLayout,
                stages: &[vk::PipelineShaderStageCreateInfo],
                flags: vk::PipelineCreateFlags,
                base_pipeline_handle: vk::Pipeline,
                (#(#pipe_argument_short,)*): Self::DynamicArguments,
            ) -> Self {
                #pipeline_definition_inner

                device.set_object_name(*pipeline, #pipe_debug_name);

                Pipeline {
                    pipeline,
                }
            }

            fn destroy(self, device: &Device) {
                self.pipeline.destroy(device);
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
    let name = syn::parse_str::<Ident>(&name).unwrap();
    let descriptor_path = descriptors
        .iter()
        .map(|x| syn::parse_str::<Path>(x).unwrap())
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
            use microprofile::scope;

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

            #spirv_code
        }
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

fn prepare_queue_manager(input: &RendererInput, dependency_graph: &DiGraph<String, DependencyType>) -> TokenStream {
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
    let edges_count = edges_definitions.len();

    let toposort_compute_virtual_queue_index = {
        let toposort = petgraph::algo::toposort(&dependency_graph, None).unwrap();
        let toposort_compute = toposort
            .iter()
            .filter(|ix| {
                let name = dependency_graph.node_weight(**ix).unwrap();
                input
                    .async_passes
                    .get(name)
                    .map(|async_pass| async_pass.queue == QueueFamily::Compute)
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
        .map(|Pass { ref name, .. }| (name, QueueFamily::Graphics))
        .chain(
            input
                .async_passes
                .values()
                .map(|AsyncPass { name, queue, .. }| (name, *queue)),
        )
        .map(|(name, queue)| {
            let graph_ix = dependency_graph
                .node_indices()
                .find(|&ix| *name == dependency_graph[ix])
                .expect("did not find pass in the dependency graph");
            let queue_def = match queue {
                QueueFamily::Graphics => quote!(renderer.device.graphics_queue()),
                QueueFamily::Compute => {
                    let virtual_queue_index = toposort_compute_virtual_queue_index.get(&graph_ix.index()).unwrap();
                    quote!(renderer.device.compute_queues[#virtual_queue_index % renderer.device.compute_queues.len()])
                }
                QueueFamily::Transfer => quote!(renderer.device.transfer_queue.as_ref().unwrap()),
            };

            let name_ident = format_ident!("{}", name);
            quote! {
                ix if ix.index() as u8 == #name_ident::INDEX => {
                    let queue = #queue_def.lock();
                    let buf = &mut [vk::CommandBuffer::null()];
                    let cmds: &[vk::CommandBuffer] = match cb {
                        Some(cmd) => {
                            buf[0] = cmd;
                            buf
                        }
                        None => &[],
                    };
                    #name_ident::Stage::queue_submit(&image_index, &renderer, *queue, cmds).expect("submit failed");
                }
            }
        })
        .collect_vec();

    quote! {
        use bevy_ecs::prelude::*;

        pub(crate) const DEPENDENCY_GRAPH: [(u8, u8); #edges_count] = [#(#edges_definitions),*];

        pub(crate) fn submit_stage_by_index(
            renderer: &RenderFrame,
            image_index: &ImageIndex,
            ix: petgraph::stable_graph::NodeIndex<u8>,
            cb: Option<vk::CommandBuffer>
        ) {
            match ix {
                #(#submit_clauses),*
                _ => panic!("Invalid pass index in queue_manager submit()"),
            }
        }
    }
}

fn grouped_toposort<G: GraphBase + IntoNeighborsDirected + IntoNodeIdentifiers + Visitable>(g: G) -> Vec<Vec<G::NodeId>>
where
    G::NodeId: std::fmt::Debug,
{
    let toposort = petgraph::algo::toposort(&g, None).unwrap();

    let mut toposort_grouped: Vec<Vec<G::NodeId>> = vec![];
    for ix in toposort.iter() {
        match toposort_grouped.last_mut() {
            // if no path bridges from the last stage to ix
            Some(last)
                if !last
                    .iter()
                    .any(|candidate| petgraph::algo::has_path_connecting(&g, *candidate, *ix, None)) =>
            {
                last.push(*ix);
            }
            _ => toposort_grouped.push(vec![*ix]),
        }
    }
    toposort_grouped
}
