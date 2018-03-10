#![allow(unused_variables)]

extern crate ash;
extern crate cgmath;
extern crate forward_renderer;
extern crate futures;
extern crate gltf;
extern crate petgraph;
extern crate specs;

use ecs::*;
use forward_renderer::*;
use render_dag::v3::{*, util::UseQueue};

use ash::{vk, version::DeviceV1_0};
use cgmath::Rotation3;
use futures::{executor::ThreadPool, future::lazy};
use std::{ptr, borrow::Cow, default::Default, mem::{size_of, transmute}, os::raw::c_void,
          path::PathBuf, sync::Arc};

fn main() {
    let mut world = specs::World::new();
    world.register::<Position>();
    world.register::<Rotation>();
    world.register::<Scale>();
    world.register::<Matrices>();
    let mut dag = RenderDAG::new();
    let window_ix = dag.new_window(800, 600);
    let (device_ix, graphics_family, compute_family, transfer_family) =
        dag.new_device(window_ix).unwrap();
    let device = match dag.graph[device_ix] {
        RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
        _ => None,
    }.unwrap();
    let swapchain_ix = dag.new_swapchain(device_ix).unwrap();
    let surface_format = match dag.graph[swapchain_ix] {
        RenderNode::Swapchain {
            ref surface_format, ..
        } => Some(surface_format.clone()),
        _ => None,
    }.expect("could not locate surface format");
    let (renderpass_ix, end_renderpass_ix, subpass_ixes) = {
        let attachment_descriptions = [
            vk::AttachmentDescription {
                format: surface_format.format,
                flags: vk::AttachmentDescriptionFlags::empty(),
                samples: vk::SAMPLE_COUNT_1_BIT,
                load_op: vk::AttachmentLoadOp::Clear,
                store_op: vk::AttachmentStoreOp::Store,
                stencil_load_op: vk::AttachmentLoadOp::DontCare,
                stencil_store_op: vk::AttachmentStoreOp::DontCare,
                initial_layout: vk::ImageLayout::Undefined,
                final_layout: vk::ImageLayout::PresentSrcKhr,
            },
        ];
        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::ColorAttachmentOptimal,
        };
        let subpass_descs = [
            vk::SubpassDescription {
                color_attachment_count: 1,
                p_color_attachments: &color_attachment_ref,
                p_depth_stencil_attachment: ptr::null(),
                flags: Default::default(),
                pipeline_bind_point: vk::PipelineBindPoint::Graphics,
                input_attachment_count: 0,
                p_input_attachments: ptr::null(),
                p_resolve_attachments: ptr::null(),
                preserve_attachment_count: 0,
                p_preserve_attachments: ptr::null(),
            },
        ];
        let subpass_dependencies = [
            vk::SubpassDependency {
                dependency_flags: Default::default(),
                src_subpass: vk::VK_SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                src_access_mask: Default::default(),
                dst_access_mask: vk::ACCESS_COLOR_ATTACHMENT_READ_BIT
                    | vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_stage_mask: vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            },
        ];
        dag.new_renderpass(
            device_ix,
            &attachment_descriptions,
            &subpass_descs,
            &subpass_dependencies,
        ).unwrap()
    };
    dag.node_names
        .insert(renderpass_ix, Cow::Borrowed("Triangle renderpass start"));
    dag.node_names
        .insert(end_renderpass_ix, Cow::Borrowed("Triangle renderpass end"));
    let (framebuffer_ix, present_ix) = dag.new_framebuffer(swapchain_ix, renderpass_ix).unwrap();
    dag.graph
        .add_edge(renderpass_ix, present_ix, Edge::Propagate);
    let graphics_command_pool_ix = dag.new_command_pool(
        device_ix,
        graphics_family,
        vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    ).unwrap();
    dag.node_names.insert(
        graphics_command_pool_ix,
        Cow::Borrowed("Graphics command pool"),
    );
    let compute_command_pool_ix = dag.new_command_pool(
        device_ix,
        compute_family,
        vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    ).unwrap();
    dag.node_names.insert(
        compute_command_pool_ix,
        Cow::Borrowed("Compute command pool"),
    );
    let transfer_command_pool_ix = dag.new_command_pool(
        device_ix,
        transfer_family,
        vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    ).unwrap();
    dag.node_names.insert(
        transfer_command_pool_ix,
        Cow::Borrowed("Transfer command pool"),
    );
    let (command_buffer_ix, submit_commands_ix) = dag.new_allocate_command_buffer(
        graphics_command_pool_ix,
        UseQueue::Graphics
    ).unwrap();
    dag.node_names
        .insert(command_buffer_ix, Cow::Borrowed("Graphics Command buffer"));
    dag.node_names.insert(
        submit_commands_ix,
        Cow::Borrowed("Submit graphics Command buffer"),
    );
    dag.graph
        .add_edge(submit_commands_ix, present_ix, Edge::Propagate);
    dag.graph
        .add_edge(framebuffer_ix, command_buffer_ix, Edge::Propagate);
    dag.graph
        .add_edge(command_buffer_ix, renderpass_ix, Edge::Propagate);
    dag.graph
        .add_edge(end_renderpass_ix, submit_commands_ix, Edge::Direct);

    // Semaphores
    let present_semaphore_ix = dag.new_persistent_semaphore(device_ix).unwrap();
    dag.node_names
        .insert(present_semaphore_ix, Cow::Borrowed("Present semaphore"));
    dag.graph
        .add_edge(framebuffer_ix, present_semaphore_ix, Edge::Direct);
    let rendering_complete_semaphore_ix = dag.new_persistent_semaphore(device_ix).unwrap();
    dag.node_names.insert(
        rendering_complete_semaphore_ix,
        Cow::Borrowed("Rendering semaphore"),
    );
    dag.graph.add_edge(
        submit_commands_ix,
        rendering_complete_semaphore_ix,
        Edge::Direct,
    );
    dag.graph
        .add_edge(rendering_complete_semaphore_ix, present_ix, Edge::Direct);

    let descriptor_set_layout_ix = dag.new_descriptor_set_layout(
        device_ix,
        &[
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1,
                stage_flags: vk::SHADER_STAGE_VERTEX_BIT,
                p_immutable_samplers: ptr::null(),
            },
        ],
    ).unwrap();

    let pipeline_layout = dag.new_pipeline_layout(
        device_ix,
        &[descriptor_set_layout_ix],
        vec![
            vk::PushConstantRange {
                stage_flags: vk::SHADER_STAGE_VERTEX_BIT,
                offset: 0,
                size: (size_of::<f32>() * 2 * 3) as u32,
            },
        ],
    ).unwrap();
    let triangle_pipeline = dag.new_graphics_pipeline(
        pipeline_layout,
        renderpass_ix,
        &[],
        &[],
        &[
            (
                vk::SHADER_STAGE_VERTEX_BIT,
                PathBuf::from(env!("OUT_DIR")).join("triangle.vert.spv"),
            ),
            (
                vk::SHADER_STAGE_FRAGMENT_BIT,
                PathBuf::from(env!("OUT_DIR")).join("triangle.frag.spv"),
            ),
        ],
    ).unwrap();
    dag.node_names.insert(
        triangle_pipeline,
        Cow::Borrowed("Triangle graphics pipeline"),
    );
    let descriptor_pool_ix = dag.new_descriptor_pool(
        device_ix,
        1,
        &[
            vk::DescriptorPoolSize {
                typ: vk::DescriptorType::UniformBuffer,
                descriptor_count: 1024,
            },
        ],
    ).unwrap();
    dag.node_names
        .insert(descriptor_pool_ix, Cow::Borrowed("Main descriptor pool"));
    let descriptor_set_ix =
        dag.new_descriptor_set(device_ix, descriptor_pool_ix, descriptor_set_layout_ix)
            .unwrap();
    dag.node_names
        .insert(descriptor_set_ix, Cow::Borrowed("Main descriptor set"));
    let ubo_set = match dag.graph[descriptor_set_ix] {
        RenderNode::DescriptorSet { handle, .. } => Some(handle),
        _ => None,
    }.unwrap();
    let ubo_buffer_ix = dag.new_buffer(
        device_ix,
        vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        alloc::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_MAPPED_BIT.0 as u32,
        alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        4 * 4 * 4 * 1024,
    ).unwrap();
    dag.node_names
        .insert(ubo_buffer_ix, Cow::Borrowed("MVP Uniform buffer"));
    let (ubo_buffer, ubo_allocation_ptr) = match dag.graph[ubo_buffer_ix] {
        RenderNode::Buffer {
            handle,
            ref allocation_info,
            ..
        } => Some((handle, allocation_info.pMappedData)),
        _ => None,
    }.unwrap();
    {
        let buffer_updates = &[
            vk::DescriptorBufferInfo {
                buffer: ubo_buffer,
                offset: 0,
                range: 1024 * size_of::<cgmath::Matrix4<f32>>() as vk::DeviceSize,
            },
        ];
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WriteDescriptorSet,
                        p_next: ptr::null(),
                        dst_set: ubo_set,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UniformBuffer,
                        p_image_info: ptr::null(),
                        p_buffer_info: buffer_updates.as_ptr(),
                        p_texel_buffer_view: ptr::null(),
                    },
                ],
                &[],
            );
        }
    }
    let (window_width, window_height) = match dag.graph[window_ix] {
        RenderNode::Instance {
            window_width,
            window_height,
            ..
        } => Some((window_width, window_height)),
        _ => None,
    }.unwrap();
    let projection = cgmath::perspective(
        cgmath::Deg(60.0),
        window_width as f32 / window_height as f32,
        0.1,
        100.0,
    );
    let view = cgmath::Matrix4::look_at(
        cgmath::Point3::new(0.0, 0.0, -5.0),
        cgmath::Point3::new(0.0, 0.0, 0.0),
        cgmath::vec3(0.0, 1.0, 0.0),
    );
    world
        .create_entity()
        .with::<Position>(Position(cgmath::Vector3::new(0.0, 0.0, 0.0)))
        .with::<Rotation>(Rotation(cgmath::Quaternion::from_angle_x(cgmath::Deg(0.0))))
        .with::<Scale>(Scale(1.0))
        .with::<Matrices>(Matrices::one())
        .build();
    let ubo_mapped =
        unsafe { transmute::<*mut c_void, *mut cgmath::Matrix4<f32>>(ubo_allocation_ptr) };
    let mut dispatcher = specs::DispatcherBuilder::new()
        .add(SteadyRotation, "steady_rotation", &[])
        .add(
            MVPCalculation { projection, view },
            "mvp",
            &["steady_rotation"],
        )
        .add(MVPUpload { dst: ubo_mapped }, "mvp_upload", &["mvp"])
        .build();
    dag.graph
        .add_edge(triangle_pipeline, end_renderpass_ix, Edge::Propagate);
    let draw_calls = dag.new_draw_calls(Arc::new(|ix, graph, world, dynamic| {
        use render_dag::v3::util::*;
        use petgraph::prelude::*;
        use futures::prelude::*;
        let device = search_deps_exactly_one(graph, ix, |node| match *node {
            RenderNode::Device { ref device, .. } => Some(Arc::clone(device)),
            _ => None,
        }).expect("Device not found in deps of NextSubpass");
        let command_buffer_locked = search_deps_exactly_one(graph, ix, |node| match *node {
            RenderNode::AllocateCommandBuffer { ref dynamic, .. } => Some(Arc::clone(dynamic)),
            _ => None,
        }).expect("Command Buffer not found in deps of NextSubpass");
        let command_buffer_fut = command_buffer_locked
            .read()
            .expect("Could not read command buffer")
            .clone();
        let pipeline_layout = search_deps_exactly_one(graph, ix, |node| match *node {
            RenderNode::PipelineLayout { handle, .. } => Some(handle),
            _ => None,
        }).expect("Command Buffer not found in deps of NextSubpass");
        let descriptor_sets =
            search_direct_deps(graph, ix, Direction::Incoming, |node| match *node {
                RenderNode::DescriptorSet { handle, .. } => Some(handle),
                _ => None,
            });
        let order_fut = wait_on_direct_deps(graph, ix);
        let mut lock = dynamic.write().expect("failed to lock present for writing");
        *lock = (Box::new(order_fut.join(command_buffer_fut).map_err(|_| ()).map(
            move |(_, cb_dyn)| {
                println!("My draw calls");
                let cb = cb_dyn.current_frame;
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::Graphics,
                        pipeline_layout,
                        0,
                        &descriptor_sets,
                        &[],
                    );
                    let constants = [1.0f32, 1.0, -1.0, 1.0, 1.0, -1.0];
                    use std::mem::transmute;
                    use std::slice::from_raw_parts;

                    let casted: &[u8] = {
                        from_raw_parts(
                            transmute::<*const f32, *const u8>(constants.as_ptr()),
                            2 * 3 * 4,
                        )
                    };
                    device.cmd_push_constants(
                        cb,
                        pipeline_layout,
                        vk::SHADER_STAGE_VERTEX_BIT,
                        0,
                        casted,
                    );
                    device.cmd_draw(cb, 3, 1, 0, 0);
                }
                ()
            },
        )) as Box<Future<Item = (), Error = ()>>)
            .shared();
    }));
    dag.node_names
        .insert(draw_calls, Cow::Borrowed("Triangle draw calls"));
    dag.graph
        .add_edge(triangle_pipeline, draw_calls, Edge::Propagate);
    dag.graph
        .add_edge(descriptor_set_ix, draw_calls, Edge::Propagate);
    dag.graph
        .add_edge(draw_calls, end_renderpass_ix, Edge::Propagate);
    {
        // Upload facilities
        let (upload_cb_ix, submit_upload_ix) = dag.new_allocate_command_buffer(
            transfer_command_pool_ix,
            UseQueue::Transfer
        ).unwrap();
        dag.node_names
            .insert(upload_cb_ix, Cow::Borrowed("Upload Command buffer"));
        dag.node_names.insert(
            submit_upload_ix,
            Cow::Borrowed("Submit Upload Command buffer"),
        );
        dag.graph
            .add_edge(framebuffer_ix, upload_cb_ix, Edge::Propagate);
        dag.graph
            .add_edge(present_semaphore_ix, submit_upload_ix, Edge::Propagate);
        let sync_upload_semaphore_ix = dag.new_persistent_semaphore(device_ix).unwrap();
        dag.node_names.insert(
            sync_upload_semaphore_ix,
            Cow::Borrowed("Sync Upload Semaphore"),
        );
        dag.graph
            .add_edge(submit_upload_ix, sync_upload_semaphore_ix, Edge::Propagate);
        dag.graph
            .add_edge(submit_upload_ix, submit_commands_ix, Edge::Direct);
        dag.graph
            .add_edge(device_ix, submit_upload_ix, Edge::Propagate);
        dag.graph.add_edge(
            sync_upload_semaphore_ix,
            submit_commands_ix,
            Edge::Propagate,
        );
    }
    println!("{}", dot(&dag.graph, &dag.node_names).unwrap());
    let mut threadpool = ThreadPool::new();
    for _i in 1..200 {
        dispatcher.dispatch(&world.res);
        let _: Result<(), ()> = threadpool.run(lazy(|| {
            dag.render_frame(&world);
            Ok(())
        }));
        if let RenderNode::PresentFramebuffer { ref dynamic, .. } = dag.graph[present_ix] {
            let lock = dynamic.read().unwrap();
            let res = threadpool.run((*lock).clone());
            println!("{:?}", res);
        } else {
            panic!("present framebuffer does not exist?")
        }
    }
}
