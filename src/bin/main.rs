extern crate ash;
extern crate cgmath;
#[macro_use]
extern crate forward_renderer;
extern crate petgraph;
extern crate specs;

use forward_renderer::*;
use ecs::*;
use mesh;

use ash::vk;
use ash::version::*;
use cgmath::One;
use std::default::Default;
use std::ptr;
use std::ffi::CString;
use std::mem;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use ash::util::*;
use std::mem::align_of;

#[derive(Clone, Debug, Copy)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

fn main() {
    let base = ExampleBase::new(1920, 1080);
    let render_dag = {
        let mut builder = RenderDAGBuilder::new();
        builder.add_node("acquire_image", Node::AcquirePresentImage);
        builder.add_node("swapchain_attachment", Node::SwapchainAttachment(0));
        builder.add_node("depth_attachment", Node::DepthAttachment(1));
        builder.add_node("subpass", Node::Subpass(1));
        builder.add_node("renderpass", Node::RenderPass);
        builder.add_node(
            "vertex_shader",
            Node::VertexShader(PathBuf::from(env!("OUT_DIR")).join("simple_color.vert.spv")),
        );
        builder.add_node(
            "fragment_shader",
            Node::FragmentShader(PathBuf::from(env!("OUT_DIR")).join("simple_color.frag.spv")),
        );
        builder.add_node("pipeline_layout", Node::PipelineLayout);
        builder.add_node(
            "vertex_binding",
            Node::VertexInputBinding(0, 3 * 4, vk::VertexInputRate::Vertex),
        );
        builder.add_node(
            "uv_binding",
            Node::VertexInputBinding(1, 2 * 4, vk::VertexInputRate::Vertex),
        );
        builder.add_node(
            "vertex_attribute",
            Node::VertexInputAttribute(0, 0, vk::Format::R32g32b32Sfloat, 0),
        );
        builder.add_node(
            "uv_attribute",
            Node::VertexInputAttribute(1, 1, vk::Format::R32g32Sfloat, 0),
        );
        builder.add_node("graphics_pipeline", Node::GraphicsPipeline);
        builder.add_node(
            "draw_commands",
            Node::DrawCommands(Arc::new(|base, world, render_dag, command_buffer| {
                use specs::Join;
                for mesh in world.read::<SimpleColorMesh>().join() {
                    unsafe {
                        base.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::Graphics,
                            render_dag
                                .pipeline_layouts
                                .get("pipeline_layout")
                                .unwrap()
                                .clone(),
                            0,
                            &[
                                render_dag
                                    .descriptor_sets
                                    .get("main_descriptor_layout")
                                    .unwrap()
                                    .clone(),
                            ],
                            &[],
                        );
                        base.device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[mesh.0.vertex_buffer.buffer(), mesh.0.tex_coords.buffer()],
                            &[0, 0],
                        );
                        base.device.cmd_bind_index_buffer(
                            command_buffer,
                            mesh.0.index_buffer.buffer(),
                            0,
                            mesh.0.index_type,
                        );
                        base.device.cmd_draw_indexed(
                            command_buffer,
                            mesh.0.index_count,
                            1,
                            0,
                            0,
                            0,
                        );
                    }
                }
            })),
        );
        builder.add_node("present_image", Node::PresentImage);
        builder.add_edge("acquire_image", "present_image");
        builder.add_edge("acquire_image", "swapchain_attachment");
        builder.add_edge("acquire_image", "depth_attachment");
        builder.add_edge("subpass", "renderpass");
        builder.add_edge("subpass", "graphics_pipeline");
        builder.add_edge("subpass", "draw_commands");
        builder.add_edge("swapchain_attachment", "renderpass");
        builder.add_edge("depth_attachment", "renderpass");
        builder.add_edge("vertex_shader", "graphics_pipeline");
        builder.add_edge("fragment_shader", "graphics_pipeline");
        builder.add_edge("renderpass", "graphics_pipeline");
        builder.add_edge("vertex_binding", "graphics_pipeline");
        builder.add_edge("uv_binding", "graphics_pipeline");
        builder.add_edge("vertex_attribute", "graphics_pipeline");
        builder.add_edge("uv_attribute", "graphics_pipeline");
        builder.add_edge("pipeline_layout", "graphics_pipeline");
        builder.add_edge("graphics_pipeline", "draw_commands");
        builder.add_edge("draw_commands", "present_image");

        {
            builder.add_node("triangle_subpass", Node::Subpass(0));
            // Triangle mesh
            builder.add_node(
                "triangle_vertex_shader",
                Node::VertexShader(PathBuf::from(env!("OUT_DIR")).join("triangle.vert.spv")),
            );
            builder.add_node(
                "triangle_fragment_shader",
                Node::FragmentShader(PathBuf::from(env!("OUT_DIR")).join("triangle.frag.spv")),
            );
            builder.add_node("triangle_pipeline_layout", Node::PipelineLayout);
            builder.add_node(
                "triangle_vertex_binding",
                Node::VertexInputBinding(0, 4 * 5, vk::VertexInputRate::Vertex),
            );
            builder.add_node(
                "triangle_vertex_attribute_pos",
                Node::VertexInputAttribute(0, 0, vk::Format::R32g32Sfloat, 0),
            );
            builder.add_node(
                "triangle_vertex_attribute_color",
                Node::VertexInputAttribute(0, 1, vk::Format::R32g32b32Sfloat, 2 * 4),
            );
            builder.add_node("triangle_graphics_pipeline", Node::GraphicsPipeline);
            builder.add_node(
                "triangle_draw_commands",
                Node::DrawCommands(Arc::new(|base, world, render_dag, command_buffer| {
                    use specs::Join;
                    for mesh in world.read::<TriangleMesh>().join() {
                        unsafe {
                            base.device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[mesh.0.vertex_buffer.buffer()],
                                &[0],
                            );
                            base.device.cmd_bind_index_buffer(
                                command_buffer,
                                mesh.0.index_buffer.buffer(),
                                0,
                                mesh.0.index_type,
                            );
                            base.device.cmd_draw_indexed(
                                command_buffer,
                                mesh.0.index_count,
                                1,
                                0,
                                0,
                                0,
                            );
                        }
                    }
                })),
            );

            builder.add_edge("triangle_subpass", "renderpass");
            builder.add_edge("triangle_subpass", "triangle_graphics_pipeline");
            builder.add_edge("triangle_subpass", "triangle_draw_commands");
            builder.add_edge("triangle_vertex_shader", "triangle_graphics_pipeline");
            builder.add_edge("triangle_fragment_shader", "triangle_graphics_pipeline");
            builder.add_edge("renderpass", "triangle_graphics_pipeline");
            builder.add_edge("triangle_vertex_binding", "triangle_graphics_pipeline");
            builder.add_edge(
                "triangle_vertex_attribute_pos",
                "triangle_graphics_pipeline",
            );
            builder.add_edge(
                "triangle_vertex_attribute_color",
                "triangle_graphics_pipeline",
            );
            builder.add_edge("triangle_pipeline_layout", "triangle_graphics_pipeline");
            builder.add_edge("triangle_graphics_pipeline", "triangle_draw_commands");
            builder.add_edge("triangle_subpass", "subpass");
        }

        builder.add_node(
            "mvp_ubo",
            Node::DescriptorBinding(
                0,
                vk::DescriptorType::UniformBuffer,
                vk::SHADER_STAGE_VERTEX_BIT,
                1,
            ),
        );
        builder.add_node(
            "color_texture",
            Node::DescriptorBinding(
                1,
                vk::DescriptorType::CombinedImageSampler,
                vk::SHADER_STAGE_FRAGMENT_BIT,
                1,
            ),
        );
        builder.add_node("main_descriptor_layout", Node::DescriptorSet);

        builder.add_edge("mvp_ubo", "main_descriptor_layout");
        builder.add_edge("color_texture", "main_descriptor_layout");
        builder.add_edge("main_descriptor_layout", "pipeline_layout");
        {
            let dot = petgraph::dot::Dot::new(&builder.graph);
            println!("{:?}", dot);
        }
        builder.build(&base)
    };
    let dot = petgraph::dot::Dot::new(&render_dag.graph);
    println!("{:?}", dot);
    unsafe {
        let renderpass = render_dag.renderpasses.get("renderpass").unwrap();
        let framebuffers: Vec<vk::Framebuffer> = base.present_image_views
            .iter()
            .map(|&present_image_view| {
                let framebuffer_attachments = [present_image_view, base.depth_image_view];
                let frame_buffer_create_info = vk::FramebufferCreateInfo {
                    s_type: vk::StructureType::FramebufferCreateInfo,
                    p_next: ptr::null(),
                    flags: Default::default(),
                    render_pass: renderpass.clone(),
                    attachment_count: framebuffer_attachments.len() as u32,
                    p_attachments: framebuffer_attachments.as_ptr(),
                    width: base.surface_resolution.width,
                    height: base.surface_resolution.height,
                    layers: 1,
                };
                base.device
                    .create_framebuffer(&frame_buffer_create_info, None)
                    .unwrap()
            })
            .collect();

        let mut world = World::new(&base.device);

        world
            .create_entity()
            .with::<Position>(Position(cgmath::Vector3::new(0.0, 0.0, 0.0)))
            .with::<Rotation>(Rotation(cgmath::Quaternion::one()))
            .with::<Scale>(Scale(1.0))
            .with::<MVP>(MVP(cgmath::Matrix4::one()))
            .with::<SimpleColorMesh>(SimpleColorMesh(
                mesh::Mesh::from_gltf(
                    &base,
                    "glTF-Sample-Models/2.0/BoxTextured/glTF/BoxTextured.gltf",
                ).unwrap(),
            ))
            .with::<TriangleMesh>(TriangleMesh(mesh::TriangleMesh::dummy(&base)));
        {
            use specs::Join;
            let mut textures = vec![];
            for mesh in world.read::<SimpleColorMesh>().join() {
                let create_info = vk::ImageViewCreateInfo {
                    s_type: vk::StructureType::ImageViewCreateInfo,
                    p_next: ptr::null(),
                    flags: Default::default(),
                    image: mesh.0.base_color_image.image,
                    view_type: vk::ImageViewType::Type2d,
                    format: vk::Format::R8g8b8a8Unorm,
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::Identity,
                        g: vk::ComponentSwizzle::Identity,
                        b: vk::ComponentSwizzle::Identity,
                        a: vk::ComponentSwizzle::Identity,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::IMAGE_ASPECT_COLOR_BIT,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                };
                textures.push((
                    mesh.0.texture_sampler,
                    base.device.create_image_view(&create_info, None).unwrap(),
                ));
            }
            let descriptor_set = render_dag
                .descriptor_sets
                .get("main_descriptor_layout")
                .unwrap();
            unsafe {
                base.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet {
                            s_type: vk::StructureType::WriteDescriptorSet,
                            p_next: ptr::null(),
                            dst_set: descriptor_set.clone(),
                            dst_binding: 1,
                            dst_array_element: 0,
                            descriptor_count: 1,
                            descriptor_type: vk::DescriptorType::CombinedImageSampler,
                            p_image_info: &vk::DescriptorImageInfo {
                                sampler: textures[0].0,
                                image_view: textures[0].1,
                                image_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
                            },
                            p_buffer_info: ptr::null(),
                            p_texel_buffer_view: ptr::null(),
                        },
                    ],
                    &[],
                );
            }
        }

        base.render_loop(&mut || {
            {
                let projection = cgmath::perspective(
                    cgmath::Deg(60.0),
                    base.surface_resolution.width as f32 / base.surface_resolution.height as f32,
                    0.1,
                    100.0,
                );
                let view = cgmath::Matrix4::look_at(
                    cgmath::Point3::new(4.0, 1.5, 3.0),
                    cgmath::Point3::new(0.0, 0.0, 0.0),
                    cgmath::vec3(0.0, 1.0, 0.0),
                );
                let mut dispatcher = specs::DispatcherBuilder::new()
                    .add(SteadyRotation, "steady_rotation", &[])
                    .add(
                        MVPCalculation { projection, view },
                        "mvp",
                        &["steady_rotation"],
                    )
                    .build();

                dispatcher.dispatch(&mut world.res);
            }
            {
                use specs::Join;
                let mut mvps = vec![];
                for mvp in world.read::<MVP>().join() {
                    mvps.push(mvp.clone());
                }

                let ubo_buffer = buffer::Buffer::upload_from::<MVP, _>(
                    &base,
                    vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    &mvps.iter().cloned(),
                );
                let ubo_mvp = render_dag
                    .descriptor_sets
                    .get("main_descriptor_layout")
                    .unwrap();
                unsafe {
                    base.device.update_descriptor_sets(
                        &[
                            vk::WriteDescriptorSet {
                                s_type: vk::StructureType::WriteDescriptorSet,
                                p_next: ptr::null(),
                                dst_set: ubo_mvp.clone(),
                                dst_binding: 0,
                                dst_array_element: 0,
                                descriptor_count: mvps.len() as u32,
                                descriptor_type: vk::DescriptorType::UniformBuffer,
                                p_image_info: ptr::null(),
                                p_buffer_info: &vk::DescriptorBufferInfo {
                                    buffer: ubo_buffer.buffer(),
                                    offset: 0,
                                    range: (mvps.len() * mem::size_of::<MVP>()) as u64,
                                },
                                p_texel_buffer_view: ptr::null(),
                            },
                        ],
                        &[],
                    );
                }
            }

            let present_index = base.swapchain_loader
                .acquire_next_image_khr(
                    base.swapchain,
                    std::u64::MAX,
                    base.present_complete_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();
            let framebuffer = framebuffers[present_index as usize];
            record_submit_commandbuffer(base.device.vk(),
                                        base.draw_command_buffer,
                                        base.present_queue,
                                        &[vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
                                        &[base.present_complete_semaphore],
                                        &[base.rendering_complete_semaphore],
                                        |device, draw_command_buffer| {
                render_dag.run(&base, &world, framebuffer, draw_command_buffer);
            });
            //let mut present_info_err = mem::uninitialized();
            let present_info = vk::PresentInfoKHR {
                s_type: vk::StructureType::PresentInfoKhr,
                p_next: ptr::null(),
                wait_semaphore_count: 1,
                p_wait_semaphores: &base.rendering_complete_semaphore,
                swapchain_count: 1,
                p_swapchains: &base.swapchain,
                p_image_indices: &present_index,
                p_results: ptr::null_mut(),
            };
            base.swapchain_loader
                .queue_present_khr(base.present_queue, &present_info)
                .unwrap();
        });

        base.device.device_wait_idle().unwrap();
        for framebuffer in framebuffers {
            base.device.destroy_framebuffer(framebuffer, None);
        }
    }
}
