use crate::ecs::components::*;
use crate::renderer::*;
use ash::vk;
use specs::{prelude::*, *};

const MAP_SIZE: u32 = 2048;

pub struct ShadowMappingData {
    depth_pipeline: Pipeline,
    renderpass: RenderPass,
    depth_image: Image,
    depth_image_view: ImageView,
    framebuffer: Framebuffer,
    pub complete_semaphore: DoubleBuffered<Semaphore>,
    previous_command_buffer: DoubleBuffered<Option<CommandBuffer>>,
    image_transitioned: bool,
}

impl shred::SetupHandler<ShadowMappingData> for ShadowMappingData {
    fn setup(world: &mut World) {
        if world.has_value::<ShadowMappingData>() {
            return;
        }

        let result = world.exec(
            |(renderer, depth_pass_data): (
                ReadExpect<RenderFrame>,
                Read<DepthPassData, DepthPassData>,
            )| {
                let renderpass = renderer.device.new_renderpass(
                    &vk::RenderPassCreateInfo::builder()
                        .attachments(unsafe {
                            &*(&[vk::AttachmentDescription::builder()
                                .format(vk::Format::D32_SFLOAT)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(
                                    vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
                                )
                                .final_layout(
                                    vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
                                )]
                                as *const [vk::AttachmentDescriptionBuilder<'_>; 1]
                                as *const [vk::AttachmentDescription; 1])
                        })
                        .subpasses(unsafe {
                            &*(&[vk::SubpassDescription::builder()
                                .depth_stencil_attachment(&vk::AttachmentReference {
                                    attachment: 0,
                                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                })
                                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)]
                                as *const [vk::SubpassDescriptionBuilder<'_>; 1]
                                as *const [vk::SubpassDescription; 1])
                        })
                        .dependencies(unsafe {
                            &*(&[vk::SubpassDependency::builder()
                                .src_subpass(vk::SUBPASS_EXTERNAL)
                                .dst_subpass(0)
                                .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
                                .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)]
                                as *const [vk::SubpassDependencyBuilder<'_>; 1]
                                as *const [vk::SubpassDependency; 1])
                        }),
                );

                let depth_pipeline = new_graphics_pipeline2(
                    Arc::clone(&renderer.device),
                    &[(
                        vk::ShaderStageFlags::VERTEX,
                        PathBuf::from(env!("OUT_DIR")).join("depth_prepass.vert.spv"),
                    )],
                    vk::GraphicsPipelineCreateInfo::builder()
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::builder()
                                .vertex_attribute_descriptions(&[
                                    vk::VertexInputAttributeDescription {
                                        location: 0,
                                        binding: 0,
                                        format: vk::Format::R32G32B32_SFLOAT,
                                        offset: 0,
                                    },
                                ])
                                .vertex_binding_descriptions(&[
                                    vk::VertexInputBindingDescription {
                                        binding: 0,
                                        stride: size_of::<f32>() as u32 * 3,
                                        input_rate: vk::VertexInputRate::VERTEX,
                                    },
                                ]),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::builder()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::builder()
                                .viewport_count(1)
                                .scissor_count(1)
                                .build(),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::builder()
                                .cull_mode(vk::CullModeFlags::BACK)
                                .front_face(vk::FrontFace::CLOCKWISE)
                                .line_width(1.0)
                                .polygon_mode(vk::PolygonMode::FILL)
                                .build(),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::builder()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                                .build(),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::builder()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                                .depth_bounds_test_enable(false)
                                .max_depth_bounds(1.0)
                                .min_depth_bounds(0.0)
                                .build(),
                        )
                        .layout(depth_pass_data.depth_pipeline_layout.handle)
                        .render_pass(renderpass.handle)
                        .subpass(0)
                        .build(),
                );
                renderer
                    .device
                    .set_object_name(depth_pipeline.handle, "Shadow mapping depth Pipeline");

                let depth_image = renderer.device.new_image(
                    vk::Format::D32_SFLOAT,
                    // 16 slots in total
                    vk::Extent3D {
                        height: MAP_SIZE * 4,
                        width: MAP_SIZE * 4,
                        depth: 1,
                    },
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageLayout::PREINITIALIZED,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_DST,
                    alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                );
                renderer
                    .device
                    .set_object_name(depth_image.handle, "Shadow mapping depth image");

                let depth_image_view = unsafe {
                    let handle = renderer
                        .device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::builder()
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(vk::Format::D32_SFLOAT)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                })
                                .image(depth_image.handle),
                            None,
                        )
                        .unwrap();

                    ImageView {
                        handle,
                        device: Arc::clone(&renderer.device),
                    }
                };

                let framebuffer = unsafe {
                    let handle = renderer
                        .device
                        .create_framebuffer(
                            &vk::FramebufferCreateInfo::builder()
                                .render_pass(renderpass.handle)
                                .attachments(&[depth_image_view.handle])
                                .width(MAP_SIZE * 4)
                                .height(MAP_SIZE * 4)
                                .layers(1),
                            None,
                        )
                        .unwrap();

                    Framebuffer {
                        handle,
                        device: Arc::clone(&renderer.device),
                    }
                };
                renderer
                    .device
                    .set_object_name(framebuffer.handle, "Shadow mapping framebuffer");

                let complete_semaphore = renderer.new_buffered(|ix| {
                    let s = renderer.device.new_semaphore();
                    renderer.device.set_object_name(
                        s.handle,
                        &format!("Shadow mapping complete semaphore - {}", ix),
                    );
                    s
                });

                let previous_command_buffer = renderer.new_buffered(|_| None);

                ShadowMappingData {
                    renderpass,
                    depth_image,
                    depth_image_view,
                    depth_pipeline,
                    framebuffer,
                    complete_semaphore,
                    previous_command_buffer,
                    image_transitioned: false,
                }
            },
        );

        world.insert(result);
    }
}

#[derive(Component)]
#[storage(VecStorage)]
/// Holds MVP data for meshes from the perspective of this light.
/// This differs from the main `MVPData` component by the origin and projection matrix setup.
pub struct ShadowMappingMVPData {
    mvp_set: DoubleBuffered<DescriptorSet>,
    mvp_buffer: DoubleBuffered<Buffer>,
}

pub struct ShadowMappingMVPCalculation;

impl<'a> System<'a> for ShadowMappingMVPCalculation {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Rotation>,
        ReadStorage<'a, Scale>,
        WriteStorage<'a, ShadowMappingMVPData>,
        ReadStorage<'a, Light>,
        Read<'a, ImageIndex>,
        ReadExpect<'a, RenderFrame>,
        Read<'a, MainDescriptorPool, MainDescriptorPool>,
        Read<'a, MVPData, MVPData>,
    );

    fn run(
        &mut self,
        (
            entities,
            positions,
            rotations,
            scales,
            mut mvps,
            lights,
            image_index,
            renderer,
            main_descriptor_pool,
            mvp_data,
        ): Self::SystemData,
    ) {
        microprofile::scope!("ecs", "shadow mapping mvp calculation");
        let mut entities_to_update = vec![];
        for (entity_id, _, ()) in (&*entities, &positions, !&mvps).join() {
            entities_to_update.push(entity_id);
        }
        for entity_id in entities_to_update.into_iter() {
            let mvp_buffer = DoubleBuffered::new(|ix| {
                let b = renderer.device.new_buffer(
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    4 * 4 * 4 * 4096,
                );
                renderer.device.set_object_name(
                    b.handle,
                    &format!(
                        "Shadow Mapping MVP Buffer - entity={:?} ix={}",
                        entity_id, ix
                    ),
                );
                b
            });
            let mvp_set = DoubleBuffered::new(|ix| {
                let s = main_descriptor_pool
                    .0
                    .allocate_set(&mvp_data.mvp_set_layout);
                renderer.device.set_object_name(
                    s.handle,
                    &format!("Shadow Mapping MVP Set - entity={:?} ix={}", entity_id, ix),
                );

                {
                    let mvp_updates = &[vk::DescriptorBufferInfo {
                        buffer: mvp_buffer.current(ix).handle,
                        offset: 0,
                        range: 4096 * size_of::<na::Matrix4<f32>>() as vk::DeviceSize,
                    }];
                    unsafe {
                        renderer.device.update_descriptor_sets(
                            &[vk::WriteDescriptorSet::builder()
                                .dst_set(s.handle)
                                .dst_binding(0)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(mvp_updates)
                                .build()],
                            &[],
                        );
                    }
                }

                s
            });

            mvps.insert(
                entity_id,
                ShadowMappingMVPData {
                    mvp_set,
                    mvp_buffer,
                },
            )
            .expect("failed to insert ShadowMappingMVPData");
        }
        (&lights, &positions, &mut mvps)
            .par_join()
            .for_each(|(_light, light_position, mvp)| {
                let l = na::Point3::new(light_position.0.x, light_position.0.y, light_position.0.z);
                let near = 0.1;
                let far = 100.0;
                let left = -10.0;
                let right = 10.0;
                let top = 10.0;
                let bottom = -10.0;
                let projection = glm::ortho_lh_zo(left, right, bottom, top, near, far);

                let view =
                    na::Isometry3::look_at_lh(&l, &na::Point3::new(0.0, 0.0, 0.0), &up_vector());

                let mut mvp_mapped = mvp
                    .mvp_buffer
                    .current_mut(image_index.0)
                    .map::<na::Matrix4<f32>>()
                    .expect("failed to map MVP buffer");
                for (entity, pos, rot, scale) in
                    (&*entities, &positions, &rotations, &scales).join()
                {
                    let translation = na::Translation3::from(pos.0.coords);
                    let model = na::Similarity3::from_parts(translation, rot.0, scale.0);
                    let mvp = projection * na::Matrix4::<f32>::from(view * model);
                    // println!("test {:?} mvp={:?}", entity, mvp);
                    mvp_mapped[entity.id() as usize] = mvp;
                }
            });
    }
}

/// Identifies stale shadow maps in the atlas and refreshes them
pub struct PrepareShadowMaps;

impl<'a> System<'a> for PrepareShadowMaps {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        Entities<'a>,
        ReadExpect<'a, RenderFrame>,
        Read<'a, DepthPassData, DepthPassData>,
        Read<'a, ImageIndex>,
        Write<'a, GraphicsCommandPool, GraphicsCommandPool>,
        Write<'a, ShadowMappingData, ShadowMappingData>,
        ReadStorage<'a, GltfMesh>,
        ReadStorage<'a, Light>,
        ReadStorage<'a, ShadowMappingMVPData>,
        Read<'a, PresentData, PresentData>,
    );

    fn run(
        &mut self,
        (
            entities,
            renderer,
            depth_pass,
            image_index,
            graphics_command_pool,
            mut shadow_mapping,
            meshes,
            lights,
            shadow_mvps,
            present_data,
        ): Self::SystemData,
    ) {
        microprofile::scope!("ecs", "shadow_mapping");
        let command_buffer = graphics_command_pool
            .0
            .record_one_time(|command_buffer| unsafe {
                if !shadow_mapping.image_transitioned {
                    renderer.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        Default::default(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::builder()
                            .image(shadow_mapping.depth_image.handle)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::DEPTH,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .old_layout(vk::ImageLayout::PREINITIALIZED)
                            .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .build()],
                    );
                    shadow_mapping.image_transitioned = true;
                }
                renderer.device.debug_marker_around(
                    command_buffer,
                    "shadow mapping",
                    [0.8, 0.1, 0.1, 1.0],
                    || {
                        renderer.device.cmd_begin_render_pass(
                            command_buffer,
                            &vk::RenderPassBeginInfo::builder()
                                .render_pass(shadow_mapping.renderpass.handle)
                                .framebuffer(shadow_mapping.framebuffer.handle)
                                .render_area(vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: vk::Extent2D {
                                        width: 4 * MAP_SIZE,
                                        height: 4 * MAP_SIZE,
                                    },
                                }),
                            vk::SubpassContents::INLINE,
                        );
                        renderer.device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            shadow_mapping.depth_pipeline.handle,
                        );

                        for (ix, (_light, shadow_mvp)) in (&lights, &shadow_mvps).join().enumerate()
                        {
                            renderer.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                depth_pass.depth_pipeline_layout.handle,
                                0,
                                &[shadow_mvp.mvp_set.current(image_index.0).handle],
                                &[],
                            );

                            let row = ix as u32 / 4;
                            let column = ix as u32 % 4;
                            let x = MAP_SIZE * column;
                            let y = MAP_SIZE * row;
                            renderer.device.cmd_set_viewport(
                                command_buffer,
                                0,
                                &[vk::Viewport {
                                    x: x as f32,
                                    y: (MAP_SIZE * (row + 1)) as f32,
                                    width: MAP_SIZE as f32,
                                    height: -(MAP_SIZE as f32),
                                    min_depth: 0.0,
                                    max_depth: 1.0,
                                }],
                            );
                            renderer.device.cmd_set_scissor(
                                command_buffer,
                                0,
                                &[vk::Rect2D {
                                    offset: vk::Offset2D {
                                        x: x as i32,
                                        y: y as i32,
                                    },
                                    extent: vk::Extent2D {
                                        width: MAP_SIZE,
                                        height: MAP_SIZE,
                                    },
                                }],
                            );
                            renderer.device.cmd_clear_attachments(
                                command_buffer,
                                &[vk::ClearAttachment::builder()
                                    .clear_value(vk::ClearValue {
                                        depth_stencil: vk::ClearDepthStencilValue {
                                            depth: 1.0,
                                            stencil: 1,
                                        },
                                    })
                                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                    .build()],
                                &[vk::ClearRect::builder()
                                    .rect(vk::Rect2D {
                                        offset: vk::Offset2D {
                                            x: x as i32,
                                            y: y as i32,
                                        },
                                        extent: vk::Extent2D {
                                            width: MAP_SIZE,
                                            height: MAP_SIZE,
                                        },
                                    })
                                    .layer_count(1)
                                    .base_array_layer(0)
                                    .build()],
                            );

                            for (entity, mesh) in (&*entities, &meshes).join() {
                                let (index_buffer, index_count) =
                                    mesh.index_buffers.last().unwrap();
                                renderer.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    index_buffer.handle,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                renderer.device.cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &[mesh.vertex_buffer.handle],
                                    &[0],
                                );
                                renderer.device.cmd_draw_indexed(
                                    command_buffer,
                                    (*index_count).try_into().unwrap(),
                                    1,
                                    0,
                                    0,
                                    entity.id(),
                                );
                            }
                        }

                        renderer.device.cmd_end_render_pass(command_buffer);
                    },
                );
            });

        let queue = renderer.device.graphics_queue.lock();

        unsafe {
            renderer
                .device
                .queue_submit(
                    *queue,
                    &*(&[vk::SubmitInfo::builder()
                        .command_buffers(&[*command_buffer])
                        .wait_semaphores(&[present_data.present_semaphore.handle])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS])
                        .signal_semaphores(&[shadow_mapping
                            .complete_semaphore
                            .current(image_index.0)
                            .handle])]
                        as *const [vk::SubmitInfoBuilder<'_>; 1]
                        as *const [vk::SubmitInfo; 1]),
                    vk::Fence::null(),
                )
                .unwrap();
        }

        *shadow_mapping
            .previous_command_buffer
            .current_mut(image_index.0) = Some(command_buffer);
    }
}
