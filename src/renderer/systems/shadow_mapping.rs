use crate::{
    ecs::components::*,
    renderer::{graphics as graphics_sync, *},
    timeline_value,
};
use ash::vk;

const MAP_SIZE: u32 = 4096;
// dimensions of the square texture, 4x4 slots = 16 in total
const DIM: u32 = 4;

pub struct ShadowMappingData {
    depth_pipeline: Pipeline,
    renderpass: RenderPass,
    depth_image: Image,
    _depth_image_view: ImageView,
    framebuffer: Framebuffer,
    previous_command_buffer: DoubleBuffered<Option<CommandBuffer>>,
    image_transitioned: bool,
    pub user_set_layout: super::super::shaders::shadow_map_set::DescriptorSetLayout,
    pub user_set: DoubleBuffered<super::super::shaders::shadow_map_set::DescriptorSet>,
    _user_sampler: helpers::Sampler,
}

impl ShadowMappingData {
    pub fn new(
        renderer: &RenderFrame,
        depth_pass_data: &DepthPassData,
        main_descriptor_pool: &mut MainDescriptorPool,
    ) -> ShadowMappingData {
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
                        .initial_layout(vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)
                        .final_layout(vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)]
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
                .vertex_input_state(&super::super::shaders::depth_pipe::vertex_input_state())
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::builder()
                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                )
                .dynamic_state(
                    &vk::PipelineDynamicStateCreateInfo::builder()
                        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
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
                .layout(depth_pass_data.depth_pipeline_layout.layout.handle)
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
                height: MAP_SIZE * DIM,
                width: MAP_SIZE * DIM,
                depth: 1,
            },
            vk::SampleCountFlags::TYPE_1,
            vk::ImageTiling::OPTIMAL,
            vk::ImageLayout::PREINITIALIZED,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
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
                        .width(MAP_SIZE * DIM)
                        .height(MAP_SIZE * DIM)
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

        let previous_command_buffer = renderer.new_buffered(|_| None);

        let user_set_layout =
            super::super::shaders::shadow_map_set::DescriptorSetLayout::new(&renderer.device);
        renderer.device.set_object_name(
            user_set_layout.layout.handle,
            "Shadow Mapping User Descriptor Set Layout",
        );

        let user_sampler = helpers::new_sampler(
            renderer.device.clone(),
            &vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .compare_enable(true)
                .compare_op(vk::CompareOp::LESS_OR_EQUAL),
        );

        let user_set = renderer.new_buffered(|ix| {
            let s = super::super::shaders::shadow_map_set::DescriptorSet::new(
                &main_descriptor_pool,
                &user_set_layout,
            );
            renderer.device.set_object_name(
                s.set.handle,
                &format!("Shadow Mapping User Descriptor Set - {}", ix),
            );

            {
                let sampler_updates = &[vk::DescriptorImageInfo::builder()
                    .image_view(depth_image_view.handle)
                    .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)
                    .sampler(user_sampler.handle)
                    .build()];
                unsafe {
                    renderer.device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::builder()
                            .dst_set(s.set.handle)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(sampler_updates)
                            .build()],
                        &[],
                    );
                }
            }

            s
        });

        ShadowMappingData {
            renderpass,
            depth_image,
            _depth_image_view: depth_image_view,
            depth_pipeline,
            framebuffer,
            previous_command_buffer,
            image_transitioned: false,
            user_set_layout,
            user_set,
            _user_sampler: user_sampler,
        }
    }
}

/// Holds Projection and view matrices for each light.
pub struct ShadowMappingLightMatrices {
    matrices_set: DoubleBuffered<super::super::shaders::camera_set::DescriptorSet>,
    matrices_buffer: DoubleBuffered<Buffer>,
}

impl ShadowMappingLightMatrices {
    pub(crate) fn new(
        renderer: &RenderFrame,
        main_descriptor_pool: &MainDescriptorPool,
        camera_matrices: &CameraMatrices,
        entity_ix: u32,
    ) -> ShadowMappingLightMatrices {
        let matrices_buffer = renderer.new_buffered(|ix| {
            let b = renderer.device.new_buffer(
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                alloc::VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                size_of::<LightMatrices>() as vk::DeviceSize,
            );
            renderer.device.set_object_name(
                b.handle,
                &format!(
                    "Shadow Mapping Light matrices Buffer - entity={:?} ix={}",
                    entity_ix, ix
                ),
            );
            b
        });
        let matrices_set = renderer.new_buffered(|ix| {
            let s = super::super::shaders::camera_set::DescriptorSet::new(
                &main_descriptor_pool,
                &camera_matrices.set_layout,
            );
            renderer.device.set_object_name(
                s.set.handle,
                &format!("Shadow Mapping MVP Set - entity={:?} ix={}", entity_ix, ix),
            );
            s.update_whole_buffer(&renderer, 0, &matrices_buffer.current(ix));
            s
        });

        ShadowMappingLightMatrices {
            matrices_buffer,
            matrices_set,
        }
    }
}

#[repr(C)]
#[allow(unused_variables)]
struct LightMatrices {
    pub projection: glm::Mat4,
    pub view: glm::Mat4,
    pub position: glm::Vec4,
}

pub struct ShadowMappingMVPCalculation;

impl ShadowMappingMVPCalculation {
    pub fn exec_system() -> Box<(dyn legion::systems::schedule::Schedulable + 'static)> {
        use legion::prelude::*;
        SystemBuilder::<()>::new("ShadowMappingMVPCalculation - exec")
            .read_resource::<ImageIndex>()
            .with_query(<(
                Read<Position>,
                Read<Rotation>,
                Write<ShadowMappingLightMatrices>,
            )>::query())
            .build(
                move |_commands, mut world, ref image_index, ref mut query| {
                    debug_assert_eq!(size_of::<LightMatrices>(), 144);
                    #[cfg(feature = "profiling")]
                    microprofile::scope!("ecs", "shadow mapping light matrices calculation");
                    for (ref light_position, ref light_rotation, ref mut light_matrix) in
                        query.iter_mut(&mut world)
                    {
                        let near = 10.0;
                        let far = 400.0;
                        let projection = glm::perspective_lh_zo(
                            1.0,
                            glm::radians(&glm::vec1(70.0)).x,
                            near,
                            far,
                        );

                        let view =
                            glm::translation(&(light_rotation.0 * (-light_position.0.coords)))
                                * light_rotation.0.to_homogeneous();
                        let mut matrices_mapped = light_matrix
                            .matrices_buffer
                            .current_mut(image_index.0)
                            .map::<LightMatrices>()
                            .expect("failed to map Light matrices buffer");
                        matrices_mapped[0] = LightMatrices {
                            projection,
                            view,
                            position: light_position.0.coords.push(1.0),
                        };
                    }
                },
            )
    }
}

/// Identifies stale shadow maps in the atlas and refreshes them
pub struct PrepareShadowMaps;

impl PrepareShadowMaps {
    pub fn exec_system() -> Box<(dyn legion::systems::schedule::Schedulable + 'static)> {
        use legion::prelude::*;
        SystemBuilder::<()>::new("PrepareShadowMaps - exec")
            .read_resource::<RenderFrame>()
            .read_resource::<DepthPassData>()
            .read_resource::<ImageIndex>()
            .write_resource::<GraphicsCommandPool>()
            .write_resource::<ShadowMappingData>()
            .read_resource::<ModelData>()
            .with_query(<(Read<DrawIndex>, Read<GltfMesh>)>::query())
            .with_query(<Read<ShadowMappingLightMatrices>>::query().filter(component::<Light>()))
            .build(move |_commands, world, resources, queries| {
                let (
                    ref renderer,
                    ref depth_pass,
                    ref image_index,
                    ref mut graphics_command_pool,
                    ref mut shadow_mapping,
                    ref model_data,
                ) = resources;
                let (ref mut mesh_query, ref mut shadow_query) = queries;
                #[cfg(feature = "profiling")]
                microprofile::scope!("ecs", "shadow_mapping");
                let command_buffer = graphics_command_pool.0.record_one_time("shadow mapping cb");
                unsafe {
                    if !shadow_mapping.image_transitioned {
                        renderer.device.cmd_pipeline_barrier(
                            *command_buffer,
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
                                .new_layout(
                                    vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
                                )
                                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .build()],
                        );
                        shadow_mapping.image_transitioned = true;
                    }
                    let _shadow_mapping_marker = renderer.device.debug_marker_around2(
                        &command_buffer,
                        "shadow mapping",
                        [0.8, 0.1, 0.1, 1.0],
                    );
                    renderer.device.cmd_begin_render_pass(
                        *command_buffer,
                        &vk::RenderPassBeginInfo::builder()
                            .render_pass(shadow_mapping.renderpass.handle)
                            .framebuffer(shadow_mapping.framebuffer.handle)
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: vk::Extent2D {
                                    width: DIM * MAP_SIZE,
                                    height: DIM * MAP_SIZE,
                                },
                            }),
                        vk::SubpassContents::INLINE,
                    );
                    renderer.device.cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        shadow_mapping.depth_pipeline.handle,
                    );

                    for (ix, shadow_mvp) in shadow_query.iter(&world).enumerate() {
                        depth_pass.depth_pipeline_layout.bind_descriptor_sets(
                            &renderer.device,
                            *command_buffer,
                            &model_data.model_set.current(image_index.0),
                            &shadow_mvp.matrices_set.current(image_index.0),
                        );

                        let row = ix as u32 / DIM;
                        let column = ix as u32 % DIM;
                        let x = MAP_SIZE * column;
                        let y = MAP_SIZE * row;
                        renderer.device.cmd_set_viewport(
                            *command_buffer,
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
                            *command_buffer,
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
                            *command_buffer,
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

                        for (draw_index, mesh) in mesh_query.iter(&world) {
                            let (index_buffer, index_count) = mesh.index_buffers.last().unwrap();
                            renderer.device.cmd_bind_index_buffer(
                                *command_buffer,
                                index_buffer.handle,
                                0,
                                vk::IndexType::UINT32,
                            );
                            renderer.device.cmd_bind_vertex_buffers(
                                *command_buffer,
                                0,
                                &[mesh.vertex_buffer.handle],
                                &[0],
                            );
                            renderer.device.cmd_draw_indexed(
                                *command_buffer,
                                (*index_count).try_into().unwrap(),
                                1,
                                0,
                                0,
                                draw_index.0,
                            );
                        }
                    }

                    renderer.device.cmd_end_render_pass(*command_buffer);
                }
                let command_buffer = command_buffer.end();

                let queue = renderer.device.graphics_queue.lock();

                let command_buffers = &[*command_buffer];
                let wait_semaphores = &[renderer.graphics_timeline_semaphore.handle];
                let wait_dst_stage_mask = &[vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS];
                let wait_semaphore_values =
                    &[timeline_value!(graphics_sync @ last renderer.frame_number => GUI_DRAW)];
                let signal_semaphore_values =
                    &[timeline_value!(graphics_sync @ renderer.frame_number => SHADOW_MAPPING)];
                let mut signal_timeline = vk::TimelineSemaphoreSubmitInfo::builder()
                    .wait_semaphore_values(wait_semaphore_values)
                    .signal_semaphore_values(signal_semaphore_values)
                    .build();
                let signal_semaphores = &[renderer.graphics_timeline_semaphore.handle];
                let submit_info = vk::SubmitInfo::builder()
                    .command_buffers(command_buffers)
                    .push_next(&mut signal_timeline)
                    .wait_semaphores(wait_semaphores)
                    .wait_dst_stage_mask(wait_dst_stage_mask)
                    .signal_semaphores(signal_semaphores)
                    .build();

                unsafe {
                    renderer
                        .device
                        .queue_submit(*queue, &[submit_info], vk::Fence::null())
                        .unwrap();
                }

                *shadow_mapping
                    .previous_command_buffer
                    .current_mut(image_index.0) = Some(command_buffer);

                // TODO: extract to another stage?
                // Update descriptor sets so that users of lights have the latest info
                // preallocate all equired memory so as not to invalidate references during iteration
                let mut mvp_updates =
                    vec![[vk::DescriptorBufferInfo::default(); 1]; DIM as usize * DIM as usize];
                let mut write_descriptors = vec![];
                for (ix, shadow_mvp) in shadow_query.iter(&world).enumerate() {
                    mvp_updates[ix] = [vk::DescriptorBufferInfo {
                        buffer: shadow_mvp.matrices_buffer.current(image_index.0).handle,
                        offset: 0,
                        range: size_of::<LightMatrices>() as vk::DeviceSize,
                    }];
                    write_descriptors.push(
                        vk::WriteDescriptorSet::builder()
                            .dst_set(shadow_mapping.user_set.current(image_index.0).set.handle)
                            .dst_binding(0)
                            .dst_array_element(ix as u32)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&mvp_updates[ix])
                            .build(),
                    );
                }

                unsafe {
                    renderer
                        .device
                        .update_descriptor_sets(&write_descriptors, &[]);
                }
            })
    }
}
