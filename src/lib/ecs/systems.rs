use super::{super::alloc, components::*};
use ash::{version::DeviceV1_0, vk};
use cgmath;
use futures::{
    executor::{self, block_on, spawn}, future::lazy,
};
use specs::prelude::*;
use std::{cmp::min, mem::size_of, ptr, slice::from_raw_parts, sync::Arc, u64};

use super::super::helpers;

pub struct SteadyRotation;

impl<'a> System<'a> for SteadyRotation {
    type SystemData = (WriteStorage<'a, Rotation>);

    fn run(&mut self, mut rotations: Self::SystemData) {
        use cgmath::Rotation3;
        let incremental = cgmath::Quaternion::from_angle_y(cgmath::Deg(1.0));
        for rot in (&mut rotations).join() {
            *rot = Rotation(incremental * rot.0);
        }
    }
}

pub struct MVPCalculation {
    pub projection: cgmath::Matrix4<f32>,
    pub view: cgmath::Matrix4<f32>,
}

impl<'a> System<'a> for MVPCalculation {
    type SystemData = (
        ReadStorage<'a, Position>,
        ReadStorage<'a, Rotation>,
        ReadStorage<'a, Scale>,
        WriteStorage<'a, Matrices>,
    );

    fn run(&mut self, (positions, rotations, scales, mut mvps): Self::SystemData) {
        for (pos, rot, scale, mvp) in (&positions, &rotations, &scales, &mut mvps).join() {
            mvp.model = cgmath::Matrix4::from_translation(pos.0)
                * cgmath::Matrix4::from(rot.0)
                * cgmath::Matrix4::from_scale(scale.0);

            mvp.mvp = self.projection * self.view * mvp.model;
            mvp.mv = self.view * mvp.model;
        }
    }
}

pub struct MVPUpload {
    pub dst_mvp: Arc<helpers::Buffer>,
    pub dst_mv: Arc<helpers::Buffer>,
    pub dst_model: Arc<helpers::Buffer>,
}

unsafe impl Send for MVPUpload {}

impl<'a> System<'a> for MVPUpload {
    type SystemData = (Entities<'a>, ReadStorage<'a, Matrices>);

    fn run(&mut self, (entities, matrices): Self::SystemData) {
        (&*entities, &matrices)
            .par_join()
            .for_each(|(entity, matrices)| {
                // println!("Writing at {:?} contents {:?}", entity.id(), matrices.mvp);
                use std::slice;
                let out_mvp = unsafe {
                    slice::from_raw_parts_mut(
                        self.dst_mvp.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>,
                        1024,
                    )
                };
                let out_mv = unsafe {
                    slice::from_raw_parts_mut(
                        self.dst_mv.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>,
                        1024,
                    )
                };
                let out_model = unsafe {
                    slice::from_raw_parts_mut(
                        self.dst_model.allocation_info.pMappedData as *mut cgmath::Matrix4<f32>,
                        1024,
                    )
                };
                out_mvp[entity.id() as usize] = matrices.mvp;
                out_mv[entity.id() as usize] = matrices.mv;
                out_model[entity.id() as usize] = matrices.model;
            });
    }
}

pub struct AssignBufferIndex;

impl<'a> System<'a> for AssignBufferIndex {
    type SystemData = (
        ReadStorage<'a, GltfMesh>,
        WriteStorage<'a, GltfMeshBufferIndex>,
    );

    fn run(&mut self, (meshes, mut indices): Self::SystemData) {
        for (ix, (_mesh, buffer_index)) in (&meshes, &mut indices).join().enumerate() {
            buffer_index.0 = ix as u32;
        }
    }
}

pub struct RenderFrame {
    pub threadpool: executor::ThreadPool,
    pub instance: Arc<helpers::Instance>,
    pub device: Arc<helpers::Device>,
    pub swapchain: Arc<helpers::Swapchain>,
    pub framebuffer: Arc<helpers::Framebuffer>,
    pub present_semaphore: Arc<helpers::Semaphore>,
    pub rendering_complete_semaphore: Arc<helpers::Semaphore>,
    pub graphics_command_pool: Arc<helpers::CommandPool>,
    pub renderpass: Arc<helpers::RenderPass>,
    pub gltf_pipeline: Arc<helpers::Pipeline>,
    pub gltf_pipeline_layout: Arc<helpers::PipelineLayout>,
    pub ubo_set: Arc<helpers::DescriptorSet>,
    pub model_set: Arc<helpers::DescriptorSet>,
    pub culled_commands_buffer: Arc<helpers::Buffer>,
    pub culled_index_buffer: Arc<helpers::Buffer>,
    pub cull_pipeline: Arc<helpers::Pipeline>,
    pub cull_pipeline_layout: Arc<helpers::PipelineLayout>,
    pub cull_set: Arc<helpers::DescriptorSet>,
}

unsafe impl Send for RenderFrame {}

impl<'a> System<'a> for RenderFrame {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, GltfMesh>,
        ReadStorage<'a, GltfMeshBufferIndex>,
    );

    fn run(&mut self, (entities, meshes, mesh_indices): Self::SystemData) {
        let image_index = unsafe {
            self.swapchain
                .handle
                .ext
                .acquire_next_image_khr(
                    self.swapchain.handle.swapchain,
                    u64::MAX,
                    self.present_semaphore.handle,
                    vk::Fence::null(),
                )
                .unwrap()
        };
        let command_buffer_future = helpers::record_one_time_cb(
            Arc::clone(&self.graphics_command_pool),
            {
                let main_renderpass = Arc::clone(&self.renderpass);
                let framebuffer = Arc::clone(&self.framebuffer);
                let instance = Arc::clone(&self.instance);
                let device = Arc::clone(&self.device);
                let ubo_set = Arc::clone(&self.ubo_set);
                let model_set = Arc::clone(&self.model_set);
                let gltf_pipeline = Arc::clone(&self.gltf_pipeline);
                let gltf_pipeline_layout = Arc::clone(&self.gltf_pipeline_layout);
                let cull_set = Arc::clone(&self.cull_set);
                let culled_index_buffer = Arc::clone(&self.culled_index_buffer);
                let culled_commands_buffer = Arc::clone(&self.culled_commands_buffer);
                let cull_pipeline = Arc::clone(&self.cull_pipeline);
                let cull_pipeline_layout = Arc::clone(&self.cull_pipeline_layout);
                move |command_buffer| unsafe {
                    device.device.debug_marker_around(
                        command_buffer,
                        "cull pass",
                        [0.0, 1.0, 0.0, 1.0],
                        || {
                            device.device.cmd_bind_pipeline(
                                command_buffer,
                                vk::PipelineBindPoint::Compute,
                                cull_pipeline.handle,
                            );
                            device.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::Compute,
                                cull_pipeline_layout.handle,
                                0,
                                &[ubo_set.handle, cull_set.handle],
                                &[],
                            );
                            device.device.cmd_fill_buffer(
                                command_buffer,
                                culled_commands_buffer.handle,
                                0,
                                size_of::<u32>() as vk::DeviceSize
                                    + size_of::<u32>() as vk::DeviceSize * 5 * 9,
                                0,
                            );
                            for (entity, mesh, mesh_index) in
                                (&*entities, &meshes, &mesh_indices).join()
                            {
                                let constants = [entity.id() as u32, mesh_index.0, 0, 0, 0];

                                let casted: &[u8] =
                                    { from_raw_parts(constants.as_ptr() as *const u8, 4) };
                                device.device.cmd_push_constants(
                                    command_buffer,
                                    gltf_pipeline_layout.handle,
                                    vk::SHADER_STAGE_VERTEX_BIT,
                                    0,
                                    casted,
                                );
                                let index_len = mesh.index_len as u32;
                                let workgroup_size = 256; // TODO: make a specialization constant, not hardcoded
                                let workgroup_count = index_len / 3 / workgroup_size
                                    + min(1, index_len / 3 % workgroup_size);
                                device
                                    .device
                                    .cmd_dispatch(command_buffer, workgroup_count, 1, 1);
                            }
                        },
                    );

                    let clear_values = &[
                        vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
                        },
                        vk::ClearValue {
                            depth: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ];
                    let begin_info = vk::RenderPassBeginInfo {
                        s_type: vk::StructureType::RenderPassBeginInfo,
                        p_next: ptr::null(),
                        render_pass: main_renderpass.handle,
                        framebuffer: framebuffer.handles[image_index as usize],
                        render_area: vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: instance.window_width,
                                height: instance.window_height,
                            },
                        },
                        clear_value_count: clear_values.len() as u32,
                        p_clear_values: clear_values.as_ptr(),
                    };

                    device.device.debug_marker_around(
                        command_buffer,
                        "main renderpass",
                        [0.0, 0.0, 1.0, 1.0],
                        || {
                            device.device.cmd_begin_render_pass(
                                command_buffer,
                                &begin_info,
                                vk::SubpassContents::Inline,
                            );
                            device.device.debug_marker_around(
                                command_buffer,
                                "gltf meshes",
                                [1.0, 0.0, 0.0, 1.0],
                                || {
                                    // gltf mesh
                                    device.device.cmd_bind_pipeline(
                                        command_buffer,
                                        vk::PipelineBindPoint::Graphics,
                                        gltf_pipeline.handle,
                                    );
                                    device.device.cmd_bind_descriptor_sets(
                                        command_buffer,
                                        vk::PipelineBindPoint::Graphics,
                                        gltf_pipeline_layout.handle,
                                        0,
                                        &[ubo_set.handle, model_set.handle],
                                        &[],
                                    );
                                    device.device.cmd_bind_index_buffer(
                                        command_buffer,
                                        culled_index_buffer.handle,
                                        size_of::<u32>() as vk::DeviceSize,
                                        vk::IndexType::Uint32,
                                    );
                                    let first_entity = (&*entities).join().next().unwrap();
                                    let mesh = meshes.get(first_entity).unwrap();
                                    device.device.cmd_bind_vertex_buffers(
                                        command_buffer,
                                        0,
                                        &[mesh.vertex_buffer.handle, mesh.normal_buffer.handle],
                                        &[0, 0],
                                    );
                                    device.device.cmd_draw_indexed_indirect(
                                        command_buffer,
                                        culled_commands_buffer.handle,
                                        0,
                                        9, // TODO: find max of GltfMeshBufferIndex
                                        size_of::<u32>() as u32 * 5,
                                    );
                                },
                            );
                            device.device.cmd_end_render_pass(command_buffer);
                        },
                    );
                    device.device.debug_marker_end(command_buffer);
                }
            },
        );
        let command_buffer = block_on(command_buffer_future).unwrap();
        unsafe {
            let wait_semaphores = &[self.present_semaphore.handle];
            let signal_semaphores = &[self.rendering_complete_semaphore.handle];
            let dst_stage_masks = vec![vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT; wait_semaphores.len()];
            let submits = [vk::SubmitInfo {
                s_type: vk::StructureType::SubmitInfo,
                p_next: ptr::null(),
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: dst_stage_masks.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &command_buffer.handle,
                signal_semaphore_count: signal_semaphores.len() as u32,
                p_signal_semaphores: signal_semaphores.as_ptr(),
            }];
            let queue = self
                .device
                .graphics_queue
                .lock()
                .expect("can't lock the submit queue");

            let submit_fence = helpers::new_fence(Arc::clone(&self.device));

            self.device
                .device
                .queue_submit(*queue, &submits, submit_fence.handle)
                .unwrap();

            {
                let device = Arc::clone(&self.device);
                self.threadpool
                    .run(lazy(move |_| {
                        spawn(lazy(move |_| {
                            // println!("dtor previous frame");
                            device
                                .device
                                .wait_for_fences(&[submit_fence.handle], true, u64::MAX)
                                .expect("Wait for fence failed.");
                            drop(command_buffer);
                            drop(submit_fence);
                            Ok(())
                        }))
                    }))
                    .unwrap();
            }

            {
                let wait_semaphores = &[self.rendering_complete_semaphore.handle];
                let present_info = vk::PresentInfoKHR {
                    s_type: vk::StructureType::PresentInfoKhr,
                    p_next: ptr::null(),
                    wait_semaphore_count: wait_semaphores.len() as u32,
                    p_wait_semaphores: wait_semaphores.as_ptr(),
                    swapchain_count: 1,
                    p_swapchains: &self.swapchain.handle.swapchain,
                    p_image_indices: &image_index,
                    p_results: ptr::null_mut(),
                };

                self.swapchain
                    .handle
                    .ext
                    .queue_present_khr(*queue, &present_info)
                    .unwrap();
            }
        }
    }
}
