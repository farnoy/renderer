use std::{hash::Hash, mem::{replace, size_of}, sync::{Arc, Weak}};

use ash::vk::{self};
use bevy_ecs::prelude::*;
use hashbrown::HashMap;
use num_traits::ToPrimitive;
use profiling::scope;
use renderer_vma::VmaMemoryUsage;

#[cfg(feature = "crash_debugging")]
use crate::renderer::CrashBuffer;
use crate::{ecs::components::ModelMatrix, renderer::{BufferType, Device, DrawIndex, GltfMesh, MainDescriptorPool, RenderFrame, RenderStage, SmartSet, SmartSetLayout, Submissions, SwapchainIndexToFrameNumber, acceleration_set, as_of_last, device::{Buffer, DoubleBuffered, StaticBuffer, StrictCommandPool}, frame_graph, helpers::command_util::CommandUtil, systems::present::ImageIndex, update_whole_buffer}};

pub(crate) struct BottomLevelAccelerationStructure {
    buffer: Buffer,
    scratch_buffer: Buffer,
    handle: vk::AccelerationStructureKHR,
}

// TODO: migrate GltfMesh with Arcs inside to Arc<GltfMesh> and change this
//       for now this is a weak ptr to the vertex buffer of GltfMesh
struct MeshHandle(Weak<Buffer>);

renderer_macros::define_resource! { TLAS = AccelerationStructure }

pub(crate) struct AccelerationStructuresInternal {
    command_util: CommandUtil,
    bottom_structures: HashMap<MeshHandle, BottomLevelAccelerationStructure>,
    top_level_buffers: DoubleBuffered<Option<Buffer>>,
    top_level_scratch_buffers: DoubleBuffered<Option<Buffer>>,
    // TODO: This is hack to further defer reclaim of old TLAS resources, as they hang the GPU if freed to early
    // previous_top_level_buffers: DoubleBuffered<Vec<Buffer>>,
    top_level_handles: DoubleBuffered<Option<vk::AccelerationStructureKHR>>,
    instances_buffer: DoubleBuffered<StaticBuffer<[vk::AccelerationStructureInstanceKHR; 4096]>>,
    random_seed: BufferType<acceleration_set::bindings::random_seed>,
}

pub(crate) struct AccelerationStructures {
    pub(crate) set_layout: SmartSetLayout<acceleration_set::Layout>,
    pub(crate) set: SmartSet<acceleration_set::Set>,
}

impl BottomLevelAccelerationStructure {
    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device
                .acceleration_structure
                .destroy_acceleration_structure(self.handle, None);
            self.handle = vk::AccelerationStructureKHR::null();
        }
        self.buffer.destroy(device);
        self.scratch_buffer.destroy(device);
    }
}

impl PartialEq for MeshHandle {
    fn eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}

impl Eq for MeshHandle {}

impl Hash for MeshHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

impl FromWorld for AccelerationStructuresInternal {
    fn from_world(world: &mut World) -> Self {
        world.resource_scope(|world, mut acceleration_structures: Mut<AccelerationStructures>| {
            let renderer = world.get_resource::<RenderFrame>().unwrap();

            let instances_buffer = renderer.new_buffered(|ix| {
                let b = renderer
                    .device
                    .new_static_buffer::<[vk::AccelerationStructureInstanceKHR; 4096]>(
                        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    );
                renderer.device.set_object_name(
                    b.buffer.handle,
                    &format!("Acceleration Structures Internal Instances Buffer - {}", ix),
                );
                b
            });

            let random_seed = renderer.device.new_static_buffer(
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            );

            update_whole_buffer::<acceleration_set::bindings::random_seed>(
                &renderer.device,
                &mut acceleration_structures.set,
                &random_seed,
            );

            AccelerationStructuresInternal {
                command_util: CommandUtil::new(renderer, renderer.device.compute_queue_family),
                bottom_structures: HashMap::new(),
                top_level_buffers: renderer.new_buffered(|_| None),
                top_level_scratch_buffers: renderer.new_buffered(|_| None),
                top_level_handles: renderer.new_buffered(|_| None),
                // previous_top_level_buffers: renderer.new_buffered(|_| vec![]),
                instances_buffer,
                random_seed,
            }
        })
    }
}

impl AccelerationStructuresInternal {
    pub(crate) fn destroy(self, device: &Device) {
        self.command_util.destroy(device);
        self.bottom_structures
            .into_iter()
            .for_each(|(_, blas)| blas.destroy(device));
        self.top_level_buffers
            .into_iter()
            .for_each(|b| b.into_iter().for_each(|b| b.destroy(device)));
        self.top_level_scratch_buffers
            .into_iter()
            .for_each(|b| b.into_iter().for_each(|b| b.destroy(device)));
        self.top_level_handles.into_iter().for_each(|b| {
            b.into_iter().for_each(|b| unsafe {
                device.acceleration_structure.destroy_acceleration_structure(b, None);
            })
        });
        self.instances_buffer.into_iter().for_each(|b| b.destroy(device));
        self.random_seed.destroy(device);
    }
}

pub(crate) fn build_acceleration_structures(
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    swapchain_indices: Res<SwapchainIndexToFrameNumber>,
    acceleration_structures: Res<AccelerationStructures>,
    mut acceleration_structures_internal: ResMut<AccelerationStructuresInternal>,
    submissions: Res<Submissions>,
    queries: QuerySet<(
        Query<&GltfMesh, With<ModelMatrix>>,
        Query<(&GltfMesh, &DrawIndex, &ModelMatrix)>,
    )>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
) {
    scope!("renderer::build_acceleration_structures");

    // TODO: there's a bug somewhere when new BLASes are being built and added to the scene while
    // actively ray tracing on screen? investigate synchronization

    let AccelerationStructuresInternal {
        ref mut command_util,
        ref mut bottom_structures,
        ref mut top_level_buffers,
        ref mut top_level_handles,
        ref mut top_level_scratch_buffers,
        // ref mut previous_top_level_buffers,
        ref mut instances_buffer,
        ref random_seed,
    } = *acceleration_structures_internal;

    // Wait for structures to have been used with this swapchain ix
    frame_graph::Main::Stage::wait_previous(&renderer, &image_index, &swapchain_indices);

        // renderer.auto_semaphores.0[frame_graph::Main::Stage::SIGNAL_AUTO_SEMAPHORE_IX]
        //     .wait(
        //         &renderer.device,
        //         as_of_last::<<frame_graph::Main::Stage as RenderStage>::SignalTimelineStage>(renderer.frame_number),
        //     )
        //     .unwrap();

    // TODO: double-buffer
    random_seed.map(&renderer.device).unwrap().seed = rand::random();

    // Free up structures used for this swapchain previously
    // TODO: inefficient and these can be reused in some cases
    // for buffer in replace(previous_top_level_buffers.current_mut(image_index.0), vec![]).into_iter() {
    //     buffer.destroy(&renderer.device);
    // }
    if let Some(previous_scratch) = top_level_scratch_buffers.current_mut(image_index.0).take() {
        previous_scratch.destroy(&renderer.device);
        // previous_top_level_buffers.current_mut(image_index.0).push(previous_scratch);
    }
    if let Some(previous_buffer) = top_level_buffers.current_mut(image_index.0).take() {
        previous_buffer.destroy(&renderer.device);
        // previous_top_level_buffers.current_mut(image_index.0).push(previous_buffer);
    }
    if let Some(previous_handle) = top_level_handles.current_mut(image_index.0).take() {
        unsafe {
            renderer
                .device
                .acceleration_structure
                .destroy_acceleration_structure(previous_handle, None);
        }
    }

    let command_buffer = command_util.reset_and_record(&renderer, &image_index);

    let all_marker = command_buffer.debug_marker_around("build_acceleration_structures", [0.3, 0.3, 0.3, 1.0]);

    // BLASes
    // TODO: force rebuild mode for profiling
    {
        let mut new_blases = HashMap::new();
        let mut geometries = vec![];
        let mut range_infos = vec![];
        let mut infos = vec![];

        for mesh in &mut queries.q0().iter() {
            let entry = bottom_structures.entry(MeshHandle(Arc::downgrade(&mesh.vertex_buffer)));

            entry.or_insert_with(|| {
                let build_sizes = unsafe {
                    let geometries = &[vk::AccelerationStructureGeometryKHR::builder()
                        .flags(vk::GeometryFlagsKHR::OPAQUE)
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(vk::AccelerationStructureGeometryDataKHR {
                            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                                .vertex_stride(3 * size_of::<f32>() as vk::DeviceSize)
                                .max_vertex(mesh.vertex_len.to_u32().unwrap())
                                .index_type(vk::IndexType::UINT32)
                                .build(),
                        })
                        .build()];
                    renderer
                        .device
                        .acceleration_structure
                        .get_acceleration_structure_build_sizes(
                            vk::AccelerationStructureBuildTypeKHR::DEVICE,
                            &vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                                .geometries(geometries),
                            &[mesh.index_buffers.last().unwrap().1.to_u32().unwrap() / 3],
                        )
                };

                debug_assert_eq!(
                    build_sizes.update_scratch_size, 0,
                    "expected update to be done in place, possible AMD specific thing"
                );

                let buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                    build_sizes.acceleration_structure_size,
                );

                let scratch_buffer = renderer.device.new_buffer(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                    build_sizes.build_scratch_size,
                );

                let handle = unsafe {
                    renderer
                        .device
                        .acceleration_structure
                        .create_acceleration_structure(
                            &vk::AccelerationStructureCreateInfoKHR::builder()
                                .buffer(buffer.handle)
                                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                                .size(build_sizes.acceleration_structure_size),
                            None,
                        )
                        .unwrap()
                };
                renderer.device.set_object_name(handle, "BLAS");

                let blas = BottomLevelAccelerationStructure {
                    buffer,
                    scratch_buffer,
                    handle,
                };

                let vertex_addr = unsafe {
                    renderer.device.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::builder().buffer(mesh.vertex_buffer.handle),
                    )
                };
                let index_addr = unsafe {
                    renderer.device.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::builder().buffer(mesh.index_buffers.last().unwrap().0.handle),
                    )
                };
                let ix = geometries.len();
                new_blases.insert(handle, ix);
                geometries.push(
                    vk::AccelerationStructureGeometryKHR::builder()
                        .flags(vk::GeometryFlagsKHR::OPAQUE)
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(vk::AccelerationStructureGeometryDataKHR {
                            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                                .vertex_stride(3 * size_of::<f32>() as vk::DeviceSize)
                                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                    device_address: vertex_addr,
                                })
                                .max_vertex(mesh.vertex_len.to_u32().unwrap())
                                .index_type(vk::IndexType::UINT32)
                                .index_data(vk::DeviceOrHostAddressConstKHR {
                                    device_address: index_addr,
                                })
                                .build(),
                        })
                        .build(),
                );
                debug_assert_eq!(range_infos.len(), ix);
                range_infos.push(vec![vk::AccelerationStructureBuildRangeInfoKHR::builder()
                    .primitive_count(mesh.index_buffers.last().unwrap().1.to_u32().unwrap() / 3)
                    .build()]);

                let scratch_addr = unsafe {
                    renderer.device.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::builder().buffer(blas.scratch_buffer.handle),
                    )
                };
                debug_assert_eq!(infos.len(), ix);
                infos.push(
                    vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                        .dst_acceleration_structure(blas.handle)
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: scratch_addr,
                        })
                        .build(),
                );

                blas
            });
        }

        let range_infos = range_infos.iter().map(|d| d.as_slice()).collect::<Vec<_>>();

        for (ix, info) in infos.iter_mut().enumerate() {
            info.geometry_count = 1;
            info.p_geometries = &geometries[ix];
        }

        #[cfg(feature = "crash_debugging")]
        crash_buffer.record(
            &renderer,
            *command_buffer,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            &image_index,
            2,
        );

        if infos.len() > 0 {
            let _blas_marker = command_buffer.debug_marker_around("BLAS", [0.1, 0.8, 0.1, 1.0]);

            unsafe {
                renderer
                    .device
                    .acceleration_structure
                    .cmd_build_acceleration_structures(*command_buffer, &infos, &range_infos);

                renderer.device.synchronization2.cmd_pipeline_barrier2(
                    *command_buffer,
                    &vk::DependencyInfoKHR::builder().memory_barriers(&[vk::MemoryBarrier2KHR::builder()
                        .src_stage_mask(vk::PipelineStageFlags2KHR::ACCELERATION_STRUCTURE_BUILD)
                        .src_access_mask(vk::AccessFlags2KHR::ACCELERATION_STRUCTURE_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2KHR::ACCELERATION_STRUCTURE_BUILD)
                        .dst_access_mask(vk::AccessFlags2KHR::ACCELERATION_STRUCTURE_READ)
                        .build()]),
                );
            }
        }

        #[cfg(feature = "crash_debugging")]
        crash_buffer.record(
            &renderer,
            *command_buffer,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            &image_index,
            3,
        );
    }

    // TLAS
    {
        let _tlas_marker = command_buffer.debug_marker_around("TLAS", [0.8, 0.1, 0.1, 1.0]);
        let _guard = renderer_macros::barrier!(
            *command_buffer,
            TLAS.build rw in BuildAccelerationStructures descriptor gltf_mesh.acceleration_set.top_level_as
        );

        fn pack(a: u32, b: u8) -> u32 {
            (a & ((1 << 24) - 1)) | (u32::from(b) << 24)
        }

        // println!("{:#010x}", (1 << 24) - 1);
        // println!("{:#010x}", pack(0xABABAB, 0xFF));
        // panic!();

        // println!("{}", model_matrix.0.rows(0, 3));

        let instances_addr = unsafe {
            renderer.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(instances_buffer.current(image_index.0).buffer.handle),
            )
        };

        let mut instance_count = 0;
        let mut mapped = instances_buffer
            .current_mut(image_index.0)
            .map(&renderer.device)
            .expect("failed to map instances buffer");
        for (ix, (mesh, draw_index, model_matrix)) in &mut queries.q1().iter().enumerate() {
            let blas = bottom_structures
                .get(&MeshHandle(Arc::downgrade(&mesh.vertex_buffer)))
                .unwrap();

            let mut matrix_data = [0f32; 12];

            // println!("Matrix\n {}", model_matrix.0.rows(0, 3));
            // println!("slice data {:?}", model_matrix.0.rows(0, 3).transpose().as_slice());
            debug_assert_eq!(model_matrix.0.rows(0, 3).transpose().as_slice().len(), 12);
            matrix_data.copy_from_slice(model_matrix.0.rows(0, 3).transpose().as_slice());

            let blas_address = unsafe {
                renderer
                    .device
                    .acceleration_structure
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::builder().acceleration_structure(blas.handle),
                    )
            };

            mapped[ix] = vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: matrix_data },
                instance_custom_index_and_mask: pack(draw_index.0, 0xFF),
                instance_shader_binding_table_record_offset_and_flags: pack(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
                        .as_raw()
                        .to_u8()
                        .unwrap(),
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas_address,
                },
            };
            instance_count += 1;
        }
        drop(mapped);

        let geometries = [vk::AccelerationStructureGeometryKHR::builder()
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instances_addr,
                    })
                    .build(),
            })
            .build()];

        let build_sizes = unsafe {
            renderer
                .device
                .acceleration_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                        .geometries(&geometries),
                    &[instance_count],
                )
        };

        // debug_assert_eq!(
        //     build_sizes.update_scratch_size, 0,
        //     "expected update to be done in place, possible AMD specific thing"
        // );

        let top_level_buffer = renderer.device.new_buffer(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            build_sizes.acceleration_structure_size,
        );

        let top_level_scratch_buffer = renderer.device.new_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            build_sizes.build_scratch_size,
        );
        
        let scratch_addr = unsafe {
            renderer.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(top_level_scratch_buffer.handle),
            )
        };

        let top_level_as = unsafe {
            renderer
                .device
                .acceleration_structure
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::builder()
                        .buffer(top_level_buffer.handle)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                        .size(build_sizes.acceleration_structure_size),
                    None,
                )
                .unwrap()
        };

        unsafe {
            renderer
                .device
                .acceleration_structure
                .cmd_build_acceleration_structures(
                    *command_buffer,
                    &[vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                        .dst_acceleration_structure(top_level_as)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                        .geometries(&geometries)
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: scratch_addr,
                        })
                        .build()],
                    &[&[vk::AccelerationStructureBuildRangeInfoKHR::builder()
                        .primitive_count(instance_count)
                        .build()]],
                );
            #[cfg(feature = "crash_debugging")]
            crash_buffer.record(
                &renderer,
                *command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                &image_index,
                4,
            );

            let structures = [top_level_as];
            let mut write_as =
                vk::WriteDescriptorSetAccelerationStructureKHR::builder().acceleration_structures(&structures);
            let mut write = vk::WriteDescriptorSet::builder()
                .push_next(&mut write_as)
                .dst_set(acceleration_structures.set.set.handle)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .build();
            write.descriptor_count = 1;

            renderer.device.update_descriptor_sets(&[write], &[]);
        }
        *top_level_buffers.current_mut(image_index.0) = Some(top_level_buffer);
        *top_level_scratch_buffers.current_mut(image_index.0) = Some(top_level_scratch_buffer);
        *top_level_handles.current_mut(image_index.0) = Some(top_level_as);
    }

    drop(all_marker);
    let command_buffer = command_buffer.end();

    submissions.submit(
        &renderer,
        &image_index,
        frame_graph::BuildAccelerationStructures::INDEX,
        Some(*command_buffer),
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
    );
}

impl AccelerationStructures {
    pub(crate) fn new(renderer: &RenderFrame, main_descriptor_pool: &MainDescriptorPool) -> Self {
        let set_layout = SmartSetLayout::new(&renderer.device);

        let set = SmartSet::new(&renderer.device, &main_descriptor_pool, &set_layout, 0);

        Self { set_layout, set }
    }

    pub(crate) fn destroy(self, device: &Device, main_descriptor_pool: &MainDescriptorPool) {
        self.set.destroy(&main_descriptor_pool.0, device);
        self.set_layout.destroy(device);
    }
}
