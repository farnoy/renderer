use std::{
    mem::size_of,
    path::{Path, PathBuf},
    sync::{Arc, Weak},
    u64,
};

use ash::vk;
use bevy_ecs::prelude::*;
use bevy_tasks::{AsyncComputeTaskPool, Task};
use hashbrown::{hash_map::Entry, HashMap};
#[cfg(feature = "compress_textures")]
use num_traits::ToPrimitive;
use profiling::scope;

#[cfg(feature = "crash_debugging")]
use crate::renderer::CrashBuffer;
use crate::{
    ecs::components::{ModelMatrix, Position, Rotation, Scale, AABB},
    renderer::{
        device::{Buffer, Device, DoubleBuffered, Image, StrictRecordingCommandBuffer, VmaMemoryUsage},
        frame_graph,
        helpers::command_util::CommandUtil,
        CoarseCulled, DrawIndex, GltfMesh, GltfMeshBaseColorTexture, GltfMeshNormalTexture, ImageIndex, RenderFrame,
        StrictCommandPool, Submissions,
    },
};

pub(crate) struct MeshToLoad {
    pub(crate) pos: [f32; 3],
    pub(crate) rot: [f32; 4],
    pub(crate) scale: f32,
    pub(crate) vertex_buffer: Vec<[f32; 3]>,
    pub(crate) normal_buffer: Vec<[f32; 3]>,
    pub(crate) uv_buffer: Vec<[f32; 2]>,
    pub(crate) tangent_buffer: Vec<[f32; 4]>,
    pub(crate) index_buffers: Vec<Vec<u32>>,
    pub(crate) vertex_len: u64,
    pub(crate) aabb: ncollide3d::bounding_volume::AABB<f32>,
    pub(crate) base_color_path: PathBuf,
    pub(crate) normal_map_path: PathBuf,
}

pub(crate) struct LoadedMesh {
    pub(crate) pos: [f32; 3],
    pub(crate) rot: [f32; 4],
    pub(crate) scale: f32,
    pub(crate) vertex_buffer: Vec<[f32; 3]>,
    pub(crate) normal_buffer: Vec<[f32; 3]>,
    pub(crate) uv_buffer: Vec<[f32; 2]>,
    pub(crate) tangent_buffer: Vec<[f32; 4]>,
    pub(crate) index_buffers: Vec<Vec<u32>>,
    pub(crate) vertex_len: u64,
    pub(crate) aabb: ncollide3d::bounding_volume::AABB<f32>,
    pub(crate) base_color: image::RgbaImage,
    pub(crate) base_color_path: PathBuf,
    pub(crate) normal_map: image::RgbaImage,
    pub(crate) normal_map_path: PathBuf,
}

/// Stores data about scenes we want to load or are in the process of loading
pub(crate) struct ScenesToLoad {
    /// Stores paths to GLTF scenes we haven't started loading
    pub(crate) scene_paths: Vec<String>,
    /// Stores the async task that parses the GLTF scene
    pub(crate) scenes: Vec<Task<(String, gltf::Document, Vec<gltf::buffer::Data>)>>,
}

/// Stores data for the UploadMeshes pass
pub(crate) struct UploadMeshesData {
    command_util: CommandUtil,
    deferred_buffers: DoubleBuffered<Vec<Buffer>>,
    /// Map from path on disk to a a pair of textures
    // TODO: these should be weak pointers so that upload does not prevent unloading
    texture_cache: HashMap<PathBuf, Arc<Image>>,
}

renderer_macros::define_resource! { IndividualGltfMeshBuffer = StaticBuffer<f32> }

impl FromWorld for UploadMeshesData {
    fn from_world(world: &mut World) -> Self {
        let renderer = world.get_resource().unwrap();

        UploadMeshesData {
            command_util: CommandUtil::new(renderer, renderer.device.compute_queue_family),
            deferred_buffers: renderer.new_buffered(|_| vec![]),
            texture_cache: HashMap::new(),
        }
    }
}

impl UploadMeshesData {
    pub(crate) fn destroy(self, device: &Device) {
        self.command_util.destroy(device);
    }
}

pub(crate) fn initiate_scene_loader(async_pool: Res<AsyncComputeTaskPool>, mut to_load: ResMut<ScenesToLoad>) {
    scope!("scene_loader::initiate_scene_loader");
    for gltf_path in std::mem::take(&mut to_load.scene_paths).into_iter() {
        let load_task = async_pool.spawn(async {
            scope!("scene_loader::async_scene_parse");
            let (loaded, buffers, _images) = gltf::import(&gltf_path).expect("Failed loading mesh");
            (gltf_path, loaded, buffers)
        });

        to_load.scenes.push(load_task);
    }
}

pub(crate) fn traverse_and_decode_scenes(
    mut commands: Commands,
    async_pool: Res<AsyncComputeTaskPool>,
    mut to_load: ResMut<ScenesToLoad>,
) {
    scope!("scene_loader::traverse_and_decode_scenes");
    let mut loaded_scenes = vec![];
    for mut load_task in std::mem::take(&mut to_load.scenes).into_iter() {
        futures_lite::future::block_on(async {
            scope!("scene_loader::traverse_and_decode_scenes::poll_once");
            match futures_lite::future::poll_once(&mut load_task).await {
                Some(data) => {
                    loaded_scenes.push(data);
                }
                None => {
                    to_load.scenes.push(load_task);
                }
            }
        });
    }
    for (gltf_path, loaded, buffers) in loaded_scenes.into_iter() {
        scope!("scene_loader::traverse_and_decode_scenes::scene_iter");
        for scene in loaded.scenes() {
            for node in scene.nodes() {
                scope!("scene_loader::traverse_and_decode_scenes::scene_iter::node_iter");
                visit_node(&mut commands, &async_pool, &buffers, &gltf_path, &node);
            }
        }
    }
}

pub(crate) fn upload_loaded_meshes(
    mut commands: Commands,
    renderer: Res<RenderFrame>,
    image_index: Res<ImageIndex>,
    submissions: Res<Submissions>,
    mut upload_data: ResMut<UploadMeshesData>,
    mut query: Query<(Entity, &mut Task<LoadedMesh>)>,
    #[cfg(feature = "crash_debugging")] crash_buffer: Res<CrashBuffer>,
) {
    scope!("scene_loader::upload_loaded_meshes");
    let mut loaded_meshes = vec![];
    for (entity, mut mesh) in query.iter_mut() {
        if loaded_meshes.len() > 8 {
            break;
        }
        futures_lite::future::block_on(async {
            match futures_lite::future::poll_once(&mut *mesh).await {
                Some(loaded_mesh) => {
                    commands.entity(entity).remove::<Task<LoadedMesh>>();
                    loaded_meshes.push(loaded_mesh);
                }
                None => {}
            }
        });
    }

    let UploadMeshesData {
        ref mut command_util,
        ref mut deferred_buffers,
        ref mut texture_cache,
    } = *upload_data;

    let command_buffer = command_util.reset_and_record(&renderer, &image_index);

    let marker = command_buffer.debug_marker_around("upload meshes", [0.0, 1.0, 0.0, 1.0]);

    let guard = renderer_macros::barrier!(
        *command_buffer,
        IndividualGltfMeshBuffer.upload w in UploadMeshes transfer copy,
    );

    let deferred_buffers = deferred_buffers.current_mut(image_index.0);

    for buffer in std::mem::take(deferred_buffers).into_iter() {
        buffer.destroy(&renderer.device);
    }

    for mesh in loaded_meshes.into_iter() {
        let vertex_len = mesh.vertex_len;
        let vertex_buffer = {
            let buf = renderer.device.new_buffer(
                vk::BufferUsageFlags::VERTEX_BUFFER
                        // | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                size_of::<f32>() as u64 * 3 * vertex_len,
            );
            renderer.device.set_object_name(buf.handle, "Gltf mesh Vertex buffer"); // TODO: extract nicer name from the GLTF scene
            scope!("scene_loader::cpu_copy_vertices");
            {
                let mut mapped = buf
                    .map::<[f32; 3]>(&renderer.device)
                    .expect("Failed to map vertex buffer");
                mapped[..mesh.vertex_buffer.len()].copy_from_slice(&mesh.vertex_buffer);
            }
            buf
        };
        let normal_buffer = {
            let buf = renderer.device.new_buffer(
                vk::BufferUsageFlags::VERTEX_BUFFER
                        // | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                size_of::<f32>() as u64 * 3 * vertex_len,
            );
            renderer.device.set_object_name(buf.handle, "Gltf mesh Normal buffer"); // TODO: extract nicer name from the GLTF scene
            {
                scope!("scene_loader::cpu_copy_normals");
                let mut mapped = buf
                    .map::<[f32; 3]>(&renderer.device)
                    .expect("Failed to map normal buffer");
                mapped[..mesh.normal_buffer.len()].copy_from_slice(&mesh.normal_buffer);
            }
            buf
        };
        let tangent_buffer = {
            let buf = renderer.device.new_buffer(
                vk::BufferUsageFlags::VERTEX_BUFFER
                        // | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                size_of::<f32>() as u64 * 4 * vertex_len,
            );
            renderer.device.set_object_name(buf.handle, "Gltf mesh Tangent buffer"); // TODO: extract nicer name from the GLTF scene
            {
                scope!("scene_loader::cpu_copy_tangents");
                let mut mapped = buf
                    .map::<[f32; 4]>(&renderer.device)
                    .expect("Failed to map tangent buffer");
                mapped[..].copy_from_slice(&mesh.tangent_buffer);
            }
            buf
        };
        let uv_buffer = {
            let buf = renderer.device.new_buffer(
                vk::BufferUsageFlags::VERTEX_BUFFER
                        // | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                size_of::<f32>() as u64 * 2 * vertex_len,
            );
            renderer.device.set_object_name(buf.handle, "Gltf mesh UV buffer"); // TODO: extract nicer name from the GLTF scene
            {
                scope!("scene_loader::cpu_copy_uvs");
                let mut mapped = buf.map::<[f32; 2]>(&renderer.device).expect("Failed to map uv buffer");
                mapped[..mesh.uv_buffer.len()].copy_from_slice(&mesh.uv_buffer);
            }
            buf
        };
        let index_buffers = mesh
            .index_buffers
            .iter()
            .enumerate()
            .map(|(ix, indices)| {
                let index_len = indices.len() as u64;
                let index_size = size_of::<u32>() as u64 * index_len;
                let index_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::INDEX_BUFFER
                        // | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    index_size,
                );
                renderer
                    .device
                    .set_object_name(index_buffer.handle, &format!("Gltf mesh index buffer LOD {}", ix));
                {
                    scope!("gltf_mesh::map_index_upload_buffer");
                    let mut mapped = index_buffer
                        .map::<u32>(&renderer.device)
                        .expect("failed to map index upload buffer");
                    mapped[0..index_len as usize].copy_from_slice(&indices);
                }

                (index_buffer, index_len)
            })
            .collect::<Vec<_>>();
        let gltf_mesh = GltfMesh {
            vertex_buffer: Arc::new(vertex_buffer),
            normal_buffer: Arc::new(normal_buffer),
            tangent_buffer: Arc::new(tangent_buffer),
            uv_buffer: Arc::new(uv_buffer),
            index_buffers: Arc::new(index_buffers),
            vertex_len,
            aabb: mesh.aabb,
        };

        let base_color_vkimage = texture_cache.entry(mesh.base_color_path.clone()).or_insert_with(|| {
            let base_color_vkimage = renderer.device.new_image(
                #[cfg(feature = "compress_textures")]
                vk::Format::BC7_UNORM_BLOCK,
                #[cfg(not(feature = "compress_textures"))]
                vk::Format::R8G8B8A8_SRGB,
                vk::Extent3D {
                    height: mesh.base_color.height(),
                    width: mesh.base_color.width(),
                    depth: 1,
                },
                vk::SampleCountFlags::TYPE_1,
                vk::ImageTiling::OPTIMAL,
                vk::ImageLayout::UNDEFINED,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            );
            renderer.device.set_object_name(
                base_color_vkimage.handle,
                &format!("Gltf mesh Base color Image {:?}", &mesh.base_color_path),
            );
            let base_color_upload_buffer = {
                let buf = renderer.device.new_buffer(
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    #[cfg(feature = "compress_textures")]
                    intel_tex::bc7::calc_output_size(mesh.base_color.width(), mesh.base_color.height())
                        .to_u64()
                        .unwrap(),
                    #[cfg(not(feature = "compress_textures"))]
                    {
                        vk::DeviceSize::from(mesh.base_color.width())
                            * vk::DeviceSize::from(mesh.base_color.height())
                            * 4
                    },
                );
                renderer
                    .device
                    .set_object_name(buf.handle, "Gltf mesh Base color image Upload buffer"); // TODO: extract nicer name from the GLTF scene
                {
                    scope!("scene_loader::cpu_copy_base_color_image");
                    let mut mapped = buf
                        .map::<image::Rgba<u8>>(&renderer.device)
                        .expect("Failed to map base color upload buffer");

                    #[cfg(not(feature = "compress_textures"))]
                    for (ix, pixel) in mesh.base_color.pixels().enumerate() {
                        mapped[ix] = *pixel;
                    }

                    #[cfg(feature = "compress_textures")]
                    intel_tex::bc7::compress_blocks_into(
                        &intel_tex::bc7::opaque_fast_settings(),
                        &intel_tex::RgbaSurface {
                            data: mesh.base_color.as_raw(),
                            width: mesh.base_color.width(),
                            height: mesh.base_color.height(),
                            stride: mesh.base_color.width() * 4,
                        },
                        &mut mapped[..],
                    );
                }
                buf
            };
            unsafe {
                scope!("scene_loader::upload_loaded_meshes::record_copy_commands");
                let device = &renderer.device;
                renderer.device.synchronization2.cmd_pipeline_barrier2(
                    *command_buffer,
                    &vk::DependencyInfoKHR::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2KHR::builder()
                        .src_stage_mask(vk::PipelineStageFlags2KHR::HOST)
                        .src_access_mask(vk::AccessFlags2KHR::HOST_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2KHR::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2KHR::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(base_color_vkimage.handle)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build()]),
                );
                device.device.cmd_copy_buffer_to_image(
                    *command_buffer,
                    base_color_upload_buffer.handle,
                    base_color_vkimage.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::builder()
                        .image_extent(vk::Extent3D {
                            height: mesh.base_color.height(),
                            width: mesh.base_color.width(),
                            depth: 1,
                        })
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build()],
                );

                renderer.device.synchronization2.cmd_pipeline_barrier2(
                    *command_buffer,
                    &vk::DependencyInfoKHR::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2KHR::builder()
                        .src_stage_mask(vk::PipelineStageFlags2KHR::TRANSFER)
                        .src_access_mask(vk::AccessFlags2KHR::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2KHR::BOTTOM_OF_PIPE)
                        .dst_access_mask(vk::AccessFlags2KHR::NONE)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(base_color_vkimage.handle)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build()]),
                );
            }
            deferred_buffers.push(base_color_upload_buffer);

            Arc::new(base_color_vkimage)
        });
        let base_color_vkimage = Arc::clone(base_color_vkimage);

        let normal_map_vkimage = texture_cache.entry(mesh.normal_map_path.clone()).or_insert_with(|| {
            let normal_map_vkimage = renderer.device.new_image(
                #[cfg(feature = "compress_textures")]
                vk::Format::BC7_UNORM_BLOCK,
                #[cfg(not(feature = "compress_textures"))]
                vk::Format::R8G8B8A8_SRGB,
                vk::Extent3D {
                    height: mesh.normal_map.height(),
                    width: mesh.normal_map.width(),
                    depth: 1,
                },
                vk::SampleCountFlags::TYPE_1,
                vk::ImageTiling::OPTIMAL,
                vk::ImageLayout::UNDEFINED,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            );
            renderer.device.set_object_name(
                normal_map_vkimage.handle,
                &format!("Gltf mesh Normal map Image {:?}", &mesh.normal_map_path),
            );
            let normal_map_upload_buffer = {
                let buf = renderer.device.new_buffer(
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    #[cfg(feature = "compress_textures")]
                    intel_tex::bc7::calc_output_size(mesh.normal_map.width(), mesh.normal_map.height())
                        .to_u64()
                        .unwrap(),
                    #[cfg(not(feature = "compress_textures"))]
                    {
                        vk::DeviceSize::from(mesh.normal_map.width())
                            * vk::DeviceSize::from(mesh.normal_map.height())
                            * 4
                    },
                );
                renderer
                    .device
                    .set_object_name(buf.handle, "Gltf mesh Normal map image Upload buffer"); // TODO: extract nicer name from the GLTF scene
                {
                    scope!("scene_loader::cpu_copy_normal_map_image");
                    let mut mapped = buf
                        .map::<image::Rgba<u8>>(&renderer.device)
                        .expect("Failed to map normal map upload buffer");

                    #[cfg(not(feature = "compress_textures"))]
                    for (ix, pixel) in mesh.normal_map.pixels().enumerate() {
                        mapped[ix] = *pixel;
                    }

                    #[cfg(feature = "compress_textures")]
                    intel_tex::bc7::compress_blocks_into(
                        &intel_tex::bc7::opaque_fast_settings(),
                        &intel_tex::RgbaSurface {
                            data: mesh.normal_map.as_raw(),
                            width: mesh.normal_map.width(),
                            height: mesh.normal_map.height(),
                            stride: mesh.normal_map.width() * 4,
                        },
                        &mut mapped[..],
                    );
                }
                buf
            };
            unsafe {
                scope!("scene_loader::upload_loaded_meshes::record_copy_commands");
                let device = &renderer.device;
                renderer.device.synchronization2.cmd_pipeline_barrier2(
                    *command_buffer,
                    &vk::DependencyInfoKHR::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2KHR::builder()
                        .src_stage_mask(vk::PipelineStageFlags2KHR::HOST)
                        .src_access_mask(vk::AccessFlags2KHR::HOST_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2KHR::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2KHR::NONE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(normal_map_vkimage.handle)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build()]),
                );
                device.device.cmd_copy_buffer_to_image(
                    *command_buffer,
                    normal_map_upload_buffer.handle,
                    normal_map_vkimage.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::builder()
                        .image_extent(vk::Extent3D {
                            height: mesh.normal_map.height(),
                            width: mesh.normal_map.width(),
                            depth: 1,
                        })
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build()],
                );

                renderer.device.synchronization2.cmd_pipeline_barrier2(
                    *command_buffer,
                    &vk::DependencyInfoKHR::builder().image_memory_barriers(&[vk::ImageMemoryBarrier2KHR::builder()
                        .src_stage_mask(vk::PipelineStageFlags2KHR::TRANSFER)
                        .src_access_mask(vk::AccessFlags2KHR::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2KHR::BOTTOM_OF_PIPE)
                        .dst_access_mask(vk::AccessFlags2KHR::NONE)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(normal_map_vkimage.handle)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build()]),
                );
            }
            deferred_buffers.push(normal_map_upload_buffer);

            Arc::new(normal_map_vkimage)
        });
        let normal_map_vkimage = Arc::clone(normal_map_vkimage);

        commands.spawn().insert_bundle((
            Position(na::Point3::from(mesh.pos)),
            Rotation(na::UnitQuaternion::new_normalize(na::Quaternion::from(mesh.rot))),
            Scale(mesh.scale),
            ModelMatrix::default(),
            GltfMeshBaseColorTexture(base_color_vkimage),
            GltfMeshNormalTexture(normal_map_vkimage),
            gltf_mesh,
            AABB::default(),
            CoarseCulled(false),
            DrawIndex::default(),
        ));
    }

    drop(guard);
    drop(marker);
    let command_buffer = command_buffer.end();

    submissions.submit(
        &renderer,
        &image_index,
        frame_graph::UploadMeshes::INDEX,
        Some(*command_buffer),
        #[cfg(feature = "crash_debugging")]
        &crash_buffer,
    );
}

fn load_mesh(mesh: MeshToLoad) -> LoadedMesh {
    scope!("scene_loader::load_mesh");
    let base_color = image::open(&mesh.base_color_path)
        .expect("failed to open base color texture")
        .to_rgba8();
    let normal_map = image::open(&mesh.normal_map_path)
        .expect("failed to open normal map texture")
        .to_rgba8();

    LoadedMesh {
        pos: mesh.pos,
        rot: mesh.rot,
        scale: mesh.scale,
        vertex_buffer: mesh.vertex_buffer,
        normal_buffer: mesh.normal_buffer,
        uv_buffer: mesh.uv_buffer,
        tangent_buffer: mesh.tangent_buffer,
        index_buffers: mesh.index_buffers,
        vertex_len: mesh.vertex_len,
        aabb: mesh.aabb,
        base_color,
        base_color_path: mesh.base_color_path,
        normal_map,
        normal_map_path: mesh.normal_map_path,
    }
}

fn visit_node(
    // renderer: &RenderFrame,
    // mesh_cache: &mut HashMap<(usize, usize), (GltfMesh, GltfMeshBaseColorTexture, GltfMeshNormalTexture)>,
    commands: &mut Commands,
    async_pool: &AsyncComputeTaskPool,
    buffers: &Vec<gltf::buffer::Data>,
    // deferred_buffers: &mut Vec<Buffer>,
    // command_buffer: &mut StrictRecordingCommandBuffer,
    path: &str,
    node: &gltf::Node,
    // budget: &mut usize,
) {
    scope!("scene_loader::visit_node");
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            scope!("scene_loader::visit_node::primitive");
            // dbg!(primitive.index());
            if primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .is_none()
            {
                // TODO: support base color factor
                continue;
            }

            let reader = {
                scope!("gltf_mesh::reader");
                primitive.reader(|buffer| Some(&buffers[buffer.index()]))
            };
            let positions = {
                scope!("gltf_mesh::read_positions");
                reader.read_positions().expect("failed to load positions")
            };
            if positions.len() < 100 {
                continue;
            }
            #[derive(Clone, Default, Debug)]
            struct Pos(pub(crate) [f32; 3]);

            impl meshopt::DecodePosition for Pos {
                fn decode_position(&self) -> [f32; 3] {
                    self.0
                }
            }
            let positions = positions.map(Pos).collect::<Vec<_>>();
            let uvs = reader
                .read_tex_coords(0)
                .expect("failed to load uvs")
                .into_f32()
                .collect::<Vec<_>>();
            let bounding_box = primitive.bounding_box();
            let aabb = ncollide3d::bounding_volume::AABB::new(
                na::Point3::from(bounding_box.min),
                na::Point3::from(bounding_box.max),
            );
            let normals = reader
                .read_normals()
                .expect("failed to load normals")
                .collect::<Vec<_>>();
            let tangents = reader
                .read_tangents()
                .expect("failed to load tangents")
                .collect::<Vec<_>>();
            let indices = reader
                .read_indices()
                .expect("failed to load indices")
                .into_u32()
                .collect::<Vec<_>>();
            let base_color_source = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .expect("failed to load base color")
                .texture()
                .source()
                .source();
            let normal_map_source = primitive
                .material()
                .normal_texture()
                .expect("failed to normal map texture")
                .texture()
                .source()
                .source();
            let base_color_path = match base_color_source {
                gltf::image::Source::Uri { uri, .. } => Path::new(path).parent().unwrap().join(uri),
                gltf::image::Source::View { .. } => {
                    unimplemented!("Reading embedded textures in gltf not supported")
                }
            };
            let normal_map_path = match normal_map_source {
                gltf::image::Source::Uri { uri, .. } => Path::new(path).parent().unwrap().join(uri),
                gltf::image::Source::View { .. } => {
                    unimplemented!("Reading embedded textures in gltf not supported")
                }
            };
            let index_lods = if true {
                let mut lods = Vec::with_capacity(6);
                for x in 1..6 {
                    let factor = 0.5f32.powf(x as f32);
                    let res = meshopt::simplify::simplify_sloppy_decoder(
                        &indices,
                        &positions,
                        (indices.len() as f32 * factor) as usize,
                    );
                    if res.len() < indices.len() && !res.is_empty() {
                        lods.push(res);
                    }
                }
                lods.insert(0, indices);
                lods
            } else {
                vec![indices]
            };

            let vertex_len = positions.len() as u64;

            let (pos, rot, scale) = node.transform().decomposed();

            let mesh_to_load = MeshToLoad {
                pos,
                rot,
                scale: scale[0],
                vertex_buffer: positions.into_iter().map(|x| x.0).collect(),
                normal_buffer: normals,
                uv_buffer: uvs,
                tangent_buffer: tangents,
                index_buffers: index_lods,
                vertex_len,
                aabb,
                base_color_path,
                normal_map_path,
            };

            let load_task: Task<LoadedMesh> = {
                scope!("scene_loader::visit_node::spawn_async_task");
                async_pool.spawn(async { load_mesh(mesh_to_load) })
            };

            commands.spawn().insert(load_task);
        }
    }

    for child in node.children() {
        visit_node(commands, async_pool, buffers, path, &child);
    }
}
