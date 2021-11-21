use std::{mem::size_of, path::Path, sync::Arc, u64};

use ash::vk;
use bevy_ecs::prelude::World;
use hashbrown::{hash_map::Entry, HashMap};
#[cfg(feature = "compress_textures")]
use num_traits::ToPrimitive;
use profiling::scope;

use super::{
    device::{Buffer, Image, StrictRecordingCommandBuffer, VmaMemoryUsage},
    RenderFrame, StrictCommandPool,
};
use crate::{
    ecs::components::{ModelMatrix, Position, Rotation, Scale, AABB},
    renderer::{CoarseCulled, DrawIndex, GltfMesh, GltfMeshBaseColorTexture, GltfMeshNormalTexture},
};

pub(crate) struct LoadedMesh {
    pub(crate) vertex_buffer: Buffer,
    pub(crate) normal_buffer: Buffer,
    pub(crate) uv_buffer: Buffer,
    pub(crate) tangent_buffer: Buffer,
    pub(crate) index_buffers: Vec<(Buffer, u64)>,
    pub(crate) vertex_len: u64,
    pub(crate) aabb: ncollide3d::bounding_volume::AABB<f32>,
    pub(crate) base_color: Image,
    pub(crate) normal_map: Image,
}

#[derive(Clone, Default, Debug)]
struct Pos(pub(crate) [f32; 3]);

impl meshopt::DecodePosition for Pos {
    fn decode_position(&self) -> [f32; 3] {
        self.0
    }
}

pub(crate) fn load(renderer: &RenderFrame, path: &str) -> LoadedMesh {
    let mut command_pool = StrictCommandPool::new(
        &renderer.device,
        renderer.device.graphics_queue_family,
        "GLTF upload CommandPool",
    );
    let (loaded, buffers, _images) = gltf::import(path).expect("Failed loading mesh");
    let mesh = loaded.meshes().next().expect("failed to get first mesh from gltf");
    let primitive = mesh
        .primitives()
        .next()
        .expect("failed to get first primitive from gltf");
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let positions = reader
        .read_positions()
        .expect("failed to load positions")
        .map(Pos)
        .collect::<Vec<_>>();
    let uvs = reader
        .read_tex_coords(0)
        .expect("failed to load uvs")
        .into_f32()
        .collect::<Vec<_>>();
    let bounding_box = primitive.bounding_box();
    let aabb =
        ncollide3d::bounding_volume::AABB::new(na::Point3::from(bounding_box.min), na::Point3::from(bounding_box.max));
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
    let _metal_roughness_source = primitive
        .material()
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
        .expect("failed to load metallic roughness texture")
        .texture()
        .source()
        .source();
    let base_color_image = match base_color_source {
        gltf::image::Source::Uri { uri, .. } => image::open(Path::new(path).parent().unwrap().join(uri))
            .expect("failed to open base color texture")
            .to_rgba8(),
        gltf::image::Source::View { .. } => {
            unimplemented!("Reading embedded textures in gltf not supported")
        }
    };
    let base_color_vkimage = renderer.device.new_image(
        #[cfg(feature = "compress_textures")]
        vk::Format::BC7_UNORM_BLOCK,
        #[cfg(not(feature = "compress_textures"))]
        vk::Format::R8G8B8A8_SRGB,
        vk::Extent3D {
            height: base_color_image.height(),
            width: base_color_image.width(),
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
        vk::ImageTiling::OPTIMAL,
        vk::ImageLayout::UNDEFINED,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
    );
    renderer
        .device
        .set_object_name(base_color_vkimage.handle, "Gltf mesh Base color image");
    let base_color_upload_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::TRANSFER_SRC,
        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        #[cfg(feature = "compress_textures")]
        intel_tex::bc7::calc_output_size(base_color_image.width(), base_color_image.height())
            .to_u64()
            .unwrap(),
        #[cfg(not(feature = "compress_textures"))]
        {
            vk::DeviceSize::from(base_color_image.width()) * vk::DeviceSize::from(base_color_image.height()) * 4
        },
    );
    renderer.device.set_object_name(
        base_color_upload_buffer.handle,
        "Gltf mesh Base color image Upload Buffer",
    );
    {
        let mut mapped = base_color_upload_buffer
            .map::<image::Rgba<u8>>(&renderer.device)
            .expect("Failed to map base color upload buffer");
        #[cfg(not(feature = "compress_textures"))]
        for (ix, pixel) in base_color_image.pixels().enumerate() {
            mapped[ix] = *pixel;
        }

        #[cfg(feature = "compress_textures")]
        intel_tex::bc7::compress_blocks_into(
            &intel_tex::bc7::opaque_fast_settings(),
            &intel_tex::RgbaSurface {
                data: base_color_image.as_raw(),
                width: base_color_image.width(),
                height: base_color_image.height(),
                stride: base_color_image.width() * 4,
            },
            &mut mapped[..],
        );
    }
    let normal_map_image = match normal_map_source {
        gltf::image::Source::Uri { uri, .. } => image::open(Path::new(path).parent().unwrap().join(uri))
            .expect("failed to open normal map texture")
            .to_rgba8(),
        gltf::image::Source::View { .. } => {
            unimplemented!("Reading embedded textures in gltf not supported")
        }
    };
    let normal_map_vkimage = renderer.device.new_image(
        #[cfg(feature = "compress_textures")]
        vk::Format::BC7_UNORM_BLOCK,
        #[cfg(not(feature = "compress_textures"))]
        vk::Format::R8G8B8A8_SRGB,
        vk::Extent3D {
            height: normal_map_image.height(),
            width: normal_map_image.width(),
            depth: 1,
        },
        vk::SampleCountFlags::TYPE_1,
        vk::ImageTiling::OPTIMAL,
        vk::ImageLayout::UNDEFINED,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
    );
    renderer
        .device
        .set_object_name(normal_map_vkimage.handle, "Gltf mesh normal map image");
    let normal_map_upload_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::TRANSFER_SRC,
        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        #[cfg(feature = "compress_textures")]
        intel_tex::bc7::calc_output_size(normal_map_image.width(), normal_map_image.height())
            .to_u64()
            .unwrap(),
        #[cfg(not(feature = "compress_textures"))]
        {
            u64::from(normal_map_image.width()) * u64::from(normal_map_image.height()) * 4
        },
    );
    renderer.device.set_object_name(
        normal_map_upload_buffer.handle,
        "Gltf mesh Normal map image Upload Buffer",
    );
    {
        let mut mapped = normal_map_upload_buffer
            .map::<image::Rgba<u8>>(&renderer.device)
            .expect("Failed to map normal map upload buffer");
        #[cfg(not(feature = "compress_textures"))]
        for (ix, pixel) in normal_map_image.pixels().enumerate() {
            mapped[ix] = *pixel;
        }

        #[cfg(feature = "compress_textures")]
        intel_tex::bc7::compress_blocks_into(
            &intel_tex::bc7::opaque_fast_settings(),
            &intel_tex::RgbaSurface {
                data: normal_map_image.as_raw(),
                width: normal_map_image.width(),
                height: normal_map_image.height(),
                stride: normal_map_image.width() * 4,
            },
            &mut mapped[..],
        );
    }
    let index_lods = if false {
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
    // Disabling as it ruins sort order of indices and affects locality
    // for mut indices in index_lods.iter_mut() {
    //     // This is a bug in the library
    //     #[allow(clippy::unnecessary_mut_passed)]
    //     meshopt::optimize_vertex_cache_in_place(&mut indices, positions.len());
    //     meshopt::optimize_overdraw_in_place_decoder(&mut indices, &positions, 1.05);
    // }
    /*
    // quoting meshopt:
    When a sequence of LOD meshes is generated that all use the original vertex buffer, care must be taken to order
    vertices optimally to not penalize mobile GPU architectures that are only capable of transforming a sequential
    vertex buffer range. It's recommended in this case to first optimize each LOD for vertex cache, then assemble all
    LODs in one large index buffer starting from the coarsest LOD (the one with fewest triangles), and call
    meshopt_optimizeVertexFetch on the final large index buffer. This will make sure that coarser LODs require a smaller
     vertex range and are efficient wrt vertex fetch and transform.
    let remap = meshopt::optimize_vertex_fetch_remap(&indices, positions.len());
    let indices_new = meshopt::remap_index_buffer(Some(&indices), positions.len(), &remap);
    let positions_new = meshopt::remap_vertex_buffer(&positions, positions.len(), &remap);
    let uvs_new = meshopt::remap_vertex_buffer(&uvs, positions.len(), &remap);
    let normals_new = meshopt::remap_vertex_buffer(&normals, positions.len(), &remap);

    // shadow with optimized buffers
    let indices = indices_new;
    let normals = normals_new;
    let uvs = uvs_new;
    let positions = positions_new;
    */

    let vertex_len = positions.len() as u64;
    let vertex_size = size_of::<f32>() as u64 * 3 * vertex_len;
    let normals_size = size_of::<f32>() as u64 * 3 * vertex_len;
    let tangents_size = size_of::<f32>() as u64 * 4 * vertex_len;
    let uvs_size = size_of::<f32>() as u64 * 2 * vertex_len;
    let vertex_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        vertex_size,
    );
    renderer
        .device
        .set_object_name(vertex_buffer.handle, "Gltf mesh Vertex buffer");
    let vertex_upload_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::TRANSFER_SRC,
        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        vertex_size,
    );
    renderer
        .device
        .set_object_name(vertex_upload_buffer.handle, "Gltf mesh Vertex upload buffer");
    {
        let mut mapped = vertex_upload_buffer
            .map::<[f32; 3]>(&renderer.device)
            .expect("Failed to map vertex upload buffer");
        for (ix, data) in positions.iter().enumerate() {
            mapped[ix] = data.0;
        }
    }
    let normal_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        normals_size,
    );
    renderer
        .device
        .set_object_name(normal_buffer.handle, "Gltf mesh Normal buffer");
    let normal_upload_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::TRANSFER_SRC,
        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        normals_size,
    );
    renderer
        .device
        .set_object_name(normal_upload_buffer.handle, "Gltf mesh Normal upload buffer");
    {
        let mut mapped = normal_upload_buffer
            .map::<[f32; 3]>(&renderer.device)
            .expect("Failed to map normal upload buffer");
        for (ix, data) in normals.iter().enumerate() {
            mapped[ix] = *data;
        }
    }
    let tangent_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        tangents_size,
    );
    renderer
        .device
        .set_object_name(tangent_buffer.handle, "Gltf mesh Tangent buffer");
    let tangent_upload_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::TRANSFER_SRC,
        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        tangents_size,
    );
    renderer
        .device
        .set_object_name(tangent_upload_buffer.handle, "Gltf mesh Tangent upload buffer");
    {
        let mut mapped = tangent_upload_buffer
            .map::<[f32; 4]>(&renderer.device)
            .expect("Failed to map tangent upload buffer");
        for (ix, data) in tangents.iter().enumerate() {
            mapped[ix] = *data;
        }
    }
    let uv_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        uvs_size,
    );
    renderer.device.set_object_name(uv_buffer.handle, "Gltf mesh UV buffer");
    let uv_upload_buffer = renderer.device.new_buffer(
        vk::BufferUsageFlags::TRANSFER_SRC,
        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        uvs_size,
    );
    renderer
        .device
        .set_object_name(uv_upload_buffer.handle, "Gltf mesh UV upload buffer");
    {
        let mut mapped = uv_upload_buffer
            .map::<[f32; 2]>(&renderer.device)
            .expect("Failed to map UV upload buffer");
        for (ix, data) in uvs.iter().enumerate() {
            mapped[ix] = *data;
        }
    }
    let index_buffers = index_lods
        .iter()
        .enumerate()
        .map(|(ix, indices)| {
            let index_len = indices.len() as u64;
            let index_size = size_of::<u32>() as u64 * index_len;
            let index_buffer = renderer.device.new_buffer(
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                index_size,
            );
            renderer.device.set_object_name(
                index_buffer.handle,
                &format!("Gltf mesh {} index buffer LOD {}", path, ix),
            );
            let index_upload_buffer = renderer.device.new_buffer(
                vk::BufferUsageFlags::TRANSFER_SRC,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                index_size,
            );
            renderer.device.set_object_name(
                index_upload_buffer.handle,
                &format!("Gltf mesh index upload buffer LOD {}", ix),
            );
            {
                let mut mapped = index_upload_buffer
                    .map::<u32>(&renderer.device)
                    .expect("failed to map index upload buffer");
                mapped[0..index_len as usize].copy_from_slice(&indices);
            }

            (index_buffer, index_upload_buffer, index_len)
        })
        .collect::<Vec<_>>();
    let command_buffer = command_pool.record_one_time(&renderer.device, "upload gltf mesh cb");
    unsafe {
        let device = &renderer.device;
        device
            .device
            .cmd_copy_buffer(*command_buffer, vertex_upload_buffer.handle, vertex_buffer.handle, &[
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: vertex_size,
                },
            ]);
        device
            .device
            .cmd_copy_buffer(*command_buffer, normal_upload_buffer.handle, normal_buffer.handle, &[
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: normals_size,
                },
            ]);
        device
            .device
            .cmd_copy_buffer(*command_buffer, tangent_upload_buffer.handle, tangent_buffer.handle, &[
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: tangents_size,
                },
            ]);
        device
            .device
            .cmd_copy_buffer(*command_buffer, uv_upload_buffer.handle, uv_buffer.handle, &[
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: uvs_size,
                },
            ]);
        for (index_buffer, index_upload_buffer, index_len) in index_buffers.iter() {
            let index_size = size_of::<u32>() as u64 * index_len;
            device
                .device
                .cmd_copy_buffer(*command_buffer, index_upload_buffer.handle, index_buffer.handle, &[
                    vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: index_size,
                    },
                ]);
        }
        device.device.cmd_pipeline_barrier(
            *command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
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
                    .build(),
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
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
                    .build(),
            ],
        );
        device.device.cmd_copy_buffer_to_image(
            *command_buffer,
            base_color_upload_buffer.handle,
            base_color_vkimage.handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy::builder()
                .image_extent(vk::Extent3D {
                    height: base_color_image.height(),
                    width: base_color_image.width(),
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
        device.device.cmd_copy_buffer_to_image(
            *command_buffer,
            normal_map_upload_buffer.handle,
            normal_map_vkimage.handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy::builder()
                .image_extent(vk::Extent3D {
                    height: normal_map_image.height(),
                    width: normal_map_image.width(),
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
        device.device.cmd_pipeline_barrier(
            *command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
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
                    .build(),
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
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
                    .build(),
            ],
        );
    }
    let mut graphics_queue = renderer.device.graphics_queue().lock();
    let upload_fence = command_buffer.submit_once(&mut *graphics_queue, "upload gltf mesh commands");
    unsafe {
        renderer
            .device
            .wait_for_fences(&[upload_fence.handle], true, u64::MAX)
            .expect("Wait for fence failed.");
    }
    upload_fence.destroy(&renderer.device);
    let index_buffers = index_buffers
        .into_iter()
        .map(|(buffer, upload_buffer, len)| {
            upload_buffer.destroy(&renderer.device);
            (buffer, len)
        })
        .collect();

    uv_upload_buffer.destroy(&renderer.device);
    normal_upload_buffer.destroy(&renderer.device);
    tangent_upload_buffer.destroy(&renderer.device);
    vertex_upload_buffer.destroy(&renderer.device);
    base_color_upload_buffer.destroy(&renderer.device);
    normal_map_upload_buffer.destroy(&renderer.device);
    command_pool.destroy(&renderer.device);

    LoadedMesh {
        vertex_buffer,
        normal_buffer,
        tangent_buffer,
        uv_buffer,
        index_buffers,
        vertex_len,
        aabb,
        base_color: base_color_vkimage,
        normal_map: normal_map_vkimage,
    }
}

pub(crate) fn load_scene(world: &mut World, renderer: &RenderFrame, path: &str) {
    let mut command_pool = StrictCommandPool::new(
        &renderer.device,
        renderer.device.graphics_queue_family,
        "GLTF upload CommandPool",
    );
    let mut command_buffer = command_pool.record_one_time(&renderer.device, "upload gltf mesh cb");
    let (loaded, buffers, _images) = gltf::import(path).expect("Failed loading mesh");
    let mut deferred_buffers = vec![];
    dbg!(loaded.meshes().count());
    dbg!(loaded.materials().count());
    for scene in loaded.scenes() {
        // let mut budget = 500;
        // cache from gltf mesh primitive index -> GltfMesh
        let mut mesh_cache = HashMap::new();
        for node in scene.nodes() {
            dbg!(&node.index());
            scope!("gltf_mesh::scene_node");
            visit_node(
                world,
                renderer,
                &mut mesh_cache,
                &buffers,
                &mut deferred_buffers,
                &mut command_buffer,
                path,
                &node,
                // &mut budget,
            );
        }
    }
    let mut graphics_queue = renderer.device.graphics_queue().lock();
    let upload_fence = command_buffer.submit_once(&mut *graphics_queue, "upload gltf mesh commands");
    unsafe {
        renderer
            .device
            .wait_for_fences(&[upload_fence.handle], true, u64::MAX)
            .expect("Wait for fence failed.");
    }
    upload_fence.destroy(&renderer.device);
    command_pool.destroy(&renderer.device);
    for buffer in deferred_buffers.into_iter() {
        buffer.destroy(&renderer.device);
    }
    dbg!("Done");
    dbg!(world.entities().len());
}

fn visit_node(
    world: &mut World,
    renderer: &RenderFrame,
    mesh_cache: &mut HashMap<(usize, usize), (GltfMesh, GltfMeshBaseColorTexture, GltfMeshNormalTexture)>,
    buffers: &Vec<gltf::buffer::Data>,
    deferred_buffers: &mut Vec<Buffer>,
    command_buffer: &mut StrictRecordingCommandBuffer,
    path: &str,
    node: &gltf::Node,
    // budget: &mut usize,
) {
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            // if *budget == 0 {
            //     dbg!(*budget);
            //     break;
            // }
            dbg!(primitive.index());
            if primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .is_none()
            {
                // TODO: support base color factor
                dbg!("refuse");
                continue;
            }

            if let Entry::Vacant(e) = mesh_cache.entry((mesh.index(), primitive.index())) {
                let reader = {
                    scope!("gltf_mesh::reader");
                    primitive.reader(|buffer| Some(&buffers[buffer.index()]))
                };
                let positions = {
                    scope!("gltf_mesh::read_positions");
                    reader.read_positions().expect("failed to load positions")
                };
                if positions.len() < 100 {
                    dbg!("2small");
                    continue;
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
                // let _metal_roughness_source = primitive
                //     .material()
                //     .pbr_metallic_roughness()
                //     .metallic_roughness_texture()
                //     .expect("failed to load metallic roughness texture")
                //     .texture()
                //     .source()
                //     .source();
                let base_color_image = match base_color_source {
                    gltf::image::Source::Uri { uri, .. } => image::open(Path::new(path).parent().unwrap().join(uri))
                        .expect("failed to open base color texture")
                        .to_rgba8(),
                    gltf::image::Source::View { .. } => {
                        unimplemented!("Reading embedded textures in gltf not supported")
                    }
                };
                let base_color_vkimage = renderer.device.new_image(
                    #[cfg(feature = "compress_textures")]
                    vk::Format::BC7_UNORM_BLOCK,
                    #[cfg(not(feature = "compress_textures"))]
                    vk::Format::R8G8B8A8_SRGB,
                    vk::Extent3D {
                        height: base_color_image.height(),
                        width: base_color_image.width(),
                        depth: 1,
                    },
                    vk::SampleCountFlags::TYPE_1,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                );
                renderer
                    .device
                    .set_object_name(base_color_vkimage.handle, "Gltf mesh Base color image");
                let base_color_upload_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    #[cfg(feature = "compress_textures")]
                    intel_tex::bc7::calc_output_size(base_color_image.width(), base_color_image.height())
                        .to_u64()
                        .unwrap(),
                    #[cfg(not(feature = "compress_textures"))]
                    {
                        vk::DeviceSize::from(base_color_image.width())
                            * vk::DeviceSize::from(base_color_image.height())
                            * 4
                    },
                );
                renderer.device.set_object_name(
                    base_color_upload_buffer.handle,
                    "Gltf mesh Base color image Upload Buffer",
                );
                {
                    scope!("gltf_mesh::base_color_upload_buffer");
                    let mut mapped = base_color_upload_buffer
                        .map::<image::Rgba<u8>>(&renderer.device)
                        .expect("Failed to map base color upload buffer");
                    #[cfg(not(feature = "compress_textures"))]
                    for (ix, pixel) in base_color_image.pixels().enumerate() {
                        mapped[ix] = *pixel;
                    }

                    #[cfg(feature = "compress_textures")]
                    intel_tex::bc7::compress_blocks_into(
                        &intel_tex::bc7::opaque_fast_settings(),
                        &intel_tex::RgbaSurface {
                            data: base_color_image.as_raw(),
                            width: base_color_image.width(),
                            height: base_color_image.height(),
                            stride: base_color_image.width() * 4,
                        },
                        &mut mapped[..],
                    );
                }
                let normal_map_image = match normal_map_source {
                    gltf::image::Source::Uri { uri, .. } => {
                        scope!("gltf_mesh::image_open_normal_map");
                        image::open(Path::new(path).parent().unwrap().join(uri))
                            .expect("failed to open normal map texture")
                            .to_rgba8()
                    }
                    gltf::image::Source::View { .. } => {
                        unimplemented!("Reading embedded textures in gltf not supported")
                    }
                };
                let normal_map_vkimage = {
                    scope!("gltf_mesh::alloc_normal_map_vkimage");
                    renderer.device.new_image(
                        #[cfg(feature = "compress_textures")]
                        vk::Format::BC7_UNORM_BLOCK,
                        #[cfg(not(feature = "compress_textures"))]
                        vk::Format::R8G8B8A8_SRGB,
                        vk::Extent3D {
                            height: normal_map_image.height(),
                            width: normal_map_image.width(),
                            depth: 1,
                        },
                        vk::SampleCountFlags::TYPE_1,
                        vk::ImageTiling::OPTIMAL,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                    )
                };
                renderer
                    .device
                    .set_object_name(normal_map_vkimage.handle, "Gltf mesh normal map image");
                let normal_map_upload_buffer = {
                    scope!("gltf_mesh::alloc_normal_map_upload_buffer");
                    renderer.device.new_buffer(
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                        #[cfg(feature = "compress_textures")]
                        intel_tex::bc7::calc_output_size(normal_map_image.width(), normal_map_image.height())
                            .to_u64()
                            .unwrap(),
                        #[cfg(not(feature = "compress_textures"))]
                        {
                            u64::from(normal_map_image.width()) * u64::from(normal_map_image.height()) * 4
                        },
                    )
                };
                renderer.device.set_object_name(
                    normal_map_upload_buffer.handle,
                    "Gltf mesh Normal map image Upload Buffer",
                );
                {
                    scope!("gltf_mesh::normal_map_upload_buffer");
                    let mut mapped = normal_map_upload_buffer
                        .map::<image::Rgba<u8>>(&renderer.device)
                        .expect("Failed to map normal map upload buffer");
                    #[cfg(not(feature = "compress_textures"))]
                    for (ix, pixel) in normal_map_image.pixels().enumerate() {
                        mapped[ix] = *pixel;
                    }

                    #[cfg(feature = "compress_textures")]
                    intel_tex::bc7::compress_blocks_into(
                        &intel_tex::bc7::opaque_fast_settings(),
                        &intel_tex::RgbaSurface {
                            data: normal_map_image.as_raw(),
                            width: normal_map_image.width(),
                            height: normal_map_image.height(),
                            stride: normal_map_image.width() * 4,
                        },
                        &mut mapped[..],
                    );
                }
                let index_lods = if false {
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
                // Disabling as it ruins sort order of indices and affects locality
                // for mut indices in index_lods.iter_mut() {
                //     // This is a bug in the library
                //     #[allow(clippy::unnecessary_mut_passed)]
                //     meshopt::optimize_vertex_cache_in_place(&mut indices, positions.len());
                //     meshopt::optimize_overdraw_in_place_decoder(&mut indices, &positions, 1.05);
                // }
                /*
                // quoting meshopt:
                When a sequence of LOD meshes is generated that all use the original vertex buffer, care must be taken to order
                vertices optimally to not penalize mobile GPU architectures that are only capable of transforming a sequential
                vertex buffer range. It's recommended in this case to first optimize each LOD for vertex cache, then assemble all
                LODs in one large index buffer starting from the coarsest LOD (the one with fewest triangles), and call
                meshopt_optimizeVertexFetch on the final large index buffer. This will make sure that coarser LODs require a smaller
                 vertex range and are efficient wrt vertex fetch and transform.
                let remap = meshopt::optimize_vertex_fetch_remap(&indices, positions.len());
                let indices_new = meshopt::remap_index_buffer(Some(&indices), positions.len(), &remap);
                let positions_new = meshopt::remap_vertex_buffer(&positions, positions.len(), &remap);
                let uvs_new = meshopt::remap_vertex_buffer(&uvs, positions.len(), &remap);
                let normals_new = meshopt::remap_vertex_buffer(&normals, positions.len(), &remap);

                // shadow with optimized buffers
                let indices = indices_new;
                let normals = normals_new;
                let uvs = uvs_new;
                let positions = positions_new;
                */

                let vertex_len = positions.len() as u64;
                let vertex_size = size_of::<f32>() as u64 * 3 * vertex_len;
                let normals_size = size_of::<f32>() as u64 * 3 * vertex_len;
                let tangents_size = size_of::<f32>() as u64 * 4 * vertex_len;
                let uvs_size = size_of::<f32>() as u64 * 2 * vertex_len;
                let vertex_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                    vertex_size,
                );
                renderer
                    .device
                    .set_object_name(vertex_buffer.handle, "Gltf mesh Vertex buffer");
                let vertex_upload_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    vertex_size,
                );
                renderer
                    .device
                    .set_object_name(vertex_upload_buffer.handle, "Gltf mesh Vertex upload buffer");
                {
                    scope!("gltf_mesh::map_vertex_upload_buffer");
                    let mut mapped = vertex_upload_buffer
                        .map::<[f32; 3]>(&renderer.device)
                        .expect("Failed to map vertex upload buffer");
                    for (ix, data) in positions.iter().enumerate() {
                        mapped[ix] = data.0;
                    }
                }
                let normal_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                    normals_size,
                );
                renderer
                    .device
                    .set_object_name(normal_buffer.handle, "Gltf mesh Normal buffer");
                let normal_upload_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    normals_size,
                );
                renderer
                    .device
                    .set_object_name(normal_upload_buffer.handle, "Gltf mesh Normal upload buffer");
                {
                    scope!("gltf_mesh::map_normal_upload_buffer");
                    let mut mapped = normal_upload_buffer
                        .map::<[f32; 3]>(&renderer.device)
                        .expect("Failed to map normal upload buffer");
                    for (ix, data) in normals.iter().enumerate() {
                        mapped[ix] = *data;
                    }
                }
                let tangent_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                    tangents_size,
                );
                renderer
                    .device
                    .set_object_name(tangent_buffer.handle, "Gltf mesh Tangent buffer");
                let tangent_upload_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    tangents_size,
                );
                renderer
                    .device
                    .set_object_name(tangent_upload_buffer.handle, "Gltf mesh Tangent upload buffer");
                {
                    let mut mapped = tangent_upload_buffer
                        .map::<[f32; 4]>(&renderer.device)
                        .expect("Failed to map tangent upload buffer");
                    for (ix, data) in tangents.iter().enumerate() {
                        mapped[ix] = *data;
                    }
                }
                let uv_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                    uvs_size,
                );
                renderer.device.set_object_name(uv_buffer.handle, "Gltf mesh UV buffer");
                let uv_upload_buffer = renderer.device.new_buffer(
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                    uvs_size,
                );
                renderer
                    .device
                    .set_object_name(uv_upload_buffer.handle, "Gltf mesh UV upload buffer");
                {
                    let mut mapped = uv_upload_buffer
                        .map::<[f32; 2]>(&renderer.device)
                        .expect("Failed to map UV upload buffer");
                    for (ix, data) in uvs.iter().enumerate() {
                        mapped[ix] = *data;
                    }
                }
                let index_buffers = index_lods
                    .iter()
                    .enumerate()
                    .map(|(ix, indices)| {
                        let index_len = indices.len() as u64;
                        let index_size = size_of::<u32>() as u64 * index_len;
                        let index_buffer = renderer.device.new_buffer(
                            vk::BufferUsageFlags::INDEX_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_DST
                                | vk::BufferUsageFlags::TRANSFER_SRC
                                | vk::BufferUsageFlags::STORAGE_BUFFER
                                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
                            index_size,
                        );
                        renderer.device.set_object_name(
                            index_buffer.handle,
                            &format!("Gltf mesh {} index buffer LOD {}", path, ix),
                        );
                        let index_upload_buffer = renderer.device.new_buffer(
                            vk::BufferUsageFlags::TRANSFER_SRC,
                            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                            index_size,
                        );
                        renderer.device.set_object_name(
                            index_upload_buffer.handle,
                            &format!("Gltf mesh index upload buffer LOD {}", ix),
                        );
                        {
                            scope!("gltf_mesh::map_index_upload_buffer");
                            let mut mapped = index_upload_buffer
                                .map::<u32>(&renderer.device)
                                .expect("failed to map index upload buffer");
                            mapped[0..index_len as usize].copy_from_slice(&indices);
                        }

                        (index_buffer, index_upload_buffer, index_len)
                    })
                    .collect::<Vec<_>>();
                unsafe {
                    scope!("gltf_mesh::record_copy_commands");
                    let device = &renderer.device;
                    device.device.cmd_copy_buffer(
                        **command_buffer,
                        vertex_upload_buffer.handle,
                        vertex_buffer.handle,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 0,
                            size: vertex_size,
                        }],
                    );
                    device.device.cmd_copy_buffer(
                        **command_buffer,
                        normal_upload_buffer.handle,
                        normal_buffer.handle,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 0,
                            size: normals_size,
                        }],
                    );
                    device.device.cmd_copy_buffer(
                        **command_buffer,
                        tangent_upload_buffer.handle,
                        tangent_buffer.handle,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 0,
                            size: tangents_size,
                        }],
                    );
                    device
                        .device
                        .cmd_copy_buffer(**command_buffer, uv_upload_buffer.handle, uv_buffer.handle, &[
                            vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: 0,
                                size: uvs_size,
                            },
                        ]);
                    for (index_buffer, index_upload_buffer, index_len) in index_buffers.iter() {
                        let index_size = size_of::<u32>() as u64 * index_len;
                        device.device.cmd_copy_buffer(
                            **command_buffer,
                            index_upload_buffer.handle,
                            index_buffer.handle,
                            &[vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: 0,
                                size: index_size,
                            }],
                        );
                    }
                    device.device.cmd_pipeline_barrier(
                        **command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[
                            vk::ImageMemoryBarrier::builder()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
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
                                .build(),
                            vk::ImageMemoryBarrier::builder()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
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
                                .build(),
                        ],
                    );
                    device.device.cmd_copy_buffer_to_image(
                        **command_buffer,
                        base_color_upload_buffer.handle,
                        base_color_vkimage.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::BufferImageCopy::builder()
                            .image_extent(vk::Extent3D {
                                height: base_color_image.height(),
                                width: base_color_image.width(),
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
                    device.device.cmd_copy_buffer_to_image(
                        **command_buffer,
                        normal_map_upload_buffer.handle,
                        normal_map_vkimage.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::BufferImageCopy::builder()
                            .image_extent(vk::Extent3D {
                                height: normal_map_image.height(),
                                width: normal_map_image.width(),
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
                    device.device.cmd_pipeline_barrier(
                        **command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[
                            vk::ImageMemoryBarrier::builder()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ)
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
                                .build(),
                            vk::ImageMemoryBarrier::builder()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ)
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
                                .build(),
                        ],
                    );
                }
                let index_buffers = index_buffers
                    .into_iter()
                    .map(|(buffer, upload_buffer, len)| {
                        deferred_buffers.push(upload_buffer);
                        (buffer, len)
                    })
                    .collect::<Vec<_>>();

                deferred_buffers.push(uv_upload_buffer);
                deferred_buffers.push(normal_upload_buffer);
                deferred_buffers.push(tangent_upload_buffer);
                deferred_buffers.push(vertex_upload_buffer);
                deferred_buffers.push(base_color_upload_buffer);
                deferred_buffers.push(normal_map_upload_buffer);

                let gltf_mesh = GltfMesh {
                    vertex_buffer: Arc::new(vertex_buffer),
                    normal_buffer: Arc::new(normal_buffer),
                    tangent_buffer: Arc::new(tangent_buffer),
                    uv_buffer: Arc::new(uv_buffer),
                    index_buffers: Arc::new(index_buffers),
                    vertex_len,
                    aabb,
                };
                let base_color = GltfMeshBaseColorTexture(Arc::new(base_color_vkimage));
                let normal_texture = GltfMeshNormalTexture(Arc::new(normal_map_vkimage));

                e.insert((gltf_mesh, base_color, normal_texture));
            }

            // *budget -= 1;

            let (pos, rot, scale) = node.transform().decomposed();
            let (gltf_mesh, base_color, normal_texture) = mesh_cache.get(&(mesh.index(), primitive.index())).unwrap();

            world.spawn().insert_bundle((
                Position(na::Point3::from(pos)),
                Rotation(na::UnitQuaternion::new_normalize(na::Quaternion::from(rot))),
                Scale(scale[0]),
                ModelMatrix::default(),
                base_color.clone(),
                normal_texture.clone(),
                gltf_mesh.clone(),
                AABB::default(),
                CoarseCulled(false),
                DrawIndex::default(),
            ));
        }
    }

    for child in node.children() {
        visit_node(
            world,
            renderer,
            mesh_cache,
            buffers,
            deferred_buffers,
            command_buffer,
            path,
            &child,
            // budget,
        );
    }
}
