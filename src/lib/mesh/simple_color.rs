use ash::vk;
use std::path::PathBuf;
use gltf;
use gltf_importer;
use gltf_utils::PrimitiveIterators;

use super::Mesh;
use super::super::ExampleBase;
use super::super::buffer::Buffer;
use super::super::device::AshDevice;
use super::super::texture::Texture;

pub struct SimpleColor {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_type: vk::IndexType,
    pub index_count: u32,
    pub base_color_image: Texture,
    pub tex_coords: Buffer,
    pub texture_sampler: vk::Sampler,
    pub tangent_buffer: Buffer,
    pub normal_buffer: Buffer,
    pub normal_texture: Texture,
}

impl Mesh for SimpleColor {
    fn from_gltf<P: Into<PathBuf> + Clone>(base: &ExampleBase, path: P) -> Option<SimpleColor> {
        let importer = gltf_importer::import(path.clone().into());
        let (loaded, buffers) = importer.unwrap();

        let mut ret = None;

        for scene in loaded.scenes() {
            for node in scene.nodes() {
                fn for_mesh<P: Into<PathBuf> + Clone>(base: &ExampleBase, buffers: &gltf_importer::Buffers, path: &P, ret: &mut Option<SimpleColor>, mesh: &gltf::Mesh) {
                    if ret.is_some() {
                        return;
                    }
                    for primitive in mesh.primitives().take(1) {
                        println!("primitive mode {:?}", primitive.mode());
                        let base_color_texture = primitive
                            .material()
                            .pbr_metallic_roughness()
                            .base_color_texture()
                            .unwrap()
                            .texture();
                        let _base_color_sampler = base_color_texture.sampler();
                        let positions = primitive.positions(buffers).unwrap();

                        let base_color_image = match base_color_texture.source().data() {
                            gltf::image::Data::View { view, .. } => {
                                let slice = buffers.view(&view).unwrap();
                                Texture::load_from_memory(
                                    base,
                                    slice,
                                    vk::IMAGE_USAGE_SAMPLED_BIT,
                                    vk::Format::R8g8b8a8Unorm,
                                )
                            }
                            gltf::image::Data::Uri { uri, .. } => {
                                let actual_path = path.clone().into().as_path().parent().unwrap().join(uri);
                                println!("actual_path {:?}", actual_path);
                                Texture::load(
                                    base,
                                    actual_path,
                                    vk::IMAGE_USAGE_SAMPLED_BIT,
                                    vk::Format::R8g8b8a8Unorm,
                                )
                            }
                        };

                        let vertex_buffer = Buffer::upload_from::<[f32; 3], _>(base, vk::BUFFER_USAGE_VERTEX_BUFFER_BIT, &positions);
                        #[cfg(feature = "validation")]
                        unsafe {
                            use std::mem::transmute;
                            base.device.set_object_name(
                                vk::DebugReportObjectTypeEXT::Buffer,
                                transmute::<_, _>(vertex_buffer.vk()),
                                "SimpleColor Vertex Buffer",
                            );
                        }

                        let indices = primitive.indices_u32(buffers).unwrap();
                        let index_count = indices.len();
                        let index_buffer = Buffer::upload_from::<u32, _>(base, vk::BUFFER_USAGE_INDEX_BUFFER_BIT, &indices);
                        let index_type = vk::IndexType::Uint32;

                        let tex_coords_iter = primitive.tex_coords_f32(0, buffers).unwrap();
                        let tex_coords = Buffer::upload_from::<[f32; 2], _>(base, vk::BUFFER_USAGE_VERTEX_BUFFER_BIT, &tex_coords_iter);

                        let sampler = {
                            use ash::version::DeviceV1_0;
                            use std::ptr;
                            let create_info = vk::SamplerCreateInfo {
                                s_type: vk::StructureType::SamplerCreateInfo,
                                p_next: ptr::null(),
                                flags: Default::default(),
                                mag_filter: vk::Filter::Linear,
                                min_filter: vk::Filter::Linear,
                                mipmap_mode: vk::SamplerMipmapMode::Linear,
                                address_mode_u: vk::SamplerAddressMode::Repeat,
                                address_mode_v: vk::SamplerAddressMode::Repeat,
                                address_mode_w: vk::SamplerAddressMode::Repeat,
                                mip_lod_bias: 0.0,
                                anisotropy_enable: 1,
                                max_anisotropy: 16.0,
                                compare_enable: 0,
                                compare_op: vk::CompareOp::Always,
                                min_lod: 0.0,
                                max_lod: 0.0,
                                border_color: vk::BorderColor::IntOpaqueBlack,
                                unnormalized_coordinates: 0,
                            };

                            unsafe { base.device.create_sampler(&create_info, None).unwrap() }
                        };

                        let tangent_buffer = Buffer::upload_from::<[f32; 4], _>(
                            base,
                            vk::BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            &primitive.tangents(buffers).unwrap(),
                        );

                        let normal_texture = primitive.material().normal_texture().unwrap();
                        let normal_texture = match normal_texture.texture().source().data() {
                            gltf::image::Data::View { view, .. } => {
                                let slice = buffers.view(&view).unwrap();
                                Texture::load_from_memory(
                                    base,
                                    slice,
                                    vk::IMAGE_USAGE_SAMPLED_BIT,
                                    vk::Format::R8g8b8a8Unorm,
                                )
                            }
                            gltf::image::Data::Uri { uri, .. } => {
                                let actual_path = path.clone().into().as_path().parent().unwrap().join(uri);
                                println!("actual_path {:?}", actual_path);
                                Texture::load(
                                    base,
                                    actual_path,
                                    vk::IMAGE_USAGE_SAMPLED_BIT,
                                    vk::Format::R8g8b8a8Unorm,
                                )
                            }
                        };

                        let normal_buffer = Buffer::upload_from::<[f32; 3], _>(
                            base,
                            vk::BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            &primitive.normals(buffers).unwrap(),
                        );

                        println!("success");

                        *ret = Some(SimpleColor {
                            vertex_buffer: vertex_buffer,
                            index_buffer: index_buffer,
                            index_type: index_type,
                            index_count: index_count as u32,
                            base_color_image: base_color_image,
                            tex_coords: tex_coords,
                            texture_sampler: sampler,
                            tangent_buffer,
                            normal_buffer,
                            normal_texture,
                        });
                    }
                }
                fn browse<P: Into<PathBuf> + Clone>(base: &ExampleBase, buffers: &gltf_importer::Buffers, path: &P, ret: &mut Option<SimpleColor>, node: &gltf::Node) {
                    if let Some(mesh) = node.mesh() {
                        for_mesh(base, buffers, path, ret, &mesh)
                    }

                    for child in node.children() {
                        browse(base, buffers, path, ret, &child);
                    }
                }

                browse(base, &buffers, &path, &mut ret, &node);
            }
        }

        ret
    }

    unsafe fn free(self, device: &AshDevice) {
        self.vertex_buffer.free(device);
        self.index_buffer.free(device);
        self.base_color_image.free(device);
        self.tex_coords.free(device);
    }
}
