use ash::vk;
use std::path::PathBuf;
use gltf;
use gltf_importer;

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
    base_color_image: Texture,
    tex_coords: Buffer,
}

impl Mesh for SimpleColor {
    fn from_gltf<P: Into<PathBuf> + Clone>(base: &ExampleBase, path: P) -> Option<SimpleColor> {
        let mut importer = gltf_importer::Importer::new(path.clone());
        let loaded = importer.import().unwrap();

        let mut ret = None;

        for scene in loaded.scenes() {
            for node in scene.nodes() {
                fn for_mesh<P: Into<PathBuf> + Clone>(base: &ExampleBase, path: P, ret: &mut Option<SimpleColor>, mesh: &gltf::Loaded<gltf::Mesh>) {
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
                        let base_color_sampler = base_color_texture.sampler();
                        let indices = primitive.indices().unwrap();
                        let positions = primitive.positions().unwrap();

                        let base_color_image = match base_color_texture.source().data() {
                            gltf::image::Data::FromBufferView { .. } => panic!("reading textures from embedded buffers is still unsupported"),
                            gltf::image::Data::External { uri, .. } => {
                                let actual_path = path.clone().into().as_path().parent().unwrap().join(uri);
                                Texture::load(base, actual_path, vk::IMAGE_USAGE_SAMPLED_BIT)
                            }
                        };

                        let vertex_buffer = Buffer::upload_from::<[f32; 3], _>(base, vk::BUFFER_USAGE_VERTEX_BUFFER_BIT, &positions);

                        let (index_buffer, index_type, index_count) = match indices {
                            gltf::mesh::Indices::U8(iter) => panic!("u8 indices are not supported"),
                            gltf::mesh::Indices::U16(iter) => (
                                Buffer::upload_from::<u16, _>(base, vk::BUFFER_USAGE_VERTEX_BUFFER_BIT, &iter),
                                vk::IndexType::Uint16,
                                iter.len(),
                            ),
                            gltf::mesh::Indices::U32(iter) => (
                                Buffer::upload_from::<u32, _>(base, vk::BUFFER_USAGE_VERTEX_BUFFER_BIT, &iter),
                                vk::IndexType::Uint32,
                                iter.len(),
                            ),
                        };

                        let tex_coords = match primitive.tex_coords(0).unwrap() {
                            gltf::mesh::TexCoords::F32(iter) => Buffer::upload_from(base, vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT, &iter),
                            _ => panic!("texture coordinates of this type are unsupported"),
                        };

                        println!("success");

                        *ret = Some(SimpleColor {
                            vertex_buffer: vertex_buffer,
                            index_buffer: index_buffer,
                            index_type: index_type,
                            index_count: index_count as u32,
                            base_color_image: base_color_image,
                            tex_coords: tex_coords,
                        });
                    }
                }
                fn browse<P: Into<PathBuf> + Clone>(base: &ExampleBase, path: P, ret: &mut Option<SimpleColor>, node: &gltf::Loaded<gltf::Node>) {
                    for mesh in node.mesh() {
                        for_mesh(base, path.clone(), ret, &mesh)
                    }

                    for child in node.children() {
                        browse(base, path.clone(), ret, &child);
                    }
                }

                browse(base, path.clone(), &mut ret, &node);
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
