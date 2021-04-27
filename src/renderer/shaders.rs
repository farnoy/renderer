use std::mem::size_of;

use ash::vk;
use static_assertions::const_assert_eq;

use super::{
    device::{DescriptorSet, DescriptorSetLayout, Device},
    Buffer, DescriptorPool, MainDescriptorPool, StaticBuffer,
};

pub(crate) type UVBuffer = [[f32; 2]; size_of::<VertexBuffer>() / size_of::<[f32; 3]>()];

// sanity checks
const_assert_eq!(size_of::<VertexBuffer>() / size_of::<[f32; 3]>(), 10 * 30_000);
const_assert_eq!(
    size_of::<VkDrawIndexedIndirectCommand>(),
    size_of::<vk::DrawIndexedIndirectCommand>(),
);

// https://users.rust-lang.org/t/can-i-conveniently-compile-bytes-into-a-rust-program-with-a-specific-alignment/24049/2
#[repr(C)] // guarantee 'bytes' comes after '_align'
struct AlignedAs<Align, Bytes: ?Sized> {
    _align: [Align; 0],
    bytes: Bytes,
}

macro_rules! include_bytes_align_as {
    ($align_ty:ty, $path:literal) => {{
        // const block expression to encapsulate the static
        use super::AlignedAs;

        // this assignment is made possible by CoerceUnsized
        static ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($path),
        };

        &ALIGNED.bytes
    }};
}

renderer_macros::define_renderer! {
    sets {
        model_set {
            model {
                type UNIFORM_BUFFER
                count 1
                stages [VERTEX, COMPUTE]
            }
        },
        camera_set {
            matrices {
                type UNIFORM_BUFFER
                count 1
                stages [VERTEX, COMPUTE]
            }
        },
        base_color_set {
            texture {
                type COMBINED_IMAGE_SAMPLER
                partially bound
                update after bind
                count 3072
                stages [FRAGMENT]
            }
        },
        cull_set {
            indirect_commands {
                type STORAGE_BUFFER
                count 1
                stages [COMPUTE]
            },
            out_index_buffer {
                type STORAGE_BUFFER
                count 1
                stages [COMPUTE]
            },
            vertex_buffer {
                type STORAGE_BUFFER
                count 1
                stages [COMPUTE]
            },
            index_buffer {
                type STORAGE_BUFFER
                count 1
                stages [COMPUTE]
            }
        },
        cull_commands_count_set {
            indirect_commands_count {
                type STORAGE_BUFFER
                count 1
                stages [COMPUTE]
            },
        },
        imgui_set {
            texture {
                type COMBINED_IMAGE_SAMPLER
                count 1
                stages [FRAGMENT]
            }
        },
        shadow_map_set {
            light_data {
                type UNIFORM_BUFFER
                partially bound
                count 16
                stages [VERTEX, FRAGMENT]
            },
            shadow_maps {
                type COMBINED_IMAGE_SAMPLER
                count 1
                stages [FRAGMENT]
            }
        }
    }
    pipelines {
        generate_work {
            descriptors [model_set, camera_set, cull_set]
            specialization_constants [1 => local_workgroup_size: u32]
            compute
        },
        compact_draw_stream {
            descriptors [cull_set, cull_commands_count_set]
            specialization_constants [
                1 => local_workgroup_size: u32,
                2 => draw_calls_to_compact: u32
            ]
            compute
        },
        depth_pipe {
            descriptors [model_set, camera_set]
            graphics
            vertex_inputs [position: vec3]
            stages [VERTEX]
            cull mode BACK
            depth test true
            depth write true
            depth compare op LESS_OR_EQUAL
        },
        gltf_mesh {
            descriptors [model_set, camera_set, shadow_map_set, base_color_set]
            specialization_constants [
                10 => shadow_map_dim: u32,
                11 => shadow_map_dim_squared: u32,
            ]
            graphics
            vertex_inputs [position: vec3, normal: vec3, uv: vec2]
            stages [VERTEX, FRAGMENT]
            cull mode BACK
            depth test true
            depth compare op EQUAL
        },
        debug_aabb {
            descriptors [camera_set]
            graphics
            stages [VERTEX, FRAGMENT]
            polygon mode LINE
        },
        imgui_pipe {
            descriptors [imgui_set]
            graphics
            vertex_inputs [pos: vec2, uv: vec2, col: vec4]
            stages [VERTEX, FRAGMENT]
        }
    }
}
