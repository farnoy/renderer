use std::mem::size_of;

use ash::vk;
use static_assertions::const_assert_eq;

use super::{
    device::{DescriptorSet, DescriptorSetLayout, Device},
    Buffer, DescriptorPool, MainDescriptorPool, StaticBuffer,
};


// renderer_macros::define_renderer! {
//     sets {
//         model_set {
//             model {
//                 type STORAGE_BUFFER
//                 count 1
//                 stages [VERTEX, COMPUTE]
//             }
//         },
//         camera_set {
//             matrices {
//                 type UNIFORM_BUFFER
//                 count 1
//                 stages [VERTEX, FRAGMENT, COMPUTE]
//             }
//         },
//         textures_set {
//             base_color {
//                 type COMBINED_IMAGE_SAMPLER
//                 partially bound
//                 update after bind
//                 count 3072
//                 stages [FRAGMENT]
//             },
//             normal_map {
//                 type COMBINED_IMAGE_SAMPLER
//                 partially bound
//                 update after bind
//                 count 3072
//                 stages [FRAGMENT]
//             }
//         },
//         cull_set {
//             indirect_commands {
//                 type STORAGE_BUFFER
//                 count 1
//                 stages [COMPUTE]
//             },
//             out_index_buffer {
//                 type STORAGE_BUFFER
//                 count 1
//                 stages [COMPUTE]
//             },
//             vertex_buffer {
//                 type STORAGE_BUFFER
//                 count 1
//                 stages [COMPUTE]
//             },
//             index_buffer {
//                 type STORAGE_BUFFER
//                 count 1
//                 stages [COMPUTE]
//             }
//         },
//         cull_commands_count_set {
//             indirect_commands_count {
//                 type STORAGE_BUFFER
//                 count 1
//                 stages [COMPUTE]
//             },
//         },
//         imgui_set {
//             texture {
//                 type COMBINED_IMAGE_SAMPLER
//                 count 1
//                 stages [FRAGMENT]
//             }
//         },
//         shadow_map_set {
//             light_data {
//                 type STORAGE_BUFFER
//                 partially bound
//                 count 16
//                 stages [VERTEX, FRAGMENT]
//             },
//             shadow_maps {
//                 type COMBINED_IMAGE_SAMPLER
//                 count 1
//                 stages [FRAGMENT]
//             }
//         }
//     }
//     pipelines {
//         generate_work {
//             descriptors [model_set, camera_set, cull_set]
//             specialization_constants [1 => local_workgroup_size: u32]
//             compute
//         },
//         compact_draw_stream {
//             descriptors [cull_set, cull_commands_count_set]
//             specialization_constants [
//                 1 => local_workgroup_size: u32,
//                 2 => draw_calls_to_compact: u32
//             ]
//             compute
//         },
//         depth_pipe {
//             descriptors [model_set, camera_set]
//             graphics
//             samples dyn
//             vertex_inputs [position: vec3]
//             stages [VERTEX]
//             cull mode BACK
//             depth test true
//             depth write true
//             depth compare op LESS_OR_EQUAL
//         },
//         gltf_mesh {
//             descriptors [model_set, camera_set, shadow_map_set, textures_set]
//             specialization_constants [
//                 10 => shadow_map_dim: u32,
//                 11 => shadow_map_dim_squared: u32,
//             ]
//             graphics
//             samples 4
//             vertex_inputs [position: vec3, normal: vec3, uv: vec2, tangent: vec4]
//             stages [VERTEX, FRAGMENT]
//             cull mode BACK
//             depth test true
//             depth compare op EQUAL
//         },
//         debug_aabb {
//             descriptors [camera_set]
//             graphics
//             samples 4
//             stages [VERTEX, FRAGMENT]
//             polygon mode LINE
//         },
//         imgui_pipe {
//             descriptors [imgui_set]
//             graphics
//             samples 4
//             vertex_inputs [pos: vec2, uv: vec2, col: vec4]
//             stages [VERTEX, FRAGMENT]
//         }
//     }
// }
