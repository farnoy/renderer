use ash::vk;

macro_rules! to_vk_format {
    (vec4) => {
        vk::Format::R32G32B32A32_SFLOAT
    };
    (vec3) => {
        vk::Format::R32G32B32_SFLOAT
    };
    (vec2) => {
        vk::Format::R32G32_SFLOAT
    };
}

macro_rules! compare_type {
    (u32, $desc: expr) => {
        *$desc == spirq::Type::Scalar(spirq::ty::ScalarType::int(4, false))
    };
    (vec4, $desc: expr) => {
        *$desc
            == spirq::Type::Vector(spirq::ty::VectorType::new(
                spirq::ty::ScalarType::float(4),
                4,
            ))
    };
    (vec3, $desc: expr) => {
        *$desc
            == spirq::Type::Vector(spirq::ty::VectorType::new(
                spirq::ty::ScalarType::float(4),
                3,
            ))
    };
    (vec2, $desc: expr) => {
        *$desc
            == spirq::Type::Vector(spirq::ty::VectorType::new(
                spirq::ty::ScalarType::float(4),
                2,
            ))
    };
}

macro_rules! to_rust_type {
    (vec4) => {
        glm::Vec4
    };
    (vec3) => {
        glm::Vec3
    };
    (vec2) => {
        glm::Vec2
    };
    ($a:ty) => {
        $a
    };
}

macro_rules! make_descriptor_set {
    ($name:ident [$($desc_count:expr $(, partially $partial:ident)? => $binding_name:ident, $t:ident, $stage:expr, $desc_type:expr);*]) => {
        pub mod $name {
            use ash::{version::DeviceV1_0, vk};
            use std::sync::Arc;

            pub const COUNT: usize = 0usize $(+ { let _ = stringify!($binding_name); 1usize } )*;

            pub const NAMES: [&'static str; COUNT] = [
                $(
                    stringify!($binding_name),
                )*
            ];

            pub const TYPE_SIZES: [vk::DeviceSize; COUNT] = [
                $(
                    std::mem::size_of::<super::$t>() as vk::DeviceSize,
                )*
            ];

            #[allow(clippy::eval_order_dependence)]
            pub fn bindings() -> [vk::DescriptorSetLayoutBinding; COUNT] {
                let mut ix = 0usize;
                [
                    $(
                        {
                            let r = vk::DescriptorSetLayoutBinding {
                                binding: ix as u32,
                                descriptor_type: $desc_type,
                                descriptor_count: $desc_count,
                                stage_flags: $stage,
                                p_immutable_samplers: std::ptr::null(),
                            };
                            #[allow(unused_assignments)]
                            {
                                ix += 1;
                            }
                            r
                        },
                    )*
                ]
            }

            pub mod bindings {
                $(
                    pub mod $binding_name {
                        use ash::vk;

                        pub type T = super::super::super::$t;
                        pub const SIZE: vk::DeviceSize = std::mem::size_of::<T>() as vk::DeviceSize;
                    }
                )*
            }

            pub struct DescriptorSetLayout {
                pub layout: super::super::DescriptorSetLayout,
            }

            impl DescriptorSetLayout {
                pub fn new(device: &Arc<super::super::Device>) -> DescriptorSetLayout {
                    let binding_flags = &[
                        $(
                            {
                                let _x = $desc_count;
                                vk::DescriptorBindingFlagsEXT::default() $( | {
                                    let _ = stringify!($partial);
                                    vk::DescriptorBindingFlagsEXT::PARTIALLY_BOUND
                                })?
                            },
                        )*
                    ];
                    let mut binding_flags =
                        vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
                            .binding_flags(binding_flags);
                    let b = bindings();
                    let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&b)
                        .push_next(&mut binding_flags);
                    DescriptorSetLayout {
                        layout: device.new_descriptor_set_layout2(&create_info)
                    }
                }
            }

            pub struct DescriptorSet {
                pub set: super::super::DescriptorSet,
            }

            impl DescriptorSet {
                pub fn new(
                    main_descriptor_pool: &super::super::MainDescriptorPool,
                    layout: &DescriptorSetLayout,
                ) -> DescriptorSet {
                    let set = main_descriptor_pool.0.allocate_set(&layout.layout);

                    DescriptorSet { set }
                }

                // TODO: add batching and return vk::WriteDescriptorSet when lifetimes are improved in ash
                pub fn update_whole_buffer(&self, renderer: &super::super::RenderFrame,
                                           binding: u32, buf: &super::super::Buffer) {
                    let buffer_updates = &[vk::DescriptorBufferInfo {
                        buffer: buf.handle,
                        offset: 0,
                        range: TYPE_SIZES[binding as usize],
                    }];
                    unsafe {
                        renderer.device.update_descriptor_sets(
                            &[vk::WriteDescriptorSet::builder()
                                .dst_set(self.set.handle)
                                .dst_binding(binding)
                                .descriptor_type(bindings()[binding as usize].descriptor_type)
                                .buffer_info(buffer_updates)
                                .build()],
                            &[],
                        );
                    }

                }
            }
        }
    };
}

fn compare_descriptor_types(lhs: vk::DescriptorType, rhs: &spirq::DescriptorType) -> bool {
    match (lhs, rhs) {
        (vk::DescriptorType::UNIFORM_BUFFER, spirq::DescriptorType::UniformBuffer(_, _)) => true,
        (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, spirq::DescriptorType::SampledImage(_, _)) => {
            true
        }
        _ => panic!(
            "compare_descriptor_types not implemented for {:?} {:?}",
            lhs, rhs
        ),
    }
}

macro_rules! make_pipe {
    (@validate_vertex_input $s:ident []) => {};
    (@validate_vertex_input $s:ident [$($e:ident : $t:ident),*]) => {
        let mut ix = 0;
        $(
            match $s.get_input(spirq::InterfaceLocation::new(ix, 0)) {
                Some(ty) => {
                    if !compare_type!($t, ty) {
                        return false;
                    }
                },
                None => {
                    println!("not found input variable: {}", stringify!($e));
                    return false;
                }
            };
            #[allow(unused_assignments)]
            {
                ix += 1;
            }
        )*
    };
    (@vertex_descs [$($e:ident : $t:ident),*]) => {
        #[allow(unused_mut, clippy::eval_order_dependence)]
        const ATTRIBUTE_DESCS: &[vk::VertexInputAttributeDescription] = {
            let mut ix = 0;
            &[
                $(
                    {
                        let r = vk::VertexInputAttributeDescription {
                            location: ix,
                            binding: ix,
                            format: to_vk_format!($t),
                            offset: 0,
                        };
                        ix += 1;
                        r
                    },
                )*
            ]
        };

        #[allow(unused_mut, clippy::eval_order_dependence)]
        const BINDING_DESCS: &[vk::VertexInputBindingDescription] = {
            let mut ix = 0;
            &[
                $(
                    {
                        let r = vk::VertexInputBindingDescription {
                            binding: ix,
                            stride: std::mem::size_of::<to_rust_type!($t)>() as u32,
                            input_rate: vk::VertexInputRate::VERTEX,
                        };
                        ix += 1;
                        r
                    },
                )*
            ]
        };

        pub fn vertex_input_state<'a>() -> vk::PipelineVertexInputStateCreateInfoBuilder<'a> {
            vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&ATTRIBUTE_DESCS)
                .vertex_binding_descriptions(&BINDING_DESCS)
        }
    };
    (@count []) => { 0 };
    (@count [$x:expr $(, $rest:expr),*]) => { 1 + make_pipe!(@count [$($rest),*]) };
    (@spec_consts [$($id:expr => $name:ident : $ty:ty),*]) => {
        use std::{mem::size_of, slice::from_raw_parts};
        use memoffset::offset_of;

        #[repr(C)]
        #[derive(PartialEq, Clone)]
        pub struct Specialization {
            $(
                pub $name : $ty,
            )*
        }

        lazy_static! {
            pub static ref SPEC_MAP: [vk::SpecializationMapEntry; make_pipe!(@count [$($id),*])] = [
                $(
                    vk::SpecializationMapEntry::builder()
                    .constant_id($id)
                    .offset(offset_of!(Specialization, $name) as u32)
                    .size(size_of::<$ty>())
                    .build()
                ),*
            ];
        }

        impl Specialization {
            pub fn get_spec_info(&self) -> vk::SpecializationInfo {
                let (left, spec_data, right) = unsafe { from_raw_parts(self as *const Specialization, 1).align_to::<u8>() };
                assert!(
                    left.is_empty() && right.is_empty(),
                    "spec constant alignment failed"
                );
                vk::SpecializationInfo::builder()
                    .map_entries(&*SPEC_MAP)
                    .data(spec_data)
                    .build()
            }
        }
    };
    ($name:ident { compute, descriptors: [$($desc:ident),*] $(, push_constants: $push:ident)? $(, specialization_constants: $spec_const:tt)?}) => {
        make_pipe!(@main $name { vertex_inputs: [], compute: true, descriptors: [$($desc),*] $(, push_constants: $push)? $(, specialization_constants: $spec_const)?});
    };
    ($name:ident { vertex_inputs: $vertex_inputs:tt, descriptors: [$($desc:ident),*] $(, push_constants: $push:ident)?}) => {
        make_pipe!(@main $name { vertex_inputs: $vertex_inputs, compute: false, descriptors: [$($desc),*] $(, push_constants: $push)?});
    };
    (@main $name:ident { vertex_inputs: $vertex_inputs:tt, compute: $compute:expr, descriptors: [$($desc:ident),*] $(, push_constants: $push:ident)? $(, specialization_constants: $spec_const:tt)?}) => {
        pub mod $name {
            use ash::{version::DeviceV1_0, vk};
            use std::{io::Read, sync::Arc};

            const IS_COMPUTE: bool = $compute;

            pub fn load_and_verify_spirv_file(path: &'static str) -> bool {
                let path = std::path::PathBuf::from(env!("OUT_DIR")).join(path);
                let file = std::fs::File::open(path).expect("Could not find shader.");
                let bytes: Vec<u8> = file.bytes().filter_map(Result::ok).collect();
                self::load_and_verify_spirv(&bytes)
            }

            pub fn load_and_verify_spirv(data: &[u8]) -> bool {
                let (l, spv_words, r) = unsafe { data.align_to::<u32>() };
                assert!(l.is_empty() && r.is_empty(), "failed to realign code");

                let spv: spirq::SpirvBinary = spv_words.into();
                let module = spv
                    .reflect()
                    .unwrap();
                let entry = module
                    .iter()
                    .find(|entry| entry.name == "main")
                    .expect("Failed to find entry point `main`");
                self::verify_spirv(&entry)
            }

            pub fn verify_spirv(s: &spirq::EntryPoint) -> bool {
                if s.exec_model == spirq::ExecutionModel::Vertex {
                    make_pipe!(@validate_vertex_input s $vertex_inputs);
                }
                let mut set_ix = 0;
                $(
                    let bindings = super::$desc::bindings();
                    for (binding, type_size) in bindings.iter().zip(super::$desc::TYPE_SIZES.iter()) {
                        if binding.descriptor_type == vk::DescriptorType::STORAGE_BUFFER {
                            println!("Skipping storage buffer validation, can't reflect on it!");
                            continue;
                        }
                        if let Some(ty) = s.get_desc(spirq::DescriptorBinding::new(set_ix, binding.binding)) {
                            if !super::compare_descriptor_types(binding.descriptor_type, ty) || ty.nbyte().map(|s| s as vk::DeviceSize != *type_size).unwrap_or(false) {
                                dbg!(binding.descriptor_type, ty);
                                return false;
                            }
                        }
                        // If shader uses a subset of available descriptors, it _should_ be safe
                    }
                    #[allow(unused_assignments)]
                    {
                        set_ix += 1;
                    }
                )*
                $(
                    match s.get_push_const() {
                        None => {
                            // TODO: definition has a push constant, but the shader does not
                        },
                        Some(spirq::Type::Struct(push_block)) => {
                            // TODO: validate offset and alignment of each block member, bigger topic
                            if std::mem::size_of::<super::$push>() != push_block.nbyte() {
                                println!("push blocks are different {} {}", std::mem::size_of::<super::$push>(),push_block.nbyte());
                                return false;
                            }
                        }
                        _ => {
                            println!("push constant block is not a struct");
                            return false;
                        }
                    }
                )?
                true
            }

            make_pipe!(@vertex_descs $vertex_inputs);
            $(
                make_pipe!(@spec_consts $spec_const);
            )?

            pub struct PipelineLayout {
                pub layout: super::super::PipelineLayout,
            }

            impl PipelineLayout {
                pub fn new(device: &Arc<super::super::Device>, $($desc: &super::$desc::DescriptorSetLayout,)*) -> PipelineLayout {
                    let layout = super::super::new_pipeline_layout(
                        Arc::clone(&device),
                        &[
                            $(
                                &$desc.layout,
                            )*
                        ],
                        &[
                            $(
                                vk::PushConstantRange {
                                    stage_flags: if IS_COMPUTE {
                                        vk::ShaderStageFlags::COMPUTE
                                    } else {
                                        vk::ShaderStageFlags::ALL_GRAPHICS // imprecise
                                    },
                                    offset: 0,
                                    size: std::mem::size_of::<super::$push>() as u32,
                                }
                            )?
                        ],
                    );

                    PipelineLayout { layout }
                }

                #[allow(unused_variables)]
                pub fn push_constants(&self, device: &super::super::Device,
                                      command_buffer: vk::CommandBuffer, $(push_constants: &super::$push)?) {
                    $(
                        unsafe {
                            let casted: &[u8] = std::slice::from_raw_parts(
                                push_constants as *const _ as *const u8, std::mem::size_of::<super::$push>(),
                            );
                            device.cmd_push_constants(
                                command_buffer,
                                self.layout.handle,
                                if IS_COMPUTE {
                                    vk::ShaderStageFlags::COMPUTE
                                } else {
                                    vk::ShaderStageFlags::ALL_GRAPHICS // imprecise
                                },
                                0,
                                casted,
                            );
                            return;
                        }
                    )?
                    #[allow(unreachable_code)]
                    {
                        panic!("PipelineLayout::push_constants() called without any push constants in the pipe");
                    }
                }

                pub fn bind_descriptor_sets(&self, device: &super::super::Device, command_buffer: vk::CommandBuffer $(, $desc: &super::$desc::DescriptorSet)*) {
                    unsafe {
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            if IS_COMPUTE {
                                vk::PipelineBindPoint::COMPUTE
                            } else {
                                vk::PipelineBindPoint::GRAPHICS
                            },
                            self.layout.handle,
                            0,
                            &[
                                $(
                                    $desc.set.handle,
                                )*
                            ],
                            &[],
                        );
                    }
                }

            }
        }
    };
}

pub struct IndirectCommands {
    pub indirect_command: [vk::DrawIndexedIndirectCommand; 2400],
}

pub type OutIndexBuffer = [[u32; 3]; 20_000_000];
pub type VertexBuffer = [[f32; 3]; 10 /* distinct meshes */ * 30_000];
pub type UVBuffer = [[f32; 2]; 10 /* distinct meshes */ * 30_000];
pub type IndexBuffer = [[u32; 3]; 10 /* distinct meshes */ * 30_000];

pub struct CameraMatrices {
    pub projection: glm::Mat4,
    pub view: glm::Mat4,
    pub position: glm::Vec4,
}

pub struct ModelMatrices {
    pub model: [glm::Mat4; 4096],
}

pub type Null = ();

make_descriptor_set!(
    model_set[
        1 => model, ModelMatrices, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE, vk::DescriptorType::UNIFORM_BUFFER
    ]
);
make_descriptor_set!(
    camera_set[
        1 => matrices, CameraMatrices, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE, vk::DescriptorType::UNIFORM_BUFFER
    ]
);
make_descriptor_set!(
    cull_set [
        1 => indirect_commands, IndirectCommands, vk::ShaderStageFlags::COMPUTE, vk::DescriptorType::STORAGE_BUFFER;
        1 => out_index_buffer, OutIndexBuffer , vk::ShaderStageFlags::COMPUTE, vk::DescriptorType::STORAGE_BUFFER;
        1 => vertex_buffer, VertexBuffer , vk::ShaderStageFlags::COMPUTE, vk::DescriptorType::STORAGE_BUFFER;
        1 => index_buffer, IndexBuffer , vk::ShaderStageFlags::COMPUTE, vk::DescriptorType::STORAGE_BUFFER
    ]
);

make_descriptor_set!(
    base_color_set [
        3072, partially bound => texture, Null, vk::ShaderStageFlags::FRAGMENT, vk::DescriptorType::COMBINED_IMAGE_SAMPLER
    ]
);

make_descriptor_set!(
    shadow_map_set [
        16, partially bound => light_data, CameraMatrices, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, vk::DescriptorType::UNIFORM_BUFFER;
        1 => shadow_maps, Null, vk::ShaderStageFlags::FRAGMENT, vk::DescriptorType::COMBINED_IMAGE_SAMPLER
    ]
);

make_descriptor_set!(
    imgui_set [
        1 => texture, Null, vk::ShaderStageFlags::FRAGMENT, vk::DescriptorType::COMBINED_IMAGE_SAMPLER
    ]
);

#[repr(C)]
pub struct GenerateWorkPushConstants {
    pub gltf_index: u32,
    pub index_count: u32,
    pub index_offset: u32,
    pub index_offset_in_output: i32,
    pub vertex_offset: i32,
}

make_pipe!(generate_work {
    compute,
    descriptors: [model_set, camera_set, cull_set],
    push_constants: GenerateWorkPushConstants,
    specialization_constants: [1 => local_workgroup_size: u32]
});

make_pipe!(depth_pipe {
    vertex_inputs: [position: vec3],
    descriptors: [model_set, camera_set]
});

make_pipe!(gltf_mesh {
    vertex_inputs: [position: vec3, normal: vec3, uv: vec2],
    descriptors: [model_set, camera_set, shadow_map_set, base_color_set]
});

#[repr(C)]
pub struct ImguiPushConstants {
    pub scale: glm::Vec2,
    pub translate: glm::Vec2,
}

make_pipe!(imgui_pipe {
    vertex_inputs: [pos: vec2, uv: vec2, col: vec4],
    descriptors: [imgui_set],
    push_constants: ImguiPushConstants
});

#[repr(C)]
pub struct DebugAABBPushConstants {
    pub center: glm::Vec3,
    pub half_extent: glm::Vec3,
}

make_pipe!(debug_aabb {
    vertex_inputs: [],
    descriptors: [camera_set],
    push_constants: DebugAABBPushConstants
});
