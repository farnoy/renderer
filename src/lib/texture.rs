use ash::version::DeviceV1_0;
use ash::vk;
use std::mem::size_of;
use image;
use std::path::Path;
use std::ptr;

use super::buffer::Buffer;
use super::command_buffer::one_time_submit_and_wait;
use super::device::AshDevice;
use super::{find_memorytype_index, ExampleBase};

#[allow(dead_code)]
pub struct Texture {
    pub image: vk::Image,
    memory: vk::DeviceMemory,
}

impl Texture {
    pub fn load<P: AsRef<Path>>(base: &ExampleBase, path: P, usage: vk::ImageUsageFlags, format: vk::Format) -> Texture {
        let loaded = image::open(path).unwrap().to_rgba();
        Texture::load_from_image(base, &loaded, usage, format)
    }

    pub fn load_from_memory(base: &ExampleBase, slice: &[u8], usage: vk::ImageUsageFlags, format: vk::Format) -> Texture {
        let loaded = image::load_from_memory(slice).unwrap().to_rgba();
        Texture::load_from_image(base, &loaded, usage, format)
    }

    pub fn load_from_image(base: &ExampleBase, loaded: &image::RgbaImage, usage: vk::ImageUsageFlags, format: vk::Format) -> Texture {
        let (w, h) = loaded.dimensions();
        let size = (loaded.len() * size_of::<image::Rgba<u8>>()) as u64;
        let extent = vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        };

        let host_buffer = unsafe {
            let mut buf = Buffer::create_specific_buffer(
                base,
                vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
                vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
                size,
            );

            buf.store_in_memory_known_count::<image::Rgba<u8>, _>(base, loaded.pixels().cloned(), loaded.len() as u64);

            buf
        };

        let image = unsafe {
            let info = vk::ImageCreateInfo {
                s_type: vk::StructureType::ImageCreateInfo,
                p_next: ptr::null(),
                flags: Default::default(),
                image_type: vk::ImageType::Type2d,
                format,
                extent: extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SAMPLE_COUNT_1_BIT,
                tiling: vk::ImageTiling::Linear,
                usage: usage | vk::IMAGE_USAGE_TRANSFER_DST_BIT,
                sharing_mode: vk::SharingMode::Exclusive,
                queue_family_index_count: 1,
                p_queue_family_indices: &base.queue_family_index,
                initial_layout: vk::ImageLayout::Undefined,
            };

            base.device.create_image(&info, None).unwrap()
        };

        let memory_req = base.device.get_image_memory_requirements(image);

        let memory = unsafe {
            let info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MemoryAllocateInfo,
                p_next: ptr::null(),
                allocation_size: memory_req.size,
                memory_type_index: find_memorytype_index(
                    &memory_req,
                    &base.device_memory_properties,
                    vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ).unwrap(),
            };

            base.device.allocate_memory(&info, None).unwrap()
        };

        unsafe {
            base.device.bind_image_memory(image, memory, 0).unwrap();
        }

        one_time_submit_and_wait(base, |cb| unsafe {
            base.device.cmd_pipeline_barrier(
                cb,
                vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk::PIPELINE_STAGE_TRANSFER_BIT,
                Default::default(),
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier {
                        s_type: vk::StructureType::ImageMemoryBarrier,
                        p_next: ptr::null(),
                        src_access_mask: Default::default(),
                        dst_access_mask: vk::ACCESS_TRANSFER_WRITE_BIT,
                        old_layout: vk::ImageLayout::Undefined,
                        new_layout: vk::ImageLayout::TransferDstOptimal,
                        src_queue_family_index: base.queue_family_index,
                        dst_queue_family_index: base.queue_family_index,
                        image: image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::IMAGE_ASPECT_COLOR_BIT,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                    },
                ],
            );
            base.device.cmd_copy_buffer_to_image(
                cb,
                host_buffer.vk(),
                image,
                vk::ImageLayout::TransferDstOptimal,
                &[
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::IMAGE_ASPECT_COLOR_BIT,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: extent,
                    },
                ],
            )
        });

        unsafe {
            host_buffer.free(&base.device);
        }

        Texture {
            image: image,
            memory: memory,
        }
    }

    pub unsafe fn free(self, device: &AshDevice) {
        device.free_memory(self.memory, None);
        device.destroy_image(self.image, None);
    }
}
