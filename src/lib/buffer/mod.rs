use ash::util::Align;
use ash::version::DeviceV1_0;
use ash::vk;
use std::default::Default;
use std::iter::{ExactSizeIterator, Iterator};
use std::mem::{align_of, size_of};
use std::ptr;
use super::command_buffer::one_time_submit_and_wait;
use super::device::AshDevice;
use super::{find_memorytype_index, ExampleBase};

#[allow(dead_code)]
pub struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl Buffer {
    pub unsafe fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn upload_from<T, I>(base: &ExampleBase, usage: vk::BufferUsageFlags, iter: &I) -> Buffer
    where
        T: Copy,
        I: ExactSizeIterator + Iterator<Item = T> + Clone,
    {
        let count = iter.len();
        let size = (count * size_of::<T>()) as u64;

        let local_buffer = unsafe {
            Buffer::create_specific_buffer(
                base,
                usage | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
                vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                size,
            )
        };

        let mut host_buffer = unsafe {
            Buffer::create_specific_buffer(
                base,
                vk::BUFFER_USAGE_TRANSFER_SRC_BIT,
                vk::MEMORY_PROPERTY_HOST_COHERENT_BIT,
                size,
            )
        };

        unsafe {
            host_buffer.store_in_memory(base, iter.clone());
        }

        one_time_submit_and_wait(base, |cb| unsafe {
            base.device.cmd_copy_buffer(
                cb,
                host_buffer.buffer(),
                local_buffer.buffer(),
                &[
                    vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size,
                    },
                ],
            )
        });

        unsafe {
            host_buffer.free(base.device.vk());
        }

        local_buffer
    }

    pub unsafe fn free(self, device: &AshDevice) {
        device.free_memory(self.memory, None);
        device.destroy_buffer(self.buffer, None);
    }

    pub unsafe fn store_in_memory<T, I>(&mut self, base: &ExampleBase, iter: I)
    where
        T: Copy,
        I: Iterator<Item = T> + Clone,
    {
        let count = iter.clone().count() as u64;
        self.store_in_memory_known_count(base, iter, count);
    }

    pub unsafe fn store_in_memory_known_count<T, I>(
        &mut self,
        base: &ExampleBase,
        iter: I,
        count: u64,
    ) where
        T: Copy,
        I: Iterator<Item = T>,
    {
        let size = count * size_of::<T>() as u64;
        let ptr = base.device
            .map_memory(self.memory, 0, size, Default::default())
            .unwrap();

        let mut align = Align::new(ptr, align_of::<T>() as u64, size);

        for (p, value) in align.iter_mut().zip(iter) {
            *p = value;
        }

        base.device.unmap_memory(self.memory);

    }

    pub unsafe fn create_specific_buffer(
        base: &ExampleBase,
        usage: vk::BufferUsageFlags,
        memory_type: vk::MemoryPropertyFlags,
        size: u64,
    ) -> Buffer {
        let buffer_create_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BufferCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            size: size,
            usage: usage | vk::BUFFER_USAGE_TRANSFER_DST_BIT,
            sharing_mode: vk::SharingMode::Exclusive,
            queue_family_index_count: 1,
            p_queue_family_indices: &base.queue_family_index,
        };

        let buffer = base.device
            .create_buffer(&buffer_create_info, None)
            .unwrap();

        let memory_req = base.device.get_buffer_memory_requirements(buffer);
        let info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MemoryAllocateInfo,
            p_next: ptr::null(),
            allocation_size: memory_req.size,
            memory_type_index: find_memorytype_index(
                &memory_req,
                &base.device_memory_properties,
                memory_type,
            ).unwrap(),
        };

        let memory = base.device.allocate_memory(&info, None).unwrap();

        base.device.bind_buffer_memory(buffer, memory, 0).unwrap();

        Buffer {
            buffer: buffer,
            memory: memory,
        }
    }
}
