#![allow(
    non_snake_case,
    non_upper_case_globals,
    non_camel_case_types,
    unused,
    clippy::all,
    unreachable_pub
)]

use std::{
    ffi::CStr,
    mem::{self, transmute, MaybeUninit},
    ptr,
};

use ash::{self, prelude, vk};
pub use renderer_vma::*;

pub fn create(
    entry: &ash::Entry,
    instance: &ash::Instance,
    device: vk::Device,
    pdevice: vk::PhysicalDevice,
) -> prelude::VkResult<VmaAllocator> {
    let vma_functions = unsafe {
        VmaVulkanFunctions {
            vkAllocateMemory: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkAllocateMemory\0").unwrap().as_ptr(),
            )),
            vkBindBufferMemory: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkBindBufferMemory\0").unwrap().as_ptr(),
            )),
            vkBindBufferMemory2KHR: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkBindBufferMemory2\0").unwrap().as_ptr(),
            )),
            vkBindImageMemory: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkBindImageMemory\0").unwrap().as_ptr(),
            )),
            vkBindImageMemory2KHR: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkBindImageMemory2\0").unwrap().as_ptr(),
            )),
            vkCmdCopyBuffer: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkCmdCopyBuffer\0").unwrap().as_ptr(),
            )),
            vkCreateBuffer: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkCreateBuffer\0").unwrap().as_ptr(),
            )),
            vkCreateImage: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkCreateImage\0").unwrap().as_ptr(),
            )),
            vkDestroyBuffer: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkDestroyBuffer\0").unwrap().as_ptr(),
            )),
            vkDestroyImage: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkDestroyImage\0").unwrap().as_ptr(),
            )),
            vkFreeMemory: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkFreeMemory\0").unwrap().as_ptr(),
            )),
            vkGetBufferMemoryRequirements: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkGetBufferMemoryRequirements\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkGetBufferMemoryRequirements2KHR: transmute(
                instance.get_device_proc_addr(
                    device,
                    CStr::from_bytes_with_nul(b"vkGetBufferMemoryRequirements2\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkGetImageMemoryRequirements: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkGetImageMemoryRequirements\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkGetImageMemoryRequirements2KHR: transmute(
                instance.get_device_proc_addr(
                    device,
                    CStr::from_bytes_with_nul(b"vkGetImageMemoryRequirements2\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkGetPhysicalDeviceMemoryProperties: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkGetPhysicalDeviceMemoryProperties\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkGetPhysicalDeviceMemoryProperties2KHR: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkGetPhysicalDeviceMemoryProperties2\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkGetPhysicalDeviceProperties: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkGetPhysicalDeviceProperties\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkMapMemory: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkMapMemory\0").unwrap().as_ptr(),
            )),
            vkUnmapMemory: transmute(entry.get_instance_proc_addr(
                instance.handle(),
                CStr::from_bytes_with_nul(b"vkUnmapMemory\0").unwrap().as_ptr(),
            )),
            vkFlushMappedMemoryRanges: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkFlushMappedMemoryRanges\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkInvalidateMappedMemoryRanges: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkInvalidateMappedMemoryRanges\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
        }
    };
    let create_info = VmaAllocatorCreateInfo {
        flags: VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT.0 as u32
            | VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT.0 as u32,
        vulkanApiVersion: vk::API_VERSION_1_2,
        instance: instance.handle(),
        device,
        physicalDevice: pdevice,
        preferredLargeHeapBlockSize: 0,
        pAllocationCallbacks: ptr::null(),
        pDeviceMemoryCallbacks: ptr::null(),
        frameInUseCount: 1,
        pHeapSizeLimit: ptr::null(),
        pVulkanFunctions: &vma_functions,
        pRecordSettings: ptr::null(),
    };
    let mut allocator: VmaAllocator = VmaAllocator(ptr::null_mut());
    let err_code = unsafe { vmaCreateAllocator(&create_info as *const _, &mut allocator as *mut _) };
    match err_code {
        vk::Result::SUCCESS => Ok(allocator),
        _ => Err(err_code),
    }
}

pub fn destroy(allocator: VmaAllocator) {
    unsafe { vmaDestroyAllocator(allocator) }
}

pub fn stats(allocator: VmaAllocator) -> VmaStats {
    let mut stats = mem::MaybeUninit::uninit();
    unsafe {
        vmaCalculateStats(allocator, stats.as_mut_ptr());
        stats.assume_init()
    }
}

pub fn create_buffer(
    allocator: VmaAllocator,
    buffer_create_info: &vk::BufferCreateInfo,
    allocation_create_info: &VmaAllocationCreateInfo,
) -> prelude::VkResult<(vk::Buffer, VmaAllocation, VmaAllocationInfo)> {
    unsafe {
        let mut buffer = MaybeUninit::<vk::Buffer>::uninit();
        let mut allocation = MaybeUninit::<VmaAllocation>::uninit();
        let mut info = MaybeUninit::<VmaAllocationInfo>::uninit();
        let err_code = vmaCreateBuffer(
            allocator,
            buffer_create_info as *const _,
            allocation_create_info as *const _,
            buffer.as_mut_ptr(),
            allocation.as_mut_ptr(),
            info.as_mut_ptr(),
        );
        match err_code {
            vk::Result::SUCCESS => Ok((buffer.assume_init(), allocation.assume_init(), info.assume_init())),
            _ => Err(err_code),
        }
    }
}

pub fn destroy_buffer(allocator: VmaAllocator, buffer: vk::Buffer, allocation: VmaAllocation) {
    unsafe { vmaDestroyBuffer(allocator, buffer, allocation) }
}

pub fn create_image(
    allocator: VmaAllocator,
    image_create_info: &vk::ImageCreateInfo,
    allocation_create_info: &VmaAllocationCreateInfo,
) -> prelude::VkResult<(vk::Image, VmaAllocation, VmaAllocationInfo)> {
    unsafe {
        let mut image = MaybeUninit::<vk::Image>::uninit();
        let mut allocation = MaybeUninit::<VmaAllocation>::uninit();
        let mut info = MaybeUninit::<VmaAllocationInfo>::uninit();
        let err_code = vmaCreateImage(
            allocator,
            image_create_info as *const _,
            allocation_create_info as *const _,
            image.as_mut_ptr(),
            allocation.as_mut_ptr(),
            info.as_mut_ptr(),
        );
        match err_code {
            vk::Result::SUCCESS => Ok((image.assume_init(), allocation.assume_init(), info.assume_init())),
            _ => Err(err_code),
        }
    }
}

pub fn destroy_image(allocator: VmaAllocator, image: vk::Image, allocation: VmaAllocation) {
    unsafe {
        vmaDestroyImage(allocator, image, allocation);
    }
}

pub fn set_current_frame_index(allocator: VmaAllocator, index: u32) {
    unsafe { vmaSetCurrentFrameIndex(allocator, index) }
}

pub fn get_allocation_info(allocator: VmaAllocator, allocation: VmaAllocation) -> VmaAllocationInfo {
    let mut info = mem::MaybeUninit::uninit();

    unsafe {
        vmaGetAllocationInfo(allocator, allocation, info.as_mut_ptr());
        info.assume_init()
    }
}
