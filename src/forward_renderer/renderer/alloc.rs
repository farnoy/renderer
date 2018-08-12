#![allow(warnings)]

use ash::{
    self, prelude,
    version::{self, DeviceV1_0, EntryV1_0, InstanceV1_0, V1_0, V1_1},
    vk,
};
use std::{
    ffi::CStr,
    mem::{self, transmute},
    ptr,
};

type VkFlags = vk::Flags;
type VkBuffer = vk::Buffer;
type VkBufferCreateInfo = vk::BufferCreateInfo;
type VkImage = vk::Image;
type VkImageCreateInfo = vk::ImageCreateInfo;
type VkResult = vk::Result;
type VkStructureType = vk::StructureType;
type VkDeviceMemory = vk::DeviceMemory;
type VkPhysicalDevice = vk::PhysicalDevice;
type VkDevice = vk::Device;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[derive(Copy, Clone)]
pub struct VmaAllocator(*mut VmaAllocator_T);

unsafe impl Send for VmaAllocator {}
unsafe impl Sync for VmaAllocator {}

#[derive(Copy, Clone, Debug)]
pub struct VmaAllocation(*mut VmaAllocation_T);

unsafe impl Send for VmaAllocation {}
unsafe impl Sync for VmaAllocation {}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct VmaAllocationInfo {
    pub memoryType: u32,
    pub deviceMemory: VkDeviceMemory,
    pub offset: VkDeviceSize,
    pub size: VkDeviceSize,
    pub pMappedData: *mut ::std::os::raw::c_void,
    pub pUserData: *mut ::std::os::raw::c_void,
}

#[repr(C)]
pub struct VmaAllocationCreateInfo {
    pub flags: VmaAllocationCreateFlagBits,
    pub usage: VmaMemoryUsage,
    pub requiredFlags: VkMemoryPropertyFlags,
    pub preferredFlags: VkMemoryPropertyFlags,
    pub memoryTypeBits: u32,
    pub pool: VmaPool,
    pub pUserData: *mut ::std::os::raw::c_void,
}

unsafe impl Send for VmaAllocationInfo {}
unsafe impl Sync for VmaAllocationInfo {}

pub fn create(
    entry: &ash::Entry<V1_0>,
    instance: &ash::Instance<V1_1>,
    device: vk::Device,
    pdevice: vk::PhysicalDevice,
) -> prelude::VkResult<VmaAllocator> {
    let vma_functions = unsafe {
        VmaVulkanFunctions {
            vkAllocateMemory: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkAllocateMemory\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkBindBufferMemory: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkBindBufferMemory\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkBindImageMemory: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkBindImageMemory\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkCreateBuffer: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkCreateBuffer\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkCreateImage: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkCreateImage\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkDestroyBuffer: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkDestroyBuffer\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkDestroyImage: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkDestroyImage\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkFreeMemory: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkFreeMemory\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
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
            vkGetPhysicalDeviceProperties: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkGetPhysicalDeviceProperties\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkMapMemory: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkMapMemory\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
            vkUnmapMemory: transmute(
                entry.get_instance_proc_addr(
                    instance.handle(),
                    CStr::from_bytes_with_nul(b"vkUnmapMemory\0")
                        .unwrap()
                        .as_ptr(),
                ),
            ),
        }
    };
    let create_info = VmaAllocatorCreateInfo {
        flags: VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT.0
            as u32,
        device: device,
        physicalDevice: pdevice,
        preferredLargeHeapBlockSize: 0,
        pAllocationCallbacks: ptr::null(),
        pDeviceMemoryCallbacks: ptr::null(),
        frameInUseCount: 1,
        pHeapSizeLimit: ptr::null(),
        pVulkanFunctions: &vma_functions,
    };
    let mut allocator: VmaAllocator = VmaAllocator(ptr::null_mut());
    let err_code =
        unsafe { vmaCreateAllocator(&create_info as *const _, &mut allocator as *mut _) };
    match err_code {
        vk::Result::SUCCESS => Ok(allocator),
        _ => Err(err_code),
    }
}

pub fn destroy(allocator: VmaAllocator) {
    unsafe { vmaDestroyAllocator(allocator) }
}

pub fn create_buffer(
    allocator: VmaAllocator,
    buffer_create_info: &vk::BufferCreateInfo,
    allocation_create_info: &VmaAllocationCreateInfo,
) -> prelude::VkResult<(vk::Buffer, VmaAllocation, VmaAllocationInfo)> {
    unsafe {
        let mut buffer = vk::Buffer::null();
        let mut allocation: VmaAllocation = VmaAllocation(ptr::null_mut());
        let mut info: VmaAllocationInfo = mem::zeroed();
        let err_code = vmaCreateBuffer(
            allocator,
            buffer_create_info as *const _,
            allocation_create_info as *const _,
            &mut buffer as *mut _,
            &mut allocation as *mut _,
            &mut info as *mut _,
        );
        match err_code {
            vk::Result::SUCCESS => Ok((buffer, allocation, info)),
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
        let mut image = vk::Image::null();
        let mut allocation: VmaAllocation = VmaAllocation(ptr::null_mut());
        let mut info: VmaAllocationInfo = mem::zeroed();
        let err_code = vmaCreateImage(
            allocator,
            image_create_info as *const _,
            allocation_create_info as *const _,
            &mut image as *mut _,
            &mut allocation as *mut _,
            &mut info as *mut _,
        );
        match err_code {
            vk::Result::SUCCESS => Ok((image, allocation, info)),
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
