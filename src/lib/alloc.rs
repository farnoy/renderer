#![allow(warnings)]

use ash::{prelude, vk};
use std::{mem, ptr};

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

unsafe impl Send for VmaAllocationInfo {}
unsafe impl Sync for VmaAllocationInfo {}

pub fn create(device: vk::Device, pdevice: vk::PhysicalDevice) -> prelude::VkResult<VmaAllocator> {
    let create_info = VmaAllocatorCreateInfo {
        flags: 0,
        device: device,
        physicalDevice: pdevice,
        preferredLargeHeapBlockSize: 0,
        pAllocationCallbacks: ptr::null(),
        pDeviceMemoryCallbacks: ptr::null(),
        frameInUseCount: 1,
        pHeapSizeLimit: ptr::null(),
        pVulkanFunctions: ptr::null(),
    };
    let mut allocator: VmaAllocator = VmaAllocator(ptr::null_mut());
    let err_code =
        unsafe { vmaCreateAllocator(&create_info as *const _, &mut allocator as *mut _) };
    match err_code {
        vk::Result::Success => Ok(allocator),
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
            vk::Result::Success => Ok((buffer, allocation, info)),
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
            vk::Result::Success => Ok((image, allocation, info)),
            _ => Err(err_code),
        }
    }
}

pub fn set_current_frame_index(allocator: VmaAllocator, index: u32) {
    unsafe { vmaSetCurrentFrameIndex(allocator, index) }
}
