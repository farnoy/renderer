#![allow(warnings)]

use ash::{prelude, vk};
use std::ptr;

type VkFlags = vk::Flags;
type VkBuffer = vk::Buffer;
type VkResult = vk::Result;
type VkStructureType = vk::StructureType;
type VkDeviceMemory = vk::DeviceMemory;
type VkPhysicalDevice = vk::PhysicalDevice;
type VkDevice = vk::Device;
type VkImage = vk::Image;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[derive(Copy, Clone)]
pub struct VmaAllocator(*mut VmaAllocator_T);

unsafe impl Send for VmaAllocator {}

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

pub fn set_current_frame_index(allocator: VmaAllocator, index: u32) {
    unsafe { vmaSetCurrentFrameIndex(allocator, index) }
}
