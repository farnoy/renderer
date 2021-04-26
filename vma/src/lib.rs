#![allow(
    non_snake_case,
    non_upper_case_globals,
    non_camel_case_types,
    unused,
    clippy::all,
    unreachable_pub
)]

use ash::vk;

type VkFlags = vk::Flags;
type VkBuffer = vk::Buffer;
type VkBufferCreateInfo = vk::BufferCreateInfo;
type VkImage = vk::Image;
type VkImageCreateInfo = vk::ImageCreateInfo;
type VkInstance = vk::Instance;
type VkResult = vk::Result;
type VkStructureType = vk::StructureType;
type VkDeviceMemory = vk::DeviceMemory;
type VkMemoryRequirements = vk::MemoryRequirements;
type VkMemoryRequirements2 = vk::MemoryRequirements2;
type VkPhysicalDevice = vk::PhysicalDevice;
type VkDevice = vk::Device;
type VkDeviceSize = vk::DeviceSize;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[derive(Copy, Clone)]
#[repr(C)]
pub struct VmaAllocator(pub *mut VmaAllocator_T);

unsafe impl Send for VmaAllocator {}
unsafe impl Sync for VmaAllocator {}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct VmaAllocation(pub *mut VmaAllocation_T);

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
