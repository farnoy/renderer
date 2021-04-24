use ash::vk;
use mem::transmute;
use std::{
    ffi::c_void,
    mem,
    ops::{Deref, DerefMut, Index, IndexMut, Range, RangeFull},
    ptr, slice,
};

use super::super::alloc;

// Wrapper to safely map buffer contents
pub(crate) struct MappedBuffer<'a, T> {
    ptr: &'a mut [T],
    allocator: alloc::VmaAllocator,
    allocation: alloc::VmaAllocation,
}

pub(crate) struct MappedStaticBuffer<'a, T> {
    ptr: &'a mut T,
    allocator: alloc::VmaAllocator,
    allocation: alloc::VmaAllocation,
}

impl<'a, T> MappedBuffer<'a, T> {
    pub(crate) fn import(
        allocator: alloc::VmaAllocator,
        allocation: alloc::VmaAllocation,
        allocation_info: &alloc::VmaAllocationInfo,
    ) -> ash::prelude::VkResult<Self> {
        assert!(allocation_info.size as usize / mem::size_of::<T>() > 0);
        unsafe {
            let mut ptr: *mut c_void = ptr::null_mut();
            match alloc::vmaMapMemory(allocator, allocation, &mut ptr) {
                vk::Result::SUCCESS => Ok(MappedBuffer {
                    ptr: slice::from_raw_parts_mut(ptr as *mut T, allocation_info.size as usize / mem::size_of::<T>()),
                    allocator,
                    allocation,
                }),
                res => Err(res),
            }
        }
    }
}

impl<'a, T> MappedStaticBuffer<'a, T> {
    pub(crate) fn import(
        allocator: alloc::VmaAllocator,
        allocation: alloc::VmaAllocation,
    ) -> ash::prelude::VkResult<Self> {
        let allocation_info = alloc::get_allocation_info(allocator, allocation);
        assert!(allocation_info.size as usize == mem::size_of::<T>());
        unsafe {
            let mut ptr: *mut c_void = ptr::null_mut();
            match alloc::vmaMapMemory(allocator, allocation, &mut ptr) {
                vk::Result::SUCCESS => Ok(MappedStaticBuffer {
                    ptr: transmute::<*mut c_void, &mut T>(ptr),
                    allocator,
                    allocation,
                }),
                res => Err(res),
            }
        }
    }
}

impl<'a, T> Index<usize> for MappedBuffer<'a, T> {
    type Output = T;

    fn index(&self, ix: usize) -> &T {
        &self.ptr[ix]
    }
}

impl<'a, T> Index<Range<usize>> for MappedBuffer<'a, T> {
    type Output = [T];

    fn index(&self, ix: Range<usize>) -> &[T] {
        &self.ptr[ix]
    }
}

impl<'a, T> Index<RangeFull> for MappedBuffer<'a, T> {
    type Output = [T];

    fn index(&self, ix: RangeFull) -> &[T] {
        &self.ptr[ix]
    }
}

impl<'a, T> IndexMut<usize> for MappedBuffer<'a, T> {
    fn index_mut(&mut self, ix: usize) -> &mut T {
        &mut self.ptr[ix]
    }
}

impl<'a, T> IndexMut<Range<usize>> for MappedBuffer<'a, T> {
    fn index_mut(&mut self, ix: Range<usize>) -> &mut [T] {
        &mut self.ptr[ix]
    }
}

impl<'a, T> IndexMut<RangeFull> for MappedBuffer<'a, T> {
    fn index_mut(&mut self, _ix: RangeFull) -> &mut [T] {
        &mut *self.ptr
    }
}

impl<'a, T> Deref for MappedStaticBuffer<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<'a, T> DerefMut for MappedStaticBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ptr
    }
}

impl<'a, T> Drop for MappedBuffer<'a, T> {
    fn drop(&mut self) {
        unsafe {
            alloc::vmaFlushAllocation(self.allocator, self.allocation, 0, vk::WHOLE_SIZE);
            alloc::vmaUnmapMemory(self.allocator, self.allocation);
        }
    }
}

impl<'a, T> Drop for MappedStaticBuffer<'a, T> {
    fn drop(&mut self) {
        unsafe {
            alloc::vmaFlushAllocation(self.allocator, self.allocation, 0, vk::WHOLE_SIZE);
            alloc::vmaUnmapMemory(self.allocator, self.allocation);
        }
    }
}
