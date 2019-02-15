use ash::vk;
use std::{
    ffi::c_void,
    mem,
    ops::{Index, IndexMut, Range, RangeFull},
    ptr, slice,
};

use super::super::alloc;

// Wrapper to safely map buffer contents
pub struct MappedBuffer<'a, T> {
    ptr: &'a mut [T],
    allocator: alloc::VmaAllocator,
    allocation: alloc::VmaAllocation,
}

impl<'a, T> MappedBuffer<'a, T> {
    pub fn import(
        allocator: alloc::VmaAllocator,
        allocation: alloc::VmaAllocation,
        allocation_info: &alloc::VmaAllocationInfo,
    ) -> ash::prelude::VkResult<Self> {
        assert!(allocation_info.size as usize / mem::size_of::<T>() > 0);
        unsafe {
            let mut ptr: *mut c_void = ptr::null_mut();
            match alloc::vmaMapMemory(allocator, allocation, &mut ptr) {
                vk::Result::SUCCESS => Ok(MappedBuffer {
                    ptr: slice::from_raw_parts_mut(
                        ptr as *mut T,
                        allocation_info.size as usize / mem::size_of::<T>(),
                    ),
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
    fn index_mut(&mut self, ix: RangeFull) -> &mut [T] {
        &mut self.ptr[ix]
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
