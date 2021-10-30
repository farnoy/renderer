use std::{
    ffi::c_void,
    mem,
    ops::{Deref, DerefMut, Index, IndexMut, Range, RangeFull},
    ptr, slice,
};

use ash::vk;
use mem::transmute;

use super::alloc;

// Wrapper to safely map buffer contents
pub(crate) struct MappedBuffer<'a, T> {
    ptr: &'a mut [T],
    allocator: alloc::VmaAllocator,
    allocation: alloc::VmaAllocation,
}

#[must_use]
pub(crate) struct MappedStaticBuffer<'a, T> {
    ptr: &'a mut T,
    allocator: alloc::VmaAllocator,
    allocation: alloc::VmaAllocation,
}

impl<T> MappedBuffer<'_, T> {
    pub(super) fn import(
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

impl<T> MappedStaticBuffer<'_, T> {
    pub(super) fn import(
        allocator: alloc::VmaAllocator,
        allocation: alloc::VmaAllocation,
    ) -> ash::prelude::VkResult<Self> {
        let allocation_info = alloc::get_allocation_info(allocator, allocation);
        assert!(allocation_info.size as usize >= mem::size_of::<T>());
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

    pub(crate) fn unmap_used_range(mut self, range: Range<vk::DeviceSize>) {
        unsafe {
            alloc::vmaFlushAllocation(self.allocator, self.allocation, range.start, range.end)
                .result()
                .unwrap();
            alloc::vmaUnmapMemory(self.allocator, self.allocation);
        }
        self.allocation = alloc::VmaAllocation(ptr::null_mut());
    }
}

impl<T> Index<usize> for MappedBuffer<'_, T> {
    type Output = T;

    fn index(&self, ix: usize) -> &T {
        &self.ptr[ix]
    }
}

impl<T> Index<Range<usize>> for MappedBuffer<'_, T> {
    type Output = [T];

    fn index(&self, ix: Range<usize>) -> &[T] {
        &self.ptr[ix]
    }
}

impl<T> Index<RangeFull> for MappedBuffer<'_, T> {
    type Output = [T];

    fn index(&self, ix: RangeFull) -> &[T] {
        &self.ptr[ix]
    }
}

impl<T> IndexMut<usize> for MappedBuffer<'_, T> {
    fn index_mut(&mut self, ix: usize) -> &mut T {
        &mut self.ptr[ix]
    }
}

impl<T> IndexMut<Range<usize>> for MappedBuffer<'_, T> {
    fn index_mut(&mut self, ix: Range<usize>) -> &mut [T] {
        &mut self.ptr[ix]
    }
}

impl<T> IndexMut<RangeFull> for MappedBuffer<'_, T> {
    fn index_mut(&mut self, _ix: RangeFull) -> &mut [T] {
        &mut *self.ptr
    }
}

impl<T> Deref for MappedStaticBuffer<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<T> DerefMut for MappedStaticBuffer<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ptr
    }
}

impl<T> Drop for MappedBuffer<'_, T> {
    fn drop(&mut self) {
        unsafe {
            alloc::vmaFlushAllocation(self.allocator, self.allocation, 0, vk::WHOLE_SIZE)
                .result()
                .unwrap();
            alloc::vmaUnmapMemory(self.allocator, self.allocation);
        }
    }
}

impl<T> Drop for MappedStaticBuffer<'_, T> {
    fn drop(&mut self) {
        if self.allocation.0 != ptr::null_mut() {
            unsafe {
                alloc::vmaFlushAllocation(self.allocator, self.allocation, 0, vk::WHOLE_SIZE)
                    .result()
                    .unwrap();
                alloc::vmaUnmapMemory(self.allocator, self.allocation);
            }
        }
    }
}
