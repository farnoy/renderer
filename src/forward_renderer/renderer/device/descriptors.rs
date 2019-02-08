use super::Device;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub struct DescriptorPool {
    pub handle: vk::DescriptorPool,
    pub device: Arc<Device>,
}

pub struct DescriptorSetLayout {
    pub handle: vk::DescriptorSetLayout,
    pub device: Arc<Device>,
}

pub struct DescriptorSet {
    pub handle: vk::DescriptorSet,
    pub pool: Arc<DescriptorPool>,
}

impl DescriptorPool {
    pub(super) fn new(
        device: &Arc<Device>,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> DescriptorPool {
        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(max_sets)
            .pool_sizes(pool_sizes);

        let handle = unsafe {
            device
                .device
                .create_descriptor_pool(&create_info, None)
                .unwrap()
        };

        DescriptorPool {
            handle,
            device: Arc::clone(device),
        }
    }

    pub fn allocate_set(self: &Arc<Self>, layout: &DescriptorSetLayout) -> DescriptorSet {
        let layouts = &[layout.handle];
        let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.handle)
            .set_layouts(layouts);

        let mut new_descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&desc_alloc_info)
                .unwrap()
        };
        let handle = new_descriptor_sets.remove(0);

        DescriptorSet {
            handle,
            pool: Arc::clone(self),
        }
    }
}

impl DescriptorSetLayout {
    pub(super) fn new(
        device: &Arc<Device>,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> DescriptorSetLayout {
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        let handle = unsafe {
            device
                .device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        };

        DescriptorSetLayout {
            handle,
            device: Arc::clone(device),
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_descriptor_pool(self.handle, None)
        }
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .destroy_descriptor_set_layout(self.handle, None)
        }
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe {
            self.pool
                .device
                .free_descriptor_sets(self.pool.handle, &[self.handle])
        }
    }
}
