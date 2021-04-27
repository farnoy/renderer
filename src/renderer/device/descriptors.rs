use ash::{version::DeviceV1_0, vk};

use super::Device;

pub(crate) struct DescriptorPool {
    pub(crate) handle: vk::DescriptorPool,
}

pub(crate) struct DescriptorSetLayout {
    pub(crate) handle: vk::DescriptorSetLayout,
}

pub(crate) struct DescriptorSet {
    pub(crate) handle: vk::DescriptorSet,
}

impl DescriptorPool {
    pub(super) fn new(device: &Device, max_sets: u32, pool_sizes: &[vk::DescriptorPoolSize]) -> DescriptorPool {
        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(
                vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET | vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
            )
            .max_sets(max_sets)
            .pool_sizes(pool_sizes);

        let handle = unsafe { device.device.create_descriptor_pool(&create_info, None).unwrap() };

        DescriptorPool { handle }
    }

    pub(crate) fn allocate_set(&self, device: &Device, layout: &DescriptorSetLayout) -> DescriptorSet {
        let layouts = &[layout.handle];
        let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.handle)
            .set_layouts(layouts);

        let mut new_descriptor_sets = unsafe { device.allocate_descriptor_sets(&desc_alloc_info).unwrap() };
        let handle = new_descriptor_sets.remove(0);

        DescriptorSet { handle }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe { device.destroy_descriptor_pool(self.handle, None) }
        self.handle = vk::DescriptorPool::null();
    }
}

impl DescriptorSetLayout {
    pub(super) fn new(device: &Device, create_info: &vk::DescriptorSetLayoutCreateInfo) -> DescriptorSetLayout {
        let handle = unsafe { device.device.create_descriptor_set_layout(create_info, None).unwrap() };

        DescriptorSetLayout { handle }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe { device.device.destroy_descriptor_set_layout(self.handle, None) }
        self.handle = vk::DescriptorSetLayout::null();
    }
}

impl DescriptorSet {
    pub(crate) fn destroy(mut self, pool: &DescriptorPool, device: &Device) {
        unsafe {
            device.free_descriptor_sets(pool.handle, &[self.handle]).unwrap();
        }
        self.handle = vk::DescriptorSet::null();
    }
}

#[cfg(debug_assertions)]
impl Drop for DescriptorPool {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::DescriptorPool::null(),
            "DescriptorPool not destroyed before Drop"
        );
    }
}

#[cfg(debug_assertions)]
impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::DescriptorSetLayout::null(),
            "DescriptorSetLayout not destroyed before Drop"
        );
    }
}

#[cfg(debug_assertions)]
impl Drop for DescriptorSet {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::DescriptorSet::null(),
            "DescriptorSet not destroyed before Drop"
        );
    }
}
