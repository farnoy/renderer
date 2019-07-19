use super::Device;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub struct Semaphore {
    pub handle: vk::Semaphore,
    pub device: Arc<Device>,
}

pub struct Fence {
    pub handle: vk::Fence,
    pub device: Arc<Device>,
}

pub struct Event {
    pub handle: vk::Event,
    pub device: Arc<Device>,
}

impl Semaphore {
    pub(super) fn new(device: &Arc<Device>) -> Semaphore {
        let create_info = vk::SemaphoreCreateInfo::builder();
        let semaphore = unsafe { device.device.create_semaphore(&create_info, None).unwrap() };

        Semaphore {
            handle: semaphore,
            device: Arc::clone(device),
        }
    }
}

impl Fence {
    pub(super) fn new(device: &Arc<Device>) -> Fence {
        let create_info = vk::FenceCreateInfo::builder();
        let fence = unsafe {
            device
                .device
                .create_fence(&create_info, None)
                .expect("Create fence failed.")
        };
        Fence {
            device: Arc::clone(device),
            handle: fence,
        }
    }
}

impl Event {
    pub(super) fn new(device: &Arc<Device>) -> Event {
        let create_info = vk::EventCreateInfo::builder();
        let event = unsafe {
            device
                .device
                .create_event(&create_info, None)
                .expect("Create event failed.")
        };
        Event {
            device: Arc::clone(device),
            handle: event,
        }
    }

    #[allow(unused)]
    pub fn signal(&self) {
        unsafe {
            self.device
                .set_event(self.handle)
                .expect("failed to signal event");
        }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_semaphore(self.handle, None);
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_fence(self.handle, None);
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_event(self.handle, None);
        }
    }
}
