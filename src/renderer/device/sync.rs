use ash::vk;
use microprofile::scope;

use super::Device;

pub(crate) struct Semaphore {
    pub(crate) handle: vk::Semaphore,
}

#[derive(PartialEq, Eq, Hash)]
pub(crate) struct TimelineSemaphore {
    pub(crate) handle: vk::Semaphore,
}

pub(crate) struct Fence {
    pub(crate) handle: vk::Fence,
}

pub(crate) struct Event {
    pub(crate) handle: vk::Event,
}

impl Semaphore {
    pub(super) fn new(device: &Device) -> Semaphore {
        let create_info = vk::SemaphoreCreateInfo::builder();
        let semaphore = unsafe { device.device.create_semaphore(&create_info, None).unwrap() };

        Semaphore { handle: semaphore }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.handle, None);
        }
        self.handle = vk::Semaphore::null();
    }
}

impl TimelineSemaphore {
    pub(super) fn new(device: &Device, initial_value: u64) -> TimelineSemaphore {
        let mut create_type_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let create_info = vk::SemaphoreCreateInfo::builder().push_next(&mut create_type_info);
        let semaphore = unsafe { device.create_semaphore(&create_info, None).unwrap() };

        TimelineSemaphore { handle: semaphore }
    }

    pub(crate) fn wait(&self, device: &Device, value: u64) -> ash::prelude::VkResult<()> {
        scope!("vk", "vkWaitSemaphores");
        let wait_ixes = &[value];
        let wait_semaphores = &[self.handle];
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(wait_semaphores)
            .values(wait_ixes);
        unsafe { device.wait_semaphores(&wait_info, std::u64::MAX) }
    }

    pub(crate) fn value(&self, device: &Device) -> ash::prelude::VkResult<u64> {
        scope!("vk", "vkGetSemaphoreCounterValue");
        unsafe { device.get_semaphore_counter_value(self.handle) }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.handle, None);
        }
        self.handle = vk::Semaphore::null();
    }
}

impl Fence {
    pub(super) fn new(device: &Device) -> Fence {
        let create_info = vk::FenceCreateInfo::builder();
        let fence = unsafe {
            device
                .device
                .create_fence(&create_info, None)
                .expect("Create fence failed.")
        };
        Fence { handle: fence }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_fence(self.handle, None);
        }
        self.handle = vk::Fence::null();
    }
}

// TODO: use this later
impl Event {
    #[allow(unused)]
    pub(super) fn new(device: &Device) -> Event {
        let create_info = vk::EventCreateInfo::builder();
        let event = unsafe {
            device
                .device
                .create_event(&create_info, None)
                .expect("Create event failed.")
        };
        Event { handle: event }
    }

    #[allow(unused)]
    pub(crate) fn signal(&self, device: &Device) {
        unsafe {
            device.set_event(self.handle).expect("failed to signal event");
        }
    }

    #[allow(unused)]
    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe {
            device.destroy_event(self.handle, None);
        }
        self.handle = vk::Event::null();
    }
}

#[cfg(debug_assertions)]
impl Drop for Semaphore {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::Semaphore::null(),
            "Semaphore not destroyed before Drop"
        );
    }
}

#[cfg(debug_assertions)]
impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::Semaphore::null(),
            "TimelineSemaphore not destroyed before Drop"
        );
    }
}

#[cfg(debug_assertions)]
impl Drop for Fence {
    fn drop(&mut self) {
        debug_assert_eq!(self.handle, vk::Fence::null(), "Fence not destroyed before Drop");
    }
}

#[cfg(debug_assertions)]
impl Drop for Event {
    fn drop(&mut self) {
        debug_assert_eq!(self.handle, vk::Event::null(), "Event not destroyed before Drop");
    }
}
