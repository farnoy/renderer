use super::Device;
use ash::{
    version::{DeviceV1_0, DeviceV1_2},
    vk,
};
use std::sync::Arc;

pub struct Semaphore {
    pub handle: vk::Semaphore,
    device: Arc<Device>,
}

pub struct TimelineSemaphore {
    pub handle: vk::Semaphore,
    device: Arc<Device>,
}

pub struct Fence {
    pub handle: vk::Fence,
    device: Arc<Device>,
}

pub struct Event {
    pub handle: vk::Event,
    device: Arc<Device>,
}

#[macro_export]
macro_rules! define_timeline {
    ($mod_name:ident $($name:ident),+) => {
        pub(crate) mod $mod_name {
            crate::define_timeline!(@define_const 1u64, $($name),+);
        }
    };
    (@define_const $ix:expr, $arg:ident) => {
        // last argument, round up to next highest power of 2
        pub const $arg: u64 = $ix.next_power_of_two();
        pub const MAX: u64 = $ix.next_power_of_two();
    };
    (@define_const $ix:expr, $arg0:ident, $($args:ident),*) => {
        pub const $arg0: u64 = $ix;
        crate::define_timeline!(@define_const ($ix + 1), $($args),*);
    };
}

#[macro_export]
macro_rules! timeline_value {
    ($module:ident @ last $frame_number:expr => $offset:ident) => {
        ($frame_number - 1) * $module::MAX + $module::$offset
    };
    ($module:ident @ $frame_number:expr => $offset:ident) => {
        $frame_number * $module::MAX + $module::$offset
    };
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

impl TimelineSemaphore {
    pub(super) fn new(device: &Arc<Device>, initial_value: u64) -> TimelineSemaphore {
        let mut create_type_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let create_info = vk::SemaphoreCreateInfo::builder().push_next(&mut create_type_info);
        let semaphore = unsafe { device.create_semaphore(&create_info, None).unwrap() };

        TimelineSemaphore {
            handle: semaphore,
            device: Arc::clone(device),
        }
    }

    pub fn wait(&self, value: u64) -> ash::prelude::VkResult<()> {
        let wait_ixes = &[value];
        let wait_semaphores = &[self.handle];
        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(wait_semaphores)
            .values(wait_ixes);
        unsafe {
            self.device
                .wait_semaphores(self.device.handle(), &wait_info, std::u64::MAX)
        }
    }

    pub fn value(&self) -> ash::prelude::VkResult<u64> {
        unsafe {
            self.device
                .get_semaphore_counter_value(self.device.handle(), self.handle)
        }
    }

    pub fn signal(&self, value: u64) -> ash::prelude::VkResult<()> {
        unsafe {
            self.device.signal_semaphore(
                self.device.handle(),
                &vk::SemaphoreSignalInfo::builder()
                    .semaphore(self.handle)
                    .value(value),
            )
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

impl Drop for TimelineSemaphore {
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
