use crate::renderer::{ImageIndex, RenderFrame};

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

pub trait Timeline {
    const MAX: u64;
}

pub trait TimelineStage<T: Timeline> {
    const VALUE: u64;

    /*
    fn advance_renderpass_2<N, F>(
        self,
        at: Attachments2<
            <Self as HasRenderTarget<0>>::Layout,
            <Self as HasRenderTarget<1>>::Layout,
        >,
        f: F,
    ) -> (
        N,
        Attachments2<<N as HasRenderTarget<0>>::Layout, <N as HasRenderTarget<1>>::Layout>,
    )
    where
        Self: HasRenderTarget<0> + HasRenderTarget<1> + Sized,
        N: SuccessorStage<T, Previous = Self> + HasRenderTarget<0> + HasRenderTarget<1>,
        F: FnOnce(
            Self,
            Attachments2<<Self as HasRenderTarget<0>>::Layout, <Self as HasRenderTarget<1>>::Layout>,
        ) -> (
            N,
            Attachments2<<N as HasRenderTarget<0>>::Layout, <N as HasRenderTarget<1>>::Layout>,
        ),
    {
        f(self, at)
    }
    */
}

pub trait SuccessorStage<T: Timeline> {
    type Previous: TimelineStage<T>;
}

#[macro_export]
macro_rules! define_timeline {
    ($mod_name:ident $($name:ident $(($ty:path))? ),+) => {
        pub mod $mod_name {
            crate::define_timeline!(@define_consts 1u64, previous, $($name $(($ty))? ),+);

            pub struct Timeline;

            impl $crate::renderer::device::Timeline for Timeline {
                const MAX: u64 = self::MAX;
            }
        }
    };
    (@define_consts $ix:expr, previous $($prev:ident)?, $arg0:ident $(($ty:path))?, $($args:ident $(($types:path))? ),+) => {
        crate::define_timeline!(@define_const $ix, previous $($prev)?, $arg0 $(($ty))? );

        crate::define_timeline!(@define_consts ($ix + 1), previous $arg0, $($args $(($types))? ),+);
    };
    (@define_consts $ix:expr, previous $($prev:ident)?, $arg:ident $(($ty:path))? ) => {
        // last argument, round up to next highest power of 2
        crate::define_timeline!(@define_const $ix.next_power_of_two(), previous $($prev)?, $arg $(($ty))? );

        #[allow(unused)]
        pub const MAX: u64 = $ix.next_power_of_two();
    };
    (@define_const $ix:expr, previous $($prev:ident)?, $arg:ident$(($argty:path))?) => {
        #[allow(unused)]
        #[derive(Debug)]
        pub struct $arg $((pub $argty))?;
        impl $crate::renderer::device::TimelineStage<Timeline> for $arg {
            const VALUE: u64 = $ix;
        }

        $(
        impl $crate::renderer::device::SuccessorStage<Timeline> for $arg {
            type Previous = $prev;
        }
        )?
    };
}

pub fn timeline_value_last<T: Timeline, S: TimelineStage<T>>(frame_number: u64) -> u64 {
    timeline_value::<T, S>(frame_number - 1)
}

pub fn timeline_value<T: Timeline, S: TimelineStage<T>>(frame_number: u64) -> u64 {
    frame_number * T::MAX + S::VALUE
}

pub fn timeline_value_previous<T: Timeline, S: TimelineStage<T>>(
    image_index: &ImageIndex,
    renderer: &RenderFrame,
) -> u64 {
    let frame_number = renderer.previous_frame_number_for_swapchain_index[image_index.0 as usize];
    timeline_value::<T, S>(frame_number)
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
        unsafe { self.device.wait_semaphores(&wait_info, std::u64::MAX) }
    }

    pub fn value(&self) -> ash::prelude::VkResult<u64> {
        unsafe { self.device.get_semaphore_counter_value(self.handle) }
    }

    pub fn signal(&self, value: u64) -> ash::prelude::VkResult<()> {
        unsafe {
            self.device.signal_semaphore(
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
