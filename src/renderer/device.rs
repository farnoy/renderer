#[cfg(target = "windows")]
use ash::extensions::Win32Surface;
use ash::{
    self, extensions,
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use parking_lot::Mutex;
use std::{ffi::CStr, ops::Deref, sync::Arc};

mod buffer;
mod commands;
mod descriptors;
mod double_buffered;
mod image;
mod mapping;
mod sync;

use super::{alloc, Instance, Surface};

pub use self::{buffer::*, commands::*, descriptors::*, double_buffered::*, image::*, sync::*};

type AshDevice = ash::Device;

pub struct Device {
    pub(super) device: AshDevice,
    instance: Arc<Instance>,
    pub(super) physical_device: vk::PhysicalDevice,
    allocator: alloc::VmaAllocator,
    graphics_queue_family: u32,
    compute_queue_family: u32,
    pub(super) graphics_queue: Mutex<vk::Queue>,
    pub(super) compute_queues: Vec<Mutex<vk::Queue>>,
    pub get_semaphore_counter_value: vk::PFN_vkGetSemaphoreCounterValue,
    pub wait_semaphores: vk::PFN_vkWaitSemaphores,
    pub signal_semaphore: vk::PFN_vkSignalSemaphore,
    // pub _transfer_queue: Arc<Mutex<vk::Queue>>,
}

pub enum QueueType {
    Graphics,
    Compute,
}

impl Device {
    pub fn new(instance: &Arc<Instance>, surface: &Surface) -> Result<Device, vk::Result> {
        let Instance { ref entry, .. } = **instance;

        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Physical device error")
        };

        let physical_device = pdevices[0];
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let graphics_queue_family = {
            queue_families
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| unsafe {
                    let supports_graphic_and_surface =
                        info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && surface
                                .ext
                                .get_physical_device_surface_support(
                                    physical_device,
                                    ix as u32,
                                    surface.surface,
                                )
                                .unwrap();
                    if supports_graphic_and_surface {
                        Some(ix as u32)
                    } else {
                        None
                    }
                })
                .next()
                .unwrap()
        };
        let compute_queues_spec = {
            queue_families
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| {
                    if info.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    {
                        Some((ix as u32, info.queue_count))
                    } else {
                        None
                    }
                })
                .next()
        };
        let queues = match compute_queues_spec {
            Some((compute_queue_family, compute_queue_len)) => vec![
                (graphics_queue_family, 1),
                (compute_queue_family, compute_queue_len),
            ],
            None => vec![(graphics_queue_family, 1)],
        };
        let device = {
            // static RASTER_ORDER: &str = "VK_AMD_rasterization_order\0";
            let timeline_semaphore_name = b"VK_KHR_timeline_semaphore\0";
            let device_extension_names_raw = [
                extensions::khr::Swapchain::name().as_ptr(),
                vk::ExtDescriptorIndexingFn::name().as_ptr(),
                unsafe { CStr::from_bytes_with_nul_unchecked(timeline_semaphore_name).as_ptr() },
            ];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                sampler_anisotropy: 1,
                depth_bounds: 0,
                multi_draw_indirect: 1,
                vertex_pipeline_stores_and_atomics: 1,
                robust_buffer_access: 1,
                fill_mode_non_solid: 1,
                draw_indirect_first_instance: 1,
                shader_storage_buffer_array_dynamic_indexing: 1,
                ..Default::default()
            };
            let mut timeline_semaphore_features =
                vk::PhysicalDeviceTimelineSemaphoreFeatures::builder().timeline_semaphore(true);
            let mut features2 = vk::PhysicalDeviceFeatures2::builder().features(features);
            let mut priorities = vec![];
            let queue_infos = queues
                .iter()
                .map(|&(ref family, ref len)| {
                    priorities.push(vec![1.0; *len as usize]);
                    let p = priorities.last().unwrap();
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(*family)
                        .queue_priorities(&p)
                        .build()
                })
                .collect::<Vec<_>>();
            let mut descriptor_indexing_features =
                vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder()
                    .runtime_descriptor_array(true)
                    .shader_storage_buffer_array_non_uniform_indexing(true)
                    .descriptor_binding_partially_bound(true)
                    .descriptor_binding_update_unused_while_pending(true);

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(&mut features2)
                .push_next(&mut timeline_semaphore_features)
                .push_next(&mut descriptor_indexing_features);

            unsafe { instance.create_device(physical_device, &device_create_info, None)? }
        };

        let allocator =
            alloc::create(entry.vk(), &**instance, device.handle(), physical_device).unwrap();
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };
        let compute_queues = match compute_queues_spec {
            Some((compute_queue_family, len)) => (0..len)
                .map(|ix| unsafe { device.get_device_queue(compute_queue_family, ix) })
                .collect::<Vec<_>>(),
            None => vec![graphics_queue],
        };

        let name = b"vkGetSemaphoreCounterValueKHR\0";
        let name_c = unsafe { CStr::from_bytes_with_nul_unchecked(name).as_ptr() };
        let addr = unsafe { instance.get_device_proc_addr(device.handle(), name_c) };
        let get_semaphore_counter_value: vk::PFN_vkGetSemaphoreCounterValue =
            unsafe { std::mem::transmute(addr.unwrap()) };

        let name = b"vkWaitSemaphoresKHR\0";
        let name_c = unsafe { CStr::from_bytes_with_nul_unchecked(name).as_ptr() };
        let addr = unsafe { instance.get_device_proc_addr(device.handle(), name_c) };
        let wait_semaphores: vk::PFN_vkWaitSemaphores =
            unsafe { std::mem::transmute(addr.unwrap()) };

        let name = b"vkSignalSemaphoreKHR\0";
        let name_c = unsafe { CStr::from_bytes_with_nul_unchecked(name).as_ptr() };
        let addr = unsafe { instance.get_device_proc_addr(device.handle(), name_c) };
        let signal_semaphore: vk::PFN_vkSignalSemaphore =
            unsafe { std::mem::transmute(addr.unwrap()) };

        let device = Device {
            device,
            instance: Arc::clone(instance),
            physical_device,
            allocator,
            graphics_queue_family,
            compute_queue_family: compute_queues_spec
                .map(|a| a.0)
                .unwrap_or(graphics_queue_family),
            graphics_queue: Mutex::new(graphics_queue),
            compute_queues: compute_queues.iter().cloned().map(Mutex::new).collect(),
            get_semaphore_counter_value,
            wait_semaphores,
            signal_semaphore,
        };
        device.set_object_name(graphics_queue, "Graphics Queue");
        for (ix, compute_queue) in compute_queues.iter().cloned().enumerate() {
            if compute_queue != graphics_queue {
                device.set_object_name(compute_queue, &format!("Compute Queue - {}", ix));
            }
        }

        Ok(device)
    }

    pub fn allocation_stats(&self) -> alloc::VmaStats {
        alloc::stats(self.allocator)
    }

    pub fn new_descriptor_pool(
        self: &Arc<Self>,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> DescriptorPool {
        DescriptorPool::new(self, max_sets, pool_sizes)
    }

    pub fn new_descriptor_set_layout(
        self: &Arc<Self>,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> DescriptorSetLayout {
        DescriptorSetLayout::new(self, bindings)
    }

    pub fn new_descriptor_set_layout2(
        self: &Arc<Self>,
        create_info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> DescriptorSetLayout {
        DescriptorSetLayout::new2(self, create_info)
    }

    fn queue_family_for(&self, t: QueueType) -> u32 {
        match t {
            QueueType::Graphics => self.graphics_queue_family,
            QueueType::Compute => self.compute_queue_family,
        }
    }

    pub fn new_command_pool(
        self: &Arc<Self>,
        queue_type: QueueType,
        flags: vk::CommandPoolCreateFlags,
    ) -> CommandPool {
        CommandPool::new(self, self.queue_family_for(queue_type), flags)
    }

    pub fn new_semaphore(self: &Arc<Self>) -> Semaphore {
        Semaphore::new(self)
    }

    pub fn new_semaphore_timeline(self: &Arc<Self>, initial_value: u64) -> TimelineSemaphore {
        TimelineSemaphore::new(self, initial_value)
    }

    pub fn new_fence(self: &Arc<Self>) -> Fence {
        Fence::new(self)
    }

    pub fn new_buffer(
        self: &Arc<Self>,
        buffer_usage: vk::BufferUsageFlags,
        allocation_usage: alloc::VmaMemoryUsage,
        size: vk::DeviceSize,
    ) -> Buffer {
        Buffer::new(self, buffer_usage, allocation_usage, size)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_image(
        self: &Arc<Self>,
        format: vk::Format,
        extent: vk::Extent3D,
        samples: vk::SampleCountFlags,
        tiling: vk::ImageTiling,
        initial_layout: vk::ImageLayout,
        usage: vk::ImageUsageFlags,
        allocation_usage: alloc::VmaMemoryUsage,
    ) -> Image {
        Image::new(
            self,
            format,
            extent,
            samples,
            tiling,
            initial_layout,
            usage,
            allocation_usage,
        )
    }

    pub fn new_event(self: &Arc<Self>) -> Event {
        Event::new(self)
    }

    pub fn new_renderpass(
        self: &Arc<Self>,
        create_info: &vk::RenderPassCreateInfoBuilder,
    ) -> RenderPass {
        RenderPass::new(self, create_info)
    }

    pub fn vk(&self) -> &AshDevice {
        &self.device
    }

    #[cfg(feature = "validation")]
    pub fn set_object_name<T: vk::Handle>(&self, handle: T, name: &str) {
        use std::ffi::CString;

        let name = CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_type(T::TYPE)
            .object_handle(handle.as_raw())
            .object_name(&name);

        unsafe {
            self.instance
                .debug_utils()
                .debug_utils_set_object_name(self.device.handle(), &name_info)
                .expect("failed to set object name");
        };
    }

    #[cfg(not(feature = "validation"))]
    pub fn set_object_name<T: vk::Handle>(&self, _handle: T, _name: &str) {}

    #[cfg(feature = "validation")]
    pub fn debug_marker_around<F: FnOnce()>(
        &self,
        command_buffer: vk::CommandBuffer,
        name: &str,
        color: [f32; 4],
        f: F,
    ) {
        unsafe {
            use std::ffi::CString;

            let name = CString::new(name).unwrap();
            {
                self.instance.debug_utils().cmd_begin_debug_utils_label(
                    command_buffer,
                    &vk::DebugUtilsLabelEXT::builder()
                        .label_name(&name)
                        .color(color),
                );
            }
            f();
            {
                self.instance
                    .debug_utils()
                    .cmd_end_debug_utils_label(command_buffer);
            }
        };
    }

    #[cfg(not(feature = "validation"))]
    pub fn debug_marker_around<R, F: FnOnce() -> R>(
        &self,
        _command_buffer: vk::CommandBuffer,
        _name: &str,
        _color: [f32; 4],
        f: F,
    ) -> R {
        f()
    }
}

impl Deref for Device {
    type Target = AshDevice;

    fn deref(&self) -> &AshDevice {
        &self.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            alloc::destroy(self.allocator);
            self.device.destroy_device(None);
        }
    }
}

pub struct RenderPass {
    pub handle: vk::RenderPass,
    device: Arc<Device>,
}

impl RenderPass {
    pub(super) fn new(
        device: &Arc<Device>,
        create_info: &vk::RenderPassCreateInfoBuilder,
    ) -> RenderPass {
        let handle = unsafe {
            device
                .device
                .create_render_pass(create_info, None)
                .expect("Failed to create renderpass")
        };
        RenderPass {
            device: Arc::clone(device),
            handle,
        }
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_render_pass(self.handle, None) }
    }
}
