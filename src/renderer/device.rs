#[cfg(target = "windows")]
use ash::extensions::Win32Surface;
use ash::{
    self, extensions,
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use parking_lot::Mutex;
use std::{ops::Deref, path::PathBuf, sync::Arc};

mod buffer;
mod commands;
mod descriptors;
mod double_buffered;
mod image;
mod mapping;
mod shader;
mod sync;

use super::{alloc, Instance, Surface};

pub use self::{
    buffer::*, commands::*, descriptors::*, double_buffered::*, image::*, shader::Shader, sync::*,
};

type AshDevice = ash::Device;

pub struct Device {
    pub(super) device: AshDevice,
    #[allow(unused)]
    instance: Arc<Instance>,
    pub(super) physical_device: vk::PhysicalDevice,
    pub allocator: alloc::VmaAllocator,
    pub limits: vk::PhysicalDeviceLimits,
    pub graphics_queue_family: u32,
    pub compute_queue_family: u32,
    pub(super) graphics_queue: Mutex<vk::Queue>,
    pub(super) compute_queues: Vec<Mutex<vk::Queue>>,
    // pub _transfer_queue: Arc<Mutex<vk::Queue>>,
}

pub enum QueueType {
    Graphics,
    Compute,
}

pub struct DebugMarkerGuard<'a> {
    #[allow(unused)]
    command_buffer: &'a RecordingCommandBuffer<'a>,
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
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        /*
        unsafe {
            for format in vk::Format::UNDEFINED.as_raw()..vk::Format::ASTC_12X12_SRGB_BLOCK.as_raw()
            {
                let format = vk::Format::from_raw(format);
                let res = instance.get_physical_device_image_format_properties(
                    physical_device,
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageUsageFlags::SAMPLED,
                    vk::ImageCreateFlags::SPARSE_BINDING & vk::ImageCreateFlags::SPARSE_RESIDENCY,
                );
                if let Ok(vk::ImageFormatProperties { ref max_extent, ..}) = res {
                    dbg!(format, max_extent);
                }
            }
        }
        */
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
            let device_extension_names_raw = [extensions::khr::Swapchain::name().as_ptr()];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                sampler_anisotropy: 1,
                // sparse_binding: 1,
                // sparse_residency_image2_d: 1,
                depth_bounds: 0,
                multi_draw_indirect: 1,
                vertex_pipeline_stores_and_atomics: 1,
                robust_buffer_access: 1, // TODO: disable at some point?
                fill_mode_non_solid: 1,
                draw_indirect_first_instance: 1,
                shader_storage_buffer_array_dynamic_indexing: 1,
                ..Default::default()
            };
            let mut features2 = vk::PhysicalDeviceFeatures2::builder().features(features);
            let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
                .descriptor_binding_partially_bound(true)
                .runtime_descriptor_array(true)
                .shader_storage_buffer_array_non_uniform_indexing(true)
                .timeline_semaphore(true)
                .scalar_block_layout(true)
                .descriptor_indexing(true);
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

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(&mut features2)
                .push_next(&mut features12);

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

        let device = Device {
            device,
            instance: Arc::clone(instance),
            physical_device,
            allocator,
            limits: properties.limits,
            graphics_queue_family,
            compute_queue_family: compute_queues_spec
                .map(|a| a.0)
                .unwrap_or(graphics_queue_family),
            graphics_queue: Mutex::new(graphics_queue),
            compute_queues: compute_queues.iter().cloned().map(Mutex::new).collect(),
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

    pub fn new_shader<F>(self: &Arc<Self>, path: &PathBuf, verify: F) -> Shader
    where
        F: Fn(&[u8]) -> bool,
    {
        Shader::new_verify(self, path, verify)
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

    #[cfg(feature = "vk_names")]
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

    #[cfg(not(feature = "vk_names"))]
    pub fn set_object_name<T: vk::Handle>(&self, _handle: T, _name: &str) {}

    #[cfg(feature = "vk_names")]
    pub fn debug_marker_around2<'a>(
        &self,
        command_buffer: &'a RecordingCommandBuffer,
        name: &str,
        color: [f32; 4],
    ) -> DebugMarkerGuard<'a> {
        unsafe {
            use std::ffi::CString;

            let name = CString::new(name).unwrap();
            {
                self.instance.debug_utils().cmd_begin_debug_utils_label(
                    **command_buffer,
                    &vk::DebugUtilsLabelEXT::builder()
                        .label_name(&name)
                        .color(color),
                );
            }
            DebugMarkerGuard { command_buffer }
        }
    }

    #[cfg(not(feature = "vk_names"))]
    pub fn debug_marker_around2<'a>(
        &self,
        command_buffer: &'a RecordingCommandBuffer,
        _name: &str,
        _color: [f32; 4],
    ) -> DebugMarkerGuard<'a> {
        DebugMarkerGuard { command_buffer }
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

impl<'a> Drop for DebugMarkerGuard<'a> {
    fn drop(&mut self) {
        #[cfg(feature = "vk_names")]
        unsafe {
            self.command_buffer
                .pool
                .device
                .instance
                .debug_utils()
                .cmd_end_debug_utils_label(**self.command_buffer);
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
