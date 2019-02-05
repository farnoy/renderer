use ash::extensions::ext::DebugMarker;
use ash::extensions::khr::Swapchain;
#[cfg(target = "windows")]
use ash::extensions::Win32Surface;
use ash::{
    self,
    extensions::khr::Surface,
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use parking_lot::Mutex;
use std::{ops::Deref, sync::Arc};

pub mod descriptors;

use super::{alloc, device::descriptors::DescriptorSetLayout, Instance};

pub type AshDevice = ash::Device;

pub struct Device {
    pub device: AshDevice,
    pub instance: Arc<Instance>,
    pub physical_device: vk::PhysicalDevice,
    pub allocator: alloc::VmaAllocator,
    pub graphics_queue_family: u32,
    pub compute_queue_family: u32,
    pub graphics_queue: Arc<Mutex<vk::Queue>>,
    pub compute_queues: Vec<Arc<Mutex<vk::Queue>>>,
    // pub _transfer_queue: Arc<Mutex<vk::Queue>>,
    #[allow(dead_code)]
    debug: Debug,
}

#[cfg(feature = "radeon-profiler")]
struct Debug {
    marker: DebugMarker,
}

#[cfg(not(feature = "radeon-profiler"))]
struct Debug;

impl Device {
    pub fn new(instance: &Arc<Instance>) -> Result<Device, vk::Result> {
        let Instance {
            ref entry,
            // ref instance,
            surface,
            ..
        } = **instance;

        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Physical device error")
        };
        let surface_loader = Surface::new(entry.vk(), &***instance);

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
                            && surface_loader.get_physical_device_surface_support(
                                physical_device,
                                ix as u32,
                                surface,
                            );
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
            let device_extension_names_raw = if cfg!(feature = "radeon-profiler") {
                vec![Swapchain::name().as_ptr(), DebugMarker::name().as_ptr()]
            } else {
                vec![Swapchain::name().as_ptr()]
            };
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                sampler_anisotropy: 1,
                geometry_shader: 1,
                depth_bounds: 1,
                multi_draw_indirect: 1,
                vertex_pipeline_stores_and_atomics: 1,
                robust_buffer_access: 1,
                fill_mode_non_solid: 0,
                ..Default::default()
            };
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
                .enabled_features(&features);
            unsafe { instance.create_device(physical_device, &device_create_info, None)? }
        };

        let allocator =
            alloc::create(entry.vk(), &**instance, device.handle(), physical_device).unwrap();
        let graphics_queue = unsafe {
            Arc::new(Mutex::new(
                device.get_device_queue(graphics_queue_family, 0),
            ))
        };
        let compute_queues = match compute_queues_spec {
            Some((compute_queue_family, len)) => (0..len)
                .map(|ix| unsafe {
                    Arc::new(Mutex::new(
                        device.get_device_queue(compute_queue_family, ix),
                    ))
                })
                .collect::<Vec<_>>(),
            None => vec![Arc::clone(&graphics_queue)],
        };

        #[cfg(feature = "radeon-profiler")]
        {
            let debug = Debug {
                marker: DebugMarker::new(instance.vk(), &device),
            };
            Ok(Device {
                device,
                instance: Arc::clone(instance),
                physical_device,
                allocator,
                graphics_queue_family,
                compute_queue_family: compute_queues_spec
                    .map(|a| a.0)
                    .unwrap_or(graphics_queue_family),
                graphics_queue,
                compute_queues,
                debug,
            })
        }

        #[cfg(not(feature = "radeon-profiler"))]
        {
            Ok(Device {
                device,
                instance: Arc::clone(instance),
                physical_device,
                allocator,
                graphics_queue_family,
                compute_queue_family: compute_queues_spec
                    .map(|a| a.0)
                    .unwrap_or(graphics_queue_family),
                graphics_queue,
                compute_queues,
                debug: Debug,
            })
        }
    }

    pub fn new_descriptor_set_layout(
        self: &Arc<Self>,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> DescriptorSetLayout {
        DescriptorSetLayout::new(self, bindings)
    }

    pub fn vk(&self) -> &AshDevice {
        &self.device
    }

    #[cfg(all(feature = "validation", not(feature = "radeon-profiler")))]
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

    #[cfg(not(all(feature = "validation", not(feature = "radeon-profiler"))))]
    pub fn set_object_name<T: vk::Handle>(&self, handle: T, _name: &str) {}

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
            #[cfg(not(feature = "radeon-profiler"))]
            {
                self.instance.debug_utils().cmd_begin_debug_utils_label(
                    command_buffer,
                    &vk::DebugUtilsLabelEXT::builder()
                        .label_name(&name)
                        .color(color),
                );
            }
            #[cfg(feature = "radeon-profiler")]
            {
                self.debug.marker.cmd_debug_marker_begin(
                    command_buffer,
                    &vk::DebugMarkerMarkerInfoEXT::builder()
                        .color(color)
                        .marker_name(&name),
                );
            }
            f();
            #[cfg(feature = "radeon-profiler")]
            {
                self.debug.marker.cmd_debug_marker_end(command_buffer);
            }
            #[cfg(not(feature = "radeon-profiler"))]
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
