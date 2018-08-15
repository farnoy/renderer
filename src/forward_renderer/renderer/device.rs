use ash::extensions::Swapchain;
#[cfg(target = "windows")]
use ash::extensions::Win32Surface;
use ash::{
    self,
    extensions::Surface,
    version::{self, DeviceV1_0, InstanceV1_0},
    vk,
};
use parking_lot::Mutex;
use std::{ops::Deref, ptr, sync::Arc};

use super::{alloc, Instance};

pub type AshDevice = ash::Device<version::V1_1>;

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
}

impl Device {
    pub fn new(instance: &Arc<Instance>) -> Result<Device, ash::DeviceError> {
        let Instance {
            ref entry,
            // ref instance,
            surface,
            ..
        } = **instance;

        let pdevices = instance
            .enumerate_physical_devices()
            .expect("Physical device error");
        let surface_loader =
            Surface::new(entry.vk(), &***instance).expect("Unable to load the Surface extension");

        let physical_device = pdevices[0];
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        let graphics_queue_family = {
            queue_families
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| {
                    let supports_graphic_and_surface =
                        info.queue_flags.subset(vk::QueueFlags::GRAPHICS) && surface_loader
                            .get_physical_device_surface_support_khr(
                                physical_device,
                                ix as u32,
                                surface,
                            );
                    if supports_graphic_and_surface {
                        Some(ix as u32)
                    } else {
                        None
                    }
                }).next()
                .unwrap()
        };
        let (compute_queue_family, compute_queue_len) = {
            queue_families
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| {
                    if info.queue_flags.subset(vk::QueueFlags::COMPUTE)
                        && !info.queue_flags.subset(vk::QueueFlags::GRAPHICS)
                    {
                        Some((ix as u32, info.queue_count))
                    } else {
                        None
                    }
                }).next()
                .expect("no suitable compute queue")
        };
        let queues = vec![
            (graphics_queue_family, 1),
            (compute_queue_family, compute_queue_len),
        ];
        let device = {
            // static RASTER_ORDER: &str = "VK_AMD_rasterization_order\0";
            let device_extension_names_raw = vec![Swapchain::name().as_ptr()];
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
                    vk::DeviceQueueCreateInfo {
                        s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                        p_next: ptr::null(),
                        flags: Default::default(),
                        queue_family_index: *family,
                        p_queue_priorities: p.as_ptr(),
                        queue_count: p.len() as u32,
                    }
                }).collect::<Vec<_>>();
            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DEVICE_CREATE_INFO,
                p_next: ptr::null(),
                flags: Default::default(),
                queue_create_info_count: queue_infos.len() as u32,
                p_queue_create_infos: queue_infos.as_ptr(),
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: device_extension_names_raw.len() as u32,
                pp_enabled_extension_names: device_extension_names_raw.as_ptr(),
                p_enabled_features: &features,
            };
            unsafe { instance.create_device(physical_device, &device_create_info, None)? }
        };

        let allocator =
            alloc::create(entry.vk(), &**instance, device.handle(), physical_device).unwrap();
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };
        let compute_queues = (0..compute_queue_len)
            .map(|ix| unsafe {
                Arc::new(Mutex::new(device.get_device_queue(compute_queue_family, 0)))
            }).collect::<Vec<_>>();

        Ok(Device {
            device,
            instance: Arc::clone(instance),
            physical_device,
            allocator,
            graphics_queue_family,
            compute_queue_family,
            graphics_queue: Arc::new(Mutex::new(graphics_queue)),
            compute_queues,
        })
    }

    pub fn vk(&self) -> &AshDevice {
        &self.device
    }

    #[cfg(feature = "validation")]
    pub fn set_object_name(&self, object_type: vk::ObjectType, object: u64, name: &str) {
        unsafe {
            use std::ffi::CString;
            use std::ptr;

            let name = CString::new(name).unwrap();
            let name_info = vk::DebugUtilsObjectNameInfoEXT {
                s_type: vk::StructureType::DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                p_next: ptr::null(),
                object_type: object_type,
                object_handle: object,
                p_object_name: name.as_ptr(),
            };
            let res = self
                .instance
                .debug_utils()
                .set_debug_utils_object_name_ext(self.device.handle(), &name_info);
            assert_eq!(res, vk::Result::SUCCESS);
        };
    }

    #[cfg(not(feature = "validation"))]
    pub fn set_object_name(&self, _object_type: vk::ObjectType, _object: u64, _name: &str) {}

    #[cfg(feature = "validation")]
    pub fn debug_marker_around<F: Fn()>(
        &self,
        command_buffer: vk::CommandBuffer,
        name: &str,
        color: [f32; 4],
        f: F,
    ) {
        unsafe {
            use std::ffi::CString;
            use std::ptr;

            let name = CString::new(name).unwrap();
            let label_info = vk::DebugUtilsLabelEXT {
                s_type: vk::StructureType::DEBUG_UTILS_LABEL_EXT,
                p_next: ptr::null(),
                p_label_name: name.as_ptr(),
                color: color,
            };
            self.instance
                .debug_utils()
                .cmd_begin_debug_utils_label_ext(command_buffer, &label_info);
            f();
            self.instance
                .debug_utils()
                .cmd_end_debug_utils_label_ext(command_buffer);
        };
    }

    #[cfg(not(feature = "validation"))]
    pub fn debug_marker_around<R, F: Fn() -> R>(
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
        use ash::version::DeviceV1_0;
        self.device.device_wait_idle().unwrap();
        alloc::destroy(self.allocator);
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
