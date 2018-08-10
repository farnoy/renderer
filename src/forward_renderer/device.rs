use ash;
use ash::extensions::Swapchain;
#[cfg(target = "windows")]
use ash::extensions::Win32Surface;
use ash::version;
use ash::version::InstanceV1_0;
use ash::vk;
use std::ops;
use std::ptr;
use std::sync::Arc;

use super::instance::Instance;

pub type AshDevice = ash::Device<version::V1_0>;

pub struct Device {
    device: AshDevice,
    #[allow(dead_code)]
    instance: Arc<Instance>,
}

impl Device {
    pub fn new(
        instance: &Arc<Instance>,
        physical_device: vk::PhysicalDevice,
        queues: &[(u32, u32)],
    ) -> Result<Arc<Device>, ash::DeviceError> {
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

        #[cfg(feature = "validation")]
        {
            let device = Device {
                device: device,
                instance: instance.clone(),
            };
            unsafe {
                use std::mem::transmute;
                device.set_object_name(
                    vk::ObjectType::DEVICE,
                    transmute(device.vk().handle()),
                    "Device",
                );
            }
            Ok(Arc::new(device))
        }
        #[cfg(not(feature = "validation"))]
        {
            Ok(Arc::new(Device {
                device,
                instance: instance.clone(),
            }))
        }
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

impl ops::Deref for Device {
    type Target = AshDevice;

    fn deref(&self) -> &AshDevice {
        &self.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        use ash::version::DeviceV1_0;
        self.device.device_wait_idle().unwrap();
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
