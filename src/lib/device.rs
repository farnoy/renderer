use ash;
#[cfg(feature = "validation")]
use ash::extensions::DebugReport;
use ash::extensions::DebugMarker;
use ash::extensions::Swapchain;
#[cfg(target = "windows")]
use ash::extensions::Win32Surface;
use ash::vk;
use ash::version;
use ash::version::InstanceV1_0;
use std::ffi::CStr;
use std::ops;
use std::ptr;
use std::sync::Arc;

use super::instance::Instance;

pub type AshDevice = ash::Device<version::V1_0>;

#[cfg(feature = "validation")]
pub struct Device {
    device: AshDevice,
    #[allow(dead_code)] // needed for Drop ordering
    instance: Arc<Instance>,
    debug_call_back: vk::DebugReportCallbackEXT,
    debug_report_loader: DebugReport,
    debug_marker_loader: Option<DebugMarker>,
}

#[cfg(not(feature = "validation"))]
pub struct Device {
    device: AshDevice,
    #[allow(dead_code)] // needed for Drop ordering
    instance: Arc<Instance>,
}

impl Device {
    pub fn new(instance: &Arc<Instance>, physical_device: vk::PhysicalDevice, queues: &[(u32, u32)]) -> Result<Arc<Device>, ash::DeviceError> {
        let debug_marker_available = cfg!(feature = "validation");

        let device = {
            let device_extension_names_raw = if cfg!(feature = "validation") {
                vec![Swapchain::name().as_ptr(), DebugMarker::name().as_ptr()]
            } else {
                vec![Swapchain::name().as_ptr()]
            };
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                sampler_anisotropy: 1,
                geometry_shader: 1,
                ..Default::default()
            };
            let mut priorities = vec![];
            let queue_infos = queues
                .iter()
                .map(|&(ref family, ref len)| {
                    priorities.push(vec![1.0; *len as usize]);
                    let p = priorities.last().unwrap();
                    vk::DeviceQueueCreateInfo {
                        s_type: vk::StructureType::DeviceQueueCreateInfo,
                        p_next: ptr::null(),
                        flags: Default::default(),
                        queue_family_index: *family,
                        p_queue_priorities: p.as_ptr(),
                        queue_count: p.len() as u32,
                    }
                })
                .collect::<Vec<_>>();
            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DeviceCreateInfo,
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
            let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

            device
        };

        #[cfg(feature = "validation")]
        {
            let debug_info = vk::DebugReportCallbackCreateInfoEXT {
                s_type: vk::StructureType::DebugReportCallbackCreateInfoExt,
                p_next: ptr::null(),
                flags: vk::DEBUG_REPORT_ERROR_BIT_EXT | vk::DEBUG_REPORT_WARNING_BIT_EXT | vk::DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
                pfn_callback: vulkan_debug_callback,
                p_user_data: ptr::null_mut(),
            };
            let debug_report_loader = DebugReport::new(instance.entry().vk(), instance.vk()).expect("Unable to load debug report");
            let debug_call_back = unsafe {
                debug_report_loader
                    .create_debug_report_callback_ext(&debug_info, None)
                    .unwrap()
            };
            let debug_marker_loader = if debug_marker_available {
                Some(DebugMarker::new(instance.vk(), &device).expect("Unable to load debug marker"))
            } else {
                None
            };

            let device = Device {
                device: device,
                instance: instance.clone(),
                debug_report_loader: debug_report_loader,
                debug_marker_loader: debug_marker_loader,
                debug_call_back: debug_call_back,
            };
            unsafe {
                use std::mem::transmute;
                device.set_object_name(
                    vk::DebugReportObjectTypeEXT::Device,
                    transmute(device.vk().handle()),
                    "Device",
                );
            }
            Ok(Arc::new(device))
        }
        #[cfg(not(feature = "validation"))]
        {
            Ok(Arc::new(Device {
                device: device,
                instance: instance.clone(),
            }))
        }
    }

    pub fn vk(&self) -> &AshDevice {
        &self.device
    }

    #[cfg(feature = "validation")]
    pub fn set_object_name(&self, object_type: vk::DebugReportObjectTypeEXT, object: u64, name: &str) {
        if self.debug_marker_loader.is_none() {
            return;
        }

        unsafe {
            use std::ffi::CString;
            use std::ptr;

            let name = CString::new(name).unwrap();
            let name_info = vk::DebugMarkerObjectNameInfoEXT {
                s_type: vk::StructureType::DebugMarkerObjectNameInfoEXT,
                p_next: ptr::null(),
                object_type: object_type,
                object: object,
                p_object_name: name.as_ptr(),
            };
            self.debug_marker_loader
                .as_ref()
                .unwrap()
                .debug_marker_set_object_name_ext(self.device.handle(), &name_info)
                .unwrap();
        };
    }

    #[cfg(feature = "validation")]
    pub fn debug_marker_around<F: Fn()>(&self, command_buffer: vk::CommandBuffer, name: &str, color: [f32; 4], f: F) {
        if self.debug_marker_loader.is_none() {
            return f();
        }

        unsafe {
            use std::ffi::CString;
            use std::ptr;

            let name = CString::new(name).unwrap();
            let marker_info = vk::DebugMarkerMarkerInfoEXT {
                s_type: vk::StructureType::DebugMarkerMarkerInfoEXT,
                p_next: ptr::null(),
                p_marker_name: name.as_ptr(),
                color: color,
            };
            self.debug_marker_loader
                .as_ref()
                .unwrap()
                .cmd_debug_marker_begin_ext(command_buffer, &marker_info);
            let res = f();
            self.debug_marker_loader
                .as_ref()
                .unwrap()
                .cmd_debug_marker_end_ext(command_buffer);
            res
        };
    }

    #[cfg(not(feature = "validation"))]
    pub fn debug_marker_around<R, F: Fn() -> R>(&self, _command_buffer: vk::CommandBuffer, _name: &str, _color: [f32; 4], f: F) -> R {
        f()
    }

    #[cfg(feature = "validation")]
    pub fn debug_marker_start(&self, command_buffer: vk::CommandBuffer, name: &str, color: [f32; 4]) {
        if self.debug_marker_loader.is_none() {
            return;
        }

        unsafe {
            use std::ffi::CString;
            use std::ptr;

            let name = CString::new(name).unwrap();
            let marker_info = vk::DebugMarkerMarkerInfoEXT {
                s_type: vk::StructureType::DebugMarkerMarkerInfoEXT,
                p_next: ptr::null(),
                p_marker_name: name.as_ptr(),
                color: color,
            };
            self.debug_marker_loader
                .as_ref()
                .unwrap()
                .cmd_debug_marker_begin_ext(command_buffer, &marker_info);
        };
    }

    #[cfg(not(feature = "validation"))]
    pub fn debug_marker_start(&self, _command_buffer: vk::CommandBuffer, _name: &str, _color: [f32; 4]) {}

    #[cfg(feature = "validation")]
    pub fn debug_marker_end(&self, command_buffer: vk::CommandBuffer) {
        if self.debug_marker_loader.is_none() {
            return;
        }

        unsafe {
            self.debug_marker_loader
                .as_ref()
                .unwrap()
                .cmd_debug_marker_end_ext(command_buffer);
        };
    }

    #[cfg(not(feature = "validation"))]
    pub fn debug_marker_end(&self, _command_buffer: vk::CommandBuffer) {}
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

        #[cfg(feature = "validation")]
        unsafe {
            self.debug_report_loader
                .destroy_debug_report_callback_ext(self.debug_call_back, None);
        }
    }
}

#[cfg(feature = "validation")]
unsafe extern "system" fn vulkan_debug_callback(
    _: vk::DebugReportFlagsEXT,
    _: vk::DebugReportObjectTypeEXT,
    _: vk::uint64_t,
    _: vk::size_t,
    _: vk::int32_t,
    _: *const vk::c_char,
    p_message: *const vk::c_char,
    _: *mut vk::c_void,
) -> u32 {
    use std::ffi::CStr;
    println!(
        "{}",
        CStr::from_ptr(p_message)
            .to_str()
            .expect("Weird validation layer message")
    );
    1
}
