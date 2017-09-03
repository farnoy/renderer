use ash;
#[cfg(feature = "validation")]
use ash::extensions::DebugReport;
use ash::extensions::Swapchain;
#[cfg(target = "windows")]
use ash::extensions::Win32Surface;
use ash::vk;
use ash::version;
use ash::version::InstanceV1_0;
use std::ffi::{CStr, CString};
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
}

#[cfg(not(feature = "validation"))]
pub struct Device {
    device: AshDevice,
    #[allow(dead_code)] // needed for Drop ordering
    instance: Arc<Instance>,
}

static VK_EXT_DEBUG_MARKER: &str = "VK_EXT_debug_marker";

impl Device {
    pub fn new(instance: &Arc<Instance>, physical_device: vk::PhysicalDevice, queue_family_index: u32) -> Result<Arc<Device>, ash::DeviceError> {
        let device = {
            let ext = CString::new(VK_EXT_DEBUG_MARKER).unwrap();
            let device_extension_names_raw = if cfg!(feature = "debug-marker") {
                unsafe {
                    if instance
                        .vk()
                        .enumerate_device_extension_properties(physical_device)
                        .unwrap()
                        .iter()
                        .any(|ext| {
                            CStr::from_ptr(ext.extension_name.as_ref().as_ptr())
                                .to_str()
                                .unwrap() == VK_EXT_DEBUG_MARKER
                        })
                    {
                        vec![Swapchain::name().as_ptr(), ext.as_ptr()]
                    } else {
                        vec![Swapchain::name().as_ptr()]

                    }
                }
            } else {
                vec![Swapchain::name().as_ptr()]
            };
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };
            let priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DeviceQueueCreateInfo,
                p_next: ptr::null(),
                flags: Default::default(),
                queue_family_index: queue_family_index as u32,
                p_queue_priorities: priorities.as_ptr(),
                queue_count: priorities.len() as u32,
            };
            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DeviceCreateInfo,
                p_next: ptr::null(),
                flags: Default::default(),
                queue_create_info_count: 1,
                p_queue_create_infos: &queue_info,
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: device_extension_names_raw.len() as u32,
                pp_enabled_extension_names: device_extension_names_raw.as_ptr(),
                p_enabled_features: &features,
            };
            let device = unsafe {
                instance.create_device(
                    physical_device,
                    &device_create_info,
                    None,
                )?
            };

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

            Ok(Arc::new(Device {
                device: device,
                instance: instance.clone(),
                debug_report_loader: debug_report_loader,
                debug_call_back: debug_call_back,
            }))
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
            self.debug_report_loader.destroy_debug_report_callback_ext(
                self.debug_call_back,
                None,
            );
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
    println!("{:?}", CStr::from_ptr(p_message));
    1
}
