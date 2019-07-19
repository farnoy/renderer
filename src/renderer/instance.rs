use ash::{
    self,
    extensions::{ext::DebugUtils, khr::Surface},
    version::{EntryV1_0, InstanceV1_0},
    vk,
};
#[cfg(feature = "validation")]
use std::borrow::Cow;
use std::{ffi::CString, ops::Deref, sync::Arc};
use winit;

use super::{entry::Entry, helpers::create_surface};

pub type AshInstance = ash::Instance;

pub struct Instance {
    handle: AshInstance,
    pub entry: Arc<Entry>,
    pub window: winit::Window,
    pub surface: vk::SurfaceKHR,
    pub window_width: u32,
    pub window_height: u32,
    #[allow(dead_code)]
    debug: Debug,
}

#[cfg(feature = "validation")]
struct Debug {
    utils: DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

#[cfg(not(feature = "validation"))]
struct Debug;

impl Instance {
    pub fn new(
        window_width: u32,
        window_height: u32,
    ) -> Result<(Instance, winit::EventsLoop), ash::InstanceError> {
        let events_loop = winit::EventsLoop::new();
        let window = winit::WindowBuilder::new()
            .with_title("Renderer v3")
            .with_dimensions((window_width, window_height).into())
            .build(&events_loop)
            .unwrap();
        let (window_width, window_height) = window.get_inner_size().unwrap().into();

        let entry = Entry::new().unwrap();

        let layer_names = if cfg!(feature = "validation",) {
            vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()]
        } else {
            vec![]
        };
        let layers_names_raw: Vec<*const i8> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();
        let extension_names_raw = extension_names();
        let name = CString::new("Renderer").unwrap();
        let appinfo = vk::ApplicationInfo::builder()
            .application_name(&name)
            .application_version(0)
            .engine_name(&name)
            .api_version(ash::vk_make_version!(1, 1, 0));
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&appinfo)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names_raw);
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let surface = unsafe { create_surface(entry.vk(), &instance, &window).unwrap() };

        #[cfg(feature = "validation")]
        {
            use std::ffi::c_void;
            let debug_utils = DebugUtils::new(entry.vk(), &instance);

            unsafe extern "system" fn vulkan_debug_callback(
                severity: vk::DebugUtilsMessageSeverityFlagsEXT,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT,
                data: *const vk::DebugUtilsMessengerCallbackDataEXT,
                _user_data: *mut c_void,
            ) -> vk::Bool32 {
                use std::ffi::CStr;
                let message_id = (*data).p_message_id_name;
                if !message_id.is_null() {
                    let s = CStr::from_ptr(message_id).to_string_lossy();
                    print!("[ {} ] ", s);
                };

                let severity_str = match severity {
                    vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "VERBOSE",
                    vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "INFO",
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "WARNING",
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "ERROR",
                    _ => "OTHER",
                };
                let type_str = match message_type {
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "GENERAL",
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VALIDATION",
                    vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "PERFORMANCE",
                    _ => "OTHER",
                };
                println!(
                    "{} & {} => {}",
                    severity_str,
                    type_str,
                    CStr::from_ptr((*data).p_message).to_string_lossy()
                );
                for ix in 0..((*data).object_count) {
                    let object = (*data).p_objects.offset(ix as isize).read();
                    let name = match object.p_object_name {
                        x if x.is_null() => Cow::Borrowed("Unknown name"),
                        x => CStr::from_ptr(x).to_string_lossy(),
                    };
                    println!(
                        "    Object[{}] - Type {:?}, Value 0x{:x?}, Name \"{}\"",
                        ix, object.object_type, object.object_handle, name
                    );
                }
                0
            }

            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_messenger = unsafe {
                debug_utils
                    .create_debug_utils_messenger(&*create_info, None)
                    .expect("failed to create debug utils messenger")
            };

            Ok((
                Instance {
                    handle: instance,
                    window,
                    surface,
                    entry: Arc::new(entry),
                    window_width,
                    window_height,
                    debug: Debug {
                        utils: debug_utils,
                        messenger: debug_messenger,
                    },
                },
                events_loop,
            ))
        }

        #[cfg(not(feature = "validation"))]
        {
            Ok((
                Instance {
                    handle: instance,
                    window,
                    surface,
                    entry: Arc::new(entry),
                    window_width,
                    window_height,
                    debug: Debug,
                },
                events_loop,
            ))
        }
    }

    pub fn vk(&self) -> &AshInstance {
        &self.handle
    }

    #[cfg(feature = "validation")]
    pub fn debug_utils(&self) -> &DebugUtils {
        &self.debug.utils
    }
}

impl Deref for Instance {
    type Target = AshInstance;

    fn deref(&self) -> &AshInstance {
        &self.handle
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        #[cfg(feature = "validation")]
        unsafe {
            self.debug
                .utils
                .destroy_debug_utils_messenger(self.debug.messenger, None);
        }

        unsafe {
            self.handle.destroy_instance(None);
        }
    }
}

fn extension_names() -> Vec<*const i8> {
    let mut base = vec![Surface::name().as_ptr()];
    if cfg!(all(unix, not(target_os = "android"))) {
        use ash::extensions::khr::XlibSurface;
        base.push(XlibSurface::name().as_ptr());
    }
    if cfg!(windows) {
        use ash::extensions::khr::Win32Surface;
        base.push(Win32Surface::name().as_ptr());
    }

    if cfg!(feature = "validation") {
        // for validation layers, use the new, consolidated extension for debugging
        base.push(DebugUtils::name().as_ptr());
    }

    base
}
