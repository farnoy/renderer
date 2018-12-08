use ash;
use ash::extensions::Surface;
#[cfg(windows)]
use ash::extensions::Win32Surface;
#[cfg(all(unix, not(target_os = "android")))]
use ash::extensions::XlibSurface;
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;
use std::ffi::CString;
#[allow(unused_imports)]
use std::mem::transmute;
use std::ops::Deref;
#[cfg(feature = "validation")]
use std::ptr;
use std::sync::Arc;
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
    utils: vk::ExtDebugUtilsFn,
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

        let layer_names = if cfg!(all(feature = "validation")) {
            vec![CString::new("VK_LAYER_LUNARG_standard_validation").unwrap()]
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
            let debug_utils = vk::ExtDebugUtilsFn::load(|name| unsafe {
                transmute(
                    entry
                        .vk()
                        .get_instance_proc_addr(instance.handle(), name.as_ptr()),
                )
            });

            unsafe extern "system" fn vulkan_debug_callback(
                severity: vk::DebugUtilsMessageSeverityFlagsEXT,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT,
                data: *const vk::DebugUtilsMessengerCallbackDataEXT,
                _user_data: *mut c_void,
            ) -> vk::Bool32 {
                use std::ffi::CStr;
                let message_id = (*data).p_message_id_name;
                if !message_id.is_null() {
                    let s = CStr::from_ptr(message_id)
                        .to_str()
                        .expect("Something weird in p_message_id_name");
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
                    CStr::from_ptr((*data).p_message)
                        .to_str()
                        .expect("Weird validation layer message")
                );
                for ix in 0..((*data).object_count) {
                    let name = (*data).p_objects.offset(ix as isize).read();
                    println!(
                        "    Object[{}] - Type {:?}, Value 0x{:x?}, Name \"{}\"",
                        ix,
                        name.object_type,
                        name.object_handle,
                        CStr::from_ptr(name.p_object_name).to_str().unwrap(),
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

            let mut debug_messenger = vk::DebugUtilsMessengerEXT::null();

            let res = unsafe {
                debug_utils.create_debug_utils_messenger_ext(
                    instance.handle(),
                    &*create_info,
                    ptr::null(),
                    &mut debug_messenger,
                )
            };
            assert_eq!(res, vk::Result::SUCCESS);

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
    pub fn debug_utils(&self) -> &vk::ExtDebugUtilsFn {
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
            self.debug.utils.destroy_debug_utils_messenger_ext(
                self.handle.handle(),
                self.debug.messenger,
                ptr::null(),
            );
        }

        unsafe {
            self.handle.destroy_instance(None);
        }
    }
}

static DEBUG_UTILS: &'static str = "VK_EXT_debug_utils\0";

#[cfg(all(unix, not(target_os = "android")))]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DEBUG_UTILS.as_ptr() as *const i8,
    ]
}

#[cfg(all(windows))]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        Win32Surface::name().as_ptr(),
        DEBUG_UTILS.as_ptr() as *const i8,
    ]
}
