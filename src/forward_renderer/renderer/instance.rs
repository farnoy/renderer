use ash;
use ash::extensions::Surface;
#[cfg(windows)]
use ash::extensions::Win32Surface;
#[cfg(all(unix, not(target_os = "android")))]
use ash::extensions::XlibSurface;
use ash::version::{EntryV1_0, InstanceFpV1_1, InstanceLoader, InstanceV1_0, V1_1};
use ash::vk;
use std::ffi::CString;
#[allow(unused_imports)]
use std::mem::transmute;
use std::ops::Deref;
use std::ptr;
use std::sync::Arc;
use winit;

use super::{entry::Entry, helpers::create_surface};

pub type AshInstance = ash::Instance<V1_1>;

pub struct Instance {
    handle: AshInstance,
    pub entry: Arc<Entry>,
    _window: winit::Window,
    pub surface: vk::SurfaceKHR,
    pub window_width: u32,
    pub window_height: u32,
    debug: Debug,
}

#[cfg(feature = "validation")]
struct Debug {
    utils: vk::extensions::ExtDebugUtilsFn,
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
            .with_dimensions(window_width, window_height)
            .build(&events_loop)
            .unwrap();
        let (window_width, window_height) = window.get_inner_size().unwrap();

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
        let appinfo = vk::ApplicationInfo {
            p_application_name: name.as_ptr(),
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            application_version: 0,
            p_engine_name: name.as_ptr(),
            engine_version: 0,
            api_version: vk_make_version!(1, 1, 0),
        };
        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            p_application_info: &appinfo,
            pp_enabled_layer_names: layers_names_raw.as_ptr(),
            enabled_layer_count: layers_names_raw.len() as u32,
            pp_enabled_extension_names: extension_names_raw.as_ptr(),
            enabled_extension_count: extension_names_raw.len() as u32,
        };
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        let fp_1_1 = unsafe {
            InstanceFpV1_1::load((*entry).static_fn(), instance.handle())
                .expect("failed to load 1.1 instance functions")
        };

        let instance_1_1 = ash::Instance::<V1_1>::from_raw(instance.handle(), fp_1_1);

        let surface = unsafe { create_surface(entry.vk(), &instance_1_1, &window).unwrap() };

        #[cfg(feature = "validation")]
        {
            let debug_utils = vk::extensions::ExtDebugUtilsFn::load(|name| unsafe {
                transmute(
                    entry
                        .vk()
                        .get_instance_proc_addr(instance.handle(), name.as_ptr()),
                )
            }).expect("DebugUtils extension not supported");

            unsafe extern "system" fn vulkan_debug_callback(
                severity: vk::DebugUtilsMessageSeverityFlagsEXT,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT,
                data: *const vk::DebugUtilsMessengerCallbackDataEXT,
                _user_data: *mut vk::c_void,
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
                1
            }

            let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
                s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                p_next: ptr::null(),
                flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::all(),
                pfn_user_callback: vulkan_debug_callback,
                p_user_data: ptr::null_mut(),
            };

            let mut debug_messenger = vk::DebugUtilsMessengerEXT::null();

            let res = unsafe {
                debug_utils.create_debug_utils_messenger_ext(
                    instance.handle(),
                    &create_info,
                    ptr::null(),
                    &mut debug_messenger,
                )
            };
            assert_eq!(res, vk::Result::SUCCESS);

            Ok((
                Instance {
                    handle: instance_1_1,
                    _window: window,
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
                    handle: instance_1_1,
                    _window: window,
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
    pub fn debug_utils(&self) -> &vk::extensions::ExtDebugUtilsFn {
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
