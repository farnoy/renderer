#[cfg(feature = "vk_names")]
use std::borrow::Cow;
use std::{ffi::CString, ops::Deref, sync::Arc};

use ash::{
    self,
    extensions::{ext::DebugUtils, khr::Surface},
    version::{EntryV1_0, InstanceV1_0},
    vk,
};

use super::entry::Entry;

pub(crate) type AshInstance = ash::Instance;

pub(crate) struct Instance {
    handle: AshInstance,
    pub(crate) entry: Arc<Entry>,
    pub(crate) window: winit::window::Window,
    #[allow(dead_code)]
    debug: Debug,
}

#[cfg(feature = "vk_names")]
struct Debug {
    utils: DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

#[cfg(not(feature = "vk_names"))]
struct Debug;

impl Instance {
    pub(crate) fn new() -> Result<(Instance, winit::event_loop::EventLoop<()>), ash::InstanceError> {
        let events_loop = winit::event_loop::EventLoop::new();
        let window = winit::window::WindowBuilder::new()
            .with_title("Renderer v3")
            .build(&events_loop)
            .expect("Failed to create window");

        let entry = Entry::new().expect("Failed to create entry");

        let layer_names = if cfg!(feature = "standard_validation",) {
            vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()]
        } else {
            vec![]
        };
        let layers_names_raw: Vec<*const i8> = layer_names.iter().map(|raw_name| raw_name.as_ptr()).collect();
        let extension_names_raw = extension_names();
        let name = CString::new("Renderer").unwrap();
        let appinfo = vk::ApplicationInfo::builder()
            .application_name(&name)
            .application_version(0)
            .engine_name(&name)
            .api_version(vk::make_version(1, 2, 0));

        #[cfg(all(feature = "silence_validation", feature = "sync_validation"))]
        compile_error!("Can't silence validation while enabling sync validation");

        #[cfg(all(feature = "gpu_validation", feature = "gpu_printf"))]
        compile_error!("Can't enable GPU validation & printf simultaneously");

        let disabled_features = vec![
            #[cfg(feature = "silence_validation")]
            vk::ValidationFeatureDisableEXT::ALL,
        ];

        let enabled_features = vec![
            #[cfg(feature = "sync_validation")]
            vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            #[cfg(feature = "gpu_validation")]
            vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
            #[cfg(feature = "gpu_validation")]
            vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT,
            #[cfg(feature = "gpu_printf")]
            vk::ValidationFeatureEnableEXT::DEBUG_PRINTF,
        ];

        let mut validation_features = vk::ValidationFeaturesEXT::builder()
            .enabled_validation_features(&enabled_features)
            .disabled_validation_features(&disabled_features);

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&appinfo)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names_raw)
            .push_next(&mut validation_features);

        let instance = unsafe { entry.create_instance(&create_info, None).expect("create instance") };

        #[cfg(feature = "vk_names")]
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

            let message_type = if cfg!(feature = "standard_validation") {
                vk::DebugUtilsMessageTypeFlagsEXT::all()
            } else {
                vk::DebugUtilsMessageTypeFlagsEXT::empty()
            };

            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | {
                        if cfg!(feature = "gpu_printf") {
                            vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        } else {
                            vk::DebugUtilsMessageSeverityFlagsEXT::empty()
                        }
                    },
                )
                .message_type(message_type)
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
                    entry: Arc::new(entry),
                    debug: Debug {
                        utils: debug_utils,
                        messenger: debug_messenger,
                    },
                },
                events_loop,
            ))
        }

        #[cfg(not(feature = "vk_names"))]
        {
            Ok((
                Instance {
                    handle: instance,
                    window,
                    entry: Arc::new(entry),
                    debug: Debug,
                },
                events_loop,
            ))
        }
    }

    pub(crate) fn vk(&self) -> &AshInstance {
        &self.handle
    }

    #[cfg(feature = "vk_names")]
    pub(crate) fn debug_utils(&self) -> &DebugUtils {
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
        #[cfg(feature = "vk_names")]
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

    if cfg!(feature = "vk_names") {
        // for validation layers, use the new, consolidated extension for debugging
        base.push(DebugUtils::name().as_ptr());
    }

    base
}
