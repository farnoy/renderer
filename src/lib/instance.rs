use ash;
use ash::extensions::{DebugReport, Surface};
#[cfg(windows)]
use ash::extensions::Win32Surface;
#[cfg(all(unix, not(target_os = "android")))]
use ash::extensions::XlibSurface;
use ash::vk;
use ash::version;
use ash::version::EntryV1_0;
use std::ffi::CString;
use std::ops;
use std::ptr;
use std::sync::Arc;
use super::entry::Entry;

pub type AshInstance = ash::Instance<version::V1_0>;

pub struct Instance {
    handle: AshInstance,
    entry: Arc<Entry>,
}

impl Instance {
    pub fn new(entry: &Arc<Entry>) -> Result<Arc<Instance>, ash::InstanceError> {
        let layer_names = if cfg!(all(feature = "validation", not(feature = "renderdoc"))) {
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
            s_type: vk::StructureType::ApplicationInfo,
            p_next: ptr::null(),
            application_version: 0,
            p_engine_name: name.as_ptr(),
            engine_version: 0,
            api_version: vk_make_version!(1, 0, 68),
        };
        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::InstanceCreateInfo,
            p_next: ptr::null(),
            flags: Default::default(),
            p_application_info: &appinfo,
            pp_enabled_layer_names: layers_names_raw.as_ptr(),
            enabled_layer_count: layers_names_raw.len() as u32,
            pp_enabled_extension_names: extension_names_raw.as_ptr(),
            enabled_extension_count: extension_names_raw.len() as u32,
        };
        let instance: AshInstance = unsafe { entry.create_instance(&create_info, None)? };

        Ok(Arc::new(Instance {
            handle: instance,
            entry: entry.clone(),
        }))
    }

    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    pub fn vk(&self) -> &AshInstance {
        &self.handle
    }
}

impl ops::Deref for Instance {
    type Target = AshInstance;

    fn deref(&self) -> &AshInstance {
        &self.handle
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        use ash::version::InstanceV1_0;
        unsafe {
            self.handle.destroy_instance(None);
        }
    }
}

#[cfg(all(unix, not(target_os = "android")))]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugReport::name().as_ptr(),
    ]
}

#[cfg(all(windows))]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        Win32Surface::name().as_ptr(),
        DebugReport::name().as_ptr(),
    ]
}
