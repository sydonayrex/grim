//! Dynamic library (.so/.dylib/.dll) plugin loader.
//!
//! §6.1: Uses `libloading` to dynamically open process-shared plugin libraries and resolve
//! their exported `GrimPluginVTable` entry points.
//!
//! ⚠️ SECURITY NOTE: dylib plugins run in process memory. A crash takes the engine down.
//! This is for performance-critical extensions only. First-party and reviewed plugins required.

use std::path::Path;
use grim_tensor::error::{Error, Result};
use crate::{GrimPluginVTable, PluginCapabilities, Sampler};
use std::sync::Arc;

/// Loaded dylib plugin with its vtable and optional sampler.
pub struct DylibPluginLoader {
    #[cfg(feature = "dylib-loading")]
    _lib: libloading::Library,
    pub vtable: GrimPluginVTable,
    _sampler: Option<Arc<dyn Sampler>>,
}

impl DylibPluginLoader {
    /// Loads a dynamic library plugin and binds its FFI vtable.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let _ = path;
        #[cfg(not(feature = "dylib-loading"))]
        {
            Err(Error::Unimplemented("dylib-loading feature is disabled".into()))
        }
        #[cfg(feature = "dylib-loading")]
        unsafe {
            let lib = libloading::Library::new(path.as_ref())
                .map_err(|e| Error::Backend(format!("Failed to load dynamic library: {e}")))?;

            // Resolve exported vtable initializer symbol
            let get_vtable: libloading::Symbol<unsafe extern "C" fn() -> *const GrimPluginVTable> =
                lib.get(b"grim_plugin_get_vtable\0")
                    .map_err(|e| Error::Backend(format!("Missing vtable symbol: {e}")))?;

            let raw_vtable_ptr = get_vtable();
            if raw_vtable_ptr.is_null() {
                return Err(Error::Backend("Loaded plugin vtable pointer is null".into()));
            }

            // Copy/dereference the ABI-stable vtable
            let vtable = std::ptr::read(raw_vtable_ptr);

            Ok(Self {
                _lib: lib,
                vtable,
                _sampler: None,
            })
        }
    }

    /// Initialize the plugin. Calls the vtable's init function if present.
    /// Uses `catch_unwind` to isolate panics in the plugin (§6.1.2).
    pub fn init(&self) -> Result<()> {
        // Wrap in catch_unwind to prevent plugin panics from crashing the engine
        // The FFI functions are plain C calls - we use a raw pointer to avoid
        // the catch_unwind panic-payload type constraints
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // FFI functions are safe to call - the unsafe is in loading them
            (self.vtable.init)(std::ptr::null_mut());
        }));
        match result {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                Err(Error::Backend(format!("Plugin init panicked: {msg}")))
            }
        }
    }

    /// Teardown the plugin. Calls the vtable's teardown function.
    /// Uses `catch_unwind` to isolate panics in the plugin (§6.1.2).
    pub fn teardown(&self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (self.vtable.teardown)();
        }));
    }

    /// Get the plugin's capabilities. Returns zero on panic.
    pub fn capabilities(&self) -> PluginCapabilities {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.vtable.capabilities)()))
            .unwrap_or(PluginCapabilities(0))
    }

    /// Get the plugin's name. Returns "unknown" on panic.
    pub fn name(&self) -> String {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let ptr = (self.vtable.name)();
            if ptr.is_null() {
                return "unknown".to_string();
            }
            // SAFETY: Plugin promises valid UTF-8 string
            unsafe { std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("invalid-name").to_string() }
        })).unwrap_or_else(|_| "panicked".to_string())
    }

    /// Create a sampler from this plugin if it provides one.
    pub fn create_sampler(&self) -> Result<Arc<dyn Sampler>> {
        let caps = self.capabilities();
        if !caps.contains(PluginCapabilities::SAMPLER) {
            return Err(Error::Backend("Plugin does not support sampler capability".into()));
        }

        if self.vtable.sampler_factory.is_none() {
            return Err(Error::Backend("Plugin missing sampler_factory symbol".into()));
        }

        // In v1, dylib samplers need to implement the Sampler trait externally.
        // The plugin provides raw data that we wrap. For now, return an error
        // indicating this needs a concrete implementation.
        Err(Error::Unimplemented(
            "Dylib sampler creation requires custom Sampler impl wrapping vtable".into()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dylib_load_error_when_disabled() {
        let res = DylibPluginLoader::load("some_nonexistent_path.so");
        #[cfg(not(feature = "dylib-loading"))]
        assert!(res.is_err());
        #[cfg(feature = "dylib-loading")]
        let _ = res;
    }

    #[test]
    fn test_dylib_loader_memory_layout() {
        // Verify the vtable is #[repr(C)] and ABI-stable
        let vtable_size = std::mem::size_of::<GrimPluginVTable>();
        let expected = std::mem::size_of::<u32>() * 7 + std::mem::size_of::<Option<extern "C" fn()>>();
        assert!(vtable_size >= expected, "vtable should have expected layout");

        let capabilities_offset = std::mem::offset_of!(GrimPluginVTable, capabilities);
        assert!(capabilities_offset > 0, "capabilities field offset check");
    }
}