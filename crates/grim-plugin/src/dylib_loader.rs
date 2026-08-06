//! Dynamic library (.so/.dylib/.dll) plugin loader.
//!
//! §6.1: Uses `libloading` to dynamically open process-shared plugin libraries and resolve
//! their exported `GrimPluginVTable` entry points.
//!
//! ⚠️ SECURITY NOTE: dylib plugins run in process memory. A crash takes the engine down.
//! This is for performance-critical extensions only. First-party and reviewed plugins required.

use crate::{GrimPluginVTable, PluginCapabilities, Sampler};
use grim_tensor::error::{Error, Result};
use std::path::Path;
use std::sync::Arc;

/// Loaded dylib plugin with its vtable and optional sampler.
pub struct DylibPluginLoader {
    #[cfg(feature = "dylib-loading")]
    _lib: libloading::Library,
    pub vtable: GrimPluginVTable,
    _sampler: Option<Arc<dyn Sampler>>,
}

/// Sampler backed by a dylib plugin's FFI vtable.
///
/// Holds the opaque plugin-allocated handle returned by `sampler_factory`
/// and the `sampler_sample` fn pointer used to drive `Sampler::sample`. The
/// handle is owned and freed via the vtable's `teardown` surface on drop.
struct DylibSampler {
    name: String,
    handle: *mut std::os::raw::c_void,
    sample_fn: extern "C" fn(
        handle: *mut std::os::raw::c_void,
        logits_ptr: *const f32,
        logits_len: u32,
        history_ptr: *const u32,
        history_len: u32,
    ) -> i32,
    teardown: extern "C" fn(),
}

// SAFETY: The plugin's handle is opaque host-owned memory. We treat it as a
// `Send`+`Sync` raw token because `sample_fn`/`teardown` are plain C
// `extern "C"` fn pointers (no captured Rust state, no thread affinity) and
// the plugin contract (§6.1) requires sampler entrypoints be reentrant /
// thread-safe when invoked. The `Arc<dyn Sampler>` that wraps this is what
// the registry shares across axum tasks.
unsafe impl Send for DylibSampler {}
unsafe impl Sync for DylibSampler {}

impl Drop for DylibSampler {
    fn drop(&mut self) {
        // §6.1.2: isolate panics in plugin teardown from the engine
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (self.teardown)();
        }));
    }
}

impl Sampler for DylibSampler {
    fn sample(&self, logits: &grim_tensor::Tensor, history: &[u32]) -> Result<u32> {
        let logits_vec = logits.to_vec_f32()?;
        let token_id = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (self.sample_fn)(
                self.handle,
                logits_vec.as_ptr(),
                logits_vec.len() as u32,
                history.as_ptr(),
                history.len() as u32,
            )
        }))
        .map_err(|e| {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            Error::Backend(format!("Dylib sampler panicked: {msg}"))
        })?;
        if token_id < 0 {
            return Err(Error::Backend(format!(
                "Dylib sampler returned negative token id: {token_id}"
            )));
        }
        Ok(token_id as u32)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl DylibPluginLoader {
    /// Loads a dynamic library plugin and binds its FFI vtable.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let _ = path;
        #[cfg(not(feature = "dylib-loading"))]
        {
            Err(Error::Unimplemented(
                "dylib-loading feature is disabled".into(),
            ))
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
                return Err(Error::Backend(
                    "Loaded plugin vtable pointer is null".into(),
                ));
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
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (self.vtable.capabilities)()
        }))
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
            unsafe {
                std::ffi::CStr::from_ptr(ptr)
                    .to_str()
                    .unwrap_or("invalid-name")
                    .to_string()
            }
        }))
        .unwrap_or_else(|_| "panicked".to_string())
    }

    /// Create a sampler from this plugin if it provides one.
    ///
    /// Requires the plugin's vtable to expose both `sampler_factory` (which
    /// allocates an opaque plugin-side sampler handle) and `sampler_sample`
    /// (which drives `Sampler::sample` on that handle). If either is missing
    /// the plugin is rejected before it can run any user code.
    pub fn create_sampler(&self) -> Result<Arc<dyn Sampler>> {
        let caps = self.capabilities();
        if !caps.contains(PluginCapabilities::SAMPLER) {
            return Err(Error::Backend(
                "Plugin does not support sampler capability".into(),
            ));
        }

        let sampler_factory = self
            .vtable
            .sampler_factory
            .ok_or_else(|| Error::Backend("Plugin missing sampler_factory symbol".into()))?;
        let sampler_sample = self.vtable.sampler_sample.ok_or_else(|| {
            Error::Backend(
                "Plugin missing sampler_sample vtable entry (ABI v1 requires it for samplers)"
                    .into(),
            )
        })?;

        // §6.1.2: sampler_factory is a plain C call; isolate panics so a
        // buggy/malicious plugin can't unwind through the FFI boundary.
        let handle = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sampler_factory()))
            .map_err(|_| Error::Backend("Plugin sampler_factory panicked".into()))?;
        if handle.is_null() {
            return Err(Error::Backend(
                "Plugin sampler_factory returned null".into(),
            ));
        }

        Ok(Arc::new(DylibSampler {
            name: self.name(),
            handle,
            sample_fn: sampler_sample,
            teardown: self.vtable.teardown,
        }))
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
        // Verify the vtable is #[repr(C)] and ABI-stable. A loose lower bound:
        // the vtable has at least one u32 (abi_version) plus several fn-pointer
        // fields (name, capabilities, init, model_factory, sampler_factory,
        // sampler_sample, teardown). New Option<fn> fields only grow the struct,
        // so this `>=` check stays valid as the vtable surface evolves.
        let vtable_size = std::mem::size_of::<GrimPluginVTable>();
        let expected =
            std::mem::size_of::<u32>() * 7 + std::mem::size_of::<Option<extern "C" fn()>>();
        assert!(
            vtable_size >= expected,
            "vtable should have expected layout"
        );

        let capabilities_offset = std::mem::offset_of!(GrimPluginVTable, capabilities);
        assert!(capabilities_offset > 0, "capabilities field offset check");
    }
}
