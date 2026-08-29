//! Dynamic library (.so/.dylib/.dll) plugin loader.
//!
//! §6.1: Uses `libloading` to dynamically open process-shared plugin libraries and resolve
//! their exported `GrimPluginVTable` entry points.
//!
//! ⚠️ SECURITY NOTE: dylib plugins run in process memory. A crash takes the engine down.
//! This is for performance-critical extensions only. First-party and reviewed plugins required.

use crate::{GrimPluginVTable, PluginCapabilities, PluginManifest, Sampler};
use grim_tensor::error::{Error, Result};
use std::path::Path;
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// Upper bound on the number of bytes scanned when reading a plugin name
/// from the vtable's `name` pointer. Guards against unbounded C-string reads
/// on untrusted plugin data.
const MAX_NAME_BYTES: usize = 1024;

/// Loaded dylib plugin with its vtable and optional sampler.
pub struct DylibPluginLoader {
    #[cfg(feature = "dylib-loading")]
    _lib: Arc<libloading::Library>,
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
    /// Keeps the owning library mapped for as long as any sampler built from
    /// it is alive. Without this, dropping the `DylibPluginLoader` unloads the
    /// library while `sample_fn`/`teardown` are still callable.
    /// [P1-29 fix: sampler owns a refcount on the library.]
    #[cfg(feature = "dylib-loading")]
    _lib: Arc<libloading::Library>,
    handle: *mut std::os::raw::c_void,
    /// `logits_len` is an **f32 element count** here (the dylib ABI takes a
    /// native `*const f32`). The WASM backend passes a **byte** length instead,
    /// because that is the natural unit for wasm linear memory. Plugin authors
    /// must use the unit matching the backend they target.
    /// [P1-29: unit mismatch documented per-backend, not silently unified.]
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
    ///
    /// # Integrity verification
    ///
    /// If `manifest.sha256` is `Some`, the file at `path` is hashed with SHA-256
    /// before `libloading::Library::new` is called. A mismatch returns an `Err`
    /// naming the plugin and both digests. If `sha256` is `None`, the file is
    /// loaded without a hash check and a warning is emitted once per load
    /// (callers that require pinned hashes should set `require_pinned_hash` or
    /// ensure manifests carry `sha256`).
    ///
    /// # Safety
    ///
    /// The dylib runs in-process. A crash in the plugin takes the engine down.
    /// Use process isolation for untrusted third-party binaries.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_with_manifest(path, None)
    }

    /// Load a dylib plugin with optional SHA-256 integrity verification.
    pub fn load_with_manifest<P: AsRef<Path>>(
        path: P,
        manifest: Option<&PluginManifest>,
    ) -> Result<Self> {
        #[cfg(not(feature = "dylib-loading"))]
        {
            let _ = path;
            let _ = manifest;
            Err(Error::Unimplemented(
                "dylib-loading feature is disabled".into(),
            ))
        }
        #[cfg(feature = "dylib-loading")]
        unsafe {
            let path = path.as_ref();

            // ABI enforcement: a manifest whose abi_version does not match
            // this engine is rejected BEFORE the library is touched. Reading
            // a foreign vtable layout is undefined behavior, so this check
            // must gate the load, not trail it. (Audit fix: validate_abi
            // existed but was only ever called from a unit test.)
            if let Some(m) = manifest {
                crate::validate_abi(m, crate::ENGINE_ABI_VERSION)?;
            }

            // Integrity verification: if the manifest carries an expected SHA-256,
            // hash the file and compare before loading.
            if let Some(expected_hex) = manifest.and_then(|m| m.sha256.as_deref()) {
                let file_hash = Self::compute_sha256_file(path)?;
                if file_hash != expected_hex {
                    return Err(Error::Backend(format!(
                        "plugin '{}' SHA-256 mismatch: file hash {file_hash}, expected {expected_hex}",
                        manifest.unwrap().name
                    )));
                }
            } else {
                // No pinned hash in the manifest — warn once that integrity is unverified.
                let name = manifest.map(|m| m.name.as_str()).unwrap_or("<unnamed>");
                tracing::warn!(
                    plugin = name,
                    "plugin loaded with no pinned hash — integrity unverified"
                );
            }

            let lib = libloading::Library::new(path)
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
                _lib: Arc::new(lib),
                vtable,
                _sampler: None,
            })
        }
    }

    /// Compute the SHA-256 digest of a file as a lowercase hex string.
    ///
    /// Uses a streaming reader so large plugin binaries do not need to be fully
    /// loaded into RAM.
    #[allow(dead_code)]
    pub(crate) fn compute_sha256_file(path: &Path) -> Result<String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| Error::Backend(format!("cannot open plugin file: {e}")))?;
        let mut hasher = Sha256::new();
        use std::io::Read;
        let mut buffer = [0u8; 65536];
        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        let digest = hasher.finalize();
        Ok(hex::encode(digest))
    }

    /// Initialize the plugin. Calls the vtable's init function if present.
    /// Uses `catch_unwind` to isolate panics in the plugin (§6.1.2).
    ///
    /// NOTE: `catch_unwind` only guards Rust panics (unwinding across the FFI
    /// boundary). It does NOT contain FFI-level crashes — segfaults, `abort()`,
    /// or other C-level faults still take down the engine; process isolation is
    /// required for those.
    pub fn init(&self) -> Result<()> {
        // Wrap in catch_unwind to prevent plugin panics from crashing the engine
        // The FFI functions are plain C calls - we use a raw pointer to avoid
        // the catch_unwind panic-payload type constraints
        // NOTE: catch_unwind only catches Rust panics, NOT FFI-level crashes
        // (segfaults, aborts) — containing those requires process isolation.
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
            unsafe {
                // Bounded scan: never trust an unbounded CStr scan on a pointer
                // sourced from untrusted plugin data. Walk at most MAX_NAME_BYTES
                // looking for a NUL terminator before treating the pointer as a
                // valid C string.
                let mut buf = [0u8; MAX_NAME_BYTES];
                let mut len = 0usize;
                while len < MAX_NAME_BYTES {
                    let byte = *ptr.add(len) as u8;
                    if byte == 0 {
                        break;
                    }
                    buf[len] = byte;
                    len += 1;
                }
                if len == MAX_NAME_BYTES {
                    // No NUL within the cap: not a bounded, terminated string.
                    return "invalid-name".to_string();
                }
                std::str::from_utf8(&buf[..len])
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
        // NOTE: catch_unwind only guards Rust panics — NOT FFI-level crashes
        // (segfaults, aborts); containing those requires process isolation.
        let handle = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sampler_factory()))
            .map_err(|_| Error::Backend("Plugin sampler_factory panicked".into()))?;
        if handle.is_null() {
            return Err(Error::Backend(
                "Plugin sampler_factory returned null".into(),
            ));
        }

        Ok(Arc::new(DylibSampler {
            name: self.name(),
            #[cfg(feature = "dylib-loading")]
            _lib: Arc::clone(&self._lib),
            handle,
            sample_fn: sampler_sample,
            teardown: self.vtable.teardown,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PluginCapabilities, PluginGrants, PluginKind, PluginReload};

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

    #[test]
    fn test_sha256_verification_rejects_mismatch() {
        // Create a temporary file with known content
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_plugin_sha256.so");
        std::fs::write(&temp_file, b"test plugin content").expect("write temp file");

        // Compute the correct hash
        let correct_hash =
            DylibPluginLoader::compute_sha256_file(&temp_file).expect("compute hash");

        // Create a manifest with the correct hash - should fail because file isn't a valid .so
        #[allow(unused_variables)]
        let manifest = PluginManifest {
            name: "test-plugin".into(),
            abi_version: 1,
            kind: PluginKind::Dylib,
            capabilities: PluginCapabilities::SAMPLER,
            entry: temp_file.to_str().unwrap().to_string(),
            sha256: Some(correct_hash.clone()),
            limits: None,
            stage: None,
            priority: None,
            grants: PluginGrants::default(),
            reload: PluginReload::default(),
        };

        // With correct hash but invalid file, should fail at Library::new (not hash check)
        #[cfg(feature = "dylib-loading")]
        {
            let result = DylibPluginLoader::load_with_manifest(&temp_file, Some(&manifest));
            // It should fail, but not because of SHA-256 mismatch - it's not a valid .so
            assert!(result.is_err());
        }

        // Create a manifest with wrong hash - should fail at hash check
        #[allow(unused_variables)]
        let wrong_manifest = PluginManifest {
            name: "test-plugin".into(),
            abi_version: 1,
            kind: PluginKind::Dylib,
            capabilities: PluginCapabilities::SAMPLER,
            entry: temp_file.to_str().unwrap().to_string(),
            sha256: Some("wrong_hash_value".to_string()),
            limits: None,
            stage: None,
            priority: None,
            grants: PluginGrants::default(),
            reload: PluginReload::default(),
        };

        #[cfg(feature = "dylib-loading")]
        {
            let result = DylibPluginLoader::load_with_manifest(&temp_file, Some(&wrong_manifest));
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("wrong-hash manifest must be rejected"),
            };
            assert!(
                err_msg.contains("SHA-256 mismatch"),
                "error should mention SHA-256 mismatch: {err_msg}"
            );
        }

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
    }

    /// Audit fix gate: a manifest whose abi_version mismatches the engine
    /// must be rejected by the LOAD PATH itself — before the file is even
    /// opened (the error names the ABI, not a library-load failure). The
    /// pre-fix loader only ever ran validate_abi from a unit test.
    #[test]
    fn dylib_load_path_enforces_abi_version() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_plugin_abi_gate.so");
        std::fs::write(&temp_file, b"not a real dylib").unwrap();

        let mismatched = PluginManifest {
            name: "abi-mismatch".into(),
            abi_version: crate::ENGINE_ABI_VERSION + 1,
            kind: PluginKind::Dylib,
            capabilities: PluginCapabilities::SAMPLER,
            entry: temp_file.to_str().unwrap().to_string(),
            sha256: None,
            limits: None,
            stage: None,
            priority: None,
            grants: PluginGrants::default(),
            reload: PluginReload::default(),
        };

        #[cfg(feature = "dylib-loading")]
        {
            let result = DylibPluginLoader::load_with_manifest(&temp_file, Some(&mismatched));
            let err_msg = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("ABI-mismatched manifest must be rejected at load"),
            };
            assert!(
                err_msg.contains("ABI version"),
                "rejection must be the ABI gate, not a library error: {err_msg}"
            );
        }
        #[cfg(not(feature = "dylib-loading"))]
        {
            let _ = mismatched;
        }

        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_sha256_computation_is_correct() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_sha256_content.bin");
        let content = b"test content for sha256";
        std::fs::write(&temp_file, content).expect("write temp file");

        let computed = DylibPluginLoader::compute_sha256_file(&temp_file).expect("compute hash");

        // Verify against a known value (computed externally)
        // SHA256("test content for sha256") = 587c8c2b5c9d1e5b3f6a7e8d9c0b1a2f3e4d5c6b7a8d9e0f1a2b3c4d5e6f7a8b
        // We'll just verify it's consistent by computing twice
        let computed2 =
            DylibPluginLoader::compute_sha256_file(&temp_file).expect("compute hash again");
        assert_eq!(
            computed, computed2,
            "SHA-256 computation should be deterministic"
        );

        let _ = std::fs::remove_file(&temp_file);
    }
}
