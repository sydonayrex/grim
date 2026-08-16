//! Compile-once, cache-to-disk `.hsaco` cache for compiled HIP kernels. [see: `(entry, gpu_target, seahash(source))`, `hipModuleLoad`]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::SystemTime;

use grim_tensor::error::Result;

/// Cache key identifying a JIT compiled kernel by entry, GPU target, hardware fingerprint, and source hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JitCacheKey {
    /// Kernel entry point name.
    pub entry: String,
    /// GPU target architecture string (e.g., "gfx1036").
    pub gpu_target: String,
    /// Hardware feature fingerprint string.
    pub hardware_fingerprint: String,
    /// Hash of the complete kernel source code.
    pub source_hash: u64,
}

impl JitCacheKey {
    /// Create a new cache key snapshot from hardware spec and source hash.
    pub fn from_spec(
        entry: &str,
        gpu_target: &str,
        spec: &crate::device::hardware_spec::HardwareSpec,
        source_hash: u64,
    ) -> Self {
        let fingerprint = format!(
            "{}:{}:{}:{}:{}",
            spec.wavefront_size,
            spec.max_shared_mem_per_block,
            spec.cu_count,
            spec.multiprocessor_count,
            spec.max_threads_per_block,
        );
        JitCacheKey {
            entry: entry.to_string(),
            gpu_target: gpu_target.to_string(),
            hardware_fingerprint: fingerprint,
            source_hash,
        }
    }

    /// Format the cache key into a unique file prefix string.
    pub fn to_key_string(&self) -> String {
        format!(
            "grim_{}_{}_{}_{:016x}",
            self.entry, self.gpu_target, self.hardware_fingerprint, self.source_hash
        )
    }
}

/// Cache for compiled .hsaco kernels. The in-memory map also stores the

/// (possibly C++-mangled) *lowered* kernel name so `hipModuleGetFunction` can
/// resolve kernels that hipRTC emits mangled (e.g. `grim_moe_fused_grouped_fp8`).
#[derive(Debug)]
pub struct HsacoKernelCache {
    cache_dir: PathBuf,
    entries: RwLock<HashMap<String, (PathBuf, SystemTime, String)>>,
}

impl HsacoKernelCache {
    pub fn new() -> Self {
        let cache_dir = std::env::var("GRIM_HSACO_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let mut dir = std::env::temp_dir();
                dir.push("grim_hsaco_cache");
                dir
            });

        if !cache_dir.exists() {
            let _ = fs::create_dir_all(&cache_dir);
        }

        // NOTE: we deliberately do NOT pre-populate `entries` from on-disk
        // .hsaco files. The lowered (possibly mangled) kernel name is computed
        // at JIT-compile time and stored in-memory here; a cold start just
        // reuses the existing on-disk .hsaco via `cache_kernel`'s exists-check.
        // This keeps `hipModuleGetFunction` pointing at the correct symbol.
        let entries_lock = RwLock::new(HashMap::new());

        Self {
            cache_dir,
            entries: entries_lock,
        }
    }

    pub fn get_cached_kernel(&self, key: &str) -> Option<(PathBuf, String)> {
        let entries = self.entries.read().unwrap();
        if let Some((path, _, lowered)) = entries.get(key) {
            if path.exists() {
                return Some((path.clone(), lowered.clone()));
            }
        }
        None
    }

    pub fn cache_kernel(
        &self,
        key: &str,
        source: &str,
        compiled: &[u8],
        lowered_name: &str,
    ) -> Result<PathBuf> {
        let hash = seahash::hash(source.as_bytes());
        let cache_key = format!("{}_{:016x}.hsaco", key, hash);
        let cache_path = self.cache_dir.join(&cache_key);

        if cache_path.exists() {
            let metadata = fs::metadata(&cache_path)?;
            let modified = metadata.modified()?;
            self.entries.write().unwrap().insert(
                key.to_string(),
                (cache_path.clone(), modified, lowered_name.to_string()),
            );
            return Ok(cache_path);
        }

        let _ = fs::create_dir_all(&self.cache_dir);
        fs::write(&cache_path, compiled)?;

        let metadata = fs::metadata(&cache_path)?;
        let modified = metadata.modified()?;
        self.entries.write().unwrap().insert(
            key.to_string(),
            (cache_path.clone(), modified, lowered_name.to_string()),
        );

        Ok(cache_path)
    }

    pub fn invalidate(&self, key: &str) {
        if let Some((path, _, _)) = self.entries.write().unwrap().remove(key) {
            let _ = fs::remove_file(path);
        }
    }
}

impl Default for HsacoKernelCache {
    fn default() -> Self {
        Self::new()
    }
}
