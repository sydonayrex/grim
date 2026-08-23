//! Compile-once, cache-to-disk `.hsaco` cache for compiled HIP kernels. [see: `(entry, gpu_target, toolchain, seahash(source))`, `hipModuleLoad`]

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
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
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
        // A code object compiled by a previous ROCm/HIPRTC build can fault in
        // `hipModuleLoad` after a driver upgrade. Keep versions separate even
        // when source and GPU target are unchanged.
        let cache_key = format!(
            "{}_{}_{:016x}.hsaco",
            key,
            crate::device::jit_cache::toolchain_fingerprint(),
            hash
        );
        let cache_path = self.cache_dir.join(&cache_key);

        let _ = fs::create_dir_all(&self.cache_dir);
        let tmp_path = self.cache_dir.join(format!(
            ".tmp_{}_{:016x}_{}_{:x}",
            key,
            hash,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::write(&tmp_path, compiled)?;
        // `rename` atomically replaces an old entry. This matters after a
        // failed or stale compile: a successful HIPRTC result must not be
        // discarded merely because a same-key file already exists.
        fs::rename(&tmp_path, &cache_path)?;

        let metadata = fs::metadata(&cache_path)?;
        let modified = metadata.modified()?;
        self.entries
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                key.to_string(),
                (cache_path.clone(), modified, lowered_name.to_string()),
            );

        Ok(cache_path)
    }

    pub fn invalidate(&self, key: &str) {
        if let Some((path, _, _)) = self
            .entries
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key)
        {
            let _ = fs::remove_file(path);
        }
    }
}

impl Default for HsacoKernelCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::hardware_spec::HardwareSpec;
    use crate::peer_access::{LinkType, P2PTopology};

    fn make_spec(
        gcn_arch: &str,
        wavefront_size: u32,
        max_shared_mem: u32,
        max_threads_per_block: u32,
        cu_count: u32,
    ) -> HardwareSpec {
        HardwareSpec {
            gcn_arch: gcn_arch.to_string(),
            wavefront_size,
            max_shared_mem_per_block: max_shared_mem,
            max_threads_per_block,
            cu_count,
            multiprocessor_count: cu_count,
            mem_bandwidth_gb_s: 500.0,
            peak_flops_fp16: 8.0e12,
            p2p_topology: P2PTopology {
                device_count: 1,
                links: vec![vec![LinkType::NoLink]],
            },
        }
    }

    #[test]
    fn jit_cache_key_from_spec_produces_key_string() {
        let spec = make_spec("gfx1036", 64, 65536, 1024, 64);
        let key = JitCacheKey::from_spec("grim_decode_gemm_f16", "gfx1036", &spec, 0xabcd1234u64);
        let s = key.to_key_string();
        assert!(
            !s.is_empty(),
            "JitCacheKey::to_key_string should produce non-empty string"
        );
        assert!(
            s.contains("grim_decode_gemm_f16"),
            "key string should contain kernel name"
        );
        assert!(s.contains("gfx1036"), "key string should contain arch");
    }

    #[test]
    fn jit_cache_key_roundtrip_via_string() {
        let spec = make_spec("gfx90a", 64, 65536, 1024, 64);
        let key =
            JitCacheKey::from_spec("grim_qkv_attention", "gfx90a", &spec, 0x1234567890abcdefu64);
        let _s = key.to_key_string();
        let key2 =
            JitCacheKey::from_spec("grim_qkv_attention", "gfx90a", &spec, 0x1234567890abcdefu64);
        assert_eq!(
            key.to_key_string(),
            key2.to_key_string(),
            "same spec  ->  same key string (cache coherence)"
        );
    }

    #[test]
    fn hsaco_kernel_cache_insert_and_get() {
        let cache = HsacoKernelCache::new();
        let key = "test_key_1";
        let _path = std::path::PathBuf::from("/tmp/test_kernel.ptx");
        let src = "kernel void test() {}";
        cache
            .cache_kernel(key, src, &[0u8; 8], "test_kernel")
            .expect("cache_kernel should succeed");
        let got = cache.get_cached_kernel(key);
        assert!(
            got.is_some(),
            "get_cached_kernel should return Some after insert"
        );
        let (got_path, got_lowered) = got.unwrap();
        assert_eq!(got_lowered, "test_kernel");
        assert!(got_path.exists());
    }

    #[test]
    fn hsaco_kernel_cache_replaces_an_existing_code_object() {
        let cache = HsacoKernelCache::new();
        let key = format!(
            "replace_existing_code_object_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let source = "kernel void test() {}";

        let path = cache
            .cache_kernel(&key, source, b"old code object", "test_kernel")
            .expect("initial code object should be cached");
        cache
            .cache_kernel(&key, source, b"fresh code object", "test_kernel")
            .expect("fresh code object should replace the old one");

        assert_eq!(
            fs::read(&path).expect("cached code object should be readable"),
            b"fresh code object"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn hsaco_kernel_cache_invalidate_removes_entry() {
        let cache = HsacoKernelCache::new();
        let key = "test_key_2";
        cache
            .cache_kernel(key, "src", &[0u8; 8], "test_kernel")
            .unwrap();
        assert!(cache.get_cached_kernel(key).is_some());
        cache.invalidate(key);
        assert!(
            cache.get_cached_kernel(key).is_none(),
            "invalidated key should no longer be found"
        );
    }

    #[test]
    fn hsaco_kernel_cache_covers_empty_get() {
        let cache = HsacoKernelCache::new();
        assert!(cache.get_cached_kernel("nonexistent").is_none());
    }
}
