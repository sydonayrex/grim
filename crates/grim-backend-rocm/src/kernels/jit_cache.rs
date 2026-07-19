//! Compile-once, cache-to-disk `.hsaco` cache for compiled HIP kernels.
//!
//! Item 2 of the ROCm spec: compiled binary persistence keyed by
//! `(entry, gpu_target, seahash(source))` so a recurring kernel
//! doesn't pay `hipModuleLoad` cost on every dispatch — the per-process
//! `RocmDevice` keeps its own in-memory module cache on top of this.
//!
//! Skill attribution:
//! - `rust-gpu-discipline` §4 — JIT cache is part of how we keep warm
//!   launches cheap; without it every cold call would re-`hc`-compile.
//! - `rust-ai-ml-inference-guide` Action 9 — runtime caching of compiled
//!   artifacts across process restarts reduces first-step latency.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::SystemTime;

use grim_tensor::error::Result;

/// Cache for compiled .hsaco kernels.
#[derive(Debug)]
pub struct HsacoKernelCache {
    cache_dir: PathBuf,
    entries: RwLock<HashMap<String, (PathBuf, SystemTime)>>,
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

        let entries_lock = RwLock::new(HashMap::new());
        if let Ok(paths) = fs::read_dir(&cache_dir) {
            let mut map = entries_lock.write().unwrap();
            for entry in paths.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().map_or(false, |ext| ext == "hsaco") {
                    if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                        if let Some(last_underscore) = filename.rfind('_') {
                            let key_part = &filename[..last_underscore];
                            if let Ok(metadata) = entry.metadata() {
                                if let Ok(modified) = metadata.modified() {
                                    map.insert(key_part.to_string(), (path.clone(), modified));
                                }
                            }
                        }
                    }
                }
            }
        }

        Self {
            cache_dir,
            entries: entries_lock,
        }
    }

    pub fn get_cached_kernel(&self, key: &str) -> Option<PathBuf> {
        let entries = self.entries.read().unwrap();
        if let Some((path, _)) = entries.get(key) {
            if path.exists() {
                return Some(path.clone());
            }
        }
        None
    }

    pub fn cache_kernel(&self, key: &str, source: &str, compiled: &[u8]) -> Result<PathBuf> {
        let hash = seahash::hash(source.as_bytes());
        let cache_key = format!("{}_{:016x}.hsaco", key, hash);
        let cache_path = self.cache_dir.join(&cache_key);

        if cache_path.exists() {
            let metadata = fs::metadata(&cache_path)?;
            let modified = metadata.modified()?;
            self.entries.write().unwrap().insert(key.to_string(), (cache_path.clone(), modified));
            return Ok(cache_path);
        }

        fs::write(&cache_path, compiled)?;

        let metadata = fs::metadata(&cache_path)?;
        let modified = metadata.modified()?;
        self.entries.write().unwrap().insert(key.to_string(), (cache_path.clone(), modified));

        Ok(cache_path)
    }

    pub fn invalidate(&self, key: &str) {
        if let Some((path, _)) = self.entries.write().unwrap().remove(key) {
            let _ = fs::remove_file(path);
        }
    }
}

impl Default for HsacoKernelCache {
    fn default() -> Self {
        Self::new()
    }
}
