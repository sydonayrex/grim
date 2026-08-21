//! Persistent disk cache for compiled ROCm HSA Code Objects (HSACO).
//!
//! Avoids expensive hipRTC multi-second recompilation stalls during cold starts
//! on consumer GPUs and APUs by caching binary bytecode indexed by SeaHash key.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Resolve the directory where compiled HSA code objects are cached.
pub fn jit_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("GRIM_JIT_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("grim").join("rocm_code_objects");
    }
    std::env::temp_dir().join("grim_rocm_code_objects")
}

/// Compute a 64-bit SeaHash cache key for a given architecture, kernel source, and options.
pub fn compute_cache_key(arch: &str, source: &str, options: &[std::ffi::CString]) -> String {
    let mut hasher = seahash::SeaHasher::new();
    use std::hash::Hasher;
    hasher.write(arch.as_bytes());
    hasher.write(b"::");
    hasher.write(source.as_bytes());
    for opt in options {
        hasher.write(b"::");
        hasher.write(opt.as_bytes());
    }
    format!("{}_{:016x}", arch, hasher.finish())
}

/// Attempt to read a cached HSA code object from disk.
pub fn load_cached_code_object(cache_key: &str) -> Option<Vec<u8>> {
    let path = jit_cache_dir().join(format!("{}.hsaco", cache_key));
    let mut file = File::open(&path).ok()?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).ok()?;
    if buffer.is_empty() {
        None
    } else {
        Some(buffer)
    }
}

/// Store a freshly compiled HSA code object binary to disk.
pub fn store_cached_code_object(cache_key: &str, bytes: &[u8]) -> std::io::Result<()> {
    let dir = jit_cache_dir();
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.hsaco", cache_key));
    let mut file = File::create(&path)?;
    file.write_all(bytes)?;
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_compute_cache_key_deterministic() {
        let opts = vec![CString::new("--std=c++17").unwrap()];
        let k1 = compute_cache_key("gfx1100", "int main() {}", &opts);
        let k2 = compute_cache_key("gfx1100", "int main() {}", &opts);
        assert_eq!(k1, k2);
        assert!(k1.starts_with("gfx1100_"));
    }

    #[test]
    fn test_store_and_load_cache_roundtrip() {
        let test_key = "test_arch_0123456789abcdef";
        let test_payload = b"\x7fELF_FAKE_HSACO_BYTES";
        store_cached_code_object(test_key, test_payload).expect("store should succeed");
        let loaded = load_cached_code_object(test_key).expect("load should succeed");
        assert_eq!(loaded, test_payload);

        // Cleanup
        let path = jit_cache_dir().join(format!("{}.hsaco", test_key));
        let _ = fs::remove_file(path);
    }
}
