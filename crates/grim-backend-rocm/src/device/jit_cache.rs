//! Persistent disk cache for compiled ROCm HSA Code Objects (HSACO).
//!
//! Avoids expensive hipRTC multi-second recompilation stalls during cold starts
//! on consumer GPUs and APUs by caching binary bytecode indexed by SeaHash key.
//!
//! WI-X9: concurrent test processes (or two grim processes) can JIT-compile the
//! same kernel simultaneously. Writes are therefore (a) cross-process serialized
//! behind an advisory `flock` on a per-cache-dir lock file and (b) atomic via
//! tmp-file + rename, so a reader never observes a partial code object.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Resolve the directory where compiled HSA code objects are cached.
pub fn jit_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("GRIM_JIT_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("grim")
            .join("rocm_code_objects");
    }
    std::env::temp_dir().join("grim_rocm_code_objects")
}

/// Advisory cross-process lock file guarding cache writes (WI-X9).
fn jit_cache_lock_file() -> PathBuf {
    jit_cache_dir().join(".write.lock")
}

/// RAII guard holding an exclusive advisory flock on the cache dir for the
/// duration of a write. Falls back to no-op locking on platforms where flock
/// is unavailable — the atomic tmp+rename still guarantees read safety.
pub struct JitCacheWriteLock {
    #[allow(dead_code)]
    file: File,
}

impl JitCacheWriteLock {
    pub fn acquire() -> std::io::Result<Self> {
        let dir = jit_cache_dir();
        fs::create_dir_all(&dir)?;
        let file = File::create(jit_cache_lock_file())?;
        lock_exclusive(&file)?;
        Ok(Self { file })
    }
}

impl Drop for JitCacheWriteLock {
    fn drop(&mut self) {
        unlock(&self.file);
    }
}

#[cfg(unix)]
fn lock_exclusive(file: &File) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // F_SETLK would block-free; we want blocking F_SETLKW semantics done
    // manually so concurrent first-device-inits serialize instead of failing.
    let rc = unsafe {
        libc_flock(file.as_raw_fd(), /* LOCK_EX */ 2)
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn unlock(file: &File) {
    use std::os::unix::io::AsRawFd;
    unsafe {
        let _ = libc_flock(file.as_raw_fd(), /* LOCK_UN */ 8);
    }
}

#[cfg(unix)]
unsafe fn libc_flock(fd: i32, operation: i32) -> i32 {
    // `flock(2)` — available on every unix target grim builds for.
    unsafe extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }
    // SAFETY: flock(2) on a valid open fd; no memory touched.
    unsafe { flock(fd, operation) }
}

#[cfg(not(unix))]
fn lock_exclusive(_file: &File) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn unlock(_file: &File) {}

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

/// Store a freshly compiled HSA code object binary to disk atomically and
/// under an advisory cross-process lock (WI-X9): tmp write → fsync → rename,
/// with all writers serialized by `.write.lock` in the cache dir. Readers need
/// no lock — rename is atomic, so they see either the old object or the new.
pub fn store_cached_code_object(cache_key: &str, bytes: &[u8]) -> std::io::Result<()> {
    let dir = jit_cache_dir();
    fs::create_dir_all(&dir)?;
    let _lock = JitCacheWriteLock::acquire()?;

    let target_path = dir.join(format!("{}.hsaco", cache_key));
    let tmp_path = dir.join(format!(
        "{}.{}.{}.tmp",
        cache_key,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));

    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all().ok(); // durability before rename; best-effort
    }

    // Atomic rename replaces the destination file safely
    fs::rename(tmp_path, target_path)
}

/// Remove stale `.tmp` files left by crashed writers (WI-X9 hygiene helper).
pub fn prune_stale_tmp_files(max_age_secs: u64) {
    let dir = jit_cache_dir();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.ends_with(".tmp") {
                continue;
            }
            let age_ok = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0)
                > max_age_secs;
            if age_ok {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::sync::Barrier;
    use std::thread;

    /// These tests mutate the process-global `GRIM_JIT_CACHE_DIR`; serialize them.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn temp_cache_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "grim_jit_cache_test_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        // SAFETY: test-only, single-threaded at this point.
        unsafe { std::env::set_var("GRIM_JIT_CACHE_DIR", &dir) };
        dir
    }

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
        let _env = env_lock();
        let dir = temp_cache_dir("roundtrip");
        let test_key = "test_arch_0123456789abcdef";
        let test_payload = b"\x7fELF_FAKE_HSACO_BYTES";
        store_cached_code_object(test_key, test_payload).expect("store should succeed");
        let loaded = load_cached_code_object(test_key).expect("load should succeed");
        assert_eq!(loaded, test_payload);

        // Cleanup
        let path = jit_cache_dir().join(format!("{}.hsaco", test_key));
        let _ = fs::remove_file(path);
        let _ = fs::remove_dir_all(dir);
        unsafe { std::env::remove_var("GRIM_JIT_CACHE_DIR") };
    }

    /// WI-X9: N threads storing distinct keys concurrently must all succeed and
    /// leave fully-readable objects (no torn writes, no lost updates).
    #[test]
    fn test_concurrent_writers_all_succeed() {
        let _env = env_lock();
        let dir = temp_cache_dir("concurrent");
        const N: usize = 8;
        let barrier = std::sync::Arc::new(Barrier::new(N));
        let mut handles = Vec::new();
        for i in 0..N {
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                let key = format!("conc_arch_{i:016x}");
                let payload = vec![i as u8; 256];
                store_cached_code_object(&key, &payload).expect("concurrent store");
                key
            }));
        }
        for h in handles {
            let key = h.join().expect("writer thread");
            let loaded = load_cached_code_object(&key).expect("object readable after store");
            assert_eq!(loaded.len(), 256, "no torn write");
        }
        let _ = fs::remove_dir_all(dir);
        unsafe { std::env::remove_var("GRIM_JIT_CACHE_DIR") };
    }

    /// Same key from many threads: exactly one wins the rename per round; the
    /// final file must be a complete payload of one of the writers.
    #[test]
    fn test_same_key_racing_writers_leave_intact_object() {
        let _env = env_lock();
        let dir = temp_cache_dir("race");
        let key = "race_arch_deadbeefdeadbeef";
        const N: usize = 8;
        let barrier = std::sync::Arc::new(Barrier::new(N));
        let mut handles = Vec::new();
        for i in 0..N {
            let barrier = barrier.clone();
            let key = key.to_string();
            handles.push(thread::spawn(move || {
                barrier.wait();
                store_cached_code_object(&key, &[i as u8 + 1; 512]).is_ok()
            }));
        }
        let ok_count = handles
            .into_iter()
            .map(|h| h.join().unwrap_or(false))
            .filter(|ok| *ok)
            .count();
        assert!(ok_count >= 1, "at least one writer must succeed");
        let loaded = load_cached_code_object(key).expect("object intact after race");
        assert_eq!(loaded.len(), 512);
        assert!(
            loaded.iter().all(|&b| b == loaded[0]),
            "file content must be one complete payload, not interleaved"
        );
        let _ = fs::remove_dir_all(dir);
        unsafe { std::env::remove_var("GRIM_JIT_CACHE_DIR") };
    }

    #[test]
    fn test_prune_stale_tmp_files_removes_only_old_tmps() {
        let _env = env_lock();
        let dir = temp_cache_dir("prune");
        fs::create_dir_all(&dir).unwrap();
        let old_tmp = dir.join("somekey.99999.123.tmp");
        let fresh_hsaco = dir.join("keepkey.hsaco");
        fs::write(&old_tmp, b"junk").unwrap();
        fs::write(&fresh_hsaco, b"data").unwrap();
        prune_stale_tmp_files(0); // max_age 0 => any mtime is stale // max_age 0 => everything tmp is stale
        assert!(!old_tmp.exists(), "stale tmp pruned");
        assert!(fresh_hsaco.exists(), "non-tmp untouched");
        let _ = fs::remove_dir_all(dir);
        unsafe { std::env::remove_var("GRIM_JIT_CACHE_DIR") };
    }
}
