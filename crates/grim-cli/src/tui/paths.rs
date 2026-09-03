//! XDG base dirs for grim state. No `dirs` crate — resolved manually,
//! same pattern as `permissions.rs`.

use std::path::PathBuf;

pub fn data_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share"))
        })?;
    Some(base.join("grim"))
}

pub fn config_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("grim"))
}

#[cfg(test)]
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Serializes tests that mutate process-global env vars. Env is shared across
/// the parallel test threads of this binary; every test that sets an XDG var
/// must hold this lock.
#[cfg(test)]
pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xdg_data_home_wins() {
        let _g = env_lock();
        unsafe { std::env::set_var("XDG_DATA_HOME", "/tmp/grim-test-data") };
        assert_eq!(data_dir(), Some(PathBuf::from("/tmp/grim-test-data/grim")));
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
    }

    #[test]
    fn xdg_config_home_wins() {
        let _g = env_lock();
        unsafe { std::env::set_var("XDG_CONFIG_HOME", "/tmp/grim-test-config") };
        assert_eq!(config_dir(), Some(PathBuf::from("/tmp/grim-test-config/grim")));
        unsafe { std::env::remove_var("XDG_CONFIG_HOME") };
    }
}
