//! `grim_paths` — canonical path resolution for grim's data directories.
//!
//! All grim crates (`grim-cli`, `grim-server`, installer) must agree on
//! where models live. This module is the single source of truth.
//!
//! Resolution order (first existing path wins):
//! 1. `$GRIM_MODELS_DIR` env var override.
//! 2. `/var/lib/grim/models` — system install (matching `dist/install.sh`).
//! 3. `$HOME/.grim/models` — user install / development.
//!
//! For the config and log directories, `grim_config_dir` and `grim_log_dir`
//! follow the same priority scheme.

use std::path::PathBuf;

/// Returns the canonical models directory.
///
/// This is the same directory that `dist/install.sh` creates and that
/// `GET /v1/models` and `grim pull` write to and read from.
pub fn grim_models_dir() -> PathBuf {
    // 1. Explicit override — useful for tests and custom layouts.
    if let Ok(dir) = std::env::var("GRIM_MODELS_DIR") {
        let p = PathBuf::from(&dir);
        if !dir.is_empty() {
            return p;
        }
    }

    // 2. System install path (created by dist/install.sh).
    let system = PathBuf::from("/var/lib/grim/models");
    if system.exists() {
        return system;
    }

    // 3. User home fallback.
    if let Some(home) = home_dir() {
        return home.join(".grim").join("models");
    }

    // Unreachable in practice — default to system path even if missing.
    PathBuf::from("/var/lib/grim/models")
}

/// Returns the canonical config directory.
pub fn grim_config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("GRIM_CONFIG_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    let system = PathBuf::from("/etc/grim");
    if system.exists() {
        return system;
    }
    home_dir()
        .map(|h| h.join(".grim"))
        .unwrap_or_else(|| PathBuf::from("/etc/grim"))
}

/// Returns the canonical log directory.
pub fn grim_log_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("GRIM_LOG_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    PathBuf::from("/var/log/grim")
}

/// Returns the canonical plugins directory.
pub fn grim_plugins_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("GRIM_PLUGINS_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    let system = PathBuf::from("/var/lib/grim/plugins");
    if system.exists() {
        return system;
    }
    home_dir()
        .map(|h| h.join(".grim").join("plugins"))
        .unwrap_or_else(|| PathBuf::from("/var/lib/grim/plugins"))
}

/// Portable home-directory probe.  
/// Returns `None` only when neither `$HOME` nor `$USERPROFILE` is set.
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_dir_env_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("GRIM_MODELS_DIR").ok();
        unsafe {
            std::env::set_var("GRIM_MODELS_DIR", "/tmp/grim_test_models");
        }
        let dir = grim_models_dir();
        assert_eq!(dir, PathBuf::from("/tmp/grim_test_models"));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("GRIM_MODELS_DIR", v),
                None => std::env::remove_var("GRIM_MODELS_DIR"),
            }
        }
    }

    #[test]
    fn models_dir_returns_pathbuf() {
        // Must always return something without panicking.
        let _ = grim_models_dir();
    }
}
