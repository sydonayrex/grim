//! Model catalog — per-model JSON sidecar written alongside every downloaded
//! model file. Allows `grim run <name>` and `GET /v1/models` to resolve a
//! friendly name to a file path without scanning for extensions.
//!
//! Sidecar path: `<models_dir>/<stem>.json`
//!
//! Contract:
//! - Written atomically (temp-file + rename) so a crash during download
//!   cannot leave a half-written catalog entry.
//! - Read-tolerant: missing fields deserialize to their Default values so
//!   older sidecars remain readable after format additions.

use std::path::{Path, PathBuf};

use crate::grim_models_dir;
use grim_tensor::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Metadata stored in the per-model JSON sidecar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// User-visible name (e.g. `"llama3:8b"`, `"mistral:7b-q4_k_m"`).
    pub name: String,
    /// Absolute path to the model file on disk.
    pub path: String,
    /// Model architecture reported by GGUF `general.architecture` (or `"unknown"`).
    #[serde(default)]
    pub arch: String,
    /// Human-readable parameter count (e.g. `"8B"`, `"70B"`).
    #[serde(default)]
    pub params: String,
    /// Quantization label (e.g. `"Q4_K_M"`, `"F16"`).
    #[serde(default)]
    pub quant: String,
    /// Context window length in tokens.
    #[serde(default)]
    pub context_length: u64,
    /// File size in bytes.
    #[serde(default)]
    pub size_bytes: u64,
    /// SHA-256 hex digest of the file at pull time.
    #[serde(default)]
    pub sha256: String,
    /// RFC-3339 timestamp of the pull.
    #[serde(default)]
    pub pulled_at: String,
    /// Registry that provided the file (`"ollama"`, `"huggingface"`, `"url"`).
    #[serde(default)]
    pub source: String,
}

impl ModelEntry {
    /// Derive the sidecar path for a given model file path.
    ///
    /// `<dir>/<stem>.json` — always lives next to the model file.
    pub fn sidecar_path_for(model_path: &Path) -> PathBuf {
        model_path.with_extension("json")
    }

    /// Write this entry to the canonical sidecar location atomically.
    ///
    /// Uses a `.tmp` suffix + rename to avoid partial writes.
    pub fn save(&self, model_path: &Path) -> Result<()> {
        let sidecar = Self::sidecar_path_for(model_path);
        let tmp = sidecar.with_extension("tmp");
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| Error::Backend(format!("catalog serialize failed: {e}")))?;
        std::fs::write(&tmp, json)
            .map_err(|e| Error::Backend(format!("catalog tmp write failed: {e}")))?;
        std::fs::rename(&tmp, &sidecar)
            .map_err(|e| Error::Backend(format!("catalog rename failed: {e}")))?;
        Ok(())
    }

    /// Load a sidecar from the given model file path. Returns `None` if the
    /// sidecar does not exist (e.g. model was placed manually without a pull).
    pub fn load_for(model_path: &Path) -> Option<Self> {
        let sidecar = Self::sidecar_path_for(model_path);
        let text = std::fs::read_to_string(sidecar).ok()?;
        serde_json::from_str(&text).ok()
    }
}

/// Resolve a model name or alias to a file path on disk.
///
/// Resolution order:
/// 1. Exact file path (absolute or relative) that exists as-is.
/// 2. Sidecar lookup in `grim_models_dir()` — walks all `.json` files and
///    matches `entry.name` exactly, then by stem prefix.
/// 3. File scan in `grim_models_dir()` — matches `<name>.gguf`, `<name>.grim`,
///    `<name_with_underscores>.gguf`, etc.
///
/// Returns `None` when no match is found. The caller should print a helpful
/// message directing the user to run `grim pull <name>`.
pub fn resolve_model_path(name: &str) -> Option<PathBuf> {
    // 1. Direct path.
    let direct = Path::new(name);
    if direct.exists() {
        return Some(direct.to_path_buf());
    }

    let models_dir = grim_models_dir();

    // 2. Sidecar lookup — accurate, includes arch/name metadata.
    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        let mut by_prefix: Option<PathBuf> = None;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Some(catalog) = ModelEntry::load_for(
                // Reconstruct the model path from the sidecar path.
                &path.with_extension("gguf"),
            )
            .or_else(|| ModelEntry::load_for(&path.with_extension("grim")))
            {
                if catalog.name == name {
                    let p = PathBuf::from(&catalog.path);
                    if p.exists() {
                        return Some(p);
                    }
                }
                // Prefix match (e.g. "llama3" matches "llama3:8b").
                if catalog.name.starts_with(name) && by_prefix.is_none() {
                    let p = PathBuf::from(&catalog.path);
                    if p.exists() {
                        by_prefix = Some(p);
                    }
                }
            }
        }
        if let Some(p) = by_prefix {
            return Some(p);
        }
    }

    // 3. File scan — extension-based fallback.
    let stem = name.replace(['/', ':'], "_");
    for ext in &["gguf", "grim"] {
        let candidate = models_dir.join(format!("{stem}.{ext}"));
        if candidate.exists() {
            return Some(candidate);
        }
        // Also try exact name without transformation.
        let candidate2 = models_dir.join(format!("{name}.{ext}"));
        if candidate2.exists() {
            return Some(candidate2);
        }
    }

    None
}

/// Resolve a model name or alias to a file path, preferring an existing
/// ROCm-optimized `.grim` conversion over a sibling `.gguf` when both are
/// present.
///
/// This is used by `grim run` so that once a model has been converted with
/// `grim oxidize convert --rocml-profile <target>`, the tuned artifact is
/// picked up automatically — the conversion step is opt-in, but once it
/// exists it should be used without the user having to remember to point at
/// the `.grim` file explicitly.
///
/// Resolution strategy mirrors [`resolve_model_path`]: direct path, then
/// sidecar lookup, then a filesystem scan — but at the filesystem-scan step
/// a `.grim` candidate takes precedence over a `.gguf` candidate for the
/// same stem.
pub fn resolve_model_preferring_grim(name: &str) -> Option<PathBuf> {
    // 1. Direct path.
    let direct = Path::new(name);
    if direct.exists() {
        // Prefer a `.grim` sibling if the user pointed at a `.gguf` directly.
        if let Some(grim_sibling) = grim_sibling_if_gguf(direct) {
            return Some(grim_sibling);
        }
        return Some(direct.to_path_buf());
    }

    let models_dir = grim_models_dir();

    // 2. Sidecar lookup — accurate, includes arch/name metadata.
    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        let mut by_prefix: Option<PathBuf> = None;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Some(catalog) = ModelEntry::load_for(&path.with_extension("gguf"))
                .or_else(|| ModelEntry::load_for(&path.with_extension("grim")))
            {
                if catalog.name == name {
                    let p = PathBuf::from(&catalog.path);
                    if p.exists() {
                        if let Some(grim_sibling) = grim_sibling_if_gguf(&p) {
                            return Some(grim_sibling);
                        }
                        return Some(p);
                    }
                }
                if catalog.name.starts_with(name) && by_prefix.is_none() {
                    let p = PathBuf::from(&catalog.path);
                    if p.exists() {
                        by_prefix = Some(p);
                    }
                }
            }
        }
        if let Some(p) = by_prefix {
            if let Some(grim_sibling) = grim_sibling_if_gguf(&p) {
                return Some(grim_sibling);
            }
            return Some(p);
        }
    }

    // 3. File scan — extension-based fallback, `.grim` wins over `.gguf`.
    let stem = name.replace(['/', ':'], "_");
    let gguf_candidate = models_dir.join(format!("{stem}.gguf"));
    let grim_candidate = models_dir.join(format!("{stem}.grim"));
    if grim_candidate.exists() {
        return Some(grim_candidate);
    }
    if gguf_candidate.exists() {
        return Some(gguf_candidate);
    }
    let gguf_candidate2 = models_dir.join(format!("{name}.gguf"));
    let grim_candidate2 = models_dir.join(format!("{name}.grim"));
    if grim_candidate2.exists() {
        return Some(grim_candidate2);
    }
    if gguf_candidate2.exists() {
        return Some(gguf_candidate2);
    }

    None
}

/// If `path` is a `.gguf` file with an existing `.grim` sibling, return the
/// `.grim` path; otherwise `None`.
fn grim_sibling_if_gguf(path: &Path) -> Option<PathBuf> {
    if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        let grim = path.with_extension("grim");
        if grim.exists() {
            return Some(grim);
        }
    }
    None
}

/// List all model entries in the models directory.
///
/// Combines sidecar metadata (when present) with a plain filesystem scan
/// for files that have no sidecar.
pub fn list_local_models() -> Vec<ModelEntry> {
    let models_dir = grim_models_dir();
    let mut out: Vec<ModelEntry> = Vec::new();
    let mut seen_paths: std::collections::HashSet<String> = std::collections::HashSet::new();

    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_string();

            if !matches!(ext.as_str(), "gguf" | "grim") {
                continue;
            }

            let path_str = path.display().to_string();
            if seen_paths.contains(&path_str) {
                continue;
            }
            seen_paths.insert(path_str.clone());

            // Prefer sidecar metadata; fall back to guessing from filename.
            if let Some(catalog) = ModelEntry::load_for(&path) {
                out.push(catalog);
            } else {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
                out.push(ModelEntry {
                    name: format!("{stem}:{ext}"),
                    path: path_str,
                    arch: String::new(),
                    params: String::new(),
                    quant: String::new(),
                    context_length: 0,
                    size_bytes,
                    sha256: String::new(),
                    pulled_at: String::new(),
                    source: String::new(),
                });
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate the process-global `GRIM_MODELS_DIR` env var
    // so concurrent test threads don't clobber each other's models directory.
    static MODELS_DIR_GUARD: Mutex<()> = Mutex::new(());

    /// WI-S6: when both a `.gguf` and a `.grim` sibling exist for a model,
    /// `resolve_model_preferring_grim` must return the `.grim` path so the
    /// ROCm-tuned conversion is used automatically once it exists.
    #[test]
    fn resolve_preferring_grim_chooses_grim_over_gguf() {
        let _guard = crate::paths::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("GRIM_MODELS_DIR").ok();
        let tmp = std::env::temp_dir().join(format!("grim_test_prefer_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        unsafe {
            std::env::set_var("GRIM_MODELS_DIR", &tmp);
        }

        let gguf = tmp.join("llama3.gguf");
        let grim = tmp.join("llama3.grim");
        std::fs::write(&gguf, b"gguf").unwrap();
        std::fs::write(&grim, b"grim").unwrap();

        let resolved = resolve_model_preferring_grim("llama3").unwrap();
        assert_eq!(resolved, grim, "expected .grim to be preferred over .gguf");

        // Cleanup.
        let _ = std::fs::remove_file(&gguf);
        let _ = std::fs::remove_file(&grim);
        let _ = std::fs::remove_dir(&tmp);
        unsafe {
            match prev {
                Some(v) => std::env::set_var("GRIM_MODELS_DIR", v),
                None => std::env::remove_var("GRIM_MODELS_DIR"),
            }
        }
    }

    /// WI-S6 regression: with only a `.gguf` present, resolution still finds it.
    #[test]
    fn resolve_preferring_grim_falls_back_to_gguf() {
        let _guard = crate::paths::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("GRIM_MODELS_DIR").ok();
        let tmp = std::env::temp_dir().join(format!("grim_test_fallback_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        unsafe {
            std::env::set_var("GRIM_MODELS_DIR", &tmp);
        }

        let gguf = tmp.join("mistral.gguf");
        std::fs::write(&gguf, b"gguf").unwrap();

        let resolved = resolve_model_preferring_grim("mistral").unwrap();
        assert_eq!(resolved, gguf);

        let _ = std::fs::remove_file(&gguf);
        let _ = std::fs::remove_dir(&tmp);
        unsafe {
            match prev {
                Some(v) => std::env::set_var("GRIM_MODELS_DIR", v),
                None => std::env::remove_var("GRIM_MODELS_DIR"),
            }
        }
    }
}
