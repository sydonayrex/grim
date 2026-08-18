//! Discovery: scan local filesystem for `.gguf`, `.grim`, and training-dataset
//! files. Returns shaped structs that the React UI consumes.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DiscoveryError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// One model on disk that the UI can offer in a dropdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Filename only — used as a stable dropdown identifier.
    pub id: String,
    /// Display name (alias for id).
    #[serde(default)]
    pub name: String,
    /// Absolute path on disk.
    pub path: String,
    /// `"gguf"` or `"grim"`.
    pub format: String,
    /// True when the file claims a `.grim` extension AND the GGUF header parses.
    pub is_grim: bool,
    /// Size on disk in bytes.
    #[serde(default)]
    pub size_bytes: u64,
}

impl ModelEntry {
    pub fn new(id: &str, path: &str, format: &str, is_grim: bool) -> Self {
        let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        Self {
            id: id.to_string(),
            name: id.to_string(),
            path: path.to_string(),
            format: format.to_string(),
            is_grim,
            size_bytes,
        }
    }
}

/// One dataset file on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntry {
    /// Filename only — used as a stable dropdown identifier.
    pub id: String,
    /// Absolute path on disk.
    pub path: String,
    /// `"jsonl"` / `"parquet"` / `"json"`.
    pub format: String,
    /// Size in bytes (for VRAM + token budgeting).
    pub size_bytes: u64,
}

/// Minimal GGUF header sanity check: the 4-byte `GGUF` magic at bytes 0-3
/// plus a little-endian u32 version >= 1 at bytes 4-7. Reads only the first
/// 8 bytes — never the whole file — so a renamed/truncated file is caught
/// cheaply without loading the model.
fn has_gguf_magic(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() || &magic != b"GGUF" {
        return false;
    }
    let mut version = [0u8; 4];
    f.read_exact(&mut version).is_ok() && u32::from_le_bytes(version) >= 1
}

fn classify_convertible_format(filename: &str) -> Option<&'static str> {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".gguf") {
        Some("gguf")
    } else if lower.ends_with(".safetensors") {
        Some("safetensors")
    } else if lower.ends_with(".bin") {
        Some("bin")
    } else if lower.ends_with(".fp16") {
        Some("fp16")
    } else if lower.ends_with(".fp8") {
        Some("fp8")
    } else if lower.ends_with(".fp4") {
        Some("fp4")
    } else if lower.ends_with(".mxfp4") {
        Some("mxfp4")
    } else if lower.ends_with(".nvfp4") {
        Some("nvfp4")
    } else if lower.contains("bitsandbytes") || lower.ends_with(".bnb") {
        Some("bitsandbytes")
    } else if lower.ends_with(".pt") || lower.ends_with(".pth") {
        Some("pytorch")
    } else if lower.ends_with(".onnx") {
        Some("onnx")
    } else {
        None
    }
}

fn classify_model_format(filename: &str) -> Option<&'static str> {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".grim") {
        Some("grim")
    } else if lower.ends_with(".gguf") {
        Some("gguf")
    } else if lower.ends_with(".safetensors") {
        Some("safetensors")
    } else {
        None
    }
}

fn classify_dataset_format(filename: &str) -> Option<&'static str> {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".jsonl") {
        Some("jsonl")
    } else if lower.ends_with(".parquet") {
        Some("parquet")
    } else if lower.ends_with(".json") {
        Some("json")
    } else {
        None
    }
}

fn scan_dir_recursive(
    dir: &Path,
    out: &mut Vec<ModelEntry>,
    is_convertible_only: bool,
    seen: &mut HashSet<PathBuf>,
) {
    scan_dir_recursive_inner(dir, out, is_convertible_only, &mut HashSet::new(), seen)
}

fn scan_dir_recursive_inner(
    dir: &Path,
    out: &mut Vec<ModelEntry>,
    is_convertible_only: bool,
    visited: &mut HashSet<PathBuf>,
    seen: &mut HashSet<PathBuf>,
) {
    let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    if !visited.insert(canonical) {
        return; // symlink cycle detected
    }
    if !dir.exists() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir_recursive_inner(&path, out, is_convertible_only, visited, seen);
        } else if path.is_file() {
            let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if is_convertible_only {
                if let Some(fmt) = classify_convertible_format(filename) {
                    let path_str = path.to_string_lossy().to_string();
                    // P2-17b: O(1) dedup via a shared HashSet — threaded from
                    // the discover_* entry points so it also covers the
                    // multiple roots scanned there — instead of the previous
                    // O(n²) `out.iter().any(...)` scan per file.
                    if seen.insert(path.clone()) {
                        out.push(ModelEntry::new(filename, &path_str, fmt, false));
                    }
                }
            } else {
                if let Some(format) = classify_model_format(filename) {
                    let path_str = path.to_string_lossy().to_string();
                    if seen.insert(path.clone()) {
                        // P2-17a: `.grim` files are GGUF under the hood — a
                        // renamed/truncated file is caught by the cheap 8-byte
                        // header check (magic `GGUF` + u32 version) instead of
                        // trusting the extension alone.
                        let is_grim = format == "grim" && has_gguf_magic(&path);
                        out.push(ModelEntry::new(filename, &path_str, format, is_grim));
                    }
                }
            }
        }
    }
}

fn is_default_dir(dir: &Path) -> bool {
    let local = Path::new("./models");
    if dir == local {
        return true;
    }
    if let Some(home) = std::env::var_os("HOME") {
        let grim_home = PathBuf::from(home).join(".grim").join("models");
        if dir == grim_home {
            return true;
        }
    }
    false
}

/// Scan `dir` for all available model files (.grim, .gguf, .safetensors).
pub fn discover_models(dir: &Path) -> Result<Vec<ModelEntry>, DiscoveryError> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    scan_dir_recursive(dir, &mut out, false, &mut seen);

    if is_default_dir(dir) {
        if let Some(home) = std::env::var_os("HOME") {
            let grim_home = PathBuf::from(home).join(".grim").join("models");
            if grim_home != dir && grim_home.exists() {
                scan_dir_recursive(&grim_home, &mut out, false, &mut seen);
            }
        }
        let local_models = PathBuf::from("./models");
        if local_models != dir && local_models.exists() {
            scan_dir_recursive(&local_models, &mut out, false, &mut seen);
        }
    }

    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Scan `dir` for raw convertible source models (.gguf, .safetensors, .bin, .fp16, .fp8, .fp4, .mxfp4, .nvfp4, bitsandbytes).
pub fn discover_convertible_models(dir: &Path) -> Result<Vec<ModelEntry>, DiscoveryError> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    scan_dir_recursive(dir, &mut out, true, &mut seen);

    if is_default_dir(dir) {
        if let Some(home) = std::env::var_os("HOME") {
            let grim_home = PathBuf::from(home).join(".grim").join("models");
            if grim_home != dir && grim_home.exists() {
                scan_dir_recursive(&grim_home, &mut out, true, &mut seen);
            }
        }
        let local_models = PathBuf::from("./models");
        if local_models != dir && local_models.exists() {
            scan_dir_recursive(&local_models, &mut out, true, &mut seen);
        }
    }

    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Scan `dir` for `.jsonl` / `.parquet` / `.json` files.
pub fn discover_datasets(dir: &Path) -> Result<Vec<DatasetEntry>, DiscoveryError> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(format) = classify_dataset_format(filename) else {
            continue;
        };
        let meta = entry.metadata()?;
        out.push(DatasetEntry {
            id: filename.to_string(),
            path: path.to_string_lossy().to_string(),
            format: format.to_string(),
            size_bytes: meta.len(),
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Resolve a default model search path. Used when the UI does not pass one.
/// Order of precedence: `GRIM_MODELS_DIR` env var → `~/.grim/models` → `./models`.
pub fn default_models_dir() -> PathBuf {
    if let Ok(p) = std::env::var("GRIM_MODELS_DIR") {
        return PathBuf::from(p);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join(".grim").join("models");
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("./models")
}

/// Resolve a default dataset search path.
pub fn default_datasets_dir() -> PathBuf {
    if let Ok(p) = std::env::var("GRIM_DATASETS_DIR") {
        return PathBuf::from(p);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join(".grim").join("datasets");
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("./datasets")
}
