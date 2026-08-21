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
use grim_format::gguf::{GgufFile, GgufValue, read_gguf};
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
    /// Preferred arithmetic dtype tagged into the `.grim` metadata at train
    /// time (`"f32"` / `"bf16"` / `"fp16"`); empty when unknown.
    #[serde(default)]
    pub preferred_dtype: String,
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

impl ModelEntry {
    /// Best-effort GGUF header enrichment.
    ///
    /// Parses only the GGUF header (magic, version, metadata KV map) — never
    /// the multi-GB tensor data section — so it is cheap enough to run on the
    /// per-model pull path and during `GET /v1/models`. Returns the fields the
    /// catalog displays: architecture (`general.architecture`), a human-readable
    /// parameter count (`general.parameter_count`, e.g. `"7B"`), and the context
    /// window (`llama.context_length` / `<arch>.context_length`).
    ///
    /// A corrupted or non-GGUF file yields `None` and the caller keeps whatever
    /// it already had (filename-derived hint, empty strings); this never fails
    /// the surrounding download/catalog operation.
    pub fn enrich_from_gguf(model_path: &Path) -> Option<GgufEnrichment> {
        let mut file = std::fs::File::open(model_path).ok()?;
        let gguf: GgufFile = read_gguf(&mut file).ok()?;

        let arch = gguf
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let params = gguf
            .metadata
            .get("general.parameter_count")
            .and_then(|v| match v {
                GgufValue::Uint64(n) => Some(human_param_count(*n)),
                GgufValue::Int64(n) if *n > 0 => Some(human_param_count(*n as u64)),
                GgufValue::Float64(f) => Some(human_param_count(*f as u64)),
                GgufValue::String(s) => Some(s.clone()),
                _ => None,
            });

        let context_length = arch
            .as_ref()
            .and_then(|a| gguf.metadata.get(&format!("{a}.context_length")))
            .and_then(|v| v.as_u32().map(|n| n as u64))
            .or_else(|| {
                gguf.metadata
                    .get("llama.context_length")
                    .and_then(|v| v.as_u32().map(|n| n as u64))
            })
            .or_else(|| {
                gguf.metadata
                    .get("general.context_length")
                    .and_then(|v| v.as_u32().map(|n| n as u64))
            });

        Some(GgufEnrichment {
            arch: arch.unwrap_or_default(),
            params: params.unwrap_or_default(),
            context_length: context_length.unwrap_or(0),
        })
    }
}

/// Fill a `ModelEntry`'s `arch` / `params` / `context_length` from a GGUF
/// header when those fields are still empty. Header-only read, so it is safe to
/// call on the per-model pull path and the filesystem-scan fallback. Never
/// overwrites data already present (e.g. a sidecar's richer value).
pub fn apply_gguf_enrichment(entry: &mut ModelEntry, model_path: &Path) {
    if !entry.arch.is_empty() && !entry.params.is_empty() && entry.context_length != 0 {
        return;
    }
    if let Some(e) = ModelEntry::enrich_from_gguf(model_path) {
        if entry.arch.is_empty() {
            entry.arch = e.arch;
        }
        if entry.params.is_empty() {
            entry.params = e.params;
        }
        if entry.context_length == 0 {
            entry.context_length = e.context_length;
        }
    }
}

/// WI-3 serve-time self-heal: if `model_path`'s catalog sidecar still has
/// an empty `arch` or zero `context_length` (an older pull, or a manually-
/// placed file whose sidecar predates WI-3), reload it, fill only the empty
/// fields from the GGUF header, and re-save. Header-only read — never the
/// multi-GB tensor section. Failure is non-fatal (returns `()`); callers in
/// the serve path invoke this after the model is already loaded, so a
/// missing sidecar or unreadable header must never break serving.
pub fn self_heal_sidecar(model_path: &Path) {
    if let Some(mut entry) = ModelEntry::load_for(model_path) {
        if entry.arch.is_empty() || entry.context_length == 0 {
            apply_gguf_enrichment(&mut entry, model_path);
            let _ = entry.save(model_path);
        }
    }
}

/// Fill a `ModelEntry`'s `preferred_dtype` from a native `.grim` file's JSON
/// metadata layer (P1 §8 tag written at train time). Header + metadata read
/// only — payload bytes are never touched — so it is cheap on the catalog
/// scan path. Non-`.grim` files or missing tags leave the entry untouched.
pub fn apply_grim_tags(entry: &mut ModelEntry, model_path: &Path) {
    if !entry.preferred_dtype.is_empty()
        || model_path.extension().map(|e| e != "grim").unwrap_or(true)
    {
        return;
    }
    if let Ok(mut file) = std::fs::File::open(model_path) {
        if let Ok(grim) = grim_format::format::GrimFile::read(&mut file) {
            if let Some(dt) = grim.metadata.preferred_dtype {
                entry.preferred_dtype = dt;
            }
        }
    }
}

/// Fields derived from a GGUF header for catalog display.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GgufEnrichment {
    pub arch: String,
    pub params: String,
    pub context_length: u64,
}

/// Format a parameter count into a compact human label (e.g. 7_000_000_000 ->
/// `"7B"`). Counts are rounded to the nearest 0.1B for cleaner labels.
fn human_param_count(n: u64) -> String {
    const B: f64 = 1_000_000_000.0;
    const M: f64 = 1_000_000.0;
    if n >= 1_000_000_000 {
        let v = (n as f64 / B * 10.0).round() / 10.0;
        format!("{}B", trim_zero(v))
    } else if n >= 1_000_000 {
        let v = (n as f64 / M * 10.0).round() / 10.0;
        format!("{}M", trim_zero(v))
    } else if n == 0 {
        String::new()
    } else {
        n.to_string()
    }
}

/// Render a float without a trailing `.0` (so `7`, not `7.0`; `6.7` stays).
fn trim_zero(v: f64) -> String {
    let s = format!("{v}");
    match s.strip_suffix(".0") {
        Some(stripped) => stripped.to_string(),
        None => s,
    }
}

fn is_safe_model_path(path: &Path, models_dir: &Path) -> bool {
    let s = path.to_string_lossy();
    if s.contains("..") {
        return false;
    }
    if let (Ok(canon_path), Ok(canon_dir)) = (path.canonicalize(), models_dir.canonicalize()) {
        canon_path.starts_with(canon_dir)
    } else {
        !path.is_absolute()
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
    let models_dir = grim_models_dir();

    // 1. Direct path (guarded against traversal).
    let direct = Path::new(name);
    if direct.exists() && is_safe_model_path(direct, &models_dir) {
        return Some(direct.to_path_buf());
    }

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
    let models_dir = grim_models_dir();

    // 0. Strip explicit format suffixes (`:grim`, `:gguf`) so that
    //    `resolve_model_preferring_grim("sleipnir:grim")` and
    //    `resolve_model_preferring_grim("sleipnir:gguf")` resolve the bare
    //    stem and then prefer/select the requested format.
    let (stem, force_ext) = strip_format_suffix(name);

    // 1. Direct path (guarded against traversal).
    let direct = Path::new(stem);
    if direct.exists() && is_safe_model_path(direct, &models_dir) {
        // Prefer a `.grim` sibling if the user pointed at a `.gguf` directly.
        if let Some(grim_sibling) = grim_sibling_if_gguf(direct) {
            return Some(resolve_with_ext(&grim_sibling, force_ext).unwrap_or(grim_sibling));
        }
        return Some(resolve_with_ext(direct, force_ext).unwrap_or_else(|| direct.to_path_buf()));
    }

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
                // Match against the suffix-stripped stem so `sleipnir:grim`
                // and `sleipnir:gguf` find the same catalog entry as `sleipnir`.
                if catalog.name == stem {
                    let p = PathBuf::from(&catalog.path);
                    if p.exists() {
                        // Honour an explicit `:grim`/`:gguf` suffix — otherwise
                        // prefer a `.grim` sibling when the catalog points at `.gguf`.
                        if let Some(resolved) = resolve_with_ext(&p, force_ext) {
                            return Some(resolved);
                        }
                        return Some(p);
                    }
                }
                if catalog.name.starts_with(stem) && by_prefix.is_none() {
                    let p = PathBuf::from(&catalog.path);
                    if p.exists() {
                        by_prefix = Some(p);
                    }
                }
            }
        }
        if let Some(p) = by_prefix {
            if let Some(resolved) = resolve_with_ext(&p, force_ext) {
                return Some(resolved);
            }
            return Some(p);
        }
    }

    // 3. File scan — extension-based fallback, `.grim` wins over `.gguf`.
    //    `stem`/`force_ext` have already had any `:grim`/`:gguf` suffix stripped
    //    in step 0, so a bare lookup like `sleipnir:grim` resolves to
    //    `sleipnir.grim` rather than the mangled `sleipnir_grim.grim`.
    let file_stem = stem.replace(['/', ':'], "_");
    let gguf_candidate = models_dir.join(format!("{file_stem}.gguf"));
    let grim_candidate = models_dir.join(format!("{file_stem}.grim"));
    if grim_candidate.exists() {
        return Some(resolve_with_ext(&grim_candidate, force_ext).unwrap_or(grim_candidate));
    }
    if gguf_candidate.exists() {
        return Some(resolve_with_ext(&gguf_candidate, force_ext).unwrap_or(gguf_candidate));
    }
    let gguf_candidate2 = models_dir.join(format!("{stem}.gguf"));
    let grim_candidate2 = models_dir.join(format!("{stem}.grim"));
    if grim_candidate2.exists() {
        return Some(resolve_with_ext(&grim_candidate2, force_ext).unwrap_or(grim_candidate2));
    }
    if gguf_candidate2.exists() {
        return Some(resolve_with_ext(&gguf_candidate2, force_ext).unwrap_or(gguf_candidate2));
    }

    None
}

/// Strip a `:grim` or `:gguf` format suffix from a model name.
///
/// Returns `(stem, force_ext)` where `stem` is the name without the suffix and
/// `force_ext` is `Some("grim")` or `Some("gguf")` when a suffix was present,
/// or `None` when the caller left the format implicit.
fn strip_format_suffix(name: &str) -> (&str, Option<&str>) {
    if let Some(rest) = name.strip_suffix(":grim") {
        (rest, Some("grim"))
    } else if let Some(rest) = name.strip_suffix(":gguf") {
        (rest, Some("gguf"))
    } else {
        (name, None)
    }
}

/// When `force_ext` is `Some`, override the candidate's extension if a file
/// with the *other* extension happens to be the one on disk. This lets
/// `sleipnir:gguf` resolve the GGUF even when `.grim` also exists.
fn resolve_with_ext(candidate: &Path, force_ext: Option<&str>) -> Option<PathBuf> {
    let desired = match force_ext {
        Some("grim") => "grim",
        Some("gguf") => "gguf",
        _ => return Some(candidate.to_path_buf()),
    };
    let current_ext = candidate.extension().and_then(|e| e.to_str());
    if current_ext == Some(desired) {
        Some(candidate.to_path_buf())
    } else {
        // Swap the extension to the one the user explicitly requested.
        let swapped = candidate.with_extension(desired);
        if swapped.exists() {
            Some(swapped)
        } else {
            Some(candidate.to_path_buf())
        }
    }
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
                let mut entry = ModelEntry {
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
                    preferred_dtype: String::new(),
                };
                // WI-3: best-effort header-derived arch/params/context_length
                // for manually-placed files that have no pull sidecar.
                apply_gguf_enrichment(&mut entry, &path);
                // P1 §8: surface the train-time `preferred_dtype` tag for
                // native `.grim` artifacts.
                apply_grim_tags(&mut entry, &path);
                out.push(entry);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WI-S6: when both a `.gguf` and a `.grim` sibling exist for a model,
    /// `resolve_model_preferring_grim` must return the `.grim` path so the
    /// ROCm-tuned conversion is used automatically once it exists.
    #[test]
    fn resolve_preferring_grim_chooses_grim_over_gguf() {
        let _guard = crate::paths::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
        let _guard = crate::paths::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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

    // ---- WI-3: GGUF-header catalog enrichment --------------------------------

    /// Build a minimal GGUF v3 byte stream with metadata KV pairs, mirroring the
    /// on-disk layout (not the library writer) so the assertion is encoder-independent.
    fn build_gguf_with_metadata(
        tensors: &[(&str, &[u64], u32, u64)],
        metadata: &[(&str, u32, Vec<u8>)],
    ) -> Vec<u8> {
        use grim_format::gguf::GGUF_MAGIC;
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // GGUF_VERSION
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, tag, val) in metadata {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(val);
        }
        for (name, dims, dtype, offset) in tensors {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&dtype.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        buf
    }

    fn gguf_kv_string<'a>(key: &'a str, val: &'a str) -> (&'a str, u32, Vec<u8>) {
        let mut v = (val.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(val.as_bytes());
        (key, 8, v)
    }

    fn gguf_kv_u64(key: &str, val: u64) -> (&str, u32, Vec<u8>) {
        (key, 10, val.to_le_bytes().to_vec())
    }

    #[test]
    fn human_param_count_formats_bands() {
        assert_eq!(human_param_count(7_000_000_000), "7B");
        assert_eq!(human_param_count(6_700_000_000), "6.7B");
        assert_eq!(human_param_count(13_000_000_000), "13B");
        assert_eq!(human_param_count(70_000_000_000), "70B");
        assert_eq!(human_param_count(1_300_000_000), "1.3B");
        assert_eq!(human_param_count(350_000_000), "350M");
        assert_eq!(human_param_count(0), "");
        assert_eq!(human_param_count(42), "42");
    }

    #[test]
    fn enrich_from_gguf_reads_arch_params_context() {
        let bytes = build_gguf_with_metadata(
            &[("token_embd.weight", &[32000, 4096], 0, 0)],
            &[
                gguf_kv_string("general.architecture", "llama"),
                gguf_kv_u64("general.parameter_count", 7_000_000_000),
                gguf_kv_u64("llama.context_length", 4096),
            ],
        );
        let tmp = std::env::temp_dir().join(format!("grim_wi3_{}.gguf", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();

        let e = ModelEntry::enrich_from_gguf(&tmp).expect("enrichment must succeed");
        assert_eq!(e.arch, "llama");
        assert_eq!(e.params, "7B");
        assert_eq!(e.context_length, 4096);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn enrich_from_gguf_prefers_arch_scoped_context_key() {
        let bytes = build_gguf_with_metadata(
            &[("tok.weight", &[10], 0, 0)],
            &[
                gguf_kv_string("general.architecture", "mistral"),
                gguf_kv_u64("mistral.context_length", 32768),
                gguf_kv_u64("llama.context_length", 4096),
            ],
        );
        let tmp = std::env::temp_dir().join(format!("grim_wi3b_{}.gguf", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();

        let e = ModelEntry::enrich_from_gguf(&tmp).unwrap();
        assert_eq!(e.arch, "mistral");
        assert_eq!(e.context_length, 32768, "architecture-scoped key must win");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn enrich_from_gguf_corrupt_file_returns_none() {
        let tmp = std::env::temp_dir().join(format!("grim_wi3c_{}.gguf", std::process::id()));
        std::fs::write(&tmp, b"not a gguf").unwrap();
        assert!(ModelEntry::enrich_from_gguf(&tmp).is_none());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn apply_gguf_enrichment_fills_empty_fields_only() {
        let bytes = build_gguf_with_metadata(
            &[("w", &[4], 0, 0)],
            &[
                gguf_kv_string("general.architecture", "llama"),
                gguf_kv_u64("general.parameter_count", 3_000_000_000),
                gguf_kv_u64("llama.context_length", 8192),
            ],
        );
        let tmp = std::env::temp_dir().join(format!("grim_wi3d_{}.gguf", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();

        let mut entry = ModelEntry {
            name: "x".into(),
            path: tmp.to_string_lossy().into_owned(),
            arch: "preset".into(),
            params: String::new(),
            quant: "Q4_K_M".into(),
            context_length: 0,
            size_bytes: 0,
            sha256: String::new(),
            pulled_at: String::new(),
            source: String::new(),
            preferred_dtype: String::new(),
        };
        apply_gguf_enrichment(&mut entry, &tmp);
        assert_eq!(
            entry.arch, "preset",
            "existing arch must not be overwritten"
        );
        assert_eq!(entry.params, "3B", "empty params filled from header");
        assert_eq!(
            entry.context_length, 8192,
            "empty context filled from header"
        );

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn self_heal_sidecar_backfills_empty_metadata() {
        let bytes = build_gguf_with_metadata(
            &[("w", &[4], 0, 0)],
            &[
                gguf_kv_string("general.architecture", "llama"),
                gguf_kv_u64("general.parameter_count", 2_000_000_000),
                gguf_kv_u64("llama.context_length", 4096),
            ],
        );
        let tmp =
            std::env::temp_dir().join(format!("grim_wi3_selfheal_{}.gguf", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();

        // Plant a sidecar with empty arch / zero context_length (the pre-WI-3
        // state for an older pull).
        let entry = ModelEntry {
            name: "x".into(),
            path: tmp.to_string_lossy().into_owned(),
            arch: String::new(),
            params: String::new(),
            quant: String::new(),
            context_length: 0,
            size_bytes: 0,
            sha256: String::new(),
            pulled_at: String::new(),
            source: String::new(),
            preferred_dtype: String::new(),
        };
        entry.save(&tmp).unwrap();

        self_heal_sidecar(&tmp);

        let healed = ModelEntry::load_for(&tmp).expect("sidecar must reload");
        assert_eq!(healed.arch, "llama", "self-heal fills empty arch");
        assert_eq!(healed.params, "2B", "self-heal fills empty params");
        assert_eq!(healed.context_length, 4096, "self-heal fills empty context");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn self_heal_sidecar_is_noop_when_already_populated() {
        let bytes = build_gguf_with_metadata(
            &[("w", &[4], 0, 0)],
            &[
                gguf_kv_string("general.architecture", "llama"),
                gguf_kv_u64("general.parameter_count", 99),
                gguf_kv_u64("llama.context_length", 7),
            ],
        );
        let tmp =
            std::env::temp_dir().join(format!("grim_wi3_selfheal2_{}.gguf", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();

        // Sidecar already has correct metadata from a WI-3 pull.
        let entry = ModelEntry {
            name: "x".into(),
            path: tmp.to_string_lossy().into_owned(),
            arch: "preset".into(),
            params: "13B".into(),
            quant: String::new(),
            context_length: 8192,
            size_bytes: 0,
            sha256: String::new(),
            pulled_at: String::new(),
            source: String::new(),
            preferred_dtype: String::new(),
        };
        entry.save(&tmp).unwrap();

        self_heal_sidecar(&tmp);

        let healed = ModelEntry::load_for(&tmp).unwrap();
        assert_eq!(healed.arch, "preset", "already-populated arch untouched");
        assert_eq!(healed.params, "13B");
        assert_eq!(healed.context_length, 8192);

        let _ = std::fs::remove_file(&tmp);
    }
}
