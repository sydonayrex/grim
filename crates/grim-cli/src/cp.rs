//! grim cp - Copy a model to a new name in the local cache.

use grim_core::catalog::{ModelEntry, list_local_models, resolve_model_preferring_grim};
use grim_core::error::{Error, Result};
use grim_core::grim_models_dir;
use std::fs;
use std::path::PathBuf;

/// Copy a model to a new name in the local cache.
pub async fn cmd_cp(src: &str, dst: &str) -> Result<()> {
    // Resolve source model
    let src_path = resolve_model_preferring_grim(src)
        .ok_or_else(|| Error::Config(format!("Source model '{}' not found", src)))?;

    // Check if destination already exists
    let models_dir = grim_models_dir();
    let dst_path_buf = PathBuf::from(dst);
    let dst_stem = dst_path_buf.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(dst);

    // Find existing extension from source
    let src_ext = src_path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("gguf");

    let dst_path = models_dir.join(format!("{}.{}", dst_stem, src_ext));
    let dst_json = models_dir.join(format!("{}.json", dst_stem));

    if dst_path.exists() || dst_json.exists() {
        return Err(Error::Config(format!(
            "Destination '{}' already exists. Use --force to overwrite",
            dst
        )));
    }

    // Copy model file
    fs::copy(&src_path, &dst_path)
        .map_err(|e| Error::Config(format!("Failed to copy model file: {e}")))?;

    // Copy or create sidecar
    let src_json = src_path.with_extension("json");
    if src_json.exists() {
        let mut entry: ModelEntry = serde_json::from_str(
            &fs::read_to_string(&src_json)
                .map_err(|e| Error::Config(format!("Failed to read source sidecar: {e}")))?
        ).map_err(|e| Error::Config(format!("Failed to parse source sidecar: {e}")))?;
        entry.name = dst.to_string();
        entry.path = dst_path.display().to_string();
        entry.save(&dst_path)?;
    } else {
        // Create minimal sidecar
        let size = fs::metadata(&dst_path).map(|m| m.len()).unwrap_or(0);
        let entry = ModelEntry {
            name: dst.to_string(),
            path: dst_path.display().to_string(),
            arch: String::new(),
            params: String::new(),
            quant: String::new(),
            context_length: 0,
            size_bytes: size,
            sha256: String::new(),
            pulled_at: chrono::Utc::now().to_rfc3339(),
            source: "local-copy".to_string(),
        };
        entry.save(&dst_path)?;
    }

    println!("Copied '{}' -> '{}'", src, dst);
    Ok(())
}