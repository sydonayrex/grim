//! grim rm - Remove a model from the local cache.

use grim_core::catalog::resolve_model_preferring_grim;
use grim_core::error::{Error, Result};
use grim_core::grim_models_dir;
use std::fs;

/// Remove a model from the local cache.
pub async fn cmd_rm(model: &str) -> Result<()> {
    // Resolve the model
    let model_path = resolve_model_preferring_grim(model)
        .ok_or_else(|| Error::Config(format!("Model '{}' not found", model)))?;

    let models_dir = grim_models_dir();
    let model_stem = model_path.file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::Config("Invalid model path".to_string()))?;

    let model_ext = model_path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("gguf");

    // Remove model file
    if model_path.exists() {
        fs::remove_file(&model_path)
            .map_err(|e| Error::Config(format!("Failed to remove model file: {e}")))?;
        println!("Removed: {}", model_path.display());
    }

    // Remove sidecar
    let sidecar = models_dir.join(format!("{}.json", model_stem));
    if sidecar.exists() {
        fs::remove_file(&sidecar)
            .map_err(|e| Error::Config(format!("Failed to remove sidecar: {e}")))?;
        println!("Removed sidecar: {}", sidecar.display());
    }

    // Remove .grim sibling if .gguf was removed
    if model_ext == "gguf" {
        let grim_sibling = model_path.with_extension("grim");
        if grim_sibling.exists() {
            fs::remove_file(&grim_sibling)
                .map_err(|e| Error::Config(format!("Failed to remove .grim sibling: {e}")))?;
            println!("Removed .grim sibling: {}", grim_sibling.display());
        }
    }

    // Remove .grim.train sidecar if exists
    let train_sidecar = models_dir.join(format!("{}.grim.train", model_stem));
    if train_sidecar.exists() {
        fs::remove_file(&train_sidecar)
            .map_err(|e| Error::Config(format!("Failed to remove .train sidecar: {e}")))?;
        println!("Removed training sidecar: {}", train_sidecar.display());
    }

    println!("Removed model '{}'", model);
    Ok(())
}