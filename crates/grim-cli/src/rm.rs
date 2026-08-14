//! grim rm — Remove a model from the local cache.

use grim_core::catalog::resolve_model_preferring_grim;
use grim_core::error::{Error, Result};
use grim_core::grim_models_dir;
use std::fs;
use std::io::{self, Write};

/// Remove a model from the local cache.
///
/// # Safety
///
/// This function permanently deletes files from disk. By default it prompts
/// for confirmation unless `force` is true.
pub async fn cmd_rm(model: &str, force: bool) -> Result<()> {
    // Resolve model
    let model_path = resolve_model_preferring_grim(model)
        .ok_or_else(|| Error::Config(format!("Model '{}' not found", model)))?;

    let models_dir = grim_models_dir();
    let model_stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::Config("Invalid model path".to_string()))?;

    let model_ext = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("gguf");

    // Collect all files that would be removed for the confirmation prompt.
    let mut targets = Vec::new();
    if model_path.exists() {
        targets.push(model_path.display().to_string());
    }
    let sidecar = models_dir.join(format!("{}.json", model_stem));
    if sidecar.exists() {
        targets.push(sidecar.display().to_string());
    }
    if model_ext == "gguf" {
        let grim_sibling = model_path.with_extension("grim");
        if grim_sibling.exists() {
            targets.push(grim_sibling.display().to_string());
        }
    }
    let train_sidecar = models_dir.join(format!("{}.grim.train", model_stem));
    if train_sidecar.exists() {
        targets.push(train_sidecar.display().to_string());
    }

    if targets.is_empty() {
        println!("Model '{}' not found (no files to remove).", model);
        return Ok(());
    }

    // Confirm unless --force was passed.
    if !force {
        println!("The following files will be permanently deleted:");
        for t in &targets {
            println!("  {}", t);
        }
        print!("Proceed? [y/N] ");
        io::stdout()
            .flush()
            .map_err(|e| Error::Config(format!("Failed to flush stdout: {e}")))?;
        let mut buf = String::new();
        io::stdin()
            .read_line(&mut buf)
            .map_err(|e| Error::Config(format!("Failed to read input: {e}")))?;
        if !buf.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

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

    // Remove .grim sibling if present
    if model_ext == "gguf" {
        let grim_sibling = model_path.with_extension("grim");
        if grim_sibling.exists() {
            fs::remove_file(&grim_sibling)
                .map_err(|e| Error::Config(format!("Failed to remove .grim sibling: {e}")))?;
            println!("Removed .grim sibling: {}", grim_sibling.display());
        }
    }

    // Remove .grim.train sidecar if present
    let train_sidecar = models_dir.join(format!("{}.grim.train", model_stem));
    if train_sidecar.exists() {
        fs::remove_file(&train_sidecar)
            .map_err(|e| Error::Config(format!("Failed to remove .train sidecar: {e}")))?;
        println!("Removed training sidecar: {}", train_sidecar.display());
    }

    println!("Removed model '{}'", model);
    Ok(())
}
