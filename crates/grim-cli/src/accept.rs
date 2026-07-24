//! `grim accept` — Validate and install a model architecture plugin into system plugins directory.

use grim_core::error::{Error, Result};
use grim_core::grim_plugins_dir;
use grim_plugin::ArchCompatSpec;
use std::fs;
use std::path::Path;

/// Validate a `.grimplugin` file and install it into the system plugin directory.
pub async fn cmd_accept(plugin_path: &str) -> Result<()> {
    let src_path = Path::new(plugin_path);
    if !src_path.exists() {
        return Err(Error::Config(format!(
            "Plugin file '{}' not found",
            plugin_path
        )));
    }

    let file_name = src_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Config("Invalid plugin filename".to_string()))?;

    // Read and validate plugin manifest or config JSON
    let content = fs::read_to_string(src_path)
        .map_err(|e| Error::Config(format!("Failed to read plugin file: {e}")))?;

    let spec = ArchCompatSpec::from_hf_config_json(&content)?;
    eprintln!(
        "[grim accept] Validated plugin spec for model_type='{}' (base='{}')",
        spec.model_type, spec.base_architecture
    );

    // Target installation directory
    let target_dir = grim_plugins_dir();
    fs::create_dir_all(&target_dir)
        .map_err(|e| Error::Config(format!("Failed to create plugins directory: {e}")))?;

    let dst_path = target_dir.join(file_name);
    fs::copy(src_path, &dst_path)
        .map_err(|e| Error::Config(format!("Failed to copy plugin file to {:?}: {e}", dst_path)))?;

    println!(
        "Successfully accepted plugin '{}' -> {}",
        spec.name,
        dst_path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmd_accept_validation() {
        let sample_json = r#"{
            "model_type": "ling",
            "hidden_size": 4096,
            "num_hidden_layers": 28
        }"#;

        let spec = ArchCompatSpec::from_hf_config_json(sample_json).unwrap();
        assert_eq!(spec.model_type, "ling");
    }
}
