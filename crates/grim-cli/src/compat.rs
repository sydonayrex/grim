//! `grim compat` — Generate a `.grimplugin` compatibility spec file from a HuggingFace `config.json`.

use grim_core::error::{Error, Result};
use grim_plugin::ArchCompatSpec;
use std::fs;
use std::path::Path;

/// Execute the `grim compat` command: ingest `config.json` and output a `.grimplugin` manifest.
pub async fn cmd_compat(config_path: &str, output_path: Option<String>) -> Result<()> {
    let path_obj = Path::new(config_path);
    if !path_obj.exists() {
        return Err(Error::Config(format!(
            "Input config file '{}' not found",
            config_path
        )));
    }

    let content = fs::read_to_string(path_obj)
        .map_err(|e| Error::Config(format!("Failed to read config file: {e}")))?;

    let spec = ArchCompatSpec::from_hf_config_json(&content)?;
    let json_output = spec.to_json()?;

    let out_filename = output_path.unwrap_or_else(|| format!("{}.grimplugin", spec.model_type));
    let out_path = Path::new(&out_filename);

    fs::write(out_path, json_output)
        .map_err(|e| Error::Config(format!("Failed to write compatibility plugin to {:?}: {e}", out_path)))?;

    println!(
        "Successfully created architecture compatibility plugin: {} (base='{}', layers={}, hidden={})",
        out_filename, spec.base_architecture, spec.num_layers, spec.hidden_size
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmd_compat_spec_generation() {
        let sample_json = r#"{
            "model_type": "ling",
            "hidden_size": 4096,
            "num_hidden_layers": 28,
            "num_attention_heads": 32,
            "num_key_value_heads": 8
        }"#;

        let spec = ArchCompatSpec::from_hf_config_json(sample_json).unwrap();
        assert_eq!(spec.model_type, "ling");
        assert_eq!(spec.num_kv_heads, 8);
    }
}
