//! `grim arch-plugin generate` — generate and install an architecture compatibility
//! plugin (.grimplugin) from a HuggingFace model repo.
//!
//! This is the CLI entry point for the `ArchCompatSpec` → `.grimplugin` workflow.
//! It fetches config.json via the HF Hub API (through `ArchCompatSpec::from_hf_model_id`),
//! validates required fields, and installs the plugin into `grim_plugins_dir()` where
//! `model_loader.rs`'s `resolve_arch_compat_spec` can discover it at model-load time.
//!
//! This is deliberately a top-level `Commands::ArchPlugin` variant, NOT folded into
//! `PluginCommands`, because the generated `.grimplugin` is model-loading metadata
//! consumed by `model_loader.rs`, not a WASM/dylib plugin consumed by
//! `PluginRegistry::scan_plugin_directory`.

use grim_core::error::{Error, Result};
use grim_plugin::ArchCompatSpec;
use std::fs;
use std::path::{Path, PathBuf};

/// Generate and install a .grimplugin from a HuggingFace model repo.
///
/// `model_id` is an `hf:org/repo` reference (e.g. `hf:Qwen/Qwen3.8-27B`).
/// Resolve the install path for a given output filename and plugins directory.
///
/// Returns `Err` if the path is absolute or escaping and does not resolve
/// inside `plugins_dir`, because such a path would not be auto-discoverable
/// by `resolve_arch_compat_spec` and we should not tell the user it is.
fn resolve_install_path(
    out_path: &Path,
    plugins_dir: &Path,
) -> std::result::Result<PathBuf, Error> {
    if out_path.is_absolute()
        || out_path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        // Absolute or escaping path — only allow if it's inside plugins_dir.
        // Use a string-prefix check on the display strings since canonicalization
        // may fail for paths that don't exist yet.
        let out_display = out_path.display().to_string();
        let plugins_display = plugins_dir.display().to_string();
        if out_display.starts_with(&plugins_display) || out_display == plugins_display {
            Ok(out_path.to_path_buf())
        } else {
            Err(Error::Config(format!(
                "Output path '{out_path:?}' is not inside plugins_dir '{plugins_display}' \
                     and therefore will not be auto-discovered by grim run/serve. \
                     Either use a bare filename (writes to '{plugins_display}/{{name}}.grimplugin'), \
                     or specify an absolute path inside '{plugins_display}'."
            )))
        }
    } else {
        // Bare filename — write into plugins_dir.
        Ok(plugins_dir.join(out_path))
    }
}

/// The command fetches config.json via the HF Hub API, validates required fields,
/// and installs the plugin into `grim_plugins_dir()`.
pub async fn cmd_arch_plugin_generate(model_id: &str, output: Option<String>) -> Result<()> {
    // Parse the hf:org/repo prefix.
    let org_repo = if model_id.starts_with("hf:") {
        model_id.strip_prefix("hf:").ok_or_else(|| {
            Error::Config(format!("model_id must start with 'hf:' (got '{model_id}')"))
        })?
    } else {
        // Accept bare org/repo as well for convenience.
        model_id
    };

    if org_repo.is_empty() || !org_repo.contains('/') {
        return Err(Error::Config(format!(
            "model_id must be 'hf:org/repo' or 'org/repo' (got '{model_id}')"
        )));
    }

    // Fetch and parse the config via the HF Hub API.
    let spec = ArchCompatSpec::from_hf_model_id(org_repo).await?;
    validate_spec(&spec)?;

    // Resolve the install path (pure function, testable in isolation).
    let plugins_dir = grim_core::paths::grim_plugins_dir();
    let out_filename = output.unwrap_or_else(|| format!("{}.grimplugin", spec.model_type));
    let out_path = Path::new(&out_filename);

    let install_path = resolve_install_path(out_path, &plugins_dir)?;

    // Ensure the plugins directory exists (needed both for the auto-discoverable
    // default path and for any explicit --output inside grim_plugins_dir()).
    let install_parent = install_path.parent().unwrap_or(&install_path);
    fs::create_dir_all(install_parent).map_err(|e| {
        Error::Config(format!(
            "Failed to create directory {:?}: {e}",
            install_parent
        ))
    })?;

    // Serialize and write the plugin.
    let json_output = spec.to_json()?;
    fs::write(&install_path, json_output)
        .map_err(|e| Error::Config(format!("Failed to write plugin to {:?}: {e}", install_path)))?;

    println!(
        "Successfully generated and installed architecture compatibility plugin: {} -> {}",
        out_filename,
        install_path.display()
    );
    println!(
        "  model_type: '{}' (base='{}', layers={}, hidden={})",
        spec.model_type, spec.base_architecture, spec.num_layers, spec.hidden_size
    );
    println!("  The next `grim run` or `grim serve` will discover this plugin automatically.");

    Ok(())
}

/// Validate that a spec has the required fields for installation.
///
/// `from_hf_model_id` does this internally, but we re-check here as a
/// defense-in-depth gate. This mirrors the validation that `compat.rs` had
/// before it was deleted.
fn validate_spec(spec: &ArchCompatSpec) -> Result<()> {
    if spec.model_type.is_empty() || spec.model_type == "custom" {
        return Err(Error::Config(
            "model_type is required but was empty or missing from config.json".into(),
        ));
    }
    if spec.num_layers == 0 {
        return Err(Error::Config("num_hidden_layers must be > 0".into()));
    }
    if spec.hidden_size == 0 {
        return Err(Error::Config("hidden_size must be > 0".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_spec_rejects_empty_model_type() {
        let spec = ArchCompatSpec {
            name: "test".into(),
            model_type: "".into(),
            base_architecture: "llama".into(),
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 16384,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            is_moe: false,
            is_ssm: false,
            is_multimodal: false,
            vision_spec: None,
            audio_spec: None,
            expert_count: None,
            expert_used_count: None,
            routed_scaling_factor: None,
            tensor_name_mapping: Default::default(),
        };
        assert!(validate_spec(&spec).is_err());
    }

    #[test]
    fn validate_spec_rejects_custom_model_type() {
        let spec = ArchCompatSpec {
            name: "test".into(),
            model_type: "custom".into(),
            base_architecture: "llama".into(),
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 16384,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            is_moe: false,
            is_ssm: false,
            is_multimodal: false,
            vision_spec: None,
            audio_spec: None,
            expert_count: None,
            expert_used_count: None,
            routed_scaling_factor: None,
            tensor_name_mapping: Default::default(),
        };
        assert!(validate_spec(&spec).is_err());
    }

    #[test]
    fn validate_spec_rejects_zero_layers() {
        let spec = ArchCompatSpec {
            name: "test".into(),
            model_type: "test".into(),
            base_architecture: "llama".into(),
            hidden_size: 4096,
            num_layers: 0,
            vocab_size: 32000,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 16384,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            is_moe: false,
            is_ssm: false,
            is_multimodal: false,
            vision_spec: None,
            audio_spec: None,
            expert_count: None,
            expert_used_count: None,
            routed_scaling_factor: None,
            tensor_name_mapping: Default::default(),
        };
        assert!(validate_spec(&spec).is_err());
    }

    #[test]
    fn validate_spec_rejects_zero_hidden_size() {
        let spec = ArchCompatSpec {
            name: "test".into(),
            model_type: "test".into(),
            base_architecture: "llama".into(),
            hidden_size: 0,
            num_layers: 32,
            vocab_size: 32000,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 16384,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            is_moe: false,
            is_ssm: false,
            is_multimodal: false,
            vision_spec: None,
            audio_spec: None,
            expert_count: None,
            expert_used_count: None,
            routed_scaling_factor: None,
            tensor_name_mapping: Default::default(),
        };
        assert!(validate_spec(&spec).is_err());
    }

    #[test]
    fn validate_spec_accepts_valid_config() {
        let spec = ArchCompatSpec {
            name: "test".into(),
            model_type: "llama".into(),
            base_architecture: "llama".into(),
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 16384,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 2048,
            is_moe: false,
            is_ssm: false,
            is_multimodal: false,
            vision_spec: None,
            audio_spec: None,
            expert_count: None,
            expert_used_count: None,
            routed_scaling_factor: None,
            tensor_name_mapping: Default::default(),
        };
        assert!(validate_spec(&spec).is_ok());
    }

    // ---------------------------------------------------------------------------
    // Tests for resolve_install_path
    // ---------------------------------------------------------------------------

    #[test]
    fn resolve_install_path_rejects_absolute_path_outside_plugins_dir() {
        let plugins_dir = std::path::PathBuf::from("/tmp/grim-plugins");
        let outside = std::path::PathBuf::from("/tmp/other-dir/plugin.grimplugin");
        let result = resolve_install_path(&outside, &plugins_dir);
        assert!(
            result.is_err(),
            "absolute path outside plugins_dir must be rejected"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("/tmp/other-dir"),
            "error should mention the outside path"
        );
        assert!(
            err.contains("/tmp/grim-plugins"),
            "error should mention the plugins_dir"
        );
    }

    #[test]
    fn resolve_install_path_rejects_relative_path_with_dotdot_outside_plugins_dir() {
        let plugins_dir = std::path::PathBuf::from("/tmp/grim-plugins");
        let outside = std::path::PathBuf::from("../other-dir/plugin.grimplugin");
        let result = resolve_install_path(&outside, &plugins_dir);
        assert!(
            result.is_err(),
            "relative path with .. outside plugins_dir must be rejected"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("..") || err.contains("other-dir"),
            "error should mention the escaping path"
        );
    }

    #[test]
    fn resolve_install_path_accepts_absolute_path_inside_plugins_dir() {
        let plugins_dir = std::path::PathBuf::from("/tmp/grim-plugins");
        let inside = std::path::PathBuf::from("/tmp/grim-plugins/subdir/plugin.grimplugin");
        let result = resolve_install_path(&inside, &plugins_dir);
        assert!(
            result.is_ok(),
            "absolute path inside plugins_dir should be accepted"
        );
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/tmp/grim-plugins/subdir/plugin.grimplugin")
        );
    }

    #[test]
    fn resolve_install_path_accepts_bare_filename() {
        let plugins_dir = std::path::PathBuf::from("/tmp/grim-plugins");
        let bare = std::path::PathBuf::from("plugin.grimplugin");
        let result = resolve_install_path(&bare, &plugins_dir);
        assert!(result.is_ok(), "bare filename should be accepted");
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/tmp/grim-plugins/plugin.grimplugin")
        );
    }

    #[test]
    fn resolve_install_path_accepts_absolute_path_equals_plugins_dir() {
        let plugins_dir = std::path::PathBuf::from("/tmp/grim-plugins");
        let exact = std::path::PathBuf::from("/tmp/grim-plugins");
        let result = resolve_install_path(&exact, &plugins_dir);
        assert!(
            result.is_ok(),
            "absolute path equal to plugins_dir should be accepted"
        );
    }
}
