//! Plugin management CLI commands.

use grim_plugin::{PluginKind, PluginRegistry, WasmPluginLoader, parse_manifest, validate_abi};
use grim_tensor::error::Result;
use sha2::{Digest, Sha256};
use std::path::Path;

fn resolve_entry(plugin_dir: &Path, entry: &str) -> Result<std::path::PathBuf> {
    let relative = Path::new(entry);
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(grim_tensor::Error::Backend(format!(
            "plugin entry escapes its directory: {entry}"
        )));
    }
    let path = plugin_dir.join(relative);
    if !path.is_file() {
        return Err(grim_tensor::Error::Backend(format!(
            "plugin entry is not a regular file: {}",
            path.display()
        )));
    }
    Ok(path)
}

fn verify_entry_digest(path: &Path, expected: Option<&str>) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(grim_tensor::Error::Backend(
            "plugin sha256 must be 64 hexadecimal characters".into(),
        ));
    }
    let bytes = std::fs::read(path)
        .map_err(|e| grim_tensor::Error::Backend(format!("failed to read plugin entry: {e}")))?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected.to_ascii_lowercase() {
        return Err(grim_tensor::Error::Backend(format!(
            "plugin entry checksum mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

/// Load plugins from a directory into the registry.
pub fn load_plugins(plugin_dir: &str, registry: &mut PluginRegistry) -> Result<usize> {
    let plugins_path = Path::new(plugin_dir);
    if !plugins_path.exists() {
        return Ok(0);
    }

    let mut count = 0;

    // Scan for plugin.grim.toml manifests
    for entry in std::fs::read_dir(plugins_path)? {
        let entry = entry?;
        let plugin_subdir = entry.path();

        if !plugin_subdir.is_dir() {
            continue;
        }

        let manifest_path = plugin_subdir.join("plugin.grim.toml");
        if !manifest_path.exists() {
            continue;
        }

        let manifest_text = std::fs::read_to_string(&manifest_path)
            .map_err(|e| grim_tensor::Error::Backend(format!("Failed to read manifest: {e}")))?;
        let manifest = parse_manifest(&manifest_text)?;
        validate_abi(&manifest, 1)
            .map_err(|e| grim_tensor::Error::Backend(format!("ABI validation failed: {e}")))?;

        // Load based on plugin kind
        match manifest.kind {
            PluginKind::Wasm => {
                let wasm_path = resolve_entry(&plugin_subdir, &manifest.entry).and_then(|p| {
                    verify_entry_digest(&p, manifest.sha256.as_deref())?;
                    Ok(p)
                })?;
                let wasm_bytes = std::fs::read(&wasm_path).map_err(|e| {
                    grim_tensor::Error::Backend(format!("Failed to read WASM: {e}"))
                })?;
                let limits = manifest.limits.clone().unwrap_or_default();
                let loader = WasmPluginLoader::new(&manifest.name, limits);

                match loader.create_sampler(&wasm_bytes) {
                    Ok(sampler) => {
                        registry.register_sampler(manifest.name.clone(), sampler);
                        let _ = registry.register_manifest(manifest);
                        count += 1;
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to load WASM plugin '{}': {}",
                            manifest.name, e
                        );
                    }
                }
            }
            PluginKind::Dylib => {
                // Dylib plugins require runtime support; register manifest for discovery.
                // Native plugins are trusted/in-process; verify their declared artifact before discovery.
                let path = resolve_entry(&plugin_subdir, &manifest.entry)?;
                verify_entry_digest(&path, manifest.sha256.as_deref())?;
                let _ = registry.register_manifest(manifest);
            }
        }
    }

    Ok(count)
}

#[allow(dead_code)]
pub fn list_plugins(registry: &PluginRegistry) -> Vec<String> {
    registry.list_samplers().into_iter().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn load_plugins_finds_no_plugins_in_empty_dir() {
        let dir = tempdir().unwrap();
        let mut registry = PluginRegistry::new();
        let count = load_plugins(dir.path().to_str().unwrap(), &mut registry).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn parse_and_validate_manifest_works() {
        let toml = r#"
[plugin]
name = "test-plugin"
abi_version = 1
kind = "wasm"
capabilities = ["sampler"]
entry = "test.wasm"
"#;
        let manifest = parse_manifest(toml).unwrap();
        assert_eq!(manifest.name, "test-plugin");
        assert!(validate_abi(&manifest, 1).is_ok());
    }

    #[test]
    fn plugin_entry_cannot_escape_directory() {
        let dir = tempdir().unwrap();
        let err = resolve_entry(dir.path(), "../outside.wasm").unwrap_err();
        assert!(err.to_string().contains("escapes"));
    }

    #[test]
    fn plugin_entry_digest_is_verified() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("plugin.wasm");
        std::fs::write(&path, b"plugin").unwrap();
        verify_entry_digest(&path, Some(&format!("{:x}", Sha256::digest(b"plugin")))).unwrap();
        assert!(verify_entry_digest(&path, Some(&"0".repeat(64))).is_err());
    }
}
