//! Plugin management CLI commands.

use grim_tensor::error::Result;
use grim_plugin::{parse_manifest, validate_abi, PluginRegistry, PluginKind, WasmPluginLoader};
use std::path::Path;

/// Load plugins from a directory and populate the registry.
pub fn load_plugins(plugin_dir: &str, registry: &mut PluginRegistry) -> Result<usize> {
    let plugins_path = Path::new(plugin_dir);
    if !plugins_path.exists() {
        return Ok(0);
    }

    let mut count = 0;

    // Scan for plugin.grim.toml files
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
        validate_abi(&manifest, 1).map_err(|e| grim_tensor::Error::Backend(format!("ABI validation failed: {e}")))?;

        // Load based on plugin kind
        match manifest.kind {
            PluginKind::Wasm => {
                let wasm_path = plugin_subdir.join(&manifest.entry);
                if wasm_path.exists() {
                    let wasm_bytes = std::fs::read(&wasm_path)
                        .map_err(|e| grim_tensor::Error::Backend(format!("Failed to read WASM: {e}")))?;
                    let limits = manifest.limits.clone().unwrap_or_default();
                    let loader = WasmPluginLoader::new(&manifest.name, limits);
                    
                    match loader.create_sampler(&wasm_bytes) {
                        Ok(sampler) => {
                            registry.register_sampler(manifest.name.clone(), sampler);
                            let _ = registry.register_manifest(manifest);
                            count += 1;
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to load WASM plugin '{}': {}", manifest.name, e);
                        }
                    }
                }
            }
            PluginKind::Dylib => {
                // Dylib plugins would be loaded differently - require runtime support
                // For now, just register the manifest for discovery
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
}