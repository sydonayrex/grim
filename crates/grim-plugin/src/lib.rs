//! `grim-plugin` — third-party extension system for Grim.
//!
//! §6 of the architecture. Two loading strategies, chosen per plugin:
//!
//! - **Dylib** (`libloading`, future) — for performance-critical extensions:
//!   new kernels, new model architectures. Process-shared memory; runs at
//!   near-native speed but a crash takes the engine down. First-party and
//!   reviewed plugins only.
//!
//! - **WASM** (`wasmtime`, future) — for control-path extensions: samplers,
//!   grammars/constrained decoding, pre/post-processors, tokenizers.
//!   Sandboxed; fuel + memory-limited; cannot touch host memory or make
//!   syscalls outside a granted capability set.
//!
//! v0 stubs here define the `GrimPluginVTable` ABI surface, the
//! `PluginCapabilities` bitflag, and the manifest schema parse. Real
//! `libloading` and `wasmtime` link land when phase 9 ships.

use grim_core::error::Result;

/// Bitflags describing what a plugin provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginCapabilities(pub u32);

impl PluginCapabilities {
    pub const MODEL_ARCHITECTURE: Self = Self(1 << 0);
    pub const BACKEND: Self = Self(1 << 1);
    pub const SAMPLER: Self = Self(1 << 2);
    pub const TOKENIZER: Self = Self(1 << 3);
    pub const PRE_POST_PROCESSOR: Self = Self(1 << 4);

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

/// Stable, `#[repr(C)]`-compatible vtable surface for dylib plugins.
/// (Rust trait objects aren't ABI-stable across compiler versions, so the
/// FFI boundary uses a C-compatible vtable — same pattern as `abi_stable`
/// / `stabby`.)
#[derive(Debug)]
#[repr(C)]
pub struct GrimPluginVTable {
    pub abi_version: u32,
    pub name: extern "C" fn() -> *const std::os::raw::c_char,
    pub capabilities: extern "C" fn() -> PluginCapabilities,
    pub init: extern "C" fn(ctx: *mut std::os::raw::c_void) -> i32,
    pub model_factory: Option<extern "C" fn(cfg: *const std::os::raw::c_char) -> *mut std::os::raw::c_void>,
    pub sampler_factory: Option<extern "C" fn() -> *mut std::os::raw::c_void>,
    pub teardown: extern "C" fn(),
}

/// Detection of plugin loading strategy from the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginKind {
    Dylib,
    Wasm,
}

/// Parsed plugin manifest.
#[derive(Debug, Clone)]
pub struct PluginManifest {
    pub name: String,
    pub abi_version: u32,
    pub kind: PluginKind,
    pub capabilities: PluginCapabilities,
    pub entry: String,
    pub limits: Option<PluginLimits>,
}

#[derive(Debug, Clone)]
pub struct PluginLimits {
    pub fuel: Option<u64>,
    pub max_memory_mb: Option<u32>,
}

/// Parse a `plugin.grim.toml` manifest (very minimal subset — full impl
/// uses `toml` + `vinto` per WASM WIT workflows).
pub fn parse_manifest(toml_text: &str) -> Result<PluginManifest> {
    let value: toml::Value = toml_text
        .parse()
        .map_err(|e: toml::de::Error| grim_core::Error::Config(format!("manifest parse: {e}")))?;
    let tbl = value
        .as_table()
        .ok_or_else(|| grim_core::Error::Config("manifest must be a TOML table".into()))?;
    let plugin = tbl
        .get("plugin")
        .and_then(|v| v.as_table())
        .ok_or_else(|| grim_core::Error::Config("missing [plugin] section".into()))?;

    let name = plugin
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| grim_core::Error::Config("plugin.name".into()))?
        .to_string();
    let abi_version = plugin
        .get("abi_version")
        .and_then(|v| v.as_integer())
        .unwrap_or(1) as u32;
    let kind = match plugin.get("kind").and_then(|v| v.as_str()) {
        Some("dylib") => PluginKind::Dylib,
        Some("wasm") => PluginKind::Wasm,
        Some(other) => {
            return Err(grim_core::Error::Config(format!(
                "unknown plugin kind '{other}'"
            )));
        }
        None => PluginKind::Wasm,
    };
    let capabilities = PluginCapabilities(
        plugin
            .get("capabilities")
            .and_then(|v| v.as_array())
            .map(|arr| {
                let mut acc = 0u32;
                for entry in arr {
                    let tag = match entry.as_str() {
                        Some("model") => PluginCapabilities::MODEL_ARCHITECTURE,
                        Some("backend") => PluginCapabilities::BACKEND,
                        Some("sampler") => PluginCapabilities::SAMPLER,
                        Some("tokenizer") => PluginCapabilities::TOKENIZER,
                        Some("pre_post") => PluginCapabilities::PRE_POST_PROCESSOR,
                        _ => PluginCapabilities(0),
                    };
                    acc |= tag.0;
                }
                acc
            })
            .unwrap_or(0),
    );
    let entry = plugin
        .get("entry")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(PluginManifest {
        name,
        abi_version,
        kind,
        capabilities,
        entry,
        limits: None,
    })
}

/// Validate that the manifest's ABI version is compatible with this engine.
pub fn validate_abi(manifest: &PluginManifest, engine_abi: u32) -> Result<()> {
    if manifest.abi_version != engine_abi {
        return Err(grim_core::Error::Config(format!(
            "plugin '{}' ABI version {} does not match engine ABI {}",
            manifest.name, manifest.abi_version, engine_abi
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_manifest() {
        let toml = r#"
[plugin]
name = "json-grammar"
abi_version = 1
kind = "wasm"
capabilities = ["sampler"]
entry = "grammar_json.wasm"
"#;
        let m = parse_manifest(toml).unwrap();
        assert_eq!(m.name, "json-grammar");
        assert_eq!(m.abi_version, 1);
        assert_eq!(m.kind, PluginKind::Wasm);
        assert!(m.capabilities.contains(PluginCapabilities::SAMPLER));
    }

    #[test]
    fn abi_validation() {
        let m = PluginManifest {
            name: "t".into(),
            abi_version: 1,
            kind: PluginKind::Wasm,
            capabilities: PluginCapabilities::SAMPLER,
            entry: "t.wasm".into(),
            limits: None,
        };
        assert!(validate_abi(&m, 1).is_ok());
        assert!(validate_abi(&m, 2).is_err());
    }
}
