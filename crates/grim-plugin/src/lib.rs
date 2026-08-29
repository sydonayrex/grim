//! `grim-plugin` — third-party extension system for Grim.
//!
//! §6 of the Grim architecture. Two loading strategies, chosen per plugin:
//!
//! - **Dylib** (`libloading`, §6.1) — for performance-critical extensions:
//!   new kernels, new model architectures. Process-shared memory; runs at
//!   near-native speed but a crash takes the engine down. First-party and
//!   reviewed plugins only.
//!
//! - **WASM** (`wasmtime`, §6.1) — for control-path extensions: samplers,
//!   grammars/constrained decoding, pre/post-processors, tokenizers.
//!   Sandboxed; fuel + memory-limited; cannot touch host memory or make
//!   syscalls outside a granted capability set.

pub mod arch_compat;
pub mod dylib_loader;
pub mod wasm_loader;

pub use arch_compat::ArchCompatSpec;
pub use dylib_loader::DylibPluginLoader;
pub use wasm_loader::WasmPluginLoader;

use grim_tensor::error::Result;
use std::collections::HashMap;
use std::sync::Arc;

// Re-export Sampler trait for plugin integration
pub use grim_core::sampler::Sampler;

/// Bitflags describing what a plugin provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
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

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl From<PluginCapabilities> for u32 {
    fn from(val: PluginCapabilities) -> Self {
        val.0
    }
}

/// Stable, `#[repr(C)]`-compatible vtable surface for dylib plugins.
/// (Rust trait objects aren't ABI-stable across compiler versions, so the
/// FFI boundary uses a C-compatible vtable — same pattern as `abi_stable`
/// / `stabby`.)
///
/// `sampler_sample` mirrors the WASM `sample` export signature
/// `(logits_ptr, logits_len, history_ptr, history_len) -> token_id`, keeping
/// the dylib and WASM sampler paths symmetric. `Option<...>` so a plugin that
/// doesn't declare the `SAMPLER` capability may leave it `None`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GrimPluginVTable {
    pub abi_version: u32,
    pub name: extern "C" fn() -> *const std::os::raw::c_char,
    pub capabilities: extern "C" fn() -> PluginCapabilities,
    pub init: extern "C" fn(ctx: *mut std::os::raw::c_void) -> i32,
    pub model_factory:
        Option<extern "C" fn(cfg: *const std::os::raw::c_char) -> *mut std::os::raw::c_void>,
    pub sampler_factory: Option<extern "C" fn() -> *mut std::os::raw::c_void>,
    /// Invoke sampling on a handle returned by `sampler_factory`. The handle
    /// is opaque host-owned state; the plugin reads `logits` (f32 slice of
    /// length `logits_len`) and `history` (u32 slice of length `history_len`)
    /// and returns a chosen token id.
    pub sampler_sample: Option<
        extern "C" fn(
            handle: *mut std::os::raw::c_void,
            logits_ptr: *const f32,
            logits_len: u32,
            history_ptr: *const u32,
            history_len: u32,
        ) -> i32,
    >,
    pub teardown: extern "C" fn(),
}
/// Detection of plugin loading strategy from the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PluginKind {
    Dylib,
    #[default]
    Wasm,
}

/// Capability grants for a plugin (§6.4, deny-by-default: all fields off).
///
/// Two manifest forms feed this struct:
/// - legacy inline: `[plugin.grants] network = true, filesystem = ["models/"]`
/// - scoped: `[grants] filesystem = true` + `[scopes] allowed_dirs = ["models/"]`
///
/// `filesystem` holds the effective scope — the exact directories that would
/// be WASI-preopened for the plugin. It is only non-empty when a grant
/// explicitly names its scopes.
#[derive(Debug, Clone, Default)]
pub struct PluginGrants {
    pub network: bool,
    pub filesystem: Vec<String>,
    pub request_metadata: bool,
}

#[derive(Debug, Clone, Default)]
pub struct PluginReload {
    pub hot_reload: bool,
}

/// Parsed plugin manifest.
#[derive(Debug, Clone)]
pub struct PluginManifest {
    pub name: String,
    pub abi_version: u32,
    pub kind: PluginKind,
    pub capabilities: PluginCapabilities,
    pub entry: String,
    /// Optional SHA-256 digest of the entry file, written as lowercase hex.
    pub sha256: Option<String>,
    pub limits: Option<PluginLimits>,
    pub stage: Option<String>,
    pub priority: Option<i32>,
    pub grants: PluginGrants,
    pub reload: PluginReload,
}

/// Execution and memory limits for WASM plugin sandboxing.
///
/// These limits are enforced by the WASM loader (`WasmPluginLoader`) when
/// present. Both fields default to `Some` with conservative values when using
/// `PluginLimits::default()`:
/// - `fuel_per_invocation`: defaults to `50_000` fuel units per invocation.
///   Controls how much "work" a plugin can do before being paused. Higher values
///   allow more complex sampling logic but increase the risk of a runaway plugin.
/// - `max_memory_mb`: defaults to `64` MB. Caps the total linear memory a plugin
///   can allocate. Prevents a malicious or buggy plugin from exhausting host RAM.
///
/// # Opt-out behavior
///
/// Both fields are `Option` types. Setting either field to `None` in a manifest
/// **disables that limit entirely**. This is an explicit opt-out — a manifest
/// that omits `limits` or sets `fuel_per_invocation = null` / `max_memory_mb = null`
/// will run without that restriction. Operators should ensure untrusted/third-party
/// plugins always carry explicit limits or be loaded in a context where the
/// `require_pinned_hash` / capability-grant enforcement provides sufficient
/// bounding.
///
/// # Trust model
///
/// For first-party and reviewed plugins, the defaults are typically sufficient.
/// For third-party or untrusted plugins, consider:
/// - Setting explicit lower limits in the manifest.
/// - Using the WASM sandbox exclusively (not dylib loading).
/// - Restricting capability grants (`network`, `filesystem`, `request_metadata`)
///   to the minimum required.
#[derive(Debug, Clone)]
pub struct PluginLimits {
    pub fuel_per_invocation: Option<u64>,
    pub max_memory_mb: Option<u32>,
}

impl Default for PluginLimits {
    fn default() -> Self {
        Self {
            fuel_per_invocation: Some(50_000),
            max_memory_mb: Some(64),
        }
    }
}

/// Parse a `plugin.grim.toml` manifest.
pub fn parse_manifest(toml_text: &str) -> Result<PluginManifest> {
    let value: toml::Value = toml_text.parse().map_err(|e: toml::de::Error| {
        grim_tensor::Error::Backend(format!("manifest parse: {e}"))
    })?;
    let tbl = value
        .as_table()
        .ok_or_else(|| grim_tensor::Error::Backend("manifest must be a TOML table".into()))?;
    let plugin = tbl
        .get("plugin")
        .and_then(|v| v.as_table())
        .ok_or_else(|| grim_tensor::Error::Backend("missing [plugin] section".into()))?;

    let name = plugin
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| grim_tensor::Error::Backend("plugin.name".into()))?
        .to_string();
    let abi_version = plugin
        .get("abi_version")
        .and_then(|v| v.as_integer())
        .unwrap_or(1) as u32;
    let kind = match plugin.get("kind").and_then(|v| v.as_str()) {
        Some("dylib") => PluginKind::Dylib,
        Some("wasm") => PluginKind::Wasm,
        Some(other) => {
            return Err(grim_tensor::Error::Backend(format!(
                "unknown plugin kind '{other}'"
            )));
        }
        None => PluginKind::Wasm,
    };
    let capabilities = match plugin.get("capabilities").and_then(|v| v.as_array()) {
        Some(arr) => {
            let mut acc = 0u32;
            for entry in arr {
                let tag = match entry.as_str() {
                    Some("model") => PluginCapabilities::MODEL_ARCHITECTURE,
                    Some("backend") => PluginCapabilities::BACKEND,
                    Some("sampler") => PluginCapabilities::SAMPLER,
                    Some("tokenizer") => PluginCapabilities::TOKENIZER,
                    Some("pre_post") => PluginCapabilities::PRE_POST_PROCESSOR,
                    Some(other) => {
                        return Err(grim_tensor::Error::Backend(format!(
                            "unknown plugin capability '{other}'"
                        )));
                    }
                    None => {
                        return Err(grim_tensor::Error::Backend(
                            "plugin capability entries must be strings".into(),
                        ));
                    }
                };
                acc |= tag.0;
            }
            PluginCapabilities(acc)
        }
        None => PluginCapabilities(0),
    };
    let entry = plugin
        .get("entry")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let sha256 = plugin
        .get("sha256")
        .and_then(|v| v.as_str())
        .map(str::to_ascii_lowercase);

    let stage = plugin
        .get("stage")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let priority = plugin
        .get("priority")
        .and_then(|v| v.as_integer())
        .map(|v| v as i32);

    // Parse limits if present
    let limits = plugin.get("limits").and_then(|limits_tbl| {
        let lt = limits_tbl.as_table()?;
        Some(PluginLimits {
            fuel_per_invocation: lt
                .get("fuel_per_invocation")
                .and_then(|v| v.as_integer())
                .map(|v| v as u64)
                .or_else(|| {
                    lt.get("fuel")
                        .and_then(|v| v.as_integer())
                        .map(|v| v as u64)
                }),
            max_memory_mb: lt
                .get("max_memory_mb")
                .and_then(|v| v.as_integer())
                .map(|v| v as u32),
        })
    });

    // Parse capabilities grants if present
    let mut grants = PluginGrants::default();
    if let Some(plugin_tbl) = tbl.get("plugin").and_then(|p| p.as_table()) {
        if let Some(grants_val) = plugin_tbl.get("grants") {
            if let Some(grants_tbl) = grants_val.as_table() {
                grants.network = grants_tbl
                    .get("network")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if let Some(fs_arr) = grants_tbl.get("filesystem").and_then(|v| v.as_array()) {
                    grants.filesystem = fs_arr
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                }
                grants.request_metadata = grants_tbl
                    .get("request_metadata")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
            }
        }
    }

    // Top-level [grants] / [scopes] form (§6.4). Grants are additive with the
    // legacy [plugin.grants] table above. A `filesystem = true` grant must
    // name its scopes: an unrestricted filesystem grant is rejected here, at
    // plugin-load time, rather than silently degrading to a trap later.
    if let Some(grants_tbl) = tbl.get("grants").and_then(|v| v.as_table()) {
        if grants_tbl
            .get("network")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            grants.network = true;
        }
        if grants_tbl
            .get("filesystem")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let allowed_dirs = tbl
                .get("scopes")
                .and_then(|v| v.as_table())
                .and_then(|s| s.get("allowed_dirs"))
                .and_then(|v| v.as_array())
                .map(|arr| -> Result<Vec<String>> {
                    arr.iter()
                        .map(|v| {
                            v.as_str().map(|s| s.to_string()).ok_or_else(|| {
                                grim_tensor::Error::Backend(
                                    "scopes.allowed_dirs entries must be strings".into(),
                                )
                            })
                        })
                        .collect()
                })
                .transpose()?
                .unwrap_or_default();
            if allowed_dirs.is_empty() {
                return Err(grim_tensor::Error::Backend(format!(
                    "plugin '{name}': filesystem grant requires [scopes].allowed_dirs in plugin.grim.toml"
                )));
            }
            grants.filesystem.extend(allowed_dirs);
        }
    }

    // Parse hot reload
    let mut reload = PluginReload::default();
    if let Some(reload_val) = tbl.get("plugin").and_then(|p| p.get("reload")) {
        if let Some(reload_tbl) = reload_val.as_table() {
            let hot = reload_tbl
                .get("hot_reload")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if kind == PluginKind::Wasm {
                reload.hot_reload = hot;
            } else {
                // Explicitly disabled by default for dylibs since they can't be sandboxed (§6.4)
                reload.hot_reload = false;
            }
        }
    }

    Ok(PluginManifest {
        name,
        abi_version,
        kind,
        capabilities,
        entry,
        sha256,
        limits,
        stage,
        priority,
        grants,
        reload,
    })
}

/// ABI version this engine's dylib loader speaks. A plugin manifest must
/// match exactly to load through the dylib path.
pub const ENGINE_ABI_VERSION: u32 = 1;

/// Validate that the manifest's ABI version is compatible with this engine.
pub fn validate_abi(manifest: &PluginManifest, engine_abi: u32) -> Result<()> {
    if manifest.abi_version != engine_abi {
        return Err(grim_tensor::Error::Backend(format!(
            "plugin '{}' ABI version {} does not match engine ABI {}",
            manifest.name, manifest.abi_version, engine_abi
        )));
    }
    Ok(())
}

/// Plugin registry that holds loaded samplers, processors, and their factories.
pub struct PluginRegistry {
    samplers: HashMap<String, Arc<dyn Sampler>>,
    manifests: HashMap<String, PluginManifest>,
    processor_chain: Vec<PluginManifest>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            samplers: HashMap::new(),
            manifests: HashMap::new(),
            processor_chain: Vec::new(),
        }
    }

    /// Register a sampler plugin by name.
    pub fn register_sampler(&mut self, name: String, sampler: Arc<dyn Sampler>) {
        self.samplers.insert(name, sampler);
    }

    /// Get a sampler by name.
    pub fn get_sampler(&self, name: &str) -> Option<Arc<dyn Sampler>> {
        self.samplers.get(name).cloned()
    }

    /// Check if a sampler is registered.
    pub fn has_sampler(&self, name: &str) -> bool {
        self.samplers.contains_key(name)
    }

    /// Register a manifest for a loaded plugin.
    pub fn register_manifest(&mut self, manifest: PluginManifest) -> Result<()> {
        // §6.3 Processing pipeline composition checks:
        // Reject duplicate plugin identity at load time, independent of whether
        // the optional stage/priority fields are present.
        if self.manifests.contains_key(&manifest.name) {
            return Err(grim_tensor::Error::Backend(format!(
                "Duplicate plugin registered: '{}'",
                manifest.name
            )));
        }
        // §6.3 Processing pipeline composition checks:
        // Reject duplicate (stage, priority) pairs at load time. A processor
        // declaring only ONE of stage/priority is still chainable — the
        // missing field defaults (stage "default", priority 0) instead of the
        // plugin silently disappearing from chain traversal.
        let mut effective = manifest.clone();
        if effective.stage.is_some() || effective.priority.is_some() {
            if effective.stage.is_none() {
                effective.stage = Some("default".to_string());
            }
            if effective.priority.is_none() {
                effective.priority = Some(0);
            }
            let stage = effective.stage.clone().expect("stage defaulted above");
            let priority = effective.priority.expect("priority defaulted above");
            for existing in &self.processor_chain {
                if existing.stage.as_ref() == Some(&stage) && existing.priority == Some(priority) {
                    return Err(grim_tensor::Error::Backend(format!(
                        "Duplicate processor stage/priority detected: ({stage}, {priority})"
                    )));
                }
            }
            self.processor_chain.push(effective.clone());
            // Maintain sorted priority order
            self.processor_chain
                .sort_by_key(|p| p.priority.unwrap_or(0));
        }

        let name = manifest.name.clone();
        self.manifests.insert(name, effective);
        Ok(())
    }

    /// Get manifest for a plugin.
    pub fn get_manifest(&self, name: &str) -> Option<&PluginManifest> {
        self.manifests.get(name)
    }

    /// List all registered sampler names.
    pub fn list_samplers(&self) -> Vec<&String> {
        self.samplers.keys().collect()
    }

    /// Scan a directory at startup to discover and load plugin manifests and binaries (§6)
    pub fn scan_plugin_directory<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let dir = path.as_ref();
        if !dir.exists() || !dir.is_dir() {
            return Ok(());
        }
        tracing::info!(dir = ?dir, "[Plugin System] scanning directory for plugins");
        for entry in std::fs::read_dir(dir)
            .map_err(|e| grim_tensor::Error::Backend(format!("read_dir failed: {e}")))?
        {
            let entry =
                entry.map_err(|e| grim_tensor::Error::Backend(format!("entry failed: {e}")))?;
            let p = entry.path();
            if p.is_dir() {
                let manifest_path = p.join("plugin.grim.toml");
                if manifest_path.exists() {
                    if let Ok(toml_content) = std::fs::read_to_string(&manifest_path) {
                        if let Ok(manifest) = parse_manifest(&toml_content) {
                            println!(
                                "[Plugin System] Discovered plugin: {} ({:?})",
                                manifest.name, manifest.kind
                            );
                            self.register_manifest(manifest)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Audit fix gate: a processor manifest with only `stage` (no priority)
    /// — or only `priority` (no stage) — must still enter processor_chain
    /// with the missing field defaulted, not silently vanish from
    /// chain traversal.
    #[test]
    fn partial_processor_manifests_still_enter_chain() {
        let mut registry = PluginRegistry::new();

        let stage_only = PluginManifest {
            name: "stage-only".into(),
            abi_version: 1,
            kind: PluginKind::Wasm,
            capabilities: PluginCapabilities::PRE_POST_PROCESSOR,
            entry: "p.wasm".into(),
            sha256: None,
            limits: None,
            stage: Some("post".into()),
            priority: None,
            grants: PluginGrants::default(),
            reload: PluginReload::default(),
        };
        registry.register_manifest(stage_only).unwrap();

        let priority_only = PluginManifest {
            name: "priority-only".into(),
            abi_version: 1,
            kind: PluginKind::Wasm,
            capabilities: PluginCapabilities::PRE_POST_PROCESSOR,
            entry: "q.wasm".into(),
            sha256: None,
            limits: None,
            stage: None,
            priority: Some(5),
            grants: PluginGrants::default(),
            reload: PluginReload::default(),
        };
        registry.register_manifest(priority_only).unwrap();

        // Both must be visible to chain traversal (2 entries), with the
        // missing fields defaulted: stage-only → priority 0, priority-only
        // → stage "default".
        assert_eq!(registry.processor_chain.len(), 2);
        let by_name = |n: &str| {
            registry
                .processor_chain
                .iter()
                .find(|m| m.name == n)
                .unwrap()
                .clone()
        };
        let so = by_name("stage-only");
        assert_eq!(so.stage.as_deref(), Some("post"));
        assert_eq!(so.priority, Some(0));
        let po = by_name("priority-only");
        assert_eq!(po.stage.as_deref(), Some("default"));
        assert_eq!(po.priority, Some(5));

        // Duplicate (stage, priority) after defaulting must still be
        // rejected: another stage-less processor also defaults to
        // ("default", 0) — but that collides with nothing here. A second
        // stage-only processor DOES collide with stage-only's ("post", 0).
        let dup = PluginManifest {
            name: "dup-post".into(),
            abi_version: 1,
            kind: PluginKind::Wasm,
            capabilities: PluginCapabilities::PRE_POST_PROCESSOR,
            entry: "r.wasm".into(),
            sha256: None,
            limits: None,
            stage: Some("post".into()),
            priority: None,
            grants: PluginGrants::default(),
            reload: PluginReload::default(),
        };
        let err = registry.register_manifest(dup).unwrap_err().to_string();
        assert!(
            err.contains("Duplicate processor stage/priority"),
            "defaulted collision must be rejected: {err}"
        );
    }

    #[test]
    fn parse_simple_manifest() {
        let toml = r#"
[plugin]
name = "grammar-constrained-json"
abi_version = 1
kind = "wasm"
capabilities = ["sampler"]
entry = "grammar_json.wasm"

[plugin.limits]
fuel = 5000000
max_memory_mb = 64
"#;
        let m = parse_manifest(toml).unwrap();
        assert_eq!(m.name, "grammar-constrained-json");
        assert_eq!(m.abi_version, 1);
        assert_eq!(m.kind, PluginKind::Wasm);
        assert!(m.capabilities.contains(PluginCapabilities::SAMPLER));
        assert!(m.limits.is_some());
        let limits = m.limits.unwrap();
        assert_eq!(limits.fuel_per_invocation, Some(5_000_000));
        assert_eq!(limits.max_memory_mb, Some(64));
    }

    #[test]
    fn abi_validation() {
        let m = PluginManifest {
            name: "t".into(),
            abi_version: 1,
            kind: PluginKind::Wasm,
            capabilities: PluginCapabilities::SAMPLER,
            entry: "t.wasm".into(),
            sha256: None,
            limits: None,
            stage: None,
            priority: None,
            grants: PluginGrants::default(),
            reload: PluginReload::default(),
        };
        assert!(validate_abi(&m, 1).is_ok());
        assert!(validate_abi(&m, 2).is_err());
    }

    #[test]
    fn plugin_registry_basic_operations() {
        use grim_core::Sampler;

        // Create a simple test sampler
        struct TestSampler;
        impl Sampler for TestSampler {
            fn sample(&self, _logits: &grim_tensor::Tensor, _history: &[u32]) -> Result<u32> {
                Ok(42)
            }
            fn name(&self) -> &str {
                "test-sampler"
            }
        }

        let mut registry: PluginRegistry = PluginRegistry::new();
        assert!(registry.get_sampler("test-sampler").is_none());

        registry.register_sampler("test-sampler".to_string(), Arc::new(TestSampler));
        assert!(registry.has_sampler("test-sampler"));
    }
}
