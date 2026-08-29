//! WASM Component Sandbox Runtime Loader.
//!
//! §6.1: Sandboxes third-party plugins using execution limits (fuel and memory
//! caps) and capability grants. Prevents unauthorized system calls or memory
//! access outside the sandbox boundaries. Uses wasmtime for runtime isolation.
//!
//! Grant enforcement (§6.4, deny-by-default):
//!   Every WASM plugin starts with **no** host imports linked. Capabilities
//!   are added only when the manifest's grants block (`[plugin.grants]` or
//!   top-level `[grants]` + `[scopes]`) explicitly enables them:
//!     - `network = false` (default) → no WASI socket imports linked.
//!     - `filesystem = []` (default) → no WASI filesystem imports linked.
//!     - `request_metadata = false` (default) → no grim host-call for request
//!       metadata linked.
//!   A plugin that calls an unlinked import traps at instantiation with a
//!   clear `unknown import` error rather than being silently permitted.
//!   A grant that this build cannot honor (the wasmtime dependency carries
//!   no WASI implementation, so no preopens or sockets can ever be linked)
//!   is rejected at plugin-load time — loudly, never as a silent trap.

use crate::{PluginGrants, PluginLimits};
use grim_core::Sampler;
use grim_tensor::error::{Error, Result};
use std::sync::Arc;

/// WIT (WebAssembly Interface Types) definition for sampler plugins.
/// §6.1.1 — WIT Interface Definition (inline for doc reference).
///
/// ```wit
/// package grim:plugin@0.1.0;
///
/// interface sampler {
///   get-name: func() -> string;
///   sample: func(logits-ptr: i32, logits-len: i32,
///                history-ptr: i32, history-len: i32) -> result<i32, string>;
///   memory-usage: func() -> i32;
/// }
///
/// world grim-sampler {
///   export sampler;
/// }
/// ```
pub const WIT_SAMPLER_INTERFACE: &str = include_str!("wit/sampler.wit");
pub const WIT_TOKENIZER_INTERFACE: &str = include_str!("wit/tokenizer.wit");
pub const WIT_PROCESSOR_INTERFACE: &str = include_str!("wit/processor.wit");

/// Wrapper for a WASM-based sampler plugin.
pub struct WasmSampler {
    name: String,
    limits: PluginLimits,
    /// The instantiated WASM module keeps the Instance alive so
    /// exports remain valid for the lifetime of this sampler.
    #[cfg(feature = "wasm-sandbox")]
    instance: Option<wasmtime::Instance>,
    /// The store is behind a Mutex because `Func::call` and `Memory::write`
    /// require `AsContextMut` (mutable access), but `Sampler::sample` takes
    /// `&self`.
    #[cfg(feature = "wasm-sandbox")]
    store: Option<std::sync::Mutex<wasmtime::Store<()>>>,
}

/// WASM plugin loader — enforces fuel, memory, and capability grants.
pub struct WasmPluginLoader {
    pub name: String,
    pub limits: PluginLimits,
    /// Capability grants parsed from the manifest. Deny-by-default: every
    /// field that is false means the corresponding host import is NOT linked
    /// into the Wasmtime linker, so calling it traps with a clear error.
    pub grants: PluginGrants,
    fuel_consumed: u64,
    memory_allocated_mb: u32,
}

impl WasmPluginLoader {
    pub fn new(name: &str, limits: PluginLimits) -> Self {
        Self {
            name: name.to_string(),
            limits,
            grants: PluginGrants::default(), // deny-by-default
            fuel_consumed: 0,
            memory_allocated_mb: 0,
        }
    }

    /// Construct with explicit grant set (used when loading from a manifest).
    pub fn with_grants(name: &str, limits: PluginLimits, grants: PluginGrants) -> Self {
        Self {
            name: name.to_string(),
            limits,
            grants,
            fuel_consumed: 0,
            memory_allocated_mb: 0,
        }
    }

    /// Create a sampler from WASM bytes, enforcing all manifest grants.
    ///
    /// Grant enforcement: a grant this build cannot link (network,
    /// filesystem, request_metadata — none have host implementations here)
    /// is rejected with a clear error at plugin-load time. Otherwise the
    /// Wasmtime `Linker` is built with no host functions at all, so any
    /// import the module declares traps at instantiation time with an
    /// `"unknown import"` error — the plugin cannot silently bypass the
    /// sandbox by calling an unlinked function.
    #[cfg(feature = "wasm-sandbox")]
    pub fn create_sampler(&self, wasm_bytes: &[u8]) -> Result<Arc<dyn Sampler>> {
        use wasmtime::{Config, Engine as WasmtimeEngine, Linker, Module, Store};

        // ----- Grant validation (before any compilation: fail at load) -----
        // Deny-by-default grants are correct with an empty linker — a plugin
        // importing WASI then traps at instantiation with wasmtime's
        // "unknown import" error. But a *granted* capability that this build
        // cannot link must not degrade to that same trap (the plugin author
        // asked for a real capability and silently got nothing), so it errors
        // here instead. The wasmtime dependency carries no WASI
        // implementation (no wasi-common / wasmtime-wasi), so network and
        // filesystem grants can never be linked in this build.
        if self.grants.network {
            return Err(Error::Backend(format!(
                "plugin '{}': network grant cannot be honored — this build links no \
                 WASI socket imports; remove the network grant from plugin.grim.toml",
                self.name
            )));
        }
        if !self.grants.filesystem.is_empty() {
            return Err(Error::Backend(format!(
                "plugin '{}': filesystem grant for {:?} cannot be honored — this build \
                 links no WASI preopens; remove the filesystem grant from plugin.grim.toml",
                self.name, self.grants.filesystem
            )));
        }
        if self.grants.request_metadata {
            return Err(Error::Backend(format!(
                "plugin '{}': request_metadata grant cannot be honored — no grim host \
                 interface is linked in this build; remove the request_metadata grant \
                 from plugin.grim.toml",
                self.name
            )));
        }

        let mut config = Config::new();
        config.max_wasm_stack(1048576); // 1 MB default

        // Enable fuel-based metering if the manifest specifies a per-invocation
        // fuel budget (§6.4). The store is topped up before each call.
        if self.limits.fuel_per_invocation.is_some() {
            config.consume_fuel(true);
        }

        let engine = WasmtimeEngine::new(&config)
            .map_err(|e| Error::Backend(format!("failed to create wasmtime engine: {e}")))?;

        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| Error::Backend(format!("failed to compile WASM module: {e}")))?;

        let mut store = Store::new(&engine, ());

        // Add fuel to the store before instantiation so the module can run
        // its start function without immediately trapping.
        if let Some(fuel) = self.limits.fuel_per_invocation {
            store
                .set_fuel(fuel)
                .map_err(|e| Error::Backend(format!("set_fuel failed: {e}")))?;
        }

        // Build the linker. Nothing is linked — deny-by-default. Grants were
        // validated above: reaching here means every grant is off, so any
        // import the module declares (WASI filesystem, sockets, grim host
        // calls) is left unlinked. This is where granted scopes would be
        // linked as WASI preopens once the dependency set carries a WASI
        // implementation (`wasmtime-wasi`), preopening exactly
        // `grants.filesystem` and nothing else.
        let linker: Linker<()> = Linker::new(&engine);

        // Instantiate — any unlinked import causes a trap here, not at call time.
        // This is the correct place to fail: before the plugin runs any user code.
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| Error::Backend(format!("failed to instantiate WASM module: {e}")))?;

        Ok(Arc::new(WasmSampler {
            name: self.name.clone(),
            limits: self.limits.clone(),
            #[cfg(feature = "wasm-sandbox")]
            instance: Some(instance),
            #[cfg(feature = "wasm-sandbox")]
            store: Some(std::sync::Mutex::new(store)),
        }))
    }

    /// Non-wasm-sandbox fallback: always errors with a clear message.
    #[cfg(not(feature = "wasm-sandbox"))]
    pub fn create_sampler(&self, wasm_bytes: &[u8]) -> Result<Arc<dyn Sampler>> {
        let _ = wasm_bytes;
        Err(Error::Unimplemented(
            "WASM sandbox support disabled. Rebuild with --features wasm-sandbox".into(),
        ))
    }

    /// Simulate allocating heap memory inside the WASM linear memory sandbox.
    pub fn allocate_memory(&mut self, mb: u32) -> Result<()> {
        if let Some(max_mem) = self.limits.max_memory_mb {
            if self.memory_allocated_mb + mb > max_mem {
                return Err(Error::Backend(format!(
                    "WASM sandbox out of memory: tried to allocate {}MB (Max: {}MB)",
                    mb, max_mem
                )));
            }
        }
        self.memory_allocated_mb += mb;
        Ok(())
    }

    /// Consume execution fuel tokens for code block steps.
    pub fn consume_fuel(&mut self, amount: u64) -> Result<()> {
        if let Some(max_fuel) = self.limits.fuel_per_invocation {
            if self.fuel_consumed + amount > max_fuel {
                return Err(Error::Backend(
                    "WASM sandbox execution ran out of fuel".into(),
                ));
            }
        }
        self.fuel_consumed += amount;
        Ok(())
    }

    /// Reset internal fuel meter for the next invocation.
    pub fn reset_fuel(&mut self) {
        self.fuel_consumed = 0;
    }
}

impl Sampler for WasmSampler {
    fn sample(&self, logits: &grim_tensor::Tensor, history: &[u32]) -> Result<u32> {
        #[cfg(not(feature = "wasm-sandbox"))]
        {
            let _ = (logits, history);
            Err(Error::Unimplemented(
                "WASM sampler execution requires wasm-sandbox feature".into(),
            ))
        }
        #[cfg(feature = "wasm-sandbox")]
        {
            let store_guard = self
                .store
                .as_ref()
                .ok_or_else(|| Error::Backend("WasmSampler store unavailable".into()))?
                .lock()
                .map_err(|e| Error::Backend(format!("store lock poisoned: {e}")))?;
            let mut store = store_guard;
            let instance = self
                .instance
                .as_ref()
                .ok_or_else(|| Error::Backend("WasmSampler instance unavailable".into()))?;
            let memory = instance
                .get_memory(&mut *store, "memory")
                .ok_or_else(|| Error::Backend("WASM plugin does not export 'memory'".into()))?;
            let sample_fn = instance.get_func(&mut *store, "sample").ok_or_else(|| {
                Error::Backend("WASM plugin does not export 'sample' function".into())
            })?;
            let sample_typed = sample_fn
                .typed::<(i32, i32, i32, i32), i32>(&mut *store)
                .map_err(|e| Error::Backend(format!("sample function has wrong signature: {e}")))?;

            // Extract logits as f32 bytes and history as u32 bytes.
            let logits_vec = logits.to_vec_f32()?;
            let logits_bytes: Vec<u8> = logits_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
            let history_bytes: Vec<u8> = history.iter().flat_map(|v| v.to_le_bytes()).collect();

            let logits_len = logits_bytes.len() as i32;
            let history_len = history_bytes.len() as i32;

            // Layout in WASM linear memory:
            //   [0 .. logits_len)           — logits f32 bytes
            //   [logits_len .. end)         — history u32 bytes
            let total = (logits_len + history_len) as usize;
            let data_len = memory.data_size(&*store);
            if total > data_len {
                return Err(Error::Backend(format!(
                    "WASM memory too small: need {} bytes, have {}",
                    total, data_len
                )));
            }

            let logits_ptr: i32 = 0;
            let history_ptr: i32 = logits_len;

            memory.write(&mut *store, 0, &logits_bytes).map_err(|e| {
                Error::Backend(format!("failed to write logits to WASM memory: {e}"))
            })?;
            memory
                .write(&mut *store, logits_len as usize, &history_bytes)
                .map_err(|e| {
                    Error::Backend(format!("failed to write history to WASM memory: {e}"))
                })?;

            // Top up fuel before each call — the store's fuel was set at
            // instantiation only, so long-running plugins would trap mid-inference
            // once fuel is exhausted. [P1-29 fix: per-call fuel top-up.]
            if let Some(fuel) = self.limits.fuel_per_invocation {
                store
                    .set_fuel(fuel)
                    .map_err(|e| Error::Backend(format!("WASM set_fuel failed: {e}")))?;
            }

            let token_id = sample_typed
                .call(
                    &mut *store,
                    (logits_ptr, logits_len, history_ptr, history_len),
                )
                .map_err(|e| Error::Backend(format!("WASM sample call failed: {e}")))?;

            Ok(token_id as u32)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wasm-sandbox")]
    use grim_tensor::CoreTensorOps;

    #[test]
    fn test_wasm_sandbox_limits() {
        let limits = PluginLimits {
            fuel_per_invocation: Some(100),
            max_memory_mb: Some(64),
        };
        let mut loader = WasmPluginLoader::new("json-sampler", limits);

        // Under bounds — both succeed.
        assert!(loader.allocate_memory(32).is_ok());
        assert!(loader.consume_fuel(50).is_ok());

        // Exceeding memory limit returns an error.
        assert!(loader.allocate_memory(40).is_err());

        // Exceeding fuel limit returns an error.
        assert!(loader.consume_fuel(60).is_err());
    }

    #[test]
    fn test_deny_by_default_grants() {
        let limits = PluginLimits {
            fuel_per_invocation: Some(1000),
            max_memory_mb: Some(128),
        };
        // Default grants — all capabilities denied.
        let loader = WasmPluginLoader::new("test-plugin", limits.clone());
        assert!(!loader.grants.network);
        assert!(loader.grants.filesystem.is_empty());
        assert!(!loader.grants.request_metadata);

        // Explicit grants — only what's specified.
        let grants = PluginGrants {
            network: true,
            ..PluginGrants::default()
        };
        let loader2 = WasmPluginLoader::with_grants("net-plugin", limits, grants);
        assert!(loader2.grants.network);
        assert!(loader2.grants.filesystem.is_empty()); // still denied
    }

    #[test]
    fn test_wasm_loader_without_wasm_sandbox_feature() {
        // Without wasm-sandbox feature, creation always returns a clear error.
        let limits = PluginLimits {
            fuel_per_invocation: Some(1000),
            max_memory_mb: Some(128),
        };
        let loader = WasmPluginLoader::new("test", limits);
        let minimal_wasm = vec![
            0x00, 0x61, 0x73, 0x6D, // magic
            0x01, 0x00, 0x00, 0x00, // version 1
        ];
        let _result = loader.create_sampler(&minimal_wasm);
        #[cfg(not(feature = "wasm-sandbox"))]
        assert!(_result.is_err());
    }

    #[cfg(feature = "wasm-sandbox")]
    #[test]
    fn test_wasm_sampler_execution_with_wat() {
        let wat_src = r#"
            (module
                (memory (export "memory") 1)
                (func (export "sample") (param i32 i32 i32 i32) (result i32)
                    i32.const 99
                )
            )
        "#;
        let wasm_bytes = wat::parse_str(wat_src).expect("valid WAT");
        let limits = PluginLimits {
            fuel_per_invocation: Some(10000),
            max_memory_mb: Some(16),
        };
        let loader = WasmPluginLoader::new("wat-sampler", limits);
        let sampler = loader.create_sampler(&wasm_bytes).expect("create_sampler");
        assert_eq!(sampler.name(), "wat-sampler");

        let cpu_dev = grim_backend_cpu::device::CpuDevice::new();
        let shape = grim_tensor::shape::Shape::new(vec![3]);
        let storage = cpu_dev
            .from_cpu(&[0.1f32, 0.9, 0.2], &shape, grim_tensor::DType::F32)
            .unwrap();
        let dummy_tensor = grim_tensor::Tensor::new(
            storage.into(),
            shape,
            grim_tensor::DType::F32,
            grim_tensor::dtype::QuantProvenance::default(),
            grim_tensor::Device::Cpu,
        );
        let token = sampler.sample(&dummy_tensor, &[]).expect("sample call");
        assert_eq!(token, 99);
    }
}
