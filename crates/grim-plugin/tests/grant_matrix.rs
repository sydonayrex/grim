//! Grant matrix tests for the WASM plugin sandbox (§6.4, deny-by-default).
//!
//! Three manifest shapes the loader must distinguish:
//!   1. default manifest → all grants denied (empty linker; WASI imports trap)
//!   2. filesystem grant without scopes → clear error at plugin-load time
//!   3. filesystem grant with scopes → manifest parses, scopes recorded
//!
//! plus loader-level enforcement of grants this build cannot link.

use grim_plugin::{PluginGrants, PluginLimits, WasmPluginLoader, parse_manifest};

const MINIMAL_MANIFEST: &str = r#"
[plugin]
name = "grant-matrix"
abi_version = 1
kind = "wasm"
capabilities = ["sampler"]
entry = "sampler.wasm"
"#;

/// Valid-but-empty WASM module (magic + version). Grant validation runs
/// before compilation, but keep the bytes valid so failures mean what
/// they say.
#[cfg(feature = "wasm-sandbox")]
fn minimal_wasm() -> Vec<u8> {
    vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]
}

#[cfg(feature = "wasm-sandbox")]
fn limits() -> PluginLimits {
    PluginLimits {
        fuel_per_invocation: Some(10_000),
        max_memory_mb: Some(16),
    }
}

#[test]
fn default_manifest_grants_all_false() {
    let m = parse_manifest(MINIMAL_MANIFEST).unwrap();
    assert!(!m.grants.network);
    assert!(m.grants.filesystem.is_empty());
    assert!(!m.grants.request_metadata);
}

#[test]
fn filesystem_grant_without_scopes_is_a_load_error() {
    let toml = r#"
[plugin]
name = "fs-no-scopes"
abi_version = 1
kind = "wasm"
entry = "s.wasm"

[grants]
filesystem = true
"#;
    let err = parse_manifest(toml).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("filesystem grant requires [scopes].allowed_dirs"),
        "{msg}"
    );
}

#[test]
fn filesystem_grant_with_empty_scopes_list_is_a_load_error() {
    let toml = r#"
[plugin]
name = "fs-empty-scopes"
abi_version = 1
kind = "wasm"
entry = "s.wasm"

[grants]
filesystem = true

[scopes]
allowed_dirs = []
"#;
    let err = parse_manifest(toml).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("filesystem grant requires [scopes].allowed_dirs"),
        "{msg}"
    );
}

#[test]
fn filesystem_grant_with_scopes_records_scopes() {
    let toml = r#"
[plugin]
name = "fs-scoped"
abi_version = 1
kind = "wasm"
entry = "s.wasm"

[grants]
network = false
filesystem = true

[scopes]
allowed_dirs = ["testdata"]
"#;
    let m = parse_manifest(toml).unwrap();
    assert_eq!(m.grants.filesystem, vec!["testdata".to_string()]);
    // Granting filesystem does not flip the other grants.
    assert!(!m.grants.network);
    assert!(!m.grants.request_metadata);
}

#[test]
fn network_grant_is_recorded() {
    let toml = r#"
[plugin]
name = "net-granted"
abi_version = 1
kind = "wasm"
entry = "s.wasm"

[grants]
network = true
"#;
    let m = parse_manifest(toml).unwrap();
    assert!(m.grants.network);
    assert!(m.grants.filesystem.is_empty());
}

// ----- Loader-level enforcement (this build links no WASI host imports) -----

#[cfg(feature = "wasm-sandbox")]
#[test]
fn default_grants_load_but_wasi_imports_trap() {
    // Deny-by-default: the linker is empty, so a module importing WASI
    // cannot be instantiated — wasmtime reports the unknown import by name.
    let wat_src = r#"
        (module
            (import "wasi_snapshot_preview1" "random_get"
                (func $random_get (param i32 i32) (result i32)))
            (memory (export "memory") 1)
            (func (export "sample") (param i32 i32 i32 i32) (result i32)
                i32.const 1
            )
        )
    "#;
    let wasm_bytes = wat::parse_str(wat_src).expect("valid WAT");
    let loader = WasmPluginLoader::new("denied-wasi", limits());
    let err = match loader.create_sampler(&wasm_bytes) {
        Err(e) => e,
        Ok(_) => panic!("WASI-importing module must not instantiate under deny-by-default grants"),
    };
    let msg = err.to_string();
    assert!(msg.contains("failed to instantiate"), "{msg}");
    assert!(msg.contains("wasi_snapshot_preview1"), "{msg}");
}

#[cfg(feature = "wasm-sandbox")]
#[test]
fn network_grant_is_rejected_at_load() {
    // A network grant cannot be linked in this build — it must error loudly
    // at plugin-load time, not degrade to a silent trap.
    let grants = PluginGrants {
        network: true,
        ..PluginGrants::default()
    };
    let loader = WasmPluginLoader::with_grants("net-granted", limits(), grants);
    let err = match loader.create_sampler(&minimal_wasm()) {
        Err(e) => e,
        Ok(_) => panic!("network grant must be rejected at plugin-load time"),
    };
    assert!(err.to_string().contains("network grant"));
}

#[cfg(feature = "wasm-sandbox")]
#[test]
fn filesystem_grant_is_rejected_at_load() {
    // Scoped or not, a filesystem grant cannot be linked in this build —
    // clear error at load time instead of an unhonored grant.
    let grants = PluginGrants {
        filesystem: vec!["testdata".to_string()],
        ..PluginGrants::default()
    };
    let loader = WasmPluginLoader::with_grants("fs-granted", limits(), grants);
    let err = match loader.create_sampler(&minimal_wasm()) {
        Err(e) => e,
        Ok(_) => panic!("filesystem grant must be rejected at plugin-load time"),
    };
    assert!(err.to_string().contains("filesystem grant"));
}
