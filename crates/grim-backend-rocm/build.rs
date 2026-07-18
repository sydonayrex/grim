/// Pure ROCm/RCCL lib-dir discovery, shared with `src/rocm_detect.rs`
/// (single source of truth, `include!`-ed — no duplicated knowledge).
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_rocm_detect.rs"));

/// Crate manifest dir is `crates/grim-backend-rocm`; workspace root is
/// two parents up.
fn workspace_root() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| manifest.clone())
}

fn main() {
    // Base system ROCm (may or may not carry MIOpen/RCCL).
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");

    // Side-by-side per-arch ROCm runtimes live in the workspace (grim's
    // .rocm-2/3/4 trees). They DO carry libMIOpen.so / librccl.so, so the
    // F9/F11 dylib links resolve against whichever runtime is active.
    // ROCM_PATH overrides; otherwise probe the local .rocm-N/lib dirs.
    let workspace = workspace_root();

    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        let lib = PathBuf::from(&rocm_path).join("lib");
        if lib.exists() {
            println!("cargo:rustc-link-search=native={}", lib.display());
        }
    }
    for dir in ["rocm-2", "rocm-3", "rocm-4"] {
        let lib = workspace.join(dir).join("lib");
        if lib.exists() {
            println!("cargo:rustc-link-search=native={}", lib.display());
        }
    }

    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rocblas");

    // grim-sonnet F11 — RCCL is OPTIONAL and discoverable (WI-R0).
    //
    // The `rccl` feature gates multi-GPU collectives. Cargo does NOT pass
    // the crate's own features to the build script via `cfg!` — build
    // scripts see them as `CARGO_FEATURE_<NAME>` env vars instead. So we
    // gate on that env var, not `#[cfg(feature = "rccl")]`.
    //
    // When the feature is OFF (the default, single-GPU consumer builds) we
    // emit nothing and never require librccl.so. When ON, we resolve the
    // lib dir from RCCL_PATH / ROCM_RCCL_PATH / ROCM_PATH / standard
    // prefixes / workspace .rocm-N, and only then emit the link directive.
    // A system without RCCL + feature ON compiles but prints a warning
    // (no hard link failure) and the runtime wrappers return
    // Error::Unsupported.
    if std::env::var("CARGO_FEATURE_RCCL").is_ok() {
        match resolve_rocm_lib_dir(&workspace) {
            Some(dir) => {
                println!("cargo:rustc-link-search=native={}", dir.display());
                println!("cargo:rustc-link-lib=dylib=rccl");
            }
            None => {
                println!(
                    "cargo:warning=rccl feature enabled but no librccl.so \
                     found via RCCL_PATH/ROCM_RCCL_PATH/ROCM_PATH; \
                     multi-GPU collectives will be unavailable"
                );
            }
        }
        println!("cargo:rerun-if-env-changed=RCCL_PATH");
        println!("cargo:rerun-if-env-changed=ROCM_RCCL_PATH");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
    }
    println!("cargo:rerun-if-changed=build.rs");

    // F9 MIOpen is loaded dynamically via libloading (dlopen) in accel_ffi.rs
    // — no link-time .so required, since no real libMIOpen.so exists in this
    // environment (only dangling symlinks in .rocm-3/.rocm-4).

    // grim-sonnet F8 — Composable Kernel (ck_tile) GEMM.
    //
    // grim is Rust-centric: there is no build-time C/C++ compilation. The
    // decode-GEMM kernel lives as an embedded HIP source literal in
    // `kernels::decode_gemm::KERNEL_SOURCE` and is JIT-compiled at runtime via
    // `hipModuleLoad` / `hipModuleGetFunction` / `hipModuleLaunchKernel` — the
    // same path as every other grim compute kernel
    // (see `kernels::compute_kernels` / `kernels::qkv_attention`). Dispatch
    // is gated by `DecodeGemmConfig::enabled` (default off) in
    // `RocmDevice::matmul`. The vendored CK headers under
    // `old/repos/rocm-libraries-develop/` remain dead reference code; grim
    // never invokes them, and no C/C++ wrapper is built.
}
