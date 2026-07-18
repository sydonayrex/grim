use std::path::PathBuf;

fn main() {
    // Base system ROCm (may or may not carry MIOpen/RCCL).
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");

    // Side-by-side per-arch ROCm runtimes live in the workspace (grim's
    // .rocm-2/3/4 trees). They DO carry libMIOpen.so / librccl.so, so the
    // F9/F11 dylib links resolve against whichever runtime is active.
    // ROCM_PATH overrides; otherwise probe the local .rocm-N/lib dirs.
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // CARGO_MANIFEST_DIR is crates/grim-backend-rocm; workspace root is two up.
    let workspace = manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| manifest.clone());

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
    // grim-sonnet F11: real libcrccl.so.1.0 in /opt/rocm/lib.
    println!("cargo:rustc-link-lib=dylib=rccl");
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
