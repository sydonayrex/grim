fn main() {
    // Only emit CUDA link directives when a CUDA toolkit is actually present
    // on the host. `grim-backend-cuda` is a hard dependency of `grim-nn`, which
    // other backends (e.g. `grim-backend-rocm`) pull in via their dev-dependency
    // graph even though they never invoke CUDA at runtime. Unconditionally
    // linking `-lcudart`/`-lcublas`/`-lcuda` makes every such consumer fail to
    // link on CUDA-less (ROCm-only / Metal-only) hosts with
    // "unable to find library -lcudart".
    if cuda_available() {
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }
}

/// True when the CUDA toolkit is discoverable on this host.
fn cuda_available() -> bool {
    if std::env::var("CUDA_HOME").is_ok() || std::env::var("CUDA_PATH").is_ok() {
        return true;
    }
    for p in [
        "/opt/cuda/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/lib/x86_64-linux-gnu/libcudart.so",
    ] {
        if std::path::Path::new(p).exists() {
            return true;
        }
    }
    false
}
