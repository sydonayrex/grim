//! Propagate backend link directives to dependents (tests, binaries).
//! Tests using grim-engine pull in both ROCm and CUDA backends, so we
//! need to resolve symbols for both.

fn main() {
    // CUDA library search path.
    for path in &[
        "/opt/cuda/lib64",
        "/opt/cuda/lib64/stubs",
        "/opt/cuda/lib",
        "/opt/cuda/lib/stubs",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib64/stubs",
        "/opt/resolve/libs",
        "/usr/local/lib/ollama/cuda_v12",
        "/usr/local/lib/ollama/cuda_v13",
    ] {
        if std::path::Path::new(path).exists() {
            println!("cargo:rustc-link-search=native={}", path);
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
        }
    }
    // Ensure libcuda.so.1 is present in OUT_DIR if only libcuda.so stub exists
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let compat_dir = std::path::Path::new(&out_dir).join("cuda_compat");
    let _ = std::fs::create_dir_all(&compat_dir);
    let target_so = compat_dir.join("libcuda.so.1");
    if !target_so.exists() {
        for stub_candidate in &[
            "/opt/cuda/lib64/stubs/libcuda.so",
            "/usr/local/cuda/lib64/stubs/libcuda.so",
        ] {
            if std::path::Path::new(stub_candidate).exists() {
                let _ = std::os::unix::fs::symlink(stub_candidate, &target_so);
                break;
            }
        }
    }
    if target_so.exists() {
        println!("cargo:rustc-link-search=native={}", compat_dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", compat_dir.display());
    }

    // CUDA runtime and math libraries.
    for lib in &["cudart", "cublas", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }

    // ROCm library search path.
    for path in &["/opt/rocm/lib", "/opt/rocm/lib64"] {
        if std::path::Path::new(path).exists() {
            println!("cargo:rustc-link-search=native={}", path);
        }
    }
    // ROCm HIP runtime compiler (used by ROCm JIT).
    if std::path::Path::new("/opt/rocm/lib/libhiprtc.so").exists() {
        println!("cargo:rustc-link-lib=dylib=hiprtc");
    }
}
