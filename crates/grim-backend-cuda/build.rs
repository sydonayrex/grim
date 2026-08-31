//! Propagate backend link directives to dependents (tests, binaries).
//! Tests using grim-engine pull in both ROCm and CUDA backends, so we
//! need to resolve symbols for both.

fn main() {
    // CUDA library search path.
    for path in &["/opt/cuda/lib64", "/opt/cuda/lib", "/usr/local/cuda/lib64"] {
        if std::path::Path::new(path).exists() {
            println!("cargo:rustc-link-search=native={}", path);
        }
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
