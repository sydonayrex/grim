// grim-hip-sys build script — the hybrid linkage core.
//
// Vendored from tracel-ai/cubecl-hip (cubecl-hip-sys), MIT OR Apache-2.0,
// with the default inverted for grim:
//   * default (static): the committed bindgen file is copied verbatim and the
//     HIP/hiprtc libraries are linked at build time, discovered with the same
//     logic grim-backend-rocm uses (`build_rocm_detect.rs`, include!-ed —
//     single source of truth, no duplicated knowledge).
//   * `dynamic` feature: cubecl's rewrite turns every `extern "C" fn hip*`
//     declaration into a libloading-resolved wrapper, and no link directives
//     are emitted — the binary builds without ROCm present.
//
// bindings selection: the HIP patch reported by `hipconfig --version` picks
// the committed `src/bindings/bindings_<patch>.rs`; fallback = the highest
// committed patch, with a warning (same posture as upstream cubecl-hip-sys).

include!("src/hipconfig.rs");

use std::{fs, path::Path};

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-changed=build.rs");
    for entry in fs::read_dir("src/bindings").expect("src/bindings exists") {
        let path = entry.expect("readable entry").path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    // Exactly one bindings file: from the installed HIP patch, else the
    // highest committed patch (compile-without-ROCm posture).
    let selected = match get_hip_patch_version() {
        Ok(ref patch) if bindings_file_exists(patch) => format!("hip_{patch}"),
        _ => extract_latest_hip_feature_from_path("Cargo.toml").unwrap(),
    };
    if std::env::var("CARGO_FEATURE_HIP_26385").is_ok() {
        panic!("HIP_XXX features must not be set manually; the build script selects them from the installed HIP patch");
    }
    println!("cargo:rustc-cfg=feature=\"{selected}\"");

    let dynamic = std::env::var("CARGO_FEATURE_DYNAMIC").is_ok();
    let out_dir = Path::new(&std::env::var_os("OUT_DIR").expect("OUT_DIR is set")).to_path_buf();
    let source =
        fs::read_to_string(format!("src/bindings/bindings_{}.rs", selected.trim_start_matches("hip_")))
            .expect("selected bindings file exists");
    let contents = if dynamic {
        // Vendored rewrite: extern "C" fn hip* -> libloading-resolved wrappers.
        rewrite_bindings(&source)
    } else {
        source
    };
    fs::write(out_dir.join("bindings.rs"), contents).expect("write bindings");

    if !dynamic {
        // Static arm: link-time linkage through grim's discovery logic.
        // The sys crate is only compiled when grim-backend-rocm's `rocm`
        // feature is on (the dep is optional + feature-forwarded), so emitting
        // the directives whenever the library exists is safe.
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("manifest dir is crates/grim-hip-sys")
            .to_path_buf();
        if let Some(dir) = resolve_hip_lib_dir(&workspace_root) {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=dylib=amdhip64");
            println!("cargo:rustc-link-lib=dylib=hiprtc");
        }
    }
}

fn bindings_file_exists(patch: &str) -> bool {
    Path::new("src/bindings").join(format!("bindings_{patch}.rs")).exists()
}

include!("build_rewrite.rs");
include!("../../crates/grim-backend-rocm/build_rocm_detect.rs");
