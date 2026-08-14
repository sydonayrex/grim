//! Build script for the Vulkan backend.
//!
//! Compiles every GLSL compute kernel to SPIR-V via `glslangValidator` at
//! compile time. The `.spv` blobs are embedded into the crate via
//! `include_bytes!`, so `spirv_for()` returns genuine SPIR-V with no runtime
//! dependency on an external compiler. All kernels share a single
//! `push_constant` `Params` block, with dynamic dispatch values supplied via
//! push constants rather than baked into shader source.

use std::path::PathBuf;
use std::process::Command;

/// Load a kernel source file as-is (already contains `#version` and `PARAMS_GLSL`).
fn load_kernel(name: &str) -> String {
    let path = PathBuf::from("kernels").join(format!("{name}.comp"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read kernel source: {}", path.display()))
}

/// (kernel name, glsl source) used for `.comp`/`.spv` files and `include_bytes!` constants.
fn kernels() -> Vec<(&'static str, String)> {
    vec![
        ("add", load_kernel("add")),
        ("mul", load_kernel("mul")),
        ("silu_mul", load_kernel("silu_mul")),
        ("rms_norm", load_kernel("rms_norm")),
        ("add_rms_norm", load_kernel("add_rms_norm")),
        ("softmax", load_kernel("softmax")),
        ("embedding", load_kernel("embedding")),
        ("matmul_32", load_kernel("matmul_tile_32")),
        ("matmul_64", load_kernel("matmul_tile_64")),
        ("matmul_64_bf16", load_kernel("matmul_tile_64_bf16")),
        ("qkv_attention", load_kernel("qkv_attention")),
        ("qkv_attention_swa", load_kernel("qkv_attention_swa")),
        ("mul_scalar", load_kernel("mul_scalar")),
        ("sqrt", load_kernel("sqrt")),
        ("recip", load_kernel("recip")),
        ("rope", load_kernel("rope")),
        ("rope_yarn", load_kernel("rope_yarn")),
        (
            "fused_dequant_gemm_q4k",
            load_kernel("fused_dequant_gemm_q4k"),
        ),
        (
            "fused_dequant_gemm_q5k",
            load_kernel("fused_dequant_gemm_q5k"),
        ),
        (
            "fused_dequant_gemm_q6k",
            load_kernel("fused_dequant_gemm_q6k"),
        ),
        (
            "fused_dequant_gemm_q8_0",
            load_kernel("fused_dequant_gemm_q8_0"),
        ),
        (
            "fused_dequant_gemm_iq4nl",
            load_kernel("fused_dequant_gemm_iq4nl"),
        ),
        (
            "fused_dequant_gemm_iq4xs",
            load_kernel("fused_dequant_gemm_iq4xs"),
        ),
        (
            "fused_dequant_gemm_iq3xxs",
            load_kernel("fused_dequant_gemm_iq3xxs"),
        ),
        (
            "fused_dequant_gemm_iq3s",
            load_kernel("fused_dequant_gemm_iq3s"),
        ),
        (
            "fused_dequant_gemm_iq2xxs",
            load_kernel("fused_dequant_gemm_iq2xxs"),
        ),
        (
            "fused_dequant_gemm_iq2xs",
            load_kernel("fused_dequant_gemm_iq2xs"),
        ),
        (
            "fused_dequant_gemm_iq2s",
            load_kernel("fused_dequant_gemm_iq2s"),
        ),
        (
            "fused_dequant_gemm_fp8_e4m3",
            load_kernel("fused_dequant_gemm_fp8_e4m3"),
        ),
        (
            "fused_dequant_gemm_mxfp4",
            load_kernel("fused_dequant_gemm_mxfp4"),
        ),
        ("kv_dequant_attention", load_kernel("kv_dequant_attention")),
        ("selective_scan", load_kernel("selective_scan")),
        ("qkv_attention_paged", load_kernel("qkv_attention_paged")),
        ("qkv_attention_paged_swa", load_kernel("qkv_attention_paged_swa")),
        ("tree_attention", load_kernel("tree_attention")),
        ("flash_attention", load_kernel("flash_attention")),
        ("silu_mul_backward", load_kernel("silu_mul_backward")),
        (
            "quantized_matmul_backward_dx",
            load_kernel("quantized_matmul_backward_dx"),
        ),
        (
            "quantized_matmul_backward_dx_q8_0",
            load_kernel("quantized_matmul_backward_dx_q8_0"),
        ),
        (
            "quantized_matmul_backward_dx_generic",
            load_kernel("quantized_matmul_backward_dx_generic"),
        ),
        ("rwkv_time_mix", load_kernel("rwkv_time_mix")),
        ("rwkv_channel_mix", load_kernel("rwkv_channel_mix")),
        ("all_reduce", load_kernel("all_reduce")),
        ("comm_fuse_reduce", load_kernel("comm_fuse_reduce")),
        ("quant_q8_0", load_kernel("quant_q8_0")),
        ("quant_fp8", load_kernel("quant_fp8")),
        (
            "fused_quant_gemm_q8_0",
            load_kernel("fused_quant_gemm_q8_0"),
        ),
        ("fused_quant_gemm_fp8", load_kernel("fused_quant_gemm_fp8")),
        ("moe_fused_dispatch", load_kernel("moe_fused_dispatch")),
    ]
}

fn main() {
    println!("cargo:rustc-link-lib=dylib=vulkan");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/add.comp");
    println!("cargo:rerun-if-changed=kernels/mul.comp");
    println!("cargo:rerun-if-changed=kernels/silu_mul.comp");
    println!("cargo:rerun-if-changed=kernels/rms_norm.comp");
    println!("cargo:rerun-if-changed=kernels/softmax.comp");
    println!("cargo:rerun-if-changed=kernels/embedding.comp");
    println!("cargo:rerun-if-changed=kernels/matmul_tile_32.comp");
    println!("cargo:rerun-if-changed=kernels/matmul_tile_64.comp");
    println!("cargo:rerun-if-changed=kernels/matmul_tile_64_bf16.comp");
    println!("cargo:rerun-if-changed=kernels/qkv_attention.comp");
    println!("cargo:rerun-if-changed=kernels/qkv_attention_swa.comp");
    println!("cargo:rerun-if-changed=kernels/mul_scalar.comp");
    println!("cargo:rerun-if-changed=kernels/sqrt.comp");
    println!("cargo:rerun-if-changed=kernels/recip.comp");
    println!("cargo:rerun-if-changed=kernels/rope.comp");
    println!("cargo:rerun-if-changed=kernels/rope_yarn.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_q4k.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_q5k.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_q6k.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_q8_0.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_iq4nl.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_iq4xs.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_iq3xxs.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_iq3s.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_iq2xxs.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_iq2xs.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_iq2s.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_fp8_e4m3.comp");
    println!("cargo:rerun-if-changed=kernels/fused_dequant_gemm_mxfp4.comp");
    println!("cargo:rerun-if-changed=kernels/kv_dequant_attention.comp");
    println!("cargo:rerun-if-changed=kernels/selective_scan.comp");
    println!("cargo:rerun-if-changed=kernels/qkv_attention_paged.comp");
    println!("cargo:rerun-if-changed=kernels/qkv_attention_paged_swa.comp");
    println!("cargo:rerun-if-changed=kernels/tree_attention.comp");
    println!("cargo:rerun-if-changed=kernels/flash_attention.comp");
    println!("cargo:rerun-if-changed=kernels/silu_mul_backward.comp");
    println!("cargo:rerun-if-changed=kernels/quantized_matmul_backward_dx.comp");
    println!("cargo:rerun-if-changed=kernels/quantized_matmul_backward_dx_q8_0.comp");
    println!("cargo:rerun-if-changed=kernels/quantized_matmul_backward_dx_generic.comp");
    println!("cargo:rerun-if-changed=kernels/rwkv_time_mix.comp");
    println!("cargo:rerun-if-changed=kernels/rwkv_channel_mix.comp");
    println!("cargo:rerun-if-changed=kernels/all_reduce.comp");
    println!("cargo:rerun-if-changed=kernels/comm_fuse_reduce.comp");
    println!("cargo:rerun-if-changed=kernels/quant_q8_0.comp");
    println!("cargo:rerun-if-changed=kernels/quant_fp8.comp");
    println!("cargo:rerun-if-changed=kernels/fused_quant_gemm_q8_0.comp");
    println!("cargo:rerun-if-changed=kernels/fused_quant_gemm_fp8.comp");
    println!("cargo:rerun-if-changed=kernels/moe_fused_dispatch.comp");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));

    let validator =
        std::env::var("GLSLANG_VALIDATOR").unwrap_or_else(|_| "glslangValidator".to_string());

    let mut gen_code = String::new();
    gen_code.push_str("// @generated by build.rs — do not edit.\n");
    gen_code.push_str("// Real SPIR-V blobs compiled from GLSL via glslangValidator.\n\n");

    let mut any_failed = false;
    for (name, glsl) in kernels() {
        let comp_path = out_dir.join(format!("{name}.comp"));
        let spv_path = out_dir.join(format!("{name}.spv"));
        std::fs::write(&comp_path, glsl.as_bytes()).expect("write .comp");

        let status = Command::new(&validator)
            .arg("-V")
            .arg("--target-env")
            .arg("vulkan1.1")
            .arg(&comp_path)
            .arg("-o")
            .arg(&spv_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                gen_code.push_str(&format!(
                    "pub const SPIRV_{}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}.spv\"));\n",
                    sanitize(name),
                    name
                ));
            }
            Ok(s) => {
                eprintln!(
                    "build.rs: glslangValidator failed for kernel `{name}` (status {s}); omitting precompiled blob"
                );
                any_failed = true;
            }
            Err(e) => {
                eprintln!(
                    "build.rs: could not invoke glslangValidator for kernel `{name}`: {e}; omitting precompiled blob"
                );
                any_failed = true;
            }
        }
    }

    let gen_path = out_dir.join("spirv_spv.rs");
    std::fs::write(&gen_path, gen_code).expect("write generated spirv module");

    if any_failed {
        // Surface a clear error so compilation failure is not silently swallowed.
        panic!(
            "build.rs: one or more Vulkan kernels failed to compile to SPIR-V. \
             Ensure `glslangValidator` is installed and on PATH (or set GLSLANG_VALIDATOR)."
        );
    }
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else if c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
