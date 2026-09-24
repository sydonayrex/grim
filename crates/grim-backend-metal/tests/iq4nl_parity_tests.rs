//! IQ4NL cross-backend parity anchor for Metal (Lapse J).
//!
//! Validates that Metal's IQ4NL dequant host path produces bit-identical
//! output to the canonical CPU decoder in `grim_quant::dequant_iq4nl`.
//!
//! Known cross-backend context (from ROCm parity_cpu_rocm.rs:267-274):
//! ROCm GPU vs CPU oracle shows up to ~0.0374 max |diff| for IQ4NL.
//! Metal currently targets bit-identity with the CPU oracle (same KVALUES
//! table, same super-block layout, same scale semantics), so any non-zero
//! max_error here is a regression to investigate. If Metal diverges from
//! the CPU oracle, record the max_error here for parity tracking.

use grim_backend_metal::MetalDevice;
use grim_quant::dequant_iq4nl;

const BLOCK_BYTES: usize = 170;
const SUPER: usize = 256;

#[test]
fn test_metal_iq4nl_dequant_parity_vs_cpu() {
    // Step 1: generate deterministic weights, quantize on CPU, dequant on Metal, compare.
    let n_blocks = 4;
    let n_weights = n_blocks * SUPER;
    let mut bytes = Vec::with_capacity(n_blocks * BLOCK_BYTES);

    // Deterministic byte pattern so the test is reproducible.
    for i in 0..(n_blocks * BLOCK_BYTES) {
        bytes.push((i.wrapping_mul(131) % 256) as u8);
    }

    // CPU oracle.
    let expected = dequant_iq4nl(&bytes, n_weights).expect("CPU dequant_iq4nl failed");

    // Metal host dequant path (runs on CPU when no Apple GPU is present;
    // the kernel path is gated behind target_vendor = "apple").
    let dev = MetalDevice::new(0).expect("MetalDevice::new(0) should succeed");
    let got = dev
        .dequantize_iq4nl_host(&bytes, n_weights)
        .expect("Metal dequantize_iq4nl_host failed");

    // Shape agreement.
    assert_eq!(
        got.len(),
        n_weights,
        "Metal dequant produced wrong element count"
    );
    assert_eq!(
        expected.len(),
        n_weights,
        "CPU oracle produced wrong element count"
    );

    // Numerical parity: Metal host path should be bit-identical to the CPU
    // oracle because both share the same KVALUES_IQ4NL table and super-block
    // layout from grim_quant. Any deviation is a regression.
    let mut max_err = 0.0f32;
    let mut max_err_idx = 0usize;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        if err > max_err {
            max_err = err;
            max_err_idx = i;
        }
    }

    eprintln!(
        "[iq4nl_parity] n_blocks={} n_weights={} max|gpu-cpu|={} at idx={}",
        n_blocks, n_weights, max_err, max_err_idx
    );

    // Metal's host dequant path uses the same constants as the CPU oracle,
    // so we expect bit-identical output (max_err == 0.0). We assert <= 0.0
    // to be explicit: any non-zero error is a failure.
    assert!(
        max_err <= 0.0,
        "IQ4NL Metal host dequant diverged from CPU oracle by max_err={max_err} at idx={max_err_idx}"
    );
}

#[test]
fn test_metal_iq4nl_kernel_and_dispatch() {
    // Sanity-check the MSL kernel + pipeline + dispatch + host-fallback wiring.
    let msl_source = include_str!("../src/kernels.msl");
    assert!(
        msl_source.contains("grim_dequant_iq4nl"),
        "MSL shader source must contain kernel declaration for grim_dequant_iq4nl"
    );

    let lib_source = include_str!("../src/lib.rs");
    assert!(
        lib_source.contains("dequant_iq4nl: Retained"),
        "MetalPipelines must contain dequant_iq4nl pipeline field"
    );
    assert!(
        lib_source.contains("get_pipeline(\"grim_dequant_iq4nl\")"),
        "MetalPipelines init must load grim_dequant_iq4nl pipeline"
    );
    assert!(
        lib_source.contains("\"iq4nl\" => &ctx.pipelines.dequant_iq4nl"),
        "dequant_gate must dispatch iq4nl to the Metal pipeline"
    );
    assert!(
        lib_source.contains("\"iq4nl\" => grim_quant::dequant_iq4nl"),
        "dequant_gate must have an iq4nl host fallback in the match arms"
    );

    // Caps gating: IQ4NL dequant runs on any GPU family capable of compute.
    // Uses gpu_family >= 1 as the gate (any Apple Silicon can run it).
    let caps = MetalDevice::new(0)
        .expect("MetalDevice::new(0) should succeed")
        .caps;
    assert!(
        caps.gpu_family >= 1,
        "IQ4NL dequant requires Apple Silicon GPU (any family)"
    );
}
