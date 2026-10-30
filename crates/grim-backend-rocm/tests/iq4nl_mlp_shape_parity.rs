//! IQ4_NL matmul parity at Gemma 4's MLP shape.
//!
//! The Gemma-4 E4B checkpoint stores most of its weights as IQ4_NL
//! (`attn_q/k/o`, `ffn_gate/up/down`). This gate pins the device GEMM to the
//! HOST one - `grim_quant::gemm_iq4nl_packed`, the path the CPU forward takes
//! and the one that matches llama.cpp byte for byte on the E4B.
//!
//! The error metric is the trap here, and the reason this file exists. The
//! device and the host sum the K terms in different orders, so the difference
//! is F32 ACCUMULATION NOISE relative to the magnitude of the SUM (up to
//! ~1e5 at K=10240) - not relative to each output's own value. Normalising
//! by the output's own value turns a CANCELLING output (whose true value is
//! ~0.03 while the sum's scale is ~4e5) into a "92% error" that is really
//! 8e-8 of the sum's scale. Measured with the metric that matches the
//! arithmetic: the mean relative error is 1e-8..4e-6 and the worst is 6e-5.
//!
//! So the device IQ4_NL GEMM is CORRECT. This gate is the regression fence
//! that keeps it so, and it covers the shapes the E4B actually uses.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device. Run it with
//! `HIP_VISIBLE_DEVICES=1` - card 0 is reserved.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, MemoryOps, QuantOps};
use grim_tensor::{DType, QuantFormat, Shape};
use std::sync::Arc;

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| Arc::new(RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")))
        .ok()
}

/// Run one shape: `m` tokens against an `[n, k]` IQ4_NL weight.
fn check(dev: &RocmDevice, m: usize, n: usize, k: usize, label: &str) -> f32 {
    // A deterministic weight spanning the range IQ4_NL encodes.
    let w_host: Vec<f32> = (0..n * k)
        .map(|i| {
            let t = i as f32;
            (t * 0.0013).sin() * 0.7 + (t * 0.011).cos() * 0.2
        })
        .collect();
    let w_packed = grim_quant::quant_iq4nl(&w_host).expect("quant_iq4nl");
    // The host dequant, kept as a second opinion on the packed bytes.
    let w_ref = grim_quant::dequant_iq4nl(&w_packed, n * k).expect("dequant_iq4nl");
    assert_eq!(w_ref.len(), n * k);

    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.0021 - 3.0).collect();
    let a_dev = dev
        .from_cpu(
            &a_host,
            &Shape::new(vec![m, k]),
            DType {
                arith: grim_tensor::dtype::ArithType::F32,
                storage: grim_tensor::dtype::Storage::Native,
            },
        )
        .expect("upload activation");
    let w_dev = dev
        .from_cpu_bytes(
            &w_packed,
            &Shape::new(vec![n, k]),
            DType {
                arith: grim_tensor::dtype::ArithType::F32,
                storage: grim_tensor::dtype::Storage::KQuant(
                    grim_tensor::dtype::KQuantScheme::IQ4NL,
                ),
            },
        )
        .expect("upload packed IQ4_NL weight");

    let out_shape = Shape::new(vec![m, n]);
    let (out, _handle) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            w_dev.as_ref(),
            &[],
            QuantFormat::Iq4Nl,
            &out_shape,
        )
        .expect("device IQ4_NL matmul");
    let got = out.to_cpu_vec_f32().expect("read output");
    assert_eq!(got.len(), m * n);

    // Host reference: the CPU's OWN packed GEMM - the path the CPU forward
    // takes, and the one that matches llama.cpp byte for byte on the E4B.
    let want_all =
        grim_quant::gemm_iq4nl_packed(&a_host, &w_packed, m, n, k).expect("host gemm_iq4nl_packed");
    // The error is F32 ACCUMULATION noise over K terms, so it scales with the
    // magnitude of the SUM, not with each output's own value. Normalising by
    // the output's own value turns a cancelling output (whose true value is
    // ~0 while the sum's scale is ~1e5) into a 90% "error" that is really
    // 1e-7 of the sum's scale. The scale is therefore the LARGEST |want| in
    // the output set.
    let scale = want_all
        .iter()
        .fold(0.0f32, |acc, w| acc.max(w.abs()))
        .max(1.0);
    let mut worst_rel = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    let mut sum_rel = 0.0f64;
    for r in 0..m {
        for c in 0..n {
            let want = want_all[r * n + c];
            let rel = ((got[r * n + c] - want).abs() / scale) as f32;
            sum_rel += rel as f64;
            if rel > worst_rel {
                worst_rel = rel;
                worst_at = (r, c);
            }
        }
    }
    let mean_rel = sum_rel / (m * n) as f64;
    eprintln!("[iq4nl-mlp] {label}: MEAN relative error {mean_rel:.8}");
    // The worst few, with the RAW values: a near-zero `want` with a small
    // absolute error is cancellation, not a wrong decode.
    let mut all: Vec<(f32, f32, usize, usize)> = Vec::new();
    for r in 0..m {
        for c in 0..n {
            let want = want_all[r * n + c];
            let have = got[r * n + c];
            all.push(((have - want).abs(), want, r, c));
        }
    }
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (abs, want, r, c) in all.iter().take(4) {
        eprintln!(
            "[iq4nl-mlp] {label}: worst abs {abs:.6} want {want:.6} got {:.6} at row {r} col {c}",
            got[r * n + c]
        );
    }
    eprintln!(
        "[iq4nl-mlp] {label} (m={m} n={n} k={k}): worst relative error {worst_rel:.6} \
         at row {} col {}",
        worst_at.0, worst_at.1
    );
    worst_rel
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn iq4nl_matmul_matches_host_at_the_mlp_shape() {
    let Some(dev) = gpu_device() else { return };

    // Gemma 4 E4B, block 0: the FFN down projection is [2560, 10240] and the
    // block's prefill is 6 tokens. The same GEMM at a decode-shaped M, and
    // the gate/up shape, for contrast.
    let shapes = [
        (1usize, 4usize, 32usize, "one block per row"),
        (1, 4, 64, "two blocks per row"),
        (1, 4, 10240, "320 blocks per row"),
        (1, 2560, 10240, "ffn_down, 1-token decode"),
        (6, 2560, 10240, "ffn_down, 6-token prefill"),
        (64, 2560, 10240, "ffn_down, 64-token prefill"),
    ];
    let mut worst = 0.0f32;
    let mut worst_label = String::new();
    for (m, n, k, label) in shapes {
        let rel = check(&dev, m, n, k, label);
        if rel > worst {
            worst = rel;
            worst_label = label.to_string();
        }
    }
    // F32 accumulation over K terms: the error is relative to the SUM's
    // scale, and the device and host sum in different orders. The bound is
    // the F32 epsilon (1.2e-7) times sqrt(K) (~100 at K=10240), with room
    // for the different summation order.
    let bound = 1e-4f32;
    assert!(
        worst < bound,
        "device IQ4_NL matmul diverges from the host by {worst} (bound {bound}) at {worst_label}"
    );
}
