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
        return None;
    }
    grim_backend_rocm::require_gpu_device().map(std::sync::Arc::new)
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

    // ── THE DISCRIMINATOR: the device route quantizes ACTIVATIONS to Q8_1
    // by design (the dot4 A operand is packed q8_1; grim_quantize_q8_1 runs
    // first in the dispatch). The host reference above multiplies RAW f32
    // activations, so device-vs-host includes that BY-DESIGN quantization
    // error, amplified by sqrt(K) — the old fixed 1e-4 bound was written
    // pre-Q8_1-route and misreads design error as decode error.
    //
    // GATE A (decode algebra): mirror the device path host-side — quantize
    // a_host to the same Q8_1 grid (d = amax/127, qi = round(a/d)), then run
    // the SAME host GEMM over the dequantized activations. Device vs mirror
    // now differs only in dot order + weight decode → tight bound.
    let mut a_q81 = vec![0.0f32; a_host.len()];
    let mut q81_max_err = 0.0f32;
    for blk_start in (0..a_host.len()).step_by(32) {
        let blk_end = (blk_start + 32).min(a_host.len());
        let amax = a_host[blk_start..blk_end]
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        // The device STORES the Q8_1 scale as f16 and reads it back f16->f32
        // (fp16_to_float_device on a_blk[0..2]) — mirror that rounding, or the
        // f16 rounding of d shows up as a per-term error in the residual.
        let d_h = half::f16::from_f32(d);
        let d_dev = f32::from(d_h);
        for (i, &v) in a_host[blk_start..blk_end].iter().enumerate() {
            let qi = if d > 1e-9 {
                (v / d).round().clamp(-127.0, 127.0)
            } else {
                0.0
            };
            let deq = qi * d_dev;
            q81_max_err = q81_max_err.max((v - deq).abs());
            a_q81[blk_start + i] = deq;
        }
    }
    let mirror_out = grim_quant::gemm_iq4nl_packed(&a_q81, &w_packed, m, n, k)
        .expect("host gemm over q8_1 activations");
    let scale_mirror = mirror_out.iter().fold(0.0f32, |acc, w| acc.max(w.abs())).max(1.0);
    let mut worst_dev_vs_mirror = 0.0f32;
    for r in 0..m {
        for c in 0..n {
            let rel = ((got[r * n + c] - mirror_out[r * n + c]).abs() / scale_mirror) as f32;
            worst_dev_vs_mirror = worst_dev_vs_mirror.max(rel);
        }
    }
    eprintln!(
        "[iq4nl-mlp] {label}: device vs Q8_1-mirrored host (decode+order only): {worst_dev_vs_mirror:.6}"
    );
    // Gate A bound: device-vs-mirror residual = dot ORDER noise (f32
    // accumulation over K terms in different orders). The random-walk bound
    // is eps32 * sqrt(K) * (per-term magnitude = max|a_q81| x max|w_ref|);
    // a real decode error (wrong nibble/table) exceeds this ~1000x.
    let a_q81_max = a_q81.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let w_max_ref = w_ref.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let gate_a = 4.0f32 * (1.2e-7 * (k as f32).sqrt() + 1e-6) * a_q81_max * w_max_ref;
    assert!(
        worst_dev_vs_mirror < gate_a,
        "{label}: device diverges from the Q8_1-mirrored host by {worst_dev_vs_mirror} \
         (bound {gate_a:.3e}) — REAL decode/dot error beyond order noise"
    );

    // GATE B (accuracy statement): device vs RAW-f64 oracle is bounded by the
    // Q8_1 activation quantization the mirror absorbed — EXACTLY, per output:
    // |Σ a_i×w_i − Σ a'_i×w_i| <= Σ |Δa_i × w_i| (triangle inequality, Δa =
    // a − a_q81). The a_host ramp makes per-block amax grow with i, so the
    // sqrt(K) uniform-error model underestimates; the exact term sum does not.
    // Normalized by the same max|f64| the host comparison uses.
    let w_f64: Vec<f64> = w_ref.iter().map(|&v| v as f64).collect();
    let mut worst_dev_vs_f64 = 0.0f64;
    let mut worst_q81_ratio = 0.0f64;
    let mut scale_f64 = 0.0f64;
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            let mut q81_abs = 0.0f64;
            for i in 0..k {
                let a = a_host[r * k + i] as f64;
                let aq = a_q81[r * k + i] as f64;
                let w = w_f64[c * k + i];
                acc += a * w;
                q81_abs += (a - aq).abs() * w.abs();
            }
            scale_f64 = scale_f64.max(acc.abs());
            let d_dev = (got[r * n + c] as f64 - acc).abs();
            worst_dev_vs_f64 = worst_dev_vs_f64.max(d_dev);
            // The ratio must hold even for cancelling outputs: normalize the
            // residual by the row's OWN q81 term bound, floored by 1e-3 of the
            // set scale (a pure-zero row can't be ratio-bounded).
            let denom = q81_abs.max(scale_f64 * 1e-3);
            worst_q81_ratio = worst_q81_ratio.max(d_dev / denom);
        }
    }
    let rel_dev_vs_f64 = worst_dev_vs_f64 / scale_f64.max(1.0);
    eprintln!(
        "[iq4nl-mlp] {label}: device vs raw-f64 {rel_dev_vs_f64:.3e} (normalized); worst residual / own Q8_1 bound {worst_q81_ratio:.3} (q81_max_err {q81_max_err:.4})"
    );
    assert!(
        worst_q81_ratio <= 1.05,
        "{label}: device residual exceeds its own Q8_1 activation-quantization bound \
         (ratio {worst_q81_ratio:.3} > 1.05) — REAL decode error beyond activation quantization"
    );
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
    eprintln!(
        "[iq4nl-mlp] worst same-order-vs-host relative error across shapes: {worst:.6} at {worst_label} (the f64-oracle gate in check() is the decode-error discriminator)"
    );
}
