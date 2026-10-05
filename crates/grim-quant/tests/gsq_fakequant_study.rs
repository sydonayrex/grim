//! Phase 0' fake-quant study (plans/citycrow-vram-deferral-plan.md): measures
//! what the GSQ-2.25 rung costs on REAL Xing4.0 expert banks, without any
//! serving path. Env-gated on XING_GGUF like xing_dequant_real_parity.
//!
//! Method: dequant the source bank to f32 (itself an IQ3_S/Q4_K decode —
//! DOUBLE-QUANT CAVEAT: these numbers are lower bounds on loss vs bf16
//! originals), apply the committed GSQ quantizer in software, dequant back,
//! and report (a) weight-space error and (b) expert-output error y = W·x for
//! unit-scale Gaussian activations. Captured real activations can replace
//! (b) via CAPTURED_ACTS=<file of f32 LE> when available.
use grim_quant::{dequant_iq3s, dequant_q4k, dequant_gsq_rco_3p5, quantize_gsq_rco_3p5_block_counted, BLOCK_BYTES_Q2_0};
use std::io::{Read, Seek};

fn load_bank(path: &str, name: &str) -> (Vec<f32>, Vec<usize>, String) {
    let f = std::fs::File::open(path).unwrap();
    let gg = grim_format::gguf::read_gguf(&f).expect("read_gguf");
    let t = gg.tensors.iter().find(|t| t.name == name)
        .unwrap_or_else(|| panic!("tensor {name} not found"));
    let mut f = std::fs::File::open(path).unwrap();
    f.seek(std::io::SeekFrom::Start(gg.data_start + t.offset)).unwrap();
    let mut raw = vec![0u8; t.size_bytes as usize];
    f.read_exact(&mut raw).unwrap();
    let n = t.shape().iter().product::<usize>();
    let f32s = match t.dtype {
        grim_format::gguf::GgufDType::IQ3_S => dequant_iq3s(&raw, n).expect("iq3s"),
        grim_format::gguf::GgufDType::Q4K => dequant_q4k(&raw, n).expect("q4k"),
        other => panic!("unexpected source dtype {other:?} for {name}"),
    };
    (f32s, t.shape().to_vec(), format!("{:?}", t.dtype))
}

/// Control: re-quantize the decoded f32 back to the SOURCE scheme. Near-zero
/// error proves the harness measures requantization damage, not harness noise.
fn fakequant_control_q4k(data: &[f32]) -> Vec<f32> {
        // Use the same public path the converter uses for Q4K targets.
    let plan = grim_quant::TensorRewritePlan {
        target: grim_quant::QuantFormat::Q4K,
        shape: vec![data.len()],
        importance: None,
        curvature: None,
    };
    let rt = grim_quant::rewrite_tensor_data(data, &plan).expect("q4k rewrite");
    match rt.target {
        grim_quant::QuantFormat::Q4K => grim_quant::dequant_q4k(&rt.bytes, data.len()).expect("q4k deq"),
        other => panic!("control produced {other:?}"),
    }
}

fn fakequant_gsq(data: &[f32]) -> (Vec<f32>, usize) {
    let mut packed = vec![0u8; data.len().div_ceil(64) * BLOCK_BYTES_Q2_0];
    let mut flushed = 0usize;
    quantize_gsq_rco_3p5_block_counted(data, &mut packed, &mut flushed).expect("gsq pack");
    let back = dequant_gsq_rco_3p5(&packed, data.len()).expect("gsq dequant");
    (back, flushed)
}

fn report(name: &str, src: &str, w: &[f32], q: &[f32], k_dim: usize, n_rows: usize, flushed: usize) {
    let amax = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let sum_abs: f64 = w.iter().zip(q).map(|(a, b)| (a - b).abs() as f64).sum();
    let max_abs: f64 = w.iter().zip(q).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
    let mean_abs = sum_abs / w.len() as f64;
    // Expert-output error on unit-scale Gaussian activations, per row.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut out_max = 0.0f64;
    let mut out_mean_acc = 0.0f64;
    let mut ref_mag_acc = 0.0f64;
    let rows = n_rows.min(64); // sample rows for time
    for r in 0..rows {
        let x: Vec<f32> = (0..k_dim).map(|_| next()).collect();
        let mut ref_y = 0.0f64; let mut q_y = 0.0f64;
        for c in 0..k_dim {
            ref_y += (w[r * k_dim + c] * x[c]) as f64;
            q_y += (q[r * k_dim + c] * x[c]) as f64;
        }
        let e = (ref_y - q_y).abs();
        out_max = out_max.max(e);
        out_mean_acc += e;
        ref_mag_acc += ref_y.abs();
    }
    // The interpretable number: output error RELATIVE to the reference
    // output magnitude, accumulated over the row set.
    let rel_mean = if ref_mag_acc > 0.0 { out_mean_acc / ref_mag_acc } else { 0.0 };
    println!(
        "{name} [{src}]: flushed={flushed} w_mean_err={mean_abs:.3e} w_max_err={max_abs:.3e} \
         (amax={amax:.3}, mean/amax={:.4}) out_rel_err_mean={:.4} out_abs_max={:.3e} (rows={rows})",
        mean_abs / amax as f64, rel_mean, out_max
    );
}

/// Corvid rung study (2026-10-05): can WhiteCrow replace Q4_K and
/// ForestRaven replace Q8_0/F32? Same fake-quant method as the GSQ study.
#[test]
fn corvid_rungs_on_real_banks() {
    let path = match std::env::var("XING_GGUF") {
        Ok(p) => p,
        Err(_) => { eprintln!("[SKIP] XING_GGUF unset"); return; }
    };

    // 1) Q4K -> WhiteCrow: attn_output [4096, 3584], k % 128 == 0.
    let (w, shape, src) = load_bank(&path, "blk.10.attn_output.weight");
    let (n, k) = (shape[0], shape[1]);
    let (qw, sc, zr) = grim_quant::quant_ostquant_w4_group128(&w, n, k).expect("wc pack");
    let wc = grim_quant::dequant_ostquant_w4a4(&qw, &sc, &zr, &[n, k], 128).expect("wc deq");
    report("blk.10.attn_output Q4K->WhiteCrow", &src, &w, &wc, k, n, 0);

    // 2) IQ3_S-decoded f32 -> ForestRaven (per-row absmax INT8): the
    //    Q8_0/F32-runG question answered on real weight distributions.
    let (w2, shape2, src2) = load_bank(&path, "blk.10.attn_q_a.weight");
    let (n2, k2) = (shape2[0], shape2[1]);
    let (codes, scales) = grim_quant::quant_forest_per_channel(&w2, n2, k2).expect("fr pack");
    // decode: framed blob = per-row absmax INT8; dequant via the quantizer's
    // own inverse: y = scale[row] * code.
    let mut fr = vec![0.0f32; w2.len()];
    for r in 0..n2 {
        let scale = f32::from_le_bytes(scales[r*4..r*4+4].try_into().unwrap());
        for c in 0..k2 {
            fr[r * k2 + c] = scale * (codes[r * k2 + c] as i8 as f32);
        }
    }
    report("blk.10.attn_q_a IQ3S->ForestRaven", &src2, &w2, &fr, k2, n2, 0);

    // 3) F32 norm -> per-tensor absmax INT8 (the user's "ForestRaven does
    //    F32"): output_norm.weight [3584], one scale for the whole vector.
    let f = std::fs::File::open(&path).unwrap();
    let gg = grim_format::gguf::read_gguf(&f).unwrap();
    let t = gg.tensors.iter().find(|t| t.name == "output_norm.weight").unwrap();
    let mut f2 = std::fs::File::open(&path).unwrap();
    f2.seek(std::io::SeekFrom::Start(gg.data_start + t.offset)).unwrap();
    let mut raw = vec![0u8; t.size_bytes as usize];
    f2.read_exact(&mut raw).unwrap();
    let norm: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let amax = norm.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = amax / 127.0;
    let q: Vec<f32> = norm.iter().map(|v| (v / scale).round() * scale).collect();
    let max_e: f32 = norm.iter().zip(&q).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let mean_e: f64 = norm.iter().zip(&q).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / norm.len() as f64;
    println!(
        "output_norm F32->INT8-per-tensor: max_rel={:.4} mean_rel={:.4} (values are elementwise-read, not GEMV)",
        max_e / amax, mean_e / (amax as f64)
    );
}

#[test]
fn gsq_fakequant_on_real_expert_banks() {
    let path = match std::env::var("XING_GGUF") {
        Ok(p) => p,
        Err(_) => { eprintln!("[SKIP] XING_GGUF unset"); return; }
    };
    // One early dense-transition expert tensor (Q4K source) and one MoE
    // expert tensor (IQ3_S source), plus an attention projection.
    let cases = [
        ("blk.2.ffn_down_exps.weight", 3584usize, 1024usize), // [k, n, 64] -> rows of k? see shape
        ("blk.10.ffn_gate_exps.weight", 3584, 1024),
        ("blk.10.attn_q_a.weight", 3584, 768),
    ];
    for (name, k_dim, n_rows) in cases {
        let (w, shape, src) = load_bank(&path, name);
        let (q, flushed) = fakequant_gsq(&w);
        // The bank is [a, b, experts] or [a, b]; the row stride for the
        // output-error probe is the LAST dim (contiguous k).
        let k_actual = *shape.last().unwrap();
        report(name, &src, &w, &q, k_actual, w.len() / k_actual, flushed);
        if src.contains("Q4K") {
            let ctrl = fakequant_control_q4k(&w);
            report(&format!("{name} CONTROL(Q4K->Q4K)"), &src, &w, &ctrl, k_actual, w.len() / k_actual, 0);
        }
        let _ = (k_dim, n_rows);
    }
}
