//! SPEED-DOT MoE parity: the dot4/sudot4 grouped kernels
//! (`grim_moe_fused_grouped_{q80,w8a8_int8,q4k}_dot4`) against the
//! already-golden scalar grouped kernels on identical packed weight bytes.
//!
//! The scalar kernels decode the quant codes to f32 and accumulate in fp32;
//! the dot4 kernels quantize the activations to Q8_1 in-thread and contract
//! via sdot4 (RDNA2) / sudot4 (RDNA3/4). Same weight bytes in, so any delta
//! beyond activation-quantization error is a dot4-path bug (nibble unpack,
//! scale index, two-dot decomposition, SiLU wiring).
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm \
//!   --test charon_dot4_grouped_parity

use grim_backend_rocm::kernels::charon::{
    dot4_entry_for, grouped_dot4_entry, CharonDot4Quant, RoutingAssignment,
};
use grim_backend_rocm::RocmDevice;
use std::panic;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

const BATCH: usize = 2;
const HIDDEN: usize = 32; // dot4 path requires multiples of 32
const INTER: usize = 64;
const NUM_EXPERTS: usize = 2;
const RSF: f32 = 1.0;

fn assignment() -> RoutingAssignment {
    // token 0 -> experts {0,1}, token 1 -> expert 0.
    RoutingAssignment {
        tokens: vec![0, 0, 1],
        experts: vec![0, 1, 0],
        weights: vec![0.7, 0.3, 0.5],
    }
}

fn activations() -> Vec<f32> {
    (0..BATCH * HIDDEN)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.11)
        .collect()
}

fn weights_f32() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mk = |n: usize, salt: f32| -> Vec<f32> {
        (0..NUM_EXPERTS * n)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.07 * salt)
            .collect()
    };
    (mk(HIDDEN * INTER, 1.0), mk(HIDDEN * INTER, 1.3), mk(INTER * HIDDEN, 0.8))
}

/// Pack f32 weights to GGUF Q8_0 (f16 scale + 32 i8 codes per block).
fn quant_q80(w: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; w.len() / 32 * 34];
    for (blk, chunk) in w.chunks(32).enumerate() {
        let amax = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = (amax / 127.0).max(1e-30);
        let bits = half::f16::from_f32(d).to_bits();
        let o = &mut out[blk * 34..blk * 34 + 34];
        o[0] = (bits & 0xFF) as u8;
        o[1] = ((bits >> 8) & 0xFF) as u8;
        for (j, &v) in chunk.iter().enumerate() {
            o[2 + j] = ((v / d).round().clamp(-127.0, 127.0) as i8) as u8;
        }
    }
    out
}

/// Pack f32 weights to Q4_K (144B per 256 weights). Simplified encoder:
/// d = dmin = 1.0 (f16), value = d*sc*q - dmin*m with q in [0,15], sc/m in
/// the 6-bit GGUF nibble-packing (mirrors the kernel's decode formulas).
fn quant_q4k(w: &[f32]) -> Vec<u8> {
    assert_eq!(w.len() % 256, 0);
    let mut out = vec![0u8; w.len() / 256 * 144];
    for (sb, superchunk) in w.chunks(256).enumerate() {
        let o = &mut out[sb * 144..sb * 144 + 144];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        o[0] = (d_bits & 0xFF) as u8;
        o[1] = ((d_bits >> 8) & 0xFF) as u8;
        o[2] = (d_bits & 0xFF) as u8; // dmin = 1.0
        o[3] = ((d_bits >> 8) & 0xFF) as u8;

        // Per-sub-block sc (6 bit) / m (6 bit).
        let mut sc_m = [(0u8, 0u8); 8];
        for (is, chunk) in superchunk.chunks(32).enumerate() {
            let wmin = chunk.iter().copied().fold(f32::INFINITY, f32::min);
            let wmax = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sc = (((wmax - wmin) / 15.0).ceil() as i32).clamp(1, 63) as u8;
            let m = ((-wmin / sc as f32).round() as i32).clamp(0, 63) as u8;
            sc_m[is] = (sc, m);
        }
        // GGUF scales packing (12 bytes): sub-blocks 0..3 → scales[is]=sc,
        // scales[is+4]=m; sub-blocks 4..7 → scales[is+4] holds sc low nibble
        // (bits 0-3) + m bits 0-3 (bits 4-7), scales[is-4] bits 6-7 = sc
        // bits 4-5, scales[is] bits 6-7 = m bits 4-5.
        for is in 0..8 {
            let (sc, m) = sc_m[is];
            if is < 4 {
                o[4 + is] = sc & 63;
                o[4 + is + 4] = m & 63;
            } else {
                o[4 + is + 4] = (sc & 0x0F) | ((m & 0x0F) << 4);
                o[4 + is - 4] = (o[4 + is - 4] & 0x3F) | ((sc >> 4) << 6);
                o[4 + is] = (o[4 + is] & 0x3F) | ((m >> 4) << 6);
            }
        }
        for (is, chunk) in superchunk.chunks(32).enumerate() {
            let (_, m) = sc_m[is];
            let group = is / 2;
            let half = is % 2;
            for (j, &v) in chunk.iter().enumerate() {
                let q = ((v + m as f32) / sc_m[is].0 as f32).round().clamp(0.0, 15.0) as u8;
                let idx = 16 + group * 32 + j;
                if half == 0 {
                    o[idx] |= q & 0x0F;
                } else {
                    o[idx] |= q << 4;
                }
            }
        }
    }
    out
}

/// Pack f32 weights to CompressedTensors W8A8 int8 (8B prefix + codes + f32 row scales).
fn quant_w8a8(rows: usize, cols: usize, w: &[f32]) -> Vec<u8> {
    let num_experts = w.len() / (rows * cols);
    let stride = 8 + rows * cols + rows * 4;
    let mut out = vec![0u8; num_experts * stride];
    for e in 0..num_experts {
        let base = e * rows * cols;
        let o = &mut out[e * stride..(e + 1) * stride];
        for r in 0..rows {
            let row = &w[base + r * cols..base + (r + 1) * cols];
            let amax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = (amax / 127.0).max(1e-30);
            o[8 + rows * cols + r * 4..8 + rows * cols + r * 4 + 4]
                .copy_from_slice(&d.to_le_bytes());
            for (c, &v) in row.iter().enumerate() {
                o[8 + r * cols + c] = ((v / d).round().clamp(-127.0, 127.0) as i8) as u8;
            }
        }
    }
    out
}

fn max_scored(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut worst_rel = 0.0f32;
    let mut worst_abs = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        worst_abs = worst_abs.max((x - y).abs());
        worst_rel = worst_rel.max((x - y).abs() / (y.abs() + 1e-3));
    }
    (worst_abs, worst_rel)
}

#[test]
fn dot4_q80_matches_scalar_grouped() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let (gw, uw, dw) = weights_f32();
    let (gw_q, uw_q, dw_q) = (quant_q80(&gw), quant_q80(&uw), quant_q80(&dw));
    let a_scale = vec![1.0f32; BATCH];
    let asg = assignment();
    let x = activations();

    let scalar = dev
        .charon_grouped_dispatch_roundtrip_q80(
            &x, &gw_q, &uw_q, &dw_q, &a_scale, &asg, BATCH, HIDDEN, INTER, RSF,
        )
        .expect("scalar q80 roundtrip");
    let dot4 = dev
        .charon_grouped_dispatch_roundtrip_dot4(
            dot4_entry_for(CharonDot4Quant::Q8_0, dev.gcn_arch(), false).expect("rdna dot4"),
            &x, &gw_q, &uw_q, &dw_q, &a_scale, &asg, BATCH, HIDDEN, INTER, NUM_EXPERTS, RSF,
        )
        .expect("dot4 q80 roundtrip");

    let (abs, rel) = max_scored(&dot4, &scalar);
    assert!(
        abs <= 1e-2 || rel <= 5e-2,
        "dot4 q80 vs scalar mismatch: abs={abs:.4e} rel={rel:.4e}\ndot4: {dot4:?}\nscalar: {scalar:?}"
    );
}

#[test]
fn dot4_q4k_matches_scalar_iqk() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let (gw, uw, dw) = weights_f32();
    let (gw_q, uw_q, dw_q) = (quant_q4k(&gw), quant_q4k(&uw), quant_q4k(&dw));
    let a_scale = vec![1.0f32; BATCH];
    let asg = assignment();
    let x = activations();

    // The scalar IQK kernel (format 7) is the dequant oracle — it is
    // golden-tested against the CPU dequant reference in golden_charon_moe_gpu.
    let scalar = dev
        .charon_grouped_dispatch_roundtrip_iqk(
            &x, &gw_q, &uw_q, &dw_q, &a_scale, &asg, BATCH, HIDDEN, INTER, 7, 144, RSF,
        )
        .expect("scalar q4k roundtrip");
    let dot4 = dev
        .charon_grouped_dispatch_roundtrip_dot4(
            grouped_dot4_entry(CharonDot4Quant::Q4K),
            &x, &gw_q, &uw_q, &dw_q, &a_scale, &asg, BATCH, HIDDEN, INTER, NUM_EXPERTS, RSF,
        )
        .expect("dot4 q4k roundtrip");

    let (abs, rel) = max_scored(&dot4, &scalar);
    assert!(
        abs <= 2e-2 || rel <= 5e-2,
        "dot4 q4k vs scalar mismatch: abs={abs:.4e} rel={rel:.4e}\ndot4: {dot4:?}\nscalar: {scalar:?}"
    );
}

#[test]
fn dot4_w8a8_int8_matches_cpu_reference() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let (gw, uw, dw) = weights_f32();
    let gw_q = quant_w8a8(INTER, HIDDEN, &gw);
    let uw_q = quant_w8a8(INTER, HIDDEN, &uw);
    let dw_q = quant_w8a8(HIDDEN, INTER, &dw);
    let a_scale = vec![1.0f32; BATCH];
    let asg = assignment();
    let x = activations();

    let dot4 = dev
        .charon_grouped_dispatch_roundtrip_dot4(
            grouped_dot4_entry(CharonDot4Quant::W8A8Int8),
            &x, &gw_q, &uw_q, &dw_q, &a_scale, &asg, BATCH, HIDDEN, INTER, NUM_EXPERTS, RSF,
        )
        .expect("dot4 int8 roundtrip");

    // CPU oracle: dequant codes (same bytes the kernel reads), fused
    // gate/up -> SiLU -> down per routed (token, expert) pair.
    let dequant_experts = |buf: &[u8], rows: usize, cols: usize| -> Vec<Vec<Vec<f32>>> {
        let stride = 8 + rows * cols + rows * 4;
        (0..NUM_EXPERTS)
            .map(|e| {
                let b = &buf[e * stride..(e + 1) * stride];
                (0..rows)
                    .map(|r| {
                        let s = f32::from_le_bytes(
                            b[8 + rows * cols + r * 4..8 + rows * cols + r * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                        (0..cols)
                            .map(|c| (b[8 + r * cols + c] as i8) as f32 * s)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let g = dequant_experts(&gw_q, INTER, HIDDEN);
    let u = dequant_experts(&uw_q, INTER, HIDDEN);
    let d = dequant_experts(&dw_q, HIDDEN, INTER);

    let mut cpu = vec![0.0f32; BATCH * HIDDEN];
    for p in 0..asg.num_pairs() {
        let (tok, e, w) = (asg.tokens[p] as usize, asg.experts[p] as usize, asg.weights[p]);
        let a = &x[tok * HIDDEN..(tok + 1) * HIDDEN];
        for h in 0..HIDDEN {
            let mut acc = 0.0f32;
            for j in 0..INTER {
                let gate: f32 = g[e][j].iter().zip(a).map(|(&wv, &av)| wv * av).sum();
                let up: f32 = u[e][j].iter().zip(a).map(|(&wv, &av)| wv * av).sum();
                acc += d[e][h][j] * (gate / (1.0 + (-gate).exp())) * up;
            }
            cpu[tok * HIDDEN + h] += RSF * w * acc;
        }
    }

    let (abs, rel) = max_scored(&dot4, &cpu);
    assert!(
        abs <= 1e-2 || rel <= 5e-2,
        "dot4 w8a8 vs CPU mismatch: abs={abs:.4e} rel={rel:.4e}\ndot4: {dot4:?}\ncpu: {cpu:?}"
    );
}

#[test]
fn dot4_rejects_unaligned_shape() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let (gw, uw, dw) = weights_f32();
    let err = dev.charon_grouped_dispatch_roundtrip_dot4(
        grouped_dot4_entry(CharonDot4Quant::Q8_0),
        &activations(),
        &quant_q80(&gw),
        &quant_q80(&uw),
        &quant_q80(&dw),
        &[1.0; BATCH],
        &assignment(),
        BATCH,
        8,  // hidden not a multiple of 32
        64,
        NUM_EXPERTS,
        RSF,
    );
    assert!(err.is_err(), "unaligned hidden must be rejected, not mis-computed");
}
