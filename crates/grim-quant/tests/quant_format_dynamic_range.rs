//! Do the 2:4-pruned formats actually beat int8 on weights, and under what?
//!
//! The hypothesis, from `old/decode-plan-universal-optimization.md` and repeated in
//! the Corvid plan, is that FP8 (E4M3) should be preferred over int8 for weights
//! because *"per-channel outliers are exactly what flattens under a per-tensor int8
//! scale"*. That claim is specifically about a **per-tensor** int8 scale, and it has
//! never been measured for weights.
//!
//! This measures format error on the host, which is the right level for the question:
//! it is about how much of the weight matrix each format preserves, not about kernel
//! throughput. Three int8 baselines make the comparison honest:
//!
//!   - **per-tensor**  one scale for the whole matrix -- the hypothesis's target, and
//!                     the worst case, since one outlier sets the step size for
//!     everything else
//!   - **per-channel** one scale per output column -- the granularity KV cache
//!                     discussions usually assume
//!   - **per-block**   Q8_0's 32-value block scale -- what grim's ForestRaven
//!     actually implements, and therefore what any ship decision really compares
//!     FP8 against
//!
//! The result that matters is the *ratio* against each baseline, not an absolute
//! pass/fail: the question is whether FP8's dynamic range buys what the hypothesis
//! says, and that only shows up relative to the granularity each baseline uses.

use grim_quant::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};

const K: usize = 1024;
const N: usize = 64;

/// Deterministic LCG, so the fixture cannot drift between runs.
fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

/// K x N row-major weights with `outlier_cols` columns scaled by `gain`.
///
/// The first draft built the fixture twice and indexed the first copy as if it were
/// N x K, leaving the outlier columns placed by row rather than by column. It was
/// harmless -- the result was discarded and rebuilt below -- but keeping it would
/// have been a trap for the next reader.
fn weights(outlier_cols: usize, gain: f32) -> Vec<f32> {
    let mut next = rng(0x243f_6a88_5a30_1234);
    let mut w = vec![0.0f32; K * N];
    for v in w.iter_mut() {
        *v = next() * 0.05;
    }
    if outlier_cols == 0 {
        return w;
    }
    let stride = (N / outlier_cols).max(1);
    for n in (0..N).step_by(stride) {
        for k in 0..K {
            w[k * N + n] *= gain;
        }
    }
    w
}

fn quantize_int8(vals: &[f32], scale: f32) -> Vec<f32> {
    vals.iter().map(|&v| ((v / scale).round().clamp(-127.0, 127.0) as i8) as f32 * scale).collect()
}

/// Round-trip through FP8 E4M3, via the crate's own encoder and decoder.
fn round_trip_fp8(vals: &[f32]) -> Vec<f32> {
    vals.iter().map(|&v| fp8_e4m3_to_f32(f32_to_fp8_e4m3(v))).collect()
}

fn rel_l2(q: &[f32], orig: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in q.iter().zip(orig) {
        let d = *a as f64 - *b as f64;
        num += d * d;
        den += *b as f64 * *b as f64;
    }
    (num / den).sqrt()
}

fn report(label: &str, w: &[f32]) {
    let fp8 = rel_l2(&round_trip_fp8(w), w);

    // Per-tensor: one scale for the entire matrix.
    let amax = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let per_tensor = rel_l2(&quantize_int8(w, amax / 127.0), w);

    // Per-channel: one scale per output column of a K x N matrix.
    let mut per_channel = vec![0.0f32; K * N];
    for n in 0..N {
        let col: Vec<f32> = (0..K).map(|k| w[k * N + n]).collect();
        let a = col.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let s = if a > 0.0 { a / 127.0 } else { 1.0 };
        for (k, v) in col.iter().enumerate() {
            per_channel[k * N + n] = ((v / s).round().clamp(-127.0, 127.0) as i8) as f32 * s;
        }
    }
    let per_channel = rel_l2(&per_channel, w);

    // Per-block: Q8_0's 32-value blocks along K.
    let mut per_block = vec![0.0f32; K * N];
    for n in 0..N {
        for g in 0..K / 32 {
            let block: Vec<f32> = (0..32).map(|j| w[(g * 32 + j) * N + n]).collect();
            let a = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let s = if a > 0.0 { a / 127.0 } else { 1.0 };
            for (j, v) in block.iter().enumerate() {
                per_block[(g * 32 + j) * N + n] =
                    ((v / s).round().clamp(-127.0, 127.0) as i8) as f32 * s;
            }
        }
    }
    let per_block = rel_l2(&per_block, w);

    println!("\n{label}");
    println!("  FP8 E4M3            {fp8:.4e}");
    println!("  int8 per-tensor     {per_tensor:.4e}   fp8/int8 = {:>6.2}x", fp8 / per_tensor);
    println!("  int8 per-channel    {per_channel:.4e}   fp8/int8 = {:>6.2}x", fp8 / per_channel);
    println!("  int8 per-block Q8_0 {per_block:.4e}   fp8/int8 = {:>6.2}x", fp8 / per_block);
}

#[test]
fn the_hypothesis_is_about_per_tensor_scaling_so_measure_every_granularity() {
    for (cols, gain, label) in [
        (0usize, 1.0f32, "no outliers (uniform range)"),
        (N / 16, 16.0, "4 of 64 columns at 16x"),
        (N / 8, 64.0, "8 of 64 columns at 64x"),
        (N / 8, 512.0, "8 of 64 columns at 512x"),
    ] {
        report(label, &weights(cols, gain));
    }
}
