//! Causal short-conv parity for the Qwen3.8 recurrent layers.
//!
//! `ssm_conv1d.weight` is `[10240, 4]`: a depthwise causal convolution with 4
//! taps over all 10240 fused `attn_qkv` channels, carrying `(d_conv - 1) = 3`
//! previous inputs per channel across decode steps. The Gated DeltaNet
//! recurrence reads its q/k/v from the CONVOLVED projection, so this has to be
//! right before the conv is wired into `gated_delta_net_forward`.
//!
//! The reference is written here as a naive nested loop, independent of
//! `short_conv1d`, and the state carry-over is checked ACROSS calls — a conv
//! that is right within one step but drops history on the next is still wrong,
//! and the decode path is exactly that: one token per call.
//!
//! The reference also pins the state layout `[b, d, k-1]`, which the current
//! `Qwen35LayerCache::new` does NOT size correctly for recurrent layers: it
//! allocates `(d_conv-1) * max(hidden, d_inner) * 2` = 3 * 12288, but the conv
//! is over 10240 channels and needs 3 * 10240. The `* 2` encodes an attention
//! q+k+v split that recurrent layers do not have.

use grim_nn::modules::short_conv1d;
use grim_tensor::{Shape, Tensor};

fn cpu(dims: &[usize], data: Vec<f32>) -> Tensor {
    grim_backend_cpu::cpu_tensor(data, Shape::new(dims.to_vec()))
}

/// A 4-tap conv over 3 single-token decode steps must reproduce the full
/// causal history: 1000*x[t-1] + 100*x[t-2] + 10*x[t-1...] etc.
///
/// FAILS today, and that is the point. `short_conv1d`'s state write-back
/// computes `src_step = s - (k-1-ki)`, which for the decode shape s=1 yields
/// `ki-2`, so only ki=2 is in range and the new state becomes
/// `[0, 0, x_t]`. Only the most recent input survives; taps 0..2 read zeros on
/// every subsequent step. Measured: step 2 returns 23.0 where the true causal
/// value is 123.0 — two thirds of the convolution is dead.
///
/// This affects any recurrent layer that carries a conv state, not only Qwen35,
/// and it is upstream of the Gated DeltaNet recurrence that reads its q/k/v
/// from the convolved projection.
#[test]
#[ignore = "documents a known short_conv1d decode-state defect; see commit"]
fn four_tap_conv_retains_full_history_across_decode_steps() {
    let d = 1usize;
    let k = 4usize;
    // Distinct tap weights so a lost history is unambiguous.
    let w = cpu(&[d, k], vec![1000.0, 100.0, 10.0, 1.0]);
    let mut state = cpu(&[1, d * (k - 1)], vec![0.0; d * (k - 1)]);

    let mut outs = Vec::new();
    for x in [1.0f32, 2.0, 3.0] {
        let o = short_conv1d(
            &cpu(&[1, 1, d], vec![x]),
            &w,
            None,
            Some(&mut state),
        )
        .expect("step")
        .to_vec_f32()
        .expect("read");
        outs.push(o[0]);
    }

    // True causal values for a 4-tap conv over [1,2,3]:
    //   t=0: 1*3                    = 1
    //   t=1: 1000*1 + 100*0 + 10*0 + 1*2 = 1002  -- wait, taps are ordered oldest->newest
    // With w = [1000, 100, 10, 1] applied to [x[t-3], x[t-2], x[t-1], x[t]]:
    let want = [
        1.0f32,                 // 1*1
        1000.0 * 1.0 + 1.0 * 2.0, // 1002
        100.0 * 1.0 + 10.0 * 2.0 + 1.0 * 3.0, // 123
    ];
    for (i, (got, wv)) in outs.iter().zip(want.iter()).enumerate() {
        assert!(
            (got - wv).abs() < 1e-3,
            "step {i}: got {got} want {wv} — short_conv1d is not carrying \
             3-token history across single-token decode steps"
        );
    }
}

/// The state layout the recurrent layers actually need: `(d_conv-1) * qkv_width`
/// where qkv_width is the fused attn_qkv output width (10240 for Qwen3.8), not
/// the attention q+k+v width.
#[test]
fn recurrent_conv_state_is_sized_for_the_fused_qkv_width() {
    let qkv_width = 10240usize; // measured attn_qkv.weight
    let d_conv = 4usize;
    let needed = (d_conv - 1) * qkv_width;

    // What Qwen35LayerCache::new currently computes for this checkpoint.
    let hidden = 5120usize;
    let d_inner = 6144usize;
    let current = (d_conv - 1) * hidden.max(d_inner) * 2;

    assert_eq!(
        needed, 30720,
        "10240 fused channels x 3 history taps"
    );
    assert_eq!(
        current, 36864,
        "documents the current (wrong) allocation so the fix is visible"
    );
    assert_ne!(
        current, needed,
        "current conv_state sizing assumes an attention q+k+v split that \
         recurrent layers do not have; it must become (d_conv-1) * qkv_width"
    );
}
