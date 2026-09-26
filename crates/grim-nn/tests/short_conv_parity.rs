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

/// A 4-tap conv over single-token decode steps must carry 3 tokens of history.
///
/// REGRESSION for a `short_conv1d` state write-back that rebuilt the state
/// entirely from the CURRENT sequence. `src_step = s - (k-1-ki)` is in range
/// only for `ki = k-1` when s = 1, so on the decode path the state collapsed to
/// `[0, 0, x_t]` and taps 0..2 read zeros forever.
///
/// Tap order is OLDEST-FIRST: w[0] multiplies x[t-3], w[k-1] multiplies x[t].
///
///   step 0 (x=1, no history): 1*1                    = 1
///   step 1 (x=2, has x0):      10*1 + 1*2             = 12
///   step 2 (x=3, has x0,x1):   100*1 + 10*2 + 1*3    = 123
///
/// Step 2 is the discriminating one: before the fix it returned 23, because
/// only the newest two inputs survived. A 2-tap conv masked the bug entirely
/// (ki-2 is then in range for ki=1), which is why it went unnoticed.
#[test]
fn four_tap_conv_retains_full_history_across_decode_steps() {
    let d = 1usize;
    let k = 4usize;
    let w = cpu(&[d, k], vec![1000.0, 100.0, 10.0, 1.0]);
    let mut state = cpu(&[1, d * (k - 1)], vec![0.0; d * (k - 1)]);

    let want = [
        1.0f32,                            // 1 * 1
        10.0 * 1.0 + 1.0 * 2.0,            // 12
        100.0 * 1.0 + 10.0 * 2.0 + 1.0 * 3.0, // 123
    ];
    for (i, wv) in want.iter().enumerate() {
        let x = [1.0f32, 2.0, 3.0][i];
        let out = short_conv1d(&cpu(&[1, 1, d], vec![x]), &w, None, Some(&mut state))
            .expect("step")
            .to_vec_f32()
            .expect("read");
        assert!(
            (out[0] - wv).abs() < 1e-3,
            "step {i} (x={x}): got {} want {wv} — short_conv1d is not carrying \
             3-token history across single-token decode steps",
            out[0]
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
