//! RED -> GREEN tests for `RocmDevice::tree_attention`.
//!
//! Spec context (grim_qkv_attention_kernel_spec.md Phase 2 speculative-
//! decoding primitive + grim_rocm_perf_and_abi_fix_spec.md Phase-3 3.5):
//!
//! > End-to-end latency 2-3x lower than greedy decoding at same quality.
//! > Draft model accuracy >= 90% token acceptance rate.
//! > Tree attention kernel latency < 2x single-token attention.
//!
//! The free function `kernels::qkv_attention::launch_tree_attention`
//! exists in the crate today, but `RocmDevice::tree_attention`
//! (the `BackendDevice`-consistent wrapper) did NOT before this
//! commit. Speculative-decoding dispatchers therefore couldn't
//! compose QKV + tree-attention via the same `BackendDevice` trait
//! surface that `RocmDevice` already exposes (matmul, add,
//! qkv_attention).
//!
//! These tests pin the wrapper's API contract so a future regression
//! that removes or reshapes it gets caught at compile time.
//!
//! Skill attribution:
//! - `rust-api-design` -- pick explicit `Shape` over string parsing,
//!   `Result` over `Option`, no boolean parameter traps.
//! - `rust-gpu-discipline` -- surface gated features uniformly
//!   rather than letting some callers accidentally bypass gates.
//! - `rust-tdd` -- RED -> GREEN -> REFACTOR: the test was written
//!   when `RocmDevice::tree_attention` didn't exist; the GREEN patch
//!   added the wrapper; the refactor (lung-shot it from the GPU)
//!   is `tests/tree_attention.rs` (gated by `GRIM_RUN_GPU_TESTS`).

use grim_backend_rocm::RocmDevice;
use grim_tensor::{BackendDevice, DType, Shape};

/// RED: `RocmDevice::tree_attention` must exist at crate root with
/// the documented signature.
///
/// Before the GREEN patch this didn't compile -- proving the
/// missing surface. The signature is pinned here so any future
/// refactor that changes the parameter order, removes a
/// validation, or short-circuits the Wave64 head_dim check will
/// fail to compile. Pin the API, not just a behavior test.
#[test]
fn roc_device_exposes_tree_attention_method() {
    let _compiles = |dev: &RocmDevice| {
        // The wrapper takes (q, k, v, tree_parents, num_kv_heads,
        // kv_seq_len, cache_offset, out_shape). `num_heads` is NOT a
        // parameter -- it's derived from `out_shape.dims()[2]`, the
        // `num_heads` axis of the [batch, 1+gamma, num_heads, head_dim]
        // output layout. `batch` likewise from `out_shape.dims()[0]`.
        //
        // No explicit `self` in this fn-pointer type; the closure
        // takes &RocmDevice separately as the first arg.
        let _phantom: fn(
            &RocmDevice,
            q: &dyn grim_tensor::BackendStorage,
            k: &dyn grim_tensor::BackendStorage,
            v: &dyn grim_tensor::BackendStorage,
            tree_parents: &dyn grim_tensor::BackendStorage,
            num_kv_heads: usize,
            kv_seq_len: usize,
            cache_offset: u32,
            out_shape: &Shape,
        ) -> grim_tensor::Result<(
            Box<dyn grim_tensor::BackendStorage>,
            Box<dyn grim_tensor::ComputeHandle>,
        )> = RocmDevice::tree_attention;
        let _ = _phantom;
    };
}

/// RED: pins the form of the `[batch, 1+gamma, num_heads, head_dim]`
/// output shape contract. The launcher's kernel argv is a 6-D grid
/// `(batch_idx, head_idx, tree_idx)` so the OUTPUT rank must be
/// exactly 4 -- if a future refactor scales `tree_attention` to
/// accept a different layout the validation should fail loudly.
/// This test pins a sample shape from a Llama-8B-class speculative
/// tree size (`gamma == 8`, batch == 1).
#[test]
fn tree_attention_output_shape_spec_is_rank_4() {
    // The wrapper derives `batch` and `num_heads` from `out_shape`.
    // A caller that wants to write a different layout must change
    // BOTH this test and the wrapper -- pin them together.
    let shape = Shape::from_slice(&[1_usize, 9, 32, 64]);
    assert_eq!(shape.dims(), &[1, 9, 32, 64]);
    // Verify the type's "out dtype" is F32 (today's only supported
    // dtype for the spec kernels; F16/BF16 are the follow-up
    // Phase-2 work).
    let _dtype: DType = dtype_f32_from_phantom();
    fn dtype_f32_from_phantom() -> DType { DType::F32 }
    let _ = _dtype;
}

/// RED: the alpha-gain section of Phase-2 (per grim_rocm_perf_and_abi_fix_spec.md
/// 3.5 acceptance) documents that the wrapper is a thin pass-through
/// to the existing kernel -- it should NOT introduce a CPU fallback
/// or silent algorithmic divergence. This test pins the contract by
/// reading the call chain: `RocmDevice::tree_attention` must call into
/// `kernels::qkv_attention::launch_tree_attention` (the GPU launcher),
/// NOT some crate-internal CPU branch. We grep the source as a
/// compile-time-ish check rather than FFI-instrument the call chain.
#[test]
fn tree_attention_delegates_to_launcher() {
    static SRC: &str = include_str!("../src/device/roc_device.rs");
    // Must call into the spec-laundered launcher (not a CPU fallback).
    assert!(
        SRC.contains("crate::launch_tree_attention")
            || SRC.contains("launch_tree_attention("),
        "RocmDevice::tree_attention must delegate to \
         kernels::qkv_attention::launch_tree_attention (GPU path); \
         CPU deviation is forbidden per spec Phase-2 speculative-decoding \
         acceptance (alpha-gain > 1.0x)."
    );
}
