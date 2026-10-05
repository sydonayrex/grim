//! Q2_0 (GGUF tag 42) ROCm launchers. [see: `q2_0_gemm`]
//!
//! Its own module rather than a arm in `dequant_k_quants.rs`, because that file
//! is shared with the `q2k`/`q3k` super-block family (84/110 bytes per 256)
//! and Q2_0 is a different geometry entirely (18 bytes per 64). Folding it in
//! would put a 64-element-block format among 256-element ones, which is the
//! same class of mistake the tiled table documents for `iq4_nl`.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;

/// Weights per Q2_0 block.
const Q2_0_BLOCK: usize = 64;

/// Bytes per Q2_0 block: `[fp16 d][16 B of 2-bit codes]`.
const Q2_0_BYTES: usize = 18;

/// Packed byte size of a `[n, k]` Q2_0 weight.
///
/// Errors rather than rounding up: a caller sizing an upload from this must not
/// be told a partial trailing block occupies a whole one, because the kernel
/// would then be pointed at the wrong byte count for the last row.
fn q2_0_packed_bytes(k: usize) -> Result<usize> {
    if !q2_0_accepts_k(k) {
        return Err(Error::Backend(format!(
            "q2_0: K={k} is not a non-zero multiple of the {Q2_0_BLOCK}-weight \
             Q2_0 block"
        )));
    }
    Ok((k / Q2_0_BLOCK) * Q2_0_BYTES)
}

/// Is `k` a whole number of Q2_0 blocks, and non-empty?
///
/// The load-bearing precondition for this kernel: its weight row stride is
/// `(k / Q2_0_BLOCK) * Q2_0_BYTES`, which silently truncates on a partial
/// trailing block and reads past the last row -- a memory-safety fault, not an
/// arithmetic error, so nothing downstream would catch it.
///
/// Split out as a free function so it can be tested without a GPU: reaching it
/// through `quantized_matmul_backward_dx` needs live device storage, so an
/// integration test could never actually exercise the refusal.
fn q2_0_accepts_k(k: usize) -> bool {
    k != 0 && k % Q2_0_BLOCK == 0
}

impl RocmDevice {
    /// `dX = dY @ W^T` for a packed Q2_0 weight, dequantizing W inline so the
    /// weight is never expanded to f32.
    ///
    /// Q2_0 is 64 weights per 18-byte block, so the kernel's row stride
    /// `(K / 64) * 18` is only exact when `K` is a multiple of 64. That is
    /// refused here rather than left to `K / 64` truncating: a trailing
    /// partial block would be read as if it were whole, which is a silent
    /// out-of-bounds read past the last row rather than an error.
    pub(crate) fn launch_fused_dequant_backward_gemm_q2_0(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // Refuse on the same predicate the size helper uses, so the two cannot
        // disagree about what a well-formed K is.
        q2_0_packed_bytes(k)?;
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q2_0",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal is the whole point of the guard, so assert the rejected set
    /// explicitly -- including the two cases that are easy to wave through:
    /// `k = 0` (a zero-length K has no block at all, and `0 % 64 == 0` would let
    /// it slip past a modulo-only check) and the off-by-ones either side of a
    /// boundary.
    #[test]
    fn q2_0_rejects_k_that_is_not_whole_blocks() {
        for k in [0usize, 1, 2, 32, 63, 65, 100, 127, 129, 191] {
            assert!(
                !q2_0_accepts_k(k),
                "K={k} is not a whole number of {Q2_0_BLOCK}-weight blocks and \
                 must be refused: (k/{Q2_0_BLOCK})*{Q2_0_BYTES} truncates and reads \
                 past the last row"
            );
            assert!(
                q2_0_packed_bytes(k).is_err(),
                "K={k} must fail the size helper for the same reason"
            );
        }
    }

    /// Accepted K must actually produce the row stride the kernel computes,
    /// which is what the two functions above are jointly claiming.
    #[test]
    fn q2_0_accepts_whole_blocks_and_sizes_them_exactly() {
        for k in [64usize, 128, 192, 256, 4096] {
            assert!(
                q2_0_accepts_k(k),
                "K={k} is whole blocks and must be accepted"
            );
            assert_eq!(
                q2_0_packed_bytes(k).expect("whole blocks must size"),
                (k / Q2_0_BLOCK) * Q2_0_BYTES,
                "row stride for K={k} must match the kernel's (K/64)*18"
            );
        }
        // 4096 is the production Llama-7B FFN shape; assert it once explicitly so
        // the stride is pinned at a size that actually occurs, not only at 64.
        assert_eq!(q2_0_packed_bytes(4096).unwrap(), 64 * Q2_0_BYTES);
    }

    /// A `(k + 63) / 64` style round-up would pass the "accepts" test above
    /// while over-reporting bytes for a rejected K. Pin that the size helper
    /// never rounds a partial block up into a whole one.
    #[test]
    fn q2_0_packed_bytes_never_rounds_a_partial_block_up() {
        // 100 weights = 1 whole block + 36 leftovers. Rounding up would claim
        // 36 bytes that hold no complete weight.
        assert!(q2_0_packed_bytes(100).is_err());
        assert!(q2_0_packed_bytes(1).is_err());
    }
}
