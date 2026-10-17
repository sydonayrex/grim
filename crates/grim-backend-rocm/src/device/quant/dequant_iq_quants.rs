//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! IQ-quant (iq2xxs..iq4xs) fused dequant GEMMs + generic elementwise dequant dispatch.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{HipDim3, arg};

/// Threads per block for a standalone IQ dequant kernel.
///
/// Load-bearing, and NOT uniform across the family. Each kernel is declared
/// `__launch_bounds__` for exactly the thread count its own geometry needs:
/// the 256-element super-blocks (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
/// IQ4_XS) need 64 threads x 4 elements each, but IQ4_NL's llama.cpp block is
/// `QK4_NL = 32` weights (ggml-common.h), so it needs 8 x 4 = 32.
///
/// Launching an `__launch_bounds__(8)` kernel with 64 threads per block is
/// rejected by the runtime, and it rejects it as a bare
/// `hipErrorLaunchFailure` (719) naming neither the kernel's launch bounds nor
/// the thread count that broke them -- which is why this used to sit in the
/// shared launcher as a hardcoded 64 and took the whole IQ4_NL path down with
/// it.
const IQ_SUPERBLOCK_THREADS: u32 = 64;
const IQ4_NL_BLOCK_THREADS: u32 = 8;

impl RocmDevice {
    pub(crate) fn launch_dequant_iq2xxs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant(
            "grim_dequant_iq2xxs",
            packed_storage,
            out_storage,
            n_blocks,
            IQ_SUPERBLOCK_THREADS,
        )
    }

    pub(crate) fn launch_dequant_iq2xs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant(
            "grim_dequant_iq2xs",
            packed_storage,
            out_storage,
            n_blocks,
            IQ_SUPERBLOCK_THREADS,
        )
    }

    pub(crate) fn launch_dequant_iq2s(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant(
            "grim_dequant_iq2s",
            packed_storage,
            out_storage,
            n_blocks,
            IQ_SUPERBLOCK_THREADS,
        )
    }

    pub(crate) fn launch_dequant_iq3xxs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant(
            "grim_dequant_iq3xxs",
            packed_storage,
            out_storage,
            n_blocks,
            IQ_SUPERBLOCK_THREADS,
        )
    }

    pub(crate) fn launch_dequant_iq3s(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant(
            "grim_dequant_iq3s",
            packed_storage,
            out_storage,
            n_blocks,
            IQ_SUPERBLOCK_THREADS,
        )
    }

    /// IQ4_NL: 18 bytes per **32** weights, so 8 threads per block -- not the
    /// family's 64.
    pub(crate) fn launch_dequant_iq4nl(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant(
            "grim_dequant_iq4nl",
            packed_storage,
            out_storage,
            n_blocks,
            IQ4_NL_BLOCK_THREADS,
        )
    }

    pub(crate) fn launch_dequant_iq4xs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant(
            "grim_dequant_iq4xs",
            packed_storage,
            out_storage,
            n_blocks,
            IQ_SUPERBLOCK_THREADS,
        )
    }

    /// Generic helper for standalone dequant kernels that take (packed, out,
    /// n_blocks).
    ///
    /// `threads_per_block` is a parameter, not a constant, because it must
    /// match each kernel's `__launch_bounds__` and that is per-kernel. Every
    /// caller states its own kernel's contract at the call site, where the
    /// kernel name is in view, so a future format with a different block width
    /// cannot inherit the wrong one by omission.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_generic_dequant(
        &self,
        name: &str,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
        threads_per_block: u32,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: packed has no device ptr", name)))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: out has no device ptr", name)))?;
        if threads_per_block == 0 {
            return Err(Error::Backend(format!(
                "{name}: threads_per_block must be non-zero"
            )));
        }
        let grid_x: u32 = n_blocks
            .try_into()
            .map_err(|_| Error::Backend(format!("{}: grid overflow", name)))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(threads_per_block, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_blk = n_blocks as i32;
        self.launch_compute_kernel(
            name,
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_blk)],
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_iq2xxs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // D2-2: group-hoisted kernel; scalar entry kept as the parity oracle.
        self.launch_fused_deq_gemm_simple(
            "grim_fused_dequant_gemm_iq2xxs_grouped",
            a,
            b,
            out,
            m,
            n,
            k,
            1,
        )
    }

    /// Scalar IQ2XXS oracle for the grouped kernel's parity gate.
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_iq2xxs_scalar(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq2xxs", a, b, out, m, n, k, 1)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_iq2xxs(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq2xxs",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_iq2xs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // D2-2: group-hoisted kernel; scalar entry kept as the parity oracle.
        self.launch_fused_deq_gemm_simple(
            "grim_fused_dequant_gemm_iq2xs_grouped",
            a,
            b,
            out,
            m,
            n,
            k,
            1,
        )
    }

    /// Scalar IQ2XS oracle for the grouped kernel's parity gate.
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_iq2xs_scalar(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq2xs", a, b, out, m, n, k, 1)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_iq2xs(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq2xs",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_iq2s(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // D2-2: group-hoisted kernel; scalar entry kept as the parity oracle.
        self.launch_fused_deq_gemm_simple(
            "grim_fused_dequant_gemm_iq2s_grouped",
            a,
            b,
            out,
            m,
            n,
            k,
            1,
        )
    }

    /// Scalar IQ2_S oracle for the grouped kernel's parity gate.
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_iq2s_scalar(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq2s", a, b, out, m, n, k, 1)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_iq2s(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq2s",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_iq3xxs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // D2-2: group-hoisted kernel (see `..._iq1s`); the scalar entry below
        // stays as the parity oracle.
        self.launch_fused_deq_gemm_simple(
            "grim_fused_dequant_gemm_iq3xxs_grouped",
            a,
            b,
            out,
            m,
            n,
            k,
            1,
        )
    }

    /// Scalar IQ3_XXS oracle for the grouped kernel's parity gate.
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_iq3xxs_scalar(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq3xxs", a, b, out, m, n, k, 1)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_iq3xxs(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq3xxs",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_iq3s(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // D2-2: group-hoisted kernel; scalar entry kept as the parity oracle.
        self.launch_fused_deq_gemm_simple(
            "grim_fused_dequant_gemm_iq3s_grouped",
            a,
            b,
            out,
            m,
            n,
            k,
            1,
        )
    }

    /// Scalar IQ3_S oracle for the grouped kernel's parity gate.
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_iq3s_scalar(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq3s", a, b, out, m, n, k, 1)
    }

    pub(crate) fn launch_fused_dequant_gemm_iq1s(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // D2-2: the GROUP-HOISTED kernel (grid-index math + IQ1S_GRID load
        // hoisted out of each 8-weight group). Bit-identical to the scalar
        // kernel — same values, same ascending accumulation order — and
        // measured much faster; the scalar entry stays reachable for the
        // parity gate via `..._iq1s_scalar`.
        self.launch_fused_deq_gemm_simple(
            "grim_fused_dequant_gemm_iq1s_grouped",
            a,
            b,
            out,
            m,
            n,
            k,
            1,
        )
    }

    /// The original per-weight scalar IQ1_S GEMM, kept as the parity oracle for
    /// the grouped kernel (and as the fallback if the grouped one regresses).
    /// Only the parity gate calls it in-tree, hence the allow.
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_iq1s_scalar(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq1s", a, b, out, m, n, k, 1)
    }

    pub(crate) fn launch_fused_dequant_gemm_iq1m(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // D2-2: group-hoisted kernel; scalar entry kept as the parity oracle.
        self.launch_fused_deq_gemm_simple(
            "grim_fused_dequant_gemm_iq1m_grouped",
            a,
            b,
            out,
            m,
            n,
            k,
            1,
        )
    }

    /// Scalar IQ1_M oracle for the grouped kernel's parity gate.
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_iq1m_scalar(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq1m", a, b, out, m, n, k, 1)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_iq1s(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq1s",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_iq1m(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq1m",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_iq3s(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq3s",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_iq4nl(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq4nl", a, b, out, m, n, k, 1)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_iq4nl(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq4nl",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_iq4xs(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // D2-2: group-hoisted kernel; scalar entry kept as the parity oracle.
        self.launch_fused_deq_gemm_simple(
            "grim_fused_dequant_gemm_iq4xs_grouped",
            a,
            b,
            out,
            m,
            n,
            k,
            1,
        )
    }

    /// Scalar IQ4_XS oracle for the grouped kernel's parity gate.
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_iq4xs_scalar(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq4xs", a, b, out, m, n, k, 1)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_iq4xs(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_iq4xs",
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
mod iq1s_grouped_parity {
    use grim_tensor::{ArithType, BackendStorage, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage};

    fn packed_iq3xxs(n: usize, k: usize) -> Vec<u8> {
        // block_iq3_xxs = d(2) + qs(64) + ss(32) = 98 B / 256 weights.
        const BLK: usize = 98;
        let row_bytes = (k / 256) * BLK;
        let mut b = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..(k / 256) {
                let off = row * row_bytes + blk * BLK;
                b[off] = 0x00;
                b[off + 1] = 0x3C; // d = 1.0f16
                for i in 2..BLK {
                    b[off + i] = ((row * 53 + blk * 29 + i * 11) & 0xFF) as u8;
                }
            }
        }
        b
    }

    fn packed_iq3s(n: usize, k: usize) -> Vec<u8> {
        // block_iq3_s = d(2) + qs(64) + qh(8) + signs(32) + scales(4) = 110 B / 256.
        const BLK: usize = 110;
        let row_bytes = (k / 256) * BLK;
        let mut b = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..(k / 256) {
                let off = row * row_bytes + blk * BLK;
                b[off] = 0x00;
                b[off + 1] = 0x3C; // d = 1.0f16
                for i in 2..BLK {
                    b[off + i] = ((row * 97 + blk * 41 + i * 13) & 0xFF) as u8;
                }
            }
        }
        b
    }

    fn packed_iq2s(n: usize, k: usize) -> Vec<u8> {
        // block_iq2_s = d(2) + qs(32) + signs(32) + qh(8) + scales(8) = 82 B / 256.
        const BLK: usize = 82;
        let row_bytes = (k / 256) * BLK;
        let mut b = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..(k / 256) {
                let off = row * row_bytes + blk * BLK;
                b[off] = 0x00;
                b[off + 1] = 0x3C;
                for i in 2..BLK {
                    b[off + i] = ((row * 71 + blk * 23 + i * 5) & 0xFF) as u8;
                }
            }
        }
        b
    }

    fn packed_fixed(n: usize, k: usize, blk_bytes: usize, seed: usize) -> Vec<u8> {
        let row_bytes = (k / 256) * blk_bytes;
        let mut b = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..(k / 256) {
                let off = row * row_bytes + blk * blk_bytes;
                b[off] = 0x00;
                b[off + 1] = 0x3C; // d = 1.0f16
                for i in 2..blk_bytes {
                    b[off + i] = ((row * seed + blk * 23 + i * 5) & 0xFF) as u8;
                }
            }
        }
        b
    }

    fn packed_iq2xs(n: usize, k: usize) -> Vec<u8> {
        packed_fixed(n, k, 74, 61)
    }
    fn packed_iq2xxs(n: usize, k: usize) -> Vec<u8> {
        packed_fixed(n, k, 66, 43)
    }

    fn packed_iq4xs(n: usize, k: usize) -> Vec<u8> {
        packed_fixed(n, k, 136, 29)
    }

    fn packed_iq1m(n: usize, k: usize) -> Vec<u8> {
        packed_fixed(n, k, 56, 37)
    }

    fn packed_iq1s(n: usize, k: usize) -> Vec<u8> {
        // Deterministic fabricated bytes in the real IQ1_S layout
        // (block_iq1_s = 2 + 32 + 16 = 50 B / 256 weights). Values are
        // arbitrary; the gate compares two KERNELS, not two values.
        const BLK: usize = 50;
        let row_bytes = (k / 256) * BLK;
        let mut b = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..(k / 256) {
                let off = row * row_bytes + blk * BLK;
                b[off] = 0x00;
                b[off + 1] = 0x3C; // d = 1.0f16
                for i in 2..BLK {
                    b[off + i] = ((row * 131 + blk * 17 + i * 7) & 0xFF) as u8;
                }
            }
        }
        b
    }

    /// D2-2 gate: the group-hoisted IQ1_S GEMM must be BIT-IDENTICAL to the
    /// scalar per-weight kernel it replaces (same dequant values, same
    /// ascending accumulation order). Any drift here is a kernel bug, not
    /// reduction-order noise.
    #[test]
    fn grouped_iq1s_gemm_matches_the_scalar_kernel() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let _lock = crate::device::util::gpu_test_lock();
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            eprintln!("[SKIP] device 0 unavailable");
            return;
        };
        for (scheme, blk_bytes, packer) in [
            (KQuantScheme::IQ1S, 50usize, packed_iq1s as fn(usize, usize) -> Vec<u8>),
            (KQuantScheme::IQ3XXS, 98usize, packed_iq3xxs),
            (KQuantScheme::IQ3S, 110usize, packed_iq3s),
            (KQuantScheme::IQ2S, 82usize, packed_iq2s),
            (KQuantScheme::IQ2XS, 74usize, packed_iq2xs),
            (KQuantScheme::IQ2XXS, 66usize, packed_iq2xxs),
            (KQuantScheme::IQ4XS, 136usize, packed_iq4xs),
            (KQuantScheme::IQ1M, 56usize, packed_iq1m),
        ] {
        let _ = blk_bytes;
        let dt = DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(scheme),
        };
        for (n, k) in [(64usize, 512usize), (256, 5120)] {
            let bytes = packer(n, k);
            let w_box = dev
                .from_cpu_bytes(&bytes, &Shape::new(vec![n, k]), dt.clone())
                .expect("packed weight");
            let w = w_box
                .as_any()
                .downcast_ref::<crate::RocmStorage>()
                .expect("RocmStorage");
            let act: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();
            let a_box = dev
                .from_cpu(&act, &Shape::new(vec![1, k]), DType::F32)
                .expect("act");
            let a = a_box
                .as_any()
                .downcast_ref::<crate::RocmStorage>()
                .expect("RocmStorage");
            let out_scalar_box = dev
                .zeros(&Shape::new(vec![1, n]), DType::F32)
                .expect("out alloc");
            let out_grouped_box = dev
                .zeros(&Shape::new(vec![1, n]), DType::F32)
                .expect("out alloc");
            let out_scalar = out_scalar_box
                .as_any()
                .downcast_ref::<crate::RocmStorage>()
                .expect("RocmStorage");
            let out_grouped = out_grouped_box
                .as_any()
                .downcast_ref::<crate::RocmStorage>()
                .expect("RocmStorage");
            match scheme {
                KQuantScheme::IQ1S => {
                    dev.launch_fused_dequant_gemm_iq1s_scalar(&a, &w, &out_scalar, 1, n, k)
                        .expect("scalar launch");
                    dev.launch_fused_dequant_gemm_iq1s(&a, &w, &out_grouped, 1, n, k)
                        .expect("grouped launch");
                }
                KQuantScheme::IQ3XXS => {
                    dev.launch_fused_dequant_gemm_iq3xxs_scalar(&a, &w, &out_scalar, 1, n, k)
                        .expect("scalar launch");
                    dev.launch_fused_dequant_gemm_iq3xxs(&a, &w, &out_grouped, 1, n, k)
                        .expect("grouped launch");
                }
                KQuantScheme::IQ3S => {
                    dev.launch_fused_dequant_gemm_iq3s_scalar(&a, &w, &out_scalar, 1, n, k)
                        .expect("scalar launch");
                    dev.launch_fused_dequant_gemm_iq3s(&a, &w, &out_grouped, 1, n, k)
                        .expect("grouped launch");
                }
                KQuantScheme::IQ2S => {
                    dev.launch_fused_dequant_gemm_iq2s_scalar(&a, &w, &out_scalar, 1, n, k)
                        .expect("scalar launch");
                    dev.launch_fused_dequant_gemm_iq2s(&a, &w, &out_grouped, 1, n, k)
                        .expect("grouped launch");
                }
                KQuantScheme::IQ2XS => {
                    dev.launch_fused_dequant_gemm_iq2xs_scalar(&a, &w, &out_scalar, 1, n, k)
                        .expect("scalar launch");
                    dev.launch_fused_dequant_gemm_iq2xs(&a, &w, &out_grouped, 1, n, k)
                        .expect("grouped launch");
                }
                KQuantScheme::IQ2XXS => {
                    dev.launch_fused_dequant_gemm_iq2xxs_scalar(&a, &w, &out_scalar, 1, n, k)
                        .expect("scalar launch");
                    dev.launch_fused_dequant_gemm_iq2xxs(&a, &w, &out_grouped, 1, n, k)
                        .expect("grouped launch");
                }
                KQuantScheme::IQ4XS => {
                    dev.launch_fused_dequant_gemm_iq4xs_scalar(&a, &w, &out_scalar, 1, n, k)
                        .expect("scalar launch");
                    dev.launch_fused_dequant_gemm_iq4xs(&a, &w, &out_grouped, 1, n, k)
                        .expect("grouped launch");
                }
                _ => {
                    dev.launch_fused_dequant_gemm_iq1m_scalar(&a, &w, &out_scalar, 1, n, k)
                        .expect("scalar launch");
                    dev.launch_fused_dequant_gemm_iq1m(&a, &w, &out_grouped, 1, n, k)
                        .expect("grouped launch");
                }
            }
            dev.synchronize();
            let s = out_scalar.to_cpu_vec_f32().expect("scalar readback");
            let g = out_grouped.to_cpu_vec_f32().expect("grouped readback");
            assert_eq!(s.len(), g.len());
            let mut worst = 0.0f32;
            let mut worst_i = 0usize;
            for (i, (x, y)) in s.iter().zip(g.iter()).enumerate() {
                let d = (x - y).abs();
                if d > worst {
                    worst = d;
                    worst_i = i;
                }
            }
            eprintln!(
                "[iq-grouped-parity] {scheme:?} n={n} k={k}: worst abs {worst:.3e} at {worst_i}"
            );
            assert!(
                worst <= 1e-6,
                "grouped {scheme:?} GEMM diverges from the scalar kernel: worst {worst:.3e} at \
                 {worst_i} (scalar {}, grouped {})",
                s[worst_i],
                g[worst_i]
            );
        }
        }
    }
}
