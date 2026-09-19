//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! K-quant (q2k..q6k, q8_0, q4k) fused dequant GEMM launchers.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::device::roc_device::{ RocmDevice };
use crate::memory::storage::RocmStorage;
use crate::{ HipDim3, arg };



impl RocmDevice {
    /// Launch the JIT compiled Q4_K fused dequantization matmul kernel (Crow Tier).
    #[allow(dead_code)] // kernel launcher, not yet wired into this build's call graph
    pub(crate) fn launch_fused_dequant_gemm_q4k(
        &self,
        a_storage: &RocmStorage,
        b_q4k_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q4k: a has no device ptr".into()))?;
        let b_ptr = b_q4k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q4k: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_q4k: out has no device ptr".into())
        })?;

        // SPEED-ROC-6: opt-in LDS-tiled prefill path (GRIM_Q4K_TILED=1).
        // The scalar kernel is one-thread-per-output and re-dequantizes the
        // weight row once per output row; the tiled kernel stages weight tiles
        // through LDS and wins at prefill shapes (m >= 16). Decode (m small)
        // and layouts the tiling cannot express stay on the scalar path.
        static TILED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let tiled_enabled = *TILED.get_or_init(|| {
            matches!(
                std::env::var("GRIM_Q4K_TILED").as_deref(),
                Ok("1" | "true" | "on")
            )
        });
        if tiled_enabled && m >= 16 && n >= 64 && k % 256 == 0 {
            return self.launch_fused_dequant_gemm_q4k_tiled(
                a_storage,
                b_q4k_storage,
                out_storage,
                m,
                n,
                k,
            );
        }

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q4k: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_gemm_q4k: grid too large for u32".to_string())
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_q4k",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// SPEED-ROC-6: LDS-tiled Q4_K forward GEMM launcher (prefill path).
    /// Grid: (ceil(N/64), ceil(M/4)), block: (64, 4, 1). See
    /// `grim_fused_dequant_gemm_q4k_tiled` in kernels::q4k_gemm.
    pub(crate) fn launch_fused_dequant_gemm_q4k_tiled(
        &self,
        a_storage: &RocmStorage,
        b_q4k_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_q4k_tiled: a has no device ptr".into())
        })?;
        let b_ptr = b_q4k_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_q4k_tiled: b has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_q4k_tiled: out has no device ptr".into())
        })?;

        let grid_x: u32 = n.div_ceil(64) as u32;
        let grid_y: u32 = m.div_ceil(4) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);
        let block_dim = HipDim3::new(64, 4, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_q4k_tiled",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled Q4_K fused dequantization backward matmul kernel (Crow Tier).
    /// `b_scales_ptr` is accepted for interface parity with the f16 fallback; KQuant blocks carry their own.
    pub(crate) fn launch_fused_dequant_backward_gemm_q4k(
        &self,
        dy_storage: &RocmStorage,
        b_q4k_storage: &RocmStorage,
        b_scales_ptr: *const c_void,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_q4k: dY has no device ptr".into())
        })?;
        let b_ptr = b_q4k_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_q4k: B has no device ptr".into())
        })?;
        let dx_ptr = dx_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_q4k: dX has no device ptr".into())
        })?;

        // SPEED-ROC-6b: opt-in tiled backward path (same GRIM_Q4K_TILED flag).
        static TILED_BWD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let tiled_enabled = *TILED_BWD.get_or_init(|| {
            matches!(
                std::env::var("GRIM_Q4K_TILED").as_deref(),
                Ok("1" | "true" | "on")
            )
        });
        if tiled_enabled && n >= 64 && k % 256 == 0 {
            return self.launch_fused_dequant_gemm_q4k_backward_tiled(
                dy_storage,
                b_q4k_storage,
                dx_storage,
                m,
                n,
                k,
            );
        }

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_backward_q4k: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_backward_q4k: grid too large for u32".to_string())
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let bsptr = b_scales_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let _ = bsptr;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_q4k",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// SPEED-ROC-6b: LDS-tiled Q4_K backward GEMM launcher.
    /// Grid: (ceil(K/64), ceil(M/4)), block: (64, 4, 1).
    pub(crate) fn launch_fused_dequant_gemm_q4k_backward_tiled(
        &self,
        dy_storage: &RocmStorage,
        b_q4k_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_q4k_tiled: dY has no device ptr".into())
        })?;
        let b_ptr = b_q4k_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_q4k_tiled: B has no device ptr".into())
        })?;
        let dx_ptr = dx_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_q4k_tiled: dX has no device ptr".into())
        })?;

        let grid_x: u32 = k.div_ceil(64) as u32;
        let grid_y: u32 = m.div_ceil(4) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);
        let block_dim = HipDim3::new(64, 4, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_q4k_backward_tiled",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Dequantize Q4_K packed bytes to F32. `n_blocks` is derived [see: `packed.bytes / 144`]
    pub(crate) fn launch_dequant_q4k(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_q4k: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_q4k: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_blocks as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_q4k: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_blk = n_blocks as i32;
        self.launch_compute_kernel(
            "grim_dequant_q4k",
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_blk)],
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_q5k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q5k", a, b, out, m, n, k)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_q5k(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q5k",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_q6k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q6k", a, b, out, m, n, k)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_q6k(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q6k",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_q2k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q2k", a, b, out, m, n, k)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_q2k(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q2k",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_q3k(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q3k", a, b, out, m, n, k)
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_q3k(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q3k",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    pub(crate) fn launch_fused_dequant_gemm_q8_0(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // SPEED-ROC: row-count-aware dispatch — when N is large (e.g. down
        // projection where N == hidden_dim), the 4-col-per-thread variant
        // shares the activation read across 4 weight dequants, halving L2
        // traffic.  Env-gated via GRIM_ROWS4_MIN_N (0 = disabled).
        let rows4_min_n: usize = std::env::var("GRIM_ROWS4_MIN_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if rows4_min_n > 0 && n >= rows4_min_n && n % 4 == 0 {
            return self.launch_fused_dequant_gemm_q8_0_rows4(a, b, out, m, n, k);
        }
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_q8_0", a, b, out, m, n, k)
    }

    /// SPEED-ROC: Q8_0 NUM_ROWS=4 launcher — each thread computes 4 consecutive
    /// output columns sharing one activation read.  Grid covers M*(N/4) slots.
    pub(crate) fn launch_fused_dequant_gemm_q8_0_rows4(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q8_0_rows4: a has no device ptr".into()))?;
        let b_ptr = b
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q8_0_rows4: b has no device ptr".into()))?;
        let out_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q8_0_rows4: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let cols_per_thread: u64 = 4;
        let n_slots = (n / cols_per_thread as usize) as u64;
        let total_slots: u64 = (m as u64)
            .checked_mul(n_slots)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_q8_0_rows4: m*(n/4) overflow".into()))?;
        let grid_x: u32 = (total_slots.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("fused_dequant_gemm_q8_0_rows4: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_q8_0_rows4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    pub(crate) fn launch_fused_dequant_backward_gemm_q8_0(
        &self,
        dy: &RocmStorage,
        b: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        self.launch_fused_deq_backward_gemm_simple(
            "grim_fused_dequant_backward_gemm_q8_0",
            dy,
            b,
            dx,
            m,
            n,
            k,
        )
    }

    /// Dequantize Q8_0 packed bytes to F32. `n_blocks` is the number of [see: `packed`, `packed.bytes / 34`]
    pub(crate) fn launch_dequant_q8_0(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_q8_0: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_q8_0: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_blocks as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_q8_0: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_blk = n_blocks as i32;

        self.launch_compute_kernel(
            "grim_dequant_q8_0",
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_blk)],
        )
    }
}
