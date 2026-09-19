//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! IQ-quant (iq2xxs..iq4xs) fused dequant GEMMs + generic elementwise dequant dispatch.

use std::ffi::c_void;

use grim_tensor::error::{Error, Result};

use crate::device::roc_device::{ RocmDevice };
use crate::memory::storage::RocmStorage;
use crate::{ HipDim3, arg };



impl RocmDevice {
    pub(crate) fn launch_dequant_iq2xxs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq2xxs", packed_storage, out_storage, n_blocks)
    }

    pub(crate) fn launch_dequant_iq2xs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq2xs", packed_storage, out_storage, n_blocks)
    }

    pub(crate) fn launch_dequant_iq2s(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq2s", packed_storage, out_storage, n_blocks)
    }

    pub(crate) fn launch_dequant_iq3xxs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq3xxs", packed_storage, out_storage, n_blocks)
    }

    pub(crate) fn launch_dequant_iq3s(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq3s", packed_storage, out_storage, n_blocks)
    }

    pub(crate) fn launch_dequant_iq4nl(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq4nl", packed_storage, out_storage, n_blocks)
    }

    pub(crate) fn launch_dequant_iq4xs(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        self.launch_generic_dequant("grim_dequant_iq4xs", packed_storage, out_storage, n_blocks)
    }

    /// Generic helper for standalone dequant kernels that take (packed, out, n_blocks).
    pub(crate) fn launch_generic_dequant(
        &self,
        name: &str,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_blocks: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: packed has no device ptr", name)))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: out has no device ptr", name)))?;
        // The IQ dequant kernels use one 64-thread block per quant block
        // (each thread decodes 4 elements with a float4 store).
        const BLOCK_SIZE: usize = 64;
        let grid_x: u32 = n_blocks
            .try_into()
            .map_err(|_| Error::Backend(format!("{}: grid overflow", name)))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
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
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq2xxs", a, b, out, m, n, k)
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
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq2xs", a, b, out, m, n, k)
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
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq2s", a, b, out, m, n, k)
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
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq3xxs", a, b, out, m, n, k)
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
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq3s", a, b, out, m, n, k)
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
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq4nl", a, b, out, m, n, k)
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
        self.launch_fused_deq_gemm_simple("grim_fused_dequant_gemm_iq4xs", a, b, out, m, n, k)
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
