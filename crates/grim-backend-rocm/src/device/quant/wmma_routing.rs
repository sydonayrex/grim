//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! WMMA capability gating, route decision helpers, shared tiled launchers,
//! and the `GpuDequant` impl. Owns `wmma_route_tests`.

use std::ffi::c_void;

use grim_tensor::dtype::{ ArithType, Storage as DTypeStorage };
use grim_tensor::error::{Error, Result};
use grim_tensor::{ BackendStorage };

use crate::device::roc_device::{ RocmDevice };
use crate::memory::storage::RocmStorage;
use crate::{ HipDim3, arg };



impl RocmDevice {
    /// Returns `true` when the activation storage holds native FP16 data
    /// (`ArithType::F16` + `DTypeStorage::Native`).  The FP16-input WMMA
    /// kernels read such activations directly, halving A-read bandwidth.
    pub(crate) fn is_fp16_activation(storage: &RocmStorage) -> bool {
        storage.dtype().arith == ArithType::F16
            && matches!(storage.dtype().storage, DTypeStorage::Native)
    }

    /// SPEED-ROC: dispatch helper for IQ-family formats.
    /// Uses WMMA kernel on RDNA3/4 for small M (decode), falls back to scalar otherwise.
    pub(crate) fn launch_iq_wmma_fallback(
        &self,
        a: &RocmStorage,
        b: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        wmma_method: fn(&Self, &RocmStorage, &RocmStorage, &RocmStorage, usize, usize, usize) -> Result<*mut c_void>,
        scalar_method: fn(&Self, &RocmStorage, &RocmStorage, &RocmStorage, usize, usize, usize) -> Result<*mut c_void>,
    ) -> Result<()> {
        let is_rdna34 = self.is_rdna34;
        if is_rdna34 && m <= 4 {
            wmma_method(self, a, b, out, m, n, k)?;
        } else {
            scalar_method(self, a, b, out, m, n, k)?;
        }
        Ok(())
    }

    /// SPEED-ROC-8: per-format opt-in flag (GRIM_Q5K_TILED, GRIM_IQ4XS_TILED,
    /// GRIM_Q8_0_TILED, ...). Cached per format tag; default off — the scalar
    /// kernels remain the shipping default until per-model benchmarks flip it.
    pub(crate) fn tiled_quant_enabled(tag: &str) -> bool {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = match cache.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        *guard.entry(tag.to_string()).or_insert_with(|| {
            let var = format!("GRIM_{}_TILED", tag.to_uppercase());
            matches!(std::env::var(&var).as_deref(), Ok("1" | "true" | "on"))
        })
    }

    /// K divisibility required by the tiled loops: the 64-wide tile stride must
    /// land on whole super-blocks for every row (256-element formats trivially
    /// satisfy it; Q8_0's 32-element blocks require K % 64 == 0).
    pub(crate) fn tiled_k_multiple(blk_elems: u32) -> usize {
        if blk_elems == 32 {
            64
        } else {
            blk_elems as usize
        }
    }

    /// SPEED-ROC-8: tiled-GEMM dispatch table for the macro-stamped kernels in
    /// kernels::quant_tiled_gemm — (scalar forward entry, format tag,
    /// super-block elements, bytes per block). The tiled entry names are
    /// derived from the tag: `grim_fused_dequant_gemm_{tag}_tiled` and
    /// `grim_fused_dequant_gemm_{tag}_backward_tiled`.
    const TILED_QUANT_TABLE: &[(&str, &str, u32, u32)] = &[
        ("grim_fused_dequant_gemm_q5k", "q5k", 256, 176),
        ("grim_fused_dequant_gemm_q6k", "q6k", 256, 210),
        ("grim_fused_dequant_gemm_iq2xxs", "iq2xxs", 256, 66),
        ("grim_fused_dequant_gemm_iq2xs", "iq2xs", 256, 74),
        ("grim_fused_dequant_gemm_iq2s", "iq2s", 256, 82),
        ("grim_fused_dequant_gemm_iq3xxs", "iq3xxs", 256, 96),
        ("grim_fused_dequant_gemm_iq3s", "iq3s", 256, 110),
        ("grim_fused_dequant_gemm_iq4nl", "iq4nl", 256, 170),
        ("grim_fused_dequant_gemm_iq4xs", "iq4xs", 256, 136),
        ("grim_fused_dequant_gemm_q8_0", "q8_0", 32, 34),
    ];

    pub(crate) fn tiled_quant_lookup(entry_or_tag: &str) -> Option<(&'static str, u32)> {
        Self::TILED_QUANT_TABLE
            .iter()
            .find(|(entry, tag, _, _)| *entry == entry_or_tag || *tag == entry_or_tag)
            .map(|(_, tag, blk, _)| (*tag, *blk))
    }

    /// Generic tiled forward launcher — grid (ceil(N/64), ceil(M/4)), block (64, 4, 1).
    pub(crate) fn launch_fused_deq_gemm_tiled(
        &self,
        kernel: &str,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: a has no device ptr", kernel)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: b has no device ptr", kernel)))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: out has no device ptr", kernel)))?;
        let grid_dim = HipDim3::new(n.div_ceil(64) as u32, m.div_ceil(4) as u32, 1);
        let block_dim = HipDim3::new(64, 4, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            kernel,
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

    /// Generic tiled backward launcher — grid (ceil(K/64), ceil(M/4)), block (64, 4, 1).
    pub(crate) fn launch_fused_deq_gemm_tiled_backward(
        &self,
        kernel: &str,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dY has no device ptr", kernel)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: b has no device ptr", kernel)))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dX has no device ptr", kernel)))?;
        let grid_dim = HipDim3::new(k.div_ceil(64) as u32, m.div_ceil(4) as u32, 1);
        let block_dim = HipDim3::new(64, 4, 1);
        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            kernel,
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

    pub(crate) fn launch_fused_deq_gemm_simple(
        &self,
        name: &str,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // SPEED-ROC-8: opt-in LDS-tiled prefill path (per-format GRIM_*_TILED).
        if let Some((tag, blk)) = Self::tiled_quant_lookup(name) {
            if Self::tiled_quant_enabled(tag)
                && m >= 16
                && n >= 64
                && k % Self::tiled_k_multiple(blk) == 0
            {
                let tiled_kernel = format!("grim_fused_dequant_gemm_{}_tiled", tag);
                return self.launch_fused_deq_gemm_tiled(
                    &tiled_kernel,
                    a_storage,
                    b_storage,
                    out_storage,
                    m,
                    n,
                    k,
                );
            }
        }
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: a has no device ptr", name)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: b has no device ptr", name)))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: out has no device ptr", name)))?;
        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend(format!("{}: m*n overflow", name)))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend(format!("{}: grid overflow", name)))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            name,
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

    /// Generic backward fused dequant GEMM launcher.
    pub(crate) fn launch_fused_deq_backward_gemm_simple(
        &self,
        name: &str,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // SPEED-ROC-8: opt-in LDS-tiled backward path (same GRIM_*_TILED flags).
        // Backward entries are "grim_fused_dequant_backward_gemm_{tag}".
        if let Some(tag) = name
            .strip_prefix("grim_fused_dequant_backward_gemm_")
            .and_then(|fmt| Self::tiled_quant_lookup(fmt))
        {
            let (tag, blk) = tag;
            if Self::tiled_quant_enabled(tag) && n >= 64 && k % Self::tiled_k_multiple(blk) == 0 {
                let tiled_kernel = format!("grim_fused_dequant_gemm_{}_backward_tiled", tag);
                return self.launch_fused_deq_gemm_tiled_backward(
                    &tiled_kernel,
                    dy_storage,
                    b_storage,
                    dx_storage,
                    m,
                    n,
                    k,
                );
            }
        }
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dY has no device ptr", name)))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: B has no device ptr", name)))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend(format!("{}: dX has no device ptr", name)))?;
        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend(format!("{}: m*k overflow", name)))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend(format!("{}: grid overflow", name)))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            name,
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


}


/// P1-WI-1: pure routing decision for the WMMA GEMM path. Extracted from [see: `RocmDevice::should_use_wmma_path`]
pub(crate) fn wmma_route_decision(
    ext: Option<&grim_format::spec::GrimTensorExt>,
    out_arith: ArithType,
    cfg_enabled: bool,
) -> bool {
    // No extension ⇒ no per-tensor hint ⇒ stick with the existing dispatcher
    let Some(ext) = ext else {
        return false;
    };
    if !cfg_enabled {
        return false;
    }
    match ext.layout_hint {
        grim_format::spec::LayoutHintTag::PackedQuantWmma { bits, .. } => {
            matches!(bits, 2 | 3 | 4 | 8) && out_arith == ArithType::F16
        }
        _ => false,
    }
}

impl grim_format::convert::GpuDequant for RocmDevice {
    fn dequantize(
        &self,
        storage: &grim_tensor::dtype::Storage,
        bytes: &[u8],
        elem_count: usize,
    ) -> grim_tensor::error::Result<Option<Vec<f32>>> {
        use grim_tensor::dtype::{BlockDtype, FloatPackScheme, KQuantScheme, Storage};
        match storage {
            Storage::KQuant(KQuantScheme::Q80) => {
                Ok(Some(self.dequantize_q8_0_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::Q4K) => {
                Ok(Some(self.dequantize_q4k_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ2XXS) => {
                Ok(Some(self.dequantize_iq2xxs_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ2XS) => {
                Ok(Some(self.dequantize_iq2xs_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ2S) => {
                Ok(Some(self.dequantize_iq2s_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ3XXS) => {
                Ok(Some(self.dequantize_iq3xxs_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ3S) => {
                Ok(Some(self.dequantize_iq3s_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ4NL) => {
                Ok(Some(self.dequantize_iq4nl_host(bytes, elem_count)?))
            }
            Storage::KQuant(KQuantScheme::IQ4XS) => {
                Ok(Some(self.dequantize_iq4xs_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::Fp8) => {
                Ok(Some(self.dequantize_fp8_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::MxFp4) => {
                Ok(Some(self.dequantize_mxfp4_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::MxFp8) => {
                Ok(Some(self.dequantize_mxfp8_host(bytes, elem_count)?))
            }
            Storage::FloatPack(FloatPackScheme::NvFp4) => {
                Ok(Some(self.dequantize_nvfp4_host(bytes, elem_count)?))
            }
            Storage::Block(BlockDtype::Fp8 | BlockDtype::Fp8Block16) => {
                Ok(Some(self.dequantize_fp8_host(bytes, elem_count)?))
            }
            _ => Ok(None),
        }
    }
}


#[cfg(test)]
mod wmma_route_tests {
    use super::*;
    use grim_format::spec::{GrimTensorExt, LayoutHintTag};

    fn ext_packed(bits: u8) -> GrimTensorExt {
        GrimTensorExt {
            tensor_name: "test.weight".into(),
            layout_hint: LayoutHintTag::PackedQuantWmma {
                bits,
                frag_m: 16,
                frag_n: 16,
            },
            ..Default::default()
        }
    }

    #[test]
    fn no_extension_routes_to_default() {
        assert!(!wmma_route_decision(None, ArithType::F16, true));
    }

    #[test]
    fn disabled_config_skips_wmma() {
        assert!(!wmma_route_decision(
            Some(&ext_packed(4)),
            ArithType::F16,
            false,
        ));
    }

    #[test]
    fn packed_4bit_f16_enabled() {
        assert!(wmma_route_decision(
            Some(&ext_packed(4)),
            ArithType::F16,
            true,
        ));
    }

    #[test]
    fn packed_2bit_supported_too() {
        assert!(wmma_route_decision(
            Some(&ext_packed(2)),
            ArithType::F16,
            true,
        ));
    }

    #[test]
    fn packed_unsupported_bpw_falls_back() {
        // 6-bit is not in {2,3,4,8} → must not dispatch to WMMA.
        assert!(!wmma_route_decision(
            Some(&ext_packed(6)),
            ArithType::F16,
            true,
        ));
    }

    #[test]
    fn non_f16_output_skips_wmma() {
        // WMMA path only registered for F16; F32 arch falls to rocBLAS.
        assert!(!wmma_route_decision(
            Some(&ext_packed(4)),
            ArithType::F32,
            true,
        ));
    }

    #[test]
    fn default_hint_skips_wmma() {
        let ext = GrimTensorExt {
            layout_hint: LayoutHintTag::Default,
            ..Default::default()
        };
        assert!(!wmma_route_decision(Some(&ext), ArithType::F16, true));
    }

    #[test]
    fn wavefront_tiled_does_not_route_wmma() {
        // WavefrontTiled goes through a different (existing) tiled path; do
        let ext = GrimTensorExt {
            layout_hint: LayoutHintTag::WavefrontTiled,
            ..Default::default()
        };
        assert!(!wmma_route_decision(Some(&ext), ArithType::F16, true));
    }
}

