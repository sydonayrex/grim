//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

//! Module root: the trait-required `QuantOps` impl, kept whole.

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use grim_tensor::backend::{ComputeHandle, ReadyHandle};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, QuantOps, Shape};

use crate::device::roc_device::{
    FUSED_BACKWARD_DISPATCH_STATS, FUSED_FORWARD_DISPATCH_STATS, RocmDevice,
};
use crate::memory::storage::RocmStorage;
use crate::{RocmHandle, arg, dtype_f32};

mod dequant_fp_quants;
mod dequant_host;
mod dequant_iq_quants;
mod dequant_k_quants;
mod gptq_awq;
mod mxfp_gemm;
mod quantize_forward;
mod wmma_routing;

#[allow(unused_imports)] // flat public API surface: `device::quant::<method>`
pub use dequant_fp_quants::*;
#[allow(unused_imports)]
pub use dequant_host::*;
#[allow(unused_imports)]
pub use dequant_iq_quants::*;
#[allow(unused_imports)]
pub use dequant_k_quants::*;
#[allow(unused_imports)]
pub use gptq_awq::*;
#[allow(unused_imports)]
pub use mxfp_gemm::*;
#[allow(unused_imports)]
pub use quantize_forward::*;
#[allow(unused_imports)]
pub use wmma_routing::*;

impl QuantOps for RocmDevice {
    fn quantize(
        &self,
        x: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        let (out, _handle) = self.quantize_on_device(x, format)?;
        Ok(out)
    }

    fn fused_quant_gemm(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.quantized_matmul(a, b, &[], format, out_shape)
    }

    fn quantized_matmul(
        &self,
        a: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        _b_scales: &[f32],
        _format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        FUSED_FORWARD_DISPATCH_STATS
            .attempts
            .fetch_add(1, Ordering::Relaxed);
        let a_storage = match a.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return self.matmul(a, b_packed, out_shape),
        };
        let b_storage = match b_packed.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return self.matmul(a, b_packed, out_shape),
        };

        let dims = out_shape.dims();
        let (m, n) = match dims.len() {
            2 => (dims[0], dims[1]),
            _ => (
                dims[..dims.len() - 1].iter().product(),
                dims[dims.len() - 1],
            ),
        };
        let k = a_storage.shape().dims().last().copied().unwrap_or(0);
        static QMM_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *QMM_TRACE.get_or_init(|| std::env::var_os("GRIM_QMM_TRACE").is_some()) {
            eprintln!(
                "[qmm] ordinal={} m={m} n={n} k={k} b_dtype={:?}",
                self.ordinal,
                b_storage.dtype().storage
            );
        }
        // SPEED-ROC: WMMA dispatch threshold; env-overridable (GRIM_WMM_MAX_M).
        let wmma_max_m: usize = self.wmma_max_m;
        // WMMA kernels are wave32-only (rocWMMA on RDNA3/4; wave64 CDNA never qualifies).
        let wave32 = matches!(
            self.props.wavefront_size,
            crate::device::handles::WavefrontSize::W32
        );

        let out_storage = RocmStorage::alloc_gpu(
            out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;

        use grim_tensor::{BlockDtype, FloatPackScheme, KQuantScheme};
        match b_storage.dtype().storage {
            DTypeStorage::KQuant(KQuantScheme::Q4K) => {
                // SPEED-DOT: On RDNA3/4, use vector dot4 GEMV for M=1 decode (100% tensor util vs 6.25% for WMMA).
                let is_rdna34 = self.is_rdna34;
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                // grim_dot4_q4k_q81_gemv is written and verified for the
                // RDNA2 APU (gfx103x) sdot4 path only — on RDNA3/4 it
                // mis-computes (scale-shuffle skews). Other arches take the
                // WMMA / scalar fused-dequant path instead.
                let is_rdna2 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_rdna2 && m == 1 && !dot_disabled && k % 256 == 0 {
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.write().unwrap_or_else(|e| e.into_inner());
                    let need_alloc = match buf_guard.as_ref() {
                        Some(s) => s.bytes < q81_bytes,
                        None => true,
                    };
                    if need_alloc {
                        *buf_guard = Some(RocmStorage::alloc_gpu(
                            &shape,
                            DType {
                                arith: ArithType::U8,
                                storage: DTypeStorage::Native,
                            },
                            &self.allocator,
                            self.ordinal,
                        )?);
                    }
                    let a_prequant = a_storage.dtype().arith == ArithType::U8;
                    if a_prequant {
                        drop(buf_guard);
                        self.launch_dot4_q4k_q81_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let q_stream = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        fence_act_quant(self, q_stream);
                        self.launch_dot4_q4k_q81_gemv(act_q81, b_storage, &out_storage, m, n, k)?;
                    }
                } else if is_rdna34 && wmma_quant_tile_ok(wave32, m, n, k, 256) {
                    self.launch_wmma_fused_dequant_q4k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_fused_dequant_gemm_q4k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::Q5K) => {
                let is_rdna34 = self.is_rdna34;
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                // grim_dot4_q4k_q81_gemv is written and verified for the
                // RDNA2 APU (gfx103x) sdot4 path only — on RDNA3/4 it
                // mis-computes (scale-shuffle skews). Other arches take the
                // WMMA / scalar fused-dequant path instead.
                let is_rdna2 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_rdna2 && m == 1 && !dot_disabled && k % 256 == 0 {
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.write().unwrap_or_else(|e| e.into_inner());
                    let need_alloc = match buf_guard.as_ref() {
                        Some(s) => s.bytes < q81_bytes,
                        None => true,
                    };
                    if need_alloc {
                        *buf_guard = Some(RocmStorage::alloc_gpu(
                            &shape,
                            DType {
                                arith: ArithType::U8,
                                storage: DTypeStorage::Native,
                            },
                            &self.allocator,
                            self.ordinal,
                        )?);
                    }
                    let a_prequant = a_storage.dtype().arith == ArithType::U8;
                    if a_prequant {
                        drop(buf_guard);
                        self.launch_dot4_q5k_q81_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let q_stream = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        fence_act_quant(self, q_stream);
                        self.launch_dot4_q5k_q81_gemv(act_q81, b_storage, &out_storage, m, n, k)?;
                    }
                } else if is_rdna34 && wmma_quant_tile_ok(wave32, m, n, k, 16) {
                    self.launch_wmma_fused_dequant_q5k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_fused_dequant_gemm_q5k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::Q6K) => {
                let is_rdna34 = self.is_rdna34;
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                // grim_dot4_q4k_q81_gemv is written and verified for the
                // RDNA2 APU (gfx103x) sdot4 path only — on RDNA3/4 it
                // mis-computes (scale-shuffle skews). Other arches take the
                // WMMA / scalar fused-dequant path instead.
                let is_rdna2 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_rdna2 && m == 1 && !dot_disabled && k % 256 == 0 {
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.write().unwrap_or_else(|e| e.into_inner());
                    let need_alloc = match buf_guard.as_ref() {
                        Some(s) => s.bytes < q81_bytes,
                        None => true,
                    };
                    if need_alloc {
                        *buf_guard = Some(RocmStorage::alloc_gpu(
                            &shape,
                            DType {
                                arith: ArithType::U8,
                                storage: DTypeStorage::Native,
                            },
                            &self.allocator,
                            self.ordinal,
                        )?);
                    }
                    let a_prequant = a_storage.dtype().arith == ArithType::U8;
                    if a_prequant {
                        drop(buf_guard);
                        self.launch_dot4_q6k_q81_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let q_stream = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        fence_act_quant(self, q_stream);
                        self.launch_dot4_q6k_q81_gemv(act_q81, b_storage, &out_storage, m, n, k)?;
                    }
                } else if is_rdna34 && wmma_quant_tile_ok(wave32, m, n, k, 16) {
                    self.launch_wmma_fused_dequant_q6k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_fused_dequant_gemm_q6k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::Q2K) => {
                let is_rdna34 = self.is_rdna34;
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                // grim_dot4_q4k_q81_gemv is written and verified for the
                // RDNA2 APU (gfx103x) sdot4 path only — on RDNA3/4 it
                // mis-computes (scale-shuffle skews). Other arches take the
                // WMMA / scalar fused-dequant path instead.
                let is_rdna2 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_rdna2 && m == 1 && !dot_disabled && k % 256 == 0 {
                    // Phase 4.5f: m==1 decode routes to the Q2_K dot4 GEMV.
                    // Activations must be pre-quantized to Q8_1; quantize on the fly otherwise.
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.write().unwrap_or_else(|e| e.into_inner());
                    let need_alloc = match buf_guard.as_ref() {
                        Some(s) => s.bytes < q81_bytes,
                        None => true,
                    };
                    if need_alloc {
                        *buf_guard = Some(RocmStorage::alloc_gpu(
                            &shape,
                            DType {
                                arith: ArithType::U8,
                                storage: DTypeStorage::Native,
                            },
                            &self.allocator,
                            self.ordinal,
                        )?);
                    }
                    let a_prequant = a_storage.dtype().arith == ArithType::U8;
                    if a_prequant {
                        drop(buf_guard);
                        self.launch_dot4_q2k_q81_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let q_stream = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        fence_act_quant(self, q_stream);
                        self.launch_dot4_q2k_q81_gemv(act_q81, b_storage, &out_storage, m, n, k)?;
                    }
                } else if is_rdna34 && wmma_quant_tile_ok(wave32, m, n, k, 16) {
                    self.launch_wmma_fused_dequant_q2k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_fused_dequant_gemm_q2k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::Q3K) => {
                let is_rdna34 = self.is_rdna34;
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                // grim_dot4_q4k_q81_gemv is written and verified for the
                // RDNA2 APU (gfx103x) sdot4 path only — on RDNA3/4 it
                // mis-computes (scale-shuffle skews). Other arches take the
                // WMMA / scalar fused-dequant path instead.
                let is_rdna2 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_rdna2 && m == 1 && !dot_disabled && k % 256 == 0 {
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.write().unwrap_or_else(|e| e.into_inner());
                    let need_alloc = match buf_guard.as_ref() {
                        Some(s) => s.bytes < q81_bytes,
                        None => true,
                    };
                    if need_alloc {
                        *buf_guard = Some(RocmStorage::alloc_gpu(
                            &shape,
                            DType {
                                arith: ArithType::U8,
                                storage: DTypeStorage::Native,
                            },
                            &self.allocator,
                            self.ordinal,
                        )?);
                    }
                    let a_prequant = a_storage.dtype().arith == ArithType::U8;
                    if a_prequant {
                        drop(buf_guard);
                        self.launch_dot4_q3k_q81_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let q_stream = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        fence_act_quant(self, q_stream);
                        self.launch_dot4_q3k_q81_gemv(act_q81, b_storage, &out_storage, m, n, k)?;
                    }
                } else if is_rdna34 && wmma_quant_tile_ok(wave32, m, n, k, 16) {
                    self.launch_wmma_fused_dequant_q3k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_fused_dequant_gemm_q3k(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XXS) => {
                self.launch_iq_wmma_fallback(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    Self::launch_wmma_fused_dequant_iq2xxs,
                    Self::launch_fused_dequant_gemm_iq2xxs,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XS) => {
                self.launch_iq_wmma_fallback(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    Self::launch_wmma_fused_dequant_iq2xs,
                    Self::launch_fused_dequant_gemm_iq2xs,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2S) => {
                self.launch_iq_wmma_fallback(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    Self::launch_wmma_fused_dequant_iq2s,
                    Self::launch_fused_dequant_gemm_iq2s,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3XXS) => {
                self.launch_iq_wmma_fallback(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    Self::launch_wmma_fused_dequant_iq3xxs,
                    Self::launch_fused_dequant_gemm_iq3xxs,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3S) => {
                self.launch_iq_wmma_fallback(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    Self::launch_wmma_fused_dequant_iq3s,
                    Self::launch_fused_dequant_gemm_iq3s,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4NL) => {
                self.launch_iq_wmma_fallback(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    Self::launch_wmma_fused_dequant_iq4nl,
                    Self::launch_fused_dequant_gemm_iq4nl,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4XS) => {
                self.launch_iq_wmma_fallback(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    Self::launch_wmma_fused_dequant_iq4xs,
                    Self::launch_fused_dequant_gemm_iq4xs,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q80) => {
                // Q8_0 uses the fused dequant+GEMM kernel (34-byte blocks → F32), matching
                // the other KQuant schemes rather than falling back to dequant+matmul.
                // SPEED-ROC: On RDNA3/4, use the WMMA fused-dequant kernel for decode (M=1)
                // and small prefill. Falls back to scalar/LDS-tiled for larger prefill.
                let is_rdna34 = self.is_rdna34;
                // SPEED-ROC: measured A/B — fp32 GEMv (20.5 ms/tok) LOSES to the
                // 4-tile WMMA kernel (14.3 ms/tok) at m=1: RDNA3/4 removed the
                // scalar `dot1-insts` feature (`__builtin_amdgcn_sdot4` needs
                // dot1-insts, absent on gfx1100/gfx1201), so the int8-dot GEMV
                // kernel can't compile. (The VOP3 vector `v_dot4_i32_i8` IS still
                // present on RDNA4, but WMMA tensor tiles are faster.) GEMV kernel
                // source removed from the JIT aggregate (P3+ cleanup).
                if is_rdna34 && m <= wmma_max_m {
                    // Decode/small-prefill: WMMA kernel is fastest for small M.
                    // SPEED-DOT (space-balls.md): M=1 decode GEMV via VOP3
                    // `v_dot4_i32_i8` — WMMA wastes 15/16 rows at m=1 (rows
                    // 1..15 multiply zeroed A, 6.25% tensor utilization); the
                    // dot4 kernel runs one wave per output column at 100%
                    // utilization with the Q8_0 scale hoisted out of the loop.
                    // SPEED-DOT: Q8_0 x Q8_1 GEMV via V_DOT4_I32_IU8 (__builtin_amdgcn_sudot4).
                    // Active by default for m <= wmma_max_m unless disabled via GRIM_DOT_GEMV=0.
                    static DOT_GEMV_CFG: std::sync::OnceLock<(bool, bool)> =
                        std::sync::OnceLock::new();
                    let (dot_disabled, use_legacy_dot2) = *DOT_GEMV_CFG.get_or_init(|| {
                        (
                            matches!(
                                std::env::var("GRIM_DOT_GEMV").as_deref(),
                                Ok("0" | "false" | "off")
                            ),
                            matches!(
                                std::env::var("GRIM_DOT_GEMV_LEGACY").as_deref(),
                                Ok("1" | "true" | "on")
                            ),
                        )
                    });
                    let dot4_max_m: usize = std::env::var("GRIM_PREFILL_DOT4_M_MAX")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(64);
                    if !dot_disabled && m <= dot4_max_m && !Self::is_fp16_activation(a_storage) {
                        // N4: K80 fallback — round K down to a 32-multiple. A
                        // tightly-packed Q8_0 buffer stores only K/32 full
                        // blocks, so the missing tail weight is zero by
                        // convention; dot4 over K_aligned is therefore exact.
                        let k_aligned = k - (k % 32);
                        if k_aligned >= 32 && !(use_legacy_dot2 && m == 1) {
                            let q81_bytes = (k_aligned / 32) * 36 * m;
                            let shape = Shape::new(vec![q81_bytes]);
                            let mut buf_guard =
                                self.act_q81_buf.write().unwrap_or_else(|e| e.into_inner());
                            let need_alloc = match buf_guard.as_ref() {
                                Some(s) => s.bytes < q81_bytes,
                                None => true,
                            };
                            if need_alloc {
                                *buf_guard = Some(RocmStorage::alloc_gpu(
                                    &shape,
                                    DType {
                                        arith: ArithType::U8,
                                        storage: DTypeStorage::Native,
                                    },
                                    &self.allocator,
                                    self.ordinal,
                                )?);
                            }
                            let a_prequant = a_storage.dtype().arith == ArithType::U8;
                            if a_prequant {
                                drop(buf_guard);
                                self.launch_dot4_q80_q81_gemv(
                                    a_storage,
                                    b_storage,
                                    &out_storage,
                                    m,
                                    n,
                                    k_aligned,
                                )?;
                            } else {
                                // Task 7 (Plan 3): Small-batch prefill & decode direct f32act dot4 GEMV.
                                // Directly quantizes in registers and computes GEMV without standalone quantize launch.
                                drop(buf_guard);
                                self.launch_dot4_q80_f32act_gemv(
                                    a_storage,
                                    b_storage,
                                    &out_storage,
                                    m,
                                    n,
                                    k_aligned,
                                )?;
                            }
                        } else if use_legacy_dot2 && m == 1 && k % 32 == 0 {
                            let act_f16 = RocmStorage::alloc_gpu(
                                &Shape::new(vec![k]),
                                DType {
                                    arith: ArithType::F16,
                                    storage: DTypeStorage::Native,
                                },
                                &self.allocator,
                                self.ordinal,
                            )?;
                            let _ = self.quantize_fp16(a_storage, &act_f16)?;
                            self.launch_dot2_q80_gemv(&act_f16, b_storage, &out_storage, n, k)?;
                            drop(act_f16);
                        } else {
                            // Q8_1 format: 36 bytes per 32-element block
                            let q81_bytes = (k / 32) * 36 * m;
                            let shape = Shape::new(vec![q81_bytes]);
                            let mut buf_guard =
                                self.act_q81_buf.write().unwrap_or_else(|e| e.into_inner());
                            let need_alloc = match buf_guard.as_ref() {
                                Some(s) => s.bytes < q81_bytes,
                                None => true,
                            };
                            if need_alloc {
                                *buf_guard = Some(RocmStorage::alloc_gpu(
                                    &shape,
                                    DType {
                                        arith: ArithType::U8,
                                        storage: DTypeStorage::Native,
                                    },
                                    &self.allocator,
                                    self.ordinal,
                                )?);
                            }
                            // SPEED-DOT-OPFUSE: a U8-typed activation is an
                            // already-packed q8_1 buffer (produced by the fused
                            // grim_rmsnorm_quant_i8 in the model layer) — skip
                            // the quantize launch entirely.
                            let a_prequant = a_storage.dtype().arith == ArithType::U8;
                            if a_prequant {
                                drop(buf_guard);
                                self.launch_dot4_q80_q81_gemv(
                                    a_storage,
                                    b_storage,
                                    &out_storage,
                                    m,
                                    n,
                                    k,
                                )?;
                            } else {
                                let act_q81 = buf_guard.as_ref().unwrap();
                                let _ = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                                self.launch_dot4_q80_q81_gemv(
                                    act_q81,
                                    b_storage,
                                    &out_storage,
                                    m,
                                    n,
                                    k,
                                )?;
                            }
                        }
                    }
                    // SPEED-ROC: if the activation is already FP16 in global
                    // memory, use the FP16-input kernel to halve A-read bandwidth.
                    else if Self::is_fp16_activation(a_storage) {
                        self.launch_wmma_fused_dequant_q8_0_fp16(
                            a_storage,
                            b_storage,
                            &out_storage,
                            m,
                            n,
                            k,
                        )?;
                    } else if std::env::var("GRIM_FP16_ACT").as_deref() == Ok("1") {
                        // Env-gated pre-quantize path: convert FP32 activations
                        // to FP16 on-device, then run the FP16-input WMMA
                        // kernel.  Proves the bandwidth-saving dispatch without
                        // requiring callers to materialize FP16 activations.
                        let fp16_buf = RocmStorage::alloc_gpu(
                            a_storage.shape(),
                            DType {
                                arith: ArithType::F16,
                                storage: DTypeStorage::Native,
                            },
                            &self.allocator,
                            self.ordinal,
                        )?;
                        let q_stream = self.quantize_fp16(a_storage, &fp16_buf)?;
                        // Quantize and GEMM run on the active stream in order;
                        // synchronize the quantize before reusing a_storage is
                        // unnecessary — a_storage is read-only here.
                        let _ = q_stream;
                        // Optional self-verify (GRIM_FP16_VERIFY=1): dequantize
                        // the FP16 buffer back to FP32 and confirm the
                        // round-trip matches the original activation within
                        // FP16 tolerance.  This exercises dequantize_fp16 and
                        // pins the quantize/dequantize kernel contract.
                        if std::env::var("GRIM_FP16_VERIFY").as_deref() == Ok("1") {
                            let n_el = a_storage.shape().elem_count();
                            let verify_buf = RocmStorage::alloc_gpu(
                                a_storage.shape(),
                                dtype_f32(),
                                &self.allocator,
                                self.ordinal,
                            )?;
                            let dv = self.dequantize_fp16(&fp16_buf, &verify_buf)?;
                            let _ = dv;
                            let original = a_storage.to_cpu_vec_f32()?;
                            let roundtrip = verify_buf.to_cpu_vec_f32()?;
                            drop(verify_buf);
                            let mut max_diff = 0.0f32;
                            for i in 0..n_el {
                                max_diff = max_diff.max((original[i] - roundtrip[i]).abs());
                            }
                            if max_diff > 1e-3_f32 {
                                return Err(Error::Backend(format!(
                                    "FP16 activation round-trip exceeded tol: max_diff={max_diff}"
                                )));
                            }
                            if std::env::var_os("GRIM_QMM_TRACE").is_some() {
                                eprintln!("[qmm] FP16 act round-trip OK max_diff={max_diff}");
                            }
                        }
                        self.launch_wmma_fused_dequant_q8_0_fp16(
                            &fp16_buf,
                            b_storage,
                            &out_storage,
                            m,
                            n,
                            k,
                        )?;
                        // fp16_buf dropped after the GEMM launch is enqueued;
                        // single-stream ordering keeps it live long enough.
                        drop(fp16_buf);
                    } else {
                        self.launch_wmma_fused_dequant_q8_0(
                            a_storage,
                            b_storage,
                            &out_storage,
                            m,
                            n,
                            k,
                        )?;
                    }
                } else {
                    self.launch_fused_dequant_gemm_q8_0(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::Block(BlockDtype::Fp8)
            | DTypeStorage::FloatPack(FloatPackScheme::Fp8) => {
                // Phase 4.5c: M=1 decode routes to the fp8 dot4 GEMV (RDNA4
                // dot11-insts V_DOT4_F32_FP8_FP8; activations quantized to
                // E4M3 in-register). WMMA/MFMA GEMM stays the prefill path.
                // Escape hatch: GRIM_DOT_GEMV=0.
                let is_rdna34 = self.is_rdna34;
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_rdna34 && m == 1 && !dot_disabled && k % 32 == 0 {
                    self.launch_dot4_fp8_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                }
                // gfx1200+ uses MFMA for FP8 throughput; other architectures use scalar.
                else if self.gpu_target.starts_with("gfx12") {
                    self.launch_fused_dequant_gemm_fp8_mfma(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_fused_dequant_gemm_fp8(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::FloatPack(FloatPackScheme::MxFp4) => {
                // The MXFP4 kernel reads one E8M0 exponent per 32-element block (block_idx = (col*K+k)/32) and expects B_codes / B_exps as separate device buffers.
                // Weight tensors carry the length-prefixed framing [u64 codes_len][codes][u64 exps_len][exps]; both segment lengths are derivable from.
                let elems = k * n;
                let codes_len = elems / 2;
                let exps_len = elems.div_ceil(32);
                let framed_len = 16 + codes_len + exps_len;
                let base = b_storage
                    .device_ptr_u64()
                    .ok_or_else(|| Error::Backend("mxfp4 gemm: b has no device ptr".into()))?;

                // Keep any transient exponent storage alive until after the kernel launch
                // below (single-stream ordering makes pooled reuse safe once this binding drops).
                let exps_storage: Option<RocmStorage>;
                let (codes_ptr, exps_ptr): (u64, u64) = if b_storage.bytes == framed_len {
                    exps_storage = None;
                    (base + 8, base + 16 + codes_len as u64)
                } else if !_b_scales.is_empty() {
                    // Caller-supplied f32 E8M0 byte values as exponents.
                    let exps_u8: Vec<u8> = _b_scales
                        .iter()
                        .map(|s| s.round().clamp(0.0, 255.0) as u8)
                        .collect();
                    let storage = RocmStorage::copy_from_host_raw_bytes(
                        &exps_u8,
                        &Shape::new(vec![exps_u8.len()]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?;
                    let ptr = storage
                        .device_ptr_u64()
                        .ok_or_else(|| Error::Backend("mxfp4 gemm: exps upload failed".into()))?;
                    exps_storage = Some(storage);
                    (base, ptr)
                } else {
                    // Legacy/empty path: zeroed dummy exponents.
                    let storage = RocmStorage::alloc_gpu(
                        &Shape::new(vec![exps_len.max(1)]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?;
                    let ptr = storage
                        .device_ptr_u64()
                        .ok_or_else(|| Error::Backend("mxfp4 gemm: dummy exps failed".into()))?;
                    exps_storage = Some(storage);
                    (base, ptr)
                };

                let use_fused = self
                    .mxfp4_fused_dequant_gemm_enabled
                    .load(Ordering::Relaxed);
                if use_fused {
                    self.launch_fused_dequant_gemm_mxfp4(
                        a_storage,
                        codes_ptr,
                        exps_ptr,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_mxfp4_gemm_tiled(
                        a_storage,
                        codes_ptr,
                        exps_ptr,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
                }
                // Keep any transient exponent storage alive until the kernel launch(es) above are enqueued on the active stream.
                // Pooled reuse is only safe once the transitive storage drops.
                drop(exps_storage);
            }
            DTypeStorage::FloatPack(FloatPackScheme::MxFp8) => {
                let dummy_exps = RocmStorage::alloc_gpu(
                    &Shape::new(vec![(k * n).max(32) / 32]),
                    DType {
                        arith: ArithType::U8,
                        storage: DTypeStorage::Native,
                    },
                    &self.allocator,
                    self.ordinal,
                )?;
                self.launch_fused_dequant_gemm_mxfp8(
                    a_storage,
                    b_storage,
                    &dummy_exps,
                    &out_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::FloatPack(FloatPackScheme::NvFp4) => {
                if m <= 4 {
                    self.launch_nvfp4_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                } else {
                    self.launch_nvfp4_gemm_tiled(a_storage, b_storage, &out_storage, m, n, k)?;
                }
            }
            DTypeStorage::FloatPack(FloatPackScheme::NutFp4) => {
                if m <= 4 {
                    self.launch_nutcracker_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                } else {
                    self.launch_nutcracker_gemm_tiled(a_storage, b_storage, &out_storage, m, n, k)?;
                }
            }
            DTypeStorage::ResidualPacked(cfg) => {
                // Generic variable-bitwidth packed + residual layout (WI-C / WI-T8): [see: `grim_fused_dequant_gemm_f16`, `enabled`]
                // Lock-free enabled check via AtomicBool shadow. [see: `fused_dequant_gemm_enabled`, `set_fused_dequant_gemm_enabled`]
                if !self.fused_dequant_gemm_enabled.load(Ordering::Relaxed) {
                    FUSED_FORWARD_DISPATCH_STATS
                        .fallback_calls
                        .fetch_add(1, Ordering::Relaxed);
                    grim_core::emit_fallback(
                        "grim-backend-rocm/quant_matmul",
                        grim_core::FallbackReason::FusedQuantGemmDisabled,
                        format!("forward m={m} n={n} k={k}"),
                    );
                    return self.matmul(a, b_packed, out_shape);
                }
                let out_f32 =
                    RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
                let residuals = grim_tensor::QuantizedMatmulBackwardResiduals::from_provenance(
                    &b_storage.provenance(),
                );
                let provenance = b_storage.provenance();
                let (primary_bytes, outlier_indices, outlier_values) = match provenance {
                    grim_tensor::QuantProvenance::WithResiduals {
                        primary_scale_bytes,
                        outlier_indices,
                        outlier_values_bits,
                        ..
                    } => (
                        primary_scale_bytes,
                        outlier_indices,
                        outlier_values_bits
                            .into_iter()
                            .map(f32::from_bits)
                            .collect::<Vec<_>>(),
                    ),
                    _ => (Vec::new(), Vec::new(), Vec::new()),
                };
                let scales_storage = if primary_bytes.is_empty() {
                    None
                } else {
                    Some(RocmStorage::copy_from_host_raw_bytes(
                        &primary_bytes,
                        &Shape::from_slice(&[primary_bytes.len()]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?)
                };
                let index_bytes: Vec<u8> = outlier_indices
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect();
                let value_bytes: Vec<u8> = outlier_values
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect();
                let indices_storage = if outlier_indices.is_empty() {
                    None
                } else {
                    Some(RocmStorage::copy_from_host_raw_bytes(
                        &index_bytes,
                        &Shape::from_slice(&[outlier_indices.len()]),
                        DType {
                            arith: ArithType::U32,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?)
                };
                let values_storage = if outlier_values.is_empty() {
                    None
                } else {
                    Some(RocmStorage::copy_from_host_raw_bytes(
                        &value_bytes,
                        &Shape::from_slice(&[outlier_values.len()]),
                        DType::F32,
                        &self.allocator,
                        self.ordinal,
                    )?)
                };
                let scale_ptr = scales_storage
                    .as_ref()
                    .and_then(|s| s.device_ptr)
                    .map(|p| p as *const c_void)
                    .unwrap_or(std::ptr::null());
                let index_ptr = indices_storage
                    .as_ref()
                    .and_then(|s| s.device_ptr)
                    .map(|p| p as *const c_void)
                    .unwrap_or(std::ptr::null());
                let value_ptr = values_storage
                    .as_ref()
                    .and_then(|s| s.device_ptr)
                    .map(|p| p as *const c_void)
                    .unwrap_or(std::ptr::null());
                let stream = self.launch_fused_dequant_gemm_f16(
                    a_storage,
                    b_storage,
                    scale_ptr,
                    &out_f32,
                    m,
                    n,
                    k,
                    cfg.bpw,
                    residuals.outlier_count,
                    index_ptr,
                    value_ptr,
                    residuals.backup1_bpw,
                    residuals.backup1_codes_offset,
                    residuals.backup1_scale_offset,
                    residuals.backup2_bpw,
                    residuals.backup2_codes_offset,
                    residuals.backup2_scale_offset,
                )?;
                FUSED_FORWARD_DISPATCH_STATS
                    .kernel_calls
                    .fetch_add(1, Ordering::Relaxed);
                FUSED_FORWARD_DISPATCH_STATS
                    .last_backup2_bpw
                    .store(residuals.backup2_bpw as usize, Ordering::Relaxed);
                FUSED_FORWARD_DISPATCH_STATS
                    .last_backup2_codes_offset
                    .store(residuals.backup2_codes_offset, Ordering::Relaxed);
                FUSED_FORWARD_DISPATCH_STATS
                    .last_backup2_scale_offset
                    .store(residuals.backup2_scale_offset, Ordering::Relaxed);
                let handle: Box<dyn ComputeHandle> = Box::new(RocmHandle::new(Some(stream)));
                return Ok((Box::new(out_f32), handle));
            }
            DTypeStorage::GroupInt(cfg) => {
                // GPTQ/EfficientQAT fused dequant-GEMM: the packed four-segment blob stays resident on-device and the kernel dequantizes in-kernel (previously
                // this arm fell through to `_ =>`, forcing a host-side F32 inflation of every GPTQ weight).
                if !matches!(cfg.bits, 2 | 4 | 8) {
                    return Err(Error::Backend(format!(
                        "gptq quantized_matmul: unsupported bit width {}",
                        cfg.bits
                    )));
                }
                let (qw_off, qz_off, sc_off, gi_off, has_g_idx) =
                    Self::gptq_segment_offsets(cfg.bits, cfg.group_size, k, n, b_storage.bytes())?;
                self.launch_gptq_dequant_gemm(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    cfg.bits,
                    cfg.group_size,
                    has_g_idx,
                    qw_off,
                    qz_off,
                    sc_off,
                    gi_off,
                )?;
            }
            DTypeStorage::Awq(cfg) => {
                // AWQ fused dequant-GEMM: packed three-segment blob ([qweight][qzeros][scales(f16)]).
                if !matches!(cfg.bits, 2 | 4 | 8) {
                    return Err(Error::Backend(format!(
                        "awq quantized_matmul: unsupported bit width {}",
                        cfg.bits
                    )));
                }
                let (qw_off, qz_off, sc_off) =
                    Self::awq_segment_offsets(cfg.bits, cfg.group_size, k, n, b_storage.bytes())?;
                self.launch_awq_dequant_gemm(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    cfg.bits,
                    cfg.group_size,
                    qw_off,
                    qz_off,
                    sc_off,
                )?;
            }
            DTypeStorage::W4A4OstQuant(cfg) => {
                // OSTQuant native W4A4 GEMV via sudot8 on RDNA4:
                // resident packed blob: [u64 qw_len][qweight][u64 sc_len][scales][u64 zr_len][zeros]
                if k % 128 != 0 {
                    return Err(Error::Backend(format!(
                        "w4a4_ostquant quantized_matmul: K={k} must be divisible by 128"
                    )));
                }
                self.launch_w4a4_ostquant_gemv_blob(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    cfg.group_size,
                )?;
            }
            DTypeStorage::W4A16(w4) => {
                // Quant workstream wiring: Marlin-style fused W4A16 GEMM over
                // the resident blob ([codes u32][scales f32], B stored [N, K/8]).
                if k % 8 != 0 {
                    return Err(Error::Backend(format!(
                        "w4a16 quantized_matmul: K={k} must be divisible by 8"
                    )));
                }
                self.launch_marlin_gemm_w4a16_blob(
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    w4.group_size,
                )?;
            }
            DTypeStorage::WNA16 => {
                // Fused dequant-GEMM over the resident blob; the header
                // carries n_bit/num_blocks.
                let (n_bit_raw, blocks_raw) = Self::wna16_read_params(b_storage, self.ordinal)?;
                let mut n_bit_v = n_bit_raw as i32;
                let mut blocks_v = blocks_raw as i32;
                self.launch_elementwise_dequant_gemm(
                    "grim_wna16_dequant_gemm",
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    &mut [arg(&mut n_bit_v), arg(&mut blocks_v)],
                )?;
            }
            DTypeStorage::EmbeddingWNA16Int => {
                // Embedding tables ride the dequant-at-load service, not a
                // GEMM — fail loudly if routed here.
                return Err(Error::Backend(
                    "quantized_matmul: EmbeddingWNA16Int is an embedding format; dequantize at load (dequant_embedding_wna16_int_to_f32)"
                        .into(),
                ));
            }
            DTypeStorage::CompressedTensorsW8A8Int8 => {
                // SmoothQuant-style W8 (int8 codes + per-output-channel f32
                // scales); activations F32 on this path.
                self.launch_elementwise_dequant_gemm(
                    "grim_w8a8_int8_dequant_gemm",
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    &mut [],
                )?;
            }
            DTypeStorage::CompressedTensorsW8A8Fp8 => {
                // OCP E4M3 codes + per-tensor f32 scale; activations F32.
                self.launch_elementwise_dequant_gemm(
                    "grim_w8a8_fp8_dequant_gemm",
                    a_storage,
                    b_storage,
                    &out_storage,
                    m,
                    n,
                    k,
                    &mut [],
                )?;
            }
            DTypeStorage::Block(bd) => {
                // No fused block-quant GEMM on this backend. Falling through to
                // `matmul` would reinterpret packed codes as F32 and emit
                // garbage, so refuse explicitly and let the caller fall back.
                return Err(Error::Unimplemented(format!(
                    "ROCm quantized_matmul: no fused kernel for block format {bd:?}"
                )));
            }
            _ => {
                return self.matmul(a, b_packed, out_shape);
            }
        }

        let handle: Box<dyn ComputeHandle> = Box::new(ReadyHandle);
        Ok((Box::new(out_storage), handle))
    }

    /// WI-F5-close: fused dequant backward dispatch (the lattice point [see: `grim-autograd::matmul_backward`]
    fn quantized_matmul_backward_dx(
        &self,
        dy: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        default_bpw: u8,
        m: usize,
        n: usize,
        k: usize,
        out_shape: &Shape,
        residuals: Option<&grim_tensor::QuantizedMatmulBackwardResiduals>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        FUSED_BACKWARD_DISPATCH_STATS
            .attempts
            .fetch_add(1, Ordering::Relaxed);

        // Both operands must already be ROCm-resident for the kernel to run
        let dy_storage = match dy.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => {
                return Err(Error::Backend(
                    "quantized_matmul_backward_dx: dy not ROCm-resident; CPU fallback expected"
                        .into(),
                ));
            }
        };
        let b_storage = match b_packed.as_any().downcast_ref::<RocmStorage>() {
            Some(s) => s,
            None => return Err(Error::Backend(
                "quantized_matmul_backward_dx: b_packed not ROCm-resident; CPU fallback expected"
                    .into(),
            )),
        };

        // Extract residual metadata or use defaults when absent.
        let outlier_count = residuals.map(|r| r.outlier_count).unwrap_or(0);
        let outlier_indices_ptr = residuals
            .and_then(|r| {
                if r.outlier_count > 0 {
                    Some(r.outlier_indices_ptr())
                } else {
                    None
                }
            })
            .unwrap_or(std::ptr::null());
        let outlier_values_ptr = residuals
            .and_then(|r| {
                if r.outlier_count > 0 {
                    Some(r.outlier_values_ptr())
                } else {
                    None
                }
            })
            .unwrap_or(std::ptr::null());
        let backup1_bpw = residuals.map(|r| r.backup1_bpw).unwrap_or(0);
        let backup1_codes_offset = residuals.map(|r| r.backup1_codes_offset).unwrap_or(0);
        let backup1_scale_offset = residuals.map(|r| r.backup1_scale_offset).unwrap_or(0);
        let backup2_bpw = residuals.map(|r| r.backup2_bpw).unwrap_or(0);
        let backup2_codes_offset = residuals.map(|r| r.backup2_codes_offset).unwrap_or(0);
        let backup2_scale_offset = residuals.map(|r| r.backup2_scale_offset).unwrap_or(0);

        // Allocate the dX output buffer (f32 row-major [M, K]).
        let dx_storage = match out_shape.dims() {
            &[mm, kk] if mm == m && kk == k => RocmStorage::alloc_gpu(
                out_shape,
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?,
            other => {
                return Err(Error::Shape(format!(
                    "quantized_matmul_backward_dx: out_shape must be [{m},{k}], got {:?}",
                    other
                )));
            }
        };

        // Pack scales into a temporary ROCm buffer so the kernel can reach them.
        // `grim_fused_dequant_backward_gemm_f16` (and the matching forward `grim_fused_dequant_gemm_f16`) read scales as `const unsigned char*` and divide by.
        let scales_storage = if b_scales.is_empty() {
            None
        } else {
            let byte_scales: Vec<u8> = b_scales
                .iter()
                .map(|&s| {
                    let n = s.clamp(0.0f32, 1.0f32) * 255.0f32;
                    n.round().clamp(0.0f32, 255.0f32) as u8
                })
                .collect();
            Some(RocmStorage::copy_from_host_raw_bytes(
                &byte_scales,
                &Shape::new(vec![byte_scales.len()]),
                DType {
                    arith: ArithType::U8,
                    storage: DTypeStorage::Native,
                },
                &self.allocator,
                self.ordinal,
            )?)
        };
        let b_scales_ptr = match &scales_storage {
            Some(s) => match s.device_ptr {
                Some(raw) => raw as *const c_void,
                None => {
                    return Err(Error::Backend(
                        "quantized_matmul_backward_dx: scales alloc missing gpu ptr".into(),
                    ));
                }
            },
            None => std::ptr::null(),
        };

        // Call the actual kernel based on quantization storage type (not bpw,
        use grim_tensor::{BlockDtype, FloatPackScheme, KQuantScheme};
        match b_storage.dtype().storage {
            DTypeStorage::KQuant(KQuantScheme::Q4K) => {
                self.launch_fused_dequant_backward_gemm_q4k(
                    dy_storage,
                    b_storage,
                    b_scales_ptr,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q80) => {
                self.launch_fused_dequant_backward_gemm_q8_0(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q5K) => {
                self.launch_fused_dequant_backward_gemm_q5k(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q6K) => {
                self.launch_fused_dequant_backward_gemm_q6k(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q2K) => {
                self.launch_fused_dequant_backward_gemm_q2k(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q3K) => {
                self.launch_fused_dequant_backward_gemm_q3k(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XXS) => {
                self.launch_fused_dequant_backward_gemm_iq2xxs(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XS) => {
                self.launch_fused_dequant_backward_gemm_iq2xs(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2S) => {
                self.launch_fused_dequant_backward_gemm_iq2s(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3XXS) => {
                self.launch_fused_dequant_backward_gemm_iq3xxs(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3S) => {
                self.launch_fused_dequant_backward_gemm_iq3s(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4NL) => {
                self.launch_fused_dequant_backward_gemm_iq4nl(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4XS) => {
                self.launch_fused_dequant_backward_gemm_iq4xs(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::Block(BlockDtype::Fp8)
            | DTypeStorage::FloatPack(FloatPackScheme::Fp8) => {
                if self.gpu_target.starts_with("gfx12") {
                    self.launch_fused_dequant_backward_gemm_fp8_mfma(
                        dy_storage,
                        b_storage,
                        &dx_storage,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.launch_fused_dequant_backward_gemm_fp8(
                        dy_storage,
                        b_storage,
                        &dx_storage,
                        m,
                        n,
                        k,
                    )?;
                }
            }
            DTypeStorage::FloatPack(FloatPackScheme::MxFp4) => {
                let exps_storage = if !b_scales.is_empty() {
                    let exps_u8: Vec<u8> = b_scales
                        .iter()
                        .map(|s| s.round().clamp(0.0, 255.0) as u8)
                        .collect();
                    RocmStorage::copy_from_host_raw_bytes(
                        &exps_u8,
                        &Shape::new(vec![exps_u8.len()]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?
                } else {
                    RocmStorage::alloc_gpu(
                        &Shape::new(vec![(k * n).max(32) / 32]),
                        DType {
                            arith: ArithType::U8,
                            storage: DTypeStorage::Native,
                        },
                        &self.allocator,
                        self.ordinal,
                    )?
                };
                self.launch_mxfp4_backward_gemm(
                    dy_storage,
                    b_storage,
                    &exps_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                )?;
            }
            DTypeStorage::ResidualPacked(cfg) => {
                // Mirror the forward `enabled` gate: when the fused backward path is disabled, fall back to a standard matmul of dY against the transposed dequantized B (same behavior as the forward fallback at line ~2252).
                // This fixes the asymmetry where the forward dispatch honors `FusedDequantGemmConfig::enabled` but the backward dispatch unconditionally.
                if !self.fused_dequant_gemm_enabled.load(Ordering::Relaxed) {
                    FUSED_BACKWARD_DISPATCH_STATS
                        .fallback_calls
                        .fetch_add(1, Ordering::Relaxed);
                    grim_core::emit_fallback(
                        "grim-backend-rocm/quant_matmul",
                        grim_core::FallbackReason::FusedQuantGemmDisabled,
                        format!("backward m={m} n={n} k={k}"),
                    );
                    return self.matmul(dy, b_packed, out_shape);
                }
                self.launch_fused_dequant_backward_gemm_f16(
                    dy_storage,
                    b_storage,
                    b_scales_ptr,
                    &dx_storage,
                    m,
                    n,
                    k,
                    cfg.bpw,
                    outlier_count,
                    outlier_indices_ptr,
                    outlier_values_ptr,
                    backup1_bpw,
                    backup1_codes_offset,
                    backup1_scale_offset,
                    backup2_bpw,
                    backup2_codes_offset,
                    backup2_scale_offset,
                )?;
            }
            DTypeStorage::Native => {
                // Unquantized weights: no dequant needed. Use straight matmul (dY @ B^T).
                return self.matmul(dy, b_packed, out_shape);
            }
            DTypeStorage::GroupInt(cfg) => {
                // GPTQ/EfficientQAT fused dequant backward: mirror the forward GroupInt arm instead of falling into the ResidualPacked-style `_`
                // catch-all, whose kernel ABI (u8 scales / backup regions) does not match the GPTQ packed layout.
                if !matches!(cfg.bits, 2 | 4 | 8) {
                    return Err(Error::Backend(format!(
                        "gptq quantized_matmul_backward_dx: unsupported bit width {}",
                        cfg.bits
                    )));
                }
                let (qw_off, qz_off, sc_off, gi_off, has_g_idx) =
                    Self::gptq_segment_offsets(cfg.bits, cfg.group_size, k, n, b_storage.bytes())?;
                self.launch_gptq_dequant_backward_gemm(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                    cfg.bits,
                    cfg.group_size,
                    has_g_idx,
                    qw_off,
                    qz_off,
                    sc_off,
                    gi_off,
                )?;
            }
            DTypeStorage::Awq(cfg) => {
                // AWQ fused dequant backward dX:
                if !matches!(cfg.bits, 2 | 4 | 8) {
                    return Err(Error::Backend(format!(
                        "awq quantized_matmul_backward_dx: unsupported bit width {}",
                        cfg.bits
                    )));
                }
                let (qw_off, qz_off, sc_off) =
                    Self::awq_segment_offsets(cfg.bits, cfg.group_size, k, n, b_storage.bytes())?;
                self.launch_awq_dequant_backward_gemm(
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                    cfg.bits,
                    cfg.group_size,
                    qw_off,
                    qz_off,
                    sc_off,
                )?;
            }
            DTypeStorage::W4A16(_)
            | DTypeStorage::EmbeddingWNA16Int
            | DTypeStorage::W4A4OstQuant(_) => {
                // Weight-only (W4A16), embedding (EmbeddingWNA16Int), and W4A4 (W4A4OstQuant) formats are
                // inference-only: no weight-gradient kernel exists. Fail loudly.
                return Err(Error::Backend(
                    "quantized_matmul_backward_dx: W4A16/EmbeddingWNA16Int have no backward \
                     kernel; these are inference-only"
                        .into(),
                ));
            }
            DTypeStorage::WNA16 => {
                // Fused dequant-GEMM backward: dX[M, K] = dY[M, N] @ deq(B)[N, K].
                let (n_bit_raw, blocks_raw) = Self::wna16_read_params(b_storage, self.ordinal)?;
                let mut n_bit_v = n_bit_raw as i32;
                let mut blocks_v = blocks_raw as i32;
                self.launch_elementwise_dequant_gemm_backward(
                    "grim_wna16_dequant_gemm_backward_dx",
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                    &mut [arg(&mut n_bit_v), arg(&mut blocks_v)],
                )?;
            }
            DTypeStorage::CompressedTensorsW8A8Int8 => {
                self.launch_elementwise_dequant_gemm_backward(
                    "grim_w8a8_int8_dequant_gemm_backward_dx",
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                    &mut [],
                )?;
            }
            DTypeStorage::CompressedTensorsW8A8Fp8 => {
                self.launch_elementwise_dequant_gemm_backward(
                    "grim_w8a8_fp8_dequant_gemm_backward_dx",
                    dy_storage,
                    b_storage,
                    &dx_storage,
                    m,
                    n,
                    k,
                    &mut [],
                )?;
            }
            _ => {
                self.launch_fused_dequant_backward_gemm_f16(
                    dy_storage,
                    b_storage,
                    b_scales_ptr,
                    &dx_storage,
                    m,
                    n,
                    k,
                    default_bpw,
                    outlier_count,
                    outlier_indices_ptr,
                    outlier_values_ptr,
                    backup1_bpw,
                    backup1_codes_offset,
                    backup1_scale_offset,
                    backup2_bpw,
                    backup2_codes_offset,
                    backup2_scale_offset,
                )?;
            }
        }

        FUSED_BACKWARD_DISPATCH_STATS
            .kernel_calls
            .fetch_add(1, Ordering::Relaxed);

        let handle: Box<dyn ComputeHandle> = Box::new(ReadyHandle);
        Ok((Box::new(dx_storage), handle))
    }
}

/// WMMA tiled-quant dispatch gate (RDNA3/4, wave32 only).
/// The tiled kernels use 128-thread (4x wave32) blocks with cross-wave LDS
/// sharing, which is only correct when every output tile is whole:
/// M % 16 == 0, N % 64 == 0, K % blk == 0 (one full super-block per row).
/// Wave64 (CDNA) devices never qualify. Everything else takes the scalar
/// fused-dequant path, which is exact for arbitrary shapes.
///
/// Fence the activation-quantize launch against the subsequent dot4 GEMV.
/// On the cold-JIT path the quantize kernel and the GEMV can end up on
/// different streams/contexts; the GEMV then reads the (zeroed) activation
/// buffer before the quantize kernel's writes land. Record an event on the
/// quantize stream and make the compute stream wait on it.
fn fence_act_quant(dev: &RocmDevice, quantize_stream: *mut c_void) {
    use crate::device::handles::{
        hipEventCreate, hipEventDestroy, hipEventRecord, hipStreamWaitEvent, hipSuccess,
    };
    use std::ptr::null_mut;
    let _dev_guard = crate::device::util::DeviceGuard::set(dev.ordinal as i32);
    unsafe {
        let mut ev: *mut c_void = null_mut();
        if hipEventCreate(&mut ev) == hipSuccess {
            if hipEventRecord(ev, quantize_stream) == hipSuccess {
                let _ = hipStreamWaitEvent(dev.active_stream(), ev, 0);
            }
            let _ = hipEventDestroy(ev);
        }
    }
}

fn wmma_quant_tile_ok(wave32: bool, m: usize, n: usize, k: usize, blk: usize) -> bool {
    wave32 && m % 16 == 0 && n % 64 == 0 && k % blk == 0
}
