//! Quantization operations and quantized GEMM dispatch for `RocmDevice`.

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use grim_tensor::backend::{ComputeHandle, ReadyHandle};
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, CoreTensorOps, QuantOps, Shape};

use crate::device::gemm_tuning::lookup_solution_index;
use crate::device::roc_device::{
    FUSED_BACKWARD_DISPATCH_STATS, FUSED_FORWARD_DISPATCH_STATS, RocmDevice,
};
use crate::memory::pinned::RocmPinnedBuffer;
use crate::memory::storage::RocmStorage;
use crate::{
    HipDim3, HipMemcpyKind, RocmHandle, arg, as_rocm, check_hip, dev_ptr, dtype_f32,
    hipMemcpyAsync, hipStreamSynchronize, linear_launch,
};

impl QuantOps for RocmDevice {
    fn quantize(
        &self,
        x: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<Box<dyn BackendStorage>> {
        let (out, _handle) = self.quantize_on_device(x, format)?;
        Ok(out)
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
        let wmma_max_m: usize = std::env::var("GRIM_WMM_MAX_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);

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
                let is_rdna34 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                let is_dot4_arch = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                        | crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_dot4_arch && m == 1 && !dot_disabled && k % 256 == 0 {
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.lock().unwrap_or_else(|e| e.into_inner());
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
                        self.launch_dot4_q4k_q81_gemv(
                            a_storage, b_storage, &out_storage, m, n, k,
                        )?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let _ = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        self.launch_dot4_q4k_q81_gemv(
                            act_q81, b_storage, &out_storage, m, n, k,
                        )?;
                    }
                } else if is_rdna34 && m <= wmma_max_m {
                    self.launch_wmma_fused_dequant_q4k(
                        a_storage, b_storage, &out_storage, m, n, k,
                    )?;
                } else {
                    self.launch_fused_dequant_gemm_q4k(a_storage, b_storage, &out_storage, m, n, k)?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::Q5K) => {
                let is_rdna34 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                let is_dot4_arch = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                        | crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_dot4_arch && m == 1 && !dot_disabled && k % 256 == 0 {
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.lock().unwrap_or_else(|e| e.into_inner());
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
                        self.launch_dot4_q5k_q81_gemv(
                            a_storage, b_storage, &out_storage, m, n, k,
                        )?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let _ = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        self.launch_dot4_q5k_q81_gemv(
                            act_q81, b_storage, &out_storage, m, n, k,
                        )?;
                    }
                } else if is_rdna34 && m <= wmma_max_m {
                    self.launch_wmma_fused_dequant_q5k(a_storage, b_storage, &out_storage, m, n, k)?;
                } else {
                    self.launch_fused_dequant_gemm_q5k(a_storage, b_storage, &out_storage, m, n, k)?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::Q6K) => {
                let is_rdna34 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                let is_dot4_arch = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                        | crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_dot4_arch && m == 1 && !dot_disabled && k % 256 == 0 {
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.lock().unwrap_or_else(|e| e.into_inner());
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
                        self.launch_dot4_q6k_q81_gemv(
                            a_storage, b_storage, &out_storage, m, n, k,
                        )?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let _ = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        self.launch_dot4_q6k_q81_gemv(
                            act_q81, b_storage, &out_storage, m, n, k,
                        )?;
                    }
                } else if is_rdna34 && m <= wmma_max_m {
                    self.launch_wmma_fused_dequant_q6k(a_storage, b_storage, &out_storage, m, n, k)?;
                } else {
                    self.launch_fused_dequant_gemm_q6k(a_storage, b_storage, &out_storage, m, n, k)?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::Q2K) => {
                let is_rdna34 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                let is_dot4_arch = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                        | crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_dot4_arch && m == 1 && !dot_disabled && k % 256 == 0 {
                    // Phase 4.5f: m==1 decode routes to the Q2_K dot4 GEMV.
                    // Activations must be pre-quantized to Q8_1; quantize on the fly otherwise.
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.lock().unwrap_or_else(|e| e.into_inner());
                    let need_alloc = match buf_guard.as_ref() {
                        Some(s) => s.bytes < q81_bytes,
                        None => true,
                    };
                    if need_alloc {
                        *buf_guard = Some(RocmStorage::alloc_gpu(
                            &shape,
                            DType { arith: ArithType::U8, storage: DTypeStorage::Native },
                            &self.allocator, self.ordinal,
                        )?);
                    }
                    let a_prequant = a_storage.dtype().arith == ArithType::U8;
                    if a_prequant {
                        drop(buf_guard);
                        self.launch_dot4_q2k_q81_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let _ = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        self.launch_dot4_q2k_q81_gemv(act_q81, b_storage, &out_storage, m, n, k)?;
                    }
                } else if is_rdna34 && m <= wmma_max_m {
                    self.launch_wmma_fused_dequant_q2k(a_storage, b_storage, &out_storage, m, n, k)?;
                } else {
                    self.launch_fused_dequant_gemm_q2k(a_storage, b_storage, &out_storage, m, n, k)?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::Q3K) => {
                let is_rdna34 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                // RDNA2 has V_DOT4_I32_I8 (signed x signed) — same builtin flags
                // as RDNA3/4 sudot4 usage; B operands are all < 128 so the
                // dot4 GEMV is sign-agnostic. WMMA stays RDNA3/4-only.
                let is_dot4_arch = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA2
                        | crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_dot4_arch && m == 1 && !dot_disabled && k % 256 == 0 {
                    let q81_bytes = (k / 32) * 36 * m;
                    let shape = Shape::new(vec![q81_bytes]);
                    let mut buf_guard = self.act_q81_buf.lock().unwrap_or_else(|e| e.into_inner());
                    let need_alloc = match buf_guard.as_ref() {
                        Some(s) => s.bytes < q81_bytes,
                        None => true,
                    };
                    if need_alloc {
                        *buf_guard = Some(RocmStorage::alloc_gpu(
                            &shape,
                            DType { arith: ArithType::U8, storage: DTypeStorage::Native },
                            &self.allocator, self.ordinal,
                        )?);
                    }
                    let a_prequant = a_storage.dtype().arith == ArithType::U8;
                    if a_prequant {
                        drop(buf_guard);
                        self.launch_dot4_q3k_q81_gemv(a_storage, b_storage, &out_storage, m, n, k)?;
                    } else {
                        let act_q81 = buf_guard.as_ref().unwrap();
                        let _ = self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                        self.launch_dot4_q3k_q81_gemv(act_q81, b_storage, &out_storage, m, n, k)?;
                    }
                } else if is_rdna34 && m <= wmma_max_m {
                    self.launch_wmma_fused_dequant_q3k(a_storage, b_storage, &out_storage, m, n, k)?;
                } else {
                    self.launch_fused_dequant_gemm_q3k(a_storage, b_storage, &out_storage, m, n, k)?;
                }
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XXS) => {
                self.launch_iq_wmma_fallback(
                    a_storage, b_storage, &out_storage, m, n, k,
                    Self::launch_wmma_fused_dequant_iq2xxs,
                    Self::launch_fused_dequant_gemm_iq2xxs,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2XS) => {
                self.launch_iq_wmma_fallback(
                    a_storage, b_storage, &out_storage, m, n, k,
                    Self::launch_wmma_fused_dequant_iq2xs,
                    Self::launch_fused_dequant_gemm_iq2xs,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ2S) => {
                self.launch_iq_wmma_fallback(
                    a_storage, b_storage, &out_storage, m, n, k,
                    Self::launch_wmma_fused_dequant_iq2s,
                    Self::launch_fused_dequant_gemm_iq2s,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3XXS) => {
                self.launch_iq_wmma_fallback(
                    a_storage, b_storage, &out_storage, m, n, k,
                    Self::launch_wmma_fused_dequant_iq3xxs,
                    Self::launch_fused_dequant_gemm_iq3xxs,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ3S) => {
                self.launch_iq_wmma_fallback(
                    a_storage, b_storage, &out_storage, m, n, k,
                    Self::launch_wmma_fused_dequant_iq3s,
                    Self::launch_fused_dequant_gemm_iq3s,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4NL) => {
                self.launch_iq_wmma_fallback(
                    a_storage, b_storage, &out_storage, m, n, k,
                    Self::launch_wmma_fused_dequant_iq4nl,
                    Self::launch_fused_dequant_gemm_iq4nl,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::IQ4XS) => {
                self.launch_iq_wmma_fallback(
                    a_storage, b_storage, &out_storage, m, n, k,
                    Self::launch_wmma_fused_dequant_iq4xs,
                    Self::launch_fused_dequant_gemm_iq4xs,
                )?;
            }
            DTypeStorage::KQuant(KQuantScheme::Q80) => {
                // Q8_0 uses the fused dequant+GEMM kernel (34-byte blocks → F32), matching
                // the other KQuant schemes rather than falling back to dequant+matmul.
                // SPEED-ROC: On RDNA3/4, use the WMMA fused-dequant kernel for decode (M=1)
                // and small prefill. Falls back to scalar/LDS-tiled for larger prefill.
                let is_rdna34 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
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
                    if !dot_disabled
                        && k % 32 == 0
                        && !Self::is_fp16_activation(a_storage)
                    {
                        if use_legacy_dot2 && m == 1 {
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
                            self.launch_dot2_q80_gemv(
                                &act_f16, b_storage, &out_storage, n, k,
                            )?;
                            drop(act_f16);
                        } else {
                            // Q8_1 format: 36 bytes per 32-element block
                            let q81_bytes = (k / 32) * 36 * m;
                            let shape = Shape::new(vec![q81_bytes]);
                            let mut buf_guard = self.act_q81_buf.lock().unwrap_or_else(|e| e.into_inner());
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
                            let a_prequant =
                                a_storage.dtype().arith == ArithType::U8;
                            if a_prequant {
                                drop(buf_guard);
                                self.launch_dot4_q80_q81_gemv(
                                    a_storage, b_storage, &out_storage, m, n, k,
                                )?;
                            } else {
                                let act_q81 = buf_guard.as_ref().unwrap();
                                let _ =
                                    self.launch_quantize_q8_1(a_storage, act_q81, m, k)?;
                                self.launch_dot4_q80_q81_gemv(
                                    act_q81, b_storage, &out_storage, m, n, k,
                                )?;
                            }
                        }
                    }
                    // SPEED-ROC: if the activation is already FP16 in global
                    // memory, use the FP16-input kernel to halve A-read bandwidth.
                    else if Self::is_fp16_activation(a_storage) {
                        self.launch_wmma_fused_dequant_q8_0_fp16(
                            a_storage, b_storage, &out_storage, m, n, k,
                        )?;
                    } else if std::env::var("GRIM_FP16_ACT").as_deref()
                        == Ok("1")
                    {
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
                        if std::env::var("GRIM_FP16_VERIFY").as_deref() == Ok("1")
                        {
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
                                max_diff =
                                    max_diff.max((original[i] - roundtrip[i]).abs());
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
                            &fp16_buf, b_storage, &out_storage, m, n, k,
                        )?;
                        // fp16_buf dropped after the GEMM launch is enqueued;
                        // single-stream ordering keeps it live long enough.
                        drop(fp16_buf);
                    } else {
                        self.launch_wmma_fused_dequant_q8_0(
                            a_storage, b_storage, &out_storage, m, n, k,
                        )?;
                    }
                } else {
                    self.launch_fused_dequant_gemm_q8_0(a_storage, b_storage, &out_storage, m, n, k)?;
                }
            }
            DTypeStorage::Block(BlockDtype::Fp8)
            | DTypeStorage::FloatPack(FloatPackScheme::Fp8) => {
                // Phase 4.5c: M=1 decode routes to the fp8 dot4 GEMV (RDNA4
                // dot11-insts V_DOT4_F32_FP8_FP8; activations quantized to
                // E4M3 in-register). WMMA/MFMA GEMM stays the prefill path.
                // Escape hatch: GRIM_DOT_GEMV=0.
                let is_rdna34 = matches!(
                    crate::quantization::gcn_arch(&self.gpu_target),
                    crate::quantization::GcnArch::RDNA3
                        | crate::quantization::GcnArch::RDNA4
                        | crate::quantization::GcnArch::UDNA
                );
                let dot_disabled = matches!(
                    std::env::var("GRIM_DOT_GEMV").as_deref(),
                    Ok("0" | "false" | "off")
                );
                if is_rdna34 && m == 1 && !dot_disabled && k % 32 == 0 {
                    self.launch_dot4_fp8_gemv(
                        a_storage,
                        b_storage,
                        &out_storage,
                        m,
                        n,
                        k,
                    )?;
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
            DTypeStorage::ResidualPacked(cfg) => {
                // Generic variable-bitwidth packed + residual layout (WI-C / WI-T8): [see: `grim_fused_dequant_gemm_f16`, `enabled`]
                // Lock-free enabled check via AtomicBool shadow. [see: `fused_dequant_gemm_enabled`, `set_fused_dequant_gemm_enabled`]
                if !self.fused_dequant_gemm_enabled.load(Ordering::Relaxed) {
                    FUSED_FORWARD_DISPATCH_STATS
                        .fallback_calls
                        .fetch_add(1, Ordering::Relaxed);
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
            DTypeStorage::W4A16(_) | DTypeStorage::EmbeddingWNA16Int => {
                // Weight-only (W4A16) and embedding (EmbeddingWNA16Int) formats are
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

impl RocmDevice {
    /// Returns `true` when the activation storage holds native FP16 data
    /// (`ArithType::F16` + `DTypeStorage::Native`).  The FP16-input WMMA
    /// kernels read such activations directly, halving A-read bandwidth.
    pub(crate) fn is_fp16_activation(storage: &RocmStorage) -> bool {
        storage.dtype().arith == ArithType::F16
            && matches!(storage.dtype().storage, DTypeStorage::Native)
    }

    /// Launch the JIT compiled fused dequantization GEMM kernel for [see: `b_storage`, `Storage::ResidualPacked`]
    pub(crate) fn launch_fused_dequant_gemm_f16(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        b_scales_ptr: *const c_void,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        default_bpw: u8,
        outlier_count: usize,
        outlier_indices_ptr: *const c_void,
        outlier_values_ptr: *const c_void,
        backup_bpw: u8,
        backup_codes_offset: usize,
        backup_scale_offset: usize,
        backup2_bpw: u8,
        backup2_codes_offset: usize,
        backup2_scale_offset: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "fused_dequant_gemm: grid too large for u32 ({} blocks)",
                    total_elems / BLOCK_SIZE as u64
                ))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut bsptr = b_scales_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        let stride_a = k; // A[M, K]
        let stride_c = n; // C[M, N]
        let mut sa = stride_a as i32;
        let mut sc = stride_c as i32;

        let mut bpw_val = default_bpw as i32;
        let mut out_cnt = outlier_count as i32;
        let mut out_idx_ptr = outlier_indices_ptr;
        let mut out_val_ptr = outlier_values_ptr;

        let mut b_bpw = backup_bpw as i32;
        let mut b_codes_off = backup_codes_offset as i32;
        let mut b_scale_off = backup_scale_offset as i32;
        let mut b2_bpw = backup2_bpw as i32;
        let mut b2_codes_off = backup2_codes_offset as i32;
        let mut b2_scale_off = backup2_scale_offset as i32;

        let solution_index = lookup_solution_index(m, n, k, &self.gpu_target, ArithType::F16);
        self.launch_compute_kernel_with_solution(
            "grim_fused_dequant_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut bsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
                arg(&mut sc),
                arg(&mut bpw_val),
                arg(&mut out_cnt),
                arg(&mut out_idx_ptr),
                arg(&mut out_val_ptr),
                arg(&mut b_bpw),
                arg(&mut b_codes_off),
                arg(&mut b_scale_off),
                arg(&mut b2_bpw),
                arg(&mut b2_codes_off),
                arg(&mut b2_scale_off),
            ],
            Some(solution_index),
            0,
        )
    }

    /// Launch the Charon fused MoE dispatch kernel (`rocm_kernel_plan.md` WI-A).
    /// Single sortless launch: each block reads its (token, expert) pair from the uploaded routing arrays.
    pub(crate) fn launch_fused_dequant_backward_gemm_f16(
        &self,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        b_scales_ptr: *const c_void,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        default_bpw: u8,
        outlier_count: usize,
        outlier_indices_ptr: *const c_void,
        outlier_values_ptr: *const c_void,
        backup_bpw: u8,
        backup_codes_offset: usize,
        backup_scale_offset: usize,
        backup2_bpw: u8,
        backup2_codes_offset: usize,
        backup2_scale_offset: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_backward: dY has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_backward: B has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_backward: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        // Grid covers M*K output elements (one thread per element of dX[M,K]).
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_backward: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend(format!(
                    "fused_dequant_backward: grid too large for u32 ({} blocks)",
                    total_elems / BLOCK_SIZE as u64
                ))
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut bsptr = b_scales_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        // dY is [M, N] row-major → stride_dy = N
        let mut sdy = n as i32;
        let mut sdx = k as i32;

        let mut bpw_val = default_bpw as i32;
        let mut out_cnt = outlier_count as i32;
        let mut out_idx_ptr = outlier_indices_ptr;
        let mut out_val_ptr = outlier_values_ptr;

        let mut b_bpw = backup_bpw as i32;
        let mut b_codes_off = backup_codes_offset as i32;
        let mut b_scale_off = backup_scale_offset as i32;

        let mut b2_bpw = backup2_bpw as i32;
        let mut b2_codes_off = backup2_codes_offset as i32;
        let mut b2_scale_off = backup2_scale_offset as i32;

        // STE: grad_scale = 1.0 for pure identity (straight-through estimator).
        // The quantize→dequantize step receives zero gradient - the upstream gradient flows straight through to the.
        let mut grad_scale: f32 = 1.0;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_f16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut bsptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sdy),
                arg(&mut sdx),
                arg(&mut bpw_val),
                arg(&mut out_cnt),
                arg(&mut out_idx_ptr),
                arg(&mut out_val_ptr),
                arg(&mut b_bpw),
                arg(&mut b_codes_off),
                arg(&mut b_scale_off),
                arg(&mut b2_bpw),
                arg(&mut b2_codes_off),
                arg(&mut b2_scale_off),
                arg(&mut grad_scale),
            ],
        )
    }

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

    /// Generic forward fused dequant GEMM launcher for simple kernels
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

    /// SPEED-ROC: dispatch helper for IQ-family formats.
    /// Uses WMMA kernel on RDNA3/4 for small M (decode), falls back to scalar otherwise.
    fn launch_iq_wmma_fallback(
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
        let is_rdna34 = matches!(
            crate::quantization::gcn_arch(&self.gpu_target),
            crate::quantization::GcnArch::RDNA3
                | crate::quantization::GcnArch::RDNA4
                | crate::quantization::GcnArch::UDNA
        );
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
    fn tiled_quant_enabled(tag: &str) -> bool {
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
    fn tiled_k_multiple(blk_elems: u32) -> usize {
        if blk_elems == 32 {
            64
        } else {
            blk_elems as usize
        }
    }

    fn tiled_quant_lookup(entry_or_tag: &str) -> Option<(&'static str, u32)> {
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

    // ─── Standalone dequant launchers ──────────────────────────────────────────

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

    /// Standalone FP8 dequant: convert FP8 E4M3 bytes to F32.
    pub(crate) fn launch_dequant_fp8(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_fp8: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_fp8: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_fp8: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_fp8",
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_w)],
        )
    }

    /// Standalone MXFP4 dequant: decompress MXFP4 codes + shared exponents to F32.
    pub(crate) fn launch_dequant_mxfp4(
        &self,
        codes_storage: &RocmStorage,
        exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let codes_ptr = codes_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp4: codes has no device ptr".into()))?;
        let exps_ptr = exps_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp4: exps has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp4: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_mxfp4: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut codes = codes_ptr;
        let mut exps = exps_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_mxfp4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut codes),
                arg(&mut exps),
                arg(&mut out),
                arg(&mut n_w),
            ],
        )
    }

    /// Standalone MXFP8 dequant: decompress MXFP8 codes + shared exponents to F32.
    pub(crate) fn launch_dequant_mxfp8(
        &self,
        codes_storage: &RocmStorage,
        exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let codes_ptr = codes_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp8: codes has no device ptr".into()))?;
        let exps_ptr = exps_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp8: exps has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_mxfp8: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_mxfp8: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut codes = codes_ptr;
        let mut exps = exps_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_mxfp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut codes),
                arg(&mut exps),
                arg(&mut out),
                arg(&mut n_w),
            ],
        )
    }

    /// Standalone NVFP4 dequant: decompress NVFP4 codes + interleaved scales to F32.
    pub(crate) fn launch_dequant_nvfp4(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_weights: usize,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_nvfp4: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant_nvfp4: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((n_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("dequant_nvfp4: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_w = n_weights as i32;
        self.launch_compute_kernel(
            "grim_dequant_nvfp4",
            grid_dim,
            block_dim,
            &mut [arg(&mut packed), arg(&mut out), arg(&mut n_w)],
        )
    }

    // ─── Standalone IQ dequant launchers ──────────────────────────

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

    // ─── Standalone compressed-tensor dequant launchers ─────────────────────

    pub(crate) fn launch_dequant_wna16(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        num_weights: usize,
        n_bit: i32,
        num_blocks: i32,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wna16 dequant: packed has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("wna16 dequant: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((num_weights as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("wna16 dequant: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_w = num_weights as i32;
        let mut n_bit_v = n_bit;
        let mut nb = num_blocks;
        self.launch_compute_kernel(
            "grim_dequant_wna16",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut packed),
                arg(&mut out),
                arg(&mut n_w),
                arg(&mut n_bit_v),
                arg(&mut nb),
            ],
        )
    }

    /// Public dequant service (quant workstream): WNA16 packed blob → F32 weights, decoded on-device.
    /// Layout contract mirrors `Storage::WNA16`: [u32 n_bit][u32 num_blocks][codes][f16 block scales][f32 tensor scale], 256-weight blocks, MSB-first codes.
    pub fn dequant_wna16_blob_to_f32(
        &self,
        blob: &dyn BackendStorage,
        num_weights: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let packed_rocm = blob
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("wna16 blob not rocm".into()))?;
        let (n_bit_raw, num_blocks_raw) = Self::wna16_read_params(packed_rocm, self.ordinal)?;
        let n_bit = n_bit_raw as u8;
        let num_blocks = num_blocks_raw as usize;
        self.dequant_wna16_to_f32(packed_rocm, num_weights, n_bit, num_blocks)
    }

    pub fn dequant_wna16_to_f32(
        &self,
        packed: &RocmStorage,
        num_weights: usize,
        n_bit: u8,
        num_blocks: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let out_shape = Shape::from_slice(&[num_weights]);
        let out = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let _h =
            self.launch_dequant_wna16(packed, &out, num_weights, n_bit as i32, num_blocks as i32)?;
        let stream = self.active_stream();
        check_hip("hipStreamSynchronize(wna16 dequant)", unsafe {
            crate::device::handles::hipStreamSynchronize(stream)
        })?;
        Ok(Box::new(out))
    }

    pub(crate) fn launch_dequant_embedding_wna16_int(
        &self,
        packed_storage: &RocmStorage,
        out_storage: &RocmStorage,
        total_elements: usize,
        n_bit: i32,
        embedding_dim: i32,
        tensor_scale: f32,
    ) -> Result<*mut c_void> {
        let packed_ptr = packed_storage.device_ptr.ok_or_else(|| {
            Error::Backend("emb_wna16_int dequant: packed has no device ptr".into())
        })?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("emb_wna16_int dequant: out has no device ptr".into()))?;
        const BLOCK_SIZE: usize = 256;
        let grid_x: u32 = ((total_elements as u64).div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("emb_wna16_int dequant: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        let mut packed = packed_ptr;
        let mut out = out_ptr;
        let mut n_el = total_elements as i32;
        let mut n_bit_v = n_bit;
        let mut dim = embedding_dim;
        let mut ts = tensor_scale;
        self.launch_compute_kernel(
            "grim_dequant_embedding_wna16_int",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut packed),
                arg(&mut out),
                arg(&mut n_el),
                arg(&mut n_bit_v),
                arg(&mut dim),
                arg(&mut ts),
            ],
        )
    }

    /// Public dequant service (quant workstream): EmbeddingWNA16Int packed blob → F32 embedding table, decoded on-device.
    /// Layout contract mirrors `Storage::EmbeddingWNA16Int`: [u32 n_bit][u32 embedding_dim][u32 num_rows][codes MSB-first].
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_embedding_wna16_int_to_f32(
        &self,
        packed: &RocmStorage,
        total_elements: usize,
        n_bit: u8,
        embedding_dim: usize,
        tensor_scale: f32,
    ) -> Result<Box<dyn BackendStorage>> {
        let out_shape = Shape::from_slice(&[total_elements]);
        let out = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let _h = self.launch_dequant_embedding_wna16_int(
            packed,
            &out,
            total_elements,
            n_bit as i32,
            embedding_dim as i32,
            tensor_scale,
        )?;
        let stream = self.active_stream();
        check_hip("hipStreamSynchronize(emb wna16 dequant)", unsafe {
            crate::device::handles::hipStreamSynchronize(stream)
        })?;
        Ok(Box::new(out))
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

    // ─── Fused dequant+GEMM kernels: Q5K ──────────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: Q6K ──────────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: Q2K ──────────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: Q3K ──────────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: IQ2_XXS ──────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: IQ2_XS ───────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: IQ2_S ────────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: IQ3_XXS ──────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: IQ3_S ────────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: IQ4_NL ──────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: IQ4_XS ──────────────────────────────────

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

    // ─── Fused dequant+GEMM kernels: Q8_0 ────────────────────────────────────

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

    /// Launch the JIT compiled Q8_0 dequantization kernel.  Reads packed [see: `packed_storage`, `n_weights`, `out_storage`, `materialize()`]
    pub fn dequantize_q8_0(&self, packed: &RocmStorage) -> Result<RocmStorage> {
        const QK8_0: usize = 32;
        // Q8_0 stores weights as packed bytes: each block is 34 bytes (2-byte [see: `n_blocks * 32`]
        let packed_bytes = packed.bytes;
        let n_blocks = packed_bytes / (QK8_0 + 2);
        let n_weights = n_blocks * QK8_0;
        let f32_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_weights]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_q8_0(packed, &f32_storage, n_blocks)?;
        Ok(f32_storage)
    }

    /// Dequantize Q8_0 packed bytes to an f32 host Vec via the ROCm kernel.
    pub fn dequantize_q8_0_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let f32_storage = self.dequantize_q8_0(&packed)?;
        let mut values = self.read_to_host_async(&f32_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Dequantize Q4_K packed bytes to F32 on the GPU.
    /// [see: `block_q4_K`] `packed` must hold `n_blocks` × 144-byte super-blocks; `out` must hold `n_blocks` × 256.
    pub fn dequantize_q4k(&self, packed: &RocmStorage) -> Result<RocmStorage> {
        const QK4_K: usize = 256;
        const BLOCK_BYTES: usize = 144;
        let packed_bytes = packed.bytes;
        let n_blocks = packed_bytes / BLOCK_BYTES;
        let n_weights = n_blocks * QK4_K;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_weights]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_q4k(packed, &out_storage, n_blocks)?;
        Ok(out_storage)
    }

    /// Dequantize Q4_K packed bytes to an f32 host Vec via the ROCm kernel.
    pub fn dequantize_q4k_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let f32_storage = self.dequantize_q4k(&packed)?;
        let mut values = self.read_to_host_async(&f32_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    // ─── Standalone dequant host wrappers (iq/fp8/mxfp) ────────────────────────

    /// Run any standalone IQ dequant kernel against `bytes` and return `elem_count` f32 values.
    fn dequantize_iq_host(
        &self,
        bytes: &[u8],
        elem_count: usize,
        block_bytes: usize,
        kernel: &str,
    ) -> Result<Vec<f32>> {
        const QK: usize = 256;
        let n_blocks = bytes.len() / block_bytes;
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_blocks * QK]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        match kernel {
            "grim_dequant_iq2xxs" => {
                self.launch_dequant_iq2xxs(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq2xs" => {
                self.launch_dequant_iq2xs(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq2s" => {
                self.launch_dequant_iq2s(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq3xxs" => {
                self.launch_dequant_iq3xxs(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq3s" => {
                self.launch_dequant_iq3s(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq4nl" => {
                self.launch_dequant_iq4nl(&packed, &out_storage, n_blocks)?;
            }
            "grim_dequant_iq4xs" => {
                self.launch_dequant_iq4xs(&packed, &out_storage, n_blocks)?;
            }
            other => {
                return Err(Error::Backend(format!(
                    "dequantize_iq_host: unknown kernel {other}"
                )));
            }
        }
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Dequantize IQ2_XXS packed bytes via the ROCm kernel. 66 bytes / 256-elem super-block.
    pub fn dequantize_iq2xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 66, "grim_dequant_iq2xxs")
    }
    /// Dequantize IQ2_XS packed bytes via the ROCm kernel. 74 bytes / 256-elem super-block.
    pub fn dequantize_iq2xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 74, "grim_dequant_iq2xs")
    }
    /// Dequantize IQ2_S packed bytes via the ROCm kernel. 82 bytes / 256-elem super-block.
    pub fn dequantize_iq2s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 82, "grim_dequant_iq2s")
    }
    /// Dequantize IQ3_XXS packed bytes via the ROCm kernel. 96 bytes / 256-elem super-block.
    pub fn dequantize_iq3xxs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 96, "grim_dequant_iq3xxs")
    }
    /// Dequantize IQ3_S packed bytes via the ROCm kernel. 110 bytes / 256-elem super-block.
    pub fn dequantize_iq3s_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 110, "grim_dequant_iq3s")
    }
    /// Dequantize IQ4_NL packed bytes via the ROCm kernel. 170 bytes / 256-elem super-block.
    pub fn dequantize_iq4nl_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 170, "grim_dequant_iq4nl")
    }
    /// Dequantize IQ4_XS packed bytes via the ROCm kernel. 178 bytes / 256-elem super-block.
    pub fn dequantize_iq4xs_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.dequantize_iq_host(bytes, elem_count, 178, "grim_dequant_iq4xs")
    }

    /// Dequantize packed FP8 bytes (4-byte f32 LE scale header, then one E4M3 code per element).
    pub fn dequantize_fp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let scale = if bytes.len() >= 4 {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            1.0
        };
        let payload = if bytes.len() >= 4 { &bytes[4..] } else { bytes };
        let packed = RocmStorage::copy_from_host_raw_bytes(
            payload,
            &Shape::new(vec![payload.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_fp8(&packed, &out_storage, elem_count)?;
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        for v in values.iter_mut() {
            *v *= scale;
        }
        Ok(values)
    }

    /// Helper to split an MXFP single-buffer (length-prefixed codes/exps segments) into two device buffers.
    /// Reuses the same framing as `grim_quant::dequant_mxfp4`/`dequant_mxfp8`.
    pub(crate) fn split_dequant_mxfp(
        &self,
        bytes: &[u8],
        elem_count: usize,
        kernel: &str,
    ) -> Result<Vec<f32>> {
        let mut cursor = 0usize;
        let read_segment = |buf: &[u8], cur: &mut usize| -> Result<Vec<u8>> {
            let len = u64::from_le_bytes(
                buf[*cur..*cur + 8]
                    .try_into()
                    .map_err(|_| Error::Backend("mxfp: bad length prefix".into()))?,
            ) as usize;
            *cur += 8;
            let seg = buf[*cur..*cur + len].to_vec();
            *cur += len;
            Ok(seg)
        };
        let codes = read_segment(bytes, &mut cursor)?;
        let exps = read_segment(bytes, &mut cursor)?;

        let codes_storage = RocmStorage::copy_from_host_raw_bytes(
            &codes,
            &Shape::new(vec![codes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let exps_storage = RocmStorage::copy_from_host_raw_bytes(
            &exps,
            &Shape::new(vec![exps.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;

        if kernel.contains("mxfp4") {
            self.launch_dequant_mxfp4(&codes_storage, &exps_storage, &out_storage, elem_count)?;
        } else {
            self.launch_dequant_mxfp8(&codes_storage, &exps_storage, &out_storage, elem_count)?;
        }
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        Ok(values)
    }

    /// Dequantize an MXFP4 single-buffer roster (length-prefixed codes/exps segments) to f32.
    pub fn dequantize_mxfp4_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.split_dequant_mxfp(bytes, elem_count, "mxfp4")
    }
    /// Dequantize an MXFP8 single-buffer roster (length-prefixed codes/exps segments) to f32.
    pub fn dequantize_mxfp8_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        self.split_dequant_mxfp(bytes, elem_count, "mxfp8")
    }

    /// Dequantize NVFP4 interleaved packed bytes (1 E8M0 scale byte + 8 codes per 16 weights) to f32.
    pub fn dequantize_nvfp4_host(&self, bytes: &[u8], elem_count: usize) -> Result<Vec<f32>> {
        let packed = RocmStorage::copy_from_host_raw_bytes(
            bytes,
            &Shape::new(vec![bytes.len()]),
            DType {
                arith: ArithType::U8,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![elem_count]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_dequant_nvfp4(&packed, &out_storage, elem_count)?;
        let mut values = self.read_to_host_async(&out_storage)?;
        values.truncate(elem_count);
        Ok(values)
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

    /// Launch the JIT compiled FP8 fused dequantization matmul kernel (Raven Tier).
    pub(crate) fn launch_fused_dequant_gemm_fp8(
        &self,
        a_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_fp8: a has no device ptr".into()))?;
        let b_ptr = b_fp8_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_fp8: b has no device ptr".into()))?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_fp8: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_fp8: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_gemm_fp8: grid too large for u32".to_string())
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
            "grim_fused_dequant_gemm_fp8",
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

    /// Launch the JIT compiled FP8 fused dequantization backward matmul kernel (Raven Tier).
    pub(crate) fn launch_fused_dequant_backward_gemm_fp8(
        &self,
        dy_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_fp8: dY has no device ptr".into())
        })?;
        let b_ptr = b_fp8_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_fp8: B has no device ptr".into())
        })?;
        let dx_ptr = dx_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_backward_fp8: dX has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_backward_fp8: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_backward_fp8: grid too large for u32".to_string())
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_fp8",
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

    /// Launch the JIT compiled MXFP4 fused dequantization matmul kernel (Jay Tier).
    #[allow(dead_code)]
    pub(crate) fn launch_fused_dequant_gemm_mxfp4(
        &self,
        a_storage: &RocmStorage,
        b_codes_ptr: u64,
        b_exps_ptr: u64,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp4: a has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp4: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_mxfp4: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_gemm_mxfp4: grid too large for u32".to_string())
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_mxfp4",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled MXFP8 fused dequantization matmul kernel (Magpie Tier).
    pub(crate) fn launch_fused_dequant_gemm_mxfp8(
        &self,
        a_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        b_exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: a has no device ptr".into())
        })?;
        let b_fp8_ptr = b_fp8_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: b_fp8 has no device ptr".into())
        })?;
        let b_exps_ptr = b_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: b_exps has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_dequant_gemm_mxfp8: out has no device ptr".into())
        })?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fused_dequant_gemm_mxfp8: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| {
                Error::Backend("fused_dequant_gemm_mxfp8: grid too large for u32".to_string())
            })?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bfp8ptr = b_fp8_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_mxfp8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bfp8ptr),
                arg(&mut bexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the JIT compiled tiled MXFP4 GEMM kernel.
    /// `b_codes_ptr` / `b_exps_ptr` are raw device pointers: either standalone storages or interior pointers into a.
    pub fn launch_mxfp4_gemm_tiled(
        &self,
        a_storage: &RocmStorage,
        b_codes_ptr: u64,
        b_exps_ptr: u64,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        // The kernels read activations as float4 and codes as uint4; both
        // require K to be a multiple of 32 (one MXFP4 micro-block).
        if k % 32 != 0 {
            return Err(Error::Backend(format!(
                "mxfp4_gemm_tiled: K must be a multiple of 32, got {k}"
            )));
        }
        // Skinny-M decode: a plain (n/16, m/16) grid leaves most CUs idle (e.g.
        // m=1, n=4096 -> 16 CTAs on a 28+ CU part).
        if m <= 8 && k >= 2048 {
            return self.launch_mxfp4_gemm_splitk(
                a_storage,
                b_codes_ptr,
                b_exps_ptr,
                out_storage,
                m,
                n,
                k,
            );
        }
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_tiled: a has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_tiled: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_x = n.div_ceil(16) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);

        let mut aptr = a_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_mxfp4_gemm_tiled",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Split-K MXFP4 GEMM for skinny-M decode (M <= 8): slice K across CUs, reduce the partials deterministically.
    /// Kept as two launches so the result is bit-stable across runs (no float atomics).
    pub(crate) fn launch_mxfp4_gemm_splitk(
        &self,
        a_storage: &RocmStorage,
        b_codes_ptr: u64,
        b_exps_ptr: u64,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_splitk: a has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_splitk: out has no device ptr".into()))?;

        let num_splits: u32 = if k >= 8192 {
            8
        } else if k >= 4096 {
            4
        } else {
            2
        };
        let partials = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[num_splits as usize, m, n]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let partials_ptr = partials
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_gemm_splitk: partials alloc failed".into()))?;

        const SPLITK_BLOCK: usize = 64;
        let grid_dim = HipDim3::new((n.div_ceil(SPLITK_BLOCK)) as u32, m as u32, num_splits);
        let block_dim = HipDim3::new(SPLITK_BLOCK as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut pptr = partials_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut splits = num_splits as i32;

        let stream = self.launch_compute_kernel(
            "grim_mxfp4_gemm_splitk",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut pptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut splits),
            ],
        )?;

        const REDUCE_BLOCK: usize = 256;
        let total = m * n;
        let reduce_grid = HipDim3::new((total.div_ceil(REDUCE_BLOCK)) as u32, 1, 1);
        let reduce_block = HipDim3::new(REDUCE_BLOCK as u32, 1, 1);

        let mut optr = out_ptr;
        let _ = self.launch_compute_kernel(
            "grim_mxfp4_splitk_reduce",
            reduce_grid,
            reduce_block,
            &mut [
                arg(&mut pptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut splits),
            ],
        )?;

        // `partials` drops to the pool here; same-stream reuse is ordered
        // after both kernels above.
        Ok(stream)
    }

    /// Fused NVFP4 GEMV with cooperative Wave reduction in LDS (for decode batch M <= 4).
    pub fn launch_nvfp4_gemv(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemv: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemv: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemv: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(256, 1, 1);
        let grid_dim = HipDim3::new(n as u32, m as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_nvfp4_gemv",
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

    /// Tiled NVFP4 GEMM for prefill batch (M > 4).
    pub fn launch_nvfp4_gemm_tiled(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemm_tiled: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemm_tiled: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("nvfp4_gemm_tiled: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_nvfp4_gemm_tiled",
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

    /// Launch the JIT compiled backward MXFP4 GEMM kernel (dA = dY @ B^T).
    pub(crate) fn launch_mxfp4_backward_gemm(
        &self,
        dy_storage: &RocmStorage,
        b_codes_storage: &RocmStorage,
        b_exps_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_backward_gemm: dy has no device ptr".into()))?;
        let b_codes_ptr = b_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("mxfp4_backward_gemm: b_codes has no device ptr".into())
        })?;
        let b_exps_ptr = b_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("mxfp4_backward_gemm: b_exps has no device ptr".into())
        })?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("mxfp4_backward_gemm: dx has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_x = k.div_ceil(16) as u32;
        let grid_y = m.div_ceil(16) as u32;
        let grid_dim = HipDim3::new(grid_x, grid_y, 1);

        let mut dyptr = dy_ptr;
        let mut bcodesptr = b_codes_ptr;
        let mut bexpsptr = b_exps_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_mxfp4_backward_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bcodesptr),
                arg(&mut bexpsptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
            ],
        )
    }

    /// Launch the fused RMSNorm + MXFP4 GEMM kernel (e.g. for MLP projections).
    pub fn launch_fused_rmsnorm_mxfp4_gemm(
        &self,
        x_storage: &RocmStorage,
        gamma_storage: &RocmStorage,
        w_codes_storage: &RocmStorage,
        w_exps_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        eps: f32,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: x has no device ptr".into())
        })?;
        let gamma_ptr = gamma_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: gamma has no device ptr".into())
        })?;
        let w_codes_ptr = w_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: w_codes has no device ptr".into())
        })?;
        let w_exps_ptr = w_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: w_exps has no device ptr".into())
        })?;
        let out_ptr = out_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_gemm: out has no device ptr".into())
        })?;

        let block_dim = HipDim3::new(64, 1, 1);
        let grid_dim = HipDim3::new(m as u32, n.div_ceil(64) as u32, 1);

        let mut xptr = x_ptr;
        let mut gammaptr = gamma_ptr;
        let mut wcodesptr = w_codes_ptr;
        let mut wexpsptr = w_exps_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut eps_val = eps;

        self.launch_compute_kernel_with_solution(
            "grim_fused_rmsnorm_mxfp4_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut gammaptr),
                arg(&mut wcodesptr),
                arg(&mut wexpsptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut eps_val),
            ],
            None,
            64 * std::mem::size_of::<f32>(),
        )
    }

    /// Launch the fused RMSNorm + MXFP4 GEMM + RoPE + direct KV cache scatter kernel.
    pub fn launch_fused_rmsnorm_mxfp4_gemm_rope_kv(
        &self,
        x_storage: &RocmStorage,
        gamma_storage: &RocmStorage,
        w_codes_storage: &RocmStorage,
        w_exps_storage: &RocmStorage,
        q_out_storage: Option<&RocmStorage>,
        k_cache_storage: Option<&RocmStorage>,
        v_cache_storage: Option<&RocmStorage>,
        out_all_storage: Option<&RocmStorage>,
        positions_storage: Option<&RocmStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq_storage: Option<&RocmStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: x has no device ptr".into())
        })?;
        let gamma_ptr = gamma_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: gamma has no device ptr".into())
        })?;
        let w_codes_ptr = w_codes_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: w_codes has no device ptr".into())
        })?;
        let w_exps_ptr = w_exps_storage.device_ptr.ok_or_else(|| {
            Error::Backend("fused_rmsnorm_mxfp4_rope_kv: w_exps has no device ptr".into())
        })?;

        let q_out_ptr = q_out_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let k_cache_ptr = k_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let v_cache_ptr = v_cache_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let out_all_ptr = out_all_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let positions_ptr = positions_storage.and_then(|s| s.device_ptr).unwrap_or(0);
        let inv_freq_ptr = inv_freq_storage.and_then(|s| s.device_ptr).unwrap_or(0);

        let n_total = (num_q_heads + 2 * num_kv_heads) * head_dim;
        let block_dim = HipDim3::new(64, 1, 1);
        let grid_dim = HipDim3::new(m as u32, n_total.div_ceil(64) as u32, 1);

        let mut xptr = x_ptr;
        let mut gammaptr = gamma_ptr;
        let mut wcodesptr = w_codes_ptr;
        let mut wexpsptr = w_exps_ptr;
        let mut qptr = q_out_ptr;
        let mut kptr = k_cache_ptr;
        let mut vptr = v_cache_ptr;
        let mut allptr = out_all_ptr;
        let mut posptr = positions_ptr;
        let mut mm = m as i32;
        let mut kk = k as i32;
        let mut nq = num_q_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut rd = rotary_dim as i32;
        let mut theta = rope_theta;
        let mut invfreqptr = inv_freq_ptr;
        let mut mscale_val = mscale;
        let mut eps_val = eps;
        let mut max_seq = max_seq_len as i32;

        self.launch_compute_kernel_with_solution(
            "grim_fused_rmsnorm_mxfp4_gemm_rope_kv",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut gammaptr),
                arg(&mut wcodesptr),
                arg(&mut wexpsptr),
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut allptr),
                arg(&mut posptr),
                arg(&mut mm),
                arg(&mut kk),
                arg(&mut nq),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut rd),
                arg(&mut theta),
                arg(&mut invfreqptr),
                arg(&mut mscale_val),
                arg(&mut eps_val),
                arg(&mut max_seq),
            ],
            None,
            64 * std::mem::size_of::<f32>(),
        )
    }

    /// Launch the GPTQ/EfficientQAT GroupInt fused dequant-GEMM (forward).
    /// `b_storage` holds the documented length-prefixed four-segment packed layout (`GpuIntConfig`); `qw/qz/sc/gi` are byte offsets of each.
    pub(crate) fn launch_gptq_dequant_gemm(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        has_g_idx: bool,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
        gi_off: i64,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq gemm: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq gemm: b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("gptq gemm: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("gptq gemm: grid too large for u32".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let values_per_word: i32 = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "gptq gemm: unsupported bit width {bits}"
                )));
            }
        };

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;
        let mut vpw = values_per_word;
        let mut has_gi = if has_g_idx { 1 } else { 0 };
        let mut qw = qw_off;
        let mut qz = qz_off;
        let mut sc = sc_off;
        let mut gi = gi_off;

        self.launch_compute_kernel(
            "grim_gptq_dequant_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut has_gi),
                arg(&mut qw),
                arg(&mut qz),
                arg(&mut sc),
                arg(&mut gi),
            ],
        )
    }

    /// Launch the GPTQ/EfficientQAT GroupInt fused dequant-GEMM (backward).
    /// Computes `dX[M, K] = dY[M, N] @ dequant(B)` from the same packed blob as [`Self::launch_gptq_dequant_gemm`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_gptq_dequant_backward_gemm(
        &self,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        has_g_idx: bool,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
        gi_off: i64,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq backward: dY has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq backward: b has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("gptq backward: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("gptq backward: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("gptq backward: grid too large for u32".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let values_per_word: i32 = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "gptq backward: unsupported bit width {bits}"
                )));
            }
        };

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;
        let mut vpw = values_per_word;
        let mut has_gi = if has_g_idx { 1 } else { 0 };
        let mut qw = qw_off;
        let mut qz = qz_off;
        let mut sc = sc_off;
        let mut gi = gi_off;

        self.launch_compute_kernel(
            "grim_gptq_dequant_backward_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut has_gi),
                arg(&mut qw),
                arg(&mut qz),
                arg(&mut sc),
                arg(&mut gi),
            ],
        )
    }

    /// Compute the length-prefixed GroupInt segment offsets for a packed weight blob of `blob_bytes` bytes.
    /// Returns `(qw_off, qz_off, sc_off, gi_off, has_g_idx)`.
    pub(crate) fn gptq_segment_offsets(
        bits: u8,
        group_size: usize,
        k: usize,
        n: usize,
        blob_bytes: usize,
    ) -> Result<(i64, i64, i64, i64, bool)> {
        const _: usize = 32; // four interleaved u64 length prefixes total
        let vpw: usize = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "gptq gemm: unsupported bit width {bits}"
                )));
            }
        };
        let qw_len = k.div_ceil(vpw) * n * 4;
        let groups = k.div_ceil(group_size);
        let qz_len = groups * n.div_ceil(vpw) * 4;
        let sc_len = groups * n * 4;

        // Each segment is preceded by ITS OWN u64 length prefix: [u64 qw_len][qweight][u64 qz_len][qzeros][u64 sc_len][scales][u64 gi_len][g_idx] so data starts
        // are 8 / (8+qw+8) / (8+qw+8+qz+8) / (+8+sc), and the blob ends with the (possibly empty-length) g_idx prefix.
        let qz_data = 8 + qw_len + 8;
        let sc_data = qz_data + qz_len + 8;
        let gi_data = sc_data + sc_len + 8;

        let no_gi_total = gi_data; // empty g_idx segment: just the zeroed u64
        let gi_total_u32 = gi_data + k * 4;
        let gi_total_u64 = gi_data + k * 8;

        let has_g_idx = if blob_bytes == no_gi_total {
            false
        } else if blob_bytes == gi_total_u32 {
            true
        } else if blob_bytes == gi_total_u64 {
            return Err(Error::Backend(
                "gptq gemm: 64-bit g_idx entries not supported by the fused kernel".into(),
            ));
        } else {
            return Err(Error::Backend(format!(
                "gptq gemm: packed blob size {blob_bytes} matches no valid \
                 GroupInt layout for bits={bits} group_size={group_size} k={k} n={n} \
                 (expected {no_gi_total}, {gi_total_u32}, or {gi_total_u64})"
            )));
        };

        Ok((8, qz_data as i64, sc_data as i64, gi_data as i64, has_g_idx))
    }

    /// Launch the AWQ fused dequant-GEMM (forward).
    /// Computes `C[M, N] = A[M, K] @ dequant(B)^T` where B is packed in the native.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_awq_dequant_gemm(
        &self,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq gemm: A has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq gemm: B has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq gemm: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("awq gemm: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("awq gemm: grid too large for u32".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let values_per_word: i32 = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "awq gemm: unsupported bit width {bits}"
                )));
            }
        };

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut outptr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;
        let mut vpw = values_per_word;
        let mut qw = qw_off;
        let mut qz = qz_off;
        let mut sc = sc_off;

        self.launch_compute_kernel(
            "grim_awq_dequant_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut outptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut qw),
                arg(&mut qz),
                arg(&mut sc),
            ],
        )
    }

    /// Launch the AWQ fused dequant-GEMM (backward dX).
    /// Computes `dX[M, K] = dY[M, N] @ dequant(B)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_awq_dequant_backward_gemm(
        &self,
        dy_storage: &RocmStorage,
        b_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
        qw_off: i64,
        qz_off: i64,
        sc_off: i64,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq backward: dY has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq backward: b has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("awq backward: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("awq backward: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("awq backward: grid too large for u32".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let values_per_word: i32 = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "awq backward: unsupported bit width {bits}"
                )));
            }
        };

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;
        let mut vpw = values_per_word;
        let mut qw = qw_off;
        let mut qz = qz_off;
        let mut sc = sc_off;

        self.launch_compute_kernel(
            "grim_awq_dequant_backward_gemm",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut dyptr),
                arg(&mut bptr),
                arg(&mut dxptr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut qw),
                arg(&mut qz),
                arg(&mut sc),
            ],
        )
    }

    /// Compute the length-prefixed AWQ segment offsets for a packed
    /// weight blob of `blob_bytes` bytes. Returns `(qw_off, qz_off, sc_off)`.
    pub fn awq_segment_offsets(
        bits: u8,
        group_size: usize,
        k: usize,
        n: usize,
        blob_bytes: usize,
    ) -> Result<(i64, i64, i64)> {
        let vpw: usize = match bits {
            2 => 16,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "awq gemm: unsupported bit width {bits}"
                )));
            }
        };
        let qw_len = k.div_ceil(vpw) * n * 4;
        let groups = k.div_ceil(group_size);
        let qz_len = groups * n.div_ceil(vpw) * 4;
        let sc_len = groups * n * 2; // f16 scales = 2 bytes each

        // Layout: [u64 qw_len][qweight][u64 qz_len][qzeros][u64 sc_len][scales (f16)]
        let qz_data = 8 + qw_len + 8;
        let sc_data = qz_data + qz_len + 8;
        let total_expected = sc_data + sc_len;

        if blob_bytes != total_expected {
            return Err(Error::Backend(format!(
                "awq gemm: packed blob size {blob_bytes} does not match expected {total_expected} \
                 for bits={bits} group_size={group_size} k={k} n={n}"
            )));
        }

        Ok((8, qz_data as i64, sc_data as i64))
    }

    /// Audit-wiring (quant workstream): W4A16 blobs are a SINGLE packed segment pair - `[codes (N*K/8 u32)][scales (N*groups f32)]` per the `Storage::W4A16` layout contract -
    /// so the dense dispatch path needs a launcher that derives the scales pointer from the same device buffer instead of requiring two separate storages.
    pub fn launch_marlin_gemm_w4a16_blob(
        &self,
        a_storage: &RocmStorage,
        blob_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<*mut c_void> {
        if k % 8 != 0 {
            return Err(Error::Backend(format!(
                "marlin_w4a16: K={k} must be divisible by 8"
            )));
        }
        let codes_bytes = n * (k / 8) * std::mem::size_of::<u32>();
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_w4a16: a has no device ptr".into()))?;
        let blob_ptr = blob_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_w4a16: blob has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_w4a16: out has no device ptr".into()))?;
        if blob_storage.bytes()
            < codes_bytes + n * (k.div_ceil(group_size)) * std::mem::size_of::<f32>()
        {
            return Err(Error::Backend(
                "marlin_w4a16: blob smaller than codes+scales segments".into(),
            ));
        }

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1);

        // Kernel contract: A row-major [M, K] f32; C [M, N] f32.
        let mut aptr = a_ptr;
        let mut bptr = blob_ptr;
        let mut sptr = (blob_ptr as usize + codes_bytes) as *mut c_void;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut gs = group_size as i32;

        self.launch_compute_kernel(
            "grim_marlin_gemm_w4a16_f32",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut sptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut gs),
            ],
        )
    }

    /// Elementwise dequant-GEMM launcher shared by the compressed-tensors
    /// W8A8 kernels (256-thread blocks over M*N outputs).
    pub(crate) fn launch_elementwise_dequant_gemm(
        &self,
        entry: &str,
        a: &RocmStorage,
        blob: &RocmStorage,
        out: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        extra_args: &mut [*mut c_void],
    ) -> Result<()> {
        let mut aptr = a
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm: a has no device ptr".into()))?
            as *mut c_void;
        let mut bptr = blob
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm: blob has no device ptr".into()))?
            as *mut c_void;
        let mut optr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm: out has no device ptr".into()))?
            as *mut c_void;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let grid_x = (m * n).div_ceil(256) as u32;
        let mut args: Vec<*mut c_void> = vec![
            arg(&mut aptr),
            arg(&mut bptr),
            arg(&mut optr),
            arg(&mut mm),
            arg(&mut nn),
            arg(&mut kk),
        ];
        args.extend_from_slice(extra_args);
        self.launch_compute_kernel(
            entry,
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(256, 1, 1),
            &mut args,
        )?;
        Ok(())
    }

    /// Elementwise dequant-GEMM backward launcher: dX[M, K] = dY[M, N] @ deq(B)[N, K].
    /// Same 256-thread blocks, but grid covers M*K outputs (the dX dimension).
    pub(crate) fn launch_elementwise_dequant_gemm_backward(
        &self,
        entry: &str,
        dy: &RocmStorage,
        blob: &RocmStorage,
        dx: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        extra_args: &mut [*mut c_void],
    ) -> Result<()> {
        let mut dyptr = dy
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm backward: dY has no device ptr".into()))?
            as *mut c_void;
        let mut bptr = blob
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm backward: blob has no device ptr".into()))?
            as *mut c_void;
        let mut dxptr = dx
            .device_ptr
            .ok_or_else(|| Error::Backend("dequant gemm backward: dX has no device ptr".into()))?
            as *mut c_void;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let grid_x = (m * k).div_ceil(256) as u32;
        let mut args: Vec<*mut c_void> = vec![
            arg(&mut dyptr),
            arg(&mut bptr),
            arg(&mut dxptr),
            arg(&mut mm),
            arg(&mut nn),
            arg(&mut kk),
        ];
        args.extend_from_slice(extra_args);
        self.launch_compute_kernel(
            entry,
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(256, 1, 1),
            &mut args,
        )?;
        Ok(())
    }

    /// Read the WNA16 blob header ([u32 n_bit][u32 num_blocks]) via pinned D2H.
    pub(crate) fn wna16_read_header(blob: &RocmStorage, ordinal: usize) -> Result<[u8; 8]> {
        let dev = RocmDevice::try_new(ordinal)?;
        let mut pinned = RocmPinnedBuffer::<u8>::alloc(8)?;
        let _g = crate::device::util::DeviceGuard::set(ordinal as i32);
        let ptr = blob
            .device_ptr
            .ok_or_else(|| Error::Backend("wna16 header: no device ptr".into()))?;
        check_hip("wna16 header D2H", unsafe {
            hipMemcpyAsync(
                pinned.as_mut_ptr() as *mut c_void,
                ptr as *const c_void,
                8,
                HipMemcpyKind::DeviceToHost,
                dev.active_stream(),
            )
        })?;
        check_hip("wna16 header sync", unsafe {
            hipStreamSynchronize(dev.active_stream())
        })?;
        let mut out = [0u8; 8];
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(pinned.as_ptr(), 8) });
        Ok(out)
    }

    /// Safely unpacks the `(n_bit, num_blocks)` tuple from a WNA16 blob header with error propagation.
    pub(crate) fn wna16_read_params(blob: &RocmStorage, ordinal: usize) -> Result<(u32, u32)> {
        let hdr = Self::wna16_read_header(blob, ordinal)?;
        let n_bit_bytes: [u8; 4] = hdr[0..4]
            .try_into()
            .map_err(|e| Error::Backend(format!("wna16 header n_bit slice error: {e}")))?;
        let blocks_bytes: [u8; 4] = hdr[4..8]
            .try_into()
            .map_err(|e| Error::Backend(format!("wna16 header num_blocks slice error: {e}")))?;
        let n_bit = u32::from_le_bytes(n_bit_bytes);
        let num_blocks = u32::from_le_bytes(blocks_bytes);
        Ok((n_bit, num_blocks))
    }

    /// Public materialization service (quant workstream): dequantize a Marlin W4A16 packed expert blob to row-major F32 [k_dim?
    /// no -] Returns Dᵀ flattened ([k_dim, n_rows] where C = A @ Dᵀ was computed.
    pub fn dequant_w4a16_blob_to_f32(
        &self,
        blob: &RocmStorage,
        n_rows: usize,
        k_dim: usize,
        group_size: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        use grim_tensor::ArithType;
        let id: Vec<f32> = (0..k_dim)
            .flat_map(|i| (0..k_dim).map(move |j| if i == j { 1.0f32 } else { 0.0 }))
            .collect();
        let a = self.from_cpu(&id, &Shape::new(vec![k_dim, k_dim]), DType::F32)?;
        let out_shape = Shape::new(vec![k_dim, n_rows]);
        let out = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let a_rocm = a
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("identity not rocm".into()))?;
        let _h = self
            .launch_marlin_gemm_w4a16_blob(a_rocm, blob, &out, k_dim, n_rows, k_dim, group_size)?;
        Ok(Box::new(out))
    }

    /// Public materialization service (quant workstream): run the GPTQ forward-dequant GEMM with an
    /// identity activation so C = D, the full row-major [n_out, k_in] dequantized weight.
    #[allow(clippy::too_many_arguments)]
    pub fn gptq_dequant_identity_to_f32(
        &self,
        blob: &RocmStorage,
        n_out: usize,
        k_in: usize,
        bits: u8,
        group_size: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        use grim_tensor::ArithType;
        let mut vpw: i32 = match bits {
            2 => 16,
            3 => 32,
            4 => 8,
            8 => 1,
            _ => {
                return Err(Error::Backend(format!(
                    "gptq dequant identity: unsupported bit width {bits}"
                )));
            }
        };
        let (qw, qz, sc, gi, has_g_idx) =
            Self::gptq_segment_offsets(bits, group_size, k_in, n_out, blob.bytes())?;
        let mut has_i = if has_g_idx { 1 } else { 0 };
        // C = A @ D with A = I[K=k_in] gives C = D row-major [k_in, n_out]
        // (the CALLER transposes to weight layout [n_out, k_in]).
        let id: Vec<f32> = (0..k_in)
            .flat_map(|i| (0..k_in).map(move |j| if i == j { 1.0f32 } else { 0.0 }))
            .collect();
        let a = self.from_cpu(&id, &Shape::new(vec![k_in, k_in]), DType::F32)?;
        let out_shape = Shape::new(vec![k_in, n_out]);
        let out = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let a_rocm = a
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("identity not rocm".into()))?;
        let mut aptr = a_rocm.device_ptr_checked()? as *mut c_void;
        let mut bptr = blob.device_ptr_checked()? as *mut c_void;
        let mut optr = out.device_ptr_checked()? as *mut c_void;
        let mut m_i = k_in as i32;
        let mut n_i = n_out as i32;
        let mut k_i = k_in as i32;
        let mut bits_i = bits as i32;
        let mut gs_i = group_size as i32;

        let mut qw_i = qw;
        let mut qz_i = qz;
        let mut sc_i = sc;
        let mut gi_i = gi;
        let grid_x = (n_out * n_out).div_ceil(256) as u32;
        self.launch_compute_kernel(
            "grim_gptq_dequant_gemm",
            HipDim3::new(grid_x, 1, 1),
            HipDim3::new(256, 1, 1),
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut optr),
                arg(&mut m_i),
                arg(&mut n_i),
                arg(&mut k_i),
                arg(&mut bits_i),
                arg(&mut gs_i),
                arg(&mut vpw),
                arg(&mut has_i),
                arg(&mut qw_i),
                arg(&mut qz_i),
                arg(&mut sc_i),
                arg(&mut gi_i),
            ],
        )?;
        Ok(Box::new(out))
    }

    /// Launch Marlin-style Interleaved W4A16 GEMM.
    pub fn launch_marlin_gemm_w4a16(
        &self,
        a_storage: &RocmStorage,
        b_w4_storage: &RocmStorage,
        scales_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: a has no device ptr".into()))?;
        let b_ptr = b_w4_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: b has no device ptr".into()))?;
        let scales_ptr = scales_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: scales has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("marlin_gemm: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut sptr = scales_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut gs = group_size as i32;

        // Select kernel based on scales dtype (what the kernel actually reads),
        // not output dtype. Out can be F16 or F32 regardless of scale precision.
        let kernel_name = match scales_storage.dtype.arith {
            grim_tensor::ArithType::F16 => "grim_marlin_gemm_w4a16",
            _ => "grim_marlin_gemm_w4a16_f32",
        };

        self.launch_compute_kernel(
            kernel_name,
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut sptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut gs),
            ],
        )
    }

    /// Launch BitNet b1.58 Ternary GEMM (W1.58A8).
    pub fn launch_bitnet_gemm_w158a8(
        &self,
        a_storage: &RocmStorage,
        b_ternary_storage: &RocmStorage,
        scale_b_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
        scale_a: f32,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: a has no device ptr".into()))?;
        let b_ptr = b_ternary_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: b has no device ptr".into()))?;
        let scale_b_ptr = scale_b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: scale_b has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("bitnet_gemm: out has no device ptr".into()))?;

        let block_dim = HipDim3::new(16, 16, 1);
        let grid_dim = HipDim3::new(n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut sbptr = scale_b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;
        let mut sa = scale_a;

        self.launch_compute_kernel(
            "grim_bitnet_gemm_w158a8",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut aptr),
                arg(&mut bptr),
                arg(&mut sbptr),
                arg(&mut optr),
                arg(&mut mm),
                arg(&mut nn),
                arg(&mut kk),
                arg(&mut sa),
            ],
        )
    }

    /// Launch standalone Q8_0 quantization HIP kernel.
    pub fn launch_quant_q8_0(
        &self,
        x: &RocmStorage,
        out: &RocmStorage,
        total: usize,
    ) -> Result<*mut c_void> {
        let n_blocks = total.div_ceil(32);
        let grid = crate::HipDim3 {
            x: n_blocks as u32,
            y: 1,
            z: 1,
        };
        let block = crate::HipDim3 { x: 32, y: 1, z: 1 };
        let mut x_ptr = dev_ptr(x)?;
        let mut out_ptr = dev_ptr(out)?;
        let mut total_i = total as i32;

        self.launch_compute_kernel(
            "grim_quant_q8_0",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut total_i)],
        )
    }

    /// Launch standalone FP8 E4M3 quantization HIP kernel.
    pub fn launch_quant_fp8(
        &self,
        x: &RocmStorage,
        out: &RocmStorage,
        total: usize,
    ) -> Result<*mut c_void> {
        let (grid, block) = linear_launch(total);
        let mut x_ptr = dev_ptr(x)?;
        let mut out_ptr = dev_ptr(out)?;
        let mut total_i = total as i32;

        self.launch_compute_kernel(
            "grim_quant_fp8",
            grid,
            block,
            &mut [arg(&mut x_ptr), arg(&mut out_ptr), arg(&mut total_i)],
        )
    }

    /// Quantize F32 tensor `x` on-device to `format`.
    pub fn quantize_on_device(
        &self,
        x: &dyn BackendStorage,
        format: grim_tensor::QuantFormat,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        if !x_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "quantize_on_device: input lacks valid device pointer".into(),
            ));
        }
        let total = x.shape().elem_count();
        use grim_tensor::{FloatPackScheme, KQuantScheme, QuantFormat};
        let (out_bytes, output_dtype) = match format {
            QuantFormat::Q8_0 => {
                let n_blocks = total.div_ceil(32);
                (
                    n_blocks * 34,
                    DType {
                        arith: ArithType::F32,
                        storage: DTypeStorage::KQuant(KQuantScheme::Q80),
                    },
                )
            }
            QuantFormat::Fp8 => (
                4 + total,
                DType {
                    arith: ArithType::F32,
                    storage: DTypeStorage::FloatPack(FloatPackScheme::Fp8),
                },
            ),
            other => {
                return Err(Error::Backend(format!(
                    "quantize_on_device: unsupported format {:?}",
                    other
                )));
            }
        };

        let out_shape = x.shape().clone();
        let out_storage = RocmStorage::alloc_gpu_with_bytes(
            &out_shape,
            output_dtype,
            out_bytes,
            &self.allocator,
            self.ordinal,
        )?;

        let stream = match format {
            QuantFormat::Q8_0 => self.launch_quant_q8_0(x_s, &out_storage, total)?,
            QuantFormat::Fp8 => self.launch_quant_fp8(x_s, &out_storage, total)?,
            _ => unreachable!(),
        };

        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(stream))),
        ))
    }

    /// Fused 3-in-1 SwiGLU activation + dynamic scale quantization HIP kernel launch.
    pub fn silu_mul_quantize_gpu(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        _format: grim_tensor::dtype::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let g_s = as_rocm(gate)?;
        let u_s = as_rocm(up)?;
        if !g_s.device_ptr_is_valid() || !u_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "silu_mul_quantize: inputs lack a valid device pointer".into(),
            ));
        }

        let total = out_shape.elem_count();
        let qout_storage = RocmStorage::alloc_gpu(
            out_shape,
            DType {
                arith: grim_tensor::dtype::ArithType::U8,
                storage: grim_tensor::dtype::Storage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        let scale_storage = RocmStorage::alloc_gpu(
            &Shape::from_slice(&[1]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;

        let mut gate_ptr = dev_ptr(g_s)?;
        let mut up_ptr = dev_ptr(u_s)?;
        let mut qout_ptr = dev_ptr(&qout_storage)?;
        let mut scale_ptr = dev_ptr(&scale_storage)?;
        let mut n_i = total as i32;

        let grid = HipDim3::new(1, 1, 1);
        let block = HipDim3::new(256, 1, 1);

        self.launch_compute_kernel(
            "grim_silu_mul_quantize",
            grid,
            block,
            &mut [
                arg(&mut gate_ptr),
                arg(&mut up_ptr),
                arg(&mut qout_ptr),
                arg(&mut scale_ptr),
                arg(&mut n_i),
            ],
        )?;

        Ok((
            Box::new(qout_storage),
            Box::new(scale_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    // ─── Phase 2: MFMA FP8 (gfx1200+) ────────────────────────────

    /// Launch the gfx1200 MFMA FP8 fused dequant GEMM kernel. [see: `should_use_wmma_path`, `rocm_device_props::gfx_level >= 12`]
    pub(crate) fn launch_fused_dequant_gemm_fp8_mfma(
        &self,
        a_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        out_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma: a has no device ptr".into()))?;
        let b_ptr = b_fp8_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma: B_fp8 has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| Error::Backend("fp8_mfma: m*n overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("fp8_mfma: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut aptr = a_ptr;
        let mut bptr = b_ptr;
        let mut optr = out_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_gemm_fp8_mfma",
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

    /// Launch the gfx1200 MFMA FP8 backward kernel.
    pub(crate) fn launch_fused_dequant_backward_gemm_fp8_mfma(
        &self,
        dy_storage: &RocmStorage,
        b_fp8_storage: &RocmStorage,
        dx_storage: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<*mut c_void> {
        let dy_ptr = dy_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: dY has no device ptr".into()))?;
        let b_ptr = b_fp8_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: B_fp8 has no device ptr".into()))?;
        let dx_ptr = dx_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: dX has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (m as u64)
            .checked_mul(k as u64)
            .ok_or_else(|| Error::Backend("fp8_mfma_bwd: m*k overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("fp8_mfma_bwd: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut dyptr = dy_ptr;
        let mut bptr = b_ptr;
        let mut dxptr = dx_ptr;
        let mut mm = m as i32;
        let mut nn = n as i32;
        let mut kk = k as i32;

        self.launch_compute_kernel(
            "grim_fused_dequant_backward_gemm_fp8_mfma",
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
