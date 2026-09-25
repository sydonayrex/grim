//! Core tensor computation, GEMM, elementwise, autograd, and optimizer operations for `RocmDevice`.
//! Fused layer ops: QKV/GateUp projections, RMSNorm fusions, cross-entropy, embedding gather.

use super::{FusedGateUpWeights, FusedQkvGateLogits, FusedQkvWeights};
use std::ffi::c_void;
use std::sync::Arc;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::memory::view::RocmStorageView;
use crate::{
    arg, as_rocm, check_hip, dev_ptr, dtype_f32, hipMemsetAsync, hipSuccess, warp_rows_launch,
    HipDim3, QkvAttentionFusionConfig, QuantMode, RmsNormMatMulFusionConfig, RocmHandle,
};

impl RocmDevice {
    /// Graph-capturable embedding gather: `out[slot*dim + j] = weight[idx[slot]*dim + j]`.
    /// Unlike `embedding()`, indices are READ FROM DEVICE MEMORY (`indices`,
    /// i32/u32 per slot), so the captured graph node sees fresh token ids on
    /// each replay. `weight` must be an F32 native table `[vocab, dim]`.
    pub fn launch_embedding_gather_dev_idx(
        &self,
        weight: &RocmStorage,
        out: &RocmStorage,
        indices: &RocmStorage,
        dim: usize,
        total: usize,
    ) -> Result<*mut c_void> {
        use grim_tensor::dtype::ArithType;
        let dt = weight.dtype();
        if dt.arith != ArithType::F32 || !matches!(dt.storage, crate::DTypeStorage::Native) {
            return Err(Error::Unimplemented(
                "embedding_gather_dev_idx: F32 native tables only (quant falls back eager)".into(),
            ));
        }
        let w_ptr = weight
            .device_ptr
            .ok_or_else(|| Error::Backend("embedding_gather: weight has no device ptr".into()))?;
        let o_ptr = out
            .device_ptr
            .ok_or_else(|| Error::Backend("embedding_gather: out has no device ptr".into()))?;
        let i_ptr = indices
            .device_ptr
            .ok_or_else(|| Error::Backend("embedding_gather: indices has no device ptr".into()))?;
        let (grid, block) = crate::device::util::linear_launch(total);
        let mut w = w_ptr;
        let mut o = o_ptr;
        let mut i = i_ptr;
        let mut d = dim as i32;
        let mut t = total as i32;
        self.launch_compute_kernel(
            "grim_embedding",
            grid,
            block,
            &mut [
                arg(&mut w),
                arg(&mut o),
                arg(&mut i),
                arg(&mut d),
                arg(&mut t),
            ],
        )
    }

    /// SPEED-DOT-FUSED (Item 1): fuse Q/K/V projections into ONE dot4 GEMV.
    ///
    /// `wqkv_q80` is the byte-concatenated weight blob `[n_q + 2·n_kv, hidden]`
    /// (all Q rows, then all K rows, then all V rows — each a Q8_0-packed row of
    /// `(hidden/32)·34` bytes). A single `grim_dot4_q80_q81_gemv` launch with
    /// `N = n_q + 2·n_kv` writes the fused `attn_out [1, n_q + 2·n_kv]` f32, which
    /// the caller slices into q/k/v via [`crate::memory::view::RocmStorageView`].
    /// Replaces three separate GEMV launches (3 → 1) on the ROCm decode path.
    ///
    /// `act_q81` is the already-packed q8_1 activation (U8-typed) shared by all
    /// three projections. Requires M==1 (decode) and `hidden % 32 == 0`.
    pub fn launch_fused_qkv_dot4(
        &self,
        act_q81: &RocmStorage,
        wqkv_q80: &RocmStorage,
        n_q: usize,
        n_kv: usize,
        hidden: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let n_total = n_q
            .checked_add(
                n_kv.checked_mul(2)
                    .ok_or_else(|| Error::Backend("launch_fused_qkv_dot4: n_kv overflow".into()))?,
            )
            .ok_or_else(|| Error::Backend("launch_fused_qkv_dot4: n_q+n_kv overflow".into()))?;
        let out_shape = Shape::new(vec![n_total]);
        let out_storage = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_fused_qkv_dot4_into(act_q81, wqkv_q80, &out_storage, n_q, n_kv, hidden)?;
        Ok(Box::new(out_storage))
    }

    /// Fused Q8_0 QKV GEMV writing into CALLER-PROVIDED `out` ([n_total]
    /// F32) — no allocation inside. Graph-capture safe.
    pub fn launch_fused_qkv_dot4_into(
        &self,
        act_q81: &RocmStorage,
        wqkv_q80: &RocmStorage,
        out: &RocmStorage,
        n_q: usize,
        n_kv: usize,
        hidden: usize,
    ) -> Result<*mut c_void> {
        if hidden == 0 || hidden % 32 != 0 {
            return Err(Error::Backend(format!(
                "launch_fused_qkv_dot4: hidden must be a non-zero multiple of 32, got {hidden}"
            )));
        }
        let n_total = n_q
            .checked_add(
                n_kv.checked_mul(2)
                    .ok_or_else(|| Error::Backend("launch_fused_qkv_dot4: n_kv overflow".into()))?,
            )
            .ok_or_else(|| Error::Backend("launch_fused_qkv_dot4: n_q+n_kv overflow".into()))?;
        if wqkv_q80.shape().elem_count() != n_total * hidden {
            return Err(Error::Backend(format!(
                "launch_fused_qkv_dot4: weight blob has {} elements, expected {} (n_total={} * hidden={})",
                wqkv_q80.shape().elem_count(),
                n_total * hidden,
                n_total,
                hidden
            )));
        }
        if out.shape().elem_count() != n_total {
            return Err(Error::Backend(format!(
                "launch_fused_qkv_dot4_into: out holds {} elems, need {n_total}",
                out.shape().elem_count()
            )));
        }
        self.launch_dot4_q80_q81_gemv(act_q81, wqkv_q80, out, 1, n_total, hidden)
    }

    /// SPEED-DOT: Q8_0 GEMV via `V_DOT2_F32_f16` at M=1 (RDNA3/4).
    /// Grid (N,1,1), block (32,1,1) — one wave per output column, 100% lane
    /// SPEED-DOT-FUSED (Item 1): build the concatenated Q8_0 weight blob
    /// `[n_q + 2·n_kv, hidden]` from the three projection weight storages.
    ///
    /// Reads the raw Q8_0 bytes of each weight (D2H), concatenates them in
    /// row order Q∥K∥V, and re-uploads ONE device buffer. One-time cost at model
    /// load. Each weight must be Q8_0 with the same `hidden` (K) dimension and
    /// `hidden % 32 == 0`. Returns the fused storage plus the per-head row counts
    /// so the caller can slice the GEMV output.
    pub fn build_fused_qkv_q80(
        &self,
        wq: &dyn BackendStorage,
        wk: &dyn BackendStorage,
        wv: &dyn BackendStorage,
    ) -> Result<FusedQkvWeights> {
        let q80 = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
        };
        for (name, w) in [("wq", wq), ("wk", wk), ("wv", wv)] {
            if w.dtype().storage != q80.storage {
                return Err(Error::Backend(format!(
                    "build_fused_qkv_q80: {name} must be Q8_0, got {:?}",
                    w.dtype().storage
                )));
            }
        }
        let q_dims = wq.shape().dims();
        let k_dims = wk.shape().dims();
        let v_dims = wv.shape().dims();
        if q_dims.len() != 2 || k_dims.len() != 2 || v_dims.len() != 2 {
            return Err(Error::Backend(
                "build_fused_qkv_q80: weights must be 2D [rows, hidden]".into(),
            ));
        }
        let n_q = q_dims[0];
        let n_k = k_dims[0];
        let n_v = v_dims[0];
        let hidden_q = q_dims[1];
        let hidden_k = k_dims[1];
        let hidden_v = v_dims[1];
        if hidden_q != hidden_k || hidden_k != hidden_v {
            return Err(Error::Backend(format!(
                "build_fused_qkv_q80: hidden dims must match (got {hidden_q}/{hidden_k}/{hidden_v})"
            )));
        }
        let hidden = hidden_q;
        if hidden == 0 || hidden % 32 != 0 {
            return Err(Error::Backend(format!(
                "build_fused_qkv_q80: hidden must be a non-zero multiple of 32, got {hidden}"
            )));
        }
        // Raw Q8_0 bytes of each weight (D2H). Each row is (hidden/32)*34 bytes.
        let q_bytes = as_rocm(wq)?.copy_to_host()?;
        let k_bytes = as_rocm(wk)?.copy_to_host()?;
        let v_bytes = as_rocm(wv)?.copy_to_host()?;
        let mut fused = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
        fused.extend_from_slice(&q_bytes);
        fused.extend_from_slice(&k_bytes);
        fused.extend_from_slice(&v_bytes);
        let n_total = n_q + n_k + n_v;
        let fused_shape = Shape::new(vec![n_total, hidden]);
        let fused_storage = RocmStorage::copy_from_host_raw_bytes(
            &fused,
            &fused_shape,
            q80,
            &self.allocator,
            self.ordinal,
        )?;
        Ok(FusedQkvWeights {
            storage: fused_storage,
            n_q,
            n_k,
            n_v,
            n_gb: 0,
            n_gw: 0,
            n_gf: 0,
            hidden,
        })
    }

    /// G4b Phase 2: quantize one f32 weight storage to Q8_0 on-device and read
    /// back its raw Q8_0 bytes (row order preserved). Shared by the gate-blob
    /// builder and used to concatenate gate rows with the QKV Q8_0 bytes.
    fn gate_weight_q80_bytes(&self, w: &dyn BackendStorage) -> Result<Vec<u8>> {
        let (quantized, handle) = self.quantize_on_device(w, grim_tensor::QuantFormat::Q8_0)?;
        handle.synchronize()?;
        let q_storage = quantized
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Backend("gate quant: output is not RocmStorage".into()))?;
        q_storage.copy_to_host()
    }

    /// G4b Phase 2: build the gate-augmented fused Q8_0 blob
    /// `[Q ∥ K ∥ V ∥ Gb ∥ Gw ∥ Gf, hidden]` consumed by one fused QKV+gate GEMV.
    ///
    /// `wq`/`wk`/`wv` must already be Q8_0 (the teacher projections); the three
    /// gate projections `w_gb`/`w_gw`/`w_gf` are f32 and are quantized to Q8_0
    /// on-device here (same quant path the parity test references). Row counts
    /// `n_gb`/`n_gw`/`n_gf` are taken from the gate weights' outer dimension so
    /// the fused GEMV output can be sliced back into gate logits.
    pub fn build_fused_gate_qkv_q80(
        &self,
        wq: &dyn BackendStorage,
        wk: &dyn BackendStorage,
        wv: &dyn BackendStorage,
        w_gb: &dyn BackendStorage,
        w_gw: &dyn BackendStorage,
        w_gf: &dyn BackendStorage,
    ) -> Result<FusedQkvWeights> {
        let q80 = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
        };
        for (name, w) in [("wq", wq), ("wk", wk), ("wv", wv)] {
            if w.dtype().storage != q80.storage {
                return Err(Error::Backend(format!(
                    "build_fused_gate_qkv_q80: {name} must be Q8_0, got {:?}",
                    w.dtype().storage
                )));
            }
        }
        let q_dims = wq.shape().dims();
        let k_dims = wk.shape().dims();
        let v_dims = wv.shape().dims();
        if q_dims.len() != 2 || k_dims.len() != 2 || v_dims.len() != 2 {
            return Err(Error::Backend(
                "build_fused_gate_qkv_q80: QKV weights must be 2D [rows, hidden]".into(),
            ));
        }
        let n_q = q_dims[0];
        let n_k = k_dims[0];
        let n_v = v_dims[0];
        let hidden = q_dims[1];
        if k_dims[1] != hidden || v_dims[1] != hidden {
            return Err(Error::Backend(format!(
                "build_fused_gate_qkv_q80: QKV hidden dims must match (got {}/{}/{})",
                q_dims[1], k_dims[1], v_dims[1]
            )));
        }
        if hidden == 0 || hidden % 32 != 0 {
            return Err(Error::Backend(format!(
                "build_fused_gate_qkv_q80: hidden must be a non-zero multiple of 32, got {hidden}"
            )));
        }
        let n_gb = w_gb.shape().dims().first().copied().unwrap_or(0);
        let n_gw = w_gw.shape().dims().first().copied().unwrap_or(0);
        let n_gf = w_gf.shape().dims().first().copied().unwrap_or(0);

        // QKV Q8_0 bytes are read back directly; gate bytes are quantized first.
        let q_bytes = as_rocm(wq)?.copy_to_host()?;
        let k_bytes = as_rocm(wk)?.copy_to_host()?;
        let v_bytes = as_rocm(wv)?.copy_to_host()?;
        let gb_bytes = self.gate_weight_q80_bytes(w_gb)?;
        let gw_bytes = self.gate_weight_q80_bytes(w_gw)?;
        let gf_bytes = self.gate_weight_q80_bytes(w_gf)?;

        let mut fused = Vec::with_capacity(
            q_bytes.len()
                + k_bytes.len()
                + v_bytes.len()
                + gb_bytes.len()
                + gw_bytes.len()
                + gf_bytes.len(),
        );
        fused.extend_from_slice(&q_bytes);
        fused.extend_from_slice(&k_bytes);
        fused.extend_from_slice(&v_bytes);
        fused.extend_from_slice(&gb_bytes);
        fused.extend_from_slice(&gw_bytes);
        fused.extend_from_slice(&gf_bytes);

        let n_total = n_q + n_k + n_v + n_gb + n_gw + n_gf;
        let fused_shape = Shape::new(vec![n_total, hidden]);
        let fused_storage = RocmStorage::copy_from_host_raw_bytes(
            &fused,
            &fused_shape,
            q80,
            &self.allocator,
            self.ordinal,
        )?;
        Ok(FusedQkvWeights {
            storage: fused_storage,
            n_q,
            n_k,
            n_v,
            n_gb,
            n_gw,
            n_gf,
            hidden,
        })
    }

    /// G4b Phase 3: one fused Q8_0/Q8_1 GEMV over the gate-augmented blob,
    /// writing `[m, n_total_with_gates]` f32 into CALLER-PROVIDED `out`. No
    /// allocation inside — graph-capture safe. Slices of `out` are the Q/K/V and
    /// gate-logit regions. `m` is the activation batch (tokens); the caller must
    /// have quantized `m` rows into `act_q81` and sized `out` to `m * n_total`.
    pub fn launch_fused_qkv_gates_dot4_into(
        &self,
        act_q81: &RocmStorage,
        weights: &FusedQkvWeights,
        out: &RocmStorage,
        m: usize,
    ) -> Result<*mut c_void> {
        if weights.hidden == 0 || weights.hidden % 32 != 0 {
            return Err(Error::Backend(format!(
                "launch_fused_qkv_gates: hidden must be a non-zero multiple of 32, got {}",
                weights.hidden
            )));
        }
        let n_total = weights.n_total_with_gates();
        if out.shape().elem_count() != m * n_total {
            return Err(Error::Backend(format!(
                "launch_fused_qkv_gates_into: out holds {} elems, need {}",
                out.shape().elem_count(),
                m * n_total
            )));
        }
        self.launch_dot4_q80_q81_gemv(act_q81, &weights.storage, out, m, n_total, weights.hidden)
    }

    /// G4b Phase 3: allocate an output buffer, run the fused QKV+gate GEMV, and
    /// return zero-copy views over the three gate-logit regions plus the full
    /// output allocation (which keeps the views alive).
    pub fn fused_qkv_gates_dot4(
        &self,
        act_q81: &RocmStorage,
        weights: &FusedQkvWeights,
    ) -> Result<FusedQkvGateLogits> {
        let n_total = weights.n_total_with_gates();
        let out_storage = RocmStorage::alloc_gpu(
            &Shape::new(vec![n_total]),
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_fused_qkv_gates_dot4_into(act_q81, weights, &out_storage, 1)?;
        let output: Arc<dyn BackendStorage> = Arc::from(out_storage);
        let gb = RocmStorageView::from_offset(
            Arc::clone(&output),
            weights.gb_offset() * 4,
            weights.n_gb * 4,
            Shape::new(vec![weights.n_gb]),
        )?;
        let gw = RocmStorageView::from_offset(
            Arc::clone(&output),
            weights.gw_offset() * 4,
            weights.n_gw * 4,
            Shape::new(vec![weights.n_gw]),
        )?;
        let gf = RocmStorageView::from_offset(
            Arc::clone(&output),
            weights.gf_offset() * 4,
            weights.n_gf * 4,
            Shape::new(vec![weights.n_gf]),
        )?;
        Ok(FusedQkvGateLogits { output, gb, gw, gf })
    }

    /// SPEED-DOT-FUSED (Phase 4c): build the concatenated Q8_0 weight blob
    /// `[n_gate + n_up, hidden]` from gate and up projection weight storages.
    pub fn build_fused_gate_up_q80(
        &self,
        w_gate: &dyn BackendStorage,
        w_up: &dyn BackendStorage,
    ) -> Result<FusedGateUpWeights> {
        let q80 = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
        };
        for (name, w) in [("w_gate", w_gate), ("w_up", w_up)] {
            if w.dtype().storage != q80.storage {
                return Err(Error::Backend(format!(
                    "build_fused_gate_up_q80: {name} must be Q8_0, got {:?}",
                    w.dtype().storage
                )));
            }
        }
        let g_dims = w_gate.shape().dims();
        let u_dims = w_up.shape().dims();
        if g_dims.len() != 2 || u_dims.len() != 2 {
            return Err(Error::Backend(
                "build_fused_gate_up_q80: weights must be 2D [rows, hidden]".into(),
            ));
        }
        let n_gate = g_dims[0];
        let n_up = u_dims[0];
        let hidden_g = g_dims[1];
        let hidden_u = u_dims[1];
        if hidden_g != hidden_u {
            return Err(Error::Backend(format!(
                "build_fused_gate_up_q80: hidden dims must match (got {hidden_g}/{hidden_u})"
            )));
        }
        let hidden = hidden_g;
        if hidden == 0 || hidden % 32 != 0 {
            return Err(Error::Backend(format!(
                "build_fused_gate_up_q80: hidden must be a non-zero multiple of 32, got {hidden}"
            )));
        }
        let n_total = n_gate + n_up;
        let fused_shape = Shape::new(vec![n_total, hidden]);
        // N1: concat on-device (hipMemcpy D2D) — no host round-trip of the
        // full weight blob at load time.
        let fused_storage =
            RocmStorage::alloc_gpu(&fused_shape, q80, &self.allocator, self.ordinal)?;
        let dst = fused_storage.device_ptr_checked()? as *mut std::ffi::c_void;
        let g_src = as_rocm(w_gate)?.device_ptr_checked()? as *const std::ffi::c_void;
        let u_src = as_rocm(w_up)?.device_ptr_checked()? as *const std::ffi::c_void;
        let g_bytes = as_rocm(w_gate)?.bytes;
        let u_bytes = as_rocm(w_up)?.bytes;
        unsafe {
            check_hip(
                "build_fused_gate_up_q80: D2D gate",
                crate::device::handles::hipMemcpy(
                    dst,
                    g_src,
                    g_bytes,
                    crate::device::handles::HipMemcpyKind::DeviceToDevice,
                ),
            )?;
            check_hip(
                "build_fused_gate_up_q80: D2D up",
                crate::device::handles::hipMemcpy(
                    dst.add(g_bytes),
                    u_src,
                    u_bytes,
                    crate::device::handles::HipMemcpyKind::DeviceToDevice,
                ),
            )?;
        }
        Ok(FusedGateUpWeights {
            storage: fused_storage,
            n_gate,
            n_up,
            hidden,
        })
    }

    /// SPEED-DOT-FUSED (Phase 4c): fuse FFN gate+up projections into ONE dot4 GEMV.
    pub fn launch_fused_gate_up_dot4(
        &self,
        act_q81: &RocmStorage,
        fused_w: &RocmStorage,
        n_gate: usize,
        n_up: usize,
        hidden: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let n_total = n_gate.checked_add(n_up).ok_or_else(|| {
            Error::Backend("launch_fused_gate_up_dot4: n_gate+n_up overflow".into())
        })?;
        let out_shape = Shape::new(vec![n_total]);
        let out_storage = RocmStorage::alloc_gpu(
            &out_shape,
            DType {
                arith: ArithType::F32,
                storage: DTypeStorage::Native,
            },
            &self.allocator,
            self.ordinal,
        )?;
        self.launch_fused_gate_up_dot4_into(act_q81, fused_w, &out_storage, n_gate, n_up, hidden)?;
        Ok(Box::new(out_storage))
    }

    /// Fused Q8_0 gate+up GEMV writing into CALLER-PROVIDED `out`
    /// ([n_gate+n_up] F32) — no allocation inside. Graph-capture safe.
    pub fn launch_fused_gate_up_dot4_into(
        &self,
        act_q81: &RocmStorage,
        fused_w: &RocmStorage,
        out: &RocmStorage,
        n_gate: usize,
        n_up: usize,
        hidden: usize,
    ) -> Result<*mut c_void> {
        if hidden == 0 || hidden % 32 != 0 {
            return Err(Error::Backend(format!(
                "launch_fused_gate_up_dot4: hidden must be a non-zero multiple of 32, got {hidden}"
            )));
        }
        let n_total = n_gate.checked_add(n_up).ok_or_else(|| {
            Error::Backend("launch_fused_gate_up_dot4: n_gate+n_up overflow".into())
        })?;
        if fused_w.shape().elem_count() != n_total * hidden {
            return Err(Error::Backend(format!(
                "launch_fused_gate_up_dot4: weight blob has {} elements, expected {} (n_total={} * hidden={})",
                fused_w.shape().elem_count(),
                n_total * hidden,
                n_total,
                hidden
            )));
        }
        if out.shape().elem_count() != n_total {
            return Err(Error::Backend(format!(
                "launch_fused_gate_up_dot4_into: out holds {} elems, need {n_total}",
                out.shape().elem_count()
            )));
        }
        self.launch_dot4_q80_q81_gemv(act_q81, fused_w, out, 1, n_total, hidden)
    }

    /// SPEED-DOT-OPFUSE (Phase 4d): fused SwiGLU + Q8_1 quantization for M=1 decode.
    pub fn launch_silu_mul_quant_q8_1(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        dst_q81: &dyn BackendStorage,
        k: usize,
    ) -> Result<*mut c_void> {
        let g_ptr = crate::device::util::dev_ptr_dyn(gate)?;
        let u_ptr = crate::device::util::dev_ptr_dyn(up)?;
        let dst_ptr = crate::device::util::dev_ptr_dyn(dst_q81)?;
        let grid_dim = HipDim3::new(1, 1, 1);
        let block_dim = HipDim3::new(32, 1, 1);
        let mut gptr = g_ptr;
        let mut uptr = u_ptr;
        let mut dptr = dst_ptr;
        let mut kk = k as i32;
        self.launch_compute_kernel(
            "grim_silu_mul_quant_q8_1",
            grid_dim,
            block_dim,
            &mut [arg(&mut gptr), arg(&mut uptr), arg(&mut dptr), arg(&mut kk)],
        )
    }

    /// WI-F1 - Fused QKV projection GEMM. `qkv_weight` must be the load-time concatenation of the per-layer Q/K/V projection weights along the
    /// output dim - row-major `[hidden, q_dim + k_dim + v_dim]`, built once at model load via [`crate::fusion::concat_qkv_weights`] (never per forward pass).
    pub fn fused_qkv_proj(
        &self,
        x: &dyn BackendStorage,
        qkv_weight: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_dims = x.shape().dims();
        let w_dims = qkv_weight.shape().dims();
        if x_dims.len() < 2 || w_dims.len() < 2 {
            return Err(Error::Shape(
                "fused_qkv_proj expects rank >= 2 inputs".into(),
            ));
        }
        let hidden = x_dims[x_dims.len() - 1];
        // SPEED-ROC-16 contract: the fused weight blob is row-major
        // [qkv_dim, hidden] (B = [N, K]); the shared K is its trailing dim.
        let k2 = w_dims[w_dims.len() - 1];
        let qkv_dim = qkv_weight.shape().elem_count() / k2;
        if hidden != k2 {
            return Err(Error::ShapeMismatch {
                expected: x_dims.to_vec(),
                got: w_dims.to_vec(),
            });
        }
        let tokens = x.shape().elem_count() / hidden;
        if out_shape.elem_count() != tokens * qkv_dim {
            return Err(Error::Shape(format!(
                "expected out elem_count {}, got {:?}",
                tokens * qkv_dim,
                out_shape.dims()
            )));
        }
        self.matmul_op(x, qkv_weight, out_shape, crate::autotune::GemmOp::Attention)
    }

    /// WI-F2 - Fused attention output projection.
    /// Runs the same fused QKV attention kernel (`grim_qkv_attention`) with the O-projection applied in the kernel.
    pub fn fused_attn_o_proj(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        o_proj: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        // P1-3: raw HIP ops below bind to the calling thread's current
        // device — pin to the owning ordinal (see matmul_op fix, 2026-08-23e).
        let _dev_guard = crate::device::util::DeviceGuard::set(self.ordinal as i32);
        let q_dims = q.shape().dims();
        let o_dims = out_shape.dims();
        if q_dims.len() != 3 || o_dims.len() != 2 {
            return Err(Error::Shape(
                "fused_attn_o_proj expects q [seq, heads, head_dim] and out [seq, o_dim]".into(),
            ));
        }
        let seq_len = q_dims[0];
        let num_heads = q_dims[1];
        let head_dim = q_dims[2];
        let o_dim = o_dims[1];
        if o_dims[0] != seq_len {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj: out rows {} must equal seq_len {seq_len}",
                o_dims[0]
            )));
        }
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || o_dim == 0 {
            return Err(Error::Shape(
                "fused_attn_o_proj: zero-sized heads / head_dim / o_dim".into(),
            ));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj: num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
            )));
        }
        if head_dim > 256 {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj supports head_dim <= 256 (got {head_dim})"
            )));
        }
        let o_s = as_rocm(o_proj)?;
        if o_proj.shape().elem_count() != num_heads * head_dim * o_dim {
            return Err(Error::Shape(format!(
                "fused_attn_o_proj: o_proj must be [num_heads*head_dim, o_dim] = {} elems (got {})",
                num_heads * head_dim * o_dim,
                o_proj.shape().elem_count()
            )));
        }
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        if !q_s.device_ptr_is_valid()
            || !k_s.device_ptr_is_valid()
            || !v_s.device_ptr_is_valid()
            || !o_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_attn_o_proj: inputs lack a valid device pointer".into(),
            ));
        }

        let config = QkvAttentionFusionConfig {
            enabled: true,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len: seq_len,
            wavefront_size: self.props.wavefront_size as u32,
            quant_mode: QuantMode::Fp32,
        };
        let launch = config.hip_launch_params();
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let out_ptr = dev_ptr(&storage)?;

        // atomicAdd accumulation across heads requires a zeroed output;
        // async memset on the active stream keeps it stream-ordered.
        let res = unsafe {
            hipMemsetAsync(
                out_ptr as *mut c_void,
                0,
                storage.bytes,
                self.active_stream(),
            )
        };
        if res != hipSuccess {
            return Err(Error::Backend(format!(
                "fused_attn_o_proj: hipMemsetAsync failed with status {res}"
            )));
        }

        let q_ptr = dev_ptr(q_s)?;
        let k_ptr = dev_ptr(k_s)?;
        let v_ptr = dev_ptr(v_s)?;
        // SPEED-ROC-16: the fused grim_qkv_attention kernel hardcodes a [K, N]
        // o_proj index (`o_proj_w[(h*head_dim+d)*o_dim + oc]`), but the new matmul
        // contract delivers o_proj in [N, K]. Transpose to [K, N] for the kernel.
        let o_proj_kn = if o_dim == num_heads * head_dim {
            let src = o_s.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
                Error::Backend("fused_attn_o_proj: o_proj is not RocmStorage".into())
            })?;
            let transposed = self.transpose_f32_2d(src, o_dim, num_heads * head_dim)?;
            let transposed_s = as_rocm(transposed.as_ref())?;
            dev_ptr(transposed_s)?
        } else {
            dev_ptr(o_s)?
        };

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut max_ptr: u64 = 0;
        let mut sum_ptr: u64 = 0;
        let mut nh = num_heads as i32;
        let mut nkv = num_kv_heads as i32;
        let mut hd = head_dim as i32;
        let mut sl = seq_len as i32;
        let mut ksl = kv_seq_len as i32;
        let mut co = cache_offset as i32;
        let mut isd: f32 = 1.0 / (head_dim as f32).sqrt();
        let mut wlo: i32 = 0;
        let mut softcap: f32 = self.attn_logit_softcap();
        let mut oproj_ptr = o_proj_kn;
        let mut odim = o_dim as i32;
        let mut fuseo: i32 = 1;
        let mut alibi_ptr: u64 = 0;
        let mut has_alibi: i32 = 0;

        let stream = self.launch_compute_kernel(
            "grim_qkv_attention",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut max_ptr),
                arg(&mut sum_ptr),
                arg(&mut nh),
                arg(&mut nkv),
                arg(&mut hd),
                arg(&mut sl),
                arg(&mut ksl),
                arg(&mut co),
                arg(&mut isd),
                arg(&mut wlo),
                arg(&mut softcap),
                arg(&mut oproj_ptr),
                arg(&mut odim),
                arg(&mut fuseo),
                arg(&mut alibi_ptr),
                arg(&mut has_alibi),
            ],
        )?;
        let _ = (
            qptr, kptr, vptr, optr, max_ptr, sum_ptr, nh, nkv, hd, sl, ksl, co, isd, wlo, softcap,
            oproj_ptr, odim, fuseo, alibi_ptr, has_alibi,
        );
        Ok((Box::new(storage), Box::new(RocmHandle::new(Some(stream)))))
    }

    /// Dispatch a fused RMSNorm + MatMul operation onto the GPU.
    pub fn rmsnorm_matmul(
        &self,
        x: &dyn BackendStorage,
        w_norm: &dyn BackendStorage,
        weight_mat: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let w_norm_s = as_rocm(w_norm)?;
        let w_mat_s = as_rocm(weight_mat)?;
        if !x_s.device_ptr_is_valid()
            || !w_norm_s.device_ptr_is_valid()
            || !w_mat_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "rmsnorm_matmul: inputs lack a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        let w_mat_dims = weight_mat.shape().dims();
        if x_dims.len() < 2 || w_mat_dims.len() < 2 {
            return Err(Error::Shape(
                "rmsnorm_matmul expects rank >= 2 inputs".into(),
            ));
        }
        let k = x_dims[x_dims.len() - 1];
        let m = x.shape().elem_count() / k;
        let n = w_mat_dims[w_mat_dims.len() - 1];
        let k2 = weight_mat.shape().elem_count() / n;
        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: x_dims.to_vec(),
                got: w_mat_dims.to_vec(),
            });
        }
        if out_shape.elem_count() != m * n {
            return Err(Error::Shape(format!(
                "expected out elem_count {}, got {:?}",
                m * n,
                out_shape.dims()
            )));
        }

        let config = RmsNormMatMulFusionConfig {
            hidden_size: k,
            intermediate_size: n,
            wavefront_size: self.props.wavefront_size as u32,
            lds_size: 65536,
        };
        let launch = config.hip_launch_params();

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_norm_ptr = dev_ptr(w_norm_s)?;
        let mut w_mat_ptr = dev_ptr(w_mat_s)?;
        let mut m_i = m as i32;
        let mut n_i = n as i32;
        let mut k_i = k as i32;
        let mut eps_f = eps;

        self.launch_compute_kernel(
            "grim_rmsnorm_matmul",
            launch.grid_dim,
            launch.block_dim,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_norm_ptr),
                arg(&mut w_mat_ptr),
                arg(&mut out_ptr),
                arg(&mut m_i),
                arg(&mut n_i),
                arg(&mut k_i),
                arg(&mut eps_f),
            ],
        )?;

        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// High-level fused RMSNorm + MXFP4 GEMM (e.g. for MLP gate/up/down projections).
    pub fn fused_rmsnorm_mxfp4_gemm(
        &self,
        x: &dyn BackendStorage,
        gamma: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        m: usize,
        n: usize,
        k: usize,
        eps: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let gamma_s = as_rocm(gamma)?;
        let w_codes_s = as_rocm(w_codes)?;
        let w_exps_s = as_rocm(w_exps)?;

        let out_shape = Shape::new(vec![m, n]);
        let out_storage =
            RocmStorage::alloc_gpu(&out_shape, dtype_f32(), &self.allocator, self.ordinal)?;

        self.launch_fused_rmsnorm_mxfp4_gemm(
            x_s,
            gamma_s,
            w_codes_s,
            w_exps_s,
            &out_storage,
            m,
            n,
            k,
            eps,
        )?;

        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// High-level fused RMSNorm + MXFP4 GEMM + RoPE + direct KV cache scatter.
    pub fn fused_rmsnorm_mxfp4_gemm_rope_kv(
        &self,
        x: &dyn BackendStorage,
        gamma: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        q_out: Option<&dyn BackendStorage>,
        k_cache: Option<&dyn BackendStorage>,
        v_cache: Option<&dyn BackendStorage>,
        out_all: Option<&dyn BackendStorage>,
        positions: Option<&dyn BackendStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq: Option<&dyn BackendStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
        rope_interleaved: bool,
    ) -> Result<Box<dyn ComputeHandle>> {
        let x_s = as_rocm(x)?;
        let gamma_s = as_rocm(gamma)?;
        let w_codes_s = as_rocm(w_codes)?;
        let w_exps_s = as_rocm(w_exps)?;

        let q_out_s = match q_out {
            Some(q) => Some(as_rocm(q)?),
            None => None,
        };
        let k_cache_s = match k_cache {
            Some(k) => Some(as_rocm(k)?),
            None => None,
        };
        let v_cache_s = match v_cache {
            Some(v) => Some(as_rocm(v)?),
            None => None,
        };
        let out_all_s = match out_all {
            Some(a) => Some(as_rocm(a)?),
            None => None,
        };
        let positions_s = match positions {
            Some(p) => Some(as_rocm(p)?),
            None => None,
        };
        let inv_freq_s = match inv_freq {
            Some(f) => Some(as_rocm(f)?),
            None => None,
        };

        self.launch_fused_rmsnorm_mxfp4_gemm_rope_kv(
            x_s,
            gamma_s,
            w_codes_s,
            w_exps_s,
            q_out_s,
            k_cache_s,
            v_cache_s,
            out_all_s,
            positions_s,
            m,
            k,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            inv_freq_s,
            mscale,
            eps,
            max_seq_len,
            rope_interleaved,
        )?;

        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// LFM2-style fused QKV projection: MXFP4 GEMM (C = x @ W_qkv) followed by per-head QK-Norm + RoPE (YaRN-aware).
    /// Mirrors `fused_rmsnorm_mxfp4_gemm_rope_kv` but applies the normalization *after* the projection (QK-norm) instead of before it, matching.
    pub fn fused_mxfp4_gemm_qk_norm_rope_kv(
        &self,
        x: &dyn BackendStorage,
        gamma_q: &dyn BackendStorage,
        gamma_k: &dyn BackendStorage,
        w_codes: &dyn BackendStorage,
        w_exps: &dyn BackendStorage,
        q_out: Option<&dyn BackendStorage>,
        k_cache: Option<&dyn BackendStorage>,
        v_cache: Option<&dyn BackendStorage>,
        out_all: Option<&dyn BackendStorage>,
        positions: Option<&dyn BackendStorage>,
        m: usize,
        k: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        inv_freq: Option<&dyn BackendStorage>,
        mscale: f32,
        eps: f32,
        max_seq_len: usize,
        rope_interleaved: bool,
    ) -> Result<Box<dyn ComputeHandle>> {
        let x_s = as_rocm(x)?;
        let gamma_q_s = as_rocm(gamma_q)?;
        let gamma_k_s = as_rocm(gamma_k)?;
        let w_codes_s = as_rocm(w_codes)?;
        let w_exps_s = as_rocm(w_exps)?;
        let q_out_s = q_out.map(|q| as_rocm(q)).transpose()?;
        let k_cache_s = k_cache.map(|k| as_rocm(k)).transpose()?;
        let v_cache_s = v_cache.map(|v| as_rocm(v)).transpose()?;
        let out_all_s = out_all.map(|a| as_rocm(a)).transpose()?;
        let positions_s = positions.map(|p| as_rocm(p)).transpose()?;
        let inv_freq_s = inv_freq.map(|f| as_rocm(f)).transpose()?;

        self.launch_fused_mxfp4_gemm_qk_norm_rope_kv(
            x_s,
            gamma_q_s,
            gamma_k_s,
            w_codes_s,
            w_exps_s,
            q_out_s,
            k_cache_s,
            v_cache_s,
            out_all_s,
            positions_s,
            m,
            k,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            rope_theta,
            inv_freq_s,
            mscale,
            eps,
            max_seq_len,
            rope_interleaved,
        )?;

        Ok(Box::new(RocmHandle::new(Some(self.active_stream()))))
    }

    /// Fused Add + RMSNorm kernel.
    /// Computes `y = x + residual` and `norm_out = RMSNorm(y, weight, eps)` in a single HIP kernel pass.
    pub fn fused_add_rms_norm(
        &self,
        x: &dyn BackendStorage,
        residual: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let x_s = as_rocm(x)?;
        let res_s = as_rocm(residual)?;
        let w_s = as_rocm(weight)?;
        if !x_s.device_ptr_is_valid() || !res_s.device_ptr_is_valid() || !w_s.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_add_rms_norm: inputs lack a valid device pointer".into(),
            ));
        }
        let x_dims = x.shape().dims();
        if x_dims.is_empty() {
            return Err(Error::Shape("fused_add_rms_norm: empty input".into()));
        }
        let row_len = x_dims
            .last()
            .copied()
            .ok_or_else(|| Error::Shape("empty tensor dims".into()))?;
        let total = out_shape.elem_count();
        let y_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let norm_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut res_ptr = dev_ptr(res_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut y_out_ptr = dev_ptr(&y_storage)?;
        let mut norm_out_ptr = dev_ptr(&norm_storage)?;
        let mut row_len_i = row_len as i32;
        let mut eps_f = eps;
        let mut total_i = total as i32;
        // grim_add_rms_norm is warp-per-row (32 lanes reduce with shuffles).
        let (grid, block) = warp_rows_launch(total / row_len.max(1));
        self.launch_compute_kernel(
            "grim_add_rms_norm",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut res_ptr),
                arg(&mut w_ptr),
                arg(&mut y_out_ptr),
                arg(&mut norm_out_ptr),
                arg(&mut row_len_i),
                arg(&mut eps_f),
                arg(&mut total_i),
            ],
        )?;
        Ok((
            Box::new(y_storage),
            Box::new(norm_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Launch the Design-A on-device linear cross-entropy forward pass.
    pub fn fused_linear_cross_entropy_forward(
        &self,
        hidden: &dyn BackendStorage,
        lm_head: &dyn BackendStorage,
        targets: &dyn BackendStorage,
        v_tile_size: i32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let h = as_rocm(hidden)?;
        let w = as_rocm(lm_head)?;
        let t = as_rocm(targets)?;
        if !h.device_ptr_is_valid() || !w.device_ptr_is_valid() || !t.device_ptr_is_valid() {
            return Err(Error::Backend(
                "fused_linear_ce: invalid input pointer".into(),
            ));
        }
        let hd = hidden.shape().dims();
        let wd = lm_head.shape().dims();
        let td = targets.shape().dims();
        if hd.len() != 2 || wd.len() != 2 || td.len() != 1 || td[0] != hd[0] || wd[1] != hd[1] {
            return Err(Error::Shape(
                "fused_linear_ce: incompatible input shapes".into(),
            ));
        }
        if v_tile_size <= 0 {
            return Err(Error::Backend(
                "fused_linear_ce: v_tile_size must be positive".into(),
            ));
        }
        let batch = hd[0];
        let loss = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let lse = RocmStorage::alloc_gpu(
            &Shape::new(vec![batch]),
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        let mut hp = dev_ptr(h)?;
        let mut wp = dev_ptr(w)?;
        let mut tp = dev_ptr(t)?;
        let mut lp = dev_ptr(&loss)?;
        let mut ep = dev_ptr(&lse)?;
        let mut k = hd[1] as i32;
        let mut v = wd[0] as i32;
        let mut tile = v_tile_size;
        let mut b = batch as i32;
        let block = crate::HipDim3 { x: 256, y: 1, z: 1 };
        let grid = crate::HipDim3 {
            x: batch as u32,
            y: 1,
            z: 1,
        };
        self.launch_compute_kernel(
            "grim_fused_linear_ce_forward",
            grid,
            block,
            &mut [
                arg(&mut hp),
                arg(&mut wp),
                arg(&mut tp),
                arg(&mut lp),
                arg(&mut ep),
                arg(&mut k),
                arg(&mut v),
                arg(&mut tile),
                arg(&mut b),
            ],
        )?;
        Ok((
            Box::new(loss),
            Box::new(lse),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Launch the Design-A on-device linear cross-entropy backward pass.
    pub fn fused_linear_cross_entropy_backward(
        &self,
        hidden: &dyn BackendStorage,
        lm_head: &dyn BackendStorage,
        targets: &dyn BackendStorage,
        lse: &dyn BackendStorage,
        v_tile_size: i32,
        inv_batch: f32,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let h = as_rocm(hidden)?;
        let w = as_rocm(lm_head)?;
        let t = as_rocm(targets)?;
        let e = as_rocm(lse)?;
        if !h.device_ptr_is_valid()
            || !w.device_ptr_is_valid()
            || !t.device_ptr_is_valid()
            || !e.device_ptr_is_valid()
        {
            return Err(Error::Backend(
                "fused_linear_ce: invalid input pointer".into(),
            ));
        }
        let hd = hidden.shape().dims();
        let wd = lm_head.shape().dims();
        if hd.len() != 2 || wd.len() != 2 || wd[1] != hd[1] || targets.shape().elem_count() != hd[0]
        {
            return Err(Error::Shape(
                "fused_linear_ce: incompatible input shapes".into(),
            ));
        }
        if v_tile_size <= 0 {
            return Err(Error::Backend(
                "fused_linear_ce: v_tile_size must be positive".into(),
            ));
        }
        let batch = hd[0];
        let grad =
            RocmStorage::alloc_gpu(hidden.shape(), dtype_f32(), &self.allocator, self.ordinal)?;
        let mut hp = dev_ptr(h)?;
        let mut wp = dev_ptr(w)?;
        let mut tp = dev_ptr(t)?;
        let mut ep = dev_ptr(e)?;
        let mut gp = dev_ptr(&grad)?;
        let mut k = hd[1] as i32;
        let mut v = wd[0] as i32;
        let mut tile = v_tile_size;
        let mut inv = inv_batch;
        let mut b = batch as i32;
        let block = crate::HipDim3 { x: 256, y: 1, z: 1 };
        let grid = crate::HipDim3 {
            x: batch as u32,
            y: 1,
            z: 1,
        };
        self.launch_compute_kernel(
            "grim_fused_linear_ce_backward",
            grid,
            block,
            &mut [
                arg(&mut hp),
                arg(&mut wp),
                arg(&mut tp),
                arg(&mut ep),
                arg(&mut gp),
                arg(&mut k),
                arg(&mut v),
                arg(&mut tile),
                arg(&mut inv),
                arg(&mut b),
            ],
        )?;
        Ok((
            Box::new(grad),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Cross-entropy loss + softmax gradient, host-staged. Returns `(avg_loss, grad)`.
    pub fn cross_entropy_gpu(
        &self,
        logits: &dyn BackendStorage,
        targets: &[usize],
        label_smoothing: Option<f32>,
    ) -> Result<(f32, Box<dyn BackendStorage>)> {
        let l_s = as_rocm(logits)?;
        if !l_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "cross_entropy_gpu: logits lack a valid device pointer".into(),
            ));
        }

        let dims = logits.shape().dims();
        if dims.len() != 2 {
            return Err(Error::Shape(
                "cross_entropy_gpu: logits must be 2-D [batch_size, vocab_size]".into(),
            ));
        }
        let batch_size = dims[0];
        let vocab_size = dims[1];
        if batch_size == 0 {
            return Err(Error::Backend(
                "cross_entropy_gpu: batch_size must be > 0".into(),
            ));
        }
        if targets.len() != batch_size {
            return Err(Error::Shape(format!(
                "cross_entropy_gpu: targets len {} != batch_size {}",
                targets.len(),
                batch_size
            )));
        }
        let smooth = label_smoothing.unwrap_or(0.0).clamp(0.0, 1.0);
        let uniform = smooth / (vocab_size as f32);
        let confident = 1.0 - smooth;

        let logits_vec = logits.to_cpu_vec_f32()?;
        if logits_vec.len() < batch_size * vocab_size {
            return Err(Error::Backend(format!(
                "cross_entropy_gpu: logits length {} < batch_size * vocab_size {}",
                logits_vec.len(),
                batch_size * vocab_size
            )));
        }

        let mut grad_vec = vec![0.0f32; batch_size * vocab_size];
        let mut total_loss = 0.0f32;
        let inv_batch = 1.0 / (batch_size as f32);

        for (b, &target_token) in targets.iter().enumerate() {
            if target_token >= vocab_size {
                return Err(Error::Backend(format!(
                    "cross_entropy_gpu: target token {} out of bounds for vocab_size {}",
                    target_token, vocab_size
                )));
            }

            let row_start = b * vocab_size;
            let row_logits = &logits_vec[row_start..row_start + vocab_size];

            // Max trick for numerical stability.
            let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut exp_logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let exp_val = (row_logits[v] - max_logit).exp();
                exp_logits[v] = exp_val;
                sum_exp += exp_val;
            }
            let log_sum_exp = max_logit + sum_exp.ln();

            // Cross-entropy with optional label smoothing: loss = -sum_v q(v) * log_softmax(v) where log_softmax(v) = row_logits[v] - log_sum_exp, and the target distribution is q(target) = confident, q(other) = uniform.
            // This collapses to: confident * (log_sum_exp - logit_target) + uniform * (vocab_size * log_sum_exp -.
            let log_target = log_sum_exp - row_logits[target_token];
            let sum_logits: f32 = row_logits.iter().sum();
            let smooth_loss = uniform * ((vocab_size as f32) * log_sum_exp - sum_logits);
            total_loss += confident * log_target + smooth_loss;

            // Gradient dL/dLogits = (softmax - q) / batch_size.
            for v in 0..vocab_size {
                let prob = exp_logits[v] / sum_exp;
                let target_q = if v == target_token {
                    confident + uniform
                } else {
                    uniform
                };
                grad_vec[row_start + v] = (prob - target_q) * inv_batch;
            }
        }

        let avg_loss = total_loss * inv_batch;
        let grad_shape = logits.shape().clone();
        let grad_storage = RocmStorage::copy_from_host(
            &grad_vec,
            &grad_shape,
            dtype_f32(),
            &self.allocator,
            self.ordinal,
        )?;
        Ok((avg_loss, Box::new(grad_storage) as Box<dyn BackendStorage>))
    }
}
