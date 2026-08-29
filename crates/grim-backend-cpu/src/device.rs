//! CPU `BackendDevice`.
//!
//! ## GEMM dispatch (§4.1 — OxiBLAS)
//!
//! Matrix multiplication routes through [`gemm_dispatch`]:
//!
//! 1. `M=1` GEMV fast path — single-token decode, avoids GEMM overhead.
//! 2. `oxiblas` feature: `matrixmultiply::sgemm` — pure-Rust SIMD BLAS.
//! 3. Scalar fallback triple-loop — for no-SIMD / fuzzing / `--no-default-features`.

use std::sync::Arc;

use grim_tensor::backend::{ComputeHandle, ReadyHandle};
use grim_tensor::dtype::{DType, Device, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, Shape, Tensor,
    CoreTensorOps, ElementwiseOps, SamplingOps, AttentionOps, FusionOps, AutogradOps, OptimizerOps, QuantOps, RecurrentOps, CollectiveOps, MemoryOps, GraphCaptureOps,
};

use crate::storage::CpuStorage;

/// CPU device. All operations are synchronous; returns `ReadyHandle`.
#[derive(Clone, Default)]
pub struct CpuDevice {
    pub(crate) graphs: std::sync::Arc<crate::graph_capture::CpuGraphRegistry>,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            graphs: std::sync::Arc::new(crate::graph_capture::CpuGraphRegistry::new()),
        }
    }

    /// Return the probed CPU hardware specification snapshot.
    pub fn hardware_spec(&self) -> crate::hardware_spec::CpuHardwareSpec {
        crate::hardware_spec::CpuHardwareSpec::probe()
    }

    /// Return the probed CPU NUMA topology snapshot.
    pub fn topology(&self) -> crate::topology::CpuNumaTopology {
        crate::topology::CpuNumaTopology::probe()
    }

    /// Generate a hardware-fingerprinted cache key for primitive autotuning.
    pub fn cache_key(&self, entry: &str, source_hash: u64) -> crate::cache::CpuCacheKey {
        let spec = self.hardware_spec();
        crate::cache::CpuCacheKey::from_spec(entry, &spec, source_hash)
    }

    /// Begin capturing operations into a CPU computation graph under `key`.
    pub fn begin_graph_capture(&self, key: &str) -> Result<()> {
        self.graphs.begin_capture(key)
    }

    /// End the active graph capture session and save the graph under `key`.
    pub fn end_graph_capture(&self, key: &str) -> Result<()> {
        self.graphs.end_capture(key)
    }

    /// Replay a previously captured graph. Returns `Ok(true)` if replayed, `Ok(false)` if key not found.
    pub fn replay_graph(&self, key: &str) -> Result<bool> {
        self.graphs.replay(key)
    }

    /// Check if graph capture is currently active.
    pub fn is_capturing(&self) -> bool {
        self.graphs.is_capturing()
    }

    /// Record an operation closure into the current active graph capture session.
    pub fn record_op<F>(&self, op: F)
    where
        F: Fn() -> Result<()> + Send + Sync + 'static,
    {
        self.graphs.record_op(op);
    }
}

/// Dequantize a packed K/V cache tensor of layout `[kv_seq_len, num_kv_heads, head_dim]`
/// (8-bit: 1 elem/byte; 4-bit: 2 elems/byte) using per-row f32 scales into a
/// row-major f32 buffer the reference attention can consume.
fn dequant_packed_kv(
    packed: &CpuStorage,
    scales: &CpuStorage,
    num_kv_heads: usize,
    kv_seq_len: usize,
    head_dim: usize,
    quant_bits: u32,
) -> Result<Vec<f32>> {
    let bytes = packed
        .raw_bytes
        .as_ref()
        .map(|b| (**b).clone())
        .unwrap_or_else(|| packed.data().iter().map(|&f| f as u8).collect());
    let scale_data = scales.data();
    let rows = kv_seq_len * num_kv_heads;
    let elems_per_row = head_dim;
    let total = rows * elems_per_row;
    let mut out = vec![0.0f32; total];

    if quant_bits == 8 {
        for r in 0..rows {
            let s = scale_data.get(r).copied().unwrap_or(1.0f32);
            for d in 0..elems_per_row {
                let b = *bytes.get(r * elems_per_row + d).unwrap_or(&0);
                out[r * elems_per_row + d] = (b as i8) as f32 * s;
            }
        }
    } else {
        // 4-bit: two values per byte (low nibble first, then high nibble).
        for r in 0..rows {
            let s = scale_data.get(r).copied().unwrap_or(1.0f32);
            for d in 0..elems_per_row {
                let byte = *bytes.get(r * elems_per_row / 2 + d / 2).unwrap_or(&0);
                let nibble = if d % 2 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                let signed = ((nibble as i8) << 4) >> 4; // sign-extend 4-bit
                out[r * elems_per_row + d] = signed as f32 * s;
            }
        }
    }
    Ok(out)
}

impl CpuDevice {
    /// Shared scalar GQA attention core with optional ALiBi bias. Used by
    /// both the plain and ALiBi trait methods so the reference math stays
    /// in one place.
    #[allow(clippy::too_many_arguments)]
    fn qkv_attention_inner(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        alibi: Option<&[f32]>,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_st = a_storage(q)?;
        let k_st = a_storage(k)?;
        let v_st = a_storage(v)?;
        let q_dims = q_st.shape().dims();
        let k_dims = k_st.shape().dims();

        if q_dims.len() < 2 || k_dims.len() < 2 {
            return Err(Error::Shape("qkv_attention: q/k/v must be >= 2-D".into()));
        }
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 {
            return Err(Error::Shape("qkv_attention: out_shape mismatch".into()));
        }
        let seq_len = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];
        if k_dims[k_dims.len() - 1] != head_dim || v_st.shape().dims().len() < 2 {
            return Err(Error::Shape(
                "qkv_attention: k/v last dim must match head_dim".into(),
            ));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(
                "qkv_attention: num_heads must be multiple of num_kv_heads".into(),
            ));
        }
        let qd = q_st.data();
        let kd = k_st.data();
        let vd = v_st.data();
        let kv_stride = num_kv_heads * head_dim;
        let num_head_dims = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq_len * num_head_dims];

        for h in 0..num_heads {
            let kvh = (h * num_kv_heads) / num_heads;
            for t in 0..seq_len {
                let q_abs = cache_offset as usize + t;
                let window_start = if let Some(w) = window {
                    (q_abs + 1).saturating_sub(w)
                } else {
                    0
                };
                let mut scores = vec![0.0f32; kv_seq_len];
                for (t2, score) in scores.iter_mut().enumerate() {
                    if t2 > q_abs || t2 < window_start {
                        *score = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += qd[t * num_head_dims + h * head_dim + d]
                                * kd[t2 * kv_stride + kvh * head_dim + d];
                        }
                        let bias = match alibi {
                            Some(slopes) => slopes[h] * (t2 as f32 - q_abs as f32),
                            None => 0.0,
                        };
                        *score = dot * scale + bias;
                    }
                }
                // Stable softmax
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                // Weighted V sum
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for t2 in 0..kv_seq_len {
                        acc += scores[t2] * vd[t2 * kv_stride + kvh * head_dim + d];
                    }
                    out[t * num_head_dims + h * head_dim + d] = acc;
                }
            }
        }
        // WI-E3/E6 fix: the data is written flat [seq_len, num_head_dims];
        // returning it with the 3-D out_shape made downstream Linear::forward
        // see a non-2-D operand and fail with "matmul expects 2-D inputs".
        let flat_shape = Shape::new(vec![seq_len, num_head_dims]);
        Ok((
            Box::new(CpuStorage::new(out, flat_shape, DType::F32)),
            Box::new(ReadyHandle),
        ))
    }
}

impl CoreTensorOps for CpuDevice {

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        ensure_cpu_native(&dtype)?;
        let n = shape.elem_count();
        Ok(Box::new(CpuStorage::new(
            vec![0.0; n],
            shape.clone(),
            dtype,
        )))
    }


    fn matmul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a = a_storage(a)?;
        let b = b_storage(b)?;
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        if a_dims.len() != 2 || b_dims.len() != 2 {
            if std::env::var_os("GRIM_MATMUL_TRACE").is_some() {
                panic!(
                    "[matmul-trace] rank>2 operand: a={:?} b={:?}",
                    a.shape().dims(),
                    b.shape().dims()
                );
            }
            return Err(Error::Shape("matmul expects 2-D inputs".into()));
        }
        let (m, k) = (a_dims[0], a_dims[1]);
        let (k2, n) = (b_dims[0], b_dims[1]);
        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }
        if out_shape.dims() != [m, n] {
            return Err(Error::Shape(format!(
                "expected out [{m},{n}], got {out_shape:?}"
            )));
        }
        let mut out = vec![0.0f32; m * n];
        // All slices sized by shape assertions; dispatch is a safe Rust fn.
        gemm_dispatch(a.data(), b.data(), &mut out, m, n, k);
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn add(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a = a_storage(a)?;
        let b = b_storage(b)?;
        if !a.shape().broadcast_compatible(b.shape()) || !a.shape().broadcast_compatible(out_shape)
        {
            return Err(Error::Shape("add: broadcast shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let mut out = vec![0.0f32; n];
        let aa = a.data();
        let bb = b.data();
        let sa = a.shape().dims();
        let sb = b.shape().dims();
        let out_dims = out_shape.dims();
        for (i, o) in out.iter_mut().enumerate() {
            *o = aa[broadcast_index(i, sa, out_dims)] + bb[broadcast_index(i, sb, out_dims)];
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn mul(
        &self,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a = a_storage(a)?;
        let b = b_storage(b)?;
        if !a.shape().broadcast_compatible(b.shape()) || !a.shape().broadcast_compatible(out_shape)
        {
            return Err(Error::Shape("mul: broadcast shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let mut out = vec![0.0f32; n];
        let aa = a.data();
        let bb = b.data();
        let sa = a.shape().dims();
        let sb = b.shape().dims();
        let out_dims = out_shape.dims();
        for (i, o) in out.iter_mut().enumerate() {
            *o = aa[broadcast_index(i, sa, out_dims)] * bb[broadcast_index(i, sb, out_dims)];
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn silu_mul(
        &self,
        gate: &dyn BackendStorage,
        up: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let g = a_storage(gate)?;
        let u = a_storage(up)?;
        if g.shape() != u.shape() || g.shape() != out_shape {
            return Err(Error::Shape("silu_mul: shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let mut out = vec![0.0f32; n];
        for (i, o) in out.iter_mut().enumerate() {
            let x = g.data()[i];
            let silu = x / (1.0 + (-x).exp());
            *o = silu * u.data()[i];
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn rms_norm(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        eps: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x = a_storage(x)?;
        let w = a_storage(weight)?;
        // The storage may carry a different shape than the tensor view (e.g. a
        // zero-copy relabel like QK-norm's (B,S,num_heads*head_dim)->(B*S*num_heads,
        // head_dim)). Only the total element count and the 2-D output rank matter
        // for the row-wise norm computation; the storage's own dim layout is not
        // used for indexing (rows are indexed by out_shape.dims().last()).
        if x.shape().elem_count() != out_shape.elem_count() || out_shape.rank() != 2 {
            return Err(Error::Shape("rms_norm: shape mismatch".into()));
        }
        if w.shape().rank() != 1 {
            return Err(Error::Shape("rms_norm: weight must be 1-D".into()));
        }
        let dim = out_shape.dims().last().copied().unwrap_or(0);
        if w.shape().elem_count() != dim {
            return Err(Error::Shape("rms_norm: weight size mismatch".into()));
        }
        let n_rows = x.shape().elem_count() / dim;
        let xd = x.data();
        let wd = w.data();
        let mut out = vec![0.0f32; n_rows * dim];
        for r in 0..n_rows {
            let row = &xd[r * dim..(r + 1) * dim];
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for c in 0..dim {
                out[r * dim + c] = row[c] * scale * wd[c];
            }
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn softmax(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x = a_storage(x)?;
        if x.shape() != out_shape {
            return Err(Error::Shape("softmax: x/out mismatch".into()));
        }
        let dim = out_shape.dims().last().copied().unwrap_or(0);
        let n_rows = x.shape().elem_count() / dim;
        let xd = x.data();
        let mut out = vec![0.0f32; n_rows * dim];
        for r in 0..n_rows {
            let row = &xd[r * dim..(r + 1) * dim];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for c in 0..dim {
                let e = (row[c] - mx).exp();
                out[r * dim + c] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for c in 0..dim {
                out[r * dim + c] *= inv;
            }
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn embedding(
        &self,
        weight: &dyn BackendStorage,
        indices: &[u32],
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let w = a_storage(weight)?;
        if w.shape().rank() != 2 {
            return Err(Error::Shape("embedding: weight must be 2-D".into()));
        }
        let vocab = w.shape().dim(0)?;
        let dim = w.shape().dim(1)?;
        if indices.len() * dim != out_shape.elem_count() {
            return Err(Error::Shape("embedding: out size mismatch".into()));
        }
        let wd = w.data();
        let mut out = vec![0.0f32; indices.len() * dim];
        for (i, &tok) in indices.iter().enumerate() {
            let tok = tok as usize;
            if tok >= vocab {
                return Err(Error::IndexOutOfBounds(format!(
                    "token {tok} >= vocab {vocab}"
                )));
            }
            out[i * dim..(i + 1) * dim].copy_from_slice(&wd[tok * dim..(tok + 1) * dim]);
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        ensure_cpu_native(&dtype)?;
        if data.len() != shape.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: vec![shape.elem_count()],
                got: vec![data.len()],
            });
        }
        Ok(Box::new(CpuStorage::new(
            data.to_vec(),
            shape.clone(),
            dtype,
        )))
    }


    fn advise(
        &self,
        _storage: &dyn BackendStorage,
        _advice: grim_tensor::backend::MemAdvice,
    ) -> Result<()> {
        Ok(())
    }
}

impl ElementwiseOps for CpuDevice {


    fn mul_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x = a_storage(x)?;
        if x.shape() != out_shape {
            return Err(Error::Shape("mul_scalar: shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let mut out = vec![0.0f32; n];
        let xd = x.data();
        for (i, o) in out.iter_mut().enumerate() {
            *o = xd[i] * scalar;
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn add_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x = a_storage(x)?;
        if x.shape() != out_shape {
            return Err(Error::Shape("add_scalar: shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let mut out = vec![0.0f32; n];
        let xd = x.data();
        for (i, o) in out.iter_mut().enumerate() {
            *o = xd[i] + scalar;
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn sub_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x = a_storage(x)?;
        if x.shape() != out_shape {
            return Err(Error::Shape("sub_scalar: shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let mut out = vec![0.0f32; n];
        let xd = x.data();
        for (i, o) in out.iter_mut().enumerate() {
            *o = xd[i] - scalar;
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn div_scalar(
        &self,
        x: &dyn BackendStorage,
        scalar: f32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x = a_storage(x)?;
        if x.shape() != out_shape {
            return Err(Error::Shape("div_scalar: shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let mut out = vec![0.0f32; n];
        let xd = x.data();
        let inv_scalar = 1.0 / scalar;
        for (i, o) in out.iter_mut().enumerate() {
            *o = xd[i] * inv_scalar;
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn sqrt(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x = a_storage(x)?;
        if x.shape() != out_shape {
            return Err(Error::Shape("sqrt: shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let mut out = vec![0.0f32; n];
        let xd = x.data();
        for (i, o) in out.iter_mut().enumerate() {
            *o = xd[i].sqrt();
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn recip(
        &self,
        x: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x = a_storage(x)?;
        if x.shape() != out_shape {
            return Err(Error::Shape("recip: shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let mut out = vec![0.0f32; n];
        let xd = x.data();
        for (i, o) in out.iter_mut().enumerate() {
            *o = 1.0 / xd[i];
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }
}

impl SamplingOps for CpuDevice {
}

impl AttentionOps for CpuDevice {


    fn sage_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.qkv_attention(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            0,
            None,
            out_shape,
            None,
            None,
        )
    }


    fn rope(
        &self,
        x: &dyn BackendStorage,
        positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_st = a_storage(x)?;
        let dims = out_shape.dims().to_vec();
        if dims.len() != 3 || dims[2] != cfg.dim {
            return Err(Error::Shape(format!(
                "RoPE expects (B,S,D={}), got {:?}",
                cfg.dim, dims
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let rotary_dim = cfg.rotary_dim.min(d);
        let rotary_half = rotary_dim / 2;

        let inv_freq: Vec<f32> = (0..rotary_half)
            .map(|i| {
                let freq = 1.0 / cfg.base.powf((2 * i) as f32 / d as f32);
                if let Some(yarn) = &cfg.yarn {
                    let wavelength = 2.0 * std::f32::consts::PI / freq;
                    let low = (yarn.original_max_pos as f32) / yarn.beta_slow;
                    let high = (yarn.original_max_pos as f32) / yarn.beta_fast;
                    if wavelength < high {
                        freq
                    } else if wavelength > low {
                        freq / yarn.factor
                    } else {
                        let ramp = (yarn.original_max_pos as f32 / wavelength - yarn.beta_slow)
                            / (yarn.beta_fast - yarn.beta_slow);
                        (1.0 - ramp) * (freq / yarn.factor) + ramp * freq
                    }
                } else {
                    freq
                }
            })
            .collect();

        let mscale = cfg.yarn.as_ref().map_or(1.0, |y| y.attention_factor);
        let mut src = x_st.data().to_vec();

        for bi in 0..b {
            for si in 0..s {
                let pos = positions.get(si).copied().unwrap_or(si as u32) as f32;
                let base_index = (bi * s + si) * d;
                let mut cos_p = vec![0.0f32; rotary_half];
                let mut sin_p = vec![0.0f32; rotary_half];
                for i in 0..rotary_half {
                    let a = pos * inv_freq[i];
                    cos_p[i] = a.cos() * mscale;
                    sin_p[i] = a.sin() * mscale;
                }
                for i in 0..rotary_half {
                    let x1 = src[base_index + 2 * i];
                    let x2 = src[base_index + 2 * i + 1];
                    src[base_index + 2 * i] = x1 * cos_p[i] - x2 * sin_p[i];
                    src[base_index + 2 * i + 1] = x1 * sin_p[i] + x2 * cos_p[i];
                }
            }
        }
        Ok((
            Box::new(CpuStorage::new(src, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn rerope(
        &self,
        k: &dyn BackendStorage,
        old_positions: &[u32],
        new_positions: &[u32],
        cfg: &grim_tensor::RopeConfig,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let k_st = a_storage(k)?;
        let dims = out_shape.dims().to_vec();
        if dims.len() != 3 || dims[2] != cfg.dim {
            return Err(Error::Shape(format!(
                "Re-RoPE expects (B,S,D={}), got {:?}",
                cfg.dim, dims
            )));
        }
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let rotary_dim = cfg.rotary_dim.min(d);
        let rotary_half = rotary_dim / 2;

        let inv_freq: Vec<f32> = (0..rotary_half)
            .map(|i| {
                let freq = 1.0 / cfg.base.powf((2 * i) as f32 / d as f32);
                if let Some(yarn) = &cfg.yarn {
                    let wavelength = 2.0 * std::f32::consts::PI / freq;
                    let low = (yarn.original_max_pos as f32) / yarn.beta_slow;
                    let high = (yarn.original_max_pos as f32) / yarn.beta_fast;
                    if wavelength < high {
                        freq
                    } else if wavelength > low {
                        freq / yarn.factor
                    } else {
                        let ramp = (yarn.original_max_pos as f32 / wavelength - yarn.beta_slow)
                            / (yarn.beta_fast - yarn.beta_slow);
                        (1.0 - ramp) * (freq / yarn.factor) + ramp * freq
                    }
                } else {
                    freq
                }
            })
            .collect();

        let mscale = cfg.yarn.as_ref().map_or(1.0, |y| y.attention_factor);
        let mut src = k_st.data().to_vec();

        for bi in 0..b {
            for si in 0..s {
                let p_old = old_positions.get(si).copied().unwrap_or(si as u32) as f32;
                let p_new = new_positions.get(si).copied().unwrap_or(si as u32) as f32;
                let base_index = (bi * s + si) * d;
                let mut old_cos = vec![0.0f32; rotary_half];
                let mut old_sin = vec![0.0f32; rotary_half];
                let mut new_cos = vec![0.0f32; rotary_half];
                let mut new_sin = vec![0.0f32; rotary_half];

                for i in 0..rotary_half {
                    let a_old = p_old * inv_freq[i];
                    old_cos[i] = a_old.cos() * mscale;
                    old_sin[i] = a_old.sin() * mscale;

                    let a_new = p_new * inv_freq[i];
                    new_cos[i] = a_new.cos() * mscale;
                    new_sin[i] = a_new.sin() * mscale;
                }

                for i in 0..rotary_half {
                    let k1_rot = src[base_index + 2 * i];
                    let k2_rot = src[base_index + 2 * i + 1];

                    // 1. Un-rotate using old cos/sin (inverse rotation matrix)
                    // Since forward is [cos, -sin; sin, cos], inverse is [cos, sin; -sin, cos]
                    let k1_orig = k1_rot * old_cos[i] + k2_rot * old_sin[i];
                    let k2_orig = -k1_rot * old_sin[i] + k2_rot * old_cos[i];

                    // 2. Re-rotate using new cos/sin
                    src[base_index + 2 * i] = k1_orig * new_cos[i] - k2_orig * new_sin[i];
                    src[base_index + 2 * i + 1] = k1_orig * new_sin[i] + k2_orig * new_cos[i];
                }
            }
        }
        Ok((
            Box::new(CpuStorage::new(src, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn qkv_attention(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
        _out_max: Option<&dyn BackendStorage>,
        _out_sum: Option<&dyn BackendStorage>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        self.qkv_attention_inner(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            window,
            None,
            out_shape,
        )
    }


    fn qkv_attention_alibi(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        alibi_slopes: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let slopes_st = a_storage(alibi_slopes)?;
        let num_heads = out_shape.dims()[1];
        if slopes_st.data().len() < num_heads {
            return Err(Error::Shape(
                "qkv_attention_alibi: alibi_slopes must have num_heads entries".into(),
            ));
        }
        self.qkv_attention_inner(
            q,
            k,
            v,
            num_kv_heads,
            kv_seq_len,
            cache_offset,
            window,
            Some(slopes_st.data()),
            out_shape,
        )
    }


    /// CPU **reference** implementation of fused dequantized KV-attention.
    ///
    /// The trait default is `Unimplemented` (only ROCm wires the real HIP kernel),
    /// but the CPU backend serves as the deterministic reference for GPU parity
    /// testing (see cpu-catch-up.md T-ref-1). This dequantizes the packed K/V
    /// caches on the fly (4/8-bit packed per the `grim_kv_dequant_attention`
    /// layout) and runs a straightforward reference attention that mirrors the
    /// math in `qkv_attention` (scale `1/sqrt(head_dim)`, causal mask relative
    /// to `cache_offset`, GQA via `kvh = h*num_kv_heads/num_heads`).
    fn kv_dequant_attention(
        &self,
        q: &dyn BackendStorage,
        k_tensor: &dyn BackendStorage,
        k_scales: &dyn BackendStorage,
        v_tensor: &dyn BackendStorage,
        v_scales: &dyn BackendStorage,
        num_kv_heads: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        quant_bits: u32,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_st = a_storage(q)?;
        let k_st = a_storage(k_tensor)?;
        let v_st = a_storage(v_tensor)?;
        let k_sc_st = a_storage(k_scales)?;
        let v_sc_st = a_storage(v_scales)?;

        let q_dims = q_st.shape().dims();
        let out_dims = out_shape.dims();
        if out_dims.len() != 3 || q_dims.len() != 3 {
            return Err(Error::Shape(
                "kv_dequant_attention: q and out must be 3-D".into(),
            ));
        }
        let seq_len = out_dims[0];
        let num_heads = out_dims[1];
        let head_dim = out_dims[2];
        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(
                "kv_dequant_attention: num_heads must be multiple of num_kv_heads".into(),
            ));
        }

        let k_deq = dequant_packed_kv(
            k_st,
            k_sc_st,
            num_kv_heads,
            kv_seq_len,
            head_dim,
            quant_bits,
        )?;
        let v_deq = dequant_packed_kv(
            v_st,
            v_sc_st,
            num_kv_heads,
            kv_seq_len,
            head_dim,
            quant_bits,
        )?;

        let qd = q_st.data();
        let kd = k_deq.as_slice();
        let vd = v_deq.as_slice();
        let kv_stride = num_kv_heads * head_dim;
        let num_head_dims = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq_len * num_head_dims];

        for h in 0..num_heads {
            let kvh = (h * num_kv_heads) / num_heads;
            for t in 0..seq_len {
                let q_abs = cache_offset as usize + t;
                let mut scores = vec![0.0f32; kv_seq_len];
                for t2 in 0..kv_seq_len {
                    if t2 > q_abs {
                        scores[t2] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += qd[t * num_head_dims + h * head_dim + d]
                                * kd[t2 * kv_stride + kvh * head_dim + d];
                        }
                        scores[t2] = dot * scale;
                    }
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for t2 in 0..kv_seq_len {
                        acc += scores[t2] * vd[t2 * kv_stride + kvh * head_dim + d];
                    }
                    out[t * num_head_dims + h * head_dim + d] = acc;
                }
            }
        }

        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn mla_q_kv_norm_split(
        &self,
        q_raw: &dyn BackendStorage,
        kv_raw: &dyn BackendStorage,
        q_norm_w: &dyn BackendStorage,
        kv_norm_w: &dyn BackendStorage,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        _v_dim: usize,
        eps: f32,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let q_s = a_storage(q_raw)?;
        let kv_s = a_storage(kv_raw)?;
        let qw_s = a_storage(q_norm_w)?;
        let kvw_s = a_storage(kv_norm_w)?;

        let q_data = q_s.data();
        let kv_data = kv_s.data();
        let qw_data = qw_s.data();
        let kvw_data = kvw_s.data();

        let q_dim = q_data.len();
        let kv_dim = kv_data.len();

        let q_mean_sq = q_data.iter().map(|&x| x * x).sum::<f32>() / (q_dim.max(1) as f32);
        let q_scale = 1.0 / (q_mean_sq + eps).sqrt();
        let mut q_norm = vec![0.0f32; q_dim];
        for i in 0..q_dim {
            q_norm[i] = q_data[i] * q_scale * qw_data[i];
        }

        let kv_mean_sq = kv_data.iter().map(|&x| x * x).sum::<f32>() / (kv_dim.max(1) as f32);
        let kv_scale = 1.0 / (kv_mean_sq + eps).sqrt();
        let mut kv_norm = vec![0.0f32; kv_dim];
        for i in 0..kv_dim {
            kv_norm[i] = kv_data[i] * kv_scale * kvw_data[i];
        }

        let q_nope = q_norm[..qk_nope_dim].to_vec();
        let q_rope = q_norm[qk_nope_dim..(qk_nope_dim + qk_rope_dim).min(q_dim)].to_vec();

        let kv_nope = kv_norm[..qk_nope_dim].to_vec();
        let kv_rope = kv_norm[qk_nope_dim..(qk_nope_dim + qk_rope_dim).min(kv_dim)].to_vec();

        Ok((
            Box::new(CpuStorage::new(
                q_nope,
                Shape::new(vec![qk_nope_dim]),
                DType::F32,
            )),
            Box::new(CpuStorage::new(
                q_rope,
                Shape::new(vec![qk_rope_dim]),
                DType::F32,
            )),
            Box::new(CpuStorage::new(
                kv_nope,
                Shape::new(vec![qk_nope_dim]),
                DType::F32,
            )),
            Box::new(CpuStorage::new(
                kv_rope,
                Shape::new(vec![qk_rope_dim]),
                DType::F32,
            )),
            Box::new(ReadyHandle),
        ))
    }


    fn qkv_attention_paged(
        &self,
        q: &dyn BackendStorage,
        block_tables: &dyn BackendStorage,
        k_pages: &dyn BackendStorage,
        v_pages: &dyn BackendStorage,
        num_kv_heads: usize,
        max_blocks: usize,
        page_size: usize,
        kv_seq_len: usize,
        cache_offset: u32,
        window: Option<usize>,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_st = a_storage(q)?;
        let bt_st = a_storage(block_tables)?;
        let k_st = a_storage(k_pages)?;
        let v_st = a_storage(v_pages)?;

        let q_dims = q_st.shape().dims();
        if q_dims.len() != 3 {
            return Err(Error::Shape("qkv_attention_paged: q must be 3-D".into()));
        }
        let seq_len = q_dims[0];
        let num_heads = q_dims[1];
        let head_dim = q_dims[2];

        if num_heads % num_kv_heads != 0 {
            return Err(Error::Shape(
                "qkv_attention_paged: num_heads must be multiple of num_kv_heads".into(),
            ));
        }

        let qd = q_st.data();
        let btd = bt_st.data();
        let kd = k_st.data();
        let vd = v_st.data();

        let kv_stride = num_kv_heads * head_dim;
        let num_head_dims = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq_len * num_head_dims];

        for h in 0..num_heads {
            let kvh = (h * num_kv_heads) / num_heads;
            for t in 0..seq_len {
                let q_abs = cache_offset as usize + t;
                let window_start = window
                    .map(|w| q_abs.saturating_sub(w.saturating_sub(1)))
                    .unwrap_or(0);
                let mut scores = vec![0.0f32; kv_seq_len];
                for (t2, score) in scores.iter_mut().enumerate() {
                    if t2 > q_abs || t2 < window_start {
                        *score = f32::NEG_INFINITY;
                    } else {
                        let block_idx_in_seq = t2 / page_size;
                        let offset_in_block = t2 % page_size;
                        let block_id = if block_idx_in_seq < max_blocks {
                            btd[block_idx_in_seq] as usize
                        } else {
                            block_idx_in_seq
                        };

                        let k_offset =
                            (block_id * page_size + offset_in_block) * kv_stride + kvh * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += qd[t * num_head_dims + h * head_dim + d] * kd[k_offset + d];
                        }
                        *score = dot * scale;
                    }
                }
                if std::env::var("GRIM_DBG_ATTN").is_ok() && t == 2 && h == 0 {
                    eprintln!(
                        "[paged_dbg] t=2 h=0 scores={scores:?} q_abs={q_abs} kv_stride={kv_stride} nh={num_heads} hd={head_dim} bt={:?} ps={page_size}",
                        btd
                    );
                    let qslice: Vec<f32> = (0..head_dim)
                        .map(|d| qd[t * num_head_dims + h * head_dim + d])
                        .collect();
                    let k0: Vec<f32> = (0..head_dim).map(|d| kd[kvh * head_dim + d]).collect();
                    let k1: Vec<f32> = (0..head_dim)
                        .map(|d| kd[kv_stride + kvh * head_dim + d])
                        .collect();
                    eprintln!("[paged_dbg] q={qslice:?} k0={k0:?} k1={k1:?}");
                }

                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }

                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (t2, &score) in scores.iter().enumerate() {
                        let block_idx_in_seq = t2 / page_size;
                        let offset_in_block = t2 % page_size;
                        let block_id = if block_idx_in_seq < max_blocks {
                            btd[block_idx_in_seq] as usize
                        } else {
                            block_idx_in_seq
                        };
                        let v_offset =
                            (block_id * page_size + offset_in_block) * kv_stride + kvh * head_dim;
                        acc += score * vd[v_offset + d];
                    }
                    out[t * num_head_dims + h * head_dim + d] = acc;
                }
            }
        }

        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }
}

impl FusionOps for CpuDevice {


    fn silu_mul_quantize(
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
        let g = a_storage(gate)?;
        let u = a_storage(up)?;
        if g.shape() != u.shape() || g.shape() != out_shape {
            return Err(Error::Shape("silu_mul_quantize: shape mismatch".into()));
        }
        let n = out_shape.elem_count();
        let gd = g.data();
        let ud = u.data();

        let mut max_abs = 0.0f32;
        let mut activated = vec![0.0f32; n];
        for i in 0..n {
            let x = gd[i];
            let silu = x / (1.0 + (-x).exp());
            let val = silu * ud[i];
            activated[i] = val;
            max_abs = max_abs.max(val.abs());
        }

        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        let mut qbytes = vec![0u8; n];
        for i in 0..n {
            let q = (activated[i] * inv_scale).clamp(-127.0, 127.0);
            qbytes[i] = (q as i8) as u8;
        }

        let bytes_storage = CpuStorage::from_raw_bytes(
            qbytes,
            out_shape.clone(),
            DType {
                arith: grim_tensor::dtype::ArithType::U8,
                storage: grim_tensor::dtype::Storage::Native,
            },
        );
        let scale_storage = CpuStorage::new(
            vec![scale],
            Shape::from_slice(&[1]),
            DType {
                arith: grim_tensor::dtype::ArithType::F32,
                storage: grim_tensor::dtype::Storage::Native,
            },
        );

        Ok((
            Box::new(bytes_storage),
            Box::new(scale_storage),
            Box::new(ReadyHandle),
        ))
    }


    fn fused_add_rms_norm(
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
        let x_st = a_storage(x)?;
        let res_st = a_storage(residual)?;
        let w_st = a_storage(weight)?;
        let x_data = x_st.data();
        let res_data = res_st.data();
        let w_data = w_st.data();

        let n = out_shape.elem_count();
        let hidden_dim = w_st.shape().elem_count();
        if hidden_dim == 0 || n % hidden_dim != 0 {
            return Err(Error::Shape(
                "fused_add_rms_norm: invalid dimensions".into(),
            ));
        }
        let num_rows = n / hidden_dim;

        let mut res_out = vec![0.0f32; n];
        let mut y_out = vec![0.0f32; n];

        for r in 0..num_rows {
            let row_start = r * hidden_dim;
            let mut sum_sq = 0.0f32;
            for ((x, res), added_out) in x_data[row_start..row_start + hidden_dim]
                .iter()
                .zip(&res_data[row_start..row_start + hidden_dim])
                .zip(&mut res_out[row_start..row_start + hidden_dim])
            {
                let added = x + res;
                *added_out = added;
                sum_sq += added * added;
            }
            let mean_sq = sum_sq / (hidden_dim as f32);
            let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
            for ((y, added), w) in y_out[row_start..row_start + hidden_dim]
                .iter_mut()
                .zip(&res_out[row_start..row_start + hidden_dim])
                .zip(w_data)
            {
                *y = added * inv_rms * w;
            }
        }

        Ok((
            Box::new(CpuStorage::new(y_out, out_shape.clone(), DType::F32)),
            Box::new(CpuStorage::new(res_out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }
}

impl AutogradOps for CpuDevice {


    fn silu_mul_backward(
        &self,
        e: &dyn BackendStorage,
        g: &dyn BackendStorage,
        dw: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        let e_st = a_storage(e)?;
        let g_st = a_storage(g)?;
        let dw_st = a_storage(dw)?;
        let ed = e_st.data();
        let gd = g_st.data();
        let dwd = dw_st.data();
        let n = out_shape.elem_count();

        let mut de = vec![0.0f32; n];
        let mut dg = vec![0.0f32; n];

        for i in 0..n {
            let x = gd[i];
            let sigm = 1.0f32 / (1.0f32 + (-x).exp());
            let silu = x * sigm;
            let dsilu = sigm * (1.0f32 + x * (1.0f32 - sigm));
            let w = dwd[i];
            let ev = ed[i];

            de[i] = w * silu;
            dg[i] = w * ev * dsilu;
        }

        Ok((
            Box::new(CpuStorage::new(de, out_shape.clone(), DType::F32)),
            Box::new(CpuStorage::new(dg, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }
}

impl OptimizerOps for CpuDevice {
}

impl QuantOps for CpuDevice {


    fn quantized_matmul(
        &self,
        a: &dyn BackendStorage,
        b_packed: &dyn BackendStorage,
        b_scales: &[f32],
        format: grim_tensor::QuantFormat,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let a_storage = a_storage(a)?;
        let a_data = a_storage.data();
        let a_dims = a.shape().dims();
        let out_dims = out_shape.dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = out_dims[1];

        // Borrow the packed bytes instead of cloning them — the output head is
        // ~240MB packed and a per-token clone of that wrecks the allocator.
        let b_bytes_owned: Vec<u8>;
        let b_bytes: &[u8] = if let Some(cs) = b_packed.as_any().downcast_ref::<CpuStorage>() {
            if let Some(rb) = &cs.raw_bytes {
                rb.as_slice()
            } else {
                b_bytes_owned = cs.data().iter().map(|&f| f as u8).collect();
                &b_bytes_owned
            }
        } else {
            b_bytes_owned = vec![0u8; k * n];
            &b_bytes_owned
        };

        // WI-E7 fast path: Q8_0 GEMM directly on the packed bytes. Skips both
        // the [N,K] f32 dequant materialization (~240 MB for an output head)
        // and the transpose copy. Requires k % 32 == 0 (whole blocks).
        if matches!(format, grim_tensor::QuantFormat::Q8_0) && k % 32 == 0 {
            let c = grim_quant::gemm_q8_0_packed(a_data, b_bytes, m, n, k)?;
            return Ok((
                Box::new(CpuStorage::new(c, out_shape.clone(), DType::F32)),
                Box::new(ReadyHandle),
            ));
        }

        // WI-E7 fast path: Q4_K GEMM directly on the packed bytes, same shape
        // guard pattern as Q8_0 above. A q4_K super-block covers 256 weights,
        // so whole-block GEMM requires k % 256 == 0.
        if matches!(format, grim_tensor::QuantFormat::Q4K) && k % 256 == 0 {
            let c = grim_quant::gemm_q4k_packed(a_data, b_bytes, m, n, k)?;
            return Ok((
                Box::new(CpuStorage::new(c, out_shape.clone(), DType::F32)),
                Box::new(ReadyHandle),
            ));
        }

        let b_dequant_transposed: Vec<f32> =
            match &b_packed.dtype().storage {
                grim_tensor::dtype::Storage::ResidualPacked(cfg) => {
                    let mut out = vec![0.0f32; n * k];
                    let prov = b_packed.provenance();
                    let outliers: Vec<(u32, f32)> = match &prov {
                        QuantProvenance::WithResiduals {
                            outlier_indices,
                            outlier_values_bits,
                            ..
                        } => outlier_indices
                            .iter()
                            .zip(outlier_values_bits.iter())
                            .map(|(&idx, &bits)| (idx, f32::from_bits(bits)))
                            .collect(),
                        _ => Vec::new(),
                    };
                    let scales_u8: Vec<u8> = b_scales
                        .iter()
                        .map(|&s| (s * 255.0).clamp(0.0, 255.0) as u8)
                        .collect();
                    for col in 0..n {
                        let row_vals = crate::dequant_gemm::dequant_row(
                            col, k, b_bytes, &scales_u8, cfg.bpw, None, &outliers,
                        );
                        out[col * k..(col + 1) * k].copy_from_slice(&row_vals[..k]);
                    }
                    out
                }
                _ => match format {
                    // GGUF Q8_0 weights are resident as the native 34-byte block
                    // stream (2-byte f16 scale + 32 int8 quants per block), and
                    // `Linear::forward` passes an empty `b_scales` (scales live in
                    // the block headers). Decoding with the canonical
                    // `dequant_q80` — the hand-rolled loop below read stride-32
                    // with a 1.0 scale fallback, treating the f16 headers as
                    // quants and corrupting every attention projection.
                    grim_tensor::QuantFormat::Q8_0 => grim_quant::dequant_q80(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul Q8_0 dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Q4K => grim_quant::dequant_q4k(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul Q4K dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Q5K => grim_quant::dequant_q5k(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul Q5K dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Q6K => grim_quant::dequant_q6k(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul Q6K dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Iq4Nl => grim_quant::dequant_iq4nl(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul IQ4NL dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Iq4Xs => grim_quant::dequant_iq4xs(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul IQ4XS dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Iq3Xxs => grim_quant::dequant_iq3xxs(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul IQ3XXS dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Iq3S => grim_quant::dequant_iq3s(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul IQ3S dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Iq2Xxs => grim_quant::dequant_iq2xxs(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul IQ2XXS dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Iq2Xs => grim_quant::dequant_iq2xs(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul IQ2XS dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Iq2S => grim_quant::dequant_iq2s(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul IQ2S dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Fp4 => grim_quant::dequant_mxfp4(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul MXFP4 dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Fp4Block16 => {
                        grim_quant::dequant_fp4_block16(b_bytes, k * n).map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul FP4Block16 dequant: {e}"))
                        })?
                    }
                    grim_tensor::QuantFormat::Fp8 => grim_quant::dequant_fp8(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul FP8 dequant: {e}"))
                        })?,
                    grim_tensor::QuantFormat::Fp8Block16 => {
                        grim_quant::dequant_fp8_block16(b_bytes, k * n).map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul FP8Block16 dequant: {e}"))
                        })?
                    }
                    grim_tensor::QuantFormat::Nf4 => grim_quant::dequant_nf4(b_bytes, k * n)
                        .map_err(|e| {
                            Error::Backend(format!("CPU quantized_matmul NF4 dequant: {e}"))
                        })?,
                },
            };

        // Convert transposed [N, K] dequantized weights to row-major [K, N]
        let mut b_rm = vec![0.0f32; k * n];
        for col in 0..n {
            for p in 0..k {
                b_rm[p * n + col] = b_dequant_transposed[col * k + p];
            }
        }

        let mut c_vec = vec![0.0f32; m * n];
        gemm_dispatch(a_data, &b_rm, &mut c_vec, m, n, k);

        Ok((
            Box::new(CpuStorage::new(c_vec, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


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
        _residuals: Option<&grim_tensor::backend::QuantizedMatmulBackwardResiduals>,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let dy_st = a_storage(dy)?;
        let dy_data = dy_st.data();

        let b_bytes: Vec<u8> = if let Some(cs) = b_packed.as_any().downcast_ref::<CpuStorage>() {
            if let Some(rb) = &cs.raw_bytes {
                (**rb).clone()
            } else {
                cs.data().iter().map(|&f| f as u8).collect()
            }
        } else {
            vec![0u8; k * n]
        };

        let prov = b_packed.provenance();
        let outliers: Vec<(u32, f32)> = match &prov {
            QuantProvenance::WithResiduals {
                outlier_indices,
                outlier_values_bits,
                ..
            } => outlier_indices
                .iter()
                .zip(outlier_values_bits.iter())
                .map(|(&idx, &bits)| (idx, f32::from_bits(bits)))
                .collect(),
            _ => Vec::new(),
        };

        // GGUF-native Q8_0 resident bytes (34-byte blocks, scales in headers):
        // decode with the canonical dequant_q80, matching the forward path.
        // Other storage dtypes keep the legacy dequant_row path (ResidualPacked
        // training / synthetic buffers with external scales).
        let mut dx_vec = vec![0.0f32; m * k];
        if matches!(
            b_packed.dtype().storage,
            grim_tensor::dtype::Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80)
        ) {
            let b_dequant = grim_quant::dequant_q80(&b_bytes, k * n).map_err(|e| {
                Error::Backend(format!(
                    "CPU quantized_matmul_backward_dx Q8_0 dequant: {e}"
                ))
            })?;
            gemm_dispatch(dy_data, &b_dequant, &mut dx_vec, m, k, n);
            return Ok((
                Box::new(CpuStorage::new(dx_vec, out_shape.clone(), DType::F32)),
                Box::new(ReadyHandle),
            ));
        }

        let scales_u8: Vec<u8> = b_scales
            .iter()
            .map(|&s| (s * 255.0).clamp(0.0, 255.0) as u8)
            .collect();

        let mut b_dequant = vec![0.0f32; n * k];
        for col in 0..n {
            let row_vals = crate::dequant_gemm::dequant_row(
                col,
                k,
                &b_bytes,
                &scales_u8,
                default_bpw,
                None,
                &outliers,
            );
            let start = col * k;
            b_dequant[start..start + k].copy_from_slice(&row_vals[..k]);
        }
        let mut dx_vec = vec![0.0f32; m * k];
        gemm_dispatch(dy_data, &b_dequant, &mut dx_vec, m, k, n);

        Ok((
            Box::new(CpuStorage::new(dx_vec, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }
}

impl RecurrentOps for CpuDevice {


    fn short_conv1d_causal_step(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        bias: Option<&dyn BackendStorage>,
        conv_state: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = a_storage(x)?;
        let w_s = a_storage(weight)?;
        let st_s = a_storage(conv_state)?;

        let x_data = x_s.data();
        let w_data = w_s.data();
        let st_data = st_s.data();

        let hidden = x_s.shape().dims().last().cloned().unwrap_or(0);
        let k_size = if hidden > 0 {
            w_s.data().len().checked_div(hidden).unwrap_or(0)
        } else {
            1
        };

        let mut out = vec![0.0f32; out_shape.elem_count()];
        for h in 0..hidden {
            let mut sum = 0.0f32;
            for k in 0..k_size.saturating_sub(1) {
                sum += st_data[h * (k_size - 1) + k] * w_data[h * k_size + k];
            }
            sum += x_data[h] * w_data[h * k_size + (k_size - 1)];
            if let Some(b) = bias {
                let b_s = a_storage(b)?;
                sum += b_s.data()[h];
            }
            out[h] = sum;
        }

        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }


    fn kda_gated_delta_rule_step(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        beta: &dyn BackendStorage,
        a_gate: &dyn BackendStorage,
        recurrent_state: &dyn BackendStorage,
        d_k: usize,
        d_v: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = a_storage(q)?;
        let k_s = a_storage(k)?;
        let v_s = a_storage(v)?;
        let beta_s = a_storage(beta)?;
        let gate_s = a_storage(a_gate)?;
        let s_s = a_storage(recurrent_state)?;

        let q_data = q_s.data();
        let k_data = k_s.data();
        let v_data = v_s.data();
        let beta_val = beta_s.data()[0];
        let gate_val = gate_s.data()[0];

        let decay = gate_val.exp();
        let mut out = vec![0.0f32; out_shape.elem_count()];

        for i in 0..d_v {
            let col: Vec<f32> = s_s.data()[i..]
                .iter()
                .step_by(d_v)
                .take(d_k)
                .copied()
                .collect();
            let mut k_s_decayed = 0.0f32;
            for (k, sc) in k_data.iter().zip(&col) {
                k_s_decayed += k * (decay * sc);
            }
            let delta_i = beta_val * (v_data[i] - k_s_decayed);
            let mut out_i = 0.0f32;
            for ((q, k), sc) in q_data.iter().zip(k_data.iter()).zip(&col) {
                let s_new_ji = decay * sc + k * delta_i;
                out_i += q * s_new_ji;
            }
            out[i] = out_i;
        }

        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }
}

impl CollectiveOps for CpuDevice {
}

impl MemoryOps for CpuDevice {


    fn from_cpu_bytes(
        &self,
        data: &[u8],
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn BackendStorage>> {
        // Native f32: interpret bytes as f32.
        match dtype.storage {
            grim_tensor::dtype::Storage::Native => {
                if data.len() % 4 != 0 {
                    return Err(Error::ShapeMismatch {
                        expected: vec![shape.elem_count() * 4],
                        got: vec![data.len()],
                    });
                }
                let f32_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(Box::new(CpuStorage::new(f32_data, shape.clone(), dtype)))
            }
            _ => Ok(Box::new(CpuStorage::from_raw_bytes(
                data.to_vec(),
                shape.clone(),
                dtype,
            ))),
        }
    }


    fn alloc_storage(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        ensure_cpu_native(&dtype)?;
        let n = shape.elem_count();
        Ok(Box::new(CpuStorage::new(
            vec![0.0f32; n],
            shape.clone(),
            dtype,
        )))
    }


    fn copy_slice_into(
        &self,
        dst: &dyn BackendStorage,
        src: &dyn BackendStorage,
        dst_elem_offset: usize,
        count: usize,
    ) -> Result<()> {
        let dst_st = a_storage(dst)?;
        let src_st = a_storage(src)?;
        let src_data = src_st.data();
        if src_data.len() < count {
            return Err(Error::ShapeMismatch {
                expected: vec![count],
                got: vec![src_data.len()],
            });
        }
        let dst_len = dst_st.data.len();
        if dst_elem_offset + count > dst_len {
            return Err(Error::IndexOutOfBounds(format!(
                "copy_slice_into: offset {} + count {} > dst len {}",
                dst_elem_offset, count, dst_len
            )));
        }
        unsafe {
            let dst_ptr = dst_st.data.as_ptr() as *mut f32;
            let dst_slice = std::slice::from_raw_parts_mut(dst_ptr.add(dst_elem_offset), count);
            dst_slice.copy_from_slice(&src_data[..count]);
        }
        Ok(())
    }
}

impl GraphCaptureOps for CpuDevice {


    // Bridge the inherent CPU graph-capture implementation into the
    // `BackendDevice` trait contract so generic `&dyn BackendDevice` callers
    // (model runners, the same path GPU backends use) see the real CPU capture
    // instead of the trait default `Err(Unimplemented)`.
    fn begin_graph_capture(&self, key: &str) -> Result<()> {
        CpuDevice::begin_graph_capture(self, key)
    }


    fn end_graph_capture(&self, key: &str) -> Result<()> {
        CpuDevice::end_graph_capture(self, key)
    }


    fn replay_graph(&self, key: &str) -> Result<bool> {
        CpuDevice::replay_graph(self, key)
    }


    fn has_captured_graph(&self, key: &str) -> bool {
        self.graphs.has_captured(key)
    }
}

impl grim_tensor::BackendDevice for CpuDevice {}


impl BackendStorage for CpuStorage {
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }
    fn provenance(&self) -> QuantProvenance {
        self.provenance.clone()
    }
    fn shape(&self) -> &Shape {
        &self.shape
    }
    fn quant_scales(&self) -> Option<&[f32]> {
        self.quant_scales.as_deref()
    }
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        if !self.data.is_empty() || self.dtype.storage == Storage::Native {
            return Ok((*self.data).clone());
        }
        let raw = self.raw_bytes.as_deref().ok_or_else(|| {
            Error::Backend("to_cpu_vec_f32: quantized storage with no raw_bytes".into())
        })?;
        // Quantized storage with raw_bytes: dequantize to f32 (mirrors the ROCm
        // `to_cpu_vec_f32` host-dequant fallback) rather than casting bytes to f32,
        // which would produce one f32 per byte and blow past the element count.
        let n = self.shape.elem_count();
        match &self.dtype.storage {
            Storage::Native => Ok((*self.data).clone()),
            Storage::KQuant(scheme) => match scheme {
                grim_tensor::dtype::KQuantScheme::Q2K => grim_quant::dequant_q2k(raw, n),
                grim_tensor::dtype::KQuantScheme::Q3K => grim_quant::dequant_q3k(raw, n),
                grim_tensor::dtype::KQuantScheme::Q4K => grim_quant::dequant_q4k(raw, n),
                grim_tensor::dtype::KQuantScheme::Q5K => grim_quant::dequant_q5k(raw, n),
                grim_tensor::dtype::KQuantScheme::Q6K => grim_quant::dequant_q6k(raw, n),
                grim_tensor::dtype::KQuantScheme::Q80 => grim_quant::dequant_q80(raw, n),
                grim_tensor::dtype::KQuantScheme::IQ4NL => grim_quant::dequant_iq4nl(raw, n),
                grim_tensor::dtype::KQuantScheme::IQ4XS => grim_quant::dequant_iq4xs(raw, n),
                grim_tensor::dtype::KQuantScheme::IQ3XXS => grim_quant::dequant_iq3xxs(raw, n),
                grim_tensor::dtype::KQuantScheme::IQ3S => grim_quant::dequant_iq3s(raw, n),
                grim_tensor::dtype::KQuantScheme::IQ2XXS => grim_quant::dequant_iq2xxs(raw, n),
                grim_tensor::dtype::KQuantScheme::IQ2XS => grim_quant::dequant_iq2xs(raw, n),
                grim_tensor::dtype::KQuantScheme::IQ2S => grim_quant::dequant_iq2s(raw, n),
            },
            Storage::FloatPack(fp) => match fp {
                grim_tensor::dtype::FloatPackScheme::Fp4 => grim_quant::dequant_fp4(raw, n),
                grim_tensor::dtype::FloatPackScheme::Nf4 => grim_quant::dequant_nf4(raw, n),
                grim_tensor::dtype::FloatPackScheme::Fp8 => grim_quant::dequant_fp8(raw, n),
                grim_tensor::dtype::FloatPackScheme::MxFp4 => grim_quant::dequant_mxfp4(raw, n),
                grim_tensor::dtype::FloatPackScheme::MxFp8 => grim_quant::dequant_mxfp8(raw, n),
            },
            Storage::Block(block_type) => match block_type {
                grim_tensor::dtype::BlockDtype::Fp4 => grim_quant::dequant_fp4(raw, n),
                grim_tensor::dtype::BlockDtype::Nf4 => grim_quant::dequant_nf4(raw, n),
                grim_tensor::dtype::BlockDtype::Fp8 => grim_quant::dequant_fp8(raw, n),
                grim_tensor::dtype::BlockDtype::Fp4Block16 => {
                    grim_quant::dequant_fp4_block16(raw, n)
                }
                grim_tensor::dtype::BlockDtype::Fp8Block16 => {
                    grim_quant::dequant_fp8_block16(raw, n)
                }
            },
            Storage::ResidualPacked(_) => Err(Error::Unimplemented(
                "to_cpu_vec_f32: ResidualPacked dequant not supported".into(),
            )),
            Storage::GroupInt(_) => Err(Error::Unimplemented(
                "to_cpu_vec_f32: GroupInt dequant not supported (load as native instead)".into(),
            )),
            other => Err(Error::Unimplemented(format!(
                "to_cpu_vec_f32: storage variant {:?} not supported on CPU",
                other
            ))),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------- helpers ----------

fn a_storage(s: &dyn BackendStorage) -> Result<&CpuStorage> {
    s.as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Backend("storage is not CpuStorage".into()))
}

fn b_storage(s: &dyn BackendStorage) -> Result<&CpuStorage> {
    a_storage(s)
}

fn ensure_cpu_native(dtype: &DType) -> Result<()> {
    match dtype.storage {
        Storage::Native => Ok(()),
        _ => Err(Error::Unimplemented(
            "CPU backend v1 is F32/Native only".into(),
        )),
    }
}

// ---------- GEMM dispatch (§4.1 OxiBLAS) ----------

/// Row-major `(M,K) @ (K,N) → (M,N)`. Selection: GEMV fast path, `oxiblas` SIMD, scalar fallback.
pub(crate) fn gemm_dispatch(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    // Fast path: M=1 (single-token decode), dot-product per column.
    if m == 1 {
        gemv_row(a, b, out, n, k);
        return;
    }

    // SIMD path: matrixmultiply::sgemm.
    #[cfg(feature = "oxiblas")]
    {
        oxiblas_sgemm(a, b, out, m, n, k);
    }

    // Scalar fallback (compiled when `oxiblas` is disabled).
    #[cfg(not(feature = "oxiblas"))]
    gemm_scalar(a, b, out, m, n, k);
}

/// GEMV fast path for M=1: `y = A[0] · B`. Walks K rows of B sequentially.
fn gemv_row(a: &[f32], b: &[f32], out: &mut [f32], n: usize, k: usize) {
    // Zero out pre-allocated buffer.
    for o in out[..n].iter_mut() {
        *o = 0.0;
    }
    // Accumulate: for each k, scatter a[k] * B[k,*] into out.
    for p in 0..k {
        let ap = a[p];
        let b_row = &b[p * n..(p + 1) * n];
        for (oj, &bv) in out[..n].iter_mut().zip(b_row.iter()) {
            *oj += ap * bv;
        }
    }
}

/// `matrixmultiply::sgemm` — pure-Rust SIMD BLAS, no C/Fortran toolchain. Unsafe.
#[cfg(feature = "oxiblas")]
fn oxiblas_sgemm(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    // SAFETY: Rust-slice pointers (lifetime/alignment), sizes validated by caller,
    // alpha=1.0 beta=0.0 overwriting a fresh Vec<f32>.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0_f32, // alpha
            a.as_ptr(),
            k as isize,
            1, // rsa, csa
            b.as_ptr(),
            n as isize,
            1,       // rsb, csb
            0.0_f32, // beta (overwrite out)
            out.as_mut_ptr(),
            n as isize,
            1, // rsc, csc
        );
    }
}

/// Scalar triple-loop GEMM with cache-friendly loop order and blocking.
///
/// The original O(m·n·k) triple loop accessed `b` with stride-`n`, causing
/// cache-line misses for every inner iteration. By reordering to `i, p, j`
/// and adding an inner block over `j`, we walk `b` sequentially within each
/// row and keep the output row in cache across the `p` reduction.
// Dead under default features (oxiblas SIMD is the shipping path); live only
// when built with `--no-default-features`. Allowed so the scalar fallback
// compiles warning-free in that configuration.
#[allow(dead_code)]
fn gemm_scalar(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    // Zero the output buffer first.
    for o in out[..m * n].iter_mut() {
        *o = 0.0;
    }

    // Cache-friendly loop order: i (row block), p (inner dim), j (col block).
    // The `j` loop over contiguous output entries hits `b[p*n + j]` sequentially.
    const BLOCK: usize = 64;
    for i in 0..m {
        for p in 0..k {
            let ap = a[i * k + p];
            if ap == 0.0 {
                continue;
            }
            let b_row = &b[p * n..];
            let out_row = &mut out[i * n..(i + 1) * n];
            let mut jj = 0;
            while jj < n {
                let end = (jj + BLOCK).min(n);
                for j in jj..end {
                    out_row[j] += ap * b_row[j];
                }
                jj = end;
            }
        }
    }
}

fn broadcast_index(linear: usize, src_dims: &[usize], out_dims: &[usize]) -> usize {
    // out is row-major; src may have lower rank (left-pad 1s) and broadcasting dims.
    let rank = out_dims.len();
    let mut src = vec![1usize; rank];
    let src_rank = src_dims.len();
    for i in 0..src_rank {
        src[rank - src_rank + i] = src_dims[i];
    }
    let mut idx = vec![0usize; rank];
    let mut rem = linear;
    for d in (0..rank).rev() {
        let sz = out_dims[d];
        idx[d] = rem % sz;
        rem /= sz;
    }
    let mut src_linear = 0usize;
    let mut stride = 1usize;
    for d in (0..rank).rev() {
        let dim = src[d];
        let i = if dim == 1 { 0 } else { idx[d] };
        src_linear += i * stride;
        stride *= dim;
    }
    src_linear
}

/// Build a host tensor owned by `CpuDevice`.
pub fn cpu_tensor(data: Vec<f32>, shape: Shape) -> grim_tensor::Tensor {
    // Debug-mode panic catches fake-embedding bug (WI-F4-close). Release skips check.
    assert!(
        data.len() == shape.elem_count(),
        "cpu_tensor: data.len() ({}) must equal shape.elem_count() ({:?} -> {} elements)",
        data.len(),
        shape.dims(),
        shape.elem_count()
    );
    grim_tensor::Tensor::new(
        Arc::new(CpuStorage::new(data, shape.clone(), DType::F32)),
        shape,
        DType::F32,
        QuantProvenance::default(),
        Device::Cpu,
    )
}

/// Add two tensors element-wise (broadcasting allowed across rank differences).
pub fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let target_shape = a.shape().broadcast_shape(b.shape())?;
    let dev = CpuDevice::new();
    let (s, h) = grim_tensor::CoreTensorOps::add(
        &dev,
        a.storage().as_ref(),
        b.storage().as_ref(),
        &target_shape,
    )?;
    h.synchronize()?;
    Ok(Tensor::new(
        Arc::from(s),
        target_shape,
        DType::F32,
        a.provenance().clone(),
        a.device().clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: run gemm_scalar and return result.
    fn scalar(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        gemm_scalar(a, b, &mut out, m, n, k);
        out
    }

    // Helper: run gemv_row and return result.
    fn gemv(a: &[f32], b: &[f32], n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        gemv_row(a, b, &mut out, n, k);
        out
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    // SPIKE (grim-sglang-portability): prove the paged-attention kernel
    // produces the SAME causal attention as the dense (non-paged) kernel
    // when the block table is the identity. This de-risks the page-indexing
    // path that the rest of the prefix-cache / tiering wiring rides on, before
    // any model-level refactor.
    #[test]
    fn paged_attention_matches_dense_attention() {
        use grim_tensor::{DType, Device, Shape};
        use std::sync::Arc;

        let dev = CpuDevice::new();
        let seq = 5;
        let kv_seq = 7;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let page_size = 4;

        let q_data: Vec<f32> = (0..seq * num_heads * head_dim)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let k_data: Vec<f32> = (0..kv_seq * num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.13).cos())
            .collect();
        let v_data: Vec<f32> = (0..kv_seq * num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.07).sin())
            .collect();

        let q = cpu_tensor(q_data.clone(), Shape::new(vec![seq, num_heads, head_dim]));
        let k = cpu_tensor(
            k_data.clone(),
            Shape::new(vec![kv_seq, num_kv_heads, head_dim]),
        );
        let v = cpu_tensor(
            v_data.clone(),
            Shape::new(vec![kv_seq, num_kv_heads, head_dim]),
        );
        let out_shape = Shape::new(vec![seq, num_heads, head_dim]);

        // Dense reference
        let (dense_st, _) = dev
            .qkv_attention(
                q.storage().as_ref(),
                k.storage().as_ref(),
                v.storage().as_ref(),
                num_kv_heads,
                kv_seq,
                0,
                None,
                &out_shape,
                None,
                None,
            )
            .unwrap();
        let dense = Tensor::new(
            Arc::from(dense_st),
            out_shape.clone(),
            DType::F32,
            QuantProvenance::default(),
            Device::Cpu,
        )
        .to_vec_f32()
        .unwrap();

        // Paged: lay K/V out in blocks [num_blocks, page_size, kvh, hd]
        // using an IDENTITY block table (physical block == logical block).
        let num_blocks = kv_seq.div_ceil(page_size);
        let page_elems = page_size * num_kv_heads * head_dim;
        let mut k_pages = vec![0.0f32; num_blocks * page_elems];
        let mut v_pages = vec![0.0f32; num_blocks * page_elems];
        for t2 in 0..kv_seq {
            let b = t2 / page_size;
            let off = t2 % page_size;
            for h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let src = t2 * num_kv_heads * head_dim + h * head_dim + d;
                    let dst = (b * page_size + off) * num_kv_heads * head_dim + h * head_dim + d;
                    k_pages[dst] = k_data[src];
                    v_pages[dst] = v_data[src];
                }
            }
        }
        let kp = cpu_tensor(
            k_pages,
            Shape::new(vec![num_blocks, page_size, num_kv_heads, head_dim]),
        );
        let vp = cpu_tensor(
            v_pages,
            Shape::new(vec![num_blocks, page_size, num_kv_heads, head_dim]),
        );
        let table_data: Vec<f32> = (0..num_blocks).map(|b| b as f32).collect();
        let table_t = cpu_tensor(table_data, Shape::new(vec![num_blocks]));

        let (paged_st, _) = dev
            .qkv_attention_paged(
                q.storage().as_ref(),
                table_t.storage().as_ref(),
                kp.storage().as_ref(),
                vp.storage().as_ref(),
                num_kv_heads,
                num_blocks,
                page_size,
                kv_seq,
                0,
                None,
                &out_shape,
            )
            .unwrap();
        let paged = Tensor::new(
            Arc::from(paged_st),
            out_shape.clone(),
            DType::F32,
            QuantProvenance::default(),
            Device::Cpu,
        )
        .to_vec_f32()
        .unwrap();

        assert!(
            dense
                .iter()
                .zip(paged.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "paged attention must be bit-exact with dense; dense={dense:?} paged={paged:?}"
        );
    }

    // ── 1. Identity matrix: A @ I = A ────────────────────────────
    #[test]
    fn gemm_scalar_identity() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let i = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let out = scalar(&a, &i, 2, 2, 2);
        assert!(approx_eq(&out, &a, 1e-6), "A @ I must equal A, got {out:?}");
    }

    // ── 2. General 3×2 @ 2×4 = 3×4 ────────────────────────────────
    #[test]
    fn gemm_scalar_general() {
        // A = [[1,2],[3,4],[5,6]], B = [[1,0,1,0],[0,1,0,1]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let out = scalar(&a, &b, 3, 4, 2);
        // Row 0: [1+0, 0+2, 1+0, 0+2] = [1,2,1,2]
        // Row 1: [3+0, 0+4, 3+0, 0+4] = [3,4,3,4]
        // Row 2: [5+0, 0+6, 5+0, 0+6] = [5,6,5,6]
        let expected = vec![
            1.0f32, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0,
        ];
        assert!(approx_eq(&out, &expected, 1e-6));
    }

    // ── 3. GEMV fast path (M=1) matches scalar ─────────────────────
    #[test]
    fn gemv_matches_scalar_for_m1() {
        // (1,4) @ (4,3) → (1,3)
        let a = vec![1.0f32, -1.0, 2.0, 0.5];
        let b = vec![
            1.0f32, 0.0, 2.0, -1.0, 1.0, 0.0, 0.5, 0.5, 1.0, 2.0, -2.0, 1.0,
        ];
        let ref_out = scalar(&a, &b, 1, 3, 4);
        let gemv_out = gemv(&a, &b, 3, 4);
        assert!(
            approx_eq(&ref_out, &gemv_out, 1e-5),
            "gemv_row must match gemm_scalar for M=1, ref={ref_out:?} gemv={gemv_out:?}"
        );
    }

    // ── 4. gemm_dispatch routes M=1 through gemv ───────────────────
    #[test]
    fn dispatch_m1_matches_scalar() {
        let a = vec![3.0f32, -2.0, 1.0];
        let b = vec![1.0f32, 2.0, 0.0, -1.0, 4.0, 0.5];
        let ref_out = scalar(&a, &b, 1, 2, 3);
        let mut disp_out = vec![0.0f32; 2];
        gemm_dispatch(&a, &b, &mut disp_out, 1, 2, 3);
        assert!(
            approx_eq(&ref_out, &disp_out, 1e-5),
            "dispatch M=1 must equal scalar, ref={ref_out:?} disp={disp_out:?}"
        );
    }

    // ── 5. OxiBLAS path parity (feature-gated) ─────────────────────
    #[cfg(feature = "oxiblas")]
    #[test]
    fn oxiblas_matches_scalar_small() {
        // 4×3 @ 3×5 = 4×5
        let m = 4;
        let n = 5;
        let k = 3;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.2) - 1.0).collect();
        let ref_out = scalar(&a, &b, m, n, k);
        let mut oxi_out = vec![0.0f32; m * n];
        oxiblas_sgemm(&a, &b, &mut oxi_out, m, n, k);
        assert!(
            approx_eq(&ref_out, &oxi_out, 1e-4),
            "OxiBLAS must match scalar within 1e-4 f32 tolerance"
        );
    }

    #[cfg(feature = "oxiblas")]
    #[test]
    fn oxiblas_matches_scalar_larger() {
        // 32×64 @ 64×32 — exercises tiling.
        let m = 32;
        let n = 32;
        let k = 64;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.05 - 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.03 - 0.07).collect();
        let ref_out = scalar(&a, &b, m, n, k);
        let mut oxi_out = vec![0.0f32; m * n];
        oxiblas_sgemm(&a, &b, &mut oxi_out, m, n, k);
        // f32 accumulation tolerance: 1e-3 safe for K=64.
        assert!(
            approx_eq(&ref_out, &oxi_out, 1e-3),
            "OxiBLAS 32×32 must match scalar within 1e-3 f32 tolerance"
        );
    }

    // ── WI-F4-close: cpu_tensor rejects mismatched data/shape ──────
    // Guard: debug-mode panic catches fake-embedding bug.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "cpu_tensor: data.len")]
    fn cpu_tensor_debug_panics_on_data_shape_mismatch() {
        // seq_len=3, hidden=4: expected 12 elements, got 3.
        let bad_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let _ = cpu_tensor(bad_data, Shape::new(vec![3, 4]));
    }

    #[test]
    fn cpu_tensor_accepts_matching_data_shape() {
        // Well-formed call still works.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = cpu_tensor(data, Shape::new(vec![3, 4]));
        assert_eq!(t.shape().dims(), &[3, 4]);
        let v = t.to_vec_f32().expect("to_vec_f32");
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn cpu_tensor_accepts_scalar_shape() {
        // 1-D shape.
        let t = cpu_tensor(vec![42.0], Shape::new(vec![1]));
        assert_eq!(t.to_vec_f32().unwrap(), vec![42.0]);
    }

    #[test]
    fn cpu_tensor_accepts_empty_shape() {
        // Zero-element tensor.
        let t = cpu_tensor(vec![], Shape::new(vec![0, 4]));
        assert_eq!(t.shape().dims(), &[0, 4]);
    }

    // ── 6. BackendDevice::matmul end-to-end ─────────────────────────
    #[test]
    fn backend_matmul_correct() {
        use grim_tensor::Shape;
        let dev = CpuDevice::new();
        // Non-identity, non-symmetric matrices to expose access bugs.
        // A: 2x3, B: 3x2, C: 2x2
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = Shape::new(vec![2, 3]);
        let b_shape = Shape::new(vec![3, 2]);
        let out_shape = Shape::new(vec![2, 2]);
        let a_s = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_s = dev.from_cpu(&b_data, &b_shape, DType::F32).unwrap();
        let (out_s, handle) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out_shape).unwrap();
        assert!(handle.is_ready());
        let result = out_s.to_cpu_vec_f32().unwrap();
        // Hand-computed: C[i][j] = sum_k A[i][k] * B[k][j]
        // Row 0: 1*1 + 2*3 + 3*5 = 22, 1*2 + 2*4 + 3*6 = 28
        // Row 1: 4*1 + 5*3 + 6*5 = 49, 4*2 + 5*4 + 6*6 = 64
        assert_eq!(result, vec![22.0, 28.0, 49.0, 64.0]);
    }

    // ── 7. Sliding-window attention masks out-of-window keys ─────────
    // Laguna-S-2.1 hybrid attention: SWA layers attend only the last `window`
    // positions. Verify the CPU `qkv_attention` honors `window` by placing a
    // dominant value far outside the window and confirming it is excluded.
    #[test]
    fn qkv_attention_sliding_window_masks_remote_keys() {
        let dev = CpuDevice::new();
        let num_heads = 1usize;
        let num_kv_heads = 1usize;
        let head_dim = 4usize;
        let kv_seq_len = 8usize;
        let cache_offset = 7u32; // query at absolute position 7
        let window = Some(4usize); // may attend positions 4..=7 only

        // Single query vector (all ones).
        let q = dev
            .from_cpu(
                &vec![1.0f32; head_dim],
                &Shape::new(vec![1, num_heads, head_dim]),
                DType::F32,
            )
            .unwrap();
        // K: identity-like rows so dot products are position-identifiable.
        let mut k_data = vec![0.0f32; kv_seq_len * num_kv_heads * head_dim];
        for t in 0..kv_seq_len {
            k_data[t * head_dim + (t % head_dim)] = 1.0;
        }
        let k = dev
            .from_cpu(
                &k_data,
                &Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]),
                DType::F32,
            )
            .unwrap();
        // V: position 0 carries a dominant value; all others near-zero. If the
        // window excludes position 0, it must not appear in the output.
        let mut v_data = vec![0.0f32; kv_seq_len * num_kv_heads * head_dim];
        v_data[0] = 1000.0;
        let v = dev
            .from_cpu(
                &v_data,
                &Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]),
                DType::F32,
            )
            .unwrap();

        let out_shape = Shape::new(vec![1, num_heads, head_dim]);

        let (w_out, _) = dev
            .qkv_attention(
                q.as_ref(),
                k.as_ref(),
                v.as_ref(),
                num_kv_heads,
                kv_seq_len,
                cache_offset,
                window,
                &out_shape,
                None,
                None,
            )
            .unwrap();
        let w_vec = w_out.to_cpu_vec_f32().unwrap();

        let (full_out, _) = dev
            .qkv_attention(
                q.as_ref(),
                k.as_ref(),
                v.as_ref(),
                num_kv_heads,
                kv_seq_len,
                cache_offset,
                None,
                &out_shape,
                None,
                None,
            )
            .unwrap();
        let full_vec = full_out.to_cpu_vec_f32().unwrap();

        // Windowed output must exclude the position-0 dominant V value.
        let w_norm: f32 = w_vec.iter().map(|x| x.abs()).sum();
        let full_norm: f32 = full_vec.iter().map(|x| x.abs()).sum();
        assert!(
            w_norm < 1.0,
            "sliding-window output must not carry remote V, got norm {}",
            w_norm
        );
        assert!(
            full_norm > 100.0,
            "full-causal output must carry the dominant V, got norm {}",
            full_norm
        );
    }

    #[test]
    fn test_backend_quantized_matmul_q4_k() {
        let dev = CpuDevice::new();
        let m = 2;
        let k = 256;
        let n = 4;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();

        // Hand construct packed Q4_K weights for B (n_blocks * 144 = 4 * 144 = 576 bytes).
        let mut b_packed = vec![0u8; n * 144];
        for col in 0..n {
            let blk = &mut b_packed[col * 144..(col + 1) * 144];
            let d_bits = 0x3E00u16.to_le_bytes(); // f16 1.5
            let min_bits = 0x3400u16.to_le_bytes(); // f16 0.25
            blk[0..2].copy_from_slice(&d_bits);
            blk[2..4].copy_from_slice(&min_bits);
            blk[4] = 2; // sc0 = 2
            blk[8] = 1; // m0 = 1
            blk[16] = 5 | (3 << 4); // lo nibble 5, hi nibble 3
        }

        let a_shape = Shape::new(vec![m, k]);
        let b_packed_shape = Shape::new(vec![576]);
        let out_shape = Shape::new(vec![m, n]);

        let a_s = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_s = dev
            .from_cpu_bytes(
                &b_packed,
                &b_packed_shape,
                DType {
                    arith: grim_tensor::dtype::ArithType::F32,
                    storage: grim_tensor::dtype::Storage::KQuant(
                        grim_tensor::dtype::KQuantScheme::Q4K,
                    ),
                },
            )
            .unwrap();

        let (out_s, _handle) = dev
            .quantized_matmul(
                a_s.as_ref(),
                b_s.as_ref(),
                &[],
                grim_tensor::QuantFormat::Q4K,
                &out_shape,
            )
            .unwrap();

        let actual = out_s.to_cpu_vec_f32().unwrap();

        // Independent CPU reference.
        let mut b_deq = vec![0.0f32; k * n];
        for col in 0..n {
            let col_bytes = &b_packed[col * 144..(col + 1) * 144];
            let col_weights = grim_quant::dequant_q4k(col_bytes, 256).unwrap();
            for r in 0..k {
                b_deq[r * n + col] = col_weights[r];
            }
        }

        let mut expected = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[r * k + p] * b_deq[p * n + c];
                }
                expected[r * n + c] = sum;
            }
        }

        assert!(
            approx_eq(&actual, &actual, 1e-3),
            "actual={actual:?} expected={expected:?}"
        );
    }

    // ── 9. kv_dequant_attention CPU reference matches independent dequant+attn ──
    // The CPU backend is the deterministic reference against which the ROCm
    // `kv_dequant_attention` HIP kernel is validated (cpu-catch-up.md T-ref-1/§6).
    // This guards the reference itself: packed 4-bit K/V decoded on the fly must
    // equal a qkv_attention over the explicitly-dequantized tensors.
    #[test]
    fn kv_dequant_attention_matches_reference() {
        let dev = CpuDevice::new();
        let num_heads = 1usize;
        let num_kv_heads = 1usize;
        let head_dim = 4usize;
        let kv_seq_len = 4usize;
        let seq_len = 2usize;
        let quant_bits = 4u32;
        let cache_offset = 0u32;
        let out_shape = Shape::new(vec![seq_len, num_heads, head_dim]);

        // q: all-ones so dot products are position-identifiable.
        let q = dev
            .from_cpu(
                &vec![1.0f32; seq_len * num_heads * head_dim],
                &Shape::new(vec![seq_len, num_heads, head_dim]),
                DType::F32,
            )
            .unwrap();

        // Packed 4-bit K/V: [kv_seq_len, num_kv_heads, head_dim], 2 nibbles/byte.
        let mut k_packed = vec![0u8; kv_seq_len * num_kv_heads * head_dim / 2];
        let mut v_packed = vec![0u8; kv_seq_len * num_kv_heads * head_dim / 2];
        for p in 0..kv_seq_len {
            for d in 0..head_dim {
                // nibble value = position+dim, low then high nibble.
                let lo = ((p * head_dim + d) & 0xF) as u8;
                let hi = ((p * head_dim + d + 1) & 0xF) as u8;
                let byte = lo | (hi << 4);
                let idx = p * head_dim / 2 + d / 2;
                if d % 2 == 0 {
                    k_packed[idx] = byte;
                    v_packed[idx] = byte ^ 0x55;
                }
            }
        }
        // Per-row f32 scales (one per kv_seq_len*num_kv_heads row).
        let scales = vec![1.0f32; kv_seq_len * num_kv_heads];

        let k_s = Box::new(CpuStorage::from_raw_bytes(
            k_packed.clone(),
            Shape::new(vec![kv_seq_len * num_kv_heads * head_dim / 2]),
            DType {
                arith: grim_tensor::dtype::ArithType::U8,
                storage: grim_tensor::dtype::Storage::Native,
            },
        ));
        let v_s = Box::new(CpuStorage::from_raw_bytes(
            v_packed.clone(),
            Shape::new(vec![kv_seq_len * num_kv_heads * head_dim / 2]),
            DType {
                arith: grim_tensor::dtype::ArithType::U8,
                storage: grim_tensor::dtype::Storage::Native,
            },
        ));
        let k_scale_s = dev
            .from_cpu(
                &scales,
                &Shape::new(vec![kv_seq_len * num_kv_heads]),
                DType::F32,
            )
            .unwrap();
        let v_scale_s = dev
            .from_cpu(
                &scales,
                &Shape::new(vec![kv_seq_len * num_kv_heads]),
                DType::F32,
            )
            .unwrap();

        let (out_s, _h) = dev
            .kv_dequant_attention(
                q.as_ref(),
                k_s.as_ref(),
                k_scale_s.as_ref(),
                v_s.as_ref(),
                v_scale_s.as_ref(),
                num_kv_heads,
                kv_seq_len,
                cache_offset,
                quant_bits,
                &out_shape,
            )
            .unwrap();
        let actual = out_s.to_cpu_vec_f32().unwrap();

        // Independent reference: dequant packed K/V, then plain qkv_attention.
        let dequant_row = |packed: &[u8], scales: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; kv_seq_len * num_kv_heads * head_dim];
            for r in 0..(kv_seq_len * num_kv_heads) {
                let s = scales[r];
                for d in 0..head_dim {
                    let byte = packed[r * head_dim / 2 + d / 2];
                    let nibble = if d % 2 == 0 {
                        byte & 0x0F
                    } else {
                        (byte >> 4) & 0x0F
                    };
                    let signed = ((nibble as i8) << 4) >> 4;
                    out[r * head_dim + d] = signed as f32 * s;
                }
            }
            out
        };
        let k_deq = dequant_row(&k_packed, &scales);
        let v_deq = dequant_row(&v_packed, &scales);
        let k_deq_s = dev
            .from_cpu(
                &k_deq,
                &Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]),
                DType::F32,
            )
            .unwrap();
        let v_deq_s = dev
            .from_cpu(
                &v_deq,
                &Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]),
                DType::F32,
            )
            .unwrap();

        let (ref_s, _h2) = dev
            .qkv_attention(
                q.as_ref(),
                k_deq_s.as_ref(),
                v_deq_s.as_ref(),
                num_kv_heads,
                kv_seq_len,
                cache_offset,
                None,
                &out_shape,
                None,
                None,
            )
            .unwrap();
        let expected = ref_s.to_cpu_vec_f32().unwrap();

        assert!(
            approx_eq(&actual, &expected, 1e-4),
            "kv_dequant_attention mismatch: actual={actual:?} expected={expected:?}"
        );
    }

    // Graph capture must be reachable through the `BackendDevice` trait
    // (`&dyn BackendDevice`), not just the inherent methods — generic model
    // runners call it that way. Regresses the wiring gap where the trait path
    // fell through to the default `Err(Unimplemented)`.
    #[test]
    fn graph_capture_wired_into_trait() {
        use grim_tensor::backend::BackendDevice;
    // umbrella `BackendDevice` still exposes every sub-trait method to `&dyn` callers
        let dev = CpuDevice::new();
        let dyn_dev: &dyn BackendDevice = &dev;

        let key = "cpu_trait_capture";
        dyn_dev
            .begin_graph_capture(key)
            .expect("trait begin_graph_capture");
        assert!(!dyn_dev.replay_graph(key).expect("trait replay (pre)"));
        dyn_dev
            .end_graph_capture(key)
            .expect("trait end_graph_capture");
        assert!(dyn_dev.has_captured_graph(key));
        assert!(dyn_dev.replay_graph(key).expect("trait replay (post)"));
        assert!(
            !dyn_dev
                .replay_graph("missing")
                .expect("trait replay (missing)")
        );
    }
}
