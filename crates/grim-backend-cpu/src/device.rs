//! CPU `BackendDevice`. Pure scalar v1 with hooks in place for SIMD
//! specialization later (`packed_simd` / `std::simd`).

use std::sync::Arc;

use grim_tensor::backend::{ComputeHandle, ReadyHandle};
use grim_tensor::dtype::{DType, Device, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendDevice, BackendStorage, Shape};

use crate::storage::CpuStorage;

/// CPU device. Operations are synchronous — the returned `ComputeHandle`
/// is always `ReadyHandle`.
#[derive(Debug, Clone, Default)]
pub struct CpuDevice;

impl CpuDevice {
    pub fn new() -> Self {
        Self
    }
}

impl BackendDevice for CpuDevice {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        ensure_cpu_native(&dtype)?;
        let n = shape.elem_count();
        Ok(Box::new(CpuStorage::new(vec![0.0; n], shape.clone(), dtype)))
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
        if out_shape.dims() != &[m, n] {
            return Err(Error::Shape(format!("expected out [{m},{n}], got {out_shape:?}")));
        }
        let mut out = vec![0.0f32; m * n];
        gemm_naive(a.data(), b.data(), &mut out, m, n, k);
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
        if !a.shape().broadcast_compatible(b.shape()) || !a.shape().broadcast_compatible(out_shape) {
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
            *o = aa[broadcast_index(i, sa, out_dims)]
                + bb[broadcast_index(i, sb, out_dims)];
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
        if !a.shape().broadcast_compatible(b.shape()) || !a.shape().broadcast_compatible(out_shape) {
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
            *o = aa[broadcast_index(i, sa, out_dims)]
                * bb[broadcast_index(i, sb, out_dims)];
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
        if x.shape() != out_shape {
            // Allow leading dim flatten: x=[B,S,H], out=[B*S,H] is valid when
            // the last dim and total elem count match.
            if x.shape().elem_count() != out_shape.elem_count()
                || x.shape().dims().last() != out_shape.dims().last()
                || out_shape.rank() != 2
            {
                return Err(Error::Shape("rms_norm: shape mismatch".into()));
            }
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
                return Err(Error::IndexOutOfBounds(format!("token {tok} >= vocab {vocab}")));
            }
            out[i * dim..(i + 1) * dim].copy_from_slice(&wd[tok * dim..(tok + 1) * dim]);
        }
        Ok((
            Box::new(CpuStorage::new(out, out_shape.clone(), DType::F32)),
            Box::new(ReadyHandle),
        ))
    }
}

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
    fn to_cpu_vec_f32(&self) -> Result<Vec<f32>> {
        Ok((*self.data).clone())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------- helpers ----------

fn a_storage<'a>(s: &'a dyn BackendStorage) -> Result<&'a CpuStorage> {
    s.as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Backend("storage is not CpuStorage".into()))
}

fn b_storage<'a>(s: &'a dyn BackendStorage) -> Result<&'a CpuStorage> {
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

fn gemm_naive(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    // Row-major (M,K) @ (K,N) -> (M,N) with no transpose.
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = s;
        }
    }
}

fn broadcast_index(linear: usize, src_dims: &[usize], out_dims: &[usize]) -> usize {
    // out is row-major; src may have lower rank (left-pad with 1s) and
    // any dim equal to 1 broadcasts.
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

/// Convenience: build a host tensor owned by `CpuDevice`.
pub fn cpu_tensor(data: Vec<f32>, shape: Shape) -> grim_tensor::Tensor {
    grim_tensor::Tensor::new(
        Arc::new(CpuStorage::new(data, shape.clone(), DType::F32)),
        shape,
        DType::F32,
        QuantProvenance::default(),
        Device::Cpu,
    )
}
