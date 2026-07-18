//! CPU `BackendDevice`.
//!
//! ## GEMM dispatch (§4.1 — OxiBLAS)
//!
//! Matrix multiplication routes through [`gemm_dispatch`], which selects
//! the fastest available path at compile time:
//!
//! 1. **`oxiblas` feature on (default):** calls `matrixmultiply::sgemm` —
//!    a pure-Rust SIMD-accelerated BLAS kernel (no Fortran, no LAPACK,
//!    no C++ linkage). This is the backend `scirs2-linalg` uses internally
//!    and matches the "OxiBLAS" target in the Grim architecture document.
//! 2. **M = 1 (GEMV fast path):** for the single-token decode step, we
//!    avoid a full M×N×K loop and instead walk rows of B in a single pass,
//!    which maps well to prefetching behaviour on both x86-64 and ARM.
//!    This path is always available regardless of the `oxiblas` feature.
//! 3. **Scalar fallback:** triple-loop GEMM, compiled when `oxiblas` is
//!    disabled. Kept so the crate builds on targets without SIMD or under
//!    `--no-default-features` for fuzzing/no-std builds.

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
        // Tier 1 — safe-by-construction: all slices are sized by shape assertions above;
        // the dispatch is a normal Rust fn with no unsafe in this call site.
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
        Ok(Box::new(CpuStorage::new(data.to_vec(), shape.clone(), dtype)))
    }

    fn advise(&self, _storage: &dyn BackendStorage, _advice: grim_tensor::backend::MemAdvice) -> Result<()> {
        // CPU backend: advice is currently a no-op (can extend to madvise mapping later if needed)
        Ok(())
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

// ---------- GEMM dispatch (§4.1 OxiBLAS) ----------

/// Top-level GEMM dispatcher. Row-major `(M,K) @ (K,N) → (M,N)`.
///
/// Selection order:
/// 1. M=1 GEMV fast path — always tried first regardless of feature flags.
/// 2. `oxiblas` feature: `matrixmultiply::sgemm` SIMD kernel.
/// 3. Scalar fallback triple-loop.
///
/// WHY: the M=1 decode-step is the hottest path in autoregressive inference;
/// giving it its own loop avoids the overhead of the full GEMM setup for a
/// single output row. For M>1 (prefill, batched decode) the SIMD kernel
/// dominates.
fn gemm_dispatch(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    // Fast path: M=1 (single-token decode). A dot-product per output column.
    if m == 1 {
        gemv_row(a, b, out, n, k);
        return;
    }

    // SIMD path: OxiBLAS / matrixmultiply::sgemm.
    #[cfg(feature = "oxiblas")]
    {
        oxiblas_sgemm(a, b, out, m, n, k);
        return;
    }

    // Scalar fallback (compiled when `oxiblas` is disabled).
    #[cfg(not(feature = "oxiblas"))]
    gemm_scalar(a, b, out, m, n, k);
}

/// GEMV fast path for M=1 (single-row A, single output row).
///
/// WHY M=1 gets its own path: the common decode step is a single token,
/// so the matmul is `(1,K) @ (K,N) → (1,N)` — one dot-product per output
/// column. Walking B column-by-column in the inner loop is cache-unfriendly;
/// instead we iterate over K (rows of B, which are contiguous in row-major
/// layout) accumulating into a result row, giving sequential memory access
/// in both A and B.
///
/// This is equivalent to `y = A[0] · B` where A[0] is the sole input row.
fn gemv_row(a: &[f32], b: &[f32], out: &mut [f32], n: usize, k: usize) {
    // Zero output first — out is pre-allocated by caller.
    for o in out[..n].iter_mut() { *o = 0.0; }
    // Accumulate: for each k-index, scatter a[k] * B[k,*] into out.
    for p in 0..k {
        let ap = a[p];
        let b_row = &b[p * n..(p + 1) * n];
        for (oj, &bv) in out[..n].iter_mut().zip(b_row.iter()) {
            *oj += ap * bv;
        }
    }
}

/// OxiBLAS (matrixmultiply::sgemm) SIMD-accelerated GEMM.
///
/// `matrixmultiply` is the pure-Rust kernel underlying `scirs2-linalg`'s
/// BLAS substrate. It compiles SIMD via stdarch auto-vectorisation without
/// requiring any C/Fortran toolchain or external `.so`.
///
/// Contract (Tier 2 — explicit unsafe with documented invariants):
/// - `a.len() >= m * k`, `b.len() >= k * n`, `out.len() >= m * n`.
/// - All inputs are row-major, no aliasing between a/b and out.
/// - `rsa`, `csa`, `rsb`, `csb`, `rsc`, `csc` are the standard row/column
///   strides for a, b, and c respectively in the matrixmultiply API.
#[cfg(feature = "oxiblas")]
fn oxiblas_sgemm(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    // SAFETY:
    // • Pointers are obtained from Rust slice references — lifetime and
    //   alignment are guaranteed by the borrow checker.
    // • Sizes m/n/k were validated by the caller (shape assertions in `matmul`).
    // • alpha=1.0, beta=0.0: pure overwrite, no accumulation into prior output.
    // • No aliasing: out is a freshly allocated Vec<f32> zeroed by the caller.
    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0_f32,                // alpha
            a.as_ptr(), k as isize, 1,  // rsa, csa
            b.as_ptr(), n as isize, 1,  // rsb, csb
            0.0_f32,                // beta (overwrite out)
            out.as_mut_ptr(), n as isize, 1, // rsc, csc
        );
    }
}

/// Scalar triple-loop GEMM. Row-major `(M,K) @ (K,N) → (M,N)`.
///
/// WHY kept: serves as the exact reference implementation that correctness
/// tests compare OxiBLAS results against, and as the compile target when
/// `--no-default-features` is set (fuzz builds, embedded, etc.).
#[allow(dead_code)]
fn gemm_scalar(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
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

    // ── 1. Identity matrix: A @ I = A ─────────────────────────────────────
    #[test]
    fn gemm_scalar_identity() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let i = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let out = scalar(&a, &i, 2, 2, 2);
        assert!(approx_eq(&out, &a, 1e-6), "A @ I must equal A, got {out:?}");
    }

    // ── 2. General 3×2 @ 2×4 = 3×4 ───────────────────────────────────────
    #[test]
    fn gemm_scalar_general() {
        // A = [[1,2],[3,4],[5,6]], B = [[1,0,1,0],[0,1,0,1]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let out = scalar(&a, &b, 3, 4, 2);
        // Row 0: [1+0, 0+2, 1+0, 0+2] = [1,2,1,2]
        // Row 1: [3+0, 0+4, 3+0, 0+4] = [3,4,3,4]
        // Row 2: [5+0, 0+6, 5+0, 0+6] = [5,6,5,6]
        let expected = vec![1.0f32,2.0,1.0,2.0, 3.0,4.0,3.0,4.0, 5.0,6.0,5.0,6.0];
        assert!(approx_eq(&out, &expected, 1e-6));
    }

    // ── 3. GEMV fast path (M=1) matches scalar ────────────────────────────
    #[test]
    fn gemv_matches_scalar_for_m1() {
        // (1,4) @ (4,3) → (1,3)
        let a = vec![1.0f32, -1.0, 2.0, 0.5];
        let b = vec![
            1.0f32, 0.0, 2.0,
           -1.0,   1.0, 0.0,
            0.5,   0.5, 1.0,
            2.0,  -2.0, 1.0,
        ];
        let ref_out = scalar(&a, &b, 1, 3, 4);
        let gemv_out = gemv(&a, &b, 3, 4);
        assert!(
            approx_eq(&ref_out, &gemv_out, 1e-5),
            "gemv_row must match gemm_scalar for M=1, ref={ref_out:?} gemv={gemv_out:?}"
        );
    }

    // ── 4. gemm_dispatch routes M=1 through gemv (same result) ────────────
    #[test]
    fn dispatch_m1_matches_scalar() {
        let a = vec![3.0f32, -2.0, 1.0];
        let b = vec![
            1.0f32, 2.0,
            0.0,   -1.0,
            4.0,    0.5,
        ];
        let ref_out = scalar(&a, &b, 1, 2, 3);
        let mut disp_out = vec![0.0f32; 2];
        gemm_dispatch(&a, &b, &mut disp_out, 1, 2, 3);
        assert!(
            approx_eq(&ref_out, &disp_out, 1e-5),
            "dispatch M=1 must equal scalar, ref={ref_out:?} disp={disp_out:?}"
        );
    }

    // ── 5. OxiBLAS path parity (only when feature is on) ──────────────────
    #[cfg(feature = "oxiblas")]
    #[test]
    fn oxiblas_matches_scalar_small() {
        // 4×3 @ 3×5 = 4×5
        let m = 4; let n = 5; let k = 3;
        let a: Vec<f32> = (0..m*k).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.2) - 1.0).collect();
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
        // 32×64 @ 64×32 — a size that exercises OxiBLAS's tiling.
        let m = 32; let n = 32; let k = 64;
        let a: Vec<f32> = (0..m*k).map(|i| ((i % 7) as f32) * 0.05 - 0.1).collect();
        let b: Vec<f32> = (0..k*n).map(|i| ((i % 5) as f32) * 0.03 - 0.07).collect();
        let ref_out = scalar(&a, &b, m, n, k);
        let mut oxi_out = vec![0.0f32; m * n];
        oxiblas_sgemm(&a, &b, &mut oxi_out, m, n, k);
        // f32 accumulation tolerance grows with K; 1e-3 is safe for K=64.
        assert!(
            approx_eq(&ref_out, &oxi_out, 1e-3),
            "OxiBLAS 32×32 must match scalar within 1e-3 f32 tolerance"
        );
    }

    // ── 6. BackendDevice::matmul end-to-end ───────────────────────────────
    #[test]
    fn backend_matmul_correct() {
        use grim_tensor::Shape;
        let dev = CpuDevice::new();
        let a_data = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0]; // 2×2
        let a_shape = Shape::new(vec![2, 2]);
        let b_shape = Shape::new(vec![2, 2]);
        let out_shape = Shape::new(vec![2, 2]);
        let a_s = dev.from_cpu(&a_data, &a_shape, DType::F32).unwrap();
        let b_s = dev.from_cpu(&b_data, &b_shape, DType::F32).unwrap();
        let (out_s, handle) = dev.matmul(a_s.as_ref(), b_s.as_ref(), &out_shape).unwrap();
        assert!(handle.is_ready());
        let result = out_s.to_cpu_vec_f32().unwrap();
        // I @ B = B
        assert!(approx_eq(&result, &b_data, 1e-6),
            "identity matmul must equal B, got {result:?}");
    }
}
