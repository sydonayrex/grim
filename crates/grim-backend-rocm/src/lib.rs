//! ROCm backend for Grim — primary GPU target per architecture §4.
//!
//! This crate currently provides the `RocmDevice` and `RocmStorage` shells
//! plus the dispatch trait surface. Real hip/rocBLAS binding work lives in:
//!
//! - `hip-sys`: FFI bindings to ROCm's HIP runtime
//! - `rocblas-sys`: FFI bindings to rocBLAS
//!
//! Both need to be vendored or generated against an ROCm 6.x install before
//! the matmul/elementwise ops here can ship. Phase 4 of the roadmap wires
//! those in; this scaffold is shape-complete but the kernels fallback to
//! an explicit error until the hip runtime is linked.

use grim_tensor::backend::ComputeHandle;
use grim_tensor::dtype::{DType, QuantProvenance};
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendDevice, BackendStorage, Shape};

/// A handle to a queued ROCm stream operation. Tracks completion for the
/// `ComputeHandle` contract — when hip-stream integration lands, this holds
/// a stream/event pair; today it is a no-op `ReadyHandle` stand-in used only
/// by the placeholder ops below.
#[derive(Debug)]
pub struct RocmHandle;

impl ComputeHandle for RocmHandle {
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    fn is_ready(&self) -> bool {
        true
    }
}

/// ROCm-side tensor storage. Holds a hipDeviceptr_t (opaque u64) plus
/// shape/dtype/provenance metadata; the GPU-side buffer is allocated via
/// `hipMalloc`/`hipFree` in the real backend (not yet active).
#[derive(Debug)]
pub struct RocmStorage {
    /// Opaque device pointer when the kernel surface is active; `None`
    /// for the current placeholder where no GPU allocation has happened.
    device_ptr: Option<u64>,
    bytes: usize,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    ordinal: usize,
}

impl RocmStorage {
    pub fn shape_metadata(&self) -> &Shape {
        &self.shape
    }

    pub fn device_ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn device_ptr(&self) -> Option<u64> {
        self.device_ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl BackendStorage for RocmStorage {
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
        // Real impl: launch a hipMemcpy DtoH. Until hip-sys is wired, fail
        // loudly so callers route through the CPU placeholder rather than
        // silently getting zero-filled data.
        Err(Error::Unimplemented(
            "grim-backend-rocm DtoH copy not yet wired (hip-sys pending). \
             Use grim-backend-cpu for now."
                .into(),
        ))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// ROCm device. Constructed per-GPU-ordinal; matmul/op implementations
/// delegate to rocBLAS once the FFI bindings are linked.
#[derive(Debug, Clone)]
pub struct RocmDevice {
    ordinal: usize,
}

impl RocmDevice {
    pub fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    /// Probe for available ROCm GPUs and return one device per ordinal.
    /// Today: returns an empty list unless `GRIM_ROCM_ORDINAL_OVERRIDE`
    /// is set in the environment, in which case that ordinal is reported.
    /// Replace with `hipGetDeviceCount` once hip-sys is wired.
    pub fn probe() -> Result<Vec<RocmDevice>> {
        if let Ok(s) = std::env::var("GRIM_ROCM_ORDINAL_OVERRIDE") {
            if let Ok(n) = s.parse::<usize>() {
                return Ok(vec![RocmDevice::new(n)]);
            }
        }
        Ok(vec![])
    }
}

impl BackendDevice for RocmDevice {
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        Ok(Box::new(RocmStorage {
            device_ptr: None,
            bytes: shape.elem_count() * dtype_byte_size(&dtype),
            shape: shape.clone(),
            dtype,
            provenance: QuantProvenance::GrimNative,
            ordinal: self.ordinal,
        }))
    }

    fn matmul(
        &self,
        _a: &dyn BackendStorage,
        _b: &dyn BackendStorage,
        _out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(Error::Unimplemented(
            "ROCM matmul pending hip-sys/rocblas-sys link".into(),
        ))
    }

    fn add(
        &self,
        _a: &dyn BackendStorage,
        _b: &dyn BackendStorage,
        _out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(Error::Unimplemented("ROCM add pending".into()))
    }

    fn mul(
        &self,
        _a: &dyn BackendStorage,
        _b: &dyn BackendStorage,
        _out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(Error::Unimplemented("ROCM mul pending".into()))
    }

    fn silu_mul(
        &self,
        _gate: &dyn BackendStorage,
        _up: &dyn BackendStorage,
        _out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(Error::Unimplemented("ROCM silu_mul pending".into()))
    }

    fn rms_norm(
        &self,
        _x: &dyn BackendStorage,
        _w: &dyn BackendStorage,
        _eps: f32,
        _out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(Error::Unimplemented("ROCM rms_norm pending".into()))
    }

    fn softmax(
        &self,
        _x: &dyn BackendStorage,
        _out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(Error::Unimplemented("ROCM softmax pending".into()))
    }

    fn embedding(
        &self,
        _weight: &dyn BackendStorage,
        _indices: &[u32],
        _out: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        Err(Error::Unimplemented("ROCM embedding pending".into()))
    }
}

fn dtype_byte_size(dtype: &DType) -> usize {
    use grim_tensor::ArithType;
    match dtype.arith {
        ArithType::F32 | ArithType::U32 => 4,
        ArithType::F16 | ArithType::BF16 => 2,
        ArithType::I64 => 8,
        ArithType::U8 => 1,
    }
}
