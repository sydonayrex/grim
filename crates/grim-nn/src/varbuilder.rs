//! `WeightSource` — depth-first cursor over a `TensorProvider`.
//!
//! Mirrors Candle's `VarBuilder` exactly: every model constructor walks a
//! config-defined layer hierarchy and pulls tensors by prefix. Per-tensor
//! dtype/provenance resolution (§4.2, §7.2) happens in `get()`.

use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_tensor::error::{Error, Result};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;
use grim_tensor::RawTensor;

use grim_backend_cpu::{cpu_tensor, CpuDevice};

/// A handle that walks a `TensorProvider` by hierarchical prefix. Models
/// call `ws.pp("model").pp("layers").pp("0").get(...)` to materialize
/// tensors; the call-site shape determines what storage type comes back.
pub struct WeightSource<'a> {
    tensors: &'a dyn grim_tensor::TensorProvider,
    prefix: Vec<String>,
    default_dtype: DType,
    default_provenance: QuantProvenance,
    device: Device,
}

impl<'a> WeightSource<'a> {
    pub fn new(
        tensors: &'a dyn grim_tensor::TensorProvider,
        default_dtype: DType,
        default_provenance: QuantProvenance,
        device: Device,
    ) -> Self {
        Self {
            tensors,
            prefix: Vec::new(),
            default_dtype,
            default_provenance,
            device,
        }
    }

    /// Root-level builder from a `TensorProvider`.
    pub fn root(
        tensors: &'a dyn grim_tensor::TensorProvider,
        device: Device,
    ) -> Self {
        Self::new(tensors, DType::F32, QuantProvenance::GrimNative, device)
    }

    /// Push a path segment and return a new `WeightSource` whose prefix is
    /// `self.prefix + [name]`. Mirrors `candle::VarBuilder::pp`.
    pub fn pp(&self, name: &str) -> WeightSource<'a> {
        let mut next = self.clone_prefix();
        next.prefix.push(name.to_owned());
        next
    }

    fn clone_prefix(&self) -> WeightSource<'a> {
        WeightSource {
            tensors: self.tensors,
            prefix: self.prefix.clone(),
            default_dtype: self.default_dtype.clone(),
            default_provenance: self.default_provenance.clone(),
            device: self.device.clone(),
        }
    }

    fn full_name(&self, leaf: &str) -> String {
        let mut s = self.prefix.join(".");
        if !s.is_empty() {
            s.push('.');
        }
        s.push_str(leaf);
        s
    }

    /// Materialize a tensor of the given `shape` and `leaf` name under the
    /// current prefix. Resolves dtype + provenance per-tensor: first from
    /// the checkpoint's per-tensor metadata, then falls back to defaults.
    pub fn get(&self, shape: impl Into<Shape>, leaf: &str) -> Result<Tensor> {
        let shape = shape.into();
        let name = self.full_name(leaf);
        let raw = self.tensors.get(&name)?;

        if raw.shape != shape.dims() {
            return Err(Error::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: raw.shape.clone(),
            });
        }
        let (dtype, provenance) = match self.tensors.meta(&name) {
            Ok(m) => (m.dtype, m.provenance),
            Err(_) => (self.default_dtype.clone(), self.default_provenance.clone()),
        };

        tensor_from_raw(raw, shape, dtype, provenance, self.device.clone())
    }
}

fn tensor_from_raw(
    raw: RawTensor,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    device: Device,
) -> Result<Tensor> {
    // v1 only knows F32/Native. Quantized / half-precision storage types
    // arrive with the ROCm/Vulkan/GPTQ loaders in phases 2 + 7.
    if dtype.is_quantized() || dtype.arith != grim_tensor::ArithType::F32 {
        return Err(Error::Unimplemented(
            "WeightSource v1 only materializes F32/Native tensors; \
             quantized formats land with grim-quant in phase 2."
                .into(),
        ));
    }
    if !device.is_cpu() {
        return Err(Error::Unimplemented(
            "WeightSource v1 only materializes CPU tensors; \
             per-device materialization lands with each backend."
                .into(),
        ));
    }

    let f32s = bytes_to_f32(&raw.bytes, raw.shape.iter().product::<usize>())?;
    let _ = CpuDevice::new(); // keep the device type reachable
    let t = cpu_tensor(f32s, shape);
    let _ = provenance; // CPU backend stamps default GrimNative; per-tensor provenance
                        // will be carried once non-Native storage lands.
    let _ = dtype;
    Ok(t)
}

fn bytes_to_f32(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() != n * std::mem::size_of::<f32>() {
        return Err(Error::Backend(format!(
            "byte buffer length {} does not match f32 count {n}",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(v);
    }
    Ok(out)
}
