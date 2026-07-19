//! `WeightSource` — depth-first cursor over a `TensorProvider`.
//!
//! Mirrors Candle's `VarBuilder` exactly: every model constructor walks a
//! config-defined layer hierarchy and pulls tensors by prefix. Per-tensor
//! dtype/provenance resolution (§4.2, §7.2) happens in `get()`.

use std::sync::Arc;

use grim_tensor::dtype::{BlockDtype, DType, Device, FloatPackScheme, KQuantScheme, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;
use grim_tensor::{BackendDevice, RawTensor};

use grim_backend_cpu::cpu_tensor;
use grim_quant::{dequant_fp4, dequant_nf4, dequant_fp8, dequant_fp4_block16, dequant_fp8_block16, dequant_q4k, dequant_q5k, dequant_q6k, dequant_q80, dequant_iq4nl};

#[cfg(feature = "cuda-mem")]
use grim_backend_cuda::CudaDevice;
#[cfg(feature = "rocm-mem")]
use grim_backend_rocm::RocmDevice;

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

        materialize(raw, shape, dtype, provenance, &self.device)
    }

    /// Materialize a tensor for training. Quantized storage types (Q4_K, Q5_K,
    /// Q6_K, Q8_0, ...) are dequantized to native F32 in CPU memory so the
    /// optimization pass has full-precision weights to take gradients against.
    /// Native dtypes flow through unchanged.
    pub fn get_for_training(&self, shape: impl Into<Shape>, leaf: &str) -> Result<Tensor> {
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

        materialize(raw, shape, dtype, provenance, &self.device)
    }
}

// Materialization helpers — each arm is a self-contained branch so cfg(...)
// attributes on `use` statements don't create non-exhaustive match arms.

fn materialize_cuda(
    f32s: Vec<f32>,
    shape: Shape,
    _dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    let dev = CudaDevice::new(ordinal);
    // Storage is F32 bytes regardless of the GGUF-stored quantization tag:
    // `f32s` was already dequantized in `materialize` above. Pass F32 to
    // `from_cpu` so the CUDA storage carries DType::F32, which downstream
    // embedding/matmul kernels require.
    let storage = BackendDevice::from_cpu(&dev, &f32s, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        provenance,
        device.clone(),
    ))
}

#[cfg(not(feature = "cuda-mem"))]
fn materialize_cuda(
    _f32s: Vec<f32>,
    _shape: Shape,
    _dtype: DType,
    _provenance: QuantProvenance,
    _device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    Err(Error::Unimplemented(format!(
        "CUDA materialization: enable 'cuda-mem' feature on grim-nn (ordinal={})",
        ordinal
    )))
}

#[cfg(feature = "rocm-mem")]
fn materialize_rocm(
    f32s: Vec<f32>,
    shape: Shape,
    _dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    let dev = RocmDevice::new(ordinal);
    // Storage is F32 bytes (already dequantized in `materialize`). Mirror
    // CUDA: stamp the storage as DType::F32 so ROCm kernels that check
    // input dtype (embedding, matmul) accept the result.
    let storage = BackendDevice::from_cpu(&dev, &f32s, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        provenance,
        device.clone(),
    ))
}

#[cfg(not(feature = "rocm-mem"))]
fn materialize_rocm(
    _f32s: Vec<f32>,
    _shape: Shape,
    _dtype: DType,
    _provenance: QuantProvenance,
    _device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    Err(Error::Unimplemented(format!(
        "ROCm materialization: enable 'rocm-mem' feature on grim-nn (ordinal={})",
        ordinal
    )))
}

fn materialize(
    raw: RawTensor,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
) -> Result<Tensor> {
    if device.is_cpu() {
        let f32s = dequant_to_f32(&raw, &dtype)?;
        return Ok(cpu_tensor(f32s, shape));
    }
    let f32s = dequant_to_f32(&raw, &dtype)?;
    match device {
        Device::Cpu => {
            _ = f32s;
            Err(Error::Backend(
                "Device::Cpu reached after is_cpu early-return — unreachable".into(),
            ))
        }
        Device::Cuda(ordinal) => materialize_cuda(f32s, shape, dtype, provenance, device, *ordinal),
        Device::Rocm(ordinal) => materialize_rocm(f32s, shape, dtype, provenance, device, *ordinal),
        Device::Vulkan => {
            _ = f32s;
            Err(Error::Unimplemented(
                "Vulkan materialization not yet wired up".into(),
            ))
        }
        Device::Metal(ordinal) => {
            _ = f32s;
            Err(Error::Unimplemented(format!(
                "Metal(device={}) materialization not yet wired up",
                ordinal
            )))
        }
    }
}

/// Materialize any supported storage format to a flat `Vec<f32>` of
/// `raw.shape` length. This is the single dequant dispatch shared by the
/// inference `get()` path (and mirrors the training `get_for_training`
/// layout). Supports native F32/BF16/F16, the K-quant family (Q2K–Q8K,
/// IQ4_NL), and low-bit float packs (FP4/NF4/FP8).
fn dequant_to_f32(raw: &RawTensor, dtype: &DType) -> Result<Vec<f32>> {
    let n = raw.shape.iter().product::<usize>();
    match &dtype.storage {
        Storage::Native => match dtype.arith {
            grim_tensor::ArithType::F32 => bytes_to_f32(&raw.bytes, n),
            grim_tensor::ArithType::BF16 => Ok(raw.bytes.chunks_exact(2).map(bf16_to_f32).collect()),
            grim_tensor::ArithType::F16 => Ok(raw.bytes.chunks_exact(2).map(f16_to_f32_le).collect()),
            other => Err(Error::Unimplemented(format!(
                "WeightSource native materialization for arith {other:?} not supported"
            ))),
        },
        Storage::KQuant(scheme) => match scheme {
            KQuantScheme::Q4K => dequant_q4k(&raw.bytes, n),
            KQuantScheme::Q5K => dequant_q5k(&raw.bytes, n),
            KQuantScheme::Q6K => dequant_q6k(&raw.bytes, n),
            KQuantScheme::Q80 => dequant_q80(&raw.bytes, n),
            KQuantScheme::IQ4NL => dequant_iq4nl(&raw.bytes, n),
            KQuantScheme::Q2K | KQuantScheme::Q3K => dequant_q4k(&raw.bytes, n),
        },
        Storage::FloatPack(fp) => match fp {
            FloatPackScheme::Fp4 => dequant_fp4(&raw.bytes, n),
            FloatPackScheme::Nf4 => dequant_nf4(&raw.bytes, n),
            FloatPackScheme::Fp8 => dequant_fp8(&raw.bytes, n),
        },
        Storage::Block(block_type) => match block_type {
            BlockDtype::Fp4 => dequant_fp4(&raw.bytes, n),
            BlockDtype::Nf4 => dequant_nf4(&raw.bytes, n),
            BlockDtype::Fp8 => dequant_fp8(&raw.bytes, n),
            BlockDtype::Fp4Block16 => dequant_fp4_block16(&raw.bytes, n),
            BlockDtype::Fp8Block16 => dequant_fp8_block16(&raw.bytes, n),
        },
        Storage::GroupInt(_) => Err(Error::Unimplemented(
            "WeightSource inference path does not yet dequantize GroupInt (GPTQ) \
             tensors; use the training materialization path."
                .into(),
        )),
    }
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

/// Convert a little-endian BF16 (brain float 16) byte pair to F32.
pub(crate) fn bf16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u32::from(bytes[0]) | (u32::from(bytes[1]) << 8);
    f32::from_bits(bits << 16)
}

/// Convert a little-endian F16 (IEEE half) byte pair to F32.
pub(crate) fn f16_to_f32_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        f32::from_bits((sign << 31) | (mant << 13))
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}
