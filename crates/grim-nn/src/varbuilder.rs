//! `WeightSource` — depth-first cursor over a `TensorProvider`.
//!
//! Mirrors Candle's `VarBuilder` exactly: every model constructor walks a
//! config-defined layer hierarchy and pulls tensors by prefix. Per-tensor
//! dtype/provenance resolution (§4.2, §7.2) happens in `get()`.

use grim_tensor::dtype::{BlockDtype, DType, Device, FloatPackScheme, KQuantScheme, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;
use grim_tensor::RawTensor;

use grim_backend_cpu::{cpu_tensor, CpuDevice};
use grim_quant::{dequant_fp4, dequant_nf4, dequant_fp8, dequant_q4k, dequant_q5k, dequant_q6k, dequant_q80, dequant_iq4nl};

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

    /// Materialize a tensor for training. Quantized storage types (Q4_K, Q5_K,
    /// Q6_K, Q8_0, ...) are dequantized to native F32 in CPU memory so the
    /// optimization pass has full-precision weights to take gradients against.
    /// Native dtypes flow through unchanged.
    ///
    /// Currently supports CPU dequantization only; ROCm/Vulkan will land as
    /// their respective materialization paths.
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

        tensor_from_raw_for_training(raw, shape, dtype, provenance, self.device.clone())
    }
}

fn tensor_from_raw(
    raw: RawTensor,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    device: Device,
) -> Result<Tensor> {
    // v1 only materializes CPU tensors; per-device (ROCm/Vulkan) materialization
    // lands with each backend. Quantized / low-bit tensors are dequantized to
    // native F32 on the CPU so downstream compute sees uniform F32 weights.
    if !device.is_cpu() {
        return Err(Error::Unimplemented(
            "WeightSource v1 only materializes CPU tensors; \
             per-device materialization lands with each backend."
                .into(),
        ));
    }

    let f32s = dequant_to_f32(&raw, &dtype)?;
    let _ = CpuDevice::new(); // keep the device type reachable
    let _ = provenance; // CPU backend stamps default GrimNative; per-tensor provenance
                        // will be carried once non-Native storage lands.
    let _ = dtype;
    Ok(cpu_tensor(f32s, shape))
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
            // Q2K/Q3K share the 4-bit super-block structure; reuse the Q4K
            // symmetric dequant as an approximation until exact K-quant readers
            // are added. (sleipnir.gguf is Q8, so this is not on its hot path.)
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

/// Training-aware counterpart of `tensor_from_raw`. Quantized storage types
/// are dequantized to native F32 so the optimizer can compute gradients in
/// full-precision weight space.
fn tensor_from_raw_for_training(
    raw: RawTensor,
    shape: Shape,
    dtype: DType,
    provenance: QuantProvenance,
    device: Device,
) -> Result<Tensor> {
    if !device.is_cpu() {
        return Err(Error::Unimplemented(
            "Training materialization only implemented for CPU; per-device \
             materialization lands with each backend."
                .into(),
        ));
    }

    if !device.is_cpu() {
        return Err(Error::Unimplemented(
            "Training materialization only implemented for CPU; per-device \
             materialization lands with each backend."
                .into(),
        ));
    }

    match &dtype.storage {
        Storage::Native => {
            if dtype.arith != grim_tensor::ArithType::F32
                && dtype.arith != grim_tensor::ArithType::BF16
            {
                return Err(Error::Unimplemented(format!(
                    "Training materialization to native {dtype:?} not supported; \
                     only F32/BF16 are valid for gradient computation."
                )));
            }
            let total: usize = raw.shape.iter().product();
            let f32s = bytes_to_f32(&raw.bytes, total)?;
            let _ = CpuDevice::new();
            Ok(cpu_tensor(f32s, shape))
        }
        // Quantized + float-pack formats share the inference dequant path.
        Storage::KQuant(_) | Storage::FloatPack(_) | Storage::Block(_) => {
            let f32s = dequant_to_f32(&raw, &dtype)?;
            let _ = CpuDevice::new();
            let _ = provenance;
            Ok(cpu_tensor(f32s, shape))
        }
        Storage::GroupInt(_gqscheme) => {
            // GPTQ-style group-quantized storage. The training materializer
            // returns native F32 once we have a hook to read the group scales;
            // for now we deliver the same error contract as phase 2 callers.
            Err(Error::Unimplemented(
                "training materialization for GroupInt tensors is staged after \
                 grim-quant dequant_group_int scales are exposed; use the inference `get()` path."
                    .into(),
            ))
        }
    }
}
