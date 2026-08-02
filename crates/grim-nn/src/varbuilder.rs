//! `WeightSource` — depth-first cursor over a `TensorProvider`.
//!
//! Mirrors Candle's `VarBuilder` exactly: every model constructor walks a
//! config-defined layer hierarchy and pulls tensors by prefix. Per-tensor
//! dtype/provenance resolution (§4.2, §7.2) happens in `get()`.

use std::sync::Arc;

use grim_tensor::dtype::{
    BlockDtype, DType, Device, FloatPackScheme, KQuantScheme, QuantProvenance, Storage,
};
use grim_tensor::error::{Error, Result};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;
use grim_tensor::{BackendDevice, RawTensor};

use grim_backend_cpu::cpu_tensor;
use grim_quant::{
    dequant_fp4, dequant_fp4_block16, dequant_fp8, dequant_fp8_block16, dequant_iq2s,
    dequant_iq2xs, dequant_iq2xxs, dequant_iq3s, dequant_iq3xxs, dequant_iq4nl, dequant_iq4xs,
    dequant_mxfp4, dequant_mxfp8, dequant_nf4, dequant_q2k, dequant_q3k, dequant_q4k, dequant_q5k,
    dequant_q6k, dequant_q80,
};

#[cfg(feature = "cuda-mem")]
use grim_backend_cuda::CudaDevice;
#[cfg(feature = "metal-mem")]
use grim_backend_metal::MetalDevice;
#[cfg(feature = "rocm-mem")]
use grim_backend_rocm::{RocmDevice, RocmStorage};
#[cfg(feature = "vulkan-mem")]
use grim_backend_vulkan::VulkanDevice;

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
    pub fn root(tensors: &'a dyn grim_tensor::TensorProvider, device: Device) -> Self {
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

    /// Returns the target device for this WeightSource (CPU or GPU).
    pub fn device(&self) -> Device {
        self.device.clone()
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
        let raw = self.tensors.get_packed(&name)?;

        if raw.shape != shape.dims() {
            eprintln!(
                "[get-trace] ShapeMismatch at name={} expected={:?} got={:?}",
                name,
                shape.dims(),
                raw.shape
            );
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

    /// Materialize a tensor without enforcing an expected shape up-front.
    ///
    /// This contract enables callers to perform dynamic shape inspection and
    /// layout normalization (for instance, [`Embedding::load`] handling
    /// transposed weights or token dimension padding in model checkpoints).
    pub fn get_unconstrained(&self, leaf: &str) -> Result<Tensor> {
        let name = self.full_name(leaf);
        let raw = self.tensors.get_packed(&name)?;
        let shape = Shape::new(raw.shape.clone());
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
        let raw = self.tensors.get_packed(&name)?;

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
    let dev = CudaDevice::new(ordinal)?;
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
fn rocm_managed_weight_mode(ordinal: usize, bytes: usize) -> bool {
    match std::env::var("GRIM_ROCM_MANAGED_WEIGHTS")
        .ok()
        .as_deref()
    {
        Some("1") | Some("true") | Some("always") => true,
        Some("auto") => {
            let (free, total) = grim_backend_rocm::vram_info(ordinal);
            let budget = std::env::var("GRIM_ROCM_VRAM_BUDGET_BYTES")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or_else(|| total.saturating_mul(9) / 10);
            free < bytes as u64 || total.saturating_sub(free) > budget
        }
        _ => false,
    }
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
    // Opt-in unified-memory residency for large model weights. HIP kernels
    // can dereference this storage normally; the runtime migrates pages
    // between VRAM and system RAM. Keep the default on ordinary VRAM until
    // a global budget policy is selected by the caller.
    let managed = rocm_managed_weight_mode(ordinal, f32s.len() * std::mem::size_of::<f32>());
    let storage = if managed {
        dev.from_cpu_managed(&f32s, &shape, DType::F32)?
    } else {
        BackendDevice::from_cpu(&dev, &f32s, &shape, DType::F32)?
    };
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

#[cfg(feature = "metal-mem")]
fn materialize_metal(
    f32s: Vec<f32>,
    shape: Shape,
    _dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    let dev = MetalDevice::try_new(ordinal)?;
    let storage = BackendDevice::from_cpu(&dev, &f32s, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        provenance,
        device.clone(),
    ))
}

#[cfg(not(feature = "metal-mem"))]
fn materialize_metal(
    _f32s: Vec<f32>,
    _shape: Shape,
    _dtype: DType,
    _provenance: QuantProvenance,
    _device: &Device,
    ordinal: usize,
) -> Result<Tensor> {
    Err(Error::Unimplemented(format!(
        "Metal materialization: enable 'metal-mem' feature on grim-nn (ordinal={})",
        ordinal
    )))
}

#[cfg(feature = "vulkan-mem")]
fn materialize_vulkan(
    f32s: Vec<f32>,
    shape: Shape,
    _dtype: DType,
    provenance: QuantProvenance,
    device: &Device,
) -> Result<Tensor> {
    let dev = VulkanDevice::new();
    let storage = BackendDevice::from_cpu(&dev, &f32s, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        provenance,
        device.clone(),
    ))
}

#[cfg(not(feature = "vulkan-mem"))]
fn materialize_vulkan(
    _f32s: Vec<f32>,
    _shape: Shape,
    _dtype: DType,
    _provenance: QuantProvenance,
    _device: &Device,
) -> Result<Tensor> {
    Err(Error::Unimplemented(
        "Vulkan materialization: enable 'vulkan-mem' feature on grim-nn".into(),
    ))
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
    if dtype.is_quantized() {
        // Q8_0 has an on-device dequant kernel on ROCm. All other KQuant
        // formats (Q4K, Q6K, IQ*, etc.) lack on-device dequant kernels, so
        // they must fall through to the CPU dequant path below, which
        // produces real F32 and uploads it via materialize_rocm.
        let is_q80 = matches!(&dtype.storage, Storage::KQuant(KQuantScheme::Q80));
        if matches!(&dtype.storage, Storage::ResidualPacked(_)) {
            if let Device::Rocm(ordinal) = device {
                #[cfg(feature = "rocm-mem")]
                {
                    let dev = RocmDevice::new(*ordinal);
                    let mut storage = dev.from_cpu_bytes(&raw.bytes, &shape, dtype.clone())?;
                    storage.set_provenance(provenance.clone());
                    return Ok(Tensor::new(Arc::from(storage), shape, dtype, provenance, device.clone()));
                }
            }
            return Err(Error::Unimplemented("ResidualPacked inference requires a ROCm device".into()));
        }
        if is_q80 {
            if let Device::Rocm(ordinal) = device {
                #[cfg(feature = "rocm-mem")]
                {
                    let dev = RocmDevice::new(*ordinal);
                    if let Ok(storage) = dev.from_cpu_bytes(&raw.bytes, &shape, dtype.clone()) {
                        let f32_storage = {
                            let roc_storage = storage
                                .as_any()
                                .downcast_ref::<RocmStorage>()
                                .ok_or_else(|| {
                                    Error::Backend(
                                        "materialize: ROCm Q80 storage is not RocmStorage".into(),
                                    )
                                })?;
                            dev.dequantize_q8_0(roc_storage)?
                        };
                        // Free the packed buffer immediately instead of
                        // letting it live until this function returns. On a
                        // 4GB card, holding both the packed (~1.06 B/elem)
                        // and freshly-dequantized F32 (4 B/elem) buffers
                        // alive at once nearly quadruples this tensor's peak
                        // VRAM footprint during load — across every Q8_0
                        // tensor in the model that adds up fast enough to
                        // OOM (`hipMalloc failed: 2`) on small consumer
                        // cards. `roc_storage` only borrowed `storage`, so
                        // dropping the owning `Box` here (rather than at
                        // end-of-scope) is what actually returns the packed
                        // allocation to the caching allocator before the
                        // next tensor loads.
                        drop(storage);
                        return Ok(Tensor::new(
                            Arc::from(f32_storage),
                            shape,
                            DType::F32,
                            provenance,
                            device.clone(),
                        ));
                    }
                }
                #[cfg(not(feature = "rocm-mem"))]
                let _ = ordinal;
            }
        }

        // CUDA: keep KQuant / FloatPack / Block storage resident on-device
        // (raw packed bytes) instead of dequantizing to F32 at load time. This
        // mirrors ROCm's Q8_0 residency and enables the CUDA fused
        // `quantized_matmul_backward_dx` path in `grim-autograd::matmul_backward`.
        // ResidualPacked is excluded (no host dequant exists); GroupInt is left
        // dequantized-to-F32 to preserve its multi-segment loader semantics.
        #[cfg(feature = "cuda-mem")]
        if let Device::Cuda(ordinal) = device {
            if dtype.is_quantized()
                && !matches!(dtype.storage, Storage::ResidualPacked(_))
                && !matches!(dtype.storage, Storage::GroupInt(_))
            {
                let dev = CudaDevice::new(*ordinal)?;
                let storage = dev.from_cpu_bytes(&raw.bytes, &shape, dtype.clone())?;
                return Ok(Tensor::new(
                    Arc::from(storage),
                    shape,
                    dtype,
                    provenance,
                    device.clone(),
                ));
            }
        }
        #[cfg(not(feature = "cuda-mem"))]
        let _ = device;

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
        Device::Vulkan => materialize_vulkan(f32s, shape, dtype, provenance, device),
        Device::Metal(ordinal) => {
            materialize_metal(f32s, shape, dtype, provenance, device, *ordinal)
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
            grim_tensor::ArithType::BF16 => {
                Ok(raw.bytes.chunks_exact(2).map(bf16_to_f32).collect())
            }
            grim_tensor::ArithType::F16 => {
                Ok(raw.bytes.chunks_exact(2).map(f16_to_f32_le).collect())
            }
            other => Err(Error::Unimplemented(format!(
                "WeightSource native materialization for arith {other:?} not supported"
            ))),
        },
        Storage::KQuant(scheme) => match scheme {
            KQuantScheme::Q2K => dequant_q2k(&raw.bytes, n),
            KQuantScheme::Q3K => dequant_q3k(&raw.bytes, n),
            KQuantScheme::Q4K => dequant_q4k(&raw.bytes, n),
            KQuantScheme::Q5K => dequant_q5k(&raw.bytes, n),
            KQuantScheme::Q6K => dequant_q6k(&raw.bytes, n),
            KQuantScheme::Q80 => dequant_q80(&raw.bytes, n),
            KQuantScheme::IQ4NL => dequant_iq4nl(&raw.bytes, n),
            KQuantScheme::IQ4XS => dequant_iq4xs(&raw.bytes, n),
            KQuantScheme::IQ3XXS => dequant_iq3xxs(&raw.bytes, n),
            KQuantScheme::IQ3S => dequant_iq3s(&raw.bytes, n),
            KQuantScheme::IQ2XXS => dequant_iq2xxs(&raw.bytes, n),
            KQuantScheme::IQ2XS => dequant_iq2xs(&raw.bytes, n),
            KQuantScheme::IQ2S => dequant_iq2s(&raw.bytes, n),
        },
        Storage::FloatPack(fp) => match fp {
            FloatPackScheme::Fp4 => dequant_fp4(&raw.bytes, n),
            FloatPackScheme::Nf4 => dequant_nf4(&raw.bytes, n),
            FloatPackScheme::Fp8 => dequant_fp8(&raw.bytes, n),
            FloatPackScheme::MxFp4 => dequant_mxfp4(&raw.bytes, n),
            FloatPackScheme::MxFp8 => dequant_mxfp8(&raw.bytes, n),
        },
        Storage::Block(block_type) => match block_type {
            BlockDtype::Fp4 => dequant_fp4(&raw.bytes, n),
            BlockDtype::Nf4 => dequant_nf4(&raw.bytes, n),
            BlockDtype::Fp8 => dequant_fp8(&raw.bytes, n),
            BlockDtype::Fp4Block16 => dequant_fp4_block16(&raw.bytes, n),
            BlockDtype::Fp8Block16 => dequant_fp8_block16(&raw.bytes, n),
        },
        Storage::ResidualPacked(_) => Err(Error::Unimplemented(
            "dequant_to_f32: ResidualPacked not yet supported".into(),
        )),
        Storage::GroupInt(cfg) => {
            // Unpack the four parallel arrays from raw.bytes
            let mut cursor = 0;
            let read_segment = |bytes: &[u8], cursor: &mut usize| -> Result<Vec<u8>> {
                if *cursor + 8 > bytes.len() {
                    return Err(Error::Backend("Truncated GPTQ packed header".into()));
                }
                let len =
                    u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap()) as usize;
                *cursor += 8;
                if *cursor + len > bytes.len() {
                    return Err(Error::Backend(format!(
                        "Truncated GPTQ packed segment (expected {len} bytes)"
                    )));
                }
                let segment = bytes[*cursor..*cursor + len].to_vec();
                *cursor += len;
                Ok(segment)
            };

            let qweight = read_segment(&raw.bytes, &mut cursor)?;
            let qzeros = read_segment(&raw.bytes, &mut cursor)?;
            let scales = read_segment(&raw.bytes, &mut cursor)?;
            let g_idx = read_segment(&raw.bytes, &mut cursor)?;

            let g_idx_opt = if g_idx.is_empty() {
                None
            } else {
                Some(&g_idx[..])
            };

            grim_quant::dequant_gptq_group_int(
                &qweight,
                &qzeros,
                &scales,
                g_idx_opt,
                &raw.shape,
                cfg.bits as u32,
                cfg.group_size,
            )
        }
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
        // Subnormal or zero. An f16 subnormal encodes `mant * 2^-24`, but
        // `f32::from_bits((sign<<31)|(mant<<13))` instead yields
        // `mant * 2^-136` (≈2^112 too small). Build the correct value; this
        // mirrors the fix in grim-quant::f16_to_f32 so the two decoders agree.
        let value = (mant as f32) * 2f32.powi(-24);
        if sign != 0 { -value } else { value }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyProvider {
        raw: RawTensor,
        dtype: DType,
    }

    impl grim_tensor::TensorProvider for DummyProvider {
        fn get(&self, _name: &str) -> Result<RawTensor> {
            Ok(self.raw.clone())
        }
        fn get_packed(&self, _name: &str) -> Result<RawTensor> {
            Ok(self.raw.clone())
        }
        fn meta(&self, _name: &str) -> Result<grim_tensor::TensorMeta> {
            Ok(grim_tensor::TensorMeta {
                dtype: self.dtype.clone(),
                provenance: QuantProvenance::GrimNative,
                shape: self.raw.shape.clone(),
                fusion_mask: 0,
            })
        }
    }

    #[test]
    fn test_quantized_weight_cpu_dequantizes() {
        let q8_dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q80),
        };
        let dummy = DummyProvider {
            raw: RawTensor {
                bytes: vec![0u8; 64],
                shape: vec![2, 16],
                dtype: q8_dtype.clone(),
                provenance: QuantProvenance::GrimNative,
            },
            dtype: q8_dtype,
        };
        let ws = WeightSource::root(&dummy, Device::Cpu);
        let tensor = ws.get(Shape::new(vec![2, 16]), "weight").unwrap();
        assert_eq!(tensor.device(), &Device::Cpu);
    }

    #[test]
    fn test_weight_source_pp_path_concatenation() {
        let dummy = DummyProvider {
            raw: RawTensor {
                bytes: vec![0u8; 16],
                shape: vec![4],
                dtype: DType::F32,
                provenance: QuantProvenance::GrimNative,
            },
            dtype: DType::F32,
        };
        let ws = WeightSource::root(&dummy, Device::Cpu);
        let scoped = ws.pp("model").pp("layers.0");
        assert_eq!(scoped.full_name("weight"), "model.layers.0.weight");
    }

    // ===== golden dequant_to_f32 tests (see header below) =====

    fn close(got: f32, want: f32, ctx: &str) {
        let abs = (got - want).abs();
        let denom = want.abs().max(1e-7);
        assert!(got.is_finite(), "{ctx}: non-finite {got:?} (want {want:?})");
        assert!(
            abs == 0.0 || (abs / denom) < 1e-5,
            "{ctx}: got {got:?} want {want:?} (abs={abs})"
        );
    }

    fn raw(bytes: Vec<u8>, shape: Vec<usize>, dtype: DType) -> RawTensor {
        RawTensor {
            bytes,
            shape,
            dtype,
            provenance: QuantProvenance::GrimNative,
        }
    }

    #[test]
    fn dequant_to_f32_q80_routes_and_applies_f16_scale() {
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q80),
        };
        let mut bytes = vec![0u8; 34];
        bytes[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
        let codes: [i8; 32] = [
            1, -1, 127, -128, 64, -64, 0, 5, 10, -10, 50, -50, 25, -25, 100, -100, 3, -3, 7, 7, 9,
            9, 12, -12, 33, -33, 11, -11, 2, 4, 8, 16,
        ];
        for (i, &c) in codes.iter().enumerate() {
            bytes[2 + i] = c as u8;
        }
        let r = raw(bytes, vec![32], dtype.clone());
        let out = dequant_to_f32(&r, &dtype).unwrap();
        assert_eq!(out.len(), 32);
        for (i, &c) in codes.iter().enumerate() {
            close(out[i], (c as f32) * 2.0, &format!("q80[{i}]"));
        }
    }

    #[test]
    fn dequant_to_f32_bf16_native_shifts_left_16() {
        let dtype = DType {
            arith: grim_tensor::ArithType::BF16,
            storage: Storage::Native,
        };
        let bytes: Vec<u8> = vec![0x80, 0x3F, 0x00, 0xC0, 0x49, 0x40];
        let r = raw(bytes, vec![3], dtype.clone());
        let out = dequant_to_f32(&r, &dtype).unwrap();
        assert_eq!(out.len(), 3);
        close(out[0], 1.0, "bf16 1.0");
        close(out[1], -2.0, "bf16 -2.0");
        close(out[2], 3.140625, "bf16 ~pi");
    }

    #[test]
    fn dequant_to_f32_f16_native_normalized_and_subnormal() {
        let dtype = DType {
            arith: grim_tensor::ArithType::F16,
            storage: Storage::Native,
        };
        let bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40, 0x48, 0x42, 0x00, 0x02, 0x00, 0xC0];
        let r = raw(bytes, vec![5], dtype.clone());
        let out = dequant_to_f32(&r, &dtype).unwrap();
        assert_eq!(out.len(), 5);
        close(out[0], 1.0, "f16 1.0");
        close(out[1], 2.0, "f16 2.0");
        close(out[2], 3.140625, "f16 ~pi");
        close(out[3], (512.0f32) * 2f32.powi(-24), "f16 subnormal 0x0200");
        close(out[4], -2.0, "f16 -2.0");
    }

    #[test]
    fn bytes_to_f32_rejects_wrong_length() {
        let dtype = DType::F32;
        let r = raw(vec![0u8, 1, 2], vec![1], dtype.clone());
        assert!(
            dequant_to_f32(&r, &dtype).is_err(),
            "non-f32-multiple must error"
        );
        let r2 = raw(vec![0u8, 0, 0x40, 0x40], vec![1], dtype.clone());
        let out = dequant_to_f32(&r2, &dtype).unwrap();
        assert_eq!(out.len(), 1);
        close(out[0], 3.0, "f32 native 3.0");
    }

    #[test]
    fn dequant_to_f32_group_int_unpacks_length_prefixed_segments() {
        use grim_tensor::dtype::{GpuIntConfig, GroupQuantScheme};
        let dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::GroupInt(GpuIntConfig {
                bits: 4,
                group_size: 8,
                scheme: GroupQuantScheme::Asymmetric,
                desc_act: false,
            }),
        };
        let mut qw: u32 = 0;
        for i in 0..8u32 {
            qw |= (i & 0xF) << (i * 4);
        }
        let qweight = qw.to_le_bytes().to_vec();
        let mut qz: u32 = 0;
        for col in 0..8u32 {
            qz |= 7u32 << (col * 4);
        }
        let qzeros = qz.to_le_bytes().to_vec();
        let scales = 0.5f32.to_le_bytes().to_vec();
        let g_idx: Vec<u8> = Vec::new();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(qweight.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&qweight);
        bytes.extend_from_slice(&(qzeros.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&qzeros);
        bytes.extend_from_slice(&(scales.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&scales);
        bytes.extend_from_slice(&(g_idx.len() as u64).to_le_bytes());
        let r = raw(bytes, vec![8, 1], dtype.clone());
        let out = dequant_to_f32(&r, &dtype).expect("group-int dequant");
        assert_eq!(out.len(), 8);
        for i in 0..8 {
            close(out[i], (i as f32 - 8.0) * 0.5, &format!("groupint[{i}]"));
        }
    }
}
