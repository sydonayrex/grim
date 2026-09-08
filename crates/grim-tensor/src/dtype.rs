//! Tensor metadata: device target and arithmetic/storage dtype configuration.

use std::fmt;

/// Hardware compute target: ROCm (primary), Vulkan (portable), CPU (reference), or CUDA/Metal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    /// ROCm primary GPU target — hip/rocBLAS-backed device ordinal.
    Rocm(usize),
    /// Vulkan, platform-agnostic compute.
    Vulkan,
    /// Optional CUDA target.
    Cuda(usize),
    /// Optional Metal target.
    Metal(usize),
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
    pub fn ordinal(&self) -> Option<usize> {
        match self {
            Device::Cpu => None,
            Device::Rocm(o) | Device::Cuda(o) | Device::Metal(o) => Some(*o),
            Device::Vulkan => None,
        }
    }
    pub fn same_kind(&self, other: &Device) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
    /// True when this target is a ROCm device.
    pub fn is_rocm(&self) -> bool {
        matches!(self, Device::Rocm(_))
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Rocm(o) => write!(f, "rocm:{o}"),
            Device::Vulkan => write!(f, "vulkan"),
            Device::Cuda(o) => write!(f, "cuda:{o}"),
            Device::Metal(o) => write!(f, "metal:{o}"),
        }
    }
}

/// The arithmetic type used for computation (what the hardware computes in).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithType {
    F32,
    F16,
    BF16,
    I64,
    U32,
    U8,
}

impl ArithType {
    pub fn is_float(&self) -> bool {
        matches!(self, ArithType::F32 | ArithType::F16 | ArithType::BF16)
    }
    pub fn is_integer(&self) -> bool {
        matches!(self, ArithType::I64 | ArithType::U32 | ArithType::U8)
    }
    pub fn byte_size(self) -> usize {
        match self {
            ArithType::F32 | ArithType::U32 => 4,
            ArithType::F16 | ArithType::BF16 => 2,
            ArithType::U8 => 1,
            ArithType::I64 => 8,
        }
    }
}

/// Physical storage encoding, separating on-disk/in-VRAM representations from arithmetic compute.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Storage {
    /// Stored in native encoding — no dequant needed.
    Native,
    /// Block-quantized K-quant format (Grim's own PTQ, llama.cpp-compatible).
    KQuant(KQuantScheme),
    /// Grouped INT weights from an external QAT pipeline (EfficientQAT, GPTQ).
    GroupInt(GpuIntConfig),
    /// Low-bit floating-point pack formats (FP4, NF4, FP8) dequantized in-kernel.
    FloatPack(FloatPackScheme),
    /// Block-quantized formats mapping FP4/NF4/FP8.
    Block(BlockDtype),
    /// Variable-bitwidth packed codes with column scale and optional outliers/residuals.
    ResidualPacked(ResidualPackedConfig),
    /// W8A8MXFP8: MXFP8 weights and activations with per-block E8M0 shared exponents.
    W8A8Mxfp8,
    /// CompressedTensors W8A8 INT8 with per-channel weight scales and per-token activation scales.
    CompressedTensorsW8A8Int8,
    /// CompressedTensors W8A8 FP8 (OCP E4M3) with per-tensor or per-block scales.
    CompressedTensorsW8A8Fp8,
    /// Marlin-style W4A16: 4-bit packed weights with per-group f32 scales.
    W4A16(W4A16Config),
    /// WNA16: weight-only N-bit quantization with per-block f16 and per-tensor f32 scales.
    WNA16,
    /// EmbeddingWNA16Int: embedding weights stored as row-major N-bit integers.
    EmbeddingWNA16Int,
    /// AWQ: Activation-aware Weight Quantization format with column-packed codes and zero-points.
    Awq(AwqStorageConfig),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockDtype {
    Fp4,
    Nf4,
    Fp8,
    Fp4Block16,
    Fp8Block16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KQuantScheme {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q80,
    /// IQ4_NL — importance-matrix-optimized 4-bit (llama.cpp `IQ4_NL`).
    IQ4NL,
    IQ4XS,
    IQ3XXS,
    IQ3S,
    IQ2XXS,
    IQ2XS,
    IQ2S,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatPackScheme {
    /// FP4 (E2M1 4-bit float).
    Fp4,
    /// NF4 (normalized float-4, Quanto/Unsloth-style).
    Nf4,
    /// FP8 (E4M3 by default; E5M2 recognized).
    Fp8,
    /// MXFP4: 4-bit float with shared E8M0 scale per 32 elements.
    /// Packed as length-prefixed codes and exponents.
    MxFp4,
    /// MXFP8: 8-bit float with shared E8M0 scale per 32 elements.
    /// Packed as length-prefixed codes and exponents.
    MxFp8,
    /// NVFP4: NVIDIA 4-bit float (E2M1) with interleaved scales per 16-element sub-block.
    NvFp4,
}

/// Target quantization format for device-side `quantize` path.
/// Selected variants have kernel acceleration; others fall back to CPU or error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Fp4,
    Nf4,
    Fp8,
    Fp4Block16,
    Fp8Block16,
    Iq4Nl,
    Iq4Xs,
    Iq3Xxs,
    Iq3S,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupQuantScheme {
    Symmetric,
    Asymmetric,
}

/// Configuration for GroupInt quantization (e.g. GPTQ or EfficientQAT).
/// Carries four length-prefixed parallel arrays (qweight, qzeros, scales, g_idx).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuIntConfig {
    pub bits: u8,
    pub group_size: usize,
    pub scheme: GroupQuantScheme,
    /// `false` for EfficientQAT (sequential `g_idx`), `true` for classic GPTQ
    /// with activation ordering.
    pub desc_act: bool,
}

/// Bitwidth configuration for `Storage::W4A16` (Marlin-style 4-bit weights).
/// Packed as contiguous 4-bit codes followed by group f32 scales without prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct W4A16Config {
    /// Number of input features per group (`k % group_size == 0`).
    pub group_size: usize,
}

/// Bitwidth and grouping configuration for `Storage::Awq`.
/// Packed into 3 length-prefixed segments: qweight, qzeros, and f16 scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AwqStorageConfig {
    pub bits: u8,
    pub group_size: usize,
}

/// Bitwidth configuration for `Storage::ResidualPacked` column-major stream.
/// Supports 256-byte aligned row strides with optional backup layers and outliers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidualPackedConfig {
    /// Bitwidth of the packed codes in `RawTensor.bytes`.
    pub bpw: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DType {
    pub arith: ArithType,
    pub storage: Storage,
}

impl DType {
    pub const F32: DType = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    pub const BF16: DType = DType {
        arith: ArithType::BF16,
        storage: Storage::Native,
    };
    pub const F16: DType = DType {
        arith: ArithType::F16,
        storage: Storage::Native,
    };
    pub const U8: DType = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    pub const U32: DType = DType {
        arith: ArithType::U32,
        storage: Storage::Native,
    };

    pub fn is_quantized(&self) -> bool {
        !matches!(self.storage, Storage::Native)
    }

    /// Calculate expected byte size for a given element count under this DType.
    /// Exact for fixed-stride formats; upper-bound for variable-metadata formats.
    pub fn expected_bytes(&self, elem_count: usize) -> usize {
        match &self.storage {
            Storage::Native => elem_count * self.arith.byte_size(),
            Storage::W4A16(cfg) => {
                // 4-bit codes, 2 per byte, plus one f32 scale per
                // (element, group): [codes][scales], no prefixes.
                let group = cfg.group_size.max(1);
                elem_count.div_ceil(2) + (elem_count.div_ceil(group)) * 4
            }
            Storage::KQuant(k) => match k {
                KQuantScheme::Q80 => (elem_count.div_ceil(32)) * 34,
                KQuantScheme::Q4K => (elem_count.div_ceil(256)) * 144,
                KQuantScheme::Q5K => (elem_count.div_ceil(256)) * 176,
                KQuantScheme::Q6K => (elem_count.div_ceil(256)) * 210,
                KQuantScheme::Q2K => (elem_count.div_ceil(256)) * 84,
                KQuantScheme::Q3K => (elem_count.div_ceil(256)) * 110,
                KQuantScheme::IQ4NL | KQuantScheme::IQ4XS => (elem_count.div_ceil(32)) * 18,
                KQuantScheme::IQ3XXS => (elem_count.div_ceil(256)) * 98,
                KQuantScheme::IQ3S => (elem_count.div_ceil(256)) * 110,
                KQuantScheme::IQ2XXS => (elem_count.div_ceil(256)) * 66,
                KQuantScheme::IQ2XS => (elem_count.div_ceil(256)) * 74,
                KQuantScheme::IQ2S => (elem_count.div_ceil(256)) * 82,
            },
            Storage::FloatPack(f) => match f {
                FloatPackScheme::Fp4 | FloatPackScheme::Nf4 => elem_count.div_ceil(2),
                FloatPackScheme::Fp8 => elem_count,
                FloatPackScheme::MxFp4 => elem_count.div_ceil(2) + (elem_count.div_ceil(32)),
                FloatPackScheme::MxFp8 => elem_count + (elem_count.div_ceil(32)),
                // NVFP4: 1 byte E8M0 scale per 16-elem sub-block + 0.5 byte per weight.
                FloatPackScheme::NvFp4 => elem_count.div_ceil(2) + elem_count.div_ceil(16),
            },
            Storage::Block(b) => match b {
                BlockDtype::Fp4 | BlockDtype::Nf4 => elem_count.div_ceil(2),
                BlockDtype::Fp8 => elem_count,
                BlockDtype::Fp4Block16 => elem_count.div_ceil(2) + (elem_count.div_ceil(16)) * 2,
                BlockDtype::Fp8Block16 => elem_count + (elem_count.div_ceil(16)) * 2,
            },
            Storage::ResidualPacked(cfg) => (elem_count * (cfg.bpw as usize)).div_ceil(8),
            _ => elem_count * self.arith.byte_size(),
        }
    }
}

impl From<QuantFormat> for Storage {
    fn from(qf: QuantFormat) -> Self {
        match qf {
            QuantFormat::Q8_0 => Storage::KQuant(KQuantScheme::Q80),
            QuantFormat::Q4K => Storage::KQuant(KQuantScheme::Q4K),
            QuantFormat::Q5K => Storage::KQuant(KQuantScheme::Q5K),
            QuantFormat::Q6K => Storage::KQuant(KQuantScheme::Q6K),
            QuantFormat::Fp4 => Storage::FloatPack(FloatPackScheme::Fp4),
            QuantFormat::Nf4 => Storage::FloatPack(FloatPackScheme::Nf4),
            QuantFormat::Fp8 => Storage::FloatPack(FloatPackScheme::Fp8),
            QuantFormat::Fp4Block16 => Storage::Block(BlockDtype::Fp4Block16),
            QuantFormat::Fp8Block16 => Storage::Block(BlockDtype::Fp8Block16),
            QuantFormat::Iq4Nl => Storage::KQuant(KQuantScheme::IQ4NL),
            QuantFormat::Iq4Xs => Storage::KQuant(KQuantScheme::IQ4XS),
            QuantFormat::Iq3Xxs => Storage::KQuant(KQuantScheme::IQ3XXS),
            QuantFormat::Iq3S => Storage::KQuant(KQuantScheme::IQ3S),
            QuantFormat::Iq2Xxs => Storage::KQuant(KQuantScheme::IQ2XXS),
            QuantFormat::Iq2Xs => Storage::KQuant(KQuantScheme::IQ2XS),
            QuantFormat::Iq2S => Storage::KQuant(KQuantScheme::IQ2S),
        }
    }
}

impl TryFrom<&Storage> for QuantFormat {
    type Error = ();

    fn try_from(s: &Storage) -> std::result::Result<Self, Self::Error> {
        match s {
            Storage::KQuant(k) => match k {
                KQuantScheme::Q80 => Ok(QuantFormat::Q8_0),
                KQuantScheme::Q4K => Ok(QuantFormat::Q4K),
                KQuantScheme::Q5K => Ok(QuantFormat::Q5K),
                KQuantScheme::Q6K => Ok(QuantFormat::Q6K),
                KQuantScheme::IQ4NL => Ok(QuantFormat::Iq4Nl),
                KQuantScheme::IQ4XS => Ok(QuantFormat::Iq4Xs),
                KQuantScheme::IQ3XXS => Ok(QuantFormat::Iq3Xxs),
                KQuantScheme::IQ3S => Ok(QuantFormat::Iq3S),
                KQuantScheme::IQ2XXS => Ok(QuantFormat::Iq2Xxs),
                KQuantScheme::IQ2XS => Ok(QuantFormat::Iq2Xs),
                KQuantScheme::IQ2S => Ok(QuantFormat::Iq2S),
                _ => Err(()),
            },
            Storage::FloatPack(f) => match f {
                FloatPackScheme::Fp4 => Ok(QuantFormat::Fp4),
                FloatPackScheme::Nf4 => Ok(QuantFormat::Nf4),
                FloatPackScheme::Fp8 => Ok(QuantFormat::Fp8),
                _ => Err(()),
            },
            Storage::Block(b) => match b {
                BlockDtype::Fp4Block16 => Ok(QuantFormat::Fp4Block16),
                BlockDtype::Fp8Block16 => Ok(QuantFormat::Fp8Block16),
                BlockDtype::Fp4 => Ok(QuantFormat::Fp4),
                BlockDtype::Nf4 => Ok(QuantFormat::Nf4),
                BlockDtype::Fp8 => Ok(QuantFormat::Fp8),
            },
            _ => Err(()),
        }
    }
}

/// Per-tensor quantization provenance carrying format origins and dequant parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum QuantProvenance {
    /// Not quantized, or produced by grim-quant's own post-training pass.
    #[default]
    GrimNative,
    /// Produced by an external QAT pipeline. Never re-quantized by grim-quant.
    ExternalQat {
        bits: u8,
        group_size: usize,
        scheme: GroupQuantScheme,
        desc_act: bool,
    },
    /// Quantized tensor with outlier overrides or residual backup layers (backup1 / backup2).
    WithResiduals {
        outlier_count: usize,
        outlier_indices_offset: usize,
        outlier_values_offset: usize,
        /// Host-decoded outlier indices and values; empty when offsets must be read from payload.
        outlier_indices: Vec<u32>,
        outlier_values_bits: Vec<u32>,
        primary_scale_offset: usize,
        primary_scale_size: usize,
        primary_row_scale_dtype: u8,
        primary_scale_bytes: Vec<u8>,
        backup1_bpw: u8,
        backup1_codes_offset: usize,
        backup1_scale_offset: usize,
        backup2_bpw: u8,
        backup2_codes_offset: usize,
        backup2_scale_offset: usize,
    },
}

impl QuantProvenance {
    pub fn is_external_qat(&self) -> bool {
        matches!(self, QuantProvenance::ExternalQat { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_properties() {
        let cpu = Device::Cpu;
        let rocm = Device::Rocm(0);
        let cuda = Device::Cuda(1);
        let metal = Device::Metal(2);
        let vulkan = Device::Vulkan;

        assert!(cpu.is_cpu());
        assert!(!rocm.is_cpu());
        assert_eq!(cpu.ordinal(), None);
        assert_eq!(rocm.ordinal(), Some(0));
        assert_eq!(cuda.ordinal(), Some(1));
        assert_eq!(metal.ordinal(), Some(2));
        assert_eq!(vulkan.ordinal(), None);

        assert!(rocm.same_kind(&Device::Rocm(9)));
        assert!(!rocm.same_kind(&cuda));
        assert_eq!(format!("{rocm}"), "rocm:0");
        assert_eq!(format!("{cpu}"), "cpu");
    }

    #[test]
    fn test_arith_type_properties() {
        assert!(ArithType::F32.is_float());
        assert!(ArithType::F16.is_float());
        assert!(ArithType::BF16.is_float());
        assert!(!ArithType::U8.is_float());

        assert!(ArithType::I64.is_integer());
        assert!(ArithType::U32.is_integer());
        assert!(ArithType::U8.is_integer());
        assert!(!ArithType::F32.is_integer());

        assert_eq!(ArithType::F32.byte_size(), 4);
        assert_eq!(ArithType::U32.byte_size(), 4);
        assert_eq!(ArithType::F16.byte_size(), 2);
        assert_eq!(ArithType::BF16.byte_size(), 2);
        assert_eq!(ArithType::U8.byte_size(), 1);
        assert_eq!(ArithType::I64.byte_size(), 8);
    }

    #[test]
    fn test_dtype_is_quantized() {
        assert!(!DType::F32.is_quantized());
        assert!(!DType::F16.is_quantized());
        assert!(!DType::BF16.is_quantized());

        let q4k = DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q4K),
        };
        assert!(q4k.is_quantized());
    }

    #[test]
    fn test_expected_bytes_golden() {
        // llama.cpp Q8_0: 32 values + f16 scale per block.
        assert_eq!(
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::Q80)
            }
            .expected_bytes(32),
            34
        );
        // Q4_K: 256 values → 144 bytes.
        assert_eq!(
            DType {
                arith: ArithType::F32,
                storage: Storage::KQuant(KQuantScheme::Q4K)
            }
            .expected_bytes(256),
            144
        );
        // FP8: 1 byte per element.
        assert_eq!(
            DType {
                arith: ArithType::F32,
                storage: Storage::FloatPack(FloatPackScheme::Fp8)
            }
            .expected_bytes(16),
            16
        );
        // MXFP4: nibble codes + one E8M0 exponent per 32-group.
        assert_eq!(
            DType {
                arith: ArithType::F32,
                storage: Storage::FloatPack(FloatPackScheme::MxFp4)
            }
            .expected_bytes(32),
            17
        );
        // W4A16: ceil(n/2) code bytes + n/group f32 scales, no prefixes.
        assert_eq!(
            DType {
                arith: ArithType::F32,
                storage: Storage::W4A16(W4A16Config { group_size: 128 }),
            }
            .expected_bytes(512),
            256 + 4 * 4
        );
        assert_eq!(DType::F32.expected_bytes(10), 40);
    }

    #[test]
    fn test_quant_provenance_default_and_variants() {
        let def = QuantProvenance::default();
        assert_eq!(def, QuantProvenance::GrimNative);
        assert!(!def.is_external_qat());

        let qat = QuantProvenance::ExternalQat {
            bits: 4,
            group_size: 128,
            scheme: GroupQuantScheme::Asymmetric,
            desc_act: false,
        };
        assert!(qat.is_external_qat());
    }
}
