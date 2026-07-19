//! Tensor metadata: device target and arithmetic/storage dtype configuration.

use std::fmt;

/// A hardware compute target. Grim's primary GPU is ROCm; Vulkan is the
/// platform-agnostic fallback; CPU is the always-available reference; CUDA
/// and Metal are optional.
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
}

/// Physical storage encoding. When storage differs from the arithmetic type,
/// dequantization is needed before compute. Splitting dtype into
/// `ArithType` + `Storage` keeps variants bounded — adding a new low-bit
/// format (MXFP4, NVFP4, ...) is one Storage variant, not a new DType that
/// forks dispatch everywhere.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Storage {
    /// Stored in native encoding — no dequant needed.
    Native,
    /// Block-quantized K-quant format (Grim's own PTQ, llama.cpp-compatible).
    KQuant(KQuantScheme),
    /// Grouped INT weights from an external QAT pipeline (EfficientQAT, GPTQ).
    GroupInt(GpuIntConfig),
    /// Low-bit floating-point pack formats (FP4 E2M1, NF4, FP8 E4M3/E5M2).
    /// Dequantized to F32 on load; kept distinct from KQuant so the dequant
    /// kernel selects the correct float-pack layout.
    FloatPack(FloatPackScheme),
    /// Block-quantized formats mapping FP4/NF4/FP8.
    Block(BlockDtype),
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatPackScheme {
    /// FP4 (E2M1 4-bit float).
    Fp4,
    /// NF4 (normalized float-4, Quanto/Unsloth-style).
    Nf4,
    /// FP8 (E4M3 by default; E5M2 recognized).
    Fp8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupQuantScheme {
    Symmetric,
    Asymmetric,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuIntConfig {
    pub bits: u8,
    pub group_size: usize,
    pub scheme: GroupQuantScheme,
    /// `false` for EfficientQAT (sequential `g_idx`), `true` for classic GPTQ
    /// with activation ordering.
    pub desc_act: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DType {
    pub arith: ArithType,
    pub storage: Storage,
}

impl DType {
    pub const F32: DType = DType { arith: ArithType::F32, storage: Storage::Native };
    pub const BF16: DType = DType { arith: ArithType::BF16, storage: Storage::Native };
    pub const F16: DType = DType { arith: ArithType::F16, storage: Storage::Native };

    pub fn is_quantized(&self) -> bool {
        !matches!(self.storage, Storage::Native)
    }
}

/// Per-tensor quantization provenance. Resolved at load time by
/// `WeightSource::get` and carried on every tensor so the dequant kernel
/// selects the correct layout per tensor (preventing re-quantization of
/// already-quantization-aware-trained weights).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuantProvenance {
    /// Not quantized, or produced by grim-quant's own post-training pass.
    GrimNative,
    /// Produced by an external QAT pipeline. Never re-quantized by grim-quant.
    ExternalQat {
        bits: u8,
        group_size: usize,
        scheme: GroupQuantScheme,
        desc_act: bool,
    },
}

impl QuantProvenance {
    pub fn is_external_qat(&self) -> bool {
        matches!(self, QuantProvenance::ExternalQat { .. })
    }
}

impl Default for QuantProvenance {
    fn default() -> Self {
        QuantProvenance::GrimNative
    }
}
