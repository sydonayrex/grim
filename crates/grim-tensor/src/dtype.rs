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
    pub fn byte_size(self) -> usize {
        match self {
            ArithType::F32 | ArithType::U32 => 4,
            ArithType::F16 | ArithType::BF16 => 2,
            ArithType::U8 => 1,
            ArithType::I64 => 8,
        }
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
    /// Kept as packed bytes on-device for residency-capable backends (ROCm /
    /// CUDA / CPU), which dequantize in-kernel; only backends without a packed
    /// residency path dequantize to F32 on load. Kept distinct from KQuant so
    /// the dequant kernel selects the correct float-pack layout.
    FloatPack(FloatPackScheme),
    /// Block-quantized formats mapping FP4/NF4/FP8.
    Block(BlockDtype),
    /// Generic variable-bitwidth packed codes with a per-column uint8 scale,
    /// optional outlier overrides, and optional backup1/backup2 residual
    /// layers — the `.grim` native packed + SpQR-residual layout consumed by
    /// `grim_fused_dequant_gemm_f16` (WI-C / WI-T8). Outlier/backup metadata
    /// rides in `QuantProvenance::WithResiduals`.
    ResidualPacked(ResidualPackedConfig),
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
    /// MXFP4 (OCP Microscaling 4-bit float with shared E8M0 scale per 32 elements - Jay tier).
    ///
    /// ### Packed Byte Layout for `RawTensor.bytes` (concatenation convention):
    /// Two length-prefixed segments, matching the kernel in
    /// `grim-backend-rocm/src/kernels/mxfp_standalone.rs` (codes packed
    /// 2-per-byte, even element in the low nibble; one E8M0 shared exponent
    /// byte per 32-element group):
    ///
    /// ```text
    /// [u64 LE: codes_len] [codes...]
    /// [u64 LE: exps_len]  [exps...]
    /// ```
    MxFp4,
    /// MXFP8 (OCP Microscaling 8-bit float with shared scale per block - Magpie tier).
    ///
    /// ### Packed Byte Layout for `RawTensor.bytes` (concatenation convention):
    /// Two length-prefixed segments, matching the kernel in
    /// `grim-backend-rocm/src/kernels/mxfp_standalone.rs` (one E4M3 code byte
    /// per element; one E8M0 shared exponent byte per 32-element group):
    ///
    /// ```text
    /// [u64 LE: codes_len] [codes...]
    /// [u64 LE: exps_len]  [exps...]
    /// ```
    MxFp8,
}

/// Target quantization format for the device-side `quantize` path.
///
/// Mirrors the CPU `grim_quant::quant_*` reference functions. Selected
/// variants (Q8_0, Fp8) have device-side kernels on CUDA/Metal/ROCm/Vulkan;
/// the rest fall back to `Err(Unimplemented)` when dispatched through
/// `BackendDevice::quantize`.
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

/// Configuration for GroupInt quantization (used by external QAT pipelines such as GPTQ).
///
/// ### Packed Byte Layout for `RawTensor.bytes` (concatenation convention):
/// When `storage` is `Storage::GroupInt(_)`, all four parallel arrays are concatenated
/// into a single `Vec<u8>` in `RawTensor.bytes`. Each array segment is prefixed with its length:
///
/// ```text
/// [u64 LE: qweight_len] [qweight_bytes...]
/// [u64 LE: qzeros_len]  [qzeros_bytes...]
/// [u64 LE: scales_len]  [scales_bytes...]
/// [u64 LE: g_idx_len]   [g_idx_bytes...]
/// ```
///
/// If `g_idx` is absent, its segment has `g_idx_len` set to 0.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuIntConfig {
    pub bits: u8,
    pub group_size: usize,
    pub scheme: GroupQuantScheme,
    /// `false` for EfficientQAT (sequential `g_idx`), `true` for classic GPTQ
    /// with activation ordering.
    pub desc_act: bool,
}

/// Bitwidth configuration for `Storage::ResidualPacked`.
///
/// # Packed residual layout contract
///
/// `ResidualPacked` is a column-major weight view as consumed by
/// `grim_fused_dequant_gemm_f16`: for a logical `[K, N]` weight, the packed
/// stream contains `N` rows, each containing `K` codes.  A code is an unsigned
/// `bpw`-bit integer, written most-significant-bit first within each byte; a
/// code which crosses a byte boundary takes its high bits from the low bits of
/// the first byte and its low bits from the high bits of the next byte.  The
/// code is normalized as `code / (2^bpw - 1) * 2 - 1`.
///
/// Each packed layer has a 256-byte row stride:
///
/// ```text
/// row_bytes(bpw, K) = align_up(ceil(K * bpw / 8), 256)
/// row_start(layer, row) = layer_codes_offset + row * row_bytes(bpw, K)
/// ```
///
/// The primary code region starts at byte offset zero of the `B_codes`
/// allocation.  Its per-output-row scale is supplied separately through the
/// `B_scales` pointer and is one `u8` per output row, decoded as
/// `scale_byte / 255.0`; a null pointer means scale `1.0` for every row.
///
/// `QuantProvenance::WithResiduals` records optional backup regions by byte
/// offsets.  `backup{1,2}_codes_offset` points to another independently
/// 256-byte-row-aligned code region using its recorded bitwidth and the same
/// MSB-first packing rule.  `backup{1,2}_scale_offset` points into the same
/// `B_codes` allocation at an array of one `u8` scale per output row,
/// decoded as `/ 255.0`; an offset of zero means unit scale in the ROCm
/// kernel.  A present backup is added to the primary decoded value.  The
/// forward kernel consumes backup1; the backward kernel consumes backup1 and
/// backup2.
///
/// Outliers are not stored in the packed code stream.  The provenance carries
/// their count and the execution path supplies two external arrays: sorted
/// `u32` flat indices and matching `f32` replacement values.  An index is
/// `row * K + k` in the logical `[N, K]` packed view.  The replacement value
/// takes precedence over the primary and backup reconstruction.
///
/// This documents the consumer-side ABI; it does not imply that a host-side
/// writer or dequantizer is implemented.
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
    /// Quantized tensor with outlier overrides or residual backup layers (backup1 / backup2).
    WithResiduals {
        outlier_count: usize,
        outlier_indices_offset: usize,
        outlier_values_offset: usize,
        /// Host-decoded outlier indices/values, when the provider has already
        /// materialized them. Empty vectors mean the offsets must be decoded
        /// from the packed payload by the backend loader.
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

impl Default for QuantProvenance {
    fn default() -> Self {
        QuantProvenance::GrimNative
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
