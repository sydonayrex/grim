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
    /// OSTQuant W4A4: 4-bit unsigned packed weights with group scales (bf16) and zeros (u8).
    W4A4OstQuant(OstQuantConfig),
    /// A format Grim can identify and size but has no backend for. Carries a
    /// human-readable reason so the failure names the format and what to do
    /// about it, instead of surfacing as an unknown-tag parse failure or, worse,
    /// as silently-wrong data reinterpreted under a similar scheme.
    Unsupported(UnsupportedFormat),
}

/// Why a recognized quant format has no backend, plus the geometry Grim does
/// know about it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnsupportedFormat {
    /// GGUF display name, e.g. `"PQ2_0"`.
    pub name: &'static str,
    /// Weights per block, when known.
    pub block_size: Option<usize>,
    /// Bytes per block, when known.
    pub bytes_per_block: Option<usize>,
    /// Explanation surfaced to the user.
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockDtype {
    Fp4,
    Nf4,
    Fp8,
    /// Legacy GGUF Q4_0 (type 2): 18-byte blocks of `[f16 scale][32 nibbles]`,
    /// one block per 32 weights, `w = (q - 8) * scale`.
    ///
    /// Its own variant rather than `KQuant(Q4K)`, which is 144-byte
    /// super-blocks: typing Q4_0 as Q4_K made every consumer slice and decode
    /// the payload with the wrong geometry. This is the format GreyCrow repacks.
    Q4_0,
    Fp4Block16,
    Fp8Block16,
    /// E4M3 codes with a 128x128 (row x col) grid of `f32` inverse scales —
    /// the DeepSeek-V2/V3 / Xing4.0 `weight_scale_inv` layout.
    ///
    /// The codes and the scale grid live in ONE buffer (see
    /// `grim_quant::pack_fp8_block128`) because the GPU quantized-matmul path
    /// takes no separate scale argument, so a two-tensor representation would
    /// force a host round-trip to pair them.
    Fp8Block128,
    /// GreyRaven (WS-E): E4M3 codes, 2:4 structured sparse, with per-group
    /// survivor-position metadata.
    ///
    /// **A distinct format from [`Self::Fp8`], not a tuning knob.** The operand
    /// geometry differs (16x32 against 16x16), the weight layout is compacted
    /// survivors plus metadata rather than a dense code plane, and the
    /// arithmetic is sparse (SWMMAC) rather than dense (WMMA). A kernel claiming
    /// both geometries is wrong, so `e0_grey_raven_is_distinct_from_white_raven`
    /// pins the separation.
    ///
    /// Density is **4.75 bpw**: 2 survivors x 8 bits of E4M3 plus 3 metadata bits
    /// per group of 4, i.e. 19 bits per 4 original weights. The plan's "6.0 bpw"
    /// assumed 2 metadata bits, which cannot encode the C(4,2) = 6 survivor
    /// patterns. `expected_bytes` below implements 4.75 and
    /// `e3_effective_bpw_is_4_75` names the number so it cannot drift silently.
    Fp8Sparse24,
    /// ForestRaven: symmetric per-output-row absmax INT8.
    ///
    /// One fp32 scale per row (`max|row| / 127`), codes `round(w/scale)` in
    /// `[-128, 127]`. A distinct format from Q8_0, not a tuning knob: Q8_0
    /// scales per 32-element block with fp16 scales, ForestRaven per output
    /// row with fp32 -- same codes, different scale streams, and a kernel
    /// reading one geometry with the other's scales produces finite,
    /// plausible, wrong weights.
    ///
    /// The framed blob (`[u64 codes_len][codes][u64 scales_len][scales]`)
    /// needs the row count to size the scale stream, which `expected_bytes`
    /// cannot see (it takes only an element count). Like Fp8Block128 it
    /// reports the codes part; the framing plus scales ride on the .grim
    /// entry's explicit `payload_size`, and GGUF-direct refuses this format
    /// because no fixed block geometry expresses row-scaled framing.
    Int8PerChannel,
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
    /// Upstream GGUF `Q2_0` (dtype tag 42): 64 weights per 18-byte block,
    /// 2.25 bpw. Block layout (`ggml-common.h:187`) is `ggml_half d` followed
    /// by `qs[QK2_0/4]`, and the decode is `y = (q - 1) * d` over a 2-bit code,
    /// so the level set is `{-1, 0, +1, +2}`.
    ///
    /// This is a distinct scheme from [`Self::GsqRco3p5`] and must never be
    /// aliased to it: GSQRCO's 2-bit codebook is `{-2, -1, 0, +1}`, i.e.
    /// `y = (q - 2) * d`. The two differ by a one-level shift of `d`, which
    /// is silently wrong rather than visibly broken.
    ///
    /// The Qwen3.8-Flash-Next GSQ-RCO-3.5bit release writes tag 42, and
    /// upstream llama.cpp reads it as this format (see the differential PPL
    /// oracle in `plans/eval/qwen4exp-reference-ppl-2026-09-26.json`).
    Q2_0,
    /// Prism-private GSQRCO at tag 81: same 64-elem/18-byte geometry as
    /// [`Self::Q2_0`] but the GSQRCO codebook `y = (q - 2) * d`.
    GsqRco3p5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatPackScheme {
    /// FP4 (E2M1 4-bit float).
    Fp4,
    /// NF4 (normalized float-4, Quanto/Unsloth-style).
    Nf4,
    /// FP8 (E4M3 by default; E5M2 recognized).
    Fp8,
    /// WhiteRaven-blocked FP8: E4M3 codes in 16x16-blocked order
    /// (`grim_quant::block_fp8_16x16`), a permutation of [`Self::Fp8`] — same
    /// byte count, different arrangement, so the WMMA fragment load is one
    /// contiguous 256B tile instead of 16 K-strided segments. Requires
    /// `n % 16 == 0` and `k % 16 == 0`. A separate variant from [`Self::Fp8`]
    /// so a blocked tensor can never reach the row-major kernel (whose loads
    /// would silently read the wrong elements, not fault).
    Fp8Blocked16,
    /// TreePie (WS-A): 5-bit E2M2 with a separate sign plane, 5.0 bpw.
    ///
    /// 32 values per 5 i32 — four payload words holding `exp|mant` and one sign
    /// plane — so the 5.0 bpw figure holds only at 32-value granularity. The
    /// dequantized value is native FP16, which is what lets the ROCm GEMV decode
    /// in-register and feed `V_DOT2_F32_F16` directly instead of uploading a
    /// dequantized copy. Requires `elem_count % 32 == 0`.
    TreePie,
    /// MXFP4: 4-bit float with shared E8M0 scale per 32 elements.
    /// Packed as length-prefixed codes and exponents.
    MxFp4,
    /// MXFP8: 8-bit float with shared E8M0 scale per 32 elements.
    /// Packed as length-prefixed codes and exponents.
    MxFp8,
    /// NVFP4: NVIDIA Blackwell 4-bit float — E2M1 elements with an **E4M3**
    /// block scale per 16 elements. GGUF type 78.
    ///
    /// Interleaved 9 bytes per 16 values: 1 E4M3 scale byte + 8 code bytes.
    NvFp4,
    /// Nutcracker: grim's internal 4-bit float — E2M1 elements with a per-16
    /// block scale byte reinterpreted as `[exp:6 | sel:2]`, where the low 2 bits
    /// are a special-value selector and the repurposed E2M1 zero code emits
    /// that value. RaZeR (arXiv:2501.04052) adapted to an E8M0-shaped byte.
    ///
    /// **Byte-identical layout to [`Self::NvFp4`], different decode.** The two
    /// schemes must never share a tag: feeding NVFP4 data to the Nutcracker
    /// decoder (or vice versa) fails silently, not loudly.
    NutFp4,
}

/// Target quantization format for device-side `quantize` path.
/// Selected variants have kernel acceleration; others fall back to CPU or error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    Q8_0,
    Q2K,
    Q3K,
    /// TreePie (WS-A): 5-bit E2M2 with a sign plane, 5.0 bpw, 32 values per group.
    TreePie,
    Q4K,
    Q5K,
    Q6K,
    Fp4,
    Nf4,
    Fp8,
    Fp4Block16,
    Fp8Block16,
    /// E4M3 with a 128x128 block scale grid; scales are embedded in the blob.
    Fp8Block128,
    /// GreyRaven 2:4: E4M3 survivors plus per-group position metadata, 4.75 bpw.
    /// A separate variant from [`Self::Fp8`] so a GreyRaven tensor cannot be
    /// mistaken for a dense one on the way to a kernel with the wrong geometry.
    Fp8Sparse24,
    /// WhiteRaven-blocked FP8: E4M3 codes in 16x16-blocked order. Same bytes
    /// as [`Self::Fp8`] permuted; feeds the blocked WMMA kernel.
    Fp8Blocked16,
    /// WhiteCrow: W4A4 OSTQuant u4×u4, group-128, feeds `V_DOT8_I32_IU4`.
    W4A4OstQuant,
    /// ForestRaven: symmetric per-output-row absmax INT8, feeds
    /// `V_DOT4_I32_IU8`. See `BlockDtype::Int8PerChannel` for why this is not
    /// Q8_0 under another name.
    Int8PerChannel,
    Iq4Nl,
    Iq4Xs,
    Iq3Xxs,
    Iq3S,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    /// Upstream GGUF `Q2_0` (dtype tag 42): 64 weights per 18-byte block,
    /// 2.25 bpw, codebook `{-1, 0, +1, +2}` scaled by an fp16 per-block delta.
    ///
    /// Carried by the Qwen3.8-Flash-Next GSQ-RCO-3.5bit release, where it is
    /// the tensor type of 62 expert banks (52.0 B params). Distinct from
    /// [`Self::Q2K`]: Q2_K is a 256-weight super-block with two scales and a
    /// min, so a Q2_K kernel reading Q2_0 bytes would produce finite garbage.
    Q2_0,
    /// GSQ-RCO 3.5-bit (GGUF tag 81): the same 18-byte block geometry as
    /// [`Self::Q2_0`] but the GSQ paper's codebook — `values = [-2, -1, 0, 1]`
    /// (old/repo/GSQ-main/src/quantization/gumbel_quantizer_2bit.py:10), so
    /// code q decodes to `(q - 2) * d`, ONE level of d below Q2_0's
    /// `(q - 1) * d` (llama.cpp ggml-quants.c:439). Same bytes, off-by-one
    /// decode: the two formats MUST NOT share a kernel. Grim's native tag,
    /// written by the oxidizer for .grim output and the `--format` default.
    GsqRco3p5,
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

/// Group configuration for `Storage::W4A4OstQuant`.
/// Packed into 3 length-prefixed segments: qweight (u32), scales (bf16), zeros (u8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OstQuantConfig {
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
                KQuantScheme::Q2_0 | KQuantScheme::GsqRco3p5 => (elem_count.div_ceil(64)) * 18,
            },
            Storage::FloatPack(f) => match f {
                FloatPackScheme::Fp4 | FloatPackScheme::Nf4 => elem_count.div_ceil(2),
                FloatPackScheme::Fp8 => elem_count,
                // Blocked is a permutation of Fp8, not a compression: same count.
                FloatPackScheme::Fp8Blocked16 => elem_count,
                // 5 i32 = 20 bytes per 32 values. Ragged tails are rejected at
                // the boundaries rather than rounded up here, because a partial
                // group would silently overstate the density the format claims.
                FloatPackScheme::TreePie => elem_count.div_ceil(32) * 5 * 4,
                FloatPackScheme::MxFp4 => elem_count.div_ceil(2) + (elem_count.div_ceil(32)),
                FloatPackScheme::MxFp8 => elem_count + (elem_count.div_ceil(32)),
                // NVFP4 and Nutcracker: 1 scale byte per 16-elem sub-block +
                // 0.5 byte per weight. Same size either way — which is exactly
                // why the old E8M0-for-E4M3 scale bug was silent.
                FloatPackScheme::NvFp4 | FloatPackScheme::NutFp4 => {
                    elem_count.div_ceil(2) + elem_count.div_ceil(16)
                }
            },
            Storage::Block(b) => match b {
                BlockDtype::Fp4 | BlockDtype::Nf4 => elem_count.div_ceil(2),
                BlockDtype::Fp8 => elem_count,
                // 18 B per 32 weights = 0.5625 B/elem, the same density as
                // IQ4_NL but a different block (f16 scale, bias 8).
                BlockDtype::Q4_0 => elem_count.div_ceil(32) * 18,
                BlockDtype::Fp4Block16 => elem_count.div_ceil(2) + (elem_count.div_ceil(16)) * 2,
                BlockDtype::Fp8Block16 => elem_count + (elem_count.div_ceil(16)) * 2,
                // codes + 128x128 scale grid; the grid extent is in the blob
                // header, so the caller must supply the real byte length.
                BlockDtype::Fp8Block128 => elem_count,
                // GreyRaven 2:4: half the weights survive as one E4M3 byte each,
                // plus 3 metadata bits per group of 4 originals:
                //   survivors = groups * 2 bytes, metadata = ceil(groups*3/8)
                // = 19/4 = 4.75 bits per original weight.
                BlockDtype::Fp8Sparse24 => {
                    let groups = elem_count.div_ceil(4);
                    groups * 2 + (groups * 3).div_ceil(8)
                }
                // ForestRaven: the codes part only. The framed blob also
                // carries 16 framing bytes plus 4 scale bytes per output row,
                // and the row count is not recoverable from an element count
                // alone -- so like Fp8Block128, callers with the shape must
                // use the real length (the .grim entry's explicit
                // `payload_size`), never this estimate, to size a read.
                BlockDtype::Int8PerChannel => elem_count,
            },
            Storage::ResidualPacked(cfg) => (elem_count * (cfg.bpw as usize)).div_ceil(8),
            Storage::Unsupported(f) => match (f.block_size, f.bytes_per_block) {
                (Some(bs), Some(bpb)) => (elem_count.div_ceil(bs)) * bpb,
                // Without geometry we cannot claim a size; report the dense
                // f32 size rather than 0, which would read as an empty tensor.
                _ => elem_count * self.arith.byte_size(),
            },
            Storage::W4A4OstQuant(cfg) => {
                let group = cfg.group_size.max(1);
                24 + elem_count.div_ceil(2)
                    + (elem_count.div_ceil(group)) * 2
                    + elem_count.div_ceil(group)
            }
            _ => elem_count * self.arith.byte_size(),
        }
    }
}

impl From<QuantFormat> for Storage {
    fn from(qf: QuantFormat) -> Self {
        match qf {
            QuantFormat::Q8_0 => Storage::KQuant(KQuantScheme::Q80),
            QuantFormat::Q2_0 => Storage::KQuant(KQuantScheme::Q2_0),
            QuantFormat::Q2K => Storage::KQuant(KQuantScheme::Q2K),
            QuantFormat::Q3K => Storage::KQuant(KQuantScheme::Q3K),
            QuantFormat::Q4K => Storage::KQuant(KQuantScheme::Q4K),
            QuantFormat::Q5K => Storage::KQuant(KQuantScheme::Q5K),
            QuantFormat::Q6K => Storage::KQuant(KQuantScheme::Q6K),
            QuantFormat::Fp4 => Storage::FloatPack(FloatPackScheme::Fp4),
            QuantFormat::Nf4 => Storage::FloatPack(FloatPackScheme::Nf4),
            QuantFormat::Fp8 => Storage::FloatPack(FloatPackScheme::Fp8),
            QuantFormat::TreePie => Storage::FloatPack(FloatPackScheme::TreePie),
            QuantFormat::Fp4Block16 => Storage::Block(BlockDtype::Fp4Block16),
            QuantFormat::Fp8Blocked16 => Storage::FloatPack(FloatPackScheme::Fp8Blocked16),
            QuantFormat::W4A4OstQuant => Storage::W4A4OstQuant(OstQuantConfig { group_size: 128 }),
            QuantFormat::Int8PerChannel => Storage::Block(BlockDtype::Int8PerChannel),
            QuantFormat::Fp8Block16 => Storage::Block(BlockDtype::Fp8Block16),
            QuantFormat::Fp8Block128 => Storage::Block(BlockDtype::Fp8Block128),
            QuantFormat::Fp8Sparse24 => Storage::Block(BlockDtype::Fp8Sparse24),
            QuantFormat::Iq4Nl => Storage::KQuant(KQuantScheme::IQ4NL),
            QuantFormat::Iq4Xs => Storage::KQuant(KQuantScheme::IQ4XS),
            QuantFormat::Iq3Xxs => Storage::KQuant(KQuantScheme::IQ3XXS),
            QuantFormat::Iq3S => Storage::KQuant(KQuantScheme::IQ3S),
            QuantFormat::Iq2Xxs => Storage::KQuant(KQuantScheme::IQ2XXS),
            QuantFormat::Iq2Xs => Storage::KQuant(KQuantScheme::IQ2XS),
            QuantFormat::Iq2S => Storage::KQuant(KQuantScheme::IQ2S),
            QuantFormat::GsqRco3p5 => Storage::KQuant(KQuantScheme::GsqRco3p5),
        }
    }
}

impl TryFrom<&Storage> for QuantFormat {
    type Error = ();

    fn try_from(s: &Storage) -> std::result::Result<Self, Self::Error> {
        match s {
            Storage::KQuant(k) => match k {
                KQuantScheme::Q80 => Ok(QuantFormat::Q8_0),
                KQuantScheme::Q2K => Ok(QuantFormat::Q2K),
                KQuantScheme::Q2_0 => Ok(QuantFormat::Q2_0),
                KQuantScheme::Q3K => Ok(QuantFormat::Q3K),
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
                // NOTE: `GsqRco3p5` deliberately has NO arm here yet, even
                // though `From<QuantFormat> for Storage` maps it and
                // `quantize_gsq_rco_3p5_block` can produce tag 81. The
                // `TryFrom<&Storage>` direction is what `Linear::forward`
                // reads, so adding it without a backend dispatch is worse
                // than leaving it out: the weight would pass the format
                // check, reach `RocmDevice::quantized_matmul`, miss the
                // `KQuantScheme::GsqRco3p5` arm, and land in the generic
                // `_ =>` arm that calls `self.matmul` on 18-byte-per-64
                // packed bytes. That trades a loud "no QuantFormat mapping"
                // error for silent garbage.
                //
                // Add this arm in the same change that adds the backend
                // dispatch, not before.
                // GSQ-RCO 3.5-bit: mapped now that BOTH backends name the
                // scheme — CPU `quantized_matmul` decodes it (device.rs
                // QuantFormat::GsqRco3p5 arm), and the ROCm
                // `quantized_matmul` arm REFUSES LOUDLY naming the missing
                // CityCrow sudot8 kernel. The refusal is the point: the
                // generic `_ =>` matmul catch-all would run on 18-byte-per-64
                // packed bytes and produce silent garbage, which is exactly
                // what this TryFrom arm previously refused to enable.
                KQuantScheme::GsqRco3p5 => Ok(QuantFormat::GsqRco3p5),

            },
            Storage::FloatPack(f) => match f {
                FloatPackScheme::Fp4 => Ok(QuantFormat::Fp4),
                FloatPackScheme::Nf4 => Ok(QuantFormat::Nf4),
                FloatPackScheme::Fp8 => Ok(QuantFormat::Fp8),
                FloatPackScheme::TreePie => Ok(QuantFormat::TreePie),
                FloatPackScheme::Fp8Blocked16 => Ok(QuantFormat::Fp8Blocked16),
                _ => Err(()),
            },
            Storage::Block(b) => match b {
                BlockDtype::Fp4Block16 => Ok(QuantFormat::Fp4Block16),
                BlockDtype::Fp8Block16 => Ok(QuantFormat::Fp8Block16),
                BlockDtype::Fp8Block128 => Ok(QuantFormat::Fp8Block128),
                BlockDtype::Fp4 => Ok(QuantFormat::Fp4),
                BlockDtype::Nf4 => Ok(QuantFormat::Nf4),
                BlockDtype::Fp8 => Ok(QuantFormat::Fp8),
                BlockDtype::Fp8Sparse24 => Ok(QuantFormat::Fp8Sparse24),
                BlockDtype::Int8PerChannel => Ok(QuantFormat::Int8PerChannel),
                // Legacy Q4_0 shares IQ4_NL's density but not its block, and no
                // canonical QuantFormat names it yet; refuse rather than
                // report a format a decoder would not honour.
                BlockDtype::Q4_0 => Err(()),
            },
            Storage::W4A4OstQuant(_) => Ok(QuantFormat::W4A4OstQuant),
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

    /// Every `KQuantScheme` must round-trip through `QuantFormat`.
    ///
    /// REGRESSION (Qwen3.8-27B Q4_K_M): `KQuantScheme::Q3K` existed in the
    /// enum and in the GGUF tag map, and `dequant_q3k` existed in grim-quant,
    /// but `TryFrom<&Storage> for QuantFormat` had no arm for it. Any Linear
    /// whose weight resolved to Q3_K therefore failed at forward time with
    /// `Unimplemented("Linear::forward: quantized storage KQuant(Q3K) has no
    /// QuantFormat mapping")` — after every weight had loaded successfully.
    ///
    /// The previous coverage was a per-variant hand-written list, so it could
    /// only ever cover the variants someone remembered. This walks the enum, so
    /// adding a K-quant without a mapping fails here instead of at inference.
    #[test]
    fn every_kquant_scheme_maps_to_a_quant_format_and_back() {
        /// K-quants deliberately WITHOUT a `QuantFormat`, with the reason.
        ///
        /// `GsqRco3p5` is here because it has a packer and a `From<QuantFormat>`
        /// mapping but no backend dispatch: adding the `TryFrom` arm alone
        /// would let a GSQRCO weight reach `RocmDevice::quantized_matmul`,
        /// miss its match arm, and fall into the generic `_ =>` arm that calls
        /// `self.matmul` on packed bytes — silent garbage instead of the loud
        /// "no QuantFormat mapping" error. Fail loudly until the dispatch
        /// exists; then remove it here and add the arm together.
        ///
        /// (An earlier note claimed tag 81 had no producer and should be
        /// retired. That was wrong — `quantize_gsq_rco_3p5_block` writes it.)
        const NO_QUANT_FORMAT: &[KQuantScheme] = &[KQuantScheme::GsqRco3p5];

        const ALL: &[KQuantScheme] = &[
            KQuantScheme::Q2K,
            KQuantScheme::Q3K,
            KQuantScheme::Q4K,
            KQuantScheme::Q5K,
            KQuantScheme::Q6K,
            KQuantScheme::Q80,
            KQuantScheme::IQ4NL,
            KQuantScheme::IQ4XS,
            KQuantScheme::IQ3XXS,
            KQuantScheme::IQ3S,
            KQuantScheme::IQ2XXS,
            KQuantScheme::IQ2XS,
            KQuantScheme::IQ2S,
            KQuantScheme::Q2_0,
            KQuantScheme::GsqRco3p5,
        ];
        for &scheme in ALL {
            let storage = Storage::KQuant(scheme);
            let Ok(fmt) = QuantFormat::try_from(&storage) else {
                assert!(
                    NO_QUANT_FORMAT.contains(&scheme),
                    "KQuantScheme::{scheme:?} has no QuantFormat mapping; if that is intended, \
                     add it to NO_QUANT_FORMAT with a reason (Linear::forward will reject it)"
                );
                continue;
            };
            assert_eq!(
                Storage::from(fmt),
                storage,
                "QuantFormat::{fmt:?} does not map back to KQuantScheme::{scheme:?}"
            );
        }
    }
}
