//! Low-level GGUF v3 binary reader. Handles the file header, metadata
//! key-value map, tensor info index, and aligned tensor data region.
//!
//! GGUF is the format standardized by llama.cpp:
//!   <header> <metadata_kv*> <tensor_info*> <aligned_tensor_data>
//!
//! This is a minimal, no-unsafe reader — we parse the file into a
//! `GgufFile` struct, then seek to offsets during `TensorProvider::get`.
//!
//! ## `.grim` ROCm Extension
//!
//! `.grim` files extend GGUF v3 with `grim.`-prefixed metadata keys.
//! Any GGUF reader ignores unknown keys silently, so `.grim` is fully
//! backward-compatible. Grim-specific keys are parsed by `read_grim_metadata()`.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use grim_tensor::dtype::DType;
use grim_tensor::dtype::{BlockDtype, KQuantScheme, Storage};
use grim_tensor::error::{Error, Result};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" LE
pub const GGUF_VERSION: u32 = 3;

/// Metadata value type tags from GGUF spec.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Uint64(v) => Some(*v as u32),
            GgufValue::Int32(v) => Some(*v as u32),
            GgufValue::Int64(v) => Some(*v as u32),
            GgufValue::Int8(v) => Some(*v as u32),
            GgufValue::Int16(v) => Some(*v as u32),
            GgufValue::Uint8(v) => Some(*v as u32),
            GgufValue::Uint16(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            GgufValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// GGUF tensor data type tags (§6 of GGUF spec).
/// https://github.com/ggml-org/gguf/blob/master/docs/gguf.md
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_upper_case_globals)]
pub enum GgufDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    Q4_2 = 21,
    Q8_1Hx = 22,
    #[allow(non_camel_case_types)]
    IQ4_NL = 35,
}

impl GgufDType {
    pub fn from_tag(tag: u32) -> Option<Self> {
        match tag {
            0 => Some(GgufDType::F32),
            1 => Some(GgufDType::F16),
            2 => Some(GgufDType::Q4_0),
            3 => Some(GgufDType::Q4_1),
            6 => Some(GgufDType::Q5_0),
            7 => Some(GgufDType::Q5_1),
            8 => Some(GgufDType::Q8_0),
            9 => Some(GgufDType::Q8_1),
            10 => Some(GgufDType::Q2K),
            11 => Some(GgufDType::Q3K),
            12 => Some(GgufDType::Q4K),
            13 => Some(GgufDType::Q5K),
            14 => Some(GgufDType::Q6K),
            15 => Some(GgufDType::Q8K),
            16 => Some(GgufDType::I8),
            17 => Some(GgufDType::I16),
            18 => Some(GgufDType::I32),
            19 => Some(GgufDType::I64),
            20 => Some(GgufDType::F64),
            21 => Some(GgufDType::Q4_2),
            22 => Some(GgufDType::Q8_1Hx),
            35 => Some(GgufDType::IQ4_NL),
            _ => None,
        }
    }

    /// Returns the number of bytes per element for this dtype (used by
    /// downstream materializers to size CPU buffers).
    pub fn elem_size(self) -> u64 {
        match self {
            GgufDType::F32 => 4,
            GgufDType::F16 => 2,
            GgufDType::I8 => 1,
            GgufDType::I16 => 2,
            GgufDType::I32 => 4,
            GgufDType::I64 => 8,
            GgufDType::F64 => 8,
            // Quantized layouts use block storage; their effective per-element cost
            // when dequantized to F32 (or how many bytes a single weight occupies in
            // a codebook) is computed via `type_size_bytes / block_size`. The values
            // 17, 21, 34, etc. were placeholder block-sizes and must NOT be used to
            // compute on-disk byte counts — see `type_size_per_block` / `block_size`.
            _ => 1,
        }
    }

    /// Number of weights stored per quantization block. Quant layouts package
    /// N weights together with shared scales/deltas; this is the N. F32/F16/I*/BF16
    /// are stored as 1-element blocks.
    pub fn block_size(self) -> u64 {
        match self {
            GgufDType::F32 | GgufDType::F16 | GgufDType::F64
            | GgufDType::I8 | GgufDType::I16 | GgufDType::I32 | GgufDType::I64
            | GgufDType::Q4_2
            | GgufDType::Q8_1Hx => 1,
            // K-quants and Q-families: 32 weights per super-block
            _ => 32,
        }
    }

    /// Bytes consumed by ONE quantization block. For F32/F16/I* kinds this is
    /// just `elem_size`. For block-quantized kinds it's the literal block layout
    /// size in the GGUF stream (scales + codebook + packed nibbles).
    ///
    /// Reference: gguf-main/src/lib.rs lines 706-748 — `type_size` per dtype.
    pub fn type_size_per_block(self) -> u64 {
        let bs = self.block_size();
        match self {
            GgufDType::F32 => 4,
            GgufDType::F16 => 2,
            GgufDType::I8 => 1,
            GgufDType::I16 => 2,
            GgufDType::I32 => 4,
            GgufDType::I64 => 8,
            GgufDType::F64 => 8,
            GgufDType::Q4_0 => 2 + bs / 2,
            GgufDType::Q4_1 => 2 + 2 + bs / 2,
            GgufDType::Q4_2 => 0,
            GgufDType::Q5_0 => 2 + 4 + bs / 2,
            GgufDType::Q5_1 => 2 + 2 + 4 + bs / 2,
            GgufDType::Q8_0 => 2 + bs,
            GgufDType::Q8_1 => 4 + 4 + bs,
            GgufDType::Q2K => bs / 16 + bs / 4 + 2 + 2,
            GgufDType::Q3K => bs / 8 + bs / 4 + 12 + 2,
            GgufDType::Q4K => 2 + 2 + 12 + bs / 2,
            GgufDType::Q5K => 2 + 2 + 12 + bs / 8 + bs / 2,
            GgufDType::Q6K => bs / 2 + bs / 4 + bs / 16 + 2,
            GgufDType::Q8K => 4 + bs + bs / 16 * 2,
            GgufDType::IQ4_NL => 2 + 16,
            _ => 0,
        }
    }
}

/// One tensor index entry from a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    /// Offset (in bytes) from the start of the file to the tensor data.
    pub offset: u64,
    /// Size of the tensor data in bytes.
    pub size_bytes: u64,
    /// GGUF tensor data type (includes quantization info).
    pub dtype: GgufDType,
}

impl GgufTensorInfo {
    pub fn shape(&self) -> Vec<usize> {
        let mut s: Vec<usize> = self.dims.iter().map(|d| *d as usize).collect();
        s.reverse();
        s
    }
    pub fn elem_count(&self) -> usize {
        self.shape().iter().product()
    }
}

/// Parsed GGUF file metadata. The raw file bytes are not kept — we store
/// tensor info (name + offset) and metadata KV pairs.
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset where the aligned tensor data section begins.
    pub data_start: u64,
}

// ---------------------------------------------------------------------------
// `.grim` ROCm extension types
// ---------------------------------------------------------------------------

/// AMD GCN architecture identifiers for ROCm profile hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrimRocmlProfile {
    /// CDNA2 compute (MI210, MI250)
    Cdna2,
    /// CDNA3 compute (MI300X)
    Cdna3,
    /// RDNA2 graphics (RX 6000 series, Steam Deck APU — gfx1036)
    Rdna2,
    /// RDNA3 graphics (RX 7900 XTX)
    Rdna3,
    /// RDNA4 graphics (RX 8000 series)
    Rdna4,
    /// All supported architectures (no specialization)
    All,
    /// Unknown — no ROCm-specific optimization hints
    Unknown,
}

impl Default for GrimRocmlProfile {
    fn default() -> Self {
        GrimRocmlProfile::Unknown
    }
}

impl GrimRocmlProfile {
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "cdna2" => GrimRocmlProfile::Cdna2,
            "cdna3" | "mi300x" => GrimRocmlProfile::Cdna3,
            "rdna2" | "gfx1036" => GrimRocmlProfile::Rdna2,
            "rdna3" => GrimRocmlProfile::Rdna3,
            "rdna4" => GrimRocmlProfile::Rdna4,
            "all" => GrimRocmlProfile::All,
            _ => GrimRocmlProfile::Unknown,
        }
    }

    pub fn wavefront_size(&self) -> u32 {
        match self {
            GrimRocmlProfile::Cdna2 => 32,
            GrimRocmlProfile::Cdna3 => 32,
            GrimRocmlProfile::Rdna2 => 64,
            GrimRocmlProfile::Rdna3 => 64,
            GrimRocmlProfile::Rdna4 => 64,
            GrimRocmlProfile::All | GrimRocmlProfile::Unknown => 0,
        }
    }

    pub fn lds_size(&self) -> u32 {
        match self {
            GrimRocmlProfile::Cdna2 => 65536,
            GrimRocmlProfile::Cdna3 => 65536,
            GrimRocmlProfile::Rdna2 => 32768,
            GrimRocmlProfile::Rdna3 => 32768,
            GrimRocmlProfile::Rdna4 => 32768,
            GrimRocmlProfile::All | GrimRocmlProfile::Unknown => 0,
        }
    }
}

/// Per-tensor quantization override decoded from `grim.quant_overrides`.
#[derive(Debug, Clone)]
pub struct GrimQuantOverride {
    /// Tensor name matching an entry in `gguf_tensor_infos`.
    pub tensor_name: String,
    /// Effective bits per weight after non-uniform re-quantization.
    pub effective_bpw: u32,
    /// GGUF dtype tag to use for this tensor (e.g. `Q5_K = 13`).
    pub override_dtype: GgufDType,
    /// Importance score from importance-matrix calibration (higher = more sensitive).
    pub importance_score: f32,
    /// Optional layout hint for ROCm LDS tiling.
    pub layout_hint: Option<GrimLayoutHint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrimLayoutHint {
    /// Reorder weights into wavefront-aligned tiles for LDS efficiency.
    WavefrontTiled,
    /// Enable block sparsity pattern for FFN layers.
    BlockSparse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrimTrainQuantMode {
    Fp4,
    Nf4,
    Fp8,
    Bf16,
}

impl GrimTrainQuantMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fp4" => Some(Self::Fp4),
            "nf4" => Some(Self::Nf4),
            "fp8" => Some(Self::Fp8),
            "bf16" => Some(Self::Bf16),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp4 => "fp4",
            Self::Nf4 => "nf4",
            Self::Fp8 => "fp8",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GrimFusionOp {
    RmsNormMatMul,
    QkvAttention,
}

impl GrimFusionOp {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rmsnorm_matmul" => Some(Self::RmsNormMatMul),
            "qkv_attention" => Some(Self::QkvAttention),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RmsNormMatMul => "rmsnorm_matmul",
            Self::QkvAttention => "qkv_attention",
        }
    }
}

/// `.grim` file metadata — parsed from `grim.*` GGUF metadata keys.
/// Absent keys indicate the file is a plain GGUF (no ROCm hints).
///
/// The on-disk file format stays at version 1 (`GRIM\x01`); the spec's
/// per-tensor capability fields (per-row scales, mixed-bitwidth rows,
/// backup streams, GPTQ-ORDERED, fusion mask, ...) ride this metadata
/// layer instead of changing the registry wire layout. See
/// `GrimTensorExt` in `spec.rs` for the capability surface.
#[derive(Debug, Clone)]
pub struct GrimMetadata {
    /// `"grim-v1"` if this is a `.grim` file; absent otherwise.
    pub magic: Option<String>,
    /// Oxidizer toolchain version that produced this file.
    pub quant_version: Option<u32>,
    /// Target ROCm profile (CDNA2, RDNA3, MI300X, etc.).
    pub rocml_profile: GrimRocmlProfile,
    /// Wavefront size from ROCm profile (0 if unknown).
    pub wavefront_size: u32,
    /// AMD GCN identifier e.g. `"gfx90a"`, `"gfx942"`.
    pub target_gcn: Option<String>,
    /// Recommended thread block size for GEMM kernels.
    pub block_size: Option<u32>,
    /// LDS (Local Data Share) size in bytes for target GPU.
    pub lds_size: Option<u32>,
    /// Whether matrix instruction units are available (WMMA/Tensile).
    pub tensor_core_enabled: bool,
    /// Quantization method used: `"evopress-gptq"`, `"uniform-kquant"`, etc.
    pub quant_method: Option<String>,
    /// Name/path of calibration dataset used for non-uniform quantization.
    pub calibration_dataset: Option<String>,
    /// Per-tensor encoding overrides (non-uniform bitwidth assignment).
    pub quant_overrides: Vec<GrimQuantOverride>,
    /// Preferred quant materialization for training-capable `.grim` artifacts.
    pub train_quant_mode: Option<GrimTrainQuantMode>,
    /// Requested fusion patterns for training or runtime lowering.
    pub train_fusion_ops: Vec<GrimFusionOp>,
    /// ROCm-specialized fusion ops baked into this artifact.
    pub rocm_fusion_ops: Vec<GrimFusionOp>,
    /// Whether XNACK was enabled on the calibration or bake target.
    pub xnack_enabled: Option<bool>,
    /// Whether the KV cache layout was pre-optimized for ROCm decode.
    pub kv_layout_optimized: Option<bool>,
    /// Whether the registry entries contain KV fields.
    pub has_kv_registry: Option<bool>,
    /// Per-tensor capability extensions attached via the JSON metadata layer.
    /// Each entry declares capabilities (per-row scales, mixed bitwidth,
    /// backup streams, GPTQ-ORDERED, fusion mask) without changing the
    /// on-disk registry layout. See `spec.rs` for the descriptor schema.
    pub ext_entries: Vec<crate::spec::GrimTensorExt>,
}

impl Default for GrimMetadata {
    fn default() -> Self {
        GrimMetadata {
            magic: None,
            quant_version: None,
            rocml_profile: GrimRocmlProfile::Unknown,
            wavefront_size: 0,
            target_gcn: None,
            block_size: None,
            lds_size: None,
            tensor_core_enabled: false,
            quant_method: None,
            calibration_dataset: None,
            quant_overrides: Vec::new(),
            train_quant_mode: None,
            train_fusion_ops: Vec::new(),
            rocm_fusion_ops: Vec::new(),
            xnack_enabled: None,
            kv_layout_optimized: None,
            has_kv_registry: None,
            ext_entries: Vec::new(),
        }
    }
}

impl GrimMetadata {
    /// Retrieve per-tensor extension capabilities for a named tensor if present.
    pub fn get_tensor_ext(&self, tensor_name: &str) -> Option<&crate::spec::GrimTensorExt> {
        self.ext_entries.iter().find(|e| e.tensor_name == tensor_name)
    }

    /// Build a `GrimMetadata` by scanning `metadata` for `grim.` keys.
    pub fn from_gguf_metadata(metadata: &HashMap<String, GgufValue>) -> Self {
        let magic = metadata.get("grim.magic").and_then(|v| v.as_str()).map(String::from);
        let quant_version = metadata.get("grim.quant_version").and_then(|v| v.as_u32());
        let rocml_profile_str = metadata.get("grim.rocml.profile").and_then(|v| v.as_str());
        let profile = rocml_profile_str
            .map(GrimRocmlProfile::from_str)
            .unwrap_or_default();
        let wavefront_size = metadata
            .get("grim.rocml.wavefront_size")
            .and_then(|v| v.as_u32())
            .unwrap_or_else(|| profile.wavefront_size());
        let target_gcn = metadata.get("grim.rocml.target_gcn").and_then(|v| v.as_str()).map(String::from);
        let block_size = metadata.get("grim.rocml.block_size").and_then(|v| v.as_u32());
        let lds_size = metadata.get("grim.rocml.lds_size").and_then(|v| v.as_u32());
        let tensor_core_enabled = metadata
            .get("grim.rocml.tensor_core_enabled")
            .and_then(|v| match v {
                GgufValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);
        let quant_method = metadata.get("grim.quant_method").and_then(|v| v.as_str()).map(String::from);
        let calibration_dataset = metadata
            .get("grim.calibration_dataset")
            .and_then(|v| v.as_str())
            .map(String::from);
        let quant_overrides = metadata
            .get("grim.quant_overrides")
            .and_then(|v| read_grim_quant_overrides(v))
            .unwrap_or_default();
        let train_quant_mode = metadata
            .get("grim.train.quant_mode")
            .and_then(|v| v.as_str())
            .and_then(GrimTrainQuantMode::from_str);
        let train_fusion_ops = metadata
            .get("grim.train.fusion_ops")
            .and_then(read_grim_fusion_ops)
            .unwrap_or_default();
        let rocm_fusion_ops = metadata
            .get("grim.rocm.fusion_ops")
            .and_then(read_grim_fusion_ops)
            .unwrap_or_default();
        let xnack_enabled = metadata.get("grim.rocm.xnack_enabled").and_then(read_bool);
        let kv_layout_optimized = metadata
            .get("grim.rocm.kv_layout_optimized")
            .and_then(read_bool);
        let has_kv_registry = metadata
            .get("grim.has_kv_registry")
            .and_then(read_bool);

        GrimMetadata {
            magic,
            quant_version,
            rocml_profile: profile,
            wavefront_size,
            target_gcn,
            block_size,
            lds_size,
            tensor_core_enabled,
            quant_method,
            calibration_dataset,
            quant_overrides,
            train_quant_mode,
            train_fusion_ops,
            rocm_fusion_ops,
            xnack_enabled,
            kv_layout_optimized,
            has_kv_registry,
            ext_entries: Vec::new(),
        }
    }

    /// Returns `true` if this is a `.grim` file (has `grim.magic`).
    pub fn is_grim(&self) -> bool {
        self.magic.is_some()
    }

    /// Look up a per-tensor override by tensor name. Returns `None` if the
    /// tensor has no override (use the GGUF dtype from the tensor info).
    pub fn override_for(&self, tensor_name: &str) -> Option<&GrimQuantOverride> {
        self.quant_overrides.iter().find(|o| o.tensor_name == tensor_name)
    }

    pub fn to_gguf_metadata(&self) -> HashMap<String, GgufValue> {
        let mut metadata = HashMap::new();
        if let Some(magic) = &self.magic {
            metadata.insert("grim.magic".into(), GgufValue::String(magic.clone()));
        }
        if let Some(version) = self.quant_version {
            metadata.insert("grim.quant_version".into(), GgufValue::Uint32(version));
        }
        metadata.insert(
            "grim.rocml.profile".into(),
            GgufValue::String(match self.rocml_profile {
                GrimRocmlProfile::Cdna2 => "cdna2",
                GrimRocmlProfile::Cdna3 => "cdna3",
                GrimRocmlProfile::Rdna2 => "rdna2",
                GrimRocmlProfile::Rdna3 => "rdna3",
                GrimRocmlProfile::Rdna4 => "rdna4",
                GrimRocmlProfile::All => "all",
                GrimRocmlProfile::Unknown => "unknown",
            }
            .to_string()),
        );
        if self.wavefront_size > 0 {
            metadata.insert(
                "grim.rocml.wavefront_size".into(),
                GgufValue::Uint32(self.wavefront_size),
            );
        }
        if let Some(target_gcn) = &self.target_gcn {
            metadata.insert("grim.rocml.target_gcn".into(), GgufValue::String(target_gcn.clone()));
        }
        if let Some(block_size) = self.block_size {
            metadata.insert("grim.rocml.block_size".into(), GgufValue::Uint32(block_size));
        }
        if let Some(lds_size) = self.lds_size {
            metadata.insert("grim.rocml.lds_size".into(), GgufValue::Uint32(lds_size));
        }
        metadata.insert(
            "grim.rocml.tensor_core_enabled".into(),
            GgufValue::Bool(self.tensor_core_enabled),
        );
        if let Some(quant_method) = &self.quant_method {
            metadata.insert("grim.quant_method".into(), GgufValue::String(quant_method.clone()));
        }
        if let Some(calibration_dataset) = &self.calibration_dataset {
            metadata.insert(
                "grim.calibration_dataset".into(),
                GgufValue::String(calibration_dataset.clone()),
            );
        }
        if !self.quant_overrides.is_empty() {
            metadata.insert(
                "grim.quant_overrides".into(),
                GgufValue::Array(
                    self.quant_overrides
                        .iter()
                        .map(|ov| {
                            GgufValue::Array(vec![
                                GgufValue::String(ov.tensor_name.clone()),
                                GgufValue::Uint32(ov.effective_bpw),
                                GgufValue::Uint32(ov.override_dtype as u32),
                                GgufValue::Float32(ov.importance_score),
                                GgufValue::String(match ov.layout_hint {
                                    Some(GrimLayoutHint::WavefrontTiled) => "wavefront-tiled",
                                    Some(GrimLayoutHint::BlockSparse) => "block-sparse",
                                    None => "none",
                                }
                                .to_string()),
                            ])
                        })
                        .collect(),
                ),
            );
        }
        if let Some(mode) = self.train_quant_mode {
            metadata.insert(
                "grim.train.quant_mode".into(),
                GgufValue::String(mode.as_str().to_string()),
            );
        }
        if !self.train_fusion_ops.is_empty() {
            metadata.insert(
                "grim.train.fusion_ops".into(),
                GgufValue::Array(
                    self.train_fusion_ops
                        .iter()
                        .map(|op| GgufValue::String(op.as_str().to_string()))
                        .collect(),
                ),
            );
        }
        if !self.rocm_fusion_ops.is_empty() {
            metadata.insert(
                "grim.rocm.fusion_ops".into(),
                GgufValue::Array(
                    self.rocm_fusion_ops
                        .iter()
                        .map(|op| GgufValue::String(op.as_str().to_string()))
                        .collect(),
                ),
            );
        }
        if let Some(xnack_enabled) = self.xnack_enabled {
            metadata.insert("grim.rocm.xnack_enabled".into(), GgufValue::Bool(xnack_enabled));
        }
        if let Some(kv_layout_optimized) = self.kv_layout_optimized {
            metadata.insert(
                "grim.rocm.kv_layout_optimized".into(),
                GgufValue::Bool(kv_layout_optimized),
            );
        }
        if let Some(has_kv_registry) = self.has_kv_registry {
            metadata.insert(
                "grim.has_kv_registry".into(),
                GgufValue::Bool(has_kv_registry),
            );
        }
        metadata
    }

    /// Serialize to a JSON object for the native `.grim` metadata layer.
    ///
    /// The native format stores metadata as a JSON blob between the header
    /// and the tensor registry (spec §1 "Metadata JSON Layer"). This is the
    /// same information `to_gguf_metadata` encodes as GGUF KV pairs, but in
    /// a self-describing representation that does not require a GGUF reader.
    pub fn to_json(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        if let Some(magic) = &self.magic {
            obj.insert("magic".into(), serde_json::Value::String(magic.clone()));
        }
        if let Some(version) = self.quant_version {
            obj.insert("quant_version".into(), serde_json::Value::Number(version.into()));
        }
        obj.insert(
            "rocml_profile".into(),
            serde_json::Value::String(match self.rocml_profile {
                GrimRocmlProfile::Cdna2 => "cdna2",
                GrimRocmlProfile::Cdna3 => "cdna3",
                GrimRocmlProfile::Rdna2 => "rdna2",
                GrimRocmlProfile::Rdna3 => "rdna3",
                GrimRocmlProfile::Rdna4 => "rdna4",
                GrimRocmlProfile::All => "all",
                GrimRocmlProfile::Unknown => "unknown",
            }.into()),
        );
        if self.wavefront_size > 0 {
            obj.insert(
                "wavefront_size".into(),
                serde_json::Value::Number(self.wavefront_size.into()),
            );
        }
        if let Some(gcn) = &self.target_gcn {
            obj.insert("target_gcn".into(), serde_json::Value::String(gcn.clone()));
        }
        if let Some(block_size) = self.block_size {
            obj.insert("block_size".into(), serde_json::Value::Number(block_size.into()));
        }
        if let Some(lds) = self.lds_size {
            obj.insert("lds_size".into(), serde_json::Value::Number(lds.into()));
        }
        obj.insert(
            "tensor_core_enabled".into(),
            serde_json::Value::Bool(self.tensor_core_enabled),
        );
        if let Some(method) = &self.quant_method {
            obj.insert("quant_method".into(), serde_json::Value::String(method.clone()));
        }
        if let Some(dataset) = &self.calibration_dataset {
            obj.insert("calibration_dataset".into(), serde_json::Value::String(dataset.clone()));
        }
        if !self.quant_overrides.is_empty() {
            obj.insert(
                "quant_overrides".into(),
                serde_json::Value::Array(
                    self.quant_overrides.iter().map(override_to_json).collect(),
                ),
            );
        }
        if let Some(mode) = self.train_quant_mode {
            obj.insert("train_quant_mode".into(), serde_json::Value::String(mode.as_str().into()));
        }
        if !self.train_fusion_ops.is_empty() {
            obj.insert(
                "train_fusion_ops".into(),
                serde_json::Value::Array(
                    self.train_fusion_ops.iter().map(|o| serde_json::Value::String(o.as_str().into())).collect(),
                ),
            );
        }
        if !self.rocm_fusion_ops.is_empty() {
            obj.insert(
                "rocm_fusion_ops".into(),
                serde_json::Value::Array(
                    self.rocm_fusion_ops.iter().map(|o| serde_json::Value::String(o.as_str().into())).collect(),
                ),
            );
        }
        if let Some(xnack) = self.xnack_enabled {
            obj.insert("xnack_enabled".into(), serde_json::Value::Bool(xnack));
        }
        if let Some(kv_opt) = self.kv_layout_optimized {
            obj.insert("kv_layout_optimized".into(), serde_json::Value::Bool(kv_opt));
        }
        if let Some(has_kv) = self.has_kv_registry {
            obj.insert("has_kv_registry".into(), serde_json::Value::Bool(has_kv));
        }
        if !self.ext_entries.is_empty() {
            // Capability extensions are tucked under a namespaced key so the
            // on-disk wire layout (header + JSON metadata + tensor registry)
            // stays unchanged.
            obj.insert(
                "grim.ext.entries".into(),
                serde_json::Value::Array(
                    self.ext_entries
                        .iter()
                        .map(crate::spec::GrimTensorExt::to_json)
                        .collect(),
                ),
            );
        }
        serde_json::Value::Object(obj)
    }

    /// Deserialize from a JSON object produced by [`to_json`](Self::to_json).
    ///
    /// Missing keys fall back to [`Default`] values, so a partial or empty
    /// JSON object yields a valid (if uninformative) `GrimMetadata`.
    pub fn from_json(value: &serde_json::Value) -> Self {
        let empty = serde_json::Map::new();
        let obj = value.as_object().unwrap_or(&empty);

        let magic = obj.get("magic").and_then(|v| v.as_str()).map(String::from);
        let quant_version = obj.get("quant_version").and_then(|v| v.as_u64()).map(|v| v as u32);
        let profile = obj
            .get("rocml_profile")
            .and_then(|v| v.as_str())
            .map(GrimRocmlProfile::from_str)
            .unwrap_or_default();
        let wavefront_size = obj
            .get("wavefront_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or_else(|| profile.wavefront_size());
        let target_gcn = obj.get("target_gcn").and_then(|v| v.as_str()).map(String::from);
        let block_size = obj.get("block_size").and_then(|v| v.as_u64()).map(|v| v as u32);
        let lds_size = obj.get("lds_size").and_then(|v| v.as_u64()).map(|v| v as u32);
        let tensor_core_enabled = obj.get("tensor_core_enabled").and_then(|v| v.as_bool()).unwrap_or(false);
        let quant_method = obj.get("quant_method").and_then(|v| v.as_str()).map(String::from);
        let calibration_dataset = obj.get("calibration_dataset").and_then(|v| v.as_str()).map(String::from);
        let quant_overrides = obj
            .get("quant_overrides")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(override_from_json).collect())
            .unwrap_or_default();
        let train_quant_mode = obj
            .get("train_quant_mode")
            .and_then(|v| v.as_str())
            .and_then(GrimTrainQuantMode::from_str);
        let train_fusion_ops = obj
            .get("train_fusion_ops")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .filter_map(GrimFusionOp::from_str)
                    .collect()
            })
            .unwrap_or_default();
        let rocm_fusion_ops = obj
            .get("rocm_fusion_ops")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .filter_map(GrimFusionOp::from_str)
                    .collect()
            })
            .unwrap_or_default();
        let xnack_enabled = obj.get("xnack_enabled").and_then(|v| v.as_bool());
        let kv_layout_optimized = obj.get("kv_layout_optimized").and_then(|v| v.as_bool());
        let has_kv_registry = obj.get("has_kv_registry").and_then(|v| v.as_bool());
        let ext_entries = obj
            .get("grim.ext.entries")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(crate::spec::GrimTensorExt::from_json).collect())
            .unwrap_or_default();

        GrimMetadata {
            magic,
            quant_version,
            rocml_profile: profile,
            wavefront_size,
            target_gcn,
            block_size,
            lds_size,
            tensor_core_enabled,
            quant_method,
            calibration_dataset,
            quant_overrides,
            train_quant_mode,
            train_fusion_ops,
            rocm_fusion_ops,
            xnack_enabled,
            kv_layout_optimized,
            has_kv_registry,
            ext_entries,
        }
    }
}

/// Serialize a [`GrimQuantOverride`] to JSON for the native metadata layer.
fn override_to_json(ov: &GrimQuantOverride) -> serde_json::Value {
    serde_json::json!({
        "tensor_name": ov.tensor_name,
        "effective_bpw": ov.effective_bpw,
        "override_dtype": ov.override_dtype as u32,
        "importance_score": ov.importance_score,
        "layout_hint": match ov.layout_hint {
            Some(GrimLayoutHint::WavefrontTiled) => "wavefront-tiled",
            Some(GrimLayoutHint::BlockSparse) => "block-sparse",
            None => "none",
        }
    })
}

/// Deserialize a [`GrimQuantOverride`] from JSON.
fn override_from_json(value: &serde_json::Value) -> Option<GrimQuantOverride> {
    let tensor_name = value.get("tensor_name")?.as_str()?.to_string();
    let effective_bpw = value.get("effective_bpw")?.as_u64()? as u32;
    let override_dtype_tag = value.get("override_dtype")?.as_u64()? as u32;
    let override_dtype = GgufDType::from_tag(override_dtype_tag)?;
    let importance_score = value.get("importance_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let layout_hint = value.get("layout_hint").and_then(|v| v.as_str()).and_then(|s| match s {
        "wavefront-tiled" => Some(GrimLayoutHint::WavefrontTiled),
        "block-sparse" => Some(GrimLayoutHint::BlockSparse),
        _ => None,
    });

    Some(GrimQuantOverride {
        tensor_name,
        effective_bpw,
        override_dtype,
        importance_score,
        layout_hint,
    })
}

/// Decode `grim.quant_overrides` from a GGUF `Array` metadata value.
/// Each entry in the array is itself a GGUF `Array` of `[name, effective_bpw, override_dtype, importance_score]`.
fn read_grim_quant_overrides(value: &GgufValue) -> Option<Vec<GrimQuantOverride>> {
    let arr = value.as_array()?;
    let mut overrides = Vec::with_capacity(arr.len());
    for entry in arr {
        let inner = entry.as_array()?;
        if inner.len() < 4 {
            continue;
        }
        let tensor_name = inner[0].as_str()?.to_string();
        let effective_bpw = inner[1].as_u32()?;
        let override_dtype_tag = inner[2].as_u32()?;
        let override_dtype = GgufDType::from_tag(override_dtype_tag)?;
        let importance_score = inner[3].as_f32().unwrap_or(0.0);

        let layout_hint = inner.get(4).and_then(|v| v.as_str()).and_then(|s| {
            match s {
                "wavefront-tiled" => Some(GrimLayoutHint::WavefrontTiled),
                "block-sparse" => Some(GrimLayoutHint::BlockSparse),
                _ => None,
            }
        });

        overrides.push(GrimQuantOverride {
            tensor_name,
            effective_bpw,
            override_dtype,
            importance_score,
            layout_hint,
        });
    }
    Some(overrides)
}

fn read_grim_fusion_ops(value: &GgufValue) -> Option<Vec<GrimFusionOp>> {
    let values = value.as_array()?;
    Some(
        values
            .iter()
            .filter_map(|v| v.as_str())
            .filter_map(GrimFusionOp::from_str)
            .collect(),
    )
}

fn read_bool(value: &GgufValue) -> Option<bool> {
    match value {
        GgufValue::Bool(v) => Some(*v),
        _ => None,
    }
}

/// Loader that reads from a reader and returns parsed GGUF structure.
pub fn read_gguf<R: Read + Seek>(mut reader: R) -> Result<GgufFile> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let magic = u32::from_le_bytes(buf);
    if magic != GGUF_MAGIC {
        return Err(Error::Backend(format!(
            "not a GGUF file: magic {:#010x}",
            magic
        )));
    }
    reader.read_exact(&mut buf[..4])?;
    let version = u32::from_le_bytes(buf);
    if version != GGUF_VERSION {
        return Err(Error::Backend(format!(
            "unsupported GGUF version {version}, expected {GGUF_VERSION}"
        )));
    }
    let tensor_count = read_u64_le(&mut reader)?;
    let metadata_kv_count = read_u64_le(&mut reader)?;

    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(&mut reader)?;
        let value = read_gguf_value(&mut reader)?;
        metadata.insert(key, value);
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut reader)?;
        let n_dims = read_u32_le(&mut reader)?;
        let mut dims: Vec<u64> = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64_le(&mut reader)?);
        }
        // Read the tensor data type (GGUF ≥ v2 stores dtype per tensor)
        let dtype_tag = read_u32_le(&mut reader)?;
        let dtype = GgufDType::from_tag(dtype_tag)
            .ok_or_else(|| Error::Backend(format!("unknown GGUF dtype tag {dtype_tag}")))?;
        let offset = read_u64_le(&mut reader)?;
        // Compute size using the dtype's BLOCK layout (per gguf-main/src/lib.rs):
        //   size_bytes = (params × type_size_per_block) / block_size
        let block_size = dtype.block_size();
        let type_size = dtype.type_size_per_block();
        let params: u64 = dims.iter().product();
        let size_bytes: u64 = if block_size == 1 {
            params * type_size
        } else if type_size == 0 {
            0
        } else {
            (params.saturating_mul(type_size)) / block_size
        };
        tensors.push(GgufTensorInfo {
            name,
            dims,
            offset,
            size_bytes,
            dtype,
        });
    }
    // data_start is at the current reader position, aligned to 32 bytes
    let pos = reader.stream_position()?;
    let data_start = (pos + 31) & !31;

    Ok(GgufFile {
        version,
        tensor_count,
        metadata,
        tensors,
        data_start,
    })
}

/// Read one tensor's raw bytes from a GGUF-backed file.
pub fn read_tensor_bytes<R: Read + Seek>(
    reader: &mut R,
    file: &GgufFile,
    info: &GgufTensorInfo,
) -> Result<Vec<u8>> {
    let start = file.data_start + info.offset;
    let size = info.size_bytes as usize;
    reader.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; size];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

// ---------- low-level helpers ----------

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).map_err(|e| {
        Error::Backend(format!("GGUF string not valid UTF-8: {e}"))
    })?)
}

fn read_gguf_value<R: Read>(r: &mut R) -> Result<GgufValue> {
    let tag = read_u32_le(r)?;
    read_gguf_value_with_tag(r, tag)
}

/// Read a single GGUF metadata value given its type tag.
///
/// GGUF stores scalars at their natural byte widths (UINT8/INT8/BOOL = 1
/// byte, UINT16/INT16 = 2, UINT32/INT32/FLOAT32 = 4, UINT64/INT64/FLOAT64
/// = 8). ARRAY elements are stored WITHOUT a repeated type tag — only the
/// array's element type (read once) precedes the count and the raw elements.
fn read_gguf_value_with_tag<R: Read>(r: &mut R, tag: u32) -> Result<GgufValue> {
    match tag {
        // GGUF metadata value type tags
        0 => Ok(GgufValue::Uint8({
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            buf[0]
        })),
        1 => Ok(GgufValue::Int8({
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            i8::from_le_bytes(buf)
        })),
        2 => Ok(GgufValue::Uint16({
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            u16::from_le_bytes(buf)
        })),
        3 => Ok(GgufValue::Int16({
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            i16::from_le_bytes(buf)
        })),
        4 => Ok(GgufValue::Uint32(read_u32_le(r)?)),
        5 => Ok(GgufValue::Int32({
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            i32::from_le_bytes(buf)
        })),
        6 => Ok(GgufValue::Float32({
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            f32::from_le_bytes(buf)
        })),
        7 => Ok(GgufValue::Bool({
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            buf[0] != 0
        })),
        8 => Ok(GgufValue::String(read_gguf_string(r)?)),
        9 => {
            // Array: element type tag (u32) + count (u64) + `count` raw
            // elements of that type (no per-element tag). Elements may
            // themselves be arrays (nested), so recurse on the element tag.
            let elem_tag = read_u32_le(r)?;
            let count = read_u64_le(r)?;
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_gguf_value_with_tag(r, elem_tag)?);
            }
            Ok(GgufValue::Array(items))
        }
        10 => Ok(GgufValue::Uint64(read_u64_le(r)?)),
        11 => Ok(GgufValue::Int64({
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            i64::from_le_bytes(buf)
        })),
        12 => Ok(GgufValue::Float64({
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            f64::from_le_bytes(buf)
        })),
        t => Err(Error::Backend(format!("unknown GGUF metadata tag {t}"))),
    }
}

/// Canonical GGUF dtype → grim `DType` mapping, preserving quantization storage.
///
/// This is the single source of truth for how GGUF tensor dtypes map to grim's
/// `DType` (arithmetic type + storage encoding). Both `map_gguf_dtype_to_grim`
/// (provenance-aware) and `tprov::dtype_from_gguf` delegate here so they cannot
/// disagree.
///
/// Unquantized types map to `Storage::Native`. Block-quantized K-quants map to
/// the appropriate `Storage::KQuant`/`Storage::Block` variant so dequant kernels
/// can select the correct layout.
pub fn map_gguf_dtype_to_storage(gguf_dtype: GgufDType) -> DType {
    match gguf_dtype {
        GgufDType::F32 => DType::F32,
        GgufDType::F16 => DType::F16,
        GgufDType::F64 => DType::F32,
        GgufDType::I8 => DType {
            arith: grim_tensor::ArithType::U8,
            storage: Storage::Native,
        },
        GgufDType::I16 | GgufDType::I32 | GgufDType::I64 => DType::F32,
        GgufDType::Q2K => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q2K),
        },
        GgufDType::Q3K => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q3K),
        },
        GgufDType::Q4K => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::Block(BlockDtype::Fp4),
        },
        GgufDType::Q4_0 | GgufDType::Q4_1 | GgufDType::Q4_2 => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q4K),
        },
        GgufDType::Q5K => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::Block(BlockDtype::Nf4),
        },
        GgufDType::Q5_0 | GgufDType::Q5_1 => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q5K),
        },
        GgufDType::Q6K => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::Block(BlockDtype::Fp8),
        },
        GgufDType::Q8K | GgufDType::Q8_0 | GgufDType::Q8_1 | GgufDType::Q8_1Hx => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q80),
        },
        GgufDType::IQ4_NL => DType {
            arith: grim_tensor::ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::IQ4NL),
        },
    }
}

/// GGUF dtype → `(DType, effective_bpw)` for provenance tracking.
///
/// Delegates to [`map_gguf_dtype_to_storage`] for the DType, then attaches
/// the effective bits-per-weight as provenance metadata.
pub fn map_gguf_dtype_to_grim(gguf_dtype: GgufDType) -> (DType, Option<u32>) {
    let dtype = map_gguf_dtype_to_storage(gguf_dtype);
    let bpw = match gguf_dtype {
        GgufDType::F32 | GgufDType::F64 | GgufDType::I32 | GgufDType::I64 => None,
        GgufDType::F16 | GgufDType::I16 => Some(16),
        GgufDType::I8 => Some(8),
        GgufDType::Q4_0 | GgufDType::Q4_1 | GgufDType::Q4_2 | GgufDType::Q4K | GgufDType::IQ4_NL => Some(4),
        GgufDType::Q5_0 | GgufDType::Q5_1 | GgufDType::Q5K => Some(5),
        GgufDType::Q6K => Some(6),
        GgufDType::Q2K => Some(2),
        GgufDType::Q3K => Some(3),
        GgufDType::Q8_0 | GgufDType::Q8_1 | GgufDType::Q8_1Hx | GgufDType::Q8K => Some(8),
    };
    (dtype, bpw)
}

/// Enum extension to determine if a GGUF dtype is quantized.
impl GgufDType {
    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            GgufDType::Q4_0
                | GgufDType::Q4_1
                | GgufDType::Q4_2
                | GgufDType::Q5_0
                | GgufDType::Q5_1
                | GgufDType::Q8_0
                | GgufDType::Q8_1
                | GgufDType::Q8_1Hx
                | GgufDType::Q2K
                | GgufDType::Q3K
                | GgufDType::Q4K
                | GgufDType::Q5K
                | GgufDType::Q6K
                | GgufDType::Q8K
                | GgufDType::IQ4_NL
        )
    }

    /// Returns the GGUF display name for this dtype.
    pub fn display_name(self) -> &'static str {
        match self {
            GgufDType::F32 => "F32",
            GgufDType::F16 => "F16",
            GgufDType::F64 => "F64",
            GgufDType::I8 => "I8",
            GgufDType::I16 => "I16",
            GgufDType::I32 => "I32",
            GgufDType::I64 => "I64",
            GgufDType::Q4_0 => "Q4_0",
            GgufDType::Q4_1 => "Q4_1",
            GgufDType::Q4_2 => "Q4_2",
            GgufDType::Q5_0 => "Q5_0",
            GgufDType::Q5_1 => "Q5_1",
            GgufDType::Q8_0 => "Q8_0",
            GgufDType::Q8_1 => "Q8_1",
            GgufDType::Q8_1Hx => "Q8_1Hx",
            GgufDType::Q2K => "Q2_K",
            GgufDType::Q3K => "Q3_K",
            GgufDType::Q4K => "Q4_K",
            GgufDType::Q5K => "Q5_K",
            GgufDType::Q6K => "Q6_K",
            GgufDType::Q8K => "Q8_K",
            GgufDType::IQ4_NL => "IQ4_NL",
            }
            }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> GrimMetadata {
        GrimMetadata {
            magic: Some("grim-v1".into()),
            quant_version: Some(1),
            rocml_profile: GrimRocmlProfile::Rdna3,
            wavefront_size: 64,
            target_gcn: Some("gfx1100".into()),
            block_size: Some(256),
            lds_size: Some(32768),
            tensor_core_enabled: true,
            quant_method: Some("evopress-gptq".into()),
            calibration_dataset: Some("wikitext".into()),
            quant_overrides: vec![GrimQuantOverride {
                tensor_name: "model.layers.0.wq".into(),
                effective_bpw: 4,
                override_dtype: GgufDType::Q4K,
                importance_score: 0.42,
                layout_hint: Some(GrimLayoutHint::WavefrontTiled),
            }],
            train_quant_mode: Some(GrimTrainQuantMode::Bf16),
            train_fusion_ops: vec![GrimFusionOp::RmsNormMatMul],
            rocm_fusion_ops: vec![GrimFusionOp::QkvAttention],
            xnack_enabled: Some(false),
            kv_layout_optimized: Some(true),
            has_kv_registry: Some(true),
            ext_entries: Vec::new(),
        }
    }

    #[test]
    fn metadata_json_round_trip_preserves_all_fields() {
        let original = sample_metadata();
        let json = original.to_json();
        let restored = GrimMetadata::from_json(&json);
        assert_eq!(original.magic, restored.magic);
        assert_eq!(original.quant_version, restored.quant_version);
        assert_eq!(original.rocml_profile, restored.rocml_profile);
        assert_eq!(original.wavefront_size, restored.wavefront_size);
        assert_eq!(original.target_gcn, restored.target_gcn);
        assert_eq!(original.block_size, restored.block_size);
        assert_eq!(original.lds_size, restored.lds_size);
        assert_eq!(original.tensor_core_enabled, restored.tensor_core_enabled);
        assert_eq!(original.quant_method, restored.quant_method);
        assert_eq!(original.calibration_dataset, restored.calibration_dataset);
        assert_eq!(original.quant_overrides.len(), restored.quant_overrides.len());
        assert_eq!(original.quant_overrides[0].tensor_name, restored.quant_overrides[0].tensor_name);
        assert_eq!(original.quant_overrides[0].effective_bpw, restored.quant_overrides[0].effective_bpw);
        assert_eq!(original.quant_overrides[0].override_dtype, restored.quant_overrides[0].override_dtype);
        assert!((original.quant_overrides[0].importance_score - restored.quant_overrides[0].importance_score).abs() < 1e-6);
        assert_eq!(original.quant_overrides[0].layout_hint, restored.quant_overrides[0].layout_hint);
        assert_eq!(original.train_quant_mode, restored.train_quant_mode);
        assert_eq!(original.train_fusion_ops, restored.train_fusion_ops);
        assert_eq!(original.rocm_fusion_ops, restored.rocm_fusion_ops);
        assert_eq!(original.xnack_enabled, restored.xnack_enabled);
        assert_eq!(original.kv_layout_optimized, restored.kv_layout_optimized);
    }

    #[test]
    fn metadata_json_round_trip_default_is_identity() {
        let original = GrimMetadata::default();
        let json = original.to_json();
        let restored = GrimMetadata::from_json(&json);
        assert_eq!(original.magic, restored.magic);
        assert_eq!(original.quant_version, restored.quant_version);
        assert_eq!(original.rocml_profile, restored.rocml_profile);
        assert_eq!(original.wavefront_size, restored.wavefront_size);
        assert!(original.quant_overrides.is_empty());
    }

    #[test]
    fn metadata_from_empty_json_yields_default() {
        let empty = serde_json::Value::Object(serde_json::Map::new());
        let restored = GrimMetadata::from_json(&empty);
        assert_eq!(restored.magic, None);
        assert_eq!(restored.quant_version, None);
        assert_eq!(restored.rocml_profile, GrimRocmlProfile::Unknown);
        assert_eq!(restored.wavefront_size, 0);
    }

    /// WI-S5: the RDNA2 (gfx1036) profile parses from both `rdna2` and
    /// `gfx1036` aliases, returns the RDNA-family numeric hints
    /// (wavefront 64, LDS 32 kB), and round-trips through JSON metadata
    /// exactly like its RDNA3/RDNA4 siblings.
    #[test]
    fn rocml_profile_rdna2_parses_aliases_and_round_trips() {
        assert_eq!(GrimRocmlProfile::from_str("rdna2"), GrimRocmlProfile::Rdna2);
        assert_eq!(GrimRocmlProfile::from_str("gfx1036"), GrimRocmlProfile::Rdna2);
        assert_eq!(GrimRocmlProfile::from_str("RDNA2"), GrimRocmlProfile::Rdna2);
        assert_eq!(GrimRocmlProfile::Rdna2.wavefront_size(), 64);
        assert_eq!(GrimRocmlProfile::Rdna2.lds_size(), 32768);

        // Round-trip through the JSON metadata layer.
        let mut original = sample_metadata();
        original.rocml_profile = GrimRocmlProfile::Rdna2;
        original.wavefront_size = GrimRocmlProfile::Rdna2.wavefront_size();
        original.lds_size = Some(GrimRocmlProfile::Rdna2.lds_size());
        let restored = GrimMetadata::from_json(&original.to_json());
        assert_eq!(restored.rocml_profile, GrimRocmlProfile::Rdna2);
        assert_eq!(restored.wavefront_size, 64);
        assert_eq!(restored.lds_size, Some(32768));
    }

    /// The spec capability extensions ride the JSON metadata layer under
    /// `grim.ext.entries`. This test proves a populated `ext_entries`
    /// round-trips through `to_json`/`from_json` with all fields intact.
    #[test]
    fn metadata_ext_entries_round_trip_through_json() {
        use crate::spec::{
            GrimTensorExt, LayoutDescriptor, LayoutHintTag, OutlierIndexEncoding,
            PayloadCompression, PerRowBpwMode, RowScaleDtype,
        };

        let original = GrimMetadata {
            ext_entries: vec![
                GrimTensorExt {
                    tensor_name: "layer.0.weight".into(),
                    row_count: 128,
                    row_stride: 4096,
                    block_size: 0,
                    per_row_bpw_mode: PerRowBpwMode::PerRowTable,
                    default_bpw: 4,
                    own_bpw_table: 1,
                    row_scale_dtype: RowScaleDtype::U8,
                    scale_offset: 8192,
                    scale_size: 128,
                    gptq_ordered: 1,
                    outlier_index_encoding: OutlierIndexEncoding::DeltaVarint,
                    outlier_residual_bpw: 8,
                    compression: PayloadCompression::Zstd,
                    fusion_mask: 0b11,
                    layout_hint: LayoutHintTag::WavefrontTiled,
                    layout_descriptor: LayoutDescriptor([1, 2, 3, 4]),
                    backup1: crate::spec::BackupLayer {
                        codes_offset: 16384,
                        codes_size: 4096,
                        bpw: 8,
                        scale_offset: 20480,
                        scale_size: 64,
                    },
                    backup2: crate::spec::BackupLayer::default(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let json = original.to_json();
        let restored = GrimMetadata::from_json(&json);

        assert_eq!(restored.ext_entries.len(), 1);
        let ext = &restored.ext_entries[0];
        assert_eq!(ext.tensor_name, "layer.0.weight");
        assert_eq!(ext.row_count, 128);
        assert_eq!(ext.row_stride, 4096);
        assert_eq!(ext.per_row_bpw_mode, PerRowBpwMode::PerRowTable);
        assert_eq!(ext.default_bpw, 4);
        assert_eq!(ext.row_scale_dtype, RowScaleDtype::U8);
        assert_eq!(ext.scale_offset, 8192);
        assert_eq!(ext.scale_size, 128);
        assert_eq!(ext.gptq_ordered, 1);
        assert_eq!(ext.outlier_index_encoding, OutlierIndexEncoding::DeltaVarint);
        assert_eq!(ext.outlier_residual_bpw, 8);
        assert_eq!(ext.compression, PayloadCompression::Zstd);
        assert_eq!(ext.fusion_mask, 0b11);
        assert_eq!(ext.layout_hint, LayoutHintTag::WavefrontTiled);
        assert_eq!(ext.layout_descriptor.0, [1, 2, 3, 4]);
        assert!(ext.backup1.is_present());
        assert_eq!(ext.backup1.codes_offset, 16384);
        assert_eq!(ext.backup1.bpw, 8);
        assert!(!ext.backup2.is_present());
    }

    /// A metadata object without `grim.ext.entries` deserializes to an
    /// empty extension list, not an error — V1 files still load cleanly.
    #[test]
    fn metadata_without_ext_entries_yields_empty_list() {
        let json = serde_json::json!({
            "magic": "grim-v1",
            "quant_version": 1,
        });
        let restored = GrimMetadata::from_json(&json);
        assert!(restored.ext_entries.is_empty());
    }
}
