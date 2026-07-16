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
            GrimRocmlProfile::Rdna3 => 64,
            GrimRocmlProfile::Rdna4 => 64,
            GrimRocmlProfile::All | GrimRocmlProfile::Unknown => 0,
        }
    }

    pub fn lds_size(&self) -> u32 {
        match self {
            GrimRocmlProfile::Cdna2 => 65536,
            GrimRocmlProfile::Cdna3 => 65536,
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
        }
    }
}

impl GrimMetadata {
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
        metadata
    }
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

/// Derive DType from GGUF metadata 'general.architecture' name + per-weight
/// metadata. This is a heuristic used when the GGUF file does not store
/// explicit per-tensor dtype.
pub fn dtype_for_gguf(name: &str) -> DType {
    let _ = name;
    // GGUF v3 stores all tensors as f32 by default (llama.cpp uses
    // quantization-specific per-tensor overrides). We default to F32
    // and let the per-tensor metadata override.
    DType::F32
}

/// Guess the GGUF tensor name from a weight path (slash-separated -> dot-separated).
pub fn gguf_tensor_name(path: &str) -> String {
    path.replace('.', ".")
}

/// Comprehensive GGUF dtype mapping to grim DType.
/// Maps GGUF quantization tags to appropriate DType + provenance metadata.
/// §7.2.4: per-tensor quantization type tag parsing.
pub fn map_gguf_dtype_to_grim(gguf_dtype: GgufDType) -> (DType, Option<u32>) {
    match gguf_dtype {
        GgufDType::F32 => (DType::F32, None),
        GgufDType::F16 => (DType::BF16, None),
        GgufDType::F64 => (DType::F32, None), // Map F64 to F32 for compat
        GgufDType::I8 => (DType::F32, Some(8)),
        GgufDType::I16 => (DType::F32, Some(16)),
        GgufDType::I32 => (DType::F32, Some(32)),
        GgufDType::I64 => (DType::F32, Some(32)),
        // K-quants and Q-family: return F32 (dequantized) with bit count for provenance
        GgufDType::Q4_0 | GgufDType::Q4_1 | GgufDType::Q4_2 => (DType::F32, Some(4)),
        GgufDType::Q5_0 | GgufDType::Q5_1 => (DType::F32, Some(5)),
        GgufDType::Q8_0 | GgufDType::Q8_1 | GgufDType::Q8_1Hx => (DType::F32, Some(8)),
        GgufDType::Q2K => (DType::F32, Some(2)),
        GgufDType::Q3K => (DType::F32, Some(3)),
        GgufDType::Q4K => (DType::F32, Some(4)),
        GgufDType::Q5K => (DType::F32, Some(5)),
        GgufDType::Q6K => (DType::F32, Some(6)),
        GgufDType::Q8K => (DType::F32, Some(8)),
        GgufDType::IQ4_NL => (DType::F32, Some(4)),
    }
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
