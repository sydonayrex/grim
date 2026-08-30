//! Model checkpoint parsers, tokenizers, metadata headers, and format converters (.gguf, .safetensors, .grim).

pub mod awq;
pub mod bank;
pub mod bolt_on;
pub mod convert;
/// Fusion-pattern detection over checkpoint tensor names (folded in from
/// the former grim-tensor-graph crate; `GrimFusionOp` lives here).
pub mod fusion;
pub mod format;
pub mod ftw;
pub mod gguf;
pub mod gptq;
pub mod onnx;
pub mod safetensors;
pub mod spec;
pub mod tokenizer;
pub mod torch;
pub mod tprov;
/// WI-R6: training-state `.grim.train` sidecar (adapters, optimizer, error matrix).
pub mod train;
pub mod weight_format;

pub use awq::{AwqConfig, AwqProvider, AwqTensorInfo, pack_awq_group_int};
pub use fusion::{FusionGroup, TensorGraphIr, build_transformer_ir};
pub use ftw::{FtwDirectLoader, FtwHeader, FtwHostBank, FtwQuantFormat};
pub use torch::{PthProvider, TorchTensorEntry};

pub use convert::{
    GpuDequant, convert_gguf_to_grim, convert_to_grim, convert_to_grim_with_dequant,
};
pub use format::normals_packed_size;
pub use format::{
    FUCKING_SORCERY, GrimFile, GrimHeader, GrimTensorEntry, WaveSize, normals_packed_size_for_wave,
    pack_row_bpw, pack_row_bpw_for_wave,
};
pub use gguf::{
    GGUF_MAGIC, GGUF_VERSION, GgufDType, GgufFile, GgufTensorInfo, GgufValue, GrimFusionOp,
    GrimLayoutHint, GrimMetadata, GrimQuantOverride, GrimRocmlProfile, GrimTrainQuantMode,
    read_gguf,
};
pub use onnx::OnnxProvider;
pub use spec::{BackupLayer, GrimTensorExt, LayoutHintTag, PayloadCompression};
pub use tokenizer::GgufTokenizer;
pub use tokenizer::{
    ChatMessage, FunctionDef, FunctionName, ToolCallMsg, ToolChoice, ToolDef, render_chat_template,
    render_messages_or_last, render_messages_or_last_with_tools, sanitize_jinja_template,
};
pub use tprov::GgufProvider;
pub use tprov::GrimProvider;
pub use train::{TrainFpFormat, TrainState};
pub use weight_format::{
    ModelFootprint, ParseWeightFormatError, QuantModeHint, WeightFormat, estimate_vram_bytes,
};
