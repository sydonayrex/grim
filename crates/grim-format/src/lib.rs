//! Model checkpoint parsers, tokenizers, metadata headers, and format converters (.gguf, .safetensors, .grim).

pub mod bolt_on;
pub mod convert;
pub mod format;
pub mod gguf;
pub mod gptq;
pub mod onnx;
pub mod safetensors;
pub mod spec;
pub mod tokenizer;
pub mod tprov;
/// WI-R6: training-state `.grim.train` sidecar (adapters, optimizer, error matrix).
pub mod train;

pub use convert::{
    GpuDequant, convert_gguf_to_grim, convert_to_grim, convert_to_grim_with_dequant,
};
pub use format::normals_packed_size;
pub use format::{
    FUCKING_SORCERY, GrimHeader, GrimTensorEntry, WaveSize, normals_packed_size_for_wave,
    pack_row_bpw, pack_row_bpw_for_wave,
};
pub use gguf::{
    GGUF_MAGIC, GGUF_VERSION, GgufDType, GgufFile, GgufTensorInfo, GgufValue, GrimFusionOp,
    GrimLayoutHint, GrimMetadata, GrimQuantOverride, GrimRocmlProfile, GrimTrainQuantMode,
};
pub use onnx::OnnxProvider;
pub use spec::{BackupLayer, GrimTensorExt, LayoutHintTag, PayloadCompression};
pub use tokenizer::GgufTokenizer;
pub use tokenizer::{
    ChatMessage, FunctionDef, FunctionName, ToolCallMsg, ToolChoice, ToolDef, render_chat_template,
    render_messages_or_last, render_messages_or_last_with_tools,
};
pub use tprov::GgufProvider;
pub use tprov::GrimProvider;
pub use train::{TrainFpFormat, TrainState};
