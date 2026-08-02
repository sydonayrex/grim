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

pub use convert::{convert_gguf_to_grim, convert_to_grim};
pub use format::normals_packed_size;
pub use format::{FUCKING_SORCERY, GrimHeader, GrimTensorEntry};
pub use gguf::{
    GGUF_MAGIC, GGUF_VERSION, GgufDType, GgufFile, GgufTensorInfo, GgufValue, GrimFusionOp,
    GrimLayoutHint, GrimMetadata, GrimQuantOverride, GrimRocmlProfile, GrimTrainQuantMode,
};
pub use onnx::OnnxProvider;
pub use spec::{BackupLayer, GrimTensorExt, LayoutHintTag, PayloadCompression};
pub use tokenizer::GgufTokenizer;
pub use tokenizer::{ChatMessage, render_chat_template, render_messages_or_last};
pub use tprov::GgufProvider;
pub use tprov::GrimProvider;
pub use train::{TrainFpFormat, TrainState};
