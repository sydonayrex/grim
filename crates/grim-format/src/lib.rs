pub mod gguf;
pub mod safetensors;
pub mod tprov;
pub mod gptq;
pub mod onnx;
pub mod tokenizer;
pub mod convert;
pub mod format;
pub mod spec;
/// WI-R6: training-state `.grim.train` sidecar (adapters, optimizer, error matrix).
pub mod train;

pub use gguf::{
    GrimFusionOp, GrimLayoutHint, GrimMetadata, GrimQuantOverride, GrimRocmlProfile,
    GrimTrainQuantMode,
};
pub use tprov::GgufProvider;
pub use tokenizer::GgufTokenizer;
pub use convert::{convert_gguf_to_grim, convert_to_grim};
pub use format::{GrimHeader, GrimTensorEntry, FUCKING_SORCERY};
