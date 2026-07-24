pub mod architecture;
pub mod catalog;
pub mod client;
pub mod config;
pub mod error;
pub mod hyperparams;
pub mod kv_cache;
pub mod model;
pub mod paths;
pub mod sampler;
pub mod session;

pub use architecture::{ModelArchitecture, TensorNamingRegistry, TensorRole};
pub use error::{Error, Result, TensorError};
pub use hyperparams::{ArchHyperparameters, HyperparameterExtractor, MetadataLookup};
pub use kv_cache::KvCache;
pub use model::{
    CausalLm, DiffusionModel, Encoder, EncoderDecoderLm, Model, ModelConfig, ModalityHint,
    NoiseScheduler, SsmState, StatefulSequence,
};
pub use paths::{grim_config_dir, grim_log_dir, grim_models_dir, grim_plugins_dir, home_dir};
pub use sampler::Sampler;
pub use session::{Session, DeterminismMode};
pub use catalog::{ModelEntry, resolve_model_path, list_local_models};
pub use client::{download_model, download_model_with_progress, DownloadProgress};


