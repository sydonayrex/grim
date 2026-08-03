//! Core model traits, session state, KV cache abstractions, sampler pipelines, and error types.

pub mod architecture;
pub mod catalog;
pub mod client;
pub mod config;
pub mod env_config;
pub mod error;
pub mod hyperparams;
pub mod kv_cache;
pub mod model;
pub mod paths;
pub mod rng;
pub mod sampler;
pub mod session;

pub use architecture::{ModelArchitecture, TensorNamingRegistry, TensorRole};
pub use catalog::{ModelEntry, list_local_models, resolve_model_path};
pub use client::{DownloadProgress, download_model, download_model_with_progress, is_bind_address_allowed};
pub use env_config::{Backend, RuntimeEnv};
pub use error::{Error, Result, TensorError};
pub use hyperparams::{ArchHyperparameters, HyperparameterExtractor, MetadataLookup};
pub use kv_cache::KvCache;
pub use model::{
    CausalLm, DiffusionModel, Encoder, EncoderDecoderLm, ModalityHint, Model, ModelConfig,
    NoiseScheduler, SsmState, StatefulSequence,
};
pub use paths::{grim_config_dir, grim_log_dir, grim_models_dir, grim_plugins_dir, home_dir};
pub use sampler::Sampler;
pub use session::{DeterminismMode, GraphBuilder, Session};
