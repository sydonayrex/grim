//! Top-level error type for `grim-core` and crates that depend on it.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("tensor error: {0}")]
    Tensor(#[from] grim_tensor::Error),

    #[error("config error: {0}")]
    Config(String),

    #[error("session error: {0}")]
    Session(String),

    #[error("kv cache error: {0}")]
    KvCache(String),

    #[error("sampler error: {0}")]
    Sampler(String),

    #[error("shape error: {0}")]
    Shape(String),

    #[error("not implemented: {0}")]
    Unimplemented(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Re-export the tensor error so callers that only depend on `grim-core`
/// can still surface failures from `grim-tensor` operations.
pub use grim_tensor::Error as TensorError;
