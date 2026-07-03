//! Grim core error type.
//!
//! Every crate in the workspace ultimately returns `grim_tensor::Result<T>`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("dtype mismatch: {0}")]
    DTypeMismatch(String),

    #[error("device mismatch: {0}")]
    DeviceMismatch(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("shape error: {0}")]
    Shape(String),

    #[error("unimplemented: {0}")]
    Unimplemented(String),

    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(String),
}

pub type Result<T> = std::result::Result<T, Error>;
