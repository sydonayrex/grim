//! Tensor and backend error types for `grim-tensor`.
//!
//! Low-level tensor computation and device abstraction errors. Crates that build
//! atop `grim-tensor` convert these into `grim_core::error::Error` via `From` / `?`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

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

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
