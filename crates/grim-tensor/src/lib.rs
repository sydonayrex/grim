//! `grim-tensor` crate — Tensor, DType, Shape, Device, and the
//! backend-agnostic trait surface (`BackendDevice` / `BackendStorage` /
//! `ComputeHandle` / `TensorProvider`).
//!
//! Designed to mirror Candle's core data-model shape, per §4.1 of the
//! Grim architecture doc.

pub mod backend;
pub mod dtype;
pub mod error;
pub mod provider;
pub mod shape;
pub mod tensor;

pub use backend::{BackendDevice, BackendStorage, ComputeHandle, ReadyHandle};
pub use dtype::{
    ArithType, Device, DType, GpuIntConfig, GroupQuantScheme, KQuantScheme, QuantProvenance, Storage,
};
pub use error::{Error, Result};
pub use provider::{RawTensor, TensorMeta, TensorProvider};
pub use shape::Shape;
pub use tensor::Tensor;
