//! Core tensor abstractions, data types, shapes, and backend-agnostic trait contracts.

pub mod backend;
pub mod dtype;
pub mod error;
pub mod provider;
pub mod shape;
pub mod softmax_merge;
pub mod tensor;
pub mod wavefront;

pub use backend::{
    BackendDevice, BackendStorage, ComputeHandle, GpuCapability, MemAdvice,
    QuantizedMatmulBackwardResiduals, ReadyHandle, ScytheLink, ScythePlacement,
};
pub use dtype::{
    ArithType, BlockDtype, DType, Device, FloatPackScheme, GpuIntConfig, GroupQuantScheme,
    KQuantScheme, QuantProvenance, Storage,
};
pub use error::{Error, Result};
pub use provider::{RawTensor, TensorMeta, TensorProvider};
pub use shape::Shape;
pub use softmax_merge::{SoftmaxPartial, merge_all, merge_partials};
pub use tensor::Tensor;
