//! Core tensor abstractions, data types, shapes, and backend-agnostic trait contracts.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::not_unsafe_ptr_arg_deref
)]

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
    QuantizedMatmulBackwardResiduals, ReadyHandle, RopeConfig, ScytheLink, ScythePlacement,
    YaRNParams,
};

pub use dtype::{
    ArithType, BlockDtype, DType, Device, FloatPackScheme, GpuIntConfig, GroupQuantScheme,
    KQuantScheme, QuantFormat, QuantProvenance, Storage,
};
pub use error::{Error, Result};
pub use provider::{RawTensor, TensorMeta, TensorProvider};
pub use shape::Shape;
pub use softmax_merge::{SoftmaxPartial, merge_all, merge_partials};
pub use tensor::Tensor;
