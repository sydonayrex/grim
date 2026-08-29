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
    AttentionOps, AutogradOps, BackendDevice, BackendStorage, CollectiveOps,
    CoreTensorOps, ElementwiseOps, FusionOps, GraphCaptureOps, GpuCapability,
    MemAdvice, MemoryOps, OptimizerOps, QuantOps, RecurrentOps, SamplingOps,
    QuantizedMatmulBackwardResiduals, ReadyHandle, RopeConfig, ScytheLink, ScythePlacement,
    YaRNParams,
};

pub use dtype::{
    ArithType, BlockDtype, DType, Device, FloatPackScheme, GpuIntConfig, GroupQuantScheme,
    KQuantScheme, QuantFormat, QuantProvenance, Storage,
};
pub use error::{Error, Result};
/// Re-export the `.gcct` compressed-tensor type tags so `Storage`'s compressed
/// variants expose a single, backend-agnostic enum.
pub use grim_compressed_tensors::CompressedTensorType;
pub use provider::{RawTensor, TensorMeta, TensorProvider};
pub use shape::Shape;
pub use softmax_merge::{SoftmaxPartial, merge_all, merge_partials};
pub use tensor::Tensor;
