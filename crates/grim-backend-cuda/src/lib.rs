#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::op_ref,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::needless_borrow,
    clippy::new_without_default,
    clippy::filter_next
)]

//! CUDA backend for Grim — NVIDIA GPU target with JIT compilation, cuBLAS integration, and graph capture.

pub mod autotune;
pub mod caps;
pub mod device;
pub mod fsdp;
pub mod graph_capture;
pub mod kernels;
pub mod memory;
pub mod nccl;

pub use fsdp::{ConsumerDpConfig, ConsumerDpGroup, ConsumerFsdpConfig, ConsumerFsdpGroup, ConsumerZeroPlanner};
pub use device::parallel_comm::{CommBackendType, HostStagingRing, ParallelCommunicator, ParallelTopology};
pub use nccl::{CudaComm, UniqueId};

#[cfg(test)]
mod device_tests;

pub use autotune::{CudaAutotuner, CudaTileConfig, GemmOp, ShapeClass};
pub use caps::CudaCaps;
pub use device::{
    compute_utilization, cudaDeviceGetAttribute, cudaDeviceSynchronize, cudaFree,
    cudaGetDeviceCount, cudaGraphCreate, cudaGraphDestroy, cudaGraphExecDestroy,
    cudaGraphInstantiate, cudaGraphLaunch, cudaMalloc, cudaMemGetInfo, cudaMemcpy,
    cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice, cudaMemcpyPeer,
    cudaMemset, cudaSetDevice, cudaStreamBeginCapture, cudaStreamCreate, cudaStreamDestroy,
    cudaStreamEndCapture, cudaStreamSynchronize, cudaSuccess, vram_info, CublasHandle, CudaDevice,
    CudaHandle, CUstream, CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_STATUS_SUCCESS,
};
pub use graph_capture::{
    CudaGraphExecutor, DecodeBatchBucket, DecodeBucketGraphPool, DecodeGraph, DecodeGraphKey,
    GraphCaptureManager,
};
pub use memory::storage::CudaStorage;

pub use grim_tensor::backend::ComputeHandle;
pub use grim_tensor::dtype::{
    ArithType, BlockDtype, DType, FloatPackScheme, KQuantScheme, QuantProvenance,
    Storage as DTypeStorage,
};
pub use grim_tensor::error::{Error, Result};
pub use grim_tensor::{
    AttentionOps, AutogradOps, BackendDevice, BackendStorage, CollectiveOps,
    CoreTensorOps, ElementwiseOps, FusionOps, GraphCaptureOps, MemoryOps, OptimizerOps, QuantOps,
    RecurrentOps, SamplingOps, Shape,
};
