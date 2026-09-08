//! Device abstraction subsystem for CUDA GPUs.

pub mod cublas;
pub mod cuda_device;
pub mod device_attention;
pub mod device_compute;
pub mod device_quant;
pub mod device_recurrent;
pub mod device_routing;
pub mod device_serve;
pub mod handles;
pub mod jit_cache;
pub mod parallel_comm;

pub use cublas::CublasHandle;
pub use cuda_device::{CudaDevice, compute_utilization, vram_info};
pub use handles::{
    CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_STATUS_SUCCESS, CUstream, CudaHandle, cudaDeviceGetAttribute,
    cudaDeviceSynchronize, cudaFree, cudaGetDeviceCount, cudaGraphCreate, cudaGraphDestroy,
    cudaGraphExecDestroy, cudaGraphInstantiate, cudaGraphLaunch, cudaMalloc, cudaMemGetInfo,
    cudaMemcpy, cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice,
    cudaMemcpyPeer, cudaMemset, cudaSetDevice, cudaStreamBeginCapture, cudaStreamCreate,
    cudaStreamDestroy, cudaStreamEndCapture, cudaStreamSynchronize, cudaSuccess,
};
pub use jit_cache::{SendCmodule, compile_and_load_kernel};
