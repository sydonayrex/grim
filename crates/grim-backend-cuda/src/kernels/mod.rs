//! CUDA kernel source definitions and submodules.

pub mod charon;
pub mod charon_backward;
pub mod charon_wmma;
pub mod compressed_gemm;
pub mod decode_gemm;
pub mod device_sampler;
pub mod flash_decode;
pub mod gptq_gemm;
pub mod iq_gemm;
pub mod kv_dequant_attention;
pub mod mla_decode;
pub mod moe_mega_kernel;
pub mod mxfp_gemm;
pub mod preshuffled_attention;
pub mod q_gemm;
pub mod sage_attention;
pub mod scythe_persistent;
pub mod source;
pub mod speculative_sampler;

pub use source::KERNELS_SOURCE;
