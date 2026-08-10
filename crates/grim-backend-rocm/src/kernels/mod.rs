//! Ground-level pyramid is `src/lib.rs`; this module holds per-kernel HIP [see: `lib.rs`]

/// Charon — P-DAFD fused MoE dispatch kernel (`rocm_kernel_plan.md` WI-A).
pub mod charon;
/// SCYTHE-2 WI-6: CommFuse decomposed P2P fan-in.
pub mod comm_fuse;
pub mod compute_kernels;
pub mod cross_attention;
pub mod decode_gemm;
pub mod fp8_gemm_rdna4;
pub mod fp8_standalone;
pub mod fused_dequant_gemm;
pub mod iq_dequant;
pub mod iq_gemm;
pub mod jit_cache;
pub mod kv_dequant_attention;
pub mod mxfp_standalone;
pub mod q2k_gemm;
pub mod q3k_gemm;
pub mod q4k_dequant;
pub mod q4k_gemm;
pub mod q5k_gemm;
pub mod q6k_gemm;
pub mod q8_0_dequant;
pub mod qkv_attention;
pub mod quant_standalone;
pub mod rwkv;
pub mod selective_scan;
pub mod shared_device_fns;
pub mod source_asm;
pub mod wmma_gemm;
