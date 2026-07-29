//! Ground-level pyramid is `src/lib.rs`; this module holds per-kernel HIP
//! sources that have been promoted out of the giant 4630-line `lib.rs` so we
//! can co-evolve their HIP body and their Rust host launcher without
//! touching the rest of the backend. Phase-1 qkv_attention lives here.

pub mod decode_gemm;
pub mod shared_device_fns;
pub mod qkv_attention;
pub mod compute_kernels;
pub mod jit_cache;
pub mod source_asm;
pub mod fused_dequant_gemm;
pub mod fp8_gemm_rdna4;
pub mod fp8_standalone;
pub mod iq_dequant;
pub mod iq_gemm;
pub mod kv_dequant_attention;
pub mod mxfp_standalone;
pub mod q2k_gemm;
pub mod q3k_gemm;
pub mod q4k_dequant;
pub mod q4k_gemm;
pub mod q5k_gemm;
pub mod q6k_gemm;
pub mod q8_0_dequant;
pub mod selective_scan;
pub mod flash_attn;
pub mod cross_attention;
pub mod rwkv;
pub mod wmma_gemm;


