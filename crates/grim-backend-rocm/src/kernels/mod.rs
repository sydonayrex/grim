//! Ground-level pyramid is `src/lib.rs`; this module holds per-kernel HIP
//! sources that have been promoted out of the giant 4630-line `lib.rs` so we
//! can co-evolve their HIP body and their Rust host launcher without
//! touching the rest of the backend. Phase-1 qkv_attention lives here.

pub mod decode_gemm;
pub mod qkv_attention;
pub mod compute_kernels;
pub mod jit_cache;
pub mod source_asm;
pub mod fused_dequant_gemm;
pub mod kv_dequant_attention;

