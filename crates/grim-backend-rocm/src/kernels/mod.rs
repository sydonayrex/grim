//! Ground-level pyramid is `src/lib.rs`; this module holds per-kernel HIP [see: `lib.rs`]

pub mod awq_gemm;
pub mod bitnet_gemm;
/// Charon — P-DAFD fused MoE dispatch kernel (`rocm_kernel_plan.md` WI-A).
pub mod charon;
/// Charon — MoE backward pass (expert-weight gradients, WI-Charon-1).
///
/// FP32 expert-weight backward: `d_gate_w`/`d_up_w`/`d_down_w`/`d_x` for the
/// grouped (token-sorted) shape, mirroring
/// `wmma_gemm::grim_fused_dequant_backward_gemm_{fp8,mxfp4,mxfp8}`'s
/// dense-GEMM backward structure adapted to Charon's per-expert
/// grouped-dispatch shape. Router backward is explicitly out of scope
/// (separate work item — non-differentiable top-k). Quantized-weight
/// backward (the 5 quantized forward variants) is phase 2, following
/// `wmma_gemm.rs`'s existing fp8/mxfp4/mxfp8 backward pattern once the
/// FP32 base case is proven. Device-gated for HIP numeric correctness.
pub mod charon_backward;
/// Charon — WMMA / tensor-core grouped forward (WI-Charon-2).
///
/// Grouped (token-sorted, matching `grim_moe_fused_grouped`'s sort/pad
/// contract) rocWMMA 16×16 tile GEMM for gate/up/down, FP32 accumulation,
/// gated behind `CharonSelector`/`CharonVariant` as a new variant option.
/// FP32 first; FP8/MXFP4/MXFP8/Q8_0/IQK WMMA variants follow the pattern
/// `wmma_gemm.rs` already establishes for dense GEMM. Does NOT touch the
/// sortless single-token `grim_moe_fused_dispatch` path (WMMA tiling does
/// not help single-token decode). Device-gated for WMMA numeric parity vs
/// the scalar grouped kernel.
pub mod charon_wmma;
/// SCYTHE-2 WI-6: CommFuse decomposed P2P fan-in.
pub mod batched_lora;
pub mod comm_fuse;
pub mod compressed_gemm;
pub mod compute_kernels;
pub mod cross_attention;
pub mod decode_gemm;
/// WI-X3: device-side stochastic sampler (`grim_sample_logits_stochastic`) —
/// temperature/top-k/top-p + Gumbel-max multinomial draw, 4-byte token readback.
pub mod device_sampler;
pub mod extend_attention;
pub mod flash_decode;
pub mod fp8_gemm_rdna4;
pub mod fp8_standalone;
pub mod fused_dequant_gemm;
pub mod fused_linear_ce;
/// GPTQ/EfficientQAT GroupInt fused dequant-GEMM (forward + backward).
pub mod gptq_gemm;
pub mod iq_dequant;
pub mod iq_gemm;
pub mod jit_cache;
pub mod kv_dequant_attention;
pub mod log_softmax_vjp;
pub mod marlin_gemm;
pub mod mla_decode;
pub mod mrope;
pub mod mxfp4_gemm;
pub mod mxfp_standalone;
pub mod preshuffled_attention;
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
pub mod sage_attention;
pub mod scythe_persistent;
pub mod silu_mul_quant;
pub mod speculative_sampler;

pub mod selective_scan;
pub mod shared_device_fns;
pub mod source_asm;
pub mod tile_picker;
pub mod wmma_gemm;
