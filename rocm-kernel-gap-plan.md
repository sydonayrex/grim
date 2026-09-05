# ROCm kernel gap plan — model coverage vs grim-backend-rocm

Companion to `gpu-followup-workitems.md`. Derived from a full sweep of
`grim-models/` (151 files) against `grim-backend-rocm/src/kernels/`. Items in
scope for the current pass are marked [NOW]; the rest are sequenced follow-ups
with the evidence needed to start them.

## Status (2026-09-04)

| Item | State | Commit |
|---|---|---|
| P0 Gemma-2 attention softcap (correctness) | **DONE** — host reference applies `cap·tanh(s/cap)`; kernel fast-path sequenced | `2d87dbb` |
| P1 MLA per-head `w_uv` stride | **DONE** — one multi-head launch, GPU parity gate | `0074924` |
| P2 `selective_scan_headed` on ROCm | **DONE** — kernel + source registration, GPU parity | `35d0faa` |
| P3 Qwen3.5-MoE onto MoeFfn/Charon | **DONE** — layer parity gate | `8f58a43` |
| P4 sage_attention trait wiring | **DONE** — trait method → `sage_attention_gpu` | `837981e` |
| Paged block-table stride-2 decode (CPU+CUDA) | **DONE** — `block_table_block_id` shared helper | `7634602`/`655542f` |
| Multi-rank VPP (R3) | **DONE** — schedule, inproc+TCP transports, benchmark | `665602b` |

Remaining items below stay as sequenced follow-ups.

## Findings summary

Covered already: standard GQA (WMMA qkv + flash_decode), paged + quantized-KV
paged, alibi, MLA decode (deepseek2/32/4, kimi_k3), Charon MoE grouped dispatch
(+ MXFP4/FP8/AWQ/Int8 arms, WMMA prefill variant, mega kernel behind
`moe-deterministic-dispatch`), IQ2/IQ3/IQ4 fused dequant GEMM, Q2K–Q8_0, FP8
(+RDNA4 MFMA), MXFP4/8, GPTQ, AWQ, Marlin W4A16, BitNet, SPQR, selective_scan
(mamba decode), rwkv kernels, cross_attention (whisper), speculative_sampler,
fused_linear_ce.

## [NOW] P0 — Gemma-2 attention softcap never applied (correctness)

`gemma2.rs` carried `attn_logit_softcapping: Some(50.0)` in the config but no
code applied it — Gemma-2 logits were wrong everywhere, not just slow.

- Fix (this pass): softcap sits between the QK product and the softmax, which
  the fused GPU kernels cannot express, so capped blocks route to the host
  reference `kv_attention::causal_attention` (now takes `softcap: Option<f32>`,
  applied as `cap * tanh(s/cap)` after scaling). Uncapped blocks keep the
  device-resident fused path.
- Regression gate: block-level test asserts capped vs uncapped logits differ
  (failed before the fix — the cap was a no-op).
- Follow-up (perf): softcap parameter on the WMMA `qkv_attention`,
  `flash_decode`, and paged HIP kernels behind a `RocmDevice::
  set_attn_logit_softcapping(Option<f32>)` setter (same pattern as
  `set_mxfp4_fused_dequant_gemm_enabled`), so capped serving returns to device
  residency. All three kernels must land together — a partial wiring would
  silently drop the cap again on the uncovered path.

## [NOW] P1 — MLA per-head `w_uv` stride (128× serial launches)

`grim_mla_absorbed_decode` indexes `w_uv` without a per-head offset, so
`deepseek2.rs` launches once per head with `num_heads=1` (documented at
`gpu_absorbed_decode`). Fix: add the head stride to the kernel ABI and issue a
single multi-head launch. Gate: per-head parity old-loop vs new launch on the
dual-ROCm box. Note `mla_decode` is decode-only; MLA prefill stays host math
(Flash-MLA style prefill kernel is a separate follow-up).

## [NOW] P2 — `selective_scan_headed` on ROCm

falcon_h1 calls `BackendDevice::selective_scan_headed` (falcon_h1.rs:743);
ROCm implements only `selective_scan` (roc_device.rs:5657), so every decode
step falls back to the host loop. Implement the headed layout on top of the
existing kernel machinery. Gate: headed vs host-loop parity on GPU.

## [NOW] P3 — wire MoE bypassers onto shared MoeFfn/Charon

These build custom expert loops and never reach `grim_moe_fused_dispatch`:
`qwen35moe.rs` (own top-k loop, `cpu_tensor` out), `glm5_2.rs`, `glm4_moe_lite.rs`,
`hyv3.rs`, `bailingmoe3.rs`, and the big ones `deepseek2.rs`
(`DeepSeek2Expert`) + `qwen38_flash_next.rs` (`Qwen38MoeBlock`). Migrate the
small ones onto `MoeBlock`/`MoeFfn` first; DeepSeek/Qwen38 need shared-expert
and latent-layout care. Gate: logits parity vs the old path (CPU) + Charon
exercised on GPU.

## [NOW] P4 — sage_attention trait wiring

`RocmDevice::sage_attention_gpu` (roc_device.rs:15366) exists but
`impl AttentionOps for RocmDevice` has no `fn sage_attention`, so the trait
default (warn + F32 fallback) always runs. Implement the trait method as the
dispatch.

## Sequenced follow-ups (not this pass)

| Item | Evidence | Notes |
|---|---|---|
| Delta-rule / gated-delta-net device kernel | `delta_net_base.rs:222` "no device kernel for the recurrence" | Unblocks real qwen3next / kimi_linear / nemotron_hmoe / minimax_m2 (today thin Llama wrappers) |
| EAGLE-3 drafter to device | `eagle3.rs` builds everything via `cpu_tensor` | Sampler kernel exists; drafter is the CPU side |
| MTP fused step | `native_mtp.rs` cpu concat + `to_vec_f32` per token (:114–123, :165–174) | Kill per-speculative-token round-trips |
| LFM2 / mamba prefill scan | `lfm2.rs:436` host loop | selective_scan wired for decode only |
| T5 cross-attention + device ReLU | `t5.rs:116` host gap; cross_attention kernel whisper-only | |
| Gemma-3n completion | AltUp / Laurel / per-layer embeddings not implemented; GeGLU host loop `gemma3n.rs:272` | Architecture work before kernels |
| IQ1_S/M, IQ2_M, IQ3_M, IQ4_K | absent from `QuantFormat`/grim-quant entirely | Format → loader → GEMM arm, end-to-end |
| WNA16 packed GEMM | elementwise dequant only (`roc_device.rs:4989`, `moe.rs:509`) | Also unblocks quantized Charon experts without host inflation |
| Dead kernel wiring | sage/preshuffled/extend/mrope-3D have launchers, no model dispatch | mrope needs trait-level 3D-grid rope |
| NF4/FP4 GEMM arms | formats exist, no kernel arm | bitsandbytes-style |
| FP8 KV-cache attention | kernels take F32/packed-int KV only | DeepSeek FP8 latent cache |
| Audio/vision/diffusion device residency | vocos/kokoro/ViT/VAE/UNet all `cpu_tensor` | Largest surface, least serving-critical |
| fused_quant_gemm / flash_attention impls | trait defaults Unimplemented on ROCm | Tiled prefill (CK FMHA style) |
| charon_backward optimization | per-weight atomicAdd sites (kernel header) | Training-side |

## Hardware gates

Dual ROCm (gfx1201 + gfx1200) available: MLA, selective_scan, Charon, and
kernel-parity gates run here under `GRIM_RUN_GPU_TEST=1`. CUDA-host and
NVIDIA-only paths are compile-verified only.
