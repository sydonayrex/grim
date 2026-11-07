# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-03

### Added
- Production readiness support for dual RDNA4 GPUs (`rocm:0` RX 9070 XT `gfx1201` and `rocm:1` RX 9060 XT `gfx1200`).
- Model tier validation matrix (`docs/models.md`):
  - **Green Tier (Full decode graph & hipGraph replay)**: Qwen2.5/3.5, Llama-3/3.1/3.2/3.3, Mistral, DeepSeek-V2/V3 MoE, StarCoder2, Gemma-2.
  - **Yellow Tier (Eager fallback / graph-refusal)**: LFM-2.5-Thinking (requires BOS token prepended in raw prompt mode), Xing4.0 (experimental YaRN positional embedding, eager execution).
  - **Red Tier (Explicit error rejection)**: Architecture unsupported by model loader.
- Memory ceiling and queue admission limits in `grim-server` (`GRIM_MAX_QUEUE=64`, `GRIM_REQUEST_TIMEOUT_SECS=300`).
- Wasmtime sandbox memory limits via `StoreLimits` (default 64 MiB cap per plugin).
- SHA-256 verification and versioned release packaging in `scripts/build-release.sh` and `dist/install.sh`.
- CPU CI workflow (`.github/workflows/ci.yml`) and `deny.toml` supply-chain license auditing.

### Changed
- Unified W4A4 GEMV default to v5 kernel, retiring fragmented legacy v2/v3 code paths.
- KV Arena ring-buffer wrap semantics verified stable under high cache churn.
- Default rustc toolchain pinned to stable (`1.98.0`) and aligned across `rust-toolchain.toml`, `Cargo.toml`, and installation documentation.

### Retired
- Deprecated undocumented runtime environment flags:
  - `GRIM_DEDUP_PARITY_DEBUG`
  - `GRIM_DEBUG_TRACE_KV`
  - `GRIM_TOTALS_TRACE`
- Retired default-on rollback switches and experimental kernel A/B flags (Phase 7):
  - `GRIM_WHITE_RAVEN_V2` (`grim_wmma_gemm_fp8_e4m3_blocked` fallback retired; `v2` is sole path)
  - `GRIM_Q5K_SUB` (`grim_dot4_q5k_q81_gemv` and `_ref` fallbacks retired; `_sub` is sole path)
  - `GRIM_DOT4_FAST` (`grim_dot4_q4k_q81_gemv_sub` fast path is sole path)
  - `GRIM_IQ_LDS` (`grim_fused_dequant_gemm_iq2s_lds` and `iq3xxs_lds` are sole paths)
  - `GRIM_WMM_QUANT` (`wmma_quant` routing is always enabled when applicable)
  - `GRIM_DEDUP_QUANT` (`linear_decode_into_prequant` deduplicated activation reuse is always enabled)
  - `GRIM_Q4K_TILED` (`grim_fused_dequant_gemm_q4k_backward_tiled` retired)
  - `GRIM_SCRUB_JAY_V2` (`grim_scrub_jay_gemv_v2` retired)
  - `GRIM_DOT4_256` (`grim_dot4_gate_up_silu_q80_f32act_gemv_256` retired)
  - `GRIM_WHITECROW_V2` / `GRIM_WHITECROW_V3` (`grim_dot8_w4a4_gemv_v2` and `v3` retired; `v5` is sole path)
