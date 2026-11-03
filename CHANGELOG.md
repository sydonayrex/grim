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
