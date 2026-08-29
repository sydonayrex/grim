# grim-core + grim-compressed-tensors audit verification & remediation record

Scope: crates/grim-core (session, kv_cache, model, sampler, catalog,
client, env_config, hyperparams, architecture) and
crates/grim-compressed-tensors, as of this verification.

## Claim-by-claim verification — grim-core

| Claim | Verdict | Action |
|---|---|---|
| No production-level logic bugs; unwraps confined to tests | **CONFIRMED** | — |
| `remap_hf_to_gguf` call sites do `.get(...).unwrap()` (panic risk) | **INCORRECT** | All three production sites (model_loader.rs:1310/2231, arch_compat.rs:267) use graceful `if let Some(mapped) = map.get(name)` with pass-through fallback. No unwrap exists. No action. |
| `gguf_name` ignores its `_arch` parameter (dead flag; wrong for non-Llama naming) | **CONFIRMED** | Documented: doc comment now states the Llama-family convention explicitly and directs non-standard architectures to `remap_hf_to_gguf`. The signature keeps `_arch` so call sites state their architecture and a future arch-aware rename is local. |
| RuntimeEnv::from_env / locate_config_file / effective_config_summary untested | **CONFIRMED** | **5 tests added** (env-var mutation serialized behind a mutex; edition-2024 `unsafe` blocks documented): defaults, env overrides incl. case-insensitive backend and list parsing, malformed-value fallbacks, toml parsing + env-over-toml precedence + `GRIM_CONFIG` location, and summary (key, value, source) rows. |
| `ArchHyperparameters::extract` untested; fallback chains unverified | **CONFIRMED** | **5 tests added** against a HashMap `MetadataLookup` mock: documented defaults, arch-specific keys beating llama.* fallbacks, llama.* fallbacks before defaults, the SmolLm2 stale-vocab special case (llama.vocab_size preferred over tokenizer.ggml.vocab_size), and GQA kv-head inheritance. |
| KvCache trait has no lightweight mock | **CONFIRMED** | **`MockKvCache` added** to grim-core::kv_cache (flat per-token rows; honors the tentative/commit/rollback speculative contract). grim-core now depends on grim-backend-cpu for the Tensor constructor — acceptable since every workspace consumer links it. Two contract tests. |
| `Graph::replay` stubbed | CONFIRMED, deliberate | It is an explicit, loud `Err(Unimplemented)` with an issue reference — the honest pattern. Implementing §4.3 graph replay is an engine feature, not an audit fix; unchanged. |

## Claim-by-claim verification — grim-compressed-tensors

| Claim | Verdict | Action |
|---|---|---|
| `from_tag` returns `fmt::Error` (wrong error type) | **CONFIRMED** | **FIXED** — all format errors now flow through a proper `GcctError` (Io / BadMagic / UnsupportedVersion / BadTag / Truncated / InvalidName / InvalidLength / UnsupportedLayout) with Display + std::error::Error. |
| No tests at all | **CONFIRMED** | **7 tests added**: tag round-trip + discriminant pins + unknown-tag rejection (0, 6, u32::MAX); container round-trip preserving name/type/metadata/data (incl. non-ASCII name, empty metadata); corruption gate (bad magic, future version, unknown in-stream tag, and EVERY truncation prefix); W8A8-Int8 per-channel exactness; E4M3 bit-pattern decode (zero, ±1, 2.0, 0.5, subnormal 2^-9, max-finite 448, NaN); Fp8 per-channel scale application; producer-owned-layout + bad-geometry rejection. |
| The container format doesn't exist (doc promises a reader) | **CONFIRMED** | **IMPLEMENTED** — `CompressedTensor` value type, `write_gcct` / `read_gcct` with magic + version check + per-tensor name/tag/metadata/data sections, strict length validation, and implausible-allocation guards. The doc's promise is now true. |
| No dequantizer dispatch | **CONFIRMED** | **IMPLEMENTED for the layouts this crate defines** — `dequantize_w8a8` covers `CompressedTensorsW8A8Int8` (signed per-channel int8 codes + f32 scales) and `CompressedTensorsW8A8Fp8` (OCP E4M3 decode + per-channel f32 scales), with the metadata layout (`num_channels u32, hidden u32`) specified in the module docs and `w8a8_metadata` as the writer-side helper. The producer-owned variants (`W8A8Mxfp8`, `WNA16`, `EmbeddingWNA16Int`) return an explicit `GcctError::UnsupportedLayout` — a loud refusal, not a guessed decode and not a silent stub; their payload layouts belong to their producers. |
| No consumers in the workspace | **STALE** | grim-tensor re-exports `CompressedTensorType` (lib.rs `pub use grim_compressed_tensors::CompressedTensorType`). The re-export compiles against the new error type unchanged (only `from_tag`'s error type changed, and grim-tensor does not call it). |

**Bug found by the new tests during the build-out:** the first Int8
dequantization implementation read codes as unsigned (`u8 as f32`), which
would have silently mis-decoded every negative weight — caught by the
per-channel exactness test and fixed to signed `(u8 as i8) as f32`.

## Verification

- `cargo check --workspace --tests` clean (grim-core's new
  grim-backend-cpu dependency introduces no cycle).
- grim-core: **36 passed, 0 failed** (was 24 — +12 new).
- grim-compressed-tensors: **7 passed, 0 failed** (was 0 tests).
- The crate remains zero-dependency (std only).

## Deliberate non-changes

- `gguf_name`'s ignored arch parameter: kept (documented) rather than
  removed — call-site churn for no semantic gain.
- `Graph::replay`: loud `Unimplemented` retained; real replay is the §4.3
  engine feature, tracked under its issue.
- grim-plugin's sampler-unit mismatch (dylib elements vs WASM bytes):
  documented at both call sites; a runtime check would need an ABI bump.
