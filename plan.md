# Fix Implementation Plan — grim open weaknesses (@ cd7cbad8)

Companion to `grim_vs_competitors.md` §8 (verified 2026-08-22). Every item below targets a weakness confirmed by source audit at commit `cd7cbad8`. Ordering is by what produces correct behavior fastest.

**Guiding principles** (matching house style already in the repo):

- Every fix lands with a regression test named after the failure mode (precedent: `roofline_cost_compute_time_uses_peak_flops_not_bandwidth`).
- Prefer explicit rejection over silent under-delivery (`grim-constrain`'s unsupported-keyword pattern; the lomo/came optimizer errors).
- Extract-and-share rather than duplicate between garage and CLI.

---

## Wave 1 — correctness quick wins (small diffs, high trust payoff)

### F1. Kill the GaLore triple-alias · fixes weakness 9

- **Root cause:** `Optimizer::new` (`grim-autograd/src/adamw.rs:306`) maps `QGaLoreAdamW8Bit | GaloreAdamW | GaloreAdamW8Bit` to one constructor; the names imply three distinct precision/projector tradeoffs.
- **Fix:**
  - `GaloreAdamW` (non-quantized spelling) becomes `Err(Unimplemented("bf16 GaLore projector not implemented; use qgalore-8bit"))` — same contract as lomo/came.
  - `galore-8bit` keeps aliasing to `QGaLoreAdamW8Bit` (semantically defensible) but emits one `eprintln!` notice at construction.
  - Remove `galore` from the `--optimizer` help string in `grim-cli/src/main.rs`.
- **Tests:** assert `Optimizer::new(GaloreAdamW, ..)` errors; assert help text no longer lists it.
- **Effort:** ~1 hour.

### F2. Close the full-parameter application loop · fixes weakness 7 (core)

- **Root cause:** gradients reach `autograd_reg.params[ParamId::base(..)]` and the optimizer steps them, but `StreamingForward::block()` serves `LlamaBlock`s built once from provider weights and cached (`block_cache`); nothing copies stepped params back, so forwards after step 1 read original weights and cross-step learning cannot move the loss.
- **Fix (two pieces):**
  1. *Write-back:* add a `RegistryOverlayProvider` wrapping the GGUF provider whose weight lookup checks `autograd_reg.params` for `ParamId::base(layer, point)` first, falling through to the inner provider. In `cmd_train`, when `scope == FullParameter`, call `streaming.clear_block_cache()` after each `optimizer.step` and route subsequent `block(provider, ..)` calls through the overlay. (Cache-clear-per-step is fine at current scale; finer-grained per-Linear update is a later optimization.)
  2. *Consumption:* teach the sidecar→model path to use base-weight entries — extend `grim merge` (or model loading) so a `.grim.train` sidecar containing `ParamId::base` entries produces a full updated checkpoint instead of being treated as adapter-only.
- **Files:** `crates/grim-engine/src/streaming_forward.rs` (overlay + cache key), `crates/grim-cli/src/train.rs` (step loop), `crates/grim-format` merge path.
- **Tests:** a `FullParameter` variant of `toy_overfit_loss_decreases` asserting (a) loss drops materially and (b) — the regression guard — `block()` weights actually differ post-step; plus a merge round-trip test.
- **Effort:** 1–2 days. Converts `--mode full-bf16/full-fp16` from decorative to functional.

### F3. Make the disagg handoff failure-safe · fixes weakness 6

- **Root cause:** in the decode pull path, `write_keys` sets the received bit on the layer-0 write even if layers ≥ 1 fail; those blocks then attend stale pages silently forever. Separately, prefill pushes each handoff twice (per-layer slices + pool-level transfer).
- **Fix:**
  - Make received marking explicit: `KvBlockPool::mark_block_received(id, bool)`. The pull path sets it only after all layers succeed and marks `false` on any failure (retry next tick). The push-side KV receiver marks only when the expected layer count for the block has arrived (thread `num_layers` through `DisaggConfig`).
  - Delete the redundant pool-level `transfer_kv_cache_real` send from the engine prefill path, keeping the per-layer slice push.
- **Prove pure transferred-KV decode:** loopback test variant where the decode engine enqueues zero local prompt tokens — decode must attend solely from transferred KV (the current test runs a local 4-token prefill, so it proves transport only).
- **Files:** `crates/grim-memory/src/lib.rs`, `crates/grim-engine/src/lib.rs` (~856–935), `crates/grim-disagg/src/lib.rs`, `crates/grim-engine/tests/disagg_engine_loopback.rs`.
- **Tests:** pool semantics unit tests (partial layers ⇒ !received; all layers ⇒ received); failed-layer pull retry test; zero-local-prefill loopback.
- **Effort:** 1–2 days.

### F4. Optional auth middleware · fixes weakness 1

- **Fix:** `[server.auth] api_keys = [...]` in `grim.toml` + `GRIM_API_KEY` env; an axum middleware layer in `build_router` (`grim-server/src/lib.rs:4499`) enforcing `Authorization: Bearer <key>` on `/v1/*` and `/api/*`, exempting `/health*`, `/readyz`, `/metrics`; OpenAI-style 401 JSON error bodies. Default stays open (preserves the loopback posture shared with Ollama), but upgrade the non-loopback warning to also fire when the bind is public **and** no keys are configured.
- **Tests:** router test matrix — no keys set → open; keys set → 401 without / 401 wrong / 200 correct; health endpoints exempt.
- **Effort:** 0.5–1 day.

### F5. Endpoint honesty triage · fixes weakness 2 (staged)

- **Audio transcriptions/translations:** these handlers never read the request body.
  - Stage 1 (half day): return explicit **501-with-guidance** exactly like embeddings — instantly honest.
  - Stage 2 (separate WI): read uploaded audio, mel-spectrogram front-end, greedy decode loop driven by logits, detokenize. Test with two fixtures asserting different transcripts (guards against any future canned output).
- **Images/generations:** minimum fix (hours): hoist `Flux2VAE::random(...)` out of the per-request path into `DIFFUSION_MODELS` state and thread the prompt through an embedding path; if no text encoder is loaded, return 501 rather than unconditioned pixels. Full text-conditioning follows when an encoder model is loadable.
- **Garage studios:** tag responses with `"demo": true` and render a visible demo-mode badge when configs are randomly initialized; wire studio generation to `/v1/models/load` checkpoints when available.
- **Effort:** stage-1 honesty 0.5 day total; real pipelines are separate WIs (audio 2–3 days, image conditioning 1–2 days).

---

## Wave 2 — reachability (make tested code usable)

### F6. Wire EAGLE3/native-MTP into serving · fixes weakness 3

- **Serve path:** `grim serve --draft-model <path> --speculative eagle3|mtp|dspark|off`. After the target loads, the server resolves the draft spec and calls the already-tested `register_eagle3_model` / `register_native_mtp_model`; `--speculative off` selects Plain (also documents the default-on wrapper's behavior).
- **Loader path:** give `ModelArchitecture::Eagle3` a dedicated load arm producing `Arc<Eagle3>` (struct fields are public; straightforward), and detect native-MTP drafts from GGUF metadata (`nextn_predict_layers`, mirroring Ollama's approach).
- **Jump-forward-lite:** in the server sampling site, when `ConstrainedSampler::lookahead_literal()` returns `Some(literal)` and the constrained mask admits it, splice the literal and skip ahead — finally consuming the dead API.
- **Files:** `crates/grim-server/src/lib.rs`, `crates/grim-cli/src/main.rs` (serve args), `crates/grim-engine/src/model_loader.rs`, sampling site for lookahead.
- **Tests:** server integration test with tiny Llama + tiny Eagle3 file asserting `speculative_telemetry()` reports the Eagle3 strategy and nonzero acceptance counts over N requests; constrain test for literal splicing.
- **Effort:** 2–4 days.

### F7. Extract the preference trainer; make CLI modes real · fixes weakness 8

- **Root cause:** the correct four-forward DPO/KTO/SimPO/ORPO/GRPO implementation exists only inside `grim-garage/src/jobs.rs`; the CLI reimplements it badly (split-half logps, −0.05 reference offsets).
- **Fix:** move the preference core (pair dataloading, policy/reference forwards, `preference_loss_and_grads`) into a shared module (e.g., `grim-autograd::preference_trainer` or a small `grim-train-core` crate). Garage's worker and `grim-cli/src/train.rs` both call it. In the CLI, preference modes **require** a Preference-format dataset (the loader already parses pairs) and hard-error without one — deleting the synthesized-vector path entirely.
- **Tests:** CLI DPO on tiny GGUF + tiny pairs JSONL reaches loss < ln(2) and diverges from the SFT baseline; mode-without-pairs is an explicit rejection test.
- **Effort:** 2–3 days. This extraction is also the foundation for F8's data-parallel fix.

### F8. Real data-parallel for `--num-gpus`; resolve FSDP pretense · fixes weakness 10

- **DP:** reuse garage's proven pieces — `fork_for_rank` (exists in the registry), deterministic shard iterator, weighted RCCL all-reduce — via the F7 shared trainer. CLI spawns one replica per `rocm:N` device, shards batches, syncs; adapters stay per-rank-identical by construction.
- **FSDP decision:** the current `execute_all_gather`/`execute_reduce_scatter` are single-process simulations. Either bind them to real `ncclAllGather`/`ncclReduceScatter` in `rccl.rs` (real work — schedule separately) or rename the module `fsdp_planner.rs` and drop the executor-sounding API so it stops implying capability it doesn't have. Recommend the rename now, real FSDP later.
- **Advanced trainers:** with the shared trainer in place, register SoulEater/Scythe1 as genuine `TrainingMode` branches calling their real implementations (each with a tiny-model loss-decreases integration test). Distillation stays explicitly rejected until a teacher-forward path exists.
- **Effort:** DP 2–3 days riding F7's refactor; trainer wiring 1–2 days.

---

## Wave 3 — depth/performance

### F9. Grammar-engine-grade constrained decoding · fixes weakness 4

- Compile JSON Schema once into an FSM whose states carry precomputed vocab masks (same architecture as JsonObject mode's `TokenMaskCache`), replacing per-novel-prefix O(V) serde validate; keep the current validator as an oracle for differential fuzzing ("never mask a valid prefix").
- Replace the `pattern` special-case heuristic with a bounded backtracking regex subset (`^ $ [ ] {m,n} . literal`, ~200 lines), falling back conservatively for unsupported constructs.
- Generalize `lookahead_literal` into schema-driven jump-forward (single-admitting states: enum values, forced key names).
- **Benchmark gate:** tokens/sec on a `json_schema` workload must beat the current memoized-mask baseline before merge.
- **Effort:** 4–6 days — the largest serving item; xgrammar-class behavior is the target, parity not required initially.

### F10. Device residency + backend parity depth · fixes weakness 5

- **KV residency (staged):**
  - Step 1 — replace whole-layer per-step uploads with a dirty-block device mirror (upload only blocks touched this step; ROCm/Vulkan paged-attention kernels already consume page tables).
  - Step 2 — graduate to fully device-resident pools behind the existing tiering API.
  - Gate each stage on ITL benchmarks.
- **CUDA graphs:** port the ROCm `graph_capture.rs` pattern to `grim-backend-cuda` (bucketed batch shapes, capture decode `step_batch`). Effort 2–3 days.
- **Hardware parity:** execute the existing Vulkan/Metal numerical suites on real adapters (gfx1036 iGPU already used for device-gated tests elsewhere; Metal needs a mac runner) instead of host-referenced checks only. Effort ~1 day + CI.
- **Total effort:** KV 3–5 days staged.

### F11. Docs truthing · fixes weakness 11 (do last, after F2/F7 land)

- Rewrite `docs/howto/train-adapter.md`'s mode table to describe actual dispatch; document the auth option (F4), draft-model flag (F6), and full-parameter semantics (F2); sync CLI help strings.
- **Effort:** half a day, repeated cheaply at each wave boundary.

---

## Sequencing and rationale

| Wave | Items | Why this order |
|---|---|---|
| 1 | F1, F2, F3, F4, F5(stage-1) | Small diffs; eliminates every "silently wrong result" surface (full-param inertness, stale disagg pages, fake endpoints, hidden aliasing) |
| 2 | F7 → F6, F8 | F7's trainer extraction unblocks both real CLI preference modes and true DP; F6 is independent |
| 3 | F9, F10, F5(stage-2) | Performance/depth arcs with benchmark gates; largest effort |

## Structural notes

- **F7 is the keystone** — extracting the shared trainer pays off three times: CLI preference correctness, true data-parallel reuse (F8), and advanced-trainer wiring.
- The recurring failure pattern behind this list ("implemented and tested but unreachable", "aliased", "simulated") argues for one standing convention going forward: **no CLI/config value may be accepted unless it changes behavior or errors loudly.** F1's rejection pattern generalized.
