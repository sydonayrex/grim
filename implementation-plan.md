# grim — Implementation Plan

Derived from `findings.md` (triaged). Style: caveman — one path, concrete
numbers, real symbols only (every cited file/function verified to exist in the
working tree). Each WI has: scope, exact touch points, verification command,
and a done-check. No TBDs.

Ordering rule from findings: FIND-1 gates FIND-5b and de-risks everything
quant. FIND-3 needs FIND-1's harness for honest numbers. Everything else is
independent.

---

## WI-E1 — Eval harness (`grim-cli eval`) · FIND-1

**Scope.** One new subcommand, one new module, one corpus file. No new crates.

**Touch points**
- `crates/grim-cli/src/eval.rs` (new) with `pub async fn cmd_eval(model, task, output) -> Result<()>`.
- `crates/grim-cli/src/main.rs`: `Commands::Eval { model, task, output }` arm + dispatch (mirror `Commands::Scheduler`).
- `crates/grim-cli/src/lib.rs`: `pub mod eval;`.

**Tasks**
1. **PPL task.** Load model via `crate::catalog::resolve_model_path` +
   `grim_engine::model_loader::load_model_from_gguf` (same as `bench.rs:13`).
   Corpus = one `docs/eval/wikitext2.sample.txt` (2 MiB fixed slice, committed).
   Tokenize once via `GgufTokenizer::encode` (`crates/grim-format/src/tokenizer.rs:303`),
   slide window of 2048 tokens stride 2048, accumulate NLL of each window's
   last-token prediction through `model.forward`. ppl = exp(sum_nll / n_windows).
   Print `ppl=<f32> windows=<n>`.
2. **gsm8k task.** 100-question slice `docs/eval/gsm8k.test100.jsonl` (committed;
   question/answer fields). For each: POST `/v1/chat/completions`
   temperature 0, max_tokens 256 against `http://127.0.0.1:<port>` (server must
   already be running — same contract as `grim-cli adapter`). Grade: extract
   final number from completion, compare to gold after normalizing commas/$.
   Print `exact_match=<f32> correct=<n>/100`.
3. **Output JSON** to `--output <path>`:
   `{"model": "...", "task": "ppl|gsm8k", "metrics": {...}, "date": "<iso>"}`.
4. **Golden capture.** `grim-cli eval --model LFM2.5-350M-Q8_0 --task ppl,gsm8k
   --output docs/eval/baseline-lfm25-350m-q8_0.json` run once by hand on
   gfx1036; that file is the regression baseline.

**Verification**
- `cargo test -p grim-cli --lib eval` — unit tests: ppl math on synthetic
  logits (known answer), gsm8k grader on 5 canned pairs.
- Live: `grim-cli eval --model LFM2.5-350M-Q8_0 --task ppl` twice → identical
  ppl (deterministic, temperature 0).

**Done when.** Both tasks print stable numbers; baseline JSON committed.

---

## WI-E2 — Serving benchmark + spec-decode telemetry · FIND-3

**Scope.** Extend `cmd_bench`, add two counters to `/api/stats`. No new crate.

**Touch points**
- `crates/grim-cli/src/bench.rs` — add serving mode.
- `crates/grim-engine/src/lib.rs` — acceptance-rate tracking in
  `drive_decode_with_outcome` path (it already computes
  `session.last_accepted_tokens()` at lib.rs:874; surface it).
- `crates/grim-server/src/lib.rs` — `/api/stats` payload extension.

**Tasks**
1. **Serving mode.** `grim-cli bench --mode serve --concurrency N --port P`
   spawns N threads POSTing sharegpt-length prompts (fixed 20-prompt list in
   `docs/eval/prompts.txt`, 200–600 tokens) to a running server for 60 s,
   measures per-request wall time → tokens/s aggregate + ITL p50/p95/p99.
   Reuse reqwest client pattern from `crates/grim-cli/src/scheduler.rs:5`.
2. **Acceptance telemetry.** Engine already tracks
   `total_tokens_generated`; add `accepted_tokens_total` accumulated from the
   existing `total_accepted += o.accepted_tokens` in `tick()` (engine lib.rs,
   both decode loops). Expose `acceptance_rate() -> f64` =
   accepted/generated when a speculative strategy is active, else `null`.
3. **Stats surface.** Add to `/api/stats` JSON next to the F-2 scheduler block:
   `"speculation": {"accepted_rate": ..., "tokens_per_sec": tps}`
   (tps already present; wire the new field into the same json! block).
4. **Parity doc.** `scripts/parity-vs-vllm.sh`: runs grim bench serve mode,
   then equivalent vllm bench_serving command, writes both to
   `docs/benchmarks/gfx1036.md` table (manual paste; script prints commands).

**Verification**
- `cargo test -p grim-cli --lib bench_serve` — ITL percentile math on canned
  latencies (known p50/p95 answers).
- `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --lib` still green
  (no kernel changes here, but keep the gate habit).
- Live: bench serve mode against `grim-cli serve` on LFM2.5-350M-Q8_0 CPU
  backend prints non-zero tokens/s and finite percentiles.

**Done when.** One command produces a tokens/sec + ITL table; /api/stats shows
acceptance_rate during a speculative run.

---

## WI-E3 — Tokenizer fast path · FIND-4

**Scope.** Two techniques from gigatoken, applied only where measured hot.
Gated behind measurement per the finding itself.

**Touch points**
- `crates/grim-format/src/tokenizer.rs` — `encode()` (line 303) gets a batched
  sibling; `decode()` (line 590) unchanged.
- New `crates/grim-cli/src/eval.rs` (WI-E1) benefits automatically.

**Tasks**
1. **Measure first.** Micro-bench: encode the 2 MiB WI-E1 corpus, report
   ms + MB/s. If ≥ 50 ms (>2% of a 2 s TTFT budget at 32K ctx), proceed;
   else stop and record the number in this WI.
2. **Parallel pre-tokenize.** Split input on newline/paragraph boundaries into
   chunks, rayon `par_iter` map each chunk through the existing single-thread
   encode, concatenate ids in order. BPE merge across boundary is prevented by
   splitting only at `\n\n` (never mid-word) — document that invariant.
   Dep: add `rayon = "1"` to `[dependencies]` of `grim-format` Cargo.toml.
3. **Arena vocab lookup.** Replace per-token `HashMap<String,u32>` lookups in
   the encoder loop with a pre-built perfect-hash via `phf` OR keep HashMap but
   hoist out of the loop (whichever profile shows). One mechanism only — caveman.

**Verification**
- `cargo test -p grim-format --lib tokenizer` — round-trip parity: parallel
  encode output == serial encode output on 10 mixed-language samples.
- Before/after timing printed in the PR description.

**Done when.** Round-trip identical; measured speedup recorded (or WI closed
as "not hot" with the measurement).

---

## WI-E4 — Garage compress job (distill wiring) · FIND-5a

**Scope.** Small plumbing WI per triage. One job type, reuse distill.rs.

**Touch points**
- `crates/grim-garage/src/jobs.rs` — `JobType`/`TrainingMode` addition
  following `grim-adapter-optimizer` skill sequence (enum variant + match arms:
  `initial_loss`, `needs_model`, `is_sft_mode` ×2, scaled_loss arm).
- `crates/grim-speculative/src/distill.rs` — expose
  `train_speculative_draft` (exists, line 131) as the inner trainer; wrap with
  teacher=full-model, student=quantized-target config struct.
- `crates/grim-garage/src/ui_state/http_client.rs` + `poller.rs` — display strings.

**Tasks**
1. Add `TrainingMode::CompressDistill` variant (doc comment: "teacher→student
   distillation with quantized student target").
2. Job flow: load teacher GGUF (fp), load student = same arch quantized target
   (Q8_0 first), run K epochs of token-level KL distillation over
   `docs/eval/wikitext2.sample.txt` (reuse WI-E1 corpus), report ppl(teacher),
   ppl(student-pre), ppl(student-post).
3. Gate: requires WI-E1's ppl function — import it or factor ppl into
   `grim-cli/src/eval.rs` as `pub(crate) fn compute_ppl(...)`.
4. UI strings: "Compress Distill" / "COMPRESS-DISTILL".

**Verification**
- `cargo test -p grim-autograd --lib compress` if loss lives there; else
  `cargo test -p grim-garage --lib jobs` for match-arm completeness.
- Live smoke: 50-step distill run on LFM2.5-230M (teacher) → Q8_0 student,
  assert student ppl ≤ pre-distill ppl + 0.5 tolerance on gfx1036.

**Done when.** Garage can start a CompressDistill job and it completes with
the three ppl numbers logged.

---

## WI-E5 — MXFP4 QAT · FIND-5b (own WI, gated on WI-E1)

**Scope.** Straight-through-estimator fake-quant in training forward; final
real-quantize must bit-match fake-quant.

**Touch points**
- `crates/grim-quant/src/lib.rs` — add `fake_quant_mxfp4(weights: &[f32]) ->
  Vec<f32>`: quantize→dequantize round trip using the EXISTING
  `grim_quant::quant_mxfp4_matrix` (verified present, used by
  lfm2.rs build_fused_qkv_pack) so fake and real share one code path. This is
  the round-trip-match guarantee for free.
- `crates/grim-autograd/src/backward.rs` — no change needed if fake-quant is
  expressed as a forward-only op with identity gradient: implement as a
  custom autograd node whose backward is `grad.clone()` (STE).
- `crates/grim-garage/src/jobs.rs` — opt-in flag on QLoRA-family modes:
  `qat_mxfp4: bool` in the job config struct (nested field, not top-level
  Option explosion — per grim-format-plan metadata rule).

**Tasks**
1. `fake_quant_mxfp4` + unit test: for random f32 vec,
   `dequant(quant(v)) == fake_quant(v)` exactly (same code path).
2. STE node in autograd: `forward = fake_quant(x)`, `backward = identity`.
   Test: gradient flows through unchanged (compare vs linear node).
3. Garage wiring: when `qat_mxfp4`, wrap each Linear weight tensor in the
   fake-quant node before matmul during training; at save time, run real
   `quant_mxfp4_matrix` on the trained weights.
4. Validation (needs WI-E1): train LoRA+QAT on LFM2.5-350M base for 200 steps,
   convert to MXFP4 pack, run `grim-cli eval --task ppl`. Acceptance: ppl(QAT)
   ≤ ppl(post-ho MXFP4) − 0.1, else the QAT isn't paying for itself.

**Verification**
- `cargo test -p grim-quant --lib fake_quant` (round-trip exactness).
- GPU: `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --lib mxfp4`
  stays green (existing mxfp4_gemm_tests untouched).
- End-to-end ppl comparison per task 4.

**Done when.** QAT-trained MXFP4 beats post-ho MXFP4 on ppl by the stated
margin, or the negative result is documented with numbers.

---

## WI-E6 — Training fused kernels (fold into fusion plan) · FIND-6

**Scope.** Three candidates handed to the existing fused-kernel/megakernel
program. This WI only tracks the garage-wiring half of candidate #1, because
the ROCm kernel already exists.

**Facts verified in tree**
- `grim_fused_linear_ce_forward` + `grim_fused_linear_ce_backward` exist in
  `crates/grim-backend-rocm/src/kernels/fused_linear_ce.rs:21,104` with
  passing parity tests (`fused_linear_ce_parity_tests`, green under cubecl
  feature per ohsheet.md Phase A).
- NOT yet wired: no references to fused_linear_ce in grim-autograd or
  grim-garage (grep clean).

**Tasks**
1. **CE wiring (this WI).** In the garage SFT forward path
   (`jobs.rs` scaled_loss block), replace `logits = output.forward(h);
   cross_entropy(logits, labels)` with a call that routes through the fused CE
   when `device == Rocm` and hidden matches kernel constraint, keeping the
   materialized-logits path as CPU fallback. Pattern = grim-nn cuda-mem cfg-
   gated fast path with CPU fallback (grim-cuda-kernels skill wiring shape).
2. **RMSNorm-bwd + GEGLU-bwd kernels** — hand off to fusion plan as candidates
   with the standard contract: cos_sim ≥ 0.999 vs fp32 reference, measured
   speedup, zero regression on existing shapes (rocm-kernel-design gates).
3. **UX half (stays here):** `docs/recipes/lora-lfm25.yaml` recipe format +
   loader in garage; dataset registry `data/dataset_info.json` pattern with
   sha256 verify. Small, independent.

**Verification**
- `cargo test -p grim-garage --lib sft_loss` — loss value identical (≤1e-5)
  between fused and fallback paths on CPU-forced run.
- Fusion-plan items follow rocm-kernel-design ledger discipline
  (candidates.jsonl + benchmark.csv), not this file.

**Done when.** Garage SFT step runs the fused CE on ROCm with identical loss
to fallback; recipes loadable from YAML.

---

## WI-E7 — Packed integer CPU GEMM (q4_K/q8_0 dotprod) · FIND-8

**Scope.** CPU backend gets ggml-cpu-style packed dot products. Doubles as
GPU-kernel oracle (FIND-1 synergy).

**Touch points**
- `crates/grim-quant/src/lib.rs` — add `gemm_q8_0_packed(a_f32, b_q80_bytes,
  m, n, k) -> Vec<f32>` and `gemm_q4k_packed(...)` operating directly on the
  packed bytes (34-byte / 144-byte blocks per ggml layout reference in
  grim-quant-kernels skill).
- `crates/grim-backend-cpu/src/device.rs` — route quantized-storage matmul to
  these instead of dequant-to-f32-then-GEMM when `k % 256 == 0`.

**Tasks**
1. q8_0 packed GEMM: scalar reference first (correctness oracle), then AVX2
   path via `std::arch::x86_64::_mm256_maddubs_epi16` style accumulation,
   NEON path via `int16x8_t vmlaq`. Runtime-isa dispatch, one `#[cfg]` per
   target, scalar always compiled as fallback.
2. q4_K packed GEMM: same structure; layout per
   `references/q5k_ggml_layout.md` conventions (ggml-quants.c is authoritative
   — old/repos/llama.cpp-master/ggml/src/ggml-quants.c `dotprod` kernels).
3. Oracle lock-in: property test `packed_gemm(q) == dequant_then_gemm(q)` to
   rel-err < 1e-2 (quant noise floor), across k ∈ {256, 512, 1536} (must
   include blocks_per_row > 1 per the self-consistent-KAT pitfall).
4. Wire into CPU device matmul dispatch behind a storage-dtype check.

**Verification**
- `cargo test -p grim-quant --lib gemm_packed` — property tests above.
- Live oracle check: `GRIM_BACKEND=cpu grim-cli run LFM2.5-350M-Q8_0.gguf
  "What is the capital of France?"` → "Paris", tokens/s before vs after
  (expect 1.3–2× on the Q8_0 GEMM-bound portion; record actual).
- Cross-check vs GPU: same prompt GRIM_BACKEND=rocm output coherent.

**Done when.** Property tests pass at all three k values; live run faster and
still coherent.

---

## WI-E8 — Per-model tool-call detector registry · FIND-9

**Scope.** Map family → detector explicitly; retire template heuristic as
primary.

**Touch points**
- `crates/grim-server/src/tool_parse.rs` — `resolve_tool_family(template)`
  (line 40) becomes lookup: `family_for_arch(arch: &str) -> ToolFamily` with a
  static table `{lfm2: BracketFirst, llama: TagDelimited, qwen: TagDelimited,
  deepseek: BareJson}`; unknown falls back to today's heuristic then Auto.
- `crates/grim-server/src/lib.rs` chat_completions — thread the loaded model's
  arch (available via `/api/stats` catalog path) into parse_ctx tuple
  `(bool, Option<String>)` → extend to include family.

**Tasks**
1. Add `family_for_arch` + static table + unit test per family.
2. Thread arch through: engine knows arch via model_loader registration;
   simplest path — store arch string in AppState at model-load time alongside
   model_name (one Mutex<Option<String>>), read in chat_completions.
3. Session persistence (second half of FIND-9): save conversations to
   `~/.grim/sessions/<id>.json` on finish_request, restore endpoint
   `GET /v1/sessions/:id`. Defer if >1 day estimate — track separately.

**Verification**
- `cargo test -p grim-server --lib tool_parse` — family_for_arch table test.
- Live: tools request against LFM2.5 (bracket-first) and a Qwen GGUF
  (tag-delimited) both produce tool_calls when the model emits its native
  format.

**Done when.** Table drives parsing; heuristic only fires for unlisted archs.

---

## WI-E9 — grim-backend-tests crate (KAT centralization) · FIND-12

**Scope.** One new crate, seeded from existing scattered quant KATs. Ranked
by bug-class prevention per triage.

**Touch points**
- `crates/grim-backend-tests/` (new): Cargo.toml dev-deps on grim-tensor,
  grim-quant, grim-backend-cpu, grim-backend-rocm (feature-gated), 
  grim-backend-cuda.
- Move/centralize: existing per-backend parity tests
  (`test_cuda_dequant_q5k_gpu_matches_cpu` pattern from
  grim-backend-correctness skill; charon_wmma_parity.rs gating convention
  `GRIM_RUN_GPU_TESTS=1`).

**Tasks**
1. Crate skeleton: `tests/parity_cpu_rocm.rs` with the established env-gate +
   bail-without-env pattern (copy from
   `crates/grim-backend-rocm/tests/charon_wmma_parity.rs:31,173`).
2. Seed matrix: for each format in {Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL, MXFP4}
   × k ∈ {256, 1536} (blocks_per_row 1 AND >1, mandatory per
   grim-quant-kernels pitfall): quantize random vec (fixed seed) → GPU
   dequant → compare vs grim-quant CPU oracle, max_rel_err thresholds from
   grim-quant-kernels (Q6K low-single-digit max_abs sanity).
3. CI hook: add to `rust-clippy.yml` workflow a CPU-only job running
   `cargo test -p grim-backend-tests` (CPU oracle legs); ROCm leg documented
   as manual gate like today.
4. Op-bench: `benches/op_timing.rs` criterion benches emitting JSON per op
   (matmul f32, dequant per format) → feeds WI-E2 numbers.

**Verification**
- `cargo test -p grim-backend-tests` green on CPU legs in sandbox.
- `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-tests` green on gfx1036
  (real hardware per house rule — no CPU-only claim).

**Done when.** One crate answers "does format X behave identically on every
backend" for all six formats.

---

## Dependency graph & order

```
WI-E1 (eval) ──gates──> WI-E5 (MXFP4 QAT)
    │
    └──feeds──> WI-E2 (serving bench uses prompts/corpus)
WI-E3 (tokenizer)     independent, measure-first gate
WI-E4 (compress job)  needs WI-E1's ppl fn
WI-E6 (fused CE wire) independent
WI-E7 (CPU packed GEMM) independent; oracle for E1 confidence
WI-E8 (tool registry) independent
WI-E9 (backend-tests) independent; do early — cheapest bug-class kill
```

Recommended execution order: **E9 → E1 → E7 → E2 → E3 → E8 → E4 → E6 → E5**.
E9 first because it's the cheapest insurance on everything after; E1 second
because it gates E5 and validates E7.

## Verification contract (all WIs)

Per grim-format-plan + grim-rocm-ffi house rules:
- CPU-testable evidence: `cargo test -p <crate> --lib <topic>` — required.
- GPU numeric claims: `GRIM_RUN_GPU_TESTS=1 cargo test ...` on gfx1036 —
  required for any kernel-touching WI; never claim verified from CPU suite
  alone.
- Workspace gate where applicable: `cargo test --workspace --exclude
  grim-backend-vulkan`.
- No ad-hoc /tmp verify scripts (blocked pattern). Use the project's own
  harnesses.
