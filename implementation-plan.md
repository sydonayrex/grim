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

---

# Part III — Barriers remediation (WI-X series) · 2026-08-21 full-project review

Derived from the five-dimension review (performance, stability, inference
speed, training VRAM, usability). Same contract as Part II: verified symbols
only, per-WI done-check. Ordering: X9 (test-race) and X10 (CI) gate all GPU
claims; X1/X2 are the throughput compounders; X12/X13 unblock training.

Dependency map:

```
X9 (device-init race)  ── gates every GRIM_RUN_GPU_TESTS claim below
X10 (CI)               ── gates regression protection for all others
X2 (arena KV)          ── independent; compounds with X1
X1 (batched decode)    ── biggest single serving ceiling
X3 (GPU sampler)       ── independent of X1/X2
X4 (prefill MFMA)      ── profile-gated decision, needs X10 for honest numbers
X5 (attn autotune)     ── after X9 (needs GPU benching)
X6 (MXFP4 default-on)  ── gated on WI-E1 eval numbers
X7 (KV-quant reach)    ── independent bug-fix class
X8 (lock hygiene)      ── mechanical sweep, do early
X11 (kimi_k3 rope)     ── correctness, small
X12 (optimizers real)  ── training VRAM headline
X13 (grad checkpoint)  ── after X12 (shares tape surgery)
X14 (OOM backoff)      ── after X12
X15 (RCCL or guard)    ── independent
X16 (env consolidation)── mechanical but wide
X17 (doctor toolchain) ── smallest; do first of usability
```

Recommended execution order:
**X17 → X8 → X11 → X9 → X10 → X2 → X7 → X3 → X12 → X13 → X14 → X1 → X5 → X6 → X15 → X16 → X4**.

---

## WI-X17 — doctor toolchain check · usability, S

**Scope.** `grim doctor` verifies the JIT compile toolchain instead of failing
at first model load.

**Touch points**
- `crates/grim-cli/src/doctor.rs`: new `check_toolchain(report)` alongside
  `check_gpu_backend` — probe `clang --version`, `llvm-config`, ROCm libs
  (`libhipblas.so`/`librocblas.so` per README requirements), write access to
  the hsaco cache dir (`kernels/jit_cache.rs` disk-cache location), and warn
  when multiple rustup toolchains have touched `target/` (stale-artifact
  E0514 class).

**Done-check.** `grim doctor` on a machine missing clang prints an actionable
remedy line naming `pacman -S clang` (or distro equivalent) before any load
attempt. `cargo test -p grim-cli --lib doctor`.

## WI-X8 — lock-poisoning hygiene sweep · stability, M

**Scope.** Eliminate panic-on-poisoned-lock across hot crates: rocm 278,
server 153, engine 73, core 51 non-test `.unwrap()` sites; mutex locks become
`unwrap_or_else(|e| e.into_inner())` (pattern already used ~200× in
grim-server).

**Touch points**
- `crates/grim-backend-rocm/src/device/roc_device.rs`,
  `crates/grim-server/src/lib.rs`, `crates/grim-engine/src/lib.rs`,
  `crates/grim-core/src/**`.
- Mechanical rule: `.lock().unwrap()` → `.lock().unwrap_or_else(|e| e.into_inner())`.
  Non-mutex unwraps audited case-by-case; genuinely infallible ones get
  `.expect("invariant: …")` with the invariant named.

**Done-check.** `grep -rn 'lock().unwrap()' crates/{server,engine,backend-rocm,core}/src | wc -l` == 0.
Full workspace test pass unchanged.

## WI-X11 — kimi_k3 rope-history correctness · stability, S

**Scope.** The causal-mask fix landed in the crashed agent's partial edit;
the rope-key caching half must be verified or completed (deepseek2.rs got the
full fix — cache rows `[latent ‖ rope_key]`; kimi_k3 keeps nope/rope split).

**Touch points**
- `crates/grim-models/transformer/src/kimi_k3.rs` (~288–337 region): confirm
  history rope keys come from cache, not `&k_rope_v[0..]`.

**Done-check.** New `#[cfg(test)]` asserting decode step 2 attends with cached
rope keys (pattern: deepseek2 `test_mla_latent_attention_matches_uncompressed`);
`cargo test -p grim-models-transformer --lib kimi_k3`.

## WI-X9 — engine GPU test device-init race · stability, M

**Scope.** Root-cause the crash running `cargo test -p grim-engine` (observed
as a memory race / abort during concurrent HIP init). Hypothesis: multiple
tests construct `RocmDevice` concurrently while hipRTC JIT compiles; the hsaco
disk cache (`kernels/jit_cache.rs`, `device/jit_cache.rs`) does read-modify-
write without cross-process locking.

**Tasks**
1. Reproduce minimally: `cargo test -p grim-engine -- --test-threads=N` sweep;
   bisect to the pair of tests that collide.
2. Add file-lock (fs2/fd-lock or advisory flock) around hsaco cache dir
   writes in `device/jit_cache.rs`; serialize first-device-init behind a
   `OnceLock<Arc<RocmDevice>>` helper in tests.
3. Re-run full engine suite 10× clean.

**Done-check.** `for i in $(seq 10); do cargo test -p grim-engine || break; done`
— 10 consecutive green runs; then flip GPU suites from env-gated to
default-on for the shared runner (see X10).

## WI-X10 — CI matrix + eval gate · stability, L

**Scope.** One workflow beyond today's lone `.github/workflows/rust-clippy.yml`:
(a) ubuntu job: `cargo test --workspace --exclude grim-backend-vulkan`;
(b) self-hosted gfx1036 job: `GRIM_RUN_GPU_TESTS=1 cargo test -p
grim-backend-rocm --tests` + one WI-E1 eval smoke (ppl delta < 2% vs golden);
(c) clippy `-D warnings` on changed crates. Concurrency-cancel + 30-min
job timeout.

**Done-check.** Red CI on an intentionally seeded attention off-by-one within
one push; green on revert.

## WI-X2 — device-arena KV for unified attention · inference speed, M

**Scope.** `shared_attention::fused_or_scalar_attention` uploads the whole
K/V history host→device every call. Give it block.rs's arena discipline
(`cache_append_kv` D2D append at `block.rs:774–810`) so loaders pay O(new
tokens) H2D, not O(context).

**Touch points**
- `crates/grim-models/transformer/src/shared_attention.rs`: accept an
  optional `KvArena` (device K/V + past_len); fall back to full upload when
  absent. Mirror `block.rs` arena struct.
- Rewire the 17 batch-1 loaders' caches to hold the arena handle (cache
  structs already carry `k_dev/v_dev` fields where fused paths exist, e.g.
  `Lfm2LayerCache::Attention`).

**Done-check.** CPU parity tests unchanged-green; GPU counter (hipProfilerSDK
marker or existing perf_gate harness) shows H2D bytes/token flat in context
length; `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test
qkv_attention`.

## WI-X7 — KV-quant reachability fix · perf/stability, S

**Scope.** Engine comment at lib.rs:199 admits the real `LloydMaxCompressor`
path was left unreachable in some construction order. Audit both compressor
wiring branches; make int4/int8 paged-KV compression actually engage under
config, else delete the dead branch.

**Done-check.** Unit test constructing the engine with kv-quant config asserts
`kv_compressor.is_some()` and a round-trip compress/decompress on a synthetic
paged block (`grim-kvquant` tests as reference).

## WI-X3 — sampler on device · inference speed, M

**Scope.** Temperature/top-k/top-p sampling currently implies logits D2H per
token. Extend the zero-CPU argmax path (recent GPU-native sampler commit)
to the stochastic pipeline: GPU top-k filter + Gumbel/phoenix-style sampled
argmax, D2H only the chosen token id.

**Touch points**
- `crates/grim-core/src/sampler.rs` (`SamplerConfig.top_p/top_k`, lines 81–103):
  add a `Device` dispatch so greedy stays on GPU argmax and stochastic uses a
  new `sample_on_device` when logits are RocmStorage.
- `crates/grim-backend-rocm/src/kernels/`: one kernel (block-reduce max,
  filter, cumulative-sum via single wave when vocab ≤ wave lanes × chunks).

**Done-check.** Distribution equivalence test: CPU-reference multinomial vs
GPU path over 100k draws, chi-square within tolerance; ITL delta measured by
`grim-cli bench` on fixed prompt.

## WI-X12 — make PagedAdamW + AdamW8bit real · training VRAM, M

**Scope.** `adamw.rs` declares the variants (~156–174) but line 327 rejects
all but dense: *"declared but not yet implemented (Phase 7)"*. Implement two:

1. `AdamW8bit`: moment buffers as Q8_0 blocks (Q8_0 quant helpers already in
   `injection.rs:406`) — 2× optimizer-state reduction vs fp32 moments.
2. `PagedAdamW`: cold moment pages to host RAM behind a dirty-set; page-in on
   touch (design mirrors grim-memory crate's pool).

**Touch points**: `crates/grim-autograd/src/adamw.rs` (match arms),
`crates/grim-cli/src/train.rs` optimizer selection.

**Done-check.** Convergence parity: 500-step tiny-LM SFT, final loss within
3% of dense AdamW; peak VRAM delta logged via rocprofiler count; unit tests
for quantize/dequantize moment round-trip.

## WI-X13 — gradient checkpointing · training VRAM, L

**Scope.** Zero checkpointing exists (grep: none in tape/backward). Add
segment-wise recompute to `Tape`: mark segment boundaries every N layers,
free intermediate activations at segment end, recompute on backward.

**Touch points**: `crates/grim-autograd/src/tape.rs` (TapeEntry flag +
`drain_and_step` two-pass), `crates/grim-cli/src/train.rs` `--checkpoint-segs N`.

**Done-check.** Activation memory scales O(N_layers / segs) on a synthetic
deep MLP (assert via allocation counter); loss identical (bitwise tolerance
1e-6) to no-checkpoint run on same seed.

## WI-X14 — train-loop OOM backoff · stability, S

**Scope.** No OOM handling in `train.rs`. Wrap the step in HIP error
classification; on `hipErrorOutOfMemory`: halve micro-batch, retry once,
then abort with a message stating the largest successful batch.

**Done-check.** Forced-OOM test (absurd batch size) exits with the actionable
message, not a panic; `cargo test -p grim-cli --lib train`.

## WI-X1 — batched decode through the scheduler · inference speed, L

**Scope.** Biggest ceiling: server funnels every request through one
`Mutex<Engine>` with `&mut step_one(request_id, …)` per token; the wired
`Scheduler` (chunked prefill, admission control) can't co-schedule because
execution is single-owner. Move to grouped execution: scheduler admits N
requests; engine gathers their next tokens into one `[sum_steps, …]` forward
(paged KV already block-addressed; `qkv_attention` takes seq_len>1), splits
results per request. Server drops to short-scope locks: enqueue → drive-batch
→ collect, never holding the lock across tokenization/rendering.

**Touch points**
- `crates/grim-engine/src/lib.rs`: new `step_batch(&mut self, [(request_id,
  input_ids, positions); N])` beside `step_one` (which becomes N=1).
- `crates/grim-server/src/lib.rs`: request loop refactored onto a worker that
  drains the scheduler queue per tick (34 lock sites shrink to the worker).
- Sampling/splitting per-request after the joint forward.

**Done-check.** Two concurrent streaming requests show interleaved tokens and
combined tok/s ≥ 1.5× single-stream on gfx1036 (bench harness, fixed prompts);
per-request outputs byte-identical to solo runs (greedy).

## WI-X5 — attention autotune parity completion · perf, M

**Scope.** Only flash-decode split counts consult the autotuner today. Extend
`KernelKey` coverage to `grim_qkv_attention` launch geometry *within safe
axes* (grid-y = num_heads is fixed by correctness; tune LDS staging chunk and
wave-partition hint exposed as kernel args), plus paged variant tile. Bench
harness: `time_kernel_ms` sweep recorded into `.autotune_cache` (ADR §5).

**Done-check.** ≥5% mean decode-kernel time improvement on the bench shape set
or documented no-win; autotune entries persist across restarts.

## WI-X6 — MXFP4 default-on · perf, S

**Scope.** `GRIM_LFM2_MXFP4_QKV` env gate defaults off, so LFM2-family serves
F32 reference math. After WI-E1 golden numbers exist: flip default to on,
keep env escape hatch (`=0`), update loader docs.

**Done-check.** Eval suite ppl delta within tolerance vs F32 goldens; bench
shows expected tok/s uplift; gate removal noted in ADR 0001 §5 follow-ups.

## WI-X15 — RCCL all-reduce or honest single-GPU guard · training, M

**Scope.** Multi-GPU training silently falls back to CPU round-trip
all-reduce (`train.rs:879–886`). Either implement RCCL collectives via
`rccl.h` (ring all-reduce on f32 grads) or hard-error multi-rank configs
without RCCL with a clear message.

**Done-check.** 2-rank gradient agreement test (same seed → identical params)
through the RCCL path, or the explicit error on missing librccl.

## WI-X16 — env-var consolidation · usability, M

**Scope.** 71 distinct `GRIM_*` vars. Move behavior flags into `grim.toml`
(template exists at repo root) behind a typed `RuntimeEnv` reader
(`grim-core::env_config::RuntimeEnv::from_env` pattern already exists):
file value first, env override second. No behavior removal — unknown keys
warn once.

**Done-check.** `grep -rhoE "GRIM_[A-Z_0-9]+" crates | sort -u | wc -l`
drops to ≤25 (debug/test escapes remain env-only); docs/configuration.md
updated; `grim doctor` prints effective config source per key.

## WI-X4 — prefill attention: profile then decide · perf, L (gated)

**Scope.** No MFMA flash-class prefill kernel (ADR deferral). With X10's CI
bench in place: rocprof prefill shapes (512/2k/8k prompt, llama-class dims)
against current kernel; record in ADR 0001. Decision rule: if TTFT share of
prefill attention > 30% and an MFMA flash port wins ≥ 25%, open the port WI;
else close the question permanently.

**Done-check.** Numbers + decision recorded in ADR 0001 appendix; either a
new WI opened or the defer made permanent.
