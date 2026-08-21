# grim — Findings (improvement backlog)

Source: comparative review against `old/repos/sglang-main/` (see session notes).
Each finding is a gap to close; this file will later be converted into an
implementation plan. Ordered by leverage.

---

## FIND-1 · Accuracy benchmark suite

**Gap.** Quantization and kernel correctness currently rests on ad-hoc KATs
(kernel-vs-dequant comparisons) and one broken-until-this-week smoke bench.
There is no perplexity or task-eval harness to prove a quant path, a new
kernel, or a loader change preserves model quality. SGLang ships
gsm8k/hellaswag/boolq/perplexity suites in `benchmark/`; grim has none.

**Why it's #1.** Every quant kernel, GGUF loader workaround, and autotune
winner is currently unverifiable end-to-end. This protects all existing work
and gates everything below.

**Needed:**
- Perplexity eval on a fixed corpus (wikitext-2 class) runnable per model/quant.
- Task evals (gsm8k at minimum) via the local server API.
- Golden-number capture per (model, quant) pair; CI-comparable output.
- `grim-cli eval` subcommand + persisted results readable by `bench`.

**Acceptance shape:** `grim-cli eval --model <id> --task ppl,gsm8k` emits
stable numbers; a regression beyond tolerance fails.

---

## FIND-2 · Attention-kernel breadth

**Gap.** SGLang delegates attention to FlashInfer/FlashAttention/CUTLASS/
DeepGEMM — MLA, DSA, preshuffled KV, dual-chunk, hybrid-linear backends come
free. Grim hand-writes every attention path; current coverage centers on
standard MHA/GQA decode + prefill, with hybrid conv/recurrent handled per-model
(LFM2, Falcon-H1, Qwen35…). Missing breadth: MLA-class compressed-KV
attention, preshuffled/KV-block layouts for large batch, sliding-window
hybrids as a shared kernel rather than per-model loops, cross-attention
batching.

**Why it matters.** Attention is the dominant cost at long context and large
batch; it is also the largest hand-maintained correctness surface. Either grim
writes these kernels or defines a controlled delegation story — but the gap
must be closed deliberately, not model-by-model.

**Needed:**
- Inventory which attention variants the 139 supported models actually need;
  rank by usage frequency.
- Shared sliding-window / hybrid-attention kernels extracted from per-model
  scalar loops (lfm2.rs-style attention loops are the template).
- MLA-class path for DeepSeek-family loaders.
- Decision recorded: delegate (vendor hsaco) vs own-kernel per variant, with
  the same JIT-cache/autotune treatment as GEMM.

---

## FIND-3 · Throughput telemetry + parity numbers

**Gap.** Continuous batching, speculative decoding, and the self-tuning
scheduler exist, but there is no published tokens/sec parity measurement
against vLLM/SGLang on identical hardware (gfx1036), and acceptance-rate /
batch-occupancy telemetry isn't surfaced to users (new-finds.md F-2 partially
closed this; scheduler snapshot exists but no ITL/acceptance histograms).
The consumer-GPU niche claim is currently unmeasurable.

**Why it's #3.** It converts the project's core claim into a number, drives
autotune priorities with real data, and makes regressions visible.

**Needed:**
- Standard benchmark harness (`grim-cli bench --mode serving`) measuring
  tokens/sec and ITL percentiles under concurrency, using a real workload
  (sharegpt-length sequences), not synthetic tensors.
- Speculative-decoding telemetry: acceptance rate, draft length, net speedup
  exposed via `/api/stats` and logged per request.
- Head-to-head run script: same model/quant/hardware vs vLLM and SGLang,
  results stored in-repo (`docs/benchmarks/gfx1036.md`), re-run on kernel
  changes.

---

*Status: findings only — no plan commitments yet. Next step: convert each into
a WI-style implementation plan with scope, milestones, and verification.*
