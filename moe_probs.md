# grim — Spec & Implementation Plan: MoE Subsystem, Laguna 2, and Verified Bug Backlog

Status: PARTIALLY IMPLEMENTED. WI-M0 (MoE primitives) is implemented and **compile- and test-verified**
in a Rust sandbox (`grim-nn` builds + `cargo test -p grim-nn moe` → 4/4 pass against
hand-computed expectations). WI-M1..M5 + WI-B1..B4 remain. Source references in this doc
were confirmed by reading the tree on 2026-08-09.

---

## 0. Headline correction to prior framing

The previous pass treated "Laguna 2 doesn't do MoE" as an isolated bug. It isn't.
Every one of the ~19 files under `grim-models/transformer/src/` with `moe` in the
name (`qwen2moe.rs`, `qwen3moe.rs`, `qwen3vl_moe.rs`, `qwen35moe.rs`, `glm4moe.rs`,
`granite_moe.rs`, `openai_moe.rs`, `phimoe.rs`, `bailingmoe.rs`, `lfm2moe.rs`,
`cohere2moe.rs`, `ernie4_5_moe.rs`, `ernie45_moe.rs`, `lladamoe.rs`, `afmoe.rs`,
`exaone_moe.rs`, `grovemoe.rs`, `hunyuan_moe.rs`, `nemotron_hmoe.rs`) plus `laguna.rs`
is the **identical 108-line dense-`Llama`-wrapper template**. `MoeConfig` (which
does carry real `expert_count`/`expert_used_count`, populated from GGUF headers at
4 call sites in `model_loader.rs`) is constructed and then discarded at every one
of these load sites. `grim-nn` has no router primitive, no expert-bank primitive,
no top-k dispatch primitive. `architecture.rs`'s HF→GGUF tensor map has no
expert-indexed tensor names (`ffn_gate_exps`, `ffn_down_exps`, `ffn_gate_inp`, etc.)
anywhere in the file.

**Conclusion: grim has no MoE inference implementation today, for any architecture.**
This plan is therefore scoped as "build the MoE subsystem once, correctly, then wire
~20 architectures to it" — not "fix Laguna." Laguna 2 is used as the acceptance-test
architecture because it's the most structurally demanding variant in the set
(sigmoid+bias router, shared expert, mixed attention pattern, per-layer head counts),
so building to Laguna's spec covers the easier architectures (standard softmax top-k,
uniform attention) as strict subsets.

---

## 1. Work item format (per project convention)

Each item: Why / Where / What-already-exists / What-to-build / Left-right-limits / Gates.
Gates ordered correctness → compile → architecture-cleanliness → performance (non-blocking).
New code paths land behind feature flags defaulting to current (dense) behavior where a
flag boundary is natural; where it isn't (e.g. a new required trait method), the fallback
is an explicit `Err(Unimplemented)` rather than silent dense execution, per the "fail
loudly, not silently" principle already established for the missing-tensor case.

---

## WI-M0 — `grim-nn`: MoE primitives (router, expert bank, dispatch)

**Why**: Nothing downstream can be correct until these exist. Every other MoE work
item depends on this one. This is the highest-leverage single item in the plan —
it turns ~20 fake architectures into real ones simultaneously.

**Where**: New module `crates/grim-nn/src/moe.rs`, exported from `grim-nn/src/lib.rs`.

**What already exists**: `Linear`, `ColumnParallelLinear`, `RowParallelLinear`,
`RmsNorm`, `Embedding` in `grim-nn` (used throughout `model.rs`/`block.rs`). No
expert-indexed weight loading; no router.

**What to build**:
1. `struct ExpertBank` — holds `Vec<Linear>` (or a single 3D-tensor-backed batched
   representation, see performance note below) for `{gate, up, down}` per expert,
   loaded via `ws.pp("ffn_gate_exps")` / `ffn_up_exps` / `ffn_down_exps`, indexed
   by expert id. Loading must read the GGUF 3D tensor `[n_experts, out, in]` layout
   directly rather than looping `n_experts` individual `get()` calls where the
   format supports it (check actual GGUF export shape for expert tensors before
   deciding; llama.cpp exports these as single 3D tensors, not per-expert 2D ones —
   confirm against a real Qwen3-MoE or Laguna GGUF header before implementing).
2. `enum RouterKind { SoftmaxTopK, SigmoidTopKWithBias { correction_bias: Tensor } }`
   — softmax top-k covers Qwen2/3-MoE, GLM4-MoE, Granite-MoE, etc.; sigmoid+bias
   covers Laguna (per HF docs: "Sigmoid MoE router with auxiliary-loss-free load
   balancing... router scores are the element-wise sigmoid of the gate logits plus
   a learned per-expert bias... added at selection time only").
3. `struct MoeRouter { gate: Linear, kind: RouterKind, top_k: usize, num_experts: usize }`
   with `fn route(&self, x: &Tensor) -> Result<(Vec<usize>, Vec<f32>)>` returning
   selected expert indices and combine weights per token. The bias-correction case
   must apply the bias **at selection time only, not at combine-weight time** — this
   is the exact place a naive port would get Laguna wrong, per the HF delta doc.
4. `struct MoeFfn { router: MoeRouter, experts: ExpertBank, shared_expert: Option<ExpertLinearTriple>, routed_scaling_factor: f32 }`
   with `fn forward(&self, x: &Tensor) -> Result<Tensor>` doing: route → gather
   selected experts' weights → weighted combine → add shared-expert output (if
   present) scaled by `routed_scaling_factor`.
5. CPU-backend dense reference forward (loop over selected experts, materialize
   each expert's contribution, weighted-sum) as the correctness baseline — this
   is what the parity test in WI-M4 checks GPU paths against. Explicitly **not**
   optimized (no batched-expert GEMM) in this item; that's WI-M5.

**Left/right limits**: This item does NOT touch attention (sliding window, per-layer
head count, softplus gating — those are WI-M2/WI-M3). It does NOT add a fused GPU
kernel (WI-M5). It only builds the router+expert-bank abstraction and a correct-but-
unoptimized CPU forward path, proven against a hand-computed reference in a unit test
using synthetic small weights (e.g. 4 experts, top-2, hidden=8) where the expected
output can be computed by hand and hardcoded in the test.

**Gates**: (1) correctness — unit test with hand-computed expected output for both
router kinds; (2) compiles cleanly as an isolated crate addition with no callers yet;
(3) `MoeFfn`/`MoeRouter` types are architecture-agnostic (no Laguna-specific or
Qwen-specific naming/assumptions baked in) so WI-M1/M2 can reuse them unmodified;
(4) performance — explicitly deferred to WI-M5, not blocking.

---

## WI-M1 — `architecture.rs`: expert-indexed HF↔GGUF tensor name mapping

**Why**: Without this, `WeightSource::get()` has no key to look up expert weights
under, regardless of what WI-M0 builds. This is the second hard blocker.

**Where**: `crates/grim-core/src/architecture.rs`, alongside the existing per-arch
`match` arms (e.g. the `ModelArchitecture::Laguna => { ... }` block at line 732).

**What already exists**: Dense per-layer name maps for every architecture (the
pattern read in the prior audit — `model.layers.{i}.mlp.gate_proj.weight` →
`blk.{i}.ffn_gate.weight`). Zero expert-indexed entries anywhere in the file.

**What to build**: For each MoE architecture, add mappings for:
- `model.layers.{i}.mlp.gate.weight` (or arch-specific router path) → `blk.{i}.ffn_gate_inp.weight`
- Expert-bank 3D tensors: `blk.{i}.ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`
- Router correction bias where applicable: `blk.{i}.exp_probs_b.bias` (verify exact
  GGUF key against a real converted Laguna checkpoint — do not guess the key name
  and ship it unverified; this is exactly the kind of unverified-claim pattern the
  project's verification discipline exists to prevent)
- Shared-expert weights where applicable: `blk.{i}.ffn_gate_shexp.weight` /
  `ffn_up_shexp.weight` / `ffn_down_shexp.weight`

Each architecture's expert count, top-k, and presence/absence of shared expert
differs — this must be table-driven per architecture, not copy-pasted 19 times
with silent per-arch drift (the copy-paste-108-lines pattern is exactly what
produced today's problem; do not repeat it at the mapping layer).

**Left/right limits**: Naming/mapping only. Does not validate that the tensors
actually exist in a given file (that's a load-time error surfaced naturally by
`WeightSource::get()`, per the existing fail-loudly behavior already confirmed
correct for the missing-tensor case).

**Gates**: (1) correctness — cross-check every new key against at least one real
downloaded GGUF header (`gguf-dump` or equivalent) per architecture family before
merging, not against memory of llama.cpp conventions; (2) compiles; (3) table-driven
structure prevents the 19x-copy-paste anti-pattern from recurring at this layer.

---

## WI-M2 — Per-architecture config structs: stop discarding MoE/attention fields

**Why**: This is the direct fix for the specific bug pattern found in `laguna.rs`
and all 19 `*moe.rs` files — real fields parsed from GGUF, then thrown away at
`load_tp`.

**Where**: `crates/grim-models/transformer/src/{laguna,qwen2moe,qwen3moe,...}.rs`
(19+1 files), plus a shared `MoeLlamaConfig`/`MoeLlamaBlock` introduced in
`grim-models/transformer/src/moe_model.rs` (new file) so the fix isn't another
19x copy-paste — it replaces the copy-pasted dense-wrapper template with a single
shared MoE-aware model implementation that each architecture's thin file configures.

**What already exists**: `MoeConfig` (has `expert_count`/`expert_used_count`, no
`top_k` distinct from used-count, no router-kind, no shared-expert flag, no
sliding-window/layer-type fields — needs extending, not replacing).

**What to build**:
1. Extend `MoeConfig` with: `router_kind: RouterKind`, `has_shared_expert: bool`,
   `shared_expert_intermediate_size: Option<usize>`, `routed_scaling_factor: f32`
   (default 1.0), `sliding_window: Option<usize>`, `layer_types: Option<Vec<String>>`,
   `heads_per_layer: Option<Vec<usize>>` (for Laguna's per-layer head-count variance).
2. New `MoeLlamaBlock` (parallel to existing `LlamaBlock` in `model.rs`) whose FFN
   sub-layer is `MoeFfn` (WI-M0) instead of the single dense `w_gate/w_up/w_down`
   triple, and whose attention sub-layer consults `sliding_window`/`layer_types`
   per-layer (this is new — see WI-M3) instead of assuming uniform global attention.
3. New `MoeLlama` model struct (parallel to `Llama`) using `Vec<MoeLlamaBlock>`.
4. Rewrite each of the 20 thin per-architecture files (`laguna.rs`, `qwen3moe.rs`,
   etc.) to build a `MoeConfig` with the correct `router_kind`/`has_shared_expert`/
   etc. for that architecture (this is real per-architecture research, not
   mechanical — e.g. Qwen3-MoE uses softmax top-k with no shared expert and no
   bias; Laguna uses sigmoid+bias with a shared expert; verify each against that
   architecture's HF `modeling_*.py` or technical report rather than assuming
   uniformity) and delegate to `MoeLlama::load_tp` instead of `Llama::load_tp`.

**Left/right limits**: This item does not change `Llama`/`LlamaBlock` (dense path
stays untouched — genuinely dense architectures like plain Llama, Mistral, etc.
keep using it unmodified). It does not implement the GPU-fused MoE kernel (WI-M5).

**Gates**: (1) correctness — per architecture, config fields populated match that
architecture's real spec (cite source per arch in code comments, per the project's
existing citation discipline seen in `q6k_gemm.rs`'s derivation comments); (2)
compiles; (3) the shared `MoeLlama`/`MoeLlamaBlock` abstraction means the next
new MoE architecture is a ~20-line config file, not another 108-line copy-paste;
(4) performance deferred.

---

## WI-M3 — Per-layer attention pattern (sliding window + global mix)

**Why**: Laguna 2 specifically requires this (36 sliding-window layers interleaved
3:1 with 12 global-attention layers, each with per-layer rotary scaling, plus
per-head softplus output gating). Several other architectures in the MoE list
also use hybrid attention patterns (verify per-arch; do not assume only Laguna
needs this — GLM4-MoE and others are known to use similar patterns and should be
checked, not assumed dense-uniform, before this item is scoped closed).

**Where**: `MoeLlamaBlock::forward` (new, from WI-M2) — the attention sub-block
needs a `window: Option<usize>` and `output_gate: Option<Linear>` field read from
`cfg.layer_types[i]`/`cfg.sliding_window`.

**What already exists**: Nothing — confirmed zero matches for "sliding_window" or
"layer_type" handling anywhere in `block.rs` in the current tree.

**What to build**: Masking logic that, when `window` is `Some(w)`, restricts
attention to the last `w` positions instead of full causal history; per-head
softplus gating applied to attention output before the output projection when
`output_gate` is present (`gate = softplus(W_gate @ x); out = gate * attn_out`
— verify exact formula against Laguna's HF modeling code, don't guess the gate
placement).

**Left/right limits**: Scoped to attention masking + output gating only. Does not
touch FFN/router (WI-M0/M2) or KV-cache paging changes beyond what's needed to
respect a shrunk window (paging logic in `grim-kvtransport`/`PagedKvCache` should
already support partial-window reads if it supports chunked prefill; confirm
rather than assume — this is a read-first item, not a rewrite of KV paging).

**Gates**: correctness (unit test: sliding-window mask matches hand-computed
attention scores for a small synthetic sequence longer than the window) →
compiles → doesn't regress existing uniform-global-attention architectures
(regression test: existing Llama/Qwen dense-attention golden outputs unchanged).

---

## WI-M4 — Correctness gate: MoE parity tests (CPU reference + real-weight smoke test)

**Why**: The project's own stated anti-pattern is "tests that assert only length,"
already partly fixed for Q3_K (real 2e-5-tolerance numeric parity test). MoE needs
the same treatment from day one, not retrofitted later.

**Where**: New `crates/grim-models/transformer/tests/moe_parity.rs` and, once a
GPU kernel exists (WI-M5), `crates/grim-backend-rocm/tests/moe_gemm_cpu_gpu_parity.rs`
following the exact structure of the existing `q3k_gemm_cpu_gpu_parity.rs` (real
numeric diff, explicit relative-error tolerance, printed for visibility).

**What to build**:
1. Synthetic small-model test: hand-constructed 2-4-expert MoE layer with known
   weights, verify router selection + combine weights + output against a hand-
   computed expected tensor (not just "output has the right shape").
2. Real-weight smoke test analogous to the existing `MiniCPM5-1B-Q4_K_M.gguf`
   real-model test in `grim-backend-cuda/src/lib.rs` (lines ~3919-4005) — download
   or reference a small real MoE GGUF (e.g. a small Qwen2-MoE variant) and diff
   grim's output logits against a known-good reference (llama.cpp CPU output for
   the same prompt, or the HF transformers reference implementation's logits for
   the first N tokens) with an explicit tolerance, not just "doesn't crash."
3. Router-specific tests: sigmoid+bias selection producing a different expert set
   than softmax top-k would for the same synthetic logits (proves the bias is
   applied at selection time only, per WI-M0's most error-prone detail).

**Left/right limits**: Test-only item. No production code changes. Blocks WI-M2/M5
from being marked done until these exist and pass.

**Gates**: correctness is the entire point of this item — it exists to make
correctness of WI-M0 through WI-M3 checkable, not to be checked itself.

---

## WI-M5 — GPU fused MoE GEMM kernel (ROCm primary, CUDA secondary) — PERFORMANCE, explicitly non-blocking

**Why**: WI-M0's CPU reference forward (looping materialized experts) will be
correct but slow — acceptable for correctness-gate purposes, not for real serving
throughput on a 256-expert model like Laguna S 2.1.

**Where**: New `crates/grim-backend-rocm/src/kernels/moe_gemm.rs`, following the
existing `q4k_gemm.rs`/`q5k_gemm.rs`/`q6k_gemm.rs` structure (derivation comments
citing the CPU reference, `#[cfg(test)]` source-contains-symbol smoke test at
minimum, real numeric parity test per WI-M4's pattern).

**What to build**: A batched/grouped GEMM that, given per-token expert assignments
(from WI-M0's router), performs the expert-selected GEMMs without materializing
every expert's full weight matrix per token — the standard "grouped GEMM" or
"expert-parallel" approach vLLM/others use. This is a substantial kernel-engineering
item in its own right; do not treat it as a copy-paste of the existing dense
fused-dequant-GEMM kernels (`should_use_wmma_path`/WMMA dispatch may be reusable
per-expert-GEMM once tokens are grouped by expert, but the grouping/sort step
itself is new).

**Left/right limits**: Explicitly gated as non-blocking performance work per the
project's own gate ordering (correctness → compile → architecture-cleanliness →
performance). WI-M0's CPU fallback path must remain correct and usable (behind a
feature flag or automatic fallback when no GPU-resident expert-grouped kernel is
available) so MoE architectures are at least functionally correct on ROCm before
this item lands, even if slow.

**Gates**: correctness (parity vs CPU reference, WI-M4 pattern, real hardware
required — `TODO(gpu-verify)` until run on real silicon per project convention) →
compiles → performance (measured tok/s on real hardware, not asserted).

---

## 2. Non-MoE bug backlog (from source re-verification this pass)

### WI-B1 — ROCm Q5_K fused GEMM parity test (currently absent)

**Why**: Prior notes claimed "219/256 wrong weights per block" for this kernel.
Hand-tracing this pass found the kernel's bit-addressing, scale sub-block indexing,
and nibble/high-bit selection match the CPU reference (`grim-quant::dequant_q5k`)
exactly. **This does not mean the kernel is correct** — it means the specific claim
could not be reproduced by static reading, and there is no test to settle it either
way (only Q3_K and Q4_K have CPU/GPU parity tests; Q5_K does not).

**Where**: New `crates/grim-backend-rocm/tests/q5k_gemm_cpu_gpu_parity.rs`, direct
copy of `q3k_gemm_cpu_gpu_parity.rs`'s structure (real numeric diff, explicit
tolerance) retargeted at Q5_K.

**What to build**: The test. Nothing in the kernel itself should be touched until
the test exists and either passes (closing the "219/256" claim as stale/already-
fixed) or fails with a concrete counterexample (reopening it with an actual
reproduction instead of a remembered figure).

**Gates**: correctness — this item's entire purpose is producing a verifiable
correctness signal. Must run on real ROCm hardware (`TODO(gpu-verify)` until then).

### WI-B2 — CUDA Q4_K `ShapeMismatch` — trace the caller, not `fused_quant_gemm` itself

**Why**: `fused_quant_gemm`'s own shape derivation (lines ~3020-3043 in
`grim-backend-cuda/src/lib.rs`) is correct on inspection and doesn't reproduce
the `[15,1536]` vs `[2048,1536]` symptom. The bug — if still live — is upstream,
in whatever builds `out_shape` before calling this function.

**Where**: Needs Syd's actual repro path (which model/layer/call site produced
the `[15, 1536]` vs `[2048, 1536]` error) — this cannot be chased further from
static source alone without that context. Flagging as **blocked on repro info**,
not closed.

**What to build**: Once repro'd, trace `out_shape` construction back to whatever
computes `15` (looks like a batch/row count) vs the expected `2048` (looks like
`out_dim`/`N`) — likely a transposed-dims or wrong-tensor-passed bug at the call
site, not in the GEMM dispatch itself.

**Gates**: Cannot proceed past "diagnose" until repro'd on real hardware with the
specific model/shapes involved.

### WI-B3 — Extend real-value parity testing beyond Q3_K

**Why**: Q3_K's parity test (2e-5 relative-error tolerance, forward + backward,
real numeric diff) is confirmed good and should be the template — but it's
unclear from this pass whether Q4_K's existing parity test (`q4k_matrix_core_parity.rs`)
and the WMMA test (`wmma_gemm_cpu_gpu_parity.rs`) use the same real-value-diff
standard or a weaker one. Confirm, don't assume, before calling the "length-only
assertions" anti-pattern fully closed project-wide.

**Where**: `crates/grim-backend-rocm/tests/q4k_matrix_core_parity.rs`,
`wmma_gemm_cpu_gpu_parity.rs`, `standalone_dequant_parity.rs`.

**What to build**: Read each file's actual assertions (not inferred from the
filename); upgrade any that are length-only or tautological to the Q3_K
numeric-diff pattern.

**Gates**: correctness — audit item, not a rewrite unless a gap is found.

### WI-B4 — `/v1/embeddings` — real encoder, or keep the honest 501

**Why**: Not a hidden bug — the `"model": "grim"` literal here is inside an
explicit `501 Not Implemented` stub with a clear error message, which is the
*correct* honest behavior (confirmed: all live completion paths correctly echo
`requested_model`). This item is a product decision, not a correctness fix:
either wire a real embeddings encoder, or leave the honest stub as-is. Listed
here only so it isn't mistaken for a regression of the already-fixed model-name-
echo bug in a future audit pass.

**Where**: `crates/grim-server/src/lib.rs:1592-1605`.

**What to build**: Nothing required. If embeddings support is wanted, that's a
separate, larger scope (encoder model support, likely BERT-family, which
`grim-models-vision`'s `Bert`/`ModernBertConfig`/`NomicBertConfig` suggests may
already be partially scaffolded — check before building new).

**Gates**: N/A — decision item, not a bug.

---

## 3. Sequencing

MoE work is strictly ordered WI-M0 → WI-M1 → WI-M2 → WI-M3 → WI-M4 (gate) → WI-M5.
M0/M1 can be developed in parallel (different files, no interdependency until M2
wires them together). M4's synthetic-weight tests should be written alongside M0
(TDD-style), not after — the real-weight smoke test in M4 is the true completion
gate for M0-M3 as a whole.

Bug-backlog items (B1-B4) are independent of the MoE track and of each other;
B1 and B3 are pure test-writing and can happen anytime. B2 is blocked on repro
info from Syd. B4 is a decision, not scheduled work.

**Recommended order if serialized**: WI-M0 → WI-M1 → WI-B1 (test-writing, low
cost, fills a real gap) → WI-M2 → WI-M3 → WI-M4 → WI-B3 (audit, low cost) → WI-M5
→ WI-B2 (once repro'd).

---

## 4. Open questions requiring your input before implementation starts

1. **Expert tensor GGUF layout**: is it 3D-batched (`ffn_gate_exps.weight` as a
   single `[n_experts, out, in]` tensor) or per-expert-indexed 2D tensors in the
   GGUF files you actually have? This determines `ExpertBank`'s loading code in
   WI-M0 and I don't want to guess it.
2. **Which real MoE GGUF do you have on disk** for the WI-M4 real-weight smoke
   test? (A small Qwen2-MoE or similar would be ideal — doesn't need to be Laguna
   itself for the first correctness gate.)
3. **Priority: breadth vs. depth** — do you want all ~20 architectures wired to
   the new `MoeLlama`/`MoeFfn` abstraction in WI-M2, or just Laguna 2 first as a
   proof point before committing to the other 19's per-architecture research?
4. **Laguna's exact router-bias GGUF key name** (`exp_probs_b.bias` is my best
   guess extrapolating from ggml naming conventions, not verified against an
   actual Laguna GGUF header) — needs confirmation before WI-M1 ships that mapping.
