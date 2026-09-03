# Performance Plan: Eliminate CPU↔GPU Round-Trips in grim-models/transformer

Status of shared path (done last session):
- `block.rs` — zero-copy rope, device KV append, cached block table. Done.
- `minicpm.rs` — rope rewritten zero-copy + device rope. Done.
- `grim-memory` — persistent per-layer KV buffers + dirty-region upload. Done.

What remains: the model-specific impls that bypass the shared machinery with
host-side math and unguarded `to_vec_f32()` calls in forward paths.

---

## Tier 1 — Structural fixes (biggest wins, few files)

### 1.1 lfm2.rs — MoE expert GEMMs on CPU (per token!)
Lines 928-931: `ffn_gate_exps.ffn_up_exps.ffn_down_exps.to_vec_f32()` pulls
ALL expert weight matrices to host on EVERY forward — then does scalar
dot-product loops in Rust (lines 944-960). This is the single worst pattern
in the crate: per-token multi-MB D2H + O(n_ff·hidden) CPU GEMM per expert.

Fix: keep experts as tensors; forward via `dev` matmul per expert only for
the winning expert (top-1) — or better, grouped GEMM. gate logits softmax
(lines 896-925) also host-side: small (n_expert) — acceptable, but the
weights must never cross. Target: zero to_vec_f32 in forward.

### 1.2 deepseek{32,4,2}.rs — identical host-math pattern (verified counts)
All three do manual host loops over attention/ffn with ~20 to_vec_f32 each.
Nearly identical code. Fix once, apply to three: extract the shared host
math into a single helper that takes device tensors and stays on-device.
If they share structure, one PR fixes 3 files.

### 1.3 qwen35.rs — attention + hybrid ssm host paths
to_vec_f32 at 363-418 in attention (q/k/v dump to host for CPU attention)
plus v1 probe paths. The working paged path exists in block.rs — route
qwen35 through the same shared helper instead of its bespoke CPU loop.

---

## Tier 2 — Unreviewed files with the same signature smell
Files with 18-22 to_vec_f32 / 0-1 from_cpu and no fallback guards —
strong signal they do full host-side forward loops:
- kimi_k3.rs, qwen35moe.rs, chameleon.rs, minimax_m3.rs, falcon_h1.rs,
  glm5_2.rs, qwen38_flash_next.rs
- Vision: qwen3vl.rs, qwen2vl.rs, hunyuan_vl.rs, cogvlm.rs

Action per file:
1. Read each forward; classify every to_vec_f32 as:
   (a) state read needed for Rust control flow — KEEP, measure,
   (b) weight/activation D2H for host math — REMOVE: replace CPU loop
       with on-device ops (relabel + Linear/matmul via dev),
   (c) debug/diagnostic — gate behind env flag.
2. Convention: forward paths must compile with zero to_vec_f32 except
   (a) — anything else is a bug.

### Lint guard
Add a CI check / crate lint that fails if a `forward*` fn contains
`to_vec_f32` not wrapped in a `debug_dump`-style helper. This prevents
regressions — the pattern is reintroduced easily during feature work.

---

## Tier 3 — Remaining shared-path polish
1. RoPE audit done; muse_glimmer already zero-copy; gemma/qwen35 mamba-style
   paths untouched (their own attention, works — don't churn).
2. Consider killing `to_vec_f32()` host verify in `weights_look_broken`
   entirely for large tensors — replace with a device reduce kernel later.
   (Sampling gate already landed.)

---

## Effort order
1. lfm2 MoE weights (Tier 1.1) — worst offender, clear fix.
2. deepseek trio share (Tier 1.2) — likely one patch for 3 files.
3. qwen35 attention routing (Tier 1.3).
4. Tier 2 sweep per-file with the lint guard.
