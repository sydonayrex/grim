# GRIM-NN & GRIM-ENGINE AUDIT

Audit date: 2026-07-26
Scope: `crates/grim-nn/`, `crates/grim-engine/`, `crates/grim-models/transformer/`, `crates/grim-speculative/`, `crates/grim-core/src/`

---

## 🔴 CRITICAL BUGS

### CRIT-1: No causal masking in Llama attention — future token leakage

**`crates/grim-models/transformer/src/block.rs:169-219`**

`prefilled_self_attention()` computes **full bidirectional** self-attention over ALL tokens. Every query attends to every key, including future positions. For a causal LM this means token i sees token j > i during prefill, leaking future information into earlier positions.

```rust
// line 188 — NO causal mask applied
for t2 in 0..total_tokens {  // attends to ALL tokens, including future
    let mut dot = 0.0f32;
    for d in 0..cfg.head_dim {
        dot += qd[t * num_head_dims + h * cfg.head_dim + d]
            * kd[t2 * kv_stride + kvh * cfg.head_dim + d];
    }
    scores[t2] = dot * scale;
}
```

**Effect:** During multi-token prefill, logits at position i are contaminated by token i+1..N. Single-token decode is unaffected since no future tokens exist.

**Fix:** Restrict inner `t2` loop to `0..=t`, or apply `f32::NEG_INFINITY` to `scores[t2]` for all `t2 > t`.

---

### CRIT-2: No RoPE applied in Llama attention — position-agnostic

**`crates/grim-models/transformer/src/block.rs:103-167`**

`Rope::forward()` exists in `grim-nn/src/modules.rs:359-408` but is **never called** by `LlamaBlock::forward()` or `prefilled_self_attention()`. Q and K tensors are used directly from linear projections without rotary positional encoding.

**Effect:** Attention is fully position-agnostic. Token at position 0 has the same representation as position 1000 given the same embedding. The model cannot distinguish word order.

**Fix:** Apply RoPE to Q and K in `prefilled_self_attention()` before computing scores.

---

### CRIT-3: Rejection sampling OOB panic when verify > 1 token

**`crates/grim-speculative/src/speculative_wrapper.rs:210-238, 300-328`**

`self.target.forward()` is called with the original single-token `input_ids`, returning logits of shape `[1, vocab]`. The rejection loop then indexes `row_start = i * vocab_size` for `i >= 1`, which is **out of bounds**.

```rust
// line 222: row_start = i * vocab_size  — OOB for i >= 1 on [1, vocab] logits
let p_target = softmax_f32_row(&target_probs[row_start..row_end])[draft_tok];
```

**Effect:** Production crash on any speculative decode with K > 1 accepted tokens.

**Root cause:** The target model receives the original short input, not the draft-extended sequence. Speculative verification requires the target to run on the draft-prefixed sequence to produce per-position probabilities at the draft positions.

---

### CRIT-4: Speculative wrapper uses global `rand::random()` — breaks determinism

**`crates/grim-speculative/src/speculative_wrapper.rs:233,323`**

```rust
if rand::random::<f32>() < p_accept {
```

Uses the global RNG instead of the per-request seeded `DeterministicRng` stored in `engine.request_rng`. In `DeterminismMode::Strict`, this makes accept/reject decisions non-reproducible.

**Effect:** Same inputs produce different outputs across runs under strict determinism.

---

### CRIT-5: DSpark verification doesn't consume draft tokens

**`crates/grim-speculative/src/speculative_wrapper.rs:200-238`**

The DSpark path:
1. Draft proposes 3 tokens stored in `scored.tokens`
2. KV cache gets `tentative_append(verify_len)` — allocates space
3. Calls `self.target.forward(session, input_ids, positions, adapters)` with **original** single-token input_ids
4. Llama CPU forward **never reads KV cache** — recomputes attention from scratch for the 1-token input
5. Rejection sampling then compares draft probabilities at positions 0..K against target probabilities at positions 0 (only one position exists)

**Effect:** The "verification" compares mismatched positions. The comparison is meaningless.

---

### CRIT-6: `SpeculativeCausalLm::forward` hardcodes dummy scheduling params

**`crates/grim-speculative/src/speculative_wrapper.rs:360-368`**

```rust
fn forward(&self, session, input_ids, positions, adapters) -> Result<Tensor> {
    self.decode_one(session, input_ids, positions, 0.0, 0, adapters)
}
```

`live_gpu_utilization` and `batch_pressure` are always 0, bypassing dynamic verify-length selection in the confidence scheduler.

**Effect:** Confidence scheduler always selects max verify length regardless of system load.

---

### CRIT-7: `SpeculativeCausalLm::forward` wraps `decode_one` but `decode_one` returns target output not adjusted for rejection

Both NativeMtp and DSpark paths return `Ok(target_logits)` which are the logits from the target forward on the *un-extended* input. The output logits reported to the engine do not correspond to the tokens that were actually accepted. The server samples from logits that represent the original input position, not the verified output position.

---

## 🟠 MAJOR ISSUES

### MAJ-1: KV cache is structural dead code in CPU Llama path

**`crates/grim-models/transformer/src/model.rs:168-224`**

`Llama::forward()` never calls `session.append_kv()` or reads from `session.kv_mut()`. Every forward pass recomputes attention from scratch over all tokens. `session.advance_pos(seq_len)` is called, but the KV cache is never populated.

**Effect:**
- Hundreds of lines of `PagedKvCache`, `tentative_append`, `commit`, `rollback_to` infrastructure is dead code on CPU
- O(N²) compute per token instead of O(N) with KV cache
- GPU backends need a completely separate attention implementation

---

### MAJ-2: Engine builds 1D tensors, model contract expects 2D `[batch, seq]`

**`crates/grim-engine/src/lib.rs:341-348`**

```rust
let ids = cpu_tensor(
    input_ids.iter().map(|&t| t as f32).collect(),
    Shape::new(vec![prompt_tokens]),  // 1D shape
);
```

GPU backends typically expect `[1, prompt_tokens]`. The Llama CPU impl works with 1D by coincidence (`ids.len()`).

---

### MAJ-3: `Llama::forward` ignores `_positions` tensor, recomputes from 0

**`crates/grim-models/transformer/src/model.rs:177,207`**

```rust
let positions: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
```

The engine passes `current_pos` for decode, but Llama ignores it and uses `0..seq_len`. During decode at position 100, the model thinks the token is at position 0.

---

### MAJ-4: Dummy MoE dispatch in LlamaBlock FFN

**`crates/grim-models/transformer/src/block.rs:123-156`**

```rust
let num_experts = 2;
for t in 0..token_count {
    let expert_idx = t % num_experts;  // round-robin, not learned routing
```

Every token gets expert `t % 2`. This is pseudo-MoE alternating between two weight subsets. On standard (non-MoE) Llama, this forces alternate tokens through different FFN parameters, producing incorrect output. On real MoE models, learned router weights are ignored.

---

### MAJ-5: `prefilled_self_attention` O(N²) memory allocation per call

**`crates/grim-models/transformer/src/block.rs:187`**

```rust
let mut scores = vec![0.0f32; total_tokens];  // per query × per head
```

At N=4096 with 32 heads: 131K allocations of 16KB each in the inner loops. Extremely slow.

---

### MAJ-6: `LlamaBlock::forward` roundtrips through CPU for every token in FFN

**`crates/grim-models/transformer/src/block.rs:127-156`**

Each token is individually copied to a device tensor, processed through FFN, downloaded back to CPU, collected. Massive inefficiency even for a CPU reference impl.

---

## 🟡 MODERATE ISSUES

### MOD-1: `add_tensors` unconditionally stamps F32 dtype

**`crates/grim-nn/src/modules.rs:43-54`**

If input tensors are BF16/F16, output is stamped F32 but values are actually FP32. Minor metadata lie — backends dequant internally.

### MOD-2: Engine always reports `accepted_tokens: 1`

**`crates/grim-engine/src/lib.rs:428-431`**

```rust
accepted_tokens: 1,  // always 1
```

Even when speculation accepts >1 token. Comment says "deferred to phase 5 hardening." The server thinks each tick produces exactly 1 token.

### MOD-3: Self-tuning controller params only partially applied

**`crates/grim-engine/src/lib.rs:297-300`**

```rust
let tuned_params = self.self_tuning_controller.tune_all();
self.scheduler.max_batched_tokens = tuned_params.max_batched_tokens;
self.scheduler.chunked_prefill_size = tuned_params.chunked_prefill_size;
```

Other tunable parameters from `tune_all()` are silently discarded.

### MOD-4: TTFT/ITL recorded as schedule time, not actual

**`crates/grim-engine/src/lib.rs:295-296`**

```rust
self.self_tuning_controller.record_ttft(schedule_elapsed...);  // wrong metric
self.self_tuning_controller.record_itl(schedule_elapsed...);   // same value for both
```

Both TTFT and ITL use the scheduler scheduling time, not the actual forward-pass wall time. Also both are set to the *identical* value.

### MOD-5: `transpose_last_two` CPU roundtrip for GPU tensors

**`crates/grim-nn/src/modules.rs:155`**

`t.to_vec_f32()` downloads GPU tensor to CPU for transpose, then re-uploads. Could use a GPU transpose kernel.

### MOD-6: `pick_device_for_storage_device` silent CPU fallback

**`crates/grim-nn/src/modules.rs:35`**

```rust
_ => Box::new(CpuDevice::new()),
```

Unknown device types silently fall back to CPU, masking config errors.

---

## 🔴 TESTING GAPS

| Gap | Related Bug | File |
|-----|-------------|------|
| No causal-mask test | CRIT-1 | `block.rs` |
| No RoPE-in-forward test | CRIT-2 | `block.rs` |
| No spec-rejection test with real logits | CRIT-3, CRIT-5 | `speculative_wrapper.rs` |
| No deterministic-mode spec test | CRIT-4 | `speculative_wrapper.rs` |
| No KV cache read/write test for Llama | MAJ-1 | `model.rs` |
| No multi-turn conversation test | MAJ-3 | `engine/src/lib.rs` |
| No multi-token vs single-token parity test | CRIT-1 downstream | `engine/tests/` |
| `sleipnir_rocm_inference` tests LFM2, not Llama | Llama bugs invisible | `engine/tests/` |
| No position-tracking-across-ticks test | MAJ-3 | `engine/src/lib.rs` |

---

## VERDICT

**7 critical bugs**, **6 major issues**, **6 moderate issues**, **9+ testing gaps**.

The most impactful finding is **CRIT-1 + CRIT-2**: the Llama CPU reference implementation produces incorrect results for any multi-token prompt. The golden decode test in `sleipnir_rocm_inference.rs` escapes detection because it tests LFM2 (not Llama), and single-token decode doesn't trigger the causal-mask bug.

All speculative decoding machinery (DSpark, NativeMtp) in `grim-speculative` contains verification logic that is structurally broken — the target model never receives the draft tokens for verification, and rejection sampling panics on accept > 1.

The codebase needs a focused correctness pass on the Llama forward path (causal mask, RoPE, KV cache integration) before any GPU backend will produce meaningful results.