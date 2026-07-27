# Fix Plan: grim-tensor-graph and grim-models crates

## Overview

The crates compile and all 15 tests pass, but the code is in an early structural phase with significant dead code, debug output in production, duplicated logic, and incomplete computations. This plan is organized by crate and severity.

---

## grim-tensor-graph

### High Priority

**1. Remove unused `thiserror` dependency**
- File: `grim-tensor-graph/Cargo.toml:12`
- `thiserror = "1"` is declared but never used (no `#[derive(Error)]` or `use thiserror` anywhere)
- Fix: Delete the line

**2. Remove unused `OpType` variants**
- File: `grim-tensor-graph/src/ir.rs:10-18`
- `OpType::Silu` and `OpType::Gelu` are defined but never matched in `identify_fusion_sequences`
- Fix: Delete `Silu` and `Gelu` from the enum

**3. Remove `TensorNode` wrapper**
- File: `grim-tensor-graph/src/lib.rs:7-9`
- `TensorNode` wraps a single `String` field with no additional behavior
- Fix: Use `String` directly in `TensorGraphIr.nodes`

**4. Remove redundant `ComputationGraph::new()`**
- File: `grim-tensor-graph/src/ir.rs:48-50`
- `new()` just calls `Self::default()` — users can call `default()` directly
- Fix: Delete the `new()` method

### Medium Priority

**5. Unify the two parallel IR representations**
- `TensorGraphIr` in `lib.rs` and `ComputationGraph` in `ir.rs` are redundant
- `TensorGraphIr` is checkpoint-name-based; `ComputationGraph` is node-based
- Fix: Merge into a single IR, keeping the richer `ComputationGraph` design

**6. Refactor `identify_fusion_sequences` to reduce duplication**
- File: `grim-tensor-graph/src/ir.rs:71-112`
- The "clear current, push, push candidate, clear" pattern is duplicated between RmsNorm+MatMul and QkvProjection+AttentionScore arms
- Fix: Extract a helper that handles the "sequence complete" logic

---

## grim-models/transformer

### High Priority

**7. Remove dead `rope` field from `Llama`**
- File: `grim-models/transformer/src/model.rs:47-48`
- `rope: Rope` is constructed but never used in `forward` (`#[allow(dead_code)]`)
- Fix: Delete the field and its construction

**8. Remove debug `println!` from production code**
- File: `grim-models/transformer/src/block.rs:129`
- `println!("[MoE Router] Routing token {} to Expert {}", t, expert_idx)`
- Fix: Delete the line

**9. Remove dead `LoRAWeights` and `align_tensor_for_rocm_gemm`**
- File: `grim-models/transformer/src/lora.rs:234-331`
- `LoRAWeights` struct, `LoRAWeights::load_for_rocm`, and `align_tensor_for_rocm_gemm` are defined but never called anywhere in the codebase
- Fix: Delete all three

**10. Remove dead `transpose_last_two`**
- File: `grim-models/transformer/src/lora.rs:199-223`
- Only used in the dead GPU path of `apply_adapters_to_logits`
- Fix: Delete it

**11. Fix Gemma `forward` to use computed hidden state**
- File: `grim-models/transformer/src/gemma.rs:155-158`
- `forward` computes `h` through layers but returns `logits` from a separate embedding lookup, ignoring `h`
- `let _ = h;` discards the computed hidden state
- Fix: Use `h` for the output projection instead of re-embedding input tokens

### Medium Priority

**12. Extract shared `add_tensors` helper**
- Files: `gpt2.rs:188`, `gemma.rs:164`, `deepseek.rs:170`, `t5.rs:178`, `lfm2.rs:446`, `rwkv.rs:180`, `whisper.rs` (implicit)
- 7 copies of the same `add_tensors` function across model files
- Fix: Add `add_tensors` to `grim-backend-cpu` or `grim-core` and replace all copies

**13. Extract SwiGLU helper in `block.rs`**
- File: `grim-models/transformer/src/block.rs:138-170`
- SwiGLU computation is duplicated between expert 0 and expert 1; only difference is `* 0.95`
- Fix: Extract `fn swiglu(gate: &[f32], up: &[f32], scale: f32) -> Vec<f32>`

**14. Remove unused variables in `block.rs`**
- File: `grim-models/transformer/src/block.rs:102`
- `_dims` computed but never used
- File: `grim-models/transformer/src/block.rs:180`
- `_dev` in `prefixed_self_attention` is unused
- Fix: Delete both

**15. Optimize GPU path in `apply_adapters_to_logits`**
- File: `grim-models/transformer/src/lora.rs:53-153`
- GPU path copies delta to host, scales, copies back to device — can be done entirely on-device
- Fix: Scale on-device using `dev.mul` or equivalent, then add on-device

**16. Remove unused variables in `vit.rs`**
- File: `grim-models/vision/src/vit.rs:107`
- `let _ = block_in;` suppresses warning
- Fix: Delete the line

---

## grim-models/vision

### High Priority

**17. Implement or remove unused attention weights in `VitBlock`**
- File: `grim-models/vision/src/vit.rs:46-58`
- `VitBlock` stores `wq, wk, wv, wo` but doesn't use them in `forward`
- `let _ = (&self.wq, &self.wk, &self.wv, &self.wo)` suppresses the warning
- Fix: Either implement attention or remove the unused weights

**18. Fix double position embedding in ViT**
- File: `grim-models/vision/src/vit.rs:252-267`
- Position embedding is added in the projection loop (line 258) and again in a separate `for_each` (lines 260-266)
- Fix: Remove the duplicate addition

---

## grim-models/audio

### High Priority

**19. Remove unused fields from `WhisperDecoderBlock`**
- File: `grim-models/audio/src/whisper.rs:110-127`
- `_self_o`, `_cross_q`, `_cross_v`, `_cross_o`, `_ffn_norm` are stored but never used
- Fix: Delete the unused fields

**20. Remove discarded results in `whisper.rs`**
- File: `grim-models/audio/src/whisper.rs:282`
- `let _ = self.enc_norm.forward(&cur)?;` discards the encoder norm result
- File: `grim-models/audio/src/whisper.rs:281`
- `let _ = dev;` discards the device
- Fix: Delete both lines

---

## grim-models/diffusion

### High Priority

**21. Fix `DownBlock::forward` convolution bug**
- File: `grim-models/diffusion/src/unet.rs:66-84`
- `weights[((i % h) * h) + k] * prev[i]` uses `prev[i]` instead of `prev[k]` — the convolution doesn't actually convolve
- Fix: Change `prev[i]` to `prev[k]` (or the correct index for the convolution)

**22. Fix `sinusoidal_timestep_embed` formula**
- File: `grim-models/diffusion/src/unet.rs:144-154`
- `(-((i as f32) * 2.0 / half as f32).exp())` applies `exp` to the frequency instead of the angle
- Correct formula: `let freq = 1.0 / (self.rope_theta.powf((2 * i) as f32 / self.head_dim as f32));` then `let angle = pos * freq;`
- Fix: Rewrite the formula to match standard sinusoidal embedding

**23. Remove unused variables in `DownBlock::forward`**
- File: `grim-models/diffusion/src/unet.rs:81-82`
- `let _ = weights; let _ = bias;` suppresses warnings
- Fix: Delete both lines

---

## grim-models/mamba

### High Priority

**24. Implement RWKV time-mix computation**
- File: `grim-models/mamba/src/rwkv.rs:82-99`
- `let _ = (k, v, r)` discards computed attention projections
- The RWKV time-mix (the core computation) is not implemented
- Fix: Implement the actual time-mix: `y = wv * (k * v) + x` with time-mix decay

**25. Remove `KvBlockPool` mock from `Mamba::step`**
- File: `grim-models/mamba/src/lib.rs:332-337`
- Creates a new `KvBlockPool` on every call with hardcoded `request_id = 999`
- This is a mock that should be replaced with real state pool integration or removed
- Fix: Remove the mock pool lookup; use the state passed in `MambaState`

**26. Remove speculative config structs**
- File: `grim-models/mamba/src/configs.rs:7-116`
- 6 config structs (`Rwkv6Config`, `Rwkv7Config`, `Mamba2Config`, `JambaConfig`, `NemotronHConfig`, `GraniteHybridConfig`) with no implementations
- `Mamba2Config` duplicates `MambaConfig` fields
- Fix: Delete all 6 structs until implementations exist

### Medium Priority

**27. Remove unused `d_param` and `dt_bias` in `MambaBlock`**
- File: `grim-models/mamba/src/lib.rs:106-107`
- `d_param` and `dt_bias` are stored but `step_block` doesn't use `dt_bias` for the SSM update
- Fix: Either implement their use or remove them

---

## Cross-cutting

### High Priority

**28. Extract shared activation helpers**
- `silu_mul` is duplicated in `deepseek.rs:177`, `lfm2.rs:453`
- `geglu` is in `gemma.rs:171`
- `relu` is in `t5.rs:185`
- Fix: Add `silu_mul`, `geglu`, `relu` to `grim-backend-cpu` or `grim-nn` and replace all copies

**29. Remove unused `conv` field usage in `MambaBlock`**
- File: `grim-models/mamba/src/lib.rs:104`
- `conv: Vec<f32>` is stored but `step_block` doesn't use conv1d (comment says "skipped in v1")
- Fix: Either implement conv1d or remove the field

---

## Summary

| Priority | Count | Key Actions |
|----------|-------|-------------|
| High | 14 | Remove dead code, fix bugs, remove debug output, implement missing computations |
| Medium | 6 | Extract shared helpers, optimize GPU path, remove unused variables |
| Cross-cutting | 2 | Extract shared activation helpers, remove speculative configs |

**Estimated lines saved: 350+**
**Dependencies removed: 1 (`thiserror` from grim-tensor-graph)**
**Duplicated functions eliminated: 8 (`add_tensors` x7, `silu_mul` x2)**
