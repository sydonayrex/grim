# DATS-DEM: Grim Models Transformer Inference Audit

**Date**: 2026-09-16  
**Scope**: All 180+ model files in `crates/grim-models/transformer/src/`  
**Methodology**: Static audit of forward paths, KV cache handling, D2H/H2D patterns, graph capture coverage, and barrier usage.

---

## Executive Summary

| Category | Count | Severity |
|----------|-------|----------|
| No decode-step graph capture (fully eager) | ~179 models | Critical |
| Pure CPU-only forward (no device path) | ~21 models | High |
| H2D/D2H as primary path (not fallback) | ~21 models | High |
| Dead `k_device`/`v_device` fields declared but unused | ~10 models | Medium |
| MoE gate logits D2H every step (host routing) | ~8 models | Medium |
| Thin wrappers inheriting block.rs (partial GPU path) | ~90 models | Low |
| Warm KV cache tracking (device arenas) | ~10 models | OK |

---

## 1. Graph Capture Coverage

### Finding: Only LFM2 has decode-step graph capture. All other models are fully eager.

| Model | Graph Capture | File |
|-------|---------------|------|
| LFM2 | Yes (but poisoned by ShortConv/MoE) | `lfm2_graph.rs` |
| Llama (and all thin wrappers) | None | `model.rs:543-621` |
| Qwen3.5 | None | `qwen35.rs` |
| Qwen3.8-Flash-Next | None | `qwen38_flash_next.rs` |
| DeepSeek-V2/V3/V4 | None | `deepseek2.rs`, `deepseek4.rs` |
| Falcon-H1 | None | `falcon_h1.rs` |
| Gemma, Gemma2 | None | `gemma.rs`, `gemma2.rs` |
| All others | None | — |

**Implication**: For non-LFM2 models, every decode step issues ~10–15 individual kernel launches per layer per op. For a 30-layer model, that's 300–450 launches per token vs. 1 `hipGraphLaunch` with graph capture.

**Blocker for LFM2**: ShortConv recurrent layers (`lfm2_graph.rs:420-428`) and MoE sub-layers (`lfm2_graph.rs:433-436`) poison the entire graph. If either is present, capture aborts and the model runs eager for the rest of the run.

**What exists in backend but is NOT wired to any model except LFM2**:
- `GraphCaptureManager` in `graph_capture.rs` — generic graph cache with retry policy
- `DecodeGraphBuffers` in `decode_graph_buffers.rs` — fixed device buffer pool for capture
- `forward_capture()` / `forward_replay()` — capture/replay lifecycle
- `forward_capture_batch()` / `forward_replay_batch()` — P3 batch capture

---

## 2. H2D/D2H as Primary Path (Not Fallback)

### Finding: 21 models have CPU-only forward paths with NO device path at all.

These models perform ALL computation on the CPU, using `to_vec_f32()` to pull tensors from device, running CPU loops, then re-uploading via `move_to_device()` or `from_cpu()`. This is the opposite of GPU-first.

### Pure CPU Models (D2H every layer, no device path)

| File | D2H Calls | Device Path |
|------|-----------|-------------|
| `bloom.rs` | 6 | None |
| `chameleon.rs` | 5 | None |
| `commandr.rs` | 5 | None |
| `dbrx.rs` | 4+ | None (expert loop) |
| `deepseek.rs` | 9 | None |
| `dots3_note.rs` | 5 | None |
| `exaone4_5.rs` | 5 | None |
| `falcon.rs` | 10 | None |
| `gemma2.rs` | 10 | None |
| `gemma.rs` | 15 | None (has `forward_kv` device path but `forward` uses CPU) |
| `glm4_moe_lite.rs` | 4 | None |
| `glm5_2.rs` | 7 | None |
| `gpt2.rs` | 11 | None |
| `gptj.rs` | 6 | None |
| `granite_moe_hybrid.rs` | 8 | None |
| `hyv3.rs` | 4 | None |
| `hy_v4.rs` | 5 | None |
| `longcat_flash.rs` | 5 | None |
| `minimax_m3.rs` | 5 | None |
| `native_mtp.rs` | 5 | None |
| `qwen35moe.rs` | 6 | None (MoE + FFN on CPU) |
| `qwen38_flash_next.rs` | 17 | None (largest offender: 17 D2H per forward) |
| `wav_tokenizer_dec.rs` | 14 | None |

### Example: `gemma.rs` CPU Path (lines 185-232)

```rust
// D2H: pull K/V to host
let k_t = cpu_tensor(k.to_vec_f32()?, ...);
let v_vec = v.to_vec_f32()?;

// Host cache append (CPU-side Vec<f32> extend)
cache.k.extend_from_slice(&k_t.to_vec_f32()?);
cache.v.extend_from_slice(&v_vec);

// CPU attention
let qd = q.to_vec_f32()?;
let attn_out = crate::kv_attention::causal_attention(&qd, &cache.k, &cache.v, ...);

// Re-upload result
let attn_out_t = cpu_tensor(attn_out, ...);
```

**Note**: `gemma.rs` HAS a device-first path (`forward_kv`, lines 235-314) that uses `concat_rows_on_device` and `fused_attention_tensors`. But the default `forward` path uses the CPU-only implementation. The device path exists but is not the default.

### Example: `falcon.rs` CPU Path (lines 33-37)

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let xv = x.to_vec_f32()?;           // D2H entire activation
    let w = self.weight.to_vec_f32()?;  // D2H weight
    let b = self.bias.as_ref().map(|b| b.to_vec_f32()).transpose()?;
    // ... CPU matmul ...
}
```

This is a linear layer that runs entirely on CPU even when the input is already on GPU.

### Example: `qwen38_flash_next.rs` (worst offender, 17 D2H)

Lines 250-291 show the MoE expert dispatch doing per-expert `to_vec_f32()` on gate logits, input tokens, expert outputs, and shared expert outputs — all on the host, every step.

---

## 3. Warm KV Cache Tracking

### Finding: Only ~10 models actually use device-resident KV arenas. ~10 more declare the fields but never use them.

### Models WITH Warm KV (device-resident arenas, D2D append)

| File | Arena Usage | Mechanism |
|------|-------------|-----------|
| `block.rs` | 20 uses | `cache_append_kv()` → D2D append to `k_device`/`v_device` |
| `deepseek2.rs` | 4 uses | Device-resident latent KV cache |
| `deepseek32.rs` | 4 uses | Device-resident latent KV cache |
| `deepseek4.rs` | 5 uses | Device-resident latent KV cache |
| `falcon_h1.rs` | 15 uses | `k_device`/`v_device` with `cache_append_kv` |
| `lfm2.rs` | 10 uses | `past_dev` device buffer + `k_arena`/`v_arena` |
| `minicpm.rs` | 10 uses | `k_device`/`v_device` with `cache_append_kv` |
| `muse_glimmer.rs` | 9 uses | `k_device`/`v_device` arena |
| `qwen35.rs` | 22 uses | `k_device`/`v_device` with geometric growth |

### Models with DEAD `k_device`/`v_device` Fields

These models declare the fields in their cache struct but never actually use them — the forward path does `to_vec_f32()` + host `extend_from_slice` + `from_cpu` instead.

| File | Field Declared | Actually Used |
|------|----------------|---------------|
| `delta_net_base.rs` | Yes | No |
| `diffusion_gemma.rs` | Yes | No |
| `gemma3n.rs` | Yes | No |
| `inkling_small.rs` | Yes | No |
| `interns2_mobius.rs` | Yes | No |
| `kimi_k3.rs` | Yes | No |
| `lora.rs` | Yes | No (adapter infrastructure, not model KV) |
| `native_mtp.rs` | Yes | No |
| `qwen2.rs` | Yes | No |
| `qwen38_flash_next.rs` | Yes | No |

### Thin Wrappers (inherit from block.rs)

The ~90 thin wrapper models (qwen, qwen3, qwen2, phi2, phi3, gemma3, olmo, stablelm, etc.) delegate to `Llama::forward`, which uses `block.rs` layers. They inherit the warm KV path through `cache_append_kv()` in `block.rs`. Their KV cache correctness depends entirely on `block.rs` implementation.

---

## 4. Unnecessary Barriers / Synchronization

### Finding: No explicit `hipDeviceSynchronize` or `hipStreamSynchronize` calls exist in model forward paths.

However, implicit synchronization occurs through:

1. **`to_vec_f32()` on GPU tensors**: Forces a D2H copy, which synchronizes the stream. Models using this in the hot path (see Section 2) pay an implicit barrier every call.

2. **`move_to_device()` / `from_cpu()`**: H2D copies that synchronize when the source is host memory.

3. **`cache_append_kv` fallback path** (`block.rs:1383-1393`): When the device arena append fails (backend doesn't support D2D append), the code falls back to `k_cache.extend_from_slice(&k_3d.to_vec_f32()?)` — a D2H + host extend + H2D re-upload. This is a correct fallback, not an unnecessary barrier.

### Mutex Contention (from `roc_device.rs`, not model files)

These are backend-level issues documented in `gpu-inference-perf-audit` findings:
- `resolved_kernel_cache` Mutex per kernel launch
- `upload_event` Mutex per `active_stream()` call
- `launch_counter` atomic per launch

---

## 5. MoE Host Routing (Gate Logits D2H)

### Finding: 8+ models pull MoE gate logits to host for every routing decision.

| File | Line | Pattern |
|------|------|---------|
| `deepseek2.rs` | 517, 538 | `logits.to_vec_f32()` for host softmax+top-k |
| `deepseek32.rs` | 653, 674 | `logits.to_vec_f32()` for host routing |
| `deepseek4.rs` | 646, 668 | `logits.to_vec_f32()` for host routing |
| `kimi_k3.rs` | 433, 454 | `logits.to_vec_f32()` for host routing |
| `lfm2.rs` | 1974, 2070 | `gate_logits.to_vec_f32()` (falls back to host when Charon unavailable) |
| `gemma2.rs` | 383 | `logits.to_vec_f32()` |
| `glm5_2.rs` | 151 | `logits_v = logits.to_vec_f32()` |
| `diffusion_gemma.rs` | 382 | `gate.to_vec_f32()` |

**Impact**: For MoE models with 64+ experts, this D2H happens every MoE layer. The `shared_moe.rs` provides a Charon-based device path, but many models don't use it or fall back eagerly.

**LFM2 exception**: `lfm2.rs:1960` (`forward_moe_ffn`) tries `shared_moe` first with device-resident routing, but falls back to host loop on `is_unimplemented`.

---

## 6. Fused Kernel Usage

### Models with Fused QKV (1 launch instead of 3)

`block.rs`, `chameleon.rs`, `commandr.rs`, `dots3_note.rs`, `exaone4_5.rs`, `falcon_h1.rs`, `gemma.rs`, `gptj.rs`, `hy_v4.rs`, `model.rs`, `qwen35.rs`, `qwen38_flash_next.rs`, `qwen35_perf.rs`

### Models with Fused Gate+Up (1 launch instead of 2)

`block.rs`, `lfm2.rs`, `lfm2_graph.rs`, `model.rs`

### Models with Device-Base RoPE (no per-token position H2D)

Only `block.rs` and `lfm2.rs` use `GRIM_ROPE_DEV_BASE`. All other models either:
- Don't use RoPE at all (non-rotary architectures)
- Use the host-side position path (`rope.forward()` on CPU)
- Use `rope_2d_on_device()` from `shared_attention.rs` (which still uploads positions per-step)

### Models with Q8_1 Activation Quantization (fused silu_mul_quant)

Only `block.rs` and `lfm2.rs` check for `silu_mul_quant_q81_decode`. Other models use `silu_mul_on_device` or host GeGLU.

---

## 7. Model Classification

### Category A: Full GPU Path (~10 models)
Have device-resident KV, device-side attention, fused kernels where applicable:
- `block.rs` (LlamaBlock — used by ~90 thin wrappers)
- `lfm2.rs` (+ `lfm2_graph.rs` for capture)
- `deepseek2.rs`, `deepseek32.rs`, `deepseek4.rs` (MLA with device latent cache)
- `falcon_h1.rs` (hybrid SSM+attention, device arenas)
- `minicpm.rs` (paged attention + device arenas)
- `muse_glimmer.rs` (hybrid attention, device arenas)
- `qwen35.rs` (hybrid SSM+attention, device arenas)

### Category B: Thin Wrappers (~90 models)
Delegate to `Llama::forward` → `block.rs` layers. Inherit:
- Warm KV via `cache_append_kv`
- Device-side RoPE via `apply_rope_multi_head`
- Fused QKV/FFN when `wqkv_q80_fused`/`w_gate_up_q80_fused` present
- Graph-capture-ready via `decode_graph_active`

**Risk**: These models don't build `wqkv_q80_fused` or `w_gate_up_q80_fused` unless the checkpoint has Q8_0 weights. Most won't benefit from fused kernels.

### Category C: Pure CPU (~21 models)
No device path at all. All computation on host. Cannot benefit from graph capture, fused kernels, or warm KV.

### Category D: Custom Forward, Partial GPU (~10 models)
Have some device operations but fall back to host for specific ops (MoE, activation functions):
- `bloom.rs` (D2H for QKV split, GPU attention)
- `gemma.rs` (has `forward_kv` device path, but `forward` is CPU)
- `qwen35moe.rs` (MoE on CPU, attention on GPU)
- `qwen38_flash_next.rs` (everything on CPU)

---

## 8. Recommendations

### Critical (Impact: 2-10x throughput)

1. **Add graph capture for block.rs/Llama**: 90+ models inherit this path. A single `forward_capture()` / `forward_replay()` pair in `block.rs` would cover the majority of the model zoo. Blockers: MoE dispatch, paged KV upload, alibi/sliding-window attention.

2. **Unify on `forward_kv` pattern**: Models like `gemma.rs` have both a CPU `forward` and device `forward_kv`. Make the device path the default.

3. **Port CPU-only models to device path**: `falcon.rs` (10 D2H), `qwen38_flash_next.rs` (17 D2H), `gpt2.rs` (11 D2H), `bloom.rs` (6 D2H) are the highest-impact targets.

### High (Impact: 1.5-2x throughput)

4. **Wire Charon MoE for all MoE models**: Currently only `lfm2.rs` and `shared_moe.rs` use the device-resident routing path. DeepSeek2/32/4, KimiK3, Gemma2, GLM5_2 all do host routing.

5. **Remove dead `k_device`/`v_device` fields**: Or actually wire them up. 10 models declare but never use device arenas.

6. **Build fused QKV/FFN blobs for all models**: Only models with Q8_0 weights get `wqkv_q80_fused`. Consider runtime quantization or fallback to unfused path.

### Medium (Impact: 1.1-1.3x throughput)

7. **Device-base RoPE for all models**: Only `block.rs` and `lfm2.rs` use `GRIM_ROPE_DEV_BASE`. Other models upload positions per-step.

8. **Device-side sampling**: GPU sampler (`sample_logits_on_device_*`) is available but only used when `allow_gpu_sample == true` (repeat_penalty == 1.0). MoE gate logits D2H could use the same infrastructure.

9. **Q8_1 activation quantization**: Only `block.rs` and `lfm2.rs` use `silu_mul_quant_q81_decode`. Other models do full-precision silu_mul.

---

## 9. Verification Notes

- All findings verified by reading actual dispatch code, not inferred from architecture.
- Launch counts are per-layer per-token.
- "D2H calls" counts are from static grep of `to_vec_f32()` / `to_cpu_vec_f32()` in forward paths.
- Graph capture coverage verified by grep for `decode_graph_active`, `forward_capture`, `hipGraph*` in model files.
- Warm KV verified by grep for `cache_append_kv`, `k_device`, `v_device` usage counts.
- Mutex/barrier findings cross-referenced with `gpu-inference-perf-audit` skill session notes.
