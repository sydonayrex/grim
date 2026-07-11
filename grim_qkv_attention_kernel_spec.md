# Task: Implement `grim_qkv_attention` (fused QKV attention HIP kernel)

## Context

`grim-backend-rocm` (a ROCm/HIP backend crate) has a fused QKV-attention path that is currently **disabled on purpose**. The Rust entry point `RocmDevice::qkv_attention()` in `src/lib.rs` hardcodes `QkvAttentionFusionConfig { enabled: false, .. }` and returns an `Err` before ever launching the kernel, because the existing `grim_qkv_attention` HIP source (in the `COMPUTE_KERNEL_SOURCE` string constant in `src/lib.rs`) is a non-functional stub:

- It never indexes `k_tensor` by sequence position — the same K vector is reused for every loop iteration.
- It never applies softmax.
- It never reads `v_tensor` at all.
- It overwrites the same `out[head * head_dim]` slot on every loop iteration instead of writing a distinct result per position.

Your job is to replace that stub with a correct, reasonably performant fused attention kernel, wire it up, and re-enable it — **only** once it's verified correct.

## Files involved

- `src/lib.rs` — contains the `COMPUTE_KERNEL_SOURCE` raw string with the HIP kernel source (find `grim_qkv_attention`), and the Rust-side `RocmDevice::qkv_attention()` method that launches it.
- `src/fusion.rs` — contains `QkvAttentionFusionConfig` (launch-geometry config, CPU-side, no device code) and its `hip_launch_params()` method.
- `tests/fusion_smoke.rs` — existing tests construct `QkvAttentionFusionConfig` instances to check launch geometry; you'll need to update these and add real correctness tests.

## Step 0 — resolve the memory layout question before writing any code

The current output shape contract (checked in `qkv_attention()`) is:

```
out_shape.dims() == [seq_len, num_heads, head_dim]
```

This implies **`seq_len` query positions per call**, not a single decode-step query — i.e. this is (or should be) general causal self-attention capable of covering both prefill and decode, not just single-token decode. But:

- The current (broken) kernel signature only takes one `seq_len` int and treats `q` as if it were a single `[num_heads, head_dim]` vector with no query-position indexing at all.
- There is no separate parameter for a KV-cache offset / total cached length distinct from the current chunk's `seq_len`, and no parameter distinguishing "number of query positions in this call" from "number of key/value positions to attend over."
- This crate has no other caller of `qkv_attention()` in this repository — the actual Q/K/V tensor layout used by callers lives in the wider workspace (the `grim-tensor` crate and whatever calls this backend for attention).

**Before implementing, find and read the calling code in the wider `grim` workspace** (search for `.qkv_attention(` or wherever attention is dispatched) to confirm:

1. Whether `q` has shape `[seq_len, num_heads, head_dim]` (matching `out_shape`) or `[num_heads, head_dim]` (single query / decode-only).
2. Whether `k`/`v` have shape `[kv_seq_len, num_kv_heads, head_dim]`, `[num_kv_heads, kv_seq_len, head_dim]`, or something else — and whether `kv_seq_len` can differ from `seq_len` (e.g. decode with a longer KV cache than the 1-token query chunk).
3. Whether causal masking is expected to happen inside this kernel, or whether the caller already slices K/V to only the valid causal range before calling.

If you cannot find the calling code, implement for the general case (points 4 below) and clearly document the assumed layout in a doc comment above the kernel, since getting this wrong will silently corrupt model outputs — exactly the class of bug you're fixing.

## Step 1 — correct algorithm

Implement standard scaled-dot-product multi-head attention with grouped-query attention (GQA) head-sharing and causal masking:

For each query position `i` in `0..seq_len`, each head `h` in `0..num_heads`:

```
kv_head = h / (num_heads / num_kv_heads)          // GQA mapping — already correct in the old code
for j in 0..kv_seq_len where j <= causal_limit(i): // causal: j <= i if this is self-attention over the same chunk;
                                                     // j <= cache_offset + i if K/V include a longer prior cache
    score[j] = dot(q[i, h, :], k[j, kv_head, :]) / sqrt(head_dim)
softmax score[] in place, numerically stable (subtract max before exp)
out[i, h, :] = sum_j softmax_score[j] * v[j, kv_head, :]
```

Key correctness requirements:
- **Numerically stable softmax**: subtract the running max before `expf`, accumulate a running sum, normalize at the end. Do not compute unnormalized `expf` over the full un-shifted scores.
- **Causal masking**: position `i` must never attend to a key position beyond its own (plus any legitimate KV-cache offset established in Step 0). Get this right — it's easy to get off-by-one here.
- **GQA head mapping**: keep the existing `kv_head = head / (num_heads / num_kv_heads)` formula; it was correct in the old code.
- Use `float` accumulation for the dot products and softmax even if inputs are lower precision, to avoid accuracy loss.

## Step 2 — parallelization strategy (this also requires changing `fusion.rs`)

The current launch geometry (`QkvAttentionFusionConfig::hip_launch_params()`) assigns a 1-D grid over heads only (`grid_x = ceil(num_heads / block_dim_x)`), with each thread serially handling one head end-to-end. That doesn't parallelize over query positions or the score/softmax reduction, and won't scale for real `seq_len` values. Replace it with:

- One thread block per `(query_position, head)` pair (2-D or flattened grid), OR one block per `head` with an inner loop over query positions if you want smaller grids — pick whichever is simpler to get correct first; note the tradeoff in a comment.
- Within a block, parallelize the `head_dim`-length dot product and the reduction across `kv_seq_len` using shared memory (LDS) for the score buffer and for the max/sum reduction, rather than a single thread doing everything serially.
- **Shared memory budget**: `ATTENTION_SHARED_MAX_BYTES` in `fusion.rs` is capped at 32768 bytes (8192 floats). If `kv_seq_len` can exceed that, a full-scores-in-shared-memory approach won't fit. In that case implement an online/streaming (flash-attention-style) softmax that maintains a running max and running weighted sum without ever materializing the full score vector — don't just silently truncate `kv_seq_len`. If you instead choose to cap supported `kv_seq_len` for this first version, make the kernel check the bound and return/assert cleanly rather than silently reading out of bounds.
- Update `QkvAttentionFusionConfig::hip_launch_params()` in `fusion.rs` to match whatever grid/block/shared-mem scheme you implement, and update `ATTENTION_SHARED_MAX_BYTES` usage accordingly if the online-softmax approach changes what you need shared memory for.

## Step 3 — wire it back up in `src/lib.rs`

- Update the `grim_qkv_attention` kernel source string inside `COMPUTE_KERNEL_SOURCE` with the corrected implementation.
- Update `RocmDevice::qkv_attention()`: pass through whatever additional parameters your Step 0/1 design needs (e.g. `kv_seq_len`, cache offset) as kernel args — don't hardcode assumptions that don't match the real caller.
- **Do not set `enabled: true` until Step 4 passes.** Leave the config gate in place; just make it flip to `true` once you're confident, rather than removing the safety check.

## Step 4 — testing (required before enabling)

1. Write a CPU-side reference implementation of causal GQA attention (plain Rust, no HIP) for small shapes.
2. Add correctness tests that: allocate small Q/K/V tensors with known values, run them through `RocmDevice::qkv_attention()`, and compare against the CPU reference within a tolerance (e.g. `1e-3` relative error for `f32`). Cover:
   - `num_heads == num_kv_heads` (no GQA) and `num_heads > num_kv_heads` (GQA) cases.
   - `seq_len == 1` (decode-style) and `seq_len > 1` (prefill-style, if applicable per Step 0) cases.
   - A `kv_seq_len` large enough to exceed the shared-memory-resident score buffer, to exercise the online-softmax path if you implemented one.
   - A causal-masking check: verify a later query position's output changes when an earlier position's K/V changes, but an earlier position's output does *not* depend on later K/V.
3. Update the existing geometry tests in `tests/fusion_smoke.rs` (`qkv_attention_w64_uses_head_count_for_grid`, `qkv_attention_w32_uses_smaller_block`, `qkv_attention_shared_mem_clamped_to_32768`) to match your new `hip_launch_params()` behavior — don't just patch them to compile, make sure they still assert something meaningful about the new grid/block/shared-mem scheme.
4. Only after all of the above pass, flip `enabled: true` in `qkv_attention()` and remove the `Err` guard (or leave the field but always set it `true`).

## Acceptance criteria

- No correctness regression vs. the CPU reference across the test matrix in Step 4.
- Kernel never reads or writes out of bounds regardless of `kv_seq_len` relative to the shared-memory cap.
- `enabled: false` guard remains as a mechanism (even if now always `true`) so a future regression can be gated off the same way without another emergency patch.
- Leave a short doc comment on the kernel documenting the exact Q/K/V memory layout it expects, since that was the root ambiguity that caused the original bug.
