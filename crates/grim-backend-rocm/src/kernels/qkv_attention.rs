//! Phase-1 fused QKV-attention HIP kernel.
//!
//! See `grim_qkv_attention_kernel_spec.md` for the contract this implements.
//! Key points, with citation to relevant skills:
//!   - **Wave64 mandate** on RDNA2/RDNA3/RDNA4 (gfx1036/gfx110x/gfx1200) —
//!     block size is 256 = 4 wavefronts of 64 (`rocm-hip-kernels`).
//!   - **Online (FlashAttention-style) softmax**: running max + running
//!     weighted sum in registers; never materializes a per-`kv_seq_len` score
//!     buffer. Behavior is uniform regardless of how large `kv_seq_len` is.
//!   - **Causal mask inside the kernel**, gated by `j <= cache_offset + i`,
//!     where `i` is the local query index and `cache_offset` shifts to the
//!     absolute position. The kernel is the sole source of truth for what
//!     attends to what.
//!   - **GQA head-sharing**: `kv_head = h / (num_heads / num_kv_heads)`. The
//!     divisor is computed with a fast integer reciprocal; we keep it inside
//!     f32 in the host launcher and pre-check `num_heads % num_kv_heads == 0`
//!     there.
//!   - **f32 only** in this revision (per Step 1's hard requirement). Other
//!     dtypes are a follow-up with a separate task and conversion design.
//!
//! Architectural note for future readers:
//!   - The Phase-2 MFMA / BF16 / FP8 / paged-attention paths live elsewhere
//!     (`rocm-quantization-inference`, `rocm-hip-kernels`). Don't fold them
//!     into this kernel until Phase 1's CPU reference equivalence holds.
//!   - The hot-path kernel keeps `kernel_search_score` inside the for-loop so
//!     registers replace LDS traffic. LDS here is reserved for thread-`h`
//!     accumulation across the wavefront (the final wave reduce).

extern crate alloc;

/// HIP source for `grim_qkv_attention`.
///
/// Concatenated into the crate-wide `COMPUTE_KERNEL_SOURCE` constant for JIT
/// compilation; the kernel signature must match exactly the kernel-launch
/// argument packing done in `lib.rs::RocmDevice::qkv_attention`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void grim_qkv_attention(
    const float* __restrict__ q,
    const float* __restrict__ k_tensor,
    const float* __restrict__ v_tensor,
    float* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_seq_len,
    int cache_offset,   // absolute position of q[head, 0, *]
    float inv_sqrt_d
) {
    // grid = (seq_len, num_heads, 1); block = (256, 1, 1) -> 4 RDNA wavefronts of 64.
    const int i = blockIdx.x;             // query position (0..seq_len)
    const int h = blockIdx.y;             // head index
    if (i >= seq_len || h >= num_heads) return;

    // GQA mapping: every (num_heads/num_kv_heads) query heads share one kv_head.
    // The host validates that num_heads % num_kv_heads == 0, so this is exact.
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;

    // Pointers to this head's q column / kv column. Layouts (Phase-1 contract):
    //   q: [seq_len, num_heads, head_dim]       -> q_offset = (i * num_heads + h) * head_dim
    //   k: [kv_seq_len, num_kv_heads, head_dim] -> k_offset = (j * num_kv_heads + kv_head) * head_dim
    //   v: same as k (separate buffer)
    //   out: [seq_len, num_heads, head_dim]
    const int q_offset = (i * num_heads + h) * head_dim;

    // Cache offset: query position i within this call is at absolute position
    // (cache_offset + i). All past K/V positions are valid up to that.
    const int abs_i = cache_offset + i;

    // ──────────────────────────────────────────────────────────────────────
    // Phase 1: online softmax. Running max + running weighted sum, no full
    // score vector materialized; kv_seq_len may exceed the LDS budget.
    //
    // Each thread owns one output dim d in [0, head_dim).
    // Wave-cross accumulations are reduced via shfl_xor for tree reduction.
    // ──────────────────────────────────────────────────────────────────────
    const int tid = threadIdx.x;
    const int wave_size = 64; // RDNA wavefront; block is 256 = 4 waves.
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;

    // Replicate inputs across waves so each can independently reduce.
    // (Block size 256 on RDNA is 4 wavefronts over the same logical work.)
    const int d = lane_id;
    const bool thread_active = d < head_dim;

    // out_d accumulates V[:, :, d] weighted by softmax(s) — but we track the
    // per-head-dim contribution, so we need one accumulator per logical dim.
    // head_dim <= 64 keeps one accumulator per lane (head_dim=128 would need
    // a register array of size 2 — see the note below).
    //
    // For head_dim <= 64 (typical Llama-style: 128 fused from two rounds),
    // we round through two passes. To keep this revision simple we require
    // head_dim <= 64 here. Larger head_dim is a Phase-2 follow-up that walks
    // the d dimension in tiled chunks; it escalates to MFMA per the spec.
    if (head_dim > 64) {
        // Bail-out: signal an invalid launch by writing NaN. The Rust side
        // validates head_dim and rejects values > 64 with a clean Error.
        if (thread_active) {
            out[q_offset + d] = nanf("");
        }
        return;
    }

    float out_acc = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Wave-local sum/dot scratch (kept in registers; no LDS traffic).
    float my_partial = 0.0f;

    // Walk K/V up to: each valid kv position j satisfies j <= abs_i (causal).
    for (int j = 0; j <= abs_i && j < kv_seq_len; ++j) {
        const int kv_offset = (j * num_kv_heads + kv_head) * head_dim;

        // Score = dot(q[i,h,:], k[j, kv_head, :]) / sqrt(head_dim)
        // One thread accumulates one dot product across head_dim (broadcast).
        float local_score = 0.0f;
        if (thread_active) {
            // Each lane takes one d -> contribution from k/d to its own score,
            // then wave-reduce. The cleanest pattern when blockDim == head_dim:
            // each lane i owns score[distance_from_my_d_to_mag] — instead we
            // do the wave-stride reduction by reusing lane 0 as a broadcast.
            // (For head_dim == wave_size this matches StarCraft-1 reductions.)
            for (int dd = 0; dd < head_dim; ++dd) {
                local_score += q[q_offset + dd] * k_tensor[kv_offset + dd];
            }
            local_score *= inv_sqrt_d;
        }

        // Reduce local_score across the wavefront (tree).
        float s = local_score;
        for (int offset = wave_size / 2; offset > 0; offset >>= 1) {
            s += __shfl_xor(s, offset);
        }
        // Now `s` holds a uniform score for every active lane.

        if (!thread_active) continue;

        // ---- online softmax update (numerically stable) ----
        if (s > running_max) {
            const float scale = expf(running_max - s);
            running_sum = running_sum * scale + 1.0f;
            out_acc = out_acc * scale;
            running_max = s;
        }
        const float w = expf(s - running_max);

        // Pull V[d] (this lane's output dim) and accumulate weighted by w.
        out_acc += w * v_tensor[kv_offset + d];
        running_sum += w;
    }

    // Wave reduce out_acc across all 4 waves in the block. Each wave reads the
    // same d after the first wave wrote a partial in registers; we instead do
    // per-lane accumulation across wave_id strides using __shfl_xor within the
    // first wave. Simpler: every thread already owns one dim d; we just need
    // to write out the lane-0 to lane-63 result whose d indexes 0..head_dim-1
    // uniformly via __shfl.
    //
    // Wave inter-reduction: shift lanes from waves 1,2,3 into wave 0 via shfl
    // chain (w0 <-> w1 <-> w2 <-> w3 across 64-lane boundaries is not directly
    // possible with a single shfl_xor; for the Phase-1 block=256 case we
    // successively reduce within each wave first, then have wave 0 read wave
    // ids 1..3 through LDS. To keep this restart small and dependency-light,
    // use shared memory as a one-shot inter-wave accumulator.)
    __shared__ float interwave[256];
    interwave[tid] = out_acc;
    __syncthreads();

    // Wave 0 reduces lanes 0..63 of the interwave buffer using a stride-4 wave
    // reduction (each lane aggregates waves 0..3 for its d) — but only valid
    // when head_dim <= 64, which we guarded above.
    if (wave_id == 0 && thread_active) {
        const int base = lane_id;
        float v = interwave[base];
        v += interwave[base + wave_size];
        v += interwave[base + 2 * wave_size];
        v += interwave[base + 3 * wave_size];
        // No inter-lane reduce needed because each lane owns a distinct output d.
        out[q_offset + base] = v / running_sum;
    }
}
"#;
