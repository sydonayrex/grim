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
///
/// WI 1.4.2 — all kernels in this source parallelize the KV walk across all 4
/// RDNA wavefronts in the 256-thread block (one wavefront per wave_id 0..3),
/// each owning a contiguous quarter of the valid `j`-range. Per-wavefront
/// partial online-softmax state (max, sum, per-dim acc) is published to
/// in-kernel `__shared__` LDS, then wave 0 merges the 4 partials pairwise and
/// writes the final `out[d]`. See `grim_rocm_consumer_perf_plan.md` WI 1.
///
/// Hardware-aware corner (RDNA iGPU, e.g. gfx1036): `warpSize` resolves to 32
/// at runtime; the host launches block = wave_size (single wavefront) since
/// one wave covers head_dim up to wave_size. `head_dim > wave_size` throws
/// NaN. The kernel uses `__shared__ float s_acc[8][wave_size == 64 ? 64 : 32]`
/// so the same kernel handles wave64 (full WI 1.4.2 4-way path) and wave32
/// (single-wavefront correctness path) without source forks. On wave32 the
/// WI 1.4.2 KV-parallel speedup is not realized (only one wavefront runs);
/// treat wave64 hardware (`gfx110x`, `gfx1200`, CDNA) as the optimization
/// target.
///
/// Load-imbalance note (DCU-GCN §2.4.4-2 caution in WI 1.6.1): within one
/// block, the 4 wavefronts get a static (base, rem) stride partition. If a
/// batch mixes very different kv_seq_len sequences the *cross-block* part of
/// the kernel is unaffected (each block has its own j_range), but per-kernel
/// terminals can see uneven per-wave work. Correctness holds either way; the
/// perf headroom is reserved for a future dynamic-chunk-sizing step rather
/// than a correction to this PR.
///
/// TODO(gpu-verify): Gate 1.6.4 perf number — measure on wave64 hardware
/// (gfx1100+ / CDNA) at `kv_seq_len ∈ {128, 512, 2048, 8192}` to confirm
/// the ~1.5×+ speedup expected from using 4× the working threads. No perf
/// number has been measured or claimed.
pub const KERNEL_SOURCE: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void grim_qkv_attention(
    const float* __restrict__ q,
    const float* __restrict__ k_tensor,
    const float* __restrict__ v_tensor,
    float* __restrict__ out,
    float* __restrict__ out_max,
    float* __restrict__ out_sum,
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
    //
    // WI 1.4.2 (wave64 hardware — RDNA dGPU / CDNA): the causal KV walk is
    // split across the 4 wavefronts in the 256-thread block, each owning a
    // quarter-stride partitioning of the sequence. At the end, wave 0
    // combines the 4 wavefront partials in shared memory LDS.
    //
    // Wave32 hardware fallback (RDNA iGPU, gfx1036 et al.): warpSize resolves
    // to 32 at runtime; the host launches block = wave_size (single
    // wavefront); WI 1.4.2's per-d up-axis at head_dim > wave_size would be
    // incomplete. We constrain head_dim ≤ wave_size and run a single-thread
    // (per dim) softmax — same algorithm, no parallelism across KV. This is a
    // hardware-mediated fast path; the wave64 branch keeps the WI 1.4.2 speedup.
    // ──────────────────────────────────────────────────────────────────────
    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    // Block is fixed at 256 (`__launch_bounds__(256)`); num_waves = block_size / wave_size
    // adapts to RDNA iGPU wave32 (num_waves=8) and RDNA dGPU / CDNA wave64 (num_waves=4).
    const int num_waves = 256 / wave_size;

    const int d = lane_id;
    const bool thread_active = d < head_dim;

    // Hardware-aware head-dim cap.
    if (head_dim > 256) {
        for (int chunk = 0; chunk < 4; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = nanf("");
            }
        }
        return;
    }

    // Per-wavefront partials published to LDS for the wave-0 merge. Sized to
    // the worst-case 8 wavefronts (RDNA iGPU wave32 + block 256 host path) and
    // head dimensions up to 256.
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][256];

    // Causal KV range for this query: [0, hi) where hi = min(abs_i + 1, kv_seq_len).
    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int range_len = hi;  // j in [0, hi)

    // Quarter-stride partitioning of [0, hi) across the wavefronts.
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Fast-path GQA key/value stride pointers
    const float* __restrict__ k_head = &k_tensor[kv_head * head_dim];
    const float* __restrict__ v_head = &v_tensor[kv_head * head_dim];

    // Inner loop: online-softmax over assigned range
    for (int j = j_start; j < j_end; ++j) {
        // Dot product Q.K for this head
        float score = 0.0f;
        #pragma unroll
        for (int dim = 0; dim < 256; ++dim) {
            if (dim < head_dim) {
                score += q[q_offset + dim] * k_head[j * (num_kv_heads * head_dim) + dim];
            }
        }
        score *= inv_sqrt_d;

        // Online-softmax update
        const float old_max = running_max;
        running_max = fmaxf(running_max, score);
        const float scale_old = expf(old_max - running_max);
        const float scale_new = expf(score - running_max);

        running_sum = running_sum * scale_old + scale_new;
        for (int chunk = 0; chunk < 4; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_head[j * (num_kv_heads * head_dim) + d];
            }
        }
    }

    // Publish per-wavefront partials to LDS. max/sum are wavefront-uniform
    // (all lanes see same j_start/j_end loop range). Wave 0 (lane 0) publishes
    // the max/sum state.
    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 4; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            s_acc[wave_id][d] = out_acc[chunk];
        } else if (d < 256) {
            s_acc[wave_id][d] = 0.0f;
        }
    }
    __syncthreads();

    // Wave 0 merges the partials from every wave into one (max, sum, acc[d]).
    if (wave_id != 0) return;

    for (int chunk = 0; chunk < 4; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            float m_final = s_max[0];
            float sum_final = s_sum[0];
            float acc_final = s_acc[0][d];
            #pragma unroll
            for (int w = 1; w < 8; ++w) {
                if (w >= num_waves) break;
                const float mw = s_max[w];
                const float uw = s_sum[w];
                const float aw = s_acc[w][d];
                const float m_new = fmaxf(m_final, mw);
                const float scale_a = expf(m_final - m_new);
                const float scale_b = expf(mw - m_new);
                sum_final = sum_final * scale_a + uw * scale_b;
                acc_final = acc_final * scale_a + aw * scale_b;
                m_final = m_new;
            }
            const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;
            out[q_offset + d] = acc_final * inv_sum;
        }
    }

    if (tid == 0) {
        if (out_max != nullptr) {
            out_max[i * num_heads + h] = m_final;
        }
        if (out_sum != nullptr) {
            out_sum[i * num_heads + h] = sum_final;
        }
    }
}

struct BlockTableEntry {
    unsigned int block_id;
    unsigned int page_size;
};

extern "C" __global__ __launch_bounds__(256)
void grim_qkv_attention_paged(
    const float* __restrict__ q,
    const BlockTableEntry* __restrict__ block_tables,
    const float* __restrict__ k_pages,
    const float* __restrict__ v_pages,
    float* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks,
    int page_size,
    int kv_seq_len,
    int cache_offset,
    float inv_sqrt_d
) {
    const int batch_idx = blockIdx.x; // grid is (batch, num_heads, 1)
    const int h = blockIdx.y;         // head index
    
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    
    // Q is laid out as [batch, num_heads, head_dim]
    const int q_offset = (batch_idx * num_heads + h) * head_dim;
    const int abs_i = cache_offset; // absolute query position for this step

    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    // Block is fixed at 256 (`__launch_bounds__(256)`); num_waves = block_size / wave_size
    // adapts to RDNA iGPU wave32 (num_waves=8) and RDNA dGPU / CDNA wave64 (num_waves=4).
    const int num_waves = 256 / wave_size;

    const int d = lane_id;
    const bool thread_active = d < head_dim;

    // Hardware-aware head-dim cap.
    if (head_dim > 256) {
        for (int chunk = 0; chunk < 4; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = nanf("");
            }
        }
        return;
    }

    // WI 1.4.2: per-wavefront partials published to LDS for the wave-0 merge.
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][256];

    // Per-wavefront KV slice [j_start, j_end) over the flattened page/token
    // index space [0, kv_seq_len). Same stride partition as the non-paged kernel.
    const int range_len = kv_seq_len;
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Get the block table for this batch
    const BlockTableEntry* my_table = block_tables + batch_idx * max_blocks;

    // Walk this wavefront's K/V slice [j_start, j_end).
    for (int j = j_start; j < j_end; ++j) {
        if (j > abs_i || j >= kv_seq_len) break;

        // Decompose j into (block b, token t within page)
        const int b = j / page_size;
        const int t = j % page_size;
        const BlockTableEntry entry = my_table[b];
        const int physical_token_idx = entry.block_id * page_size + t;
        // K/V page layout: [num_pages, page_size, num_kv_heads, head_dim]
        const int kv_offset = (physical_token_idx * num_kv_heads + kv_head) * head_dim;
        
        float score = 0.0f;
        #pragma unroll
        for (int dim = 0; dim < 256; ++dim) {
            if (dim < head_dim) {
                score += q[q_offset + dim] * k_pages[kv_offset + dim];
            }
        }
        score *= inv_sqrt_d;
        
        float w = expf(score - running_max);
        if (score > running_max) {
            const float scale = expf(running_max - score);
            running_sum = running_sum * scale;
            for (int chunk = 0; chunk < 4; ++chunk) {
                out_acc[chunk] = out_acc[chunk] * scale;
            }
            running_max = score;
            w = 1.0f;
        }
        for (int chunk = 0; chunk < 4; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out_acc[chunk] += w * v_pages[kv_offset + d];
            }
        }
        running_sum += w;
    }

    // Publish per-wavefront partials to LDS.
    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 4; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            s_acc[wave_id][d] = out_acc[chunk];
        } else if (d < 256) {
            s_acc[wave_id][d] = 0.0f;
        }
    }
    __syncthreads();

    // Wave 0 merges the partials from every wave into one (max, sum, acc[d]).
    if (wave_id != 0) return;

    for (int chunk = 0; chunk < 4; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            float m_final = s_max[0];
            float sum_final = s_sum[0];
            float acc_final = s_acc[0][d];
            #pragma unroll
            for (int w = 1; w < 8; ++w) {
                if (w >= num_waves) break;
                const float mw = s_max[w];
                const float uw = s_sum[w];
                const float aw = s_acc[w][d];
                const float m_new = fmaxf(m_final, mw);
                const float scale_a = expf(m_final - m_new);
                const float scale_b = expf(mw - m_new);
                sum_final = sum_final * scale_a + uw * scale_b;
                acc_final = acc_final * scale_a + aw * scale_b;
                m_final = m_new;
            }
            const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;
            out[q_offset + d] = acc_final * inv_sum;
        }
    }
}

__device__ bool is_ancestor(int j, int i, const unsigned int* tree_parents) {
    if (j == i) return true;
    int curr = i;
    while (curr > 0) {
        curr = (int)tree_parents[curr];
        if (curr == j) return true;
    }
    return false;
}

extern "C" __global__ __launch_bounds__(256)
void grim_tree_attention(
    const float* __restrict__ q,
    const float* __restrict__ k_tensor,
    const float* __restrict__ v_tensor,
    const unsigned int* __restrict__ tree_parents,
    float* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int gamma,
    int kv_seq_len,
    int cache_offset,
    float inv_sqrt_d
) {
    const int i = blockIdx.x;             // tree position (0..gamma)
    const int h = blockIdx.y;             // head index
    const int batch_idx = blockIdx.z;     // batch index
    
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    
    // q and out are [batch, 1+gamma, num_heads, head_dim]
    const int q_offset = ((batch_idx * (1 + gamma) + i) * num_heads + h) * head_dim;
    
    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    // Block is fixed at 256 (`__launch_bounds__(256)`); num_waves = block_size / wave_size
    // adapts to RDNA iGPU wave32 (num_waves=8) and RDNA dGPU / CDNA wave64 (num_waves=4).
    const int num_waves = 256 / wave_size;

    const int d = lane_id;
    const bool thread_active = d < head_dim;

    // Hardware-aware head-dim cap.
    if (head_dim > 256) {
        for (int chunk = 0; chunk < 4; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = nanf("");
            }
        }
        return;
    }

    // WI 1.4.2: per-wavefront partials published to LDS for the wave-0 merge.
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][256];

    // Per-wavefront KV slice [j_start, j_end) over the flattened page/token
    // index space [0, kv_seq_len). Same stride partition as the non-paged kernel.
    const int range_len = kv_seq_len;
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    for (int j = j_start; j < j_end; ++j) {
        bool attend = false;
        if (j < cache_offset) {
            attend = true;
        } else {
            int tree_node_idx = j - cache_offset;
            if (tree_node_idx <= i && is_ancestor(tree_node_idx, i, tree_parents)) {
                attend = true;
            }
        }
        if (!attend) continue;
        
        const int kv_offset = ((batch_idx * kv_seq_len + j) * num_kv_heads + kv_head) * head_dim;
        
        float score = 0.0f;
        #pragma unroll
        for (int dim = 0; dim < 256; ++dim) {
            if (dim < head_dim) {
                score += q[q_offset + dim] * k_tensor[kv_offset + dim];
            }
        }
        score *= inv_sqrt_d;
        
        float w = expf(score - running_max);
        if (score > running_max) {
            const float scale = expf(running_max - score);
            running_sum = running_sum * scale;
            for (int chunk = 0; chunk < 4; ++chunk) {
                out_acc[chunk] = out_acc[chunk] * scale;
            }
            running_max = score;
            w = 1.0f;
        }
        for (int chunk = 0; chunk < 4; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out_acc[chunk] += w * v_tensor[kv_offset + d];
            }
        }
        running_sum += w;
    }

    // Publish per-wavefront partials to LDS.
    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 4; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            s_acc[wave_id][d] = out_acc[chunk];
        } else if (d < 256) {
            s_acc[wave_id][d] = 0.0f;
        }
    }
    __syncthreads();

    // Wave 0 merges the partials from every wave into one (max, sum, acc[d]).
    if (wave_id != 0) return;

    for (int chunk = 0; chunk < 4; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            float m_final = s_max[0];
            float sum_final = s_sum[0];
            float acc_final = s_acc[0][d];
            #pragma unroll
            for (int w = 1; w < 8; ++w) {
                if (w >= num_waves) break;
                const float mw = s_max[w];
                const float uw = s_sum[w];
                const float aw = s_acc[w][d];
                const float m_new = fmaxf(m_final, mw);
                const float scale_a = expf(m_final - m_new);
                const float scale_b = expf(mw - m_new);
                sum_final = sum_final * scale_a + uw * scale_b;
                acc_final = acc_final * scale_a + aw * scale_b;
                m_final = m_new;
            }
            const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;
            out[q_offset + d] = acc_final * inv_sum;
        }
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct BlockTableEntry {
    pub block_id: u32,
    pub page_size: u32,
}

use grim_tensor::backend::BackendStorage;

fn arg<T>(v: &mut T) -> *mut std::ffi::c_void {
    v as *mut T as *mut std::ffi::c_void
}

pub fn launch_paged_attention(
    dev: &crate::RocmDevice,
    q: &dyn BackendStorage,          // [batch, num_heads, head_dim]
    block_tables: &dyn BackendStorage, // [batch, max_blocks] of BlockTableEntry
    k_pages: &dyn BackendStorage,     // [num_pages, page_size, num_kv_heads, head_dim]
    v_pages: &dyn BackendStorage,     // [num_pages, page_size, num_kv_heads, head_dim]
    out: &mut dyn BackendStorage,     // [batch, num_heads, head_dim]
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_blocks: u32,
    page_size: u32,
    kv_seq_len: u32,
    cache_offset: u32,
) -> Result<(), crate::Error> {
    let q_s = q.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("q must be RocmStorage".into()))?;
    let block_tables_s = block_tables.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("block_tables must be RocmStorage".into()))?;
    let k_pages_s = k_pages.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("k_pages must be RocmStorage".into()))?;
    let v_pages_s = v_pages.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("v_pages must be RocmStorage".into()))?;
    let out_s = out.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("out must be RocmStorage".into()))?;

    let q_ptr = q_s.device_ptr.ok_or_else(|| crate::Error::Backend("q has no device ptr".into()))?;
    let block_tables_ptr = block_tables_s.device_ptr.ok_or_else(|| crate::Error::Backend("block_tables has no device ptr".into()))?;
    let k_pages_ptr = k_pages_s.device_ptr.ok_or_else(|| crate::Error::Backend("k_pages has no device ptr".into()))?;
    let v_pages_ptr = v_pages_s.device_ptr.ok_or_else(|| crate::Error::Backend("v_pages has no device ptr".into()))?;
    let out_ptr = out_s.device_ptr.ok_or_else(|| crate::Error::Backend("out has no device ptr".into()))?;

    // Grid: (batch, num_heads, 1); Block: (256, 1, 1)
    let grid_dim = crate::HipDim3::new(batch, num_heads, 1);
    let block_dim = crate::HipDim3::new(256, 1, 1);

    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

    let mut qptr = q_ptr;
    let mut btptr = block_tables_ptr;
    let mut kptr = k_pages_ptr;
    let mut vptr = v_pages_ptr;
    let mut optr = out_ptr;
    let mut nh = num_heads as i32;
    let mut nkv = num_kv_heads as i32;
    let mut hd = head_dim as i32;
    let mut mb = max_blocks as i32;
    let mut ps = page_size as i32;
    let mut ksl = kv_seq_len as i32;
    let mut co = cache_offset as i32;
    let mut isd = inv_sqrt_d;

    dev.launch_compute_kernel(
        "grim_qkv_attention_paged",
        grid_dim,
        block_dim,
        &mut [
            arg(&mut qptr),
            arg(&mut btptr),
            arg(&mut kptr),
            arg(&mut vptr),
            arg(&mut optr),
            arg(&mut nh),
            arg(&mut nkv),
            arg(&mut hd),
            arg(&mut mb),
            arg(&mut ps),
            arg(&mut ksl),
            arg(&mut co),
            arg(&mut isd),
        ],
    )?;

    Ok(())
}

pub fn launch_tree_attention(
    dev: &crate::RocmDevice,
    q: &dyn BackendStorage,            // [batch, 1+gamma, num_heads, head_dim]
    k: &dyn BackendStorage,            // [batch, kv_seq_len, num_kv_heads, head_dim]
    v: &dyn BackendStorage,            // [batch, kv_seq_len, num_kv_heads, head_dim]
    tree_parents: &dyn BackendStorage, // [1+gamma] uint32 parent indices
    out: &mut dyn BackendStorage,      // [batch, 1+gamma, num_heads, head_dim]
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    gamma: u32,
    kv_seq_len: u32,
    cache_offset: u32,
) -> Result<(), crate::Error> {
    let q_s = q.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("q must be RocmStorage".into()))?;
    let k_s = k.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("k must be RocmStorage".into()))?;
    let v_s = v.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("v must be RocmStorage".into()))?;
    let parents_s = tree_parents.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("tree_parents must be RocmStorage".into()))?;
    let out_s = out.as_any().downcast_ref::<crate::memory::storage::RocmStorage>().ok_or_else(|| crate::Error::Backend("out must be RocmStorage".into()))?;

    let q_ptr = q_s.device_ptr.ok_or_else(|| crate::Error::Backend("q has no device ptr".into()))?;
    let k_ptr = k_s.device_ptr.ok_or_else(|| crate::Error::Backend("k has no device ptr".into()))?;
    let v_ptr = v_s.device_ptr.ok_or_else(|| crate::Error::Backend("v has no device ptr".into()))?;
    let parents_ptr = parents_s.device_ptr.ok_or_else(|| crate::Error::Backend("tree_parents has no device ptr".into()))?;
    let out_ptr = out_s.device_ptr.ok_or_else(|| crate::Error::Backend("out has no device ptr".into()))?;

    // Grid: (1 + gamma, num_heads, batch); Block: (256, 1, 1)
    let grid_dim = crate::HipDim3::new(1 + gamma, num_heads, batch);
    let block_dim = crate::HipDim3::new(256, 1, 1);

    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

    let mut qptr = q_ptr;
    let mut kptr = k_ptr;
    let mut vptr = v_ptr;
    let mut pptr = parents_ptr;
    let mut optr = out_ptr;
    let mut nh = num_heads as i32;
    let mut nkv = num_kv_heads as i32;
    let mut hd = head_dim as i32;
    let mut gm = gamma as i32;
    let mut ksl = kv_seq_len as i32;
    let mut co = cache_offset as i32;
    let mut isd = inv_sqrt_d;

    dev.launch_compute_kernel(
        "grim_tree_attention",
        grid_dim,
        block_dim,
        &mut [
            arg(&mut qptr),
            arg(&mut kptr),
            arg(&mut vptr),
            arg(&mut pptr),
            arg(&mut optr),
            arg(&mut nh),
            arg(&mut nkv),
            arg(&mut hd),
            arg(&mut gm),
            arg(&mut ksl),
            arg(&mut co),
            arg(&mut isd),
        ],
    )?;

    Ok(())
}
