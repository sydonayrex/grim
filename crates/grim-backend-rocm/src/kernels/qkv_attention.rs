//! Phase-1 fused QKV-attention HIP kernel. [see: `grim_qkv_attention_kernel_spec.md`, `rocm-hip-kernels`]

extern crate alloc;

/// HIP source for `grim_qkv_attention`. [see: `COMPUTE_KERNEL_SOURCE`, `lib.rs::RocmDevice::qkv_attention`, `j`, `__shared__`]
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
    float inv_sqrt_d,
    int window_lo,      // sliding-window lower bound: max(0, abs_i - window + 1).
                        // Pass 0 for full causal attention (no window).
    // WI-F2 — fused O-projection epilogue. When fuse_o != 0, `out` is
    // [seq_len, o_dim] (host pre-zeroed) and the per-head normalized
    // attention vector is multiplied by this head's slice of `o_proj_w`
    // (row-major [num_heads*head_dim, o_dim]) and accumulated across heads
    // with atomicAdd instead of being written per-head. Pass o_proj_w=null,
    // o_dim=0, fuse_o=0 for the unfused per-head output path.
    const float* __restrict__ o_proj_w,
    int o_dim,
    int fuse_o
) {
    // grid = (seq_len, num_heads, 1); block = (blockDim.x, 1, 1).
    // The host launches block_dim_x = 128 on RDNA2 (gfx1036, Wave32: 4 wavefronts)
    // or 256 on CDNA (Wave64: 4 wavefronts) — see fusion.rs:78 / roc_device.rs:8145.
    // num_waves is derived at runtime from blockDim.x (line ~74), so the LDS
    // wave-merge loop sees the true wave count on either arch.
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
    // Wave-cross accumulations within a block are reduced via shfl_xor (per
    // wavefront) then combined across wavefronts in LDS by wave 0.
    //
    // The causal KV walk is split across all `num_waves` wavefronts in the
    // block (quarter-stride partitioning for 4 waves; generalizes to N), each
    // owning a slice of the sequence. At the end, wave 0 combines the
    // per-wavefront partials in shared memory LDS.
    //
    // Wave size is resolved at runtime via warpSize: 32 on RDNA2 (gfx1036),
    // 64 on CDNA. The host launch sets block_dim_x = 128 on Wave32 / 256 on
    // Wave64 (fusion.rs:78, roc_device.rs:8145), so num_waves = blockDim.x /
    // wave_size is 4 on either arch. No single-wavefront fallback path
    // exists — head_dim is capped at 256 and partitioned across lanes.
    // ──────────────────────────────────────────────────────────────────────
    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    // num_waves = actual launched block size / wave size. Uses blockDim.x (not a
    // compile-time constant) so it matches the real launch: 128 on gfx1036
    // (Wave32 -> 4 wavefronts) or 256 on CDNA (Wave64 -> 4 wavefronts). The
    // host sets block_dim_x = 128 for Wave32 (fusion.rs:78, roc_device.rs:8145),
    // so this resolves to 4 waves on gfx1036 — the value the LDS merge loop needs.
    const int num_waves = blockDim.x / wave_size;

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
    // worst-case 8 wavefronts (RDNA iGPU Wave32 + block 256 host path) and
    // head dimensions up to 256.
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][256];

    // Causal KV range for this query: [lo, hi) where:
    //   hi = min(abs_i + 1, kv_seq_len)  (standard causal upper bound)
    //   lo = window_lo                    (0 for full attention; sliding lower bound for SWA)
    // `window_lo` is pre-computed on the host as max(0, abs_i - window + 1) so
    // the kernel stays branch-free for the common full-causal case (window_lo == 0).
    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int lo = window_lo;             // 0 for full causal; >= 0 for SWA
    const int range_len = hi - lo;        // may be 0 if lo >= hi (empty window)

    // Quarter-stride partitioning of [lo, hi) across the wavefronts.
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    // j_start / j_end are offsets into [0, range_len); add `lo` when accessing KV.
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Fast-path GQA key/value stride pointers
    const float* __restrict__ k_head = &k_tensor[kv_head * head_dim];
    const float* __restrict__ v_head = &v_tensor[kv_head * head_dim];
    const int kv_stride = num_kv_heads * head_dim;

    // Stage this lane's strided slice of q into registers ONCE. The previous
    // form re-fetched the whole q head from global memory on every KV token
    // and walked a 256-iteration branchy loop per lane; here each lane does
    // <=8 MACs per token and the wavefront-uniform score is produced by a
    // __shfl_xor butterfly.
    float q_reg[8];
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
        int dd = lane_id + chunk * wave_size;
        q_reg[chunk] = (dd < head_dim) ? q[q_offset + dd] : 0.0f;
    }

    // Inner loop: online-softmax over assigned range [lo + j_start, lo + j_end)
    for (int j = lo + j_start; j < lo + j_end; ++j) {
        // Lane-strided partial dot product Q.K, then butterfly reduction.
        float partial = 0.0f;
        #pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
            int dd = lane_id + chunk * wave_size;
            if (dd < head_dim) {
                partial += q_reg[chunk] * k_head[j * kv_stride + dd];
            }
        }
        const unsigned long long shfl_mask = 0xffffffffffffffffULL;
        #pragma unroll
        for (int off = wave_size >> 1; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(shfl_mask, partial, off);
        }
        float score = partial * inv_sqrt_d;

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

    float m_final = s_max[0];
    float sum_final = s_sum[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) {
        if (w >= num_waves) break;
        const float mw = s_max[w];
        const float uw = s_sum[w];
        const float m_new = fmaxf(m_final, mw);
        const float scale_a = expf(m_final - m_new);
        const float scale_b = expf(mw - m_new);
        sum_final = sum_final * scale_a + uw * scale_b;
        m_final = m_new;
    }

    const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;

    // Reconstruct this lane's slice of the normalized attention vector.
    float attn_reg[4];
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        attn_reg[chunk] = 0.0f;
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            float acc_final = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w >= num_waves) break;
                acc_final += s_acc[w][d] * expf(s_max[w] - m_final);
            }
            attn_reg[chunk] = acc_final * inv_sum;
        }
    }

    if (fuse_o == 0) {
        for (int chunk = 0; chunk < 4; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = attn_reg[chunk];
            }
        }
    } else {
        // WI-F2 fused O-projection epilogue. Each wave-0 lane owns its
        // strided slice of the head_dim axis; per output column oc the lane
        // partial is butterfly-reduced across the wavefront and lane 0
        // atomically accumulates into the (host-zeroed) fused output row.
        for (int oc = 0; oc < o_dim; ++oc) {
            float partial = 0.0f;
            #pragma unroll
            for (int chunk = 0; chunk < 4; ++chunk) {
                int d = lane_id + chunk * wave_size;
                if (d < head_dim) {
                    partial += attn_reg[chunk] * o_proj_w[(h * head_dim + d) * o_dim + oc];
                }
            }
            #pragma unroll
            for (int off = wave_size >> 1; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, off);
            }
            if (lane_id == 0 && partial != 0.0f) {
                atomicAdd(&out[i * o_dim + oc], partial);
            }
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
    float inv_sqrt_d,
    int window_lo       // sliding-window lower bound: max(0, abs_i - window + 1).
                        // Pass 0 for full causal attention (no window).
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
    // num_waves = actual launched block size / wave size. Uses blockDim.x (not a
    // compile-time constant) so it matches the real launch: 128 on gfx1036
    // (Wave32 -> 4 wavefronts) or 256 on CDNA (Wave64 -> 4 wavefronts). The
    // host sets block_dim_x = 128 for Wave32 (fusion.rs:78, roc_device.rs:8145),
    // so this resolves to 4 waves on gfx1036 — the value the LDS merge loop needs.
    const int num_waves = blockDim.x / wave_size;

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

    // Per-wavefront KV slice over [window_lo, abs_i+1), partitioned by wavefront.
    // window_lo == 0 is full causal; >=0 is the sliding-window lower bound.
    const int lo = window_lo;
    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int range_len = hi - lo;  // may be 0 if lo >= hi (empty window)
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Get the block table for this batch
    const BlockTableEntry* my_table = block_tables + batch_idx * max_blocks;

    // Walk this wavefront's K/V slice [lo + j_start, lo + j_end).
    for (int j = lo + j_start; j < lo + j_end; ++j) {
        // KV index space is [0, kv_seq_len); causal upper bound is abs_i.
        // (lo already lowers the bound in the partition; this guards the edge on
        // the last wave where j_end may overshoot abs_i+1 if range_len % num_waves.)
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
    // num_waves = actual launched block size / wave size. Uses blockDim.x (not a
    // compile-time constant) so it matches the real launch: 128 on gfx1036
    // (Wave32 -> 4 wavefronts) or 256 on CDNA (Wave64 -> 4 wavefronts). The
    // host sets block_dim_x = 128 for Wave32 (fusion.rs:78, roc_device.rs:8145),
    // so this resolves to 4 waves on gfx1036 — the value the LDS merge loop needs.
    const int num_waves = blockDim.x / wave_size;

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
    q: &dyn BackendStorage,            // [batch, num_heads, head_dim]
    block_tables: &dyn BackendStorage, // [batch, max_blocks] of BlockTableEntry
    k_pages: &dyn BackendStorage,      // [num_pages, page_size, num_kv_heads, head_dim]
    v_pages: &dyn BackendStorage,      // [num_pages, page_size, num_kv_heads, head_dim]
    out: &mut dyn BackendStorage,      // [batch, num_heads, head_dim]
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_blocks: u32,
    page_size: u32,
    kv_seq_len: u32,
    cache_offset: u32,
    window_lo: i32, // sliding-window lower bound; 0 = full causal
) -> Result<(), crate::Error> {
    // The kernel bakes in a hard cap at head_dim > 256 (writes NaN + returns).
    // Reject unsupported head_dim at the wrapper so callers get a clear error
    // rather than silent NaN output. [P1-9 fix.]
    if head_dim > 256 {
        return Err(crate::Error::Backend(format!(
            "qkv_attention: head_dim {} exceeds kernel cap of 256",
            head_dim
        )));
    }
    let q_s = q
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("q must be RocmStorage".into()))?;
    let block_tables_s = block_tables
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("block_tables must be RocmStorage".into()))?;
    let k_pages_s = k_pages
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("k_pages must be RocmStorage".into()))?;
    let v_pages_s = v_pages
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("v_pages must be RocmStorage".into()))?;
    let out_s = out
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("out must be RocmStorage".into()))?;

    let q_ptr = q_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("q has no device ptr".into()))?;
    let block_tables_ptr = block_tables_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("block_tables has no device ptr".into()))?;
    let k_pages_ptr = k_pages_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("k_pages has no device ptr".into()))?;
    let v_pages_ptr = v_pages_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("v_pages has no device ptr".into()))?;
    let out_ptr = out_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("out has no device ptr".into()))?;

    let wf = dev.wavefront_size() as u32;
    let grid_dim = crate::HipDim3::new(batch, num_heads, 1);
    let block_dim = crate::HipDim3::new(wf * 4, 1, 1);
    // Wavefront-aware: W32→128 threads (4×Wave32), W64→256 threads (4×Wave64).
    // The kernel's LDS sizing assumes exactly 4 wavefronts; assert the invariant.
    debug_assert!(
        block_dim.x == wf * 4,
        "qkv_attention: block_dim.x ({}) != wf*4 ({}) — LDS sizing assumes 4 waves",
        block_dim.x,
        wf * 4
    );

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
    // Sliding-window lower bound (0 for full causal; >=0 for SWA). Mirrors the
    // non-paged wrapper's host-side `window_lo_i` computation. Laguna-S-2.1
    // uses seq_len==1 decode, so the block-wide bound is exact.
    let mut wlo = window_lo;

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
            arg(&mut wlo),
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
    let q_s = q
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("q must be RocmStorage".into()))?;
    let k_s = k
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("k must be RocmStorage".into()))?;
    let v_s = v
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("v must be RocmStorage".into()))?;
    let parents_s = tree_parents
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("tree_parents must be RocmStorage".into()))?;
    let out_s = out
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("out must be RocmStorage".into()))?;

    let q_ptr = q_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("q has no device ptr".into()))?;
    let k_ptr = k_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("k has no device ptr".into()))?;
    let v_ptr = v_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("v has no device ptr".into()))?;
    let parents_ptr = parents_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("tree_parents has no device ptr".into()))?;
    let out_ptr = out_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("out has no device ptr".into()))?;

    let wf = dev.wavefront_size() as u32;
    // Wavefront-aware: W32→128 threads, W64→256 threads
    let grid_dim = crate::HipDim3::new(1 + gamma, num_heads, batch);
    let block_dim = crate::HipDim3::new(wf * 4, 1, 1);

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
