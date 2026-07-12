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
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;

    // Only let wave 0 perform the computation.
    if (wave_id > 0) return;

    const int d = lane_id;
    const bool thread_active = d < head_dim;

    if (head_dim > 64) {
        if (thread_active) {
            out[q_offset + d] = nanf("");
        }
        return;
    }

    float out_acc = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Walk K/V up to: each valid kv position j satisfies j <= abs_i (causal).
    for (int j = 0; j <= abs_i && j < kv_seq_len; ++j) {
        const int kv_offset = (j * num_kv_heads + kv_head) * head_dim;

        // Score = dot(q[i,h,:], k[j, kv_head, :])
        float local_score = 0.0f;
        if (thread_active) {
            local_score = q[q_offset + d] * k_tensor[kv_offset + d];
        }

        // Reduce local_score across the wavefront (tree).
        float s = local_score;
        for (int offset = wave_size / 2; offset > 0; offset >>= 1) {
            s += __shfl_xor(s, offset);
        }
        s *= inv_sqrt_d;

        if (!thread_active) continue;

        // ---- online softmax update (numerically stable) ----
        float w = expf(s - running_max);
        if (s > running_max) {
            const float scale = expf(running_max - s);
            running_sum = running_sum * scale;
            out_acc = out_acc * scale;
            running_max = s;
            w = 1.0f;
        }
        out_acc += w * v_tensor[kv_offset + d];
        running_sum += w;
    }

    if (thread_active) {
        out[q_offset + d] = out_acc / running_sum;
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

    // Only let wave 0 perform the computation.
    if (wave_id > 0) return;

    const int d = lane_id;
    const bool thread_active = d < head_dim;

    if (head_dim > 64) {
        if (thread_active) {
            out[q_offset + d] = nanf("");
        }
        return;
    }

    float out_acc = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Get the block table for this batch
    const BlockTableEntry* my_table = block_tables + batch_idx * max_blocks;
    
    // Loop over the pages/blocks
    int num_blocks = (kv_seq_len + page_size - 1) / page_size;
    for (int b = 0; b < num_blocks; ++b) {
        BlockTableEntry entry = my_table[b];
        int num_tokens = entry.page_size;
        
        for (int t = 0; t < num_tokens; ++t) {
            int j = b * page_size + t;
            if (j > abs_i || j >= kv_seq_len) break;
            
            // K/V page layout: [num_pages, page_size, num_kv_heads, head_dim]
            const int physical_token_idx = entry.block_id * page_size + t;
            const int kv_offset = (physical_token_idx * num_kv_heads + kv_head) * head_dim;
            
            float local_score = 0.0f;
            if (thread_active) {
                local_score = q[q_offset + d] * k_pages[kv_offset + d];
            }
            
            float s = local_score;
            for (int offset = wave_size / 2; offset > 0; offset >>= 1) {
                s += __shfl_xor(s, offset);
            }
            s *= inv_sqrt_d;
            
            if (!thread_active) continue;
            
            float w = expf(s - running_max);
            if (s > running_max) {
                const float scale = expf(running_max - s);
                running_sum = running_sum * scale;
                out_acc = out_acc * scale;
                running_max = s;
                w = 1.0f;
            }
            out_acc += w * v_pages[kv_offset + d];
            running_sum += w;
        }
    }

    if (thread_active) {
        out[q_offset + d] = out_acc / running_sum;
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

    // Only let wave 0 perform the computation.
    if (wave_id > 0) return;

    const int d = lane_id;
    const bool thread_active = d < head_dim;

    if (head_dim > 64) {
        if (thread_active) {
            out[q_offset + d] = nanf("");
        }
        return;
    }

    float out_acc = 0.0f;
    float running_max = -1e30f;
    float running_sum = 0.0f;

    for (int j = 0; j < kv_seq_len; ++j) {
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
        
        float local_score = 0.0f;
        if (thread_active) {
            local_score = q[q_offset + d] * k_tensor[kv_offset + d];
        }
        
        float s = local_score;
        for (int offset = wave_size / 2; offset > 0; offset >>= 1) {
            s += __shfl_xor(s, offset);
        }
        s *= inv_sqrt_d;
        
        if (!thread_active) continue;
        
        float w = expf(s - running_max);
        if (s > running_max) {
            const float scale = expf(running_max - s);
            running_sum = running_sum * scale;
            out_acc = out_acc * scale;
            running_max = s;
            w = 1.0f;
        }
        out_acc += w * v_tensor[kv_offset + d];
        running_sum += w;
    }

    if (thread_active) {
        out[q_offset + d] = out_acc / running_sum;
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
