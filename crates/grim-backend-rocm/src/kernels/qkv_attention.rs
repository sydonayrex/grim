//! Phase-1 fused QKV-attention HIP kernel. [see: `grim_qkv_attention_kernel_spec.md`, `rocm-hip-kernels`]

extern crate alloc;

/// Pure parser for the legacy `GRIM_F16_KV` flag. Split out of
/// [`kv_f16_enabled`] so the mapping is unit-testable without touching the
/// process-global environment (which parallel tests share), and so the
/// Qwen35 resolver can share one definition instead of re-parsing the flag
/// with different spellings.
///
/// `None` (unset) is `false`. Only an explicit `0`/`false`/`off`/`no`
/// (case-insensitive) disables; any other value — including garbage — reads
/// as "enabled".
pub fn parse_f16_kv_env(v: Option<&str>) -> bool {
    match v {
        Some(v) => {
            let v = v.trim();
            !(v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no"))
        }
        None => false,
    }
}

/// GRAVE Phase 1: store the LFM2-family KV arena in f16 (2 bytes/element)
/// instead of f32 — halves attention bytes. Read at arena allocation sites
/// (launchers derive representation from storage instead). LFM2-family
/// only; other models' arenas are unaffected while the default is off.
pub fn kv_f16_enabled() -> bool {
    parse_f16_kv_env(std::env::var("GRIM_F16_KV").ok().as_deref())
}

/// The `GRIM_F16_KV` flag read in the **disabled** sense: `true` only when the
/// value is an explicit off spelling. Models whose DEFAULT is f16 (they opt
/// OUT of f16 rather than in) must use this, because
/// [`parse_f16_kv_env`] treats *unset* as "off" — correct for the f32-default
/// backend, wrong for a f16-default model.
///
/// Invariant (pinned by `f16_kv_flag_senses_are_complements`):
/// `parse_f16_kv_env(v) == !f16_kv_explicitly_disabled(v)`.
pub fn f16_kv_explicitly_disabled(v: Option<&str>) -> bool {
    match v {
        Some(v) => {
            let v = v.trim();
            v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no")
        }
        None => false,
    }
}

/// Phase-3 KV-eviction ring policy handed to the graph kernels.
///
/// `cap_rows <= 0` means "no cap": the arena grows with the context and every
/// kernel behaves exactly as before eviction existed. Otherwise the arena
/// holds `cap_rows` token rows — `sink_rows` pinned attention sinks followed
/// by the newest `cap_rows - sink_rows` tokens, overwritten in place. See
/// `grim_kv_ring_row` for the mapping and why the live set stays contiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvRingPolicy {
    pub cap_rows: i32,
    pub sink_rows: i32,
}

impl KvRingPolicy {
    /// No eviction: the identity for every ring kernel.
    pub const UNLIMITED: Self = Self {
        cap_rows: 0,
        sink_rows: 0,
    };

    /// Build a policy, clamping sinks into `[0, cap_rows - 1]` so a bad
    /// `GRIM_KV_SINKS` can never consume the whole arena (which would make the
    /// ring degenerate to `t % cap_rows` and drop the sinks).
    pub fn new(cap_rows: usize, sink_rows: usize) -> Self {
        if cap_rows == 0 {
            return Self::UNLIMITED;
        }
        let cap = i32::try_from(cap_rows).unwrap_or(i32::MAX);
        let sink = i32::try_from(sink_rows)
            .unwrap_or(i32::MAX)
            .min(cap - 1)
            .max(0);
        Self {
            cap_rows: cap,
            sink_rows: sink,
        }
    }

    pub fn is_limited(self) -> bool {
        self.cap_rows > 0
    }

    /// Host twin of `grim_kv_ring_row`: the arena row an ABSOLUTE token row
    /// maps to. `KvRingPolicy::row_matches_device_arithmetic` pins this against
    /// the inlined device formula, and every host-side caller (the eager
    /// append, the capture-topology validators) goes through here so there is
    /// exactly one copy of the mapping in Rust.
    pub fn row(self, t: usize) -> usize {
        if self.cap_rows <= 0 {
            return t;
        }
        let t = t as i64;
        let cap = self.cap_rows as i64;
        if t < cap {
            return t as usize;
        }
        let span = cap - self.sink_rows as i64;
        if span <= 0 {
            return (t % cap) as usize;
        }
        (self.sink_rows as i64 + (t - self.sink_rows as i64) % span) as usize
    }
}

/// Dense KV-arena representation on ROCm.
///
/// `GRIM_KV_ARENA=f32|f16|int8` wins when set; otherwise the legacy
/// `GRIM_F16_KV` maps on (any non-`0`/`false`) to F16 and anything else to
/// F32. Global default is F32 — models opt into narrower formats through
/// their own `kv_arena_dtype` (Qwen35 defaults to F16; int8 is always
/// explicit). Launchers never read this: they derive the representation
/// from arena storage (storage truth).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvArenaFormat {
    F32,
    F16,
    Int8,
    Int4,
    /// Asymmetric 2-bit keys / 4-bit values (WS-3A). The representation is
    /// declared here so the selector is first-class and cannot be mistaken for
    /// another format, but its DEVICE KERNELS DO NOT EXIST YET — every arena
    /// entry point must refuse it rather than fall through (int2v4 and int8 are
    /// both byte-addressed, so a fallthrough would serve 2-bit codes as 8-bit).
    Int2V4,
}

impl KvArenaFormat {
    /// Bytes per stored element, excluding sidecars (scales ride separately).
    /// `Int2V4` is the average of its two widths: 2-bit K + 4-bit V.
    pub fn bytes_per_elem(self) -> f32 {
        match self {
            KvArenaFormat::F32 => 4.0,
            KvArenaFormat::F16 => 2.0,
            KvArenaFormat::Int8 => 1.0,
            KvArenaFormat::Int4 => 0.5,
            KvArenaFormat::Int2V4 => 0.375,
        }
    }

    /// True for representations whose device kernels are implemented. Every
    /// declared format is implemented as of WS-3A (`int2v4` gained its 2-bit-K
    /// quantizer and its mixed-width attention kernel); the query stays so a
    /// future format can declare itself incomplete and be refused centrally.
    pub fn has_device_kernels(self) -> bool {
        match self {
            KvArenaFormat::F32
            | KvArenaFormat::F16
            | KvArenaFormat::Int8
            | KvArenaFormat::Int4
            | KvArenaFormat::Int2V4 => true,
        }
    }
}

/// Pure parser for an explicit `GRIM_KV_ARENA` value. `None` means "not one
/// of the known spellings" — the caller decides whether that is a typo
/// (warn, fall back) or simply unset. This is the SINGLE place the explicit
/// format spellings live; the Qwen35 model resolver delegates here too, so an
/// explicit `GRIM_KV_ARENA` can never be read two different ways.
pub fn parse_kv_arena_env(v: &str) -> Option<KvArenaFormat> {
    match v {
        "f32" => Some(KvArenaFormat::F32),
        "f16" => Some(KvArenaFormat::F16),
        "int8" | "i8" | "q8" => Some(KvArenaFormat::Int8),
        "int4" | "i4" | "q4" => Some(KvArenaFormat::Int4),
        // WS-3A: asymmetric bits. Resolves, then every arena entry point
        // refuses it (`has_device_kernels` == false) instead of mis-serving.
        "int2v4" | "i2v4" | "k2v4" => Some(KvArenaFormat::Int2V4),
        _ => None,
    }
}

/// Pure core of [`kv_arena_format`]: resolve from already-read env strings.
/// Kept pure so the mapping is unit-testable without the process-global
/// environment.
pub(crate) fn resolve_kv_arena_format(arena: Option<&str>, f16_kv: Option<&str>) -> KvArenaFormat {
    match arena {
        Some(v) => parse_kv_arena_env(v).unwrap_or_else(|| {
            eprintln!("[kv] GRIM_KV_ARENA={v:?} unrecognized (f32|f16|int8|int4); using F32");
            KvArenaFormat::F32
        }),
        None => {
            if parse_f16_kv_env(f16_kv) {
                KvArenaFormat::F16
            } else {
                KvArenaFormat::F32
            }
        }
    }
}

/// Resolve the dense-arena representation (see [`KvArenaFormat`]).
pub fn kv_arena_format() -> KvArenaFormat {
    let arena = std::env::var("GRIM_KV_ARENA").ok();
    let f16 = std::env::var("GRIM_F16_KV").ok();
    resolve_kv_arena_format(arena.as_deref(), f16.as_deref())
}

/// HIP source for `grim_qkv_attention`. [see: `COMPUTE_KERNEL_SOURCE`, `lib.rs::RocmDevice::qkv_attention`, `j`, `__shared__`]
pub const KERNEL_SOURCE: &str = r#"
// HIPRTC compiles this unit standalone: provide the infinity constant the
// masked-tile softmax kernels rely on (C <math.h> is not pulled in).
#ifndef INFINITY
#define INFINITY (__builtin_huge_valf())
#endif
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
    float softcap,      // logit softcapping cap (e.g. 50.0f for Gemma-2). Pass <=0.0f to disable.
                        // WI-F2 - fused O-projection epilogue.
    const float* __restrict__ o_proj_w,
    int o_dim,
    int fuse_o,
    // ALiBi per-head slopes [num_heads]. When has_alibi != 0 the score for
    // query abs_i / key j gains slopes[h] * (j - abs_i). Pass null/0 to skip.
    const float* __restrict__ alibi_slopes,
    int has_alibi
) {
    // grid = (seq_len, num_heads, 1); block = (blockDim.x, 1, 1).
    // The host launches block_dim_x = 128 on RDNA2 (gfx1036, Wave32: 4 wavefronts) or 256 on.
    const int i = blockIdx.x;             // query position (0..seq_len)
    const int h = blockIdx.y;             // head index
    if (i >= seq_len || h >= num_heads) return;

    // GQA mapping: every (num_heads/num_kv_heads) query heads share one kv_head.
    // The host validates that num_heads % num_kv_heads == 0, so this is exact.
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;

    // Pointers to this head's q column / kv column.
    // Layouts (Phase-1 contract): q: [seq_len, num_heads, head_dim] -> q_offset = (i * num_heads + h).
    const int q_offset = (i * num_heads + h) * head_dim;

    // Cache offset: query position i within this call is at absolute position
    // (cache_offset + i). All past K/V positions are valid up to that.
    const int abs_i = cache_offset + i;

    // ────────────────────────────────────────────────────────────────────── Phase 1: online softmax.
    // Running max + running weighted sum, no full score vector materialized; kv_seq_len may exceed the.
    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    // num_waves = actual launched block size / wave size.
    // Uses blockDim.x (not a compile-time constant) so it matches the real launch: 128 on gfx1036.
    const int num_waves = blockDim.x / wave_size;

    // Every head_dim axis loop must cover the SAME range as the q staging loop
    // (8 chunks). It used to be 4, which covered only 4*wave_size = 128 dims, so
    // for head_dim 256 the upper half of every output row was never written and
    // read back as whatever the allocation held - observed as exactly half the
    // output elements wrong. s_acc[8][260] and the head_dim>256 guard already
    // assumed 256.
    const int d = lane_id;
    const bool thread_active = d < head_dim;

    // Hardware-aware head-dim cap.
    if (head_dim > 256) {
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = nanf("");
            }
        }
        return;
    }

    // Per-wavefront partials published to LDS for the wave-0 merge.
    // Sized to worst-case 8 wavefronts (RDNA iGPU Wave32 + block 256 host path) and head.
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][260];

    // Causal KV range for this query: [lo, hi) where: hi = min(abs_i + 1, kv_seq_len) (standard causal upper bound) lo = window_lo (0 for full attention; sliding lower
    // bound for SWA) `window_lo` is pre-computed on the host as max(0, abs_i - window + 1) so the kernel stays branch-free for the common full-causal case (window_lo == 0).
    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int lo = window_lo;             // 0 for full causal; >= 0 for SWA
    const int range_len = hi - lo;        // may be 0 if lo >= hi (empty window)

    // Quarter-stride partitioning of [lo, hi) across the wavefronts.
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    // j_start / j_end are offsets into [0, range_len); add `lo` when accessing KV.
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Fast-path GQA key/value stride pointers
    const float* __restrict__ k_head = &k_tensor[kv_head * head_dim];
    const float* __restrict__ v_head = &v_tensor[kv_head * head_dim];
    const int kv_stride = num_kv_heads * head_dim;

    // Stage this lane's strided slice of q into registers ONCE.
    // The previous form re-fetched the whole q head from global memory on every KV token.
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
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        if (has_alibi) {
            score += alibi_slopes[h] * (float)(j - abs_i);
        }

        // Online-softmax update
        const float old_max = running_max;
        running_max = fmaxf(running_max, score);
        const float scale_old = expf(old_max - running_max);
        const float scale_new = expf(score - running_max);

        running_sum = running_sum * scale_old + scale_new;
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_head[j * (num_kv_heads * head_dim) + d];
            }
        }
    }

    // Publish per-wavefront partials to LDS. max/sum are wavefront-uniform (all lanes see same j_start/j_end loop range).
    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 8; ++chunk) {
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
    float attn_reg[8];
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
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
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = attn_reg[chunk];
            }
        }
    } else {
        // WI-F2 fused O-projection epilogue. Each wave-0 lane owns its strided slice of the head_dim axis; per output column oc
        // the lane partial is butterfly-reduced across the wavefront and lane 0 atomically accumulates into the (host-zeroed) fused output row.
        for (int oc = 0; oc < o_dim; ++oc) {
            float partial = 0.0f;
            #pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
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


// int8-KV variant of `grim_qkv_attention` (Phase 2 KV shrink): identical
// math, K/V rows read as biased-u8 codes with f32 per-(row,kv_head)
// scales. q, out, and all epilogues are untouched f32.
extern "C" __global__ __launch_bounds__(256)
void grim_qkv_attention_i8kv(
    const float* __restrict__ q,
    const float* __restrict__ k_tensor,
    const float* __restrict__ v_tensor,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
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
    float softcap,      // logit softcapping cap (e.g. 50.0f for Gemma-2). Pass <=0.0f to disable.
                        // WI-F2 - fused O-projection epilogue.
    const float* __restrict__ o_proj_w,
    int o_dim,
    int fuse_o,
    // ALiBi per-head slopes [num_heads]. When has_alibi != 0 the score for
    // query abs_i / key j gains slopes[h] * (j - abs_i). Pass null/0 to skip.
    const float* __restrict__ alibi_slopes,
    int has_alibi
) {
    // grid = (seq_len, num_heads, 1); block = (blockDim.x, 1, 1).
    // The host launches block_dim_x = 128 on RDNA2 (gfx1036, Wave32: 4 wavefronts) or 256 on.
    const int i = blockIdx.x;             // query position (0..seq_len)
    const int h = blockIdx.y;             // head index
    if (i >= seq_len || h >= num_heads) return;

    // GQA mapping: every (num_heads/num_kv_heads) query heads share one kv_head.
    // The host validates that num_heads % num_kv_heads == 0, so this is exact.
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;

    // Pointers to this head's q column / kv column.
    // Layouts (Phase-1 contract): q: [seq_len, num_heads, head_dim] -> q_offset = (i * num_heads + h).
    const int q_offset = (i * num_heads + h) * head_dim;

    // Cache offset: query position i within this call is at absolute position
    // (cache_offset + i). All past K/V positions are valid up to that.
    const int abs_i = cache_offset + i;

    // ────────────────────────────────────────────────────────────────────── Phase 1: online softmax.
    // Running max + running weighted sum, no full score vector materialized; kv_seq_len may exceed the.
    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    // num_waves = actual launched block size / wave size.
    // Uses blockDim.x (not a compile-time constant) so it matches the real launch: 128 on gfx1036.
    const int num_waves = blockDim.x / wave_size;

    // Every head_dim axis loop must cover the SAME range as the q staging loop
    // (8 chunks). It used to be 4, which covered only 4*wave_size = 128 dims, so
    // for head_dim 256 the upper half of every output row was never written and
    // read back as whatever the allocation held - observed as exactly half the
    // output elements wrong. s_acc[8][260] and the head_dim>256 guard already
    // assumed 256.
    const int d = lane_id;
    const bool thread_active = d < head_dim;

    // Hardware-aware head-dim cap.
    if (head_dim > 256) {
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = nanf("");
            }
        }
        return;
    }

    // Per-wavefront partials published to LDS for the wave-0 merge.
    // Sized to worst-case 8 wavefronts (RDNA iGPU Wave32 + block 256 host path) and head.
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][260];

    // Causal KV range for this query: [lo, hi) where: hi = min(abs_i + 1, kv_seq_len) (standard causal upper bound) lo = window_lo (0 for full attention; sliding lower
    // bound for SWA) `window_lo` is pre-computed on the host as max(0, abs_i - window + 1) so the kernel stays branch-free for the common full-causal case (window_lo == 0).
    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int lo = window_lo;             // 0 for full causal; >= 0 for SWA
    const int range_len = hi - lo;        // may be 0 if lo >= hi (empty window)

    // Quarter-stride partitioning of [lo, hi) across the wavefronts.
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    // j_start / j_end are offsets into [0, range_len); add `lo` when accessing KV.
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Fast-path GQA key/value stride pointers
    // int8-KV variant (Phase 2): codes are u8 (biased +128, same element
    // layout as f32 rows, 1 B/elem); scales are f32 per (row, kv_head).
    const unsigned char* __restrict__ k_u8 =
        ((const unsigned char*)k_tensor) + (long long)kv_head * head_dim;
    const unsigned char* __restrict__ v_u8 =
        ((const unsigned char*)v_tensor) + (long long)kv_head * head_dim;
    const int kv_stride = num_kv_heads * head_dim;

    // Stage this lane's strided slice of q into registers ONCE.
    // The previous form re-fetched the whole q head from global memory on every KV token.
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
                float k_dq = ((float)((int)k_u8[j * kv_stride + dd] - 128) / 127.0f)
                    * k_scales[(long long)j * num_kv_heads + kv_head];
                partial += q_reg[chunk] * k_dq;
            }
        }
        const unsigned long long shfl_mask = 0xffffffffffffffffULL;
        #pragma unroll
        for (int off = wave_size >> 1; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(shfl_mask, partial, off);
        }
        float score = partial * inv_sqrt_d;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        if (has_alibi) {
            score += alibi_slopes[h] * (float)(j - abs_i);
        }

        // Online-softmax update
        const float old_max = running_max;
        running_max = fmaxf(running_max, score);
        const float scale_old = expf(old_max - running_max);
        const float scale_new = expf(score - running_max);

        running_sum = running_sum * scale_old + scale_new;
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                float v_dq = ((float)((int)v_u8[j * (num_kv_heads * head_dim) + d] - 128) / 127.0f)
                    * v_scales[(long long)j * num_kv_heads + kv_head];
                out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_dq;
            }
        }
    }

    // Publish per-wavefront partials to LDS. max/sum are wavefront-uniform (all lanes see same j_start/j_end loop range).
    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 8; ++chunk) {
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
    float attn_reg[8];
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
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
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = attn_reg[chunk];
            }
        }
    } else {
        // WI-F2 fused O-projection epilogue. Each wave-0 lane owns its strided slice of the head_dim axis; per output column oc
        // the lane partial is butterfly-reduced across the wavefront and lane 0 atomically accumulates into the (host-zeroed) fused output row.
        for (int oc = 0; oc < o_dim; ++oc) {
            float partial = 0.0f;
            #pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
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

// int4-KV variant of `grim_qkv_attention` (Phase 3 KV shrink: ~2.2GB @ 262K):
// K/V rows read as packed nibbles (2 values per byte, low nibble first,
// biased by +7, range -7..+7 mapped to 0..14).
// Scales are f32 per (row, kv_head).
// q, out, and all epilogues are untouched f32.
extern "C" __global__ __launch_bounds__(256)
void grim_qkv_attention_i4kv(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_tensor,
    const unsigned char* __restrict__ v_tensor,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ out,
    float* __restrict__ out_max,
    float* __restrict__ out_sum,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_seq_len,
    int cache_offset,
    float inv_sqrt_d,
    int window_lo,
    float softcap,
    const float* __restrict__ o_proj_w,
    int o_dim,
    int fuse_o,
    const float* __restrict__ alibi_slopes,
    int has_alibi
) {
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    if (i >= seq_len || h >= num_heads) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = (i * num_heads + h) * head_dim;
    const int abs_i = cache_offset + i;

    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    const int num_waves = blockDim.x / wave_size;

    if (head_dim > 256) {
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = nanf("");
            }
        }
        return;
    }

    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][260];

    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int lo = window_lo;
    const int range_len = hi - lo;

    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Packed int4 storage: 2 values per byte. Row stride in bytes is (num_kv_heads * head_dim) / 2.
    // Offset for head is (kv_head * head_dim) / 2 bytes.
    const int kv_stride_bytes = (num_kv_heads * head_dim) / 2;
    const int head_offset_bytes = (kv_head * head_dim) / 2;
    const unsigned char* __restrict__ k_bytes = k_tensor + head_offset_bytes;
    const unsigned char* __restrict__ v_bytes = v_tensor + head_offset_bytes;

    float q_reg[8];
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
        int dd = lane_id + chunk * wave_size;
        q_reg[chunk] = (dd < head_dim) ? q[q_offset + dd] : 0.0f;
    }

    for (int j = lo + j_start; j < lo + j_end; ++j) {
        float partial = 0.0f;
        const long long row_base_bytes = (long long)j * kv_stride_bytes;
        const float k_sc = k_scales[(long long)j * num_kv_heads + kv_head];
        #pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
            int dd = lane_id + chunk * wave_size;
            if (dd < head_dim) {
                unsigned char packed = k_bytes[row_base_bytes + (dd >> 1)];
                int code = (dd & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                float k_dq = ((float)(code - 7) / 7.0f) * k_sc;
                partial += q_reg[chunk] * k_dq;
            }
        }
        const unsigned long long shfl_mask = 0xffffffffffffffffULL;
        #pragma unroll
        for (int off = wave_size >> 1; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(shfl_mask, partial, off);
        }
        float score = partial * inv_sqrt_d;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        if (has_alibi) {
            score += alibi_slopes[h] * (float)(j - abs_i);
        }

        const float old_max = running_max;
        running_max = fmaxf(running_max, score);
        const float scale_old = expf(old_max - running_max);
        const float scale_new = expf(score - running_max);

        running_sum = running_sum * scale_old + scale_new;
        const float v_sc = v_scales[(long long)j * num_kv_heads + kv_head];
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                unsigned char packed = v_bytes[row_base_bytes + (d >> 1)];
                int code = (d & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                float v_dq = ((float)(code - 7) / 7.0f) * v_sc;
                out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_dq;
            }
        }
    }

    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 8; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            s_acc[wave_id][d] = out_acc[chunk];
        } else if (d < 256) {
            s_acc[wave_id][d] = 0.0f;
        }
    }
    __syncthreads();

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

    float attn_reg[8];
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
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
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = attn_reg[chunk];
            }
        }
    } else {
        for (int oc = 0; oc < o_dim; ++oc) {
            float partial = 0.0f;
            #pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
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

// f16-KV variant of `grim_qkv_attention` (Phase 1 KV shrink): bit-identical
// math, K/V rows read as fp16 through u16 views. q, out, and all
// epilogues (max/sum/softcap/alibi/fused-O) are untouched f32.
extern "C" __global__ __launch_bounds__(256)
void grim_qkv_attention_f16kv(
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
    float softcap,      // logit softcapping cap (e.g. 50.0f for Gemma-2). Pass <=0.0f to disable.
                        // WI-F2 - fused O-projection epilogue.
    const float* __restrict__ o_proj_w,
    int o_dim,
    int fuse_o,
    // ALiBi per-head slopes [num_heads]. When has_alibi != 0 the score for
    // query abs_i / key j gains slopes[h] * (j - abs_i). Pass null/0 to skip.
    const float* __restrict__ alibi_slopes,
    int has_alibi
) {
    // grid = (seq_len, num_heads, 1); block = (blockDim.x, 1, 1).
    // The host launches block_dim_x = 128 on RDNA2 (gfx1036, Wave32: 4 wavefronts) or 256 on.
    const int i = blockIdx.x;             // query position (0..seq_len)
    const int h = blockIdx.y;             // head index
    if (i >= seq_len || h >= num_heads) return;

    // GQA mapping: every (num_heads/num_kv_heads) query heads share one kv_head.
    // The host validates that num_heads % num_kv_heads == 0, so this is exact.
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;

    // Pointers to this head's q column / kv column.
    // Layouts (Phase-1 contract): q: [seq_len, num_heads, head_dim] -> q_offset = (i * num_heads + h).
    const int q_offset = (i * num_heads + h) * head_dim;

    // Cache offset: query position i within this call is at absolute position
    // (cache_offset + i). All past K/V positions are valid up to that.
    const int abs_i = cache_offset + i;

    // ────────────────────────────────────────────────────────────────────── Phase 1: online softmax.
    // Running max + running weighted sum, no full score vector materialized; kv_seq_len may exceed the.
    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    // num_waves = actual launched block size / wave size.
    // Uses blockDim.x (not a compile-time constant) so it matches the real launch: 128 on gfx1036.
    const int num_waves = blockDim.x / wave_size;

    // Every head_dim axis loop must cover the SAME range as the q staging loop
    // (8 chunks). It used to be 4, which covered only 4*wave_size = 128 dims, so
    // for head_dim 256 the upper half of every output row was never written and
    // read back as whatever the allocation held - observed as exactly half the
    // output elements wrong. s_acc[8][260] and the head_dim>256 guard already
    // assumed 256.
    const int d = lane_id;
    const bool thread_active = d < head_dim;

    // Hardware-aware head-dim cap.
    if (head_dim > 256) {
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = nanf("");
            }
        }
        return;
    }

    // Per-wavefront partials published to LDS for the wave-0 merge.
    // Sized to worst-case 8 wavefronts (RDNA iGPU Wave32 + block 256 host path) and head.
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][260];

    // Causal KV range for this query: [lo, hi) where: hi = min(abs_i + 1, kv_seq_len) (standard causal upper bound) lo = window_lo (0 for full attention; sliding lower
    // bound for SWA) `window_lo` is pre-computed on the host as max(0, abs_i - window + 1) so the kernel stays branch-free for the common full-causal case (window_lo == 0).
    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int lo = window_lo;             // 0 for full causal; >= 0 for SWA
    const int range_len = hi - lo;        // may be 0 if lo >= hi (empty window)

    // Quarter-stride partitioning of [lo, hi) across the wavefronts.
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    // j_start / j_end are offsets into [0, range_len); add `lo` when accessing KV.
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Fast-path GQA key/value stride pointers
    const float* __restrict__ k_head = &k_tensor[kv_head * head_dim];
    const float* __restrict__ v_head = &v_tensor[kv_head * head_dim];
    // f16-KV variant: same math, K/V read through u16 views (arenas hold
    // fp16 when GRIM_F16_KV is on; q stays f32).
    //
    // STRIDE CORRECTNESS (do not "simplify"): the head base MUST be computed
    // in u16 units from the tensor base. `&k_tensor[kv_head*head_dim]` advances
    // kv_head*head_dim FLOATS (= ×4 bytes); reinterpreting that address as u16
    // puts every kv_head>0 at 2x its true offset (head 0 accidentally exact,
    // all later heads out-of-bounds). Index from the u16 base instead.
    const unsigned short* __restrict__ k_head16 =
        ((const unsigned short*)k_tensor) + kv_head * head_dim;
    const unsigned short* __restrict__ v_head16 =
        ((const unsigned short*)v_tensor) + kv_head * head_dim;
    (void)k_head;
    (void)v_head;
    const int kv_stride = num_kv_heads * head_dim;

    // Stage this lane's strided slice of q into registers ONCE.
    // The previous form re-fetched the whole q head from global memory on every KV token.
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
                partial += q_reg[chunk] * fp16_bits_to_float_device(k_head16[j * kv_stride + dd]);
            }
        }
        const unsigned long long shfl_mask = 0xffffffffffffffffULL;
        #pragma unroll
        for (int off = wave_size >> 1; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(shfl_mask, partial, off);
        }
        float score = partial * inv_sqrt_d;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        if (has_alibi) {
            score += alibi_slopes[h] * (float)(j - abs_i);
        }

        // Online-softmax update
        const float old_max = running_max;
        running_max = fmaxf(running_max, score);
        const float scale_old = expf(old_max - running_max);
        const float scale_new = expf(score - running_max);

        running_sum = running_sum * scale_old + scale_new;
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * fp16_bits_to_float_device(v_head16[j * (num_kv_heads * head_dim) + d]);
            }
        }
    }

    // Publish per-wavefront partials to LDS. max/sum are wavefront-uniform (all lanes see same j_start/j_end loop range).
    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 8; ++chunk) {
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
    float attn_reg[8];
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
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
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = attn_reg[chunk];
            }
        }
    } else {
        // WI-F2 fused O-projection epilogue. Each wave-0 lane owns its strided slice of the head_dim axis; per output column oc
        // the lane partial is butterfly-reduced across the wavefront and lane 0 atomically accumulates into the (host-zeroed) fused output row.
        for (int oc = 0; oc < o_dim; ++oc) {
            float partial = 0.0f;
            #pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
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
    // num_waves = actual launched block size / wave size.
    // Uses blockDim.x (not a compile-time constant) so it matches the real launch: 128 on gfx1036.
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
    __shared__ float s_acc[8][260];

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
        // (lo already lowers the bound in the partition; this guards the edge on the last.
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
    // num_waves = actual launched block size / wave size.
    // Uses blockDim.x (not a compile-time constant) so it matches the real launch: 128 on gfx1036.
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
    __shared__ float s_acc[8][260];

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

extern "C" __global__ __launch_bounds__(64)
void grim_qkv_attention_wmma(
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
    int cache_offset,
    float inv_sqrt_d,
    int window_lo,
    const float* __restrict__ o_proj_w,
    int o_dim,
    int fuse_o
) {
    // One block = one wave = one 16-query tile of a single head.
    // The 16-query x 16-key score tile is staged in shared memory and filled with explicit.
    (void)o_proj_w; (void)o_dim; (void)fuse_o; (void)out_max; (void)out_sum;

    const int h = blockIdx.y;
    if (h >= num_heads) return;
    const int i0 = blockIdx.x * 16;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int kv_stride = num_kv_heads * head_dim;
    const int q_stride = num_heads * head_dim;
    const float* q_tile = q + ((long)i0 * num_heads + h) * head_dim;
    const float* k_tile = k_tensor + kv_head * head_dim;
    const float* v_tile = v_tensor + kv_head * head_dim;

    const int tid = threadIdx.x;
    const int wave_size = warpSize;

    __shared__ float s_scores[16][16];
    __shared__ float s_m[16];
    __shared__ float s_sum[16];
    __shared__ float s_acc[16][256];

    for (int idx = tid; idx < 16 * 256; idx += wave_size) {
        s_acc[idx / 256][idx % 256] = 0.0f;
    }
    for (int r = tid; r < 16; r += wave_size) {
        s_m[r] = -INFINITY;
        s_sum[r] = 0.0f;
    }

    const int last_row = i0 + 15 < seq_len ? i0 + 15 : seq_len - 1;
    const int abs_last = cache_offset + last_row;
    const int range_lo = window_lo > 0 ? window_lo : 0;
    int range_hi = abs_last + 1 < kv_seq_len ? abs_last + 1 : kv_seq_len;
    if (range_hi < range_lo) range_hi = range_lo;

    for (int j0 = range_lo; j0 < range_hi; j0 += 16) {
        int k_count = 16;
        if (j0 + 16 > range_hi) k_count = range_hi - j0;

        // Fill the score tile: s_scores[k][r] = Q[i0+r] . K[j0+k], masked
        // entries set to -INFINITY so softmax drops them.
        for (int idx = tid; idx < 256; idx += wave_size) {
            int k = idx / 16;
            int r = idx % 16;
            if (k >= k_count || i0 + r >= seq_len || j0 + k > cache_offset + i0 + r) {
                s_scores[k][r] = -INFINITY;
            } else {
                const float* q_row = q_tile + (long)r * q_stride;
                const float* k_row = k_tile + (long)(j0 + k) * kv_stride;
                float acc = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    acc += q_row[d] * k_row[d];
                }
                s_scores[k][r] = acc;
            }
        }
        __syncthreads();

        // Per-row online softmax over this key chunk. Key j attends to query
        // row i0+r only when j <= cache_offset + i0 + r (causal).
        for (int r = 0; r < 16 && i0 + r < seq_len; ++r) {
            const int abs_i = cache_offset + i0 + r;
            float m_new = s_m[r];
            for (int k = 0; k < k_count; ++k) {
                if (j0 + k <= abs_i) {
                    m_new = fmaxf(m_new, s_scores[k][r] * inv_sqrt_d);
                }
            }
            if (m_new == -INFINITY) continue;  // this chunk is fully masked

            const float scale = expf(s_m[r] - m_new);
            s_sum[r] *= scale;
            for (int d = tid; d < head_dim; d += wave_size) {
                s_acc[r][d] *= scale;
            }
            s_m[r] = m_new;

            for (int k = 0; k < k_count; ++k) {
                const int j = j0 + k;
                if (j > abs_i) continue;
                const float w = expf(s_scores[k][r] * inv_sqrt_d - m_new);
                s_sum[r] += w;
                const float* v_row = v_tile + (long)j * kv_stride;
                for (int d = tid; d < head_dim; d += wave_size) {
                    s_acc[r][d] += w * v_row[d];
                }
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < 16 && i0 + r < seq_len; ++r) {
        const float inv = s_sum[r] > 0.0f ? 1.0f / s_sum[r] : 0.0f;
        float* out_row = out + ((long)(i0 + r) * num_heads + h) * head_dim;
        for (int d = tid; d < head_dim; d += wave_size) {
            out_row[d] = s_acc[r][d] * inv;
        }
    }
}

__device__ inline float fp8_e4m3_scale_dev(unsigned char b) {
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0x0F;
    int mant = b & 0x07;
    float v;
    if (exp == 0xF) {
        v = (mant == 7) ? 0.0f / 0.0f : 448.0f;
    } else if (exp != 0) {
        v = (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, exp - 7);
    } else {
        v = (float)mant / 512.0f;
    }
    return sign ? -v : v;
}

__device__ inline float dequant_kv_element(
    const unsigned char* data,
    int idx,
    int quant_format,
    float scale,
    float bias
) {
    if (quant_format == 0) { // INT8 (signed i8)
        signed char val = (signed char)data[idx];
        return ((float)val) * scale + bias;
    } else if (quant_format == 1) { // INT4 / W4A16 (packed signed nibbles)
        int byte_idx = idx >> 1;
        unsigned char b = data[byte_idx];
        int nibble = (idx & 1) ? ((int)(b >> 4)) : ((int)(b & 0x0F));
        if (nibble >= 8) nibble -= 16;
        return ((float)nibble) * scale + bias;
    } else if (quant_format == 2) { // FP8_E4M3
        unsigned char b = data[idx];
        int sign = (b >> 7) & 1;
        int exp = (b >> 3) & 0x0F;
        int mant = b & 0x07;
        float val = 0.0f;
        if (exp == 0) {
            val = (float)mant / 8.0f * (1.0f / 64.0f);
        } else if (exp == 15 && mant == 7) {
            val = 0.0f;
        } else {
            val = (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, exp - 7);
        }
        return (sign ? -val : val) * scale;
    } else if (quant_format == 3) { // FP8_E5M2
        unsigned char b = data[idx];
        int sign = (b >> 7) & 1;
        int exp = (b >> 2) & 0x1F;
        int mant = b & 0x03;
        float val = 0.0f;
        if (exp == 0) {
            val = (float)mant / 4.0f * (1.0f / 16384.0f);
        } else if (exp == 31) {
            val = 0.0f;
        } else {
            val = (1.0f + (float)mant / 4.0f) * ldexpf(1.0f, exp - 15);
        }
        return (sign ? -val : val) * scale;
    } else if (quant_format == 4) { // FP4_E2M1 (packed 4-bit)
        int byte_idx = idx >> 1;
        unsigned char b = data[byte_idx];
        int nibble = (idx & 1) ? (b >> 4) : (b & 0x0F);
        int sign = (nibble >> 3) & 1;
        int exp = (nibble >> 1) & 0x03;
        int mant = nibble & 1;
        // OCP E2M1 subnormal step is 0.5 (codes 0x1/0x9 -> +/-0.5), matching
        // mxfp4_to_float_hip and the CUDA NVFP4 LUT.
        float val = (exp == 0) ? ((float)mant * 0.5f) : ((1.0f + (float)mant * 0.5f) * ldexpf(1.0f, exp - 1));
        return (sign ? -val : val) * scale;
    } else if (quant_format == 5) { // MXFP4 (OCP E2M1 + E8M0 scale per 32)
        // 17 bytes per 32 values: 1 E8M0 scale byte then 16 code bytes.
        // This arm previously ignored the block scale and used the caller's
        // per-tensor `scale`, which is not MXFP4 at all.
        const int MXFP4_GROUP = 32;
        const int MXFP4_BYTES = 17;
        int grp = idx / MXFP4_GROUP;
        int in_grp = idx % MXFP4_GROUP;
        const unsigned char* g = data + grp * MXFP4_BYTES;
        float blk_scale = exp2f((float)(int)g[0] - 127.0f);
        unsigned char cb = g[1 + (in_grp >> 1)];
        int nibble = (in_grp & 1) ? (cb >> 4) : (cb & 0x0F);
        int sign = (nibble >> 3) & 1;
        int exp = (nibble >> 1) & 0x03;
        int mant = nibble & 1;
        float val = (exp == 0) ? ((float)mant * 0.5f) : ((1.0f + (float)mant * 0.5f) * ldexpf(1.0f, exp - 1));
        return (sign ? -val : val) * blk_scale * scale;
    } else if (quant_format == 6) { // MXFP8 (OCP E4M3 + E8M0 scale per 32)
        // 33 bytes per 32 values: 1 E8M0 scale byte then 32 E4M3 code bytes.
        const int MXFP8_GROUP = 32;
        const int MXFP8_BYTES = 33;
        int grp = idx / MXFP8_GROUP;
        int in_grp = idx % MXFP8_GROUP;
        const unsigned char* g = data + grp * MXFP8_BYTES;
        float blk_scale = exp2f((float)(int)g[0] - 127.0f);
        unsigned char b = g[1 + in_grp];
        int sign = (b >> 7) & 1;
        // E4M3 has FOUR exponent bits. This was masked with 0x07, truncating
        // every value >= 2^8 to the wrong binade.
        int exp = (b >> 3) & 0x0F;
        int mant = b & 0x07;
        float val = (exp == 0) ? ((float)mant / 512.0f) : ((1.0f + (float)mant / 8.0f) * ldexpf(1.0f, exp - 7));
        return (sign ? -val : val) * blk_scale * scale;
    } else if (quant_format == 7) { // NVFP4 (E2M1 + inline E4M3 scale per 16)
        // 9 bytes per 16 values: [E4M3 scale byte][8 code bytes]. Zero codes
        // decode to a real zero.
        int sub = idx >> 4;
        int in_sub = idx & 15;
        const unsigned char* blk = data + sub * 9;
        float blk_scale = fp8_e4m3_scale_dev(blk[0]);
        unsigned char code_byte = blk[1 + (in_sub >> 1)];
        int nibble = (in_sub & 1) ? (code_byte >> 4) : (code_byte & 0x0F);
        int sign = (nibble >> 3) & 1;
        int exp = (nibble >> 1) & 0x03;
        int mant = nibble & 1;
        float val = (exp == 0) ? ((float)mant * 0.5f) : ((1.0f + (float)mant * 0.5f) * ldexpf(1.0f, exp - 1));
        return (sign ? -val : val) * blk_scale;
    } else if (quant_format == 8) { // NUTCRACKER (E2M1 + inline per-16 block scale)
        // 9 bytes per 16 values: [scale byte][8 code bytes]. The scale byte is
        // [exp:6 | sel:2] — the low bits are stolen as the special-value
        // selector, so the caller's per-tensor `scale` is deliberately unused.
        int sub = idx >> 4;
        int in_sub = idx & 15;
        const unsigned char* blk = data + sub * 9;
        int sel = blk[0] & 0x3;
        float blk_scale = ldexpf(1.0f, (int)(blk[0] >> 2) - 31);
        unsigned char code_byte = blk[1 + (in_sub >> 1)];
        int nibble = (in_sub & 1) ? (code_byte >> 4) : (code_byte & 0x0F);
        if ((nibble & 0x7) == 0) {
            // Repurposed zero encoding: emit this block's special value. The
            // sign comes from the selector, not from the nibble's sign bit.
            float mag = (sel & 0x1) ? 2.5f : 5.0f;
            return ((sel & 0x2) ? -mag : mag) * blk_scale;
        }
        int sign = (nibble >> 3) & 1;
        int exp = (nibble >> 1) & 0x03;
        int mant = nibble & 1;
        float val = (exp == 0) ? ((float)mant * 0.5f) : ((1.0f + (float)mant * 0.5f) * ldexpf(1.0f, exp - 1));
        return (sign ? -val : val) * blk_scale;
    }
    return 0.0f;
}

extern "C" __global__ __launch_bounds__(256)
void grim_qkv_attention_paged_quant(
    const float* __restrict__ q,
    const BlockTableEntry* __restrict__ block_tables,
    const unsigned char* __restrict__ k_pages,
    const unsigned char* __restrict__ v_pages,
    float* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks,
    int page_size,
    int kv_seq_len,
    int cache_offset,
    float inv_sqrt_d,
    int window_lo,
    int quant_format,
    float k_scale,
    float k_bias,
    float v_scale,
    float v_bias
) {
    const int batch_idx = blockIdx.x;
    const int h = blockIdx.y;
    if (batch_idx >= 1 && blockIdx.z > 0) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = (batch_idx * num_heads + h) * head_dim;
    const int abs_i = cache_offset;

    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    const int num_waves = blockDim.x / wave_size;

    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][260];

    const int range_lo = window_lo > 0 ? window_lo : 0;
    const int range_hi = abs_i + 1 < kv_seq_len ? abs_i + 1 : kv_seq_len;
    const int range_len = range_hi > range_lo ? range_hi - range_lo : 0;

    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    const int j_start = range_lo + wave_id * base + (wave_id < rem ? wave_id : rem);
    const int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    const int page_stride = page_size * num_kv_heads * head_dim;

    for (int j = j_start; j < j_end; ++j) {
        const int page_idx = j / page_size;
        const int page_offset = j % page_size;
        if (page_idx >= max_blocks) continue;

        const BlockTableEntry entry = block_tables[batch_idx * max_blocks + page_idx];
        const int physical_block = entry.block_id;

        const int elem_offset = physical_block * page_stride + (page_offset * num_kv_heads + kv_head) * head_dim;

        float score = 0.0f;
        #pragma unroll
        for (int dim = 0; dim < 256; ++dim) {
            if (dim < head_dim) {
                float k_val = dequant_kv_element(k_pages, elem_offset + dim, quant_format, k_scale, k_bias);
                score += q[q_offset + dim] * k_val;
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
                float v_val = dequant_kv_element(v_pages, elem_offset + d, quant_format, v_scale, v_bias);
                out_acc[chunk] += w * v_val;
            }
        }
        running_sum += w;
    }

    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 4; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            s_acc[wave_id][d] = out_acc[chunk];
        }
    }
    __syncthreads();

    if (wave_id == 0) {
        for (int chunk = 0; chunk < 4; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d >= head_dim) continue;

            float m_final = s_max[0];
            float sum_final = s_sum[0];
            float acc_final = s_acc[0][d];

            for (int w = 1; w < num_waves; ++w) {
                float mw = s_max[w];
                float uw = s_sum[w];
                float aw = s_acc[w][d];
                if (uw == 0.0f) continue;
                if (sum_final == 0.0f) {
                    m_final = mw; sum_final = uw; acc_final = aw;
                    continue;
                }
                float m_new = m_final > mw ? m_final : mw;
                float scale_a = expf(m_final - m_new);
                float scale_b = expf(mw - m_new);
                sum_final = sum_final * scale_a + uw * scale_b;
                acc_final = acc_final * scale_a + aw * scale_b;
                m_final = m_new;
            }
            float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;
            out[q_offset + d] = acc_final * inv_sum;
        }
    }
}


// Phase 3 (KV eviction): map an ABSOLUTE token row to its arena row under a
// fixed-capacity ring with pinned attention sinks.
//
//   rows [0, sink_rows)            -> the first tokens, never evicted
//   rows [sink_rows, cap_rows)     -> the newest (cap_rows - sink_rows) tokens,
//                                     overwritten in place
//
// The live set is therefore ALWAYS the contiguous range [0, min(t, cap_rows)),
// which is why the attention kernels only need a capacity clamp and no gather.
// cap_rows <= 0 means "no cap" (today's grow-with-context behavior).
//
// The eager path passes absolute offsets and maps on the host; only the graph
// kernels need this on-device.
__device__ inline int grim_kv_ring_row(int t, int cap_rows, int sink_rows) {
    if (cap_rows <= 0 || t < cap_rows) return t;
    const int span = cap_rows - sink_rows;
    if (span <= 0) return t % cap_rows;  // sinks consumed the arena
    return sink_rows + (t - sink_rows) % span;
}

// Phase 2 (int8 KV arenas): append-quantize. One block per (slot, step, kv_head)
// head-vector quantizes `head_dim` f32 values to symmetric int8
// (q = clamp(round(x/scale), -127, 127), scale = maxabs/127, all-zero row
// stores scale 1.0 + zero codes) into the codes arena, plus one f32 scale
// into the scales sidecar. `head_dim` <= 256 (block 256, one thread/elem).
// Strides are in ELEMENTS of their own buffer.
extern "C" __global__ void grim_kv_append_i8(
    const float* __restrict__ k_rot,
    const int* __restrict__ past_dev,
    unsigned char* __restrict__ codes,
    float* __restrict__ scales,
    int kv_stride,
    int nkv_heads,
    int head_dim,
    int steps,
    int batch,
    int arena_slot_stride,
    int scale_slot_stride,
    int cap_rows,
    int sink_rows) {
    const int vec = blockIdx.x;  // one head-vector per block
    const int tid = threadIdx.x;
    const int vecs_per_slot = steps * nkv_heads;
    const int total_vecs = batch * vecs_per_slot;
    if (vec >= total_vecs) return;
    const int slot = vec / vecs_per_slot;
    const int rem = vec - slot * vecs_per_slot;
    const int step = rem / nkv_heads;
    const int kh = rem - step * nkv_heads;
    const int past = past_dev[slot];
    // Phase 3: codes and scales are indexed by the SAME (ring-mapped) token
    // row — they must never drift apart, or the reader pairs a row's codes
    // with another row's scale.
    const int abs_row = grim_kv_ring_row(past + step, cap_rows, sink_rows);
    const long long src_base = ((long long)slot * steps + step) * kv_stride + (long long)kh * head_dim;
    const long long dst_base = (long long)slot * arena_slot_stride + (long long)abs_row * kv_stride + (long long)kh * head_dim;
    const long long sc_idx = (long long)slot * scale_slot_stride + (long long)abs_row * nkv_heads + kh;
    const int wave_size = warpSize;
    const int num_waves = blockDim.x / wave_size;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    float v = (tid < head_dim) ? k_rot[src_base + tid] : 0.0f;
    float amax = fabsf(v);
    #pragma unroll
    for (int off = wave_size >> 1; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffULL, amax, off));
    // Cross-wave fold through shared (mirrors grim_qkv_attention), then
    // broadcast the winning scale: every quantizing thread needs the SAME
    // denominator, not its wave's partial max.
    __shared__ float s_max[8];
    __shared__ float s_scale;
    if (lane_id == 0) s_max[wave_id] = amax;
    __syncthreads();
    float mx = (tid < num_waves) ? s_max[tid] : 0.0f;
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffULL, mx, off));
    // NOTE: lanes outside wave 0 hold a partial max here; only lane 0's
    // value is final (contributors tids 0..num_waves-1 all live in wave 0
    // when num_waves <= wave_size, i.e. block 256 on wave32/wave64).
    float scale = 1.0f;
    if (tid == 0) {
        // Legacy int8 convention (matches `grim_kv_dequant_attention`): the
        // stored scale is the vector maxabs and codes are fractions of it,
        // dequant = (code-128)/127*scale. NOT the step size.
        scale = (mx > 1e-30f) ? mx : 1.0f;
        scales[sc_idx] = scale;
        s_scale = scale;
    }
    __syncthreads();
    scale = s_scale;
    if (tid < head_dim) {
        // Reciprocal of the BROADCAST winning scale (not the thread's
        // partial mx): every code in the vector shares one denominator.
        // Codes are fractions of maxabs: q = round(v/scale*127).
        float inv = (scale > 1e-30f) ? (127.0f / scale) : 0.0f;
        int q = (int)__builtin_roundf(v * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        // Stored biased by 128 (u8); readers subtract 128 (matches the
        // legacy int8 dequant convention `(int8-128)/127*scale` with the
        // sign folded into the byte).
        codes[dst_base + tid] = (unsigned char)(q + 128);
    }
}

// Item 3 — device-side KV append. Replaces the host-offset `copy_slice_into`:
// each thread reads `past = past_dev[slot]` ON DEVICE, computes its own destination
// offset `slot*arena_slot_stride + past*kv_stride + local`, and copies one element.
// Offsets are computed entirely on-device so the append is graph-capturable
// (no host scalar baked in). `batch==1` → slot is always 0 (legacy layout).
extern "C" __global__ void grim_kv_append(
    float* __restrict__ k_arena,
    const float* __restrict__ k_rot,
    const int* __restrict__ past_dev,
    int kv_stride,
    int steps,
    int batch,
    int arena_slot_stride,
    int f16,
    int cap_rows,
    int sink_rows
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int per_slot = steps * kv_stride;
    const int total = batch * per_slot;
    if (tid >= total) return;
    const int slot = (batch > 1) ? (tid / per_slot) : 0;
    const int local = tid - slot * per_slot;
    const int past = past_dev[slot];
    // Phase 3: map the token row through the eviction ring. `local` walks the
    // whole steps x kv_stride span, so the row advances every kv_stride
    // elements and the in-row column is the remainder.
    const int row_local = local / kv_stride;
    const int col = local - row_local * kv_stride;
    const int dst_tok = grim_kv_ring_row(past + row_local, cap_rows, sink_rows);
    long long off = (long long)slot * arena_slot_stride + (long long)dst_tok * kv_stride + col;
    if (f16) {
        // GRAVE Phase 1: arena holds f16 (2-byte) elements at the same
        // element offsets; convert on store.
        unsigned short* a16 = (unsigned short*)k_arena;
        a16[off] = f32_to_fp16_bits_device(k_rot[tid]);
    } else {
        k_arena[off] = k_rot[tid];
    }
}

// Item 3 — device-total attention variant. Identical math to `grim_qkv_attention`
// EXCEPT the KV length `kv_seq_len` is read from a device scalar (`total_dev`)
// instead of being baked in as a host parameter. This makes the attention node
// graph-capturable even though `past` changes every token: total_dev is a graph
// input pointer, replayed with the live value. Only the kv_seq_len parameter is
// replaced; cache_offset is dropped (decode is always steps==1 at abs position
// total-1, which the caller passes as part of total_dev semantics via total only).

// ─────────────────────────────────────────────────────────────────────────────
// Query-tiled prefill attention.
//
// `grim_qkv_attention` launches one block per (query, head) and each block reads
// its whole causal KV range from DRAM, so a K/V element is re-read once per
// query: at 3928 tokens that is 126.4 GB per launch, and the measured 287 ms is
// exactly that traffic at ~440 GB/s (this card is 256-bit GDDR6, ~640 GB/s
// theoretical). The kernel is at its memory roofline — the only lever is to
// read K/V less often.
//
// This variant puts QT consecutive queries in one block, so a K/V element read
// from DRAM is reused QT times out of registers. Traffic falls by ~QT.
//
// Contract: head_dim <= 256, block = 128 (4 waves), no sliding window, no
// softcap, no alibi — callers that need those keep the scalar kernel. Each
// query keeps its own online-softmax state and its own causal bound
// (abs_i = cache_offset + i), so the tiling does not change the result.
extern "C" __global__ void grim_qkv_attention_tiled(
    const float* __restrict__ q,
    const float* __restrict__ k_tensor,
    const float* __restrict__ v_tensor,
    float* __restrict__ out,
    int num_heads, int num_kv_heads, int head_dim, int seq_len, int kv_seq_len,
    int cache_offset, float inv_sqrt_d, int qt
) {
    constexpr int QMAX = 4;
    const int h = blockIdx.y;
    const int i0 = blockIdx.x * qt;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int num_waves = blockDim.x >> 5;
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int kv_stride = num_kv_heads * head_dim;
    const int nqt = (qt < QMAX) ? qt : QMAX;
    // QMAX is the UNROLL width: unrolling wider than the tile makes every K/V
    // row do the wider tile's work and blows the register arrays (measured:
    // QMAX=8 with qt=4 was slower than the scalar kernel it replaced).

    __shared__ float s_m[QMAX][4];
    __shared__ float s_s[QMAX][4];
    __shared__ float s_acc[QMAX][4][260];

    // Per-query state: q in registers, online-softmax max/sum, and the
    // value accumulator (8 chunks of head_dim/32).
    float qreg[QMAX][8];
    float acc[QMAX][8];
    float rmax[QMAX];
    float rsum[QMAX];
    #pragma unroll
    for (int t = 0; t < QMAX; ++t) {
        rmax[t] = -1e30f;
        rsum[t] = 0.0f;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            acc[t][c] = 0.0f;
            const int d = lane + c * 32;
            const int i = i0 + t;
            qreg[t][c] = (t < nqt && i < seq_len && d < head_dim)
                ? q[((long long)i * num_heads + h) * head_dim + d] : 0.0f;
        }
    }

    // Block KV range: [0, hi_max). A query t only owns keys up to its own
    // abs_i, and abs_i grows with t, so for a given j the participating
    // queries are exactly t >= j - (cache_offset + i0).
    const int abs_i_last = cache_offset + i0 + nqt - 1;
    const int hi = (abs_i_last + 1 < kv_seq_len) ? (abs_i_last + 1) : kv_seq_len;
    const int range_len = (hi > 0) ? hi : 0;
    const int base = range_len / num_waves;
    const int rem = range_len % num_waves;
    const int j_start = wave * base + (wave < rem ? wave : rem);
    const int j_end = j_start + base + (wave < rem ? 1 : 0);

    const unsigned short* k16 = (const unsigned short*)k_tensor;
    const unsigned short* v16 = (const unsigned short*)v_tensor;

    // Software-pipelined loads: row j+1 is fetched while row j is consumed.
    // Without this the loop serializes on its two HBM loads per iteration —
    // the kernel was latency-bound at ~50 GB/s achieved, well under the
    // ~440 GB/s the traffic alone would allow.
    auto load_kv_row = [&](int j, float* kr, float* vr) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int d = lane + c * 32;
            if (d < head_dim) {
                const long long off = (long long)kv_head * head_dim + (long long)j * kv_stride + d;
                kr[c] = fp16_bits_to_float_device(k16[off]);
                vr[c] = fp16_bits_to_float_device(v16[off]);
            } else {
                kr[c] = 0.0f;
                vr[c] = 0.0f;
            }
        }
    };
    float kreg[8];
    float vreg[8];
    if (j_start < j_end) load_kv_row(j_start, kreg, vreg);
    for (int j = j_start; j < j_end; ++j) {
        float knext[8];
        float vnext[8];
        if (j + 1 < j_end) load_kv_row(j + 1, knext, vnext);
        const int t_min = (j - cache_offset - i0 > 0) ? (j - cache_offset - i0) : 0;
        #pragma unroll
        for (int t = 0; t < QMAX; ++t) {
            if (t >= nqt || t < t_min) continue;
            float score = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; ++c) score += qreg[t][c] * kreg[c];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) score += __shfl_xor(score, off);
            score *= inv_sqrt_d;
            const float m_new = fmaxf(rmax[t], score);
            const float sa = expf(rmax[t] - m_new);
            const float sb = expf(score - m_new);
            rmax[t] = m_new;
            rsum[t] = rsum[t] * sa + sb;
            #pragma unroll
            for (int c = 0; c < 8; ++c) acc[t][c] = acc[t][c] * sa + sb * vreg[c];
        }
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            kreg[c] = knext[c];
            vreg[c] = vnext[c];
        }
    }

    // Publish per-(query, wave) partials, then let wave 0 merge them.
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < QMAX; ++t) {
            s_m[t][wave] = rmax[t];
            s_s[t][wave] = rsum[t];
        }
    }
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        const int d = lane + c * 32;
        if (d < 256) {
            #pragma unroll
            for (int t = 0; t < QMAX; ++t) {
                s_acc[t][wave][d] = (t < nqt && d < head_dim) ? acc[t][c] : 0.0f;
            }
        }
    }
    __syncthreads();
    if (wave != 0) return;

    #pragma unroll
    for (int t = 0; t < QMAX; ++t) {
        if (t >= nqt) continue;
        const int i = i0 + t;
        if (i >= seq_len) continue;
        float m_final = s_m[t][0];
        float sum_final = s_s[t][0];
        #pragma unroll
        for (int w = 1; w < 4; ++w) {
            if (w >= num_waves) break;
            const float mw = s_m[t][w];
            const float uw = s_s[t][w];
            const float m_new = fmaxf(m_final, mw);
            const float sa = expf(m_final - m_new);
            const float sb = expf(mw - m_new);
            sum_final = sum_final * sa + uw * sb;
            m_final = m_new;
        }
        const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;
        float* orow = out + ((long long)i * num_heads + h) * head_dim;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int d = lane + c * 32;
            if (d < head_dim) {
                float a = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    if (w >= num_waves) break;
                    a += s_acc[t][w][d] * expf(s_m[t][w] - m_final);
                }
                orow[d] = a * inv_sum;
            }
        }
    }
}

extern "C" __global__ void grim_qkv_attention_dev(
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
    const int* __restrict__ total_dev,
    int steps,
    float inv_sqrt_d,
    int window_lo,
    float softcap,
    const float* __restrict__ o_proj_w,
    int o_dim,
    int fuse_o,
    const float* __restrict__ alibi_slopes,
    int has_alibi,
    int batch,
    int arena_slot_stride,
    int kv_f16,
    int cap_rows   // Phase 3 ring capacity in token rows; <= 0 means unlimited
) {
    // grid = (seq_len, num_heads, batch); block = (blockDim.x, 1, 1).
    // P3: blockIdx.z is the batch slot — every slot has its own arena region
    // (`slot * arena_slot_stride`) and its own position scalar (`total_dev[slot]`).
    const int slot = blockIdx.z;
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    if (i >= seq_len || h >= num_heads) return;
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = slot * (seq_len * num_heads * head_dim)
        + (i * num_heads + h) * head_dim;
    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    const int num_waves = blockDim.x / wave_size;
    const int d = lane_id;
    const bool thread_active = d < head_dim;
    if (head_dim > 256) {
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) out[q_offset + d] = nanf("");
        }
        return;
    }
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][260];
    // Device-driven KV length. `total_dev` holds the per-slot pre-step past
    // count (bumped by grim_bump_i32 AFTER this attention node), and `steps` is
    // the number of rows the preceding kv_append node appended IN EACH SLOT —
    // so the live KV length during this attention is `total_dev[slot] + steps`.
    // `steps` is a graph-safe constant (1 for decode), so replay N times
    // re-derives the right total from the live per-slot total_dev each time.
    int live_rows = total_dev[slot] + steps;
    // Phase 3: under the eviction ring the arena is a fixed capacity and the
    // live tokens are the contiguous prefix [0, min(live_rows, cap_rows)) —
    // sinks plus the newest tokens. Clamping here is the whole read-side
    // change; no gather is needed. cap_rows <= 0 means unlimited.
    const int kv_seq_len = (cap_rows > 0 && live_rows > cap_rows) ? cap_rows : live_rows;
    // Decode: single query at the last live row (all live rows are <= it).
    const int abs_i = kv_seq_len - 1;
    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    // A window lower bound computed from the ABSOLUTE position can exceed the
    // live range once tokens have been evicted; clamp so the range is never
    // inverted (hi < lo).
    const int lo = (window_lo < kv_seq_len) ? window_lo : kv_seq_len;
    const int range_len = hi - lo;
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);
    float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;
    const unsigned short* k_head16 = (const unsigned short*)k_tensor;
    const unsigned short* v_head16 = (const unsigned short*)v_tensor;
    const float* __restrict__ k_head = &k_tensor[slot * arena_slot_stride + kv_head * head_dim];
    const float* __restrict__ v_head = &v_tensor[slot * arena_slot_stride + kv_head * head_dim];
    const int kv_stride = num_kv_heads * head_dim;
    float q_reg[8];
    if (thread_active) {
        for (int c = 0; c < 8; ++c) {
            int idx = d + c * wave_size;
            q_reg[c] = (idx < head_dim) ? q[q_offset + idx] : 0.0f;
        }
    }
    for (int j = j_start; j < j_end; ++j) {
        const int kv_idx = lo + j;
        float score = 0.0f;
        for (int c = 0; c < 8; ++c) {
            int idx = d + c * wave_size;
            if (idx < head_dim) {
                float kvv = kv_f16
                    ? fp16_bits_to_float_device(k_head16[(long long)(slot * arena_slot_stride + kv_head * head_dim) + kv_idx * kv_stride + idx])
                    : k_head[kv_idx * kv_stride + idx];
                score += q_reg[c] * kvv;
            }
        }
        // Wave-uniform total score (every lane holds the sum).
        for (int off = wave_size / 2; off > 0; off >>= 1)
            score += __shfl_xor(score, off);
        score *= inv_sqrt_d;
        if (thread_active) {
            if (softcap > 0.0f) {
                float t = score / softcap;
                score = softcap * logf(1.0f + expf((t > -20.0f) ? t : -20.0f));
            }
            if (has_alibi) score += alibi_slopes[h] * ((float)kv_idx - (float)abs_i);
            float mw = fmaxf(running_max, score);
            float scale_a = expf(running_max - mw);
            float scale_b = expf(score - mw);
            running_sum = running_sum * scale_a + scale_b;
            for (int c = 0; c < 8; ++c) {
                int idx = d + c * wave_size;
                if (idx < head_dim) {
                    float vv = kv_f16
                        ? fp16_bits_to_float_device(v_head16[(long long)(slot * arena_slot_stride + kv_head * head_dim) + kv_idx * kv_stride + idx])
                        : v_head[kv_idx * kv_stride + idx];
                    out_acc[c] = out_acc[c] * scale_a + scale_b * vv;
                }
            }
            running_max = mw;
        }
    }
    // Publish per-wavefront partials: max/sum are wave-uniform (lane 0 stores
    // the wave's scalars); acc is per-lane strided. Empty waves keep
    // running_max = -1e30 and are skipped by the merge (scale 0).
    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int c = 0; c < 8; ++c) {
        int idx = d + c * wave_size;
        if (idx < head_dim) {
            s_acc[wave_id][idx] = out_acc[c];
        } else if (idx < 256) {
            s_acc[wave_id][idx] = 0.0f;
        }
    }
    __syncthreads();
    if (wave_id != 0) return;

    // Wave-0 merge of per-wave scalars (same pattern as grim_qkv_attention).
    float m_final = s_max[0];
    float sum_final = s_sum[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) {
        if (w >= num_waves) break;
        const float mw = s_max[w];
        const float uw = s_sum[w];
        const float m_new = fmaxf(m_final, mw);
        sum_final = sum_final * expf(m_final - m_new) + uw * expf(mw - m_new);
        m_final = m_new;
    }
    const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;

    // Each lane reconstructs its strided slice of the normalized attention row.
    float attn_reg[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        attn_reg[c] = 0.0f;
        int idx = d + c * wave_size;
        if (idx < head_dim) {
            float acc_final = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w >= num_waves) break;
                acc_final += s_acc[w][idx] * expf(s_max[w] - m_final);
            }
            attn_reg[c] = acc_final * inv_sum;
        }
    }

    if (fuse_o == 0) {
        for (int c = 0; c < 8; ++c) {
            int idx = d + c * wave_size;
            if (idx < head_dim) out[q_offset + idx] = attn_reg[c];
        }
    } else {
        // WI-F2 fused O-projection epilogue: each wave-0 lane owns a strided
        // slice of head_dim; per output column the lane partial is
        // butterfly-reduced and lane 0 atomically accumulates into the
        // (host-zeroed) fused output row. `out` here is [seq_len, o_dim].
        for (int oc = 0; oc < o_dim; ++oc) {
            float partial = 0.0f;
            for (int c = 0; c < 8; ++c) {
                int idx = d + c * wave_size;
                if (idx < head_dim) {
                    partial += attn_reg[c] * o_proj_w[(h * head_dim + idx) * o_dim + oc];
                }
            }
            for (int off = wave_size >> 1; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, off);
            }
            if (lane_id == 0 && partial != 0.0f) {
                atomicAdd(&out[slot * (seq_len * o_dim) + i * o_dim + oc], partial);
            }
        }
    }
}


// int8-KV device-total variant of `grim_qkv_attention_dev` (Phase 2):
// identical math and capture semantics, K/V rows read as biased-u8
// codes with f32 per-(slot,row,kv_head) scales. No f16/f32 legs.
extern "C" __global__ void grim_qkv_attention_dev_i8(
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
    const int* __restrict__ total_dev,
    int steps,
    float inv_sqrt_d,
    int window_lo,
    float softcap,
    const float* __restrict__ o_proj_w,
    int o_dim,
    int fuse_o,
    const float* __restrict__ alibi_slopes,
    int has_alibi,
    int batch,
    int arena_slot_stride,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    int scale_slot_stride,
    int cap_rows   // Phase 3 ring capacity in token rows; <= 0 means unlimited
) {
    // grid = (seq_len, num_heads, batch); block = (blockDim.x, 1, 1).
    // P3: blockIdx.z is the batch slot — every slot has its own arena region
    // (`slot * arena_slot_stride`) and its own position scalar (`total_dev[slot]`).
    const int slot = blockIdx.z;
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    if (i >= seq_len || h >= num_heads) return;
    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = slot * (seq_len * num_heads * head_dim)
        + (i * num_heads + h) * head_dim;
    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    const int num_waves = blockDim.x / wave_size;
    const int d = lane_id;
    const bool thread_active = d < head_dim;
    if (head_dim > 256) {
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) out[q_offset + d] = nanf("");
        }
        return;
    }
    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][260];
    // Device-driven KV length. `total_dev` holds the per-slot pre-step past
    // count (bumped by grim_bump_i32 AFTER this attention node), and `steps` is
    // the number of rows the preceding kv_append node appended IN EACH SLOT —
    // so the live KV length during this attention is `total_dev[slot] + steps`.
    // `steps` is a graph-safe constant (1 for decode), so replay N times
    // re-derives the right total from the live per-slot total_dev each time.
    int live_rows = total_dev[slot] + steps;
    // Phase 3: under the eviction ring the arena is a fixed capacity and the
    // live tokens are the contiguous prefix [0, min(live_rows, cap_rows)) —
    // sinks plus the newest tokens. Clamping here is the whole read-side
    // change; no gather is needed. cap_rows <= 0 means unlimited.
    const int kv_seq_len = (cap_rows > 0 && live_rows > cap_rows) ? cap_rows : live_rows;
    // Decode: single query at the last live row (all live rows are <= it).
    const int abs_i = kv_seq_len - 1;
    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    // A window lower bound computed from the ABSOLUTE position can exceed the
    // live range once tokens have been evicted; clamp so the range is never
    // inverted (hi < lo).
    const int lo = (window_lo < kv_seq_len) ? window_lo : kv_seq_len;
    const int range_len = hi - lo;
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);
    float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;
    // int8-KV variant (Phase 2): codes are biased-u8 (same element layout
    // as f32 rows, 1 B/elem); scales are f32 per (slot, row, kv_head).
    const unsigned char* __restrict__ k_u8 = (const unsigned char*)k_tensor;
    const unsigned char* __restrict__ v_u8 = (const unsigned char*)v_tensor;
    const int kv_stride = num_kv_heads * head_dim;
    float q_reg[8];
    if (thread_active) {
        for (int c = 0; c < 8; ++c) {
            int idx = d + c * wave_size;
            q_reg[c] = (idx < head_dim) ? q[q_offset + idx] : 0.0f;
        }
    }
    for (int j = j_start; j < j_end; ++j) {
        const int kv_idx = lo + j;
        float score = 0.0f;
        for (int c = 0; c < 8; ++c) {
            int idx = d + c * wave_size;
            if (idx < head_dim) {
                float kvv = ((float)((int)k_u8[(long long)slot * arena_slot_stride + (long long)kv_idx * kv_stride + (long long)kv_head * head_dim + idx] - 128) / 127.0f)
                    * k_scales[(long long)slot * scale_slot_stride + (long long)kv_idx * num_kv_heads + kv_head];
                score += q_reg[c] * kvv;
            }
        }
        // Wave-uniform total score (every lane holds the sum).
        for (int off = wave_size / 2; off > 0; off >>= 1)
            score += __shfl_xor(score, off);
        score *= inv_sqrt_d;
        if (thread_active) {
            if (softcap > 0.0f) {
                float t = score / softcap;
                score = softcap * logf(1.0f + expf((t > -20.0f) ? t : -20.0f));
            }
            if (has_alibi) score += alibi_slopes[h] * ((float)kv_idx - (float)abs_i);
            float mw = fmaxf(running_max, score);
            float scale_a = expf(running_max - mw);
            float scale_b = expf(score - mw);
            running_sum = running_sum * scale_a + scale_b;
            for (int c = 0; c < 8; ++c) {
                int idx = d + c * wave_size;
                if (idx < head_dim) {
                    float vv = ((float)((int)v_u8[(long long)slot * arena_slot_stride + (long long)kv_idx * kv_stride + (long long)kv_head * head_dim + idx] - 128) / 127.0f)
                        * v_scales[(long long)slot * scale_slot_stride + (long long)kv_idx * num_kv_heads + kv_head];
                    out_acc[c] = out_acc[c] * scale_a + scale_b * vv;
                }
            }
            running_max = mw;
        }
    }
    // Publish per-wavefront partials: max/sum are wave-uniform (lane 0 stores
    // the wave's scalars); acc is per-lane strided. Empty waves keep
    // running_max = -1e30 and are skipped by the merge (scale 0).
    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int c = 0; c < 8; ++c) {
        int idx = d + c * wave_size;
        if (idx < head_dim) {
            s_acc[wave_id][idx] = out_acc[c];
        } else if (idx < 256) {
            s_acc[wave_id][idx] = 0.0f;
        }
    }
    __syncthreads();
    if (wave_id != 0) return;

    // Wave-0 merge of per-wave scalars (same pattern as grim_qkv_attention).
    float m_final = s_max[0];
    float sum_final = s_sum[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) {
        if (w >= num_waves) break;
        const float mw = s_max[w];
        const float uw = s_sum[w];
        const float m_new = fmaxf(m_final, mw);
        sum_final = sum_final * expf(m_final - m_new) + uw * expf(mw - m_new);
        m_final = m_new;
    }
    const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;

    // Each lane reconstructs its strided slice of the normalized attention row.
    float attn_reg[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        attn_reg[c] = 0.0f;
        int idx = d + c * wave_size;
        if (idx < head_dim) {
            float acc_final = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w >= num_waves) break;
                acc_final += s_acc[w][idx] * expf(s_max[w] - m_final);
            }
            attn_reg[c] = acc_final * inv_sum;
        }
    }

    if (fuse_o == 0) {
        for (int c = 0; c < 8; ++c) {
            int idx = d + c * wave_size;
            if (idx < head_dim) out[q_offset + idx] = attn_reg[c];
        }
    } else {
        // WI-F2 fused O-projection epilogue: each wave-0 lane owns a strided
        // slice of head_dim; per output column the lane partial is
        // butterfly-reduced and lane 0 atomically accumulates into the
        // (host-zeroed) fused output row. `out` here is [seq_len, o_dim].
        for (int oc = 0; oc < o_dim; ++oc) {
            float partial = 0.0f;
            for (int c = 0; c < 8; ++c) {
                int idx = d + c * wave_size;
                if (idx < head_dim) {
                    partial += attn_reg[c] * o_proj_w[(h * head_dim + idx) * o_dim + oc];
                }
            }
            for (int off = wave_size >> 1; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, off);
            }
            if (lane_id == 0 && partial != 0.0f) {
                atomicAdd(&out[slot * (seq_len * o_dim) + i * o_dim + oc], partial);
            }
        }
    }
}

// P3: per-slot position bump. Grid (batch,1,1)/(1,1,1) — thread b bumps
// `past_dev[b] += steps`. batch=1 callers launch grid (1,1,1): slot 0.
extern "C" __global__ void grim_bump_i32(
    int* __restrict__ past_dev,
    int steps
) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    past_dev[slot] += steps;
}

// ScytheRing: write one 64-byte task descriptor into ring slot `slot_idx`
// entirely on-device. Replaces host-pack + H2D so the enqueue is HIP-graph
// capturable. Thread 0 writes the packed fields; the slot array is pre-zeroed
// at alloc so status=PENDING and trailing bytes need no explicit clear.
// Layout (native-endian, matches pack_gemm_descriptor):
//   [0..4)  opcode  [4..8)  m  [8..12)  n  [12..16)  k
//   [16..24) input_ptr  [24..32) weight_ptr  [32..40) output_ptr  [40..48) peer_ptr
extern "C" __global__ void grim_scythe_write_slot(
    unsigned char* __restrict__ slots_base,
    unsigned int slot_idx,
    unsigned int opcode,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    unsigned long long input_ptr,
    unsigned long long weight_ptr,
    unsigned long long output_ptr,
    unsigned long long peer_ptr
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned char* slot = slots_base + (unsigned long long)slot_idx * 64ull;
        *(unsigned int*)(slot + 0)  = opcode;
        *(unsigned int*)(slot + 4)  = m;
        *(unsigned int*)(slot + 8)  = n;
        *(unsigned int*)(slot + 12) = k;
        *(unsigned long long*)(slot + 16) = input_ptr;
        *(unsigned long long*)(slot + 24) = weight_ptr;
        *(unsigned long long*)(slot + 32) = output_ptr;
        *(unsigned long long*)(slot + 40) = peer_ptr;
    }
}

// ScytheRing: publish the head counter on-device. Writes a single u32 to
// head_dev so the persistent dispatch wave sees the new task count. On-device
// (graph-capturable) instead of the legacy pinned-staging + H2D publish.
extern "C" __global__ void grim_scythe_publish_head(
    unsigned int* __restrict__ head_dev,
    unsigned int head_value
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *head_dev = head_value;
    }
}

// Phase 2 (eager int8 append): quantize explicit f32 rows to biased-u8 codes
// + f32 per-head scales at explicit destination offsets (host-computed, like
// `copy_slice_range`). One block per (row, kv_head) head-vector; block 256.
extern "C" __global__ void grim_kv_quantize_rows_i8(
    const float* __restrict__ src,
    int src_off,
    unsigned char* __restrict__ codes,
    int codes_off,
    float* __restrict__ scales,
    int scales_off,
    int n_rows,
    int kv_stride,
    int nkv_heads,
    int head_dim) {
    const int vec = blockIdx.x;  // one head-vector per block
    const int tid = threadIdx.x;
    const int vecs = n_rows * nkv_heads;
    if (vec >= vecs) return;
    const int row = vec / nkv_heads;
    const int kh = vec - row * nkv_heads;
    const long long src_base = (long long)src_off + (long long)row * kv_stride + (long long)kh * head_dim;
    const long long dst_base = (long long)codes_off + (long long)row * kv_stride + (long long)kh * head_dim;
    const long long sc_idx = (long long)scales_off + (long long)row * nkv_heads + kh;
    const int wave_size = warpSize;
    const int num_waves = blockDim.x / wave_size;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    float v = (tid < head_dim) ? src[src_base + tid] : 0.0f;
    float amax = fabsf(v);
    #pragma unroll
    for (int off = wave_size >> 1; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffULL, amax, off));
    __shared__ float s_max[8];
    __shared__ float s_scale;
    if (lane_id == 0) s_max[wave_id] = amax;
    __syncthreads();
    float mx = (tid < num_waves) ? s_max[tid] : 0.0f;
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffULL, mx, off));
    float scale = 1.0f;
    if (tid == 0) {
        // Legacy int8 convention (matches `grim_kv_dequant_attention`): the
        // stored scale is the vector maxabs and codes are fractions of it,
        // dequant = (code-128)/127*scale. NOT the step size.
        scale = (mx > 1e-30f) ? mx : 1.0f;
        scales[sc_idx] = scale;
        s_scale = scale;
    }
    __syncthreads();
    scale = s_scale;
    if (tid < head_dim) {
        // Reciprocal of the BROADCAST winning scale (not the thread's
        // partial mx): every code in the vector shares one denominator.
        // Codes are fractions of maxabs: q = round(v/scale*127).
        float inv = (scale > 1e-30f) ? (127.0f / scale) : 0.0f;
        int q = (int)__builtin_roundf(v * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        codes[dst_base + tid] = (unsigned char)(q + 128);
    }
}

// Phase 3 (eager int4 append): quantize explicit f32 rows to packed u4 nibbles
// (2 values per byte, low nibble first, biased by +7) + f32 per-head scales.
// One block per (row, kv_head) head-vector; block 256.
extern "C" __global__ void grim_kv_quantize_rows_i4(
    const float* __restrict__ src,
    int src_off,
    unsigned char* __restrict__ codes,
    int codes_off,
    float* __restrict__ scales,
    int scales_off,
    int n_rows,
    int kv_stride,
    int nkv_heads,
    int head_dim) {
    const int vec = blockIdx.x;
    const int tid = threadIdx.x;
    const int vecs = n_rows * nkv_heads;
    if (vec >= vecs) return;
    const int row = vec / nkv_heads;
    const int kh = vec - row * nkv_heads;
    const long long src_base = (long long)src_off + (long long)row * kv_stride + (long long)kh * head_dim;
    // codes row stride in bytes is kv_stride / 2. Head offset is kh * head_dim / 2.
    const long long dst_base = (long long)codes_off + (long long)row * (kv_stride / 2) + (long long)kh * (head_dim / 2);
    const long long sc_idx = (long long)scales_off + (long long)row * nkv_heads + kh;
    const int wave_size = warpSize;
    const int num_waves = blockDim.x / wave_size;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;

    float v = (tid < head_dim) ? src[src_base + tid] : 0.0f;
    float amax = fabsf(v);
    #pragma unroll
    for (int off = wave_size >> 1; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffULL, amax, off));
    __shared__ float s_max[8];
    __shared__ float s_scale;
    if (lane_id == 0) s_max[wave_id] = amax;
    __syncthreads();
    float mx = (tid < num_waves) ? s_max[tid] : 0.0f;
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffULL, mx, off));
    float scale = 1.0f;
    if (tid == 0) {
        scale = (mx > 1e-30f) ? mx : 1.0f;
        scales[sc_idx] = scale;
        s_scale = scale;
    }
    __syncthreads();
    scale = s_scale;

    // Quantize to [-7, +7] -> [0, 14]
    int q = 7;
    if (tid < head_dim) {
        float inv = (scale > 1e-30f) ? (7.0f / scale) : 0.0f;
        int val = (int)__builtin_roundf(v * inv);
        val = val < -7 ? -7 : (val > 7 ? 7 : val);
        q = val + 7;
    }

    // Pack adjacent pairs (even/odd lane) into single bytes
    // Only even lanes write to codes buffer
    int q_odd = __shfl_xor_sync(0xffffffffffffffffULL, q, 1);
    if ((tid & 1) == 0 && tid < head_dim) {
        unsigned char packed = (unsigned char)((q & 0x0f) | ((q_odd & 0x0f) << 4));
        codes[dst_base + (tid >> 1)] = packed;
    }
}

// ── WS-3A (K2V4): asymmetric 2-bit KEY / 4-bit VALUE dense arena ──────────
//
// The int4 kernels hardcode 2 codes/byte and `(code - 7) / 7`. K2V4 needs a
// different width on each side, so the width is a RUNTIME parameter here and
// the packing stride follows from it. `k_bits`/`v_bits` are uniform across the
// launch, so the branches below are not divergence.

// Dequantize one element of a packed signed code row.
//   bits == 2 -> 4 codes/byte, levels [-1, +1] stored biased +1
//   bits == 4 -> 2 codes/byte, levels [-7, +7] stored biased +7
__device__ inline float grim_dequant_iN(
    const unsigned char* __restrict__ base,
    long long row_bytes,
    int d,
    int bits,
    float sc) {
    if (bits == 2) {
        unsigned char packed = base[row_bytes + (d >> 2)];
        int code = (packed >> ((d & 3) * 2)) & 0x03;
        return ((float)(code - 1)) * sc;
    }
    unsigned char packed = base[row_bytes + (d >> 1)];
    int code = (d & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    return ((float)(code - 7) / 7.0f) * sc;
}

// Bytes per element for a packed signed code of `bits` width.
__device__ inline int grim_bytes_per_elem_iN(int bits) { return bits / 8; }

// WS-3A: 2-bit KEY quantizer. Same per-(row, kv_head) amax scheme as the i4
// quantizer, but 3 used levels (signed [-1, +1], biased +1) packed FOUR codes
// per byte, lowest 2 bits first. head_dim must be a multiple of 4.
extern "C" __global__ void grim_kv_quantize_rows_i2(
    const float* __restrict__ src,
    int src_off,
    unsigned char* __restrict__ codes,
    int codes_off,
    float* __restrict__ scales,
    int scales_off,
    int n_rows,
    int kv_stride,
    int nkv_heads,
    int head_dim) {
    const int vec = blockIdx.x;
    const int tid = threadIdx.x;
    const int vecs = n_rows * nkv_heads;
    if (vec >= vecs) return;
    const int row = vec / nkv_heads;
    const int kh = vec - row * nkv_heads;
    const long long src_base = (long long)src_off + (long long)row * kv_stride + (long long)kh * head_dim;
    // 2-bit codes: row stride is kv_stride / 4 bytes, head offset kh * head_dim / 4.
    const long long dst_base = (long long)codes_off + (long long)row * (kv_stride / 4) + (long long)kh * (head_dim / 4);
    const long long sc_idx = (long long)scales_off + (long long)row * nkv_heads + kh;
    const int wave_size = warpSize;
    const int num_waves = blockDim.x / wave_size;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;

    float v = (tid < head_dim) ? src[src_base + tid] : 0.0f;
    float amax = fabsf(v);
    #pragma unroll
    for (int off = wave_size >> 1; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffULL, amax, off));
    __shared__ float s_max2[8];
    __shared__ float s_scale2;
    if (lane_id == 0) s_max2[wave_id] = amax;
    __syncthreads();
    float mx = (tid < num_waves) ? s_max2[tid] : 0.0f;
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffffffffffULL, mx, off));
    float scale = 1.0f;
    if (tid == 0) {
        scale = (mx > 1e-30f) ? mx : 1.0f;
        scales[sc_idx] = scale;
        s_scale2 = scale;
    }
    __syncthreads();
    scale = s_scale2;

    // Quantize to [-1, +1] -> [0, 2]; stored biased +1 so the reader is a plain
    // (code - 1) * scale, the 2-bit analogue of the i4 convention.
    int q = 1;
    if (tid < head_dim) {
        float inv = (scale > 1e-30f) ? (1.0f / scale) : 0.0f;
        int val = (int)__builtin_roundf(v * inv);
        val = val < -1 ? -1 : (val > 1 ? 1 : val);
        q = val + 1;
    }

    // Pack FOUR adjacent lanes into one byte, lowest 2 bits first.
    const unsigned long long m = 0xffffffffffffffffULL;
    int q1 = __shfl_xor_sync(m, q, 1);
    int q2 = __shfl_xor_sync(m, q, 2);
    int q3 = __shfl_xor_sync(m, q1, 2);
    if ((tid & 3) == 0 && tid < head_dim) {
        unsigned char packed = (unsigned char)((q & 0x03) | ((q1 & 0x03) << 2) | ((q2 & 0x03) << 4) | ((q3 & 0x03) << 6));
        codes[dst_base + (tid >> 2)] = packed;
    }
}

// WS-3A: dense-arena attention over a MIXED-WIDTH packed KV row. Same body as
// `grim_qkv_attention_i4kv` with the K side generalized to `k_bits`; with
// k_bits == v_bits == 4 it must reproduce that kernel bit for bit, which is
// what the parity gate pins.
extern "C" __global__ void grim_qkv_attention_ixkv(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_tensor,
    const unsigned char* __restrict__ v_tensor,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ out,
    float* __restrict__ out_max,
    float* __restrict__ out_sum,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_seq_len,
    int cache_offset,
    float inv_sqrt_d,
    int window_lo,
    float softcap,
    const float* __restrict__ o_proj_w,
    int o_dim,
    int fuse_o,
    const float* __restrict__ alibi_slopes,
    int has_alibi,
    int k_bits,
    int v_bits
) {
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    if (i >= seq_len || h >= num_heads) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = (i * num_heads + h) * head_dim;
    const int abs_i = cache_offset + i;

    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    const int num_waves = blockDim.x / wave_size;

    if (head_dim > 256) {
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = nanf("");
            }
        }
        return;
    }

    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][260];

    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int lo = window_lo;
    const int range_len = hi - lo;

    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // Packed storage: the K and V sides may use DIFFERENT widths, so each side
    // carries its own row stride and head offset.
    const int k_elems_per_byte = (k_bits == 2) ? 4 : 2;
    const int v_elems_per_byte = (v_bits == 2) ? 4 : 2;
    const int k_stride_bytes = (num_kv_heads * head_dim) / k_elems_per_byte;
    const int v_stride_bytes = (num_kv_heads * head_dim) / v_elems_per_byte;
    const int k_head_bytes = (kv_head * head_dim) / k_elems_per_byte;
    const int v_head_bytes = (kv_head * head_dim) / v_elems_per_byte;
    const unsigned char* __restrict__ k_bytes = k_tensor + k_head_bytes;
    const unsigned char* __restrict__ v_bytes = v_tensor + v_head_bytes;

    float q_reg[8];
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
        int dd = lane_id + chunk * wave_size;
        q_reg[chunk] = (dd < head_dim) ? q[q_offset + dd] : 0.0f;
    }

    for (int j = lo + j_start; j < lo + j_end; ++j) {
        float partial = 0.0f;
        const long long k_row_bytes = (long long)j * k_stride_bytes;
        const float k_sc = k_scales[(long long)j * num_kv_heads + kv_head];
        #pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
            int dd = lane_id + chunk * wave_size;
            if (dd < head_dim) {
                float k_dq = grim_dequant_iN(k_bytes, k_row_bytes, dd, k_bits, k_sc);
                partial += q_reg[chunk] * k_dq;
            }
        }
        const unsigned long long shfl_mask = 0xffffffffffffffffULL;
        #pragma unroll
        for (int off = wave_size >> 1; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(shfl_mask, partial, off);
        }
        float score = partial * inv_sqrt_d;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        if (has_alibi) {
            score += alibi_slopes[h] * (float)(j - abs_i);
        }

        const float old_max = running_max;
        running_max = fmaxf(running_max, score);
        const float scale_old = expf(old_max - running_max);
        const float scale_new = expf(score - running_max);

        running_sum = running_sum * scale_old + scale_new;
        const float v_sc = v_scales[(long long)j * num_kv_heads + kv_head];
        const long long v_row_bytes = (long long)j * v_stride_bytes;
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                float v_dq = grim_dequant_iN(v_bytes, v_row_bytes, d, v_bits, v_sc);
                out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_dq;
            }
        }
    }

    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 8; ++chunk) {
        int d = lane_id + chunk * wave_size;
        if (d < head_dim) {
            s_acc[wave_id][d] = out_acc[chunk];
        } else if (d < 256) {
            s_acc[wave_id][d] = 0.0f;
        }
    }
    __syncthreads();

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

    float attn_reg[8];
    #pragma unroll
    for (int chunk = 0; chunk < 8; ++chunk) {
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
        for (int chunk = 0; chunk < 8; ++chunk) {
            int d = lane_id + chunk * wave_size;
            if (d < head_dim) {
                out[q_offset + d] = attn_reg[chunk];
            }
        }
    } else {
        for (int oc = 0; oc < o_dim; ++oc) {
            float partial = 0.0f;
            #pragma unroll
            for (int chunk = 0; chunk < 8; ++chunk) {
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
    // WI-X5: autotuner block-dim override. `None` keeps the wavefront-aware default of 4 waves per block.
    block_dim_override: Option<u32>,
) -> Result<(), crate::Error> {
    // The kernel bakes in a hard cap at head_dim > 256 (writes NaN + returns).
    // Reject unsupported head_dim at the wrapper so callers get a clear error rather than silent.
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
    // WI-X5: honor the autotuner override when it respects the kernel's wavefront-multiple /
    // 8-wave LDS invariants; otherwise keep the default 4-waves-per-block launch (W32→128 threads, W64→256 threads).
    let block_x = match block_dim_override {
        Some(d) if d % wf == 0 && d / wf <= 8 => d,
        _ => wf * 4,
    };
    let block_dim = crate::HipDim3::new(block_x, 1, 1);
    // The kernel derives num_waves = blockDim.x / wavefront at runtime and its
    // LDS accumulators are sized for up to 8 waves; assert the invariant.
    debug_assert!(
        block_dim.x % wf == 0 && block_dim.x / wf <= 8,
        "qkv_attention_paged: block_dim.x ({}) violates wavefront/LDS invariant (wf={}, max 8 waves)",
        block_dim.x,
        wf
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
    // Sliding-window lower bound (0 for full causal; >=0 for SWA).
    // Mirrors the non-paged wrapper's host-side `window_lo_i` computation.
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

/// Host launcher for the WMMA tensor-core fused QKV attention kernel.
pub fn launch_qkv_attention_wmma(
    dev: &crate::RocmDevice,
    q: &dyn BackendStorage,
    k: &dyn BackendStorage,
    v: &dyn BackendStorage,
    out: &mut dyn BackendStorage,
    out_max: Option<&mut dyn BackendStorage>,
    out_sum: Option<&mut dyn BackendStorage>,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_seq_len: u32,
    cache_offset: u32,
    window_lo: i32,
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
    let out_ptr = out_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("out has no device ptr".into()))?;

    let max_ptr = match out_max {
        Some(m) => m
            .as_any()
            .downcast_ref::<crate::memory::storage::RocmStorage>()
            .and_then(|s| s.device_ptr)
            .unwrap_or(0),
        None => 0,
    };
    let sum_ptr = match out_sum {
        Some(s) => s
            .as_any()
            .downcast_ref::<crate::memory::storage::RocmStorage>()
            .and_then(|st| st.device_ptr)
            .unwrap_or(0),
        None => 0,
    };

    let wf = dev.wavefront_size() as u32;
    // One block per 16-query tile of one head; a single wave owns the tile.
    let q_tiles = seq_len.div_ceil(16);
    let grid_dim = crate::HipDim3::new(q_tiles, num_heads, 1);
    let block_dim = crate::HipDim3::new(wf, 1, 1);
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

    let mut qptr = q_ptr;
    let mut kptr = k_ptr;
    let mut vptr = v_ptr;
    let mut optr = out_ptr;
    let mut mptr = max_ptr;
    let mut sptr = sum_ptr;
    let mut nh = num_heads as i32;
    let mut nkv = num_kv_heads as i32;
    let mut hd = head_dim as i32;
    let mut sl = seq_len as i32;
    let mut ksl = kv_seq_len as i32;
    let mut co = cache_offset as i32;
    let mut isd = inv_sqrt_d;
    let mut wl = window_lo;
    let mut o_proj: u64 = 0;
    let mut o_dim: i32 = 0;
    let mut fuse_o: i32 = 0;

    dev.launch_compute_kernel(
        "grim_qkv_attention_wmma",
        grid_dim,
        block_dim,
        &mut [
            arg(&mut qptr),
            arg(&mut kptr),
            arg(&mut vptr),
            arg(&mut optr),
            arg(&mut mptr),
            arg(&mut sptr),
            arg(&mut nh),
            arg(&mut nkv),
            arg(&mut hd),
            arg(&mut sl),
            arg(&mut ksl),
            arg(&mut co),
            arg(&mut isd),
            arg(&mut wl),
            arg(&mut o_proj),
            arg(&mut o_dim),
            arg(&mut fuse_o),
        ],
    )?;

    Ok(())
}

/// Quantization formats for KV-cache attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum KvCacheQuantFormat {
    Int8 = 0,
    Int4 = 1, // W4A16 packed 4-bit nibbles
    Fp8E4M3 = 2,
    Fp8E5M2 = 3,
    Fp4E2M1 = 4,
    MxFp4 = 5,
    MxFp8 = 6,
    /// NVFP4: E2M1 codes with an **E4M3** block scale stored inline per 16
    /// values (9 bytes per 16). Same layout as [`Self::NutFp4`], different
    /// scale decode — never share a page buffer between the two.
    NvFp4 = 7,
    /// Nutcracker: E2M1 codes with a per-16 block scale stored inline in the
    /// packed buffer (9 bytes per 16 values), whose low 2 bits are stolen as a
    /// special-value selector. 4-bit element, so this tags as FP4.
    ///
    /// Unlike every other variant here, this one reads its scale out of the
    /// buffer and ignores the caller's per-tensor `k_scale`/`v_scale`.
    NutFp4 = 8,
    /// WhiteCrow: RDNA 3/4 native unsigned 4-bit group-128 OSTQuant layout
    /// (discriminant matches `grim_tensor::PagedKvQuantFormat::WhiteCrow`).
    /// The paged-attention kernel has no WhiteCrow decode leg yet, so the
    /// launcher refuses it loudly rather than misreading packed pages.
    WhiteCrow = 9,
}

impl KvCacheQuantFormat {
    /// Packed bytes needed to store `num_values` elements in this format.
    ///
    /// The per-tensor `k_scale`/`v_scale` is supplied separately by the caller
    /// and is not counted here.
    ///
    /// Use this for page allocation; the decode arms in
    /// `dequant_kv_element` index with exactly these strides.
    pub fn packed_bytes(self, num_values: usize) -> usize {
        let bits = self.element_bits() as usize;
        // Number of inline scale bytes. `Fp4E2M1` and the unpacked formats
        // carry no inline scale — the caller's per-tensor `k_scale`/`v_scale`
        // covers them.
        let groups = match self {
            // Per-16 sub-blocks with one inline scale byte.
            Self::NvFp4 | Self::NutFp4 => (num_values + 15) / 16,
            // Per-32 microscale groups with one inline E8M0 scale byte.
            Self::MxFp4 | Self::MxFp8 => (num_values + 31) / 32,
            Self::Int8 | Self::Int4 | Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Fp4E2M1 => 0,
            // WhiteCrow: per 128 values - 64 B codes + 2 B scale + 1 B zero
            // (the codes ride u32 words, so the arithmetic below collapses to
            // num_values/2 + num_values/128*2 + num_values/128).
            Self::WhiteCrow => (num_values + 127) / 128,
        };
        let payload = (num_values * bits).div_ceil(8);
        payload + groups
    }

    /// Bits per stored element (before any per-tensor scaling).
    pub fn element_bits(self) -> u32 {
        match self {
            Self::Int8 | Self::Fp8E4M3 | Self::Fp8E5M2 | Self::MxFp8 => 8,
            Self::Int4 | Self::Fp4E2M1 | Self::MxFp4 | Self::NvFp4 | Self::NutFp4 => 4,
            // WhiteCrow codes are unsigned 4-bit group-128.
            Self::WhiteCrow => 4,
        }
    }
}

impl From<grim_tensor::PagedKvQuantFormat> for KvCacheQuantFormat {
    /// The neutral `grim_tensor` enum and this backend's enum share the same
    /// discriminants (they are the values the kernel switches on), so the
    /// conversion is a discriminant cast rather than a translation table.
    fn from(f: grim_tensor::PagedKvQuantFormat) -> Self {
        match f {
            grim_tensor::PagedKvQuantFormat::Int8 => KvCacheQuantFormat::Int8,
            grim_tensor::PagedKvQuantFormat::Int4 => KvCacheQuantFormat::Int4,
            grim_tensor::PagedKvQuantFormat::Fp8E4M3 => KvCacheQuantFormat::Fp8E4M3,
            grim_tensor::PagedKvQuantFormat::Fp8E5M2 => KvCacheQuantFormat::Fp8E5M2,
            grim_tensor::PagedKvQuantFormat::Fp4E2M1 => KvCacheQuantFormat::Fp4E2M1,
            grim_tensor::PagedKvQuantFormat::MxFp4 => KvCacheQuantFormat::MxFp4,
            grim_tensor::PagedKvQuantFormat::MxFp8 => KvCacheQuantFormat::MxFp8,
            grim_tensor::PagedKvQuantFormat::NvFp4 => KvCacheQuantFormat::NvFp4,
            grim_tensor::PagedKvQuantFormat::NutFp4 => KvCacheQuantFormat::NutFp4,
            grim_tensor::PagedKvQuantFormat::WhiteCrow => KvCacheQuantFormat::WhiteCrow,
        }
    }
}

/// Host launcher for quantized and microscaled paged KV-cache attention.
pub fn launch_paged_attention_quant(
    dev: &crate::RocmDevice,
    q: &dyn BackendStorage,
    block_tables: &dyn BackendStorage,
    k_pages: &dyn BackendStorage,
    v_pages: &dyn BackendStorage,
    out: &mut dyn BackendStorage,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_blocks: u32,
    page_size: u32,
    kv_seq_len: u32,
    cache_offset: u32,
    window_lo: i32,
    quant_format: KvCacheQuantFormat,
    k_scale: f32,
    k_bias: f32,
    v_scale: f32,
    v_bias: f32,
) -> Result<(), crate::Error> {
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
    let mut wl = window_lo;
    // WhiteCrow pages have no kernel decode leg yet (the format exists so
    // grim-memory's packed-pool producer can round-trip through the neutral
    // enum). Refuse loudly: a silent fallthrough would make the kernel read
    // OSTQuant u4 group-128 pages as whichever dense layout shares the
    // discriminant - finite, plausible, wrong attention.
    if quant_format == KvCacheQuantFormat::WhiteCrow {
        return Err(crate::Error::Backend(
            "launch_paged_attention_quant: WhiteCrow KV pages have no kernel decode leg yet; \
             run with GRIM_PACKED_POOL=0 (dense KV) until the WhiteCrow leg lands"
                .into(),
        ));
    }
    let mut qf = quant_format as i32;
    let mut ks = k_scale;
    let mut kb = k_bias;
    let mut vs = v_scale;
    let mut vb = v_bias;

    dev.launch_compute_kernel(
        "grim_qkv_attention_paged_quant",
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
            arg(&mut wl),
            arg(&mut qf),
            arg(&mut ks),
            arg(&mut kb),
            arg(&mut vs),
            arg(&mut vb),
        ],
    )?;

    Ok(())
}

/// Item 3 launcher: `grim_kv_append`. Copies `steps*kv_stride` elements from
/// `k_rot` into `k_arena` at the device-computed offset `*past_dev * kv_stride`.
/// P0-2 (PLAN-improve-grim-perf): accept both owning `RocmStorage` and
/// non-owning `RocmStorageView` for the rotated-K/V operands — the fused
/// Q8_0 QKV decode path produces `RocmStorageView` slices of one fused
/// output buffer, and a hard downcast to `RocmStorage` turned that into a
/// fatal error instead of a working append. Views never free, so borrowing
/// their parent-backed device pointer here is sound.
fn kv_rot_device_ptr(
    s: &dyn BackendStorage,
    label: &str,
) -> std::result::Result<*mut std::ffi::c_void, crate::Error> {
    if let Some(r) = s
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
    {
        return r
            .device_ptr
            .map(|p| p as *mut std::ffi::c_void)
            .ok_or_else(|| crate::Error::Backend(format!("kv_append: {label} has no device ptr")));
    }
    if let Some(v) = s
        .as_any()
        .downcast_ref::<crate::memory::view::RocmStorageView>()
    {
        let p = v.device_ptr_u64();
        if p != 0 {
            return Ok(p as *mut std::ffi::c_void);
        }
    }
    Err(crate::Error::Backend(format!(
        "kv_append: {label} must be RocmStorage or RocmStorageView"
    )))
}

/// M3: KV-arena representation from STORAGE, not env. Uniform F32 or
/// uniform F16 arenas are accepted regardless of `GRIM_F16_KV` (the flag is
/// only an allocation hint); anything else — mixed dtypes, packed/quant
/// storages that the dense kernels would reinterpret — fails loud here and
/// routes callers to the host fallback instead of garbage. Returns true for
/// f16 arenas (kernels take the f16 leg) and false for f32.
fn kv_arena_f16(arena: &dyn BackendStorage, what: &str) -> Result<bool, crate::Error> {
    use crate::memory::storage::RocmStorage;
    let s = arena
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| crate::Error::Backend(format!("{what}: K/V arena must be RocmStorage")))?;
    match s.dtype.arith {
        grim_tensor::dtype::ArithType::F16 => Ok(true),
        grim_tensor::dtype::ArithType::F32 => Ok(false),
        other => Err(crate::Error::Backend(format!(
            "{what}: K/V arena dtype {other:?} needs the paged quant path, not dense kernels"
        ))),
    }
}

pub fn launch_kv_append(
    dev: &crate::RocmDevice,
    k_arena: &dyn BackendStorage,
    k_rot: &dyn BackendStorage,
    past_dev: &dyn BackendStorage,
    kv_stride: usize,
    steps: usize,
    ring: KvRingPolicy,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let mut f16_flag = kv_arena_f16(k_arena, "kv_append")? as i32;
    let total = steps * kv_stride;
    let arena_s = k_arena
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("kv_append: k_arena must be RocmStorage".into()))?;
    let mut rot_ptr = kv_rot_device_ptr(k_rot, "k_rot")?;
    let past_s = past_dev
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("kv_append: past_dev must be RocmStorage".into()))?;
    let mut arena_ptr = arena_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("kv_append: k_arena has no device ptr".into()))?;
    let mut past_ptr = past_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("kv_append: past_dev has no device ptr".into()))?;
    let mut stride_i = kv_stride as i32;
    let mut steps_i = steps as i32;
    // Phase 3 ring (cap_rows <= 0 disables eviction).
    let mut cap_rows_i = ring.cap_rows;
    let mut sink_rows_i = ring.sink_rows;
    let (grid, block) = crate::device::util::linear_launch(total);
    // Legacy single-sequence path: batch=1 (slot always 0 in the kernel).
    dev.launch_compute_kernel(
        "grim_kv_append",
        grid,
        block,
        &mut [
            arg(&mut arena_ptr),
            arg(&mut rot_ptr),
            arg(&mut past_ptr),
            arg(&mut stride_i),
            arg(&mut steps_i),
            arg(&mut 1i32),
            arg(&mut 0i32),
            arg(&mut f16_flag),
            arg(&mut cap_rows_i),
            arg(&mut sink_rows_i),
        ],
    )
}

/// P3 batched KV append: `k_rot` is `[batch, steps*kv_stride]` flat, arena is
/// per-slot regions of `arena_slot_stride` elements each, `past_dev` has one
/// i32 per slot. batch=1 with stride 0 matches the legacy kernel exactly.
#[allow(clippy::too_many_arguments)]
pub fn launch_kv_append_batch(
    dev: &crate::RocmDevice,
    k_arena: &dyn BackendStorage,
    k_rot: &dyn BackendStorage,
    past_dev: &dyn BackendStorage,
    kv_stride: usize,
    steps: usize,
    batch: usize,
    arena_slot_stride: usize,
    ring: KvRingPolicy,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    // Phase 3 ring (cap_rows <= 0 disables eviction).
    let mut cap_rows_i = ring.cap_rows;
    let mut sink_rows_i = ring.sink_rows;
    let mut f16_flag = kv_arena_f16(k_arena, "kv_append_batch")? as i32;
    let arena_s = k_arena
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("kv_append_batch: k_arena must be RocmStorage".into())
        })?;
    let mut rot_ptr = kv_rot_device_ptr(k_rot, "k_rot_batch")?;
    let past_s = past_dev
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("kv_append_batch: past_dev must be RocmStorage".into())
        })?;
    let mut arena_ptr = arena_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_append_batch: k_arena has no device ptr".into())
    })?;
    let mut past_ptr = past_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_append_batch: past_dev has no device ptr".into())
    })?;
    let mut stride_i = kv_stride as i32;
    let mut steps_i = steps as i32;
    let mut batch_i = batch.max(1) as i32;
    let mut slot_stride_i = arena_slot_stride as i32;
    let total = batch.max(1) * steps * kv_stride;
    let (grid, block) = crate::device::util::linear_launch(total);
    dev.launch_compute_kernel(
        "grim_kv_append",
        grid,
        block,
        &mut [
            arg(&mut arena_ptr),
            arg(&mut rot_ptr),
            arg(&mut past_ptr),
            arg(&mut stride_i),
            arg(&mut steps_i),
            arg(&mut batch_i),
            arg(&mut slot_stride_i),
            arg(&mut f16_flag),
            arg(&mut cap_rows_i),
            arg(&mut sink_rows_i),
        ],
    )
}

// Phase 2 launcher: `grim_kv_append_i8`. Quantizes `steps` f32 rows per slot
/// (each (step, kv_head) head-vector independently, symmetric int8 with an
/// f32 per-vector scale) into the u8 codes arena + f32 scales sidecar.
/// Block 256 required (one thread per head_dim element, `head_dim` <= 256).
#[allow(clippy::too_many_arguments)]
/// Phase 2 launcher: `grim_kv_quantize_rows_i8`. Host-offset sibling of
/// `launch_kv_append_i8` for the eager path (offsets computed on host like
/// `copy_slice_range`; no past_dev). Quantizes `n_rows` f32 head-vectors.
#[allow(clippy::too_many_arguments)]
pub fn launch_kv_quantize_rows_i8(
    dev: &crate::RocmDevice,
    src: &dyn BackendStorage,
    src_off: usize,
    codes: &dyn BackendStorage,
    codes_off: usize,
    scales: &dyn BackendStorage,
    scales_off: usize,
    n_rows: usize,
    kv_stride: usize,
    nkv_heads: usize,
    head_dim: usize,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    use crate::memory::storage::RocmStorage;
    use grim_tensor::dtype::ArithType;
    let src_s = src.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i8: src must be RocmStorage".into())
    })?;
    if src_s.dtype.arith != ArithType::F32 {
        return Err(crate::Error::Backend(
            "kv_quantize_rows_i8: src must be F32".into(),
        ));
    }
    let codes_s = codes
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("kv_quantize_rows_i8: codes must be RocmStorage".into())
        })?;
    if codes_s.dtype.arith != ArithType::U8 {
        return Err(crate::Error::Backend(
            "kv_quantize_rows_i8: codes must be U8".into(),
        ));
    }
    let scales_s = scales
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("kv_quantize_rows_i8: scales must be RocmStorage".into())
        })?;
    if scales_s.dtype.arith != ArithType::F32 {
        return Err(crate::Error::Backend(
            "kv_quantize_rows_i8: scales must be F32".into(),
        ));
    }
    if head_dim == 0 || head_dim > 256 {
        return Err(crate::Error::Backend(format!(
            "kv_quantize_rows_i8: head_dim {head_dim} not in 1..=256"
        )));
    }
    let mut sptr = src_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i8: src has no device ptr".into())
    })?;
    let mut cptr = codes_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i8: codes have no device ptr".into())
    })?;
    let mut scptr = scales_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i8: scales have no device ptr".into())
    })?;
    let mut src_off_i = src_off as i32;
    let mut codes_off_i = codes_off as i32;
    let mut scales_off_i = scales_off as i32;
    let mut n_rows_i = n_rows as i32;
    let mut stride_i = kv_stride as i32;
    let mut nkv_i = nkv_heads as i32;
    let mut hd_i = head_dim as i32;
    let grid = crate::HipDim3::new((n_rows * nkv_heads) as u32, 1, 1);
    let block = crate::HipDim3::new(256, 1, 1);
    dev.launch_compute_kernel(
        "grim_kv_quantize_rows_i8",
        grid,
        block,
        &mut [
            arg(&mut sptr),
            arg(&mut src_off_i),
            arg(&mut cptr),
            arg(&mut codes_off_i),
            arg(&mut scptr),
            arg(&mut scales_off_i),
            arg(&mut n_rows_i),
            arg(&mut stride_i),
            arg(&mut nkv_i),
            arg(&mut hd_i),
        ],
    )
}

/// Phase 3 launcher: `grim_kv_quantize_rows_i4`.
/// Quantizes `n_rows` f32 head-vectors into packed 4-bit nibbles + f32 per-vector scales.
#[allow(clippy::too_many_arguments)]
pub fn launch_kv_quantize_rows_i4(
    dev: &crate::RocmDevice,
    src: &dyn BackendStorage,
    src_off: usize,
    codes: &dyn BackendStorage,
    codes_off: usize,
    scales: &dyn BackendStorage,
    scales_off: usize,
    n_rows: usize,
    kv_stride: usize,
    nkv_heads: usize,
    head_dim: usize,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    use crate::memory::storage::RocmStorage;
    use grim_tensor::dtype::ArithType;
    let src_s = src.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i4: src must be RocmStorage".into())
    })?;
    if src_s.dtype.arith != ArithType::F32 {
        return Err(crate::Error::Backend(
            "kv_quantize_rows_i4: src must be F32".into(),
        ));
    }
    let codes_s = codes
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("kv_quantize_rows_i4: codes must be RocmStorage".into())
        })?;
    if codes_s.dtype.arith != ArithType::U8 {
        return Err(crate::Error::Backend(
            "kv_quantize_rows_i4: codes must be U8 (byte storage for packed u4)".into(),
        ));
    }
    let scales_s = scales
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("kv_quantize_rows_i4: scales must be RocmStorage".into())
        })?;
    if scales_s.dtype.arith != ArithType::F32 {
        return Err(crate::Error::Backend(
            "kv_quantize_rows_i4: scales must be F32".into(),
        ));
    }
    if head_dim == 0 || head_dim > 256 || head_dim % 2 != 0 {
        return Err(crate::Error::Backend(format!(
            "kv_quantize_rows_i4: head_dim {head_dim} must be even and in 1..=256"
        )));
    }
    let mut sptr = src_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i4: src has no device ptr".into())
    })?;
    let mut cptr = codes_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i4: codes have no device ptr".into())
    })?;
    let mut scptr = scales_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i4: scales have no device ptr".into())
    })?;
    let mut src_off_i = src_off as i32;
    let mut codes_off_i = codes_off as i32;
    let mut scales_off_i = scales_off as i32;
    let mut n_rows_i = n_rows as i32;
    let mut stride_i = kv_stride as i32;
    let mut nkv_i = nkv_heads as i32;
    let mut hd_i = head_dim as i32;
    let grid = crate::HipDim3::new((n_rows * nkv_heads) as u32, 1, 1);
    let block = crate::HipDim3::new(256, 1, 1);
    dev.launch_compute_kernel(
        "grim_kv_quantize_rows_i4",
        grid,
        block,
        &mut [
            arg(&mut sptr),
            arg(&mut src_off_i),
            arg(&mut cptr),
            arg(&mut codes_off_i),
            arg(&mut scptr),
            arg(&mut scales_off_i),
            arg(&mut n_rows_i),
            arg(&mut stride_i),
            arg(&mut nkv_i),
            arg(&mut hd_i),
        ],
    )
}

/// WS-3A: 2-bit KEY quantizer launcher. Same per-(row, kv_head) amax scheme
/// as the i4 launcher, packing four codes per byte (`head_dim % 4 == 0`).
pub fn launch_kv_quantize_rows_i2(
    dev: &crate::RocmDevice,
    src: &dyn BackendStorage,
    src_off: usize,
    codes: &dyn BackendStorage,
    codes_off: usize,
    scales: &dyn BackendStorage,
    scales_off: usize,
    n_rows: usize,
    kv_stride: usize,
    nkv_heads: usize,
    head_dim: usize,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    use crate::memory::storage::RocmStorage;
    use grim_tensor::dtype::ArithType;
    let src_s = src.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i2: src must be RocmStorage".into())
    })?;
    if src_s.dtype.arith != ArithType::F32 {
        return Err(crate::Error::Backend(
            "kv_quantize_rows_i2: src must be F32".into(),
        ));
    }
    let codes_s = codes
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("kv_quantize_rows_i2: codes must be RocmStorage".into())
        })?;
    if codes_s.dtype.arith != ArithType::U8 {
        return Err(crate::Error::Backend(
            "kv_quantize_rows_i2: codes must be U8 (byte storage for packed u2)".into(),
        ));
    }
    let scales_s = scales
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("kv_quantize_rows_i2: scales must be RocmStorage".into())
        })?;
    if scales_s.dtype.arith != ArithType::F32 {
        return Err(crate::Error::Backend(
            "kv_quantize_rows_i2: scales must be F32".into(),
        ));
    }
    // 2-bit packing puts FOUR codes per byte, so head_dim must divide by 4.
    if head_dim == 0 || head_dim > 256 || head_dim % 4 != 0 {
        return Err(crate::Error::Backend(format!(
            "kv_quantize_rows_i2: head_dim {head_dim} must be a multiple of 4 and in 1..=256"
        )));
    }
    let mut sptr = src_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i2: src has no device ptr".into())
    })?;
    let mut cptr = codes_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i2: codes have no device ptr".into())
    })?;
    let mut scptr = scales_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("kv_quantize_rows_i2: scales have no device ptr".into())
    })?;
    let mut src_off_i = src_off as i32;
    let mut codes_off_i = codes_off as i32;
    let mut scales_off_i = scales_off as i32;
    let mut n_rows_i = n_rows as i32;
    let mut stride_i = kv_stride as i32;
    let mut nkv_i = nkv_heads as i32;
    let mut hd_i = head_dim as i32;
    let grid = crate::HipDim3::new((n_rows * nkv_heads) as u32, 1, 1);
    let block = crate::HipDim3::new(256, 1, 1);
    dev.launch_compute_kernel(
        "grim_kv_quantize_rows_i2",
        grid,
        block,
        &mut [
            arg(&mut sptr),
            arg(&mut src_off_i),
            arg(&mut cptr),
            arg(&mut codes_off_i),
            arg(&mut scptr),
            arg(&mut scales_off_i),
            arg(&mut n_rows_i),
            arg(&mut stride_i),
            arg(&mut nkv_i),
            arg(&mut hd_i),
        ],
    )
}

pub fn launch_kv_append_i8(
    dev: &crate::RocmDevice,
    k_rot: &dyn BackendStorage,
    past_dev: &dyn BackendStorage,
    codes: &dyn BackendStorage,
    scales: &dyn BackendStorage,
    kv_stride: usize,
    nkv_heads: usize,
    head_dim: usize,
    steps: usize,
    batch: usize,
    arena_slot_stride: usize,
    scale_slot_stride: usize,
    ring: KvRingPolicy,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    // Phase 3 ring (cap_rows <= 0 disables eviction).
    let mut cap_rows_i = ring.cap_rows;
    let mut sink_rows_i = ring.sink_rows;
    use crate::memory::storage::RocmStorage;
    use grim_tensor::dtype::ArithType;
    let codes_s = codes
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("kv_append_i8: codes must be RocmStorage".into()))?;
    if codes_s.dtype.arith != ArithType::U8 {
        return Err(crate::Error::Backend(
            "kv_append_i8: codes arena must be U8 (int8 format)".into(),
        ));
    }
    let scales_s = scales
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("kv_append_i8: scales must be RocmStorage".into()))?;
    if scales_s.dtype.arith != ArithType::F32 {
        return Err(crate::Error::Backend(
            "kv_append_i8: scales sidecar must be F32".into(),
        ));
    }
    if head_dim == 0 || head_dim > 256 {
        return Err(crate::Error::Backend(format!(
            "kv_append_i8: head_dim {head_dim} not in 1..=256"
        )));
    }
    let mut rot_ptr = kv_rot_device_ptr(k_rot, "k_rot_i8")?;
    let past_s = past_dev
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("kv_append_i8: past_dev must be RocmStorage".into())
        })?;
    let mut codes_ptr = codes_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("kv_append_i8: codes have no device ptr".into()))?;
    let mut scales_ptr = scales_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("kv_append_i8: scales have no device ptr".into()))?;
    let mut past_ptr = past_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("kv_append_i8: past_dev has no device ptr".into()))?;
    let mut stride_i = kv_stride as i32;
    let mut nkv_i = nkv_heads as i32;
    let mut hd_i = head_dim as i32;
    let mut steps_i = steps as i32;
    let mut batch_i = batch.max(1) as i32;
    let mut slot_stride_i = arena_slot_stride as i32;
    let mut sc_slot_stride_i = scale_slot_stride as i32;
    let grid = crate::HipDim3::new((batch.max(1) * steps * nkv_heads) as u32, 1, 1);
    let block = crate::HipDim3::new(256, 1, 1);
    dev.launch_compute_kernel(
        "grim_kv_append_i8",
        grid,
        block,
        &mut [
            arg(&mut rot_ptr),
            arg(&mut past_ptr),
            arg(&mut codes_ptr),
            arg(&mut scales_ptr),
            arg(&mut stride_i),
            arg(&mut nkv_i),
            arg(&mut hd_i),
            arg(&mut steps_i),
            arg(&mut batch_i),
            arg(&mut slot_stride_i),
            arg(&mut sc_slot_stride_i),
            arg(&mut cap_rows_i),
            arg(&mut sink_rows_i),
        ],
    )
}

/// Item 3 launcher: `grim_qkv_attention_dev`. Same math as `launch_qkv_attention`
/// but reads the KV length from a device scalar `total_dev` instead of a host
/// `kv_seq_len` parameter, making the node graph-capturable.
pub fn launch_qkv_attention_dev(
    dev: &crate::RocmDevice,
    q: &dyn BackendStorage,
    k_tensor: &dyn BackendStorage,
    v_tensor: &dyn BackendStorage,
    out: &dyn BackendStorage,
    out_max: &dyn BackendStorage,
    out_sum: &dyn BackendStorage,
    total_dev: &dyn BackendStorage,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    steps: u32,
    inv_sqrt_d: f32,
    window_lo: i32,
    softcap: f32,
    o_proj_w: &dyn BackendStorage,
    o_dim: u32,
    fuse_o: u32,
    alibi_slopes: &dyn BackendStorage,
    has_alibi: u32,
    ring: KvRingPolicy,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    launch_qkv_attention_dev_batch(
        dev,
        q,
        k_tensor,
        v_tensor,
        out,
        out_max,
        out_sum,
        total_dev,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        steps,
        inv_sqrt_d,
        window_lo,
        softcap,
        o_proj_w,
        o_dim,
        fuse_o,
        alibi_slopes,
        has_alibi,
        1,
        0,
        ring,
    )
}

/// P3 batched device-total attention: grid (seq_len, num_heads, batch).
/// Per-slot: q/out stride `seq_len*num_heads*head_dim`, arena regions of
/// `arena_slot_stride` elements, `total_dev[slot]` per-slot KV length.
/// batch=1 with stride 0 matches the legacy kernel exactly.
#[allow(clippy::too_many_arguments)]
pub fn launch_qkv_attention_dev_batch(
    dev: &crate::RocmDevice,
    q: &dyn BackendStorage,
    k_tensor: &dyn BackendStorage,
    v_tensor: &dyn BackendStorage,
    out: &dyn BackendStorage,
    out_max: &dyn BackendStorage,
    out_sum: &dyn BackendStorage,
    total_dev: &dyn BackendStorage,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    steps: u32,
    inv_sqrt_d: f32,
    window_lo: i32,
    softcap: f32,
    o_proj_w: &dyn BackendStorage,
    o_dim: u32,
    fuse_o: u32,
    alibi_slopes: &dyn BackendStorage,
    has_alibi: u32,
    batch: usize,
    arena_slot_stride: usize,
    ring: KvRingPolicy,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    const KERNEL: &str = "grim_qkv_attention_dev";
    if head_dim > 256 {
        return Err(crate::Error::Backend(format!(
            "{}: head_dim {} exceeds kernel cap of 256",
            KERNEL, head_dim
        )));
    }
    let q_s = q
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend(format!("{}: q must be RocmStorage", KERNEL)))?;
    let k_f16 = kv_arena_f16(k_tensor, "grim_qkv_attention_dev k")?;
    let v_f16 = kv_arena_f16(v_tensor, "grim_qkv_attention_dev v")?;
    if k_f16 != v_f16 {
        return Err(crate::Error::Backend(
            "grim_qkv_attention_dev: K/V arenas must share one dtype (storage truth)".into(),
        ));
    }
    let mut f16_flag = k_f16 as i32;
    let k_s = k_tensor
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend(format!("{}: k must be RocmStorage", KERNEL)))?;
    let v_s = v_tensor
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend(format!("{}: v must be RocmStorage", KERNEL)))?;
    let out_s = out
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend(format!("{}: out must be RocmStorage", KERNEL)))?;
    let om_s = out_max
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend(format!("{}: out_max must be RocmStorage", KERNEL)))?;
    let os_s = out_sum
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend(format!("{}: out_sum must be RocmStorage", KERNEL)))?;
    let total_s = total_dev
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend(format!("{}: total_dev must be RocmStorage", KERNEL))
        })?;
    let oproj_s = o_proj_w
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend(format!("{}: o_proj_w must be RocmStorage", KERNEL))
        })?;
    let alibi_s = alibi_slopes
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend(format!("{}: alibi_slopes must be RocmStorage", KERNEL))
        })?;
    let mut q_ptr = q_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("q has no ptr".into()))?;
    let mut k_ptr = k_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("k has no ptr".into()))?;
    let mut v_ptr = v_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("v has no ptr".into()))?;
    let mut o_ptr = out_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("out has no ptr".into()))?;
    let mut om_ptr = om_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("out_max has no ptr".into()))?;
    let mut os_ptr = os_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("out_sum has no ptr".into()))?;
    let mut tot_ptr = total_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("total_dev has no ptr".into()))?;
    let mut nh = num_heads as i32;
    let mut nkv = num_kv_heads as i32;
    let mut hd = head_dim as i32;
    let mut sl = seq_len as i32;
    let mut steps_i = steps as i32;
    let mut isd = inv_sqrt_d;
    let mut wl = window_lo;
    let mut sc = softcap;
    let mut od = o_dim as i32;
    let mut fo = fuse_o as i32;
    let mut asl = alibi_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("alibi has no ptr".into()))?;
    let mut ha = has_alibi as i32;
    let mut opptr = oproj_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("o_proj_w has no ptr".into()))?;
    let mut batch_i = batch.max(1) as i32;
    let mut slot_stride_i = arena_slot_stride as i32;
    // f16_flag derived from the arenas above (storage truth).
    let grid_dim = crate::HipDim3::new(sl as u32, nh as u32, batch.max(1) as u32);
    let block_dim = crate::HipDim3::new(128, 1, 1);
    // Phase 3 ring (cap_rows <= 0 disables eviction).
    let mut cap_rows_i = ring.cap_rows;
    dev.launch_compute_kernel(
        KERNEL,
        grid_dim,
        block_dim,
        &mut [
            arg(&mut q_ptr),
            arg(&mut k_ptr),
            arg(&mut v_ptr),
            arg(&mut o_ptr),
            arg(&mut om_ptr),
            arg(&mut os_ptr),
            arg(&mut nh),
            arg(&mut nkv),
            arg(&mut hd),
            arg(&mut sl),
            arg(&mut tot_ptr),
            arg(&mut steps_i),
            arg(&mut isd),
            arg(&mut wl),
            arg(&mut sc),
            arg(&mut opptr),
            arg(&mut od),
            arg(&mut fo),
            arg(&mut asl),
            arg(&mut ha),
            arg(&mut batch_i),
            arg(&mut slot_stride_i),
            arg(&mut f16_flag),
            arg(&mut cap_rows_i),
        ],
    )
}

/// Phase-2 launcher: `grim_qkv_attention_dev_i8`. Device-total twin of the
/// scalar int8 attend: K/V are biased-u8 codes with f32 per-(slot,row,kv_head)
/// scales. Same capture semantics as `launch_qkv_attention_dev_batch`.
#[allow(clippy::too_many_arguments)]
pub fn launch_qkv_attention_dev_batch_i8(
    dev: &crate::RocmDevice,
    q: &dyn BackendStorage,
    k_codes: &dyn BackendStorage,
    k_scales: &dyn BackendStorage,
    v_codes: &dyn BackendStorage,
    v_scales: &dyn BackendStorage,
    out: &dyn BackendStorage,
    out_max: &dyn BackendStorage,
    out_sum: &dyn BackendStorage,
    total_dev: &dyn BackendStorage,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    steps: u32,
    inv_sqrt_d: f32,
    window_lo: i32,
    softcap: f32,
    o_proj_w: &dyn BackendStorage,
    o_dim: u32,
    fuse_o: u32,
    alibi_slopes: &dyn BackendStorage,
    has_alibi: u32,
    batch: usize,
    arena_slot_stride: usize,
    scale_slot_stride: usize,
    ring: KvRingPolicy,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    // Phase 3 ring (cap_rows <= 0 disables eviction).
    let mut cap_rows_i = ring.cap_rows;
    const KERNEL: &str = "grim_qkv_attention_dev_i8";
    use crate::memory::storage::RocmStorage;
    use grim_tensor::dtype::ArithType;
    if head_dim > 256 {
        return Err(crate::Error::Backend(format!(
            "{}: head_dim {} exceeds kernel cap of 256",
            KERNEL, head_dim
        )));
    }
    let q_s = q
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| crate::Error::Backend(format!("{}: q must be RocmStorage", KERNEL)))?;
    for (buf, what, want) in [
        (k_codes, "k_codes", ArithType::U8),
        (v_codes, "v_codes", ArithType::U8),
        (k_scales, "k_scales", ArithType::F32),
        (v_scales, "v_scales", ArithType::F32),
    ] {
        let r = buf.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
            crate::Error::Backend(format!("{KERNEL}: {what} must be RocmStorage"))
        })?;
        if r.dtype.arith != want {
            return Err(crate::Error::Backend(format!(
                "{KERNEL}: {what} must be {want:?}, got {:?}",
                r.dtype.arith
            )));
        }
        if !r.device_ptr_is_valid() {
            return Err(crate::Error::Backend(format!(
                "{KERNEL}: {what} lacks a valid device pointer"
            )));
        }
    }
    fn dc<'a>(
        b: &'a dyn BackendStorage,
        what: &'static str,
    ) -> Result<&'a RocmStorage, crate::Error> {
        b.as_any().downcast_ref::<RocmStorage>().ok_or_else(|| {
            crate::Error::Backend(format!(
                "grim_qkv_attention_dev_i8: {what} must be RocmStorage"
            ))
        })
    }
    let out_s = dc(out, "out")?;
    let om_s = dc(out_max, "out_max")?;
    let os_s = dc(out_sum, "out_sum")?;
    let total_s = dc(total_dev, "total_dev")?;
    let oproj_s = dc(o_proj_w, "o_proj_w")?;
    let alibi_s = dc(alibi_slopes, "alibi_slopes")?;
    let kq = dc(k_codes, "k_codes")?;
    let ks = dc(k_scales, "k_scales")?;
    let vq = dc(v_codes, "v_codes")?;
    let vs = dc(v_scales, "v_scales")?;
    let mut q_ptr = q_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("q has no ptr".into()))?;
    let mut k_ptr = kq
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("k_codes have no ptr".into()))?;
    let mut ksc_ptr = ks
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("k_scales have no ptr".into()))?;
    let mut v_ptr = vq
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("v_codes have no ptr".into()))?;
    let mut vsc_ptr = vs
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("v_scales have no ptr".into()))?;
    let mut o_ptr = out_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("out has no ptr".into()))?;
    let mut om_ptr = om_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("out_max has no ptr".into()))?;
    let mut os_ptr = os_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("out_sum has no ptr".into()))?;
    let mut tot_ptr = total_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("total_dev has no ptr".into()))?;
    let mut nh = num_heads as i32;
    let mut nkv = num_kv_heads as i32;
    let mut hd = head_dim as i32;
    let mut sl = seq_len as i32;
    let mut steps_i = steps as i32;
    let mut isd = inv_sqrt_d;
    let mut wl = window_lo;
    let mut sc = softcap;
    let mut od = o_dim as i32;
    let mut fo = fuse_o as i32;
    let mut asl = alibi_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("alibi_slopes has no ptr".into()))?;
    let mut ha = has_alibi as i32;
    let mut batch_i = batch.max(1) as i32;
    let mut slot_stride_i = arena_slot_stride as i32;
    let mut sc_slot_stride_i = scale_slot_stride as i32;
    let mut opptr = oproj_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("o_proj_w has no ptr".into()))?;
    let grid_dim = crate::HipDim3::new(sl as u32, nh as u32, batch.max(1) as u32);
    let block_dim = crate::HipDim3::new(128, 1, 1);
    dev.launch_compute_kernel(
        KERNEL,
        grid_dim,
        block_dim,
        &mut [
            // Argument order MUST match `grim_qkv_attention_dev_i8`'s parameter
            // list exactly. The i8 kernel takes the code/scale pointers LAST
            // (after the batch/slot strides), not next to k/v: emitting them
            // early made the kernel treat `k_scales` as `out`, `num_heads` as a
            // pointer, and fault at the first dereference.
            arg(&mut q_ptr),
            arg(&mut k_ptr),
            arg(&mut v_ptr),
            arg(&mut o_ptr),
            arg(&mut om_ptr),
            arg(&mut os_ptr),
            arg(&mut nh),
            arg(&mut nkv),
            arg(&mut hd),
            arg(&mut sl),
            arg(&mut tot_ptr),
            arg(&mut steps_i),
            arg(&mut isd),
            arg(&mut wl),
            arg(&mut sc),
            arg(&mut opptr),
            arg(&mut od),
            arg(&mut fo),
            arg(&mut asl),
            arg(&mut ha),
            arg(&mut batch_i),
            arg(&mut slot_stride_i),
            arg(&mut ksc_ptr),
            arg(&mut vsc_ptr),
            arg(&mut sc_slot_stride_i),
            arg(&mut cap_rows_i),
        ],
    )
}

/// Item 3 launcher: `grim_bump_i32`. Increments `*past_dev += steps`. Single
/// thread; the mutable `past_dev` pointer is a graph input so each replay bumps
/// the live counter.
pub fn launch_bump_i32(
    dev: &crate::RocmDevice,
    past_dev: &dyn BackendStorage,
    steps: usize,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    launch_bump_i32_slots(dev, past_dev, steps, 1)
}

/// P3: bump one i32 per batch slot. Grid (batch,1,1), thread b bumps
/// `past_dev[b] += steps`. batch=1 matches the legacy single-thread kernel.
pub fn launch_bump_i32_slots(
    dev: &crate::RocmDevice,
    past_dev: &dyn BackendStorage,
    steps: usize,
    batch: usize,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let past_s = past_dev
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| crate::Error::Backend("bump_i32: past_dev must be RocmStorage".into()))?;
    let mut past_ptr = past_s
        .device_ptr
        .ok_or_else(|| crate::Error::Backend("bump_i32: past_dev has no device ptr".into()))?;
    let mut steps_i = steps as i32;
    let b = batch.max(1) as u32;
    dev.launch_compute_kernel(
        "grim_bump_i32",
        crate::HipDim3::new(b, 1, 1),
        crate::HipDim3::new(1, 1, 1),
        &mut [arg(&mut past_ptr), arg(&mut steps_i)],
    )
}

/// ScytheRing head publish — fully on-device (no pinned staging, no H2D).
/// Writes a single u32 head counter to head_dev. Launched as grid
/// (1,1,1)/block (1,1,1): thread 0 writes the value. HIP-graph capturable.
/// `head_dev_ptr` is the raw device address of the head scalar.
pub fn launch_scythe_publish_head(
    dev: &crate::RocmDevice,
    head_dev_ptr: u64,
    head_value: u32,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let mut hp = head_dev_ptr;
    let mut hv = head_value;
    let handle = dev.launch_compute_kernel(
        "grim_scythe_publish_head",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(1, 1, 1),
        &mut [arg(&mut hp), arg(&mut hv)],
    )?;
    crate::device::helpers::check_hip("scythe_publish_head: sync", unsafe {
        crate::hipStreamSynchronize(dev.active_stream())
    })?;
    Ok(handle)
}

/// ScytheRing descriptor write — fully on-device (no host pack, no H2D).
/// Writes one 64-byte task descriptor into ring slot `slot_idx`. Launched as
/// grid (1,1,1)/block (1,1,1): thread 0 packs the fields. This is HIP-graph
/// capturable, unlike the legacy pinned-staging + H2D path.
pub fn launch_scythe_write_slot(
    dev: &crate::RocmDevice,
    slots_base: &dyn BackendStorage,
    slot_idx: usize,
    opcode: u32,
    m: u32,
    n: u32,
    k: u32,
    input_ptr: u64,
    weight_ptr: u64,
    output_ptr: u64,
    peer_ptr: u64,
) -> Result<*mut std::ffi::c_void, crate::Error> {
    let slots_s = slots_base
        .as_any()
        .downcast_ref::<crate::memory::storage::RocmStorage>()
        .ok_or_else(|| {
            crate::Error::Backend("scythe_write_slot: slots must be RocmStorage".into())
        })?;
    let mut base_ptr = slots_s.device_ptr.ok_or_else(|| {
        crate::Error::Backend("scythe_write_slot: slots have no device ptr".into())
    })?;
    let mut slot = slot_idx as u32;
    let mut op = opcode;
    let mut mi = m;
    let mut nn = n;
    let mut kk = k;
    let mut inp = input_ptr;
    let mut wep = weight_ptr;
    let mut outp = output_ptr;
    let mut pep = peer_ptr;
    dev.launch_compute_kernel(
        "grim_scythe_write_slot",
        crate::HipDim3::new(1, 1, 1),
        crate::HipDim3::new(1, 1, 1),
        &mut [
            arg(&mut base_ptr),
            arg(&mut slot),
            arg(&mut op),
            arg(&mut mi),
            arg(&mut nn),
            arg(&mut kk),
            arg(&mut inp),
            arg(&mut wep),
            arg(&mut outp),
            arg(&mut pep),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::dtype::{ArithType, DType, Storage};
    use grim_tensor::{AttentionOps, CoreTensorOps, MemoryOps, Shape};

    /// WS-0: the explicit-format spelling table is the single source of truth.
    /// Every alias must resolve, and an unknown value must NOT silently become
    /// a valid format (it is `None`; callers warn and fall back).
    #[test]
    fn kv_arena_env_aliases_resolve() {
        use KvArenaFormat::*;
        for (s, want) in [
            ("f32", F32),
            ("f16", F16),
            ("int8", Int8),
            ("i8", Int8),
            ("q8", Int8),
            ("int4", Int4),
            ("i4", Int4),
            ("q4", Int4),
        ] {
            assert_eq!(parse_kv_arena_env(s), Some(want), "GRIM_KV_ARENA={s}");
        }
        assert_eq!(parse_kv_arena_env("bogus"), None);
        // Spellings are case-sensitive; an uppercase value is a typo, not f16.
        assert_eq!(parse_kv_arena_env("F16"), None);
        // WS-3A: the asymmetric format resolves as a first-class selector and,
        // as of the K2V4 kernels landing, reports itself implemented.
        assert_eq!(parse_kv_arena_env("int2v4"), Some(KvArenaFormat::Int2V4));
        assert_eq!(parse_kv_arena_env("k2v4"), Some(KvArenaFormat::Int2V4));
        assert!(KvArenaFormat::Int2V4.has_device_kernels());
        for f in [
            KvArenaFormat::F32,
            KvArenaFormat::F16,
            KvArenaFormat::Int8,
            KvArenaFormat::Int4,
        ] {
            assert!(f.has_device_kernels(), "{f:?} should have device kernels");
        }
        // Density: 2-bit K + 4-bit V averages 0.375 B/elem, half of int4's.
        assert!((KvArenaFormat::Int2V4.bytes_per_elem() - 0.375).abs() < 1e-6);
    }

    /// WS-0: the legacy f16 flag and the arena resolver must not drift. The
    /// backend resolver's only two inputs are these strings, so pinning the
    /// pure core pins the public `kv_arena_format`/`kv_f16_enabled` pair.
    #[test]
    fn kv_arena_resolver_is_consistent_with_f16_flag() {
        use KvArenaFormat::*;
        // Unset arena -> driven entirely by the legacy flag.
        assert_eq!(resolve_kv_arena_format(None, None), F32);
        assert_eq!(resolve_kv_arena_format(None, Some("1")), F16);
        assert_eq!(resolve_kv_arena_format(None, Some("off")), F32);
        // Explicit arena wins over the flag, in every spelling.
        assert_eq!(resolve_kv_arena_format(Some("f32"), Some("1")), F32);
        assert_eq!(resolve_kv_arena_format(Some("int8"), Some("1")), Int8);
        assert_eq!(resolve_kv_arena_format(Some("int4"), Some("0")), Int4);
        // Unrecognized explicit value falls back to F32 (with a warning).
        assert_eq!(resolve_kv_arena_format(Some("q7"), None), F32);
    }

    /// The f16 flag's disabled spellings are ONE set, not two — this is the
    /// drift that let `GRIM_F16_KV=off` mean f16 to the backend and f32 to the
    /// model before WS-0.
    #[test]
    fn f16_kv_flag_disabled_spellings() {
        for s in ["0", "false", "FALSE", "off", "no", " off "] {
            assert!(
                !parse_f16_kv_env(Some(s)),
                "GRIM_F16_KV={s:?} should disable"
            );
        }
        for s in ["1", "true", "yes", "on", "garbage"] {
            assert!(parse_f16_kv_env(Some(s)), "GRIM_F16_KV={s:?} should enable");
        }
        assert!(!parse_f16_kv_env(None), "unset must not enable f16");
    }

    /// WS-0: for an explicit value the two senses of the flag are exact
    /// complements, so a f16-default model and the f32-default backend can
    /// never disagree about what a spelling means. UNSET is where the two
    /// defaults legitimately differ (backend reads it as f32, a f16-default
    /// model as f16), and there BOTH parsers report false.
    #[test]
    fn f16_kv_flag_senses_are_complements() {
        for v in [
            "0", "false", "off", "no", "1", "true", "yes", "garbage", " off ",
        ] {
            assert_eq!(
                parse_f16_kv_env(Some(v)),
                !f16_kv_explicitly_disabled(Some(v)),
                "GRIM_F16_KV={v:?}"
            );
        }
        assert!(!parse_f16_kv_env(None));
        assert!(!f16_kv_explicitly_disabled(None));
    }

    #[test]
    fn test_wmma_qkv_attention_source_contains_tensor_core_and_fallback() {
        assert!(KERNEL_SOURCE.contains("void grim_qkv_attention_wmma"));
        assert!(KERNEL_SOURCE.contains("void grim_qkv_attention_paged_quant"));
        // Scalar fused-attention fallback still present alongside the tile.
        assert!(KERNEL_SOURCE.contains("for (int dim = 0; dim < 256; ++dim)"));
    }

    /// Structural regression for the attention score path (WI-SB6 follow-up).
    /// History, in order: (1) fragments loaded from single-token vectors with `frag_qk.x[0]` as the score -.
    #[test]
    fn wmma_attention_source_rejects_vector_as_matrix_fragments() {
        // The original one-vector fragment loads and single-lane score read.
        assert!(
            !KERNEL_SOURCE.contains("load_matrix_sync(frag_q, q + q_offset"),
            "matrix_a fragment must not be loaded from a single query vector"
        );
        assert!(
            !KERNEL_SOURCE.contains("load_matrix_sync(frag_k, k_tensor + kv_offset"),
            "matrix_b fragment must not be loaded from a single KV vector"
        );
        assert!(
            !KERNEL_SOURCE.contains("frag_qk.x[0]"),
            "a single accumulator element is not the Q·K dot product"
        );
        // Fragment-based scoring must not return: on gfx1201 it produced an
        // all-zero accumulator instead of an error.
        assert!(
            !KERNEL_SOURCE.contains("load_matrix_sync(frag_"),
            "no fragment loads in the attention kernel"
        );
        // The corrected tiling invariants: full 16x16 score tile staged in
        // shared memory with causal/edge masking into the softmax.
        assert!(
            KERNEL_SOURCE.contains("s_scores[k][r] = acc"),
            "score tile must be filled from explicit dot products"
        );
        assert!(
            KERNEL_SOURCE.contains("-INFINITY"),
            "edge tiles must mask invalid rows/keys out of the softmax"
        );
    }

    // PASSED: 2026-08-20 on gfx1036 (ROCm)
    #[test]
    fn test_wmma_qkv_attention_gpu_parity() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            return;
        };

        let seq_len = 2usize;
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 16usize;
        let kv_seq_len = 4usize;

        let q_size = seq_len * num_heads * head_dim;
        let kv_size = kv_seq_len * num_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..q_size).map(|i| ((i as f32) * 0.1).sin()).collect();
        let k_data: Vec<f32> = (0..kv_size).map(|i| ((i as f32) * 0.15).cos()).collect();
        let v_data: Vec<f32> = (0..kv_size)
            .map(|i| ((i as f32) * 0.08).sin() + 0.5)
            .collect();

        let q_shape = Shape::new(vec![seq_len, num_heads, head_dim]);
        let k_shape = Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]);
        let v_shape = Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]);
        let out_shape = Shape::new(vec![seq_len, num_heads, head_dim]);

        let q_storage = dev.from_cpu(&q_data, &q_shape, DType::F32).unwrap();
        let k_storage = dev.from_cpu(&k_data, &k_shape, DType::F32).unwrap();
        let v_storage = dev.from_cpu(&v_data, &v_shape, DType::F32).unwrap();
        let mut out_wmma_storage = dev.alloc_storage(&out_shape, DType::F32).unwrap();

        // 1. Run WMMA fused QKV attention
        launch_qkv_attention_wmma(
            &dev,
            q_storage.as_ref(),
            k_storage.as_ref(),
            v_storage.as_ref(),
            out_wmma_storage.as_mut(),
            None,
            None,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            seq_len as u32,
            kv_seq_len as u32,
            0,
            0,
        )
        .unwrap();
        dev.synchronize();

        // 2. Run standard fused QKV attention for baseline comparison
        let (out_ref_storage, handle) = dev
            .qkv_attention(
                q_storage.as_ref(),
                k_storage.as_ref(),
                v_storage.as_ref(),
                num_kv_heads,
                kv_seq_len,
                0,
                None,
                &out_shape,
                None,
                None,
            )
            .unwrap();
        handle.synchronize().unwrap();

        let wmma_res = out_wmma_storage.to_cpu_vec_f32().unwrap();
        let ref_res = out_ref_storage.to_cpu_vec_f32().unwrap();

        // CPU ground truth (causal softmax + GQA key-head mapping) computed
        // independently of both device paths; see the assertions below.
        let inv_sqrt_d_test = 1.0f32 / (head_dim as f32).sqrt();
        let mut host_exp = vec![0f32; seq_len * num_heads * head_dim];
        for i in 0..seq_len {
            for hh in 0..num_heads {
                let kh = hh / (num_heads / num_kv_heads);
                let scores: Vec<f32> = (0..=i)
                    .map(|j| {
                        let mut s = 0f32;
                        for d in 0..head_dim {
                            s += q_data[(i * num_heads + hh) * head_dim + d]
                                * k_data[(j * num_kv_heads + kh) * head_dim + d];
                        }
                        s * inv_sqrt_d_test
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for d in 0..head_dim {
                    let mut acc = 0f32;
                    for (j, e) in exps.iter().enumerate() {
                        acc += e * v_data[(j * num_kv_heads + kh) * head_dim + d];
                    }
                    host_exp[(i * num_heads + hh) * head_dim + d] = acc / sum;
                }
            }
        }
        // Host ground truth gates BOTH device paths independently: the tiled kernel and the scalar reference must
        // each match this CPU softmax, so a shared regression in one cannot hide behind the other.
        for (i, (w, h)) in wmma_res.iter().zip(&host_exp).enumerate() {
            assert!(
                (w - h).abs() < 1e-4,
                "tiled kernel diverges from host truth at {i}: {w} vs {h}"
            );
        }
        for (i, (r, h)) in ref_res.iter().zip(&host_exp).enumerate() {
            assert!(
                (r - h).abs() < 1e-4,
                "scalar reference diverges from host truth at {i}: {r} vs {h}"
            );
        }

        assert_eq!(wmma_res.len(), ref_res.len());
        for i in 0..wmma_res.len() {
            assert!(
                (wmma_res[i] - ref_res[i]).abs() < 1e-4,
                "Mismatch at index {i}: wmma={}, ref={}",
                wmma_res[i],
                ref_res[i]
            );
        }
    }

    /// Edge-tile companion to `test_wmma_qkv_attention_gpu_parity`: head_dim leaves a scalar tail (48 % 16), seq_len spans two 16-row query tiles, and a nonzero cache_offset shifts the causal horizon.
    /// PASSED first on gfx1201 with the tiled rewrite (2026-08-23).
    #[test]
    fn test_wmma_qkv_attention_gpu_parity_untiled_shapes() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            return;
        };

        let seq_len = 17usize;
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 48usize;
        let kv_seq_len = 35usize;
        let cache_offset = 3u32;

        let q_size = seq_len * num_heads * head_dim;
        let kv_size = kv_seq_len * num_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..q_size).map(|i| ((i as f32) * 0.037).sin()).collect();
        let k_data: Vec<f32> = (0..kv_size).map(|i| ((i as f32) * 0.041).cos()).collect();
        let v_data: Vec<f32> = (0..kv_size)
            .map(|i| ((i as f32) * 0.029).sin() + 0.5)
            .collect();

        let q_shape = Shape::new(vec![seq_len, num_heads, head_dim]);
        let k_shape = Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]);
        let v_shape = Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]);
        let out_shape = Shape::new(vec![seq_len, num_heads, head_dim]);

        let q_storage = dev.from_cpu(&q_data, &q_shape, DType::F32).unwrap();
        let k_storage = dev.from_cpu(&k_data, &k_shape, DType::F32).unwrap();
        let v_storage = dev.from_cpu(&v_data, &v_shape, DType::F32).unwrap();
        let mut out_wmma_storage = dev.alloc_storage(&out_shape, DType::F32).unwrap();

        launch_qkv_attention_wmma(
            &dev,
            q_storage.as_ref(),
            k_storage.as_ref(),
            v_storage.as_ref(),
            out_wmma_storage.as_mut(),
            None,
            None,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            seq_len as u32,
            kv_seq_len as u32,
            cache_offset,
            0,
        )
        .unwrap();
        dev.synchronize();

        let (out_ref_storage, handle) = dev
            .qkv_attention(
                q_storage.as_ref(),
                k_storage.as_ref(),
                v_storage.as_ref(),
                num_kv_heads,
                kv_seq_len,
                cache_offset,
                None,
                &out_shape,
                None,
                None,
            )
            .unwrap();
        handle.synchronize().unwrap();

        let wmma_res = out_wmma_storage.to_cpu_vec_f32().unwrap();
        let ref_res = out_ref_storage.to_cpu_vec_f32().unwrap();
        assert_eq!(wmma_res.len(), ref_res.len());
        for i in 0..wmma_res.len() {
            assert!(
                (wmma_res[i] - ref_res[i]).abs() < 1e-4,
                "Mismatch at index {i}: wmma={}, ref={}",
                wmma_res[i],
                ref_res[i]
            );
        }
    }

    // PASSED: 2026-08-20 on gfx1036 (ROCm)
    #[test]
    fn test_paged_attention_quant_formats_gpu() {
        if !crate::gpu_test_enabled() {
            return;
        }
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            return;
        };

        let batch = 1u32;
        let num_heads = 2u32;
        let num_kv_heads = 2u32;
        let head_dim = 16u32;
        let page_size = 4u32;
        let max_blocks = 2u32;
        let kv_seq_len = 4u32;

        let q_data = vec![1.0f32; (batch * num_heads * head_dim) as usize];
        let q_shape = Shape::new(vec![batch as usize, num_heads as usize, head_dim as usize]);
        let q_storage = dev.from_cpu(&q_data, &q_shape, DType::F32).unwrap();

        let block_tables_data = [
            BlockTableEntry {
                block_id: 0,
                page_size,
            },
            BlockTableEntry {
                block_id: 1,
                page_size,
            },
        ];
        let bt_bytes: Vec<u8> = block_tables_data
            .iter()
            .flat_map(|b| {
                let mut v = Vec::new();
                v.extend_from_slice(&b.block_id.to_ne_bytes());
                v.extend_from_slice(&b.page_size.to_ne_bytes());
                v
            })
            .collect();
        let bt_storage = dev
            .from_cpu_bytes(
                &bt_bytes,
                &Shape::new(vec![block_tables_data.len() * 8]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )
            .unwrap();

        let total_page_elems = (max_blocks * page_size * num_kv_heads * head_dim) as usize;
        let k_bytes = vec![2u8; total_page_elems];
        let v_bytes = vec![3u8; total_page_elems];

        let k_storage = dev
            .from_cpu_bytes(
                &k_bytes,
                &Shape::new(vec![total_page_elems]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )
            .unwrap();
        let v_storage = dev
            .from_cpu_bytes(
                &v_bytes,
                &Shape::new(vec![total_page_elems]),
                DType {
                    arith: ArithType::U8,
                    storage: Storage::Native,
                },
            )
            .unwrap();

        let out_shape = Shape::new(vec![batch as usize, num_heads as usize, head_dim as usize]);
        let mut out_storage = dev.alloc_storage(&out_shape, DType::F32).unwrap();

        // Verify across INT8, INT4 / W4A16, FP8, FP4, MXFP4, MXFP8, Nutcracker FP4
        for fmt in [
            KvCacheQuantFormat::Int8,
            KvCacheQuantFormat::Int4,
            KvCacheQuantFormat::Fp8E4M3,
            KvCacheQuantFormat::Fp8E5M2,
            KvCacheQuantFormat::Fp4E2M1,
            KvCacheQuantFormat::MxFp4,
            KvCacheQuantFormat::MxFp8,
            KvCacheQuantFormat::NvFp4,
            KvCacheQuantFormat::NutFp4,
        ] {
            launch_paged_attention_quant(
                &dev,
                q_storage.as_ref(),
                bt_storage.as_ref(),
                k_storage.as_ref(),
                v_storage.as_ref(),
                out_storage.as_mut(),
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                max_blocks,
                page_size,
                kv_seq_len,
                0,
                0,
                fmt,
                0.5f32,
                0.0f32,
                0.5f32,
                0.0f32,
            )
            .unwrap();
            dev.synchronize();

            let out_vec = out_storage.to_cpu_vec_f32().unwrap();
            assert_eq!(out_vec.len(), (batch * num_heads * head_dim) as usize);
            assert!(out_vec[0].is_finite());
        }
    }

    // ── Nutcracker / E2M1 decode guards ───────────────────────────────────

    /// The paged-KV FP4 arms must use the OCP E2M1 subnormal step of 0.5.
    /// This caught a real 2x error on codes 0x1/0x9 (they decoded as +/-0.25).
    #[test]
    fn e2m1_subnormal_step_is_half_not_quarter() {
        // No E2M1 arm may reintroduce the 0.25 step.
        assert!(
            !KERNEL_SOURCE.contains("* 0.25f"),
            "E2M1 subnormal step regressed to 0.25; OCP E2M1 uses 0.5"
        );
        // Pin the corrected step in each of the four 4-bit arms (Fp4E2M1,
        // MxFp4, NvFp4, NutFp4). The subnormal ternary head is the unambiguous
        // marker: the normalized branch also contains "* 0.5f" as a mantissa
        // addend, so a bare substring count double-counts each arm.
        let count = KERNEL_SOURCE.matches("? ((float)mant * 0.5f) :").count();
        assert_eq!(
            count, 4,
            "expected Fp4E2M1, MxFp4, NvFp4 and NutFp4 to use the 0.5 subnormal step"
        );
    }

    /// Every E2M1 decoder in grim must agree on the subnormal step.
    #[test]
    fn e2m1_decoders_agree_with_grim_quant() {
        for code in [0x1u8, 0x9] {
            assert_eq!(
                grim_quant::mxfp4_e2m1_to_f32(code, 127),
                if code & 0x8 != 0 { -0.5 } else { 0.5 },
                "grim-quant E2M1 subnormal step drifted for code {code:#04x}"
            );
        }
    }

    /// Nutcracker is a 4-bit element format, so it must tag as FP4 and not FP8.
    #[test]
    fn nutcracker_kv_tag_is_four_bit() {
        assert_eq!(KvCacheQuantFormat::NvFp4 as i32, 7);
        assert_eq!(KvCacheQuantFormat::NutFp4 as i32, 8);
        assert!(KvCacheQuantFormat::NutFp4 as i32 > KvCacheQuantFormat::NvFp4 as i32);
        // Distinct tag: reusing an existing one would silently decode as that
        // format, since the wire layout is byte-identical to NvFp4.
        let all = [
            KvCacheQuantFormat::Int8,
            KvCacheQuantFormat::Int4,
            KvCacheQuantFormat::Fp8E4M3,
            KvCacheQuantFormat::Fp8E5M2,
            KvCacheQuantFormat::Fp4E2M1,
            KvCacheQuantFormat::MxFp4,
            KvCacheQuantFormat::MxFp8,
            KvCacheQuantFormat::NvFp4,
            KvCacheQuantFormat::NutFp4,
        ];
        let mut seen = std::collections::BTreeSet::new();
        for f in all {
            assert!(
                seen.insert(f as i32),
                "duplicate KvCacheQuantFormat tag {f:?}"
            );
        }
    }

    /// The kernel's Nutcracker arm must keep matching the grim-quant reference.
    ///
    /// There is no GPU in this test path, so this is a two-part guard: the
    /// behavioural check pins the intended semantics against
    /// `grim_quant::nutcracker_e2m1_to_f32`, and the source assertions below
    /// pin the kernel text so the two cannot drift apart silently.
    #[test]
    fn nutcracker_kv_arm_matches_grim_quant_reference() {
        // Transcribe the kernel arm's index math verbatim and check it against
        // the reference decoder for every code x selector pair.
        for sel in 0..4u8 {
            for code in 0u8..16 {
                let scale_byte = (31u8 << 2) | sel; // exp = 31 -> scale 1.0
                let packed = nutcracker_pack_reference(scale_byte, code);

                let sub = 0i32;
                let in_sub = code as i32; // one value per idx, idx < 16
                let blk = &packed[(sub * 9) as usize..];
                let sel_k = blk[0] & 0x3;
                let blk_scale = (2.0f32).powi((blk[0] >> 2) as i32 - 31);
                let code_byte = blk[1 + (in_sub >> 1) as usize];
                let nibble = if in_sub & 1 != 0 {
                    code_byte >> 4
                } else {
                    code_byte & 0x0F
                };

                let kernel_val = if nibble & 0x7 == 0 {
                    let mag = if sel_k & 0x1 != 0 { 2.5f32 } else { 5.0f32 };
                    (if sel_k & 0x2 != 0 { -mag } else { mag }) * blk_scale
                } else {
                    let sign = (nibble >> 3) & 1;
                    let exp = (nibble >> 1) & 0x03;
                    let mant = nibble & 1;
                    let val = if exp == 0 {
                        mant as f32 * 0.5
                    } else {
                        (1.0 + mant as f32 * 0.5) * (2.0f32).powi(exp as i32 - 1)
                    };
                    (if sign != 0 { -val } else { val }) * blk_scale
                };

                assert_eq!(
                    kernel_val,
                    grim_quant::nutcracker_e2m1_to_f32(code, scale_byte),
                    "sel {sel} code {code:#04x} diverged between the KV kernel and grim-quant"
                );
            }
        }
    }

    /// Pin the kernel source literals so the arm cannot be edited in isolation.
    #[test]
    fn nutcracker_kv_arm_source_is_pinned() {
        assert!(KERNEL_SOURCE.contains("quant_format == 8"));
        // Reads its own scale: 9 bytes per 16 values.
        assert!(KERNEL_SOURCE.contains("const unsigned char* blk = data + sub * 9;"));
        // Selector steals the low 2 bits, bias 31, matching grim-quant.
        assert!(KERNEL_SOURCE.contains("int sel = blk[0] & 0x3;"));
        assert!(KERNEL_SOURCE.contains("ldexpf(1.0f, (int)(blk[0] >> 2) - 31)"));
        // Repurposed-zero test keys on the low 3 bits, never the sign.
        assert!(KERNEL_SOURCE.contains("if ((nibble & 0x7) == 0)"));
    }

    /// Regression guards for the two pre-existing paged-KV decode bugs.
    #[test]
    fn mxfp8_uses_a_four_bit_exponent_mask() {
        // E4M3 has 4 exponent bits. This was masked with 0x07, truncating every
        // value >= 2^8 into the wrong binade.
        assert!(
            KERNEL_SOURCE.contains("(b >> 3) & 0x0F"),
            "MXFP8 must mask the exponent with 0x0F (E4M3 has 4 exponent bits)"
        );
        assert!(
            !KERNEL_SOURCE.contains("int exp = (b >> 3) & 0x07;"),
            "MXFP8 regressed to a 3-bit exponent mask"
        );
    }

    #[test]
    fn mxfp4_reads_its_block32_scale() {
        // The arm previously applied only the caller's per-tensor scale, which
        // is not MXFP4. It must read the inline E8M0 byte per 32 values.
        assert!(KERNEL_SOURCE.contains("const int MXFP4_GROUP = 32;"));
        assert!(KERNEL_SOURCE.contains("const int MXFP4_BYTES = 17;"));
        assert!(KERNEL_SOURCE.contains("exp2f((float)(int)g[0] - 127.0f)"));
        assert!(KERNEL_SOURCE.contains("const int MXFP8_GROUP = 32;"));
        assert!(KERNEL_SOURCE.contains("const int MXFP8_BYTES = 33;"));
    }

    /// Page allocation must agree with the decode strides.
    #[test]
    fn packed_bytes_matches_the_kernel_strides() {
        use KvCacheQuantFormat::*;
        for n in [16usize, 32, 64, 100] {
            // 9 bytes per 16 values for the per-16 inline-scale formats.
            assert_eq!(NvFp4.packed_bytes(n), n.div_ceil(2) + n.div_ceil(16));
            assert_eq!(NutFp4.packed_bytes(n), n.div_ceil(2) + n.div_ceil(16));
            // 17 bytes per 32 for MxFp4, 33 per 32 for MxFp8.
            assert_eq!(MxFp4.packed_bytes(n), n.div_ceil(2) + n.div_ceil(32));
            assert_eq!(MxFp8.packed_bytes(n), n + n.div_ceil(32));
            // No inline scale byte for the caller-scaled formats.
            assert_eq!(Fp4E2M1.packed_bytes(n), n.div_ceil(2));
            assert_eq!(Int8.packed_bytes(n), n);
            assert_eq!(Fp8E4M3.packed_bytes(n), n);
        }
    }

    /// Build a 9-byte Nutcracker sub-block carrying a single code at `idx`.
    fn nutcracker_pack_reference(scale_byte: u8, idx: u8) -> [u8; 9] {
        let mut blk = [0u8; 9];
        blk[0] = scale_byte;
        blk[1 + (idx as usize / 2)] = if idx % 2 == 0 { idx & 0x0F } else { idx << 4 };
        blk
    }

    // ── Item 3 parity tests ────────────────────────────────────────────────

    fn as_u8_slice<T>(slice: &[T]) -> &[u8] {
        let len = std::mem::size_of_val(slice);
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, len) }
    }

    /// `grim_kv_append` with a device-side past must match the host-offset
    /// `copy_slice_into` result, byte-exact, for past ∈ {0, 17, 500}.
    #[test]
    fn test_kv_append_dev_parity() {
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            eprintln!("[SKIP] requires GPU");
            return;
        };
        use grim_tensor::MemoryOps;
        let kv_stride = 128usize;
        let steps = 1usize;
        let max_past = 600usize;
        let arena_rows = max_past + steps;

        let mut seed = 0xAFF1CEu64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6363136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let k_rot: Vec<f32> = (0..steps * kv_stride).map(|_| rand()).collect();

        for past in [0usize, 17, 500] {
            // Host reference: copy k_rot into a fresh arena at offset past*stride.
            let mut arena_ref = vec![0.0f32; arena_rows * kv_stride];
            arena_ref[past * kv_stride..past * kv_stride + steps * kv_stride]
                .copy_from_slice(&k_rot);

            // Device path: grim_kv_append reads past from device memory.
            let arena_dev = vec![0.0f32; arena_rows * kv_stride];
            let arena_storage = dev
                .from_cpu(
                    &arena_dev,
                    &Shape::new(vec![arena_rows * kv_stride]),
                    DType::F32,
                )
                .unwrap();
            let rot_storage = dev
                .from_cpu(&k_rot, &Shape::new(vec![steps * kv_stride]), DType::F32)
                .unwrap();
            let past_val = past as u32;
            let past_dev = MemoryOps::from_cpu_bytes(
                &dev,
                as_u8_slice(&[past_val]),
                &Shape::new(vec![1]),
                DType::U32,
            )
            .unwrap();
            launch_kv_append(
                &dev,
                arena_storage.as_ref(),
                rot_storage.as_ref(),
                past_dev.as_ref(),
                kv_stride,
                steps,
                KvRingPolicy::UNLIMITED,
            )
            .unwrap();
            dev.synchronize();

            let arena_got = arena_storage.to_cpu_vec_f32().unwrap();
            assert_eq!(
                arena_got, arena_ref,
                "kv_append dev-offset mismatch at past={past}"
            );
        }
    }

    /// `grim_bump_i32` increments a device scalar. After 5 bumps of 1 the value
    /// must read back as 5 (counter contract).
    #[test]
    fn test_bump_monotonic() {
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            eprintln!("[SKIP] requires GPU");
            return;
        };
        use grim_tensor::MemoryOps;
        let steps = 1usize;
        let past_val = 0u32;
        let past_storage = MemoryOps::from_cpu_bytes(
            &dev,
            as_u8_slice(&[past_val]),
            &Shape::new(vec![1]),
            DType::U32,
        )
        .unwrap();
        for _ in 0..5 {
            launch_bump_i32(&dev, past_storage.as_ref(), steps).unwrap();
        }
        dev.synchronize();
        let raw = past_storage
            .as_any()
            .downcast_ref::<crate::memory::storage::RocmStorage>()
            .unwrap()
            .copy_to_host()
            .unwrap();
        let got = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        assert_eq!(got, 5, "past_dev should be 5 after 5 bumps of 1");
    }

    /// `grim_qkv_attention_dev` (device-total attention) must match the
    /// host-scalar `dev.qkv_attention` at the same total KV length, max_diff ≤ 1e-5.
    /// past ∈ {0, 17, 500}: total = past + steps.
    #[test]
    fn test_qkv_attention_dev_parity() {
        let Ok(dev) = crate::RocmDevice::try_new(0) else {
            eprintln!("[SKIP] requires GPU");
            return;
        };
        use grim_tensor::{AttentionOps, CoreTensorOps, MemoryOps};
        let num_heads = 4usize;
        let num_kv_heads = 4usize;
        let head_dim = 64usize;
        let steps = 1usize;
        let kv_stride = num_kv_heads * head_dim;

        let mut seed = 0xDECAFEu64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6363136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        for past in [0usize, 17, 500] {
            let total = past + steps;
            // K/V arenas sized to total rows (the appended row occupies [past, total)).
            let k_data: Vec<f32> = (0..total * kv_stride).map(|_| rand()).collect();
            let v_data: Vec<f32> = (0..total * kv_stride).map(|_| rand()).collect();
            let q_data: Vec<f32> = (0..steps * num_heads * head_dim).map(|_| rand()).collect();

            let k_shape = Shape::new(vec![total, num_kv_heads, head_dim]);
            let v_shape = k_shape.clone();
            let q_shape = Shape::new(vec![steps, num_heads, head_dim]);
            let out_shape = Shape::new(vec![steps, num_heads * head_dim]);

            let q_st = dev.from_cpu(&q_data, &q_shape, DType::F32).unwrap();
            let k_st = dev.from_cpu(&k_data, &k_shape, DType::F32).unwrap();
            let v_st = dev.from_cpu(&v_data, &v_shape, DType::F32).unwrap();

            // Reference: host-scalar attention with the same total + cache_offset.
            let (ref_st, _rh) = dev
                .qkv_attention(
                    q_st.as_ref(),
                    k_st.as_ref(),
                    v_st.as_ref(),
                    num_kv_heads,
                    total,
                    past as u32,
                    None,
                    &out_shape,
                    None,
                    None,
                )
                .unwrap();
            let ref_v = ref_st.to_cpu_vec_f32().unwrap();

            // Device path: grim_qkv_attention_dev reads total = *past_dev + steps.
            let out_st = dev.alloc_storage(&out_shape, DType::F32).unwrap();
            let out_max = dev
                .alloc_storage(&Shape::new(vec![num_heads]), DType::F32)
                .unwrap();
            let out_sum = dev
                .alloc_storage(&Shape::new(vec![num_heads]), DType::F32)
                .unwrap();
            let dummy = dev.alloc_storage(&Shape::new(vec![1]), DType::F32).unwrap();
            let past_val = past as u32;
            let past_dev_st = MemoryOps::from_cpu_bytes(
                &dev,
                as_u8_slice(&[past_val]),
                &Shape::new(vec![1]),
                DType::U32,
            )
            .unwrap();
            let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
            let stream = launch_qkv_attention_dev(
                &dev,
                q_st.as_ref(),
                k_st.as_ref(),
                v_st.as_ref(),
                out_st.as_ref(),
                out_max.as_ref(),
                out_sum.as_ref(),
                past_dev_st.as_ref(),
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                steps as u32,
                steps as u32,
                inv_sqrt_d,
                0,
                0.0,
                dummy.as_ref(),
                0,
                0,
                dummy.as_ref(),
                0,
                KvRingPolicy::UNLIMITED,
            )
            .unwrap();
            let _ = stream;
            dev.synchronize();

            let got = out_st.to_cpu_vec_f32().unwrap();
            let diff = got
                .iter()
                .zip(&ref_v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[qkv-attn-dev-parity] past={past} total={total} max_diff={diff:.8}");
            assert!(
                diff < 1e-5,
                "qkv_attention_dev diverges at past={past}: max_diff={diff}"
            );
        }
    }
}

#[cfg(test)]
mod kv_kernel_source_structure_tests {
    use super::KERNEL_SOURCE;

    /// Every `grim_*` kernel in this TU must carry `extern "C" __global__`.
    /// A script-generated duplicate once lost its prefix line: the TU still
    /// parsed (the body became a host function), hiprtc failed every entry
    /// compiled afterwards with `__shared__ not allowed in __host__`, and the
    /// failure surfaced far from its cause. String-level, runs everywhere.
    #[test]
    fn every_kernel_has_global_prefix() {
        for (i, line) in KERNEL_SOURCE.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("void grim_") {
                let prev: Vec<&str> = KERNEL_SOURCE.lines().take(i).collect();
                let mut j = prev.len();
                while j > 0 && prev[j - 1].trim().is_empty() {
                    j -= 1;
                }
                let pre = prev[j - 1];
                assert!(
                    pre.contains("__global__"),
                    "kernel without __global__ prefix near TU line {}: {t} (previous: {pre})",
                    i + 1,
                );
            }
        }
    }
}

#[cfg(test)]
mod kv_arena_dtype_tests {
    use super::kv_arena_f16;
    use grim_tensor::dtype::{ArithType, Storage};
    use grim_tensor::{CoreTensorOps, DType, Shape};

    fn gpu_dev() -> Option<crate::RocmDevice> {
        crate::RocmDevice::try_new(0).ok()
    }

    fn arena_of(dev: &crate::RocmDevice, arith: ArithType) -> Box<dyn grim_tensor::BackendStorage> {
        let shape = Shape::new(vec![4, 64]);
        let data = vec![0.5f32; 4 * 64];
        dev.from_cpu(
            &data,
            &shape,
            DType {
                arith,
                storage: Storage::Native,
            },
        )
        .unwrap()
    }

    /// Storage truth: uniform F32 or uniform F16 arenas are accepted
    /// regardless of `GRIM_F16_KV` (no env mutation here, so no races with
    /// parallel tests). Packed/quant storages fail loud — the dense kernels
    /// would reinterpret their bytes.
    #[test]
    fn kv_arena_dtype_storage_truth() {
        let Some(dev) = gpu_dev() else {
            eprintln!("[SKIP] requires GPU");
            return;
        };
        assert_eq!(
            kv_arena_f16(arena_of(&dev, ArithType::F32).as_ref(), "m3-test").unwrap(),
            false,
            "F32 arena must take the f32 leg"
        );
        assert_eq!(
            kv_arena_f16(arena_of(&dev, ArithType::F16).as_ref(), "m3-test").unwrap(),
            true,
            "F16 arena must take the f16 leg"
        );
    }

    /// The flag is an allocation hint, not a behavior switch: F32 arenas
    /// keep working with the flag ON, and F16 arenas with it OFF.
    #[test]
    fn kv_arena_dtype_ignores_env() {
        let Some(dev) = gpu_dev() else {
            eprintln!("[SKIP] requires GPU");
            return;
        };
        temp_env::with_var("GRIM_F16_KV", Some("1"), || {
            assert_eq!(
                kv_arena_f16(arena_of(&dev, ArithType::F32).as_ref(), "m3-test").unwrap(),
                false
            );
        });
        temp_env::with_var("GRIM_F16_KV", Some("0"), || {
            assert_eq!(
                kv_arena_f16(arena_of(&dev, ArithType::F16).as_ref(), "m3-test").unwrap(),
                true
            );
        });
    }
}
