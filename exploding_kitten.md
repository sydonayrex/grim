# Exploding Kittens: Porting Mixture-of-Kittens Megakernel Patterns to grim's RDNA Backend

> A full-Rust technical integration guide for rewriting grim's ROCm kernels with Wave32 rocWMMA targeting gfx1200 and newer GPUs. All kernel source is HIP embedded in Rust string literals; all host logic is Rust.

---

## 1. Architectural Premise

Mixture-of-Kittens (MoK) is built for NVIDIA Blackwell NVL72s—SM100/SM103, warp-32, CUDA 13. grim targets AMD ROCm on RDNA3/4. The ISAs do not interoperate, but the **control-flow patterns, data layouts, and overlap strategies** transfer cleanly because both architectures converge on 32-thread SIMD execution units.

| MoK (Blackwell) | grim target (RDNA4) |
|---|---|
| Warp = 32 threads | Wave32 = 32 threads |
| WGMMA / Tensor Core | rocWMMA (gfx1200+) |
| NVLink P2P multicast | PCIe/Infinity Fabric `atomicAdd` |
| Shared memory 228 KB / SM | LDS 128 KB / CU |
| MXFP8 block scales `[E*M/128, N/128, 32, 16]` | Same layout, scale lookup in wave registers |

rocWMMA is available on **gfx1200 and newer** (RDNA4). On gfx1100/gfx1103 (RDNA3), rocWMMA provides FP16/BF16 16×16 tiles but lacks native FP8 matrix instructions. This document assumes a **gfx1200+ baseline** where rocWMMA exposes the full tile path.

---

## 2. Wave32 Execution Model

RDNA organizes work into **Wave32** units—exactly 32 lanes executing in lockstep. This is a 1:1 semantic match with NVIDIA warps, which means MoK's warp-level scheduling logic ports with minimal translation:

- **One wave** = one rocWMMA 16×16 tile operation.
- **Four waves** = a 128-thread block (4× Wave32), matching MoK's 4-warp blocks.
- **Wavefront size** is always 32 on RDNA; grim's block sizing emits 32, 64, 96, or 128, never 256.

```rust
// grim-backend-rocm/src/kernels/charon_wmma.rs

/// Wave32-aligned block dimension for RDNA.
///
/// RDNA uses Wave32 (32 threads). rocWMMA 16×16 tiles map to one wave.
/// A block of 128 threads = 4 waves, which saturates a CU while leaving
/// LDS headroom for fragment staging.
pub const WAVE_SIZE: u32 = 32;
pub const WAVES_PER_BLOCK: u32 = 4;
pub const BLOCK_DIM: u32 = WAVE_SIZE * WAVES_PER_BLOCK; // 128
```

A block of 128 threads on RDNA4 can hold **four concurrent 16×16 WMMA tiles** in flight, using ~32 KB of LDS for fragment staging—well under the 128 KB limit.

---

## 3. rocWMMA on gfx1200+: Tile Geometry

rocWMMA on gfx1200 supports the following fragment shapes:

| Fragment | Dimensions | Data Types |
|---|---|---|
| `matrix_a` | 16 × 16 × 16 | `__half`, `__hip_bfloat16`, `__hip_fp8_e4m3_fnuz` |
| `matrix_b` | 16 × 16 × 16 | `__half`, `__hip_bfloat16`, `__hip_fp8_e4m3_fnuz` |
| `accumulator` | 16 × 16 × 16 | `float` |

Each 16×16×16 `mma_sync` consumes one wave (32 threads). The K-dimension is unrolled in steps of 16.

### 3.1 WMMA MoE Tile Stack

A single wave processes one (token, expert) pair through three projections:

```
Wave lanes 0-31:
  ├─ Load activations [1, 16] into frag_a
  ├─ MMA with gate_w [16, 16] → frag_gate (FP32 accum)
  ├─ MMA with up_w   [16, 16] → frag_up   (FP32 accum)
  ├─ Store gate/up to LDS
  ├─ In-LDS SiLU(gate) * up → hidden [1, 16]
  ├─ Load hidden into frag_a_down
  ├─ MMA with down_w [16, 16] → frag_out (FP32 accum)
  └─ AtomicAdd result to output buffer (local or peer)
```

All three MMAs happen in the **same kernel launch** without HBM round-trips for intermediate activations. This is the MoK megakernel pattern adapted to Wave32.

---

## 4. Kernel Rewrite: `charon.rs` → `charon_wmma.rs`

The current `grim_moe_fused_dispatch` in `charon.rs` uses scalar triple-nested loops. The rewrite targets rocWMMA fragments with Wave32 scheduling.

### 4.1 HIP Source in Rust

```rust
// grim-backend-rocm/src/kernels/charon_wmma.rs

//! Charon WMMA — P-DAFD fused MoE dispatch kernel using rocWMMA on gfx1200+.
//!
//! One kernel launch carries every routed token to its expert via Wave32
//! block-to-expert assignment. Gate + up projections use WMMA fragments,
//! SiLU fusion happens in LDS, down projection uses WMMA, and the result
//! is accumulated into the token's output row via atomicAdd. Peer P2P
//! writes are inlined for expert-parallel combine.

use std::ffi::c_void;
use grim_tensor::error::{Error, Result};

// ---------------------------------------------------------------------------
// HIP source — grim_moe_wmma_dispatch
// ---------------------------------------------------------------------------

pub const KERNEL_SOURCE: &str = r#"
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

extern "C" __global__ void grim_moe_wmma_dispatch(
    const __hip_bfloat16* __restrict__ activations,
    const __hip_bfloat16* __restrict__ expert_gate_w,
    const __hip_bfloat16* __restrict__ expert_up_w,
    const __hip_bfloat16* __restrict__ expert_down_w,
    const unsigned int* __restrict__ router_tokens,
    const unsigned int* __restrict__ router_experts,
    const float* __restrict__ router_weights,
    float* __restrict__ out,
    float* __restrict__ peer_out,
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor,
    int n_total, int col_offset)
{
    const unsigned long long pair =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= (unsigned long long)num_pairs) return;

    const unsigned int tok = router_tokens[pair];
    const unsigned int exp = router_experts[pair];
    const float w = router_weights[pair];

    // Wave32: each thread handles 8 elements of a 16x16 tile.
    fragment<matrix_a, 16, 16, 16, __hip_bfloat16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __hip_bfloat16, col_major> frag_b_gate;
    fragment<matrix_b, 16, 16, 16, __hip_bfloat16, col_major> frag_b_up;
    fragment<accumulator, 16, 16, 16, float> frag_gate;
    fragment<accumulator, 16, 16, 16, float> frag_up;

    fill_fragment(frag_gate, 0.0f);
    fill_fragment(frag_up, 0.0f);

    for (int k = 0; k < hidden; k += 16) {
        load_matrix_sync(frag_a, activations + (unsigned long long)tok * hidden + k, hidden);
        load_matrix_sync(frag_b_gate,
            expert_gate_w + (unsigned long long)exp * inter * hidden + k, hidden);
        mma_sync(frag_gate, frag_a, frag_b_gate, frag_gate);

        load_matrix_sync(frag_b_up,
            expert_up_w + (unsigned long long)exp * inter * hidden + k, hidden);
        mma_sync(frag_up, frag_a, frag_b_up, frag_up);
    }

    // LDS partitioned by wave ID: 4 waves × 256 elements × 2 bytes = 2 KB
    __shared__ __hip_bfloat16 s_pool[4 * 16 * 16];
    const int wave_id = threadIdx.x / 32;
    __hip_bfloat16* s_gate = s_pool + wave_id * 256;
    __hip_bfloat16* s_up   = s_pool + wave_id * 256;
    __hip_bfloat16* s_hidden = s_pool + wave_id * 256;

    store_matrix_sync(s_gate, frag_gate, 16, mem_row_major);
    store_matrix_sync(s_up, frag_up, 16, mem_row_major);
    __syncthreads();

    // In-LDS SiLU fusion: each thread processes 8 elements
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        float g = __bfloat162float(s_gate[i]);
        float u = __bfloat162float(s_up[i]);
        float silu = g / (1.0f + expf(-g));
        s_hidden[i] = __float2bfloat16(silu * u);
    }
    __syncthreads();

    // Down projection
    fragment<matrix_a, 16, 16, 16, __hip_bfloat16, row_major> frag_a_down;
    fragment<matrix_b, 16, 16, 16, __hip_bfloat16, col_major> frag_b_down;
    fragment<accumulator, 16, 16, 16, float> frag_out;

    fill_fragment(frag_out, 0.0f);
    for (int k = 0; k < inter; k += 16) {
        load_matrix_sync(frag_a_down, s_hidden + k, 16);
        load_matrix_sync(frag_b_down,
            expert_down_w + (unsigned long long)exp * hidden * inter + k, inter);
        mma_sync(frag_out, frag_a_down, frag_b_down, frag_out);
    }

    // Weighted accumulate + atomicAdd (local and optional peer)
    store_matrix_sync(s_gate, frag_out, 16, mem_row_major);
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        float val = routed_scaling_factor * w * __bfloat162float(s_gate[i]);
        int h = i / 16;
        int col = i % 16;
        unsigned long long out_idx = (unsigned long long)tok * hidden + h * 16 + col;
        atomicAdd(out + out_idx, val);

        if (peer_out != nullptr) {
            unsigned long long peer_idx =
                (unsigned long long)tok * n_total + col_offset + h * 16 + col;
            atomicAdd(peer_out + peer_idx, val);
        }
    }
}
"#;
```

### 4.2 Host Launch Planner (Rust)

```rust
// grim-backend-rocm/src/kernels/charon_wmma.rs (continued)

/// Resolved kernel launch parameters for one WMMA fused dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharonWmmaLaunchPlan {
    /// Grid x = ceil(num_pairs / block_dim). 128-thread blocks (4 waves).
    pub grid_x: u32,
    pub block_x: u32,
    /// Peer P2P mapping for expert-parallel combine. None = local-only.
    pub peer: Option<PeerMapping>,
}

/// Peer GPU memory mapping for inline P2P atomic accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerMapping {
    /// Device pointer to peer's combine buffer (via hipDeviceEnablePeerAccess).
    pub peer_out: *mut c_void,
    /// Column offset of this rank's shard in the peer buffer.
    pub col_offset: usize,
    /// Total columns in the peer buffer.
    pub n_total: usize,
}

/// Choose the Wave32-aligned block dimension for a fused dispatch.
///
/// RDNA is always Wave32. Cap at 4 wavefronts (128 threads) to leave LDS
/// budget for 4 concurrent tile pipelines.
pub(crate) fn choose_block_dim(num_pairs: usize) -> u32 {
    const WAVES_MAX: u32 = 4;
    let one_wave = WAVE_SIZE.max(1);
    if num_pairs == 0 {
        return one_wave;
    }
    let target = num_pairs.max(one_wave as usize) as u32;
    let mut block = one_wave;
    while block < target && block < one_wave * WAVES_MAX {
        block *= 2;
    }
    block.min(one_wave * WAVES_MAX)
}

/// Pure planner: resolve grid/block for a fused dispatch.
pub(crate) fn plan_fused_dispatch(
    assignment: &RoutingAssignment,
    peer: Option<PeerMapping>,
) -> CharonWmmaLaunchPlan {
    let n = assignment.num_pairs();
    let block_x = choose_block_dim(n);
    let grid_x = if n == 0 {
        0
    } else {
        ((n as u32 + block_x - 1) / block_x) as u32
    };
    CharonWmmaLaunchPlan { grid_x, block_x, peer }
}

/// Validate host-side inputs before any device pointer is dereferenced.
pub(crate) fn validate_launch_inputs(
    activations: *mut c_void,
    expert_gate_w: *mut c_void,
    expert_up_w: *mut c_void,
    expert_down_w: *mut c_void,
    out: *mut c_void,
    assignment: &RoutingAssignment,
    hidden: usize,
    inter: usize,
) -> Result<()> {
    for (label, p) in [
        ("activations", activations),
        ("expert_gate_w", expert_gate_w),
        ("expert_up_w", expert_up_w),
        ("expert_down_w", expert_down_w),
        ("out", out),
    ] {
        if p.is_null() {
            return Err(Error::Backend(format!(
                "charon_wmma_dispatch: {label} is null"
            )));
        }
    }
    if hidden == 0 || inter == 0 {
        return Err(Error::Backend(format!(
            "charon_wmma_dispatch: degenerate shape (hidden={hidden}, inter={inter})"
        )));
    }
    let _ = assignment.num_pairs();
    Ok(())
}
```

---

## 5. FP8 / MXFP8 on gfx1200 with rocWMMA

gfx1200 rocWMMA may expose `__hip_fp8_e4m3_fnuz` fragments. If not yet available, use **dequantization-on-load** into BF16 fragments.

### 5.1 Native FP8 WMMA Kernel Source (gfx1200+)

```rust
// grim-backend-rocm/src/kernels/fp8_wmma.rs

//! FP8 WMMA GEMM kernel for gfx1200+ using native rocWMMA FP8 fragments.

pub const FP8_WMMA_SOURCE: &str = r#"
#if defined(__gfx1200__) || defined(__gfx1201__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

extern "C" __global__ void grim_fp8_wmma_gemm(
    const __hip_fp8_e4m3_fnuz* __restrict__ A,
    const __hip_fp8_e4m3_fnuz* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (tile_row * 16 >= M || tile_col * 16 >= N) return;

    fragment<matrix_a, 16, 16, 16, __hip_fp8_e4m3_fnuz, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __hip_fp8_e4m3_fnuz, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;

    fill_fragment(frag_c, 0.0f);

    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(frag_a, A + tile_row * 16 * stride_a + k, stride_a);
        load_matrix_sync(frag_b, B + tile_col * 16 + k * stride_b, stride_b);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    __shared__ float c_tile[16 * 16];
    store_matrix_sync(c_tile, frag_c, 16, mem_row_major);

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            C[(tile_row * 16 + i) * stride_c + (tile_col * 16 + j)] = c_tile[i * 16 + j];
        }
    }
}
#else
// Fallback: scalar loop for pre-gfx1200 architectures
extern "C" __global__ void grim_fp8_wmma_gemm(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int row = idx / N;
    const int col = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = fp8_e4m3_to_float_hip(A[row * stride_a + k]);
        float b_val = fp8_e4m3_to_float_hip(B[k * stride_b + col]);
        acc += a_val * b_val;
    }
    C[row * stride_c + col] = acc;
}
#endif
"#;
```

### 5.2 MXFP8 → BF16 Dequant-on-Load (Fallback)

```rust
// grim-backend-rocm/src/kernels/mxfp8_wmma_dequant.rs

//! MXFP8 dequantization + BF16 WMMA fused kernel.
//! Loads FP8 codes + block scales, converts to BF16 in LDS, then issues WMMA.

pub const MXFP8_DEQUANT_WMMA_SOURCE: &str = r#"
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

__device__ inline float mxfp8_scale(const unsigned char* scales,
                                     int e, int row, int col,
                                     int num_e, int m, int n) {
    int global_row = e * m + row;
    int bm = global_row / 128;
    int bn = col / 128;
    int li = (global_row % 128) / 4;
    int lj = (col % 128) / 8;
    int idx = ((bm * (n / 128) + bn) * 32 + li) * 16 + lj;
    unsigned char exp = scales[idx];
    return powf(2.0f, (float)exp - 127.0f);
}

extern "C" __global__ void grim_mxfp8_wmma_gemm(
    const unsigned char* __restrict__ A_codes,
    const unsigned char* __restrict__ A_scales,
    const __hip_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int num_e, int m, int n)
{
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    if (tile_row * 16 >= M || tile_col * 16 >= N) return;

    fragment<matrix_a, 16, 16, 16, __hip_bfloat16, row_major> frag_a;
    fragment<matrix_b, 16, 16, 16, __hip_bfloat16, col_major> frag_b;
    fragment<accumulator, 16, 16, 16, float> frag_c;

    fill_fragment(frag_c, 0.0f);

    __shared__ __hip_bfloat16 s_a_bf16[16 * 16];

    for (int kk = 0; kk < K; kk += 16) {
        // Each wave lane dequantizes 8 FP8 values
        int lane = threadIdx.x % 32;
        int local_base = lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int local_k = local_base + i;
            int global_k = kk + local_k;
            int global_row = tile_row * 16 + local_k / 16;
            int global_col = tile_row * 16 + local_k % 16; // row-major tile
            float scale = mxfp8_scale(A_scales, 0, global_row, global_col, num_e, m, n);
            float val = fp8_e4m3_to_float_hip(A_codes[global_row * K + global_k]) * scale;
            s_a_bf16[local_k] = __float2bfloat16(val);
        }
        __syncthreads();

        load_matrix_sync(frag_a, s_a_bf16, 16);
        load_matrix_sync(frag_b, B + tile_col * 16 + kk * N, N);
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    __shared__ float c_tile[16 * 16];
    store_matrix_sync(c_tile, frag_c, 16, mem_row_major);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            C[(tile_row * 16 + i) * N + (tile_col * 16 + j)] = c_tile[i * 16 + j];
        }
    }
}
"#;
```

---

## 6. Inline P2P Communication (Replacing `comm_fuse.rs`)

Instead of a separate `grim_comm_fuse_p2p_epilogue` kernel, peer writes are inlined into the megakernel via an optional `peer_out` pointer.

### 6.1 Rust Peer Mapping

```rust
// grim-backend-rocm/src/kernels/charon_wmma.rs

/// Peer GPU memory mapping for inline P2P atomic accumulation.
/// Initialized once at model load via `hipDeviceEnablePeerAccess`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerMapping {
    /// Device pointer to peer's combine buffer.
    pub peer_out: *mut c_void,
    pub col_offset: usize,
    pub n_total: usize,
}

/// Workspace cache for symmetric peer buffers across an EP group.
/// Mirrors MoK's `get_workspace` caching semantics.
pub struct PeerWorkspace {
    pub mappings: Vec<PeerMapping>,
    pub ep_size: usize,
}

impl PeerWorkspace {
    pub fn new(ep_size: usize) -> Result<Self> {
        if !matches!(ep_size, 4 | 8 | 16 | 32 | 64) {
            return Err(Error::Backend(
                "PeerWorkspace: ep_size must be 4, 8, 16, 32, or 64".into()
            ));
        }
        Ok(Self {
            mappings: Vec::with_capacity(ep_size),
            ep_size,
        })
    }
}
```

### 6.2 Deprecating `comm_fuse.rs`

The standalone `grim_comm_fuse_p2p_epilogue` kernel and its host orchestrator (`comm_fuse_fan_in`) are no longer needed. The megakernel handles accumulation directly:

```rust
// In the launcher:
// If peer mapping exists, pass peer_out to the kernel.
// If local-only, pass nullptr for peer_out.
let peer_out_ptr = plan.peer.map(|p| p.peer_out).unwrap_or(std::ptr::null_mut());
```

---

## 7. Schedule-Once Token Routing

MoK builds a `schedule` once per layer and reuses it for forward and backward. grim currently rebuilds `RoutingAssignment` every launch.

### 7.1 Device-Resident Schedule Buffer (Rust)

```rust
// grim-backend-rocm/src/kernels/schedule.rs

//! Device-resident MoE schedule, built once per forward pass and cached.

use grim_tensor::device::DeviceBuffer;

/// A persistent device schedule for MoE dispatch/combine.
///
/// Built from `topk_all: [ep_size, num_local_tokens, topk]` once per layer.
/// Reused for both forward and backward passes.
pub struct MoESchedule {
    pub peer_rank: DeviceBuffer<i32>,
    pub peer_token_idx: DeviceBuffer<i32>,
    pub tokens_per_expert: DeviceBuffer<i32>,
    pub num_tokens: DeviceBuffer<i32>,
    pub capacity: usize,
}

/// Schedule cache keyed by (ep_group, num_tokens, topk, num_local_experts).
/// Mirrors MoK's `get_workspace` caching.
pub struct ScheduleCache {
    // HashMap key → MoESchedule
}

impl MoESchedule {
    /// `schedule_capacity` is sized to worst-case routing.
    /// MoK uses `schedule_capacity_multiplier` (default 0.5) to pad.
    pub fn build(
        topk_all: &[Vec<Vec<u32>>], // [ep_size][num_local_tokens][topk]
        num_local_experts: usize,
        capacity_multiplier: f32,
    ) -> Result<Self> {
        let ep_size = topk_all.len();
        let num_local_tokens = topk_all.first().map(|v| v.len()).unwrap_or(0);
        let topk = topk_all.first().and_then(|v| v.first().map(|v| v.len())).unwrap_or(0);
        let capacity = ((num_local_tokens * topk) as f32 * capacity_multiplier.max(0.5)) as usize;
        let capacity = ((capacity + 255) / 256) * 256; // align to 256

        // Flatten routing into peer_rank / peer_token_idx arrays
        let mut peer_rank = Vec::with_capacity(capacity);
        let mut peer_token_idx = Vec::with_capacity(capacity);
        // ... flattening logic

        Ok(Self {
            peer_rank: DeviceBuffer::from_vec(peer_rank)?,
            peer_token_idx: DeviceBuffer::from_vec(peer_token_idx)?,
            tokens_per_expert: DeviceBuffer::new(num_local_experts)?,
            num_tokens: DeviceBuffer::from_vec(vec![0i32])?,
            capacity,
        })
    }
}
```

### 7.2 Kernel Consumption

The WMMA kernel reads the schedule instead of separate `router_tokens` / `router_experts` arrays:

```rust
// In charon_wmma.rs KERNEL_SOURCE, replace router arrays with:
//
// const int schedule_slot = blockIdx.x * blockDim.x + threadIdx.x;
// if (schedule_slot >= num_tokens) return;
// int peer = schedule_peer_rank[schedule_slot];
// int tok  = schedule_peer_token_idx[schedule_slot];
```

This removes host-side flattening and enables a fixed grid sized to `schedule_capacity`.

---

## 8. Block-Scale MXFP8 Layout (MoK-Compatible)

Adopt MoK's scale tensor shape so grim can consume weights prepared by MoK's `mxfp8_quantize`:

```
weights_fp8:   float8_e4m3fn  [E, M, N]
weights_sc:    uint8          [E * M // 128, N // 128, 32, 16]
weights_fp8_t: float8_e4m3fn  [E, N, M]
weights_sc_t:  uint8          [E * N // 128, M // 128, 32, 16]
```

### 8.1 Rust Host Struct

```rust
// grim-backend-rocm/src/kernels/mxfp_weights.rs

//! MoK-compatible MXFP8 weight layout for grim.

use grim_tensor::device::DeviceBuffer;

/// MXFP8 expert weights with block-scale layout.
///
/// Scale tensor shape: `[E * M // 128, N // 128, 32, 16]`
/// Compatible with MoK's `ops.mxfp8_quantize` output.
pub struct Mxfp8ExpertWeights {
    pub codes: DeviceBuffer<u8>,
    pub scales: DeviceBuffer<u8>,
    pub shape: (usize, usize, usize), // (num_experts, rows, cols)
}

impl Mxfp8ExpertWeights {
    /// Compute flat scale index for element (e, row, col).
    pub fn scale_idx(&self, e: usize, row: usize, col: usize) -> usize {
        let (num_e, m, n) = self.shape;
        let global_row = e * m + row;
        let bm = global_row / 128;
        let bn = col / 128;
        let li = (global_row % 128) / 4;
        let lj = (col % 128) / 8;
        ((bm * (n / 128) + bn) * 32 + li) * 16 + lj
    }

    /// Validate shapes match MoK convention.
    pub fn validate(&self) -> Result<()> {
        let (e, m, n) = self.shape;
        if m % 128 != 0 || n % 128 != 0 {
            return Err(Error::Backend(
                "Mxfp8ExpertWeights: M and N must be divisible by 128".into()
            ));
        }
        let expected_scales = e * (m / 128) * (n / 128) * 32 * 16;
        if self.scales.len() != expected_scales {
            return Err(Error::Backend(format!(
                "Mxfp8ExpertWeights: scales len {} != expected {}",
                self.scales.len(), expected_scales
            )));
        }
        Ok(())
    }
}
```

### 8.2 Prequantization Op (Rust Host)

```rust
// grim-backend-rocm/src/ops/mxfp8_quantize.rs

//! Host-side MXFP8 prequantization, mirroring `mok::ops.mxfp8_quantize`.

use grim_tensor::Tensor;

/// Quantize a BF16 weight matrix to MXFP8.
///
/// Returns `(codes, scales)` compatible with `Mxfp8ExpertWeights`.
pub fn mxfp8_quantize(x: &Tensor, return_transposed: bool) -> Result<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)> {
    // x: BF16 [M, N] or [E, M, N]
    // codes: FP8 [M, N] or [E, M, N]
    // scales: U8 [M//128, N//128, 32, 16] or [E*M//128, N//128, 32, 16]
    todo!("HIP kernel dispatch for block-scale quantization")
}
```

---

## 9. Backward Pass Kernel (Training)

MoK provides fused backward kernels. grim currently lacks training kernels. A Wave32 WMMA backward kernel fuses six operations into one launch:

```rust
// grim-backend-rocm/src/kernels/charon_backward.rs

//! Fused Wave32 WMMA backward pass for MoE training.

pub const BACKWARD_KERNEL_SOURCE: &str = r#"
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

extern "C" __global__ void grim_moe_wmma_backward(
    const float* __restrict__ d_y,
    const __hip_bfloat16* __restrict__ gate,
    const __hip_bfloat16* __restrict__ up,
    const __hip_bfloat16* __restrict__ expert_gate_w,
    const __hip_bfloat16* __restrict__ expert_up_w,
    const __hip_bfloat16* __restrict__ expert_down_w,
    const unsigned int* __restrict__ schedule_peer_rank,
    const unsigned int* __restrict__ schedule_peer_token_idx,
    float* __restrict__ d_x,
    float* __restrict__ d_gate_w,
    float* __restrict__ d_up_w,
    float* __restrict__ d_down_w,
    int hidden, int inter, int num_tokens,
    float routed_scaling_factor)
{
    const unsigned long long slot =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= (unsigned long long)num_tokens) return;

    const unsigned int tok = schedule_peer_token_idx[slot];
    // ... d_y @ down^T → d_hidden via WMMA
    // ... d_hidden * up → d_gate (SiLU derivative in LDS)
    // ... d_hidden * gate → d_up
    // ... d_hidden @ gate_w^T + d_hidden @ up_w^T → d_x
    // ... outer products for d_gate_w, d_up_w, d_down_w
}
"#;
```

The `MoESchedule` from forward is reused for backward routing.

---

## 10. Implementation Checklist

| Task | File | Priority |
|---|---|---|
| Replace scalar loops with rocWMMA fragments | `charon_wmma.rs` | P0 |
| Add inline P2P `atomicAdd` peer writes | `charon_wmma.rs` | P0 |
| Implement MXFP8 block-scale layout + prequantization | `mxfp_weights.rs`, `mxfp8_quantize.rs` | P0 |
| Add FP8/BF16 WMMA path for gfx1200 | `fp8_wmma.rs` | P1 |
| Build device-resident `MoESchedule` cache | `schedule.rs` | P1 |
| Deprecate standalone `comm_fuse.rs` kernel | `comm_fuse.rs` | P2 |
| Fused backward WMMA kernel | `charon_backward.rs` | P2 |
| Wave cost model + device-side variant selector | `charon.rs` host logic | P3 |

---

## 11. Performance Targets

Based on MoK's reported speedups on Blackwell, grim's RDNA4 WMMA rewrite should target:

- **1.5–2.0×** over per-expert rocBLAS + separate `comm_fuse` kernel for BF16 forward.
- **1.3–1.6×** for MXFP8 forward (bandwidth-bound; dequant overhead applies if native FP8 WMMA is unavailable).
- **Zero host synchronization** per MoE layer: one kernel launch, one schedule build, cached workspace.

---

*Document generated for grim-backend-rocm RDNA4 rewrite. All kernel source is HIP embedded in Rust string literals; all host logic is Rust. Targets rocWMMA on gfx1200+ with Wave32 execution.*
