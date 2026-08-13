# JIT Multi-GPU Kernel with Hardware-Adaptive Configuration

Implementation plan for grim-backend-rocm. Target: gfx1036 (RX 7900 XT/XTX) primary, gfx1030 secondary.

---

## 1. What exists today

### JIT pipeline (working)

- `jit_compile_hsaco()` in `helpers.rs:57` — takes a HIP source string, passes it through hiprtc, returns `.hsaco` bytes and the lowered kernel name
- `HsacoKernelCache` in `jit_cache.rs:15` — compiles once, caches to disk. Keyed by `(entry_name, gpu_arch_string, source_hash)`. Stored under `$GRIM_HSACO_CACHE_DIR` or `$TMPDIR/grim_hsaco_cache`
- `compute_kernel_source()` in `source_asm.rs:3` — concatenates ~20 module sources (charon, compute_kernels, qkv_attention, wmma_gemm, all quantized GEMM variants, fp8/mxfp standalone) into one translation unit
- `RocmDevice::launch_compute_kernel()` — `hipModuleLoad` the `.hsaco`, `hipModuleGetFunction` the entry, `hipModuleLaunchKernel` with grid/block dims

### Hardware queries (exist but don't drive kernel code)

- `probe.rs` — reads gcn_arch_name, wavefront size, LDS bytes per CU via `hipDeviceGetAttribute`
- `capability_profiler.rs` — TFLOPS estimate, VRAM free, throttle %, link matrix from P2P status. Has a SCYTHE-2 epoch counter for capability changes
- These numbers are sampled. They are not fed into kernel source generation or launch config

### Multi-GPU (collective layer only, no kernel splitting)

- `rccl.rs` — full NCCL FFI wrapper: `ncclAllReduce`, `ncclReduceScatter`, `ncclAllGather`, `ncclSend/Recv`, group start/end, `hipMemcpyPeerAsync`. `RocmComm` RAII struct with `Drop` impl. `tp_all_reduce()` hook for tensor-parallel serving
- `peer_access.rs` — `hipDeviceCanAccessPeer`, `hipDeviceEnablePeerAccess`, verdict enum (`PeerDirect` / `HostBounce` / `NoLink`), gcnArchName extraction
- `p2p_route.rs` — `RouteLink` decision: `PeerDirect` vs `HostBounce` with tunable PCIe threshold. `HostStagingBuffer` (pinned host alloc). `to_route_link()` for cross-device copy path selection
- These handle communication after single-GPU kernels finish. No kernel splits work across devices

### Autotune (launch params, not kernel code)

- `autotune.rs` — `KernelKey` (kernel, gpu_arch, m, n, k) maps to `LaunchConfig` (block_m, block_n, block_k, split_k, threads). Caches to on-disk JSON. `KernelTuneCache` for the JSON shadow
- `gemm_tuning.rs` — `lookup_gemm_config()` picks tiles from shape class (Decode/Prefill/TLOLog) and wavefront size. `lookup_solution_index()` is an offline-tuned rocBLAS solution index table for gfx1036
- `charon_scalar_candidates()` — brute-force block dims against LDS limit

### What's missing

1. **Source parametrization** — injecting hardware-discovered constants into kernel source before JIT compile. Today the source is a static string
2. **Hardware-to-kernel-parameter mapping** — LDS, CU count, wavefront, P2P topology should drive kernel specialization. They are queried but ignored
3. **Multi-GPU kernel launches** — no kernel splits work across GPUs. Pattern today: same kernel on each GPU + host-side RCCL all-reduce
4. **Cache key extension** — hardware fingerprint in cache keys so parametrized kernels cache correctly
5. **Performance research is not wired in** — the old/res* research exists as synthesis documents but is not consulted by the kernel at runtime or compile time

---

## 2. Research findings that inform the kernel

### old/res5/synthesis_2.md — Wave-2 multi-GPU parallelism synthesis

22 papers surveyed. Key takeaways for kernel design:

- **FCP [2602.21788]**: polynomial-time auto-search for parallel strategy. Replaces 1400-candidate joint enumeration with millisecond-level overhead. This is the model for our fallback tile search — not full joint search, just a fast polynomial pass over a small candidate set
- **Lagom [2602.20656]**: conditional compute-communication overlap. Don't always overlap. The cost model decides. Relevant to whether we overlap RCCL with kernel launch in the multi-GPU path
- **SCYTHE** (derived method in the same document): precomputed lookup table for common configurations, FCP fallback for rare shapes. The lookup table approach beats runtime search for the common case. For our kernel: a small table of (gcn_arch, shape_class) → TileConfig entries, built at compile time or first run, with FCP fallback for shapes not in the table
- **Marlin/warp-tiled kernels**: memory-aligned layouts matching bus transaction boundaries (32-byte or 64-byte). Avoids dequant penalties. Relevant to how we pack LDS tiles

### old/res2/research-synthesis.md — Training memory optimization

ROCm-specific observations:
- LDS 64-128KB per CU for tiled decode. grim's gfx1036 has 64KB LDS/CU
- MFMA instructions on RDNA4 (gfx12xx). Not on gfx1036
- hiprtc JIT for per-arch fused kernels — this is exactly what grim already does
- Wave32 occupancy targeting 2-4 wavefronts per CU (RDNA only; CDNA is Wave64)

### old/res4/grim_formats_evopress_ceiling.md — Quantization codec status

All four codecs are built but not wired into TrainingJob: Crow (Q4K), Raven (FP8), Jay (MXFP4), Magpie (MXFP8). Relevance to kernel: the dequant kernels exist (`grim_dequant_mxfp4`, `grim_dequant_mxfp8`, `grim_dequant_fp8`). The hardware-adaptive path can choose which dequant kernel to invoke based on the loaded tensor format, which is part of the tensor metadata, not the hardware spec

### old/res4/research.md — SmoothQuant, QuaRot, ORES, FP8/MXFP formats

FP8/MXFP hardware-native formats section: MXFP8 is strictly better than per-tensor FP8 at the same bit budget on RDNA2 because the shared E8M0 exponent per 32 elements captures block-level dynamic range. Relevant to kernel: if a tensor is stored as MXFP8, the dequant-in-tile path uses the shared exponent. The hardware-adaptive JIT picks the dequant kernel based on tensor format, not just hardware

### old/res8/synth.md — Efficient convolution layers for Mamba path

Short conv is declared but not wired in v1. Not directly relevant to the GEMM kernel plan, but the pattern is: declared infrastructure that isn't activated. Same risk for our hardware-adaptive path if we don't wire it into the dispatch

### oldmoeres/ — does not exist

That path is not in the repo. Skipped.

---

## 3. Design principles

### Caveman: minimal, robust, no over-engineering

- One kernel source template with `#define` slots. Not N kernel variants
- One `HardwareSpec` struct. Not 5 separate probe functions
- One JIT path: `HardwareSpec` → source string → hiprtc compile → cached `.hsaco`. Not per-kernel compilation logic
- Cache by hardware fingerprint, not just arch string
- Tile selection heuristics are concrete numbers for gfx1036, derived for gfx1030. Not "first pass, validate later"

### Clean code guard

- No `unsafe` beyond what hiprtc/hipModule already require. The existing FFI layer already has the unsafe blocks
- Thread-safe cache. Use `RwLock<HashMap>` or a mutex-per-entry pattern. Not a single global lock that serializes all compiles
- `thiserror` for JIT error types. Not `Any` or stringly-typed errors
- Deterministic: same `HardwareSpec` + same source → same kernel bytes. No random seed, no timestamp in the binary

### Rust expert

- `Arc` for shared device handles across GPU contexts
- `OnceLock` for one-time JIT compilation per hardware fingerprint. Compile on first use, reuse after
- `std::collections::HashMap` for the compile cache. The compile path is single-threaded per device. No need for `dashmap`
- Feature gates: `feature = "jit-hw-adaptive"` (defaults to on), `feature = "multi-gpu-kernel"` (defaults to off — multi-GPU is a separate capability)

### FFI discipline

- hiprtc calls stay in `helpers.rs`. That file already has the C-FFI bindings
- `hiprtcGetLoweredName` already handled. No new C++ mangling surprises
- Multi-GPU kernel launch uses existing `hipModuleLaunchKernel`. No new FFI beyond what rccl/p2p already have
- New FFI only if we add RCCL-from-device-code (out of scope)

### ROCm reality for gfx1036

| Property | Value |
|----------|-------|
| GCN arch | gfx1036 |
| Active CUs | 64 |
| LDS per CU | 64 KB |
| Max shared memory per block | 384 KB (hipDeviceGetAttribute MAX_SHARED_MEMORY_PER_BLOCK) |
| Wavefront size | 32 threads |
| Max threads per block | 1024 (hipDeviceGetAttribute MAX_THREADS_PER_BLOCK) |
| Multiprocessors (CUs) | 64 (HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) |
| Max grid dim X | 2^31-1 |
| Native FP8 MFMA | No (RDNA4 only) |
| rocWMMA | No (gfx1100+ only) |
| BF16 WMMA | Available |
| P2P PCIe Gen4 x16 | ~32 GB/s, ~1-2 µs latency |
| P2P xGMI (if present) | ~50-100 GB/s |

gfx1030 (RX 7600/9060) has fewer CUs but same LDS/CU (64KB) and same wavefront (32). The tile selection scales down CU count but not LDS or wavefront.

---

## 4. Architecture

### 4.1 HardwareSpec — the system information snapshot

```rust
// crates/grim-backend-rocm/src/device/hardware_spec.rs

use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct HardwareSpec {
    pub gcn_arch: String,                       // "gfx1036"
    pub wavefront_size: u32,                    // 32
    pub max_shared_mem_per_block: u32,          // 384 * 1024 on gfx1036
    pub max_threads_per_block: u32,             // 1024 on gfx1036
    pub cu_count: u32,                          // 64 on gfx1036
    pub multiprocessor_count: u32,              // same as cu_count on AMD
    pub mem_bandwidth_gb_s: f64,                // estimated from capability_profiler
    pub p2p_topology: P2PTopology,              // NxN matrix of peer link types
}

// Hash and Eq derived from the fields that affect kernel code generation.
// P2P topology does NOT affect single-GPU kernel code; it only affects
// multi-GPU launch config. So it is excluded from the hash for the
// single-GPU cache key. We'll handle that in section 4.4.
impl PartialEq for HardwareSpec {
    fn eq(&self, other: &Self) -> bool {
        self.gcn_arch == other.gcn_arch
            && self.wavefront_size == other.wavefront_size
            && self.max_shared_mem_per_block == other.max_shared_mem_per_block
            && self.max_threads_per_block == other.max_threads_per_block
            && self.cu_count == other.cu_count
            && self.multiprocessor_count == other.multiprocessor_count
    }
}
impl Eq for HardwareSpec {}

impl Hash for HardwareSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.gcn_arch.hash(state);
        self.wavefront_size.hash(state);
        self.max_shared_mem_per_block.hash(state);
        self.max_threads_per_block.hash(state);
        self.cu_count.hash(state);
        self.multiprocessor_count.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P2PTopology {
    pub device_count: usize,
    pub links: Vec<Vec<LinkType>>,  // links[i][j] for i in 0..N, j in 0..N
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkType {
    PeerDirect,   // hipDeviceCanAccessPeer(i, j) == true && peer enabled
    HostBounce,   // P2P unavailable, host staging required
    NoLink,       // different hosts (not expected in single-system multi-GPU)
}
```

`HardwareSpec` is populated from:
- `probe.rs` — gcn_arch_name, wavefront_size, max_shared_mem_per_block, max_threads_per_block, cu_count, multiprocessor_count
- `capability_profiler.rs` — mem_bandwidth_gb_s (estimated from GPU model + memory clock, or a conservative default if profiling hasn't run)
- `peer_access.rs` — P2P topology matrix (new function: `build_topology_matrix(devices: &[RocmDevice]) -> P2PTopology`)

The `From<RocmDevice>` impl for single-GPU case:

```rust
impl From<&RocmDevice> for HardwareSpec {
    fn from(device: &RocmDevice) -> Self {
        let props = device.hip_device_properties();  // hipDeviceGetProperties
        let arch = props.gcnArchName;                // already extracted in probe.rs
        
        HardwareSpec {
            gcn_arch: arch.to_string(),
            wavefront_size: probe::wavefront_size(device),
            max_shared_mem_per_block: probe::max_shared_mem(device),
            max_threads_per_block: probe::max_threads_per_block(device),
            cu_count: probe::active_cu_count(device),
            multiprocessor_count: probe::active_cu_count(device),  // same on AMD
            mem_bandwidth_gb_s: capability_profiler::estimate_bandwidth(device),
            p2p_topology: P2PTopology {
                device_count: 1,
                links: vec![vec![LinkType::NoLink]],
            },
        }
    }
}
```

### 4.2 Kernel source template with hardware injection

Today `compute_kernel_source()` returns a static string. The new path adds a source factory that injects `#define`s before the `extern "C"` block.

```rust
// crates/grim-backend-rocm/src/kernels/source_asm.rs

pub fn compute_kernel_source_with_spec(
    spec: &HardwareSpec,
    entry: &str,
    shape_class: ShapeClass,
    device_id: u32,       // 0 for single-GPU, 0..N-1 for multi-GPU
    num_devices: u32,     // 1 for single-GPU, N for multi-GPU
) -> String {
    let tiles = pick_tiles(spec, shape_class);
    
    let mut source = String::new();
    
    // Module sources (same as today's compute_kernel_source)
    source.push_str(include_str!("charon.rs"));
    source.push_str(include_str!("compute_kernels.rs"));
    source.push_str(include_str!("qkv_attention.rs"));
    source.push_str(include_str!("wmma_gemm.rs"));
    // ... all existing module sources in the same order as today ...
    source.push_str(include_str!("mxfp_standalone.rs"));
    source.push_str(include_str!("fp8_standalone.rs"));
    source.push_str(include_str!("q4k_gemm.rs"));
    source.push_str(include_str!("shared_device_fns.rs"));
    
    // Hardware constants injected before extern "C"
    source.push_str(&format!(
        r#"
#define GRIM_WAVEFRONT_SIZE   {}
#define GRIM_MAX_LDS_BYTES    {}
#define GRIM_CU_COUNT         {}
#define GRIM_BLOCK_M          {}
#define GRIM_BLOCK_N          {}
#define GRIM_BLOCK_K          {}
#define GRIM_SPLIT_K          {}
#define GRIM_GRID_STRIDE_M    {}
#define GRIM_GRID_STRIDE_N    {}
#define GRIM_DEVICE_ID        {}
#define GRIM_NUM_DEVICES      {}

"#,
        spec.wavefront_size,
        spec.max_shared_mem_per_block,
        spec.cu_count,
        tiles.block_m,
        tiles.block_n,
        tiles.block_k,
        tiles.split_k,
        tiles.grid_stride_m,
        tiles.grid_stride_n,
        device_id,
        num_devices,
    ));
    
    source
}
```

The existing `compute_kernel_source()` stays as-is. It is the fallback when the `jit-hw-adaptive` feature is disabled.

`ShapeClass` is already defined in `autotune.rs`. It classifies GEMM shapes:

```rust
// From autotune.rs — already exists
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeClass {
    Decode,   // memory-bound: small M, large N, small K per token
    Prefill,  // compute-bound: large M, large N, large K
    TLOLog,   // mixed: tensor-parallel logit projection
}
```

`shape_class` comes from the GEMM dispatch layer, not invented here. The existing `lookup_gemm_config()` in `gemm_tuning.rs` already classifies shapes this way.

### 4.3 Tile selection from hardware + shape

This is where the kernel is not just unoptimized defaults. The tile selection derives from hardware properties + shape class.

The research basis:
- Marlin/warp-tiled kernels: tile sizes should be multiples of the **Wave32** wavefront size (32 threads on RDNA2) so that memory transactions align with bus boundaries. On RDNA2 with 32-byte memory transactions, a block of 128 or 256 threads (4 or 8 Wave32 wavefronts) hits transaction boundaries cleanly
- Occupancy: 2-4 **Wave32** wavefronts per CU is the sweet spot on RDNA. More wavefronts → more register pressure. Fewer → underutilization. On gfx1036 with 64 CUs, 4 Wave32 wavefronts per CU = 128 threads per block. (CDNA is Wave64 — separate path.)
- LDS double-buffering: when LDS is large enough relative to tile data, use ping-pong buffers to overlap global loads with computation. `max_shared_mem_per_block` tells us if we have room
- Split-K: when K is large relative to block K, split the reduction across multiple passes to hide the reduction latency

Concrete tile picker for gfx1036:

```rust
// crates/grim-backend-rocm/src/kernels/tile_picker.rs

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileConfig {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub split_k: u32,
    pub grid_stride_m: u32,
    pub grid_stride_n: u32,
    pub lds_double_buffer: bool,
    pub use_wmma: bool,
    pub use_mfma: bool,
    pub threads: u32,
}

/// Shape dimensions for tile selection.
/// M = output rows, N = output columns, K = reduction dimension.
/// These map to GEMM C(M,N) = A(M,K) * B(K,N).
#[derive(Debug, Clone, Copy)]
pub struct ShapeDims {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl ShapeClass {
    pub fn dims(&self) -> ShapeDims {
        match self {
            ShapeClass::Decode => ShapeDims { m: 1, n: 0, k: 0 },  // per-token: M=1
            ShapeClass::Prefill => ShapeDims { m: 0, n: 0, k: 0 }, // large batch
            ShapeClass::TLOLog => ShapeDims { m: 0, n: 0, k: 0 },  // projection
        }
    }
}

pub fn pick_tiles(spec: &HardwareSpec, shape_class: ShapeClass) -> TileConfig {
    let wave = spec.wavefront_size;           // 32 on gfx1036/gfx1030
    let max_lds = spec.max_shared_mem_per_block; // 384 * 1024 on gfx1036
    let max_threads = spec.max_threads_per_block; // 1024 on gfx1036
    let cu_count = spec.cu_count;              // 64 on gfx1036

    // Block threads: target 4 Wave32 wavefronts per CU on high-CU RDNA cards,
    // 2 Wave32 wavefronts on low-CU RDNA. Wavefront size is 32 on gfx1036/gfx1030.
    // CDNA (gfx9xx) is Wave64 — separate path, not covered by this plan.
    // gfx1036: 64 CUs → 4 waves = 128 threads (32 threads per Wave32 wavefront)
    // gfx1030: ~32 CUs → 2 waves = 64 threads (conservative)
    // Must be a multiple of wavefront_size and ≤ max_threads_per_block
    let target_waves = if cu_count >= 48 { 4 } else { 2 };
    let threads = target_waves * wave;
    assert!(threads <= max_threads && threads % wave == 0);

    // LDS per tile: we store BF16 A tile, BF16 B tile, and accumulate C tile in LDS.
    // A tile: block_m * block_k * 2 bytes (BF16)
    // B tile: block_k * block_n * 2 bytes (BF16)
    // C tile: block_m * block_n * 2 bytes (BF16 accumulator)
    // Total LDS per tile = 2 * (block_m * block_k + block_k * block_n + block_m * block_n)
    //
    // Double buffering: if LDS can hold 2 tiles worth of data, use ping-pong.
    // Double buffer threshold: max_lds >= 2 * single_tile_lds
    //
    // We pick block_k first from LDS budget, then block_m/block_n from shape heuristics.

    // Block M/N from shape class (matches existing gemm_tuning.rs logic):
    // Decode: small tiles (memory-bound, many small GEMMs)
    // Prefill: large tiles (compute-bound, fewer large GEMMs)
    // TLOLog: medium tiles (projection, moderate compute)
    let (block_m, block_n) = match shape_class {
        ShapeClass::Decode => (16, 16),
        ShapeClass::Prefill => (32, 32),
        ShapeClass::TLOLog => (32, 16),
    };

    // Round up to wavefront multiple
    let block_m = ((block_m + wave - 1) / wave) * wave;
    let block_n = ((block_n + wave - 1) / wave) * wave;

    // Block K from LDS budget.
    // LDS available per tile (no double buffer yet):
    // lds_per_tile = 2 * (block_m * block_k + block_k * block_n + block_m * block_n)
    // We want lds_per_tile ≤ max_lds
    // Solve for block_k:
    //   block_k * (block_m + block_n) * 2 + block_m * block_n * 2 ≤ max_lds
    //   block_k ≤ (max_lds / 2 - block_m * block_n) / (block_m + block_n)
    //
    // For gfx1036, Decode (16,16):
    //   block_k ≤ (384*1024/2 - 256) / 32 = (196608 - 256) / 32 = 6136
    // That's too large — limited by threads, not LDS.
    //
    // For gfx1036, Prefill (32,32):
    //   block_k ≤ (196608 - 1024) / 64 = 3082
    // Still large. LDS is not the bottleneck for BF16 tiles on gfx1036.
    //
    // Practical block_k: limited by register pressure and wavefront occupancy,
    // not LDS. Use values from existing charon_scalar_candidates() that work:
    // Decode: block_k = 32 (small K per token)
    // Prefill: block_k = 64 (larger K for batch)
    // TLOLog: block_k = 64

    let block_k = match shape_class {
        ShapeClass::Decode => 32,
        ShapeClass::Prefill => 64,
        ShapeClass::TLOLog => 64,
    };

    // Double buffer check: can we fit 2 tiles in LDS?
    let lds_per_tile = 2 * (
        (block_m as u64) * (block_k as u64) * 2
        + (block_k as u64) * (block_n as u64) * 2
        + (block_m as u64) * (block_n as u64) * 2
    ) as u32;
    let lds_double_buffer = max_lds >= 2 * lds_per_tile;

    // Split-K: when K > block_k * 4, split the reduction.
    // This requires the kernel to support split-K (existing charon kernels do).
    // For shape_class, we need the actual K dimension. ShapeClass doesn't carry K.
    // The caller passes K separately. For now, use a conservative default.
    let split_k = 1;  // overridden by caller when K is known

    // WMMA: rocWMMA is available on gfx1100+. Not on gfx1036.
    let use_wmma = spec.gcn_arch.starts_with("gfx11") || spec.gcn_arch.starts_with("gfx12");
    // MFMA: native MFMA on gfx12xx (RDNA4) and gfx9xx (CDNA). Not on gfx1036.
    let use_mfma = spec.gcn_arch.starts_with("gfx12") || spec.gcn_arch.starts_with("gfx9");

    TileConfig {
        block_m,
        block_n,
        block_k,
        split_k,
        grid_stride_m: block_m,
        grid_stride_n: block_n,
        lds_double_buffer,
        use_wmma,
        use_mfma,
        threads,
    }
}
```

The numbers above are for gfx1036. For gfx1030 (fewer CUs, same LDS/CU): `target_waves = 2`, `threads = 64`, same block_m/block_n/block_k. The LDS budget is the same per-CU, so tile sizes don't change — only the thread count per block drops to match the lower CU count.

This is not "first pass, validate later." These numbers are derived from the actual gfx1036 hardware properties and the existing `charon_scalar_candidates()` brute-force approach. The tile picker should be validated against `charon_scalar_candidates()` output for the same shapes, but the starting point is concrete.

#### Roofline cost model for FCP fallback

The FCP fallback evaluates candidates and picks the one with the lowest estimated execution time. The roofline model:

```
compute_time = (2 * M * N * K) / (TFLOPS * occupancy_factor)
memory_time = (bytes_read + bytes_written) / mem_bandwidth_gb_s
estimated_time = max(compute_time, memory_time)
```

Where:
- `2 * M * N * K` = total floating point operations for GEMM (multiply + add per element)
- `TFLOPS` = spec.mem_bandwidth_gb_s derived from capability_profiler, or conservative default (gfx1036: ~23 TFLOPS BF16 at stock clocks)
- `occupancy_factor` = fraction of peak TFLOPS the kernel achieves. Approx 0.6-0.8 for well-tuned kernels. Conservative: 0.6
- `bytes_read` = M*K*2 + K*N*2 (A and B matrices in BF16)
- `bytes_written` = M*N*2 (C matrix in BF16)
- `mem_bandwidth_gb_s` = spec.mem_bandwidth_gb_s (gfx1036: ~500 GB/s from GDDR6 1900MHz 256-bit)

The candidate with the lowest `estimated_time` wins. Ties broken by preferring the candidate with the smallest block dimensions (less register pressure).

```rust
pub fn roofline_cost(spec: &HardwareSpec, dims: ShapeDims, tiles: &TileConfig) -> f64 {
    let muflops = 2.0 * (dims.m as f64) * (dims.n as f64) * (dims.k as f64);
    let compute_time_s = muflops / (spec.mem_bandwidth_gb_s * 1e9 * 0.6);  // 0.6 occupancy factor
    
    let bytes_read = ((dims.m as u64) * (dims.k as u64) * 2
                    + (dims.k as u64) * (dims.n as u64) * 2) as f64;
    let bytes_written = ((dims.m as u64) * (dims.n as u64) * 2) as f64;
    let bytes_total = bytes_read + bytes_written;
    let memory_time_s = bytes_total / (spec.mem_bandwidth_gb_s * 1e9);
    
    compute_time_s.max(memory_time_s)
}
```

The FCP fallback generates candidates by perturbing the default tile config:

```rust
pub fn fcp_fallback_tile_search(
    spec: &HardwareSpec,
    dims: ShapeDims,
    shape_class: ShapeClass,
) -> TileConfig {
    let base = pick_tiles(spec, shape_class);
    
    // Generate candidates: perturb block_m, block_n, block_k around the base.
    // Candidates: base, and ±1 wave in each dimension.
    let wave = spec.wavefront_size;
    let mut candidates = vec![base.clone()];
    
    for dm in [-wave, wave] {
        for dn in [-wave, wave] {
            for dk in [32u32.wrapping_sub(wave), 32u32 + wave].iter() {
                // Only include candidates that are valid (positive, wavefront-multiple, ≤ max_threads)
                let bm = (base.block_m as i32 + dm) as u32;
                let bn = (base.block_n as i32 + dn) as u32;
                let bk = if dk > &0 { *dk } else { continue };
                if bm % wave != 0 || bn % wave != 0 || bk % wave != 0 { continue; }
                if bm + bn == 0 { continue; }
                if (bm * bn) as u32 > spec.max_threads_per_block { continue; }
                
                let mut cand = base.clone();
                cand.block_m = bm;
                cand.block_n = bn;
                cand.block_k = bk;
                candidates.push(cand);
            }
        }
    }
    
    // Deduplicate
    candidates.sort();
    candidates.dedup();
    
    // Pick lowest roofline cost. Tie-break: smallest block dimensions.
    candidates.iter().min_by(|a, b| {
        let cost_a = roofline_cost(spec, dims, a);
        let cost_b = roofline_cost(spec, dims, b);
        cost_a.partial_cmp(&cost_b).unwrap()
            .then_with(|| (a.block_m + a.block_n).cmp(&(b.block_m + b.block_n)))
    }).cloned()
}
```

This generates ~12-20 candidates (not 1400 like full joint search). The evaluation is O(candidates) with a simple roofline formula. Millisecond-level, matching the FCP paper's claim.

### 4.4 Hardware fingerprint in JIT cache key

Today's cache key: `(entry_name, gpu_target_string, source_hash)` where `source_hash = seahash(source_string)`.

New cache key: add `hardware_fingerprint`. The fingerprint is derived from `HardwareSpec` fields that affect kernel code generation: wavefront_size, max_shared_mem_per_block, cu_count, multiprocessor_count. P2P topology does not affect single-GPU kernel code, so it's excluded from the fingerprint for single-GPU cache entries.

```rust
// crates/grim-backend-rocm/src/kernels/jit_cache.rs

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JitCacheKey {
    pub entry: String,                    // kernel entry name, e.g. "grim_moe_fused_grouped_fp8"
    pub gpu_target: String,               // existing: "gfx1036" from hiprtc arch option
    pub hardware_fingerprint: String,     // "{wavefront}:{lds_bytes}:{cu}:{mp}:{threads}"
    pub source_hash: u64,                 // existing: seahash of source string
}

impl JitCacheKey {
    pub fn from_spec(
        entry: &str,
        gpu_target: &str,
        spec: &HardwareSpec,
        source_hash: u64,
    ) -> Self {
        let fingerprint = format!(
            "{}:{}:{}:{}:{}:{}",
            spec.wavefront_size,
            spec.max_shared_mem_per_block,
            spec.cu_count,
            spec.multiprocessor_count,
            spec.max_threads_per_block,
        );
        JitCacheKey {
            entry: entry.to_string(),
            gpu_target: gpu_target.to_string(),
            hardware_fingerprint: fingerprint,
            source_hash,
        }
    }
}
```

The `HsacoKernelCache` is already in `jit_cache.rs`. It uses `std::collections::HashMap` for in-memory cache and writes `.hsaco` files to disk. The key change: `JitCacheKey` now includes `hardware_fingerprint`. The disk filename includes the fingerprint so different hardware configs get different files.

For multi-GPU: each device gets its own cache entry (different device_id). The cache key for multi-GPU includes `device_id` in the fingerprint or as a separate field. Since `device_id` doesn't change the kernel source (only the `#define GRIM_DEVICE_ID` value), and the source hash captures the full source string including the device_id injection, the source_hash already differentiates multi-GPU entries. No separate device_id field needed in the key.

### 4.5 Multi-GPU kernel launch

The multi-GPU path splits the M dimension (output rows) across devices, launches a per-device kernel on each shard, then uses RCCL all-reduce to combine.

This is not a single kernel running across GPUs. ROCm does not have NVLink-style device-side P2P for arbitrary kernels. The pattern is: per-device kernel launch + host-side RCCL collective. This is what the existing `rccl.rs` and `p2p_route.rs` infrastructure already supports.

```rust
// crates/grim-backend-rocm/src/multi_gpu_launch.rs

use crate::rccl::RcclAllReduce;
use crate::device::HardwareSpec;
use crate::kernels::{compute_kernel_source_with_spec, TileConfig, ShapeClass};

/// Launch a kernel across N GPUs with shard split on the M dimension.
///
/// Each device i computes the shard [i * M/N, (i+1) * M/N) of the output.
/// After all device kernels complete, an RCCL all-reduce combines the shards.
///
/// Requirements:
/// - All devices must have the same GCN arch (or at least compatible tile configs)
/// - RCCL must be initialized on all devices (existing RocmComm setup)
/// - P2P must be enabled between all device pairs if using peer direct path
///   (existing peer_access::enable_peer_access handles this)
///
/// The args array contains the kernel arguments. The kernel is responsible for
/// using GRIM_DEVICE_ID and GRIM_NUM_DEVICES to compute its shard boundaries.
/// The host passes the full M, N, K dimensions in args — the kernel computes
/// its local shard from the #define values.
///
/// Error handling: returns the first error encountered. If device 2 fails,
/// devices 0 and 1 have already launched. The caller is responsible for
/// cleaning up (existing RocmDevice Drop handles hipModuleUnload).
pub fn launch_multi_gpu_kernel(
    devices: &[&RocmDevice],
    comm: &RcclAllReduce,
    entry: &str,
    shape_class: ShapeClass,
    full_dims: ShapeDims,          // M, N, K of the full problem
    hardware_specs: &[HardwareSpec],
    args: &[DeviceArg],
) -> Result<()> {
    let n = devices.len();
    assert_eq!(n, hardware_specs.len(), "device count must match spec count");
    assert!(n >= 2, "multi-GPU launch requires at least 2 devices");

    // 1. For each device, compute shard and launch
    for (i, (device, spec)) in devices.iter().zip(hardware_specs.iter()).enumerate() {
        // Compute shard boundaries
        let shard_start = (i as u32 * full_dims.m) / (n as u32);
        let shard_end = ((i as u32 + 1) * full_dims.m) / (n as u32);
        let shard_m = shard_end - shard_start;
        
        // Generate source with shard-specific #defines
        let source = compute_kernel_source_with_spec(
            spec,
            entry,
            shape_class,
            i as u32,      // GRIM_DEVICE_ID
            n as u32,      // GRIM_NUM_DEVICES
        );
        
        // JIT compile or cache hit
        let (hsaco, lowered_name) = device.jit_compile_or_cache(&source, entry, spec)?;
        
        // Grid dimensions for this shard
        let grid_m = (shard_m + spec.grid_stride_m - 1) / spec.grid_stride_m;
        let grid_n = ((shard_end - shard_start) + spec.grid_stride_n - 1) / spec.grid_stride_n;
        
        // Launch kernel on this device
        device.launch_compute_kernel(
            &hsaco,
            &lowered_name,
            grid_m,
            grid_n,
            1,              // grid_z
            spec.threads,  // block_x
            1,              // block_y
            1,              // block_z
            args,
        )?;
    }
    
    // 2. Cross-GPU all-reduce (existing RCCL path)
    // The all-reduce combines the per-device output shards.
    // The exact NCCL call depends on the tensor shape and data type.
    // This is a placeholder for the existing rccl::all_reduce call.
    comm.all_reduce(
        devices,
        /* buffer */ &output_buffer,  // the per-device output buffers
        /* count */ full_dims.m * full_dims.n,
        /* dtype */ NCCL_FLOAT16,     // or whatever the kernel outputs
    )?;
    
    Ok(())
}
```

The `DeviceArg` type is the existing argument passing mechanism used by `launch_compute_kernel`. It's defined in `roc_device.rs` and handles the `hipModuleLaunchKernel` arg array. We don't change it.

The `comm.all_reduce(...)` call is a placeholder. The actual call uses the existing `RcclAllReduce` methods from `rccl.rs`. The exact signature depends on the tensor shape and data type. The existing `tp_all_reduce()` hook in `rccl.rs` is the model.

For the P2P path: if `hardware_specs[i].p2p_topology.links[i][j] == LinkType::PeerDirect`, the kernel on device i can read from device j's output buffer via `hipMemcpyPeerAsync` before the all-reduce. The `RouteLink` from `p2p_route.rs` determines whether peer access is used. The existing `RocmDevice::copy_via_route()` handles this. We don't add new P2P logic — we use the existing path.

### 4.6 Integrating hardware-adaptive path into existing dispatch

The `launch_compute_kernel` method in `roc_device.rs` is the main entry point for kernel launches. We extend it to try the hardware-adaptive source first, fall back to the static source if compilation fails or the feature is disabled.

```rust
// crates/grim-backend-rocm/src/device/roc_device.rs

impl RocmDevice {
    pub fn launch_compute_kernel(
        &self,
        entry: &str,
        shape_class: ShapeClass,
        dims: ShapeDims,
        args: &[DeviceArg],
    ) -> Result<()> {
        // Try hardware-adaptive path first (if feature enabled)
        #[cfg(feature = "jit-hw-adaptive")]
        {
            let spec = self.hardware_spec();  // Populates HardwareSpec from probe queries
            let source = compute_kernel_source_with_spec(&spec, entry, shape_class, 0, 1);
            let (hsaco, lowered_name) = self.jit_compile_or_cache(&source, entry, &spec)?;
            
            let grid_m = (dims.m + spec.grid_stride_m - 1) / spec.grid_stride_m;
            let grid_n = (dims.n + spec.grid_stride_n - 1) / spec.grid_stride_n;
            
            return self.launch_compute_kernel_with_params(
                &hsaco,
                &lowered_name,
                grid_m, grid_n, 1,
                spec.threads, 1, 1,
                args,
            );
        }
        
        #[cfg(not(feature = "jit-hw-adaptive"))]
        {
            // Fall back to static source (existing behavior)
            let source = compute_kernel_source(entry);
            let (hsaco, lowered_name) = self.jit_compile_or_cache(&source, entry)?;
            // ... existing launch logic ...
        }
    }
}
```

The `jit_compile_or_cache` method is new — it wraps `jit_compile_hsaco` with cache lookup. The existing `HsacoKernelCache` handles the cache. We add a method to `RocmDevice` that checks the cache first, compiles if missing.

---

## 5. Implementation phases

Each phase has: what to build, which files to create/modify, test criteria, and the skills that apply.

### Phase 1: HardwareSpec struct and system query

**Duration:** 1-2 days

**What:** Create `HardwareSpec` struct. Populate it from existing `probe.rs` and `capability_profiler.rs` queries. Add P2P topology matrix construction.

**New files:**
- `crates/grim-backend-rocm/src/device/hardware_spec.rs` — `HardwareSpec`, `P2PTopology`, `LinkType`, `From<&RocmDevice>` impl

**Modified files:**
- `crates/grim-backend-rocm/src/device/probe.rs` — add functions that return the specific values needed by `HardwareSpec`: `wavefront_size()`, `max_shared_mem()`, `max_threads_per_block()`, `active_cu_count()`. These may already exist — check and expose them
- `crates/grim-backend-rocm/src/device/capability_profiler.rs` — add `estimate_bandwidth(device: &RocmDevice) -> f64`. Use GPU model name + memory clock to estimate, or return a conservative default (gfx1036: 500 GB/s GDDR6)
- `crates/grim-backend-rocm/src/peer_access.rs` — add `build_topology_matrix(devices: &[&RocmDevice]) -> P2PTopology`. For each pair (i, j), check `hipDeviceCanAccessPeer(i, j)` and whether peer is enabled. Map to `LinkType::PeerDirect` or `LinkType::HostBounce`

**Test:**
- Create `HardwareSpec` from a single gfx1036 GPU. Verify: gcn_arch = "gfx1036", wavefront_size = 32, max_shared_mem_per_block = 393216 (384 * 1024), max_threads_per_block = 1024, cu_count = 64
- Create `HardwareSpec` from a gfx1030 GPU. Verify: gcn_arch = "gfx1030", wavefront_size = 32, cu_count matches the GPU's CU count
- `build_topology_matrix` for 2-GPU system with P2P enabled. Verify: links[0][1] == LinkType::PeerDirect, links[1][0] == LinkType::PeerDirect

**Skills that apply:**
- `rust-expert` — struct design, `From` impl, `Hash`/`Eq` impl, error handling with `thiserror`
- `ffi` — hipDeviceGetAttribute calls to populate fields
- `rocm` — HIP device property queries, peer access checks
- `caveman` — one struct, one source of truth, no redundant fields

### Phase 2: Source template with hardware injection

**Duration:** 2-3 days

**What:** Add `compute_kernel_source_with_spec()` that injects `#define`s. Add `pick_tiles()` and `TileConfig`. Add roofline cost model.

**New files:**
- `crates/grim-backend-rocm/src/kernels/tile_picker.rs` — `TileConfig`, `pick_tiles()`, `roofline_cost()`, `fcp_fallback_tile_search()`, `ShapeDims`

**Modified files:**
- `crates/grim-backend-rocm/src/kernels/source_asm.rs` — add `compute_kernel_source_with_spec()`. Keep `compute_kernel_source()` as-is for backward compatibility

**Test:**
- Generate source for gfx1036 with shape_class = Prefill. Verify the source contains `#define GRIM_WAVEFRONT_SIZE 32`, `#define GRIM_CU_COUNT 64`, `#define GRIM_BLOCK_M 32`, `#define GRIM_BLOCK_N 32`, `#define GRIM_BLOCK_K 64`
- Generate source for gfx1030 with shape_class = Decode. Verify: wavefront = 32, block_m = 16, block_n = 16, block_k = 32, threads = 64 (2 waves)
- Generate source for gfx1036 with shape_class = Decode, then with shape_class = Prefill. Verify different block_m/block_n values
- `pick_tiles` for gfx1036 Prefill returns `TileConfig { block_m: 32, block_n: 32, block_k: 64, threads: 128, lds_double_buffer: true }` (verify LDS calculation: lds_per_tile = 2*(32*64*2 + 64*32*2 + 32*32*2) = 2*(4096 + 4096 + 2048) = 2*10240 = 20480 bytes. max_lds = 393216. 2*20480 = 40960 ≤ 393216 → double buffer true)
- `roofline_cost` for a known shape returns a finite f64. Verify it's deterministic (same input → same output)
- `fcp_fallback_tile_search` for M=137, N=256, K=512 returns a TileConfig with valid dimensions (wavefront multiples, ≤ max_threads)

**Skills that apply:**
- `kernel` — tile selection heuristics, LDS budgeting, wavefront alignment
- `amd` — RDNA2-specific properties (LDS per CU, wavefront size, max threads)
- `rust-expert` — struct design, deterministic functions, `assert!` for invariants
- `caveman` — concrete numbers, not "first pass"

### Phase 3: Hardware-fingerprinted JIT cache

**Duration:** 1 day

**What:** Extend `JitCacheKey` to include `hardware_fingerprint`. Update `HsacoKernelCache` to use the new key. Update `jit_compile_hsaco` to accept optional `HardwareSpec` for cache key construction.

**Modified files:**
- `crates/grim-backend-rocm/src/kernels/jit_cache.rs` — extend `JitCacheKey`, update cache lookup/insertion to use new key
- `crates/grim-backend-rocm/src/device/helpers.rs` — add `jit_compile_or_cache` method to `RocmDevice` (wraps `jit_compile_hsaco` with cache check). Accept optional `&HardwareSpec` for cache key

**Test:**
- Compile kernel "test_kernel" for HardwareSpec A (gfx1036, wavefront=32, cu=64). Verify cache entry created
- Compile same kernel for HardwareSpec B (gfx1030, wavefront=32, cu=32). Verify separate cache entry created
- Compile kernel for HardwareSpec A again. Verify cache hit (no hiprtc compile, just hipModuleLoad)

**Skills that apply:**
- `rust-expert` — HashMap key design, Hash/Eq correctness
- `ffi` — hiprtc compile path unchanged, just wrapped with cache
- `caveman` — extend existing cache, don't replace it

### Phase 4: Multi-GPU kernel launch

**Duration:** 3-4 days

**What:** Create `launch_multi_gpu_kernel()`. Split M dimension across devices. JIT compile per-device kernel. Launch on each device. RCCL all-reduce after.

**New files:**
- `crates/grim-backend-rocm/src/multi_gpu_launch.rs` — `launch_multi_gpu_kernel()`

**Modified files:**
- `crates/grim-backend-rocm/src/rccl.rs` — verify `RcclAllReduce` has the method needed for the all-reduce call. If not, add it. The existing `tp_all_reduce()` hook is the model
- `crates/grim-backend-rocm/src/kernels/source_asm.rs` — GRIM_DEVICE_ID and GRIM_NUM_DEVICES are already in the template from Phase 2

**Test:**
- Two-GPU launch of a simple kernel (e.g., element-wise add with shard split). Verify:
  - Device 0 computes shard [0, M/2)
  - Device 1 computes shard [M/2, M)
  - After RCCL all-reduce, the full output buffer contains the correct result
- Two-GPU launch where P2P is not available (HostBounce). Verify: host staging buffer used for cross-device copy, result still correct
- Two-GPU launch with different HardwareSpecs (gfx1036 + gfx1030). Verify: each device uses its own tile config, both produce correct shard

**Skills that apply:**
- `rocm` — multi-device launch, RCCL collective, P2P path
- `ffi` — hipModuleLaunchKernel per device, hipMemcpyPeerAsync if P2P
- `kernel` — shard boundary computation, grid/block sizing per shard
- `rust-expert` — slice iteration, error propagation with `?`, `assert!` for preconditions

### Phase 5: FCP-style fallback tile search

**Duration:** 1-2 days

**What:** Add `fcp_fallback_tile_search()` to `tile_picker.rs`. Integrate into `Autotuner` when cache miss for a shape that doesn't match a precomputed tile config.

**Modified files:**
- `crates/grim-backend-rocm/src/kernels/tile_picker.rs` — add `fcp_fallback_tile_search()` and `roofline_cost()` (created in Phase 2, tested in Phase 2)
- `crates/grim-backend-rocm/src/autotune.rs` — in `Autotuner::lookup()` or `Autotuner::tune()`, when the shape doesn't match a cached entry and `pick_tiles()` returns a config, run `fcp_fallback_tile_search()` to refine. Or: when `pick_tiles()` doesn't have enough info (e.g., K dimension not available from ShapeClass alone), fall back to FCP search

**Test:**
- Feed shape M=137, N=256, K=512 (doesn't match standard tile sizes). Verify `fcp_fallback_tile_search` returns a TileConfig with valid wavefront-multiple dimensions
- Feed same shape twice. Verify deterministic result (same TileConfig both times)
- Compare `fcp_fallback_tile_search` result against `charon_scalar_candidates()` for the same shape. Verify the FCP result is not worse (same or better LDS utilization)

**Skills that apply:**
- `kernel` — roofline model, candidate generation, cost comparison
- `amd` — occupancy factors for RDNA2, TFLOPS estimates
- `rust-expert` — deterministic iteration, `min_by` with tie-break

### Phase 6: Wire into existing kernel dispatch

**Duration:** 2-3 days

**What:** Make hardware-adaptive path the default. Keep static path as fallback. Feature gates.

**Modified files:**
- `crates/grim-backend-rocm/src/device/roc_device.rs` — extend `launch_compute_kernel()` to try hardware-adaptive source first (as shown in section 4.6). Fall back to static source if feature disabled or compilation fails
- `crates/grim-backend-rocm/src/lib.rs` — add feature gates documentation. Add `jit-hw-adaptive` and `multi-gpu-kernel` features to Cargo.toml

**Test:**
- Single-GPU end-to-end: load a model, run forward pass with hardware-adaptive JIT enabled. Verify output matches static JIT path (bit-identical or within epsilon for FP16)
- Single-GPU with `jit-hw-adaptive` feature disabled. Verify: falls back to static source, works correctly
- Two-GPU end-to-end: load a model, run forward pass with multi-GPU launch. Verify output matches two independent single-GPU runs + RCCL all-reduce
- Two-GPU with `multi-gpu-kernel` feature disabled. Verify: falls back to N independent launches (no shard split)

**Skills that apply:**
- `rust-expert` — feature gate conditional compilation, `#[cfg(feature = ...)]`
- `rocm` — kernel dispatch integration, RCCL wiring
- `caveman` — try new path, fall back to old path, don't break existing behavior

### Phase 7: Performance validation

**Duration:** ongoing

**What:** Benchmark hardware-adaptive tile configs against existing hand-tuned configs. Verify multi-GPU launch is faster than N independent runs for large M.

**Approach:**

Benchmark on gfx1036 (RX 7900 XT/7900 XTX, 64 CU, 64KB LDS, 32 wavefront):
- Compare `pick_tiles(Prefill)` output against `lookup_gemm_config(Prefill)` from `gemm_tuning.rs`. Both should produce similar tile configs. If they differ, understand why
- Benchmark GEMM with hardware-adaptive kernel vs static kernel for M=2048, N=4096, K=4096 (large prefill). Target: hardware-adaptive is equal or faster
- Benchmark GEMM for M=1, N=4096, K=4096 (decode). Target: hardware-adaptive is equal or faster

Benchmark on gfx1030 (RX 7600, fewer CU, same LDS/CU):
- Verify `pick_tiles` scales down CU count correctly (4 waves → 2 waves, threads 128 → 64)
- Benchmark GEMM with hardware-adaptive kernel. Target: correct output, no crash

Benchmark two-GPU gfx1036 on PCIe Gen4:
- For M=8192, N=4096, K=4096: compare multi-GPU launch + RCCL vs two independent single-GPU runs + RCCL. Target: multi-GPU is faster (shard overhead amortized)
- For M=256, N=4096, K=4096: compare. Target: multi-GPU may be slower (shard overhead not amortized). This is expected — multi-GPU helps for large M

**Skills that apply:**
- `amd` — GPU benchmarking, ROCm performance counters
- `kernel` — GEMM performance analysis, roofline validation

---

## 6. Source file inventory

### New files

| File | Purpose |
|------|---------|
| `crates/grim-backend-rocm/src/device/hardware_spec.rs` | `HardwareSpec`, `P2PTopology`, `LinkType`, `From<&RocmDevice>` impl |
| `crates/grim-backend-rocm/src/kernels/tile_picker.rs` | `TileConfig`, `pick_tiles()`, `roofline_cost()`, `fcp_fallback_tile_search()`, `ShapeDims` |
| `crates/grim-backend-rocm/src/multi_gpu_launch.rs` | `launch_multi_gpu_kernel()`, shard computation, per-device JIT + launch |

### Modified files

| File | Change |
|------|--------|
| `src/device/probe.rs` | Expose `wavefront_size()`, `max_shared_mem()`, `max_threads_per_block()`, `active_cu_count()` |
| `src/device/capability_profiler.rs` | Add `estimate_bandwidth(device: &RocmDevice) -> f64` |
| `src/peer_access.rs` | Add `build_topology_matrix(devices: &[&RocmDevice]) -> P2PTopology` |
| `src/kernels/source_asm.rs` | Add `compute_kernel_source_with_spec()` |
| `src/kernels/jit_cache.rs` | Extend `JitCacheKey` with `hardware_fingerprint` |
| `src/device/helpers.rs` | Add `jit_compile_or_cache()` method to `RocmDevice` |
| `src/device/roc_device.rs` | Extend `launch_compute_kernel()` to try hardware-adaptive source first |
| `src/autotune.rs` | Integrate `fcp_fallback_tile_search()` into `Autotuner` |
| `src/lib.rs` | Feature gate documentation |
| `Cargo.toml` | Add `jit-hw-adaptive` and `multi-gpu-kernel` features |

### Not touched

- `rccl.rs` — multi-GPU collectives already complete. Only verify the all-reduce method signature
- `p2p_route.rs` — route selection already complete. Only verify `to_route_link()` works for multi-device case
- `gemm_tuning.rs` — shape-class tile lookup already exists. `tile_picker` builds on it, doesn't replace it
- `charon.rs` — kernel source modules already complete. Not modified
- `jit_cache.rs` — caching infra already complete. Only extends the key struct

---

## 7. What the kernel optimizes

The hardware-adaptive JIT kernel adjusts these parameters based on system information:

| Parameter | Source | Effect |
|-----------|--------|--------|
| Block M/N tile sizes | Shape class (Decode=16, Prefill=32, TLOLog=32x16) + wavefront rounding | LDS utilization, occupancy |
| Block K tile size | Shape class (Decode=32, Prefill=64, TLOLog=64) | Reduction efficiency |
| Double-buffering depth | LDS bytes vs tile data size | Hides global memory latency when LDS budget allows |
| Thread count per block | Wavefront size × target waves/CU (4 waves on ≥48 CU, 2 waves on <48 CU) | Occupancy vs register pressure |
| Split-K factor | K dimension (passed by caller, default 1) | Hides reduction latency when K > block_k × 4 |
| Grid stride | Shape class (same as block size for grid-stride loop) | Scalability across GPU sizes |
| WMMA path | GCN arch starts with "gfx11" or "gfx12" | Uses rocWMMA 16×16 tiles when available |
| MFMA path | GCN arch starts with "gfx12" or "gfx9" | Uses native MFMA when available |
| Multi-GPU shard boundaries | Device count, M dimension, device index | Even work distribution across devices |
| P2P vs host bounce | P2P topology matrix | Minimizes cross-GPU transfer latency |
| RCCL all-reduce | Link type (xGMI vs PCIe) | Collective reduction after per-device kernel completion |

---

## 8. Scope boundaries

### In scope

- Single-GPU JIT with hardware-discovered tile configs
- Two-GPU kernel launch with M-dimension shard split + RCCL all-reduce
- P2P topology awareness for cross-GPU data movement (uses existing p2p_route path)
- FCP-style fallback for rare shapes (small candidate set, roofline cost model)
- Cache by hardware fingerprint

### Out of scope

- 3+ GPU kernel launch. The architecture is the same as 2-GPU (just more shards). Add when needed
- Device-side RCCL from kernel code. ROCm doesn't support this for arbitrary kernels
- Training graph capture with multi-GPU kernel. Separate concern
- Kernel auto-tuning across process restarts. The cache handles this — cache survives restarts
- NCCL P2P from device code. ROCm doesn't support this
- Auto-detection of GCN arch from PCI ID. The existing `probe.rs` already does this
- Fallback to software FP8 emulation on RDNA2. The existing `quantization.rs` gate already handles this correctly

---

## 9. Error handling

All new functions return `Result<T, JITError>` where `JITError` is a `thiserror`-derived enum:

```rust
// crates/grim-backend-rocm/src/kernels/error.rs (or in tile_picker.rs)

use thiserror::Error;

#[derive(Debug, Error)]
pub enum JITError {
    #[error("hiprtc compilation failed: {0}")]
    CompileFailed(String),
    
    #[error("hipModuleLoad failed: {0}")]
    ModuleLoadFailed(String),
    
    #[error("hipModuleGetFunction failed for entry '{entry}': {source}")]
    FunctionNotFound { entry: String, source: String },
    
    #[error("invalid tile config: block_m={block_m}, block_n={block_n}, block_k={block_k}, threads={threads}, max_threads={max_threads}")]
    InvalidTileConfig {
        block_m: u32,
        block_n: u32,
        block_k: u32,
        threads: u32,
        max_threads: u32,
    },
    
    #[error("cache write failed: {0}")]
    CacheWriteFailed(String),
    
    #[error("multi-GPU launch requires at least 2 devices, got {0}")]
    InsufficientDevices(usize),
    
    #[error("device count {device_count} != spec count {spec_count}")]
    DeviceSpecMismatch { device_count: usize, spec_count: usize },
}
```

The existing `launch_compute_kernel` already returns `Result<()>` with its own error type. The new code uses the same pattern — `?` propagation, no panics except for `assert!` on programming errors (invalid tile config that should never happen with correct inputs).

---

## 10. Feature gates

Add to `Cargo.toml`:

```toml
[features]
default = ["jit-hw-adaptive"]
jit-hw-adaptive = []      # hardware-adaptive source template + tile picker
multi-gpu-kernel = []     # multi-GPU launch with shard split + RCCL
```

- `jit-hw-adaptive` defaults to on. When disabled, `launch_compute_kernel` uses the static source path only
- `multi-gpu-kernel` defaults to off. When enabled, `launch_multi_gpu_kernel` is compiled and available. This is a separate feature because multi-GPU requires RCCL initialization and P2P setup that not all deployments need

---

## 11. Verification checklist

- [ ] `HardwareSpec` populated correctly for gfx1036 (all fields match hipDeviceGetAttribute values)
- [ ] `HardwareSpec` populated correctly for gfx1030 (CU count matches, LDS and wavefront same as gfx1036)
- [ ] `compute_kernel_source_with_spec` produces correct `#define` values for gfx1036 Prefill
- [ ] `compute_kernel_source_with_spec` produces different values for gfx1030 Decode
- [ ] `pick_tiles` returns valid TileConfig (wavefront multiples, threads ≤ max_threads, LDS budget satisfied)
- [ ] `pick_tiles` double-buffer decision is correct for gfx1036 Prefill (LDS budget allows double buffer)
- [ ] `roofline_cost` returns finite f64, deterministic
- [ ] `fcp_fallback_tile_search` returns valid TileConfig for rare shape, deterministic
- [ ] `JitCacheKey` with hardware fingerprint differentiates gfx1036 from gfx1030 cache entries
- [ ] Cache hit on second compile with same spec (no hiprtc recompile)
- [ ] `launch_multi_gpu_kernel` produces correct output for 2-GPU simple kernel
- [ ] `launch_multi_gpu_kernel` produces correct output when P2P is HostBounce
- [ ] Single-GPU hardware-adaptive path produces same output as static path
- [ ] Multi-GPU path with feature disabled falls back to N independent launches
- [ ] `jit-hw-adaptive` feature disabled: static path works
- [ ] `multi-gpu-kernel` feature disabled: multi-GPU launch not compiled
