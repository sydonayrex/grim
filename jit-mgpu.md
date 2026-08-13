# JIT Multi-GPU Kernel with Hardware-Adaptive Configuration

Implementation plan for grim-backend-rocm. Target: gfx1036 (RX 7900 XT/XTX) primary, gfx1030 secondary.

> **UPDATE (patched against current `crates.zip` source):** Two factual inaccuracies in the original
> draft's "what exists today" claims were found and corrected in place, marked inline as **PATCH NOTE**
> blocks:
> 1. `ShapeClass::TLOLog` was described as an existing variant of `autotune.rs`'s `ShapeClass`. It does
>    not exist — real `ShapeClass` has only `Decode`/`Prefill`. Added as explicit new-work item #0 in
>    "What's missing," rather than assumed pre-existing.
> 2. `capability_profiler::estimate_bandwidth(device)` was cited as an existing function. It does not
>    exist under that name. The underlying data (`hbm_bandwidth_gbps`) already exists on `GpuCapability`
>    via `CapabilityProfiler::capabilities()` — the fix is to read that field, not add a new function.
>
> Everything else in this document was spot-checked against source (`jit_compile_hsaco`, `HsacoKernelCache`,
> `fusion.rs`'s `block_dim_x` ternary, `WavefrontTiledLayout`/`PackedQuantLayout` in `layout.rs`, the
> MFMA/WMMA arch-gating logic) and confirmed accurate — no changes made to those sections.

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
- `gemm_tuning.rs` — `lookup_gemm_config()` picks tiles from shape class and wavefront size. `lookup_solution_index()` is an offline-tuned rocBLAS solution index table for gfx1036

> **PATCH NOTE (verified against source):** `autotune.rs`'s real `ShapeClass` currently has only two variants — `Decode` (m==1) and `Prefill` (m>1). `TLOLog` does **not** exist in source today; every reference to it below is this plan's proposal to add it as a genuine third variant (the `lm_head` / logit-projection GEMM — a real, distinct shape pattern, not an existing thing being described). Item 0 in "What's missing" below reflects this. The classifier is **op-identity** (`ShapeClass::from_op(GemmOp, m)`, `GemmOp::LmHead → TLOLog`) — not the earlier draft's `from_m`-only rule, which cannot fire for lm_head (M alone can't tell it from an attention/ffn GEMM of equal m). Treat every `TLOLog` arm + `GemmOp`/`from_op` as new code to write, not code already present in `autotune.rs`.
- `charon_scalar_candidates()` — brute-force block dims against LDS limit

### What's missing

0. **`ShapeClass::TLOLog` variant** — `autotune.rs`'s `ShapeClass` enum currently has only `Decode`/`Prefill`. This plan adds a third variant for the `lm_head` / logit-projection GEMM. It is **new work, not existing infrastructure** — reviewers should not look for it in current `autotune.rs`. The addition is: (a) the `TLOLog` enum arm; (b) a `GemmOp` enum + `ShapeClass::from_op(op, m)` **op-identity** classifier (the `from_m`-only rule cannot distinguish lm_head from an attention/ffn GEMM of equal m, so the class MUST be tagged at the dispatch layer — see the §4.2 "Why TLOLog is a separate bucket" section); (c) a `lookup_gemm_config` + `pick_tiles` arm with the distinct TLOLog tile `(16, 64)`, block_k 64. The tile is justified there: N=vocab is the dominant wide dim, K=hidden is reused across it, so the optimal tile is *small block_m / wide block_n* — the inverse of the Decode/Prefill square tiling.
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

### Research verification against codebase (old/res5)

Each research claim from `old/res5/synthesis_2.md` verified against `grim-backend-rocm/src`:

| Research claim | Paper ID | In codebase? | Where / status |
|---------------|----------|-------------|-----------------|
| FCP polynomial-time auto-search, millisecond-level overhead | 2602.21788 | **No** | No references to FCP, polynomial-time search, or 2602.21788 found in any source file. The `autotune.rs` brute-force search and `charon.rs` cost model are related but do not implement FCP's polynomial-time approach. Plan Phase 5 (FCP fallback tile search) is the first integration. |
| Lagom conditional compute-communication overlap, cost model decides | 2602.20656 | **No** | No references to Lagom, 2602.20656, or conditional overlap in any source file. No overlap decision logic exists. Noted as future consideration for multi-GPU RCCL overlap. |
| Marlin/warp-tiled memory-aligned layouts matching bus transaction boundaries | — | **Partially** | `WavefrontTiledLayout` (`layout.rs`) is a wavefront-tiled layout but simpler than Marlin's micro-kernel. `PackedQuantLayout` (`layout.rs:188`) handles Wave64-aligned row segments for quantized data. The plan's tile picker with wavefront-multiple sizes is the first Marlin-style alignment for compute kernels. Old/mockdud.md explicitly identifies the gap: "GrimLayoutHint::WavefrontTiled exists but simpler." |
| SCYTHE precomputed lookup table + FCP fallback | (derived in synthesis_2.md) | **Partially** | SCYTHE-2 infrastructure is in the codebase: `CapabilityProfiler` (WI-2, `capability_profiler.rs`), `estimate_gemm_latency_ms` (WI-1/WI-6, `roc_device.rs:3900`), `comm_fuse_reduce` (WI-6, `roc_device.rs:3942`), `comm_fuse.rs` (WI-6 kernel). BUT the C²PLR controller (per-layer-per-shape routing) is NOT in `grim-backend-rocm` — it's referenced as living in `grim_engine::scythe2` via `#[see: ...]` doc comments. The backend-rocm crate has the profiling, latency estimation, and comm_fuse infra; the controller that uses them is in a different crate. |
| SCYTHE fills the gap: tree-drafting + disaggregation + topology-awareness + auto-search combined | (derived) | **Yes (claimed by SCYTHE-2)** | `launch_tree_attention` (roc_device.rs:4109) = tree drafting. `grim-disagg::DisaggRouter` (referenced in scythe2.md) = disaggregation. `link_matrix()` (capability_profiler.rs:107) + `ScytheLink` = topology-awareness. `estimate_gemm_latency_ms` + SCYTHE-2 controller = auto-search. Per `old/scythe2.md`: "Fuses FCP's speed with TriRoute's granularity." |

**Bottom line**: None of the research from old/res5 is fully wired into `grim-backend-rocm` today. SCYTHE-2 has the most integration (profiling, latency estimation, comm_fuse), but the C²PLR controller that ties it together lives in `grim-engine`. FCP and Lagom are not in the codebase at all — the plan's Phase 5 is the first integration of FCP concepts.

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
            // PATCH NOTE (verified against source): `capability_profiler::estimate_bandwidth`
            // does not exist. The real data lives in `GpuCapability.hbm_bandwidth_gbps`,
            // populated by `arch_tflops_table()` and retrieved via `CapabilityProfiler::capabilities()`.
            // Use that instead of inventing a new function:
            mem_bandwidth_gb_s: CapabilityProfiler::new()
                .capabilities()
                .first()
                .map(|cap| cap.hbm_bandwidth_gbps as f64)
                .unwrap_or(500.0), // conservative gfx1036 GDDR6 default if no GPU capability reported yet
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
    dims: ShapeDims,      // M,N,K of the problem — drives K-derived split-K (see #4)
    device_id: u32,       // 0 for single-GPU, 0..N-1 for multi-GPU
    num_devices: u32,     // 1 for single-GPU, N for multi-GPU
    tiles: Option<&TileConfig>,  // None -> pick_tiles(); Some(c) -> force a config (FCP search)
) -> String {
    let tiles = match tiles {
        Some(t) => t.clone(),
        None => pick_tiles(spec, shape_class, dims),
    };
    
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

`ShapeClass` is defined in `autotune.rs` (real `Decode`/`Prefill` today; this plan adds `TLOLog` — see "What's missing" #0 and the classifier below). It classifies GEMM shapes. `shape_class` is produced by the GEMM dispatch layer, not invented here:

```rust
// autotune.rs — real enum after this plan's #0 addition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShapeClass {
    Decode,   // m == 1: per-token GEMM (attention, ffn, lm_head during decode)
    Prefill,  // m > 1:  large-batch GEMM (attention, ffn, lm_head during prefill)
    TLOLog,   // lm_head / logit-projection ONLY — tagged by op-identity, NOT by m (see from_op)
}

/// Which GEMM an op is, known at the dispatch layer (the engine launches lm_head as a
/// matmul with weight [vocab, hidden]). M alone cannot distinguish lm_head from an
/// attention/ffn GEMM of the same m, so the class is tagged op-identity here and passed
/// into `matmul` (roc_device.rs:1488), which forwards it to `lookup_gemm_config` /
/// `pick_tiles`. This is the classifier for TLOLog — it fires for every lm_head GEMM
/// regardless of m (decode: m==1; prefill: m==steps), which `from_m` can never do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmOp {
    Attention,
    Ffn,
    LmHead,    // -> ShapeClass::TLOLog
    Other,
}

impl ShapeClass {
    /// Backward-compatible: used by attention/ffn callers (m is sufficient to bin them).
    pub fn from_m(m: usize) -> Self {
        if m == 1 { Self::Decode } else { Self::Prefill }
    }
    /// Op-aware classifier. LmHead is TLOLog no matter its m; everything else bins by m.
    pub fn from_op(op: GemmOp, m: usize) -> Self {
        match op {
            GemmOp::LmHead => Self::TLOLog,
            _ => Self::from_m(m),
        }
    }
}
```

`lookup_gemm_config()` in `gemm_tuning.rs` gains a `shape: ShapeClass` parameter (or a `from_op`-derived class) so the TLOLog tile is selected for lm_head. The existing `from_m`-only callers keep working via `ShapeClass::from_m`.

#### Why TLOLog is a separate bucket (justification)

The `lm_head` GEMM is `C[steps, vocab] = X[steps, hidden] · W[vocab, hidden]ᵀ`. Its shape profile is **structurally different** from any attention/ffn GEMM, so a shared Decode/Prefill tile is sub-optimal:

- **N = vocab is the dominant, widest dimension** (vocab 32000–128000 ≫ hidden 4096 ≫ steps). Attention GEMMs are near-square (N≈hidden); ffn up-proj has N≈4·hidden. Only lm_head has N *much larger* than M and than K.
- The reduction dim **K = hidden is reused across the vast N column** — tiling should *widen block_n* to amortize the K-load over many output columns, and *keep block_m small* because M is 1 at decode (≤ steps at prefill).
- Consequence: the optimal tile is **(small block_m, wide block_n)** — the inverse of the attention/ffn tiling the Decode/Prefill arms pick. Hard-coding lm_head into Decode (N≈hidden square tile) wastes the wide-N reuse; hard-coding it into Prefill (block_m=32) over-subscribes M when m==1. Hence a distinct arm.

The hand-written TLOLog default below is the starting point; the empirical FCP pass (#2) refines it. It is **(16, 64)**: block_m=16 (small — M is 1 at decode), block_n=64 (wide — N=vocab dominated), block_k=64 (K=hidden, reused across N). (The earlier draft's `(32,16)` was backwards — it narrowed the output column exactly where widening helps; corrected here.)

### 4.3 Tile selection from hardware + shape

This is where the kernel is not just unoptimized defaults. The tile selection derives from hardware properties + shape class.

The research basis:
- Marlin/warp-tiled kernels: tile sizes should be multiples of the **Wave32** wavefront size (32 threads on RDNA2) so that memory transactions align with bus boundaries. On RDNA2 with 32-byte memory transactions, a block of 128 or 256 threads (4 or 8 Wave32 wavefronts) hits transaction boundaries cleanly
- Occupancy: 2-4 **Wave32** wavefronts per CU is the sweet spot on RDNA. More wavefronts → more register pressure. Fewer → underutilization. On gfx1036 with 64 CUs, the codebase uses 256 threads/block (8 Wave32 wavefronts), which exceeds the 2-4 wavefront sweet spot and targets higher occupancy for compute-bound GEMMs. For memory-bound decode paths, 128 threads (4 Wave32 wavefronts) is preferred — matching `fusion.rs:56` which sets `block_dim_x = 128` when `wavefront_size == 32`. (CDNA is Wave64 — separate path.)
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
    /// Placeholder dims for the hand-written tile defaults below. ONLY Decode/Prefill use
    /// these in tests; TLOLog is never constructed with placeholder dims — its real M/N/K
    /// arrive via the `dims` argument to `pick_tiles` from the dispatch layer (the lm_head
    /// GEMM's actual [steps, vocab, hidden]). The `m:0` placeholders here are NOT a real
    /// shape; do not read them for TLOLog.
    pub fn dims(&self) -> ShapeDims {
        match self {
            ShapeClass::Decode => ShapeDims { m: 1, n: 0, k: 0 },  // per-token: M=1
            ShapeClass::Prefill => ShapeDims { m: 0, n: 0, k: 0 }, // large batch
            ShapeClass::TLOLog => ShapeDims { m: 0, n: 0, k: 0 },  // real dims from dispatch
        }
    }
}

pub fn pick_tiles(spec: &HardwareSpec, shape_class: ShapeClass, dims: ShapeDims) -> TileConfig {
    let wave = spec.wavefront_size;           // 32 on gfx1036/gfx1030
    // NOTE: max_shared_mem_per_block (384 KB) is the per-BLOCK request ceiling, but
    // physical LDS is 64 KB/CU on gfx1036/gfx1030. Co-residency is bounded by the
    // per-CU figure, so the double-buffer gate and the #6 candidate-validity check
    // (§4.3 empirical pass) must use lds_per_cu, NOT the per-block attribute.
    let lds_per_cu = 64 * 1024; // physical LDS per CU on gfx1036/gfx1030
    let max_lds = lds_per_cu;
    let max_threads = spec.max_threads_per_block; // 1024 on gfx1036
    let cu_count = spec.cu_count;              // 64 on gfx1036

    // Block thread count (target_waves) is NOT hardcoded here — it is derived from the
    // tile's LDS + VGPR + thread bounds once block_m/block_k/lds_per_tile are known
    // (see the [#6] occupancy derivation right after the LDS double-buffer gate below).
    // CDNA (Wave64) is out of scope for this plan; gfx1036/gfx1030 are Wave32, so
    // wave = 32 throughout.

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

    // Block M/N from shape class:
    // Decode:  (16, 16) small tiles (memory-bound, M=1 per token)
    // Prefill: (32, 32) large tiles (compute-bound, fewer large GEMMs)
    // TLOLog:  (16, 64) small block_m (M=1 at decode), WIDE block_n (N=vocab dominated,
    //          K=hidden reused across the wide N column) — see "Why TLOLog is a separate
    //          bucket" above. This is the inverse of the Decode/Prefill square-ish tiling.
    let (block_m, block_n) = match shape_class {
        ShapeClass::Decode => (16, 16),
        ShapeClass::Prefill => (32, 32),
        ShapeClass::TLOLog => (16, 64),
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
    //   block_k ≤ (64*1024/2 - 256) / 32 = (32768 - 256) / 32 = 1016
    // That's still large — limited by threads/VGPR, not LDS.
    //
    // For gfx1036, Prefill (32,32):
    //   block_k ≤ (32768 - 1024) / 64 = 496
    // Still large. LDS is not the bottleneck for BF16 tiles on gfx1036.
    //
    // Practical block_k: limited by register pressure and wavefront occupancy,
    // not LDS. Use values from existing charon_scalar_candidates() that work:
    // Decode:  block_k = 32 (small K per token)
    // Prefill: block_k = 64 (larger K for batch)
    // TLOLog:  block_k = 64 (K=hidden, reused across the wide N=vocab column)

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

    // [#6] Derive occupancy (target_waves -> threads) from HARDWARE BOUNDS, not a
    // hard-coded 4. vLLM's LL4MI kernel does exactly this wave-aware partitioning;
    // jit-mgpu.md targets the same on gfx1036 (Wave32). No threading race: pick_tiles
    // is pure in (spec, dims), single-threaded per device. This is a launch-validity
    // + occupancy guard (see candidate_valid() in the FCP section for the same bounds
    // used as a search pre-filter).
    //
    // Occupancy on one CU is bounded by three independent ceilings; the real wave count
    // is the MINIMUM of all three, then clamped to a sane range:
    //
    //   (a) VGPR ceiling:    waves <= VGPR_FILE / (vgpr_per_thread * wave)
    //       VGPR_FILE = 512 per SIMD (RDNA2). vgpr_per_thread is kernel-specific
    //       (estimate from tile size; refined empirically by the FCP pass — see #2).
    //   (b) LDS ceiling:     waves <= (lds_per_cu / (lds_per_tile + double_buf))
    //       co-resident tiles must fit in physical 64 KB/CU, not the 384 KB request cap.
    //   (c) Thread ceiling:  waves <= max_threads_per_block / wave
    //
    // Then clamp to [1, 4] (RDNA sweet spot per rocm-kernels; 4 Wave32 = 128 threads,
    // which matches the existing fusion.rs/charon.rs launch pattern). The empirical FCP
    // pass (#2) is the final arbiter — this just picks a safe, occupancy-aware default
    // so the base (non-searched) path is correct, not over- or under-subscribed.
    let vgpr_per_thread = estimate_vgpr_per_thread(block_m, block_n, block_k); // (a)
    let vgpr_file: u32 = 512;                  // RDNA2 per-SIMD VGPR file
    let waves_vgpr = vgpr_file / (vgpr_per_thread * wave).max(1);
    let waves_lds = (lds_per_cu / (lds_per_tile + if lds_double_buffer { lds_per_tile } else { 0 }).max(1)) as u32; // (b)
    let waves_thread = max_threads / wave;     // (c)
    let target_waves = waves_vgpr
        .min(waves_lds)
        .min(waves_thread)
        .clamp(1, 4);                          // RDNA occupancy sweet spot
    let threads = target_waves * wave;
    assert!(threads <= max_threads && threads % wave == 0);

    // Split-K: derive from the ACTUAL K dimension (dims.k), not a hard-coded 1.
    // vLLM's q_gemm_rdna3.cu uses compute_wmma_k_split / compute_wmma_k_split_mn:
    // split when K exceeds what one block_k pass can hide. Same rule of thumb here:
    // split when K > block_k * 4, with the split count capped so per-split work
    // stays balanced. The kernel already supports split-K (charon kernels do).
    let split_k = if dims.k > block_k * 4 {
        // ceil(K / (block_k * 4)), clamped to [1, 16] to bound atomic/epilogue cost.
        ((dims.k + block_k * 4 - 1) / (block_k * 4)).clamp(1, 16)
    } else {
        1
    };

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

/// [#6] Estimate VGPRs/thread for a tile, so the occupancy derivation can bound
/// wave count by the VGPR file. This is a cheap static estimate from tile geometry;
/// the empirical FCP pass (#2) is the real arbiter of register pressure (it skips
/// candidates that fail to compile/launch under VGPR overflow via `Err(_) => continue`).
/// Larger tiles (more accumulators / A-B fragments) cost more VGPRs; clamp to the
/// RDNA2 per-thread max of 256 so the divisor in `waves_vgpr` stays valid.
fn estimate_vgpr_per_thread(block_m: u32, block_n: u32, block_k: u32) -> u32 {
    // Heuristic: per-thread fragments scale with the per-thread tile area after
    // wavefront decomposition. ~1 VGPR per 4 (block_m*block_n)/wave elements for the
    // C accumulator + ~1 per 4 (block_k*wave)/... for A/B staging. Keep it simple and
    // conservative; the FCP pass corrects it.
    let per_thread_area = ((block_m * block_n) / 32).max(1) + ((block_k * 32) / 64).max(1);
    per_thread_area.clamp(32, 256)   // RDNA2 VGPR/thread range
}
```

The numbers above are for gfx1036. For gfx1030 (fewer CUs, same LDS/CU): the same formula applies — `target_waves` is derived from the VGPR/LDS/thread ceilings above, and on gfx1030 it typically resolves to 2 (→ 64 threads) because the per-CU budgets are identical but the lower CU count drives smaller grids; it is NOT a separate literal. The LDS budget is the same per-CU, so tile sizes don't change — only the derived occupancy (and thus `threads`) drops. Verify `pick_tiles` on gfx1030 returns `threads == 64`.

This is not "first pass, validate later." These numbers are derived from the actual gfx1036 hardware properties and the existing `charon_scalar_candidates()` brute-force approach. The tile picker should be validated against `charon_scalar_candidates()` output for the same shapes, but the starting point is concrete.

#### Roofline cost model — pre-filter for FCP (NOT the final selector)  [#2]

The roofline model is a cheap **pre-filter** that drops obviously bad candidates before
the empirical GPU measurement pass (below). It is no longer the winner selector — that
role belongs to measured kernel time. The model:

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

#### Empirical autotune — FCP measures, it does not estimate  [#2]

The prior FCP design picked the winner by `roofline_cost` — a static model that
cannot see register pressure, LDS bank conflicts, or instruction scheduling. That
selection is structurally wrong for real shapes. helion's autotuner (and vLLM's
runtime-tuned rocBLAS solution index in `gemm_tuning.rs`) *measure* candidate kernels
on the GPU and persist the winner. The corrected `fcp_fallback_tile_search` does the
same:

1. Generate a small, **constrained** candidate set (not a 1400-way joint search).
2. Pre-filter by roofline (cheap) to drop obviously bad configs.
3. For survivors, JIT-compile + launch + GPU-time each, keep the fastest.
4. **[#3] Write the winner into `KernelTuneCache`** (`autotune.rs`), keyed by the real
   shape `(entry, gpu_arch, m, n, k)` so repeat shapes hit the lookup table (SCYTHE
   principle) and skip the search entirely.

```rust
pub fn fcp_fallback_tile_search(
    device: &RocmDevice,        // needed to compile + launch + time candidates
    spec: &HardwareSpec,
    entry: &str,
    dims: ShapeDims,
    shape_class: ShapeClass,
) -> TileConfig {
    let base = pick_tiles(spec, shape_class, dims);

    // --- Candidate generation: constrained search space ---
    // block_k ∈ {16,32,64,128} (K a multiple of 16 for Wave/MFMA alignment);
    // block_m/block_n are wavefront multiples only; split_k ∈ {1, base.split_k, 2*base.split_k}
    // (clamped [1,16]). This kills invalid candidates and matches vLLM's templated
    // BLOCK_SIZE discipline (vllm-port.md §2).
    let wave = spec.wavefront_size;
    let block_k_choices = [16u32, 32, 64, 128];
    let mut candidates: Vec<TileConfig> = Vec::new();
    for &bm in &[base.block_m, base.block_m.saturating_add(wave)] {
        for &bn in &[base.block_n, base.block_n.saturating_add(wave)] {
            if bm == 0 || bn == 0 || bm % wave != 0 || bn % wave != 0 { continue; }
            if (bm * bn) as u32 > spec.max_threads_per_block { continue; }
            for &bk in block_k_choices.iter() {
                if bk % wave != 0 { continue; }
                for &sk in [1u32, base.split_k, (base.split_k * 2).clamp(1, 16)].iter() {
                    let mut cand = base.clone();
                    cand.block_m = bm; cand.block_n = bn; cand.block_k = bk; cand.split_k = sk;
                    if candidate_valid(spec, &cand) {     // [#6] resource gate (below)
                        candidates.push(cand);
                    }
                }
            }
        }
    }

    // --- Pre-filter by roofline (cheap; keeps the measured set small) ---
    candidates.sort_by(|a, b| {
        roofline_cost(spec, dims, a).partial_cmp(&roofline_cost(spec, dims, b)).unwrap()
    });
    candidates.truncate(candidates.len().min(16));   // cap measured candidates
    candidates.dedup();

    // --- Empirical measurement: compile + launch + GPU-time each survivor ---
    // DESIGN NOTE: compute_kernel_source_with_spec must accept an explicit TileConfig
    // (add Option<TileConfig>; None -> pick_tiles). The loop passes Some(cand) so each
    // candidate is compiled with its own #defines, not the base config.
    let mut best: Option<(TileConfig, f64)> = None;
    for cand in &candidates {
        let source = compute_kernel_source_with_spec(spec, entry, shape_class, dims, 0, 1, Some(cand));
        let (hsaco, lowered) = match device.jit_compile_or_cache(&source, entry, spec) {
            Ok(v) => v,
            Err(_) => continue,   // VGPR/compile failure (see [#6]) -> skip, don't panic
        };
        let t_ms = device.time_kernel_ms(&hsaco, &lowered, dims, cand);
        if best.as_ref().map_or(true, |(_, bt)| t_ms < *bt) {
            best = Some((cand.clone(), t_ms));
        }
    }
    let winner = best.expect("at least one valid candidate").0;

    // --- [#3] Persist the winner into KernelTuneCache ---
    // autotune.rs already has KernelKey(kernel, gpu_arch, m, n, k) -> LaunchConfig
    // and KernelTuneCache (JSON shadow). Map TileConfig -> LaunchConfig and store
    // keyed by the REAL shape so the next miss on this (entry,arch,m,n,k) is a table
    // hit, not a re-measure.
    device.store_tune_cache(entry, spec, dims, &winner);

    winner
}

/// [#6] Reject candidates that overcommit GPU resources.
/// This is NOT a threading race — pick_tiles / fcp_fallback_tile_search are pure in
/// (spec, dims), and the compile path is single-threaded per device (§3). It is a
/// launch-validity + occupancy guard:
///  - physical LDS is 64 KB/CU on gfx1036/gfx1030, NOT the 384 KB per-block ceiling.
///    With co-resident blocks, the double buffer + 1 resident tile must fit in 64 KB/CU.
///  - block threads <= max_threads_per_block.
///  - VGPR/register pressure is not modeled here; the backstop is the hiprtc
///    compile/launch failure caught by `match Err(_) => continue` above. If a
///    candidate's VGPRs/thread exceed the per-thread max (256 on ROCm), compilation or
///    launch fails hard — we skip it rather than panic, so the search still returns a
///    valid winner.
fn candidate_valid(spec: &HardwareSpec, cand: &TileConfig) -> bool {
    let lds_per_cu = 64 * 1024;  // physical LDS/CU on RDNA2
    let lds_per_tile = 2 * (
        (cand.block_m as u64) * (cand.block_k as u64) * 2
        + (cand.block_k as u64) * (cand.block_n as u64) * 2
        + (cand.block_m as u64) * (cand.block_n as u64) * 2
    );
    if 2 * lds_per_tile > lds_per_cu { return false; }   // double buffer + co-residency
    if cand.threads > spec.max_threads_per_block { return false; }
    if cand.block_m == 0 || cand.block_n == 0 || cand.block_k == 0 { return false; }
    true
}
```

The result: a **measured** best config (not an estimated one), persisted for reuse. The
cost is O(survivors) GPU launches on a cache miss — milliseconds-to-tens-of-ms for ≤16
candidates — matching FCP's "millisecond-level overhead" claim, and it runs only once
per (entry, arch, m, n, k) thanks to #3.

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
            full_dims,    // real M,N,K for split-K (#4) + FCP search shaping
            i as u32,      // GRIM_DEVICE_ID
            n as u32,      // GRIM_NUM_DEVICES
            None,          // multi-GPU path uses pick_tiles(); FCP loop passes Some(cand)
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
            let source = compute_kernel_source_with_spec(&spec, entry, shape_class, dims, 0, 1, None);
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
- `crates/grim-backend-rocm/src/device/hardware_spec.rs` — populate `mem_bandwidth_gb_s` from the existing `CapabilityProfiler::capabilities()[..].hbm_bandwidth_gbps` (already computed by `arch_tflops_table()`); no new function needed in `capability_profiler.rs` itself. Fall back to a conservative default (gfx1036: 500 GB/s GDDR6) when no capability has been reported yet.
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
- `crates/grim-backend-rocm/src/kernels/tile_picker.rs` — `TileConfig`, `pick_tiles()`, `roofline_cost()`, `estimate_vgpr_per_thread()`, `fcp_fallback_tile_search()`, `candidate_valid()`, `ShapeDims`

**Modified files:**
- `crates/grim-backend-rocm/src/kernels/source_asm.rs` — add `compute_kernel_source_with_spec()`. Keep `compute_kernel_source()` as-is for backward compatibility

**Test:**
- Generate source for gfx1036 with shape_class = Prefill. Verify the source contains `#define GRIM_WAVEFRONT_SIZE 32`, `#define GRIM_CU_COUNT 64`, `#define GRIM_BLOCK_M 32`, `#define GRIM_BLOCK_N 32`, `#define GRIM_BLOCK_K 64`
- Generate source for gfx1030 with shape_class = Decode. Verify: wavefront = 32, block_m = 16, block_n = 16, block_k = 32, threads = 64 (2 waves)
- Generate source for gfx1036 with shape_class = Decode, then with shape_class = Prefill. Verify different block_m/block_n values
- `pick_tiles` for gfx1036 Prefill returns `TileConfig { block_m: 32, block_n: 32, block_k: 64, lds_double_buffer: true, ... }`. Verify LDS: lds_per_tile = 2*(32*64*2 + 64*32*2 + 32*32*2) = 20480 bytes; physical lds_per_cu = 64*1024 = 65536; 2*20480 = 40960 ≤ 65536 → double buffer true. **[#6]** Verify `threads` is derived (not hardcoded): `target_waves` = min(VGPR ceiling, LDS ceiling, thread ceiling) clamped [1,4]; for Prefill on gfx1036 this resolves to 128 (4 Wave32). Assert `threads % wave == 0 && threads <= max_threads`.
- **[#6]** `pick_tiles` occupancy derivation: inject a tile that would demand >512 VGPRs/thread (e.g. oversized block_m/block_n) and verify `target_waves` drops accordingly (waves_vgpr ceiling bites) — proves occupancy is bound-driven, not the old constant 4.
- `pick_tiles` for gfx1030 (Decode) returns `threads == 64` (derived 2 Wave32), not a hardcoded literal.
- **[TLOLog]** `ShapeClass::from_op(GemmOp::LmHead, 1)` == `ShapeClass::TLOLog` even though `m == 1` (which `from_m` would bin as Decode) — proves the classifier is op-identity, not M-derived. And `ShapeClass::from_op(GemmOp::LmHead, 4096)` is also `TLOLog` (fires at prefill m too). `ShapeClass::from_op(GemmOp::Attention, 1)` == `Decode` (unchanged behavior for non-lm_head).
- **[TLOLog]** `pick_tiles` for gfx1036 with `shape_class = ShapeClass::TLOLog` and real `dims { m: 1, n: vocab(=32000), k: 4096 }` returns `block_m == 16, block_n == 64, block_k == 64` — the distinct wide-N tile, confirmed different from both Decode (16,16) and Prefill (32,32). Assert `block_n == 64` (wide output for N=vocab) and `block_m == 16` (small, M=1 at decode).
- **[TLOLog]** Wire-through: a `matmul` call on `roc_device.rs:1488` tagged `GemmOp::LmHead` results in a JIT source containing `#define GRIM_BLOCK_M 16` + `#define GRIM_BLOCK_N 64` (not the Decode/Prefill defines), proving the op tag propagates end-to-end through `lookup_gemm_config`/`pick_tiles`/`compute_kernel_source_with_spec`.
- `roofline_cost` for a known shape returns a finite f64. Verify it's deterministic (same input → same output)
- `fcp_fallback_tile_search` for M=137, N=256, K=512 returns a TileConfig with valid dimensions (wavefront multiples, ≤ max_threads) and `candidate_valid() == true`

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

### Phase 5: Empirical FCP fallback tile search  [#2,#3]

**Duration:** 1-2 days

**What:** Add `fcp_fallback_tile_search()` to `tile_picker.rs` as a **measured** search (GPU-time candidates, keep fastest), and self-persist the winner into `KernelTuneCache` so repeat shapes hit the lookup table. This replaces the prior roofline-only design (the roofline model is now just a pre-filter — see §4.3). Add the `candidate_valid()` resource gate (see #6).

**New files / new fns:**
- `crates/grim-backend-rocm/src/kernels/tile_picker.rs`:
  - `fcp_fallback_tile_search(device, spec, entry, dims, shape_class)` — compile + launch + GPU-time each candidate, return fastest
  - `candidate_valid(spec, &TileConfig)` — LDS-per-CU / threads / non-zero gate (backstops VGPR failure via the `match Err(_) => continue` path in the search)
- `crates/grim-backend-rocm/src/device/helpers.rs`: add `RocmDevice::time_kernel_ms()` (hipEventRecord around launch, returns ms) and `RocmDevice::store_tune_cache()` (maps `TileConfig` -> `LaunchConfig`, writes `KernelTuneCache` keyed by `(entry, arch, m, n, k)`)
- `crates/grim-backend-rocm/src/kernels/source_asm.rs`: `compute_kernel_source_with_spec` gains `tiles: Option<&TileConfig>` (None -> `pick_tiles()`; Some -> force a per-candidate config for the FCP loop)

**Modified files:**
- `crates/grim-backend-rocm/src/autotune.rs` — `Autotuner::lookup()`: on key miss, call `fcp_fallback_tile_search()` (which self-persists via `store_tune_cache`), then return the stored `LaunchConfig`. No separate "refine" branch needed — persistence is inside the search.

**Test:**
- Feed shape M=137, N=256, K=512. Verify `fcp_fallback_tile_search` returns a TileConfig with valid wavefront-multiple dimensions and `candidate_valid()==true`
- Feed same shape twice. Verify: (a) deterministic winner, (b) the second call is a **cache hit** in `KernelTuneCache` — no GPU re-measure (assert `time_kernel_ms` not invoked)
- Compare winner against `charon_scalar_candidates()` for the same shape: same or better LDS utilization, and the measured time is ≤ the base `pick_tiles` config time
- Inject a bogus candidate with 256 KB/CU LDS demand; verify `candidate_valid()` rejects it before any compile
- Inject a candidate that exceeds VGPR headroom; verify the search skips it via `Err(_) => continue` and still returns a valid winner

**Skills that apply:**
- `kernel` — empirical tuning, candidate generation over constrained space, measured cost
- `amd` — occupancy factors for RDNA2, TFLOPS estimates (pre-filter only)
- `ffi` — hipEvent timing, hiprtc compile-per-candidate
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
| `crates/grim-backend-rocm/src/kernels/tile_picker.rs` | `TileConfig`, `pick_tiles()`, `roofline_cost()`, `estimate_vgpr_per_thread()`, `fcp_fallback_tile_search()`, `candidate_valid()`, `ShapeDims` |
| `crates/grim-backend-rocm/src/multi_gpu_launch.rs` | `launch_multi_gpu_kernel()`, shard computation, per-device JIT + launch |

### Modified files

| File | Change |
|------|--------|
| `src/device/probe.rs` | Expose `wavefront_size()`, `max_shared_mem()`, `max_threads_per_block()`, `active_cu_count()` |
| `src/device/capability_profiler.rs` | No changes needed — `hbm_bandwidth_gbps` already exists on `GpuCapability` |
| `src/autotune.rs` | Add `ShapeClass::TLOLog` variant (#0); add `GemmOp` enum + `ShapeClass::from_op(op, m)` op-identity classifier (replaces the inoperable `from_m`-only rule for TLOLog) |
| `src/device/gemm_tuning.rs` | `lookup_gemm_config(m, n, k, wave)` gains a `shape: ShapeClass` param; select the TLOLog tile arm for `ShapeClass::TLOLog` (lm_head). `from_m`-only callers unchanged via `ShapeClass::from_m` |
| `src/device/roc_device.rs` | `matmul` (line 1488) takes a `GemmOp` and forwards it into `lookup_gemm_config`/`pick_tiles` so lm_head is tagged TLOLog regardless of m; extend `launch_compute_kernel()` to try hardware-adaptive source first |
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
- `gemm_tuning.rs` — NOT in the "not touched" list: `lookup_gemm_config` is modified to accept `ShapeClass` (see Modified files). The existing shape-class tables for Decode/Prefill are unchanged; only the new TLOLog arm + the `shape` param are added.
- `charon.rs` — kernel source modules already complete. Not modified
- `jit_cache.rs` — caching infra already complete. Only extends the key struct

---

## 7. What the kernel optimizes

The hardware-adaptive JIT kernel adjusts these parameters based on system information:

| Parameter | Source | Effect |
|-----------|--------|--------|
| Block M/N tile sizes | Shape class (Decode=16x16, Prefill=32x32, TLOLog=16x64) + wavefront rounding | LDS utilization, occupancy |
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
