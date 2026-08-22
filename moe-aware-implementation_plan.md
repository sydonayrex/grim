# Implementation Plan: FreeToken MoE-Aware Bandwidth-Adaptive Execution in GRIM

Incorporate FreeToken edge-native MoE serving architecture into GRIM. Structured into 5 prioritized phases aligning with existing GRIM crates (`grim-memory`, `grim-scheduler`, `grim-engine`, `grim-backend-rocm`, `grim-backend-cpu`, `grim-format`).

```mermaid
flowchart TD
    subgraph P1 ["Phase 1: Semantic State Caching"]
        RTree["grim-memory::RadixTree"] --> AnchorPool["SemanticAnchorPool (Recurrent State Checkpoints)"]
        AnchorPool --> AnchorMatch["Anchor Match on Special Tokens (<think>, </tool_call>)"]
    end

    subgraph P2 ["Phase 2: Bandwidth-Adaptive Decode (q* Policy)"]
        BWProf["grim-scheduler::BandwidthProfile (B_P, B_H)"]
        BWProf --> QStar["q* = round(m * B_P / B_H)"]
        QStar -->|Set F (q*)| H2D_Miss["PCIe Cache Fill -> GPU MoE"]
        QStar -->|Set C (m - q*)| CPU_Pool["grim-backend-cpu Worker Pool (SIMD)"]
        CPU_Pool --> Merge["Exact Partial Sum Reduction (y_GPU + y_CPU)"]
    end

    subgraph P3 ["Phase 3: Elastic GPU Budget Rebalancing"]
        SafePoint["Scheduler Safe Point (Step Boundary)"]
        SafePoint --> Rebalance["Dynamic Split: VRAM -> KV Cache vs Expert Cache Slots"]
    end

    subgraph P4 ["Phase 4: Prefill Double-Buffering"]
        StreamL["Compute Stream: Layer l"]
        StreamL1["Transfer Stream: Layer l+1 Full Expert Set"]
        StreamL1 -.->|Overlap| StreamL
    end

    subgraph P5 ["Phase 5: FTW Layout & Fast Bootstrap"]
        FTW["Flat [L*E] Bank Format -> Direct I/O -> Post-Pin"]
    end
```

---

## User Review Required

> [!IMPORTANT]
> **Semantic Anchoring vs KV Blocks**:
> Standard attention reuses KV blocks via `RadixTree`. Recurrent / linear-attention layers (DeltaNet, Kimi Delta, Mamba/SWA) compress full sequence history into a monolithic state. Checkpointing every token is memory-prohibitive. We attach full recurrent-state checkpoints **only** at semantic anchor tokens (`<think>`, `</think>`, `<tool_call>`, `</tool_output>`, message turn boundaries).

> [!TIP]
> **Runtime Policy for $q^*$**:
> $B_P$ (PCIe bandwidth) and $B_H$ (Host memory GEMV bandwidth) are measured at engine initialization on actual tensor sizes. $q^*$ is calculated as a runtime policy: $q^* = \text{round}\left(m \cdot \frac{B_P}{B_H}\right)$, ensuring graceful degradation from pure GPU cache to pure CPU offload across heterogeneous machines.

---

## Phased Implementation Breakdown

### Phase 1: Semantic-Aware State Caching for Recurrent/Hybrid Models
**Primary Crates:** `grim-memory`, `grim-core`, `grim-models`

1. **[NEW] [`semantic_anchor.rs`](file:///D/rex/projects/grim/crates/grim-memory/src/semantic_anchor.rs)**:
   - Define `SemanticAnchorToken` registry (special token IDs for `<think>`, `</think>`, `<tool_call>`, `</tool_output>`, turn boundaries).
   - Implement `RecurrentStateCheckpointPool`: fixed-size LRU pool of layer recurrent states attached to `RadixNode` IDs.
   - Extend `RadixTree` traversal: when prefix matching a prompt, identify the deepest valid node holding a recurrent checkpoint.

2. **[MODIFY] [`radix.rs`](file:///D/rex/projects/grim/crates/grim-memory/src/radix.rs)**:
   - Add optional `recurrent_state_id: Option<usize>` slot to `RadixNode`.
   - On context editing/truncation (common in agent tool loops), evict child nodes but preserve valid root prefix and nearest anchor state.

---

### Phase 2: Bandwidth-Adaptive Decode ($q^*$ Miss Policy) & CPU Co-Execution
**Primary Crates:** `grim-scheduler`, `grim-engine`, `grim-backend-cpu`, `grim-backend-rocm`

1. **[NEW] [`bandwidth_policy.rs`](file:///D/rex/projects/grim/crates/grim-scheduler/src/bandwidth_policy.rs)**:
   - `BandwidthProfile`: stores probed $B_P$ (MB/s) and $B_H$ (MB/s).
   - Calculate $q^* = \text{clamp}\left(\text{round}\left(m \cdot \frac{B_P}{B_H}\right), 1, m\right)$.
   - Partition miss set $M$ into cache-fill set $\mathcal{F}$ ($|\mathcal{F}|=q^*$) and CPU execution set $\mathcal{C}$ ($|\mathcal{C}|=m - q^*$).

2. **[MODIFY] [`moe_dispatch.rs`](file:///D/rex/projects/grim/crates/grim-backend-cpu/src/moe_dispatch.rs)**:
   - Multithreaded persistent worker pool pinned to 1 thread per physical core.
   - Streaming SIMD dequant GEMV (BF16, MXFP4, NVFP4, Q4_0) with cache prefetching.
   - Partial gate-weighted reduction output buffer for set $\mathcal{C}$.

3. **[NEW] [`moe_hybrid_exec.rs`](file:///D/rex/projects/grim/crates/grim-backend-rocm/src/device/moe_hybrid_exec.rs)**:
   - Issue D2H signal for set $\mathcal{C}$ tokens to CPU worker pool.
   - Concurrently stream set $\mathcal{F}$ expert weights over PCIe and evaluate resident $\mathcal{H} \cup \mathcal{F}$ on GPU.
   - Receive CPU partial sum and perform exact reduction $y = y_{\text{GPU}} + y_{\text{CPU}}$.

---

### Phase 3: Elastic GPU Memory Management
**Primary Crates:** `grim-memory`, `grim-kvtransport`, `grim-engine`

1. **[MODIFY] [`moe_budget.rs`](file:///D/rex/projects/grim/crates/grim-memory/src/moe_budget.rs)**:
   - Implement `ElasticMoEAllocation`: dynamically shift VRAM allocation between KV cache blocks and expert cache slots.
   - Expose `reconfigure_budget(new_kv_envelope, new_expert_slots)` callable at scheduler step boundaries (safe points).
   - No engine restart or host-pool reloading needed on resize.

---

### Phase 4: Full-Layer Double-Buffered Prefill Transfer Pipelining
**Primary Crates:** `grim-engine`, `grim-backend-rocm`, `grim-backend-cuda`

1. **[NEW] [`moe_prefill_pipeline.rs`](file:///D/rex/projects/grim/crates/grim-engine/src/pipelines/moe_prefill_pipeline.rs)**:
   - Two full-layer slot buffers in GPU cache (`Buffer_A`, `Buffer_B`).
   - Layer $l$ compute on default compute stream while Layer $l+1$ full expert weights stream asynchronously on dedicated DMA stream.
   - Fallback to on-demand paging when GPU VRAM lacks space for 2 full layers.

---

### Phase 5: Optional FTW Format & Fast Bootstrap
**Primary Crates:** `grim-format`, `grim-models`

1. **[NEW] [`ftw.rs`](file:///D/rex/projects/grim/crates/grim-format/src/ftw.rs)**:
   - FreeToken Weight (FTW) pre-merged bank format: flat contiguous rows indexed by $lE + e$.
   - Direct I/O reading into host memory with subsequent page pinning (`mlock` / `hipHostRegister`).

---

## Verification Plan

### Automated Tests
1. **Semantic Anchor State Matching**:
   ```bash
   cargo test -p grim-memory --lib semantic_anchor
   ```
2. **Bandwidth Partitioning ($q^*$) Unit Tests**:
   ```bash
   cargo test -p grim-scheduler --lib bandwidth_policy
   ```
3. **CPU SIMD MoE Kernel Parity**:
   ```bash
   cargo test -p grim-backend-cpu --lib moe_dispatch
   ```
4. **End-to-End Hybrid MoE Correctness**:
   - Compare output of hybrid split ($\mathcal{H} \cup \mathcal{F}$ on GPU, $\mathcal{C}$ on CPU) against reference dense MoE. Max absolute error $< 10^{-4}$.
5. **Compilation & Safety**:
   ```bash
   cargo check --workspace
   ```
