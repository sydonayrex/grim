# GPU follow-up work items

These two items are the GPU-execution counterparts of recommendations already
landed on the hot path (R2 deterministic MoE dispatch, R3 VPP). The CPU-side
foundations are built, tested, and pushed; what remains is the multi-GPU
execution that needs >=2 GPUs to verify. Scope is honest: these are weeks of
work, and neither can be validated on a single GPU.

---

## Work item 1 — MoE fused comm-compute mega-kernel (R2 GPU)

**Goal.** Replace the per-slot CPU loop in `MoeFfn::forward_deterministic`
with a single persistent-SM HIP kernel that packs tokens by expert, evaluates,
and combines — so heterogeneous MoE adapters share one kernel dispatch the way
the CPU path already shares one `DeterministicTokenMap`. Bitwise-identical to the
CPU reference (proven by the existing property test).

**Prerequisites in place (do not rebuild):**
- `crates/grim-nn/src/moe_deterministic.rs` — `DeterministicTokenMap::build`,
  `pack_activate_activations`, `combine_expert_outputs`, `ScoreboardSync`.
- `crates/grim-nn/src/moe.rs:forward_deterministic` — CPU reference that uploads
  nothing and calls `expert_forward` per slot. The feature flag
  `moe-deterministic-dispatch` (in `grim-nn/Cargo.toml`) already switches
  `forward` to use it.
- `crates/grim-nn/src/moe.rs::test_deterministic_dispatch_is_bitwise_identical`
  — the parity gate the kernel must still pass.

**File-level steps:**

1. `crates/grim-backend-rocm/src/kernels/` — new `moe_mega_kernel.rs`:
   - HIP kernel source string `MOE_MEGA_KERNEL_SOURCE` implementing the
     persistent-Worker model: a 1D grid of `NSM` threadblocks, each polling a
     global task cursor (`atomic_add` on a `LOCK_PTR`). Task space linearized as
     `[0, Ncomm)` = communication tasks, `[Ncomm, Ncomm+Ncomp)` = compute tasks.
     Decode task ID → role (Comm / Comp / Relay).
   - Comm-Worker: read token from the rank-local staging buffer at
     `destination_slots[instance]`, write to the expert's packed buffer at
     `global_offsets[expert] + local_cursor`.
   - Comp-Worker: poll `ScoreboardSync` token-arrival counter; on tile-ready,
     launch the expert GroupGEMM via the existing CK/wmma GEMM path
     (`launch_charon_fused_dispatch` in `roc_device.rs:6588` is the template).
   - Relay-Worker: when multiple experts for one token are on-GPU, multicast
     once instead of re-fetching over the inter-rank link.
   - Compile via the existing `jit_compile_hsaco` / `jit_compile_or_cache`
     path in `device/helpers.rs` and `device/jit_cache.rs`; the disk cache
     handles cold-start cost automatically.

2. `crates/grim-nn/src/moe.rs` — extend `forward_deterministic` with a device
   branch:
   - When `x.device().is_rocm()` (gated on feature `rocm-mem`), upload
     `destination_slots`, `global_offsets`, `expert_counts` to device buffers
     (`upload_device_buffer` in `device/helpers.rs`), build the `ScoreboardSync`
     state, then launch `moe_mega_kernel` instead of the per-slot CPU loop.
   - Keep the CPU path as the verified fallback (it already passes the parity
     test). Structure: `#[cfg(feature = "rocm-mem")] { launch kernel } else { cpu loop }`.

3. `crates/grim-backend-rocm/src/device/roc_device.rs` — add a
   `launch_moe_mega_dispatch` method modeled on
   `launch_charon_fused_dispatch` (line 6588): it pins the device context, plans
   the launch grid, uploads the routing + scoreboard buffers, and calls the
   generic `launch_compute_kernel` helper. Reuse `upload_device_buffer` and
   `DeviceGuard::set` discipline already there.

4. `crates/grim-nn/src/moe_deterministic.rs` — no logic change, but expose a
   `ScoreboardSync::to_device_buffers(&self) -> (Vec<u32>, Vec<u32>)` so the
   kernel can read arrival/ready flags over PCIe without host round-trips.

5. Autotune integration (`crates/grim-backend-rocm/src/autotune.rs`): add a
   `MoeMegaKnob` search over the UniEP tuple `(wdisp, Ndisp, Nrelay)`.
   Reuse the existing `ShapeDims` / `TileConfig` / autotuner infrastructure;
   the kernel is parametrized by SM role counts, so the same launch path runs
   every candidate.

6. `crates/grim-nn/Cargo.toml` — document that `moe-deterministic-dispatch`
   produces correct CPU output without `rocm-mem`; with `rocm-mem` it additionally
   exercises the kernel. Add a `moe-mega-kernel = ["rocm-mem"]` feature if a
   separate gate is desired.

**Acceptance gates:**
- `test_deterministic_dispatch_is_bitwise_identical_to_reference` still passes
  with `moe-deterministic-dispatch` on (CPU path unchanged).
- New `test_moe_mega_kernel_parity` (under `rocm-mem`): run `forward_deterministic`
  on a tiny MoE with the kernel, compare `to_bits()` against the CPU loop.
  Skips without a GPU.
- Golden-charon-MoE test in `crates/grim-backend-rocm/tests/golden_charon_moe_gpu.rs`
  stays green (the mega-kernel must not regress the non-deterministic path).

**Hardware needed:** >=1 ROCm GPU to launch; >=2 GPUs (or ROCm peer-to-peer on
one APU) to exercise the inter-rank Comm-Worker path. Cannot be verified
single-GPU because the all-to-all is the point.

---

## Work item 2 — Multi-rank VPP execution (R3 multi-node)

**Goal.** Drive the existing `VirtualPlannerPlan` V-traversal across >=2 physical
ranks with async bidirectional communication at the fold points, so chunk *k*'s
heavy middle on rank *r* overlaps chunk *k-1*'s tail and chunk *k+1*'s head on
rank *r-1*. Measure bubble ratio on long-context prefill.

**Prerequisites in place (do not rebuild):**
- `crates/grim-engine/src/pipeline_engine.rs` — `VirtualPipelinePlan::plan`
  (V-fold layer→rank mapping), `stages_for_rank`, `VirtualPipelineCoordinator`,
  `forward_vpp`.
- Tests `test_virtual_pipeline_plan_fold_back_mapping` and
  `test_virtual_pipeline_forward_vpp_equivalence` (single-node, passing).
- `crates/grim-kvtransport/src/lib.rs` — the TCP send/recv primitives
  (`send_block_remote`, `fetch_block_remote`) and the shared-memory P2P path
  already added for P1.

**File-level steps:**

1. `crates/grim-engine/src/pipeline_engine.rs` —
   `VirtualPipelineCoordinator::forward_vpp` currently runs every virtual stage
   inline on one device. Split it:
   - Keep the single-device path for `num_physical_ranks == 1`.
   - Add a multi-rank path that, for each `VppStep`, (a) `recv_activations`
     from the predecessor virtual stage's rank if `recv_from` is `None` on this
     rank, (b) run the local layers, (c) `send_activations` to the successor
     virtual stage's rank. The activation source/destination ranks come from
     `VirtualStage::physical_rank` in the plan — no new routing logic.

2. `crates/grim-engine/src/pipeline_engine.rs` — VPP-Async reordering. After the
   basic multi-rank path works, apply the paper's tail/head swap: when chunk
   Ck-1's tail stage and chunk Ck+1's head stage are both pending on the same
   rank, schedule the head first so its send fires while the peer rank computes
   Ck's middle. This is a reordering of `VppStep` execution within the existing
   scheduler loop; no new transport.

3. Cross-rank transport. Two options, one file each:
   - `crates/grim-kvtransport/src/lib.rs` — reuse the existing
     `send_block_remote` / `fetch_block_remote` over TCP (already tested in
     `test_fetch_block_remote_roundtrip_against_live_server`). Lowest effort;
     latency is acceptable for prefill where chunks are large.
   - Or ROCm peer-to-peer / RCCL for the inter-rank activation exchange.
     This is the higher-performance path and should mirror how the existing
     `parallel_comm.rs` `ParallelCommunicator` does P2P on ROCm.

4. `crates/grim-memory/src/` — per-rank KV isolation. Each rank's
   `PipelineStageRunner` already holds its own `block_pool`
   (`pipeline_engine.rs::PipelineStageRunner::new`). Confirm the paged KV pool
   is allocated on `config.device_ordinal` (it is, via `KvBlockPool::new_on_device`).
   No cross-rank KV fetch is needed for the first version — each rank holds the
   KV for its own virtual stages, matching how the single-node path works.

5. `crates/grim-scheduler/src/lib.rs` — expose chunk boundaries. The
   continuous-batcher `Scheduler::schedule()` emits `prefill_ids`; the
   coordinator needs to know each id's chunk offset so it can interleave two
   requests' chunks for drain-window packing. Minimal: a
   `Scheduler::chunk_offsets(&self) -> HashMap<u64, (usize, usize)>` derived from
   each running request's `consumed_tokens`.

6. Benchmark harness `crates/grim-engine/tests/` — new `vpp_benchmark.rs`:
   - Run the VPP coordinator on a 512K-token prefill across 2+ ranks.
   - Measure pipeline bubble ratio = (idle_steps / total_steps) and compare
     against single-device and naive PP baselines.
   - The synthesis headline (98% bubble reduction vs DCPP) is the target; this
     test can only run with >=2 GPUs and should skip otherwise.

**Acceptance gates:**
- `test_virtual_pipeline_plan_fold_back_mapping` and
  `test_virtual_pipeline_forward_vpp_equivalence` stay green (single-node path
  untouched).
- New `test_vpp_multi_rank_forward` (skips without >=2 GPUs): run a toy
  multi-layer model through the VPP coordinator across 2 ranks via TCP
  transport; assert output matches the single-node `forward_vpp`.
- `vpp_benchmark.rs` runs on >=2 GPUs and reports bubble ratio; does not gate
  CI (hardware-gated) but documents the measured number.

**Hardware needed:** >=2 ROCm GPUs. The planner and single-node forward are
already verified; this item cannot be validated without the second GPU.

---

## Sequencing recommendation

Do Work item 1 first. It is self-contained on the MoE path, has a clear parity
gate (the existing bitwise property test), and the CPU reference already proves
the algorithm. Work item 2 depends on cross-rank transport that the P1
multi-transport work (shared-memory P2P) has already started, and it needs the
benchmark to be meaningful — without it, VPP is a planner with no evidence it
helps, exactly as the synthesis cautioned.
