# dats-demm.md — Inference Pipeline Code Review (MoE + Dense, ROCm backend)

> Date: 2026-09-17
> Base: `main @ 6dc612cf` (includes who-dat.md fixes)
> Scope: inference pipeline across `grim-backend-rocm`, `grim-models/transformer`, `grim-engine`, `grim-server`, `grim-cli`, `grim-scheduler`, `grim-core`, `grim-quant`, `grim-tensor`, `grim-speculative`, `grim-disagg`, `grim-kvtransport`, `grim-format`
> Method: full-workspace grep sweeps (1,361 raw `unwrap`/`expect` sites reduced to 193 production sites, each read in context), definition-vs-callsite analysis per identifier, `#[allow(dead_code)]` census, env-gate inventory. Test/bench/example code excluded from findings unless the *production* item is only test-reachable.
> Companion to: `who-dat.md` (speed/ROCm perf audit). This doc is correctness/hygiene, not perf.

---

## 0. MoE vs Dense perspective

**Dense path (Llama-family, Qwen, Gemma, LFM2 dense, …):** mature. Fused QKV/Gate-Up dot4 GEMVs wired, KV arena path default-on for ROCm, decode graph replay live (who-dat fixes landed 6dc612cf). The dense path's remaining hygiene is unwrap-heavy session-state downcasts and dense-FFN expects on the per-token path (§2, tier 1).

**MoE path (LFM2-MoE, DeepSeek, Mixtral, Ling, bailingmoe*):** significantly more speculative code mass. Three *complete, compiled, exported* MoE optimization subsystems are unwired: the `scythe_route` P2P ring-routing helpers, the `moe_fused_grouped_dispatch_{w8a8_int8,w8a8_fp8,awq}` launchers, and the charon variant-selector/dispatch family (`CharonSelector`, `plan_grouped_dispatch`, …). Host-side top-k routing in `shared_moe.rs` remains the only live MoE dispatch (matches who-dat.md finding #2.3: MoE poisons graph capture). The unwired-but-complete GPU dispatch code inflates the crate, suggests features that don't run, and still gets compiled/reviewed per change — classic "complete but test-only" (§3B).

**Cross-cutting speculative/PP/disagg subsystems** (DSpark, MTP, EAGLE3, pipeline-parallel, GDS, disagg bloom/coherence/lookup) are built and compiled into production binaries but unreachable from any production entrypoint — `SpeculativeCausalLm::auto` is always constructed with `draft=None` from the only production registration path, so `Strategy::DSpark`/`NativeMtp` arms in `decode_one` are dead in prod, and `GRIM_PP_SIZE > 1` hard-panics in `Engine::new` (grim-engine/src/lib.rs:541-549).

---

## 1. Unnecessary `unwrap()` / `expect()` calls

### Severity tiers

- **T1 — per-token/per-layer hot path, panics on model/session state:** can kill the server mid-generation.
- **T2 — lock poisoning contrary to codebase convention** (`.lock().unwrap_or_else(|e| e.into_inner())` is the house pattern): a poisoned mutex panics a request instead of recovering.
- **T3 — untrusted input / fallible IO.**
- **T4 — cleanup:** `?` available in a `Result` fn.

### T1 — hot-path panics on model/session state

| # | File:line | Code | Fix |
|---|-----------|------|-----|
| 1 | `crates/grim-models/transformer/src/block.rs:774,779,801` | `.as_ref().expect("dense FFN enabled")` on `w_gate/w_up/w_down` in `forward_with_kv_paged` (MoE-vs-dense invariant spread across struct) | `ok_or_else(...)?` |
| 2 | `crates/grim-models/transformer/src/block.rs:965,971` | `.expect("norm_x is RocmStorage on fused path")`, `.expect("act_q81 is RocmStorage")` in `fused_qkv_dot4_decode` | `?` |
| 3 | `crates/grim-models/transformer/src/block.rs:1056,1062` | same two expects in `fused_gate_up_dot4_decode` | `?` |
| 4 | `crates/grim-models/transformer/src/block.rs:1565` | `cache.past_dev.as_ref().unwrap()` in `device_graph_decode_attention` | capture binding before branch, or `ok_or?` (low severity) |
| 5 | `crates/grim-models/transformer/src/qwen35.rs:541,548,600,601` | `cache.{k,v}_device.as_ref().unwrap()` in `Qwen35Block::forward` | `if let (Some(k), Some(v)) = ...` |
| 6 | Session downcast expects in `fn forward(...) -> Result<Tensor>`: `qwen35.rs:921`, `chameleon.rs:376`, `deepseek.rs:416`, `falcon.rs:439`, `falcon_h1.rs:295`, `gemma.rs:424`, `gpt2.rs:424` | `.expect("... model_state must be ...")` — a corrupt/restored session kills the process | `.ok_or_else(...)?` |
| 7 | `crates/grim-models/transformer/src/solar_open2.rs:236,251,252` | `self.llama_block.as_ref().unwrap()`, `self.delta_net.as_ref().unwrap() ×2` in `forward` | `ok_or?` |
| 8 | `crates/grim-engine/src/streaming_forward.rs:684,690,722,728,771,777` | `w_gate/w_up/w_down.as_ref().expect("dense FFN")` ×6 (training forward) | `ok_or?` |
| 9 | `crates/grim-engine/src/lib.rs:2335,2377` | `self.decode_graphs.get_mut(&k).unwrap()`, `get(&k).unwrap()` — graph-slot miss panics instead of eager fallback the surrounding code already implements | `let Some(g) = ... else { return self.drive_forward(...) }` |
| 10 | `crates/grim-models/transformer/src/shared_attention.rs:44,50` | downcast expects in `fused_qkv_dot4_decode` | `?` |
| 11 | `crates/grim-models/transformer/src/shared_attention.rs:716` | `a.shape().dims().last().expect("non-empty tensor")` in `concat_rows_on_device` | `ok_or?` |
| 12 | `crates/grim-models/transformer/src/kv_attention.rs:214,221,229,230` | `arena.{k,v}_dev.as_ref().unwrap()` ×4 — panics when arena never grew (`tokens_this_step == 0` edge) | `ok_or?` |

### T2 — lock poisoning vs house convention

| # | File:line | Code |
|---|-----------|------|
| 13 | `crates/grim-scheduler/src/lib.rs:105,130,161,170` | `throughput_estimate.lock().unwrap()` ×4 (admission per request) |
| 14 | `crates/grim-scheduler/src/readiness_dispatch.rs:150,164,170,177` | `arrival_counter`/`ready_set.lock().unwrap()` ×4 (per microbatch) |
| 15 | `crates/grim-server/src/lib.rs:1481,1491` | `state.tokenizer.lock().unwrap()` ×2 (per request) |
| 16 | `crates/grim-backend-rocm/src/device/capability_profiler.rs:83,114,143` | `.expect("CapabilityProfiler lock poisoned")` ×3 |

All: `.lock().unwrap_or_else(|e| e.into_inner())`.

### T3 — untrusted input / fallible IO

| # | File:line | Code |
|---|-----------|------|
| 17 | `crates/grim-backend-rocm/src/memory/storage.rs:155,189` | `self.device_ptr.unwrap()` in `write_host_f32{,_async}` (fn returns `Result`; siblings check `device_ptr_u64()`) — per-token host-write path | `ok_or_else(...)?` |
| 18 | `crates/grim-cli/src/run.rs:720, 1371` | `tokens.last().unwrap()` — panics on empty prompt (post-templating) | `?` / hoist check |
| 19 | `crates/grim-cli/src/run.rs:1316` | `stdin().read_line(...).unwrap()` — EOF panics interactive mode | `?` or `unwrap_or(0)` + break |
| 20 | `crates/grim-server/src/lib.rs:574` | `partial_cmp(...).unwrap()` on logits — NaN logit aborts server (debug-gated, low) | `unwrap_or(Ordering::Equal)` |

### T4 — where `?` is already available

| # | File:line | Code |
|---|-----------|------|
| 21 | `crates/grim-backend-rocm/src/device/cubecl.rs:127,151,177,213,249,298,446,610,774,832` | `client.read_one(oh).unwrap()` ×10 in fallback kernels (add/mul/silu_mul/embedding/rms_norm/softmax/qkv_attention/paged_attention/tree_attention/gptq_correction) these fns return `Vec<f32>`; should return `Result<Vec<f32>>` |
| 22 | `crates/grim-engine/src/pipeline_engine.rs:816` | `queue.pop_front().expect("front checked non-empty")` → `ok_or?` |
| 23 | `crates/grim-cli/src/run.rs:741-746, 1395-1412` | 4× `decode_input/decode_pos.as_ref().unwrap()` per token in both decode loops — guarded by `is_some()` but noisy | hoist once per iteration |

### Idiomatic (won't panic, but needless test-then-unwrap on hot path)

`if x.is_some() { x.as_ref().unwrap() }` fused-QKV pattern — rewrite as `if let Some(fused) = ...`:
`block.rs:629,768`, `chameleon.rs:174`, `commandr.rs:177`, `dots3_note.rs:167`, `gemma.rs:248`, `gptj.rs:151`, `hy_v4.rs:167`, `qwen35.rs:475`, `qwen38_flash_next.rs:378`, `falcon_h1.rs:357`.

### Justified (not findings)

Lock-poison `into_inner()` recoveries; unwraps inside `#[cfg(test)]`; init-time `.unwrap()` on local GGUF paths; `Response::builder().body(...).unwrap()`; infallible-by-construction patterns where the `Some` is assigned in the immediately preceding block; `scythe2.rs:190`, `engine/lib.rs:1396`, `model_loader.rs:4261` (documented invariants). WAV parser (`grim-server/src/audio.rs:51-61`): `try_into().unwrap()` on length-checked slices — cannot panic; optional hardening.

Total: **22 finding groups** (~60 individual sites) after excluding test code.

---

## 2. Dead code (file:line)

### 2.1 Dead subsystems (delete-or-wire candidates, ranked)

1. **`crates/grim-backend-rocm/src/device/segment_replay.rs` (whole file incl. `run_segment` :33) + env `GRIM_SEGMENT_GRAPH` (:47).** Opt-in segment-graph replay; `run_segment` has **zero call sites** — setting the env var changes nothing. Orphaned.
2. **`crates/grim-backend-rocm/src/sampler_zero_probe.rs` (whole file) + `GRIM_PROBE_ORDINAL`/`GRIM_PROBE_TRIALS` (:17,27).** Permanent test-only probe module compiled into production.
3. **`crates/grim-engine/src/pipeline_engine.rs` (whole module)** + **`crates/grim-kvtransport/src/activation_channel.rs` (`TcpActivationTransport` :27)** + env **`GRIM_PP_SIZE`** (engine lib.rs:146, panic at :541-549). Pipeline-parallel execution: compiles, has tests, panics at runtime if enabled. Only tests + benches touch it.
4. **`crates/grim-cli/src/service.rs:15-61`** — service-supervision subsystem (`RestartPolicy`, `HealthCheckConfig`, `ServiceConfig`, `ServiceStatus`, `reload_config`), all `#[allow(dead_code)]`, never wired to any subcommand.
5. **`crates/grim-kvtransport`** — complete but unused families:
   - `bitmask_index.rs` whole (`TierMask` :9, `ChunkEntry` :83, `BitmaskChunkIndex` :92 + methods 20-193)
   - `gds.rs` whole (`GdsTier` :15, `is_direct` :76, `promote_block` :133, `demote_block` :81) and `gds_ffi.rs` whole (`HipFileHandle` :13, `HipFileLib` :39, `load` :69, `probe_available` :57, `driver_open/close` :136/:142, register/deregister :148-170, `read_direct` :176, `write_direct` :191)
   - `pin_lease.rs` whole (`LeaseStatus` :10, `PinnedLease` :18, `PinLeaseMonitor` :29, `SharedPinLeaseMonitor` :100)
   - `lib.rs`: `RdmaPinnedRegion` :96, `grimvise_advise` :41, `compute_checksum` :564, `compute_checksum_bytes` :553, `retarget_block_positions` :368, `NvmeWeightStreamer` :1501 + `prefetch_layer_async` :1547, `commit_and_swap` :1620, `set_bandwidth_usage` :1629, `retrieve_unit` :1611, `get_unit_tier` :1605, `KvBlockHeader` :495, `KvWireTransport` :1717, `TcpWireTransport` :1726, `SharedMemWireTransport` :1750, `shm_root`/`shm_inbox_for_port`/`start_shm_inbox_poller` :1765/:1777/:2041, `start_kv_receiver_server_with_prompts` :1158, `LocalSpillManager` :121 (only `SharedSpillManager` is consumed).
6. **`crates/grim-disagg`** — `bloom.rs` whole (`BloomFilter` :6, insert :55, might_contain :68), `coherence.rs` whole (`InvalidationMsg` :10, `CacheCoherenceManager` :63 + methods), `lookup.rs` whole (`LookupClient` :8 + impl), and `lib.rs`: `RetryPolicy` :23, `ReMPMigrationBatch` :73, `PoolAssignment` :170, `LayerPipelinedKvStreamer` :225 + `stream_layer_block` :248, `DisaggRouterT` :375, `transfer_kv_p2p_direct` :538, `transfer_paged_cache_real` :581, `transfer_kv_colocated` :768, `with_pool` :427, `with_retry_policy` :440, `effective_role` :356, `take_prompt_tokens` :866. (Engine uses only `KvReceiverServer`/`DisaggOrchestrator`/`PoolRole`.)
7. **`crates/grim-speculative/`** — `mamba_speculative.rs` whole (`MambaStepState` :8, `MambaSpeculativeEngine` :14 + impls); `confidence_scheduler.rs` whole (`ThroughputProfile` :9, `SpeculationConfig` :27, `ConfidenceScheduler` :45 + :58/:84); `depth_tuner.rs` (`SpeculativeDepthPidConfig` :6, controller :39/:50/:63/:68/:73/:79); `distill.rs` everything except `train_speculative_draft` (`AdaptationSignal` :9, `DraftRefreshInput/Outcome` :21/:34, `refresh_draft` :46, `compress_distill_report` :80); `speculative_wrapper.rs`: `plain` :71, `with_dspark` :88, `with_dspark_and_tuner` :107, `auto_with_native_mtp` :146; `eagle3_drafter.rs:36` `draft_block_with_fusion`. Production only ever instantiates strategy `Plain` (engine lib.rs:1139 passes `None,None,None`).
8. **`crates/grim-models/transformer/src/attention_dispatcher.rs`** (`AttentionDispatcher` :…, `dispatch_gqa` :63, `select_tier` :96; re-exported lib.rs:36-38) — tiered attention dispatch referenced only by `tests/transformer_e2e_integration.rs`.
9. **`crates/grim-backend-rocm/src/device/scythe_route.rs`** ring-routing helper set: `RingChannel` :34, `pack_gemm_descriptor` :135, `channel_for_ordinal` :245, `peer_link_type` :252/:253, `route_gemm_to` :262, `route_commfuse` :280, `route_peer_reduce` :344, `route_peer_broadcast` :411/:412, `route_peer_gather` :474/:475, `record_event_on` :540, `stream_wait_event` :563, `destroy_event` :586.
10. **`crates/grim-backend-rocm/src/device/parallel_comm.rs`** — `single_device` :113, `with_rccl` :143, `all_reduce_sum_storage` :212, `send_recv_p2p_device` :418.
11. **`crates/grim-backend-rocm/src/kernels/charon.rs`** — `plan_fused_dispatch` :1916, `plan_fused_dispatch_with_autotuner` :1933, `plan_grouped_dispatch` :1986, `validate_grouped_inputs` :2002, `validate_launch_inputs` :2047, `CharonVariant` :2087, `default_variant_table` :2231, `build_variant_table_from_autotuner` :2269, `routing_skew` :2341, `CharonSelector` :2361 (all `#[allow(dead_code)]`).

### 2.2 Dead individual items

| File:line | Item |
|---|---|
| `crates/grim-backend-rocm/src/kernels/compressed_gemm.rs:7,30` | `decode_msb_nbit`, `decode_msb_nbit_row` |
| `crates/grim-backend-rocm/src/device/device_compute.rs:1763,1994,2081,2144,2195,3688,3804,3852` | `launch_decode_gemm_f16`, `launch_wmma_gemm_b_transposed`, `launch_wmma_gemm_b_transposed_rdna4`, `launch_wmma_fused_dequant_q8_0`, `launch_wmma_fused_dequant_q8_0_fp16`, `launch_wmma_gemm_fp8_e4m3`, `launch_fp8_gemm_rdna4`, `launch_madam_update_f32` (zero call sites) |
| `crates/grim-backend-rocm/src/device/device_compute.rs:3506-3622` | 13× `launch_wmma_fused_dequant_{q4k,q5k,q2k,q3k,q6k,iq2xxs,iq2xs,iq2s,iq3xxs,iq3s,iq4nl,iq4xs}` launcher family |
| `crates/grim-backend-rocm/src/device/device_quant.rs:1687,3571` | `launch_fused_dequant_gemm_q4k` (comment admits "not yet wired"), `launch_fused_dequant_gemm_mxfp4` |
| `crates/grim-backend-rocm/src/device/device_routing.rs:604,684,735,806,995,3309` | MoE grouped dispatch family `moe_fused_grouped_dispatch_{w8a8_int8,w8a8_fp8,awq}`, `launch_charon_grouped_dispatch{,_fp8}`, `reorder_batch` |
| `crates/grim-backend-rocm/src/memory/view.rs:183` | `alloc_parent` |
| `crates/grim-backend-rocm/src/peer_access.rs:175` | `build_topology_matrix` |
| `crates/grim-backend-rocm/src/gptq_kernel.rs:225,234` | `compile_gptq_correction_kernel`, `compile_gptq_scale_fit_kernel` |
| `crates/grim-backend-rocm/src/autotune.rs:262,467,691,716,764` | `candidate_grid`, `occupancy_fields`, `set_cache_dir`, `list_keys`, `get_or_tune_moe` (last three test/bench-only) |
| `crates/grim-backend-rocm/src/fsdp.rs:240` | `fits_vram_budget` |
| `crates/grim-backend-rocm/src/graph_capture.rs:48,51,488,496,654` | `exec_handle`, `graph_handle`, `get_buffers`, `get_stream`, field `device_ordinal` |
| `crates/grim-backend-rocm/src/device/accel_features.rs:25,61,139` | `mfma_supported_on_device`, `wmma_supported_on_device`, `rccl_device_count` |
| `crates/grim-backend-rocm/src/device/roc_device.rs:217,463,895` | field `tuning`, `new_best`, `set_split_k_enabled` |
| `crates/grim-backend-rocm/src/device/jit_cache.rs:43` | field `file` |
| `crates/grim-backend-rocm/src/device/helpers.rs:230` | `_arc_pinned` |
| `crates/grim-backend-rocm/src/device/capability_profiler.rs:19,27` | consts `HIP_DEVICE_ATTR_THROTTLE_REMOVED`, `HIP_DEVICE_ATTR_TOTAL_CONST_MEM` |
| `crates/grim-backend-rocm/src/device/handles.rs:133` | `hipGetLastError` binding never invoked |
| `crates/grim-backend-rocm/src/perf_gate.rs:221` | `fails_ci` |
| `crates/grim-backend-rocm/src/quantization.rs:69` | `QuantMode::n_per_block` |
| `crates/grim-backend-rocm/src/rccl.rs:167,514,823` | `raw_comm`, field `comms` (feature-gated read), `scale_gradients` |
| `crates/grim-backend-rocm/src/speculative.rs:244,612,617` | field `pickup`, `stats_mut`, `reset_stats` |
| `crates/grim-backend-rocm/src/fusion.rs:71` | `with_enabled` |
| `crates/grim-backend-rocm/src/decode_graph_buffers.rs:819,832` | `write_input_async`, `write_input_batch_async` |
| `crates/grim-backend-rocm/src/trace.rs:379` | `_self_file_preview_used_in_warnings_only` |
| `crates/grim-models/transformer/src/kv_attention.rs:118,129,180` | `as_3d`, `as_2d`, `arena_append` |
| `crates/grim-models/transformer/src/native_mtp.rs:127,143` | `MtpLayer::forward_tensor`, `forward_step` (prod uses `forward_step_tensor`) |
| `crates/grim-models/transformer/src/muse_glimmer.rs:213,871` | const `MUSE_GLIMMER_30B_TENSOR_KEYS`, `embed_token` |
| `crates/grim-models/transformer/src/bailingmoe3.rs:148` | const `LING3_TINY_TENSOR_KEYS` |
| `crates/grim-models/transformer/src/falcon_h1.rs:36` | `head_group` |
| `crates/grim-models/transformer/src/lfm2_graph.rs:851` | `write_batch_embeddings` |
| `crates/grim-quant/src/lib.rs:2472,2785,2788,5747` | `quant_packed_symmetric`, const `GPTQ_PROXY_COLUMN_GROUP`, `BlockQuantization`, `gemm_q8_0_packed_scalar` |
| `crates/grim-engine/src/scythe2.rs:1194,1244,1302,1394` | `submit_col_gemm`, `submit_add`, `launch_resident`, `wait_completed` |
| `crates/grim-engine/src/scythe_ab.rs:8` | `default_results_path` |
| `crates/grim-engine/src/lib.rs:1018,3189` | `clear_latency_trace`, `enqueue_remote_prefill_request` |
| `crates/grim-engine/src/pipelines/moe_prefill_pipeline.rs:110` | `execute_pipelined` |
| `crates/grim-scheduler/src/lib.rs:629` | `plan_hybrid_attention_step` (see §3) |
| `crates/grim-cli/src/recipe.rs:11,210` | fields `recipe_version`, `description` |
| `crates/grim-cli/src/oxidizer.rs:807,842` | `quant_format_for_bitwidth`, `materialize_f32` (bench-only) |
| `crates/grim-cli/src/tui/worker.rs:887` | `build_prompt_ids` |
| `crates/grim-cli/src/plugin.rs:118` | `list_plugins` |
| `crates/grim-server/src/tool_parse.rs:12,124,138` | field, `RunawayReason` + `as_str` |
| `crates/grim-format/src/bank.rs:32,66,81`; `gguf.rs:631,1865`; `torch.rs:250`; `tokenizer.rs:1512` | `mmap_lazy`, `fill_from_disk`, `fill_from_slice`; `set_v2`, `map_gguf_dtype_to_grim`; `Val`; `char_byte_offset` |

---

## 3. Complete but unwired — only tests/benches call it

Definition-level evidence: production-reachable call sites = zero; only `#[cfg(test)]`/`tests/`/`benches/` reference.

| Item | File:line | Assessment |
|---|---|---|
| `Engine::register_with_dspark` | `crates/grim-engine/src/lib.rs:1098` | The ONLY production-shaped entry to `Strategy::DSpark`; called only by lib.rs:3815 (test). Speculative decoding (§5.3.2 DSpark) is dead in production. |
| `Engine::register_native_mtp_model` | `crates/grim-engine/src/lib.rs:1165` | MTP registration; only tests (:4209). |
| `Engine::register_eagle3_model` | `crates/grim-engine/src/lib.rs:1197` | reachable only via `load_and_register_speculative` … |
| `Engine::load_and_register_speculative` | `crates/grim-engine/src/lib.rs:1208` | …which has **zero callers**. Speculative model loading is complete and never dispatched from server/CLI model loading. |
| `SpeculativeCausalLm::{auto with None×3}` — connsequence | `(grim-engine/src/lib.rs:1139)` | Production `decode_one` can only ever be `Strategy::Plain`; `DSpark`/`NativeMtp` arms (`speculative_wrapper.rs:247+`) unreachable; server's `GRIM_SPEC` telemetry flag then always reports "disabled". |
| Confidence/PID adaptive speculation | `crates/grim-speculative/src/confidence_scheduler.rs:9-176`, `depth_tuner.rs:6-79` | Fully implemented adaptation loop; wired into nothing. |
| Draft refresh feedback loop | `crates/grim-speculative/src/distill.rs:46,80` | Only `train_speculative_draft` (one-shot) is live; the feedback loop is open. |
| `MtpLayer::forward_tensor`/`forward_step` | `crates/grim-models/transformer/src/native_mtp.rs:127,143` | CPU/tensor MTP step variants; production uses `forward_step_tensor`; the named pair are dead siblings. |
| `AttentionDispatcher` + `dispatch_gqa`/`select_tier` | `crates/grim-models/transformer/src/attention_dispatcher.rs:…,63,96` | Tiered GQA/MLA/Sage attention dispatch — complete module, only called by `tests/transformer_e2e_integration.rs`. No model forward calls it. |
| `plan_hybrid_attention_step` | `crates/grim-scheduler/src/lib.rs:629` | Documented "WI 3.4.1/3.4.5 hybrid CPU/GPU attention offload"; only callers: own `#[cfg(test)]` (:674,690,698,715). Never dispatched from grim-engine's attention path. |
| scythe2 ring-resident API | `crates/grim-engine/src/scythe2.rs:1194,1244,1302,1394` | Ring-resident kernel submission; only exercised by `tests/scythe_ring_loop.rs` under `GRIM_SCYTHE_RING_RESIDENT`. |
| `Engine::enqueue_remote_prefill_request` | `crates/grim-engine/src/lib.rs:3189` | Test-only. |
| PP execution stack | `crates/grim-engine/src/pipeline_engine.rs` (whole) + `grim-kvtransport/src/activation_channel.rs:27` | See §2.1.3 — panic-gated; only tests/benches run it. |
| `grim-format::bank` lazy-mmap weight banking | `bank.rs:32,66,81` | full `FillFlags` path; the loader never uses it. |
| MoE grouped-GEMM dispatchers | `device_routing.rs:604,684,735` + charon family (§2.1.11) | Complete GPU MoE dispatch (w8a8 int8/fp8, AWQ) — production MoE stays on host top-k + plain GEMM (see who-dat.md #2.3). This is the biggest MoE-specific unwired asset. |
| `RingChannel`/route_* P2P set (§2.1.9), `parallel_comm` set (§2.1.10) | scythe_route.rs / parallel_comm.rs | Multi-GPU routing/coordinator APIs; production TP goes through different primitives. |
| MoE autotune entry | `autotune.rs:764 get_or_tune_moe` (+ `set_cache_dir` :691, `list_keys` :716) | Only moe_autotune_gpu tests/benches. |

---

## 4. Env-gate leftovers (supporting evidence)

Suspicious/leftover knobs (live hot gates omitted):

| Env var | File:line | Verdict |
|---|---|---|
| `GRIM_SEGMENT_GRAPH` | `device/segment_replay.rs:47` | dead — nothing calls `run_segment` |
| `GRIM_PROBE_ORDINAL` / `GRIM_PROBE_TRIALS` | `sampler_zero_probe.rs:17,27` | test-only probe compiled in prod |
| `GRIM_SPEC` | `server/lib.rs:3812,5416` | telemetry-only, decorates status JSON; never gates behavior |
| `GRIM_PP_SIZE` | `engine/lib.rs:146` | >1 panics in `Engine::new`; selects a panic |
| `GRIM_DOT_GEMV_LEGACY` | `device_quant.rs:393` | escape hatch for long-retired dot2 path |
| `GRIM_FUSED_QKV` vs `GRIM_QKV_FUSED` | `block.rs:474`, `gemma.rs:104`, `shared_attention.rs:144` vs `shared_attention.rs:195` | duplicate spellings for the same switch — one is leftover |
| `GRIM_SCYTHE_RING_RESIDENT` | `grim-engine/tests/scythe_ring_loop.rs:176` | test-only knob |
| `GRIM_TEST_FREE_DEVICE_BYTES` | `engine/lib.rs:71, 4929` | test hook in production engine |
| `GRIM_DBG_ATTN`, `GRIM_TRACE_FUSED_QKV`, `GRIM_HIP_TRACE`, `GRIM_ALLOC_TRACE`, `GRIM_QMM_TRACE`, `GRIM_STEP_TRACE`, `GRIM_DEBUG_PROMPT`, `GRIM_COMPRESS_PROMPT`, `GRIM_CPU_SAMPLER`, `GRIM_TOKEN_PACING_MS` | various | undocumented ad-hoc debug switches — fine to keep, worth documenting |
| `GRIM_Q4K_TILED`/`Q5K_TILED`/`Q8_0_TILED`/`IQ4XS_TILED`, `GRIM_MXFP*`, `GRIM_ROPE_DEV_BASE`, `GRIM_RMSNORM_ROPE`, `GRIM_ROWS4_MIN_N`, `GRIM_FP16_*`, `GRIM_WAVEFRONT_SIZE` | various device_quant/device_compute | tuning knobs; several target launchers that are themselves dead (§2.2) — knobs that tune unreachable code |

---

## 5. Recommendations (ranked)

1. **T1 unwrap sweep (§1, items 1-12):** mechanical `expect → ok_or([...])?` on ~25 sites; zero behavior change, removes server-kill panic paths. Add a regression test: session-state downcast with a deliberately wrong state vec must `Err`, not panic (`block.rs`/`qwen35.rs` forward).
2. **T2 lock convention sweep (items 13-16):** 13 sites, one mechanical pattern; matches house style.
3. **Delete-or-wire the dead MoE/PP/speculative mass (§2.1 + §3):** either wire `moe_fused_grouped_dispatch_*` + charon into `shared_moe.rs` (pairs with who-dat.md P0-3), or delete the rest. Either direction shrinks review surface and compile time. Highest value: (a) delete segment_replay + probes, (b) decide PP (wire properly or delete pipeline_engine + activation_channel + GRIM_PP_SIZE), (c) decide speculative (wire `load_and_register_speculative` into server model load or delete the strategy arms).
4. **Deduplicate env gates:** collapse `GRIM_FUSED_QKV`/`GRIM_QKV_FUSED`; document the surviving debug knobs.
5. **After deletions:** rerun `cargo check --workspace` — several `#[allow(dead_code)]` annotations then disappear with their targets.

## 6. Caveats

- One `GRIM_…`-flag-election (e.g. a WIP feature branch flag) may be intentional future work; where upstream docs (plans/, docs/) reference them, defer to those plans before deleting.
- Zero-caller evidence is workspace-wide grep on this commit; external consumers of `grim-*` as libraries would re-mark some items as API surface — none of these crates are published.

*End of review.*
