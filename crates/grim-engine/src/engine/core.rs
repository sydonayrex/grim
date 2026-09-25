//! Construction and top-level config.

use crate::*;

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let mut pool = KvBlockPool::new(
            config.block_pool_capacity,
            config.num_kv_heads,
            config.head_dim,
        );

        // KV-cache quantization (§kv-int8). `EngineConfig.kv_compressor` takes precedence; otherwise honor `GRIM_KV_QUANT=int8` which attaches a
        // Lloyd-Max int4/int8 compressor so the paged KV pool compresses admitted blocks before spill.
        let mut compressor: Option<Arc<dyn grim_kvquant::KvCompressor>> =
            config.kv_compressor.clone();
        if compressor.is_none() {
            if let Ok(mode) = std::env::var("GRIM_KV_QUANT") {
                match mode.trim().to_ascii_lowercase().as_str() {
                    "int8" => {
                        let cfg = grim_kvquant::KvQuantConfig {
                            key_bits: 3,
                            value_bits: 8,
                            group_size: 64,
                            qk_compute_bits: 8,
                        };
                        compressor = Some(Arc::new(grim_kvquant::LloydMaxCompressor::new(cfg)));
                        log::info!(
                            "[grim-engine] kv-int8: attached LloydMaxCompressor (key_bits=3, value_bits=8, group=64)"
                        );
                    }
                    "int4" => {
                        let cfg = grim_kvquant::KvQuantConfig {
                            key_bits: 2,
                            value_bits: 4,
                            group_size: 64,
                            qk_compute_bits: 8,
                        };
                        compressor = Some(Arc::new(grim_kvquant::LloydMaxCompressor::new(cfg)));
                        log::info!(
                            "[grim-engine] kv-int8: attached LloydMaxCompressor (key_bits=2, value_bits=4, group=64)"
                        );
                    }
                    "off" | "" => {}
                    other => eprintln!(
                        "[grim-engine] GRIM_KV_QUANT='{other}' not recognized (expected int8|int4|off)"
                    ),
                }
            }
        }
        if let Some(comp) = &compressor {
            pool.attach_compressor(comp.clone());
        }

        // WI-HYBRID-ATTENTION-OFFLOAD (Phase 1):
        // Attach a SharedSpillManager when GRIM_KV_SPILL=1 (or true/on).
        let kv_spill_enabled = std::env::var("GRIM_KV_SPILL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on"))
            .unwrap_or(false);
        if kv_spill_enabled {
            let scratch_dir = std::env::var("GRIM_KV_SPILL_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| {
                    std::env::temp_dir().join(format!("grim_kv_spill_{}", std::process::id()))
                });
            let block_elems = BLOCK_SIZE * config.num_kv_heads * config.head_dim;
            match grim_kvtransport::SharedSpillManager::new(scratch_dir.clone(), block_elems) {
                Ok(spill_mgr) => {
                    log::info!(
                        "[grim-engine] attached SharedSpillManager (scratch: {:?}, block_elems: {})",
                        scratch_dir,
                        block_elems
                    );
                    pool.attach_spill(std::sync::Arc::new(spill_mgr));
                }
                Err(e) => {
                    log::warn!("[grim-engine] failed to create SharedSpillManager: {e}");
                }
            }
        }

        // Tensor-parallel bootstrap (§c-tp-scope, WI-TP-4).
        // Design A (multi-process): one OS process per rank.
        let tp_config: Option<grim_nn::TensorParallelConfig> = if config.tp_size > 1 {
            let tp = grim_nn::TensorParallelConfig::from_env().unwrap_or_default();
            if let Err(msg) = tp.validate() {
                // TP requested but structurally invalid - hard fail so the operator sees the mismatch immediately instead of silently loading the wrong shard.
                // Engine::new returns Self (not Result), so we panic; this is an unrecoverable config error.
                log::info!(
                    "[grim-engine] INVALID TP config (GRIM_TP_SIZE={}): {msg}",
                    config.tp_size
                );
                panic!(
                    "invalid tensor-parallel configuration (GRIM_TP_SIZE / GRIM_TP_RANK): {msg}"
                );
            }
            // Resolve this rank's GPU ordinal (mirrors model_loader's logic)
            // for diagnostics — the actual device is built in the loader.
            let gpus: Vec<usize> = std::env::var("GRIM_GPUS")
                .ok()
                .map(|s| {
                    s.split(',')
                        .filter_map(|t| t.trim().parse::<usize>().ok())
                        .collect()
                })
                .unwrap_or_default();
            let my_ordinal = gpus.get(tp.rank).copied().unwrap_or(tp.rank);
            log::info!(
                "[grim-engine] TP rank {}/{} on ordinal {} (device built in model_loader)",
                tp.rank,
                tp.world_size,
                my_ordinal
            );
            Some(tp)
        } else {
            None
        };
        let block_pool = Arc::new(std::sync::Mutex::new(pool));

        // Disaggregation: start a background KV receiver server when configured for Decode or Colocated roles.
        // The receiver writes incoming KV blocks into the engine's block_pool, enabling cross-node KV handoff.
        let kv_receiver = if let Some(ref dc) = config.disagg_config {
            let role = dc.role;
            let listen_addr = if role == grim_disagg::PoolRole::Decode {
                &dc.decode_addr
            } else {
                &dc.prefill_addr
            };
            match grim_disagg::KvReceiverServer::new(listen_addr, block_pool.clone()) {
                Ok(srv) => {
                    log::info!(
                        "[grim-engine] disagg: KV receiver server started on {} (role={:?})",
                        srv.listen_addr(),
                        role
                    );
                    Some(srv)
                }
                Err(e) => {
                    log::warn!(
                        "[grim-engine] disagg: failed to start KV receiver on {listen_addr}: {e}"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Disaggregation cluster orchestrator (§5.6): use the caller's, or
        // auto-construct from disagg_config, or none.
        let disagg_orchestrator = match (&config.disagg_orchestrator, &config.disagg_config) {
            (Some(o), _) => Some(o.clone()),
            (None, Some(dc)) => Some(Arc::new(std::sync::Mutex::new(
                grim_disagg::DisaggOrchestrator::new(dc.clone()),
            ))),
            (None, None) => None,
        };
        let disagg_effective_role = std::sync::Mutex::new(
            config
                .disagg_config
                .as_ref()
                .map(|dc| dc.role)
                .unwrap_or(grim_disagg::PoolRole::Colocated),
        );

        let admission =
            grim_scheduler::AdmissionController::new(config.target_ttft_ms, config.target_itl_ms);
        let mut scheduler = grim_scheduler::Scheduler::new(
            config.max_batched_tokens,
            config.max_num_seqs,
            admission,
        );
        scheduler.determinism_mode = config.determinism_mode;
        // §5.2 real KV pressure: pool occupancy feeds the scheduler's pressure signal
        // so preemption/chunked draining react to actual KV exhaustion, not just prompt-token sums.
        scheduler.set_kv_pressure(Arc::new(grim_scheduler::PoolKvPressure::new({
            let block_pool = block_pool.clone();
            move || {
                let guard = block_pool.lock().unwrap_or_else(|e| e.into_inner());
                guard.used_count() as f32 / guard.capacity().max(1) as f32
            }
        })));
        let target_ttft = config.target_ttft_ms as f64;
        let target_itl = config.target_itl_ms as f64;

        let is_multi_gpu = tp_config
            .as_ref()
            .map(|tp| tp.world_size > 1)
            .unwrap_or(false)
            || config.tp_size > 1;
        let scythe_inference_flag = std::env::var("GRIM_SCYTHE_INFERENCE")
            .map(|v| v == "1" || v == "true" || v == "on")
            .unwrap_or(false);

        // WI-INF1: the profiler is the only thing constructed on the default path, and only when more than one GPU is visible.
        // A single-GPU box pays zero probe cost (gate: test_single_gpu_capability_profiler_is_none).
        let capability_profiler = if is_multi_gpu || scythe_inference_flag {
            Some(Arc::new(grim_backend_rocm::CapabilityProfiler::new()))
        } else {
            None
        };

        // WI-INF2: the controller routes activations across GPUs *in this process* (SCYTHE-2's P2P-ring execution model), so it is armed on the count of ROCm devices visible here - not on `TP world_size`, which under Design A counts one GPU per OS process.
        // Fewer than two visible GPUs leaves nothing to route between; the controller stays `None` even.
        const SCYTHE_NUM_LAYERS_PLACEHOLDER: usize = 32;
        let visible_gpus = capability_profiler
            .as_ref()
            .map(|p| p.capabilities().len())
            .unwrap_or(0);
        let scythe_ctrl = if scythe_inference_flag && visible_gpus > 1 {
            log::info!(
                "[scythe2] inference routing armed over {visible_gpus} visible GPUs \
                 (GRIM_SCYTHE_INFERENCE=1)"
            );
            Some(crate::scythe2::C2plrController::new(
                SCYTHE_NUM_LAYERS_PLACEHOLDER,
                visible_gpus,
                config.target_itl_ms as f64,
            ))
        } else {
            None
        };

        Self {
            config,
            scheduler,
            block_pool,
            models: HashMap::new(),
            sessions: HashMap::new(),
            adapters: HashMap::new(),
            last_outcomes: HashMap::new(),
            request_rng: HashMap::new(),
            request_model_ids: HashMap::new(),
            request_adapters: HashMap::new(),
            request_input_ids: HashMap::new(),
            prefill_progress: HashMap::new(),
            request_last_token: HashMap::new(),
            self_tuning_controller: grim_scheduler::SelfTuningController::new(
                target_ttft,
                target_itl,
            ),
            tuned_speculative_block_len: 5,
            tuned_kv_compression_bit_width: 4,
            tokens_per_sec_ema: 0.0,
            total_tokens_generated: 0,
            prefill_tokens_per_sec_ema: 0.0,
            total_tokens_prefilled: 0,
            accepted_tokens_total: 0,
            last_ttft_ms: None,
            last_itl_ms: None,
            tp_config,
            kv_receiver,
            disagg_orchestrator,
            disagg_effective_role,
            capability_profiler,
            scythe_ctrl,
            scythe_replicas: HashMap::new(),
            scythe_pin: HashMap::new(),
            scythe_pin_cooldown: Vec::new(),
            scythe_vram_waitlist: Vec::new(),
            radix_enabled: std::env::var("GRIM_RADIX")
                .map(|v| v != "0" && v != "false" && v != "off")
                .unwrap_or(true),
            radix_seeded_blocks: HashMap::new(),
            radix_registered_blocks: HashMap::new(),
            radix_lookups: 0,
            radix_hit_requests: 0,
            radix_hit_tokens: 0,
            decode_graph_input_buffers: HashMap::new(),
            graph_capture_logits: HashMap::new(),
            decode_graphs: HashMap::new(),
            request_session: HashMap::new(),
            session_slot_last_use: HashMap::new(),
            batch_graph_pools: HashMap::new(),
            batch_bucket_slots: HashMap::new(),
        }
    }

    /// Tensor-parallel configuration resolved once at `Engine::new` from the env.
    /// `GRIM_TP_RANK` selects this process's shard index; `GRIM_TP_SIZE` selects the world size.
    pub fn tp_config(&self) -> Option<grim_nn::TensorParallelConfig> {
        self.tp_config
    }
}
