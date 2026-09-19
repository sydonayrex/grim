//! Scheduler loops: prefill/decode drives, graph capture, bucket groups, step.

use crate::*;

impl Engine {
    /// Run one engine iteration. For each scheduled prefill or decode request,
    /// drive the speculative wrapper against the request's session and capture per-request outcomes.
    pub fn tick(&mut self) -> Result<grim_scheduler::SchedulerOutput> {
        let tick_start = Instant::now();

        // Background capability profiler tick check (~100ms cadence)
        if let Some(ref profiler) = self.capability_profiler {
            if profiler.age() >= Duration::from_millis(100) {
                profiler.tick();
                // WI-INF2: pull the (possibly bumped) capability epoch into the placement cache.
                // This is the existing mode-B staleness path - `sync_epoch` clears the fast slots so the.
                if let Some(ref mut ctrl) = self.scythe_ctrl {
                    ctrl.cache.sync_epoch(grim_backend_rocm::current_epoch());
                }
            }
        }

        // WI-SB2: give parked requests a chance to place before this pass's
        // admission, so freed VRAM is picked up in the same tick.
        self.retry_scythe_vram_waitlist();

        // Disagg failover (§5.6): refresh the effective role each tick from observed peer heartbeats.
        // A silent peer fails the node over to colocated execution, which gates the remote KV.
        {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            self.disagg_evaluate_failover(now_ms);
        }

        let output = self.scheduler.schedule();
        let schedule_elapsed = tick_start.elapsed();

        // Run prefill, then decode in a single deterministic pass - for §5.3 correctness, prefills share block pool and decode uses the KV they just wrote.
        // We process them in the order the scheduler produced them so a paused predicate is.
        let prefill = output.prefill_ids.clone();
        let had_prefill = !prefill.is_empty();
        let mut prefill_elapsed = Duration::ZERO;
        let mut prefill_tokens_in_tick = 0usize;
        for id in prefill {
            if self.scheduler.is_paused(id) {
                continue;
            }
            let pf_start = Instant::now();
            let n_prefilled = self.drive_prefill(id)?;
            prefill_tokens_in_tick += n_prefilled;
            prefill_elapsed += pf_start.elapsed();
        }
        let mut decode_elapsed = Duration::ZERO;
        let mut total_accepted = 0usize;
        let mut decode_count = 0usize;

        // Process active decode steps across all scheduled requests via step_batch (WI-X1)
        let mut decode_items = Vec::new();
        for &id in &output.decode_ids {
            if !self.scheduler.is_paused(id) {
                if let Some((model_id, _)) = self.model_for_request(id) {
                    let start_pos = self.sessions.get(&id).map(|s| s.current_pos()).unwrap_or(0);
                    let next_token = self
                        .request_last_token
                        .get(&id)
                        .copied()
                        .unwrap_or(start_pos as u32);
                    let ids = grim_backend_cpu::cpu_tensor(
                        vec![next_token as f32],
                        grim_tensor::Shape::new(vec![1]),
                    );
                    let positions = grim_backend_cpu::cpu_tensor(
                        vec![start_pos as f32],
                        grim_tensor::Shape::new(vec![1]),
                    );
                    decode_items.push((id, model_id.to_string(), ids, positions));
                }
            }
        }

        if !decode_items.is_empty() {
            let dec_start = Instant::now();
            let batch_refs: Vec<(u64, &str, &grim_tensor::Tensor, &grim_tensor::Tensor)> =
                decode_items
                    .iter()
                    .map(|(id, m, ids, pos)| (*id, m.as_str(), ids, pos))
                    .collect();
            let batch_results = self.step_batch(&batch_refs)?;
            decode_count = batch_results.len();
            for (id, outcome) in batch_results {
                total_accepted += outcome.accepted_tokens;
                self.last_outcomes.insert(id, outcome);
            }
            decode_elapsed = dec_start.elapsed();
        }

        // MIN-4: Record actual forward-pass wall time, not schedule time.
        // TTFT = time to first token (prefill), ITL = inter-token latency (decode).
        let ttft_ms = prefill_elapsed.as_secs_f64() * 1000.0;
        if had_prefill {
            self.last_ttft_ms = Some(ttft_ms);
        }
        let itl_ms = if decode_count > 0 {
            let itl = decode_elapsed.as_secs_f64() * 1000.0 / decode_count as f64;
            self.last_itl_ms = Some(itl);
            itl
        } else {
            0.0
        };
        self.self_tuning_controller.record_ttft(ttft_ms);
        self.self_tuning_controller.record_itl(itl_ms);

        // MIN-3: Apply ALL tuned params, not just max_batched_tokens and
        // chunked_prefill_size.
        let tuned_params = self.self_tuning_controller.tune_all();
        self.scheduler.max_batched_tokens = tuned_params.max_batched_tokens;
        self.scheduler.chunked_prefill_size = tuned_params.chunked_prefill_size;
        // Speculative block length and KV compression bit width are stored
        // on the engine for the speculative wrapper to pick up at decode time.
        self.tuned_speculative_block_len = tuned_params.speculative_block_len;
        self.tuned_kv_compression_bit_width = tuned_params.kv_compression_bit_width;

        // Accumulate prefill tokens and update prefill tokens/sec EMA
        if prefill_tokens_in_tick > 0 && prefill_elapsed.as_secs_f32() > 0.0 {
            let inst_prefill_tps = (prefill_tokens_in_tick as f32) / prefill_elapsed.as_secs_f32();
            if self.prefill_tokens_per_sec_ema == 0.0 {
                self.prefill_tokens_per_sec_ema = inst_prefill_tps;
            } else {
                self.prefill_tokens_per_sec_ema =
                    0.7 * self.prefill_tokens_per_sec_ema + 0.3 * inst_prefill_tps;
            }
            self.total_tokens_prefilled += prefill_tokens_in_tick as u64;
        } else if prefill_tokens_in_tick > 0 {
            self.total_tokens_prefilled += prefill_tokens_in_tick as u64;
        }

        // WI-E2: accumulate accepted tokens for the acceptance-rate metric.
        self.accepted_tokens_total += total_accepted as u64;
        let _ = (schedule_elapsed, total_accepted);
        let tick_elapsed = tick_start.elapsed();
        if total_accepted > 0 && tick_elapsed.as_secs_f32() > 0.0 {
            let inst_tps = (total_accepted as f32) / tick_elapsed.as_secs_f32();
            if self.tokens_per_sec_ema == 0.0 {
                self.tokens_per_sec_ema = inst_tps;
            } else {
                self.tokens_per_sec_ema = 0.7 * self.tokens_per_sec_ema + 0.3 * inst_tps;
            }
            self.total_tokens_generated += total_accepted as u64;
        }
        Ok(output)
    }

    /// WI-M2 drift watch (gguf_multigpu_context_plan.md): hold the process-wide prefill latch up for the duration of the pass.
    /// While it is set, ANY HIP context switch to a non-zero device on any thread.
    /// Returns the number of newly prefilled prompt tokens processed in this pass.
    pub(crate) fn drive_prefill(&mut self, id: u64) -> Result<usize> {
        grim_backend_rocm::set_prefill_in_flight(true);
        let outcome = self.drive_prefill_inner(id);
        grim_backend_rocm::set_prefill_in_flight(false);
        outcome
    }

    pub(crate) fn drive_prefill_inner(&mut self, id: u64) -> Result<usize> {
        eprintln!("[pin-dbg] prefill req {id} session_recorded={}", self.request_session.contains_key(&id));
        // Chunked prefill (F9 follow-on): the scheduler may carry several running copies of `id` (one per
        // pass, each with the cumulative consumed count), so take the LATEST bound, not the first copy's.
        let mut prompt_tokens = None;
        let mut consumed_tokens = 0usize;
        for r in self.scheduler.running.iter().filter(|r| r.id == id) {
            prompt_tokens = Some(r.prompt_tokens);
            consumed_tokens = consumed_tokens.max(r.consumed_tokens);
        }
        let prompt_tokens = match prompt_tokens {
            Some(p) => p,
            None => return Ok(0),
        };
        if prompt_tokens == 0 {
            return Ok(0);
        }
        // Only the tokens the scheduler has budgeted but the engine has not yet prefilled run through the model.
        // Everything else (radix matching, KV registration, disagg handoff) still sees the full prompt below.
        let already0 = self.prefill_progress.get(&id).copied().unwrap_or(0);
        let target = consumed_tokens.min(prompt_tokens);
        // Build the full input_ids tensor: use real token IDs if provided,
        // otherwise fall back to synthetic position indices (0..prompt_tokens) for backward compatibility.
        let full_input: Vec<u32> = self
            .request_input_ids
            .get(&id)
            .cloned()
            .filter(|v| !v.is_empty() && v.len() == prompt_tokens)
            .unwrap_or_else(|| (0..prompt_tokens as u32).collect());
        let has_real_ids = self
            .request_input_ids
            .get(&id)
            .filter(|v| !v.is_empty() && v.len() == prompt_tokens)
            .is_some();

        // Layer 1.5 (session-continuity design): CONSUME the radix prefix
        // cache — it was registration-only before this and `seed_prefix` had
        // no callers. On the FIRST prefill pass of a request with real token
        // ids, claim matched prefix blocks (tree refcount +1), seed + hydrate
        // the session's paged KV from the shared pool, and start the prefill
        // cursor past the matched tokens.
        let mut already = already0;
        if self.radix_enabled && already0 == 0 && has_real_ids && !full_input.is_empty() {
            let (matched_blocks, matched_tokens, promoted) = {
                let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                // ponytail: match_prefix_promoting = match + LRU-promote in one call,
                // so a hit doesn't simultaneously make itself eviction-eligible.
                pool.match_prefix_promoting(&full_input)
            };
            self.radix_lookups += 1;
            // Seeding is only sound when (a) the model keeps all its state in
            // the paged KV pages (hybrid models with recurrent/conv state are
            // excluded via model_uses_hybrid_kv) and (b) at least one block
            // remains to prefill — the first decode token's logits come from the
            // last prefill pass. The skip is block-aligned: `matched_tokens` is
            // a multiple of BLOCK_SIZE.
            let skip_cap = full_input.len().saturating_sub(1);
            let skip_tokens = {
                let raw = matched_tokens.min(skip_cap);
                raw - (raw % grim_memory::BLOCK_SIZE)
            };
            let seed_ok = skip_tokens >= grim_memory::BLOCK_SIZE
                && !self.model_uses_hybrid_kv(id);
            if seed_ok {
                let seed_blocks = &matched_blocks[..skip_tokens / grim_memory::BLOCK_SIZE];
                // Claim the matched nodes for the lifetime of this request.
                {
                    let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                    pool.insert_prefix(&full_input[..skip_tokens], seed_blocks);
                }
                if let Some(session) = self.sessions.get_mut(&id) {
                    // `seed_prefix` refcounts the blocks and wires the block
                    // table; the pool's K/V bytes are staged lazily per layer
                    // on the session's first appends, when the model's real
                    // stride is known (the arena-bridge problem is unsolvable
                    // at engine level because EngineConfig kv geometry may
                    // differ from the registered model's).
                    if let Some(kv) = session.kv_mut() {
                        kv.seed_prefix(seed_blocks);
                    }
                    session.advance_pos(skip_tokens);
                    self.prefill_progress.insert(id, skip_tokens);
                    self.radix_seeded_blocks.insert(id, seed_blocks.len());
                    self.radix_hit_requests += 1;
                    self.radix_hit_tokens += skip_tokens as u64;
                    already = skip_tokens;
                    log::info!(
                        "[grim-engine] req {id} radix hit+reuse: skipping {} of {} prompt tokens ({} blocks, promoted={})",
                        skip_tokens,
                        full_input.len(),
                        seed_blocks.len(),
                        promoted,
                    );
                }
            } else if matched_tokens > 0 {
                log::info!(
                    "[grim-engine] req {id} radix match not consumed ({}/{} tokens; skip_tokens={}, hybrid={}, promoted={})",
                    matched_tokens,
                    full_input.len(),
                    skip_tokens,
                    self.model_uses_hybrid_kv(id),
                    promoted,
                );
            }
        }
        if target <= already {
            return Ok(0); // this pass budgeted no new prompt tokens
        }

        // The chunk actually prefilled this pass: tokens [already, target) with their true positions.
        // Models place KV by sequential append, so the session position after this call equals `target`.
        let chunk: Vec<u32> = full_input[already..target].to_vec();
        let chunk_len = chunk.len();
        let ids = grim_backend_cpu::cpu_tensor(
            chunk.iter().map(|&t| t as f32).collect::<Vec<f32>>(),
            grim_tensor::Shape::new(vec![chunk_len]),
        );
        let positions = grim_backend_cpu::cpu_tensor(
            (already..target).map(|t| t as f32).collect::<Vec<f32>>(),
            grim_tensor::Shape::new(vec![chunk_len]),
        );
        if let Some((model_id, _)) = self.model_for_request(id) {
            let model_id = model_id.to_string();
            let outcome = self.drive_forward(&model_id, id, &ids, &positions)?;
            // Record progress only after the forward succeeded — a failed
            // pass must retry the same chunk, not skip it.
            self.prefill_progress.insert(id, target);
            // `current_pos` is owned by the model/session - the underlying forward already advanced it via `session.advance_pos(seq_len)`.
            // The engine does *not* double-count.
            self.last_outcomes.insert(id, outcome);

            // Radix prefix cache: register computed KV blocks and semantic state anchors.
            // Only blocks beyond (seeded + previously registered) are inserted —
            // re-inserting matched/already-registered nodes would double their
            // tree refcount and leak their eviction eligibility (Layer 1.5).
            if self.radix_enabled && !full_input.is_empty() {
                let skip_blocks = self.radix_seeded_blocks.get(&id).copied().unwrap_or(0)
                    + self.radix_registered_blocks.get(&id).copied().unwrap_or(0);
                if let Some(session) = self.sessions.get(&id) {
                    if let Some(block_table) = session.block_table() {
                        if block_table.len() > skip_blocks {
                            let usize_blocks: Vec<usize> =
                                block_table.iter().map(|&b| b as usize).collect();
                            let registered_now = usize_blocks.len() - skip_blocks;
                            let mut pool =
                                self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                            pool.insert_prefix_with_recurrent_state_at(
                                &full_input,
                                &usize_blocks,
                                Vec::new(),
                                skip_blocks,
                            );
                            self.radix_registered_blocks
                                .insert(id, skip_blocks + registered_now);
                        }
                    }
                }
            }

            // Disaggregation handoff: if disagg_router is configured for Prefill role, stream real KV blocks generated during prefill over the network to the decode node.
            // Failover gate (§5.6): once the orchestrator fails this node over to Colocated (decode peer silent.
            if let Some(router) = &self.config.disagg_router {
                if router.pool_role == grim_disagg::PoolRole::Prefill
                    && self.disagg_effective_role() == grim_disagg::PoolRole::Prefill
                {
                    // The pool is shared across concurrent requests, so the handoff must carry only
                    // this request's physical blocks - a full-pool scan would leak other requests' KV cache.
                    let block_ids: Vec<usize> = self
                        .sessions
                        .get(&id)
                        .and_then(|s| s.block_table())
                        .map(|t| t.iter().map(|&b| b as usize).collect())
                        .unwrap_or_default();
                    if !block_ids.is_empty() {
                        if let Some(session) = self.sessions.get(&id) {
                            if let Some(kv) = session.kv_cache() {
                                let num_layers = kv.num_layers();
                                for layer in 0..num_layers {
                                    for &b_id in &block_ids {
                                        if let Some((k_slice, v_slice)) =
                                            kv.layer_block_slice(layer, b_id)
                                        {
                                            // Carry the block's valid token count on the wire;
                                            // a partially-filled tail block must not arrive marked as full.
                                            let num_tokens = kv
                                                .block_num_tokens(b_id)
                                                .unwrap_or(0)
                                                .min(k_slice.len());
                                            if let Err(e) = router.send_layer_block_remote(
                                                b_id,
                                                layer as u32,
                                                k_slice,
                                                v_slice,
                                                num_tokens,
                                            ) {
                                                log::warn!(
                                                    "[grim-engine] Disagg prefill KV transfer failed for req {id}, layer {layer}, block {b_id}: {e}"
                                                );
                                            } else {
                                                // A landed transfer proves the decode peer is alive (§5.6).
                                                self.disagg_record_peer_heartbeat(
                                                    grim_disagg::PoolRole::Decode,
                                                    std::time::SystemTime::now()
                                                        .duration_since(std::time::UNIX_EPOCH)
                                                        .map(|d| d.as_millis() as u64)
                                                        .unwrap_or(0),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // WI-HYBRID-ATTENTION-OFFLOAD (Phase 1):
            // Watermark policy: demote only prefix blocks beyond watermark (never the last N blocks).
            // Default keep_tail = 4 blocks (64 tokens).
            if let Ok(watermark_str) = std::env::var("GRIM_KV_WATERMARK_BLOCKS") {
                if let Ok(watermark) = watermark_str.parse::<usize>() {
                    let block_ids: Vec<usize> = self
                        .sessions
                        .get(&id)
                        .and_then(|s| s.block_table())
                        .map(|t| t.iter().map(|&b| b as usize).collect())
                        .unwrap_or_default();
                    if block_ids.len() > watermark {
                        let keep_tail = std::env::var("GRIM_KV_KEEP_TAIL_BLOCKS")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or(2);
                        let demote_end = block_ids.len().saturating_sub(keep_tail);
                        let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                        if pool.has_spill() {
                            for &bid in &block_ids[..demote_end] {
                                pool.demote_block(bid);
                            }
                        }
                    }
                }
            }

            // WI-HYBRID Layer 2: session-tagged requests pin their KV blocks
            // (idle-time pin) so the radix LRU can't reclaim the cached
            // prefix between turns under other traffic's pressure. Advisory
            // only — rollback/free still reclaims pinned pages.
            if self.request_session.contains_key(&id) {
                let pin_secs = session_pin_secs();
                let block_ids: Vec<usize> = self
                    .sessions
                    .get(&id)
                    .and_then(|s| s.block_table())
                    .map(|t| t.iter().map(|&b| b as usize).collect())
                    .unwrap_or_default();
                if !block_ids.is_empty() {
                    let mut pool =
                        self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                    pool.pin_blocks(&block_ids, pin_secs);
                    eprintln!(
                        "[pin-dbg] prefill pinned {block_ids:?} for {pin_secs}s; is_pinned now: {:?}",
                        block_ids.iter().map(|&b| pool.is_block_pinned(b)).collect::<Vec<_>>()
                    );
                }
            }

            return Ok(chunk_len);
        }
        Ok(0)
    }

    /// Drives a decode step for sequence `id`, recording the outcome step.
    pub fn drive_decode(&mut self, id: u64) -> Result<()> {
        let outcome = self.drive_decode_with_outcome(id)?;
        if let Some(outcome) = outcome {
            self.last_outcomes.insert(id, outcome);
        }
        Ok(())
    }

    pub(crate) fn drive_decode_with_outcome(&mut self, id: u64) -> Result<Option<StepOutcome>> {
        // Disaggregated decode: ensure required KV blocks are present in the local pool before executing the decode step.
        // When this is a Decode node, the KV cache was generated on the Prefill node.
        if let Some(ref router) = self.config.disagg_router {
            if router.pool_role == grim_disagg::PoolRole::Decode {
                let elem_per_token = self.config.num_kv_heads * self.config.head_dim;
                let block_elems = elem_per_token * BLOCK_SIZE;
                let req_blocks: Vec<usize> = self
                    .sessions
                    .get(&id)
                    .and_then(|s| s.block_table())
                    .map(|t| t.iter().map(|&b| b as usize).collect())
                    .unwrap_or_default();
                let num_layers = self
                    .sessions
                    .get(&id)
                    .and_then(|s| s.kv_cache())
                    .map(|kv| kv.num_layers())
                    .unwrap_or(1)
                    .max(1);
                for &block_id in &req_blocks {
                    let already_received = {
                        let pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                        pool.block_is_received(block_id)
                    };
                    if already_received {
                        continue;
                    }
                    let mut fetch_ok = true;
                    for layer in 0..num_layers {
                        match router.fetch_kv_block(block_id, layer as u32, block_elems) {
                            // V3 wire carries the block's valid token count; storing `block_elems
                            // / elem_per_token` instead would mark every (partially-filled) block as fully valid.
                            Ok((k_data, v_data, num_tokens)) => {
                                if layer == 0 {
                                    let mut pool =
                                        self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                                    pool.write_keys(block_id, &k_data, num_tokens);
                                    pool.write_values(block_id, &v_data);
                                }
                                if let Some(session) = self.sessions.get_mut(&id) {
                                    if let Some(kv) = session.kv_mut() {
                                        let _ =
                                            kv.write_layer_block(layer, block_id, &k_data, &v_data);
                                    }
                                }
                            }
                            Err(e) => {
                                // F3: a failed layer must not leave the block marked received (write_keys
                                // auto-marks on layer 0), or that block would attend stale pages forever.
                                fetch_ok = false;
                                log::warn!(
                                    "[grim-engine] Disagg decode KV fetch failed for req {id}, layer {layer}, block {block_id}: {e}"
                                );
                            }
                        }
                    }
                    {
                        let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                        pool.set_received(block_id, fetch_ok);
                    }
                }
            }
        }

        let start_pos = self.sessions.get(&id).map(|s| s.current_pos()).unwrap_or(0);
        // Use the previously sampled token if available, otherwise fall back
        // to position index for backward compatibility.
        let next_token = self
            .request_last_token
            .get(&id)
            .copied()
            .unwrap_or(start_pos as u32);

        let model_id = self.model_for_request(id).map(|(m, _)| m.to_string());
        let model_id = model_id.as_deref();

        // Graph capture path: when GRIM_CAPTURE_GRAPH=1 and the model is on a
        // ROCm device, use persistent GPU buffers for inputs so the graph can
        // be replayed with in-place updates (matching the verified semantics
        // from tests/test_hip_graph_capture.rs).
        let graph_capture = std::env::var("GRIM_CAPTURE_GRAPH")
            .map(|v| v != "0" && v != "false" && v != "off")
            .unwrap_or(true)
            && model_id.is_some_and(|mid| {
                self.models
                    .get(mid)
                    .map(|m| matches!(m.device, grim_tensor::dtype::Device::Rocm(_)))
                    .unwrap_or(false)
            });

        if graph_capture {
            let capture_key = format!("decode_req_{id}");
            let model_id = model_id.ok_or_else(|| Error::Config("no model for request".into()))?;
            // Get or create persistent GPU buffers for this request, update them
            // in-place with the new token/position, then clone the tensor handles
            // for the forward call (avoids borrow conflict with self).
            let (input_ids, positions) = {
                let buffers = self.get_or_create_graph_input_buffers(id, model_id)?;
                let ordinal = match buffers.input_ids.device() {
                    grim_tensor::dtype::Device::Rocm(ord) => *ord,
                    _ => return Err(Error::Backend("graph buffers must be on ROCm".into())),
                };
                // Update the persistent GPU buffers in-place with the new token/position.
                // Async H2D on the active stream, ordered vs later launches — no host sync.
                let rocm = grim_backend_rocm::device::roc_device::RocmDevice::shared(ordinal);
                rocm.write_f32_into_async(
                    &**buffers.input_ids.storage(),
                    &[next_token as f32],
                )?;
                rocm.write_f32_into_async(
                    &**buffers.positions.storage(),
                    &[start_pos as f32],
                )?;
                // Clone the tensor handles (Arc clones are cheap) to release the borrow.
                (buffers.input_ids.clone(), buffers.positions.clone())
            };
            let outcome = self.drive_forward_graph_capture(
                model_id, id, &input_ids, &positions, &capture_key,
            )?;
            Ok(Some(outcome))
        } else {
            let ids = grim_backend_cpu::cpu_tensor(
                vec![next_token as f32],
                grim_tensor::Shape::new(vec![1]),
            );
            let positions = grim_backend_cpu::cpu_tensor(
                vec![start_pos as f32],
                grim_tensor::Shape::new(vec![1]),
            );
            if let Some(model_id) = model_id {
                let outcome = self.drive_forward(model_id, id, &ids, &positions)?;
                Ok(Some(outcome))
            } else {
                Ok(None)
            }
        }
    }

    /// P4: per-item graph fast-lane for `step_batch`. Adapter-free single-token
    /// decode items D2D-copy their inputs into the request's persistent GPU
    /// buffers, then replay the captured graph. Returns `None` for anything
    /// unsuitable (non-ROCm, disabled, multi-token prefill chunk, copy miss)
    /// so the caller falls back to the eager batched path. Zero host traffic:
    /// the 1-elem copies stay device-resident.
    pub(crate) fn try_graph_decode_item(
        &mut self,
        req_id: u64,
        model_id: &str,
        input_ids: &grim_tensor::Tensor,
        positions: &grim_tensor::Tensor,
    ) -> Option<StepOutcome> {
        if input_ids.shape().elem_count() != 1 || positions.shape().elem_count() != 1 {
            return None;
        }
        let effective = self.effective_model_id(req_id, model_id);
        let ordinal = match self.models.get(&effective).map(|m| &m.device) {
            Some(grim_tensor::dtype::Device::Rocm(ord)) => *ord,
            _ => return None,
        };
        let rocm = grim_backend_rocm::device::roc_device::RocmDevice::shared(ordinal);
        if !rocm.graph_capture_enabled() {
            return None;
        }
        let (input_clone, pos_clone) = {
            let buffers = self
                .get_or_create_graph_input_buffers(req_id, &effective)
                .ok()?;
            grim_tensor::MemoryOps::copy_slice_into(
                &rocm,
                &**buffers.input_ids.storage(),
                input_ids.storage().as_ref(),
                0,
                1,
            )
            .ok()?;
            grim_tensor::MemoryOps::copy_slice_into(
                &rocm,
                &**buffers.positions.storage(),
                positions.storage().as_ref(),
                0,
                1,
            )
            .ok()?;
            (buffers.input_ids.clone(), buffers.positions.clone())
        };
        let capture_key = format!("decode_req_{req_id}");
        self.drive_forward_graph_capture(model_id, req_id, &input_clone, &pos_clone, &capture_key)
            .ok()
    }

    /// P3: batch graph fast-lane for `step_batch`. Groups adapter-free items
    /// by (model_id, DecodeBatchBucket) and replays through a captured
    /// DecodeBucketGraphPool. Returns `None` when batch graph capture/replay
    /// is not available (non-ROCm, disabled, wrong batch size, etc.).
    pub(crate) fn try_batch_graph_decode(
        &mut self,
        req_id: u64,
        model_id: &str,
        input_ids: &grim_tensor::Tensor,
        positions: &grim_tensor::Tensor,
    ) -> Option<StepOutcome> {
        // Only single-token items qualify for batch graph decode.
        if input_ids.shape().elem_count() != 1 || positions.shape().elem_count() != 1 {
            return None;
        }
        let effective = self.effective_model_id(req_id, model_id);
        let ordinal = match self.models.get(&effective).map(|m| &m.device) {
            Some(grim_tensor::dtype::Device::Rocm(ord)) => *ord,
            _ => return None,
        };
        let rocm = grim_backend_rocm::device::roc_device::RocmDevice::shared(ordinal);
        if !rocm.graph_capture_enabled() {
            return None;
        }
        // Determine batch size from the pool: use the item's token as batch=1
        // for the per-request graph path, or fall through to batch grouping
        // in step_batch for multi-item batches. Here we handle the case where
        // a batch graph pool already exists for this model and the item fits.
        let tid: u32 = self
            .request_last_token
            .get(&req_id)
            .copied()
            .or_else(|| {
                input_ids
                    .to_vec_f32()
                    .ok()
                    .and_then(|v| v.first().copied())
                    .map(|f| f as u32)
            })
            .unwrap_or(0);

        // Replay through an existing bucket graph pool: batch=1 (B1) bucket.
        // Returns None when the pool or the captured graph does not exist yet —
        // step_batch's grouping path creates pools on first multi-item batch.
        let bucket = grim_backend_rocm::DecodeBatchBucket::B1;
        let pool = self.batch_graph_pools.get_mut(&effective)?;
        if !pool.contains_bucket(bucket) {
            return None;
        }
        let logits_storage = pool.replay_batch(bucket, &rocm, &[tid]).ok()?;
        use grim_tensor::backend::BackendStorage as _;
        let logits = logits_storage.to_cpu_vec_f32().ok()?;
        let vocab = logits.len();
        let logits_tensor = grim_backend_cpu::cpu_tensor(
            logits,
            grim_tensor::Shape::new(vec![1, vocab]),
        );
        let accepted = self
            .sessions
            .get_mut(&req_id)
            .map(|s| s.as_mut().last_accepted_tokens())
            .unwrap_or(1);
        Some(StepOutcome {
            logits: Some(std::sync::Arc::new(logits_tensor)),
            accepted_tokens: accepted,
            speculative: false,
        })
    }

    /// Get or create persistent GPU buffers for graph-captured decode inputs.
    pub(crate) fn get_or_create_graph_input_buffers(
        &mut self,
        request_id: u64,
        model_id: &str,
    ) -> Result<&mut GraphCaptureInputBuffers> {
        if !self.decode_graph_input_buffers.contains_key(&request_id) {
            let loaded = self
                .models
                .get(model_id)
                .ok_or_else(|| Error::Config(format!("unknown model {model_id}")))?;
            let ordinal = match loaded.device {
                grim_tensor::dtype::Device::Rocm(ord) => ord,
                _ => return Err(Error::Config("graph capture requires ROCm device".into())),
            };
            let rocm = grim_backend_rocm::device::roc_device::RocmDevice::shared(ordinal);
            // Create persistent GPU buffers for token ID and position (1 x F32 each).
            // These are allocated once per request and updated in-place between replays,
            // matching the verified semantics from tests/test_hip_graph_capture.rs.
            let input_ids_storage = grim_tensor::CoreTensorOps::from_cpu(
                &rocm,
                &[0.0f32],
                &grim_tensor::Shape::new(vec![1]),
                grim_tensor::DType::F32,
            )?;
            let input_ids = grim_tensor::Tensor::new(
                std::sync::Arc::from(input_ids_storage),
                grim_tensor::Shape::new(vec![1]),
                grim_tensor::DType::F32,
                grim_tensor::dtype::QuantProvenance::default(),
                grim_tensor::dtype::Device::Rocm(ordinal),
            );
            let positions_storage = grim_tensor::CoreTensorOps::from_cpu(
                &rocm,
                &[0.0f32],
                &grim_tensor::Shape::new(vec![1]),
                grim_tensor::DType::F32,
            )?;
            let positions = grim_tensor::Tensor::new(
                std::sync::Arc::from(positions_storage),
                grim_tensor::Shape::new(vec![1]),
                grim_tensor::DType::F32,
                grim_tensor::dtype::QuantProvenance::default(),
                grim_tensor::dtype::Device::Rocm(ordinal),
            );
            let buffers = GraphCaptureInputBuffers { input_ids, positions };
            self.decode_graph_input_buffers.insert(request_id, buffers);
        }
        self.decode_graph_input_buffers
            .get_mut(&request_id)
            .ok_or_else(|| Error::Backend("graph input buffer missing after creation".into()))
    }

    pub(crate) fn drive_forward(
        &mut self,
        model_id: &str,
        request_id: u64,
        input_ids: &grim_tensor::Tensor,
        positions: &grim_tensor::Tensor,
    ) -> Result<StepOutcome> {
        // Resolve adapters for this specific request from the adapter registry
        let adapter_ids = self
            .request_adapters
            .get(&request_id)
            .cloned()
            .unwrap_or_default();
        let adapters = { self.resolve_adapters(&adapter_ids).unwrap_or_default() };
        self.drive_forward_with_adapters(model_id, request_id, input_ids, positions, &adapters)
    }

    /// `drive_forward` with explicit adapters: the batched-LoRA decode pass (`step_batch`) drives *base* forwards
    /// through this with `&[]` and applies adapter deltas once per adapter segment afterwards.
    pub(crate) fn drive_forward_with_adapters(
        &mut self,
        model_id: &str,
        request_id: u64,
        input_ids: &grim_tensor::Tensor,
        positions: &grim_tensor::Tensor,
        adapters: &[AdapterHandle],
    ) -> Result<StepOutcome> {
        // SCYTHE-2 farm mode: a pinned request executes on its replica, not
        // the base registration (same weights, different device).
        let model_id = self.effective_model_id(request_id, model_id);
        let model_id = model_id.as_str();
        let was_speculative_path = match self.models.get(model_id) {
            Some(m) => m.model.strategy() != Strategy::Plain,
            None => return Err(Error::Config(format!("unknown model {model_id}"))),
        };
        let session = self
            .sessions
            .get_mut(&request_id)
            .ok_or_else(|| Error::Config("no session for request".into()))?
            .as_mut();
        let loaded = self
            .models
            .get(model_id)
            .ok_or_else(|| Error::Config(format!("unknown model {model_id}")))?;
        let live = self.scheduler.running.len() as f32 / self.config.max_num_seqs.max(1) as f32;
        let logits = loaded.model.decode_one(
            session,
            input_ids,
            positions,
            live,
            self.scheduler.running.len(),
            adapters,
        )?;
        // MIN-2: Report the actual accepted token count from the session (set by the speculative wrapper's decode_one).
        // Non-speculative paths default to 1.
        let accepted_tokens = session.last_accepted_tokens();
        let _ = (loaded, was_speculative_path);
        Ok(StepOutcome {
            logits: Some(Arc::new(logits)),
            accepted_tokens,
            speculative: was_speculative_path,
        })
    }

    /// Graph-capture-aware decode forward: captures the full decode step on the
    /// first call, replays it on subsequent calls with in-place input updates.
    ///
    /// The capture semantics follow `tests/test_hip_graph_capture.rs`:
    /// 1. First call: `begin_graph_capture` → `decode_one` → `end_graph_capture` → `replay_graph`
    /// 2. Subsequent calls: update input buffers in-place → `replay_graph`
    ///
    /// All ops inside `decode_one` automatically dispatch on the capture stream
    /// (via `active_stream()` → `capture_stream` when `capture_active`), so the
    /// model code is capture-unaware.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn drive_forward_graph_capture(
        &mut self,
        model_id: &str,
        request_id: u64,
        input_ids: &grim_tensor::Tensor,
        positions: &grim_tensor::Tensor,
        capture_key: &str,
    ) -> Result<StepOutcome> {
        let _step_t0 = if std::env::var("GRIM_STEP_TRACE").is_ok() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let adapter_ids = self
            .request_adapters
            .get(&request_id)
            .cloned()
            .unwrap_or_default();
        let adapters = { self.resolve_adapters(&adapter_ids).unwrap_or_default() };
        let model_id_eff = self.effective_model_id(request_id, model_id);
        let model_id = model_id_eff.as_str();
        let loaded = self
            .models
            .get(model_id)
            .ok_or_else(|| Error::Config(format!("unknown model {model_id}")))?;

        // WI-HYBRID Layer 2: session-scoped slot key (None when untagged).
        let session_slot_key = self
            .request_session
            .get(&request_id)
            .map(|h| format!("{model_id}#s{h}"));

        // Only ROCm devices support graph capture.
        let ordinal = match &loaded.device {
            grim_tensor::dtype::Device::Rocm(ord) => *ord,
            _ => {
                // Non-ROCm: fall back to eager.
                return self.drive_forward(model_id, request_id, input_ids, positions);
            }
        };

        let rocm = grim_backend_rocm::device::roc_device::RocmDevice::shared(ordinal);
        if !rocm.graph_capture_enabled() {
            return self.drive_forward(model_id, request_id, input_ids, positions);
        }

        // P2: fixed-buffer DecodeGraph for Lfm2 on ROCm
        if let Some(lfm2) = loaded
            .model
            .target()
            .as_any()
            .downcast_ref::<grim_models_transformer::Lfm2>()
        {
            let tid: u32 = self
                .request_last_token
                .get(&request_id)
                .copied()
                .or_else(|| {
                    input_ids
                        .to_vec_f32()
                        .ok()
                        .and_then(|v| v.first().copied())
                        .map(|f| f as u32)
                })
                .unwrap_or(0);

            let graph_slot_key = session_slot_key
                .clone()
                .unwrap_or_else(|| model_id.to_string());
            if !self.decode_graphs.contains_key(&graph_slot_key) {
                let max_ctx = 4096;
                match lfm2.get_or_create_decode_graph(max_ctx, 1) {
                    Ok(mut g) => {
                        // A5 Phase 2: seed graph KV arenas from the eager
                        // device caches (fail-closed to eager). Runs outside
                        // the capture bracket. `current_pos` on the session
                        // (advanced by prior forwards) = valid arena rows.
                        let seed_ok: std::result::Result<(), String> = (|| {
                            let sess = self
                                .sessions
                                .get(&request_id)
                                .ok_or_else(|| "no session".to_string())?;
                            let valid_rows = sess.current_pos() as u32;
                            let caches = sess
                                .model_state()
                                .and_then(|s| {
                                    s.downcast_ref::<
                                        Vec<Option<
                                            grim_models_transformer::Lfm2LayerCache,
                                        >>,
                                    >()
                                })
                                .ok_or_else(|| "no LFM2 caches".to_string())?;
                            let srcs = lfm2
                                .eager_kv_seed_sources(caches, valid_rows)
                                .map_err(|e| format!("export: {e}"))?;
                            let dev = grim_backend_rocm::RocmDevice::shared(ordinal);
                            g.buffers
                                .seed_kv_arena_from_eager(&dev, &srcs)
                                .map_err(|e| format!("seed: {e}"))
                        })();
                        if let Err(e) = seed_ok {
                            eprintln!("[grim] decode-graph: KV seed failed for request {request_id} ({e}); eager fallback");
                            return self.drive_forward(model_id, request_id, input_ids, positions);
                        }
                        if g.begin_capture().is_err() {
                            return self.drive_forward(model_id, request_id, input_ids, positions);
                        }
                        let ok = lfm2.forward_capture(&mut g, tid).is_ok() && g.end_capture().is_ok();
                        if !ok {
                            let _ = g.abort_capture();
                            return self.drive_forward(model_id, request_id, input_ids, positions);
                        }
                        // `current_pos` already set by the KV seed
                        // (`= valid_rows`); replays append after it.
                        self.decode_graphs.insert(graph_slot_key.clone(), g);
                        if let Some(g) = self.decode_graphs.get(&graph_slot_key) {
                            let _ = g.replay();
                        }
                    }
                    Err(_) => {
                        return self.drive_forward(model_id, request_id, input_ids, positions);
                    }
                }
            } else {
                let Some(g) = self.decode_graphs.get_mut(&graph_slot_key) else {
                    return self.drive_forward(model_id, request_id, input_ids, positions);
                };
                // P2-1 Layer 1: the slot is reused across requests, so its
                // arenas must be re-bound to THIS request's prefill state
                // (same seed path as the miss branch, cheap D2D, outside any
                // capture bracket). Without it the replay would attend over
                // the previous request's KV rows.
                let seed_ok: std::result::Result<(), String> = (|| {
                    let sess = self
                        .sessions
                        .get(&request_id)
                        .ok_or_else(|| "no session".to_string())?;
                    let valid_rows = sess.current_pos() as u32;
                    let caches = sess
                        .model_state()
                        .and_then(|s| {
                            s.downcast_ref::<
                                Vec<Option<grim_models_transformer::Lfm2LayerCache>>,
                            >()
                        })
                        .ok_or_else(|| "no LFM2 caches".to_string())?;
                    let srcs = lfm2
                        .eager_kv_seed_sources(caches, valid_rows)
                        .map_err(|e| format!("export: {e}"))?;
                    let dev = grim_backend_rocm::RocmDevice::shared(ordinal);
                    g.buffers
                        .seed_kv_arena_from_eager(&dev, &srcs)
                        .map_err(|e| format!("seed: {e}"))
                })();
                if let Err(e) = seed_ok {
                    eprintln!(
                        "[grim] decode-graph: slot re-seed failed for {graph_slot_key} ({e}); eager step"
                    );
                    self.decode_graphs.remove(&graph_slot_key);
                    return self.drive_forward(model_id, request_id, input_ids, positions);
                }
                if lfm2.forward_replay(g, tid).is_err() {
                    self.decode_graphs.remove(&graph_slot_key);
                    return self.drive_forward(model_id, request_id, input_ids, positions);
                }
                g.buffers.current_pos = g.buffers.current_pos.wrapping_add(1);
            }

            let Some(g) = self.decode_graphs.get(&graph_slot_key) else {
                return self.drive_forward(model_id, request_id, input_ids, positions);
            };
            let logits_arc = match self.graph_capture_logits.get(capture_key) {
                Some(cached) => cached.clone(),
                None => {
                    let t = g.logits_tensor()?;
                    let arc = Arc::new(t);
                    self.graph_capture_logits
                        .insert(capture_key.to_string(), arc.clone());
                    arc
                }
            };
            let accepted = self
                .sessions
                .get_mut(&request_id)
                .map(|s| s.as_mut().last_accepted_tokens())
                .unwrap_or(1);
            return Ok(StepOutcome {
                logits: Some(logits_arc),
                accepted_tokens: accepted,
                speculative: false,
            });
        }

        // Replay path: inputs already updated async above. No decode_one,
        // no D2H — one hipGraphLaunch rewrites the capture-time output
        // buffers, so the cached logits Arc stays valid with fresh data.
        if rocm.has_captured_graph(capture_key) {
            let replayed = rocm.replay_graph(capture_key)?;
            if replayed {
                if let Some(cached) = self.graph_capture_logits.get(capture_key).cloned() {
                    let accepted = self
                        .sessions
                        .get_mut(&request_id)
                        .map(|s| s.as_mut().last_accepted_tokens())
                        .unwrap_or(1);
                    return Ok(StepOutcome {
                        logits: Some(cached),
                        accepted_tokens: accepted,
                        speculative: false,
                    });
                }
                // Graph present but logits evicted — fall through to re-capture.
                self.graph_capture_logits.remove(capture_key);
                let _ = rocm.drop_captured_graph(capture_key);
            } else {
                self.graph_capture_logits.remove(capture_key);
            }
        }

        // Capture path: run decode_one with the capture stream active.
        // All ops dispatch on the capture stream automatically.
        //
        // P2-2 action item (PLAN-improve-grim-perf): models whose decode step
        // is not capture-clean (e.g. Qwen3.5 dense still has per-step D2H
        // host round-trips) make capture fail HERE — previously a hard `Err`
        // with no fallback, killing the request. Degrade to the eager step
        // instead; a clean capture path (LFM2) is unaffected.
        if rocm.begin_graph_capture(capture_key).is_err() {
            eprintln!("[grim] graph capture: begin failed for {capture_key}; eager step");
            return self.drive_forward(model_id, request_id, input_ids, positions);
        }
        // Scope borrows: session + model disjoint from logits cache insert below.
        let (result, accepted, running) = {
            let session = self
                .sessions
                .get_mut(&request_id)
                .ok_or_else(|| Error::Config("no session for request".into()))?
                .as_mut();
            let live =
                self.scheduler.running.len() as f32 / self.config.max_num_seqs.max(1) as f32;
            let running = self.scheduler.running.len();
            let loaded = self
                .models
                .get(model_id)
                .ok_or_else(|| Error::Config(format!("unknown model {model_id}")))?;
            let out = loaded.model.decode_one(
                session, input_ids, positions, live, running, &adapters,
            );
            let acc = session.last_accepted_tokens();
            (out, acc, running)
        };
        if rocm.end_graph_capture(capture_key).is_err() {
            eprintln!("[grim] graph capture: end failed for {capture_key}; eager step");
            return self.drive_forward(model_id, request_id, input_ids, positions);
        }
        // Replay immediately to execute the captured graph.
        if rocm.replay_graph(capture_key).is_err() {
            eprintln!("[grim] graph capture: replay failed for {capture_key}; eager step");
            return self.drive_forward(model_id, request_id, input_ids, positions);
        }
        let logits = result?;
        // Cache output handle: replay rewrites the same device buffers, so
        // future replays return this Arc with fresh contents — zero transfers.
        self.graph_capture_logits
            .insert(capture_key.to_string(), Arc::new(logits.clone()));
        let _ = running;
        if let Some(t0) = _step_t0 {
            eprintln!(
                "[grim] step trace: total {:?} (generic capture bracket)",
                t0.elapsed()
            );
        }
        Ok(StepOutcome {
            logits: Some(Arc::new(logits)),
            accepted_tokens: accepted,
            speculative: false,
        })
    }

    /// P3 pre-pass for `step_batch`: groups adapter-free, single-token Lfm2/ROCm
    /// decode items per (model, power-of-2 bucket) and replays one captured batch
    /// graph per group. Every failure leaves the items unprocessed for the
    /// per-item eager/graph loop.
    pub(crate) fn drive_batch_bucket_groups(
        &mut self,
        items: &[(u64, &str, &grim_tensor::Tensor, &grim_tensor::Tensor)],
        out: &mut HashMap<usize, StepOutcome>,
    ) {
        if !grim_backend_rocm::decode_graph_enabled() {
            return;
        }
        use grim_backend_rocm::DecodeBatchBucket;
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, &(req_id, model_id, input_ids, positions)) in items.iter().enumerate() {
            if input_ids.shape().elem_count() != 1 || positions.shape().elem_count() != 1 {
                continue;
            }
            let adapter_free = self
                .request_adapters
                .get(&req_id)
                .map(|a| a.is_empty())
                .unwrap_or(true);
            if !adapter_free {
                continue;
            }
            let effective = self.effective_model_id(req_id, model_id);
            let eligible = self.models.get(&effective).is_some_and(|m| {
                matches!(m.device, grim_tensor::dtype::Device::Rocm(_))
                    && m.model
                        .target()
                        .as_any()
                        .downcast_ref::<grim_models_transformer::Lfm2>()
                        .is_some()
            });
            if !eligible {
                continue;
            }
            groups.entry(effective).or_default().push(idx);
        }
        for (model_id, idxs) in groups {
            if idxs.len() < 2 {
                continue;
            }
            // Slice into power-of-2 bucket chunks, largest first.
            let mut rest = idxs.as_slice();
            while rest.len() >= 2 {
                let n = rest.len();
                let bucket_size = [32usize, 16, 8, 4, 2]
                    .into_iter()
                    .find(|&b| b <= n)
                    .unwrap();
                let chunk = &rest[..bucket_size];
                let bucket = match DecodeBatchBucket::from_batch_size(bucket_size) {
                    Some(b) => b,
                    None => {
                        rest = &rest[1..];
                        continue;
                    }
                };
                let served = self
                    .replay_bucket_group(&model_id, bucket, chunk, items, out)
                    .unwrap_or(false);
                if served {
                    rest = &rest[bucket_size..];
                } else {
                    // Shrink: drop one item to eager rather than looping.
                    rest = &rest[1..];
                }
            }
        }
    }

    /// Replay one batch bucket graph for `idxs` (all items single-token,
    /// adapter-free, same model). Returns Ok(true) when every item got logits.
    /// Slot assignment (request → bucket row) is pinned on first capture so
    /// each slot's KV arena stays bound to one request across steps.
    pub(crate) fn replay_bucket_group(
        &mut self,
        model_id: &str,
        bucket: grim_backend_rocm::DecodeBatchBucket,
        idxs: &[usize],
        items: &[(u64, &str, &grim_tensor::Tensor, &grim_tensor::Tensor)],
        out: &mut HashMap<usize, StepOutcome>,
    ) -> Result<bool> {
        let bsz = bucket.batch_size();
        if idxs.len() != bsz {
            return Ok(false);
        }
        // Slot pinning: replay only when the request set matches the captured
        // pin (order-insensitive membership, slot by pinned index).
        let key = (model_id.to_string(), bsz as u32);
        let pinned = self.batch_bucket_slots.get(&key).cloned();
        let req_ids: Vec<u64> = idxs.iter().map(|&i| items[i].0).collect();
        let order: Vec<usize> = if let Some(pin) = &pinned {
            if pin.len() != bsz {
                return Ok(false);
            }
            let mut ord = Vec::with_capacity(bsz);
            for want in pin {
                match req_ids.iter().position(|r| r == want) {
                    Some(pos) => ord.push(pos),
                    None => return Ok(false),
                }
            }
            ord
        } else {
            (0..bsz).collect()
        };

        let ordinal = match &self.models.get(model_id).map(|m| m.device.clone()) {
            Some(grim_tensor::dtype::Device::Rocm(ord)) => *ord,
            _ => return Ok(false),
        };
        let rocm = grim_backend_rocm::device::roc_device::RocmDevice::shared(ordinal);
        if !rocm.graph_capture_enabled() {
            return Ok(false);
        }
        let lfm2 = match self
            .models
            .get(model_id)
            .and_then(|m| m.model.target().as_any().downcast_ref::<grim_models_transformer::Lfm2>())
        {
            Some(l) => l,
            None => return Ok(false),
        };

        // Tokens + positions in pinned slot order.
        let mut token_ids: Vec<u32> = Vec::with_capacity(bsz);
        let mut positions_u32: Vec<u32> = Vec::with_capacity(bsz);
        for &oi in &order {
            let (req_id, _model_id, input_ids, positions) = items[idxs[oi]];
            let tid = self
                .request_last_token
                .get(&req_id)
                .copied()
                .or_else(|| {
                    input_ids
                        .to_vec_f32()
                        .ok()
                        .and_then(|v| v.first().copied())
                        .map(|f| f as u32)
                })
                .unwrap_or(0);
            let pos = positions
                .to_vec_f32()
                .ok()
                .and_then(|v| v.first().copied())
                .map(|f| f as u32)
                .unwrap_or(0);
            token_ids.push(tid);
            positions_u32.push(pos);
        }

        // Lazily allocate + capture the bucket graph once.
        {
            let pool = self
                .batch_graph_pools
                .entry(model_id.to_string())
                .or_default();
            if !pool.contains_bucket(bucket) {
                let cfg = &lfm2.cfg;
                let max_ctx: usize = 4096;
                let n_q = cfg.num_heads * cfg.head_dim;
                let n_k = cfg.num_kv_heads * cfg.head_dim;
                pool.allocate_buffers_for_bucket(
                    bucket,
                    &rocm,
                    cfg.num_layers,
                    cfg.hidden_size,
                    n_q,
                    n_k,
                    n_k,
                    cfg.intermediate_size,
                    max_ctx,
                    cfg.vocab_size,
                    cfg.num_heads,
                )?;
                let seed = token_ids.clone();
                pool.capture_batch_graph(bucket, |g| {
                    lfm2
                        .forward_capture_batch(g, &seed)
                        .map_err(|e| grim_backend_rocm::Error::Backend(format!("{e}")))
                })?;
                self.batch_bucket_slots.insert(key, req_ids.clone());
            }
        }

        let pool = self
            .batch_graph_pools
            .get_mut(model_id)
            .ok_or_else(|| Error::Backend("batch pool missing after capture".into()))?;
        let logits_storage = pool.replay_batch_with_pos(
            bucket,
            &rocm,
            &token_ids,
            &positions_u32,
        )?;
        use grim_tensor::backend::BackendStorage as _;
        let logits = logits_storage.to_cpu_vec_f32()?;
        let vocab = lfm2.cfg.vocab_size;
        if vocab == 0 || logits.len() != bsz * vocab {
            return Err(Error::Backend(format!(
                "batch graph logits len {} != {bsz}*{vocab}",
                logits.len()
            )));
        }
        for (slot, &oi) in order.iter().enumerate() {
            let idx = idxs[oi];
            let (req_id, ..) = items[idx];
            let row = logits[slot * vocab..(slot + 1) * vocab].to_vec();
            let t = grim_backend_cpu::cpu_tensor(
                row,
                grim_tensor::Shape::new(vec![1, vocab]),
            );
            let accepted = self
                .sessions
                .get_mut(&req_id)
                .map(|s| s.as_mut().last_accepted_tokens())
                .unwrap_or(1);
            out.insert(
                idx,
                StepOutcome {
                    logits: Some(Arc::new(t)),
                    accepted_tokens: accepted,
                    speculative: false,
                },
            );
        }
        Ok(true)
    }

    /// Public stepping API: drive one forward pass for `request_id` against a caller-supplied target model id, with caller-supplied adapters and an explicit input tensor.
    /// Returns the speculative wrapper's emitted logits.
    pub fn step_one(
        &mut self,
        request_id: u64,
        target_model_id: &str,
        input_ids: &grim_tensor::Tensor,
        positions: &grim_tensor::Tensor,
    ) -> Result<StepOutcome> {
        self.drive_forward(target_model_id, request_id, input_ids, positions)
    }

    /// Execute a grouped batch step across multiple co-scheduled requests (WI-X1).
    /// Drives decoding across up to N requests in a single scheduling tick, returning each request's.
    pub fn step_batch(
        &mut self,
        items: &[(u64, &str, &grim_tensor::Tensor, &grim_tensor::Tensor)],
    ) -> Result<Vec<(u64, StepOutcome)>> {
        #[derive(Clone)]
        enum Slot {
            /// Fully stepped in phase A (legacy path or no logits).
            Done(u64, StepOutcome),
            /// Base logits staged; adapter delta applied in phase B.
            Staged {
                request_id: u64,
                model_id: String,
                base: Arc<grim_tensor::Tensor>,
                adapter: u32,
                accepted_tokens: usize,
            },
        }
        let mut slots: Vec<Slot> = Vec::with_capacity(items.len());

        // P3 pre-pass: group adapter-free, single-token, ROCm Lfm2 decode items
        // per (effective model, power-of-2 bucket) and replay ONE captured batch
        // graph per group. Items the graph cannot serve stay in the per-item
        // loop below (adapters/eager fallbacks unaffected).
        // WI-HYBRID Layer 2: session-slot affinity bookkeeping for every
        // item — device-agnostic, and it runs BEFORE any graph consumption
        // this tick so LRU eviction sees fresh recency.
        for &(req_id, model_id, _, _) in items {
            if let Some(h) = self.request_session.get(&req_id).copied() {
                let key = format!("{model_id}#s{h}");
                let prefix = format!("{model_id}#s");
                let graph_keys: Vec<String> =
                    self.decode_graphs.keys().cloned().collect();
                if let Some(victim) = session_slot_victim(
                    &prefix,
                    &graph_keys,
                    &self.session_slot_last_use,
                    session_graph_slots(),
                    &key,
                ) {
                    self.decode_graphs.remove(&victim);
                    self.session_slot_last_use.remove(&victim);
                }
                self.session_slot_last_use
                    .insert(key, std::time::Instant::now());
            }
        }

        let mut prebatched: HashMap<usize, StepOutcome> = HashMap::new();
        self.drive_batch_bucket_groups(items, &mut prebatched);

        for (item_idx, &(req_id, model_id, input_ids, positions)) in items.iter().enumerate() {
            if let Some(outcome) = prebatched.remove(&item_idx) {
                slots.push(Slot::Done(req_id, outcome));
                continue;
            }
            let effective = self.effective_model_id(req_id, model_id);
            let strategy_plain = self
                .models
                .get(&effective)
                .map(|m| m.model.strategy() == Strategy::Plain)
                .unwrap_or(false);
            let adapter_ids = self
                .request_adapters
                .get(&req_id)
                .cloned()
                .unwrap_or_default();

            if !strategy_plain || adapter_ids.len() > 1 {
                // Legacy path: adapters applied inside decode_one.
                let outcome = self.drive_forward(model_id, req_id, input_ids, positions)?;
                slots.push(Slot::Done(req_id, outcome));
                continue;
            }

            // P4 graph fast-lane: adapter-free decode items replay the
            // per-request captured graph (1 launch) instead of a full eager
            // forward. Anything unsuitable returns None -> eager below.
            if adapter_ids.is_empty()
                && let Some(outcome) =
                    self.try_graph_decode_item(req_id, model_id, input_ids, positions)
            {
                let Some(base) = outcome.logits else {
                    slots.push(Slot::Done(req_id, outcome));
                    continue;
                };
                slots.push(Slot::Staged {
                    request_id: req_id,
                    model_id: effective,
                    base,
                    adapter: 0,
                    accepted_tokens: outcome.accepted_tokens,
                });
                continue;
            }

            // P3 batch graph fast-lane: group adapter-free items by model +
            // DecodeBatchBucket, replay through DecodeBucketGraphPool.
            if adapter_ids.is_empty()
                && let Some(outcome) =
                    self.try_batch_graph_decode(req_id, model_id, input_ids, positions)
            {
                let Some(base) = outcome.logits else {
                    slots.push(Slot::Done(req_id, outcome));
                    continue;
                };
                slots.push(Slot::Staged {
                    request_id: req_id,
                    model_id: effective,
                    base,
                    adapter: 0,
                    accepted_tokens: outcome.accepted_tokens,
                });
                continue;
            }

            // Batched-LoRA path: base forward now, delta applied per segment.
            let outcome =
                self.drive_forward_with_adapters(model_id, req_id, input_ids, positions, &[])?;
            let Some(base) = outcome.logits else {
                slots.push(Slot::Done(req_id, outcome));
                continue;
            };
            slots.push(Slot::Staged {
                request_id: req_id,
                model_id: effective,
                base,
                adapter: adapter_ids.first().copied().unwrap_or(0),
                accepted_tokens: outcome.accepted_tokens,
            });
        }

        // Phase B: group staged rows by model (vocab is per-model constant),
        // stable-sort by adapter id so equal ids are contiguous, apply once.
        let mut staged_count = 0usize;
        let mut group_order: Vec<String> = Vec::new();
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (slot_idx, slot) in slots.iter().enumerate() {
            let Slot::Staged { model_id, .. } = slot else {
                continue;
            };
            staged_count += 1;
            let entry = groups.entry(model_id.clone()).or_insert_with(|| {
                group_order.push(model_id.clone());
                Vec::new()
            });
            entry.push(slot_idx);
        }

        if staged_count > 0 {
            // Scratch space per group: (slot index, staged rows, adapter id).
            let mut stacked: Vec<f32> = Vec::new();
            let mut row_adapters: Vec<u32> = Vec::new();
            let mut layout: Vec<(usize, usize)> = Vec::new();

            for model_id in &group_order {
                let slot_idxs = &groups[model_id];
                let mut sorted: Vec<usize> = slot_idxs.clone();
                let adapter_of = |s: &Slot| match s {
                    Slot::Staged { adapter, .. } => *adapter,
                    Slot::Done(..) => 0,
                };
                sorted.sort_by_key(|&i| adapter_of(&slots[i]));

                // Group width: every staged tensor of the same model shares its vocab.
                // A mismatch would mean cross-model corruption - fail loudly instead.
                let dim_of = |s: &Slot| -> usize {
                    match s {
                        Slot::Staged { base, .. } => {
                            base.shape().dims().last().copied().unwrap_or(0)
                        }
                        Slot::Done(..) => 0,
                    }
                };
                let dim = dim_of(&slots[sorted[0]]);
                if dim == 0 {
                    return Err(Error::Config(format!(
                        "step_batch: model {model_id} produced a zero-width logits row"
                    )));
                }

                stacked.clear();
                row_adapters.clear();
                layout.clear();
                for &i in &sorted {
                    let Slot::Staged { base, .. } = &slots[i] else {
                        continue;
                    };
                    let rows = base.shape().elem_count() / dim;
                    let data = base.to_vec_f32()?;
                    if data.len() != rows * dim {
                        return Err(Error::Config(format!(
                            "step_batch: model {model_id} staged logits len {} is not \
                             rows*{dim} — refusing to stack",
                            data.len()
                        )));
                    }
                    stacked.extend_from_slice(&data);
                    let adapter = adapter_of(&slots[i]);
                    row_adapters.extend(std::iter::repeat_n(adapter, rows));
                    layout.push((i, rows));
                }

                // The adapter deltas must land on the device the model's
                // logits live on, so downstream on-device sampling (WI-X3) keeps working.
                let model_device = self.models.get(model_id).map(|m| m.device.clone());
                self.apply_batched_lora_to_rows(
                    &mut stacked,
                    &row_adapters,
                    dim,
                    model_device.as_ref(),
                )?;

                // Scatter adapted rows back into the staged slots.
                let mut offset = 0usize;
                for &(i, rows) in &layout {
                    let (request_id, accepted_tokens, base) = match &slots[i] {
                        Slot::Staged {
                            request_id,
                            base,
                            accepted_tokens,
                            ..
                        } => (*request_id, *accepted_tokens, base.clone()),
                        Slot::Done(..) => continue,
                    };
                    let shape = base.shape().clone();
                    let data = stacked[offset..offset + rows * dim].to_vec();
                    offset += rows * dim;
                    let adapted = Self::logits_tensor_like(data, shape, &base)?;
                    slots[i] = Slot::Done(
                        request_id,
                        StepOutcome {
                            logits: Some(Arc::new(adapted)),
                            accepted_tokens,
                            speculative: false,
                        },
                    );
                }
            }
        }

        Ok(slots
            .into_iter()
            .map(|slot| match slot {
                Slot::Done(id, outcome) => (id, outcome),
                Slot::Staged { .. } => unreachable!("staged slot left unfilled by phase B"),
            })
            .collect())
    }
}
