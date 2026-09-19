//! Request enqueue/admission/session hydration lifecycle.

use crate::*;

impl Engine {
    pub fn enqueue_request(&mut self, request: grim_scheduler::Request) -> Result<()> {
        self.enqueue_request_with_kv(request)
    }

    /// Allocate a session with a paged KV cache wired in and prefix caching active (§5.1).
    pub fn enqueue_request_with_kv(&mut self, request: grim_scheduler::Request) -> Result<()> {
        // WI-HYBRID Layer 2: record the session hash (if any) so the decode
        // loop can affinity the graph slot and prefill can pin the blocks.
        if let Some(sess) = &request.session {
            let model = request.model_id.as_deref().unwrap_or("");
            let mut h = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&model, &mut h);
            std::hash::Hash::hash(&sess, &mut h);
            self.request_session
                .insert(request.id, std::hash::Hasher::finish(&h));
        }
        // SCYTHE-2 farm mode: pin the request to a controller-chosen replica BEFORE the session exists, so its
        // KV pages are allocated on the pinned replica's device and stay there for the request's lifetime.
        let base_for_pin = request
            .model_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| self.models.keys().next().cloned());
        let mut pin_rank = None;
        if let Some(base) = base_for_pin.as_deref() {
            if self.scythe_armed() && self.scythe_replicas.contains_key(base) {
                // WI-SB2: consult the VRAM guard before anything is allocated.
                let caps_raw = self
                    .capability_profiler
                    .as_ref()
                    .map(|p| p.capabilities())
                    .unwrap_or_default();
                match self.scythe_admission_decision(
                    base,
                    request.prompt_tokens,
                    request.max_new_tokens,
                    &caps_raw,
                ) {
                    ScytheAdmission::Pin(rank) => {
                        self.scythe_pin.insert(request.id, rank);
                        log::info!(
                            "[scythe2] request {} ({} tok) pinned to farm rank {rank}",
                            request.id,
                            request.prompt_tokens
                        );
                        pin_rank = Some(rank);
                    }
                    ScytheAdmission::WaitVram => {
                        // Hold the request out of the scheduler entirely: no session, no pin, no admission
                        // - a rank must be able to hold it before it enters the queue.
                        log::info!(
                            "[scythe2] request {} parked on VRAM waitlist (WI-SB2)",
                            request.id
                        );
                        self.scythe_vram_waitlist.push(request);
                        return Ok(());
                    }
                    ScytheAdmission::Bypass => {}
                }
            }
        }
        self.admit_placed_request(request, pin_rank)
    }

    /// Session creation + scheduler entry for an already-placed request.
    /// `pin_rank` (farm mode) selects the pinned replica's device for the KV.
    pub(crate) fn admit_placed_request(
        &mut self,
        request: grim_scheduler::Request,
        pin_rank: Option<usize>,
    ) -> Result<()> {
        // Honor the model's actual device instead of silently forcing CPU.
        // Under a farm pin, that is the pinned replica's device.
        let base_for_pin = request
            .model_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| self.models.keys().next().cloned());
        let device = if pin_rank.is_some() {
            base_for_pin
                .as_deref()
                .map(|base| self.effective_model_id(request.id, base))
                .and_then(|rid| self.models.get(&rid).map(|m| m.device.clone()))
                .unwrap_or(grim_tensor::Device::Cpu)
        } else {
            match request.model_id.as_deref() {
                Some(id) if !id.is_empty() => self
                    .models
                    .get(id)
                    .map(|m| m.device.clone())
                    .unwrap_or(grim_tensor::Device::Cpu),
                _ => self
                    .models
                    .values()
                    .next()
                    .map(|m| m.device.clone())
                    .unwrap_or(grim_tensor::Device::Cpu),
            }
        };
        log::info!(
            "[grim-engine] admit_placed_request: request {} model_id={:?} resolved device={:?}",
            request.id,
            request.model_id,
            device
        );
        // R4 - memory-sovereign admission gate. If the model reported hyperparameters and the backend can probe current
        // free device memory, certify this request's footprint (prompt + max_tokens) fits within what is *currently* free.
        if let Some(base_model) = request
            .model_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .and_then(|id| self.models.get(id))
            .or_else(|| self.models.values().next())
        {
            if let Some(hparams) = &base_model.arch_hyperparams {
                let target_seq_len = request.prompt_tokens.saturating_add(request.max_new_tokens);
                if let Some(free_device) = free_device_memory(&device) {
                    let host_allowance = std::env::var("GRIM_HOST_ALLOWANCE_GB")
                        .ok()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(8)
                        * 1024
                        * 1024
                        * 1024;
                    let reserve = std::env::var("GRIM_MEMORY_RESERVE_GB")
                        .ok()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(2)
                        * 1024
                        * 1024
                        * 1024;
                    let live_boundary =
                        BoundaryVector::standard(host_allowance, free_device, reserve);
                    if let Err(e) = MemoryCertificate::certify(
                        hparams,
                        live_boundary,
                        target_seq_len,
                        1,
                        2,
                        request.id.to_string(),
                    ) {
                        return Err(Error::Config(format!(
                            "request {} ({} tokens) exceeds current memory envelope: {}",
                            request.id, target_seq_len, e
                        )));
                    }
                }
            }
        }

        // WI-HYBRID Layer 2 (step 2): pool-level admission accounting. The
        // device-memory certificate above knows nothing about the block pool,
        // so a pool exhausted by pinned + live blocks could admit a request
        // whose KV demand can never be satisfied (the append-crash scenario).
        // Demand = one block per BLOCK_SIZE tokens of (prompt + max_tokens).
        // If allocatable capacity falls short, drop the OLDEST pins (spec:
        // "pin sweep drops oldest pins when admission can't be satisfied").
        // Still short after the rescue: log and proceed — over-demand is the
        // scheduler/scythe-waitlist's contract to queue, and the atomic
        // append makes any true exhaustion a loud step error, never silent
        // KV corruption.
        let needed_blocks = request
            .prompt_tokens
            .saturating_add(request.max_new_tokens)
            .div_ceil(grim_memory::BLOCK_SIZE);
        {
            let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
            let allocatable = pool.allocatable_blocks();
            if allocatable < needed_blocks {
                let dropped = pool.drop_oldest_pins(needed_blocks - allocatable);
                log::info!(
                    "[grim-engine] admission: pool short by {} blocks for request {} — dropped {} oldest session pins (allocatable now {})",
                    needed_blocks.saturating_sub(pool.allocatable_blocks()),
                    request.id,
                    dropped,
                    pool.allocatable_blocks()
                );
            }
        }
        let mut kv = grim_memory::PagedKvCache::new(
            self.block_pool.clone(),
            self.config.num_kv_heads,
            self.config.head_dim,
            BLOCK_SIZE,
        );
        let backend = grim_nn::pick_device_for_storage_device(&device);
        kv.set_device(device.clone(), backend);
        let session = Box::new(grim_core::session::Inner::with_kv(device, Box::new(kv)));

        self.sessions.insert(request.id, session);
        self.request_model_ids
            .insert(request.id, request.model_id.clone().unwrap_or_default());
        self.request_adapters
            .insert(request.id, request.adapter_ids.clone());
        // Store the real input token IDs if provided
        if let Some(input_ids) = request.input_ids.clone() {
            self.request_input_ids.insert(request.id, input_ids);
        }
        self.request_rng.insert(
            request.id,
            DeterministicRng::from_seed(request.id.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        self.scheduler.enqueue(request);
        Ok(())
    }

    /// F3b helper: copy pool layer storage into the session's page tensors for every received block (all layers present per block).
    /// Mirror of what the pull path does for un-received blocks.
    pub(crate) fn hydrate_session_from_pool(&mut self, id: u64) {
        let num_blocks = {
            let pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.num_blocks()
        };
        // Collect under the pool lock, then write into the session.
        let mut payload: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = Vec::new();
        {
            let pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
            for b in 0..num_blocks {
                if !pool.block_is_received(b) {
                    continue;
                }
                let mut layer = 0usize;
                while let Some(k) = pool.read_layer_keys(b, layer) {
                    match pool.read_layer_values(b, layer) {
                        Some(v) => payload.push((b, layer, k.to_vec(), v.to_vec())),
                        None => break,
                    }
                    layer += 1;
                }
            }
        }
        if payload.is_empty() {
            return;
        }
        if let Some(session) = self.sessions.get_mut(&id) {
            if let Some(kv) = session.kv_mut() {
                for (b, layer, k, v) in payload {
                    let _ = kv.write_layer_block(layer, b, &k, &v);
                }
            }
        }
    }

    pub fn finish_request(&mut self, id: u64) {
        self.scheduler.finish(id);
        // Layer 1.5: release this request's radix-tree claims BEFORE the KV
        // rollback below — the tree's refcounts (+1 per seeded claim, +1 per
        // registered chunk) balance exactly one `remove_prefix` over the full
        // block table, after which unreferenced cached prefixes become
        // LRU-evictable again.
        let had_radix_claim = self.radix_seeded_blocks.contains_key(&id)
            || self.radix_registered_blocks.contains_key(&id);
        if self.radix_enabled && had_radix_claim {
            let blocks: Option<Vec<usize>> = self
                .sessions
                .get(&id)
                .and_then(|s| s.block_table().map(|t| t.to_vec()))
                .map(|t| t.iter().map(|&b| b as usize).collect());
            if let Some(blocks) = blocks {
                let mut pool = self.block_pool.lock().unwrap_or_else(|e| e.into_inner());
                pool.remove_prefix(&blocks);
            }
        }
        self.radix_seeded_blocks.remove(&id);
        self.radix_registered_blocks.remove(&id);
        if let Some(session) = self.sessions.get_mut(&id) {
            let _ = session.rollback_kv_to(0);
        }
        self.sessions.remove(&id);
        // P2-1: keep the per-model decode graph warm across requests. The old
        // behavior removed it here, forcing the next request to pay alloc +
        // capture + seed again (~170 ms of warm TTFT). Callers that unload a
        // model drop the whole engine instance, which drops the graph with it.
        self.last_outcomes.remove(&id);
        self.request_rng.remove(&id);
        self.request_model_ids.remove(&id);
        self.request_adapters.remove(&id);
        self.request_input_ids.remove(&id);
        self.prefill_progress.remove(&id);
        // WI-HYBRID Layer 2: the request→session mapping dies with the
        // request; the session-scoped slot + its LRU entry persist by design
        // (that IS the affinity).
        self.request_session.remove(&id);
        self.request_last_token.remove(&id);
        self.decode_graph_input_buffers.remove(&id);
        // Release the farm slot so the controller's load view stays honest.
        // The rank stays counted for a short cooldown (see `scythe_admission_decision`) so the NEXT admission still.
        if let Some(rank) = self.scythe_pin.remove(&id) {
            self.scythe_pin_cooldown
                .push((rank, std::time::Instant::now()));
        }
        // A cancelled request must not linger on the WI-SB2 VRAM waitlist.
        self.scythe_vram_waitlist.retain(|r| r.id != id);
    }
}
