//! Scythe disaggregated prefill/decode routing and farm registration.

use crate::*;

impl Engine {
    /// Access the disaggregated KV receiver server instance if running in disaggregated decode role.
    pub fn kv_receiver(&self) -> Option<&grim_disagg::KvReceiverServer> {
        self.kv_receiver.as_ref()
    }

    /// Record an incoming heartbeat from a peer role indicating that the peer role is alive (a successful KV transfer to the decode node, an ingested block from the prefill node, or a server-layer health probe).
    /// `tick()` evaluates these timestamps against [`EngineConfig::disagg_heartbeat_timeout_ms`].
    pub fn disagg_record_peer_heartbeat(&self, role: grim_disagg::PoolRole, now_ms: u64) {
        if let Some(orch) = &self.disagg_orchestrator {
            let mut guard = orch.lock().unwrap_or_else(|p| p.into_inner());
            guard.record_heartbeat(role, now_ms);
        }
    }

    /// Evaluate failover against `now_ms` and cache the effective role.
    /// Returns the role this node should execute as.
    pub fn disagg_evaluate_failover(&self, now_ms: u64) -> grim_disagg::PoolRole {
        if let Some(orch) = &self.disagg_orchestrator {
            let effective = orch
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .evaluate_failover(now_ms, self.config.disagg_heartbeat_timeout_ms);
            *self
                .disagg_effective_role
                .lock()
                .unwrap_or_else(|p| p.into_inner()) = effective;
        }
        *self
            .disagg_effective_role
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    /// The effective role after failover evaluation: `Colocated` means the
    /// remote peer is presumed dead and remote handoff is gated off.
    pub fn disagg_effective_role(&self) -> grim_disagg::PoolRole {
        *self
            .disagg_effective_role
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    /// True when SCYTHE-2 inference routing is armed (`GRIM_SCYTHE_INFERENCE`
    /// set and more than one ROCm GPU visible).
    pub fn scythe_armed(&self) -> bool {
        self.scythe_ctrl.is_some()
    }

    /// Hand the engine's SCYTHE-2 controller to a streaming executor as a [`ScytheRoute`](crate::streaming_forward::ScytheRoute) (WI-INF3).
    /// The route snapshots capabilities/links from the profiler lazily (on capability-epoch change only) and maps rank.
    pub fn attach_scythe_route(
        &mut self,
        sfb: &mut crate::streaming_forward::StreamingBlockForward,
    ) -> bool {
        let Some(ctrl) = self.scythe_ctrl.take() else {
            return false;
        };
        let Some(ref profiler) = self.capability_profiler else {
            // Armed controller without a profiler cannot happen by
            // construction; put the controller back rather than dropping it.
            self.scythe_ctrl = Some(ctrl);
            return false;
        };
        sfb.attach_scythe_route(crate::streaming_forward::ScytheRoute {
            ctrl: Arc::new(std::sync::Mutex::new(ctrl)),
            profiler: Some(Arc::clone(profiler)),
            caps: Vec::new(),
            links: Vec::new(),
            caps_epoch: u32::MAX, // force first-use refresh from the profiler
            device_for_rank: Arc::new(grim_tensor::Device::Rocm),
        });
        true
    }

    /// SCYTHE-2 farm mode (WI-INF3 serving integration).
    /// A dense model's blocks live on one device, so per-layer placement needs the weight-sharded ring.
    pub fn register_model_with_farm(&mut self, id: &str, primary: Box<dyn CausalLm>, path: &str) {
        self.register_model_with_farm_inner(id, primary, path, None);
    }

    /// Load a farm while keeping speculative decoding on rank 0.
    /// Replica models are intentionally plain: the drafter is coupled to rank 0's device and is.
    pub fn load_and_register_scythe_farm_speculative(
        &mut self,
        id: &str,
        base_path: &str,
        draft_path: Option<&str>,
        lookahead: bool,
    ) -> Result<()> {
        let base_model = crate::model_loader::load_from_path(base_path)?;
        let drafter = if let Some(d_path) = draft_path {
            match crate::model_loader::load_eagle3_from_path(d_path, base_model.device().clone()) {
                Ok(eagle3) => Some(Arc::new(grim_speculative::Eagle3Drafter::new(eagle3))
                    as Arc<dyn DraftBackbone>),
                Err(_) => {
                    let _ = crate::model_loader::load_from_path(d_path)?;
                    Some(Arc::new(grim_speculative::TinyDraftBackbone::new(
                        128256, 2048, 4, 42,
                    )) as Arc<dyn DraftBackbone>)
                }
            }
        } else {
            None
        };
        let _ = lookahead;
        self.register_model_with_farm_inner(id, base_model, base_path, drafter);
        Ok(())
    }

    pub(crate) fn register_model_with_farm_inner(
        &mut self,
        id: &str,
        primary: Box<dyn CausalLm>,
        path: &str,
        drafter: Option<Arc<dyn DraftBackbone>>,
    ) {
        let devices = crate::model_loader::visible_rocm_devices();
        let primary_ordinal = match (self.scythe_armed(), primary.device()) {
            (true, grim_tensor::Device::Rocm(ord)) if devices.len() > 1 => *ord,
            _ => {
                self.register_model(id, primary);
                return;
            }
        };
        // Rank order: primary's own ordinal first, remaining GPUs after.
        let mut ordered: Vec<grim_tensor::Device> =
            vec![grim_tensor::Device::Rocm(primary_ordinal)];
        for dev in devices {
            if dev != grim_tensor::Device::Rocm(primary_ordinal) {
                ordered.push(dev);
            }
        }
        if let Some(drafter) = drafter {
            self.register_speculative(id, primary, Some(drafter), None, None);
        } else {
            self.register_model(id, primary);
        }
        let mut replica_ids = vec![id.to_string()];
        for (rank, dev) in ordered.iter().enumerate().skip(1) {
            match crate::model_loader::load_from_path_on_device(path, dev.clone()) {
                Ok(replica) => {
                    let rid = Self::scythe_replica_id(id, rank);
                    self.register_model(&rid, replica);
                    replica_ids.push(rid);
                }
                Err(e) => {
                    // A failed replica must not silently shrink the farm below
                    // what the controller was told to route over — fail loudly.
                    log::warn!(
                        "[scythe2] farm replica {rid} on {dev:?} failed to load: {e}; \
                         removing partial farm",
                        rid = Self::scythe_replica_id(id, rank)
                    );
                    for rid in &replica_ids[1..] {
                        self.models.remove(rid);
                    }
                    self.scythe_replicas.remove(id);
                    return;
                }
            }
        }
        // The controller routes over exactly the registered replica count.
        if let Some(ctrl) = self.scythe_ctrl.as_mut() {
            let n = replica_ids.len();
            if ctrl.num_gpus() != n {
                let num_layers = ctrl.layer_fps.len();
                let budget = ctrl.budget_ms;
                *ctrl = crate::scythe2::C2plrController::new(num_layers, n, budget);
            }
        }
        log::info!(
            "[scythe2] farm armed: {} replica(s) of {id} across GPUs {:?}",
            replica_ids.len(),
            ordered
        );
        self.scythe_replicas.insert(id.to_string(), replica_ids);
    }

    pub(crate) fn scythe_replica_id(base: &str, rank: usize) -> String {
        format!("{base}#scythe{rank}")
    }

    /// WI-SB2 admission guard for one request against the farm's live caps.
    /// [`ScytheAdmission::Pin`] carries the controller-chosen rank; [`ScytheAdmission::WaitVram`] means every rank failed the footprint check (or the.
    pub(crate) fn scythe_admission_decision(
        &mut self,
        base: &str,
        seq_len: usize,
        max_new_tokens: usize,
        caps_raw: &[grim_tensor::backend::GpuCapability],
    ) -> ScytheAdmission {
        let Some(ids) = self.scythe_replicas.get(base) else {
            return ScytheAdmission::Bypass;
        };
        let n = ids.len();
        if n <= 1 {
            return ScytheAdmission::Pin(0);
        }
        if !self.scythe_armed() {
            return ScytheAdmission::Bypass;
        }
        if caps_raw.is_empty() {
            log::info!("[scythe2] farm present but profiler sees no GPUs; leaving request queued");
            return ScytheAdmission::WaitVram;
        }
        // Active pins plus pins released inside the cooldown window, plus
        // external (non-farm) utilization folded in at a fixed weight.
        self.scythe_pin_cooldown
            .retain(|(_, t)| t.elapsed() < SCYTHE_PIN_COOLDOWN);
        let external_busy: Vec<Option<u32>> =
            (0..n).map(grim_backend_rocm::compute_utilization).collect();
        let effective_loads = scythe_effective_loads(
            self.scythe_pin.values().copied(),
            &self.scythe_pin_cooldown,
            SCYTHE_PIN_COOLDOWN,
            &external_busy,
            n,
            SCYTHE_EXTERNAL_BUSY_WEIGHT,
        );
        let any_external_busy = external_busy
            .iter()
            .any(|b| matches!(b, Some(pct) if *pct >= 25));
        let any_load = effective_loads.iter().any(|&l| l > 0.0);

        // Reserve KV plus the activation working-set floor before placement.
        let (layers, hidden_hint) = self.models.get(base).map_or((1, None), |m| {
            (
                m.model.num_layers_hint().unwrap_or(1) as u64,
                m.model.hidden_size_hint(),
            )
        });
        let footprint = scythe_request_footprint_bytes(
            seq_len,
            max_new_tokens,
            self.config.num_kv_heads,
            self.config.head_dim,
            hidden_hint,
            layers,
        );
        let mut caps = load_adjusted_caps(caps_raw, n, &effective_loads);
        let feasible = scythe_vram_feasible(&caps, footprint);
        for (cap, ok) in caps.iter_mut().zip(&feasible) {
            if !ok {
                cap.tflops_fp16 = 0.0;
            }
        }
        if feasible.iter().all(|&ok| !ok) {
            log::info!(
                "[scythe2] no farm rank holds ~{} MiB; leaving request queued",
                footprint / (1024 * 1024)
            );
            return ScytheAdmission::WaitVram;
        }
        let links = grim_backend_rocm::CapabilityProfiler::link_matrix(n);
        let epoch = grim_backend_rocm::current_epoch();
        // Proxy shape: at pass granularity only the sequence length carries
        // signal (the bucket), and relative TFLOPS ordering drives the pick.
        let shape = [1usize, seq_len.max(1), 1, 1];
        let Some(ctrl) = self.scythe_ctrl.as_mut() else {
            return ScytheAdmission::Bypass;
        };
        // The shape-keyed PlacementCache is load-blind; while ANY rank
        // carries load (pins or external busy) the decision must run fresh.
        let placement = if any_load || any_external_busy {
            ctrl.decide_forced(0, &shape, &caps, &links, epoch)
        } else {
            ctrl.decide(0, &shape, &caps, &links, epoch)
        };
        let chosen = placement.ranks.first().copied();
        // WI-SB1 load spreading: ON by default since the P1-3 guard sweep fixed cross-device FFI (verification 2026-08-23f: rank-1 pins served cleanly, GPU1 sampled at 88-93 % under sustained load, 18/17 rank split over 35 live requests).
        // Opt out with GRIM_SCYTHE_SPREAD=0.
        let spread_enabled = std::env::var("GRIM_SCYTHE_SPREAD")
            .map(|v| v != "0")
            .unwrap_or(true);
        let chosen = match chosen {
            Some(r) if r != 0 && !spread_enabled => {
                log::info!(
                    "[scythe2] load favored rank {r} but spreading is disabled \
                     (GRIM_SCYTHE_SPREAD=0); clamping to rank 0"
                );
                Some(0)
            }
            other => other,
        };
        if let Some(r) = chosen {
            log::info!(
                "[scythe2] admission loads {:?} (external busy {:?}) -> rank {}",
                effective_loads,
                external_busy,
                r
            );
        }
        chosen
            .map(|r| r.min(n - 1))
            .map_or(ScytheAdmission::WaitVram, ScytheAdmission::Pin)
    }

    /// Pinned farm rank for a request, if any. Telemetry/status surface.
    pub fn scythe_pin_of(&self, request_id: u64) -> Option<usize> {
        self.scythe_pin.get(&request_id).copied()
    }

    /// Number of farm replicas registered for `base` (0 = not a farm model).
    pub fn scythe_farm_size(&self, base: &str) -> usize {
        self.scythe_replicas.get(base).map_or(0, Vec::len)
    }

    /// WI-SB2: retry requests parked on the VRAM waitlist at tick start - finished sessions have freed their ranks by now.
    /// Order-stable backfill: entries are scanned in arrival order and admitted individually as soon as some.
    pub(crate) fn retry_scythe_vram_waitlist(&mut self) {
        if self.scythe_vram_waitlist.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.scythe_vram_waitlist);
        for request in pending {
            let id = request.id;
            let base_for_pin = request
                .model_id
                .as_deref()
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .or_else(|| self.models.keys().next().cloned());
            let decision = base_for_pin
                .as_deref()
                .filter(|base| self.scythe_armed() && self.scythe_replicas.contains_key(*base))
                .map(|base| {
                    let caps_raw = self
                        .capability_profiler
                        .as_ref()
                        .map(|p| p.capabilities())
                        .unwrap_or_default();
                    self.scythe_admission_decision(
                        base,
                        request.prompt_tokens,
                        request.max_new_tokens,
                        &caps_raw,
                    )
                })
                .unwrap_or(ScytheAdmission::Bypass);
            match decision {
                ScytheAdmission::Pin(rank) => {
                    self.scythe_pin.insert(id, rank);
                    log::info!("[scythe2] waitlisted request {id} admitted on farm rank {rank}");
                    if let Err(e) = self.admit_placed_request(request, Some(rank)) {
                        log::warn!("[scythe2] waitlisted request {id} failed to admit: {e}");
                    }
                }
                ScytheAdmission::Bypass => {
                    if let Err(e) = self.admit_placed_request(request, None) {
                        log::warn!("[scythe2] waitlisted request {id} failed to admit: {e}");
                    }
                }
                ScytheAdmission::WaitVram => self.scythe_vram_waitlist.push(request),
            }
        }
    }

    /// Number of requests currently held on the WI-SB2 VRAM waitlist - no farm rank could hold their footprint when they arrived.
    /// Status and observability surface; nonzero means serving capacity is exhausted for that prompt size, not.
    pub fn scythe_vram_waitlist_len(&self) -> usize {
        self.scythe_vram_waitlist.len()
    }

    /// F3b: Enqueue a request whose prefill already ran on a remote Prefill node.
    /// Creates the local session/KV structures without any local prompt forward (`prompt_tokens = 0`, so `drive_prefill`.
    pub fn enqueue_remote_prefill_request(
        &mut self,
        id: u64,
        prompt_len: usize,
        model_id: Option<String>,
    ) -> Result<()> {
        let request = grim_scheduler::Request {
            id,
            prompt_tokens: 0,
            priority: 0,
            model_id,
            ..Default::default()
        };
        self.enqueue_request_with_kv(request)?;
        if let Some(s) = self.sessions.get_mut(&id) {
            s.as_mut().advance_pos(prompt_len);
        }
        self.hydrate_session_from_pool(id);
        Ok(())
    }
}
