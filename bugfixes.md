Logic Bug Hunt — grim-garage crate
Hunt performed across jobs.rs, routes.rs, rocm.rs, discovery.rs, ui_state/*, view_model/*. Findings ranked by severity, every one cited against actual code I read.

HIGH severity
H1 — cancel_job reuses JobStatus::Failed but the worker keeps running and can resurrect the status
routes.rs:310-322 calls update_status(id, JobStatus::Failed). There is no cancellation token passed to the spawned worker (routes.rs:272). The worker run_training_worker (jobs.rs:253-410) reads status once at start (jobs.rs:255) and never re-checks it inside the metric loop (jobs.rs:313-392). So:

The worker runs all total_steps to completion, still calling append_metric and broadcasting events.
The client gets "status": "failed" with no way to distinguish cancel from genuine failure (JobStatus has no Cancelled variant).
On natural completion, the worker calls update_status(Completed) at jobs.rs:405 — overwriting the failed/cancelled status. A UI polling /status sees failed → completed. Cancelled jobs resurrect.
Fix: Add a CancellationToken (or Arc<AtomicBool>) on TrainingJob, set it in cancel_job, check at the top of each loop iteration; on cancel, exit the loop and return without calling update_status(Completed). Add JobStatus::Cancelled.

H2 — SSE stream terminates on Lagged, emits spurious end, and never ends on real completion
routes.rs:552-558:


rust
Err(_) => { yield Ok(Event::default().event("end").data("done")); break; }
But tokio::broadcast::Receiver::recv() returns Err(Lagged(_)) for slow subscribers and Err(Closed) only when all senders are dropped. metrics_tx lives in JobRegistry forever (never dropped) → Closed essentially never fires. Two compounding paths:

Burst (worker emits >1024 events faster than client drains): Lagged → stream emits end/done and breaks. UI concludes training finished; training keeps running. False terminal.
Natural completion: worker calls update_status(Completed) (jobs.rs:405) which doesn't broadcast (see H3). A well-behaved client that drains in time sits forever on recv().await and never gets an end.
Fix:


rust
Err(broadcast::error::RecvError::Lagged(_)) => continue,
Err(broadcast::error::RecvError::Closed) => { yield end_event(); break; }
Plus broadcast a terminal MetricStreamEvent carrying status = Completed/Cancelled/Failed at jobs.rs:405 and jobs.rs:301.

H3 — SSE never receives the terminal Completed event; clients learn completion only by polling
append_metric (jobs.rs:205-218) snapshots let status = job.status; before push_metric and broadcasts that captured status. The final loop step broadcasts Running. Then update_status(Completed) (jobs.rs:405) and update_status(Failed) (jobs.rs:301, 387) do not broadcast anything. Result: an SSE subscriber cannot tell when a job finished on the happy path.

Fix: Add update_status_and_broadcast that, under one write lock, mutates status and sends a MetricStreamEvent { status, … } (using the last metric or a sentinel). Call it on the Completed/Failed/Cancelled transitions.

H4 — UI thread blocks on backend during entire poll; mutex held across network awaits
poller.rs:131-142 holds let mut s = state.lock().await across poll_once(...).await, which itself awaits get_models().await, get_datasets().await, get_devices().await, get_jobs().await in sequence (poller.rs:37-55). GarageClient::get_json (http_client.rs:110-121) does TcpStream::connect + read_to_end with no timeout and Connection: close. Any UI read of DisplayState also takes the same Mutex. A hung (not refused) TCP connect (OS default ≈75s) freezes the UI for the whole duration. The poller's own docstring ("the loop swallows it — no UI death") contradicts this reality.

Fix: Fetch unlocked inside tokio::time::timeout(Duration::from_millis(800), …), then take state.lock().await only for the synchronous set_* writes.

H5 — Stale jobs accumulate forever; DisplayState has no prune / set_jobs path
poll_once only calls state.upsert_job(...) for jobs returned by GET /api/train/jobs (poller.rs:55-58). DisplayState::upsert_job is a plain HashMap::insert (display.rs:44-46); there is no set_jobs/retain_jobs/remove_job API and no intersection with existing keys. When the backend prunes a job, the dead UiJob stays in the registry forever; ViewModel::history_cards keeps re-surfacing terminated jobs whose status never changes. live_metrics per-id memory also leaks (mod.rs:33).

Fix: Add DisplayState::set_jobs(HashMap<String, UiJob>) that replaces the map, and prune live_metrics for ids not in the new set. In poll_once, build the full map from client.get_jobs then call set_jobs (not upsert_job).

MEDIUM severity
M1 — start_training skips path-traversal validation; worker writes arbitrary .train sidecar
routes.rs:251-284 accepts req.model_path/req.dataset_path verbatim — unlike get_bolt_ons (routes.rs:324-333) and convert_model_route (routes.rs:584) which reject ..///\. The stored model_path is used by the worker at jobs.rs:395-399: format!("{}.train", job.model_path) → Path::new(...).parent() → create_dir_all(parent) → train_state.write(&sidecar_path). A body {"model_path":"../../etc/cron.d/x", ...} results in arbitrary file write of a .train file under an attacker-chosen directory.

Fix: Apply sanitize_model_id-style validation (or resolve against default_models_dir() with escape rejection) to both fields before registry.create.

M2 — SSE subscribe-after-get race permanently loses first metric(s)
routes.rs:534-541 does registry.get(&jid).await and only then subscribe_metrics(). Between spawn (routes.rs:272) and subscribe, the worker can dispatch step-0's append_metric. metrics_tx.send is let _ = send (jobs.rs:212) — if there's no subscriber the event is silently dropped, never replayed. The first data point in the UI starts at some step >0.

Fix: On SSE attach, snapshot job.metrics and emit them as initial replay before subscribing live; or always subscribe first, then get.

M3 — RL-mode loss stays stuck at 0.0 forever after the first autograd error
jobs.rs:311 seeds loss = initial_loss(mode); for RL modes initial_loss returns 0.0 (jobs.rs:233). Each branch's error fallback is Err(_) => loss * 0.9 (jobs.rs:361, 370, 379). So 0.0 * 0.9 = 0.0 on any first error, and the variable can never climb back via the fallback. The UI loss curve looks indistinguishable from perfect reward, misleading operators. For SFT modes (initial 2.3) the same fallback works fine.

Fix: E.g. Err(_) => (loss.abs() * 0.9).max(1e-3) for RL modes, or decay-from-initial Err(_) => initial_loss(mode) * 0.9 rather than from last value.

M4 — rand_noise is dead; loss curves are deterministic, contradicting doc comment
jobs.rs:412-422 defines rand_noise with comment "minimal pseudo-random noise for the step simulator", but grep shows zero call sites in the crate. Every emitted loss is either identical (SFT, fixed 0.01 logits against same targets, modulo optimizer step) or a deterministic linear function of step (RL, e.g. step as f32 * 0.05). Worker contract at jobs.rs:250 claims "Metric event per simulated step" — intent was jitter, reality is deterministic.

Fix: Apply rand_noise(0.05) to the emitted loss field, or delete the function and remove the misleading comment. Note: subsec_nanos is also a poor RNG primitive (low resolution when called back-to-back), so deleting is safer than activating.

M5 — errors == 4 magic constant in poll_once breaks silently if a 5th endpoint is added
poller.rs:35-67 increments errors per failed endpoint, then if errors == 4 { Err(AllFailed) }. Adding a future endpoint (e.g. /api/conversions) without bumping the threshold means the AllFailed branch can never fire. Partial failure (errors ∈ {1,2,3}) is also silently swallowed — caller learns nothing about which endpoint failed.

Fix: const POLL_ENDPOINT_COUNT: usize = 4; ... if errors == POLL_ENDPOINT_COUNT { Err(AllFailed) }. Better: carry a partial(Vec<&str>) on PollError so partial failures surface.

M6 — Status wire string has no .to_ascii_lowercase() seam; badge silently degrades on case drift
poller.rs:81-96 copies s.status (a free String) verbatim into UiJob.status. JobCardV1::badge_label (job_card.rs:30-39) is a literal match with other => format!("? {other}") fallback. The server today lowercases via status_label (routes.rs:564-572), so it works — but the moment any refactor returns "Failed", every failed job badge becomes ? Failed instead of ✗ failed. No normalization anywhere on the read path.

Fix: Normalize at the poller seam: status: s.status.to_ascii_lowercase(). Better: add a typed JobStatusView enum serialized alongside training_mode.

M7 — lora_rank = 0 ships to autograd (div-by-zero); normalized() is opt-in and never called
HyperparamFormV1::from_training_config (hyperparam.rs:54-68) copies c.lora_rank verbatim — does NOT call .normalized(). VALID_LORA_RANKS/normalized() exist as helpers but no code path on read or submit invokes them. GarageClient::build_start_training_request (http_client.rs:166-194) accepts lora_rank: u32 with no guard. A non-form code path setting lora_rank = 0 submits a div-by-zero-triggering value to apply_and_record_lora. No QLoRA×rank bound anywhere either.

Fix: Call .normalized() in from_training_config and add if lora_rank == 0 { return Err(...) } in build_start_training_request. Best: make LoraRank a newtype constructor that rejects 0.

LOW severity
L1 — rocm.rs per-agent state leaks across rocminfo agents
query_rocminfo_gpus (rocm.rs:286-376) declares compute_units = 36, wavefront_size = 32, vram_bytes = 8 GiB once outside the agent loop. On a new "Agent " line, the code resets is_gpu = false, name.clear(), marketing_name.clear() — but not compute_units/wavefront_size/vram_bytes. If agent #2's lines don't re-set those keys (e.g. missing "Compute Unit:" line on some kernels), agent #2 inherits agent #1's values.

Fix: Reset the three vars (compute_units=36; wavefront_size=32; vram_bytes=8*GiB) inside the "Agent " start branch.

L2 — rocminfo "Size:" parsing likely sets VRAM to L2 cache size; every AMD card reports 8 GiB default
rocm.rs:341 checks trimmed.starts_with("Size:") && trimmed.ends_with("KB"). rocminfo's "Size:" lines report L1/L2 cache sizes in KB, not VRAM (the VRAM line is "Memory Size:"). The if kb > 100_000 guard (~100 MB) is meant to filter caches out, which mostly means vram_bytes stays at the default 8 GiB for every AMD card regardless of actual VRAM.

Fix: Match "Memory Size:" explicitly (it's reported in KB); fall back to sysfs mem_info_vram_total if available; otherwise default.

L3 — query_amd_vram_used aliases distinct AMD cards to the last slot when ordinal >= count
rocm.rs:186-188: let idx = (ordinal as usize).min(amd_used_bytes.len() - 1);. If lspci enumerates more AMD cards than /sys/class/drm/card*/device/mem_info_vram_used files exist (e.g. iGPU without that sysfs node + dGPU with it), both ordinals 0 and 1 return the dGPU's usage. The iGPU reports the dGPU's VRAM-used.

Fix: if (ordinal as usize) < amd_used_bytes.len() { amd_used_bytes[ordinal as usize] } else { 0 }.

L4 — Duplicate mode_options lists desync silently; unknown mode falls through to Bf16
training_panel.rs:28-65 keeps a hardcoded ["LoRA","QLoRA","Bf16-Full","GRPO","DPO","ORPO"] and a match with _ => defaulting any unrecognized string to Bf16-Full. A stale config file containing a typo like "Lo-RA" silently trains in Bf16 mode with the quant picker hidden. Three separate lists (enum, poller labels, panel options) must stay in lockstep with no compile-time check.

Fix: Centralize into a TrainingModeView enum; replace the _ fallthrough with an explicit other => that surfaces the unknown value rather than masking as Bf16.

L5 — Registry race yields ghost JobSummary with empty paths
routes.rs:236-243: if registry.list() returns a (id, status) but registry.get(&id) returns None (small window between list's read lock release and the per-job get), the server emits a JobSummary with empty model_path/dataset_path and a TrainingMode::Lora placeholder. JobCardV1::subtitle (job_card.rs:42-53) then renders blank — rsplit_once('/') on "" returns None and unwrap_or keeps the empty string, producing a ghost "-titled" card.

Fix: Skip jobs with empty model_path in ViewModel::from, or in list_jobs take the data under one read lock rather than N+1 lock acquisitions.
