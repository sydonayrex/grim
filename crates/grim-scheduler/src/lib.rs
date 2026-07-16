//! Continuous-batching scheduler with latency-aware admission control.
//!
//! Architecture §5.2:
//! - Three-queue iteration-level scheduling (waiting/running/swapped).
//! - `AdmissionController` predicts TTFT per request and defers requests
//!   that would exceed the budget, self-calibrating from real prefill timings.
//! - Preemption under memory pressure, lowest-priority-first.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use grim_core::error::{Error, Result};
use grim_core::DeterminismMode;

pub mod self_tuning;
pub use self_tuning::SelfTuningController;

/// A request in the scheduler system.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt_tokens: usize,
    pub priority: i32,
}

/// Admission decision for an incoming request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit,
    Defer,
}

/// Batch token backlog: sum of remaining prefill tokens for waiting requests
/// plus unprocessed chunked-prefill tokens.
#[derive(Debug, Default, Clone)]
pub struct BatchTokenBacklog {
    pub total: usize,
}

/// Latency-aware admission controller (§5.2).
pub struct AdmissionController {
    pub target_ttft_ms: u64,
    pub target_itl_ms: u64,
    throughput_estimate: Mutex<f64>,
}

impl AdmissionController {
    pub fn new(target_ttft_ms: u64, target_itl_ms: u64) -> Self {
        Self {
            target_ttft_ms,
            target_itl_ms,
            throughput_estimate: Mutex::new(1000.0),
        }
    }

    pub fn predict_ttft(&self, prompt_tokens: usize, batch_token_backlog: usize) -> Duration {
        let total = batch_token_backlog + prompt_tokens;
        let rate = *self.throughput_estimate.lock().unwrap();
        Duration::from_secs_f64(total as f64 / rate.max(1.0))
    }

    pub fn admit(&self, request: &Request, backlog: &BatchTokenBacklog) -> AdmissionDecision {
        // Solo-prompt predicted TTFT check (§5.2): if a single request's prompt length
        // is so large that its predicted TTFT alone exceeds the target_ttft_ms,
        // it would be deferred forever causing livelock.
        // We bypass the defer decision and admit it if no other requests are waiting in the backlog.
        let solo_predicted = self.predict_ttft(request.prompt_tokens, 0);
        if backlog.total <= request.prompt_tokens && solo_predicted.as_millis() as u64 > self.target_ttft_ms {
            println!("[AdmissionController] Solo-prompt livelock bypass: admitting oversized request {}", request.id);
            return AdmissionDecision::Admit;
        }

        if self.target_ttft_ms == 0 {
            return AdmissionDecision::Admit;
        }

        let predicted = self.predict_ttft(request.prompt_tokens, backlog.total);
        
        // ITL (Inter-Token Latency) check (§5.2): verify expected decode latency does not exceed target limit
        let rate = *self.throughput_estimate.lock().unwrap();
        let expected_itl_ms = if rate > 0.0 { (1000.0 / rate) as u64 } else { 0 };
        if self.target_itl_ms > 0 && expected_itl_ms > self.target_itl_ms {
            println!("[AdmissionController] Deferring request {} due to ITL constraint violation (expected {}ms > target {}ms)", request.id, expected_itl_ms, self.target_itl_ms);
            return AdmissionDecision::Defer;
        }

        if predicted.as_millis() as u64 <= self.target_ttft_ms {
            AdmissionDecision::Admit
        } else {
            AdmissionDecision::Defer
        }
    }

    pub fn observe_prefill(&self, prompt_tokens: usize, wall_duration: Duration) {
        let measured_tps = prompt_tokens as f64 / wall_duration.as_secs_f64();
        const EMA_ALPHA: f64 = 0.3;
        let mut est = self.throughput_estimate.lock().unwrap();
        *est = *est * (1.0 - EMA_ALPHA) + measured_tps * EMA_ALPHA;
    }

    pub fn throughput_estimate(&self) -> f64 {
        *self.throughput_estimate.lock().unwrap()
    }
}

/// The scheduler: manages waiting/running/swapped/paused queues per §5.2.
pub struct Scheduler {
    pub waiting: VecDeque<Request>,
    pub running: Vec<Request>,
    pub swapped: VecDeque<Request>,
    pub paused: VecDeque<Request>,   // §5.2.1 — explicitly paused, KV retained
    pub max_batched_tokens: usize,
    pub max_num_seqs: usize,
    /// Tuned by [`SelfTuningController::chunked_prefill_size`](crate::self_tuning::KnobKind::ChunkedPrefillSize)
    /// (§5.7): how many tokens from any one prompt are drained per
    /// schedule pass. Drives prefill-vs-decode TTFT balance.
    pub chunked_prefill_size: usize,
    pub admission: AdmissionController,
    pub determinism_mode: DeterminismMode,
}

/// Result of one `schedule()` call — the engine uses this to run the batch.
#[derive(Debug, Default)]
pub struct SchedulerOutput {
    pub prefill_ids: Vec<u64>,
    pub decode_ids: Vec<u64>,
    pub preempted_ids: Vec<u64>,
}

impl SchedulerOutput {
    pub fn is_empty(&self) -> bool {
        self.prefill_ids.is_empty() && self.decode_ids.is_empty()
    }
}

impl Scheduler {
    pub fn new(
        max_batched_tokens: usize,
        max_num_seqs: usize,
        admission: AdmissionController,
    ) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            swapped: VecDeque::new(),
            paused: VecDeque::new(),
            max_batched_tokens,
            max_num_seqs,
            chunked_prefill_size: 512,
            admission,
            determinism_mode: DeterminismMode::Relaxed,
        }
    }

    pub fn enqueue(&mut self, request: Request) {
        self.waiting.push_back(request);
    }

    pub fn compute_token_backlog(&self) -> BatchTokenBacklog {
        let mut total = 0usize;
        for r in &self.waiting {
            total += r.prompt_tokens;
        }
        BatchTokenBacklog { total }
    }

    /// Called once per engine tick. Decides what runs this step.
    pub fn schedule(&mut self) -> SchedulerOutput {
        if self.determinism_mode == DeterminismMode::Strict {
            // Sort waiting queue deterministically by request ID
            let mut temp: Vec<Request> = self.waiting.drain(..).collect();
            temp.sort_by_key(|r| r.id);
            self.waiting = temp.into();

            // Sort running list deterministically by request ID
            self.running.sort_by_key(|r| r.id);
        }

        let backlog = self.compute_token_backlog();
        let total_running_tokens: usize = self.running.iter().map(|r| r.prompt_tokens).sum();
        let pressure_active = backlog.total > self.max_batched_tokens || self.waiting.len() > 10 || total_running_tokens > self.max_batched_tokens;

        // 0. Admission control: defer requests that would bust the TTFT budget.
        let mut admitted = VecDeque::new();
        while let Some(r) = self.waiting.pop_front() {
            if self.admission.admit(&r, &backlog) == AdmissionDecision::Admit {
                admitted.push_back(r);
            } else {
                self.waiting.push_front(r);
                break;
            }
        }

        let mut output = SchedulerOutput::default();

        // Preemption check (§5.2): swap lowest-priority running sequences to swapped queue under pressure
        // (Simulate memory/token pressure when total running tokens exceed batch limits)
        if pressure_active && total_running_tokens > self.max_batched_tokens && !self.running.is_empty() {
            // Sort running sequences by priority ascending (lowest first)
            self.running.sort_by_key(|r| r.priority);
            let preempted = self.running.remove(0);
            output.preempted_ids.push(preempted.id);
            println!("[Scheduler] Preemption: Swapping request {} to host queue (priority {})", preempted.id, preempted.priority);
            self.swapped.push_back(preempted);
        }

        // 1. Admit from admitted queue up to budget.
        let mut total_prefill = 0usize;
        let current_running = self.running.len();
        while let Some(mut r) = admitted.pop_front() {
            if current_running + output.prefill_ids.len() >= self.max_num_seqs {
                self.waiting.push_back(r);
                continue;
            }

            // Chunked prefill (Sarathi-Serve style, §5.2): drain tokens up to chunked_prefill_size only under load
            let chunk_size = if pressure_active {
                r.prompt_tokens.min(self.chunked_prefill_size)
            } else {
                r.prompt_tokens
            };
            if total_prefill + chunk_size > self.max_batched_tokens {
                self.waiting.push_back(r);
                break;
            }
            
            total_prefill += chunk_size;
            let remaining_tokens = r.prompt_tokens.saturating_sub(chunk_size);
            r.prompt_tokens = chunk_size;
            output.prefill_ids.push(r.id);
            self.running.push(r.clone());
            
            if pressure_active {
                // Return all other admitted requests back to the front of the waiting queue
                while let Some(leftover) = admitted.pop_back() {
                    self.waiting.push_front(leftover);
                }
                if remaining_tokens > 0 {
                    let mut remainder_req = r.clone();
                    remainder_req.prompt_tokens = remaining_tokens;
                    self.waiting.push_front(remainder_req);
                }
                break;
            } else if remaining_tokens > 0 {
                let mut remainder_req = r.clone();
                remainder_req.prompt_tokens = remaining_tokens;
                self.waiting.push_front(remainder_req);
            }
        }

        // 2. Return decode IDs for already-running sequences.
        for r in &self.running {
            if !output.prefill_ids.contains(&r.id) {
                output.decode_ids.push(r.id);
            }
        }

        // 3. Batched LoRA sub-batching grouping (§4.5 requirements)
        // Group running sequences by hypothetical adapter properties to optimize fused kernel pipelines
        let mut adapter_batches: std::collections::HashMap<u32, Vec<u64>> = std::collections::HashMap::new();
        for r in &self.running {
            let mock_adapter_id = (r.id % 2) as u32; // Alternating mock adapter IDs
            adapter_batches.entry(mock_adapter_id).or_default().push(r.id);
        }
        if !adapter_batches.is_empty() {
            println!("[Scheduler] Batched adapter sub-batches: {:?}", adapter_batches);
        }

        output
    }

    /// Called after a sequence completes.
    pub fn finish(&mut self, id: u64) {
        self.running.retain(|r| r.id != id);
        self.swapped.retain(|r| r.id != id);
        self.paused.retain(|r| r.id != id);
    }

    /// Pause a running request — moves it to `paused` queue, keeping its
    /// KV state alive (ref-counted, per §5.4). The request will not be
    /// selected for the running batch until `resume` is called.
    pub fn pause(&mut self, id: u64) -> bool {
        if let Some(pos) = self.running.iter().position(|r| r.id == id) {
            let r = self.running.remove(pos);
            self.paused.push_back(r);
            return true;
        }
        false
    }

    /// Resume a paused request — moves it back to running. O(1), KV
    /// blocks stay alive since the request was never evicted.
    pub fn resume(&mut self, id: u64) -> bool {
        if let Some(pos) = self.paused.iter().position(|r| r.id == id) {
            if let Some(r) = self.paused.remove(pos) {
                self.running.push(r);
                return true;
            }
        }
        false
    }

    /// Returns true if the request is currently paused.
    pub fn is_paused(&self, id: u64) -> bool {
        self.paused.iter().any(|r| r.id == id)
    }
}

// ---------------------------------------------------------------------------
// WI 3.4.1 / 3.4.5 — Hybrid CPU/GPU attention offload (APEX-style).
//
// The *decision* of how to partition a sequence's KV blocks between GPU and
// CPU for a hybrid decode step lives here in the scheduler (Gate 3.6.4:
// scheduling policy must not live in a backend crate). The backend crates
// expose the primitives ("run this partial on CPU", "run that partial on
// GPU"); this module decides *which* blocks go where using the existing
// tier-tracking API from `grim-kvtransport`.
// ---------------------------------------------------------------------------

/// Partition a sequence's physical block list into device-resident and
/// host-offloaded halves, based on each block's current `CacheTier`.
///
/// Per WI 3.4.1: for a given decode step, once some KV blocks are on the
/// `HostRam`/`NvMe` tier and some remain on-device, the attention computation
/// needs contributions from both. This function uses the existing
/// `SharedSpillManager::get_tier` API to classify each block — no new
/// tier-tracking mechanism is added.
///
/// **Tier inference rule** (matches the spill manager's contract):
/// - `get_tier(id) == None` → device/GPU-resident (a block that was `alloc`'d
///   and never demoted has no tier entry).
/// - `get_tier(id) == Some(HostRam)` or `Some(NvMe)` → host/offloaded.
/// - `Some(Gpu)` is theoretically in the enum but never written by this spill
///   manager; treated as device-resident (same as `None`) for safety.
/// - `Some(NvMeWeightStream)` is for weight tensors, not KV blocks; treated as
///   device-resident (should not appear for KV block IDs).
///
/// Returns `(device_blocks, host_blocks)` — the two partitions, preserving
/// the input order within each.
pub fn plan_hybrid_attention_step(
    physical_ids: &[grim_kvtransport::BlockId],
    spill: &grim_kvtransport::SharedSpillManager,
) -> (Vec<grim_kvtransport::BlockId>, Vec<grim_kvtransport::BlockId>) {
    use grim_kvtransport::CacheTier;

    let mut device_blocks = Vec::new();
    let mut host_blocks = Vec::new();
    for &id in physical_ids {
        match spill.get_tier(id) {
            // Offloaded tiers → host side.
            Some(CacheTier::HostRam) | Some(CacheTier::NvMe) => host_blocks.push(id),
            // GPU-resident (explicit Gpu, NvMeWeightStream, or None for fresh
            // alloc) → device side.
            Some(CacheTier::Gpu) | Some(CacheTier::NvMeWeightStream) | None => {
                device_blocks.push(id)
            }
        }
    }
    (device_blocks, host_blocks)
}

#[cfg(test)]
mod hybrid_tests {
    use super::*;
    use std::path::PathBuf;

    fn make_spill() -> grim_kvtransport::SharedSpillManager {
        let dir = std::env::temp_dir().join(format!(
            "grim_hybrid_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        grim_kvtransport::SharedSpillManager::new(dir, 64).unwrap()
    }

    #[test]
    fn all_gpu_resident_blocks_go_to_device_partition() {
        let spill = make_spill();
        // No demotions: all blocks are None (GPU-resident).
        let ids = vec![0usize, 1, 2, 3];
        let (device, host) = plan_hybrid_attention_step(&ids, &spill);
        assert_eq!(device, ids);
        assert!(host.is_empty());
    }

    #[test]
    fn demoted_blocks_go_to_host_partition() {
        let spill = make_spill();
        // Demote blocks 1 and 3 to HostRam.
        spill.demote_to_host(1, vec![0.0; 64], vec![0.0; 64]).unwrap();
        spill.demote_to_host(3, vec![0.0; 64], vec![0.0; 64]).unwrap();
        let ids = vec![0usize, 1, 2, 3];
        let (device, host) = plan_hybrid_attention_step(&ids, &spill);
        assert_eq!(device, vec![0, 2], "GPU-resident blocks");
        assert_eq!(host, vec![1, 3], "offloaded blocks");
    }

    #[test]
    fn empty_block_list_returns_empty_partitions() {
        let spill = make_spill();
        let (device, host) = plan_hybrid_attention_step(&[], &spill);
        assert!(device.is_empty());
        assert!(host.is_empty());
    }

    #[test]
    fn mixed_nvme_and_hostram_all_go_to_host() {
        let spill = make_spill();
        spill.demote_to_host(0, vec![0.0; 64], vec![0.0; 64]).unwrap();
        spill.demote_to_host(1, vec![0.0; 64], vec![0.0; 64]).unwrap();
        spill.demote_to_nvme(1).unwrap(); // block 1 → NvMe
        // block 0 stays HostRam, block 1 → NvMe, block 2 is GPU-resident.
        let ids = vec![0usize, 1, 2];
        let (device, host) = plan_hybrid_attention_step(&ids, &spill);
        assert_eq!(device, vec![2]);
        assert_eq!(host, vec![0, 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admit_under_load() {
        let ctrl = AdmissionController::new(2000, 100);
        let backlog = BatchTokenBacklog { total: 0 };
        let req = Request { id: 1, prompt_tokens: 100, priority: 0 };
        assert_eq!(ctrl.admit(&req, &backlog), AdmissionDecision::Admit);
    }

    #[test]
    fn schedule_basic() {
        let ctrl = AdmissionController::new(2000, 100);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.enqueue(Request { id: 1, prompt_tokens: 128, priority: 0 });
        sched.enqueue(Request { id: 2, prompt_tokens: 256, priority: 0 });
        let out = sched.schedule();
        assert_eq!(out.prefill_ids.len(), 2);
    }

    #[test]
    fn scheduler_budget_limit() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(128, 2, ctrl);
        sched.enqueue(Request { id: 1, prompt_tokens: 128, priority: 0 });
        sched.enqueue(Request { id: 2, prompt_tokens: 128, priority: 0 });
        let out = sched.schedule();
        assert_eq!(out.prefill_ids.len(), 1);
        let out2 = sched.schedule();
        assert_eq!(out2.prefill_ids.len(), 1);
    }

    #[test]
    fn pause_and_resume_moves_request() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.enqueue(Request { id: 1, prompt_tokens: 128, priority: 0 });
        let _ = sched.schedule();
        assert_eq!(sched.running.len(), 1);
        assert_eq!(sched.paused.len(), 0);

        assert!(sched.pause(1));
        assert_eq!(sched.running.len(), 0);
        assert_eq!(sched.paused.len(), 1);
        assert!(sched.is_paused(1));

        assert!(sched.resume(1));
        assert_eq!(sched.running.len(), 1);
        assert_eq!(sched.paused.len(), 0);
        assert!(!sched.is_paused(1));
    }

    #[test]
    fn pause_unknown_request_is_noop() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        assert!(!sched.pause(42));
        assert!(!sched.resume(42));
        assert!(!sched.is_paused(42));
    }

    #[test]
    fn paused_requests_are_not_rescheduled() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.enqueue(Request { id: 1, prompt_tokens: 128, priority: 0 });
        let _ = sched.schedule();
        assert_eq!(sched.running.len(), 1);

        sched.pause(1);
        let out = sched.schedule();
        assert!(out.decode_ids.is_empty(), "paused request must not run");
        assert_eq!(sched.paused.len(), 1);
    }

    #[test]
    fn test_strict_queue_sorting() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.determinism_mode = DeterminismMode::Strict;

        // Enqueue requests out of ID order
        sched.enqueue(Request { id: 3, prompt_tokens: 128, priority: 0 });
        sched.enqueue(Request { id: 1, prompt_tokens: 128, priority: 0 });
        sched.enqueue(Request { id: 2, prompt_tokens: 128, priority: 0 });

        let out = sched.schedule();
        // They should be admitted in order: 1, 2, 3
        assert_eq!(out.prefill_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_scheduler_solo_prompt_floor_check() {
        // Target TTFT = 50ms, throughput rate = 1000 tokens/sec
        // Oversized single request = 100 tokens -> predicted TTFT = 100ms
        let ctrl = AdmissionController::new(50, 0);
        // Force throughput estimate to 100.0 so 100 tokens = 1000ms > 50ms target
        *ctrl.throughput_estimate.lock().unwrap() = 100.0;
        
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.enqueue(Request { id: 1, prompt_tokens: 100, priority: 0 });
        let out = sched.schedule();
        // Livelock floor bypass: should still admit it since backlog is empty
        assert_eq!(out.prefill_ids, vec![1]);
    }

    #[test]
    fn test_chunked_prefill_draining() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.chunked_prefill_size = 50;
        
        // Enqueue multiple items to active pressure (pressure_active = true)
        for i in 0..15 {
            sched.enqueue(Request { id: i, prompt_tokens: 120, priority: 0 });
        }
        let out = sched.schedule();
        // First schedule pass: should consume 50 tokens of request 0, return ID, request stays in queue
        assert_eq!(out.prefill_ids, vec![0]);
        assert_eq!(sched.waiting[0].prompt_tokens, 70);
    }

    #[test]
    fn test_scheduler_preemption() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(100, 8, ctrl);
        
        sched.running.push(Request { id: 1, prompt_tokens: 60, priority: 2 });
        sched.running.push(Request { id: 2, prompt_tokens: 60, priority: 1 }); // Lowest priority

        let out = sched.schedule();
        // Total active tokens (120) > max (100) -> lowest priority (id=2) preempted
        assert_eq!(out.preempted_ids, vec![2]);
        assert_eq!(sched.swapped[0].id, 2);
    }
}