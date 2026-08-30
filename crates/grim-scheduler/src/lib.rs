//! Continuous-batching request scheduler, chunked prefill tracking, and latency-aware admission control.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use grim_core::DeterminismMode;

pub mod readiness_dispatch;
pub mod self_tuning;

pub use readiness_dispatch::{
    MicrobatchTask, ReadinessDispatcher, ReadySet, ScheduleHint, TaskKind,
};
pub use self_tuning::{SelfTuningController, TunableKnob};

/// Real KV-memory pressure source (§5.2): reports current KV pool occupancy
/// in `[0.0, 1.0]`. The engine wires this to the live `KvBlockPool`; tests
/// supply synthetic values. `None` on the scheduler keeps the legacy
/// token-arithmetic-only pressure signal.
pub trait KvPressureSource: Send + Sync {
    fn kv_occupancy(&self) -> f32;
}

/// Adapts a shared KV block pool into a [`KvPressureSource`] via an occupancy
/// closure (e.g. `used_count() / capacity()` over the pool guard). Defined
/// over a closure so the scheduler keeps no grim-memory dependency.
pub struct PoolKvPressure {
    occupancy: Box<dyn Fn() -> f32 + Send + Sync>,
}

impl PoolKvPressure {
    pub fn new(f: impl Fn() -> f32 + Send + Sync + 'static) -> Self {
        Self {
            occupancy: Box::new(f),
        }
    }
}

impl KvPressureSource for PoolKvPressure {
    fn kv_occupancy(&self) -> f32 {
        (self.occupancy)().clamp(0.0, 1.0)
    }
}

/// KV occupancy above which the scheduler treats memory as under real
/// pressure regardless of token arithmetic (default 0.9).
pub const KV_PRESSURE_THRESHOLD: f32 = 0.9;

/// Readiness-driven dispatch tuning constant (RRFP idea). Decode tasks are
/// submitted with high priority so the dispatcher arbitrates them ahead of
/// contending prefill work under pressure.
const READINESS_DECODE_PRIORITY: i32 = 100;

/// A request in the scheduler system.
#[derive(Debug, Clone, Default)]
pub struct Request {
    pub id: u64,
    pub prompt_tokens: usize,
    /// Maximum number of tokens this request may generate, used by admission
    /// guards that reserve KV capacity before selecting a device.
    pub max_new_tokens: usize,
    /// Scheduling priority. One policy governs both consumers: admission
    /// ordering (higher priority first, arrival order breaking ties) and
    /// preemption victim selection (lowest priority, earliest-arrived first).
    /// A request is never starved by admission deferral or rescued by
    /// priority alone — the swap-in path below guarantees re-entry.
    pub priority: i32,
    /// Tokens consumed so far in the current prefill pass (chunked prefill tracking).
    pub consumed_tokens: usize,
    /// Target model id for multi-model setups. None defaults to the first registered model.
    pub model_id: Option<String>,
    /// Adapter ids this request uses for LoRA/batch fusion.
    pub adapter_ids: Vec<u32>,
    /// Actual input token IDs for the prompt. If provided, these are used
    /// instead of synthetic position indices during prefill. Length must match
    /// `prompt_tokens` when present.
    pub input_ids: Option<Vec<u32>>,
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
        if backlog.total <= request.prompt_tokens
            && solo_predicted.as_millis() as u64 > self.target_ttft_ms
        {
            println!(
                "[AdmissionController] Solo-prompt livelock bypass: admitting oversized request {}",
                request.id
            );
            return AdmissionDecision::Admit;
        }

        if self.target_ttft_ms == 0 {
            return AdmissionDecision::Admit;
        }

        let predicted = self.predict_ttft(request.prompt_tokens, backlog.total);

        // ITL (Inter-Token Latency) check (§5.2): verify expected decode latency does not exceed target limit
        let rate = *self.throughput_estimate.lock().unwrap();
        let expected_itl_ms = if rate > 0.0 {
            (1000.0 / rate) as u64
        } else {
            0
        };
        if self.target_itl_ms > 0 && expected_itl_ms > self.target_itl_ms {
            println!(
                "[AdmissionController] Deferring request {} due to ITL constraint violation (expected {}ms > target {}ms)",
                request.id, expected_itl_ms, self.target_itl_ms
            );
            return AdmissionDecision::Defer;
        }

        if predicted.as_millis() as u64 <= self.target_ttft_ms {
            AdmissionDecision::Admit
        } else {
            AdmissionDecision::Defer
        }
    }

    pub fn observe_prefill(&self, prompt_tokens: usize, wall_duration: Duration) {
        let secs = wall_duration.as_secs_f64();
        if secs <= 0.0 || prompt_tokens == 0 {
            return;
        }
        let measured_tps = prompt_tokens as f64 / secs;
        if !measured_tps.is_finite() || measured_tps <= 0.0 {
            return;
        }
        const EMA_ALPHA: f64 = 0.3;
        let mut est = self.throughput_estimate.lock().unwrap();
        if !est.is_finite() || *est <= 0.0 {
            *est = measured_tps;
        } else {
            *est = *est * (1.0 - EMA_ALPHA) + measured_tps * EMA_ALPHA;
        }
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
    pub paused: VecDeque<Request>, // §5.2.1 — explicitly paused, KV retained
    pub max_batched_tokens: usize,
    pub max_num_seqs: usize,
    /// Tuned by [`SelfTuningController::chunked_prefill_size`](crate::self_tuning::KnobKind::ChunkedPrefillSize)
    /// (§5.7): how many tokens from any one prompt are drained per
    /// schedule pass. Drives prefill-vs-decode TTFT balance.
    pub chunked_prefill_size: usize,
    pub admission: AdmissionController,
    pub determinism_mode: DeterminismMode,
    /// Cumulative admission events since scheduler creation. A request counts
    /// once on first admission; a preempted request that is swapped back in
    /// counts again (it is a fresh admission into the batch).
    admitted_total: usize,
    /// Live KV occupancy source wired by the engine (see
    /// [`Scheduler::set_kv_pressure`]). `None` = legacy token-arithmetic-only
    /// pressure.
    kv_pressure: Option<Arc<dyn KvPressureSource>>,
    /// Readiness-driven dispatcher (RRFP idea): under pressure, arbitrates
    /// decode (always ready) vs prefill (ready only if budget fits) instead of
    /// the fixed prefill-first order. `None` keeps the legacy fixed-order path.
    readiness: Option<ReadinessDispatcher>,
}

/// Why readiness-driven dispatch matters here.
///
/// The legacy `schedule()` admits prefill chunks up to the token budget, then
/// returns decode IDs — a fixed prefill-first order. Under pressure a large
/// prefill can starve decode and blow ITL. The RRFP insight ("schedule as a
/// hint, dispatch ready work, skip blocked work") adapts to this single-stage
/// batcher as: decode tasks are always ready (no dependencies); prefill tasks
/// are ready only while budget remains. `ReadinessDispatcher::arbitrate` then
/// picks decode-first under pressure, eliminating the head-of-line prefill
/// block. Enabled with [`Scheduler::set_readiness_dispatch`].
const _READINESS_DOC: () = ();

/// Contiguous segment of sequences sharing a LoRA adapter ID. Advisory
/// summary of one `schedule()` call — execution plans row-level segments
/// via [`LoraRowSegment::plan_for_rows`] from the forwarded batch layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraSegment {
    /// Primary adapter ID (0 = base model, > 0 = fine-tuned adapter).
    pub adapter_id: u32,
    /// Starting sequence offset within the batch.
    pub seq_start: usize,
    /// Number of sequences in this segment.
    pub seq_count: usize,
    /// Sequence IDs belonging to this segment.
    pub seq_ids: Vec<u64>,
}

/// Contiguous segment of *batch rows* sharing one LoRA adapter, for fused
/// multi-LoRA kernel execution (S-LoRA/Punica style) over a stacked
/// `[rows, dim]` matrix.
///
/// Unlike [`LoraSegment`] (sequence-level, advisory), row segments are the
/// execution contract: `row_start`/`row_count` index packed rows directly, so
/// the consumer must plan them from the batch layout it actually forwards
/// ([`LoraRowSegment::plan_for_rows`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraRowSegment {
    /// Adapter applied to every row in this segment (0 = base model, no delta).
    pub adapter_id: u32,
    /// First row of the segment within the stacked matrix.
    pub row_start: usize,
    /// Number of consecutive rows in the segment.
    pub row_count: usize,
}

impl LoraRowSegment {
    /// Plan contiguous row segments from per-row primary adapter ids.
    ///
    /// Rows are consumed in the given order; equal adapter ids adjacent in
    /// that order coalesce into one segment. This is a pure grouping pass —
    /// callers that want maximal contiguity should sort their rows by adapter
    /// id (stable, to keep determinism) before calling.
    pub fn plan_for_rows(row_adapters: &[u32]) -> Vec<Self> {
        let mut segments: Vec<Self> = Vec::new();
        for (row, &adapter_id) in row_adapters.iter().enumerate() {
            match segments.last_mut() {
                Some(seg) if seg.adapter_id == adapter_id => seg.row_count += 1,
                _ => segments.push(Self {
                    adapter_id,
                    row_start: row,
                    row_count: 1,
                }),
            }
        }
        segments
    }
}

/// Result of one `schedule()` call — the engine uses this to run the batch.
#[derive(Debug, Default)]
pub struct SchedulerOutput {
    pub prefill_ids: Vec<u64>,
    pub decode_ids: Vec<u64>,
    pub preempted_ids: Vec<u64>,
    /// Advisory grouping of running sequence IDs by primary LoRA adapter ID
    /// (0 = base model). Nothing in the execution path is required to consume
    /// this: fused batched LoRA dispatch plans its own row segments from the
    /// batch layout it actually forwards (`LoraRowSegment::plan_for_rows`).
    /// This field exists for observability and as a scheduling-level summary.
    pub adapter_batches: std::collections::HashMap<u32, Vec<u64>>,
    /// Contiguous sequence-level segment descriptors (advisory, same contract
    /// as `adapter_batches`).
    pub lora_segments: Vec<LoraSegment>,
}

/// Read-only queue counts for status and observability surfaces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerSnapshot {
    pub active_requests: usize,
    pub waiting_requests: usize,
    /// Cumulative admissions since scheduler creation (`Scheduler::
    /// admitted_total`) — matches the `grim_scheduler_admitted_requests`
    /// counter semantics on the serve surface.
    pub admitted_requests: usize,
    pub paused_requests: usize,
}

impl SchedulerOutput {
    pub fn is_empty(&self) -> bool {
        self.prefill_ids.is_empty() && self.decode_ids.is_empty()
    }
}

impl Scheduler {
    /// Return queue counts without exposing the scheduler's collections to
    /// status consumers.
    pub fn snapshot(&self) -> SchedulerSnapshot {
        SchedulerSnapshot {
            active_requests: self.running.len(),
            waiting_requests: self.waiting.len(),
            admitted_requests: self.admitted_total,
            paused_requests: self.paused.len(),
        }
    }

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
            admitted_total: 0,
            kv_pressure: None,
            readiness: None,
        }
    }

    /// Enable readiness-driven dispatch (RRFP idea). When set, `schedule()`
    /// uses the [`ReadinessDispatcher`] to arbitrate decode-before-prefill under
    /// pressure instead of the legacy fixed prefill-first order. Pass `None` to
    /// restore legacy behavior.
    pub fn set_readiness_dispatch(&mut self, dispatcher: Option<ReadinessDispatcher>) {
        self.readiness = dispatcher;
    }

    /// Wire a live KV occupancy source (§5.2): when occupancy exceeds
    /// [`KV_PRESSURE_THRESHOLD`], the scheduler enters memory pressure
    /// regardless of token arithmetic — preemption and chunked draining
    /// then react to actual pool exhaustion, not just prompt-token sums.
    pub fn set_kv_pressure(&mut self, source: Arc<dyn KvPressureSource>) {
        self.kv_pressure = Some(source);
    }

    /// Current effective pressure signal: `true` when token arithmetic OR
    /// wired KV occupancy indicates pressure.
    fn kv_memory_pressure(&self) -> bool {
        self.kv_pressure
            .as_ref()
            .map(|s| s.kv_occupancy() >= KV_PRESSURE_THRESHOLD)
            .unwrap_or(false)
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
        } else {
            // Admission policy (see `Request::priority`): higher priority
            // first, arrival order breaking ties (stable sort). Checked
            // before sorting so the common already-ordered case stays
            // allocation-free.
            let has_inversion = (1..self.waiting.len())
                .any(|i| self.waiting[i - 1].priority < self.waiting[i].priority);
            if has_inversion {
                let mut temp: Vec<Request> = self.waiting.drain(..).collect();
                temp.sort_by_key(|r| std::cmp::Reverse(r.priority));
                self.waiting = temp.into();
            }
        }

        let backlog = self.compute_token_backlog();
        let total_running_tokens: usize = self
            .running
            .iter()
            .map(|r| r.prompt_tokens.saturating_sub(r.consumed_tokens))
            .sum();
        let pressure_active = backlog.total > self.max_batched_tokens
            || self.waiting.len() > 10
            || total_running_tokens > self.max_batched_tokens
            || self.kv_memory_pressure();

        // Swap-in (§5.2): once pressure lifts, preempted requests re-enter
        // admission from the FRONT of `waiting` (older than new arrivals).
        // `consumed_tokens` is preserved across the swap, so chunked prefill
        // resumes at the true offset, and a fully-prefilled swap-in skips
        // straight to decode (handled in the admission loop below). Before
        // this path existed, `finish()` was the only remover of `swapped` —
        // a preempted request was stranded until it was aborted.
        //
        // Entries already tracked elsewhere (a preempted mid-prefill request
        // whose chunk remainder is still in `waiting`) are dropped: the
        // remaining copy is the live one, and keeping both would schedule
        // the same id twice per pass.
        if !pressure_active {
            let mut swap_back = Vec::new();
            while let Some(r) = self.swapped.pop_front() {
                let already_tracked = self.waiting.iter().any(|w| w.id == r.id)
                    || self.running.iter().any(|w| w.id == r.id);
                if !already_tracked {
                    swap_back.push(r);
                }
            }
            for r in swap_back.into_iter().rev() {
                self.waiting.push_front(r);
            }
        }

        // 0. Admission control: defer requests that would bust the TTFT budget.
        // Push deferred requests to the back of the queue so they don't
        // starve newer requests (livelock prevention).
        let mut admitted = VecDeque::new();
        while let Some(r) = self.waiting.pop_front() {
            if self.admission.admit(&r, &backlog) == AdmissionDecision::Admit {
                admitted.push_back(r);
            } else {
                self.waiting.push_back(r);
                break;
            }
        }

        let mut output = SchedulerOutput::default();

        // Preemption check (§5.2): swap the lowest-priority, earliest-arrived
        // running sequence to `swapped` under token pressure. Victims return
        // via the swap-in path above once pressure lifts (see
        // `Request::priority` for the single ordering policy).
        if pressure_active
            && total_running_tokens > self.max_batched_tokens
            && !self.running.is_empty()
        {
            // Sort running sequences by priority ascending (lowest first;
            // stable sort keeps earliest-arrived first within a priority).
            self.running.sort_by_key(|r| r.priority);
            let preempted = self.running.remove(0);
            output.preempted_ids.push(preempted.id);
            println!(
                "[Scheduler] Preemption: Swapping request {} to host queue (priority {})",
                preempted.id, preempted.priority
            );
            self.swapped.push_back(preempted);
        }

        // 1. Admit from admitted queue up to budget.
        let mut total_prefill = 0usize;
        let current_running = self.running.len();

        // Readiness-driven decode-first interleaving (RRFP idea). The engine
        // runs all admitted prefills before any decode this tick, so under
        // pressure a large newly-admitted prefill chunk head-of-line-blocks
        // ready decode work and spikes ITL. When the dispatcher is enabled and
        // decode work is ready under pressure, interleave: submit the ready
        // decode tasks, let the dispatcher arbitrate decode-first, and if it
        // does, defer new prefill admission to next tick so decode runs now.
        // This is the RRFP "dispatch ready work, skip blocked work" insight
        // applied to the prefill/decode contention — not a new pipeline stage.
        // Without the dispatcher (or off pressure) prefill proceeds greedily.
        let defer_prefill_this_tick = if self.readiness.is_some() && pressure_active {
            let decode_ready: Vec<&Request> = self
                .running
                .iter()
                .filter(|r| r.consumed_tokens >= r.prompt_tokens)
                .collect();
            if decode_ready.is_empty() {
                false
            } else if let Some(ref readiness) = self.readiness {
                // Submit one ready-decode task per decode-eligible request,
                // keyed by request id so they don't collide in the ready set.
                for r in &decode_ready {
                    readiness.submit_task(
                        r.id,
                        r.id,
                        TaskKind::ForwardDecode,
                        READINESS_DECODE_PRIORITY,
                        0,
                    );
                }
                // Arbitration: decode (no dependencies) always wins over a
                // contending prefill, so a decode task is returned.
                readiness
                    .arbitrate()
                    .map(|t| t.kind == TaskKind::ForwardDecode)
                    .unwrap_or(false)
            } else {
                false
            }
        } else {
            false
        };
        let decode_deferred_prefill = defer_prefill_this_tick;

        while let Some(r) = admitted.pop_front() {
            if current_running + output.prefill_ids.len() >= self.max_num_seqs {
                self.waiting.push_back(r);
                continue;
            }

            // Honor the decode-first arbitration: when decode won, defer new
            // prefill admission to next tick so decode runs this tick.
            if decode_deferred_prefill {
                self.waiting.push_back(r);
                continue;
            }
            // Chunked prefill (Sarathi-Serve style, §5.2): drain tokens up to
            // chunked_prefill_size only under load. Chunks operate on the
            // REMAINING unconsumed tokens — F9 (audit): the previous code
            // assigned `consumed_tokens = chunk_size` and sized chunks off the
            // full prompt, so a request scheduled on a 3rd+ pass had its
            // consumed count reset backward and reprocessed the entire prompt
            // from offset 0.
            let remaining_before = r.prompt_tokens.saturating_sub(r.consumed_tokens);
            if remaining_before == 0 {
                // Fully prefilled before leaving the batch (a swap-in): no
                // prefill work remains, so it re-enters `running` directly
                // and becomes decode-eligible this pass instead of paying a
                // zero-token prefill pass.
                match self.running.iter().position(|e| e.id == r.id) {
                    Some(pos) => self.running[pos] = r,
                    None => {
                        self.admitted_total += 1;
                        self.running.push(r);
                    }
                }
                continue;
            }
            let chunk_size = if pressure_active {
                remaining_before.min(self.chunked_prefill_size)
            } else {
                remaining_before
            };
            if total_prefill + chunk_size > self.max_batched_tokens {
                self.waiting.push_back(r);
                break;
            }

            total_prefill += chunk_size;
            let new_consumed = r.consumed_tokens + chunk_size;
            let remaining_tokens = r.prompt_tokens.saturating_sub(new_consumed);
            output.prefill_ids.push(r.id);
            let mut running_req = r.clone();
            running_req.consumed_tokens = new_consumed;
            // One running entry per request: a chunked request re-enters
            // through `waiting` every pass, and pushing a fresh copy each
            // time accumulated duplicates (a 3-pass prefill left 3 entries,
            // tripling that request's decode work every tick and letting
            // preemption swap one copy while another kept running). Replace
            // the prior entry instead of appending.
            match self.running.iter().position(|e| e.id == r.id) {
                Some(pos) => self.running[pos] = running_req,
                None => {
                    self.admitted_total += 1;
                    self.running.push(running_req);
                }
            }

            if pressure_active {
                // Return all other admitted requests back to the front of the waiting queue
                while let Some(leftover) = admitted.pop_back() {
                    self.waiting.push_back(leftover);
                }
                if remaining_tokens > 0 {
                    let mut remainder_req = r.clone();
                    remainder_req.consumed_tokens = new_consumed;
                    self.waiting.push_back(remainder_req);
                }
                break;
            } else if remaining_tokens > 0 {
                let mut remainder_req = r.clone();
                remainder_req.consumed_tokens = new_consumed;
                self.waiting.push_back(remainder_req);
            }
        }

        // 2. Return decode IDs for already-running sequences. A request
        // still mid-prefill (chunked, remainder waiting) must NOT decode:
        // its prompt is not fully in KV yet, and the engine's decode step
        // would feed a prompt token as if it were generated. It idles
        // until its remainder is scheduled.
        for r in &self.running {
            if !output.prefill_ids.contains(&r.id) && r.consumed_tokens >= r.prompt_tokens {
                output.decode_ids.push(r.id);
            }
        }

        // 3. Batched LoRA sub-batching grouping (§4.5 requirements)
        // Group running sequences by actual adapter properties to optimize fused kernel pipelines
        let mut adapter_batches: std::collections::HashMap<u32, Vec<u64>> =
            std::collections::HashMap::new();
        for r in &self.running {
            let primary_adapter = r.adapter_ids.first().copied().unwrap_or(0);
            adapter_batches
                .entry(primary_adapter)
                .or_default()
                .push(r.id);
        }

        // Build deterministic contiguous segments for fused Multi-LoRA kernel execution
        let mut sorted_adapters: Vec<u32> = adapter_batches.keys().copied().collect();
        sorted_adapters.sort_unstable();

        let mut lora_segments = Vec::new();
        let mut seq_offset = 0;
        for adapter_id in sorted_adapters {
            if let Some(seq_ids) = adapter_batches.get(&adapter_id) {
                let count = seq_ids.len();
                lora_segments.push(LoraSegment {
                    adapter_id,
                    seq_start: seq_offset,
                    seq_count: count,
                    seq_ids: seq_ids.clone(),
                });
                seq_offset += count;
            }
        }

        output.adapter_batches = adapter_batches;
        output.lora_segments = lora_segments;

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
) -> (
    Vec<grim_kvtransport::BlockId>,
    Vec<grim_kvtransport::BlockId>,
) {
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
        spill
            .demote_to_host(1, vec![0.0; 64], vec![0.0; 64])
            .unwrap();
        spill
            .demote_to_host(3, vec![0.0; 64], vec![0.0; 64])
            .unwrap();
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
        spill
            .demote_to_host(0, vec![0.0; 64], vec![0.0; 64])
            .unwrap();
        spill
            .demote_to_host(1, vec![0.0; 64], vec![0.0; 64])
            .unwrap();
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
        let req = Request {
            id: 1,
            prompt_tokens: 100,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        };
        assert_eq!(ctrl.admit(&req, &backlog), AdmissionDecision::Admit);
    }

    #[test]
    fn schedule_basic() {
        let ctrl = AdmissionController::new(2000, 100);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 128,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });
        sched.enqueue(Request {
            id: 2,
            prompt_tokens: 256,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });
        let out = sched.schedule();
        assert_eq!(out.prefill_ids.len(), 2);
    }

    #[test]
    fn scheduler_budget_limit() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(128, 2, ctrl);
        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 128,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });
        sched.enqueue(Request {
            id: 2,
            prompt_tokens: 128,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });
        let out = sched.schedule();
        assert_eq!(out.prefill_ids.len(), 1);
        let out2 = sched.schedule();
        assert_eq!(out2.prefill_ids.len(), 1);
    }

    #[test]
    fn pause_and_resume_moves_request() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 128,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });
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
        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 128,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });
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
        sched.enqueue(Request {
            id: 3,
            prompt_tokens: 128,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });
        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 128,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });
        sched.enqueue(Request {
            id: 2,
            prompt_tokens: 128,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });

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
        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 100,
            priority: 0,
            consumed_tokens: 0,
            ..Default::default()
        });
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
            sched.enqueue(Request {
                id: i,
                prompt_tokens: 120,
                priority: 0,
                consumed_tokens: 0,
                ..Default::default()
            });
        }
        let out = sched.schedule();
        // First schedule pass: should consume 50 tokens of request 0, return ID, request stays in queue
        assert_eq!(out.prefill_ids, vec![0]);
        // Request 0 has 120 total tokens; 50 consumed, 70 remaining
        let req0_waiting = sched
            .waiting
            .iter()
            .find(|r| r.id == 0)
            .expect("request 0 in waiting");
        assert_eq!(req0_waiting.consumed_tokens, 50);
        assert_eq!(req0_waiting.prompt_tokens, 120);
    }

    /// F9 (audit): consumed_tokens must ACCUMULATE across scheduling passes,
    /// not reset to the current chunk's size. A 120-token prompt under
    /// pressure with chunk size 50 drains 50 → 100 → 120 across three
    /// passes; the pre-fix code produced 50 → 50 → 20 (reset on every pass)
    /// and, once pressure lifted mid-drain, reprocessed the whole prompt
    /// because chunk sizing ignored already-consumed tokens.
    #[test]
    fn test_chunked_prefill_accumulates_across_passes() {
        let ctrl = AdmissionController::new(0, 0);
        // Small max_batched_tokens so backlog alone (120 > 100) keeps
        // pressure_active on every pass — with one request in the queue the
        // remainder is always the head of `waiting` next pass.
        let mut sched = Scheduler::new(100, 8, ctrl);
        sched.chunked_prefill_size = 50;

        sched.enqueue(Request {
            id: 7,
            prompt_tokens: 120,
            ..Default::default()
        });

        // Pass 1: chunk of 50.
        let out1 = sched.schedule();
        assert_eq!(out1.prefill_ids, vec![7]);
        let r = sched
            .waiting
            .iter()
            .find(|r| r.id == 7)
            .expect("remainder after pass 1");
        assert_eq!(r.consumed_tokens, 50, "pass 1 must consume 50");

        // Pass 2: next 50 → cumulative 100, NOT a reset to 50.
        let out2 = sched.schedule();
        assert_eq!(out2.prefill_ids, vec![7]);
        let r = sched
            .waiting
            .iter()
            .find(|r| r.id == 7)
            .expect("remainder after pass 2");
        assert_eq!(r.consumed_tokens, 100, "pass 2 must accumulate to 100");

        // Pass 3: final 20 → cumulative 120; no remainder left in waiting.
        let out3 = sched.schedule();
        assert_eq!(out3.prefill_ids, vec![7]);
        assert!(
            !sched.waiting.iter().any(|r| r.id == 7),
            "fully-consumed request must not return to waiting"
        );
        // One running entry per request (dedup across chunk passes), and
        // once fully consumed the request becomes decode-eligible.
        let copies = sched.running.iter().filter(|r| r.id == 7).count();
        assert_eq!(
            copies, 1,
            "chunk passes must replace, not accumulate, running entries"
        );
        let running = sched
            .running
            .iter()
            .find(|r| r.id == 7)
            .expect("request 7 in running after final chunk");
        assert_eq!(running.consumed_tokens, 120);
        let out4 = sched.schedule();
        assert_eq!(out4.decode_ids, vec![7], "fully-prefilled request decodes");
    }

    /// Decode-mid-prefill exclusion: a request whose prompt is only partly
    /// consumed (its remainder is still in `waiting`) must NOT appear in
    /// `decode_ids` — the engine would run a decode step against an
    /// incomplete KV and feed a prompt token as if it were generated.
    /// A fully-consumed request decodes normally.
    #[test]
    fn test_decode_excludes_partially_prefilled_requests() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(100, 8, ctrl);
        sched.chunked_prefill_size = 50;

        // Mid-prefill request (chunked) + a fully-prefilled one.
        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 120,
            ..Default::default()
        });
        let out1 = sched.schedule();
        assert_eq!(out1.prefill_ids, vec![1]);
        assert!(out1.decode_ids.is_empty());

        // Enqueue request 2. Note queue order: under pressure the leftover
        // admissions return to `waiting` BEFORE the processed request's
        // remainder, so the order below is [remainder(1), 2] and after the
        // next pass [2, remainder(1)].
        sched.enqueue(Request {
            id: 2,
            prompt_tokens: 40,
            ..Default::default()
        });

        // Pass 2 drains request 1's next chunk (50 → 100 consumed). While
        // any part of its prompt is unconsumed it must NOT decode — the
        // pre-fix scheduler listed it here with a half-filled KV.
        let out2 = sched.schedule();
        assert_eq!(out2.prefill_ids, vec![1]);
        assert!(
            out2.decode_ids.is_empty(),
            "mid-prefill request must not decode (got {:?})",
            out2.decode_ids
        );

        // Pass 3 prefills request 2 in one pass. Request 1 is STILL
        // mid-prefill (100/120) and stays excluded even though another
        // runnable request exists; request 2 was just prefilled this pass.
        let out3 = sched.schedule();
        assert_eq!(out3.prefill_ids, vec![2]);
        assert!(out3.decode_ids.is_empty());

        // Pass 4 completes request 1's prompt; request 2 now decodes.
        let out4 = sched.schedule();
        assert_eq!(out4.prefill_ids, vec![1]);
        assert_eq!(out4.decode_ids, vec![2]);

        // Both complete: both decode.
        let out5 = sched.schedule();
        assert_eq!(out5.decode_ids, vec![1, 2]);
    }
    #[test]
    fn test_scheduler_preemption() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(100, 8, ctrl);

        sched.running.push(Request {
            id: 1,
            prompt_tokens: 60,
            priority: 2,
            consumed_tokens: 0,
            ..Default::default()
        });
        sched.running.push(Request {
            id: 2,
            prompt_tokens: 60,
            priority: 1,
            consumed_tokens: 0,
            ..Default::default()
        }); // Lowest priority

        let out = sched.schedule();
        // Total active tokens (120) > max (100) -> lowest priority (id=2) preempted
        assert_eq!(out.preempted_ids, vec![2]);
        assert_eq!(sched.swapped[0].id, 2);
    }

    /// Swap-in round trip: a preempted request must re-enter the batch once
    /// pressure lifts — `swapped` may not be a terminal state (before the
    /// swap-in path existed, `finish()` was the only remover, so preemption
    /// stranded the request until it was aborted).
    #[test]
    fn test_preempted_request_reenters_after_pressure_lifts() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(100, 8, ctrl);

        // Preempted mid-decode: prompt fully consumed, KV state warm.
        sched.swapped.push_back(Request {
            id: 2,
            prompt_tokens: 60,
            priority: 1,
            consumed_tokens: 60,
            ..Default::default()
        });

        // No pressure (waiting empty, running idle) → swap-in this pass.
        let out = sched.schedule();
        assert!(
            sched.swapped.is_empty(),
            "swapped must drain once pressure lifts"
        );
        assert!(
            sched.running.iter().any(|r| r.id == 2),
            "swap-in must re-enter running"
        );
        // Fully-prefilled swap-in skips the prefill pass and decodes.
        assert!(
            out.decode_ids.contains(&2),
            "fully-prefilled swap-in decodes without a 0-token prefill pass"
        );
    }

    /// A preempted request whose chunked-prefill remainder is still in
    /// `waiting` must not be duplicated by the swap-in: the remainder copy
    /// is the live one, and keeping both would schedule the id twice.
    #[test]
    fn test_swap_in_dedups_against_live_remainder() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);

        sched.swapped.push_back(Request {
            id: 5,
            prompt_tokens: 120,
            consumed_tokens: 50,
            ..Default::default()
        });
        sched.waiting.push_back(Request {
            id: 5,
            prompt_tokens: 120,
            consumed_tokens: 50,
            ..Default::default()
        });

        let out = sched.schedule();
        let prefilled = out.prefill_ids.iter().filter(|&&id| id == 5).count();
        assert_eq!(prefilled, 1, "request 5 must be scheduled at most once");
    }

    /// Under pressure, `swapped` holds; only when the pressure signal lifts
    /// do swapped requests come back.
    #[test]
    fn test_swapped_holds_until_pressure_lifts() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(100, 8, ctrl);

        sched.swapped.push_back(Request {
            id: 9,
            prompt_tokens: 500,
            consumed_tokens: 500,
            ..Default::default()
        });
        // Backlog 600 > max_batched_tokens 100 → pressure active.
        sched.enqueue(Request {
            id: 10,
            prompt_tokens: 600,
            ..Default::default()
        });

        let _ = sched.schedule();
        assert_eq!(sched.swapped.len(), 1, "pressure active → swap-in holds");

        // Drain the backlog → pressure lifts → swap-in fires.
        sched.waiting.clear();
        let out = sched.schedule();
        assert!(sched.swapped.is_empty());
        assert!(
            out.decode_ids.contains(&9),
            "swap-in decodes after re-entry"
        );
    }

    /// `admitted_requests` is a real cumulative counter, not a hardcoded 0:
    /// first admission counts once, chunked re-passes don't recount,
    /// swap-in re-entry counts as a fresh admission.
    #[test]
    fn test_snapshot_admitted_requests_counts_admissions() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        assert_eq!(sched.snapshot().admitted_requests, 0);

        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 128,
            ..Default::default()
        });
        sched.enqueue(Request {
            id: 2,
            prompt_tokens: 128,
            ..Default::default()
        });
        let _ = sched.schedule();
        assert_eq!(sched.snapshot().admitted_requests, 2);

        // Chunked re-admission of an already-running request must not
        // inflate the counter (one running entry per request is replaced).
        let out = sched.schedule();
        let _ = out;
        assert_eq!(sched.snapshot().admitted_requests, 2);

        // Swap-in re-entry is a fresh admission event.
        sched.swapped.push_back(Request {
            id: 3,
            prompt_tokens: 64,
            consumed_tokens: 64,
            ..Default::default()
        });
        let _ = sched.schedule();
        assert_eq!(sched.snapshot().admitted_requests, 3);
    }

    /// Single priority policy: admission order is priority-descending with
    /// arrival order breaking ties (stable), while victim selection evicts
    /// the lowest priority. High-priority work is admitted before older
    /// low-priority work.
    #[test]
    fn test_priority_orders_admission_arrival_breaks_ties() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);

        // Arrival order 1(p0), 2(p5), 3(p0).
        for (id, priority) in [(1u64, 0i32), (2, 5), (3, 0)] {
            sched.enqueue(Request {
                id,
                prompt_tokens: 128,
                priority,
                ..Default::default()
            });
        }
        let out = sched.schedule();
        assert_eq!(
            out.prefill_ids,
            vec![2, 1, 3],
            "priority 5 first; arrival order 1-before-3 within priority 0"
        );
    }

    struct FixedKvPressure(f32);
    impl KvPressureSource for FixedKvPressure {
        fn kv_occupancy(&self) -> f32 {
            self.0
        }
    }

    /// A wired KV source above [`KV_PRESSURE_THRESHOLD`] puts the scheduler
    /// under memory pressure even with a small token backlog: chunked
    /// draining engages. Below the threshold (or unwired), the same request
    /// prefills in a single pass.
    #[test]
    fn test_kv_memory_pressure_drives_scheduling() {
        // 120-token prompt, chunk 50, huge token budget → token arithmetic
        // says no pressure; only the KV signal can force chunked draining.
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.chunked_prefill_size = 50;
        sched.set_kv_pressure(Arc::new(FixedKvPressure(0.95)));
        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 120,
            ..Default::default()
        });

        let out = sched.schedule();
        assert_eq!(out.prefill_ids, vec![1]);
        let remainder = sched
            .waiting
            .iter()
            .find(|r| r.id == 1)
            .expect("KV pressure must chunk-drain: remainder stays in waiting");
        assert_eq!(
            remainder.consumed_tokens, 50,
            "first chunk = 50 despite tiny backlog — KV occupancy drives pressure"
        );
    }

    #[test]
    fn test_low_kv_occupancy_keeps_legacy_pressure_only() {
        let ctrl = AdmissionController::new(0, 0);
        let mut sched = Scheduler::new(4096, 8, ctrl);
        sched.set_kv_pressure(Arc::new(FixedKvPressure(0.1)));
        sched.enqueue(Request {
            id: 1,
            prompt_tokens: 120,
            ..Default::default()
        });
        let out = sched.schedule();
        assert_eq!(out.prefill_ids, vec![1], "fully prefilled in one pass");
        assert!(
            sched.waiting.iter().all(|r| r.id != 1),
            "no remainder without pressure"
        );
    }

    #[test]
    fn test_schedule_adapter_batches_grouping() {
        let ctrl = AdmissionController::new(1000, 1000);
        let mut sched = Scheduler::new(4096, 8, ctrl);

        sched.running.push(Request {
            id: 10,
            prompt_tokens: 10,
            adapter_ids: vec![101],
            ..Default::default()
        });
        sched.running.push(Request {
            id: 20,
            prompt_tokens: 10,
            adapter_ids: vec![101],
            ..Default::default()
        });
        sched.running.push(Request {
            id: 30,
            prompt_tokens: 10,
            adapter_ids: vec![202],
            ..Default::default()
        });
        sched.running.push(Request {
            id: 40,
            prompt_tokens: 10,
            adapter_ids: Vec::new(),
            ..Default::default()
        });

        let out = sched.schedule();
        assert_eq!(out.adapter_batches.get(&101), Some(&vec![10, 20]));
        assert_eq!(out.adapter_batches.get(&202), Some(&vec![30]));
        assert_eq!(out.adapter_batches.get(&0), Some(&vec![40]));

        // Verify contiguous segment layout (0, 101, 202)
        assert_eq!(out.lora_segments.len(), 3);
        assert_eq!(out.lora_segments[0], LoraSegment {
            adapter_id: 0,
            seq_start: 0,
            seq_count: 1,
            seq_ids: vec![40],
        });
        assert_eq!(out.lora_segments[1], LoraSegment {
            adapter_id: 101,
            seq_start: 1,
            seq_count: 2,
            seq_ids: vec![10, 20],
        });
        assert_eq!(out.lora_segments[2], LoraSegment {
            adapter_id: 202,
            seq_start: 3,
            seq_count: 1,
            seq_ids: vec![30],
        });
    }

    #[test]
    fn test_lora_row_segment_planning() {
        // Empty batch -> no segments.
        assert!(LoraRowSegment::plan_for_rows(&[]).is_empty());

        // Adjacent equal ids coalesce; every id change opens a new segment.
        let segs = LoraRowSegment::plan_for_rows(&[7, 7, 0, 0, 0, 9]);
        assert_eq!(
            segs,
            vec![
                LoraRowSegment { adapter_id: 7, row_start: 0, row_count: 2 },
                LoraRowSegment { adapter_id: 0, row_start: 2, row_count: 3 },
                LoraRowSegment { adapter_id: 9, row_start: 5, row_count: 1 },
            ]
        );

        // Single base-only batch: one passthrough segment covering all rows.
        assert_eq!(
            LoraRowSegment::plan_for_rows(&[0, 0]),
            vec![LoraRowSegment { adapter_id: 0, row_start: 0, row_count: 2 }]
        );
    }

    /// R1 validation: with the readiness dispatcher enabled, a decode-eligible
    /// running request wins arbitration over a contending new prefill under
    /// pressure, so the prefill is deferred to protect ITL (RRFP decode-first
    /// interleaving). Without the dispatcher, the prefill is admitted greedily.
    #[test]
    fn test_readiness_dispatch_defers_prefill_under_pressure() {
        use crate::readiness_dispatch::ReadinessDispatcher;

        let make_sched = |with_readiness: bool| -> Scheduler {
            let ctrl = AdmissionController::new(1000, 1000);
            let mut sched = Scheduler::new(64, 8, ctrl);
            if with_readiness {
                sched.set_readiness_dispatch(Some(ReadinessDispatcher::new(0, None)));
            }
            sched.max_batched_tokens = 64;
            sched.chunked_prefill_size = 32;
            // One decode-eligible running request (fully prefilled).
            sched.running.push(Request {
                id: 1,
                prompt_tokens: 16,
                consumed_tokens: 16,
                ..Default::default()
            });
            // A new large prefill waiting to be admitted.
            sched.waiting.push_back(Request {
                id: 2,
                prompt_tokens: 100,
                ..Default::default()
            });
            sched
        };

        // Without readiness dispatch: prefill is admitted greedily (legacy).
        {
            let mut sched = make_sched(false);
            let out = sched.schedule();
            assert!(
                out.prefill_ids.contains(&2),
                "legacy path should admit the prefill"
            );
            assert!(out.decode_ids.contains(&1), "decode should be returned");
        }

        // With readiness dispatch + pressure: decode wins arbitration, prefill
        // is deferred to protect ITL.
        {
            let mut sched = make_sched(true);
            // Force pressure so the decode-first gate is active: backlog from the
            // waiting request exceeds the token budget.
            let out = sched.schedule();
            assert!(
                out.decode_ids.contains(&1),
                "decode-eligible request must be returned"
            );
            assert!(
                !out.prefill_ids.contains(&2),
                "prefill must be deferred when decode won arbitration under pressure"
            );
        }
    }
}
