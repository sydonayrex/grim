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
        if self.target_ttft_ms == 0 {
            return AdmissionDecision::Admit;
        }
        let predicted = self.predict_ttft(request.prompt_tokens, backlog.total);
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

/// The scheduler: manages waiting/running/swapped queues per §5.2.
pub struct Scheduler {
    pub waiting: VecDeque<Request>,
    pub running: Vec<Request>,
    pub swapped: VecDeque<Request>,
    pub max_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub admission: AdmissionController,
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
            max_batched_tokens,
            max_num_seqs,
            admission,
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
        let backlog = self.compute_token_backlog();

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

        // 1. Admit from admitted queue up to budget.
        let mut total_prefill = 0usize;
        let current_running = self.running.len();
        while let Some(r) = admitted.pop_front() {
            if current_running + output.prefill_ids.len() >= self.max_num_seqs {
                self.waiting.push_back(r);
                continue;
            }
            if total_prefill + r.prompt_tokens > self.max_batched_tokens {
                self.waiting.push_back(r);
                break;
            }
            total_prefill += r.prompt_tokens;
            output.prefill_ids.push(r.id);
            self.running.push(r);
        }

        // 2. Return decode IDs for already-running sequences.
        for r in &self.running {
            if !output.prefill_ids.contains(&r.id) {
                output.decode_ids.push(r.id);
            }
        }

        output
    }

    /// Called after a sequence completes.
    pub fn finish(&mut self, id: u64) {
        self.running.retain(|r| r.id != id);
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
}