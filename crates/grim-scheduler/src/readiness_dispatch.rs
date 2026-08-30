//! Readiness-Driven Runtime for Pipeline-Parallel Dispatch (RRFP).
//!
//! Implements out-of-order readiness-driven stage scheduling based on
//! Liu et al. (arXiv:2605.18750). Treats pipeline schedules as non-binding
//! priority hints, dispatching ready microbatches from a `ReadySet` to absorb
//! PCIe/interconnect communication jitter without stalling physical stages.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use grim_core::error::{Error, Result};

/// Classification of computation and communication tasks in a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskKind {
    /// Prefill / chunked prefill forward compute.
    ForwardPrefill,
    /// Autoregressive decode forward compute.
    ForwardDecode,
    /// MoE expert routing and dispatch.
    MoERouteAndDispatch,
    /// Inter-stage activation communication (send/recv).
    ActivationComm,
    /// Backward gradient compute (training/speculative verify).
    BackwardCompute,
}

/// A discrete microbatch execution task with precedence constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicrobatchTask {
    /// Unique microbatch identifier.
    pub microbatch_id: u64,
    /// Originating request identifier.
    pub request_id: u64,
    /// Pipeline stage assigned to execute this task.
    pub stage_id: usize,
    /// Task operation type.
    pub kind: TaskKind,
    /// Scheduling priority (higher executes first).
    pub priority: i32,
    /// Number of unsatisfied predecessor dependencies.
    pub pending_dependencies: usize,
    /// Arrival sequence counter for FIFO tie-breaking.
    pub arrival_epoch: u64,
}

impl MicrobatchTask {
    /// Check if this task has all dependencies satisfied and is ready for dispatch.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.pending_dependencies == 0
    }
}

/// Buffer set holding ready, pending, and completed tasks for a pipeline stage.
#[derive(Debug, Default)]
pub struct ReadySet {
    /// Unfinished tasks keyed by `(microbatch_id, kind)`.
    pub all_tasks: HashMap<(u64, TaskKind), MicrobatchTask>,
    /// Set of ready microbatch IDs organized by task kind.
    pub ready_by_kind: HashMap<TaskKind, VecDeque<(u64, i32, u64)>>, // (microbatch_id, priority, arrival_epoch)
}

impl ReadySet {
    /// Insert a new task with its dependency count.
    pub fn insert(&mut self, task: MicrobatchTask) {
        let key = (task.microbatch_id, task.kind);
        if task.is_ready() {
            self.ready_by_kind
                .entry(task.kind)
                .or_default()
                .push_back((task.microbatch_id, task.priority, task.arrival_epoch));
        }
        self.all_tasks.insert(key, task);
    }

    /// Satisfy a dependency for a task. If pending reaches 0, moves task to ready queue.
    pub fn mark_dependency_satisfied(&mut self, microbatch_id: u64, kind: TaskKind) -> bool {
        if let Some(task) = self.all_tasks.get_mut(&(microbatch_id, kind)) {
            if task.pending_dependencies > 0 {
                task.pending_dependencies -= 1;
                if task.pending_dependencies == 0 {
                    self.ready_by_kind
                        .entry(kind)
                        .or_default()
                        .push_back((task.microbatch_id, task.priority, task.arrival_epoch));
                    return true;
                }
            }
        }
        false
    }

    /// Total number of currently ready tasks across all kinds.
    pub fn ready_count(&self) -> usize {
        self.ready_by_kind.values().map(|q| q.len()).sum()
    }
}

/// Priority hint order for scanning executable work.
#[derive(Debug, Clone)]
pub struct ScheduleHint {
    /// Preferred execution ordering of task kinds.
    pub priority_kinds: Vec<TaskKind>,
}

impl Default for ScheduleHint {
    fn default() -> Self {
        Self {
            priority_kinds: vec![
                TaskKind::ForwardDecode,       // Latency-sensitive decode first
                TaskKind::MoERouteAndDispatch, // Dispatch experts early to overlap GEMMs
                TaskKind::ForwardPrefill,      // High-throughput prefill
                TaskKind::ActivationComm,
                TaskKind::BackwardCompute,
            ],
        }
    }
}

/// Readiness-driven stage dispatcher implementing RRFP out-of-order execution.
pub struct ReadinessDispatcher {
    /// Stage identifier.
    pub stage_id: usize,
    /// Active ready set.
    pub ready_set: Mutex<ReadySet>,
    /// Non-binding priority hint order.
    pub hint: ScheduleHint,
    /// Monotonic arrival counter.
    arrival_counter: Mutex<u64>,
}

impl ReadinessDispatcher {
    /// Create a dispatcher for a pipeline stage.
    pub fn new(stage_id: usize, hint: Option<ScheduleHint>) -> Self {
        Self {
            stage_id,
            ready_set: Mutex::new(ReadySet::default()),
            hint: hint.unwrap_or_default(),
            arrival_counter: Mutex::new(0),
        }
    }

    /// Submit a task into the pipeline stage with predecessor requirements.
    pub fn submit_task(
        &self,
        microbatch_id: u64,
        request_id: u64,
        kind: TaskKind,
        priority: i32,
        dependencies: usize,
    ) {
        let mut arrival = self.arrival_counter.lock().unwrap();
        *arrival += 1;
        let epoch = *arrival;

        let task = MicrobatchTask {
            microbatch_id,
            request_id,
            stage_id: self.stage_id,
            kind,
            priority,
            pending_dependencies: dependencies,
            arrival_epoch: epoch,
        };

        let mut rs = self.ready_set.lock().unwrap();
        rs.insert(task);
    }

    /// Notify that an incoming activation or predecessor operation has arrived.
    pub fn on_predecessor_completed(&self, microbatch_id: u64, kind: TaskKind) {
        let mut rs = self.ready_set.lock().unwrap();
        rs.mark_dependency_satisfied(microbatch_id, kind);
    }

    /// Arbitrate and select the next executable task according to the priority hint.
    ///
    /// If the highest-priority kind has no ready tasks, skips to the next kind
    /// in the hint list, completely eliminating idle wait bubbles.
    pub fn arbitrate(&self) -> Option<MicrobatchTask> {
        let mut rs = self.ready_set.lock().unwrap();

        for &kind in &self.hint.priority_kinds {
            if let Some(queue) = rs.ready_by_kind.get_mut(&kind) {
                if let Some((mb_id, _, _)) = queue.pop_front() {
                    if let Some(task) = rs.all_tasks.remove(&(mb_id, kind)) {
                        return Some(task);
                    }
                }
            }
        }

        // Fallback: check any remaining ready task
        let ready_kinds: Vec<TaskKind> = rs.ready_by_kind.keys().cloned().collect();
        for kind in ready_kinds {
            if let Some(queue) = rs.ready_by_kind.get_mut(&kind) {
                if let Some((mb_id, _, _)) = queue.pop_front() {
                    if let Some(task) = rs.all_tasks.remove(&(mb_id, kind)) {
                        return Some(task);
                    }
                }
            }
        }

        None
    }

    /// Coordinate microbatch selection across Tensor-Parallel ranks to ensure
    /// identical collective invocation order.
    pub fn coordinate_tp_selection(
        &self,
        local_choice: Option<&MicrobatchTask>,
        peer_choices: &[Option<MicrobatchTask>],
    ) -> Result<Option<MicrobatchTask>> {
        // All non-None choices must agree on microbatch_id and kind
        let mut elected: Option<MicrobatchTask> = local_choice.cloned();
        for peer in peer_choices {
            if let Some(p) = peer {
                if let Some(ref e) = elected {
                    if e.microbatch_id != p.microbatch_id || e.kind != p.kind {
                        return Err(Error::Config(format!(
                            "TP Coordination divergence: local ({:?}, mb {}) vs peer ({:?}, mb {})",
                            e.kind, e.microbatch_id, p.kind, p.microbatch_id
                        )));
                    }
                } else {
                    elected = Some(p.clone());
                }
            }
        }
        Ok(elected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_readiness_dispatch_out_of_order_execution() {
        let dispatcher = ReadinessDispatcher::new(0, None);

        // Submit microbatch 1 with 1 pending dependency (blocked)
        dispatcher.submit_task(1, 100, TaskKind::ForwardPrefill, 10, 1);
        // Submit microbatch 2 with 0 dependencies (ready)
        dispatcher.submit_task(2, 101, TaskKind::ForwardPrefill, 5, 0);

        // Arbitrate: microbatch 2 must be dispatched immediately even though MB 1 has higher priority
        let first = dispatcher.arbitrate().expect("MB 2 should be ready");
        assert_eq!(first.microbatch_id, 2);

        // Now satisfy MB 1's dependency
        dispatcher.on_predecessor_completed(1, TaskKind::ForwardPrefill);

        // Arbitrate: microbatch 1 is now dispatched
        let second = dispatcher.arbitrate().expect("MB 1 should be ready now");
        assert_eq!(second.microbatch_id, 1);

        // Queue is now empty
        assert!(dispatcher.arbitrate().is_none());
    }

    #[test]
    fn test_decode_priority_over_prefill() {
        let dispatcher = ReadinessDispatcher::new(0, None);

        // Submit ready prefill task and ready decode task
        dispatcher.submit_task(1, 100, TaskKind::ForwardPrefill, 0, 0);
        dispatcher.submit_task(2, 101, TaskKind::ForwardDecode, 0, 0);

        // Decode must be selected first according to ScheduleHint
        let task = dispatcher.arbitrate().unwrap();
        assert_eq!(task.kind, TaskKind::ForwardDecode);
        assert_eq!(task.microbatch_id, 2);
    }

    #[test]
    fn test_tp_coordination_consensus_and_divergence() {
        let dispatcher = ReadinessDispatcher::new(0, None);
        let task_a = MicrobatchTask {
            microbatch_id: 42,
            request_id: 1,
            stage_id: 0,
            kind: TaskKind::ForwardPrefill,
            priority: 1,
            pending_dependencies: 0,
            arrival_epoch: 1,
        };

        // Consensus case
        let result = dispatcher
            .coordinate_tp_selection(Some(&task_a), &[Some(task_a.clone())])
            .unwrap();
        assert_eq!(result.unwrap().microbatch_id, 42);

        // Divergence case
        let mut task_b = task_a.clone();
        task_b.microbatch_id = 43;
        let err = dispatcher
            .coordinate_tp_selection(Some(&task_a), &[Some(task_b)])
            .unwrap_err();
        assert!(err.to_string().contains("TP Coordination divergence"));
    }
}
