//! Device-side memory subsystem (Phase-3 §3.1 of the QKV spec).
//!
//! Lives under `memory/` to set up the modularization that the spec's
//! anti-pattern rule requires: every new file under 500 lines, no kernel
//! or pool logic bleeding back into the giant 4630-line `lib.rs`.
//!
//! Modules:
//! - [`pool`] — `DeviceScratchPool`, a thread-safe scratch-buffer pool
//!   with power-of-2 bucketization used by the fused-QKV decode hot
//!   path to replace per-call `hipMalloc`/`hipFree` churn.
//!
//! Skill attribution: see the QKV spec §"Skills map" — `rust-ai-ml-inference-guide`
//! Action 3, `rust-gpu-parallelism` (stream-ordered memory), `rocm-profiling-perf`
//! (allocation is in the optimizer's hot path).

pub mod pool;
