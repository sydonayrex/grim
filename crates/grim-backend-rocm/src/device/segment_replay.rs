//! SPEED-GRAPH Option B: per-segment HIP-graph capture/replay for the decode
//! path (llama.cpp "graphs reused" mechanism, applied to static layer
//! segments — norms + GEMV chains — while dynamic segments (RoPE, KV append,
//! attention) stay eager).
//!
//! Mechanics: the caller wraps a static segment's host code in
//! [`run_segment`]. First call records the segment's device launches into a
//! HIP graph under `key` (relaxed capture — pooled allocations allowed) and
//! then replays once to execute. Later calls run the SAME host code with
//! kernel launches SUPPRESSED (deterministic host sequence → the pooled
//! allocator hands back the captured addresses) and enqueue one
//! `hipGraphLaunch` instead of N launches.

use std::sync::atomic::{AtomicBool, Ordering};


use crate::device::roc_device::RocmDevice;

/// When set, `launch_compute_kernel_with_solution` returns without launching.
/// Scoped strictly inside [`run_segment`]'s replay branch.
pub static SEGMENT_SUPPRESS: AtomicBool = AtomicBool::new(false);

/// One-time flag per device? No — global: decode is single-threaded and the
/// capture failure just downgrades that segment to eager for this process.
static CAPTURE_POISON: AtomicBool = AtomicBool::new(false);

impl RocmDevice {
    /// Execute `f` as a captured segment: first call captures + replays,
    /// later calls suppress the launches inside `f` and replay the recorded
    /// graph instead. Falls back to plain eager execution when capture is
    /// disabled, was poisoned by a prior failure, or the host code cannot be
    /// captured (e.g. it host-syncs — capture then aborts).
    pub fn run_segment<T, E>(
        &self,
        key: &str,
        f: impl Fn() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<grim_tensor::error::Error>,
    {
        // MEASURED (gfx1201, LFM2.5-350M): 32 small segment graphs replay at
        // roughly the same cost as the ~256 eager fast-path launches they
        // replace — net +0.4 ms/tok. HIP graph launch has a fixed cost that
        // only amortizes over LARGE node counts. Off by default; opt in with
        // GRIM_SEGMENT_GRAPH=1 to re-measure.
        if !matches!(
            std::env::var("GRIM_SEGMENT_GRAPH").as_deref(),
            Ok("1" | "true" | "on")
        ) || !self.graph_capture_enabled()
            || CAPTURE_POISON.load(Ordering::Relaxed)
        {
            return f();
        }

        // Replay phase: host bookkeeping runs, launches suppressed, one
        // hipGraphLaunch enqueues the whole recorded segment.
        if self.has_captured_graph(key) {
            SEGMENT_SUPPRESS.store(true, Ordering::SeqCst);
            let r = f();
            SEGMENT_SUPPRESS.store(false, Ordering::SeqCst);
            let r = r?;
            self.replay_graph(key)?;
            return Ok(r);
        }

        // Capture phase: record the segment, replay once to execute it.
        match self.begin_graph_capture(key) {
            Ok(()) => {
                let r = f();
                match self.end_graph_capture(key) {
                    Ok(()) if self.has_captured_graph(key) => {
                        self.replay_graph(key)?;
                        r
                    }
                    // Capture aborted (host sync inside f, capture already
                    // active, ...). The recorded launches were invalidated, so
                    // re-run the segment eagerly and poison this path.
                    _ => {
                        CAPTURE_POISON.store(true, Ordering::Relaxed);
                        eprintln!(
                            "[speed-graph] segment '{key}' capture failed — eager for this process"
                        );
                        f()
                    }
                }
            }
            // begin failed (capture disabled races, already active, ...).
            Err(_) => f(),
        }
    }
}
