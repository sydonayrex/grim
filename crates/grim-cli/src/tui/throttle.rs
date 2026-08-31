//! Frame throttle: gate `term.draw` to at most 60 FPS (16ms interval).
//!
//! Input handling stays latency-sensitive via `request_immediate`, which
//! bypasses the throttle on the next `should_render` check.

use std::time::{Duration, Instant};

/// Minimum interval between rendered frames. 8ms gives ~120 FPS for
/// snappy picker navigation and typing response while still avoiding
/// excessive CPU use during idle streaming.
pub const MIN_FRAME_INTERVAL: Duration = Duration::from_millis(8);

/// Synchronous render scheduler. Lives on the UI thread, no background task.
#[derive(Debug)]
pub struct RenderScheduler {
    /// When the last frame was actually drawn.
    last_render: Instant,
    /// Whether a frame has been requested since the last draw.
    pending: bool,
    /// Whether the next frame should bypass the throttle.
    immediate: bool,
}

impl Default for RenderScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderScheduler {
    /// Create a scheduler that will allow the first frame immediately.
    pub fn new() -> Self {
        Self {
            // Far enough in the past that the first should_render returns true.
            last_render: Instant::now() - MIN_FRAME_INTERVAL - Duration::from_millis(1),
            pending: false,
            immediate: false,
        }
    }

    /// Mark a frame as needed. Throttled to `MIN_FRAME_INTERVAL`.
    pub fn request_render(&mut self) {
        self.pending = true;
    }

    /// Mark a frame as needed and bypass the throttle on the next check.
    ///
    /// Use for input events where latency matters more than frame budget.
    pub fn request_immediate(&mut self) {
        self.pending = true;
        self.immediate = true;
    }

    /// True when a frame should be drawn now. Resets pending state when true.
    pub fn should_render(&mut self) -> bool {
        if !self.pending {
            return false;
        }
        if self.immediate {
            self.immediate = false;
            self.pending = false;
            self.last_render = Instant::now();
            return true;
        }
        if self.last_render.elapsed() >= MIN_FRAME_INTERVAL {
            self.pending = false;
            self.last_render = Instant::now();
            return true;
        }
        false
    }

    /// Force the next pending frame to render regardless of interval.
    ///
    /// Call on terminal resize or explicit full redraw.
    pub fn reset(&mut self) {
        self.last_render = Instant::now() - MIN_FRAME_INTERVAL - Duration::from_millis(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_scheduler_wants_first_frame() {
        let mut s = RenderScheduler::new();
        // First frame should always render (last_render is far in the past).
        s.request_render();
        assert!(
            s.should_render(),
            "first pending frame should render immediately"
        );
    }

    #[test]
    fn throttle_suppresses_rapid_second_frame() {
        let mut s = RenderScheduler::new();
        s.request_render();
        assert!(s.should_render()); // first frame renders
        s.request_render();
        // Second request immediately after should be throttled.
        assert!(
            !s.should_render(),
            "second frame within 16ms should be suppressed"
        );
    }

    #[test]
    fn immediate_bypasses_throttle() {
        let mut s = RenderScheduler::new();
        s.request_render();
        assert!(s.should_render());
        s.request_render();
        assert!(!s.should_render()); // throttled
        s.request_immediate();
        assert!(s.should_render(), "immediate should bypass throttle");
    }

    #[test]
    fn reset_forces_next_frame() {
        let mut s = RenderScheduler::new();
        s.request_render();
        assert!(s.should_render());
        // No pending frame, but reset forces the next one.
        s.reset();
        s.request_render();
        assert!(s.should_render());
    }

    #[test]
    fn no_pending_means_no_render() {
        let mut s = RenderScheduler::new();
        // Never requested, should not render even after reset is not called.
        assert!(!s.should_render());
    }
}
