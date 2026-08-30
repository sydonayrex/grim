//! Fixed-capacity circular ring tracking generation speeds for sparkline widget.
//!
//! Stores history of tokens-per-second samples for real-time visualization
//! in the diagnostics sidebar.

/// Fixed-capacity buffer storing recent tokens-per-second integer metrics.
#[derive(Debug, Clone)]
pub struct SpeedHistory {
    buffer: Vec<u64>,
    capacity: usize,
}

impl Default for SpeedHistory {
    fn default() -> Self {
        Self::new(32)
    }
}

impl SpeedHistory {
    /// Create new history buffer with a maximum number of data points.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Record a speed sample in tok/s.
    pub fn record(&mut self, tps: u64) {
        if self.buffer.len() >= self.capacity {
            self.buffer.remove(0);
        }
        self.buffer.push(tps);
    }

    /// View history as a continuous slice for `ratatui::widgets::Sparkline`.
    pub fn as_slice(&self) -> &[u64] {
        &self.buffer
    }

    /// Clear all recorded speed samples.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speed_history_capacity() {
        let mut hist = SpeedHistory::new(3);
        hist.record(10);
        hist.record(20);
        hist.record(30);
        hist.record(40);
        assert_eq!(hist.as_slice(), &[20, 30, 40]);
    }
}
