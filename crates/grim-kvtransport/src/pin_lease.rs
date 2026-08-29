//! Pinned Buffer Lease and Stale Lock Monitor.
//!
//! Tracks active host-pinned / device-pinned buffer leases during asynchronous
//! cross-node transfers and device copies. Automatically reclaims abandoned leases
//! on worker timeouts or connection drops to prevent permanent memory deadlocks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Status of a tracked pinned buffer lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseStatus {
    Active,
    Released,
    TimedOut,
}

/// Metadata for an in-flight pinned buffer lease.
#[derive(Debug, Clone)]
pub struct PinnedLease {
    pub buffer_id: usize,
    pub tier: crate::CacheTier,
    pub size_bytes: usize,
    pub acquired_at: Instant,
    pub timeout: Duration,
    pub status: LeaseStatus,
}

/// Monitors and reclaims active pinned memory leases.
#[derive(Debug)]
pub struct PinLeaseMonitor {
    leases: HashMap<usize, PinnedLease>,
    default_timeout: Duration,
}

impl PinLeaseMonitor {
    pub fn new(default_timeout: Duration) -> Self {
        Self {
            leases: HashMap::new(),
            default_timeout,
        }
    }

    /// Acquire a new lease on a pinned buffer.
    pub fn acquire(&mut self, buffer_id: usize, tier: crate::CacheTier, size_bytes: usize) {
        self.leases.insert(
            buffer_id,
            PinnedLease {
                buffer_id,
                tier,
                size_bytes,
                acquired_at: Instant::now(),
                timeout: self.default_timeout,
                status: LeaseStatus::Active,
            },
        );
    }

    /// Explicitly release a pinned lease.
    pub fn release(&mut self, buffer_id: usize) {
        if let Some(lease) = self.leases.get_mut(&buffer_id) {
            lease.status = LeaseStatus::Released;
        }
        self.leases.remove(&buffer_id);
    }

    /// Sweep expired leases and return the IDs of timed-out buffers to be force-reclaimed.
    pub fn sweep_timed_out(&mut self) -> Vec<usize> {
        let now = Instant::now();
        let mut expired = Vec::new();

        for (&id, lease) in self.leases.iter_mut() {
            if lease.status == LeaseStatus::Active
                && now.checked_duration_since(lease.acquired_at).unwrap_or_default() > lease.timeout
            {
                lease.status = LeaseStatus::TimedOut;
                expired.push(id);
            }
        }

        for &id in &expired {
            self.leases.remove(&id);
        }

        expired
    }

    /// Count of active leases.
    pub fn active_count(&self) -> usize {
        self.leases
            .values()
            .filter(|l| l.status == LeaseStatus::Active)
            .count()
    }
}

/// Shared thread-safe pin lease monitor.
#[derive(Debug, Clone)]
pub struct SharedPinLeaseMonitor(pub Arc<Mutex<PinLeaseMonitor>>);

impl SharedPinLeaseMonitor {
    pub fn new(default_timeout: Duration) -> Self {
        Self(Arc::new(Mutex::new(PinLeaseMonitor::new(default_timeout))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_lease_acquire_release() {
        let mut monitor = PinLeaseMonitor::new(Duration::from_millis(50));
        monitor.acquire(1, crate::CacheTier::HostRam, 4096);
        assert_eq!(monitor.active_count(), 1);

        monitor.release(1);
        assert_eq!(monitor.active_count(), 0);
    }

    #[test]
    fn test_pin_lease_timeout_sweep() {
        let mut monitor = PinLeaseMonitor::new(Duration::from_millis(10));
        monitor.acquire(10, crate::CacheTier::Gpu, 8192);

        std::thread::sleep(Duration::from_millis(20));

        let expired = monitor.sweep_timed_out();
        assert_eq!(expired, vec![10]);
        assert_eq!(monitor.active_count(), 0);
    }
}
