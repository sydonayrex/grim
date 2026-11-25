//! Pointer-based speculative rollback for recurrent (SSM/KDA/GDN) state.
//!
//! ReplaySSM's key insight is that rejected draft records need no state
//! reconstruction: append all `T` draft inputs behind the `h` committed
//! records, then commit by advancing `h` by the accepted count. Records beyond
//! `h` are overwritten on the next append. The flush decision depends only on
//! `h` and `T`, so it can be taken before sampling and stays graph-safe.
//!
//! This module owns only the bookkeeping. Binding it to device-side recurrent
//! state is backend/model work; no low-precision checkpoint reads are added
//! here (they are a net loss at batch 1, grim's decode regime).

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayRing<T: Clone> {
    records: Vec<T>,
    /// Number of committed records at the front of `records`.
    committed: usize,
    capacity: usize,
    flushes: u64,
}

impl<T: Clone> ReplayRing<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            records: Vec::new(),
            committed: 0,
            capacity: capacity.max(1),
            flushes: 0,
        }
    }

    pub fn committed_len(&self) -> usize {
        self.committed
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn flushes(&self) -> u64 {
        self.flushes
    }

    pub fn record(&self, i: usize) -> Option<&T> {
        self.records.get(i)
    }

    /// Whether this append would overflow the ring and require folding the
    /// committed prefix into the checkpoint. Deterministic in `h` and `T`.
    pub fn needs_flush(&self, draft_len: usize) -> bool {
        self.committed + draft_len > self.capacity
    }

    /// Append draft records after the committed frontier. Pre-existing
    /// speculative records are overwritten (that *is* the rollback).
    pub fn append_draft(&mut self, drafts: &[T]) {
        let start = self.committed;
        let end = start + drafts.len();
        if self.records.len() < end {
            self.records.resize(end, drafts.first().cloned().unwrap());
        }
        for (slot, rec) in self.records[start..end].iter_mut().zip(drafts.iter()) {
            *slot = rec.clone();
        }
    }

    /// Commit `accepted` draft records. Rejected records stay beyond the
    /// frontier and are overwritten by the next `append_draft`.
    pub fn commit(&mut self, accepted: usize) {
        let n = accepted.min(self.records.len().saturating_sub(self.committed));
        self.committed += n;
    }

    /// Fold the committed prefix into the checkpoint and start a fresh window.
    /// The caller must materialise the state for `records[..committed]` first.
    pub fn flush(&mut self) {
        self.records.drain(..self.committed);
        self.committed = 0;
        self.flushes += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_records_are_overwritten_not_rebuilt() {
        let mut ring = ReplayRing::new(8);
        ring.append_draft(&[1, 2, 3, 4]);
        ring.commit(2);
        assert_eq!(ring.committed_len(), 2);
        assert_eq!(ring.record(2), Some(&3)); // still physically present
        ring.append_draft(&[9, 9]);
        assert_eq!(ring.record(2), Some(&9)); // overwritten in place
        assert_eq!(ring.committed_len(), 2);
    }

    #[test]
    fn flush_is_decidable_before_sampling() {
        let mut ring = ReplayRing::new(4);
        ring.append_draft(&[1, 2, 3]);
        ring.commit(3);
        assert!(!ring.needs_flush(1)); // 3 + 1 == capacity
        assert!(ring.needs_flush(2)); // 3 + 2 > capacity
        ring.append_draft(&[4, 5]);
        ring.commit(1);
        ring.flush();
        assert_eq!(ring.flushes(), 1);
        assert_eq!(ring.committed_len(), 0);
        assert_eq!(ring.record(0), Some(&5));
    }

    #[test]
    fn empty_ring_is_total() {
        let mut ring: ReplayRing<u32> = ReplayRing::new(2);
        assert!(ring.is_empty());
        ring.commit(5); // must not run past the frontier
        assert_eq!(ring.committed_len(), 0);
        ring.append_draft(&[]);
        assert!(ring.is_empty());
    }
}
