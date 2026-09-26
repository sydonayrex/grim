//! Device-allocation ledger: which pointer range belongs to which GPU.
//!
//! A HIP memory fault reports a *virtual address*, not a tensor name. Without a
//! record of what was allocated where, that address resolves to nothing and the
//! only available response is to guess. The KMD's `HsaMemoryAccessFault`
//! (`hsakmttypes.h`) carries `NodeId`, `VirtualAddress` and a
//! `Failure` bitfield, so the address can be looked up here and compared
//! against the owning allocation's ordinal. Agreement points at buffer
//! lifetime; disagreement is a cross-device placement bug, proven rather than
//! hypothesised.
//!
//! Entries are registered from the two real allocation paths in
//! `storage.rs` and dropped on release, so the ledger describes actual VRAM
//! rather than an intention.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// One live device allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocRecord {
    /// Base device pointer.
    pub ptr: u64,
    /// Size of the allocation in bytes.
    pub bytes: u64,
    /// Device ordinal the allocation was made on.
    pub ordinal: usize,
    /// True when the allocation is HIP managed (host-backed) rather than VRAM.
    pub managed: bool,
    /// Caller-supplied label, e.g. `"layer 12 attn_q"`.
    pub owner: String,
}

impl AllocRecord {
    /// Does `addr` fall inside this allocation? Half-open, so the byte one past
    /// the end belongs to whatever is allocated next.
    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.ptr && addr < self.ptr.saturating_add(self.bytes)
    }

    /// Byte offset of `addr` within this allocation.
    pub fn offset(&self, addr: u64) -> u64 {
        addr.saturating_sub(self.ptr)
    }
}

/// Result of attributing a faulting address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Attribution {
    /// The address is not inside any live allocation. Either the buffer was
    /// already freed (a lifetime bug) or the address is not ours.
    Unowned { addr: u64 },
    /// The address is inside an allocation on `ordinal`, and the faulting node
    /// agrees with it.
    Owned {
        record: AllocRecord,
        offset: u64,
        fault_ordinal: usize,
    },
    /// The address is inside an allocation, but the allocation's device is not
    /// the device that faulted. This is the cross-device placement bug.
    WrongDevice {
        record: AllocRecord,
        offset: u64,
        fault_ordinal: usize,
    },
    /// More than one live allocation contains the address. Should not happen
    /// for non-overlapping VRAM, so it is reported rather than guessed at.
    Ambiguous {
        addr: u64,
        candidates: Vec<AllocRecord>,
    },
}

fn table() -> &'static Mutex<HashMap<u64, AllocRecord>> {
    static TABLE: OnceLock<Mutex<HashMap<u64, AllocRecord>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Live allocation count, for tests and for a load-time summary.
static TRACKED: AtomicU64 = AtomicU64::new(0);

/// Record a device allocation. `ptr` is the key; re-registering the same
/// pointer replaces the previous record.
pub fn register(ptr: u64, bytes: u64, ordinal: usize, managed: bool, owner: &str) {
    let mut t = table().lock().unwrap_or_else(|e| e.into_inner());
    if t.insert(ptr, AllocRecord { ptr, bytes, ordinal, managed, owner: owner.to_string() })
        .is_none()
    {
        TRACKED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Drop a record when its allocation is released.
pub fn deregister(ptr: u64) {
    let mut t = table().lock().unwrap_or_else(|e| e.into_inner());
    if t.remove(&ptr).is_some() {
        TRACKED.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Every allocation whose range contains `addr`, narrowest first.
///
/// Narrowest-first matters when ranges nest: the more specific allocation is
/// the more likely owner of a faulting address.
pub fn candidates(addr: u64) -> Vec<AllocRecord> {
    let t = table().lock().unwrap_or_else(|e| e.into_inner());
    let mut hits: Vec<AllocRecord> = t.values().filter(|r| r.contains(addr)).cloned().collect();
    hits.sort_by_key(|r| r.bytes);
    hits
}

/// Attribute a faulting address to an allocation and compare devices.
///
/// `fault_ordinal` is the faulting device in *our* ordinal numbering, not the
/// raw `NodeId` from the KMD - the caller maps that first, because the two
/// numbering schemes are not the same.
pub fn attribute(addr: u64, fault_ordinal: usize) -> Attribution {
    match candidates(addr).as_slice() {
        [] => Attribution::Unowned { addr },
        [only] => {
            let offset = only.offset(addr);
            if only.ordinal == fault_ordinal {
                Attribution::Owned { record: only.clone(), offset, fault_ordinal }
            } else {
                Attribution::WrongDevice {
                    record: only.clone(),
                    offset,
                    fault_ordinal,
                }
            }
        }
        many => Attribution::Ambiguous { addr, candidates: many.to_vec() },
    }
}

/// Snapshot every live allocation, ordered by pointer.
pub fn snapshot() -> Vec<AllocRecord> {
    let t = table().lock().unwrap_or_else(|e| e.into_inner());
    let mut all: Vec<AllocRecord> = t.values().cloned().collect();
    all.sort_by_key(|r| r.ptr);
    all
}

/// How many allocations are currently tracked.
pub fn tracked() -> u64 {
    TRACKED.load(Ordering::Relaxed)
}

/// Clear the ledger (test hook).
pub fn reset() {
    let mut t = table().lock().unwrap_or_else(|e| e.into_inner());
    t.clear();
    TRACKED.store(0, Ordering::Relaxed);
}


/// Shared by every module whose tests touch the global ledger. One lock, not
/// one per module: separate per-module locks do not serialize against each
/// other, and the ledger is a single process-global table. That mistake made
/// the faultlog tests fail intermittently while the suite was otherwise green.
#[cfg(test)]
pub(crate) static LEDGER_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;


    /// The whole point of the ledger: a raw faulting address, the only thing the
    /// KMD gives us, must resolve to a named allocation.
    #[test]
    fn address_resolves_to_its_allocation() {
        let _g = super::LEDGER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        register(0x1000, 4096, 1, false, "layer 12 attn_q");
        match attribute(0x1000 + 17, 1) {
            Attribution::Owned { record, offset, .. } => {
                assert_eq!(record.owner, "layer 12 attn_q");
                assert_eq!(record.ordinal, 1);
                assert_eq!(offset, 17);
            }
            other => panic!("expected Owned, got {other:?}"),
        }
        assert_eq!(tracked(), 1);
    }

    /// The fault address is the start of the access, so offset 0 must resolve.
    #[test]
    fn base_address_resolves() {
        let _g = super::LEDGER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        register(0x2000, 512, 0, false, "ssm_state");
        assert!(matches!(attribute(0x2000, 0), Attribution::Owned { offset: 0, .. }));
    }

    /// Half-open range: the byte one past the end is not ours. An off-by-one
    /// here would attribute a neighbouring allocation's fault to us.
    #[test]
    fn address_past_the_end_does_not_resolve() {
        let _g = super::LEDGER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        register(0x3000, 256, 0, false, "a");
        assert!(matches!(attribute(0x3000 + 256, 0), Attribution::Unowned { .. }));
        assert!(matches!(attribute(0x2fff, 0), Attribution::Unowned { .. }));
    }

    /// The cross-device case. The allocation is on ordinal 0 and the fault came
    /// from ordinal 1 - that disagreement is the placement bug, and it must be
    /// distinguishable from a clean hit.
    #[test]
    fn fault_on_the_wrong_device_is_flagged() {
        let _g = super::LEDGER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        register(0x4000, 1024, 0, false, "layer 3 k");
        match attribute(0x4000, 1) {
            Attribution::WrongDevice { record, fault_ordinal, .. } => {
                assert_eq!(record.ordinal, 0);
                assert_eq!(fault_ordinal, 1);
            }
            other => panic!("expected WrongDevice, got {other:?}"),
        }
    }

    /// Released memory must stop resolving, or every use-after-free looks like a
    /// live allocation and the ledger cannot tell lifetime from placement.
    #[test]
    fn deregistered_memory_stops_resolving() {
        let _g = super::LEDGER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        register(0x5000, 256, 1, false, "kv_k");
        assert!(matches!(attribute(0x5000, 1), Attribution::Owned { .. }));
        deregister(0x5000);
        assert!(matches!(attribute(0x5000, 1), Attribution::Unowned { .. }));
        assert_eq!(tracked(), 0);
    }

    /// Managed allocations are host-backed and must be distinguishable from
    /// VRAM, because they fail differently.
    #[test]
    fn managed_allocations_are_distinguishable() {
        let _g = super::LEDGER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        register(0x6000, 144, 1, true, "attn_q (spilled)");
        match attribute(0x6070, 1) {
            Attribution::Owned { record, .. } => {
                assert!(record.managed, "spilled allocation must report managed");
            }
            other => panic!("expected Owned, got {other:?}"),
        }
    }

    /// Nested ranges should report the specific owner rather than picking one.
    #[test]
    fn overlapping_ranges_are_reported_not_guessed() {
        let _g = super::LEDGER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        register(0x7000, 8192, 0, false, "outer");
        register(0x7100, 512, 0, false, "inner");
        assert_eq!(candidates(0x7150).len(), 2);
        match attribute(0x7150, 0) {
            Attribution::Ambiguous { candidates, .. } => {
                // Narrowest first, so the more specific owner leads.
                assert_eq!(candidates[0].owner, "inner");
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    /// Every registration must show up in the snapshot, or a load-time report
    /// would silently under-count.
    #[test]
    fn snapshot_lists_every_live_allocation() {
        let _g = super::LEDGER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        register(0x9000, 16, 0, false, "one");
        register(0x8000, 16, 1, false, "two");
        let all = snapshot();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].ptr, 0x8000, "snapshot is pointer-ordered");
        assert_eq!(tracked(), 2);
    }
}
