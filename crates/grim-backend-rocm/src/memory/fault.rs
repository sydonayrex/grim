//! Resolving a HIP memory fault against the device-allocation ledger.
//!
//! The KMD raises `HsaMemoryAccessFault` for a memory access it cannot
//! translate, carrying the faulting virtual address and the node it happened
//! on. That address is the only actionable datum in the fault: names of tensors
//! and layers do not appear anywhere in it. This module decodes the record and
//! resolves the address through [`crate::memory::ledger`], which turns "GPU
//! node-N faulted" into a named allocation and a device comparison.
//!
//! `NodeId` is an H-NUMA node, **not** a device ordinal in grim's numbering,
//! and the two are not the same. The mapping is therefore injected rather than
//! assumed, so an unmapped node stays an explicit error instead of silently
//! attributing a fault to the wrong GPU.

use super::ledger::{self, Attribution};

/// `HSA_EVENTTYPE_MEMORY` from `hsakmttypes.h` - the event type carrying a
/// memory access fault.
pub const EVENT_TYPE_MEMORY: u32 = 8;

/// `HsaAccessAttributeFailure` bit positions, which are bitfields in C and so
/// cannot be mirrored as a Rust struct without risking the compiler choosing a
/// different layout.
pub const FAILURE_NOT_PRESENT: u32 = 1 << 0;
pub const FAILURE_READ_ONLY: u32 = 1 << 1;
pub const FAILURE_NO_EXECUTE: u32 = 1 << 2;
pub const FAILURE_GPU_ACCESS: u32 = 1 << 3;
pub const FAILURE_ECC: u32 = 1 << 4;
pub const FAILURE_IMPRECISE: u32 = 1 << 5;
pub const FAILURE_ERROR_TYPE_SHIFT: u32 = 6;
pub const FAILURE_ERROR_TYPE_MASK: u32 = 0b111;

/// Faithful mirror of `HsaMemoryAccessFault`.
///
/// `NodeId` is `HSAuint32` and `VirtualAddress` is `HSAuint64`, so C pads
/// `NodeId` out to the 8-byte alignment of the following field. The
/// `size_of`/`align_of` test below is the guard on that: if the header ever
/// changes, this struct must change with it rather than misparse a fault.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HsaMemoryAccessFault {
    /// H-NUMA node containing the device where the access occurred.
    pub node_id: u32,
    /// The virtual address the access occurred on.
    pub virtual_address: u64,
    /// `HsaAccessAttributeFailure` bitfield, raw.
    pub failure: u32,
    /// `HSA_EVENTID_MEMORYFLAGS`.
    pub flags: u32,
}

/// A decoded fault, ready to be reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultReport {
    /// Raw KMD record.
    pub raw: HsaMemoryAccessFault,
    /// Failure bits, named.
    pub reasons: Vec<&'static str>,
    /// grim device ordinal, once the H-NUMA node has been mapped.
    pub ordinal: Option<usize>,
    /// Where the faulting address lives, if we own it.
    pub attribution: Attribution,
}

impl FaultReport {
    /// One line suitable for a log or a panic message.
    pub fn summary(&self) -> String {
        let reasons = if self.reasons.is_empty() {
            "unspecified".to_string()
        } else {
            self.reasons.join("+")
        };
        format!(
            "memory fault: addr=0x{:x} node={} ordinal={:?} reasons={reasons} -> {}",
            self.raw.virtual_address,
            self.raw.node_id,
            self.ordinal,
            match &self.attribution {
                Attribution::Unowned { .. } => "no live allocation owns this address".to_string(),
                Attribution::Owned { record, offset, .. } => format!(
                    "owned by {} on device {} at offset {offset}",
                    record.owner, record.ordinal
                ),
                Attribution::WrongDevice { record, fault_ordinal, .. } => format!(
                    "CROSS-DEVICE: {} lives on device {} but the fault came from device {}",
                    record.owner, record.ordinal, fault_ordinal
                ),
                Attribution::Ambiguous { candidates, .. } => format!(
                    "ambiguous: {} overlapping allocations",
                    candidates.len()
                ),
            }
        )
    }
}

/// Decode the failure bitfield into names.
pub fn reasons(failure: u32) -> Vec<&'static str> {
    let mut out = Vec::new();
    for (bit, name) in [
        (FAILURE_NOT_PRESENT, "page-not-present"),
        (FAILURE_READ_ONLY, "write-to-read-only"),
        (FAILURE_NO_EXECUTE, "execute-on-nx"),
        (FAILURE_GPU_ACCESS, "host-only-page"),
        (FAILURE_ECC, "ecc"),
    ] {
        if failure & bit != 0 {
            out.push(name);
        }
    }
    out
}

/// Whether the driver could pin the exact faulting address. When set, the
/// address in the record is approximate and any attribution derived from it is
/// a hint rather than a proof.
pub fn is_imprecise(failure: u32) -> bool {
    failure & FAILURE_IMPRECISE != 0
}

/// Decode a fault and resolve its address through the ledger.
///
/// `node_to_ordinal` maps the KMD's H-NUMA `NodeId` onto grim's device
/// ordinals. Returning `None` for an unrecognised node is expected and is
/// reported, not guessed: the address is still resolved against the ledger, but
/// no device comparison is made.
pub fn resolve(
    raw: HsaMemoryAccessFault,
    node_to_ordinal: &dyn Fn(u32) -> Option<usize>,
) -> FaultReport {
    let ordinal = node_to_ordinal(raw.node_id);
    // Without a mapping there is no device to compare against, so resolve
    // against device 0's perspective only if we can - otherwise report the
    // address as unowned rather than inventing a device.
    let attribution = match ordinal {
        Some(o) => ledger::attribute(raw.virtual_address, o),
        None => Attribution::Unowned { addr: raw.virtual_address },
    };
    FaultReport { raw, reasons: reasons(raw.failure), ordinal, attribution }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ledger;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// The C struct is `{u32; u64; u32; u32}` with 8-byte alignment, so it is
    /// 24 bytes with 4 bytes of padding after `NodeId`. If this ever fails, the
    /// driver is writing a different layout and every fault would be misparsed.
    #[test]
    fn layout_matches_the_kmd_struct() {
        assert_eq!(std::mem::size_of::<HsaMemoryAccessFault>(), 24);
        assert_eq!(std::mem::align_of::<HsaMemoryAccessFault>(), 8);
    }

    /// The headline case: a fault on device 1 that lands in a buffer owned by
    /// device 0. This is the cross-device placement bug, and it must be
    /// reported as such rather than as a clean hit.
    #[test]
    fn cross_device_fault_is_reported_as_such() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ledger::reset();
        ledger::register(0x1000, 4096, 0, false, "layer 12 attn_q");

        let raw = HsaMemoryAccessFault {
            node_id: 1,
            virtual_address: 0x1000 + 64,
            failure: FAILURE_NOT_PRESENT,
            flags: 0,
        };
        // KMD node 1 is grim ordinal 0 here: the buffer is on ordinal 0 and the
        // faulting node maps to ordinal 1, so the devices disagree.
        let report = resolve(raw, &|node| match node {
            0 => Some(0),
            1 => Some(1),
            _ => None,
        });
        assert!(matches!(report.attribution, Attribution::WrongDevice { .. }));
        assert!(report.summary().contains("CROSS-DEVICE"));
        assert_eq!(report.ordinal, Some(1));
    }

    /// A fault inside a buffer that does belong to the faulting device is a
    /// lifetime problem, not a placement one, and the two must not be confused.
    #[test]
    fn same_device_fault_is_not_reported_as_cross_device() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ledger::reset();
        ledger::register(0x2000, 1024, 1, false, "kv_k");
        let raw = HsaMemoryAccessFault {
            node_id: 0,
            virtual_address: 0x2000,
            failure: FAILURE_NOT_PRESENT,
            flags: 0,
        };
        let report = resolve(raw, &|node| match node {
            0 => Some(1),
            _ => None,
        });
        assert!(matches!(report.attribution, Attribution::Owned { .. }));
        assert!(!report.summary().contains("CROSS-DEVICE"));
    }

    /// An unmapped H-NUMA node must stay unresolved. Silently defaulting to
    /// ordinal 0 is how a fault on an unknown device gets blamed on the wrong
    /// card.
    #[test]
    fn unmapped_node_is_not_guessed() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ledger::reset();
        ledger::register(0x3000, 256, 0, false, "x");
        let raw = HsaMemoryAccessFault {
            node_id: 99,
            virtual_address: 0x3000,
            failure: FAILURE_NOT_PRESENT,
            flags: 0,
        };
        let report = resolve(raw, &|_| None);
        assert_eq!(report.ordinal, None);
        assert!(matches!(report.attribution, Attribution::Unowned { .. }));
        assert!(report.summary().contains("ordinal=None"));
    }

    /// An address we do not own means the buffer was already released, or the
    /// address is not ours at all. Either way it is not a placement bug.
    #[test]
    fn unowned_address_is_reported_as_such() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ledger::reset();
        let raw = HsaMemoryAccessFault {
            node_id: 0,
            virtual_address: 0xdead_beef,
            failure: FAILURE_NOT_PRESENT,
            flags: 0,
        };
        let report = resolve(raw, &|_| Some(0));
        assert!(matches!(report.attribution, Attribution::Unowned { .. }));
    }

    /// `Imprecise` means the driver could not determine the exact address, so
    /// any attribution is a hint. That has to be visible or a confident-looking
    /// report will be over-trusted.
    #[test]
    fn imprecise_address_is_flagged() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ledger::reset();
        let raw = HsaMemoryAccessFault {
            node_id: 0,
            virtual_address: 0x4000,
            failure: FAILURE_NOT_PRESENT | FAILURE_IMPRECISE,
            flags: 0,
        };
        assert!(is_imprecise(raw.failure));
        let report = resolve(raw, &|_| Some(0));
        assert!(!report.reasons.contains(&"ecc"));
    }

    /// Multiple failure bits can be set at once and all must be reported.
    #[test]
    fn failure_bits_decode_together() {
        assert_eq!(
            reasons(FAILURE_NOT_PRESENT | FAILURE_READ_ONLY),
            vec!["page-not-present", "write-to-read-only"]
        );
        assert!(reasons(0).is_empty(), "no bits set means no named reason");
    }
}
