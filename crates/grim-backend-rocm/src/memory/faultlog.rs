//! Parse the kernel's `amdgpu` page-fault record into something the ledger can
//! resolve.
//!
//! A GPU fault surfaces in two places: a process-visible runtime string, and
//! the kernel ring buffer. Only the ring buffer carries the faulting virtual
//! address and the device, which are the two things attribution needs - and it
//! is readable unprivileged, so no `/dev/kfd` ioctl or async handler is
//! required to get them.
//!
//! The record layout is kernel-version-specific and not a stable interface, so
//! every match here is deliberately loose and a line that does not fit is
//! skipped rather than treated as an error. A parser that panics on a kernel
//! version bump would be worse than one that reports nothing.

use super::fault::{self, HsaMemoryAccessFault, FAILURE_NOT_PRESENT};
use super::ledger::Attribution;

/// One `amdgpu` page fault, as far as the kernel record exposes it. Every field
/// is optional except `pci`: the record is a best-effort view of a
/// kernel-version-specific format, so a partially-recognised block still yields
/// a record rather than being discarded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuPageFault {
    /// PCI address of the faulting device, e.g. `0000:0a:00.0`.
    pub pci: String,
    /// The faulting virtual address, when the record carries it.
    pub address: Option<u64>,
    /// Offending process name.
    pub process: Option<String>,
    /// Offending process id.
    pub pid: Option<u32>,
    /// Offending thread name.
    pub thread: Option<String>,
    /// `PERMISSION_FAULTS` counter from the record.
    pub permission_faults: Option<u32>,
    /// `MAPPING_ERROR` counter from the record.
    pub mapping_error: Option<u32>,
    /// `RW` field from the record, kept raw - its direction encoding is not
    /// documented, so interpreting it here would be a guess.
    pub rw: Option<u32>,
}

/// A parsed fault resolved against the allocation ledger.
#[derive(Debug, Clone)]
pub struct ResolvedFault {
    pub fault: GpuPageFault,
    /// grim device ordinal, or `None` when the PCI address is not recognised.
    pub ordinal: Option<usize>,
    pub attribution: Attribution,
    /// One line suitable for a log or a panic message.
    pub summary: String,
}

/// Split `amdgpu <pci>: <message>`.
///
/// The PCI address itself contains colons but never a colon-space, so the first
/// `: ` is always the end of the address. That keeps this independent of the
/// exact field count (`0000:0a:00.0` vs `0000:03:00.0` are both 12 chars, but
/// the scan does not care).
fn split_pci(rest: &str) -> Option<(&str, &str)> {
    let (pci, msg) = rest.split_once(": ")?;
    if pci.len() < 7 || !pci.contains(':') {
        return None;
    }
    Some((pci, msg))
}

/// Read `address 0x<hex>` out of an `in page starting at address ...` line.
fn parse_address(msg: &str) -> Option<u64> {
    let after = msg.split_once("address 0x")?.1;
    let hex: String = after.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    u64::from_str_radix(&hex, 16).ok()
}

/// Read a `NAME: 0x<hex>` counter line.
fn parse_flag(msg: &str, name: &str) -> Option<u32> {
    let after = msg.split_once(name)?.1;
    let rest = after.trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix("0x").unwrap_or(rest);
    let hex: String = rest.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    u32::from_str_radix(&hex, 16).ok()
}

/// Parse a `Process <name> pid N thread <name> pid M` line.
fn parse_process(msg: &str) -> Option<(String, Option<u32>, Option<String>)> {
    let after = msg.trim_start().strip_prefix("Process ")?;
    let tok: Vec<&str> = after.split_whitespace().collect();
    let process = tok.first()?.to_string();
    let pid = tok.get(2).and_then(|v| v.parse::<u32>().ok());
    let thread = match tok.get(3) {
        Some(&"thread") => tok.get(4).map(|s| s.to_string()),
        _ => None,
    };
    Some((process, pid, thread))
}

/// Parse every `amdgpu` page-fault block in `text`.
///
/// A `[gfxhub] page fault (...)` header starts a record; the lines that follow
/// for the same device fill it in. Lines that match nothing are skipped, so a
/// kernel that words this differently degrades to silence rather than to a
/// wrong answer.
pub fn parse(text: &str) -> Vec<GpuPageFault> {
    let mut out: Vec<GpuPageFault> = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.split_once("amdgpu ").map(|(_, r)| r) else {
            continue;
        };
        let Some((pci, msg)) = split_pci(rest) else {
            continue;
        };
        if msg.contains("page fault") {
            out.push(GpuPageFault { pci: pci.to_string(), ..Default::default() });
            continue;
        }
        // Continuation lines attach to the newest open record for this device.
        let Some(cur) = out.iter_mut().rev().find(|f| f.pci == pci) else {
            continue;
        };
        if let Some(addr) = parse_address(msg) {
            cur.address = Some(addr);
        }
        if let Some(v) = parse_flag(msg, "PERMISSION_FAULTS") {
            cur.permission_faults = Some(v);
        }
        if let Some(v) = parse_flag(msg, "MAPPING_ERROR") {
            cur.mapping_error = Some(v);
        }
        if let Some(v) = parse_flag(msg, "RW") {
            cur.rw = Some(v);
        }
        if let Some((process, pid, thread)) = parse_process(msg) {
            cur.process = Some(process);
            cur.pid = pid;
            cur.thread = thread;
        }
    }
    out
}

/// Resolve parsed faults against the ledger.
///
/// `pci_to_ordinal` maps the record's PCI address onto grim's device ordinals.
/// An unmapped PCI stays unresolved on purpose: defaulting to ordinal 0 is how a
/// fault gets attributed to the wrong card.
///
/// The record's `HSA_NODEID` equivalent is not available from the kernel log,
/// so the fault is routed through [`fault::resolve`] with the mapping supplied
/// here and a placeholder node id that the closure overrides.
pub fn resolve_all(
    faults: &[GpuPageFault],
    pci_to_ordinal: &dyn Fn(&str) -> Option<usize>,
) -> Vec<ResolvedFault> {
    faults
        .iter()
        .map(|f| {
            let ordinal = pci_to_ordinal(&f.pci);
            // PERMISSION_FAULTS and MAPPING_ERROR are the two counters the
            // record exposes, and both mean the page was not present. There is
            // no separate mapping bit in the HSA failure bitfield.
            let failure = if f.permission_faults.unwrap_or(0) != 0
                || f.mapping_error.unwrap_or(0) != 0
            {
                FAILURE_NOT_PRESENT
            } else {
                0
            };
            let raw = HsaMemoryAccessFault {
                node_id: 0,
                virtual_address: f.address.unwrap_or(0),
                failure,
                flags: 0,
            };
            let report = fault::resolve(raw, &|_| ordinal);
            let summary = format!(
                "amdgpu {} pid={:?} thread={:?} -> {}",
                f.pci, f.pid, f.thread, report.summary()
            );
            ResolvedFault { fault: f.clone(), ordinal, attribution: report.attribution, summary }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ledger;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Verbatim capture from the deliberate OOB probe, `dmesg` form.
    const REAL_DMESG: &str = "\
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:14617)
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0:  Process grim_backend_ro pid 2600701 thread lib_internal_te pid 2600702
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0:   in page starting at address 0x00007fea88000000 from client 10
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0:          Faulty UTCL2 client ID: TCP (0x8)
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0:        MORE_FAULTS: 0x1
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0:        WALKER_ERROR: 0x0
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0:        PERMISSION_FAULTS: 0x3
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0:        MAPPING_ERROR: 0x0
Sep 26 12:39:54 syd-beasty kernel: amdgpu 0000:0a:00.0:        RW: 0x0
";

    /// A real dmesg line must yield the faulting address, the device, and the
    /// offending process. The address is the whole point: without it there is
    /// nothing to resolve.
    #[test]
    fn parses_a_real_amdgpu_page_fault() {
        let faults = parse(REAL_DMESG);
        assert_eq!(faults.len(), 1, "one fault block, one record");
        let f = &faults[0];
        assert_eq!(f.pci, "0000:0a:00.0");
        assert_eq!(f.address, Some(0x0000_7fea_8800_0000));
        assert_eq!(f.process.as_deref(), Some("grim_backend_ro"));
        assert_eq!(f.pid, Some(2600701));
        assert_eq!(f.permission_faults, Some(3));
    }

    /// `journalctl -k` prefixes lines differently from `dmesg`. Both are things
    /// a user will actually run, so both have to work.
    #[test]
    fn parses_journalctl_prefixed_lines() {
        let text = "\
2026-09-26T12:39:54+0000 syd-beasty kernel: amdgpu 0000:03:00.0: [gfxhub] page fault (vmid:1)
2026-09-26T12:39:54+0000 syd-beasty kernel: amdgpu 0000:03:00.0:   in page starting at address 0x00007f0000000000 from client 3
";
        let faults = parse(text);
        assert_eq!(faults.len(), 1);
        assert_eq!(faults[0].pci, "0000:03:00.0");
        assert_eq!(faults[0].address, Some(0x0000_7f00_0000_0000));
    }

    /// Unrelated kernel chatter must not become a fault. A parser that treats
    /// every amdgpu line as a fault invents incidents that never happened.
    #[test]
    fn ignores_unrelated_kernel_lines() {
        let text = "\
Sep 26 12:12:52 syd-beasty kernel: amdgpu 0000:03:00.0: VM memory stats for proc antigravity(1997958) is non-zero when fini
Sep 26 12:40:01 syd-beasty kernel: amdgpu 0000:0a:00.0: initialized amdgpu
Sep 26 12:40:02 syd-beasty kernel: usb 1-3: new high-speed USB device
";
        assert!(parse(text).is_empty(), "no page fault header, no record");
    }

    /// A header with no address line still happened, and dropping it would hide
    /// a real fault. It is reported with `address: None` so the caller can say
    /// "a fault occurred but the address was not in the record".
    #[test]
    fn a_fault_without_an_address_is_still_reported() {
        let text = "\
Sep 26 12:39:54 host kernel: amdgpu 0000:0a:00.0: [gfxhub] page fault (vmid:8)
";
        let faults = parse(text);
        assert_eq!(faults.len(), 1);
        assert_eq!(faults[0].address, None);
    }

    /// Two separate fault headers are two incidents, not one record with two
    /// addresses. The real log repeats a block when MORE_FAULTS is set.
    #[test]
    fn separate_headers_are_separate_faults() {
        let text = "\
Sep 26 12:39:54 host kernel: amdgpu 0000:0a:00.0: [gfxhub] page fault (vmid:8)
Sep 26 12:39:54 host kernel: amdgpu 0000:0a:00.0:   in page starting at address 0x00007fea88000000 from client 10
Sep 26 12:39:54 host kernel: amdgpu 0000:0a:00.0: [gfxhub] page fault (vmid:8)
Sep 26 12:39:54 host kernel: amdgpu 0000:0a:00.0:   in page starting at address 0x00007fea88001000 from client 10
";
        let faults = parse(text);
        assert_eq!(faults.len(), 2);
        assert_eq!(faults[0].address, Some(0x7fea_8800_0000));
        assert_eq!(faults[1].address, Some(0x7fea_8800_1000));
    }

    /// End to end: a parsed address, resolved against the ledger, on a device
    /// that does not own it. This is the cross-device verdict, reached from
    /// kernel text.
    #[test]
    fn parsed_fault_resolves_to_a_cross_device_allocation() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ledger::reset();
        // The buffer lives on ordinal 0; the fault came from PCI 0000:0a:00.0,
        // which this box maps to ordinal 1.
        ledger::register(0x0000_7fea_8800_0000, 4096, 0, false, "layer 12 attn_q");

        let faults = parse(REAL_DMESG);
        let resolved = resolve_all(&faults, &|pci| match pci {
            "0000:0a:00.0" => Some(1),
            _ => None,
        });
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].ordinal, Some(1));
        assert!(resolved[0].summary.contains("CROSS-DEVICE"));
    }

    /// A fault inside a buffer the faulting device does own is a lifetime
    /// problem. The two must never be reported the same way.
    #[test]
    fn parsed_fault_on_the_owning_device_is_not_cross_device() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ledger::reset();
        ledger::register(0x0000_7fea_8800_0000, 4096, 1, false, "kv_k");
        let faults = parse(REAL_DMESG);
        let resolved = resolve_all(&faults, &|_| Some(1));
        assert!(resolved[0].summary.contains("owned by"));
        assert!(!resolved[0].summary.contains("CROSS-DEVICE"));
    }

    /// Garbage in, nothing out, no panic. A kernel version that changes the
    /// wording must degrade to silence, not take the process down mid-load.
    #[test]
    fn unrecognised_text_yields_nothing_without_panicking() {
        let junk = "not a kernel line\n\x00\x01\x02 random bytes\namdgpu\n";
        assert!(parse(junk).is_empty());
    }

    /// A PCI string that maps to no known device must stay unresolved rather
    /// than defaulting to ordinal 0 and blaming the wrong card.
    #[test]
    fn unknown_pci_is_not_guessed() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ledger::reset();
        ledger::register(0x0000_7fea_8800_0000, 4096, 0, false, "x");
        let faults = parse(REAL_DMESG);
        let resolved = resolve_all(&faults, &|_| None);
        assert_eq!(resolved[0].ordinal, None);
        assert!(resolved[0].summary.contains("no live allocation"));
    }
}

    /// End-to-end against the live kernel ring buffer rather than a captured
    /// copy: the log contains thousands of unrelated lines from every device,
    /// and the parser has to find the fault inside that noise without inventing
    /// one. Diagnostic only - the log is empty on a machine that never faulted.
    #[test]
    fn live_kernel_log_parses_without_inventing_faults() {
        if std::env::var("GRIM_FAULTLOG_LIVE").as_deref() != Ok("1") {
            return;
        }
        // `dmesg` itself is gated by kernel.dmesg_restrict on this host, so it
        // returns nothing without privileges. `journalctl -k` reads the same
        // ring buffer and is world-readable, so it is the source that works.
        let mut text = String::new();
        for cmd in ["journalctl -k --no-pager", "dmesg"] {
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            if let Ok(out) = std::process::Command::new(parts[0]).args(&parts[1..]).output() {
                let t = String::from_utf8_lossy(&out.stdout).to_string();
                if t.contains("amdgpu") {
                    text = t;
                    eprintln!("[faultlog] reading kernel log via `{cmd}`");
                    break;
                }
            }
        }
        let faults = parse(&text);
        eprintln!("[faultlog] {} kernel lines -> {} page fault(s)", text.lines().count(), faults.len());
        for f in &faults {
            eprintln!(
                "[faultlog] pci={} addr={:?} proc={:?} pid={:?} perm_faults={:?}",
                f.pci, f.address, f.process, f.pid, f.permission_faults
            );
        }
        let resolved = resolve_all(&faults, &|pci| match pci {
            "0000:03:00.0" => Some(0),
            "0000:0a:00.0" => Some(1),
            "0000:78:00.0" => Some(2),
            _ => None,
        });
        for r in &resolved {
            eprintln!("[faultlog] {}", r.summary);
        }
        // The probe faults at a pointer outside every allocation, so the only
        // correct verdict is that nothing owns it.
        assert!(
            resolved.iter().any(|r| r.fault.address.is_some()),
            "the deliberate OOB fault from GRIM_FAULT_PROBE should be in the log"
        );
    }
