//! Deliberate out-of-bounds probe: what does a real memory fault actually
//! leave behind, and can the ledger attribute it?
//!
//! The decoder in `memory::fault` is unit-tested against synthetic records, but
//! that only proves it handles the shapes we hand it. The question that
//! matters is whether a *real* fault produces an address we can resolve. This
//! launches a kernel at a pointer far outside any live allocation and reports
//! what the driver surfaced.
//!
//! Deliberately destructive: a device page fault can leave the context unusable
//! and can trip a GPU reset, so this runs on its own, gated behind
//! `GRIM_FAULT_PROBE=1`, and asserts nothing about the surrounding state. It is
//! a diagnostic, not a regression gate - the KFD message goes to the kernel
//! ring buffer, which is read out-of-band.
//!
//! Needs `GRIM_GPU_TEST=1` and `GRIM_FAULT_PROBE=1`.

use crate::device::util::DeviceGuard;
use grim_tensor::CoreTensorOps;
use crate::memory::ledger;

/// Far enough past the end of any real allocation to be unmapped, but still a
/// plausible device virtual address.
const OOB_OFFSET: u64 = 1 << 32;

#[test]
fn deliberate_oob_launch_surfaces_attributable_evidence() {
    if std::env::var("GRIM_GPU_TEST").as_deref() != Ok("1") {
        return;
    }
    if std::env::var("GRIM_FAULT_PROBE").as_deref() != Ok("1") {
        eprintln!("[fault-probe] skipped (set GRIM_FAULT_PROBE=1 to run)");
        return;
    }
    let dev = match crate::device::roc_device::RocmDevice::try_new(1) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[fault-probe] no usable device on ordinal 1: {e}");
            return;
        }
    };
    let _guard = DeviceGuard::set(dev.ordinal() as i32);

    ledger::reset();

    // A real, small, ledger-registered allocation - the innocent bystander that
    // the faulting address must NOT be attributed to.
    let rows = 4usize;
    let cols = 64usize;
    let shape = crate::Shape::from_slice(&[rows, cols]);
    let real = crate::memory::storage::RocmStorage::copy_from_host(
        &vec![0.5f32; rows * cols],
        &shape,
        crate::DType::F32,
        &dev.allocator,
        dev.ordinal(),
    )
    .expect("real allocation");
    let real_ptr = real.device_ptr_u64().expect("real ptr");
    eprintln!("[fault-probe] real allocation at 0x{real_ptr:x} ({} bytes)", real.bytes);

    // Fabricate a storage whose pointer is outside every live allocation.
    let mut bad = unsafe { std::ptr::read(&real) };
    let bad_ptr = real_ptr + OOB_OFFSET;
    bad.device_ptr = Some(bad_ptr);
    std::mem::forget(real); // the clone owns the same allocation; do not free it
    eprintln!("[fault-probe] launching kernel at 0x{bad_ptr:x} (outside all allocations)");

    let w = crate::memory::storage::RocmStorage::copy_from_host(
        &vec![1.0f32; cols],
        &crate::Shape::from_slice(&[cols]),
        crate::DType::F32,
        &dev.allocator,
        dev.ordinal(),
    )
    .expect("gamma upload");

    let launch = dev.rms_norm(&bad, &w, 1e-5, &shape);
    match launch {
        Ok((_out, handle)) => match handle.synchronize() {
            Ok(()) => eprintln!(
                "[fault-probe] UNEXPECTED: out-of-bounds kernel completed without error"
            ),
            Err(e) => eprintln!("[fault-probe] surfaced at sync: {e}"),
        },
        Err(e) => eprintln!("[fault-probe] surfaced at launch: {e}"),
    }

    // Whatever the driver said, report the live allocations so an out-of-band
    // address from dmesg can be checked against them.
    let all = ledger::snapshot();
    eprintln!(
        "[fault-probe] ledger holds {} allocation(s):",
        all.len()
    );
    for r in all {
        eprintln!(
            "[fault-probe]   0x{:x}..0x{:x} device {} managed={} {}",
            r.ptr,
            r.ptr + r.bytes,
            r.ordinal,
            r.managed,
            r.owner
        );
    }
    eprintln!(
        "[fault-probe] the faulting VA should be ~0x{:x} and must match none of the above",
        bad_ptr
    );
}
