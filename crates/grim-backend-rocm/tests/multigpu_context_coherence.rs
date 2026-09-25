//! Multi-GPU context-coherence regression (Qwen3.8 Q4_K `hipModuleLoad` 209).
//!
//! Failure this pins down: loading the 27B Qwen checkpoint across two GPUs
//! failed with
//!
//! ```text
//! hipModuleLoad failed: 209 (entry=grim_fused_dequant_gemm_q4k, gpu_target=gfx1201)
//! ```
//!
//! HIP status 209 is `hipErrorNoBinaryForGpu`. A standalone HIP program confirms
//! the code object itself is fine: the same `gfx1201` file loads with status 0
//! on the `gfx1201` device and 209 on the `gfx1200` device. So the object was
//! being loaded while the thread sat on the *wrong* ordinal.
//!
//! Root cause: `DeviceGuard` caches the current device in a thread-local
//! (`CUR_DEV`) to skip redundant `hipSetDevice` calls, but `raw_set_device` (used
//! by the device constructor) moved the real HIP context without updating that
//! cache. The cache then claimed a device the thread was not on, `DeviceGuard`
//! skipped the switch as a no-op, and a `gfx1201` module was loaded against
//! `gfx1200`.
//!
//! Gated: `GRIM_GPU_TEST=1` and at least two visible ROCm devices.

use grim_backend_rocm::device::handles::hipGetDevice;
use grim_backend_rocm::{DeviceGuard, enumerate_devices, gpu_test_enabled, raw_set_device};

/// Read the real HIP current device for this thread, bypassing any cache.
fn real_current_device() -> i32 {
    let mut d: i32 = -1;
    unsafe {
        let _ = hipGetDevice(&mut d);
    }
    d
}

/// Skip unless GPU tests are enabled and two devices are actually present.
fn two_gpu_gate() -> bool {
    if !gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return false;
    }
    match enumerate_devices() {
        Ok(n) if n >= 2 => true,
        Ok(n) => {
            eprintln!("[SKIP] needs 2 ROCm devices, found {n}");
            false
        }
        Err(e) => {
            eprintln!("[SKIP] device enumeration failed: {e}");
            false
        }
    }
}

/// The stale-cache scenario, in the exact order that triggers it:
///
/// 1. a guard for ordinal 0 populates `CUR_DEV` with 0;
/// 2. `raw_set_device(1)` moves the real HIP context to ordinal 1 (this is what
///    the device constructor and the P2P probe do) but, before the fix, left
///    `CUR_DEV` still reading 0;
/// 3. a guard for ordinal 0 sees `CUR_DEV == 0`, takes the "already pinned"
///    fast path, issues no `hipSetDevice`, and the thread stays on ordinal 1.
///
/// Loading a `gfx1201` module at that point fails with HIP error 209
/// (`hipErrorNoBinaryForGpu`) because the object does not match the device.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1 on a 2-GPU box"]
fn device_guard_pins_thread_after_a_raw_switch_moved_it() {
    if !two_gpu_gate() {
        return;
    }

    let original = real_current_device();

    // Step 1: cache now records whatever device this guard settled on.
    {
        let _g0 = DeviceGuard::set(0);
        assert_eq!(real_current_device(), 0);
    }

    // Step 2: an unguarded switch to the other GPU (constructor / P2P probe).
    let status = raw_set_device(1);
    assert_eq!(status, grim_backend_rocm::hipSuccess, "raw_set_device(1) failed");
    assert_eq!(real_current_device(), 1);

    // Step 3: ask for ordinal 0 again. This must actually move the thread back.
    let guard = DeviceGuard::set(0);
    assert_eq!(
        real_current_device(),
        0,
        "DeviceGuard::set(0) must re-pin the thread after an unguarded switch to \
         ordinal 1; a stale CUR_DEV skipped the hipSetDevice and made gfx1201 \
         module loads fail with HIP error 209"
    );
    drop(guard);

    let _ = raw_set_device(original);
}

/// `raw_set_device` must leave the cache agreeing with the real HIP context, so
/// that a following guard for the *same* ordinal is a genuine no-op rather than
/// a skipped required switch.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1 on a 2-GPU box"]
fn raw_set_device_settles_the_thread_on_the_requested_ordinal() {
    if !two_gpu_gate() {
        return;
    }

    let original = real_current_device();

    let status = raw_set_device(1);
    assert!(
        status == grim_backend_rocm::hipSuccess,
        "raw_set_device(1) failed with {status}"
    );

    assert_eq!(
        real_current_device(),
        1,
        "raw_set_device(1) must leave the thread on ordinal 1"
    );

    // The guard for the ordinal we are already on should be a cheap no-op and
    // must not move us anywhere.
    {
        let _g = DeviceGuard::set(1);
        assert_eq!(real_current_device(), 1);
    }

    // Restore the caller's original context.
    let _ = raw_set_device(original);
}
