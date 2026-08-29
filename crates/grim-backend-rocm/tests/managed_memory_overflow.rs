//! Hardware-gated validation for the ROCm host-backed overflow tier.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm
//! --test managed_memory_overflow -- --nocapture` on a ROCm machine.

use grim_backend_rocm::{RocmDevice, RocmStorage};
use grim_tensor::{CoreTensorOps, DType, Shape};
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn managed_storage_accepts_prefetch_request() {
    let _guard = TEST_LOCK.lock().expect("managed-memory test lock poisoned");
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    let device = RocmDevice::new(0);
    let host = vec![1.0f32; 128];
    let storage = match device.from_cpu_managed(&host, &Shape::from_slice(&[128]), DType::F32) {
        Ok(storage) => storage,
        Err(error) => {
            eprintln!("[skipped: ROCm device unavailable: {error}]");
            return;
        }
    };

    let rocm = storage
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("managed upload must return RocmStorage");
    assert!(rocm.is_managed());
    storage
        .prefetch_to_device()
        .expect("managed storage prefetch should succeed");
}

#[test]
fn global_policy_routes_ordinary_allocations_to_managed_memory() {
    let _guard = TEST_LOCK.lock().expect("managed-memory test lock poisoned");
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    unsafe { std::env::set_var("GRIM_ROCM_MANAGED_ALLOCATIONS", "always") };
    let device = RocmDevice::new(0);
    let result = device.from_cpu(&[2.0f32; 8], &Shape::from_slice(&[8]), DType::F32);
    unsafe { std::env::remove_var("GRIM_ROCM_MANAGED_ALLOCATIONS") };

    let storage = match result {
        Ok(storage) => storage,
        Err(error) => {
            eprintln!("[skipped: ROCm device unavailable: {error}]");
            return;
        }
    };
    let rocm = storage
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("ordinary ROCm allocation must return RocmStorage");
    assert!(rocm.is_managed());
}

/// WI-P3: routing an allocation through the managed fallback must (a) mark the
/// storage managed and (b) record the fallback in the process-wide
/// instrumentation + one-time warning. The negative case (ordinary allocation)
/// must leave the instrumentation untouched.
#[test]
fn managed_fallback_warning_and_instrumentation_fire() {
    let _guard = TEST_LOCK.lock().expect("managed-memory test lock poisoned");
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }
    grim_backend_rocm::memory::budget::reset_managed_fallback_instrumentation();

    let baseline = grim_backend_rocm::memory::budget::managed_fallback_count();

    // Forced managed mode: even a tiny allocation goes through hipMallocManaged.
    // The env scope must cover BOTH device construction (env read once at
    // `RocmDevice::new`) and the allocation itself.
    let storage = temp_env::with_var("GRIM_ROCM_MANAGED_ALLOCATIONS", Some("always"), || {
        let device = RocmDevice::new(0);
        device
            .from_cpu(&vec![1.0f32; 64], &Shape::from_slice(&[64]), DType::F32)
            .expect("forced-managed alloc must succeed")
    });
    let rocm = storage
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("must be RocmStorage");
    assert!(
        rocm.is_managed(),
        "forced mode must produce managed storage"
    );
    assert!(
        grim_backend_rocm::memory::budget::managed_fallback_count() > baseline,
        "managed fallback must be recorded in the WI-P3 counter"
    );
    assert!(
        grim_backend_rocm::memory::budget::managed_fallback_warned(),
        "managed fallback must have surfaced the one-time user warning"
    );
}

/// WI-P3 negative case on real hardware: with the default (auto / budget-fit)
/// policy a small allocation must NOT route through managed memory and must
/// not touch the instrumentation.
#[test]
fn small_auto_allocation_stays_unmanaged_and_uninstrumented() {
    let _guard = TEST_LOCK.lock().expect("managed-memory test lock poisoned");
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }
    grim_backend_rocm::memory::budget::reset_managed_fallback_instrumentation();

    let storage = temp_env::with_var("GRIM_ROCM_MANAGED_ALLOCATIONS", Some("auto"), || {
        let device = RocmDevice::new(0);
        device
            .from_cpu(&vec![1.0f32; 64], &Shape::from_slice(&[64]), DType::F32)
            .expect("small auto alloc must succeed")
    });
    let rocm = storage
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("must be RocmStorage");
    assert!(
        !rocm.is_managed(),
        "a 64-element f32 alloc must fit VRAM under the default budget"
    );
    assert_eq!(
        grim_backend_rocm::memory::budget::managed_fallback_count(),
        0,
        "no managed fallback recorded for a normal fit"
    );
    assert!(
        !grim_backend_rocm::memory::budget::managed_fallback_warned(),
        "no fallback warning for a normal fit"
    );
}
