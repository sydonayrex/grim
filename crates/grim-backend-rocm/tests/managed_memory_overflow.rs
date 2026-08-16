//! Hardware-gated validation for the ROCm host-backed overflow tier.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm
//! --test managed_memory_overflow -- --nocapture` on a ROCm machine.

use grim_backend_rocm::{RocmDevice, RocmStorage};
use grim_tensor::{BackendDevice, DType, Shape};
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
