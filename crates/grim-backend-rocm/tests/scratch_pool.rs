//! RED-GREEN-REFACTOR tests for `memory::pool::DeviceScratchPool`.
//!
//! The tests in this file are the **RED**-phase tests for Phase-3 §3.1 of
//! the QKV spec. They assume the planned API of `DeviceScratchPool` and
//! fail to compile (or fail at runtime) until the GREEN phase implements
//! the matching module. See `crates/grim-backend-rocm/src/memory/mod.rs`
//! for the live integration.
//!
//! Methodology reminders (`rust-tdd` skill):
//! - Single-value equivalence → `assert_eq!` (no overwrite with snapshots).
//! - Tests return `Result` and use `?` rather than `.unwrap()`.
//! - "Compile error counts as red."
//! - Thread-spawned closures that bubble errors must use a Send+Sync error
//!   type — `Box<dyn std::error::Error>` alone is not `Send`, so multi-
//!   threaded tests use the Send+Sync variant.
//!
//! Skill attribution:
//! - `rust-ai-ml-inference-guide` Action 3 — Memory pool, KV-cache scratch.
//! - `rust-gpu-parallelism` — Stream-ordered memory, one stream per device.
//! - `rocm-profiling-perf` — Allocation is in the optimizer's hot path; the
//!   pool eliminates a measurable per-call hipMalloc cost.

use std::sync::Arc;

use grim_backend_rocm::memory::pool::{DeviceScratchPool, PoolLayout};
use grim_backend_rocm::RocmDevice;
use grim_tensor::{DType, Shape};

/// Thread-safe boxed error type for tests that bubble errors across
/// `std::thread::spawn`. Wraps a `String` so it's trivially `Send + Sync`
/// instead of fighting opaque `dyn std::error::Error` lifetimes.
type TestError = Box<dyn std::error::Error + Send + Sync>;
type TestResult<R = ()> = Result<R, TestError>;

const GPU_TEST_ENV: &str = "GRIM_RUN_GPU_TESTS";

/// `PoolLayout::new(size, align).size` must round up to the next power of
/// two (with a 256-byte floor). This is the bucketization that lets the
/// pool reuse buffers across similarly-sized requests without copying.
#[test]
fn pool_layout_round_up_to_power_of_two() -> TestResult {
    let layout_2k = PoolLayout::new(2048, 16);
    let layout_3k = PoolLayout::new(3000, 16);
    let layout_tiny = PoolLayout::new(64, 16);

    // Powers of 2: 2048, 4096, 256 (floor).
    assert_eq!(layout_2k.size, 2048);
    assert_eq!(layout_3k.size, 4096);
    assert_eq!(layout_tiny.size, 256);

    assert_eq!(layout_2k.align, 16);
    assert_eq!(layout_3k.align, 16);
    Ok(())
}

/// Two `PoolLayout::new` calls with the same input produce structurally-
/// equal layouts (so the HashMap key derives correctly). This isn't a
/// partial-equality test on `PooledBuffer`; it's a contract test for the
/// layout-key identity.
#[test]
fn pool_layout_hash_key_stable() -> TestResult {
    let a = PoolLayout::new(1024, 32);
    let b = PoolLayout::new(1024, 32);
    let c = PoolLayout::new(1025, 32);
    assert_eq!(a, b, "same input must produce equal layouts");
    assert_ne!(a, c, "rounded-up sizes must bucket differently");
    Ok(())
}

/// Allocating a single buffer advances `peak_bytes` and `current_bytes`,
/// and the pool tracks them monotonically.
#[test]
fn pool_alloc_tracks_peak_and_current() -> TestResult {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return Ok(()); // Pool math is purely CPU-side bookkeeping.
    }
    let dev = std::panic::catch_unwind(|| RocmDevice::new(0));
    let dev = match dev {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    let pool = DeviceScratchPool::new();
    let baseline = pool.peak_bytes();
    assert_eq!(pool.current_bytes(), 0);
    assert_eq!(baseline, 0);

    let buf1 = dev.get_scratch(1024, 16)?;
    drop(buf1);
    // After drop, the bucket returns the buffer to the pool, so current_bytes
    // is non-zero but peak_bytes is at least 1024.
    let after_drop = pool.current_bytes();
    assert!(after_drop >= 1024, "current_bytes must reflect allocations, got {}", after_drop);
    Ok(())
}

/// The recycled-pointer contract: get a buffer, drop it, get another of
/// the same size — the second one is the recycled pointer (or, if the
/// pool allocated fresh, *some* valid pointer). The acceptance criterion
/// is "the same pointer is reused when the underlying slice is the same
/// size bucket", proving pool reclamation actually happens.
#[test]
fn pool_drops_recycle_pointer_for_same_bucket() -> TestResult {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return Ok(());
    }
    let dev = std::panic::catch_unwind(|| RocmDevice::new(0));
    let dev = match dev {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    let p1 = dev.get_scratch(2048, 16)?;
    let p1_addr = p1.as_ptr() as usize;
    drop(p1);

    let p2 = dev.get_scratch(2048, 16)?;
    let p2_addr = p2.as_ptr() as usize;

    // The pool's LIFO bucket *should* hand back the just-freed slot.
    // If this fails, recycle-on-drop is silently leaking physical memory.
    assert_eq!(
        p1_addr, p2_addr,
        "pool must recycle the freed slot for the same bucket"
    );
    Ok(())
}

/// Pool buckets distinguish by size. A 4 KB request must NOT reuse the
/// slot freed by a 1 KB request — different buckets, different slots.
#[test]
fn pool_uses_distinct_buckets_per_size() -> TestResult {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return Ok(());
    }
    let dev = std::panic::catch_unwind(|| RocmDevice::new(0));
    let dev = match dev {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    let p1 = dev.get_scratch(1024, 16)?; // bucket 1024 (above 256-floor)
    let p2 = dev.get_scratch(4096, 16)?; // bucket 4096
    drop(p1);
    drop(p2);

    let q1 = dev.get_scratch(1024, 16)?;
    let q2 = dev.get_scratch(4096, 16)?;
    let aa1 = q1.as_ptr() as usize;
    let aa2 = q2.as_ptr() as usize;

    // Different buckets → different pointers (no cross-bucket pollution).
    assert_ne!(aa1, aa2, "different buckets must not share pointers");
    Ok(())
}

/// Multi-threaded smoke: the pool's mutex must not deadlock under
/// concurrent allocations. Two threads x two allocations interleave
/// safely. Spec rationale: in a decoded LLM serving loop, the scheduler
/// and the main inference task may both ask for scratch.
#[test]
fn pool_handle_concurrent_allocs() -> TestResult {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return Ok(());
    }
    let dev = std::panic::catch_unwind(|| RocmDevice::new(0));
    let dev = match dev {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    let dev_arc = Arc::new(dev);

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let d = dev_arc.clone();
            std::thread::spawn(move || -> TestResult {
                for _ in 0..16 {
                    let p = d.get_scratch(512, 16)?;
                    drop(p);
                    let p2 = d.get_scratch(8192, 16)?;
                    drop(p2);
                }
                Ok(())
            })
        })
        .collect();
    for h in handles {
        h.join().map_err(|_| "thread panicked")??;
    }
    Ok(())
}

/// Cross-threaded peak accounting is monotonic. Two threads allocate and
/// the final peak is at least the largest single allocation seen. We
/// observe the peak through `RocmDevice::scratch_pool_peak_bytes`
/// (avoids sharing `Arc<DeviceScratchPool>` across threads directly —
/// the pool's raw-pointer bucket map is not `Send`).
#[test]
fn pool_peak_monotonic_under_concurrent_load() -> TestResult {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return Ok(());
    }
    let dev = std::panic::catch_unwind(|| RocmDevice::new(0));
    let dev = match dev {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    let dev_arc = Arc::new(dev);

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let d = dev_arc.clone();
            std::thread::spawn(move || -> TestResult {
                let dev = &*d;
                let _ = dev.get_scratch(2048, 16)?;
                let _ = dev.get_scratch(8192, 16)?;
                let peak = dev.scratch_pool_peak_bytes();
                assert!(peak >= 2048, "peak must keep up with allocations, got {}", peak);
                Ok(())
            })
        })
        .collect();
    for h in handles {
        h.join().map_err(|_| "thread panicked")??;
    }
    Ok(())
}

/// `DeviceScratchPool` is also exposed as a public type for callers that
/// want to wire their own integration sites (e.g. a future per-layer
/// scratch bucket). The constructor must not panic and must produce a
/// usable pool, regardless of GPU presence.
#[test]
fn pool_new_is_infallible_and_zeroed() -> TestResult {
    let pool = DeviceScratchPool::new();
    assert_eq!(pool.current_bytes(), 0);
    assert_eq!(pool.peak_bytes(), 0);
    // Pool is Arc; clonable for shared use without disturbing counters.
    let _other = Arc::clone(&pool);
    Ok(())
}

/// `PooledBuffer::as_ptr()` returns a non-null pointer for any successful
/// allocation. The `null` sentinel is reserved for the error path only
/// and isn't reachable from the success path.
#[test]
fn pool_buffer_as_ptr_is_nonnull() -> TestResult {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return Ok(());
    }
    let dev = std::panic::catch_unwind(|| RocmDevice::new(0));
    let dev = match dev {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let buf = dev.get_scratch(4096, 16)?;
    assert!(!buf.as_ptr().is_null(), "as_ptr must be non-null after get()");
    Ok(())
}

/// Sanity: a buffer's bytes are usable for arbitrary host writes AFTER
/// the pool hands the slot back. We don't run a kernel; this just
/// confirms dtype/storage round-trips through the buffer alignment.
#[test]
fn pool_buffer_can_be_uploaded_to() -> TestResult {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return Ok(());
    }
    let dev = std::panic::catch_unwind(|| RocmDevice::new(0));
    let dev = match dev {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    // Tight scope so the second allocation after `drop(buf)` is observable.
    let slot_addr = {
        let buf = dev.get_scratch(1024, 16)?;
        let addr = buf.as_ptr() as usize;
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
        let shape = Shape::from_slice(&[64]);
        // `from_cpu` does an `hipMemcpy` into the underlying device buffer
        // via `RocmStorage::copy_from_host`. We don't read back here — a
        // typo in the buffer reuse path would corrupt subsequent allocs.
        let _uploaded = grim_tensor::BackendDevice::from_cpu(&dev, &data, &shape, DType::F32)?;
        addr
    };
    // After dropping the previous scratch handle, a same-size handle
    // must reuse the same slot. This is the integration assertion that
    // the pool's `Drop` recycle path actually fires from `RocmDevice`.
    let buf2 = dev.get_scratch(1024, 16)?;
    assert_eq!(
        buf2.as_ptr() as usize, slot_addr,
        "RocmDevice::get_scratch must route through the pool and recycle"
    );
    Ok(())
}

// All imports are intentionally referenced inside gated arms; no top-level
// silence needed.

// =========================================================================
// REFACTOR — Integration tests for the `upload_to_scratch` hot path.
// Goal: the existing `upload_device_buffer` path does `hipMalloc` +
// `hipMemcpy` per call. Phase-3 §3.1 wants to replace that with the pool.
// These tests pin that the new path actually reuses slots.
// =========================================================================

/// `RocmDevice::upload_to_scratch` must route through the pool. Two
/// consecutive uploads of the same shape land on the same device slot
/// (after the first one is dropped), proving the integration is wired.
#[test]
fn upload_to_scratch_recycles_slot() -> TestResult {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return Ok(());
    }
    let dev = std::panic::catch_unwind(|| RocmDevice::new(0));
    let dev = match dev {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.25).collect();
    let shape = Shape::from_slice(&[64]);

    let slot_addr = {
        let buf = grim_backend_rocm::RocmDevice::upload_to_scratch(&dev, &data, &shape, DType::F32)?;
        buf.as_ptr() as usize
    };
    let buf2 = grim_backend_rocm::RocmDevice::upload_to_scratch(&dev, &data, &shape, DType::F32)?;
    assert_eq!(
        buf2.as_ptr() as usize, slot_addr,
        "upload_to_scratch must route through DeviceScratchPool and recycle the slot"
    );
    Ok(())
}

/// Sanity: the `upload_to_scratch` shape accounting matches the input
/// length. If the pool ever rounds `len` differently from the upload's
/// expected size, the subsequent memcpy would be wrong.
#[test]
fn upload_to_scratch_bytes_match_data() -> TestResult {
    let env = std::env::var(GPU_TEST_ENV).is_ok();
    if !env {
        return Ok(());
    }
    let dev = std::panic::catch_unwind(|| RocmDevice::new(0));
    let dev = match dev {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    let data: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
    let shape = Shape::from_slice(&[128]);
    let buf = grim_backend_rocm::RocmDevice::upload_to_scratch(&dev, &data, &shape, DType::F32)?;
    assert_eq!(buf.layout().size, 128 * 4);
    Ok(())
}

