//! Audit §2.5/§4 — append-growth guard.
//!
//! `launch_kv_append` / `launch_kv_append_batch` are no-capacity launches:
//! the HIP kernel `grim_kv_append` computes `off = slot*slot_stride +
//! dst_tok*kv_stride + col` with no bounds check (audit §: kernels cannot
//! bounds-check). Two honest guards exist instead:
//!
//! * the PHASE-3 RING: `KvRingPolicy::new(cap_rows, sink_rows)` maps an
//!   absolute row through `grim_kv_ring_row` (host twin `KvRingPolicy::row`)
//!   so a full arena recycles rows modulo rather than wrapping (never a raw
//!   overwrite of the wrong row, never a silent truncate) — this is the
//!   kernel-level "full arena" behavior, and it is documented on the
//!   kernel itself.
//! * the HOST GUARD `decode_graph_buffers::launch_qkv_gemv_ring(k_arena,
//!   pos, max_ctx, ring)`: bounds-checks the row that *will* be written
//!   (`ring.row(pos) >= max_ctx` → Err). With `UNLIMITED` this is the
//!   `pos < max_ctx` check; under a ring it validates the mapped row, so a
//!   caller cannot walk off the arena even though the kernel cannot
//!   bounds-check.
//!
//! Tests (a)-(c) below pin the *documented* behavior only; nothing is
//! silently wrapped or silently truncated — overflow with UNLIMITED fails
//! loud at the host guard, overflow under a capped ring evicts by policy.
//!
//! Gated: `GRIM_GPU_TEST=1` + ROCm device; the host-guard twins compile and
//! run on CPU wire-only as well (GUARD D1 is host-reachable).

use grim_backend_rocm::{KvRingPolicy, RocmDevice, as_rocm, gpu_test_enabled, launch_kv_append};
use grim_backend_rocm::decode_graph_buffers::launch_qkv_gemv_ring;
use grim_backend_rocm::RocmStorage;
use grim_tensor::{CoreTensorOps, DType, MemoryOps, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

/// Upload a u32 device counter (`past`) the way `qkv_attention.rs` does.
fn u32_past(dev: &RocmDevice, v: u32) -> Box<dyn grim_tensor::BackendStorage> {
    MemoryOps::from_cpu_bytes(
        dev,
        {
            let bytes = v.to_le_bytes();
            unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) }
        },
        &Shape::from_slice(&[1]),
        DType::U32,
    )
    .expect("past upload")
}

#[test]
#[ignore]
fn append_capacity_one_succeeds() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    // Capacity-1 arena: [1 row, stride]. One append at past=0 lands row 0.
    let (stride, cap) = (4usize, 1usize);
    let arena = CoreTensorOps::zeros(&dev, &Shape::from_slice(&[cap, stride]), DType::F32).unwrap();
    let rot = CoreTensorOps::from_cpu(
        &dev,
        &[7.0f32; 4],
        &Shape::from_slice(&[1, stride]),
        DType::F32,
    )
    .unwrap();
    let past = u32_past(&dev, 0);
    launch_kv_append(
        &dev,
        arena.as_ref(),
        rot.as_ref(),
        past.as_ref(),
        stride,
        1,
        KvRingPolicy::UNLIMITED,
    )
    .expect("capacity-1 append must succeed");
    dev.synchronize();
    let got = arena.to_cpu_vec_f32().expect("d2h");
    assert!(
        got.iter().all(|v| *v == 7.0),
        "capacity-1 append wrote row 0 incorrectly: {got:?}"
    );
}

#[test]
#[ignore]
fn append_full_unlimited_arena_fails_loud_at_host_guard() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    // Capacity-N arena filled to N tokens. The launcher has no capacity
    // check (kernel cannot bounds-check), so the fail-loud surface is the
    // HOST guard `launch_qkv_gemv_ring`: a pos mapping to row >= max_ctx
    // must return Err (never silent wrap / never silent truncate).
    let (stride, cap) = (4usize, 4usize);
    let arena = CoreTensorOps::zeros(&dev, &Shape::from_slice(&[cap, stride]), DType::F32).unwrap();
    let arena_rocm: &RocmStorage = as_rocm(arena.as_ref()).unwrap();
    let rot = CoreTensorOps::from_cpu(
        &dev,
        &[7.0f32; 4],
        &Shape::from_slice(&[1, stride]),
        DType::F32,
    )
    .unwrap();
    for t in 0..cap as u32 {
        let past = u32_past(&dev, t);
        launch_kv_append(
            &dev,
            arena.as_ref(),
            rot.as_ref(),
            past.as_ref(),
            stride,
            1,
            KvRingPolicy::UNLIMITED,
        )
        .expect("append within capacity");
    }
    dev.synchronize();
    // One more token at pos == cap: under UNLIMITED this is out of bounds.
    let err = launch_qkv_gemv_ring(arena_rocm, cap as u32, cap, KvRingPolicy::UNLIMITED);
    assert!(
        err.is_err(),
        "host guard must fail loud for pos == cap under UNLIMITED (kernel would write OOB)"
    );
    let msg = format!("{}", err.err().unwrap());
    eprintln!("[append-guard] full-arena UNLIMITED host-guard Err: {msg}");
}

#[test]
#[ignore]
fn append_full_capped_ring_evicts_by_policy_not_corrupts() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    // Capacity-N arena under a capped ring: a full-arena append is the ring
    // EVICT path — row (pos) maps back into the arena by policy, so the write
    // lands in-bounds and succeeds. This is the kernel's documented
    // "full arena" contract (eviction, not an error, not a wrap-onto-foreign
    // row). Uses the Phase-3 ring shape from qkv_attention.rs: cap=4, sink=1.
    let (cap, sinks, stride) = (4usize, 1usize, 4usize);
    let ring = KvRingPolicy::new(cap, sinks);
    let arena = CoreTensorOps::zeros(&dev, &Shape::from_slice(&[cap, stride]), DType::F32).unwrap();
    // Fill to capacity (tokens 0..cap-1), then overflow with tokens cap..2*cap.
    for t in 0..(cap * 2) as u32 {
        let src: Vec<f32> = vec![(t + 1) as f32; stride];
        let rot = CoreTensorOps::from_cpu(&dev, &src, &Shape::from_slice(&[1, stride]), DType::F32).unwrap();
        let past = u32_past(&dev, t);
        launch_kv_append(
            &dev,
            arena.as_ref(),
            rot.as_ref(),
            past.as_ref(),
            stride,
            1,
            ring,
        )
        .expect("ring append must succeed even when full (eviction policy)");
    }
    dev.synchronize();
    let got = arena.to_cpu_vec_f32().expect("d2h");
    // Sink row 0 still holds token 0.
    let row0 = &got[0..stride];
    assert!(
        row0.iter().all(|v| *v == 1.0),
        "ring sink row 0 must keep token 0 after overflow: {row0:?}"
    );
    // Every row holds exactly one token (constant row value) — no torn write.
    for r in 0..cap {
        let row = &got[r * stride..(r + 1) * stride];
        assert!(
            row.iter().all(|v| *v == row[0]),
            "ring row {r} must be uniformly one token after overflow: {row:?}"
        );
    }
}
