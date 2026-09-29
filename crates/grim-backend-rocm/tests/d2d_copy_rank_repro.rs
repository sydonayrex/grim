//! `copy_slice_into` D2D repro: the exact reshape that aborts the 9B run is a
//! rank-3 `[1,16,256]` source into a rank-2 `[1,4096]` destination — 16384
//! bytes, element offsets 0.
//!
//! `copy_slice_range` has no rank logic: it derives a flat byte count from the
//! element count and hands the two device pointers to `hipMemcpyAsync`. So if
//! the rank-3 -> rank-2 pair is refused, either the refusal is not about rank
//! (and the rank-2 -> rank-2 control at the same byte count fails too), or the
//! SOURCE storage differs in some way that rank only correlates with. The
//! controls below are therefore the point of this file: same device, same byte
//! count, same dtype, same stream, one variable changed per case.
//!
//! Every case READS THE RESULT BACK. A copy that returns `Ok` without moving
//! the bytes is a different defect from one that is refused, and in a
//! `?`-chained caller the two are indistinguishable.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, MemoryOps, Shape};
use std::sync::{Arc, OnceLock};

/// One device for the whole binary: five `RocmDevice::try_new(0)` calls racing
/// on one card would test the allocator, not the copy.
fn gpu_device() -> Option<Arc<RocmDevice>> {
    static DEV: OnceLock<Option<Arc<RocmDevice>>> = OnceLock::new();
    DEV.get_or_init(|| {
        if !grim_backend_rocm::gpu_test_enabled() {
            eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
            return None;
        }
        let _lock = grim_backend_rocm::device::util::gpu_test_lock();
        Some(Arc::new(RocmDevice::try_new(0).expect(
            "GRIM_GPU_TEST=1 is set but RocmDevice::try_new(0) failed. Failing loudly rather \
         than catch_unwind().ok(): a swallowed init failure turns this gate GREEN with \
         zero assertions run, which is how a real defect hides behind a passing test.",
        )))
    })
    .clone()
}

/// Deterministic, non-symmetric filler: a palindromic pattern would let a copy
/// that moves nothing still compare equal.
fn payload(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.5 - 7.25).collect()
}

/// One D2D copy of `count` elements, then verify the bytes actually landed.
fn copy_and_check(
    dev: &RocmDevice,
    label: &str,
    src_dims: Vec<usize>,
    dst_dims: Vec<usize>,
) -> Result<(), String> {
    let count: usize = src_dims.iter().product();
    assert_eq!(
        dst_dims.iter().product::<usize>(),
        count,
        "{label}: this is only a copy when both sides hold the same element count"
    );
    let want = payload(count);

    let src = dev
        .from_cpu(&want, &Shape::new(src_dims.clone()), DType::F32)
        .map_err(|e| format!("{label}: upload src {src_dims:?}: {e}"))?;
    let dst = dev
        .alloc_storage(&Shape::new(dst_dims.clone()), DType::F32)
        .map_err(|e| format!("{label}: alloc dst {dst_dims:?}: {e}"))?;

    dev.copy_slice_into(dst.as_ref(), src.as_ref(), 0, count)
        .map_err(|e| {
            format!(
                "{label}: copy_slice_into REFUSED {src_dims:?} -> {dst_dims:?} ({count} elems): {e}"
            )
        })?;
    dev.synchronize();

    let got = dst
        .to_cpu_vec_f32()
        .map_err(|e| format!("{label}: read back: {e}"))?;
    if got.len() != count {
        return Err(format!(
            "{label}: read back {} elems, want {count}",
            got.len()
        ));
    }
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if g != w {
            return Err(format!(
                "{label}: {src_dims:?} -> {dst_dims:?} returned Ok but element {i} is {g}, \
                 want {w} — the bytes did not move"
            ));
        }
    }
    eprintln!("[d2d] {label}: {src_dims:?} -> {dst_dims:?} ({count} elems) OK, bytes verified");
    Ok(())
}

/// The shape pair the 9B run actually hits.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn d2d_rank3_to_rank2_matches_the_failing_run_shape() {
    let Some(dev) = gpu_device() else { return };
    copy_and_check(&dev, "run-shape", vec![1, 16, 256], vec![1, 4096]).unwrap();
}

/// Control: identical byte count and dtypes, source rank already equal to the
/// destination rank. If this passes while the case above fails, rank is
/// implicated. If it fails too, the refusal is about something else entirely
/// and the "rank-3 to rank-2" framing is wrong.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn d2d_rank2_to_rank2_same_byte_count_is_the_control() {
    let Some(dev) = gpu_device() else { return };
    copy_and_check(&dev, "control-2d", vec![1, 4096], vec![1, 4096]).unwrap();
}

/// Second control: rank-3 to rank-3 at the same byte count. Separates "the
/// source is rank 3" from "the destination is rank 2".
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn d2d_rank3_to_rank3_same_byte_count_is_the_second_control() {
    let Some(dev) = gpu_device() else { return };
    copy_and_check(&dev, "control-3d", vec![1, 16, 256], vec![1, 16, 256]).unwrap();
}

/// The copy must work for any rank pairing, not just this one shape — a
/// `reshaped_view` fast path is only correct if the D2D under it is.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn d2d_copy_is_rank_agnostic() {
    let Some(dev) = gpu_device() else { return };
    let cases: Vec<(&str, Vec<usize>, Vec<usize>)> = vec![
        ("1d->2d", vec![4096], vec![1, 4096]),
        ("2d->1d", vec![1, 4096], vec![4096]),
        ("2d->3d", vec![16, 256], vec![1, 16, 256]),
        ("3d->2d", vec![1, 16, 256], vec![16, 256]),
        ("4d->2d", vec![1, 1, 16, 256], vec![16, 256]),
        ("3d->3d-regrouped", vec![1, 16, 256], vec![16, 1, 256]),
    ];
    let mut failures = Vec::new();
    for (label, s, d) in cases {
        if let Err(e) = copy_and_check(&dev, label, s, d) {
            failures.push(e);
        }
    }
    assert!(
        failures.is_empty(),
        "D2D copy is not rank-agnostic:\n  {}",
        failures.join("\n  ")
    );
}

/// A copy at a NON-ZERO destination element offset is the KV-append shape.
/// `reshaped_view` always copies at offset 0, so this does not explain the 9B
/// failure, but an offset defect here would later read as a rank defect and
/// this pair differs in exactly one variable.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn d2d_rank3_to_rank2_at_nonzero_dst_offset() {
    let Some(dev) = gpu_device() else { return };
    let count = 4096usize;
    let want = payload(count);
    let src = dev
        .from_cpu(&want, &Shape::new(vec![1, 16, 256]), DType::F32)
        .expect("upload src");
    let dst = dev
        .alloc_storage(&Shape::new(vec![2, 4096]), DType::F32)
        .expect("alloc dst");
    dev.copy_slice_into(dst.as_ref(), src.as_ref(), 4096, count)
        .expect("D2D at dst element offset 4096");
    dev.synchronize();
    let got = dst.to_cpu_vec_f32().expect("read back");
    assert_eq!(got.len(), 2 * count, "dst holds two rows");
    for i in 0..count {
        assert!(
            got[count + i] == want[i],
            "row 1 element {i}: got {}, want {}",
            got[count + i],
            want[i]
        );
    }
}
