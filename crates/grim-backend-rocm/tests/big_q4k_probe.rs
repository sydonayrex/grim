//! LDS-stage dump probe for the 64x64x32 big-tile kernel (dump build).
//!
//! FINDING (2026-10-08): with a dump build (sA[0]/sA[1] copied to C after the
//! fills), BOTH LDS stages are CORRECT at k=64 — stage 0 = A k-slice 0, stage 1
//! = A k-slice 1, f16-rounded, zero inf/nan. The INF in C therefore arises in
//! the CONSUME (fragment loads / mma_sync with 2 M-frags x 2 N-frags per wave),
//! not in the fills. The fills were verified independently: the synthetic-block
//! probe (analytic block: d=1.0, scales=4, nibbles=4 -> weight=16) shows the
//! single-K-step path (k=32) EXACT for all rows (max_rel 1e-4), so fill, MMA,
//! store and the masked write are all sound at one K step.
//!
//! NEXT: bisect the consume — (a) single-buffered 3-barrier variant (no
//! prefetch) to separate the pipeline from the fragment loads; (b) 1 M-frag x
//! 4 N-frags per wave (drop the mi loop) to test whether the [2][2] fragment
//! arrays under HIPRTC+rocWMMA miscompile; (c) device printf on the fragment
//! values after load.
//!
//! This dump build requires the kernel's PROBE DUMP block (temporarily
//! unconditional in kernels/wmma_big_gemm.rs — restore the `#ifdef` when done).
use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage};

#[test]
#[ignore]
fn big_q4k_minimal() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() { return; }
    let dev = RocmDevice::try_new(0).unwrap();
    let n = 64usize;
    for k in [32usize, 64usize] {
        let m = 64usize;
        let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
        let b_packed = grim_quant::quant_q4k(&b_host).unwrap();

        let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32).unwrap();
        let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q4K) };
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype).unwrap();
        let dump_len = 64 * 40 + 64 * 40;
        let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![dump_len]), DType::F32).unwrap();
        let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
        let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
        let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();

        dev.launch_wmma_big_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
        let got = out.to_cpu_vec_f32().unwrap();

        // sA stage layout: [stage][rr][kk], rr 0..64, kk 0..40 (ld=40, kk<32 live)
        for stage in 0..2usize {
            let base = stage * 64 * 40;
            let mut inf_count = 0usize; let mut nan_count = 0usize;
            let mut bad_rows = std::collections::BTreeSet::new();
            for rr in 0..64usize {
                for kk in 0..32usize {
                    let v = got[base + rr * 40 + kk];
                    if v.is_infinite() { inf_count += 1; bad_rows.insert(rr); }
                    if v.is_nan() { nan_count += 1; bad_rows.insert(rr); }
                }
            }
            eprintln!("[dump] k={k} stage={stage} inf={inf_count} nan={nan_count} bad_rows={:?}",
                      bad_rows.iter().take(8).collect::<Vec<_>>());
        }
        // spot: stage0 row0 k0..7 vs A[0][0..8]
        eprintln!("[dump] sA[0] row0 k0..8 = {:?}", &got[0..8]);
        eprintln!("[dump] sA[1] row0 k0..8 = {:?}", &got[2560..2568]);
        eprintln!("[dump] expect A[0][0..8] = {:?}", &a_host[0..8]);
    }
}

/// DUMP probe: the kernel (dump build) returns sA stage 0 + stage 1 in C.
/// Locates the inf: is stage 1's A tile (K slice 1..) garbage after the
/// pipeline handoff, while stage 0 is clean?
#[test]
#[ignore]
fn big_q4k_dump_read() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() { return; }
    let dev = RocmDevice::try_new(0).unwrap();
    let n = 64usize;
    let dump_len = 64 * 40 + 64 * 40;
    for k in [64usize, 2048usize] {
        let m = 64usize;
        let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
        let b_packed = grim_quant::quant_q4k(&b_host).unwrap();
        let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32).unwrap();
        let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q4K) };
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype).unwrap();
        let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![dump_len]), DType::F32).unwrap();
        let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
        let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
        let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();
        dev.launch_wmma_big_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
        let got = out.to_cpu_vec_f32().unwrap();

        for stage in 0..2usize {
            let base = stage * 64 * 40;
            let _kslice = stage;
            // spot-check: stage s should hold A[k = stage*32 .. +32] for the
            // last fill of that stage. We check finiteness + the first fill's
            // expected value is gone, so just report.
            let mut inf = 0usize; let mut nan = 0usize;
            let mut maxv = f32::NEG_INFINITY; let mut minv = f32::INFINITY;
            for i in 0..64 * 40 {
                let v = got[base + i];
                if v.is_infinite() { inf += 1; }
                if v.is_nan() { nan += 1; }
                if !v.is_infinite() && !v.is_nan() { maxv = maxv.max(v); minv = minv.min(v); }
            }
            eprintln!("[dump] k={k} stage={stage} inf={inf} nan={nan} range=[{minv:.4}, {maxv:.4}]");
        }
        // row 0 k 0..8 of each stage vs expected A[0][k0..]
        for stage in 0..2usize {
            let k0 = stage * 32;
            let vals: Vec<f32> = (0..8).map(|i| got[stage * 64 * 40 + i]).collect();
            eprintln!("[dump] k={k} stage={stage} row0 = {:?} (expect A[0][{}..] = {:?})",
                      vals, k0, &a_host[k0..k0 + 8.min(k)]);
        }
    }
}
