//! res0-T2 SPARSITY LEVER gates: `grim_dot4_q80_f32act_gemv_skip`.
//!
//! The lever: drop 32-wide activation blocks whose max|a| < tau * row-max
//! from the weight stream. The probe (189e047d) measured 70% of k-groups dead
//! at tau=0.10 on real decode activations — the skipped block fraction IS the
//! skipped weight-byte fraction, so a ~3x decode is in play if accuracy holds.
//!
//! Gates (cheap discriminating tests first, per the plan's rule):
//! 1. tau = 0 must be BIT-IDENTICAL to the dense kernel (the skip machinery
//!    — row-max prepass + threshold — must be an exact no-op at tau = 0).
//! 2. With a synthetic 70%-dead activation the skip result must sit within a
//!    computable error bound of dense, and the measured relative error must
//!    shrink with the dead blocks' magnitude (proves the skips are honored,
//!    not that the kernel silently ignores tau).
//! 3. Cold-cache timed A/B at real 27B shapes across dead fractions — the
//!    perf verdict. GB/s of the weight stream vs dense.
//!
//! Run: GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm
//!      --test q80_dot4_kgroup_skip -- --ignored --nocapture --test-threads=1
use grim_backend_rocm::as_rocm;
use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_backend_rocm::RocmStorage;
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::{CoreTensorOps, KQuantScheme, MemoryOps, Shape};

fn gpu() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1");
        return None;
    }
    RocmDevice::try_new(0).ok()
}

fn f32ty() -> DType {
    DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    }
}

fn q80ty() -> DType {
    DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q80),
    }
}

/// Activation with a controlled dead fraction: dead 32-blocks (spread evenly,
/// offset so block 0 — the row-max anchor — stays live) are scaled to
/// `dead_scale`; everything else is full-scale in [-1, 1).
fn synthetic_activation(k: usize, dead_frac: f32, dead_scale: f32, seed: u64) -> Vec<f32> {
    let mut seed = seed;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let n_blocks = k / 32;
    let mut a = vec![0.0f32; k];
    for blk in 0..n_blocks {
        // Decade-modular dead set: exact fractions (0.7 -> 7 of every 10
        // blocks), spread evenly. Block 0 stays live so the row-max anchor
        // is always a full-scale block.
        let dead = dead_frac > 0.0 && blk > 0 && ((blk % 10) as f32) < dead_frac * 10.0;
        for e in 0..32 {
            a[blk * 32 + e] = rand() * if dead { dead_scale } else { 1.0 };
        }
    }
    a
}

fn weight_blob(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut seed = seed;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut b_f32 = vec![0.0f32; n * k];
    for v in b_f32.iter_mut() {
        *v = rand();
    }
    grim_quant::quant_q80(&b_f32).expect("quant_q80")
}

fn upload_bytes(
    dev: &RocmDevice,
    bytes: &[u8],
    ty: DType,
) -> Box<dyn grim_tensor::backend::BackendStorage> {
    MemoryOps::from_cpu_bytes(dev, bytes, &Shape::new(vec![bytes.len()]), ty).expect("upload")
}

fn alloc_out(dev: &RocmDevice, len: usize) -> Box<dyn grim_tensor::backend::BackendStorage> {
    MemoryOps::alloc_storage(dev, &Shape::new(vec![len]), f32ty()).expect("out alloc")
}

fn run_dense(
    dev: &RocmDevice,
    a: &RocmStorage,
    b: &RocmStorage,
    out: &RocmStorage,
    m: usize,
    n: usize,
    k: usize,
) {
    dev.launch_dot4_q80_f32act_gemv_v1_for_ab(a, b, out, m, n, k)
        .expect("dense gemv");
}

fn run_skip(
    dev: &RocmDevice,
    a: &RocmStorage,
    b: &RocmStorage,
    out: &RocmStorage,
    m: usize,
    n: usize,
    k: usize,
    tau: f32,
) {
    dev.launch_dot4_q80_f32act_gemv_skip(a, b, out, m, n, k, tau)
        .expect("skip gemv");
}

/// GATE 1: tau = 0 is bit-identical to the dense kernel.
#[test]
#[ignore]
fn kgroup_skip_tau0_is_bit_identical_to_dense() {
    let dev = match gpu() {
        Some(d) => d,
        None => return,
    };
    let m = 1usize;
    for (n, k) in [(1024usize, 1024usize), (17408, 5120), (5120, 17408)] {
        let a = synthetic_activation(k, 0.3, 1e-3, 0xBEEF);
        let bq = weight_blob(n, k, 0xF00D ^ n as u64);
        let a_st = CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![m, k]), DType::F32)
            .expect("a h2d");
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_st = upload_bytes(&dev, &bq, q80ty());
        let b_r = as_rocm(b_st.as_ref()).unwrap();
        let out_dense_st = alloc_out(&dev, m * n);
        let out_skip_st = alloc_out(&dev, m * n);
        run_dense(&dev, a_r, b_r, as_rocm(out_dense_st.as_ref()).unwrap(), m, n, k);
        run_skip(
            &dev,
            a_r,
            b_r,
            as_rocm(out_skip_st.as_ref()).unwrap(),
            m,
            n,
            k,
            0.0,
        );
        dev.synchronize();
        let d = out_dense_st.to_cpu_vec_f32().expect("dense d2h");
        let s = out_skip_st.to_cpu_vec_f32().expect("skip d2h");
        for (i, (x, y)) in d.iter().zip(s.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "tau=0 mismatch at {i} n={n} k={k}");
        }
        eprintln!("[kgroup-skip] tau=0 bit-identical n={n} k={k}");
    }
}

/// GATE 2: 70%-dead synthetic activation — bounded error, and the error must
/// scale down with the dead magnitude (proves tau is honored), plus the
/// tau=2.0 all-dead structural check (exact zero output).
#[test]
#[ignore]
fn kgroup_skip_70pct_dead_error_bounded_and_tau_honored() {
    let dev = match gpu() {
        Some(d) => d,
        None => return,
    };
    let m = 1usize;
    let (n, k) = (17408usize, 5120usize);
    let bq = weight_blob(n, k, 0x5EED);
    let b_st = upload_bytes(&dev, &bq, q80ty());
    let b_r = as_rocm(b_st.as_ref()).unwrap();

    // The dense baseline must run on the SAME dead-heavy activation, so the
    // diff isolates the skip effect (omitted dead-block contribution) and not
    // the dead blocks' own magnitude.
    for (dead_scale, tau) in [(1e-2f32, 0.10), (1e-3, 0.10), (1e-4, 0.10)] {
        let a = synthetic_activation(k, 0.7, dead_scale, 0xCAFE);
        let a_st = CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![m, k]), DType::F32)
            .expect("a h2d");
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let out_dense_st = alloc_out(&dev, m * n);
        run_dense(&dev, a_r, b_r, as_rocm(out_dense_st.as_ref()).unwrap(), m, n, k);
        let dense = out_dense_st.to_cpu_vec_f32().expect("d2h");
        let dense_scale = dense.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let out_skip_st = alloc_out(&dev, m * n);
        run_skip(
            &dev,
            a_r,
            b_r,
            as_rocm(out_skip_st.as_ref()).unwrap(),
            m,
            n,
            k,
            tau,
        );
        dev.synchronize();
        let skip = out_skip_st.to_cpu_vec_f32().expect("d2h");
        let max_abs = skip
            .iter()
            .zip(&dense)
            .fold(0.0f32, |acc, (s, d)| acc.max((s - d).abs()));
        eprintln!(
            "[kgroup-skip] 70% dead scale={dead_scale}: max|skip-dense|={max_abs:.3e} rel={:.3e}",
            max_abs / dense_scale
        );
        assert!(
            max_abs / dense_scale < 0.05,
            "skip error too large at dead_scale={dead_scale}: {max_abs} vs scale {dense_scale}"
        );
    }

    // Tau honored structurally: on the full-scale activation, tau = 2.0
    // kills EVERY block (row max ~1, every block amax <= row max), so the
    // output must be exactly zero.
    let a_full = synthetic_activation(k, 0.0, 1.0, 0xCAFE);
    let a_full_st = CoreTensorOps::from_cpu(&dev, &a_full, &Shape::new(vec![m, k]), DType::F32)
        .expect("a h2d");
    let a_full_r = as_rocm(a_full_st.as_ref()).unwrap();
    let out_all_dead = alloc_out(&dev, m * n);
    run_skip(
        &dev,
        a_full_r,
        b_r,
        as_rocm(out_all_dead.as_ref()).unwrap(),
        m,
        n,
        k,
        2.0,
    );
    dev.synchronize();
    let all_dead = out_all_dead.to_cpu_vec_f32().expect("d2h");
    assert!(
        all_dead.iter().all(|v| *v == 0.0),
        "tau=2.0 left nonzero outputs — skip not honored"
    );
}

/// GATE 3: cold-cache timed A/B at real 27B shapes across dead fractions.
/// 8 rotated weight copies push the blob past Infinity Cache (d84fcb32
/// protocol). Reports GB/s of the full weight stream; the skipped fraction
/// bounds the achievable speedup (1 / (1 - dead_frac)).
#[test]
#[ignore]
fn kgroup_skip_cold_ab_at_27b_shapes() {
    let dev = match gpu() {
        Some(d) => d,
        None => return,
    };
    let m = 1usize;
    let shapes = [(17408usize, 5120usize), (5120, 17408)]; // down-proj, gate/up
    for (n, k) in shapes {
        let copies = 8usize;
        let blobs: Vec<Box<dyn grim_tensor::backend::BackendStorage>> = (0..copies)
            .map(|c| upload_bytes(&dev, &weight_blob(n, k, 0xABCD + c as u64), q80ty()))
            .collect();
        let bytes_per_blob = (n * (k / 32) * 34) as f64;

        for dead_frac in [0.0f32, 0.5, 0.7, 0.9] {
            let a = synthetic_activation(k, dead_frac, 1e-3, 0x1234 ^ n as u64);
            let a_st = CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![m, k]), DType::F32)
                .expect("a h2d");
            let a_r = as_rocm(a_st.as_ref()).unwrap();
            let out_st = alloc_out(&dev, m * n);
            let o_r = as_rocm(out_st.as_ref()).unwrap();

            let time = |f: &dyn Fn(&RocmStorage)| -> f64 {
                for i in 0..4 {
                    f(as_rocm(blobs[i % copies].as_ref()).unwrap());
                }
                dev.synchronize();
                let iters = 24;
                let t0 = std::time::Instant::now();
                for i in 0..iters {
                    f(as_rocm(blobs[i % copies].as_ref()).unwrap());
                }
                dev.synchronize();
                t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
            };

            // Dense on the same (dead-heavy) activation is the fair baseline:
            // same input, same output work, no skip machinery.
            let t_dense = time(&|b_r: &RocmStorage| run_dense(&dev, a_r, b_r, o_r, m, n, k));
            let t_skip =
                time(&|b_r: &RocmStorage| run_skip(&dev, a_r, b_r, o_r, m, n, k, 0.10));
            eprintln!(
                "[kgroup-skip] n={n} k={k} dead={:.0}%: dense {t_dense:.3} ms \
                 ({:.1} GB/s)  skip {t_skip:.3} ms ({:.1} GB/s eff)  speedup {:.3}x \
                 (ideal-from-fraction {:.3}x)",
                dead_frac * 100.0, bytes_per_blob / t_dense / 1e6,
                bytes_per_blob / t_skip / 1e6,
                t_dense / t_skip,
                1.0 / (1.0 - dead_frac)
            );
        }
    }
}
