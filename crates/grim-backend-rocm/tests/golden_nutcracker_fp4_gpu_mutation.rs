//! Golden mutation-resistant test for the Nutcracker FP4 GPU dequantization
//! GEMM, matching the standard set by the other corvid codecs
//! (`golden_raven_fp8_gpu_mutation.rs`, `golden_jay_magpie_gpu_mutation.rs`).
//!
//! Builds packed Nutcracker weights, computes a CPU ground truth from
//! `dequant_nutcracker`, and compares against the ROCm `grim_nutcracker_gemv` /
//! `_gemm_tiled` results. Non-square, asymmetric dimensions so a stride or
//! transposition bug cannot pass.
//!
//! The comparison is deliberately against the *reference decoder*, not against
//! a re-implementation of the kernel: if the kernel and the reference ever
//! drift into using the same wrong scale semantics, the shared
//! `dequant_nutcracker` call is what both are checked against.

use grim_backend_rocm::RocmDevice;
use grim_quant::{dequant_nutcracker, quant_nutcracker};
use grim_tensor::{CoreTensorOps, MemoryOps, QuantOps};
use grim_tensor::{
    Shape,
    dtype::{ArithType, DType, FloatPackScheme, Storage},
};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm"))
        .ok()
}

/// Spread values across several orders of magnitude so blocks differ and the
/// per-block selector actually has to choose.
fn weights(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let base = (t * 9.0 - 4.5) * 2.0f32.powi((i % 5) as i32 - 2);
            base + 0.15 * ((i % 7) as f32 - 3.0)
        })
        .collect()
}

/// A [m,k] @ [k,n] ground truth computed in f32.
fn ground_truth(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[mi * k + ki] * b[ni * k + ki];
            }
            c[mi * n + ni] = sum;
        }
    }
    c
}

fn run(m: usize, k: usize, n: usize, label: &str) -> TestResult {
    let a_data: Vec<f32> = weights(m * k);
    let b_orig: Vec<f32> = weights(k * n);

    // Pack with the production packer, then decode with the reference. The GPU
    // gets the exact same bytes.
    let b_packed = quant_nutcracker(&b_orig)?;
    let b_dequant = dequant_nutcracker(&b_packed, k * n)?;
    let expected = ground_truth(&a_data, &b_dequant, m, k, n);

    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] {label}: no GPU");
        return Ok(());
    };

    let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &Shape::from_slice(&[m, k]), DType::F32)?;
    let b_dev = MemoryOps::from_cpu_bytes(
        &dev,
        &b_packed,
        &Shape::from_slice(&[n, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::FloatPack(FloatPackScheme::NutFp4),
        },
    )?;
    let out_shape = Shape::from_slice(&[m, n]);

    // No separate scale tensor: Nutcracker carries its block scale inline in
    // the packed buffer, so the dummy is empty by design.
    let (out, handle) = dev.quantized_matmul(
        a_dev.as_ref(),
        b_dev.as_ref(),
        &[],
        grim_tensor::QuantFormat::Fp4,
        &out_shape,
    )?;
    handle.synchronize()?;
    let actual = out.to_cpu_vec_f32()?;

    assert_eq!(actual.len(), expected.len(), "{label}: output length");
    let mut max_err: f32 = 0.0;
    let mut peak: f32 = 0.0;
    for (a, b) in expected.iter().zip(actual.iter()) {
        max_err = max_err.max((a - b).abs());
        peak = peak.max(a.abs());
    }
    let bound = (peak * 0.15).max(1e-4);
    assert!(
        max_err <= bound,
        "Nutcracker GPU matmul max error {max_err} exceeds {bound} \
         (peak {peak}) for {label}"
    );
    Ok(())
}

/// Decode path: M=1 (GEMV) and M=3 (cooperative reduction), non-square.
#[test]
#[ignore = "requires AMD GPU"]
fn test_nutcracker_gpu_gemm_golden_mutation_resistant() -> TestResult {
    for (m, k, n) in [(1usize, 256usize, 128usize), (3, 256, 64)] {
        run(m, k, n, &format!("m={m} k={k} n={n}"))?;
    }
    Ok(())
}

/// Prefill path: M>4 dispatches to the tiled GEMM.
#[test]
#[ignore = "requires AMD GPU"]
fn test_nutcracker_gpu_tiled_gemm_golden_mutation_resistant() -> TestResult {
    run(16, 512, 32, "tiled m=16 k=512 n=32")?;
    Ok(())
}

/// The packer's selector must not be inert, and must not collapse: a broken
/// kernel that ignored the selector bits would still round-trip through the
/// reference decoder, so this checks the *data* has both selector variety and a
/// healthy error profile before any GPU result is trusted.
#[test]
fn test_nutcracker_packed_data_is_well_formed() -> TestResult {
    let w = weights(512);
    let packed = quant_nutcracker(&w)?;
    assert_eq!(
        packed.len(),
        512usize.div_ceil(16) * 9,
        "9 bytes per 16 values"
    );

    // Selector variety across blocks.
    let sels: std::collections::BTreeSet<u8> = packed.chunks(9).map(|b| b[0] & 0x3).collect();
    assert!(
        sels.len() > 1,
        "selector never varied across blocks: {sels:?} — the packer sweep is inert"
    );

    // Reconstructed values must be finite and track the originals.
    let rec = dequant_nutcracker(&packed, w.len())?;
    for (i, &b) in rec.iter().enumerate() {
        assert!(b.is_finite(), "elem {i} reconstructed as {b}");
    }

    // NRMSE, not per-element relative error. A block's shared exponent is
    // pinned by its largest element, so a small element in a block with a large
    // max can carry unbounded *relative* error while being perfectly correct in
    // absolute terms. Relative-per-element would flag that as a failure.
    let mut se = 0.0f32;
    for (a, b) in w.iter().zip(rec.iter()) {
        se += (a - b) * (a - b);
    }
    let rmse = (se / w.len() as f32).sqrt();
    let rms = (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt();
    let nrmse = rmse / rms;
    assert!(nrmse < 0.15, "NRMSE {nrmse:.4} is implausible for 4.5 bpw");
    Ok(())
}
