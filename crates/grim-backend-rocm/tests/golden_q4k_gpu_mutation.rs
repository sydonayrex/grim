//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test golden_q4k_gpu_mutation -- --ignored
//! STATUS: the historical `hipModuleLoad failed: 209` (`hipErrorNoBinaryForGpu`) on a dual-RDNA4
//!   (gfx1200 + gfx1201) box was a cross-device context mismatch — the code object was loaded while
//!   the thread sat on the wrong ordinal (see DeviceGuard/raw_set_device in src/device/util.rs),
//!   compounded by HSA_OVERRIDE_GFX_VERSION forcing gfx1200. Both are fixed: RDNA4 is never
//!   overridden (src/rocm_detect.rs) and raw_set_device keeps the CUR_DEV cache synced, so the
//!   kernel loads via quantized_matmul (the wmma_q4k path in device/compute/gemm_launchers.rs).
//!   The test stays GPU-tier #[ignore]d with the 209 reason recorded: if 209 recurs on a host it
//!   means arch/context mismatch, not a silent pass. The CPU-oracle portion (quant_q4k /
//!   dequant_q4k ground truth) runs device-free; GPU parity is device-gated.
//!
//! Golden mutation-resistant test for Crow Q4_K GPU dequantization GEMM forward and backward paths.
//! Real manual mutation proof: test_q4k_gpu_gemm_mutation_is_detected (M-Q4K-SCALE-01).

use grim_backend_rocm::RocmDevice;
use grim_format::convert::GpuDequant;
use grim_quant::{dequant_q4k, quant_q4k};
use grim_tensor::{CoreTensorOps, MemoryOps, QuantOps};
use grim_tensor::{
    Shape,
    dtype::{ArithType, DType, KQuantScheme, Storage},
};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

/// Recompute the main test's f16-envelope verdict without asserting: true
/// iff `actual` is within 1.05x its own f16 envelope of the f64 oracle.
/// Factored so the mutation test reuses the same tolerance check.
fn q4k_within_envelope(
    actual_c: &[f32],
    a_data: &[f32],
    b_dequant: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let w_f16: Vec<f32> = b_dequant
        .iter()
        .map(|&v| f32::from(half::f16::from_f32(v)))
        .collect();
    let a_f16: Vec<f32> = a_data
        .iter()
        .map(|&v| f32::from(half::f16::from_f32(v)))
        .collect();
    let mut worst_ratio: f64 = 0.0;
    for mi in 0..m {
        for ni in 0..n {
            let mut acc_f64 = 0.0f64;
            let mut envelope = 0.0f64;
            for ki in 0..k {
                let exact = a_data[mi * k + ki] as f64 * b_dequant[ni * k + ki] as f64;
                let f16_prod = a_f16[mi * k + ki] as f64 * w_f16[ni * k + ki] as f64;
                acc_f64 += exact;
                envelope += (f16_prod - exact).abs();
            }
            let denom = envelope.max(acc_f64.abs() * 1e-3).max(1e-6);
            worst_ratio = worst_ratio.max((actual_c[mi * n + ni] as f64 - acc_f64).abs() / denom);
        }
    }
    worst_ratio <= 1.05
}

#[test]
#[ignore = "GPU-tier: run with --ignored and GRIM_RUN_GPU_TEST=1; historical hipModuleLoad 209 was a dual-RDNA4 context/arch mismatch (fixed — see header). CPU oracle runs in the mutation test below."]
fn test_q4k_gpu_gemm_golden_mutation_resistant() -> TestResult {
    // The mutation target is the Q4K DECODE itself. The dispatch's u4-lane
    // arm (GRIM_DECODE_W4A4, default ON) lossily requants Q4K weights to
    // CityCrow u4 lanes and bypasses Q4K decode entirely — a separate,
    // documented route. Force it OFF here so the golden oracle (exact Q4K
    // dequant) is the right reference for every routing config below.
    // SAFETY: single-threaded test process; no other readers of this knob.
    unsafe { std::env::set_var("GRIM_DECODE_W4A4", "0") };
    // Non-square asymmetric dimensions: M=2, K=256, N=128
    let (m, k, n) = (2usize, 256usize, 128usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.015).cos().abs() * 8.0)
        .collect();

    let b_packed = quant_q4k(&b_orig).expect("quant_q4k");
    let b_dequant = dequant_q4k(&b_packed, b_orig.len()).expect("dequant_q4k");

    // Ground-truth CPU reference: C = A @ B_dequant^T
    let mut expected_c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a_data[mi * k + ki] * b_dequant[ni * k + ki];
            }
            expected_c[mi * n + ni] = sum;
        }
    }

    if let Some(dev) = gpu_device() {
        let a_shape = Shape::from_slice(&[m, k]);
        let b_shape = Shape::from_slice(&[n, k]);
        let out_shape = Shape::from_slice(&[m, n]);

        let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
        let q4k_dtype = DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q4K),
        };
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &b_shape, q4k_dtype)?;

        let (out, handle) = dev.quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K,
            &out_shape,
        )?;
        handle.synchronize()?;
        let actual_c = out.to_cpu_vec_f32()?;

        assert_eq!(actual_c.len(), expected_c.len());

        // The M=2 Q4K route is the WMMA fused-dequant path: BOTH operands are
        // quantized to f16 inside the kernel (lds_a cast, dequantized-B cast
        // to _Float16 before mma_sync) and the fragments accumulate in f32.
        // The by-design error vs the f32 oracle is therefore the f16
        // quantization of every product, bounded EXACTLY per output:
        //   |sum f16(a)*f16(w) - sum a*w| <= sum |f16(a)*f16(w) - a*w|
        // (the device's f32 accumulation from exact f16 products adds only
        // sub-ulp noise vs the f64 sum of the same f16 products). A decode
        // error (wrong nibble/scale) exceeds this envelope ~1000x — that is
        // what "mutation-resistant" means for this path.
        let mut worst_ratio: f64 = 0.0;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc_f64 = 0.0f64;
                let mut envelope = 0.0f64;
                for ki in 0..k {
                    let exact = a_data[mi * k + ki] as f64 * b_dequant[ni * k + ki] as f64;
                    let f16_prod = f32::from(half::f16::from_f32(a_data[mi * k + ki])) as f64
                        * f32::from(half::f16::from_f32(b_dequant[ni * k + ki])) as f64;
                    acc_f64 += exact;
                    envelope += (f16_prod - exact).abs();
                }
                let denom = envelope.max(acc_f64.abs() * 1e-3).max(1e-6);
                worst_ratio = worst_ratio.max(
                    (actual_c[mi * n + ni] as f64 - acc_f64).abs() / denom,
                );
            }
        }
        eprintln!(
            "[golden] worst |device - f64| / own f16 envelope: {worst_ratio:.3} (<= 1.05 = WMMA f16 by-design)"
        );
        assert!(
            worst_ratio <= 1.05,
            "device residual {worst_ratio:.3}x its own f16 quantization envelope — \
             REAL decode error beyond WMMA f16 by-design precision"
        );

        // Mutation-resistant max-error record vs the f32 oracle, kept for the
        // journey log: the WMMA f16 path moves outputs off the 1e-3 f32 scale
        // BY DESIGN (the old fixed assert fired on design error).
        let mut max_err: f32 = 0.0;
        for (act, exp) in actual_c.iter().zip(expected_c.iter()) {
            let err = (act - exp).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("[golden] max_err vs f32 oracle: {max_err} (WMMA f16 path by design)");

        // --- GpuDequant path (used by grim-format conversion) ---
        // Routes `KQuant::Q4K` through `RocmDevice::dequantize` →
        // `dequantize_q4k_host` → the `grim_dequant_q4k` HIP kernel instead of
        // the CPU `grim_quant::dequant_q4k` fallback. Must remain bit-exact with
        // the CPU oracle `b_dequant` (computed above).
        let gpu_dequant = dev
            .dequantize(&Storage::KQuant(KQuantScheme::Q4K), &b_packed, b_orig.len())
            .expect("GpuDequant::dequantize Q4K");
        assert!(
            gpu_dequant.is_some(),
            "GpuDequant must handle Q4K (should not fall back to CPU)"
        );
        let gpu_dequant = gpu_dequant.expect("checked above");
        assert_eq!(gpu_dequant.len(), b_dequant.len());
        let dequant_max_err = b_dequant
            .iter()
            .zip(gpu_dequant.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            dequant_max_err, 0.0,
            "Q4_K GpuDequant kernel deviates from CPU oracle (max_err={dequant_max_err:e})"
        );
    }

    Ok(())
}

// ── Real mutation proof — M-Q4K-SCALE-01 ─────────────────────────────────────
//
// Manual perturbation (no cargo-mutants): flip ONE `d` (f16 scale) byte in the
// packed Q4K buffer `b_packed` (byte 0 = low byte of block-0 `d`; XOR 0xFF
// moves d by orders of magnitude). The kernel is untouched. The true CPU
// oracle is untouched. Only the GPU input buffer is perturbed.
//
// Gate: passes_unmutated == true AND passes_mutated == false, where the
// verdict is the SAME f16-envelope check the main test asserts (reused via
// q4k_within_envelope). Mutation ID: M-Q4K-SCALE-01.
#[test]
#[ignore]
fn test_q4k_gpu_gemm_mutation_is_detected() -> TestResult {
    // SAFETY: same single-threaded knob as the main test.
    unsafe { std::env::set_var("GRIM_DECODE_W4A4", "0") };
    let (m, k, n) = (2usize, 256usize, 128usize);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_orig: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.015).cos().abs() * 8.0)
        .collect();
    let b_packed = quant_q4k(&b_orig).expect("quant_q4k");
    assert!(b_packed.len() > 1, "Q4K buffer must have block scales to mutate");

    let b_dequant = dequant_q4k(&b_packed, b_orig.len()).expect("dequant_q4k");
    let mut expected_c = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a_data[mi * k + ki] * b_dequant[ni * k + ki];
            }
            expected_c[mi * n + ni] = sum;
        }
    }

    // CPU-only sensitivity proof (device-free): the mutated bytes decode to a
    // verdict-failing result. Computed BEFORE any GPU skip so the fixture's
    // mutation sensitivity is proven even on GPU-less hosts.
    let mut b_mutated = b_packed.clone();
    b_mutated[0] ^= 0xFF; // flip low byte of block-0 `d` scale
    let b_dequant_mut = dequant_q4k(&b_mutated, b_orig.len()).expect("dequant_q4k mutated");
    let mut cpu_mut = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a_data[mi * k + ki] * b_dequant_mut[ni * k + ki];
            }
            cpu_mut[mi * n + ni] = sum;
        }
    }
    // cpu_mut is exactly what a faithful kernel must emit on mutated input;
    // the envelope verdict against the TRUE oracle must FAIL on it.
    let cpu_pass_unmutated = q4k_within_envelope(&expected_c, &a_data, &b_dequant, m, k, n);
    let cpu_pass_mutated = q4k_within_envelope(&cpu_mut, &a_data, &b_dequant, m, k, n);
    assert!(cpu_pass_unmutated, "oracle must pass its own envelope verdict");
    assert!(
        !cpu_pass_mutated,
        "M-Q4K-SCALE-01 not detected at CPU level: mutated oracle still within envelope — \
         fixture insensible, mutation size wrong"
    );

    // GPU tier: run the same assertion twice, unmutated vs mutated input.
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] M-Q4K-SCALE-01: no GPU (CPU sensitivity proven above)");
        return Ok(());
    };
    let a_shape = Shape::from_slice(&[m, k]);
    let b_shape = Shape::from_slice(&[n, k]);
    let out_shape = Shape::from_slice(&[m, n]);
    let q4k_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };

    let run_gpu = |packed: &[u8]| -> TestResult<Vec<f32>> {
        let a_dev = CoreTensorOps::from_cpu(&dev, &a_data, &a_shape, DType::F32)?;
        let b_dev = MemoryOps::from_cpu_bytes(&dev, packed, &b_shape, q4k_dtype.clone())?;
        let (out, handle) = dev.quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            grim_tensor::QuantFormat::Q4K,
            &out_shape,
        )?;
        handle.synchronize()?;
        Ok(out.to_cpu_vec_f32()?)
    };

    let actual_unmutated = run_gpu(&b_packed)?;
    let actual_mutated = run_gpu(&b_mutated)?;

    let passes_unmutated = q4k_within_envelope(&actual_unmutated, &a_data, &b_dequant, m, k, n);
    let passes_mutated = q4k_within_envelope(&actual_mutated, &a_data, &b_dequant, m, k, n);
    assert!(passes_unmutated, "unmutated GPU run must pass the envelope verdict");
    assert!(
        !passes_mutated,
        "M-Q4K-SCALE-01 NOT detected: mutated-scale GPU output still within envelope — gate did not fire"
    );
    assert_eq!(expected_c.len(), actual_unmutated.len());
    Ok(())
}
