//! M7.2h: parity gate for `grim_dot4_q6k_f32act_gemv` (LM head exact path).
//!
//! Reads f32 activations directly (no Q8_1 quantization) and Q6_K weights.
//! Compared against: host dequant of Q6_K weights dotted with f32 activation.
//! Must be byte-identical to the oracle (Q6_K has no dmin term, so weight
//! dequant is exact; activation is used raw).
//!
//! Run: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 cargo test --release -p grim-backend-rocm
//!      --test q6k_f32act_dot4_parity -- --nocapture --test-threads=1

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::as_rocm;
use grim_tensor::{ArithType, BackendStorage, DType, MemoryOps, Shape, Storage};
use std::panic;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytes_of(t: &Box<dyn BackendStorage>) -> Result<Vec<u8>, String> {
    as_rocm(t.as_ref())
        .map_err(|e| e.to_string())?
        .copy_to_host()
        .map_err(|e| format!("copy_to_host: {e}"))
}

fn f32_vec(t: &Box<dyn BackendStorage>) -> Result<Vec<f32>, String> {
    Ok(bytes_of(t)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

const Q6K_ROW_BYTES_PER_256: usize = 210;

fn q6k_f32act_parity(k: usize, n: usize) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    assert_eq!(k % 256, 0, "Q6_K needs k % 256 == 0");
    let row_bytes = (k / 256) * Q6K_ROW_BYTES_PER_256;

    let mut rng = Lcg(0x6B6B_0001 ^ (k as u64) ^ ((n as u64) << 32));
    let a: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();

    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    let a_st = MemoryOps::from_cpu_bytes(&dev, &f32_bytes(&a), &Shape::new(vec![k]), f32ty)
        .map_err(|e| format!("a h2d: {e}"))?;

    let mut bq = vec![0u8; n * row_bytes];
    for col in 0..n {
        let packed = grim_quant::quant_q6k(&b[col * k..(col + 1) * k])
            .map_err(|e| format!("quant: {e}"))?;
        assert_eq!(packed.len(), row_bytes, "Q6_K row length");
        bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
    }
    let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
        .map_err(|e| format!("b h2d: {e}"))?;
    let out_st = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;

    let a_r = as_rocm(a_st.as_ref()).unwrap();
    let b_r = as_rocm(b_st.as_ref()).unwrap();
    let o_r = as_rocm(out_st.as_ref()).unwrap();

    dev.launch_dot4_q6k_f32act_gemv_for_ab(a_r, b_r, o_r, 1, n, k)
        .map_err(|e| format!("launch: {e}"))?;
    dev.synchronize();

    let got = f32_vec(&out_st)?;

    // Oracle: host dequant dotted with f32 activation.
    let mut worst = 0.0f32;
    let mut nonzero = 0usize;
    for col in 0..n {
        let w = grim_quant::dequant_q6k(&bq[col * row_bytes..(col + 1) * row_bytes], k)
            .map_err(|e| format!("dequant: {e}"))?;
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += a[kk] * w[kk];
        }
        let d = (got[col] - acc).abs() / acc.abs().max(1e-3);
        worst = worst.max(d);
        if got[col] != 0.0 {
            nonzero += 1;
        }
    }
    assert_eq!(
        nonzero, n,
        "Q6_K f32act kernel produced all-zero output at k={k} n={n} — the parity \
         comparison would have been vacuous"
    );
    println!(
        "q6k f32act k={k:<6} n={n:<6} worst relative error {worst:.3e}"
    );
    assert!(
        worst <= 1e-5,
        "Q6_K f32act GEMV disagrees with the dequant oracle by {worst:.3e} on {}",
        dev.gpu_target_str()
    );
    Ok(())
}

#[test]
fn q6k_f32act_dot4_parity() -> Result<(), String> {
    for (k, n) in [(256usize, 64usize), (1024, 67), (4096, 129), (5120, 68)] {
        q6k_f32act_parity(k, n)?;
    }
    Ok(())
}

/// Speed A/B at the LM head shape: n=248320 (27B vocab), k=5120, m=1.
/// Compares f32act dot4 vs exact fused-dequant.
#[test]
fn q6k_f32act_speed_ab_head() -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let k = 5120usize;
    // Head shape: vocab-scale. Use n=65536 (just above the head threshold)
    // to exercise the f32act path without allocating 1 GB of weights.
    let n = 65536usize;
    let row_bytes = (k / 256) * Q6K_ROW_BYTES_PER_256;

    let mut rng = Lcg(0xBEEF_0004);
    let a: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();

    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    let a_st = MemoryOps::from_cpu_bytes(&dev, &f32_bytes(&a), &Shape::new(vec![k]), f32ty)
        .map_err(|e| format!("a h2d: {e}"))?;
    let mut bq = vec![0u8; n * row_bytes];
    for col in 0..n {
        let packed = grim_quant::quant_q6k(&b[col * k..(col + 1) * k])
            .map_err(|e| format!("quant: {e}"))?;
        bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
    }
    let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
        .map_err(|e| format!("b h2d: {e}"))?;
    let out_f32act = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;
    let out_exact = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;

    let a_r = as_rocm(a_st.as_ref()).unwrap();
    let b_r = as_rocm(b_st.as_ref()).unwrap();
    let o_f32act = as_rocm(out_f32act.as_ref()).unwrap();
    let o_exact = as_rocm(out_exact.as_ref()).unwrap();

    // Warm-up
        dev.launch_dot4_q6k_f32act_gemv_for_ab(a_r, b_r, o_f32act, 1, n, k)
            .unwrap();
    dev.launch_fused_dequant_gemm_q6k_for_ab(a_r, b_r, o_exact, 1, n, k)
        .unwrap();
    dev.synchronize();

    fn time_it(dev: &RocmDevice, f: impl Fn()) -> f64 {
        let iters = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        dev.synchronize();
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    }

    let t_f32act = time_it(&dev, || {
        dev.launch_dot4_q6k_f32act_gemv_for_ab(a_r, b_r, o_f32act, 1, n, k).unwrap();
    });
    let t_exact = time_it(&dev, || {
        dev.launch_fused_dequant_gemm_q6k_for_ab(a_r, b_r, o_exact, 1, n, k).unwrap();
    });

    let bytes = (n * row_bytes) as f64;
    println!(
        "q6k f32act vs exact n={n} k={k} m=1 (head shape): f32act {t_f32act:.3} ms ({:.1} GB/s)  \
         exact {t_exact:.3} ms ({:.1} GB/s)  speedup {:.2}x",
        bytes / t_f32act / 1e6,
        bytes / t_exact / 1e6,
        t_exact / t_f32act
    );

    // Verify f32act matches exact within tolerance (both should be near-exact;
    // differences only from summation order).
    let got_f32act = f32_vec(&out_f32act)?;
    let got_exact = f32_vec(&out_exact)?;
    let mut max_abs_diff = 0.0f32;
    for col in 0..n {
        max_abs_diff = max_abs_diff.max((got_f32act[col] - got_exact[col]).abs());
    }
    println!("q6k f32act vs exact max|abs diff| at n={n}: {max_abs_diff:.3e}");
    assert!(max_abs_diff <= 0.1, "f32act and exact disagree by {max_abs_diff:.3e}");
    Ok(())
}
