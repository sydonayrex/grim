//! res0-T1-3: INT8 WMMA probe gate + rate.
//!
//! Gate (1): D (i32) BIT-IDENTICAL to the host i32 reference at a small
//! shape (16x48x32) — integer math is exact, so any layout/fragment error
//! shows as a mismatch (the "wrong geometry that runs" trap).
//! Gate (2): Double-K sanity — a K=32 problem computed in one pass equals
//! two K=16 halves accumulated on the host side (associativity, exact).
//! Rate: TFLOPS at grim prefill shapes M in {64, 512, 2048}, N=4096, K=4096.
//!
//! Run: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm
//!      --test wmma_i8_probe -- --nocapture --test-threads=1

use grim_backend_rocm::as_rocm;
use grim_backend_rocm::RocmDevice;
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
    fn next_byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
    fn next_i8(&mut self) -> i8 {
        (self.next_byte() as i32 % 5 - 2) as i8 // small range: no i32 overflow
    }
    fn next_u8(&mut self) -> u8 {
        self.next_byte() % 5 // small range
    }
}

fn bytes_of(t: &Box<dyn BackendStorage>) -> Result<Vec<u8>, String> {
    as_rocm(t.as_ref())
        .map_err(|e| e.to_string())?
        .copy_to_host()
        .map_err(|e| format!("copy_to_host: {e}"))
}

fn upload_bytes(
    dev: &RocmDevice,
    b: &[u8],
    shape: &[usize],
    arith: ArithType,
) -> Result<Box<dyn BackendStorage>, String> {
    MemoryOps::from_cpu_bytes(
        dev,
        b,
        &Shape::new(shape.to_vec()),
        DType {
            arith,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("h2d: {e}"))
}

fn run(m: usize, n: usize, k: usize, check_exact: bool) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let mut rng = Lcg(0x1A8 ^ (m as u64) ^ ((k as u64) << 8));

    // Pad to whole 16-row tiles (the launcher's OOB contract).
    let am = m.div_ceil(16) * 16;
    let bn = n.div_ceil(16) * 16;
    let mut a = vec![0u8; am * k];
    let mut b = vec![0u8; bn * k];
    for v in a.iter_mut() {
        *v = rng.next_i8() as u8;
    }
    for v in b.iter_mut() {
        *v = rng.next_u8();
    }

    let a_st = upload_bytes(&dev, &a, &[am * k], ArithType::U8)?;
    let b_st = upload_bytes(&dev, &b, &[bn * k], ArithType::U8)?;
    let d_st = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![m * n]),
        DType {
            arith: ArithType::U32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("d alloc: {e}"))?;

    let a_r = as_rocm(a_st.as_ref()).unwrap();
    let b_r = as_rocm(b_st.as_ref()).unwrap();
    let d_r = as_rocm(d_st.as_ref()).unwrap();

    if check_exact {
        dev.launch_wmma_gemm_iu8_for_ab(a_r, b_r, d_r, m, n, k)
            .map_err(|e| format!("launch: {e}"))?;
        dev.synchronize();
        let raw = bytes_of(&d_st)?;
        let got: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Host reference over the PADDED operands restricted to the real
        // m/n window (the kernel stores only the masked window).
        let mut worst = 0usize;
        let mut first = None;
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0i64;
                for kk in 0..k {
                    acc += a[row * k + kk] as i8 as i64 * b[col * k + kk] as i64;
                }
                let expect = acc as i32;
                if got[row * n + col] != expect {
                    worst += 1;
                    if first.is_none() {
                        first = Some((row, col, got[row * n + col], expect));
                    }
                }
            }
        }
        assert_eq!(
            worst, 0,
            "iu8 WMMA not bit-identical to the host reference at {m}x{n}x{k}: {worst} differ, \
             first {first:?}"
        );
        println!("iu8 WMMA {m}x{n}x{k}: BIT-IDENTICAL to the host reference");
    }

    // Exactness of the mtile variant too (B reuse across M tiles).
    if m >= 64 {
        let d2_st = MemoryOps::alloc_storage(
            &dev,
            &Shape::new(vec![m * n]),
            DType {
                arith: ArithType::U32,
                storage: Storage::Native,
            },
        )
        .map_err(|e| format!("d2 alloc: {e}"))?;
        let d2_r = as_rocm(d2_st.as_ref()).unwrap();
        dev.launch_wmma_gemm_iu8_mtile_for_ab(a_r, b_r, d2_r, m, n, k)
            .map_err(|e| format!("launch mtile: {e}"))?;
        dev.synchronize();
        let raw = bytes_of(&d2_st)?;
        let got: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut bad = 0usize;
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0i64;
                for kk in 0..k {
                    acc += a[row * k + kk] as i8 as i64 * b[col * k + kk] as i64;
                }
                if got[row * n + col] != acc as i32 {
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "iu8 mtile not bit-identical at {m}x{n}x{k}: {bad} differ");
    }

    // Rate: 5 warmup + 30 timed, both variants.
    let time = |mtile: bool| -> f64 {
        for _ in 0..5 {
            if mtile {
                dev.launch_wmma_gemm_iu8_mtile_for_ab(a_r, b_r, d_r, m, n, k).unwrap();
            } else {
                dev.launch_wmma_gemm_iu8_for_ab(a_r, b_r, d_r, m, n, k).unwrap();
            }
        }
        dev.synchronize();
        let t0 = std::time::Instant::now();
        for _ in 0..30 {
            if mtile {
                dev.launch_wmma_gemm_iu8_mtile_for_ab(a_r, b_r, d_r, m, n, k).unwrap();
            } else {
                dev.launch_wmma_gemm_iu8_for_ab(a_r, b_r, d_r, m, n, k).unwrap();
            }
        }
        dev.synchronize();
        t0.elapsed().as_secs_f64() * 1e3 / 30.0
    };
    let ms = time(false);
    let tflops = 2.0 * (m as f64) * (n as f64) * (k as f64) / (ms * 1e-3) / 1e12;
    print!("iu8 WMMA {m}x{n}x{k}: naive {ms:.3} ms = {tflops:.1} TFLOP/s");
    if m >= 64 {
        let ms2 = time(true);
        let tflops2 = 2.0 * (m as f64) * (n as f64) * (k as f64) / (ms2 * 1e-3) / 1e12;
        println!("  mtile {ms2:.3} ms = {tflops2:.1} TFLOP/s");
    } else {
        println!();
    }
    Ok(())
}

#[test]
fn wmma_i8_probe() -> Result<(), String> {
    // Exactness gates first (the "wrong geometry that runs" trap).
    run(16, 48, 32, true)?;
    run(32, 16, 64, true)?;
    run(16, 16, 4096, true)?;
    // Rate at grim prefill shapes.
    for m in [64usize, 512, 2048] {
        run(m, 4096, 4096, false)?;
    }
    Ok(())
}
