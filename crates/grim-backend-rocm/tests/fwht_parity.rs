//! Task 3 gate: FWHT-1024 GPU parity against the Task-1 host oracle
//! (`grim_quant::hadamard::fwht_signed_f32`). The GPU kernel is a port of
//! PrismML fwht.cu (`fwht_cuda_smem` + `fwht_cuda_block`) for N=1024; both
//! the signed (signs tensor, n_blk modulo rows) and unsigned paths are
//! checked at decode (few rows) and prefill (many rows) shapes.

use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_quant::hadamard::fwht_signed_f32;
use grim_tensor::{CoreTensorOps, DType, Shape};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

#[test]
#[ignore]
fn fwht_1024_gpu_matches_host_oracle_signed_and_unsigned() -> TestResult {
    let Some(dev) = gpu_device() else { return Ok(()); };
    let n = 1024usize;
    let mut state = 0x243F6A8885A308D3u64;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32 * 4.0 - 2.0
    };
    // Deterministic ±1 signs of width 5120 — but the kernel transforms ONE
    // row of n at a time over `rows` rows; the reference signs tensor covers
    // n_blk * n elements and row r uses signs[(r % n_blk)*n ..]. Use n_blk=5
    // (mimicking width 5120 = 5 blocks of 1024, the Bonsai attn shape).
    let n_blk = 5usize;
    let signs: Vec<i32> = (0..n_blk * n)
        .map(|_| if rand() < 0.5 { -1 } else { 1 })
        .collect();
    let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();

    // 256 exercises the prefill (LDS smem) variant; 1/7/64 cover the decode
    // (block) variant at and below its rows<=64 selection threshold.
    for &rows in &[1usize, 7usize, 64usize, 256usize] {
        let src: Vec<f32> = (0..rows * n).map(|_| rand()).collect();

        for use_signs in [false, true] {
            let mut want = src.clone();
            if use_signs {
                // Oracle applies the same per-row modulo: row r uses block (r % n_blk).
                // The kernel's n_blk semantics live in fwht.cu:100
                // (`signs + (r % n_blk) * N`) — mirror it exactly.
                for r in 0..rows {
                    let off = (r % n_blk) * n;
                    let row = &mut want[r * n..(r + 1) * n];
                    for (i, v) in row.iter_mut().enumerate() {
                        *v *= signs[off + i] as f32;
                    }
                    fwht_signed_f32(row, &vec![1i32; n]); // signs already applied; now pure H
                }
            } else {
                for r in 0..rows {
                    fwht_signed_f32(&mut want[r * n..(r + 1) * n], &vec![1i32; n]);
                }
            }

            let src_dev = CoreTensorOps::from_cpu(&dev, &src, &Shape::new(vec![rows * n]), DType::F32)?;
            let dst_dev = CoreTensorOps::zeros(&dev, &Shape::new(vec![rows * n]), DType::F32)?;
            let signs_dev = grim_backend_rocm::MemoryOps::from_cpu_bytes(
                &dev,
                &signs_f32.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(),
                &Shape::new(vec![n_blk * n]),
                DType::F32,
            )?;
            let s = grim_backend_rocm::device::util::as_rocm(src_dev.as_ref())?;
            let d = grim_backend_rocm::device::util::as_rocm(dst_dev.as_ref())?;
            let sg = grim_backend_rocm::device::util::as_rocm(signs_dev.as_ref())?;
            grim_backend_rocm::device::hadamard_launch::launch_fwht_signed(
                &dev, s, d, if use_signs { Some(sg) } else { None }, n as u32, rows,
            )?;
            dev.synchronize();
            let got = dst_dev.to_cpu_vec_f32()?;
            let mut worst = 0.0f64;
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                let err = (g - w).abs();
                worst = worst.max(err as f64 / (w.abs().max(1.0)) as f64);
                assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "rows={rows} signs={use_signs} idx={i}: gpu={g} oracle={w}");
            }
            println!("rows={rows} signs={use_signs}: worst rel err {worst:.3e}");
        }
    }
    println!("fwht-1024 GPU parity ok (signed + unsigned, rows 1/7/64/256)");
    Ok(())
}
