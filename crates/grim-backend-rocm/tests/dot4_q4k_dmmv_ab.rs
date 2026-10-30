//! res0-T1-4: the Q4_K DMMV gate + A/B (zolotukhin's chunked-prefill lever).
//!
//! Gate: `grim_dot4_q4k_q81_gemv_dmmv` at M=8 must be BYTE-IDENTICAL to 8
//! single-row `grim_dot4_q4k_q81_gemv_fast` launches — row r's accumulation
//! order is unchanged; only the weight-read count differs (1x vs 8x).
//! A/B: one dmmv launch vs 8 fast launches at n=4096, k=5120 (chunked
//! prefill chunk size 8), plus M in {2, 4}.
//!
//! Run: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm
//!      --test dot4_q4k_dmmv_ab -- --nocapture --test-threads=1

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

fn f32_vec(t: &Box<dyn BackendStorage>) -> Result<Vec<f32>, String> {
    Ok(as_rocm(t.as_ref())
        .map_err(|e| e.to_string())?
        .copy_to_host()
        .map_err(|e| format!("copy_to_host: {e}"))?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

const K: usize = 5120;

fn run_case(m: usize, n: usize, check_gate: bool) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let mut rng = Lcg(0xD000 ^ (m as u64) ^ ((n as u64) << 8));

    // Quantize M activation rows and N weight rows to Q8_1 / Q4_K.
    let q81_len = m * (K / 32) * 36;
    let mut a_q81 = vec![0u8; q81_len];
    {
        // Host q8_1 quantization (same layout the device quantize kernel
        // produces: f16 d, f16 sum, 32 i8 codes per 32-block).
        let a: Vec<f32> = (0..m * K).map(|_| rng.next_f32()).collect();
        for row in 0..m {
            for blk in 0..(K / 32) {
                let base = row * (K / 32) * 36 + blk * 36;
                let mut max_abs = 0f32;
                for j in 0..32 {
                    max_abs = max_abs.max(a[row * K + blk * 32 + j].abs());
                }
                let d = (max_abs / 127.0).max(1e-30);
                let bits = half::f16::from_f32(d).to_bits();
                a_q81[base] = (bits & 0xFF) as u8;
                a_q81[base + 1] = ((bits >> 8) & 0xFF) as u8;
                let mut sum = 0i32;
                for j in 0..32 {
                    let code =
                        ((a[row * K + blk * 32 + j] / d).round().clamp(-127.0, 127.0) as i8) as u8;
                    a_q81[base + 4 + j] = code;
                    sum += code as i8 as i32;
                }
                let sbits = half::f16::from_f32(sum as f32).to_bits();
                a_q81[base + 2] = (sbits & 0xFF) as u8;
                a_q81[base + 3] = ((sbits >> 8) & 0xFF) as u8;
            }
        }
    }
    let mut bq = vec![0u8; n * (K / 256) * 144];
    for col in 0..n {
        let w: Vec<f32> = (0..K).map(|_| rng.next_f32()).collect();
        let packed = grim_quant::quant_q4k(&w).map_err(|e| format!("quant: {e}"))?;
        bq[col * (K / 256) * 144..(col + 1) * (K / 256) * 144].copy_from_slice(&packed);
    }

    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    // Discriminating experiment: zero row 1's codes (its true answer must be
    // exactly 0.0) when gating — shows whether dmmv's output rows are
    // shifted or its accumulation skipped r=0.
    if check_gate && m >= 2 {
        let row_bytes = (K / 32) * 36;
        for b in a_q81[row_bytes..2 * row_bytes].iter_mut() {
            *b = 0;
        }
    }
    let a_st = MemoryOps::from_cpu_bytes(&dev, &a_q81, &Shape::new(vec![q81_len]), u8ty.clone())
        .map_err(|e| format!("a h2d: {e}"))?;
    // Per-row buffers: the single-row reference launches must each read their
    // OWN activation row (a launcher takes a storage, not an offset).
    let mut a_row_sts: Vec<Box<dyn BackendStorage>> = Vec::new();
    for row in 0..m {
        let row_bytes = (K / 32) * 36;
        let st = MemoryOps::from_cpu_bytes(
            &dev,
            &a_q81[row * row_bytes..(row + 1) * row_bytes],
            &Shape::new(vec![row_bytes]),
            u8ty.clone(),
        )
        .map_err(|e| format!("a row h2d: {e}"))?;
        a_row_sts.push(st);
    }
    let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
        .map_err(|e| format!("b h2d: {e}"))?;
    let mk_out = move |dev: &RocmDevice| -> Result<Box<dyn BackendStorage>, String> {
        MemoryOps::alloc_storage(
            dev,
            &Shape::new(vec![m * n]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Native,
            },
        )
        .map_err(|e| format!("out alloc: {e}"))
    };

    let a_r = as_rocm(a_st.as_ref()).unwrap();
    let b_r = as_rocm(b_st.as_ref()).unwrap();

    if check_gate {
        // M single-row fast launches, EACH with its own n-length output (a
        // m=1 launch writes C[0..n); sharing one (m*n) storage made every
        // launch overwrite row 0).
        let mut row_outs: Vec<Box<dyn BackendStorage>> = Vec::new();
        for ast in a_row_sts.iter() {
            let arow = as_rocm(ast.as_ref()).unwrap();
            let out_r = MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![n]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )
            .map_err(|e| format!("row out alloc: {e}"))?;
            let or = as_rocm(out_r.as_ref()).unwrap();
            dev.launch_dot4_q4k_q81_gemv_for_ab(arow, b_r, or, 1, n, K)
                .map_err(|e| format!("single-row launch: {e}"))?;
            row_outs.push(out_r);
        }
        dev.synchronize();
        // ... vs ONE dmmv launch.
        let out_dmmv = mk_out(&dev)?;
        let o_dmmv = as_rocm(out_dmmv.as_ref()).unwrap();
        let got_rows: Vec<f32> = {
            let mut v = Vec::new();
            for out_r in row_outs.iter() {
                v.extend(f32_vec(out_r)?);
            }
            v
        };
        dev.launch_dot4_q4k_q81_gemv_dmmv_for_ab(a_r, b_r, o_dmmv, m, n, K)
            .map_err(|e| format!("dmmv launch: {e}"))?;
        dev.synchronize();

        let got_dmmv = f32_vec(&out_dmmv)?;
        let mut differing = 0usize;
        let mut first = None;
        let mut per_row = vec![0usize; m];
        for (i, (x, y)) in got_rows.iter().zip(got_dmmv.iter()).enumerate() {
            if x.to_bits() != y.to_bits() {
                differing += 1;
                per_row[i / n] += 1;
                if first.is_none() {
                    first = Some((i, *x, *y));
                }
            }
        }
        if m >= 2 {
            eprintln!(
                "[dmmv-gate] per-row diffs: {per_row:?}  rows[0][0]={} rows[1][0]={} dmmv[0][0]={} dmmv[1][0]={}",
                got_rows[0], got_rows[n], got_dmmv[0], got_dmmv[n]
            );
        } else {
            eprintln!(
                "[dmmv-gate] per-row diffs: {per_row:?}  dmmv[0][0]={}",
                got_dmmv[0]
            );
        }
        for r in 0..m {
            for rr in 0..m {
                let same =
                    (0..4).all(|c| got_rows[r * n + c].to_bits() == got_dmmv[rr * n + c].to_bits());
                if same {
                    eprintln!("[dmmv-gate] dmmv row {rr} matches single-row row {r}");
                }
            }
        }
        assert_eq!(
            differing,
            0,
            "q4k DMMV is not byte-identical to the single-row launches at M={m} n={n}: \
             {differing}/{} differ, first {first:?}",
            got_rows.len(),
            first = first
        );
        println!(
            "q4k dmmv M={m} n={n}: BYTE-IDENTICAL to {} single-row launches",
            m
        );
    }

    // A/B: dmmv(1 launch) vs M fast launches.
    let time = |dmmv: bool| -> f64 {
        let out = mk_out(&dev).unwrap();
        let o = as_rocm(out.as_ref()).unwrap();
        let f = || {
            if dmmv {
                dev.launch_dot4_q4k_q81_gemv_dmmv_for_ab(a_r, b_r, o, m, n, K)
            } else {
                for ast in a_row_sts.iter() {
                    let arow = as_rocm(ast.as_ref()).unwrap();
                    dev.launch_dot4_q4k_q81_gemv_for_ab(arow, b_r, o, 1, n, K)?;
                }
                Ok(0 as *mut std::ffi::c_void)
            }
        };
        for _ in 0..3 {
            f().unwrap();
        }
        dev.synchronize();
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f().unwrap();
        }
        dev.synchronize();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    let t_rows = time(false);
    let t_dmmv = time(true);
    let bytes = (n * (K / 256) * 144) as f64;
    println!(
        "q4k dmmv M={m} n={n} k={K}: {m}x-single {t_rows:.1} us (weight read {m}x, {:.0} GB/s)  \
         dmmv {t_dmmv:.1} us (weight read 1x, {:.0} GB/s)  speedup {:.2}x",
        bytes * m as f64 / t_rows / 1e3,
        bytes / t_dmmv / 1e3,
        t_rows / t_dmmv
    );
    Ok(())
}

#[test]
fn dot4_q4k_dmmv_parity_and_ab() -> Result<(), String> {
    for (m, gate) in [(1usize, true), (2, true), (4, true), (8, true)] {
        run_case(m, 4096, gate)?;
    }
    run_case(8, 17408, false)?;
    Ok(())
}
