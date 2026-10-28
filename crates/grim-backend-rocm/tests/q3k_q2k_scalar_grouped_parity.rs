//! M7.2g: the Q3_K/Q2_K scalar group-hoist parity gate.
//!
//! The Q3_K/Q2_K scalar fused-dequant GEMMs re-decode the super-block
//! header (d + all 16 scales, plus dmin for Q2_K) once per weight. The IQ
//! recipe hoists that decode once per super-block. These two kernels are
//! bit-identical: same per-weight values, same ascending accumulation order
//! over k.
//!
//! Two claims are gated here:
//!
//!  1. BYTE-IDENTITY. The grouped kernel and the retained pre-change scalar
//!     kernel (`grim_fused_dequant_gemm_q3k` / `grim_fused_dequant_gemm_q2k`,
//!     shipped and on-dispatch) must agree bit for bit at several shapes.
//!     The oracle is the real pre-change kernel, not a hand-written model.
//!  2. CORRECTNESS. Both must agree with the llama.cpp-faithful
//!     `grim_quant::dequant_*` dotted against the same f32 activation — so
//!     byte-identity cannot be satisfied by two identically wrong kernels.
//!
//! Run: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm
//!      --test q3k_q2k_scalar_grouped_parity -- --nocapture --test-threads=1

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

fn scheme(
    name: &str,
) -> (
    usize, // bytes per 256-weight super-block
    fn(&[f32]) -> grim_tensor::Result<Vec<u8>>,
    fn(&[u8], usize) -> grim_tensor::Result<Vec<f32>>,
) {
    match name {
        "q3k" => (110, grim_quant::quant_q3k, grim_quant::dequant_q3k),
        "q2k" => (84, grim_quant::quant_q2k, grim_quant::dequant_q2k),
        other => panic!("unknown scheme {other}"),
    }
}

/// k must be a multiple of 256; n is the number of output columns; m=1
/// decode. n can be non-divisible-by-4 to exercise the tail path.
fn grouped_parity(scheme_name: &str, k: usize, n: usize) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    assert_eq!(k % 256, 0, "K-quants need k % 256 == 0");
    let (blk_bytes, quant, dequant) = scheme(scheme_name);
    let row_bytes = (k / 256) * blk_bytes;

    let mut rng = Lcg(0x4B51_0003 ^ (k as u64) ^ ((n as u64) << 32));
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
        let packed = quant(&b[col * k..(col + 1) * k]).map_err(|e| format!("quant: {e}"))?;
        assert_eq!(packed.len(), row_bytes, "{scheme_name} row length");
        bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
    }
    let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
        .map_err(|e| format!("b h2d: {e}"))?;

    let mk_out = || -> Result<Box<dyn BackendStorage>, String> {
        MemoryOps::alloc_storage(
            &dev,
            &Shape::new(vec![n]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Native,
            },
        )
        .map_err(|e| format!("out alloc: {e}"))
    };
    let out_new = mk_out()?;
    let out_ret = mk_out()?;

    let a_r = as_rocm(a_st.as_ref()).unwrap();
    let b_r = as_rocm(b_st.as_ref()).unwrap();
    let o_new = as_rocm(out_new.as_ref()).unwrap();
    let o_ret = as_rocm(out_ret.as_ref()).unwrap();

    match scheme_name {
        "q3k" => {
            dev.launch_fused_dequant_gemm_q3k_for_ab(a_r, b_r, o_ret, 1, n, k)
                .map_err(|e| format!("launch retained: {e}"))?;
            dev.launch_fused_dequant_gemm_q3k_grouped_for_ab(a_r, b_r, o_new, 1, n, k)
                .map_err(|e| format!("launch grouped: {e}"))?;
        }
        "q2k" => {
            dev.launch_fused_dequant_gemm_q2k_for_ab(a_r, b_r, o_ret, 1, n, k)
                .map_err(|e| format!("launch retained: {e}"))?;
            dev.launch_fused_dequant_gemm_q2k_grouped_for_ab(a_r, b_r, o_new, 1, n, k)
                .map_err(|e| format!("launch grouped: {e}"))?;
        }
        _ => unreachable!(),
    }
    dev.synchronize();

    let got_new = f32_vec(&out_new)?;
    let got_ret = f32_vec(&out_ret)?;

    // (1) BYTE-IDENTITY against the retained pre-change scalar kernel.
    let mut differing = 0usize;
    let mut first_diff = None;
    for (i, (x, y)) in got_new.iter().zip(got_ret.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            differing += 1;
            if first_diff.is_none() {
                first_diff = Some((i, *x, *y));
            }
        }
    }
    assert_eq!(
        differing,
        0,
        "{scheme_name} grouped scalar kernel is not byte-identical to the retained scalar kernel \
         at k={k} n={n}: {differing}/{} elements differ, first at {:?}",
        got_new.len(),
        first_diff
    );

    // (2) CORRECTNESS against the llama.cpp-faithful dequant dotted with the
    // same f32 activation.
    let mut worst = 0.0f32;
    let mut nonzero = 0usize;
    for col in 0..n {
        let w = dequant(&bq[col * row_bytes..(col + 1) * row_bytes], k)
            .map_err(|e| format!("dequant: {e}"))?;
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += a[kk] * w[kk];
        }
        let got = got_new[col];
        if got != 0.0 {
            nonzero += 1;
        }
        let d = (got - acc).abs() / acc.abs().max(1e-3);
        worst = worst.max(d);
    }
    assert_eq!(
        nonzero, n,
        "{scheme_name} grouped scalar kernel produced all-zero output at k={k} n={n} — the parity \
         comparison would have been vacuous"
    );
    println!(
        "{scheme_name} grouped scalar k={k:<6} n={n:<6} byte-identical to retained scalar, \
         worst relative error {worst:.3e}"
    );
    assert!(
        worst <= 2e-2,
        "{scheme_name} grouped scalar GEMV disagrees with the dequant oracle by {worst:.3e} on {}",
        dev.gpu_target_str()
    );
    Ok(())
}

#[test]
fn q3k_q2k_grouped_scalar_matches_retained() -> Result<(), String> {
    // Powers-of-two and a non-divisible-by-4 n to exercise tail paths.
    for (k, n) in [(256usize, 64usize), (1024, 67), (4096, 129), (5120, 2000)] {
        for s in ["q3k", "q2k"] {
            grouped_parity(s, k, n)?;
        }
    }
    Ok(())
}

#[test]
fn q3k_q2k_grouped_scalar_speed_ab() -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    // The Q3_K/Q2_K prompt path at the 27B's shape: m=1 decode, n=17408,
    // k=5120.
    let k = 5120usize;
    let n = 17408usize;
    let time_it = |new: bool, scheme_name: &str| -> Result<f64, String> {
        let (blk_bytes, quant, _dequant) = scheme(scheme_name);
        let row_bytes = (k / 256) * blk_bytes;

        let mut rng = Lcg(0xBEEF_0003 ^ (scheme_name.as_bytes()[0] as u64));
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
            let packed = quant(&b[col * k..(col + 1) * k]).map_err(|e| format!("quant: {e}"))?;
            bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
        }
        let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
            .map_err(|e| format!("b h2d: {e}"))?;
        let mk_out = || -> Result<Box<dyn BackendStorage>, String> {
            MemoryOps::alloc_storage(
                &dev,
                &Shape::new(vec![n]),
                DType {
                    arith: ArithType::F32,
                    storage: Storage::Native,
                },
            )
            .map_err(|e| format!("out alloc: {e}"))
        };
        let out_new = mk_out()?;
        let out_ret = mk_out()?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_r = as_rocm(b_st.as_ref()).unwrap();
        let o_new = as_rocm(out_new.as_ref()).unwrap();
        let o_ret = as_rocm(out_ret.as_ref()).unwrap();

        for _ in 0..2 {
            match scheme_name {
                "q3k" => {
                    if new {
                        dev.launch_fused_dequant_gemm_q3k_grouped_for_ab(a_r, b_r, o_new, 1, n, k)
                            .unwrap();
                    } else {
                        dev.launch_fused_dequant_gemm_q3k_for_ab(a_r, b_r, o_ret, 1, n, k)
                            .unwrap();
                    }
                }
                "q2k" => {
                    if new {
                        dev.launch_fused_dequant_gemm_q2k_grouped_for_ab(a_r, b_r, o_new, 1, n, k)
                            .unwrap();
                    } else {
                        dev.launch_fused_dequant_gemm_q2k_for_ab(a_r, b_r, o_ret, 1, n, k)
                            .unwrap();
                    }
                }
                _ => unreachable!(),
            }
        }
        dev.synchronize();
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            match scheme_name {
                "q3k" => {
                    if new {
                        dev.launch_fused_dequant_gemm_q3k_grouped_for_ab(a_r, b_r, o_new, 1, n, k)
                            .unwrap();
                    } else {
                        dev.launch_fused_dequant_gemm_q3k_for_ab(a_r, b_r, o_ret, 1, n, k)
                            .unwrap();
                    }
                }
                "q2k" => {
                    if new {
                        dev.launch_fused_dequant_gemm_q2k_grouped_for_ab(a_r, b_r, o_new, 1, n, k)
                            .unwrap();
                    } else {
                        dev.launch_fused_dequant_gemm_q2k_for_ab(a_r, b_r, o_ret, 1, n, k)
                            .unwrap();
                    }
                }
                _ => unreachable!(),
            }
        }
        dev.synchronize();
        Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
    };

    for scheme_name in ["q3k", "q2k"] {
        let t_new = time_it(true, scheme_name)?;
        let t_ret = time_it(false, scheme_name)?;
        let blk_bytes = scheme(scheme_name).0;
        let row_bytes = (k / 256) * blk_bytes;
        let bytes = (n * row_bytes) as f64;
        println!(
            "{scheme_name} grouped scalar n={n} k={k} m=1: grouped {t_new:.3} ms ({:.1} GB/s)  \
             retained {t_ret:.3} ms ({:.1} GB/s)  speedup {:.2}x",
            bytes / t_new / 1e6,
            bytes / t_ret / 1e6,
            t_ret / t_new
        );
    }
    Ok(())
}
