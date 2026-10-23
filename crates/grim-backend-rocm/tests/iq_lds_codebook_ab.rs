//! res0-T1: the IQ codebook-cache LDS gate + A/B (VQ-LLM / SPHQuant levers).
//!
//! The `_grouped_lds` variants stage each format's hot tables (IQ3XXS grid +
//! signs, IQ3S grid, IQ2S grid) in __shared__ once per block. The math, the
//! index expressions and the accumulation order are identical to the
//! production `_grouped` kernels, so the pair must be BYTE-IDENTICAL — that
//! is gate (1). Gate (2) is the A/B: production grouped vs LDS, at the
//! census shape (n=4096, k=5120, m=1) and the 27B expert shape (n=17408).
//!
//! Run: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm
//!      --test iq_lds_codebook_ab -- --nocapture --test-threads=1

use grim_backend_rocm::as_rocm;
use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, BackendStorage, DType, KQuantScheme, MemoryOps, Shape, Storage};
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

fn run_case(
    scheme_name: &str,
    scheme: KQuantScheme,
    quant: fn(&[f32]) -> grim_tensor::Result<Vec<u8>>,
    n: usize,
) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let m = 1usize;

    let mut rng = Lcg(0xC0DE ^ (n as u64));
    let vals: Vec<f32> = (0..n * K).map(|_| rng.next_f32()).collect();
    let packed = quant(&vals).map_err(|e| format!("quant: {e}"))?;
    let row_bytes = packed.len() / n;

    let a: Vec<f32> = (0..K).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let iq_ty = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(scheme),
    };
    let a_st = MemoryOps::from_cpu_bytes(&dev, &f32_bytes(&a), &Shape::new(vec![m * K]), f32ty)
        .map_err(|e| format!("a h2d: {e}"))?;
    let w_st = MemoryOps::from_cpu_bytes(&dev, &packed, &Shape::new(vec![n, K]), iq_ty)
        .map_err(|e| format!("w h2d: {e}"))?;

    let mk_out = || -> Result<Box<dyn BackendStorage>, String> {
        MemoryOps::alloc_storage(
            &dev,
            &Shape::new(vec![m * n]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Native,
            },
        )
        .map_err(|e| format!("out alloc: {e}"))
    };
    let out_stock = mk_out()?;
    let out_lds = mk_out()?;

    let a_r = as_rocm(a_st.as_ref()).unwrap();
    let w_r = as_rocm(w_st.as_ref()).unwrap();
    let o_stock = as_rocm(out_stock.as_ref()).unwrap();
    let o_lds = as_rocm(out_lds.as_ref()).unwrap();

    // Gate (1): byte-identity, both sides through launchers.
    match scheme_name {
        "iq3xxs" => {
            dev.launch_fused_dequant_gemm_iq3xxs_stock_for_ab(a_r, w_r, o_stock, m, n, K)
                .map_err(|e| format!("stock: {e}"))?;
            dev.launch_fused_dequant_gemm_iq3xxs_lds_for_ab(a_r, w_r, o_lds, m, n, K)
                .map_err(|e| format!("lds: {e}"))?;
        }
        "iq3s" => {
            dev.launch_fused_dequant_gemm_iq3s_stock_for_ab(a_r, w_r, o_stock, m, n, K)
                .map_err(|e| format!("stock: {e}"))?;
            dev.launch_fused_dequant_gemm_iq3s_lds_for_ab(a_r, w_r, o_lds, m, n, K)
                .map_err(|e| format!("lds: {e}"))?;
        }
        _ => {
            dev.launch_fused_dequant_gemm_iq2s_stock_for_ab(a_r, w_r, o_stock, m, n, K)
                .map_err(|e| format!("stock: {e}"))?;
            dev.launch_fused_dequant_gemm_iq2s_lds_for_ab(a_r, w_r, o_lds, m, n, K)
                .map_err(|e| format!("lds: {e}"))?;
        }
    }
    dev.synchronize();

    let got_stock = f32_vec(&out_stock)?;
    let got_lds = f32_vec(&out_lds)?;
    let mut differing = 0usize;
    let mut first = None;
    for (i, (x, y)) in got_stock.iter().zip(got_lds.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            differing += 1;
            if first.is_none() {
                first = Some((i, *x, *y));
            }
        }
    }
    assert_eq!(
        differing, 0,
        "{scheme_name} LDS variant is not byte-identical to the production grouped kernel at \
         n={n}: {differing}/{} differ, first {first:?}",
        got_stock.len(),
        first = first
    );

    // Gate (2): A/B. 5 warmup + 50 timed launches each, fresh output each.
    let time_it = |lds: bool| -> f64 {
        let f = |out: &Box<dyn BackendStorage>| {
            let o = as_rocm(out.as_ref()).unwrap();
            match scheme_name {
                "iq3xxs" => {
                    if lds {
                        dev.launch_fused_dequant_gemm_iq3xxs_lds_for_ab(a_r, w_r, o, m, n, K)
                    } else {
                        dev.launch_fused_dequant_gemm_iq3xxs_stock_for_ab(a_r, w_r, o, m, n, K)
                    }
                }
                "iq3s" => {
                    if lds {
                        dev.launch_fused_dequant_gemm_iq3s_lds_for_ab(a_r, w_r, o, m, n, K)
                    } else {
                        dev.launch_fused_dequant_gemm_iq3s_stock_for_ab(a_r, w_r, o, m, n, K)
                    }
                }
                _ => {
                    if lds {
                        dev.launch_fused_dequant_gemm_iq2s_lds_for_ab(a_r, w_r, o, m, n, K)
                    } else {
                        dev.launch_fused_dequant_gemm_iq2s_stock_for_ab(a_r, w_r, o, m, n, K)
                    }
                }
            }
        };
        let out = mk_out().unwrap();
        for _ in 0..5 {
            f(&out).unwrap();
        }
        dev.synchronize();
        let t0 = std::time::Instant::now();
        for _ in 0..50 {
            f(&out).unwrap();
        }
        dev.synchronize();
        t0.elapsed().as_secs_f64() * 1e6 / 50.0
    };
    let t_stock = time_it(false);
    let t_lds = time_it(true);
    let bytes = (n * row_bytes) as f64;
    println!(
        "{scheme_name} LDS-codebook n={n} k={K} m=1: stock {t_stock:.1} us ({:.1} GB/s)  lds \
         {t_lds:.1} us ({:.1} GB/s)  speedup {:.2}x  [byte-identical]",
        bytes / t_stock / 1e3,
        bytes / t_lds / 1e3,
        t_stock / t_lds
    );
    Ok(())
}

#[test]
fn iq_lds_codebook_ab() -> Result<(), String> {
    for n in [4096usize, 17408usize] {
        run_case(
            "iq3xxs",
            KQuantScheme::IQ3XXS,
            grim_quant::quant_iq3xxs,
            n,
        )?;
        run_case("iq3s", KQuantScheme::IQ3S, grim_quant::quant_iq3s, n)?;
        run_case("iq2s", KQuantScheme::IQ2S, grim_quant::quant_iq2s, n)?;
    }
    Ok(())
}
