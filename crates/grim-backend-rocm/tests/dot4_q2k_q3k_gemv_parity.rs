//! Parity gates for the Q2_K and Q3_K dot4 GEMV decode arms
//! (`grim_dot4_q2k_q81_gemv`, `grim_dot4_q3k_q81_gemv`).
//!
//! The dispatch gates linear_decode's K-quant route to `is_dot4_arch` with
//! `k % 256 == 0`, so on RDNA3/4 these two GEMVs ARE the decode path for
//! Q2_K/Q3_K weights when GRIM_DECODE_W4A4 declines or is off. The q3k
//! launcher's own comment admits its parity claim was never settled on this
//! hardware; this file settles both.
//!
//! Oracle: dequantize the SAME packed bytes with `grim_quant::dequant_q2k` /
//! `dequant_q3k` (llama.cpp-faithful) and dot them with the SAME q8_1
//! activation codes the kernel consumed. Tolerance 2e-2 relative, the same
//! bar `dot4_q4k_arch_probe` applies (int8 dot4 rounding on Q2_K's 2-bit
//! weights is coarse by design).

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
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

/// Decode the packed q8_1 activation layout: per 32-value block, an f16
/// scale, an f16 sum, then 32 i8 codes.
fn dequant_q81(packed: &[u8], len: usize) -> Vec<f32> {
    let f16 = |b: &[u8]| {
        let h = (b[0] as u16) | ((b[1] as u16) << 8);
        let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f32;
        s * if e == 0 { m / 1024.0 * 2f32.powi(-14) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
    };
    let mut out = vec![0f32; len];
    for (bi, chunk) in out.chunks_mut(32).enumerate() {
        let base = bi * 36;
        let scale = f16(&packed[base..base + 2]);
        for (i, o) in chunk.iter_mut().enumerate() {
            *o = scale * (packed[base + 4 + i] as i8 as f32);
        }
    }
    out
}

fn dot4_kquant_gemv_parity(scheme: &str, k: usize) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let (quant, dequant): (
        fn(&[f32]) -> grim_tensor::Result<Vec<u8>>,
        fn(&[u8], usize) -> grim_tensor::Result<Vec<f32>>,
    ) = match scheme {
        "q2k" => (grim_quant::quant_q2k, grim_quant::dequant_q2k),
        "q3k" => (grim_quant::quant_q3k, grim_quant::dequant_q3k),
        other => return Err(format!("unknown scheme {other}")),
    };
    let row_bytes = match scheme {
        "q2k" => (k / 256) * 84,
        _ => (k / 256) * 110,
    };

    const M: usize = 1;
    let mut rng = Lcg(if scheme == "q2k" { 0xBEEF } else { 0xCAFE });
    let n = 64usize;
    let a: Vec<f32> = (0..M * k).map(|_| rng.next_f32()).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();

    let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
    let u8ty = DType { arith: ArithType::U8, storage: Storage::Native };
    let a_st = MemoryOps::from_cpu_bytes(&dev, &f32_bytes(&a), &Shape::new(vec![M * k]), f32ty)
        .map_err(|e| format!("a h2d: {e}"))?;
    let q81_len = M * (k / 32) * 36;
    let q81_st = MemoryOps::from_cpu_bytes(
        &dev,
        &vec![0u8; q81_len],
        &Shape::new(vec![q81_len]),
        u8ty.clone(),
    )
    .map_err(|e| format!("q81 alloc: {e}"))?;
    dev.launch_quantize_q8_1(
        grim_backend_rocm::as_rocm(a_st.as_ref()).unwrap(),
        grim_backend_rocm::as_rocm(q81_st.as_ref()).unwrap(),
        M,
        k,
    )
    .map_err(|e| format!("quantize: {e}"))?;
    dev.synchronize();
    let q81 = bytes_of(&q81_st)?;

    let mut bq = vec![0u8; n * row_bytes];
    for col in 0..n {
        let packed = quant(&b[col * k..(col + 1) * k]).map_err(|e| format!("quant: {e}"))?;
        assert_eq!(packed.len(), row_bytes, "{scheme} row length");
        bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
    }
    let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
        .map_err(|e| format!("b h2d: {e}"))?;
    let out_st = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![M * n]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out alloc: {e}"))?;

    let a_r = grim_backend_rocm::as_rocm(q81_st.as_ref()).unwrap();
    let b_r = grim_backend_rocm::as_rocm(b_st.as_ref()).unwrap();
    let o_r = grim_backend_rocm::as_rocm(out_st.as_ref()).unwrap();
    match scheme {
        "q2k" => dev
            .launch_dot4_q2k_q81_gemv_for_ab(a_r, b_r, o_r, M, n, k)
            .map_err(|e| format!("launch: {e}"))?,
        _ => dev
            .launch_dot4_q3k_q81_gemv_for_ab(a_r, b_r, o_r, M, n, k)
            .map_err(|e| format!("launch: {e}"))?,
    };
    dev.synchronize();
    let got = f32_vec(&out_st)?;

    let a_deq = dequant_q81(&q81, M * k);
    let mut worst = 0.0f32;
    for col in 0..n {
        let w = dequant(&bq[col * row_bytes..(col + 1) * row_bytes], k)
            .map_err(|e| format!("dequant: {e}"))?;
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += a_deq[kk] * w[kk];
        }
        let d = (got[col] - acc).abs();
        worst = worst.max(d / acc.abs().max(1e-3));
    }
    println!("{scheme} k={k:<6} worst relative error {worst:.3e}");
    assert!(
        worst <= 2e-2,
        "{scheme} k={k}: dot4 GEMV disagrees with the oracle by {worst:.3e} on {}",
        dev.gpu_target_str()
    );
    Ok(())
}

#[test]
fn dot4_q2k_gemv_parity() -> Result<(), String> {
    for k in &[256usize, 1024, 4096] {
        dot4_kquant_gemv_parity("q2k", *k)?;
    }
    Ok(())
}

#[test]
fn dot4_q3k_gemv_parity() -> Result<(), String> {
    for k in &[256usize, 1024, 4096] {
        dot4_kquant_gemv_parity("q3k", *k)?;
    }
    Ok(())
}
