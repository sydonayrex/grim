//! M7.2b: the Q5_K group-hoisting gate.
//!
//! The Q5_K dot4 decode kernel used to fetch the four 4-bit codes and the
//! four `qh` high bits of every 4-weight group with EIGHT separate 1-byte
//! loads. Q5_K spends 2 bytes per weight against Q4_K's 0.75, and the
//! measured decode gap tracked that ratio: 8.7 GB/s against 24.5 at the 27B's
//! dominant shape.
//!
//! The change widens those to two dword loads and hoists the extraction into
//! whole-word masks. Two independent claims are gated here:
//!
//!  1. BYTE-IDENTITY. The new kernel and the retained pre-change oracle
//!     (`grim_dot4_q5k_q81_gemv_ref`, the old body kept alive in the JIT
//!     source) must agree bit for bit. The oracle is the real pre-change
//!     kernel, not a hand-written model of it.
//!  2. CORRECTNESS. Both must agree with the llama.cpp-faithful
//!     `grim_quant::dequant_q5k` dotted against the same q8_1 activation
//!     codes — so byte-identity cannot be satisfied by two kernels that are
//!     identically wrong (e.g. both reading zeros).
//!
//! Run: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm
//!      --test dot4_q5k_grouped_parity -- --nocapture --test-threads=1

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

/// Decode the packed q8_1 activation layout: per 32-value block, an f16
/// scale, an f16 sum, then 32 i8 codes.
fn dequant_q81(packed: &[u8], len: usize) -> Vec<f32> {
    let f16 = |b: &[u8]| {
        let h = (b[0] as u16) | ((b[1] as u16) << 8);
        let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f32;
        s * if e == 0 {
            m / 1024.0 * 2f32.powi(-14)
        } else {
            (1.0 + m / 1024.0) * 2f32.powi(e - 15)
        }
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

const Q5K_ROW_BYTES_PER_256: usize = 176;

fn q5k_grouped_parity(k: usize, n: usize, m: usize) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    assert_eq!(k % 256, 0, "Q5_K needs k % 256 == 0");
    let row_bytes = (k / 256) * Q5K_ROW_BYTES_PER_256;

    let mut rng = Lcg(0x5A5A_1234 ^ (k as u64) ^ ((n as u64) << 32));
    let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();

    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    let a_st = MemoryOps::from_cpu_bytes(&dev, &f32_bytes(&a), &Shape::new(vec![m * k]), f32ty)
        .map_err(|e| format!("a h2d: {e}"))?;
    let q81_len = m * (k / 32) * 36;
    let q81_st = MemoryOps::from_cpu_bytes(
        &dev,
        &vec![0u8; q81_len],
        &Shape::new(vec![q81_len]),
        u8ty.clone(),
    )
    .map_err(|e| format!("q81 alloc: {e}"))?;
    dev.launch_quantize_q8_1(
        as_rocm(a_st.as_ref()).unwrap(),
        as_rocm(q81_st.as_ref()).unwrap(),
        m,
        k,
    )
    .map_err(|e| format!("quantize: {e}"))?;
    dev.synchronize();
    let q81 = bytes_of(&q81_st)?;

    let mut bq = vec![0u8; n * row_bytes];
    for col in 0..n {
        let packed = grim_quant::quant_q5k(&b[col * k..(col + 1) * k])
            .map_err(|e| format!("quant: {e}"))?;
        assert_eq!(packed.len(), row_bytes, "Q5_K row length");
        bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
    }
    let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
        .map_err(|e| format!("b h2d: {e}"))?;

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
    let out_new = mk_out()?;
    let out_ref = mk_out()?;

    let a_r = as_rocm(q81_st.as_ref()).unwrap();
    let b_r = as_rocm(b_st.as_ref()).unwrap();
    dev.launch_dot4_q5k_q81_gemv_for_ab(
        a_r,
        b_r,
        as_rocm(out_new.as_ref()).unwrap(),
        m,
        n,
        k,
    )
    .map_err(|e| format!("launch new: {e}"))?;
    dev.launch_dot4_q5k_q81_gemv_ref_for_ab(
        a_r,
        b_r,
        as_rocm(out_ref.as_ref()).unwrap(),
        m,
        n,
        k,
    )
    .map_err(|e| format!("launch ref: {e}"))?;
    dev.synchronize();

    let got_new = f32_vec(&out_new)?;
    let got_ref = f32_vec(&out_ref)?;

    // (1) BYTE-IDENTITY against the retained pre-change oracle.
    let mut differing = 0usize;
    let mut first_diff = None;
    for (i, (x, y)) in got_new.iter().zip(got_ref.iter()).enumerate() {
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
        "Q5_K grouped hoist is not byte-identical to the retained oracle at k={k} n={n} m={m}: \
         {differing}/{} elements differ, first at {:?}",
        got_new.len(),
        first_diff
    );

    // (2) CORRECTNESS of the (identical) pair against the llama.cpp-faithful
    // dequant dotted with the same activation codes. Without this, two
    // kernels that are identically wrong would satisfy (1).
    let a_deq = dequant_q81(&q81, m * k);
    let mut worst = 0.0f32;
    let mut nonzero = 0usize;
    for row in 0..m {
        for col in 0..n {
            let w = grim_quant::dequant_q5k(&bq[col * row_bytes..(col + 1) * row_bytes], k)
                .map_err(|e| format!("dequant: {e}"))?;
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_deq[row * k + kk] * w[kk];
            }
            let got = got_new[row * n + col];
            if got != 0.0 {
                nonzero += 1;
            }
            let d = (got - acc).abs() / acc.abs().max(1e-3);
            worst = worst.max(d);
        }
    }
    assert_eq!(
        nonzero, m * n,
        "Q5_K grouped kernel produced all-zero output at k={k} n={n} m={m} — the parity \
         comparison would have been vacuous"
    );
    println!(
        "q5k grouped k={k:<6} n={n:<5} m={m:<3} byte-identical to the retained oracle, \
         worst relative error {worst:.3e}"
    );
    assert!(
        worst <= 2e-2,
        "Q5_K grouped GEMV disagrees with the dequant oracle by {worst:.3e} on {}",
        dev.gpu_target_str()
    );
    Ok(())
}

#[test]
fn q5k_grouped_matches_retained_oracle() -> Result<(), String> {
    for (k, n, m) in [
        (256usize, 64usize, 1usize),
        (1024, 64, 1),
        (1024, 64, 4),
        (4096, 68, 1),
        (4096, 68, 8),
    ] {
        q5k_grouped_parity(k, n, m)?;
    }
    Ok(())
}
