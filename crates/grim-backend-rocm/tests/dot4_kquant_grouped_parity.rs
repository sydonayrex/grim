//! M7.2c: the K-quant group-hoisting gates (Q4_K, Q5_K, Q6_K).
//!
//! The dot4 decode GEMVs for the K-quants fetched each 4-weight group's
//! codes with one 1-byte load PER WEIGHT. The byte ratio tracks the measured
//! gap exactly: Q5_K spends 2 bytes per weight against Q4_K's 0.75, and
//! Q5_K ran at 8.7 GB/s where Q4_K ran at 24.5.
//!
//! The hoist replaces those with the fewest wide loads the block layout
//! allows (Q4_K: one dword; Q5_K: one dword per field; Q6_K: two ushorts
//! per field, because its 210-byte super-block stride only guarantees
//! 2-byte alignment).
//!
//! Two independent claims are gated here:
//!
//!  1. BYTE-IDENTITY. The hoisted kernel and the retained pre-change oracle
//!     (`grim_dot4_*_q81_gemv_ref` — the old body kept alive in the JIT
//!     source) must agree bit for bit. The oracle is the real pre-change
//!     kernel, not a hand-written model of it.
//!  2. CORRECTNESS. Both must agree with the llama.cpp-faithful
//!     `grim_quant::dequant_*` dotted against the same q8_1 activation
//!     codes — so byte-identity cannot be satisfied by two kernels that are
//!     identically wrong (e.g. both reading zeros).
//!
//! Run: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm
//!      --test dot4_kquant_grouped_parity -- --nocapture --test-threads=1

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

/// bytes per 256-weight super-block, and the quant/dequant pair.
fn scheme(
    name: &str,
) -> (
    usize,
    fn(&[f32]) -> grim_tensor::Result<Vec<u8>>,
    fn(&[u8], usize) -> grim_tensor::Result<Vec<f32>>,
) {
    match name {
        "q4k" => (144, grim_quant::quant_q4k, grim_quant::dequant_q4k),
        "q5k" => (176, grim_quant::quant_q5k, grim_quant::dequant_q5k),
        "q6k" => (210, grim_quant::quant_q6k, grim_quant::dequant_q6k),
        // Q2_K: 84 bytes per 256 weights, one 2-bit code per BYTE in qs[64]
        // at offset 16. The word-wide-mask variant reads a 4-weight group
        // with ONE 32-bit load and `& 0x03030303` instead of four 1-byte
        // loads — the same mask the Q5_K `qh` field already uses.
        "q2k" => (84, grim_quant::quant_q2k, grim_quant::dequant_q2k),
        // IQ4_XS: 136 bytes per 256 weights — d f16 @0, scales_h[2] @2,
        // scales_l[8] @4, qs[128] @8. The codes are NIBBLES: weights 0..15
        // take the LOW nibble of qs[0..15], weights 16..31 the HIGH nibble of
        // the SAME bytes; values go through kvalues_iq4nl (NOT linear in the
        // code), so the dot4 kernel packs kvalues[code] into the i8 operand.
        "iq4xs" => (136, grim_quant::quant_iq4xs, grim_quant::dequant_iq4xs),
        // Q3_K: 110 bytes per 256 — hmask[32] (sign bits, stays per-byte),
        // qs[64] (one 2-bit magnitude per BYTE, so a 4-weight group is four
        // consecutive bytes: one 32-bit load + & 0x03030303), scales[12],
        // f16 d @108.
        "q3k" => (110, grim_quant::quant_q3k, grim_quant::dequant_q3k),
        other => panic!("unknown scheme {other}"),
    }
}

fn grouped_parity(scheme_name: &str, k: usize, n: usize, m: usize) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    assert_eq!(k % 256, 0, "K-quants need k % 256 == 0");
    let (blk_bytes, quant, dequant) = scheme(scheme_name);
    let row_bytes = (k / 256) * blk_bytes;

    let mut rng = Lcg(0x5A5A_1234 ^ (k as u64) ^ ((n as u64) << 32) ^ (m as u64));
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
        let packed = quant(&b[col * k..(col + 1) * k]).map_err(|e| format!("quant: {e}"))?;
        assert_eq!(packed.len(), row_bytes, "{scheme_name} row length");
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
    let o_new = as_rocm(out_new.as_ref()).unwrap();
    let o_ref = as_rocm(out_ref.as_ref()).unwrap();
    match scheme_name {
        "q4k" => {
            dev.launch_dot4_q4k_q81_gemv_ref_for_ab(a_r, b_r, o_ref, m, n, k)
                .map_err(|e| format!("launch ref: {e}"))?;
            // The stock Q4_K kernel is not what the default dispatch runs
            // (that is the `_fast` 2-column variant), so it is driven
            // through its own launcher rather than the process-wide
            // GRIM_DOT4_FAST OnceLock.
            dev.launch_dot4_q4k_q81_gemv_stock_for_ab(a_r, b_r, o_new, m, n, k)
                .map_err(|e| format!("launch new: {e}"))?;
        }
        "q2k" => {
            // Both sides go through a LAUNCHER, so the gate sees the grid the
            // dispatch actually uses. The Q4_K lesson: a gate that called
            // the kernel with its own grid stayed green while the launcher
            // fed it a grid covering half the columns.
            dev.launch_dot4_q2k_q81_gemv_stock_for_ab(a_r, b_r, o_ref, m, n, k)
                .map_err(|e| format!("launch ref: {e}"))?;
            dev.launch_dot4_q2k_q81_gemv_for_ab(a_r, b_r, o_new, m, n, k)
                .map_err(|e| format!("launch new: {e}"))?;
        }
        "iq4xs" => {
            // Both dot4 sides through launchers (the Q4_K lesson). The
            // production baseline the A/B is judged against is the f32-act
            // `_grouped` fused kernel (103.9 GB/s at n=4096/k=5120), timed
            // separately in `iq4xs_dot4_vs_grouped_speed_ab`.
            dev.launch_dot4_iq4xs_q81_gemv_stock_for_ab(a_r, b_r, o_ref, m, n, k)
                .map_err(|e| format!("launch ref: {e}"))?;
            dev.launch_dot4_iq4xs_q81_gemv_for_ab(a_r, b_r, o_new, m, n, k)
                .map_err(|e| format!("launch new: {e}"))?;
        }
        "q3k" => {
            // "ref" = the stock per-byte magnitude loads; "new" = the
            // word-wide-mask default through the launcher's OnceLock.
            dev.launch_dot4_q3k_q81_gemv_stock_for_ab(a_r, b_r, o_ref, m, n, k)
                .map_err(|e| format!("launch ref: {e}"))?;
            dev.launch_dot4_q3k_q81_gemv_for_ab(a_r, b_r, o_new, m, n, k)
                .map_err(|e| format!("launch new: {e}"))?;
        }
        "q5k" => {
            dev.launch_dot4_q5k_q81_gemv_ref_for_ab(a_r, b_r, o_ref, m, n, k)
                .map_err(|e| format!("launch ref: {e}"))?;
            dev.launch_dot4_q5k_q81_gemv_for_ab(a_r, b_r, o_new, m, n, k)
                .map_err(|e| format!("launch new: {e}"))?;
        }
        _ => {
            dev.launch_dot4_q6k_q81_gemv_ref_for_ab(a_r, b_r, o_ref, m, n, k)
                .map_err(|e| format!("launch ref: {e}"))?;
            dev.launch_dot4_q6k_q81_gemv_for_ab(a_r, b_r, o_new, m, n, k)
                .map_err(|e| format!("launch new: {e}"))?;
        }
    }
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
        "{scheme_name} grouped hoist is not byte-identical to the retained oracle at k={k} n={n} \
         m={m}: {differing}/{} elements differ, first at {:?}",
        got_new.len(),
        first_diff
    );

    // (2) CORRECTNESS of the (identical) pair against the llama.cpp-faithful
    // dequant dotted with the same activation codes.
    let a_deq = dequant_q81(&q81, m * k);
    let mut worst = 0.0f32;
    let mut worst_case = (0usize, 0usize, 0f32, 0f32, 0f32);
    let mut nonzero = 0usize;
    for row in 0..m {
        for col in 0..n {
            let w = dequant(&bq[col * row_bytes..(col + 1) * row_bytes], k)
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
            if d > worst {
                worst = d;
                worst_case = (row, col, got, acc, (got - acc).abs());
            }
        }
    }
    if worst > 1e-2 {
        let (row, col, got, acc, absd) = worst_case;
        eprintln!(
            "{scheme_name} worst k={k} n={n} m={m}: ({row},{col}) got={got} oracle={acc} \
             abs_diff={absd:.3e} rel={worst:.3e}"
        );
    }
    assert_eq!(
        nonzero,
        m * n,
        "{scheme_name} grouped kernel produced all-zero output at k={k} n={n} m={m} — the parity \
         comparison would have been vacuous"
    );
    println!(
        "{scheme_name} grouped k={k:<6} n={n:<5} m={m:<3} byte-identical to the retained oracle, \
         worst relative error {worst:.3e}"
    );
    assert!(
        worst <= 2e-2,
        "{scheme_name} grouped GEMV disagrees with the dequant oracle by {worst:.3e} on {}",
        dev.gpu_target_str()
    );
    Ok(())
}

#[test]
fn kquant_grouped_matches_retained_oracle() -> Result<(), String> {
    for (k, n, m) in [
        (256usize, 64usize, 1usize),
        (1024, 64, 1),
        (1024, 64, 4),
        (4096, 68, 1),
        (4096, 68, 8),
    ] {
        for s in ["q4k", "q5k", "q6k", "q2k", "iq4xs", "q3k"] {
            grouped_parity(s, k, n, m)?;
        }
    }
    Ok(())
}

/// The A/B the hoist has to justify: the hoisted kernel against the
/// retained pre-change oracle, at the 27B's dominant projection shape
/// (n=17408, k=5120, decode m=1).
fn speed_ab(scheme_name: &str) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let (blk_bytes, quant, _dequant) = scheme(scheme_name);
    let k = 5120usize;
    let n = 17408usize;
    let m = 1usize;
    let row_bytes = (k / 256) * blk_bytes;

    let mut rng = Lcg(0xBEEF_0001);
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

    let mut bq = vec![0u8; n * row_bytes];
    for col in 0..n {
        let packed = quant(&b[col * k..(col + 1) * k]).map_err(|e| format!("quant: {e}"))?;
        bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
    }
    let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
        .map_err(|e| format!("b h2d: {e}"))?;
    let out_st = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![m * n]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("out alloc: {e}"))?;

    let a_r = as_rocm(q81_st.as_ref()).unwrap();
    let b_r = as_rocm(b_st.as_ref()).unwrap();
    let o_r = as_rocm(out_st.as_ref()).unwrap();

    let (new_fn, ref_fn): (
        fn(
            &RocmDevice,
            &grim_backend_rocm::RocmStorage,
            &grim_backend_rocm::RocmStorage,
            &grim_backend_rocm::RocmStorage,
            usize,
            usize,
            usize,
        ) -> Result<*mut std::ffi::c_void, String>,
        fn(
            &RocmDevice,
            &grim_backend_rocm::RocmStorage,
            &grim_backend_rocm::RocmStorage,
            &grim_backend_rocm::RocmStorage,
            usize,
            usize,
            usize,
        ) -> Result<*mut std::ffi::c_void, String>,
    ) = match scheme_name {
        "q4k" => (
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q4k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q4k_q81_gemv_ref_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
        ),
        "q5k" => (
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q5k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q5k_q81_gemv_ref_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
        ),
        // Q2_K: "new" is the word-wide-mask variant (the default through the
        // launcher), "ref" is the stock 4-byte-load kernel. Both are driven
        // through launchers so the grid is the dispatch's own.
        "q2k" => (
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q2k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q2k_q81_gemv_stock_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
        ),
        "iq4xs" => (
            |d, a, b, o, m, n, k| {
                d.launch_dot4_iq4xs_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
            |d, a, b, o, m, n, k| {
                d.launch_dot4_iq4xs_q81_gemv_stock_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
        ),
        "q3k" => (
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q3k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q3k_q81_gemv_stock_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
        ),
        _ => (
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q6k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
            |d, a, b, o, m, n, k| {
                d.launch_dot4_q6k_q81_gemv_ref_for_ab(a, b, o, m, n, k)
                    .map_err(|e| e.to_string())
            },
        ),
    };

    let time_it = |f: &dyn Fn(
        &RocmDevice,
        &grim_backend_rocm::RocmStorage,
        &grim_backend_rocm::RocmStorage,
        &grim_backend_rocm::RocmStorage,
        usize,
        usize,
        usize,
    ) -> Result<*mut std::ffi::c_void, String>|
     -> f64 {
        // warm-up
        for _ in 0..2 {
            f(&dev, a_r, b_r, o_r, m, n, k).unwrap();
        }
        dev.synchronize();
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f(&dev, a_r, b_r, o_r, m, n, k).unwrap();
        }
        dev.synchronize();
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    };

    let t_new = time_it(&new_fn);
    let t_ref = time_it(&ref_fn);
    let bytes = (n * row_bytes) as f64;
    println!(
        "{scheme_name} dot4 n={n} k={k} m={m}: hoisted {t_new:.3} ms ({:.1} GB/s)  \
         retained {t_ref:.3} ms ({:.1} GB/s)  speedup {:.2}x",
        bytes / t_new / 1e6,
        bytes / t_ref / 1e6,
        t_ref / t_new
    );
    Ok(())
}

#[test]
fn kquant_grouped_speed_ab() -> Result<(), String> {
    for s in ["q4k", "q5k", "q6k", "q2k", "iq4xs", "q3k"] {
        speed_ab(s)?;
    }
    Ok(())
}

/// The REAL A/B for IQ4_XS: the dot4 fast kernel against the production
/// f32-act `_grouped` fused kernel (103.9 GB/s at n=4096/k=5120 per the
/// census). The dot4-vs-stock comparison in `speed_ab` only proves the word
/// unpack is faster than its own byte twin; this one decides whether the
/// dot4 kernel earns a default switch at all.
#[test]
fn iq4xs_dot4_vs_grouped_speed_ab() -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let (blk_bytes, quant, _dequant) = scheme("iq4xs");
    let k = 5120usize;
    let m = 1usize;

    let time_shape = |n: usize| -> Result<(), String> {
        let f32ty = DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        };
        let u8ty = DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        };
        let row_bytes = (k / 256) * blk_bytes;
        let mut rng = Lcg(0xBEEF_0042 ^ (n as u64));
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();
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
        let mut bq = vec![0u8; n * row_bytes];
        for col in 0..n {
            let packed = quant(&b[col * k..(col + 1) * k]).map_err(|e| format!("quant: {e}"))?;
            bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
        }
        let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
            .map_err(|e| format!("b h2d: {e}"))?;
        let out_st = MemoryOps::alloc_storage(
            &dev,
            &Shape::new(vec![m * n]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Native,
            },
        )
        .map_err(|e| format!("out alloc: {e}"))?;
        let a_r = as_rocm(q81_st.as_ref()).unwrap();
        let b_r = as_rocm(b_st.as_ref()).unwrap();
        let a_f = as_rocm(a_st.as_ref()).unwrap();
        let o_r = as_rocm(out_st.as_ref()).unwrap();

        let time_it = |f: &dyn Fn() -> Result<*mut std::ffi::c_void, String>| -> f64 {
            for _ in 0..5 {
                f().unwrap();
            }
            dev.synchronize();
            let iters = 50;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f().unwrap();
            }
            dev.synchronize();
            t0.elapsed().as_secs_f64() * 1e6 / iters as f64
        };
        let bytes = (n * row_bytes) as f64;
        let t_dot4 = time_it(&|| {
            dev.launch_dot4_iq4xs_q81_gemv_sub_v2_for_ab(a_r, b_r, o_r, m, n, k)
                .map_err(|e| e.to_string())
        });
        let t_grouped = time_it(&|| {
            dev.launch_fused_dequant_gemm_iq4xs_for_ab(a_f, b_r, o_r, m, n, k)
                .map_err(|e| e.to_string())
        });
        println!(
            "iq4xs dot4-vs-grouped n={n} k={k} m={m}: dot4 {t_dot4:.1} us ({:.1} GB/s)  \
             grouped-f32 {t_grouped:.1} us ({:.1} GB/s)  speedup {:.2}x",
            bytes / t_dot4 / 1e3,
            bytes / t_grouped / 1e3,
            t_grouped / t_dot4
        );
        Ok(())
    };
    for n in [4096usize, 17408usize] {
        time_shape(n)?;
    }
    Ok(())
}

type GemvFn = fn(
    &RocmDevice,
    &grim_backend_rocm::RocmStorage,
    &grim_backend_rocm::RocmStorage,
    &grim_backend_rocm::RocmStorage,
    usize,
    usize,
    usize,
) -> grim_tensor::Result<*mut std::ffi::c_void>;

/// Step 2.1 A/B: sub-block-per-lane kernels vs the retained oracle
/// (tolerance: float association differs) + bandwidth.
///
/// Bandwidth is reported twice: WARM (same weights every launch, so the
/// 32 MB Infinity Cache serves part of a 61-73 MB tensor) and COLD (launches
/// rotate across `ROT` weight copies totalling > 240 MB, so every launch
/// streams from DRAM). Only COLD is comparable to the DRAM spec peak.
fn sub_lane_ab(
    name: &str,
    blk: usize,
    quant: fn(&[f32]) -> grim_tensor::Result<Vec<u8>>,
    stock: GemvFn,
    oracle: GemvFn,
    sub: GemvFn,
) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    const ROT: usize = 4;
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    for &(n, k, m, timed) in &[
        (64usize, 256usize, 1usize, false),
        (68, 1024, 1, false),
        (68, 4096, 4, false),
        (67, 5120, 1, false),
        (17408, 5120, 1, true),
        (5120, 17408, 1, true),
    ] {
        let row_bytes = (k / 256) * blk;
        let mut rng = Lcg(0xC0FFEE ^ (n * 31 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        let mut bq = vec![0u8; n * row_bytes];
        for col in 0..n {
            let w: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
            let p = quant(&w).map_err(|e| e.to_string())?;
            bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&p);
        }
        let a_st =
            MemoryOps::from_cpu_bytes(&dev, f32_bytes(&a), &Shape::new(vec![m * k]), f32ty.clone())
                .map_err(|e| e.to_string())?;
        let q81_len = m * (k / 32) * 36;
        let q81 = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; q81_len],
            &Shape::new(vec![q81_len]),
            u8ty.clone(),
        )
        .map_err(|e| e.to_string())?;
        dev.launch_quantize_q8_1(
            as_rocm(a_st.as_ref()).unwrap(),
            as_rocm(q81.as_ref()).unwrap(),
            m,
            k,
        )
        .map_err(|e| e.to_string())?;
        let copies = if timed { ROT } else { 1 };
        let b_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty.clone())
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o_ref = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o_new = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(q81.as_ref()).unwrap();
        let b_rs: Vec<_> = b_sts.iter().map(|b| as_rocm(b.as_ref()).unwrap()).collect();
        let (or_, on_) = (
            as_rocm(o_ref.as_ref()).unwrap(),
            as_rocm(o_new.as_ref()).unwrap(),
        );
        oracle(&dev, a_r, b_rs[0], or_, m, n, k).map_err(|e| e.to_string())?;
        sub(&dev, a_r, b_rs[0], on_, m, n, k).map_err(|e| e.to_string())?;
        dev.synchronize();
        let (r, s) = (f32_vec(&o_ref)?, f32_vec(&o_new)?);
        let scale = r.iter().fold(0f32, |x, v| x.max(v.abs())).max(1e-6);
        let worst = r
            .iter()
            .zip(&s)
            .fold(0f32, |x, (p, q)| x.max((p - q).abs()))
            / scale;
        println!("{name} sub n={n} k={k} m={m}: worst rel-to-max {worst:.3e}");
        assert!(
            worst < 1e-4,
            "{name} sub n={n} k={k} m={m}: worst rel-to-max {worst:.3e}"
        );
        if timed {
            let time = |f: GemvFn, rotate: bool| {
                for i in 0..ROT {
                    f(&dev, a_r, b_rs[i], on_, m, n, k).unwrap();
                }
                dev.synchronize();
                let iters = 48;
                let t0 = std::time::Instant::now();
                for i in 0..iters {
                    let b = if rotate { b_rs[i % ROT] } else { b_rs[0] };
                    f(&dev, a_r, b, on_, m, n, k).unwrap();
                }
                dev.synchronize();
                t0.elapsed().as_secs_f64() / iters as f64
            };
            let gb = (n * row_bytes) as f64 / 1e9;
            for (label, rot) in [("warm", false), ("cold", true)] {
                let ts = time(stock, rot);
                let tn = time(sub, rot);
                println!(
                    "{name} {label} n={n} k={k}: stock {:.1} us ({:.1} GB/s)  sub {:.1} us ({:.1} GB/s)  {:.2}x",
                    ts * 1e6,
                    gb / ts,
                    tn * 1e6,
                    gb / tn,
                    ts / tn
                );
            }
        }
    }
    Ok(())
}

#[test]
fn q5k_sub_lane_parity_and_speed() -> Result<(), String> {
    sub_lane_ab(
        "q5k",
        176,
        grim_quant::quant_q5k,
        |d, a, b, o, m, n, k| d.launch_dot4_q5k_q81_gemv_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q5k_q81_gemv_ref_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q5k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn q6k_sub_lane_parity_and_speed() -> Result<(), String> {
    sub_lane_ab(
        "q6k",
        210,
        grim_quant::quant_q6k,
        |d, a, b, o, m, n, k| d.launch_dot4_q6k_q81_gemv_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q6k_q81_gemv_ref_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q6k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn q4k_sub_lane_parity_and_speed() -> Result<(), String> {
    sub_lane_ab(
        "q4k",
        144,
        grim_quant::quant_q4k,
        |d, a, b, o, m, n, k| d.launch_dot4_q4k_q81_gemv_stock_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q4k_q81_gemv_ref_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q4k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn q3k_sub_lane_parity_and_speed() -> Result<(), String> {
    sub_lane_ab(
        "q3k",
        110,
        grim_quant::quant_q3k,
        |d, a, b, o, m, n, k| d.launch_dot4_q3k_q81_gemv_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q3k_q81_gemv_stock_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q3k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn iq4xs_sub_lane_parity_and_speed() -> Result<(), String> {
    sub_lane_ab(
        "iq4xs",
        136,
        grim_quant::quant_iq4xs,
        |d, a, b, o, m, n, k| d.launch_dot4_iq4xs_q81_gemv_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_iq4xs_q81_gemv_stock_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_iq4xs_q81_gemv_sub_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn q2k_sub_lane_parity_and_speed() -> Result<(), String> {
    sub_lane_ab(
        "q2k",
        84,
        grim_quant::quant_q2k,
        |d, a, b, o, m, n, k| d.launch_dot4_q2k_q81_gemv_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q2k_q81_gemv_stock_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q2k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
    )
}

/// Step 2.2 (#1 + #2) A/B gate across all K-quant schemes:
/// 1. First verifies `_sub_v2` is BIT-IDENTICAL to `_sub` AND within 1e-4 of
///    the retained GPU oracle (`_ref` / `_stock`) AND within 2e-2 of the host
///    `dequant_*` dot `dequant_q81` oracle across all boundary shapes before timing.
/// 2. Then runs a direct A/B benchmark between `_sub` (`sub_v1`) and `_sub_v2`
///    (both warm and cold 4x-rotated >200 MB DRAM).
fn sub_v2_parity_then_ab(
    name: &str,
    blk: usize,
    quant: fn(&[f32]) -> grim_tensor::Result<Vec<u8>>,
    dequant: fn(&[u8], usize) -> grim_tensor::Result<Vec<f32>>,
    oracle: GemvFn,
    sub_v1: GemvFn,
    sub_v2: GemvFn,
) -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    const ROT: usize = 4;
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };

    for &(n, k, m, timed) in &[
        (64usize, 256usize, 1usize, false),
        (68, 1024, 1, false),
        (68, 4096, 4, false),
        (67, 5120, 1, false),
        (17408, 5120, 1, true),
        (5120, 17408, 1, true),
    ] {
        let row_bytes = (k / 256) * blk;
        let mut rng = Lcg(0xC0FFEE ^ (n * 31 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        let mut bq = vec![0u8; n * row_bytes];
        for col in 0..n {
            let w: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
            let p = quant(&w).map_err(|e| e.to_string())?;
            bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&p);
        }
        let a_st =
            MemoryOps::from_cpu_bytes(&dev, f32_bytes(&a), &Shape::new(vec![m * k]), f32ty.clone())
                .map_err(|e| e.to_string())?;
        let q81_len = m * (k / 32) * 36;
        let q81 = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; q81_len],
            &Shape::new(vec![q81_len]),
            u8ty.clone(),
        )
        .map_err(|e| e.to_string())?;
        dev.launch_quantize_q8_1(
            as_rocm(a_st.as_ref()).unwrap(),
            as_rocm(q81.as_ref()).unwrap(),
            m,
            k,
        )
        .map_err(|e| e.to_string())?;

        let copies = if timed { ROT } else { 1 };
        let b_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty.clone())
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o_ref = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o_sub = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o_v2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(q81.as_ref()).unwrap();
        let b_rs: Vec<_> = b_sts.iter().map(|b| as_rocm(b.as_ref()).unwrap()).collect();
        let (or_, os_, ov2_) = (
            as_rocm(o_ref.as_ref()).unwrap(),
            as_rocm(o_sub.as_ref()).unwrap(),
            as_rocm(o_v2.as_ref()).unwrap(),
        );

        oracle(&dev, a_r, b_rs[0], or_, m, n, k).map_err(|e| e.to_string())?;
        sub_v1(&dev, a_r, b_rs[0], os_, m, n, k).map_err(|e| e.to_string())?;
        sub_v2(&dev, a_r, b_rs[0], ov2_, m, n, k).map_err(|e| e.to_string())?;
        dev.synchronize();

        let (r_ref, r_sub, r_v2) = (f32_vec(&o_ref)?, f32_vec(&o_sub)?, f32_vec(&o_v2)?);
        let bit_diffs = r_sub
            .iter()
            .zip(&r_v2)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            bit_diffs, 0,
            "{name} sub_v2 is NOT bit-identical to sub at n={n} k={k} m={m} ({bit_diffs} diffs)"
        );

        let scale = r_ref.iter().fold(0f32, |x, v| x.max(v.abs())).max(1e-6);
        let worst_vs_ref = r_ref
            .iter()
            .zip(&r_v2)
            .fold(0f32, |x, (p, q)| x.max((p - q).abs()))
            / scale;
        assert!(
            worst_vs_ref < 1e-4,
            "{name} sub_v2 vs ref worst rel-to-max {worst_vs_ref:.3e}"
        );

        if !timed {
            let q81_host = bytes_of(&q81)?;
            let a_deq = dequant_q81(&q81_host, m * k);
            let mut worst_deq = 0f32;
            for row in 0..m {
                for col in 0..n {
                    let w = dequant(&bq[col * row_bytes..(col + 1) * row_bytes], k)
                        .map_err(|e| e.to_string())?;
                    let mut acc = 0f32;
                    for kk in 0..k {
                        acc += a_deq[row * k + kk] * w[kk];
                    }
                    let got = r_v2[row * n + col];
                    worst_deq = worst_deq.max((got - acc).abs() / acc.abs().max(1e-3));
                }
            }
            println!(
                "{name:<5} sub_v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to sub_v1, vs_ref={worst_vs_ref:.3e}, vs_host_dequant={worst_deq:.3e}"
            );
            assert!(
                worst_deq <= 2e-2,
                "{name} sub_v2 vs host dequant {worst_deq:.3e}"
            );
        } else {
            println!(
                "{name:<5} sub_v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to sub_v1, vs_ref={worst_vs_ref:.3e}"
            );
            let time = |f: GemvFn, rotate: bool| {
                for i in 0..ROT {
                    f(&dev, a_r, b_rs[i], ov2_, m, n, k).unwrap();
                }
                dev.synchronize();
                let iters = 48;
                let t0 = std::time::Instant::now();
                for i in 0..iters {
                    let b = if rotate { b_rs[i % ROT] } else { b_rs[0] };
                    f(&dev, a_r, b, ov2_, m, n, k).unwrap();
                }
                dev.synchronize();
                t0.elapsed().as_secs_f64() / iters as f64
            };
            let gb = (n * row_bytes) as f64 / 1e9;
            for (label, rot) in [("warm", false), ("cold", true)] {
                let t_v1 = time(sub_v1, rot);
                let t_v2 = time(sub_v2, rot);
                println!(
                    "{name:<5} A/B {label:<4} n={n:<5} k={k:<5}: sub_v1 {:.1} us ({:.1} GB/s)  sub_v2 {:.1} us ({:.1} GB/s)  {:.3}x",
                    t_v1 * 1e6,
                    gb / t_v1,
                    t_v2 * 1e6,
                    gb / t_v2,
                    t_v1 / t_v2
                );
            }
        }
    }
    Ok(())
}

#[test]
fn q4k_sub_v2_parity_then_ab() -> Result<(), String> {
    sub_v2_parity_then_ab(
        "q4k",
        144,
        grim_quant::quant_q4k,
        grim_quant::dequant_q4k,
        |d, a, b, o, m, n, k| d.launch_dot4_q4k_q81_gemv_ref_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q4k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q4k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn q5k_sub_v2_parity_then_ab() -> Result<(), String> {
    sub_v2_parity_then_ab(
        "q5k",
        176,
        grim_quant::quant_q5k,
        grim_quant::dequant_q5k,
        |d, a, b, o, m, n, k| d.launch_dot4_q5k_q81_gemv_ref_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q5k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q5k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn q6k_sub_v2_parity_then_ab() -> Result<(), String> {
    sub_v2_parity_then_ab(
        "q6k",
        210,
        grim_quant::quant_q6k,
        grim_quant::dequant_q6k,
        |d, a, b, o, m, n, k| d.launch_dot4_q6k_q81_gemv_ref_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q6k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q6k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn q3k_sub_v2_parity_then_ab() -> Result<(), String> {
    sub_v2_parity_then_ab(
        "q3k",
        110,
        grim_quant::quant_q3k,
        grim_quant::dequant_q3k,
        |d, a, b, o, m, n, k| d.launch_dot4_q3k_q81_gemv_stock_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q3k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q3k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn q2k_sub_v2_parity_then_ab() -> Result<(), String> {
    sub_v2_parity_then_ab(
        "q2k",
        84,
        grim_quant::quant_q2k,
        grim_quant::dequant_q2k,
        |d, a, b, o, m, n, k| d.launch_dot4_q2k_q81_gemv_stock_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q2k_q81_gemv_sub_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_q2k_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k),
    )
}

#[test]
fn iq4xs_sub_v2_parity_then_ab() -> Result<(), String> {
    sub_v2_parity_then_ab(
        "iq4xs",
        136,
        grim_quant::quant_iq4xs,
        grim_quant::dequant_iq4xs,
        |d, a, b, o, m, n, k| d.launch_dot4_iq4xs_q81_gemv_stock_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_iq4xs_q81_gemv_sub_for_ab(a, b, o, m, n, k),
        |d, a, b, o, m, n, k| d.launch_dot4_iq4xs_q81_gemv_sub_v2_for_ab(a, b, o, m, n, k),
    )
}

/// Step 2.3: Non-K-quant / native GEMV v2 parity + A/B gate (`fp32`, `bf16`,
/// `fp8`, `forest`, `q80_q81`, `q80_f32act`):
/// 1. First verifies `v2` is 100% BIT-IDENTICAL (`0` bit diffs) to `v1` across
///    all boundary shapes (`(64,256,1)`, `(68,1024,1)`, `(68,4096,4)`, `(67,5120,1)`,
///    plus benchmark shapes) before timing.
/// 2. Then runs warm + cold (4x rotated >200 MB DRAM) A/B benchmarks on Card 1.
#[test]
fn non_kquant_gemv_v2_parity_then_ab() -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    const ROT: usize = 4;
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };

    let time_ab = |name: &str,
                   n: usize,
                   k: usize,
                   m: usize,
                   row_bytes: usize,
                   a_r: &grim_backend_rocm::RocmStorage,
                   b_rs: &[&grim_backend_rocm::RocmStorage],
                   o_r: &grim_backend_rocm::RocmStorage,
                   v1: GemvFn,
                   v2: GemvFn| {
        let time = |f: GemvFn, rotate: bool| {
            for i in 0..ROT {
                f(&dev, a_r, b_rs[i], o_r, m, n, k).unwrap();
            }
            dev.synchronize();
            let iters = 48;
            let t0 = std::time::Instant::now();
            for i in 0..iters {
                let b = if rotate { b_rs[i % ROT] } else { b_rs[0] };
                f(&dev, a_r, b, o_r, m, n, k).unwrap();
            }
            dev.synchronize();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        let gb = (n * row_bytes) as f64 / 1e9;
        for (label, rot) in [("warm", false), ("cold", true)] {
            let t1 = time(v1, rot);
            let t2 = time(v2, rot);
            println!(
                "{name:<10} A/B {label:<4} n={n:<5} k={k:<5}: v1 {:.1} us ({:.1} GB/s)  v2 {:.1} us ({:.1} GB/s)  {:.3}x",
                t1 * 1e6,
                gb / t1,
                t2 * 1e6,
                gb / t2,
                t1 / t2
            );
        }
    };

    // 1. FP32 GEMV (`grim_fp32_gemv` vs `grim_fp32_gemv_v2`)
    for &(n, k, m, timed) in &[
        (64usize, 256usize, 1usize, false),
        (68, 1024, 1, false),
        (68, 4096, 4, false),
        (67, 5120, 1, false),
        (4096, 5120, 1, true),
        (5120, 4096, 1, true),
    ] {
        let mut rng = Lcg(0xA11CE001 ^ (n * 31 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();
        let b_bytes = f32_bytes(&b);
        let a_st =
            MemoryOps::from_cpu_bytes(&dev, f32_bytes(&a), &Shape::new(vec![m * k]), f32ty.clone())
                .map_err(|e| e.to_string())?;
        let copies = if timed { ROT } else { 1 };
        let b_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(&dev, b_bytes, &Shape::new(vec![n * k]), f32ty.clone())
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_rs: Vec<_> = b_sts.iter().map(|s| as_rocm(s.as_ref()).unwrap()).collect();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());
        dev.launch_fp32_gemv_v1_for_ab(a_r, b_rs[0], o1_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_fp32_gemv_v2_for_ab(a_r, b_rs[0], o2_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_out, v2_out) = (f32_vec(&o1)?, f32_vec(&o2)?);
        let diffs = v1_out
            .iter()
            .zip(&v2_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "fp32_gemv_v2 not bit-identical at n={n} k={k} m={m}"
        );
        println!("fp32       v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");
        if timed {
            time_ab(
                "fp32",
                n,
                k,
                m,
                k * 4,
                a_r,
                &b_rs,
                o2_r,
                |d, a, b, o, m, n, k| d.launch_fp32_gemv_v1_for_ab(a, b, o, m, n, k),
                |d, a, b, o, m, n, k| d.launch_fp32_gemv_v2_for_ab(a, b, o, m, n, k),
            );
        }
    }

    // 2. BF16 GEMV (`grim_dot2_bf16_gemv` vs `grim_dot2_bf16_gemv_v2`)
    for &(n, k, m, timed) in &[
        (64usize, 256usize, 1usize, false),
        (68, 1024, 1, false),
        (68, 4096, 4, false),
        (67, 5120, 1, false),
        (8192, 5120, 1, true),
        (5120, 8192, 1, true),
    ] {
        let mut rng = Lcg(0xA11CE002 ^ (n * 31 + k) as u64);
        let a_bf16: Vec<u8> = (0..m * k)
            .flat_map(|_| half::bf16::from_f32(rng.next_f32()).to_le_bytes())
            .collect();
        let b_bf16: Vec<u8> = (0..n * k)
            .flat_map(|_| half::bf16::from_f32(rng.next_f32()).to_le_bytes())
            .collect();
        let a_st =
            MemoryOps::from_cpu_bytes(&dev, &a_bf16, &Shape::new(vec![a_bf16.len()]), u8ty.clone())
                .map_err(|e| e.to_string())?;
        let copies = if timed { ROT } else { 1 };
        let b_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(
                    &dev,
                    &b_bf16,
                    &Shape::new(vec![b_bf16.len()]),
                    u8ty.clone(),
                )
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n * 2]), u8ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n * 2]), u8ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_rs: Vec<_> = b_sts.iter().map(|s| as_rocm(s.as_ref()).unwrap()).collect();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());
        dev.launch_dot2_bf16_gemv_v1_for_ab(a_r, b_rs[0], o1_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_dot2_bf16_gemv_v2_for_ab(a_r, b_rs[0], o2_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_out, v2_out) = (bytes_of(&o1)?, bytes_of(&o2)?);
        assert_eq!(
            v1_out, v2_out,
            "bf16_gemv_v2 not bit-identical at n={n} k={k} m={m}"
        );
        println!("bf16       v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");
        if timed {
            time_ab(
                "bf16",
                n,
                k,
                m,
                k * 2,
                a_r,
                &b_rs,
                o2_r,
                |d, a, b, o, m, n, k| d.launch_dot2_bf16_gemv_v1_for_ab(a, b, o, m, n, k),
                |d, a, b, o, m, n, k| d.launch_dot2_bf16_gemv_v2_for_ab(a, b, o, m, n, k),
            );
        }
    }

    // 3. FP8 GEMV (`grim_dot4_fp8_gemv` vs `grim_dot4_fp8_gemv_v2`)
    for &(n, k, m, timed) in &[
        (64usize, 256usize, 1usize, false),
        (68, 1024, 1, false),
        (68, 4096, 4, false),
        (67, 5120, 1, false),
        (17408, 5120, 1, true),
        (5120, 17408, 1, true),
    ] {
        let mut rng = Lcg(0xA11CE003 ^ (n * 31 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        // Valid non-NaN E4M3 codes (avoid 0x7F and 0xFF)
        let b_fp8: Vec<u8> = (0..n * k).map(|i| ((i * 13 + 7) % 120) as u8).collect();
        let a_st =
            MemoryOps::from_cpu_bytes(&dev, f32_bytes(&a), &Shape::new(vec![m * k]), f32ty.clone())
                .map_err(|e| e.to_string())?;
        let copies = if timed { ROT } else { 1 };
        let b_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(
                    &dev,
                    &b_fp8,
                    &Shape::new(vec![b_fp8.len()]),
                    u8ty.clone(),
                )
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_rs: Vec<_> = b_sts.iter().map(|s| as_rocm(s.as_ref()).unwrap()).collect();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());
        dev.launch_dot4_fp8_gemv_v1_for_ab(a_r, b_rs[0], o1_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_dot4_fp8_gemv_v2_for_ab(a_r, b_rs[0], o2_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_out, v2_out) = (f32_vec(&o1)?, f32_vec(&o2)?);
        let diffs = v1_out
            .iter()
            .zip(&v2_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "fp8_gemv_v2 not bit-identical at n={n} k={k} m={m}"
        );
        println!("fp8        v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");
        if timed {
            time_ab(
                "fp8",
                n,
                k,
                m,
                k,
                a_r,
                &b_rs,
                o2_r,
                |d, a, b, o, m, n, k| d.launch_dot4_fp8_gemv_v1_for_ab(a, b, o, m, n, k),
                |d, a, b, o, m, n, k| d.launch_dot4_fp8_gemv_v2_for_ab(a, b, o, m, n, k),
            );
        }
    }

    // 4. ForestRaven GEMV (`grim_dot4_forest_gemv` vs `grim_dot4_forest_gemv_v2`)
    for &(n, k, m, timed) in &[
        (64usize, 256usize, 1usize, false),
        (68, 1024, 1, false),
        (68, 4096, 4, false),
        (67, 5120, 1, false),
        (17408, 5120, 1, true),
        (5120, 17408, 1, true),
    ] {
        let mut rng = Lcg(0xA11CE004 ^ (n * 31 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();
        let (codes, scales) =
            grim_quant::quant_forest_per_channel(&w, n, k).map_err(|e| e.to_string())?;
        let mut b_blob = Vec::with_capacity(16 + codes.len() + scales.len());
        b_blob.extend_from_slice(&(codes.len() as u64).to_le_bytes());
        b_blob.extend_from_slice(&codes);
        b_blob.extend_from_slice(&(scales.len() as u64).to_le_bytes());
        b_blob.extend_from_slice(&scales);
        let a_st =
            MemoryOps::from_cpu_bytes(&dev, f32_bytes(&a), &Shape::new(vec![m * k]), f32ty.clone())
                .map_err(|e| e.to_string())?;
        let copies = if timed { ROT } else { 1 };
        let b_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(
                    &dev,
                    &b_blob,
                    &Shape::new(vec![b_blob.len()]),
                    u8ty.clone(),
                )
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_rs: Vec<_> = b_sts.iter().map(|s| as_rocm(s.as_ref()).unwrap()).collect();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());
        dev.launch_dot4_forest_gemv_v1_for_ab(a_r, b_rs[0], o1_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_dot4_forest_gemv_v2_for_ab(a_r, b_rs[0], o2_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_out, v2_out) = (f32_vec(&o1)?, f32_vec(&o2)?);
        let diffs = v1_out
            .iter()
            .zip(&v2_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "forest_gemv_v2 not bit-identical at n={n} k={k} m={m}"
        );
        println!("forest     v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");
        if timed {
            time_ab(
                "forest",
                n,
                k,
                m,
                k + 4,
                a_r,
                &b_rs,
                o2_r,
                |d, a, b, o, m, n, k| d.launch_dot4_forest_gemv_v1_for_ab(a, b, o, m, n, k),
                |d, a, b, o, m, n, k| d.launch_dot4_forest_gemv_v2_for_ab(a, b, o, m, n, k),
            );
        }
    }

    // 5. Q8_0 (both `q80_q81` and `q80_f32act`)
    for &(n, k, m, timed) in &[
        (64usize, 256usize, 1usize, false),
        (68, 1024, 1, false),
        (68, 4096, 4, false),
        (67, 5120, 1, false),
        (17408, 5120, 1, true),
        (5120, 17408, 1, true),
    ] {
        let row_bytes = (k / 32) * 34;
        let mut rng = Lcg(0xA11CE005 ^ (n * 31 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        let mut b_q80 = vec![0u8; n * row_bytes];
        for col in 0..n {
            let w: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
            let p = grim_quant::quant_q80(&w).map_err(|e| e.to_string())?;
            b_q80[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&p);
        }
        let a_st =
            MemoryOps::from_cpu_bytes(&dev, f32_bytes(&a), &Shape::new(vec![m * k]), f32ty.clone())
                .map_err(|e| e.to_string())?;
        let q81_len = m * (k / 32) * 36;
        let q81_st = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; q81_len],
            &Shape::new(vec![q81_len]),
            u8ty.clone(),
        )
        .map_err(|e| e.to_string())?;
        dev.launch_quantize_q8_1(
            as_rocm(a_st.as_ref()).unwrap(),
            as_rocm(q81_st.as_ref()).unwrap(),
            m,
            k,
        )
        .map_err(|e| e.to_string())?;
        let copies = if timed { ROT } else { 1 };
        let b_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(
                    &dev,
                    &b_q80,
                    &Shape::new(vec![b_q80.len()]),
                    u8ty.clone(),
                )
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_f32_r = as_rocm(a_st.as_ref()).unwrap();
        let a_q81_r = as_rocm(q81_st.as_ref()).unwrap();
        let b_rs: Vec<_> = b_sts.iter().map(|s| as_rocm(s.as_ref()).unwrap()).collect();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());

        dev.launch_dot4_q80_q81_gemv_v1_for_ab(a_q81_r, b_rs[0], o1_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_dot4_q80_q81_gemv_v2_for_ab(a_q81_r, b_rs[0], o2_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_out, v2_out) = (f32_vec(&o1)?, f32_vec(&o2)?);
        let diffs = v1_out
            .iter()
            .zip(&v2_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "q80_q81_gemv_v2 not bit-identical at n={n} k={k} m={m}"
        );
        println!("q80_q81    v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");

        dev.launch_dot4_q80_f32act_gemv_v1_for_ab(a_f32_r, b_rs[0], o1_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_dot4_q80_f32act_gemv_v2_for_ab(a_f32_r, b_rs[0], o2_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_f32_out, v2_f32_out) = (f32_vec(&o1)?, f32_vec(&o2)?);
        let diffs_f32 = v1_f32_out
            .iter()
            .zip(&v2_f32_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs_f32, 0,
            "q80_f32act_gemv_v2 not bit-identical at n={n} k={k} m={m}"
        );
        println!("q80_f32act v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");

        if timed {
            time_ab(
                "q80_q81",
                n,
                k,
                m,
                row_bytes,
                a_q81_r,
                &b_rs,
                o2_r,
                |d, a, b, o, m, n, k| d.launch_dot4_q80_q81_gemv_v1_for_ab(a, b, o, m, n, k),
                |d, a, b, o, m, n, k| d.launch_dot4_q80_q81_gemv_v2_for_ab(a, b, o, m, n, k),
            );
            time_ab(
                "q80_f32act",
                n,
                k,
                m,
                row_bytes,
                a_f32_r,
                &b_rs,
                o2_r,
                |d, a, b, o, m, n, k| d.launch_dot4_q80_f32act_gemv_v1_for_ab(a, b, o, m, n, k),
                |d, a, b, o, m, n, k| d.launch_dot4_q80_f32act_gemv_v2_for_ab(a, b, o, m, n, k),
            );
        }
    }

    Ok(())
}

/// Corvid kernels `_v2` parity + warm/cold A/B benchmark:
/// - `WhiteCrow` / `CityCrow` (`grim_dot8_w4a4_gemv_v2`)
/// - `TreePie` (`grim_tree_pie_gemv_v2`)
/// - `ScrubJay` (`grim_scrub_jay_gemv_v2`)
/// - `GreyRaven` (`grim_grey_raven_gemm_v2`)
/// - `WhiteRaven` (`grim_wmma_gemm_fp8_e4m3_blocked_v2`)
#[test]
#[ignore = "requires AMD GPU; run with GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=1"]
fn corvid_gemv_v2_parity_then_ab() -> Result<(), String> {
    if std::env::var("GRIM_RUN_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set GRIM_RUN_GPU_TEST=1 to run");
        return Ok(());
    }
    let dev = RocmDevice::try_new(0).map_err(|e| format!("no ROCm device: {e}"))?;
    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let f16ty = DType {
        arith: ArithType::F16,
        storage: Storage::Native,
    };
    let bf16ty = DType {
        arith: ArithType::BF16,
        storage: Storage::Native,
    };
    let u32ty = DType {
        arith: ArithType::U32,
        storage: Storage::Native,
    };
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };

    const WARMUP: usize = 10;
    const ITERS: usize = 60;
    const ROT: usize = 4;

    println!(
        "\n=== Corvid Kernels v1 vs v2 Parity + A/B (device={}) ===",
        dev.gcn_arch()
    );
    println!(
        "{:<16} {:<15} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7}",
        "Kernel", "Shape (NxKxM)", "W v1", "W v2", "W v3", "C v1", "C v2", "C v3"
    );
    println!("{}", "-".repeat(98));

    // 1. WhiteCrow / CityCrow (`grim_dot8_w4a4_gemv` vs `grim_dot8_w4a4_gemv_v2`)
    for &(n, k, m, timed) in &[
        (64usize, 256usize, 1usize, false),
        (67, 512, 1, false),
        (68, 4096, 4, false),
        (65, 5120, 1, false),
        (17408, 5120, 1, true),
        (5120, 17408, 1, true),
    ] {
        let n_groups = k / 128;
        let words_per_col = k / 8;
        let row_bytes = words_per_col * 4 + n_groups * 2 + n_groups;
        let mut rng = Lcg(0xC0271D01 ^ (n * 31 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();
        let (qw_bytes, sc_bytes, zr_bytes) =
            grim_quant::quant_ostquant_w4_group128(&w, n, k).map_err(|e| e.to_string())?;

        let a_st =
            MemoryOps::from_cpu_bytes(&dev, f32_bytes(&a), &Shape::new(vec![m * k]), f32ty.clone())
                .map_err(|e| e.to_string())?;
        let act_codes =
            MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n_groups * 16]), u32ty.clone())
                .map_err(|e| e.to_string())?;
        let act_scales =
            MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n_groups]), f32ty.clone())
                .map_err(|e| e.to_string())?;
        let act_sums =
            MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n_groups]), u32ty.clone())
                .map_err(|e| e.to_string())?;
        let (ac_r, as_r, asum_r) = (
            as_rocm(act_codes.as_ref()).unwrap(),
            as_rocm(act_scales.as_ref()).unwrap(),
            as_rocm(act_sums.as_ref()).unwrap(),
        );
        dev.launch_quantize_u4_group128(as_rocm(a_st.as_ref()).unwrap(), ac_r, as_r, asum_r, m, k)
            .map_err(|e| e.to_string())?;

        let copies = if timed { ROT } else { 1 };
        let b_qw_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(
                    &dev,
                    &qw_bytes,
                    &Shape::new(vec![n * words_per_col]),
                    u32ty.clone(),
                )
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let b_sc_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(
                    &dev,
                    &sc_bytes,
                    &Shape::new(vec![n * n_groups]),
                    bf16ty.clone(),
                )
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let b_zr_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(
                    &dev,
                    &zr_bytes,
                    &Shape::new(vec![n * n_groups]),
                    u8ty.clone(),
                )
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;

        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o3 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let (o1_r, o2_r, o3_r) = (
            as_rocm(o1.as_ref()).unwrap(),
            as_rocm(o2.as_ref()).unwrap(),
            as_rocm(o3.as_ref()).unwrap(),
        );

        dev.launch_dot8_w4a4_gemv_v1_for_ab(
            ac_r,
            as_r,
            asum_r,
            as_rocm(b_qw_sts[0].as_ref()).unwrap(),
            as_rocm(b_sc_sts[0].as_ref()).unwrap(),
            as_rocm(b_zr_sts[0].as_ref()).unwrap(),
            o1_r,
            m,
            n,
            k,
        )
        .map_err(|e| e.to_string())?;
        dev.launch_dot8_w4a4_gemv_v2_for_ab(
            ac_r,
            as_r,
            asum_r,
            as_rocm(b_qw_sts[0].as_ref()).unwrap(),
            as_rocm(b_sc_sts[0].as_ref()).unwrap(),
            as_rocm(b_zr_sts[0].as_ref()).unwrap(),
            o2_r,
            m,
            n,
            k,
        )
        .map_err(|e| e.to_string())?;
        dev.launch_dot8_w4a4_gemv_v3_for_ab(
            ac_r,
            as_r,
            asum_r,
            as_rocm(b_qw_sts[0].as_ref()).unwrap(),
            as_rocm(b_sc_sts[0].as_ref()).unwrap(),
            as_rocm(b_zr_sts[0].as_ref()).unwrap(),
            o3_r,
            m,
            n,
            k,
        )
        .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_out, v2_out, v3_out) = (f32_vec(&o1)?, f32_vec(&o2)?, f32_vec(&o3)?);
        let diffs = v1_out
            .iter()
            .zip(&v2_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "whitecrow_v2 not bit-identical at n={n} k={k} m={m}"
        );
        println!("whitecrow      v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");
        let diffs3 = v1_out
            .iter()
            .zip(&v3_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs3, 0,
            "whitecrow_v3 not bit-identical to v1 at n={n} k={k} m={m}"
        );
        println!("whitecrow      v3 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");

        if timed {
            let bytes = (n * row_bytes + m * (n_groups * 64 + n_groups * 8) + m * n * 4) as f64;
            for _ in 0..WARMUP {
                dev.launch_dot8_w4a4_gemv_v1_for_ab(
                    ac_r,
                    as_r,
                    asum_r,
                    as_rocm(b_qw_sts[0].as_ref()).unwrap(),
                    as_rocm(b_sc_sts[0].as_ref()).unwrap(),
                    as_rocm(b_zr_sts[0].as_ref()).unwrap(),
                    o2_r,
                    m,
                    n,
                    k,
                )
                .unwrap();
                dev.launch_dot8_w4a4_gemv_v2_for_ab(
                    ac_r,
                    as_r,
                    asum_r,
                    as_rocm(b_qw_sts[0].as_ref()).unwrap(),
                    as_rocm(b_sc_sts[0].as_ref()).unwrap(),
                    as_rocm(b_zr_sts[0].as_ref()).unwrap(),
                    o2_r,
                    m,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                dev.launch_dot8_w4a4_gemv_v1_for_ab(
                    ac_r,
                    as_r,
                    asum_r,
                    as_rocm(b_qw_sts[0].as_ref()).unwrap(),
                    as_rocm(b_sc_sts[0].as_ref()).unwrap(),
                    as_rocm(b_zr_sts[0].as_ref()).unwrap(),
                    o2_r,
                    m,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let w_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                dev.launch_dot8_w4a4_gemv_v2_for_ab(
                    ac_r,
                    as_r,
                    asum_r,
                    as_rocm(b_qw_sts[0].as_ref()).unwrap(),
                    as_rocm(b_sc_sts[0].as_ref()).unwrap(),
                    as_rocm(b_zr_sts[0].as_ref()).unwrap(),
                    o2_r,
                    m,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let w_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                dev.launch_dot8_w4a4_gemv_v3_for_ab(
                    ac_r,
                    as_r,
                    asum_r,
                    as_rocm(b_qw_sts[0].as_ref()).unwrap(),
                    as_rocm(b_sc_sts[0].as_ref()).unwrap(),
                    as_rocm(b_zr_sts[0].as_ref()).unwrap(),
                    o2_r,
                    m,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let w_v3 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                let idx = i % ROT;
                dev.launch_dot8_w4a4_gemv_v1_for_ab(
                    ac_r,
                    as_r,
                    asum_r,
                    as_rocm(b_qw_sts[idx].as_ref()).unwrap(),
                    as_rocm(b_sc_sts[idx].as_ref()).unwrap(),
                    as_rocm(b_zr_sts[idx].as_ref()).unwrap(),
                    o2_r,
                    m,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let c_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                let idx = i % ROT;
                dev.launch_dot8_w4a4_gemv_v2_for_ab(
                    ac_r,
                    as_r,
                    asum_r,
                    as_rocm(b_qw_sts[idx].as_ref()).unwrap(),
                    as_rocm(b_sc_sts[idx].as_ref()).unwrap(),
                    as_rocm(b_zr_sts[idx].as_ref()).unwrap(),
                    o2_r,
                    m,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let c_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                let idx = i % ROT;
                dev.launch_dot8_w4a4_gemv_v3_for_ab(
                    ac_r,
                    as_r,
                    asum_r,
                    as_rocm(b_qw_sts[idx].as_ref()).unwrap(),
                    as_rocm(b_sc_sts[idx].as_ref()).unwrap(),
                    as_rocm(b_zr_sts[idx].as_ref()).unwrap(),
                    o2_r,
                    m,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let c_v3 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            println!(
                "{:<16} {:<15} {:>7.1} {:>7.1} {:>7.1} | {:>7.1} {:>7.1} {:>7.1}",
                "whitecrow",
                format!("{n}x{k}x{m}"),
                w_v1,
                w_v2,
                w_v3,
                c_v1,
                c_v2,
                c_v3
            );
        }
    }

    // 2. TreePie (`grim_tree_pie_gemv` vs `grim_tree_pie_gemv_v2`)
    for &(n, k, timed) in &[
        (64usize, 256usize, false),
        (67, 512, false),
        (65, 5120, false),
        (17408, 5120, true),
        (5120, 17408, true),
    ] {
        let mut rng = Lcg(0xC0271D02 ^ (n * 31 + k) as u64);
        let a_f16: Vec<u16> = (0..k)
            .map(|_| half::f16::from_f32(rng.next_f32()).to_bits())
            .collect();
        let w_f32: Vec<f32> = (0..n * k).map(|_| rng.next_f32() * 6.0).collect();
        let packed_i32 = grim_quant::tree_pie::pack_tree_pie(&w_f32);
        let packed_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(packed_i32.as_ptr() as *const u8, packed_i32.len() * 4)
        };
        let a_bytes: Vec<u8> = a_f16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let a_st = MemoryOps::from_cpu_bytes(&dev, &a_bytes, &Shape::new(vec![k]), f16ty.clone())
            .map_err(|e| e.to_string())?;
        let copies = if timed { ROT } else { 1 };
        let b_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(
                    &dev,
                    packed_bytes,
                    &Shape::new(vec![packed_i32.len()]),
                    u32ty.clone(),
                )
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_rs: Vec<_> = b_sts.iter().map(|s| as_rocm(s.as_ref()).unwrap()).collect();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());

        dev.launch_tree_pie_gemv_v1_for_ab(a_r, b_rs[0], o1_r, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_tree_pie_gemv_v2_for_ab(a_r, b_rs[0], o2_r, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_out, v2_out) = (f32_vec(&o1)?, f32_vec(&o2)?);
        let diffs = v1_out
            .iter()
            .zip(&v2_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "tree_pie_gemv_v2 not bit-identical at n={n} k={k}"
        );
        println!("tree_pie       v2 parity n={n:<5} k={k:<5} m=1: bit-identical to v1 (0 diffs)");

        if timed {
            let row_bytes = (k / 32) * 20;
            let bytes = (n * row_bytes + k * 2 + n * 4) as f64;
            for _ in 0..WARMUP {
                dev.launch_tree_pie_gemv_v1_for_ab(a_r, b_rs[0], o2_r, n, k)
                    .unwrap();
                dev.launch_tree_pie_gemv_v2_for_ab(a_r, b_rs[0], o2_r, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                dev.launch_tree_pie_gemv_v1_for_ab(a_r, b_rs[0], o2_r, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let w_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                dev.launch_tree_pie_gemv_v2_for_ab(a_r, b_rs[0], o2_r, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let w_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                dev.launch_tree_pie_gemv_v1_for_ab(a_r, b_rs[i % ROT], o2_r, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let c_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                dev.launch_tree_pie_gemv_v2_for_ab(a_r, b_rs[i % ROT], o2_r, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let c_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            println!(
                "{:<16} {:<15} {:>7.1} GB/s {:>7.1} GB/s {:>7.3}x | {:>7.1} GB/s {:>7.1} GB/s {:>7.3}x",
                "tree_pie",
                format!("{n}x{k}x1"),
                w_v1,
                w_v2,
                w_v2 / w_v1,
                c_v1,
                c_v2,
                c_v2 / c_v1
            );
        }
    }

    // 3. ScrubJay (`grim_scrub_jay_gemv` vs `grim_scrub_jay_gemv_v2`)
    for &(n, k, timed) in &[
        (64usize, 256usize, false),
        (67, 512, false),
        (65, 5120, false),
        (8192, 5120, true),
        (5120, 8192, true),
    ] {
        let nbc = k / 8;
        let mut rng = Lcg(0xC0271D03 ^ (n * 31 + k) as u64);
        let a: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
        let sel: Vec<u8> = (0..n * nbc).map(|i| (i % 16) as u8).collect();
        let idx: Vec<u8> = (0..n * k).map(|i| ((i * 7 + 3) % 16) as u8).collect();
        let sgn: Vec<u8> = (0..n * nbc).map(|i| ((i * 13 + 5) & 0xFF) as u8).collect();
        let scl: Vec<f32> = (0..n * nbc)
            .map(|_| rng.next_f32().abs() * 0.1 + 0.01)
            .collect();

        let a_st =
            MemoryOps::from_cpu_bytes(&dev, f32_bytes(&a), &Shape::new(vec![k]), f32ty.clone())
                .map_err(|e| e.to_string())?;
        let copies = if timed { ROT } else { 1 };
        let sel_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(&dev, &sel, &Shape::new(vec![sel.len()]), u8ty.clone())
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let idx_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(&dev, &idx, &Shape::new(vec![idx.len()]), u8ty.clone())
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let sgn_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(&dev, &sgn, &Shape::new(vec![sgn.len()]), u8ty.clone())
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let scl_sts: Vec<_> = (0..copies)
            .map(|_| {
                MemoryOps::from_cpu_bytes(
                    &dev,
                    f32_bytes(&scl),
                    &Shape::new(vec![scl.len()]),
                    f32ty.clone(),
                )
            })
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());

        dev.launch_scrub_jay_gemv_v1_for_ab(
            a_r,
            as_rocm(sel_sts[0].as_ref()).unwrap(),
            as_rocm(idx_sts[0].as_ref()).unwrap(),
            as_rocm(sgn_sts[0].as_ref()).unwrap(),
            as_rocm(scl_sts[0].as_ref()).unwrap(),
            o1_r,
            1.0,
            n,
            k,
        )
        .map_err(|e| e.to_string())?;
        dev.launch_scrub_jay_gemv_v2_for_ab(
            a_r,
            as_rocm(sel_sts[0].as_ref()).unwrap(),
            as_rocm(idx_sts[0].as_ref()).unwrap(),
            as_rocm(sgn_sts[0].as_ref()).unwrap(),
            as_rocm(scl_sts[0].as_ref()).unwrap(),
            o2_r,
            1.0,
            n,
            k,
        )
        .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_out, v2_out) = (f32_vec(&o1)?, f32_vec(&o2)?);
        let diffs = v1_out
            .iter()
            .zip(&v2_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "scrub_jay_gemv_v2 not bit-identical at n={n} k={k}"
        );
        println!("scrub_jay      v2 parity n={n:<5} k={k:<5} m=1: bit-identical to v1 (0 diffs)");

        if timed {
            // ScrubJay planes read from VRAM: B_sel (1B/8) + B_sgn (1B/8) + B_scl (4B/8) + B_idx (8B/8) = 14B per 8 values
            let row_bytes = nbc * 14;
            let bytes = (n * row_bytes + k * 4 + n * 4) as f64;
            for _ in 0..WARMUP {
                dev.launch_scrub_jay_gemv_v1_for_ab(
                    a_r,
                    as_rocm(sel_sts[0].as_ref()).unwrap(),
                    as_rocm(idx_sts[0].as_ref()).unwrap(),
                    as_rocm(sgn_sts[0].as_ref()).unwrap(),
                    as_rocm(scl_sts[0].as_ref()).unwrap(),
                    o2_r,
                    1.0,
                    n,
                    k,
                )
                .unwrap();
                dev.launch_scrub_jay_gemv_v2_for_ab(
                    a_r,
                    as_rocm(sel_sts[0].as_ref()).unwrap(),
                    as_rocm(idx_sts[0].as_ref()).unwrap(),
                    as_rocm(sgn_sts[0].as_ref()).unwrap(),
                    as_rocm(scl_sts[0].as_ref()).unwrap(),
                    o2_r,
                    1.0,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                dev.launch_scrub_jay_gemv_v1_for_ab(
                    a_r,
                    as_rocm(sel_sts[0].as_ref()).unwrap(),
                    as_rocm(idx_sts[0].as_ref()).unwrap(),
                    as_rocm(sgn_sts[0].as_ref()).unwrap(),
                    as_rocm(scl_sts[0].as_ref()).unwrap(),
                    o2_r,
                    1.0,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let w_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                dev.launch_scrub_jay_gemv_v2_for_ab(
                    a_r,
                    as_rocm(sel_sts[0].as_ref()).unwrap(),
                    as_rocm(idx_sts[0].as_ref()).unwrap(),
                    as_rocm(sgn_sts[0].as_ref()).unwrap(),
                    as_rocm(scl_sts[0].as_ref()).unwrap(),
                    o2_r,
                    1.0,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let w_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                let r = i % ROT;
                dev.launch_scrub_jay_gemv_v1_for_ab(
                    a_r,
                    as_rocm(sel_sts[r].as_ref()).unwrap(),
                    as_rocm(idx_sts[r].as_ref()).unwrap(),
                    as_rocm(sgn_sts[r].as_ref()).unwrap(),
                    as_rocm(scl_sts[r].as_ref()).unwrap(),
                    o2_r,
                    1.0,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let c_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                let r = i % ROT;
                dev.launch_scrub_jay_gemv_v2_for_ab(
                    a_r,
                    as_rocm(sel_sts[r].as_ref()).unwrap(),
                    as_rocm(idx_sts[r].as_ref()).unwrap(),
                    as_rocm(sgn_sts[r].as_ref()).unwrap(),
                    as_rocm(scl_sts[r].as_ref()).unwrap(),
                    o2_r,
                    1.0,
                    n,
                    k,
                )
                .unwrap();
            }
            dev.synchronize();
            let c_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            println!(
                "{:<16} {:<15} {:>7.1} GB/s {:>7.1} GB/s {:>7.3}x | {:>7.1} GB/s {:>7.1} GB/s {:>7.3}x",
                "scrub_jay",
                format!("{n}x{k}x1"),
                w_v1,
                w_v2,
                w_v2 / w_v1,
                c_v1,
                c_v2,
                c_v2 / c_v1
            );
        }
    }

    // 4. GreyRaven (`grim_grey_raven_gemm` vs `grim_grey_raven_gemm_v2`)
    // Note: SWMMAC (`v_swmmac_f32_16x16x32_fp8_fp8_w32`) requires PCIe atomics on gfx12
    // and is not supported by the AQL command processor on secondary discrete GPUs (e.g. Card 1).
    // We probe support first.
    let grey_raven_supported = {
        let n = 32usize;
        let k = 64usize;
        let m = 1usize;
        let n_tiles = n.div_ceil(16);
        let n_windows = k.div_ceil(32);
        let w_bytes_len = n_tiles * n_windows * 260;
        let b_frag_len = 1 * n_windows * 512;
        let w_blob = vec![0u8; w_bytes_len];
        let b_frag = vec![0u8; b_frag_len];
        let w_st =
            MemoryOps::from_cpu_bytes(&dev, &w_blob, &Shape::new(vec![w_blob.len()]), u8ty.clone())
                .unwrap();
        let b_st =
            MemoryOps::from_cpu_bytes(&dev, &b_frag, &Shape::new(vec![b_frag.len()]), u8ty.clone())
                .unwrap();
        let o_st = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone()).unwrap();
        let res = dev.launch_grey_raven_gemm_v1_for_ab(
            as_rocm(w_st.as_ref()).unwrap(),
            as_rocm(b_st.as_ref()).unwrap(),
            as_rocm(o_st.as_ref()).unwrap(),
            m,
            n,
            k,
            n_tiles,
            n_windows,
        );
        match res {
            Ok(_) => {
                let _ = dev.synchronize();
                true
            }
            Err(e) => {
                println!(
                    "grey_raven     SKIPPED: AQL dispatch rejected ({e}) - SWMMAC requires PCIe atomics not present on this device"
                );
                false
            }
        }
    };

    if grey_raven_supported {
        for &(n, k, m, timed) in &[
            (32usize, 64usize, 1usize, false),
            (48, 96, 1, false),
            (64, 256, 16, false),
            (17408, 5120, 1, true),
            (17408, 5120, 16, true),
        ] {
            let n_tiles = n.div_ceil(16);
            let n_windows = k.div_ceil(32);
            let m_tiles = m.div_ceil(16);
            let w_bytes_len = n_tiles * n_windows * 260;
            let b_frag_len = m_tiles * n_windows * 512;

            // Valid 2:4 patterns: pair p in 0..4 gets slots (0,1), (0,2), (1,2), (1,3)
            let sidx_word: u32 = (0 | (1 << 2))
                | ((0 | (2 << 2)) << 4)
                | ((1 | (2 << 2)) << 8)
                | ((1 | (3 << 2)) << 12);
            let mut w_blob = vec![0u8; w_bytes_len];
            for t in 0..(n_tiles * n_windows) {
                let base = t * 260;
                for i in 0..256 {
                    w_blob[base + i] = (((t * 17 + i * 5) % 64) + 0x20) as u8;
                }
                w_blob[base + 256..base + 260].copy_from_slice(&sidx_word.to_le_bytes());
            }
            let b_frag: Vec<u8> = (0..b_frag_len)
                .map(|i| (((i * 11 + 3) % 64) + 0x20) as u8)
                .collect();

            let b_st = MemoryOps::from_cpu_bytes(
                &dev,
                &b_frag,
                &Shape::new(vec![b_frag.len()]),
                u8ty.clone(),
            )
            .map_err(|e| e.to_string())?;
            let copies = if timed { ROT } else { 1 };
            let w_sts: Vec<_> = (0..copies)
                .map(|_| {
                    MemoryOps::from_cpu_bytes(
                        &dev,
                        &w_blob,
                        &Shape::new(vec![w_blob.len()]),
                        u8ty.clone(),
                    )
                })
                .collect::<grim_tensor::Result<_>>()
                .map_err(|e| e.to_string())?;
            let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
                .map_err(|e| e.to_string())?;
            let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
                .map_err(|e| e.to_string())?;
            let b_r = as_rocm(b_st.as_ref()).unwrap();
            let w_rs: Vec<_> = w_sts.iter().map(|s| as_rocm(s.as_ref()).unwrap()).collect();
            let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());

            dev.launch_grey_raven_gemm_v1_for_ab(w_rs[0], b_r, o1_r, m, n, k, n_tiles, n_windows)
                .map_err(|e| e.to_string())?;
            dev.launch_grey_raven_gemm_v2_for_ab(w_rs[0], b_r, o2_r, m, n, k, n_tiles, n_windows)
                .map_err(|e| e.to_string())?;
            dev.synchronize();
            let (v1_out, v2_out) = (f32_vec(&o1)?, f32_vec(&o2)?);
            let diffs = v1_out
                .iter()
                .zip(&v2_out)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                diffs, 0,
                "grey_raven_gemm_v2 not bit-identical at n={n} k={k} m={m}"
            );
            println!(
                "grey_raven     v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)"
            );

            if timed {
                let bytes = (w_bytes_len + b_frag_len + m * n * 4) as f64;
                for _ in 0..WARMUP {
                    dev.launch_grey_raven_gemm_v1_for_ab(
                        w_rs[0], b_r, o2_r, m, n, k, n_tiles, n_windows,
                    )
                    .unwrap();
                    dev.launch_grey_raven_gemm_v2_for_ab(
                        w_rs[0], b_r, o2_r, m, n, k, n_tiles, n_windows,
                    )
                    .unwrap();
                }
                dev.synchronize();
                let t0 = std::time::Instant::now();
                for _ in 0..ITERS {
                    dev.launch_grey_raven_gemm_v1_for_ab(
                        w_rs[0], b_r, o2_r, m, n, k, n_tiles, n_windows,
                    )
                    .unwrap();
                }
                dev.synchronize();
                let w_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

                let t0 = std::time::Instant::now();
                for _ in 0..ITERS {
                    dev.launch_grey_raven_gemm_v2_for_ab(
                        w_rs[0], b_r, o2_r, m, n, k, n_tiles, n_windows,
                    )
                    .unwrap();
                }
                dev.synchronize();
                let w_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

                let t0 = std::time::Instant::now();
                for i in 0..ITERS {
                    dev.launch_grey_raven_gemm_v1_for_ab(
                        w_rs[i % ROT],
                        b_r,
                        o2_r,
                        m,
                        n,
                        k,
                        n_tiles,
                        n_windows,
                    )
                    .unwrap();
                }
                dev.synchronize();
                let c_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

                let t0 = std::time::Instant::now();
                for i in 0..ITERS {
                    dev.launch_grey_raven_gemm_v2_for_ab(
                        w_rs[i % ROT],
                        b_r,
                        o2_r,
                        m,
                        n,
                        k,
                        n_tiles,
                        n_windows,
                    )
                    .unwrap();
                }
                dev.synchronize();
                let c_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

                println!(
                    "{:<16} {:<15} {:>7.1} GB/s {:>7.1} GB/s {:>7.3}x | {:>7.1} GB/s {:>7.1} GB/s {:>7.3}x",
                    "grey_raven",
                    format!("{n}x{k}x{m}"),
                    w_v1,
                    w_v2,
                    w_v2 / w_v1,
                    c_v1,
                    c_v2,
                    c_v2 / c_v1
                );
            }
        }
    }

    // 5. WhiteRaven blocked (`grim_wmma_gemm_fp8_e4m3_blocked` vs `grim_wmma_gemm_fp8_e4m3_blocked_v2`)
    for &(n, k, m, timed) in &[
        (32usize, 64usize, 1usize, false),
        (48, 96, 1, false),
        (64, 256, 16, false),
        (17408, 5120, 1, true),
        (17408, 5120, 16, true),
    ] {
        let m_pad = m.div_ceil(16) * 16;
        let a_fp8: Vec<u8> = (0..m_pad * k)
            .map(|i| (((i * 7 + 3) % 64) + 0x20) as u8)
            .collect();
        let b_fp8: Vec<u8> = (0..n * k)
            .map(|i| (((i * 13 + 5) % 64) + 0x20) as u8)
            .collect();
        let a_st =
            MemoryOps::from_cpu_bytes(&dev, &a_fp8, &Shape::new(vec![m_pad, k]), u8ty.clone())
                .map_err(|e| e.to_string())?;
        let copies = if timed { ROT } else { 1 };
        let b_sts: Vec<_> = (0..copies)
            .map(|_| MemoryOps::from_cpu_bytes(&dev, &b_fp8, &Shape::new(vec![n, k]), u8ty.clone()))
            .collect::<grim_tensor::Result<_>>()
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m, n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m, n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_rs: Vec<_> = b_sts.iter().map(|s| as_rocm(s.as_ref()).unwrap()).collect();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());

        dev.launch_wmma_gemm_fp8_e4m3_blocked_v1_for_ab(a_r, b_rs[0], o1_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_wmma_gemm_fp8_e4m3_blocked_v2_for_ab(a_r, b_rs[0], o2_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let (v1_out, v2_out) = (f32_vec(&o1)?, f32_vec(&o2)?);
        let diffs = v1_out
            .iter()
            .zip(&v2_out)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "white_raven_blocked_v2 not bit-identical at n={n} k={k} m={m}"
        );
        println!("white_raven    v2 parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");

        if timed {
            let bytes = (n * k + m_pad * k + m * n * 4) as f64;
            for _ in 0..WARMUP {
                dev.launch_wmma_gemm_fp8_e4m3_blocked_v1_for_ab(a_r, b_rs[0], o2_r, m, n, k)
                    .unwrap();
                dev.launch_wmma_gemm_fp8_e4m3_blocked_v2_for_ab(a_r, b_rs[0], o2_r, m, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                dev.launch_wmma_gemm_fp8_e4m3_blocked_v1_for_ab(a_r, b_rs[0], o2_r, m, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let w_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                dev.launch_wmma_gemm_fp8_e4m3_blocked_v2_for_ab(a_r, b_rs[0], o2_r, m, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let w_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                dev.launch_wmma_gemm_fp8_e4m3_blocked_v1_for_ab(a_r, b_rs[i % ROT], o2_r, m, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let c_v1 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                dev.launch_wmma_gemm_fp8_e4m3_blocked_v2_for_ab(a_r, b_rs[i % ROT], o2_r, m, n, k)
                    .unwrap();
            }
            dev.synchronize();
            let c_v2 = bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9;

            println!(
                "{:<16} {:<15} {:>7.1} GB/s {:>7.1} GB/s {:>7.3}x | {:>7.1} GB/s {:>7.1} GB/s {:>7.3}x",
                "white_raven",
                format!("{n}x{k}x{m}"),
                w_v1,
                w_v2,
                w_v2 / w_v1,
                c_v1,
                c_v2,
                c_v2 / c_v1
            );
        }
    }

    Ok(())
}
