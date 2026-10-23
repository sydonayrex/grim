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
fn scheme(name: &str) -> (usize, fn(&[f32]) -> grim_tensor::Result<Vec<u8>>, fn(&[u8], usize) -> grim_tensor::Result<Vec<f32>>) {
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
        nonzero, m * n,
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
        fn(&RocmDevice, &grim_backend_rocm::RocmStorage, &grim_backend_rocm::RocmStorage, &grim_backend_rocm::RocmStorage, usize, usize, usize) -> Result<*mut std::ffi::c_void, String>,
        fn(&RocmDevice, &grim_backend_rocm::RocmStorage, &grim_backend_rocm::RocmStorage, &grim_backend_rocm::RocmStorage, usize, usize, usize) -> Result<*mut std::ffi::c_void, String>,
    ) = match scheme_name {
        "q4k" => (
            |d, a, b, o, m, n, k| d.launch_dot4_q4k_q81_gemv_stock_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
            |d, a, b, o, m, n, k| d.launch_dot4_q4k_q81_gemv_ref_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
        ),
        "q5k" => (
            |d, a, b, o, m, n, k| d.launch_dot4_q5k_q81_gemv_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
            |d, a, b, o, m, n, k| d.launch_dot4_q5k_q81_gemv_ref_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
        ),
        // Q2_K: "new" is the word-wide-mask variant (the default through the
        // launcher), "ref" is the stock 4-byte-load kernel. Both are driven
        // through launchers so the grid is the dispatch's own.
        "q2k" => (
            |d, a, b, o, m, n, k| d.launch_dot4_q2k_q81_gemv_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
            |d, a, b, o, m, n, k| d.launch_dot4_q2k_q81_gemv_stock_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
        ),
        "iq4xs" => (
            |d, a, b, o, m, n, k| d.launch_dot4_iq4xs_q81_gemv_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
            |d, a, b, o, m, n, k| d.launch_dot4_iq4xs_q81_gemv_stock_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
        ),
        "q3k" => (
            |d, a, b, o, m, n, k| d.launch_dot4_q3k_q81_gemv_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
            |d, a, b, o, m, n, k| d.launch_dot4_q3k_q81_gemv_stock_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
        ),
        _ => (
            |d, a, b, o, m, n, k| d.launch_dot4_q6k_q81_gemv_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
            |d, a, b, o, m, n, k| d.launch_dot4_q6k_q81_gemv_ref_for_ab(a, b, o, m, n, k).map_err(|e| e.to_string()),
        ),
    };

    let time_it = |f: &dyn Fn(&RocmDevice, &grim_backend_rocm::RocmStorage, &grim_backend_rocm::RocmStorage, &grim_backend_rocm::RocmStorage, usize, usize, usize) -> Result<*mut std::ffi::c_void, String>| -> f64 {
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
            dev.launch_dot4_iq4xs_q81_gemv_for_ab(a_r, b_r, o_r, m, n, k)
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