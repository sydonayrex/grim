//! M7.2e: WHY is Q6_K's dot4 leg disabled?
//!
//! The dispatch claims the Q6_K dot4 GEMV "is not bit-exact at production
//! shapes" and cites one number: the Xing4.0-29B lm_head (real
//! output.weight, k=3584) at rel 3.2e-3 / max_abs 1.5 at m=1, while the
//! fused-dequant path agrees with the host dequant at 3.9e-7.
//!
//! That single comparison cannot tell three different things apart, so this
//! probe separates them, on the REAL 27B tensors:
//!
//!   A. the KERNEL's own error   — dot4 vs the host dequant dotted with the
//!                                 SAME q8_1 codes the kernel consumed. This
//!                                 is the only comparison that can find a
//!                                 bug in the decode.
//!   B. the ACTIVATION quantization — dot4 vs the host dequant dotted with
//!                                 the ORIGINAL f32 activation.
//!   C. the SUMMATION order      — the exact (f32) path vs the same oracle.
//!
//! It runs the same three for Q5_K and Q4_K, whose dot4 legs are ENABLED,
//! so the verdict is comparative rather than absolute.
//!
//! Usage: HIP_VISIBLE_DEVICES=1 ./target/release/examples/kquant_dot4_vs_exact_probe \
//!            /tmp/kqf/q6k_output_weight.bin q6k 4096 5120

use grim_backend_rocm::{RocmDevice, as_rocm};
use grim_tensor::{ArithType, DType, MemoryOps, Shape, Storage};
use std::time::Instant;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: kquant_dot4_vs_exact_probe <weights.bin> <q4k|q5k|q6k> [n] [k]");
        std::process::exit(2);
    });
    let scheme = args.get(2).cloned().unwrap_or_else(|| "q6k".into());
    let n: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let k: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5120);

    let (bpb, dequant): (usize, fn(&[u8], usize) -> grim_tensor::Result<Vec<f32>>) = match scheme.as_str()
    {
        "q4k" => (144, grim_quant::dequant_q4k),
        "q5k" => (176, grim_quant::dequant_q5k),
        "q6k" => (210, grim_quant::dequant_q6k),
        other => {
            eprintln!("unknown scheme {other}");
            std::process::exit(2);
        }
    };
    let row_bytes = (k / 256) * bpb;

    let bytes = std::fs::read(&path).expect("read weights");
    assert!(
        bytes.len() >= n * row_bytes,
        "file holds {} bytes, need {} for n={n} k={k}",
        bytes.len(),
        n * row_bytes
    );
    let bytes = &bytes[..n * row_bytes];

    let dev = RocmDevice::try_new(0).expect("try_new");
    let mut rng = Lcg(0x1234_5678);
    let act: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();

    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    let a_st = MemoryOps::from_cpu_bytes(
        &dev,
        unsafe { std::slice::from_raw_parts(act.as_ptr() as *const u8, act.len() * 4) },
        &Shape::new(vec![k]),
        f32ty,
    )
    .expect("a h2d");
    let q81_len = (k / 32) * 36;
    let q81_st = MemoryOps::from_cpu_bytes(
        &dev,
        &vec![0u8; q81_len],
        &Shape::new(vec![q81_len]),
        u8ty.clone(),
    )
    .expect("q81 alloc");
    dev.launch_quantize_q8_1(
        as_rocm(a_st.as_ref()).unwrap(),
        as_rocm(q81_st.as_ref()).unwrap(),
        1,
        k,
    )
    .expect("quantize");
    dev.synchronize();
    let q81_bytes = as_rocm(q81_st.as_ref())
        .unwrap()
        .copy_to_host()
        .expect("q81 d2h");
    let act_q81 = dequant_q81(&q81_bytes, k);

    let b_st = MemoryOps::from_cpu_bytes(&dev, bytes, &Shape::new(vec![n * k]), u8ty)
        .expect("b h2d");
    let mk_out = || {
        MemoryOps::alloc_storage(
            &dev,
            &Shape::new(vec![n]),
            DType {
                arith: ArithType::F32,
                storage: Storage::Native,
            },
        )
        .expect("out alloc")
    };
    let out_dot4 = mk_out();
    let out_exact = mk_out();

    let a_r = as_rocm(a_st.as_ref()).unwrap();
    let b_r = as_rocm(b_st.as_ref()).unwrap();
    let o_dot4 = as_rocm(out_dot4.as_ref()).unwrap();
    let o_exact = as_rocm(out_exact.as_ref()).unwrap();

    // Warm up: grim JITs a kernel on its first launch, and a cold launch
    // measures the compile, not the GEMV.
    for _ in 0..2 {
        match scheme.as_str() {
        "q4k" => {
            dev.launch_dot4_q4k_q81_gemv_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                b_r,
                o_dot4,
                1,
                n,
                k,
            )
            .expect("dot4 q4k");
        }
        "q5k" => {
            dev.launch_dot4_q5k_q81_gemv_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                b_r,
                o_dot4,
                1,
                n,
                k,
            )
            .expect("dot4 q5k");
        }
        _ => {
            dev.launch_dot4_q6k_q81_gemv_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                b_r,
                o_dot4,
                1,
                n,
                k,
            )
            .expect("dot4 q6k");
        }
        }
    }
    dev.synchronize();
    let t0 = Instant::now();
    let iters = 10;
    for _ in 0..iters {
        match scheme.as_str() {
        "q4k" => { dev.launch_dot4_q4k_q81_gemv_for_ab(as_rocm(q81_st.as_ref()).unwrap(), b_r, o_dot4, 1, n, k).expect("d"); }
        "q5k" => { dev.launch_dot4_q5k_q81_gemv_for_ab(as_rocm(q81_st.as_ref()).unwrap(), b_r, o_dot4, 1, n, k).expect("d"); }
        _ => { dev.launch_dot4_q6k_q81_gemv_for_ab(as_rocm(q81_st.as_ref()).unwrap(), b_r, o_dot4, 1, n, k).expect("d"); }
        }
    }
    dev.synchronize();
    let t_dot4 = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    for _ in 0..2 {
    match scheme.as_str() {
        "q4k" => {
            dev.launch_fused_dequant_gemm_q4k_for_ab(a_r, b_r, o_exact, 1, n, k)
                .expect("exact q4k");
        }
        "q5k" => {
            dev.launch_fused_dequant_gemm_q5k_for_ab(a_r, b_r, o_exact, 1, n, k)
                .expect("exact q5k");
        }
        _ => {
            dev.launch_fused_dequant_gemm_q6k_for_ab(a_r, b_r, o_exact, 1, n, k)
                .expect("exact q6k");
        }
    }
    }
    dev.synchronize();
    let t0 = Instant::now();
    for _ in 0..iters {
        match scheme.as_str() {
        "q4k" => { dev.launch_fused_dequant_gemm_q4k_for_ab(a_r, b_r, o_exact, 1, n, k).expect("e"); }
        "q5k" => { dev.launch_fused_dequant_gemm_q5k_for_ab(a_r, b_r, o_exact, 1, n, k).expect("e"); }
        _ => { dev.launch_fused_dequant_gemm_q6k_for_ab(a_r, b_r, o_exact, 1, n, k).expect("e"); }
        }
    }
    dev.synchronize();
    let t_exact = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let got_dot4 = as_rocm(out_dot4.as_ref())
        .unwrap()
        .copy_to_host()
        .expect("d2h dot4");
    let got_exact = as_rocm(out_exact.as_ref())
        .unwrap()
        .copy_to_host()
        .expect("d2h exact");
    let g = |b: &[u8], i: usize| f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);

    // The host oracle, per column.
    let mut a_kernel = 0.0f64; // dot4 vs oracle(q81 codes)
    let mut b_act = 0.0f64; // dot4 vs oracle(f32 activation)
    let mut c_sum = 0.0f64; // exact vs oracle(f32 activation)
    let mut a_absmax = 0.0f64;
    let mut b_absmax = 0.0f64;
    let mut c_absmax = 0.0f64;
    let mut mag = 0.0f64;
    let mut a_sq = 0.0f64;
    let mut b_sq = 0.0f64;
    let mut c_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for col in 0..n {
        let w = dequant(&bytes[col * row_bytes..(col + 1) * row_bytes], k).expect("dequant");
        let mut acc_f = 0.0f64;
        let mut acc_q = 0.0f64;
        for kk in 0..k {
            acc_f += act[kk] as f64 * w[kk] as f64;
            acc_q += act_q81[kk] as f64 * w[kk] as f64;
        }
        let d = g(&got_dot4, col);
        let e = g(&got_exact, col);
        let a_e = (d as f64 - acc_q).abs();
        let b_e = (d as f64 - acc_f).abs();
        let c_e = (e as f64 - acc_f).abs();
        a_kernel = a_kernel.max(a_e);
        b_act = b_act.max(b_e);
        c_sum = c_sum.max(c_e);
        a_absmax = a_absmax.max(a_e);
        b_absmax = b_absmax.max(b_e);
        c_absmax = c_absmax.max(c_e);
        a_sq += a_e * a_e;
        b_sq += b_e * b_e;
        c_sq += c_e * c_e;
        ref_sq += acc_f * acc_f;
        mag = mag.max(acc_f.abs());
    }
    let bytes_total = (n * row_bytes) as f64;
    println!("=== {scheme}  n={n} k={k}  (real {path})");
    let rms = (ref_sq / n as f64).sqrt().max(1e-12);
    let rms_rel = |sq: f64| (sq / n as f64).sqrt() / rms;
    println!(
        "    A. kernel's own error   dot4 vs host-dequant(q81 codes): rms-rel {:.3e}  absmax {:.3e}",
        rms_rel(a_sq), a_absmax
    );
    println!(
        "    B. + activation quant   dot4 vs host-dequant(f32 act)  : rms-rel {:.3e}  absmax {:.3e}",
        rms_rel(b_sq), b_absmax
    );
    println!(
        "    C. summation order      exact vs host-dequant(f32 act) : rms-rel {:.3e}  absmax {:.3e}",
        rms_rel(c_sq), c_absmax
    );
    println!(
        "    (logit magnitude max {mag:.3e})   dot4 {t_dot4:.3} ms ({:.1} GB/s)  exact {t_exact:.3} ms ({:.1} GB/s)",
        bytes_total / t_dot4 / 1e6,
        bytes_total / t_exact / 1e6
    );
}
