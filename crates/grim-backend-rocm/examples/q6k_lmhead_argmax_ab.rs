//! M7.2e: does the Q6_K dot4 leg change the LM head's ANSWER?
//!
//! The Q6_K dot4 leg is now on by default. Its kernel is exact (the host
//! dequant with the same q8_1 codes agrees to ~1e-7), but the leg quantizes
//! the activation to Q8_1, and that quantization is worth ~3.6e-3 rms
//! relative on the output. The question that decides whether the leg is
//! safe is not "is the kernel right" but "does the head still pick the same
//! token".
//!
//! This runs the REAL 27B Q6_K head (output.weight: n=248320, k=5120, 1.04
//! GB) over a batch of realistic final hidden states, and compares the two
//! paths' LOGITS: the argmax (the token the head picks), the top-1 margin,
//! and the error scaled by that margin.
//!
//! Usage: HIP_VISIBLE_DEVICES=1 ./target/release/examples/q6k_lmhead_argmax_ab \
//!            /tmp/kqf/q6k_real.bin [n_hidden_states] [k] [h_rms]

use grim_backend_rocm::{RocmDevice, as_rocm};
use grim_tensor::{ArithType, DType, MemoryOps, Shape, Storage};

struct Lcg {
    s: u64,
    rms: f32,
}
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.s = self
            .s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // A roughly normal sample, scaled to the requested RMS.
        let u1 = ((self.s >> 11) as f64 / (1u64 << 53) as f64).max(1e-9);
        let u2 = ((self.s >> 7) as f64 / (1u64 << 53) as f64) % 1.0;
        let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        (g as f32) * self.rms
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/kqf/q6k_real.bin".into());
    let n_hidden: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5120);
    let h_rms: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let row_bytes = (k / 256) * 210;
    let bytes = std::fs::read(&path).expect("read head");
    let n = bytes.len() / row_bytes;
    println!(
        "[q6k-head] {} bytes -> n={n} k={k} ({} MB of Q6_K weights)",
        bytes.len(),
        bytes.len() / 1_000_000
    );
    assert_eq!(
        bytes.len() % row_bytes,
        0,
        "head is not a whole number of rows"
    );

    let dev = RocmDevice::try_new(0).expect("try_new");
    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    let b_st = MemoryOps::from_cpu_bytes(&dev, &bytes, &Shape::new(vec![n * k]), u8ty.clone())
        .expect("head h2d");
    let b_r = as_rocm(b_st.as_ref()).unwrap();

    let mut flips = 0usize;
    let mut worst_margin_ratio = 0.0f64;
    let mut worst_logit_err = 0.0f64;
    let mut tot = 0usize;

    for hi in 0..n_hidden {
        let mut rng = Lcg {
            s: 0x51ED_0000 ^ (hi as u64),
            rms: h_rms,
        };
        let h: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
        let h_st = MemoryOps::from_cpu_bytes(
            &dev,
            unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 4) },
            &Shape::new(vec![k]),
            f32ty.clone(),
        )
        .expect("h h2d");
        let h_r = as_rocm(h_st.as_ref()).unwrap();

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
        let out_exact = mk_out();
        let out_dot4 = mk_out();
        // The q8_1 activation the dot4 leg consumes.
        let q81_len = (k / 32) * 36;
        let q81_st = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; q81_len],
            &Shape::new(vec![q81_len]),
            u8ty.clone(),
        )
        .expect("q81 alloc");

        for _ in 0..1 {
            dev.launch_quantize_q8_1(h_r, as_rocm(q81_st.as_ref()).unwrap(), 1, k)
                .expect("quantize");
            dev.synchronize();
            dev.launch_fused_dequant_gemm_q6k_for_ab(
                h_r,
                b_r,
                as_rocm(out_exact.as_ref()).unwrap(),
                1,
                n,
                k,
            )
            .expect("exact");
            dev.launch_dot4_q6k_q81_gemv_for_ab(
                as_rocm(q81_st.as_ref()).unwrap(),
                b_r,
                as_rocm(out_dot4.as_ref()).unwrap(),
                1,
                n,
                k,
            )
            .expect("dot4");
            dev.synchronize();
        }

        let le = as_rocm(out_exact.as_ref())
            .unwrap()
            .copy_to_host()
            .expect("d2h exact");
        let ld = as_rocm(out_dot4.as_ref())
            .unwrap()
            .copy_to_host()
            .expect("d2h dot4");
        let g = |b: &[u8], i: usize| {
            f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
        };
        // argmax of each path
        let mut ai_e = 0usize;
        let mut ai_d = 0usize;
        let mut top1 = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for i in 0..n {
            let e = g(&le, i);
            let d = g(&ld, i);
            if e > top1 {
                second = top1;
                top1 = e;
                ai_e = i;
            } else if e > second {
                second = e;
            }
            if d > g(&ld, ai_d) {
                ai_d = i;
            }
        }
        let margin = (top1 - second) as f64;
        let mut err = 0.0f64;
        for i in 0..n {
            err = err.max((g(&ld, i) as f64 - g(&le, i) as f64).abs());
        }
        let flip = ai_e != ai_d;
        if flip {
            flips += 1;
        }
        worst_margin_ratio = worst_margin_ratio.max(err / margin.max(1e-6));
        worst_logit_err = worst_logit_err.max(err);
        tot += 1;
        println!(
            "[q6k-head] h#{hi}: argmax exact={ai_e} dot4={ai_d} {}  top1={:.4} margin={:.4} \
             max|dlogit|={:.4}  err/margin={:.3}",
            if flip { "FLIP" } else { "same" },
            top1,
            margin,
            err,
            err / margin.max(1e-6)
        );
    }
    println!(
        "\n[q6k-head] {tot} hidden states: {flips} argmax flips, worst max|dlogit| {worst_logit_err:.4}, \
         worst err/margin {worst_margin_ratio:.3}"
    );
}
