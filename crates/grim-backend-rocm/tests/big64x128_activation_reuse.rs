//! The 64x128 big tile memoizes its f32->f16 activation conversion, and the
//! memo key is the SOURCE ALLOCATION — `(device_ptr, bytes, elems, ordinal)`,
//! see `RocmDevice::f32_activation_as_f16_ptr`. A pooled allocator hands the
//! same address to a later activation of the same shape, so the key says
//! "same tensor" about two different tensors and the second GEMM launch reads
//! the first one's halves.
//!
//! This gate is the discriminator: two launches through the PRODUCTION path
//! (`a_f16 = 1`, the memo) with the A buffer's CONTENTS rewritten in between,
//! and the second result checked against the oracle for the SECOND contents.
//!
//! It has to go through the production entry, not `..._q4k_for_ab`: that one
//! passes six arguments to a seven-parameter kernel and leaves `a_f16`
//! undefined, so it exercises the fill-converts-A path and would stay green
//! with the memo as broken as it likes.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage};

/// Host reference for C = A @ dequant(B)^T.
fn oracle(a: &[f32], b_deq: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[r * k + kk] * b_deq[c * k + kk];
            }
            out[r * n + c] = acc;
        }
    }
    out
}

fn worst_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn big64x128_a16_memo_does_not_survive_a_rewrite_of_a() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[SKIP] no ROCm device 0: {e}");
            return;
        }
    };
    let (m, n, k) = (64usize, 128usize, 256usize);

    // Two DIFFERENT activations of the same shape, plus a constant weight.
    let a1: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin()).collect();
    let a2: Vec<f32> = (0..m * k)
        .map(|i| 1.5 + (i as f32 * 0.031).cos().abs())
        .collect();
    let b_host: Vec<f32> = (0..k * n)
        .map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0)
        .collect();
    let b_packed = grim_quant::quant_q4k(&b_host).expect("quant_q4k");
    let b_deq = grim_quant::dequant_q4k(&b_packed, k * n).expect("dequant_q4k");

    let a_dev = CoreTensorOps::from_cpu(&dev, &a1, &Shape::new(vec![m, k]), DType::F32)
        .expect("from_cpu a1");
    let a2_dev = CoreTensorOps::from_cpu(&dev, &a2, &Shape::new(vec![m, k]), DType::F32)
        .expect("from_cpu a2");
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };
    let b_dev =
        MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype)
            .expect("from_cpu_bytes q4k");
    let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * n]), DType::F32).expect("zeros");

    let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
    let a2_r = grim_backend_rocm::as_rocm(a2_dev.as_ref()).unwrap();
    let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
    let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();

    // Launch 1: the memo is cold, so this one populates it from a1.
    dev.launch_wmma_big64x128_q4k_a16_for_ab(a_r, b_r, o_r, m, n, k)
        .expect("launch 1");
    let got1 = out.to_cpu_vec_f32().expect("d2h 1");
    let want1 = oracle(&a1, &b_deq, m, n, k);
    let d1 = worst_diff(&got1, &want1);
    assert!(
        d1 < 0.6,
        "harness: the FIRST launch does not match its own oracle (worst {d1}); \
         the reference below would be meaningless"
    );

    // Rewrite A IN PLACE — same allocation, same address, new contents — and
    // re-launch. The memo must not answer with a1's halves.
    MemoryOps::copy_slice_range(&dev, a_r, 0, a2_r, 0, m * k).expect("rewrite A in place");
    dev.launch_wmma_big64x128_q4k_a16_for_ab(a_r, b_r, o_r, m, n, k)
        .expect("launch 2");
    let got2 = out.to_cpu_vec_f32().expect("d2h 2");
    let want2 = oracle(&a2, &b_deq, m, n, k);
    let d2 = worst_diff(&got2, &want2);
    let stale = worst_diff(&got2, &want1);

    assert!(
        d2 < 0.6,
        "the 64x128 a_f16 memo answered with the PREVIOUS contents of A: second \
         launch is {d2} from the oracle for the rewritten A (and {stale} from the \
         result for the old A). The memo key is the source allocation, which the \
         pooled allocator reuses for a different activation of the same shape."
    );

    // The a16 path's justification is that pre-converting A is EQUIVALENT to
    // letting the fill convert it — the source comment claims "bit-identical".
    // Pin that, because it is the only reason the extra pass is allowed to
    // exist: same A, same B, the `a_f16 = 0` entry against the `a_f16 = 1` one.
    dev.launch_wmma_big64x128_q4k_for_ab(a_r, b_r, o_r, m, n, k)
        .expect("launch a_f16=0");
    let got_fill = out.to_cpu_vec_f32().expect("d2h a_f16=0");
    let d_paths = worst_diff(&got_fill, &got2);
    let scale = got2.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
    eprintln!(
        "[a16-equivalence] a_f16=0 vs a_f16=1: worst abs {d_paths:.6}, \
         worst rel {:.2e} (output amax {scale:.3})",
        d_paths / scale
    );
    assert!(
        d_paths / scale < 1e-3,
        "the pre-converted A path and the fill-converts-A path disagree by \
         {d_paths} ({:.2e} relative) on identical inputs; one of them is wrong",
        d_paths / scale
    );
}

/// What the pre-conversion is worth once the memo is gone.
///
/// The memo deduplicated the conversion across the launches that share an
/// activation (q/k/v, gate/up). The fix converts on every call, so the question
/// is whether the pass still pays for itself at all — the pre-convert exists
/// because the fill re-reads A once per column block, and at these shapes that
/// is worth more than one conversion. Ignored: a bench, not a gate.
#[test]
#[ignore]
fn big64x128_a16_conversion_still_pays_for_itself() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[SKIP] no ROCm device 0: {e}");
            return;
        }
    };
    // attn_q [4096->4096] and ffn_gate/up [4096->12288] at a 256-row window —
    // the two shapes the 4K prefill spends its launches on. The A:B traffic
    // ratio is m-independent, so 256 rows rank them the same way 3928 does.
    const ITERS: usize = 20;
    for (m, n, k) in [(256usize, 4096usize, 4096usize), (256, 12288, 4096)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let b_host: Vec<f32> = (0..k * n)
            .map(|i| 1.0 + (i as f32 * 0.0007).cos().abs() * 6.0)
            .collect();
        let b_packed = grim_quant::quant_q4k(&b_host).expect("quant_q4k");
        let q_dtype = DType {
            arith: ArithType::F32,
            storage: Storage::KQuant(KQuantScheme::Q4K),
        };
        let a_dev = CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![m, k]), DType::F32)
            .expect("from_cpu a");
        let b_dev =
            MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype)
                .expect("from_cpu_bytes b");
        let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * n]), DType::F32).expect("zeros");
        let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
        let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
        let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();

        let time = |label: &str, a16: bool| -> f64 {
            // Warm, then time; each iteration syncs on the D2H.
            for _ in 0..3 {
                let _ = if a16 {
                    dev.launch_wmma_big64x128_q4k_a16_for_ab(a_r, b_r, o_r, m, n, k)
                } else {
                    dev.launch_wmma_big64x128_q4k_for_ab(a_r, b_r, o_r, m, n, k)
                };
            }
            let _ = out.to_cpu_vec_f32().expect("sync");
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                let _ = if a16 {
                    dev.launch_wmma_big64x128_q4k_a16_for_ab(a_r, b_r, o_r, m, n, k)
                } else {
                    dev.launch_wmma_big64x128_q4k_for_ab(a_r, b_r, o_r, m, n, k)
                };
                let _ = out.to_cpu_vec_f32().expect("sync");
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
            eprintln!("[a16-cost] {label}: {ms:.3} ms/launch at m={m} n={n} k={k}");
            ms
        };
        let fill = time("a_f16=0 (fill converts A)", false);
        let pre = time("a_f16=1 (one conversion per call, no memo)", true);
        eprintln!(
            "[a16-cost] m={m} n={n} k={k}: a_f16=0 {fill:.3} ms vs a_f16=1 {pre:.3} ms \
             -> pre-convert is {} ({:.3}x)",
            if pre < fill { "a win" } else { "a LOSS" },
            fill / pre
        );
        assert!(
            pre < fill,
            "the pre-conversion no longer pays for itself at m={m} n={n} k={k} \
             ({pre:.3} vs {fill:.3} ms): the extra per-call pass costs more than the \
             A traffic it saves, so the 64x128 arm should go back to a_f16 = 0"
        );
    }
}
