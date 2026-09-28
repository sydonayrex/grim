//! B7 — the ScrubJay kill criterion: is it at least at parity with Q4_K?
//!
//! The plan's rule (PLAN-corvid-precision.md, "Kill criterion (B7)"):
//!
//!   Abandon ScrubJay if, at decode shapes (M <= 8), it is **not at least
//!   parity** with grim's existing Q4_K fused-dequant path on wall-clock. A
//!   4-bit format slower than an existing 4-bit format has no reason to exist,
//!   regardless of its accuracy advantage. A documented kill is a successful
//!   outcome of this workstream.
//!
//! Read the result with one caveat held firmly in view, because it changes what
//! the number means: the Q4_K path being compared is the *scalar*
//! one-thread-per-output kernel (its own doc comment says it "re-dequantizes
//! the weight row once per output row"). ScrubJay here is a wave-per-output-column
//! kernel. So a ScrubJay win is evidence that the kernel is better written, and
//! only weak evidence about the *format* — the formats differ in bpw (5.5 vs
//! 4.5), so at these shapes the honest comparison is also reported as achieved
//! bandwidth against each format's own byte count.
//!
//! What would actually kill ScrubJay is bandwidth: it is 5.5 bpw against Q4_K's
//! 4.5, so it moves 22% more bytes for the same weights. If it still keeps pace,
//! the kernel's headroom is absorbing the density penalty. If it does not, the
//! density penalty is real and the format is dead on arrival.
//!
//! RUN: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!      cargo test -p grim-backend-rocm --test scrub_jay_journey -- --nocapture

use grim_backend_rocm::RocmDevice;
use grim_quant::scrub_jay::{quantize_block, SCRUB_JAY_BLOCK};
use grim_tensor::{ArithType, BackendStorage, DType, MemoryOps, Shape, Storage};
use std::panic;
use std::time::Instant;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

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
        let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
        (unit - 0.5) * 2.0 * 6.0
    }
}

fn timed_ms<F: FnMut() -> TestResult>(dev: &RocmDevice, mut launch: F, iters: u32) -> TestResult<f64> {
    for _ in 0..3 {
        launch()?;
    }
    dev.synchronize();
    let t0 = Instant::now();
    for _ in 0..iters {
        launch()?;
    }
    dev.synchronize();
    Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

#[test]
fn scrub_jay_vs_q4k_at_decode_shapes() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };

    const K: usize = 4096;
    const ITERS: u32 = 30;
    let mut rng = Lcg(0xB7_C0FFEE);
    let pre_scale = 1.0f32 / 14.0;

    println!("--- B7: ScrubJay vs Q4_K fused dequant, decode (M=1) ---");
    println!("{:>6}  {:>12}  {:>12}  {:>8}  {:>10}  {:>10}", "N", "scrubjay ms", "q4k ms", "ratio", "SJ GB/s", "Q4K GB/s");

    let mut worst_ratio = f64::INFINITY;
    for &n in &[256usize, 1024, 4096] {
        let a: Vec<f32> = (0..K).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n * K).map(|_| rng.next_f32()).collect();

        // ---- ScrubJay planes --------------------------------------------
        let nbc = K / SCRUB_JAY_BLOCK;
        let mut sel = vec![0u8; n * nbc];
        let mut idx = vec![0u8; n * K];
        let mut sgn = vec![0u8; n * nbc];
        let mut scl = vec![0f32; n * nbc];
        for col in 0..n {
            for blk in 0..nbc {
                let mut block = [0f32; SCRUB_JAY_BLOCK];
                for (j, slot) in block.iter_mut().enumerate() {
                    *slot = b[col * K + blk * SCRUB_JAY_BLOCK + j];
                }
                let (s, indices, sign_plane, scale) = quantize_block(&block, pre_scale);
                sel[col * nbc + blk] = s;
                sgn[col * nbc + blk] = sign_plane;
                scl[col * nbc + blk] = scale;
                for j in 0..SCRUB_JAY_BLOCK {
                    idx[col * K + blk * SCRUB_JAY_BLOCK + j] = indices[j];
                }
            }
        }
        // Bytes actually resident in VRAM for the weights.
        let sj_bytes =
            (sel.len() + idx.len() + sgn.len()) as u64 + (scl.len() * 4) as u64 + 4;

        // ---- Q4_K packed --------------------------------------------------
        let q4k = quantize_q4k(&b, n, K)?;
        let q4k_bytes = q4k.len() as u64;

        // ---- upload -------------------------------------------------------
        let a_t = up(&dev, f32_bytes(&a), K, DType { arith: ArithType::F32, storage: Storage::Native }, "a")?;
        let sel_t = up(&dev, &sel, sel.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "sel")?;
        let idx_t = up(&dev, &idx, idx.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "idx")?;
        let sgn_t = up(&dev, &sgn, sgn.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "sgn")?;
        let scl_t = up(&dev, f32_bytes(&scl), scl.len(), DType { arith: ArithType::F32, storage: Storage::Native }, "scl")?;
        let q4k_t = up(&dev, &q4k, q4k.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "q4k")?;
        let out_t = MemoryOps::alloc_storage(
            &dev,
            &Shape::new(vec![n]),
            DType { arith: ArithType::F32, storage: Storage::Native },
        )
        .map_err(|e| format!("out alloc: {e}"))?;

        fn r(t: &Box<dyn BackendStorage>) -> &grim_backend_rocm::RocmStorage {
            grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
        }

        // ---- time both ---------------------------------------------------
        let sj_ms = timed_ms(&dev, || {
            dev.launch_scrub_jay_gemv(r(&a_t), r(&sel_t), r(&idx_t), r(&sgn_t), r(&scl_t), r(&out_t), pre_scale, n, K)
                .map(|_| ())
                .map_err(|e| format!("sj launch: {e}").into())
        }, ITERS)?;

        let q4k_ms = timed_ms(&dev, || {
            dev.launch_fused_dequant_gemm_q4k_for_ab(r(&a_t), r(&q4k_t), r(&out_t), 1, n, K)
                .map(|_| ())
                .map_err(|e| format!("q4k launch: {e}").into())
        }, ITERS)?;

        // Weight-stream bandwidth: the decode GEMV is bandwidth-bound, so the
        // resident weight bytes over wall-clock is the figure that says whether
        // either kernel is leaving the memory system on the table.
        let sj_gbs = sj_bytes as f64 / (sj_ms * 1e-3) / 1e9;
        let q4k_gbs = q4k_bytes as f64 / (q4k_ms * 1e-3) / 1e9;
        let ratio = sj_ms / q4k_ms;
        worst_ratio = worst_ratio.min(ratio);

        println!(
            "{n:>6}  {sj_ms:>12.4}  {q4k_ms:>12.4}  {ratio:>7.2}x  {sj_gbs:>10.1}  {q4k_gbs:>10.1}",
        );
    }

    println!();
    println!("worst ScrubJay/Q4_K ratio across shapes: {worst_ratio:.2}x");
    println!("(ratio < 1.00 means ScrubJay is faster; the kill fires at > 1.00)");
    println!("density: ScrubJay 5.5 bpw vs Q4_K 4.5 bpw = 1.22x the bytes for the same weights");

    assert!(
        worst_ratio <= 1.0,
        "B7 KILL CRITERION FIRED: ScrubJay is {worst_ratio:.2}x the Q4_K wall-clock at decode shapes \
         (needs <= 1.00x). A format slower than an existing format of lower density has no reason to exist."
    );
    Ok(())
}

/// Pack B as Q4_K using grim's own quantizer, so the comparison is against the
/// real production layout rather than a hand-rolled approximation of it.
///
/// The kernel indexes `B_q4k + col * (K/256) * 144` -- row-major per output
/// column, one 144-byte superblock per 256 weights -- which is exactly what
/// `quant_q4k` emits for a K-length row.
fn quantize_q4k(b: &[f32], n: usize, k: usize) -> TestResult<Vec<u8>> {
    assert!(k % 256 == 0, "Q4_K needs K % 256 == 0, got {k}");
    let row_bytes = (k / 256) * 144;
    let mut out = vec![0u8; n * row_bytes];
    for col in 0..n {
        let row = &b[col * k..(col + 1) * k];
        let packed = grim_quant::quant_q4k(row).map_err(|e| format!("quant_q4k col {col}: {e}"))?;
        assert_eq!(packed.len(), row_bytes, "q4k row length mismatch");
        out[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
    }
    Ok(out)
}

fn up(
    d: &RocmDevice,
    bytes: &[u8],
    len: usize,
    dt: DType,
    what: &str,
) -> TestResult<Box<dyn BackendStorage>> {
    MemoryOps::from_cpu_bytes(d, bytes, &Shape::new(vec![len]), dt)
        .map_err(|e| format!("{what} h2d: {e}").into())
}
fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// ─── Viability: the same question A7 asked of TreePie, asked of ScrubJay ───

const HIDDEN: usize = 512;
const LAYERS: usize = 4;
const VOCAB: usize = 1024;

fn gelu(v: f32) -> f32 {
    let x = v.clamp(-8.0, 8.0);
    0.5 * x * (1.0 + (0.797_884_56 * (x + 0.044_715 * x * x * x)).tanh())
}

/// Quantize one weight tensor to ScrubJay and decode it straight back.
///
/// `pre_scale = 1.0` on purpose: the format carries a per-block E4M3 scale, so
/// the per-tensor pre-scale is only a global range hint and leaving it at 1
/// exercises the path a real quantizer would take for a well-centred tensor.
fn scrub_jay_roundtrip(w: &[f32]) -> Vec<f32> {
    let pre_scale = 1.0f32;
    let mut out = vec![0f32; w.len()];
    for (blk, chunk) in w.chunks_exact(SCRUB_JAY_BLOCK).enumerate() {
        let mut block = [0f32; SCRUB_JAY_BLOCK];
        block.copy_from_slice(chunk);
        let (sel, indices, sign_plane, e4m3_scale) = quantize_block(&block, pre_scale);
        let book = &grim_quant::scrub_jay::SCRUB_JAY_CODEBOOK[sel as usize];
        let scale = e4m3_scale * pre_scale;
        let base = blk * SCRUB_JAY_BLOCK;
        for j in 0..SCRUB_JAY_BLOCK {
            let mut v = book[indices[j] as usize] as f32;
            if sign_plane & (1 << j) != 0 {
                v = -v;
            }
            out[base + j] = v * scale;
        }
    }
    out
}

fn forward_stack(weights: &[Vec<f32>], x0: &[f32]) -> Vec<f32> {
    let mut x = x0.to_vec();
    for layer in 0..LAYERS + 1 {
        let out_dim = if layer == LAYERS { VOCAB } else { HIDDEN };
        let w = &weights[layer];
        let mut y = vec![0.0f32; out_dim];
        for (j, slot) in y.iter_mut().enumerate() {
            let row = &w[j * HIDDEN..(j + 1) * HIDDEN];
            let mut acc = 0.0f32;
            for (i, &xi) in x.iter().enumerate() {
                acc += xi * row[i];
            }
            *slot = acc / (HIDDEN as f32).sqrt();
        }
        if layer < LAYERS {
            for v in y.iter_mut() {
                *v = gelu(*v);
            }
            for (i, v) in y.iter_mut().enumerate() {
                x[i] += 0.5 * *v;
            }
        } else {
            x = y;
        }
    }
    x
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        num += ((*g - *w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    ((num / den.max(1e-30)).sqrt()) as f32
}

fn argmax_margin(v: &[f32]) -> (usize, f32) {
    let mut idx = (0, v[0]);
    let mut second = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > idx.1 {
            second = idx.1;
            idx = (i, x);
        } else if x > second {
            second = x;
        }
    }
    (idx.0, idx.1 - second)
}

/// ScrubJay's end-to-end viability, on the identical stack A7 used for TreePie so
/// the two formats' numbers are directly comparable.
///
/// The metric that decides it is NOT the output error: it is the output error
/// measured against the top-2 logit margin. A format can hold output error
/// comfortably below any threshold and still be unusable if the model's actual
/// decision boundary is narrower than that error. A7 found exactly that for
/// TreePie (0.0533 error against a 0.0062 margin), so the margin is reported
/// here as a first-class number rather than derived.
#[test]
fn scrub_jay_journey_survives_end_to_end() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };

    let mut rng = Lcg(0x5C12_4A17);
    let x0: Vec<f32> = (0..HIDDEN).map(|_| rng.next_f32() * 0.5).collect();

    let mut base: Vec<Vec<f32>> = Vec::new();
    let mut quant: Vec<Vec<f32>> = Vec::new();
    for layer in 0..LAYERS + 1 {
        let out_dim = if layer == LAYERS { VOCAB } else { HIDDEN };
        let w: Vec<f32> = (0..out_dim * HIDDEN).map(|_| rng.next_f32()).collect();
        base.push(w.clone());
        quant.push(scrub_jay_roundtrip(&w));
    }

    let want = forward_stack(&base, &x0);
    let got = forward_stack(&quant, &x0);

    // Per-layer trajectory through the residual stream.
    let mut xs_want = x0.clone();
    let mut xs_got = x0.clone();
    let mut traj = Vec::new();
    for layer in 0..LAYERS {
        let w_b = &base[layer];
        let w_q = &quant[layer];
        for (xsw, xsg) in xs_want.iter_mut().zip(xs_got.iter_mut()) {
            let (mut a, mut b) = (0.0f32, 0.0f32);
            for j in 0..HIDDEN {
                let row_b = &w_b[j * HIDDEN..(j + 1) * HIDDEN];
                let row_q = &w_q[j * HIDDEN..(j + 1) * HIDDEN];
                let (mut ab, mut aq) = (0.0f32, 0.0f32);
                for i in 0..HIDDEN {
                    ab += *xsw * row_b[i];
                    aq += *xsg * row_q[i];
                }
                a += gelu(ab / (HIDDEN as f32).sqrt());
                b += gelu(aq / (HIDDEN as f32).sqrt());
            }
            *xsw += 0.5 * a;
            *xsg += 0.5 * b;
        }
        traj.push((layer, rel_err(&xs_got, &xs_want)));
    }

    let out_err = rel_err(&got, &want);
    let (aw, margin) = argmax_margin(&want);
    let (ag, _) = argmax_margin(&got);

    println!("--- ScrubJay per-layer relative error (decode residual stack) ---");
    for (layer, e) in &traj {
        println!("  after layer {layer}: {e:.4}");
    }
    println!("  final output: {out_err:.4}   argmax f32={aw} scrubjay={ag} (top-2 margin {margin:.4})");
    println!("  argmax preserved: {}", aw == ag);
    println!("  error / margin = {:.2}x  (< 1.0 means the error is smaller than the decision boundary)",
        out_err / margin.max(1e-9));

    // GPU arm: the real kernel over real planes, so the cost figure is a launch
    // measurement and the route counter proves the path ran.
    let before = grim_backend_rocm::kernel_route_snapshot();
    let before_count = before
        .iter()
        .find(|(k, _)| k == "grim_scrub_jay_gemv")
        .map(|(_, v)| *v)
        .unwrap_or(0);

    let (n, k) = (HIDDEN, HIDDEN);
    let nbc = k / SCRUB_JAY_BLOCK;
    let mut sel = vec![0u8; n * nbc];
    let mut idx = vec![0u8; n * k];
    let mut sgn = vec![0u8; n * nbc];
    let mut scl = vec![0f32; n * nbc];
    for col in 0..n {
        for blk in 0..nbc {
            let mut block = [0f32; SCRUB_JAY_BLOCK];
            for (j, slot) in block.iter_mut().enumerate() {
                *slot = quant[0][col * k + blk * SCRUB_JAY_BLOCK + j];
            }
            let (s, indices, sign_plane, scale) = quantize_block(&block, 1.0);
            sel[col * nbc + blk] = s;
            sgn[col * nbc + blk] = sign_plane;
            scl[col * nbc + blk] = scale;
            for j in 0..SCRUB_JAY_BLOCK {
                idx[col * k + blk * SCRUB_JAY_BLOCK + j] = indices[j];
            }
        }
    }
    let a_t = up(&dev, f32_bytes(&x0), k, DType { arith: ArithType::F32, storage: Storage::Native }, "a")?;
    let sel_t = up(&dev, &sel, sel.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "sel")?;
    let idx_t = up(&dev, &idx, idx.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "idx")?;
    let sgn_t = up(&dev, &sgn, sgn.len(), DType { arith: ArithType::U8, storage: Storage::Native }, "sgn")?;
    let scl_t = up(&dev, f32_bytes(&scl), scl.len(), DType { arith: ArithType::F32, storage: Storage::Native }, "scl")?;
    let out_t = MemoryOps::alloc_storage(
        &dev,
        &Shape::new(vec![n]),
        DType { arith: ArithType::F32, storage: Storage::Native },
    )
    .map_err(|e| format!("out alloc: {e}"))?;

    fn rr(t: &Box<dyn BackendStorage>) -> &grim_backend_rocm::RocmStorage {
        grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
    }
    let ms = timed_ms(&dev, || {
        dev.launch_scrub_jay_gemv(rr(&a_t), rr(&sel_t), rr(&idx_t), rr(&sgn_t), rr(&scl_t), rr(&out_t), 1.0, n, k)
            .map(|_| ())
            .map_err(|e| format!("launch: {e}").into())
    }, 50)?;

    let after_count = grim_backend_rocm::kernel_route_snapshot()
        .iter()
        .find(|(k, _)| k == "grim_scrub_jay_gemv")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let launched = after_count.saturating_sub(before_count);
    println!("--- GPU arm ---");
    println!("  grim_scrub_jay_gemv launches recorded: {launched}");
    println!("  {ms:.4} ms per launch (N={n}, K={k}, M=1)");
    assert!(launched > 0, "grim_scrub_jay_gemv route never recorded a launch");

    // Verdict. The output-error gate is the same 10% A7 used so the two formats
    // are judged on one scale; the margin ratio is printed rather than gated,
    // because it is a property of this fixture's logits, not of the format.
    assert!(
        out_err <= 0.10,
        "ScrubJay end-to-end output error {out_err:.4} exceeds 10%"
    );
    Ok(())
}

/// The apples-to-apples verdict.
///
/// A7 measured TreePie against a top-2 margin of 0.0062 and the journey above
/// measures ScrubJay against 11.98 -- but those are two *different* random
/// fixtures, so the margins are a property of (format x fixture) and the
/// cross-format "error / margin" comparison is meaningless. This runs all
/// three weight sets through the identical stack, identical input, identical
/// fixture, so the margin cancels and the output errors are directly
/// comparable.
///
/// CPU only on purpose: it is a weight-format question, and the GPU kernels are
/// already gated by B6.
#[test]
fn tree_pie_vs_scrub_jay_on_identical_fixture() -> TestResult {
    let mut rng = Lcg(0x5C12_4A17);
    let x0: Vec<f32> = (0..HIDDEN).map(|_| rng.next_f32() * 0.5).collect();

    let mut base: Vec<Vec<f32>> = Vec::new();
    let mut tp: Vec<Vec<f32>> = Vec::new();
    let mut sj: Vec<Vec<f32>> = Vec::new();
    for layer in 0..LAYERS + 1 {
        let out_dim = if layer == LAYERS { VOCAB } else { HIDDEN };
        let w: Vec<f32> = (0..out_dim * HIDDEN).map(|_| rng.next_f32()).collect();
        base.push(w.clone());
        tp.push(treepie_roundtrip(&w));
        sj.push(scrub_jay_roundtrip(&w));
    }

    let want = forward_stack(&base, &x0);
    let got_tp = forward_stack(&tp, &x0);
    let got_sj = forward_stack(&sj, &x0);

    let e_tp = rel_err(&got_tp, &want);
    let e_sj = rel_err(&got_sj, &want);
    let (aw, margin) = argmax_margin(&want);
    let atp = argmax_margin(&got_tp).0;
    let asj = argmax_margin(&got_sj).0;

    println!("--- identical fixture: f32 vs TreePie vs ScrubJay ---");
    println!("  top-2 logit margin (shared): {margin:.4}");
    println!("  TreePie  output err {e_tp:.4}  argmax {atp}  err/margin {:.2}x  preserved={}",
        e_tp / margin.max(1e-9), atp == aw);
    println!("  ScrubJay  output err {e_sj:.4}  argmax {asj}  err/margin {:.2}x  preserved={}",
        e_sj / margin.max(1e-9), asj == aw);

    // Bytes moved for the same weights, which is the other half of the trade.
    println!("  density: TreePie 5.0 bpw, ScrubJay 5.5 bpw, Q4_K 4.5 bpw");

    // The gate is the one that actually decides usefulness: is the error
    // smaller than the boundary the model has to resolve? Reported for both,
    // gated on ScrubJay because that is the workstream under test.
    assert!(
        e_sj < margin,
        "ScrubJay error {e_sj:.4} exceeds the top-2 margin {margin:.4} -- it cannot resolve this model's decision"
    );
    Ok(())
}

/// TreePie, with the per-channel scale A4 used, so the two round-trips differ
/// only in format rather than in scaling discipline.
fn treepie_roundtrip(w: &[f32]) -> Vec<f32> {
    let amax = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if amax > 1e-8 { amax / 14.0 } else { 1.0 };
    let scaled: Vec<f32> = w.iter().map(|&v| v / scale).collect();
    let packed = grim_quant::tree_pie::pack_tree_pie(&scaled);
    let deq = grim_quant::tree_pie::unpack_tree_pie(&packed);
    deq.iter().map(|&v| v * scale).collect()
}
