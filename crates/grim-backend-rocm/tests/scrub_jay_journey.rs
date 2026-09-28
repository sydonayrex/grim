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
