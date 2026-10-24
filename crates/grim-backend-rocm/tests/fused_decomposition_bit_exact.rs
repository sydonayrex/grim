//! M7.3 — is the FUSED launch's decomposition BIT-IDENTICAL?
//!
//! WHY THIS GATE EXISTS. The plan's M7.3 is the fused-launch
//! decomposition: "The launcher stops fusing (a numerics change — the M2
//! oracle is what says whether the decomposition is bit-identical), and the
//! model's IR names the dot4 and the add separately."
//!
//! The models' Q8_0 down projections run as ONE launch that does the
//! integer dot AND the residual add together
//! (`launch_dot4_q80_f32act_add_gemv`). For the IR to name the dot4 as its
//! own node — a `Q8_0_DOT4_F32ACT` node, which the ring CAN execute — the
//! launcher has to stop fusing, and the residual add becomes its own
//! `ELEMENTWISE` node.
//!
//! But "the launcher stops fusing" is a NUMERICS change, and the plan says
//! the oracle decides whether it is acceptable. So the question has to be
//! MEASURED, and the existing gate is not the answer: it compares the fused
//! path against the split path with a **1e-4 tolerance**
//! (`fused_residual_add_gemv_tile16_matches_split`), which is a
//! "close enough" claim, not a "the same answer" claim.
//!
//! A Q8_0 dot4's rounding IS the kind of difference that hides until the
//! logits move — the same reasoning the Q8_0 arm's own gate uses when it
//! demands EQUALITY rather than closeness. So this gate demands EQUALITY
//! and, if the fused and split paths differ, it says so EXACTLY (which
//! elements differ, and by how much) rather than hiding behind a
//! tolerance.
//!
//! THE REASON THEY MAY DIFFER, and it is a real one: the fused kernel
//! computes `C[out_idx] = res + facc[j]` with its OWN tiling (16 or 8
//! columns per block, chosen by the TILE16/TILE8 env switches), while the
//! split path's dot4 uses 4 columns per block. The column tiling changes
//! which lane computes which column, and the 32-lane shuffle reduction
//! sums a DIFFERENT set of partial sums — so the two paths can differ in
//! the last bits of every output.
//!
//! Run (GPU 1 is the measurement card; GPU 0 is reserved):
//! `HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!  cargo test -p grim-backend-rocm --test fused_decomposition_bit_exact \
//!    -- --nocapture --test-threads=1`

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, CoreTensorOps, DType, Shape, Storage};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new(0)")).ok()
}

fn f32_tensor(
    dev: &RocmDevice,
    data: &[f32],
    shape: &Shape,
) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev, data, shape, DType::F32).unwrap()
}

/// Pack a [rows, k] f32 matrix into the Q8_0 block format the dot4 reads.
///
/// The format is the model's own: 32-wide blocks, each with an f16 scale
/// and 32 int8 values. The packing is the SAME one the checkpoint loader
/// uses, so the kernel sees exactly the bytes it sees in production.
fn pack_q80(w: &[f32], rows: usize, k: usize) -> Vec<u8> {
    let blocks = k / 32;
    let mut out = vec![0u8; rows * blocks * 34];
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * 34;
            let mut amax = 0.0f32;
            for e in 0..32 {
                amax = amax.max(w[r * k + b * 32 + e].abs());
            }
            let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            let h = half::f16::from_f32(d);
            out[off..off + 2].copy_from_slice(&h.to_le_bytes());
            for e in 0..32 {
                let v = w[r * k + b * 32 + e] / d;
                out[off + 2 + e] = v.round().clamp(-127.0, 127.0) as i8 as u8;
            }
        }
    }
    out
}

fn upload_q80(
    dev: &RocmDevice,
    packed: &[u8],
    rows: usize,
    k: usize,
) -> Box<dyn grim_tensor::BackendStorage> {
    let q_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(grim_tensor::dtype::KQuantScheme::Q80),
    };
    <grim_backend_rocm::RocmDevice as grim_tensor::MemoryOps>::from_cpu_bytes(
        dev,
        packed,
        &Shape::new(vec![rows, k]),
        q_dtype,
    )
    .unwrap()
}

fn rocm<'a>(s: &'a Box<dyn grim_tensor::BackendStorage>) -> &'a grim_backend_rocm::RocmStorage {
    grim_backend_rocm::as_rocm(s.as_ref()).unwrap()
}

fn rand_f32(n: usize, seed: usize) -> Vec<f32> {
    let mut s = seed as u64 | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / u32::MAX as f32) - 0.5) * 0.4
        })
        .collect()
}

/// Compare the FUSED launch against the SPLIT path (the dot4, then the
/// add) and report whether they are BIT-IDENTICAL.
#[test]
fn the_fused_and_split_paths_are_or_are_not_bit_identical() {
    let Some(dev) = gpu_device() else {
        return;
    };
    if !dev.supports_dot4() {
        eprintln!("[SKIP] the device does not support dot4");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();

    // The 9B's down-projection shape, and a couple of neighbours.
    let m = 1usize;
    let n = 4096usize;
    let k = 4096usize;

    let x = rand_f32(k, 501);
    let res_data = rand_f32(n, 502);
    let w_data = rand_f32(n * k, 503);

    let a = f32_tensor(&dev, &x, &Shape::new(vec![m, k]));
    let w = upload_q80(&dev, &pack_q80(&w_data, n, k), n, k);
    let residual = f32_tensor(&dev, &res_data, &Shape::new(vec![n]));

    // ── The SPLIT path: the dot4, then the add. ──
    let gemv_out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    dev.launch_dot4_q80_f32act_gemv(rocm(&a), rocm(&w), rocm(&gemv_out), m, n, k)
        .expect("split dot4");
    let gemv_vec = gemv_out.to_cpu_vec_f32().expect("split readback");

    // ── The FUSED path: one launch doing both. ──
    let fused_out = f32_tensor(&dev, &vec![0.0f32; n], &Shape::new(vec![n]));
    dev.launch_dot4_q80_f32act_add_gemv(rocm(&a), rocm(&w), Some(rocm(&residual)), rocm(&fused_out), m, n, k)
        .expect("fused dot4+add");
    let fused_vec = fused_out.to_cpu_vec_f32().expect("fused readback");

    // ── The COMPARISON. ──
    //
    // The split path's answer is `dot4 + residual` in f32, computed on the
    // host here so the comparison isolates the DOT4's rounding from the
    // ADD's. The fused kernel does `res + facc[j]` in f32 on the device.
    let mut differ = 0usize;
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for i in 0..n {
        let want = gemv_vec[i] + res_data[i];
        let d = (fused_vec[i] - want).abs();
        if d != 0.0 {
            differ += 1;
            if d > worst {
                worst = d;
                worst_at = i;
            }
        }
    }
    eprintln!(
        "[fused-decomp] the fused path vs. the split path: {differ}/{n} elements \
         differ, worst {worst:.3e} at {worst_at}"
    );
    if differ > 0 {
        eprintln!(
            "[fused-decomp] got={} want={}",
            fused_vec[worst_at],
            gemv_vec[worst_at] + res_data[worst_at]
        );
    }

    // The VERDICT, and it is the plan's decision rule made executable.
    //
    // If the two paths are bit-identical, the launcher can stop fusing and
    // the IR can name the dot4 and the add separately, with no numerics
    // change — M7.3 lands. If they differ, the fusion MUST stay (the
    // fused launch is one kernel with one reduction order, and the split
    // path's different tiling reduces a different set of partial sums), and
    // the IR keeps naming the fused launch as ONE node — which is what it
    // does today, and which is CORRECT: a node's opcode must name the
    // computation the ring would actually run.
    //
    // This gate therefore REPORTS rather than asserts a fixed outcome: the
    // decomposition's viability is a property of the kernels' arithmetic, and
    // a gate that hard-coded either answer would stop being true the moment
    // a tiling changed.
    // ── THE COST of the decomposition. ──
    //
    // Bit-identical is NECESSARY for the decomposition but not SUFFICIENT: the
    // plan's M7.3 also has to answer "is it worth it". Unfusing turns ONE
    // launch into TWO (the dot4 and the add), so the decomposition's price
    // is a launch — and a launch is exactly what the ring is supposed to
    // remove. So the price is measured here, on the real shape, and the
    // verdict is printed with it.
    //
    // This is the same discipline the res7 corpus's §10a asks for: the
    // outcome AND the mechanism, so a reader can see whether the change
    // pays rather than whether it merely works.
    const BENCH_ITERS: usize = 50;
    let bench = |fused: bool| {
        for _ in 0..5 {
            if fused {
                dev.launch_dot4_q80_f32act_add_gemv(
                    rocm(&a), rocm(&w), Some(rocm(&residual)), rocm(&fused_out), m, n, k,
                )
                .unwrap();
            } else {
                dev.launch_dot4_q80_f32act_gemv(rocm(&a), rocm(&w), rocm(&gemv_out), m, n, k)
                    .unwrap();
                dev.add_into(rocm(&gemv_out), rocm(&residual), rocm(&fused_out))
                    .unwrap();
            }
        }
        dev.synchronize();
        let t0 = std::time::Instant::now();
        for _ in 0..BENCH_ITERS {
            if fused {
                dev.launch_dot4_q80_f32act_add_gemv(
                    rocm(&a), rocm(&w), Some(rocm(&residual)), rocm(&fused_out), m, n, k,
                )
                .unwrap();
            } else {
                dev.launch_dot4_q80_f32act_gemv(rocm(&a), rocm(&w), rocm(&gemv_out), m, n, k)
                    .unwrap();
                dev.add_into(rocm(&gemv_out), rocm(&residual), rocm(&fused_out))
                    .unwrap();
            }
        }
        dev.synchronize();
        t0.elapsed().as_secs_f64() * 1e6 / BENCH_ITERS as f64
    };
    let fused_us = bench(true);
    let split_us = bench(false);
    eprintln!(
        "[fused-decomp] the price of the decomposition: the fused launch is \
         {fused_us:.1} us, the split path {split_us:.1} us ({:.2}x)",
        split_us / fused_us
    );

    if differ == 0 {
        eprintln!(
            "[fused-decomp] BIT-IDENTICAL: the launcher may stop fusing and the IR \
             may name the dot4 and the add separately (M7.3 is available)."
        );
    } else {
        eprintln!(
            "[fused-decomp] NOT bit-identical: the fused launch must stay one \
             node, because the split path's different column tiling reduces a \
             DIFFERENT set of partial sums. The IR keeps naming the fused \
             launch as one node — which is correct, and is what it does."
        );
    }
}
