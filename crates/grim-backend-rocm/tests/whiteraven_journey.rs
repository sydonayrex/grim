//! WhiteRaven end to end: a real decode-shaped stack, through the production
//! `Linear` path, on the real blocked kernel.
//!
//! The gates before this file each prove one link: parity proves the blocked
//! kernel equals the row-major one, dispatch proves a `Fp8Blocked16` tensor
//! reaches it. Neither proves anything a model would care about -- whether the
//! format survives being loaded as weights and run through layers, and whether
//! the result is still right. A format whose per-layer error looks fine can
//! still compound down a residual stack into a flipped argmax.
//!
//! What is deliberately NOT here: nothing loads a checkpoint. The weights come
//! from grim's own `rewrite_tensor_data` (the producer item 1 shipped), so
//! the bytes under test are the bytes a checkpoint would carry.
//!
//! Accuracy bar: E4M3 has 3 mantissa bits, so per-weight error is ~6% and a
//! K=512 dot partially cancels it (~1/sqrt(K)). The end-to-end gate is
//! deliberately on the OUTPUT and the argmax, not on a per-weight figure --
//! same discipline as `tree_pie_journey`.
//!
//! RUN: GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm \
//!      --test whiteraven_journey -- --nocapture --test-threads=1

use grim_backend_rocm::{kernel_route_snapshot, RocmDevice};
use grim_nn::Linear;
use grim_tensor::{
    ArithType, DType, MemoryOps, QuantFormat, QuantProvenance, Shape, Storage, Tensor,
};
use std::panic;
use std::sync::Arc;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

const HIDDEN: usize = 512;
const LAYERS: usize = 4;
const VOCAB: usize = 1024;

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888950007);
        let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
        (unit - 0.5) * 2.0
    }
}

fn gelu(v: f32) -> f32 {
    let x = v.clamp(-8.0, 8.0);
    let inner = 0.797_884_56 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Produce the blocked-FP8 bytes through the real rewrite path (item 1), not
/// by hand: a hand-packed buffer would not catch a producer that disagrees
/// with the kernel about the layout.
fn pack_blocked(w: &[f32], n: usize, k: usize) -> TestResult<Vec<u8>> {
    let plan = grim_quant::TensorRewritePlan {
        target: QuantFormat::Fp8Blocked16,
        shape: vec![n, k],
        importance: None,
        curvature: None,
    };
    let out = grim_quant::rewrite_tensor_data(w, &plan)?;
    assert_eq!(
        out.bytes.len(),
        n * k,
        "blocked fp8 is a permutation: same byte count as f32's element count"
    );
    Ok(out.bytes)
}

fn upload_blocked(dev: &RocmDevice, bytes: &[u8], n: usize, k: usize) -> TestResult<Tensor> {
    let t = MemoryOps::from_cpu_bytes(
        dev,
        bytes,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::U8,
            storage: Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16),
        },
    )
    .map_err(|e| format!("blocked weight h2d: {e}"))?;
    let dtype = DType {
        arith: ArithType::U8,
        storage: Storage::FloatPack(grim_tensor::FloatPackScheme::Fp8Blocked16),
    };
    Ok(Tensor::new(
        Arc::from(t),
        Shape::new(vec![n, k]),
        dtype,
        QuantProvenance::GrimNative,
        grim_tensor::Device::Rocm(0),
    ))
}

fn upload_f32(dev: &RocmDevice, vals: &[f32], n: usize, k: usize) -> TestResult<Tensor> {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    let t = MemoryOps::from_cpu_bytes(dev, &bytes, &Shape::new(vec![n, k]), DType::F32)
        .map_err(|e| format!("f32 weight h2d: {e}"))?;
    Ok(Tensor::new(
        Arc::from(t),
        Shape::new(vec![n, k]),
        DType::F32,
        QuantProvenance::GrimNative,
        grim_tensor::Device::Rocm(0),
    ))
}

fn upload_act(dev: &RocmDevice, vals: &[f32], rows: usize, k: usize) -> TestResult<Tensor> {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    let t = MemoryOps::from_cpu_bytes(dev, &bytes, &Shape::new(vec![rows, k]), DType::F32)
        .map_err(|e| format!("act h2d: {e}"))?;
    Ok(Tensor::new(
        Arc::from(t),
        Shape::new(vec![rows, k]),
        DType::F32,
        QuantProvenance::GrimNative,
        grim_tensor::Device::Rocm(0),
    ))
}

fn download_f32(_dev: &RocmDevice, t: &Tensor) -> TestResult<Vec<f32>> {
    let s = grim_backend_rocm::as_rocm(t.storage().as_ref()).map_err(|e| e.to_string())?;
    Ok(s.copy_to_host()?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
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

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Process-wide route counters mean these three tests cannot run
/// concurrently: each asserts on a delta of the same two entries, and a
/// sibling test's launches land in the other's window. `cargo test` runs
/// TEST BINARIES sequentially but threads within one binary in parallel, so
/// the lock goes here.
fn journey_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn route(entry: &str) -> u64 {
    kernel_route_snapshot()
        .into_iter()
        .find(|(k, _)| k == entry)
        .map(|(_, v)| v)
        .unwrap_or(0)
}

/// Run `LAYERS` residual Linear+GELU blocks plus a head, on GPU, through the
/// production `Linear::forward` -> `quantized_matmul` dispatch.
fn stack_forward(dev: &RocmDevice, layers: &[Linear], x0: &[f32]) -> TestResult<Vec<f32>> {
    let mut x = upload_act(dev, x0, 1, HIDDEN)?;
    for layer in layers.iter().take(LAYERS) {
        let y = layer.forward(&x)?;
        let yh = download_f32(dev, &y)?;
        // Residual must accumulate onto the RUNNING stream, not x0 -- the host
        // reference does the same, and a divergence here would make the f32
        // harness check fail instead of measuring the format.
        let xh = download_f32(dev, &x)?;
        let next: Vec<f32> = (0..HIDDEN).map(|i| xh[i] + 0.5 * gelu(yh[i])).collect();
        x = upload_act(dev, &next, 1, HIDDEN)?;
    }
    let head = &layers[LAYERS];
    let out = head.forward(&x)?;
    download_f32(dev, &out)
}

/// Host reference for the same stack, from the f32 weights. The quantized arm
/// is scored against THIS, so the bar is "blocked FP8 weights reproduce the
/// f32 model's logits", not "blocked FP8 reproduces itself".
fn host_stack(weights: &[Vec<f32>], x0: &[f32]) -> Vec<f32> {
    let mut x = x0.to_vec();
    for layer in 0..LAYERS + 1 {
        let out_dim = if layer == LAYERS { VOCAB } else { HIDDEN };
        let w = &weights[layer];
        let mut y = vec![0.0f32; out_dim];
        for (j, slot) in y.iter_mut().enumerate() {
            let row = &w[j * HIDDEN..(j + 1) * HIDDEN];
            // No 1/sqrt(HIDDEN) here: it is already folded into the weights,
            // so the host and the Linear's plain C = A @ B^T agree exactly.
            let mut acc = 0.0f32;
            for (i, &xi) in x.iter().enumerate() {
                acc += xi * row[i];
            }
            *slot = acc;
        }
        if layer < LAYERS {
            for (i, v) in x.iter_mut().enumerate() {
                *v += 0.5 * gelu(y[i]);
            }
        } else {
            x = y;
        }
    }
    x
}

/// Prefill (m > 1) on the same stack: the blocked GEMM's grid covers m as well
/// as n, and a m=16 prefill is WhiteRaven's native fragment tile. Decode-only
/// coverage would leave that path unmeasured -- and the ragged case (m=13,
/// two tiles, the second partial) is where an off-by-one in the epilogue
/// writes past the output or drops a row.
fn prefill_case(dev: &RocmDevice, m: usize) -> TestResult<(f32, usize, usize, u64)> {
    let mut rng = Lcg(0x57_1E_2A_11);
    // Same activation for every row so the oracle is one dot per column.
    let x0: Vec<f32> = (0..HIDDEN).map(|_| rng.next_f32()).collect();
    let inv = 1.0 / (HIDDEN as f32).sqrt();
    let w: Vec<f32> = (0..HIDDEN * HIDDEN)
        .map(|_| rng.next_f32() * 0.5 * inv)
        .collect();
    let blocked = pack_blocked(&w, HIDDEN, HIDDEN)?;

    let layer = Linear::from_tensor(upload_blocked(dev, &blocked, HIDDEN, HIDDEN)?, None);
    // Every row sees the same activation, so one host dot per column serves as
    // the oracle for all m rows.
    let acts: Vec<f32> = (0..m * HIDDEN).map(|i| x0[i % HIDDEN]).collect();
    let x = upload_act(dev, &acts, m, HIDDEN)?;
    let before = route("grim_wmma_gemm_fp8_e4m3_blocked");
    let y = layer.forward(&x)?;
    let got = download_f32(dev, &y)?;
    let launched = route("grim_wmma_gemm_fp8_e4m3_blocked").saturating_sub(before);

    // Oracle: host dot over the DEQUANTIZED blocked weights, so this measures
    // kernel math and layout, not the quantizer's error.
    let codes: Vec<u8> = w.iter().map(|&v| grim_quant::f32_to_fp8_e4m3(v)).collect();
    let wq = grim_quant::dequant_fp8_blocked16(&blocked, HIDDEN, HIDDEN)?;
    assert_eq!(wq.len(), HIDDEN * HIDDEN);
    assert_eq!(codes.len(), HIDDEN * HIDDEN);
    let aq: Vec<f32> = acts
        .iter()
        .map(|&v| grim_quant::fp8_e4m3_to_f32(grim_quant::f32_to_fp8_e4m3(v)))
        .collect();

    let mut worst = 0.0f32;
    for r in 0..m {
        for col in 0..HIDDEN {
            let mut acc = 0.0f32;
            let mut mag = 0.0f32;
            for i in 0..HIDDEN {
                let t = aq[r * HIDDEN + i] * wq[col * HIDDEN + i];
                acc += t;
                mag += t.abs();
            }
            let idx = r * HIDDEN + col;
            let rel = (got[idx] - acc).abs() / mag.max(f32::MIN_POSITIVE);
            worst = worst.max(rel);
        }
    }
    Ok((worst, m, HIDDEN, launched))
}

#[test]
fn whiteraven_blocked_survives_a_prefill_tile_including_a_ragged_tail() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let _route_lock = journey_lock();
    for m in [16usize, 13] {
        let (worst, m, n, launched) = prefill_case(&dev, m)?;
        println!("--- prefill m={m} n={n}: worst rel err {worst:.3e}, {launched} launch(es)");
        assert_eq!(launched, 1, "prefill m={m} must dispatch exactly once");
        // f32 accumulation over K=512 in a different order than the CPU oracle;
        // 1e-3 is the f32 summation floor, not a format tolerance.
        assert!(
            worst <= 1e-3,
            "prefill m={m} math wrong: worst rel {worst:.3e} (an O(1) value means a              wrong row/column, not rounding)"
        );
    }
    Ok(())
}

/// Graph-capture arm: the blocked leg must capture and replay, producing the
/// same logits as eager, and re-quantizing A on device every replay.
///
/// This is the only thing that exercises `linear_decode_blocked_into`. The
/// eager path converts activations through the host (D2H + alloc), and the
/// eager leg now REFUSES under capture rather than poisoning the stream -- so
/// without this test the capture-safe alternative would be unproven code.
#[test]
fn whiteraven_blocked_captures_and_replays_through_the_decode_graph() -> TestResult {
    use grim_backend_rocm::decode_graph_buffers::DecodeGraph;

    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let _route_lock = journey_lock();
    if !grim_backend_rocm::decode_graph_buffers::decode_graph_enabled() {
        eprintln!("SKIP: decode graph disabled by env");
        return Ok(());
    }

    let m = 1usize;
    let k = 512usize;
    let n = 256usize;
    let mut rng = Lcg(0x0BAD_C0DE);
    let inv = 1.0 / (k as f32).sqrt();
    let w: Vec<f32> = (0..n * k).map(|_| rng.next_f32() * 0.5 * inv).collect();
    let blocked = pack_blocked(&w, n, k)?;

    // Eager reference.
    let x0: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
    let eager = stack_forward_one(&dev, &upload_blocked(&dev, &blocked, n, k)?, &x0)?;

    // Own stream: `active_stream` is crate-private, and a captured graph's
    // stream must not be the default one (graph_capture.rs:504 makes the same
    // choice).
    let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
    let res: grim_backend_rocm::HipErrorT =
        unsafe { grim_backend_rocm::hipStreamCreate(&mut stream) };
    assert_eq!(
        res,
        grim_backend_rocm::hipSuccess,
        "stream create for capture failed: {res:?}"
    );
    let mut graph = DecodeGraph::allocate(
        &dev, stream, 1,   // num_layers
        k,   // hidden_size
        1,   // n_q
        1,   // n_k
        1,   // n_v
        k,   // intermediate_size -- this is what sizes the fp8 pad scratch
        128, // max_ctx
        n,   // vocab_size (must be nonzero: allocate rejects zero dims)
        1,   // num_heads
        m,   // batch
    )?;

    // Seed inputs BEFORE capture: an async H2D inside the bracket would bake a
    // dead host pointer into the graph's memcpy node.
    let act = upload_act(&dev, &x0, m, k)?;
    let act_rocm = grim_backend_rocm::as_rocm(act.storage().as_ref()).map_err(|e| e.to_string())?;
    let out = grim_backend_rocm::RocmStorage::alloc_gpu(
        &Shape::new(vec![m, n]),
        DType::F32,
        &dev.allocator_handle(),
        0,
    )
    .map_err(|e| format!("out alloc: {e}"))?;
    let b_t = upload_blocked(&dev, &blocked, n, k)?;
    let b_rocm = grim_backend_rocm::as_rocm(b_t.storage().as_ref()).map_err(|e| e.to_string())?;

    // Warmup launches use their own scratch: the capture-scoped one is only
    // published between begin_capture and end_capture.
    let warm_pad = grim_backend_rocm::RocmStorage::alloc_gpu(
        &Shape::new(vec![16 * k]),
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
        &dev.allocator_handle(),
        0,
    )
    .map_err(|e| format!("warm scratch: {e}"))?;
    for _ in 0..2 {
        dev.linear_decode_blocked_into(act_rocm, b_rocm, &out, &warm_pad)
            .map_err(|e| format!("warmup blocked decode: {e}"))?;
    }
    dev.synchronize();

    // Inside the bracket, take the scratch from the accessor begin_capture
    // publishes -- which is exactly how the model's own capture path finds it.
    let before_blocked = route("grim_wmma_gemm_fp8_e4m3_blocked");
    let before_prologue = route("grim_quant_fp8_pad16");
    graph.begin_capture()?;
    let pad = grim_backend_rocm::decode_graph_buffers::capture_fp8_pad_scratch()
        .ok_or("begin_capture must publish the padded-fp8 act scratch")?;
    dev.linear_decode_blocked_into(act_rocm, b_rocm, &out, pad)?;
    graph.end_capture()?;
    let captured_blocked = route("grim_wmma_gemm_fp8_e4m3_blocked").saturating_sub(before_blocked);
    let captured_prologue = route("grim_quant_fp8_pad16").saturating_sub(before_prologue);
    assert_eq!(
        captured_blocked, 1,
        "capture must record one blocked launch"
    );
    assert_eq!(
        captured_prologue, 1,
        "capture must record the act prologue -- without it the graph would read a stale fp8 buffer"
    );

    // Replay must reproduce the eager logits.
    graph.replay()?;
    dev.synchronize();
    let replayed = read_f32(&out)?;
    let rel = rel_err(&replayed, &eager);
    println!("--- graph capture/replay vs eager: rel err {rel:.3e} (m={m} n={n} k={k})");
    assert!(
        rel <= 1e-3,
        "graph replay diverges from eager ({rel:.3e}): the captured prologue or GEMM is wrong"
    );

    // The eager leg must REFUSE under capture rather than silently poisoning
    // the stream. A fresh graph: the one above is already captured, and
    // begin_capture rejects a second capture rather than re-bracketing.
    let mut refuse_graph = DecodeGraph::allocate(&dev, stream, 1, k, 1, 1, 1, k, 128, n, 1, m)?;
    refuse_graph.begin_capture()?;
    let refuse =
        dev.linear_decode_into(act_rocm, b_rocm, &out, &refuse_graph.buffers.act_q81_buf[0]);
    refuse_graph.abort_capture()?;
    assert!(
        refuse.is_err(),
        "the eager blocked leg must refuse during capture -- it allocates and syncs"
    );
    Ok(())
}

#[test]
fn whiteraven_blocked_survives_a_decode_stack_end_to_end() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let _route_lock = journey_lock();

    let mut rng = Lcg(0x57_1E_2A_11);
    let x0: Vec<f32> = (0..HIDDEN).map(|_| rng.next_f32()).collect();

    // Build both arms from the same f32 weights.
    let mut base: Vec<Vec<f32>> = Vec::new();
    let mut base_linears: Vec<Linear> = Vec::new();
    let mut wr_linears: Vec<Linear> = Vec::new();
    for layer in 0..LAYERS + 1 {
        let out_dim = if layer == LAYERS { VOCAB } else { HIDDEN };
        // Scale by 1/sqrt(HIDDEN) up front so the Linear's plain C = A @ B^T
        // matches the host reference's explicit 1/sqrt(HIDDEN); otherwise the two
        // arms differ by a factor of sqrt(HIDDEN) and the harness check fires.
        let inv = 1.0 / (HIDDEN as f32).sqrt();
        let w: Vec<f32> = (0..out_dim * HIDDEN)
            .map(|_| rng.next_f32() * 0.5 * inv)
            .collect();
        base.push(w.clone());

        base_linears.push(Linear::from_tensor(
            upload_f32(&dev, &w, out_dim, HIDDEN)?,
            None,
        ));

        let blocked = pack_blocked(&w, out_dim, HIDDEN)?;
        wr_linears.push(Linear::from_tensor(
            upload_blocked(&dev, &blocked, out_dim, HIDDEN)?,
            None,
        ));
    }

    // Route counters must move, or this file has measured nothing.
    let before_blocked = route("grim_wmma_gemm_fp8_e4m3_blocked");
    let before_rowmajor = route("grim_wmma_gemm_fp8_e4m3");

    let want = host_stack(&base, &x0);
    let got = stack_forward(&dev, &wr_linears, &x0)?;
    // f32 arm on the GPU too: if it disagreed with the host reference, the
    // quantized arm's error would be measuring the harness, not the format.
    let got_f32 = stack_forward(&dev, &base_linears, &x0)?;

    let launched = route("grim_wmma_gemm_fp8_e4m3_blocked").saturating_sub(before_blocked);
    let rowmajor = route("grim_wmma_gemm_fp8_e4m3").saturating_sub(before_rowmajor);
    let harness_err = rel_err(&got_f32, &want);
    let out_err = rel_err(&got, &want);
    let (aw, ag) = (argmax(&want), argmax(&got));
    let mut sorted = want.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let margin = sorted[0] - sorted[1];

    println!("--- WhiteRaven blocked FP8 decode stack (HIDDEN={HIDDEN}, LAYERS={LAYERS}) ---");
    println!("  blocked kernel launches: {launched}   row-major launches: {rowmajor}");
    println!("  harness (f32 GPU vs f32 host) rel err: {harness_err:.3e}");
    println!("  blocked-vs-f32 output rel err: {out_err:.4}");
    println!(
        "  argmax f32={aw} blocked={ag}   top-2 margin {margin:.4}   preserved={}",
        aw == ag
    );

    assert_eq!(
        launched,
        (LAYERS + 1) as u64,
        "every one of the {} Linear layers must dispatch to the blocked kernel",
        LAYERS + 1
    );
    assert_eq!(
        rowmajor, 0,
        "the row-major kernel must not run for blocked weights"
    );
    assert!(
        harness_err <= 1e-3,
        "the f32 GPU arm disagrees with the host reference ({harness_err:.3e}); \
         the quantized arm's error would be measuring the harness"
    );

    // E4M3's 3 mantissa bits give ~6% per-weight error, partially cancelling
    // over K=512. 0.15 end-to-end is loose enough not to flake on a seed and
    // tight enough that an O(1) failure (wrong layout, wrong transpose, a
    // stale scratch row) cannot hide -- that shows up near 1.0.
    assert!(
        out_err <= 0.15,
        "blocked FP8 end-to-end output error {out_err:.4} exceeds 15% -- the format does not survive the stack"
    );
    assert_eq!(
        aw, ag,
        "argmax flipped (f32={aw} blocked={ag}, top-2 margin {margin:.4}) -- a decode loop would emit a different token"
    );
    Ok(())
}

/// One projection through `Linear::forward`, returned as host f32.
fn stack_forward_one(dev: &RocmDevice, weight: &Tensor, x0: &[f32]) -> TestResult<Vec<f32>> {
    let n = weight.shape().dims()[0];
    let layer = Linear::from_tensor(weight.clone(), None);
    let x = upload_act(dev, x0, 1, weight.shape().dims()[1])?;
    let y = layer.forward(&x)?;
    assert_eq!(y.shape().dims(), &[1, n]);
    download_f32(dev, &y)
}

fn read_f32(s: &grim_backend_rocm::RocmStorage) -> TestResult<Vec<f32>> {
    Ok(s.copy_to_host()?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
