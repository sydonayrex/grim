//! Phase 5 dispatch A/B (decode-plan-universal-optimization.md).
//!
//! Times the three `quantized_matmul` paths per K-quant format on gfx1201:
//!   - dot4 GEMV      (m==1, GRIM_DOT_GEMV=1, k%256==0)
//!   - WMMA fused     (m <= GRIM_WMM_MAX_M)
//!   - scalar per-quant GEMM (m > GRIM_WMM_MAX_M — the q2k/q3k/q4k/q5k/q6k
//!     scalar kernels Phase 5 proposes to delete)
//!
//! Both env knobs are read per-call inside `quantized_matmul`, so a single
//! process can A/B the paths by toggling env between timed runs. The verdict
//! recorded in the plan: if WMMA beats the scalar GEMM at every prefill shape
//! (and dot4 wins at m==1), the scalar per-quant kernels become removable by
//! collapsing the m>1 dispatch to WMMA.
//!
//! Runs only under GRIM_RUN_GPU_TESTS=1; otherwise prints SKIP and passes.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, Shape, Storage};
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref AB_MUTEX: Mutex<()> = Mutex::new(());
}

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Fmt {
    fn name(self) -> &'static str {
        match self {
            Fmt::Q2K => "Q2_K",
            Fmt::Q3K => "Q3_K",
            Fmt::Q4K => "Q4_K",
            Fmt::Q5K => "Q5_K",
            Fmt::Q6K => "Q6_K",
        }
    }
    fn dtype(self) -> DType {
        let scheme = match self {
            Fmt::Q2K => KQuantScheme::Q2K,
            Fmt::Q3K => KQuantScheme::Q3K,
            Fmt::Q4K => KQuantScheme::Q4K,
            Fmt::Q5K => KQuantScheme::Q5K,
            Fmt::Q6K => KQuantScheme::Q6K,
        };
        DType { arith: ArithType::F32, storage: Storage::KQuant(scheme) }
    }
}

/// Encode `n` columns of `k` weights into this format's packed layout.
fn encode_weights(fmt: Fmt, n: usize, k: usize) -> Vec<u8> {
    let mut seed = 0xABCD_1234u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    match fmt {
        // Encoders exist — use them (canonical layouts).
        Fmt::Q4K | Fmt::Q5K | Fmt::Q6K => {
            // Encode ONE row of k weights, then tile it per column — the
            // encoder consumes the whole slice, so passing n*k here would
            // produce n× the packed size (tens of GB at prefill shapes).
            let vals: Vec<f32> = (0..k).map(|_| rand()).collect();
            let packed = match fmt {
                Fmt::Q4K => grim_quant::quant_q4k(&vals).expect("quant_q4k"),
                Fmt::Q5K => grim_quant::quant_q5k(&vals).expect("quant_q5k"),
                Fmt::Q6K => grim_quant::quant_q6k(&vals).expect("quant_q6k"),
                _ => unreachable!(),
            };
            let row_len = packed.len();
            let mut out = vec![0u8; n * row_len];
            for c in 0..n {
                out[c * row_len..(c + 1) * row_len].copy_from_slice(&packed);
            }
            out
        }
        // No encoders — build valid blocks directly (layout matches
        // grim_quant::dequant_q2k / dequant_q3k; content is perf-only).
        Fmt::Q2K => {
            let sb = k / 256;
            let mut out = vec![0u8; n * sb * 84];
            for c in 0..n {
                for b in 0..sb {
                    let blk = &mut out[(c * sb + b) * 84..(c * sb + b) * 84 + 84];
                    for i in 0..16 {
                        blk[i] = (rand() as u8) % 16 | (((rand() as u8) % 16) << 4);
                    }
                    for i in 0..64 {
                        blk[16 + i] = rand().to_bits() as u8;
                    }
                    let d = 0.02f32;
                    blk[80..82].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                    let dmin = 0.01f32;
                    blk[82..84].copy_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                }
            }
            out
        }
        Fmt::Q3K => {
            let sb = k / 256;
            let mut out = vec![0u8; n * sb * 110];
            for c in 0..n {
                for b in 0..sb {
                    let blk = &mut out[(c * sb + b) * 110..(c * sb + b) * 110 + 110];
                    for i in 0..32 {
                        blk[i] = rand().to_bits() as u8; // hmask
                    }
                    for i in 0..64 {
                        blk[32 + i] = rand().to_bits() as u8; // qs
                    }
                    for i in 0..12 {
                        blk[96 + i] = rand().to_bits() as u8; // scales
                    }
                    let d = 0.01f32;
                    blk[108..110].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                }
            }
            out
        }
    }
}

/// Set the dispatch env knobs for one A/B configuration.
fn set_config(dot4: bool, wmma_max_m: usize) {
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", if dot4 { "1" } else { "0" });
        std::env::set_var("GRIM_WMM_MAX_M", wmma_max_m.to_string());
    }
}

fn time_config(
    dev: &RocmDevice,
    a: &dyn grim_tensor::BackendStorage,
    b: &dyn grim_tensor::BackendStorage,
    m: usize,
    n: usize,
    iters: usize,
) -> (f64, Vec<f32>) {
    let out_shape = Shape::new(vec![m, n]);
    // Warmup + correctness sample.
    let (st, handle) = dev
        .quantized_matmul(a, b, &[], grim_tensor::QuantFormat::Q8_0, &out_shape)
        .expect("quantized_matmul");
    handle.synchronize().expect("sync");
    let sample = st.to_cpu_vec_f32().expect("d2h");
    let _ = &sample;

    // Per-iteration REAL device barrier (RocmDevice::synchronize =
    // hipDeviceSynchronize). Two constraints drive this shape:
    //  1. ComputeHandle::synchronize is a no-op when the handle carries no
    //     stream, so it cannot time kernel completion;
    //  2. enqueueing iterations without a barrier between them defeats the
    //     caching allocator's recycling (freed outputs still have kernels in
    //     flight) and OOMs the host. Per-iter sync keeps allocation strictly
    //     sequential. Sync overhead is IDENTICAL across all A/B configs, so
    //     the comparison stays fair.
    dev.synchronize();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let (_st, _h) = dev
            .quantized_matmul(a, b, &[], grim_tensor::QuantFormat::Q8_0, &out_shape)
            .expect("quantized_matmul timed iter");
        dev.synchronize();
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    (ms, sample)
}

#[test]
fn phase5_dispatch_ab_benchmark() {
    let _lock = AB_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TESTS=1 + GPU");
        return;
    };

    let formats = [Fmt::Q2K, Fmt::Q3K, Fmt::Q4K, Fmt::Q5K, Fmt::Q6K];
    // (M, K, N) — K % 256 == 0 for the dot4 paths; N=1024 (hidden-ish), K=4096 (inter-ish).
    let shapes: [(usize, usize, usize); 4] = [
        (1, 1024, 1024),    // small-model decode GEMV
        (1, 4096, 4096),    // 7-8B-class decode GEMV
        (64, 4096, 4096),   // short prefill
        (512, 4096, 4096),  // prefill chunk
    ];
    const ITERS_SMALL: usize = 30; // m == 1 (~0.1ms kernels + sync)
    const ITERS_MID: usize = 20;   // m == 64
    const ITERS_BIG: usize = 5;    // m >= 512 (multi-ms kernels)

    let mut report = String::from(
        "| Format | M | K | N | dot4 ms | WMMA ms | scalar ms | fastest (M=1 / prefill) |\n|---|---|---|---|---|---|---|---|\n",
    );

    for &fmt in &formats {
        // Largest shape defines the buffers; upload once per format.
        let (_m_max, k_max, n_max) = shapes.iter().fold((0usize, 0, 0), |acc, &(m, k, n)| {
            (acc.0.max(m), acc.1.max(k), acc.2.max(n))
        });
        let b_bytes = encode_weights(fmt, n_max, k_max);
        let b_dev = MemoryOps::from_cpu_bytes(
            &dev,
            &b_bytes,
            &Shape::new(vec![b_bytes.len()]),
            fmt.dtype(),
        )
        .expect("upload weights");

        for &(m, k, n) in &shapes {
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.08).collect();
            let a_dev = CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![m, k]), DType::F32)
                .expect("upload act");

            let iters = if m == 1 { ITERS_SMALL } else if m < 512 { ITERS_MID } else { ITERS_BIG };

            // dot4 (m==1 only; at m>1 the dispatch skips it regardless).
            set_config(true, 1);
            let (dot4_ms, dot4_out) = if m == 1 {
                time_config(&dev, a_dev.as_ref(), b_dev.as_ref(), m, n, iters)
            } else {
                (f64::NAN, Vec::new())
            };

            // WMMA (wmma_max_m >= m).
            set_config(false, m.max(4));
            let (wmma_ms, wmma_out) = time_config(&dev, a_dev.as_ref(), b_dev.as_ref(), m, n, iters);

            // Scalar per-quant GEMM (wmma_max_m=0 forces m > threshold).
            set_config(false, 0);
            let (scalar_ms, scalar_out) = time_config(&dev, a_dev.as_ref(), b_dev.as_ref(), m, n, iters);

            // Cross-path sanity: all-finite, and loose parity between WMMA and
            // scalar (same math, different accumulation order).
            for (label, v) in [("dot4", &dot4_out), ("wmma", &wmma_out), ("scalar", &scalar_out)] {
                if !v.is_empty() && !v.iter().all(|x| x.is_finite()) {
                    // Non-finite output = kernel-vs-host layout drift. Reported,
                    // not fatal: timing is unaffected, and correctness is owned
                    // by the dedicated parity tests (dot_gemv_parity.rs) which
                    // compare against the AUTHORITATIVE host dequant.
                    eprintln!(
                        "[phase5-ab] WARN non-finite output: {label} ({fmt:?} m={m} k={k} n={n}) — layout drift vs host dequant"
                    );
                }
            }
            if !wmma_out.is_empty() && !scalar_out.is_empty() {
                let diff = wmma_out
                    .iter()
                    .zip(scalar_out.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                if diff > 5.0 {
                    // Layout drift between scalar and WMMA kernels — reported,
                    // not fatal: the A/B measures memory traffic & throughput,
                    // and the authoritative layout is the host dequant.
                    eprintln!(
                        "[phase5-ab] WARN layout drift {fmt:?} m={m}: wmma-vs-scalar max_diff={diff}"
                    );
                }
            }

            let m1_winner = if m == 1 {
                if dot4_ms <= wmma_ms && dot4_ms <= scalar_ms { "dot4" }
                else if wmma_ms <= scalar_ms { "WMMA" } else { "scalar" }
            } else if wmma_ms <= scalar_ms { "WMMA" } else { "scalar" };

            report.push_str(&format!(
                "| {} | {} | {} | {} | {:.4} | {:.4} | {:.4} | {} |\n",
                fmt.name(), m, k, n, dot4_ms, wmma_ms, scalar_ms, m1_winner
            ));
            eprintln!(
                "[phase5-ab] {} m={} k={} n={}: dot4={:.4}ms wmma={:.4}ms scalar={:.4}ms -> {}",
                fmt.name(), m, k, n, dot4_ms, wmma_ms, scalar_ms, m1_winner
            );
        }
    }

    // Reset to defaults so later tests in the same process see stock behavior.
    set_config(true, 4);

    eprintln!("\n[phase5-ab] RESULTS TABLE\n{report}");
    // Persist for the plan document (test runs from crate root).
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("phase5_ab_results.md");
    let _ = std::fs::write(&out_path, format!("# Phase 5 dispatch A/B (gfx1201)\n\n{report}"));
    eprintln!("[phase5-ab] wrote {}", out_path.display());
}
