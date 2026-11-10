//! astrum P1 Steps 5-7: dim collection (qwen35-9b) + hipBLASLt heuristic/algo
//! enumeration + measured run (≥3% or negative) + PPL sanity notes.
//!
//! Run (GPU1, sequential):
//!   HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm \
//!     --test tune_blaslt_qwen35 -- --ignored --nocapture --test-threads=1
//!
//! Outputs a measured table JSON candidate and per-shape timings. Default OFF;
//! this is a tuning/measured-run harness, not a correctness gate.

use std::ffi::c_void;
use std::time::Instant;

use grim_backend_rocm::device::blaslt::{
    get_heuristic_algos_f32, matmul_col_major_f32, matmul_col_major_f32_with_algo, probe_blaslt,
};
use grim_backend_rocm::device::gemm_tuning_measured::{MeasuredEntry, MeasuredTable};
use grim_backend_rocm::memory::storage::RocmStorage;
use grim_backend_rocm::RocmDevice;
use grim_tensor::backend::BackendStorage;
use grim_tensor::{CoreTensorOps, DType, Shape};

fn alloc_f32(dev: &RocmDevice, len: usize) -> Box<dyn BackendStorage> {
    let data = vec![1.0f32; len];
    dev.from_cpu(&data, &Shape::new(vec![len]), DType::F32)
        .expect("alloc/from_cpu f32 buffer")
}

#[test]
#[ignore]
fn tune_blaslt_qwen35_dims_and_heuristics() {
    // Step 5: dim collection (qwen35-9b / Qwen3.5-9B-Q4_K_M.gguf)
    // GGUF: hidden=4096, ffn=12288, heads=16, kv_heads=4, key_len=256, value_len=256,
    // 32 blocks, full_attention_interval=4. Key 2D weight shapes (ne0,ne1):
    // (4096,4096),(4096,8192),(12288,4096),(4096,12288),(4096,1024),(4096,248320),(4096,32).
    // Representative GEMM (M,N,K) for prefill tuning (col-major canonical C[M,N]=A[M,K]*B[K,N]):
    // treat weight [N,K] -> N=ne0/ne1 per projection; we cover dominant (K,N) pairs.
    let base: &[(usize, usize)] = &[
        (4096, 4096),  // attn/ssm out, gate
        (4096, 8192),  // qkv-like
        (4096, 12288), // ffn gate/up (K,N)
        (12288, 4096), // ffn down
        (4096, 1024),  // k/v proj (full-attn layers)
    ];
    let m_values: &[usize] = &[1, 4, 8, 16, 32, 64, 128];

    let probe = probe_blaslt();
    eprintln!("[P1] blaslt probe: {}", probe.describe());
    let version = probe.version.unwrap_or(0);
    let dev = RocmDevice::new(0);
    let arch = dev.gcn_arch().to_string();
    eprintln!("[P1] arch={arch} blaslt_version={version}");
    let stream = dev.create_non_blocking_stream().expect("create stream");

    let mut entries: Vec<MeasuredEntry> = Vec::new();
    let mut any_improve = false;
    let mut any_negative = false;

    for &(k, n) in base {
        for &m in m_values {
            // Step 6: heuristic/algo enumeration
            let heur = match get_heuristic_algos_f32(m, n, k, 8, 0) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("[P1] heuristics failed m={m} n={n} k={k}: {e}");
                    continue;
                }
            };
            if heur.is_empty() {
                eprintln!("[P1] no heuristics m={m} n={n} k={k}");
                continue;
            }

            let a = alloc_f32(&dev, m * k);
            let b = alloc_f32(&dev, k * n);
            let d = alloc_f32(&dev, m * n);
            let a_r = a.as_any().downcast_ref::<RocmStorage>().expect("a rocm");
            let b_r = b.as_any().downcast_ref::<RocmStorage>().expect("b rocm");
            let d_r = d.as_any().downcast_ref::<RocmStorage>().expect("d rocm");
            let a_ptr = a_r.device_ptr_checked().expect("a ptr") as *const c_void;
            let b_ptr = b_r.device_ptr_checked().expect("b ptr") as *const c_void;
            let d_ptr = d_r.device_ptr_checked().expect("d ptr") as *mut c_void;

            // warmup default
            let _ = matmul_col_major_f32(stream, a_ptr, b_ptr, d_ptr, m, n, k);
            dev.synchronize();
            // baseline (default algo)
            let iters = if m <= 8 { 50 } else { 20 };
            let t0 = Instant::now();
            for _ in 0..iters {
                if let Err(e) = matmul_col_major_f32(stream, a_ptr, b_ptr, d_ptr, m, n, k) {
                    eprintln!("[P1] baseline failed m={m} n={n} k={k}: {e}");
                    break;
                }
            }
            dev.synchronize();
            let base_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let mut best_us = base_us;
            let mut best_idx: i32 = -1; // -1 = default
            for (i, cand) in heur.iter().enumerate() {
                // try candidate algo (no workspace; skip if requires workspace)
                if cand.workspace_size > 0 {
                    continue;
                }
                let _ = matmul_col_major_f32_with_algo(
                    stream, a_ptr, b_ptr, d_ptr, m, n, k, Some(&cand.algo), std::ptr::null_mut(), 0,
                );
                dev.synchronize();
                let t1 = Instant::now();
                let mut ok = true;
                for _ in 0..iters {
                    if let Err(e) = matmul_col_major_f32_with_algo(
                        stream, a_ptr, b_ptr, d_ptr, m, n, k, Some(&cand.algo), std::ptr::null_mut(), 0,
                    ) {
                        eprintln!("[P1] algo[{i}] failed m={m} n={n} k={k}: {e}");
                        ok = false;
                        break;
                    }
                }
                dev.synchronize();
                if !ok {
                    continue;
                }
                let us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
                if us < best_us {
                    best_us = us;
                    best_idx = i as i32;
                }
            }

            let speedup = if best_us > 0.0 { base_us / best_us } else { 1.0 };
            if speedup >= 1.03 {
                any_improve = true;
            } else if speedup < 1.0 {
                any_negative = true;
            }
            eprintln!(
                "[P1] m={m:>3} n={n:<5} k={k:<5} base={base_us:>9.2}us best={best_us:>9.2}us speedup={speedup:.4}x heur={} best_idx={best_idx}",
                heur.len()
            );
            // record winner when a heuristic algo beats default by >=3%
            if best_idx >= 0 && speedup >= 1.03 {
                entries.push(MeasuredEntry {
                    m,
                    n,
                    k,
                    arith: "f32".to_string(),
                    solution: best_idx,
                    speedup,
                });
            }
        }
    }

    dev.destroy_stream(stream);

    // Step 7: measured run summary (≥3% or negative result) + table candidate
    let table = MeasuredTable {
        arch: arch.clone(),
        library: "blaslt".to_string(),
        blaslt_version: version.to_string(),
        entries: entries.clone(),
    };
    match serde_json::to_string_pretty(&table) {
        Ok(json) => eprintln!("[P1] measured table candidate:\n{json}"),
        Err(e) => eprintln!("[P1] table serialize failed: {e}"),
    }
    eprintln!(
        "[P1] summary: winning_entries={} any_>=3%_improvement={} any_negative_vs_default={}",
        entries.len(),
        any_improve,
        any_negative
    );
    // PPL sanity: no HIP/kernel/model path changed by this harness run; default
    // build remains bit-identical. Full model PPL sanity is a separate gate run
    // under GRIM_ASTRUM_{BLASLT,ROCBLAS}_TABLE (not applied here).
    eprintln!("[P1] PPL sanity: harness-only GPU timing (no model/kernel edits, no table applied) — default path unchanged.");
}
