use grim_backend_rocm::{CoreTensorOps, RocmDevice};
use grim_backend_rocm::device::capability_profiler::CapabilityProfiler;
use grim_tensor::{ArithType, DType, Shape, dtype::Storage as DTypeStorage};
use std::time::Instant;

fn main() {
    // Ordinal is an argument: the 2026-08-25 recalibration ran both cards
    // (the pre-DeviceGuard-fix ordinal-1 data was suspect, and the split-K
    // F32 corruption invalidates any tuning measured before it).
    let ordinal: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // WI-SB0 capability calibration for every visible device (measured
    // fresh — the cache is per-process). Printed as JSON lines so runs can
    // be appended to docs/benchmarks verbatim.
    let profiler = CapabilityProfiler::new();
    for cap in profiler.capabilities() {
        println!(
            "{{\"bench\":\"sb0_calibration\",\"ordinal\":{},\"tflops_fp16\":{:.4},\"tflops_fp8\":{:.4},\"hbm_gbps\":{:.2},\"throttle_pct\":{:.4},\"vram_free_mib\":{}}}",
            cap.ordinal,
            cap.tflops_fp16,
            cap.tflops_fp8,
            cap.hbm_bandwidth_gbps,
            cap.throttle_pct,
            cap.vram_free_bytes >> 20
        );
    }

    let dev = match RocmDevice::try_new(ordinal) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("RocmDevice::try_new({ordinal}) failed: {e}");
            std::process::exit(1);
        }
    };

    // We tune some representative shapes seen in Llama / Gemma models:
    let shapes_to_tune = vec![
        (1, 4096, 4096),
        (8, 4096, 4096),
        (1, 11008, 4096),
        (8, 11008, 4096),
    ];

    println!("Starting offline GEMM solution sweep...");
    println!("------------------------------------------------------------");

    for &(m, n, k) in &shapes_to_tune {
        println!("Tuning shape: M={}, N={}, K={}...", m, n, k);

        let a_host = vec![1.0f32; m * k];
        let b_host = vec![1.0f32; k * n];

        let dtype = DType {
            arith: ArithType::F32,
            storage: DTypeStorage::Native,
        };
        let a_s = dev
            .from_cpu(&a_host, &Shape::from_slice(&[m, k]), dtype.clone())
            .unwrap();
        let b_s = dev
            .from_cpu(&b_host, &Shape::from_slice(&[k, n]), dtype.clone())
            .unwrap();
        let out_shape = Shape::from_slice(&[m, n]);

        let mut best_index = 0;
        let mut best_time = f64::MAX;

        // Sweep solution indices 0 to 80
        for index in 0..80 {
            // Run a few warmups
            let mut success = true;
            for _ in 0..2 {
                if dev
                    .matmul_with_solution(a_s.as_ref(), b_s.as_ref(), &out_shape, index)
                    .is_err()
                {
                    success = false;
                    break;
                }
            }

            if !success {
                continue; // Skip invalid solution indices
            }

            // Benchmark
            let start = Instant::now();
            let num_iters = 10;
            for _ in 0..num_iters {
                let _ = dev
                    .matmul_with_solution(a_s.as_ref(), b_s.as_ref(), &out_shape, index)
                    .unwrap();
            }
            let elapsed = start.elapsed().as_secs_f64() / (num_iters as f64);

            if elapsed < best_time {
                best_time = elapsed;
                best_index = index;
            }
        }

        println!(
            "  -> Best Solution Index: {} (avg time: {:.3} ms)",
            best_index,
            best_time * 1000.0
        );
    }
}
