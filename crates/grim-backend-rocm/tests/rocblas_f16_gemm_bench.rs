//! Benchmark: rocBLAS f16 GEMM at the model's prefill shapes, on this card.
//!
//! Decision input for "dequant-once-to-f16 + rocBLAS" vs a hand-written
//! 128x128 WMMA kernel for large-M prefill. If rocBLAS sustains ~20-30 TFLOP/s
//! at [M, K] x [N, K] (f16 in, f32/f16 out), the staged-dequant route reaches
//! Ollama-class 4K prefill (~2.4 s) without a new GEMM kernel, and it is
//! Rule-0-compliant (GEMM stays in rocBLAS).
//!
//! Shapes are the 9B's dominant prefill GEMMs: FFN gate/up is [M,4096] x
//! [12288,4096]^T; FFN down and attention are [M,4096] x [4096,4096]^T.
//!
//! RESULT (gfx1201, ROCm default build, rocblas_gemm_ex f16-in/f16-out, median
//! of 5): ~2 TFLOP/s at EVERY shape — 1.98 TFLOP/s at [4056 x 12288x4096], and
//! flat down to M=16. That MEASURED number eliminates "dequant-once-to-f16 +
//! rocBLAS" as the large-M prefill fix: 4K tokens need 72 TFLOP, which at
//! 2 TFLOP/s is ~36 s — no better than the shipped fused-WMMA path (39.45 s).
//! It also matches the reason the AMD kernel blogs exist: ROCm's library GEMM
//! does not have tuned RDNA4 coverage, so the path to Ollama-class prefill
//! (their ~30 TFLOP/s effective on the same 72 TFLOP) is a hand-written
//! large-M-tile WMMA kernel — i.e. porting the STRUCTURE of FlyDSL
//! `kernels/gemm/rdna_f16_gemm.py` (128x128x32, 4 waves 2x2, double-buffered
//! LDS, a_k_pad=8) with the Q4_K dequant fused into the LDS fill.
//!
//! RUN: GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-rocm --test rocblas_f16_gemm_bench -- --ignored --nocapture

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, BackendStorage, CoreTensorOps, DType, MemoryOps, Shape, Storage};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    Some(RocmDevice::try_new(0).expect("RocmDevice::try_new"))
}

/// f16 bytes for `len` elements: 0x3800 = 0.5, a normal value so nothing
/// takes a denormal fast path.
fn f16_bytes(len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(2 * len);
    for _ in 0..len {
        v.push(0x00);
        v.push(0x38);
    }
    v
}

fn f16_tensor(
    dev: &RocmDevice,
    bytes: &[u8],
    shape: &Shape,
) -> Box<dyn BackendStorage> {
    MemoryOps::from_cpu_bytes(
        dev,
        bytes,
        shape,
        DType {
            arith: ArithType::F16,
            storage: Storage::Native,
        },
    )
    .expect("upload f16")
}

#[test]
#[ignore]
fn rocblas_f16_prefill_shapes() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TESTS=1 + GPU");
        return;
    };

    let shapes = [
        (4056usize, 12288usize, 4096usize),
        (4056, 4096, 4096),
        (1024, 12288, 4096),
        (251, 12288, 4096),
        (64, 12288, 4096),
        (16, 12288, 4096),
        (5, 12288, 4096),
    ];

    eprintln!(
        "{:>6} {:>12} {:>10} {:>10} {:>11}",
        "M", "NxK", "ms", "TFLOP/s", "GB/s moved"
    );
    for (m, n, k) in shapes {
        let a_bytes = f16_bytes(m * k);
        let b_bytes = f16_bytes(n * k);
        let a = f16_tensor(&dev, &a_bytes, &Shape::new(vec![m, k]));
        let b = f16_tensor(&dev, &b_bytes, &Shape::new(vec![n, k]));
        let out_shape = Shape::new(vec![m, n]);

        for _ in 0..2 {
            let (_, h) = dev
                .matmul(a.as_ref(), b.as_ref(), &out_shape)
                .expect("matmul");
            h.synchronize().expect("sync");
        }

        let mut samples = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let (_, h) = dev
                .matmul(a.as_ref(), b.as_ref(), &out_shape)
                .expect("matmul");
            h.synchronize().expect("sync");
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|x, y| x.total_cmp(y));
        let ms = samples[samples.len() / 2];

        let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms / 1e3) / 1e12;
        let gb = (2.0 * (m * k + n * k) as f64 * 2.0 + 4.0 * (m * n) as f64) / 1e9;
        eprintln!(
            "{m:>6} {n:>6}x{k:<6} {ms:>10.3} {tflops:>10.2} {gb:>11.2}"
        );
    }
}
