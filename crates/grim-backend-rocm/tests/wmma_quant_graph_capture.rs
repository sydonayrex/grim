//! SPEED-ROC: graph-capture acceptance for the consolidated WMMA
//! fused-dequant quant GEMM kernels (`grim_wmma_fused_dequant_*`).
//!
//! Proves the quant WMMA path can be captured into a HIP graph and replayed
//! with output matching the eager path within FP16 quantizer tolerance — the
//! wiring precondition for replaying decode-step quant GEMMs with near-zero
//! launch overhead.
//!
//! PASSED: 2026-09-09 on gfx1201 (ROCm).
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test wmma_quant_graph_capture

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, KQuantScheme, MemoryOps, QuantOps, QuantFormat, Shape};
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref CAP_MUTEX: Mutex<()> = Mutex::new(());
}

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    unsafe {
        std::env::set_var("GRIM_CAPTURE_GRAPH", "1");
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

/// Build FP32 activation A[m,k] and pack B[n,k] to Q8_0 on the host.
fn build_q80_case(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32 - 6.0) * 0.04).collect();

    // Q8_0 pack: per 32-elem block -> fp16 scale + 32 i8 codes.
    let blocks = k / 32;
    let row_bytes = blocks * 34;
    let mut packed = vec![0u8; n * row_bytes];
    for col in 0..n {
        let row_start = col * k;
        for blk in 0..blocks {
            let base = blk * 32;
            let mut max_a = 0.0f32;
            for j in 0..32 {
                max_a = max_a.max((b_f32[row_start + base + j]).abs());
            }
            let d = (max_a / 127.0).max(1e-30);
            // fp16 scale (little-endian)
            let bits = half::f16::from_f32(d).to_bits();
            let out = &mut packed[col * row_bytes + blk * 34..col * row_bytes + blk * 34 + 34];
            out[0] = (bits & 0xFF) as u8;
            out[1] = ((bits >> 8) & 0xFF) as u8;
            for j in 0..32 {
                let code = ((b_f32[row_start + base + j] / d).round().clamp(-127.0, 127.0) as i8) as u8;
                out[2 + j] = code;
            }
        }
    }
    (a, packed)
}

/// CPU reference: C[m,n] = sum_k A[m,k] * (d_b[n] * code).
fn reference_q80(a: &[f32], b_packed: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let blocks = k / 32;
    let row_bytes = blocks * 34;
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let brow = &b_packed[col * row_bytes..(col + 1) * row_bytes];
            let mut acc = 0.0f32;
            for blk in 0..blocks {
                let base: &[u8] = &brow[blk * 34..blk * 34 + 34];
                let scale_bits = u16::from_le_bytes([base[0], base[1]]);
                let d = half::f16::from_bits(scale_bits).to_f32();
                for j in 0..32 {
                    let code = base[2 + j] as i8 as f32;
                    acc += a[row * k + blk * 32 + j] * d * code;
                }
            }
            c[row * n + col] = acc;
        }
    }
    c
}

#[test]
fn wmma_quant_graph_capture_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let _lock = CAP_MUTEX.lock().unwrap();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return Ok(());
    };

    // Decode-shaped: m=1 routes through WMMA (m <= wmma_max_m default 4).
    let (m, n, k) = (1, 1024, 1024);
    let (a, b_packed) = build_q80_case(m, n, k);
    let want = reference_q80(&a, &b_packed, m, n, k);

    let a_shape = Shape::new(vec![m, k]);
    let a_dev = CoreTensorOps::from_cpu(&dev, &a, &a_shape, DType::F32).expect("upload A");

    let b_dev = {
        let q_dtype = DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::KQuant(KQuantScheme::Q80),
        };
        MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype)
            .expect("upload packed B")
    };

    // Eager reference (warms JIT + resolved-kernel cache).
    let out_shape = Shape::new(vec![m, n]);
    let (out_eager_h, h_eager) = dev
        .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &[], QuantFormat::Q8_0, &out_shape)
        .expect("eager quant matmul");
    h_eager.synchronize().expect("sync eager");
    let eager = out_eager_h.to_cpu_vec_f32().expect("read eager");

    // Capture: only the quant matmul runs inside the bracket (inputs already
    // on-device; active_stream() returns the capture stream during capture).
    let key = "wmma_q80_decode";
    dev.begin_graph_capture(key).expect("begin capture");
    let (out_cap_h, h_cap) = dev
        .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &[], QuantFormat::Q8_0, &out_shape)
        .expect("captured quant matmul");
    dev.end_graph_capture(key).expect("end capture");
    assert!(dev.replay_graph(key)?, "replay must launch captured graph");
    h_cap.synchronize().expect("sync replay");
    let replayed = out_cap_h.to_cpu_vec_f32().expect("read replay");

    let check = |label: &str, got: &[f32]| {
        assert_eq!(want.len(), got.len(), "{label}: length mismatch");
        let mut max_diff = 0.0f32;
        for i in 0..want.len() {
            max_diff = max_diff.max((want[i] - got[i]).abs());
        }
        // FP16 accumulation over k=1024 terms -> ~1e-2 tolerance.
        assert!(
            max_diff < 5e-2,
            "{label}: max_diff={max_diff} exceeds FP16 quant tolerance"
        );
        eprintln!("[{label}] max_diff={max_diff} OK");
    };
    check("eager", &eager);
    check("replayed", &replayed);
    dev.synchronize();
    Ok(())
}
