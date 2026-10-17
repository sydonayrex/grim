//! D2-1 attribution probe: how long does ONE IQ1_S fused-dequant GEMV take at
//! the 27B's dominant decode shape ([17408, 5120], M=1)? The 27B graph arm
//! measures 4395 ms/token over 64 layers with 2.4 GB/s "achieved GEMV
//! bandwidth"; this decides whether that is the GEMV itself or something else
//! per token.
//!
//! The packed bytes are FABRICATED (grid indices from a deterministic pattern,
//! d = 1.0) — this is a TIMING probe, not a correctness one; the kernel's cost
//! depends on the byte layout and the shapes, not the values.
//!
//! Usage: cargo run --release -p grim-backend-rocm --example iq1s_gemv_probe [ordinal] [iters]

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, MemoryOps, QuantOps, Shape};
use std::time::Instant;

fn main() {
    let ordinal: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    // The 27B's recurrent-layer attn_qkv shape (IQ1_S in the GSQ-RCO build).
    // Optional n override (arg 3) for the A-read scaling test: if the cost is
    // dominated by A re-reads (every output thread walks the whole activation
    // row), time stays ~flat as n shrinks at fixed k.
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(17408);
    let k = 5120usize;
    let dev = RocmDevice::try_new(ordinal).expect("try_new");

    const BLK_ELEMS: usize = 256;
    const BLK_BYTES: usize = 50; // llama.cpp block_iq1_s: 2 + 32 + 16
    assert_eq!(k % BLK_ELEMS, 0);
    let row_bytes = (k / BLK_ELEMS) * BLK_BYTES;
    let mut bytes = vec![0u8; n * row_bytes];
    for row in 0..n {
        for blk in 0..(k / BLK_ELEMS) {
            let off = row * row_bytes + blk * BLK_BYTES;
            // d = 1.0 as f16 (0x3C00), little-endian.
            bytes[off] = 0x00;
            bytes[off + 1] = 0x3C;
            for i in 2..BLK_BYTES {
                bytes[off + i] = ((row.wrapping_mul(31).wrapping_add(blk * 7 + i)) & 0xFF) as u8;
            }
        }
    }

    let packed_dt = DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::KQuant(grim_tensor::KQuantScheme::IQ1S),
    };
    let w = dev
        .from_cpu_bytes(&bytes, &Shape::new(vec![n, k]), packed_dt)
        .expect("upload packed weights");

    let act: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let a = dev
        .from_cpu(&act, &Shape::new(vec![1, k]), DType::F32)
        .expect("upload act");
    let out_shape = Shape::new(vec![1, n]);

    // Warmup (JIT + residency).
    for _ in 0..2 {
        let (_o, _h) = dev
            .quantized_matmul(
                a.as_ref(),
                w.as_ref(),
                &[],
                grim_tensor::QuantFormat::Iq1S,
                &out_shape,
            )
            .expect("warmup gemv");
    }
    dev.synchronize();

    let weight_bytes = n * row_bytes;
    let t0 = Instant::now();
    for _ in 0..iters {
        let (_o, _h) = dev
            .quantized_matmul(
                a.as_ref(),
                w.as_ref(),
                &[],
                grim_tensor::QuantFormat::Iq1S,
                &out_shape,
            )
            .expect("gemv");
    }
    dev.synchronize();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gbs = (weight_bytes as f64) / (ms / 1000.0) / 1e9;
    println!(
        "[iq1s-gemv] n={n} k={k} m=1: {ms:.3} ms/gemv, {gbs:.2} GB/s over {iters} iters \
         ({:.1} MB weight bytes)",
        weight_bytes as f64 / 1e6
    );
    println!(
        "[iq1s-gemv] 64 layers x 1 such weight per token => {:.0} ms/token if this is the \
         whole cost (27B graph arm measured 4395 ms/token)",
        ms * 64.0
    );
}
