//! K-quant GEMV probe: how fast is each K-quant format at the 27B's decode
//! shape, and which kernel does it actually take?
//!
//! WHY THIS PROBE EXISTS. The group-hoisting recipe (hoist the per-weight
//! block header + scale decode out of the inner loop) was proven on the EIGHT
//! IQ formats: it took the 27B's dominant GEMV from ~40 ms per tensor to
//! ~0.5 ms, a 6.3x win on the prompt path, bit-identically.
//!
//! **The K-quants never got the recipe.** Q4_K, Q5_K, Q6_K, Q3_K and Q2_K
//! still route through the WMMA fused-dequant leg, which the 27B's profile
//! showed at ~2.5 GFLOP/s and 19.6 s of a 21.6 s run. So the build that
//! actually ships the 27B (Q4_K_M) is the one carrying the slow leg — and
//! the Q4_K_M 27B is the build that needs TWO CARDS, so its slow leg is
//! paid across the interconnect.
//!
//! This probe measures every K-quant format at the 27B's dominant decode
//! shape ([17408, 5120], M=1) and at the prompt shape (M=64), so the
//! before/after is a number and not an impression.
//!
//! The packed bytes are FABRICATED (deterministic pattern, d = 1.0) — this is
//! a TIMING probe, not a correctness one; the kernel's cost depends on the
//! byte layout and the shapes, not the values.
//!
//! Usage:
//!   cargo run --release -p grim-backend-rocm --example kquant_gemv_probe \
//!     [ordinal] [iters]

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, MemoryOps, QuantOps, Shape};
use std::time::Instant;

/// One K-quant format: its super-block width, its packed bytes per block,
/// and the QuantFormat the routing takes.
struct KQuant {
    name: &'static str,
    format: grim_tensor::QuantFormat,
    block_elems: usize,
    block_bytes: usize,
}

const K_QUANTS: &[KQuant] = &[
    KQuant {
        name: "Q4_K",
        format: grim_tensor::QuantFormat::Q4K,
        block_elems: 256,
        block_bytes: 144,
    },
    KQuant {
        name: "Q5_K",
        format: grim_tensor::QuantFormat::Q5K,
        block_elems: 256,
        block_bytes: 176,
    },
    KQuant {
        name: "Q6_K",
        format: grim_tensor::QuantFormat::Q6K,
        block_elems: 256,
        block_bytes: 210,
    },
    KQuant {
        name: "Q3_K",
        format: grim_tensor::QuantFormat::Q3K,
        block_elems: 256,
        block_bytes: 110,
    },
    KQuant {
        name: "Q2_K",
        format: grim_tensor::QuantFormat::Q2K,
        block_elems: 256,
        block_bytes: 84,
    },
];

fn main() {
    let ordinal: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    // The 27B's dominant projection shape: n = 17408 output columns, k = 5120.
    let n: usize = 17408;
    let k: usize = 5120;

    let dev = RocmDevice::try_new(ordinal).expect("try_new");

    println!(
        "[kquant-gemv] n={n} k={k} on ordinal {ordinal}\n\
         [kquant-gemv] the 27B's dominant projection shape. The IQ formats' grouped\
         \n[kquant-gemv] scalar recipe reached ~0.5 ms/tensor; the K-quants are the\
         \n[kquant-gemv] formats that never got it."
    );

    for q in K_QUANTS {
        for m in [1usize, 64] {
            let row_bytes = (k / q.block_elems) * q.block_bytes;
            let mut bytes = vec![0u8; n * row_bytes];
            // A deterministic non-uniform pattern so a kernel that reads
            // garbage cannot look cheap by reading zeros.
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = ((i.wrapping_mul(31).wrapping_add(7)) & 0xFF) as u8;
            }
            // The fp16 scale header must be a valid finite value or a
            // dequant that multiplies by it yields inf/NaN and the timing
            // measures a different kernel (the NaN path can be faster).
            for blk in 0..(k / q.block_elems) {
                let off = blk * q.block_bytes;
                bytes[off] = 0x00;
                bytes[off + 1] = 0x3C; // f16 1.0
            }

            let packed_dt = DType {
                arith: grim_tensor::ArithType::F32,
                storage: grim_tensor::Storage::from(q.format),
            };
            let w = dev
                .from_cpu_bytes(&bytes, &Shape::new(vec![n, k]), packed_dt)
                .expect("upload packed weights");
            let act: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
            let a = dev
                .from_cpu(&act, &Shape::new(vec![m, k]), DType::F32)
                .expect("upload act");
            let out_shape = Shape::new(vec![m, n]);

            // Warmup (JIT + residency).
            for _ in 0..2 {
                let (_o, _h) = dev
                    .quantized_matmul(a.as_ref(), w.as_ref(), &[], q.format, &out_shape)
                    .expect("warmup gemv");
            }
            dev.synchronize();

            let t0 = Instant::now();
            for _ in 0..iters {
                let (_o, _h) = dev
                    .quantized_matmul(a.as_ref(), w.as_ref(), &[], q.format, &out_shape)
                    .expect("gemv");
            }
            dev.synchronize();
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            let gbs = (bytes.len() as f64) / (ms / 1000.0) / 1e9;
            println!(
                "[kquant-gemv] {:<5} m={:<3} {:>8.3} ms/gemv  {:>7.2} GB/s  ({} MB weights)",
                q.name,
                m,
                ms,
                gbs,
                bytes.len() / 1_000_000
            );
        }
    }
}
