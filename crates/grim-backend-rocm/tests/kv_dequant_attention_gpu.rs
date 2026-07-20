//! P1-WI-2 acceptance test: the wired `grim_kv_dequant_attention` HIP kernel
//! produces results matching a pure-float fused-attention reference on the real
//! GPU (gfx1036 in this env).
//!
//! The GPU path (via `BackendDevice::kv_dequant_attention` ->
//! `RocmDevice::kv_dequant_attention`) re-packs the dequantized K/V as
//! signed 8-bit with a per-buffer scale, so the kernel's signed 8-bit
//! dequant path reproduces the f32 values up to `scale/255` quantization
//! error (tiny for the small-magnitude K/V this compressor emits). The
//! reference below computes *true* float attention on the same dequantized
//! K/V — it does NOT use the SageAttention INT8 simulation that the
//! `LloydMaxCompressor` CPU path folds in, so the two represent the same
//! math and differ only by float/quantization noise.
//!
//! Skill attribution:
//! - `rust-tdd` / `strong-tests` — assert GPU == CPU within a tight
//!   tolerance; no snapshot, no `unwrap()` in the test body.
//! - `rust-gpu-discipline` — runs on the real device, never a CPU fallback.

use std::sync::Arc;

use grim_backend_cpu::CpuDevice;
use grim_backend_rocm::RocmDevice;
use grim_kvquant::{KvCompressor, KvQuantConfig, KvDequantAttentionConfig, LloydMaxCompressor};
use grim_tensor::{ArithType, BackendDevice, DType, Device, QuantProvenance, Shape, Tensor};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn f32_dtype() -> DType {
    DType { arith: ArithType::F32, storage: grim_tensor::Storage::Native }
}

/// Pure-float fused attention on already-dequantized f32 K/V, matching the
/// causal contract the GPU kernel implements (cache_offset = kv_seq_len-1,
/// so every query position attends the full KV cache -> effectively unmasked
/// here). Returns `[seq, heads, head_dim]`.
fn float_reference(
    q: &[f32], k: &[f32], v: &[f32],
    num_tokens: usize, num_heads: usize, head_dim: usize,
) -> Vec<f32> {
    let num_kv_heads = k.len() / (num_tokens * head_dim);
    let q_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; num_tokens * num_heads * head_dim];
    for t in 0..num_tokens {
        for h in 0..num_heads {
            let kv_head = h / q_per_kv;
            let mut scores = vec![0.0f32; num_tokens];
            let mut max_score = f32::NEG_INFINITY;
            for kt in 0..num_tokens {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = (t * num_heads + h) * head_dim + d;
                    let k_idx = (kt * num_kv_heads + kv_head) * head_dim + d;
                    dot += q[q_idx] * k[k_idx];
                }
                let s = dot * scale;
                scores[kt] = s;
                if s > max_score { max_score = s; }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for kt in 0..num_tokens {
                    let v_idx = (kt * num_kv_heads + kv_head) * head_dim + d;
                    val += scores[kt] * v[v_idx];
                }
                let o_idx = (t * num_heads + h) * head_dim + d;
                out[o_idx] = val / sum;
            }
        }
    }
    out
}

#[test]
fn gpu_fused_attention_matches_cpu_reference() -> TestResult {
    // (num_heads, num_kv_heads, head_dim): covers 1:1 heads and
    // GQA (kv_heads < heads), and head_dim <= 64 (single chunk)
    // vs > 64 (multi-chunk).
    for &(num_heads, num_kv_heads, head_dim) in &[
        (4usize, 4usize, 64usize),
        (8usize, 2usize, 128usize),
        (4usize, 1usize, 96usize),
    ] {
        run_case(num_heads, num_kv_heads, head_dim)?;
    }
    Ok(())
}

fn run_case(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> TestResult {
    let dev = RocmDevice::new(0);

    let num_tokens = 4usize;

    let shape = Shape::new(vec![num_tokens, num_kv_heads, head_dim]);
    let dtype = f32_dtype();

    // Synthetic f32 K/V with a small magnitude so the signed 8-bit packing
    // round-trips with negligible error.
    let synth = |seed, heads| {
        (0..num_tokens * heads * head_dim)
            .map(|i| ((i as f32).sin() * 0.5 + (seed as f32) * 1e-3))
            .collect::<Vec<f32>>()
    };
    let k_data = synth(1, num_kv_heads);
    let v_data = synth(2, num_kv_heads);
    // Query has the full `num_heads` heads (GQA: > num_kv_heads).
    let q_data = synth(3, num_heads);

    let cpu = CpuDevice::new();
    let k_storage = Arc::from(cpu.from_cpu(&k_data, &shape, dtype.clone())?);
    let v_storage = Arc::from(cpu.from_cpu(&v_data, &shape, dtype.clone())?);
    let keys = Tensor::new(k_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);
    let values = Tensor::new(v_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);

    let gpu_compressor = LloydMaxCompressor::with_gpu_attn(
        KvQuantConfig::default(),
        KvDequantAttentionConfig { enabled: true },
    );
    let block = gpu_compressor.compress(&keys, &values)?;

    // Query has [num_tokens, num_heads, head_dim] layout.
    let q_shape = Shape::new(vec![num_tokens, num_heads, head_dim]);
    let q_storage = Arc::from(cpu.from_cpu(&q_data, &q_shape, dtype.clone())?);
    let query = Tensor::new(q_storage, q_shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);

    let gpu_dev: &dyn grim_tensor::BackendDevice = &dev;
    let gpu_out = gpu_compressor.fused_attention(&block, &query, gpu_dev, Device::Rocm(0))?;

    // Pure-float reference on the dequantized K/V (no INT8 simulation).
    let (ref_keys, ref_values) = gpu_compressor.dequantize_for_attention(&block, &cpu, Device::Cpu)?;
    let ref_k = ref_keys.to_vec_f32()?;
    let ref_v = ref_values.to_vec_f32()?;
    let cpu_vec = float_reference(&q_data, &ref_k, &ref_v, num_tokens, num_heads, head_dim);

    let gpu_vec = gpu_out.to_vec_f32()?;
    assert_eq!(gpu_vec.len(), cpu_vec.len(), "output length mismatch");

    let mut max_err = 0.0f32;
    for (g, c) in gpu_vec.iter().zip(cpu_vec.iter()) {
        max_err = max_err.max((g - c).abs());
    }
    assert!(
        max_err < 0.05,
        "GPU fused attention diverged from CPU reference (heads={num_heads}, kv={num_kv_heads}, dim={head_dim}): max_err={max_err}"
    );
    Ok(())
}
