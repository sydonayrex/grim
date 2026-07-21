//! P1-WI-2 TODO(gpu-verify): local perf methodology on gfx1036 (RDNA2).
//!
//! Validates the fused-dequant-attention kernel's *decode throughput* and the
//! *KV-memory reduction* at 2-8 bit, on the only GPU present here (RDNA2,
//! no WMMA). Absolute tok/s will be lower than RDNA3/4 (which have matrix
//! cores); the speedup ratio vs the CPU dense baseline is the portable signal.
//!
//! Run: `cargo test -p grim-backend-rocm --test kv_dequant_perf -- --ignored --nocapture`

use std::time::Instant;

use grim_kvquant::{
    KvCompressor, KvDequantAttentionConfig, KvQuantConfig, LloydMaxCompressor,
};
use grim_backend_cpu::CpuDevice;
use grim_tensor::{BackendDevice, Device, DType, QuantProvenance, Shape, Tensor};

use grim_backend_rocm::RocmDevice;

/// Dense float attention for a single decode token vs a full KV cache.
/// `q` is [num_heads, head_dim]; `k`/`v` are [kv_len, num_kv_heads, head_dim].
fn dense_attn_1tok(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_len: usize,
    num_kv_heads: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let q_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let kv_head = h / q_per_kv;
        let mut scores = vec![0.0f32; kv_len];
        let mut max_s = f32::NEG_INFINITY;
        for j in 0..kv_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[h * head_dim + d]
                    * k[(j * num_kv_heads + kv_head) * head_dim + d];
            }
            let s = dot * scale;
            scores[j] = s;
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for j in 0..kv_len {
                val += scores[j] * v[(j * num_kv_heads + kv_head) * head_dim + d];
            }
            out[h * head_dim + d] = val / sum;
        }
    }
    out
}

fn main_config(
    dev: &RocmDevice,
    cache_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    key_bits: u8,
    value_bits: u8,
    steps: usize,
) {
    let dtype = DType::F32;
    let shape = Shape::new(vec![cache_len, num_kv_heads, head_dim]);

    let synth = |seed: f32| {
        (0..cache_len * num_kv_heads * head_dim)
            .map(|i| ((i as f32).sin() * 0.4 + seed))
            .collect::<Vec<f32>>()
    };
    let k_data = synth(1.0);
    let v_data = synth(2.0);

    let cpu = CpuDevice::new();
    let k_storage = std::sync::Arc::from(cpu.from_cpu(&k_data, &shape, dtype.clone()).unwrap());
    let v_storage = std::sync::Arc::from(cpu.from_cpu(&v_data, &shape, dtype.clone()).unwrap());
    let keys = Tensor::new(
        k_storage,
        shape.clone(),
        dtype.clone(),
        QuantProvenance::GrimNative,
        Device::Cpu,
    );
    let values = Tensor::new(
        v_storage,
        shape.clone(),
        dtype.clone(),
        QuantProvenance::GrimNative,
        Device::Cpu,
    );

    let compressor = LloydMaxCompressor::with_gpu_attn(
        KvQuantConfig {
            key_bits,
            value_bits,
            ..Default::default()
        },
        KvDequantAttentionConfig { enabled: true },
    );
    let block = compressor.compress(&keys, &values).unwrap();

    let compressed_bytes =
        block.key_bits.len() + block.value_bits.len() + (block.key_meta.len() + block.value_meta.len()) * 4;
    let dense_bytes = cache_len * num_kv_heads * head_dim * 4 * 2;
    let host_ratio = dense_bytes as f32 / compressed_bytes as f32;

    // What the GPU actually ingests = the dispatcher-repacked buffer. Matches
    // dispatch_gpu_fused_attention's bitwidth selection rule (both ≤4 and even
    // head_dim -> 4-bit nibble pair, else 8-bit signed byte).
    let quant_bits: u32 = if key_bits <= 4 && value_bits <= 4 && head_dim % 2 == 0 { 4 } else { 8 };
    let per_elem = if quant_bits == 8 { 1.0 } else { 0.5 };
    let k_bytes = (cache_len * num_kv_heads * head_dim) as f32 * per_elem;
    let v_bytes = k_bytes;
    let scales_bytes = 2.0 * (cache_len * num_kv_heads * 4) as f32; // k + v scales, f32
    let gpu_pack_bytes = k_bytes + v_bytes + scales_bytes;
    let gpu_ratio = dense_bytes as f32 / gpu_pack_bytes as f32;

    // --- GPU decode loop (fused dequant attention) ---
    let q_shape = Shape::new(vec![1, num_heads, head_dim]);
    let q_data = (0..num_heads * head_dim)
        .map(|i| (i as f32).sin() * 0.4 + 0.7)
        .collect::<Vec<f32>>();
    let q_storage = std::sync::Arc::from(cpu.from_cpu(&q_data, &q_shape, dtype.clone()).unwrap());
    let query = Tensor::new(
        q_storage,
        q_shape.clone(),
        dtype.clone(),
        QuantProvenance::GrimNative,
        Device::Cpu,
    );

    // warmup
    for _ in 0..16 {
        let _ = compressor
            .fused_attention(&block, &query, dev, Device::Rocm(0))
            .unwrap();
    }
    let t0 = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..steps {
        let o = compressor
            .fused_attention(&block, &query, dev, Device::Rocm(0))
            .unwrap();
        let v = o.to_vec_f32().unwrap();
        acc += v[0];
    }
    let gpu_elapsed = t0.elapsed();
    let gpu_tok_s = steps as f32 / gpu_elapsed.as_secs_f32();

    // --- CPU dense float baseline (attention only; KV dequantized once) ---
    let (dk, dv) = compressor
        .dequantize_for_attention(&block, &cpu, Device::Cpu)
        .unwrap();
    let dk_v = dk.to_vec_f32().unwrap();
    let dv_v = dv.to_vec_f32().unwrap();

    for _ in 0..16 {
        let _ = dense_attn_1tok(
            &q_data,
            &dk_v,
            &dv_v,
            cache_len,
            num_kv_heads,
            num_heads,
            head_dim,
        );
    }
    let t0 = Instant::now();
    let mut acc2 = 0.0f32;
    for _ in 0..steps {
        let o = dense_attn_1tok(
            &q_data,
            &dk_v,
            &dv_v,
            cache_len,
            num_kv_heads,
            num_heads,
            head_dim,
        );
        acc2 += o[0];
    }
    let cpu_elapsed = t0.elapsed();
    let cpu_tok_s = steps as f32 / cpu_elapsed.as_secs_f32();

    println!(
        "  k{}v{} (GPU {}b) | dense {:.2}MB -> host {:.2}MB ({:.1}x) | GPU-pack {:.2}MB ({:.1}x) | GPU {:.0} tok/s | CPU {:.0} tok/s | {:.1}x",
        key_bits,
        value_bits,
        quant_bits,
        dense_bytes as f32 / 1e6,
        compressed_bytes as f32 / 1e6,
        host_ratio,
        gpu_pack_bytes as f32 / 1e6,
        gpu_ratio,
        gpu_tok_s,
        cpu_tok_s,
        gpu_tok_s / cpu_tok_s,
    );
    // keep the compiler honest about unused accumulators
    assert!(acc.is_finite() && acc2.is_finite());
}

#[test]
#[ignore]
fn gpu_fused_attn_decode_throughput_vs_dense() {
    let dev = RocmDevice::new(0);

    let cache_len = 512;
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = 128;
    let steps = 200;

    println!(
        "\n[gpu-verify] fused-dequant-attn decode on gfx1036 (RDNA2): {} tok cache, {} heads / {} kv-heads, dim {}",
        cache_len, num_heads, num_kv_heads, head_dim
    );
    println!("  config | KV memory | decode throughput (vs CPU dense float baseline)\n");

    // Near-dense GPU reference: 8-bit K/V still flows through the fused kernel.
    main_config(&dev, cache_len, num_heads, num_kv_heads, head_dim, 8, 8, steps);
    // Aggressive low-bit (the KV-quant sweet spot).
    main_config(&dev, cache_len, num_heads, num_kv_heads, head_dim, 4, 4, steps);
    main_config(&dev, cache_len, num_heads, num_kv_heads, head_dim, 3, 4, steps);
    main_config(&dev, cache_len, num_heads, num_kv_heads, head_dim, 2, 2, steps);

    println!("\n[gpu-verify] note: 8-bit GPU run is the 'dense-equivalent' compute cost; lower bits cut KV memory at ~constant kernel cost.");
}
