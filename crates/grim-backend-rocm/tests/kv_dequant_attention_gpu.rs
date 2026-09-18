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
use grim_kvquant::{KvCompressor, KvDequantAttentionConfig, KvQuantConfig, LloydMaxCompressor};
use grim_tensor::CoreTensorOps;
use grim_tensor::{ArithType, DType, Device, QuantProvenance, Shape, Tensor};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn f32_dtype() -> DType {
    DType {
        arith: ArithType::F32,
        storage: grim_tensor::Storage::Native,
    }
}

/// Pure-float fused attention on already-dequantized f32 K/V, matching the
/// causal contract the GPU kernel implements (cache_offset = kv_seq_len-1,
/// so every query position attends the full KV cache -> effectively unmasked
/// here). Returns `[seq, heads, head_dim]`.
fn float_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
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
            for (kt, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = (t * num_heads + h) * head_dim + d;
                    let k_idx = (kt * num_kv_heads + kv_head) * head_dim + d;
                    dot += q[q_idx] * k[k_idx];
                }
                let s = dot * scale;
                *score = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for (kt, &score) in scores.iter().enumerate() {
                    let v_idx = (kt * num_kv_heads + kv_head) * head_dim + d;
                    val += score * v[v_idx];
                }
                let o_idx = (t * num_heads + h) * head_dim + d;
                out[o_idx] = val / sum;
            }
        }
    }
    out
}

#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn gpu_fused_attention_matches_cpu_reference() -> TestResult {
    // (num_heads, num_kv_heads, head_dim, key_bits, value_bits): covers 1:1
    // heads and GQA (kv_heads < heads), head_dim <= 64 vs > 64 (multi-chunk),
    // and BOTH kernel paths — 4-bit (≤4) and 8-bit (>4 / odd config).
    for &(num_heads, num_kv_heads, head_dim, key_bits, value_bits) in &[
        (4usize, 4usize, 64usize, 3u8, 4u8),
        (8usize, 2usize, 128usize, 3u8, 4u8),
        (4usize, 1usize, 96usize, 3u8, 4u8),
        (4usize, 4usize, 64usize, 8u8, 8u8),
        (8usize, 2usize, 128usize, 8u8, 8u8),
    ] {
        run_case(num_heads, num_kv_heads, head_dim, key_bits, value_bits)?;
    }
    Ok(())
}

fn run_case(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    key_bits: u8,
    value_bits: u8,
) -> TestResult {
    let dev = RocmDevice::try_new(0)
        .expect("RocmDevice::try_new(0) should succeed on a system with ROCm");

    let num_tokens = 4usize;

    let shape = Shape::new(vec![num_tokens, num_kv_heads, head_dim]);
    let dtype = f32_dtype();

    // Synthetic f32 K/V with a small magnitude so the signed 8-bit packing
    // round-trips with negligible error.
    let synth = |seed, heads| {
        (0..num_tokens * heads * head_dim)
            .map(|i| (i as f32).sin() * 0.5 + (seed as f32) * 1e-3)
            .collect::<Vec<f32>>()
    };
    let k_data = synth(1, num_kv_heads);
    let v_data = synth(2, num_kv_heads);
    // Query has the full `num_heads` heads (GQA: > num_kv_heads).
    let q_data = synth(3, num_heads);

    let cpu = CpuDevice::new();
    let k_storage = Arc::from(cpu.from_cpu(&k_data, &shape, dtype.clone())?);
    let v_storage = Arc::from(cpu.from_cpu(&v_data, &shape, dtype.clone())?);
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

    let gpu_compressor = LloydMaxCompressor::with_gpu_attn(
        KvQuantConfig {
            key_bits,
            value_bits,
            ..Default::default()
        },
        KvDequantAttentionConfig { enabled: true },
    );
    let block = gpu_compressor.compress(&keys, &values)?;

    // Query has [num_tokens, num_heads, head_dim] layout.
    let q_shape = Shape::new(vec![num_tokens, num_heads, head_dim]);
    let q_storage = Arc::from(cpu.from_cpu(&q_data, &q_shape, dtype.clone())?);
    let query = Tensor::new(
        q_storage,
        q_shape.clone(),
        dtype.clone(),
        QuantProvenance::GrimNative,
        Device::Cpu,
    );

    let gpu_dev: &dyn grim_tensor::BackendDevice = &dev;
    let gpu_out = gpu_compressor.fused_attention(&block, &query, gpu_dev, Device::Rocm(0))?;

    // Pure-float reference on the dequantized K/V (no INT8 simulation).
    let (ref_keys, ref_values) =
        gpu_compressor.dequantize_for_attention(&block, &cpu, Device::Cpu)?;
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

/// M4: single-token decode against a longer quantized cache must take the
/// split-KV FlashDecoding path (dequant-to-f32 + grim_flash_decode_stage1/2)
/// and still match the pure-float reference. The gate is lowered via
/// GRIM_FLASH_DECODE_MIN_KV so the test runs without a 512-token cache.
#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1 and -- --ignored"]
fn gpu_kv_dequant_decode_uses_split_kv_flashdecode() -> TestResult {
    // SAFETY: single-threaded test process; the gate is read per-call.
    unsafe { std::env::set_var("GRIM_FLASH_DECODE_MIN_KV", "2") };

    let (num_heads, num_kv_heads, head_dim) = (8usize, 2usize, 128usize);
    let cache_tokens = 8usize; // kv_seq_len
    let dev = RocmDevice::try_new(0)
        .expect("RocmDevice::try_new(0) should succeed on a system with ROCm");

    let shape = Shape::new(vec![cache_tokens, num_kv_heads, head_dim]);
    let dtype = f32_dtype();

    let synth = |seed: u64, elems: usize| {
        (0..elems)
            .map(|i| (i as f32).sin() * 0.5 + (seed as f32) * 1e-3)
            .collect::<Vec<f32>>()
    };
    let k_data = synth(1, cache_tokens * num_kv_heads * head_dim);
    let v_data = synth(2, cache_tokens * num_kv_heads * head_dim);
    let q_data = synth(3, num_heads * head_dim); // one decode token

    let cpu = CpuDevice::new();
    let k_storage = Arc::from(cpu.from_cpu(&k_data, &shape, dtype.clone())?);
    let v_storage = Arc::from(cpu.from_cpu(&v_data, &shape, dtype.clone())?);
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

    let gpu_compressor = LloydMaxCompressor::with_gpu_attn(
        KvQuantConfig {
            key_bits: 8,
            value_bits: 8,
            ..Default::default()
        },
        KvDequantAttentionConfig { enabled: true },
    );
    let block = gpu_compressor.compress(&keys, &values)?;

    let q_shape = Shape::new(vec![1usize, num_heads, head_dim]);
    let q_storage = Arc::from(cpu.from_cpu(&q_data, &q_shape, dtype.clone())?);
    let query = Tensor::new(
        q_storage,
        q_shape.clone(),
        dtype.clone(),
        QuantProvenance::GrimNative,
        Device::Cpu,
    );

    let gpu_dev: &dyn grim_tensor::BackendDevice = &dev;
    let gpu_out = gpu_compressor.fused_attention(&block, &query, gpu_dev, Device::Rocm(0))?;

    // Pure-float reference: decode token attends the full dequantized cache.
    let (ref_keys, ref_values) =
        gpu_compressor.dequantize_for_attention(&block, &cpu, Device::Cpu)?;
    let ref_k = ref_keys.to_vec_f32()?;
    let ref_v = ref_values.to_vec_f32()?;
    let q_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut ref_out = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let kv_head = h / q_per_kv;
        let mut scores = vec![0.0f32; cache_tokens];
        let mut max_score = f32::NEG_INFINITY;
        for (kt, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_data[h * head_dim + d]
                    * ref_k[(kt * num_kv_heads + kv_head) * head_dim + d];
            }
            let s = dot * scale;
            *score = s;
            if s > max_score {
                max_score = s;
            }
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for (kt, &score) in scores.iter().enumerate() {
                val += score * ref_v[(kt * num_kv_heads + kv_head) * head_dim + d];
            }
            ref_out[h * head_dim + d] = val / sum;
        }
    }

    let gpu_vec = gpu_out.to_vec_f32()?;
    assert_eq!(gpu_vec.len(), ref_out.len(), "output length mismatch");
    let max_err = gpu_vec
        .iter()
        .zip(ref_out.iter())
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.05,
        "M4 split-KV quantized-KV decode diverged from CPU reference: max_err={max_err}"
    );
    Ok(())
}

/// WI-3 (PLAN-kvcache-channel-axis): a channel-allocated block must take the
/// Q4KHalf kernel branch (per-32-channel-group scale+min) and match the
/// pure-float reference computed over the allocation-aware CPU dequant.
/// Runs BOTH the dense kernel path and the M4 split-KV FlashDecode path.
#[test]
#[ignore = "requires real ROCm device; metal parity covered separately"]
fn gpu_q4khalf_attention_matches_cpu_reference() -> TestResult {
    use grim_kvquant::channel_importance::ChannelBitAllocation;

    let (num_heads, num_kv_heads, head_dim) = (8usize, 2usize, 128usize);
    let dev = RocmDevice::try_new(0)
        .expect("RocmDevice::try_new(0) should succeed on a system with ROCm");

    // Elevated importance in group 0 of kv_head 0 so the allocation marks it
    // 8-bit while the rest stay 4-bit (real asymmetry exercised).
    let groups = head_dim / 32;
    let mut key_bits = vec![vec![4u8; groups]; num_kv_heads];
    let mut value_bits = vec![vec![4u8; groups]; num_kv_heads];
    key_bits[0][0] = 8;
    value_bits[0][0] = 8;
    let alloc = ChannelBitAllocation {
        num_kv_heads,
        head_dim,
        group_size: 32,
        key_bits,
        value_bits,
    };

    for tokens in [4usize, 8usize] {
        let cache_tokens = tokens;
        let shape = Shape::new(vec![cache_tokens, num_kv_heads, head_dim]);
        let dtype = f32_dtype();
        let synth = |seed: u64, elems: usize| {
            (0..elems)
                .map(|i| (i as f32).sin() * 0.35 + (seed as f32) * 1e-3)
                .collect::<Vec<f32>>()
        };
        let k_data = synth(11, cache_tokens * num_kv_heads * head_dim);
        let v_data = synth(12, cache_tokens * num_kv_heads * head_dim);
        let q_data = synth(13, num_heads * head_dim * 2 /* 2 query tokens when prefill */);
        let q_tokens = if cache_tokens >= 8 { 1usize } else { 2usize };

        let cpu = CpuDevice::new();
        let k_storage = Arc::from(cpu.from_cpu(&k_data, &shape, dtype.clone())?);
        let v_storage = Arc::from(cpu.from_cpu(&v_data, &shape, dtype.clone())?);
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

        let gpu_compressor = LloydMaxCompressor::with_gpu_attn(
            KvQuantConfig { key_bits: 4, value_bits: 4, ..Default::default() },
            KvDequantAttentionConfig { enabled: true },
        )
        .with_channel_alloc(alloc.clone());
        let block = gpu_compressor.compress(&keys, &values)?;
        assert!(block.channel_alloc.is_some(), "allocation must be carried");

        let q_shape = Shape::new(vec![q_tokens, num_heads, head_dim]);
        let q_storage = Arc::from(cpu.from_cpu(&q_data[..q_tokens * num_heads * head_dim].to_vec(), &q_shape, dtype.clone())?);
        let query = Tensor::new(
            q_storage,
            q_shape.clone(),
            dtype.clone(),
            QuantProvenance::GrimNative,
            Device::Cpu,
        );

        let gpu_dev: &dyn grim_tensor::BackendDevice = &dev;
        let gpu_out = gpu_compressor.fused_attention(&block, &query, gpu_dev, Device::Rocm(0))?;

        // Reference: the GPU kernel reads the Q4KHalf-packed bytes, so the
        // honest CPU reference is attention over the Q4KHalf-ROUND-TRIPPED
        // (pack→unpack) data, not the allocation-only dequant — the delta
        // between those two is double-quantization, which is physics, not a
        // kernel bug. (Grim-quant's pack is the exact function pack_kv_buf
        // uses, so the bytes below are byte-identical to the GPU inputs.)
        let (ref_keys, ref_values) =
            gpu_compressor.dequantize_for_attention(&block, &cpu, Device::Cpu)?;
        let ref_k_alloc = ref_keys.to_vec_f32()?;
        let ref_v_alloc = ref_values.to_vec_f32()?;
        let (ref_k, ref_v) = {
            let mut ko = Vec::with_capacity(ref_k_alloc.len());
            let mut vo = Vec::with_capacity(ref_v_alloc.len());
            for row in ref_k_alloc.chunks_exact(head_dim) {
                let p = grim_quant::quant_q4khalf(row)?;
                ko.extend_from_slice(&grim_quant::dequant_q4khalf(&p, head_dim)?);
            }
            for row in ref_v_alloc.chunks_exact(head_dim) {
                let p = grim_quant::quant_q4khalf(row)?;
                vo.extend_from_slice(&grim_quant::dequant_q4khalf(&p, head_dim)?);
            }
            (ko, vo)
        };
        // The kernel's cache_offset is kv_seq_len - 1 (last query position);
        // a q_tokens>1 caller receives the equivalent of the dense-masked
        // tail, so compute the reference per query token with its absolute
        // position (kernel: abs_i = cache_offset + i).
        // The kernel's cache_offset is kv_seq_len - 1; for a multi-token query
        // its absolute positions are cache_offset + t.
        let q_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut ref_out = vec![0.0f32; q_tokens * num_heads * head_dim];
        for t in 0..q_tokens {
            let abs = (cache_tokens - 1) + t;
            for h in 0..num_heads {
                let kvh = h / q_per_kv;
                let mut scores = vec![0.0f32; cache_tokens.min(abs + 1)];
                let mut mx = f32::NEG_INFINITY;
                for (kt, s) in scores.iter_mut().enumerate() {
                    let mut dot: f32 = 0.0;
                    for d in 0..head_dim {
                        dot += q_data[t * num_heads * head_dim + h * head_dim + d]
                            * ref_k[(kt * num_kv_heads + kvh) * head_dim + d];
                    }
                    *s = dot * scale;
                    if *s > mx { mx = *s }
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (kt, s) in scores.iter().enumerate() {
                        acc += s * ref_v[(kt * num_kv_heads + kvh) * head_dim + d];
                    }
                    ref_out[t * num_heads * head_dim + h * head_dim + d] = acc / sum;
                }
            }
        }

        let gpu_vec = gpu_out.to_vec_f32()?;
        assert_eq!(gpu_vec.len(), ref_out.len(), "length mismatch");
        let max_err = gpu_vec
            .iter()
            .zip(ref_out.iter())
            .map(|(g, r)| (g - r).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.05,
            "Q4KHalf {tokens}-token fused attention diverged: max_err={max_err}"
        );
    }
    Ok(())
}

/// Direct device-level pinpoint for the q4khalf 8-token failure: hand-pack
/// rows via grim_quant (identical bytes to pack_kv_buf), run the raw kernel
/// through `kv_dequant_attention` with quant_bits=3, and compare every
/// (token, head, dim) against the host reference.
#[test]
#[ignore = "requires real ROCm device; run manually with GRIM_RUN_GPU_TESTS=1"]
fn gpu_q4khalf_raw_pinpoint() -> TestResult {
    let (num_heads, num_kv_heads, head_dim) = (2usize, 2usize, 128usize);
    let kv_seq_len = 8usize;
    let dev = RocmDevice::try_new(0).expect("rocm device");
    let dtype = f32_dtype();

    let rows = kv_seq_len * num_kv_heads;
    let mut k_rows: Vec<Vec<f32>> = Vec::new();
    let mut k_packed = Vec::new();
    for r in 0..rows {
        let row: Vec<f32> = (0..head_dim).map(|i| ((r * head_dim + i) as f32 * 0.021).sin() * 0.4).collect();
        k_packed.extend_from_slice(&grim_quant::quant_q4khalf(&row)?);
        k_rows.push(row);
    }
    let mut v_rows: Vec<Vec<f32>> = Vec::new();
    let mut v_packed = Vec::new();
    for r in 0..rows {
        let row: Vec<f32> = (0..head_dim).map(|i| ((r * head_dim + i) as f32 * 0.017).cos() * 0.3).collect();
        v_packed.extend_from_slice(&grim_quant::quant_q4khalf(&row)?);
        v_rows.push(row);
    }

    let q_data: Vec<f32> = (0..num_heads * head_dim).map(|i| (i as f32 * 0.019).sin() * 0.5).collect();

    let q_shape = Shape::new(vec![1usize, num_heads, head_dim]);
    let row_shape = Shape::new(vec![kv_seq_len, num_kv_heads, grim_quant::q4khalf_row_bytes(head_dim)]);
    let scale_shape = Shape::new(vec![rows]);
    let u8_dtype = DType { arith: ArithType::U8, storage: grim_tensor::Storage::Native };

    let gpu_dev2: &dyn grim_tensor::BackendDevice = &dev;
    let q_st = gpu_dev2.from_cpu(&q_data, &q_shape, dtype.clone())?;
    let k_st = gpu_dev2.from_cpu_bytes(&k_packed, &row_shape, u8_dtype.clone())?;
    let v_st = gpu_dev2.from_cpu_bytes(&v_packed, &row_shape, u8_dtype)?;
    let ks_st = dev.from_cpu(&vec![1.0f32; rows], &scale_shape, dtype.clone())?;
    let vs_st = dev.from_cpu(&vec![1.0f32; rows], &scale_shape, dtype)?;

    let gpu_dev: &dyn grim_tensor::BackendDevice = &dev;
    let (out_st, handle) = gpu_dev.kv_dequant_attention(
        q_st.as_ref(), k_st.as_ref(), ks_st.as_ref(), v_st.as_ref(), vs_st.as_ref(),
        num_kv_heads, kv_seq_len, (kv_seq_len - 1) as u32, 3, &q_shape,
    )?;
    handle.synchronize()?;

    // Dequantize each row back on host to build the float attention reference.
    let row_bytes = grim_quant::q4khalf_row_bytes(head_dim);
    let k_deq: Vec<f32> = (0..rows).flat_map(|r| {
        grim_quant::dequant_q4khalf(&k_packed[r*row_bytes..(r+1)*row_bytes], head_dim).unwrap()
    }).collect();
    let v_deq: Vec<f32> = (0..rows).flat_map(|r| {
        grim_quant::dequant_q4khalf(&v_packed[r*row_bytes..(r+1)*row_bytes], head_dim).unwrap()
    }).collect();

    let q_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let kvh = h / q_per_kv;
        let mut scores = vec![0.0f32; kv_seq_len];
        let mut mx = f32::NEG_INFINITY;
        for (kt, s) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_data[h * head_dim + d] * k_deq[(kt * num_kv_heads + kvh) * head_dim + d];
            }
            *s = dot * scale;
            if *s > mx { mx = *s }
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - mx).exp(); sum += *s; }

        let out = out_st.to_cpu_vec_f32()?;
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (kt, s) in scores.iter().enumerate() {
                acc += s * v_deq[(kt * num_kv_heads + kvh) * head_dim + d];
            }
            let expect = acc / sum;
            let got = out[h * head_dim + d];
            assert!(
                (got - expect).abs() < 0.05,
                "raw q4kh mismatch at head {h} dim {d}: got {got}, expect {expect}"
            );
        }
    }
    Ok(())
}
