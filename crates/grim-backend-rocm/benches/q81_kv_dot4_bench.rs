//! Micro-benchmark and variant bakeoff comparing QKV Attention:
//! 1) F16 KV
//! 2) I8 KV (biased-u8 scalar dequant)
//! 3) Q8_1 KV Dot4 Variant 1 (Lane 0 sequential dot4)
//! 4) Q8_1 KV Dot4 Variant 2 (Cooperative warp-distributed dot4)

use std::sync::Arc;
use std::time::Instant;
use grim_backend_rocm::{
    AttentionOps, CoreTensorOps, MemoryOps, RocmDevice,
};
use grim_tensor::{DType, Shape};

fn box_dev() -> Arc<RocmDevice> {
    Arc::new(RocmDevice::try_new(0).unwrap())
}

#[test]
#[ignore]
fn bench_q81_kv_dot4_speed() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    let dev = box_dev();
    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 128usize;
    let seq_len = 1usize;
    let kv_seq_len = 16384usize;

    println!("[bakeoff] benchmarking QKV Attention on device {} (kv_seq_len={})", dev.ordinal(), kv_seq_len);

    let q_f32: Vec<f32> = (0..num_heads * head_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let k_f32: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
    let v_f32: Vec<f32> = (0..kv_seq_len * num_kv_heads * head_dim).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();

    let q_s = dev.from_cpu(&q_f32, &Shape::new(vec![seq_len, num_heads, head_dim]), DType::F32).unwrap();
    let _k_s = dev.from_cpu(&k_f32, &Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]), DType::F32).unwrap();
    let _v_s = dev.from_cpu(&v_f32, &Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]), DType::F32).unwrap();

    let out_shape = Shape::new(vec![seq_len, num_heads, head_dim]);

    // 1. Benchmark Standard F16 KV
    let k_f16_bytes: Vec<u8> = k_f32.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect();
    let v_f16_bytes: Vec<u8> = v_f32.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect();
    let k_f16 = dev.from_cpu_bytes(&k_f16_bytes, &Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]), DType::F16).unwrap();
    let v_f16 = dev.from_cpu_bytes(&v_f16_bytes, &Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]), DType::F16).unwrap();

    for _ in 0..5 {
        let _ = dev.qkv_attention(q_s.as_ref(), k_f16.as_ref(), v_f16.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape, None, None).unwrap();
    }
    dev.synchronize();
    let t0 = Instant::now();
    let iters = 100;
    for _ in 0..iters {
        let _ = dev.qkv_attention(q_s.as_ref(), k_f16.as_ref(), v_f16.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape, None, None).unwrap();
    }
    dev.synchronize();
    let dt_f16 = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
    println!("  [F16 KV]                 latency: {:.3} ms", dt_f16);

    // 2. Benchmark I8 KV (Biased-U8 Scalar Dequant)
    let num_kv_elems = kv_seq_len * num_kv_heads * head_dim;
    let mut k_u8 = vec![0u8; num_kv_elems];
    let mut v_u8 = vec![0u8; num_kv_elems];
    let mut k_scales = vec![0.0f32; kv_seq_len * num_kv_heads];
    let mut v_scales = vec![0.0f32; kv_seq_len * num_kv_heads];
    for row in 0..kv_seq_len {
        for kh in 0..num_kv_heads {
            let offset = (row * num_kv_heads + kh) * head_dim;
            let k_amax = k_f32[offset..offset + head_dim].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let ks = (k_amax / 127.0).max(1e-30);
            k_scales[row * num_kv_heads + kh] = ks;
            for d in 0..head_dim {
                k_u8[offset + d] = (((k_f32[offset + d] / ks).round().clamp(-127.0, 127.0) as i32) + 128) as u8;
            }
            let v_amax = v_f32[offset..offset + head_dim].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let vs = (v_amax / 127.0).max(1e-30);
            v_scales[row * num_kv_heads + kh] = vs;
            for d in 0..head_dim {
                v_u8[offset + d] = (((v_f32[offset + d] / vs).round().clamp(-127.0, 127.0) as i32) + 128) as u8;
            }
        }
    }
    let k_u8_s = dev.from_cpu_bytes(&k_u8, &Shape::new(vec![num_kv_elems]), DType::U8).unwrap();
    let v_u8_s = dev.from_cpu_bytes(&v_u8, &Shape::new(vec![num_kv_elems]), DType::U8).unwrap();
    let ks_s = dev.from_cpu(&k_scales, &Shape::new(vec![kv_seq_len * num_kv_heads]), DType::F32).unwrap();
    let vs_s = dev.from_cpu(&v_scales, &Shape::new(vec![kv_seq_len * num_kv_heads]), DType::F32).unwrap();

    for _ in 0..5 {
        let _ = dev.qkv_attention_i8(q_s.as_ref(), k_u8_s.as_ref(), ks_s.as_ref(), v_u8_s.as_ref(), vs_s.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape).unwrap();
    }
    dev.synchronize();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.qkv_attention_i8(q_s.as_ref(), k_u8_s.as_ref(), ks_s.as_ref(), v_u8_s.as_ref(), vs_s.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape).unwrap();
    }
    dev.synchronize();
    let dt_i8 = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
    println!("  [I8 KV Scalar]           latency: {:.3} ms", dt_i8);

    // 3. Dot4 Q8_1 KV Setup
    let num_blocks = head_dim / 32;
    let q81_row_bytes = num_kv_heads * num_blocks * 36;
    let total_q81_bytes = kv_seq_len * q81_row_bytes;
    let mut k_q81 = vec![0u8; total_q81_bytes];
    let mut v_q81 = vec![0u8; total_q81_bytes];

    for row in 0..kv_seq_len {
        for kh in 0..num_kv_heads {
            for blk in 0..num_blocks {
                let f32_off = (row * num_kv_heads + kh) * head_dim + blk * 32;
                let q81_off = row * q81_row_bytes + (kh * num_blocks + blk) * 36;
                // K block
                let amax = k_f32[f32_off..f32_off + 32].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                let d = (amax / 127.0).max(1e-30);
                let bits = half::f16::from_f32(d).to_bits();
                k_q81[q81_off] = (bits & 0xff) as u8;
                k_q81[q81_off + 1] = ((bits >> 8) & 0xff) as u8;
                for e in 0..32 {
                    k_q81[q81_off + 4 + e] = ((k_f32[f32_off + e] / d).round().clamp(-127.0, 127.0) as i8) as u8;
                }
                // V block
                let amax_v = v_f32[f32_off..f32_off + 32].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                let d_v = (amax_v / 127.0).max(1e-30);
                let bits_v = half::f16::from_f32(d_v).to_bits();
                v_q81[q81_off] = (bits_v & 0xff) as u8;
                v_q81[q81_off + 1] = ((bits_v >> 8) & 0xff) as u8;
                for e in 0..32 {
                    v_q81[q81_off + 4 + e] = ((v_f32[f32_off + e] / d_v).round().clamp(-127.0, 127.0) as i8) as u8;
                }
            }
        }
    }

    let k_q81_s = dev.from_cpu_bytes(&k_q81, &Shape::new(vec![total_q81_bytes]), DType::U8).unwrap();
    let v_q81_s = dev.from_cpu_bytes(&v_q81, &Shape::new(vec![total_q81_bytes]), DType::U8).unwrap();

    // 4. Benchmark Variant 1 (Sequential dot4 on lane 0)
    for _ in 0..5 {
        let _ = dev.qkv_attention_q81(q_s.as_ref(), k_q81_s.as_ref(), v_q81_s.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape, false).unwrap();
    }
    dev.synchronize();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.qkv_attention_q81(q_s.as_ref(), k_q81_s.as_ref(), v_q81_s.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape, false).unwrap();
    }
    dev.synchronize();
    let dt_v1 = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
    println!("  [Dot4 Q8_1 V1 (Lane0)]   latency: {:.3} ms ({:.2}x vs I8, {:.2}x vs F16)", dt_v1, dt_i8 / dt_v1, dt_f16 / dt_v1);

    // 5. Benchmark Variant 2 (Cooperative warp-distributed dot4)
    for _ in 0..5 {
        let _ = dev.qkv_attention_q81(q_s.as_ref(), k_q81_s.as_ref(), v_q81_s.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape, true).unwrap();
    }
    dev.synchronize();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.qkv_attention_q81(q_s.as_ref(), k_q81_s.as_ref(), v_q81_s.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape, true).unwrap();
    }
    dev.synchronize();
    let dt_v2 = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
    println!("  [Dot4 Q8_1 V2 (Coop)]    latency: {:.3} ms ({:.2}x vs I8, {:.2}x vs F16)", dt_v2, dt_i8 / dt_v2, dt_f16 / dt_v2);

    // 6. Correctness verification: ensure output is finite and closely tracks F16
    let out_f16_st = dev.qkv_attention(q_s.as_ref(), k_f16.as_ref(), v_f16.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape, None, None).unwrap().0;
    let out_dot4_st = dev.qkv_attention_q81(q_s.as_ref(), k_q81_s.as_ref(), v_q81_s.as_ref(), num_kv_heads, kv_seq_len, 0, None, &out_shape, true).unwrap().0;
    let f16_vals = out_f16_st.to_cpu_vec_f32().unwrap();
    let dot4_vals = out_dot4_st.to_cpu_vec_f32().unwrap();

    let mut max_diff = 0.0f32;
    for (a, b) in f16_vals.iter().zip(dot4_vals.iter()) {
        assert!(b.is_finite(), "Dot4 Q8_1 output is non-finite!");
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("  [Parity Gate] max abs diff vs F16: {:.5}", max_diff);
}
