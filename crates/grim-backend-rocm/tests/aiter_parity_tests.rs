//! Integration tests for AMD ROCm/aiter operator parity features:
//! `silu_mul_quantize` and `sage_attention`.

use grim_backend_cpu::CpuDevice;
use grim_backend_rocm::RocmDevice;
use grim_tensor::dtype::{DType, QuantFormat};
use grim_tensor::{BackendDevice, BackendStorage, Shape};

#[test]
fn test_silu_mul_quantize_parity() {
    let cpu_dev = CpuDevice::new();
    let rocm_dev = RocmDevice::new(0);

    let shape = Shape::new(vec![1, 16]);
    let gate_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.2).collect();
    let up_data: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let g_cpu = cpu_dev.from_cpu(&gate_data, &shape, DType::F32).unwrap();
    let u_cpu = cpu_dev.from_cpu(&up_data, &shape, DType::F32).unwrap();

    let (q_bytes_cpu, s_cpu, h_cpu) = cpu_dev
        .silu_mul_quantize(g_cpu.as_ref(), u_cpu.as_ref(), QuantFormat::Q8_0, &shape)
        .unwrap();
    h_cpu.synchronize().unwrap();

    let scale_cpu_val = s_cpu.to_cpu_vec_f32().unwrap()[0];
    // Downcast CPU storage to get raw bytes for comparison
    let q_cpu_storage = q_bytes_cpu
        .as_any()
        .downcast_ref::<grim_backend_cpu::CpuStorage>()
        .unwrap();
    let q_cpu_elem_count = q_cpu_storage.shape().elem_count();

    let g_rocm = rocm_dev.from_cpu(&gate_data, &shape, DType::F32).unwrap();
    let u_rocm = rocm_dev.from_cpu(&up_data, &shape, DType::F32).unwrap();

    let (q_bytes_rocm, s_rocm, h_rocm) = rocm_dev
        .silu_mul_quantize_gpu(g_rocm.as_ref(), u_rocm.as_ref(), QuantFormat::Q8_0, &shape)
        .unwrap();
    h_rocm.synchronize().unwrap();

    let scale_rocm_val = s_rocm.to_cpu_vec_f32().unwrap()[0];
    // Copy ROCm storage to CPU for comparison
    let q_rocm_vec = q_bytes_rocm.to_cpu_vec_f32().unwrap();
    let q_rocm_elem_count = q_bytes_rocm.shape().elem_count();

    eprintln!(
        "CPU: q_elem_count={}, scale={}",
        q_cpu_elem_count, scale_cpu_val
    );
    eprintln!(
        "ROCm: q_elem_count={}, scale={}",
        q_rocm_elem_count, scale_rocm_val
    );

    assert!((scale_cpu_val - scale_rocm_val).abs() < 1e-4);
    assert_eq!(q_cpu_elem_count, q_rocm_elem_count);
    // Compare quantized values (both as f32, then cast to i32 for tolerance check)
    assert!(
        q_cpu_storage
            .data()
            .iter()
            .zip(q_rocm_vec.iter())
            .all(|(c, r)| {
                let ci = (*c as u8) as i32;
                let ri = (*r as f32).round() as i32;
                (ci - ri).abs() <= 1
            })
    );
}

#[test]
fn test_sage_attention_parity() {
    let cpu_dev = CpuDevice::new();
    let rocm_dev = RocmDevice::new(0);

    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = 8;
    let seq_len = 2;
    let kv_seq_len = 2;

    let shape_q = Shape::new(vec![seq_len, num_heads, head_dim]);
    let shape_out = Shape::new(vec![seq_len, num_heads, head_dim]);

    let q_data = vec![0.1f32; seq_len * num_heads * head_dim];
    let k_data = vec![0.1f32; seq_len * num_kv_heads * head_dim];
    let v_data = vec![0.2f32; seq_len * num_kv_heads * head_dim];

    let q_cpu = cpu_dev.from_cpu(&q_data, &shape_q, DType::F32).unwrap();
    let k_cpu = cpu_dev.from_cpu(&k_data, &shape_q, DType::F32).unwrap();
    let v_cpu = cpu_dev.from_cpu(&v_data, &shape_q, DType::F32).unwrap();

    let (out_cpu, h_cpu) = cpu_dev
        .sage_attention(
            q_cpu.as_ref(),
            k_cpu.as_ref(),
            v_cpu.as_ref(),
            num_kv_heads,
            kv_seq_len,
            &shape_out,
        )
        .unwrap();
    h_cpu.synchronize().unwrap();

    let q_rocm = rocm_dev.from_cpu(&q_data, &shape_q, DType::F32).unwrap();
    let k_rocm = rocm_dev.from_cpu(&k_data, &shape_q, DType::F32).unwrap();
    let v_rocm = rocm_dev.from_cpu(&v_data, &shape_q, DType::F32).unwrap();

    let (out_rocm, h_rocm) = rocm_dev
        .sage_attention_gpu(
            q_rocm.as_ref(),
            k_rocm.as_ref(),
            v_rocm.as_ref(),
            num_kv_heads,
            kv_seq_len,
            &shape_out,
        )
        .unwrap();
    h_rocm.synchronize().unwrap();

    let vec_cpu = out_cpu.to_cpu_vec_f32().unwrap();
    let vec_rocm = out_rocm.to_cpu_vec_f32().unwrap();

    assert_eq!(vec_cpu.len(), vec_rocm.len());
    for i in 0..vec_cpu.len() {
        assert!(
            (vec_cpu[i] - vec_rocm[i]).abs() < 1e-4,
            "mismatch at {i}: cpu {} != rocm {}",
            vec_cpu[i],
            vec_rocm[i]
        );
    }
}

#[test]
fn test_fused_allreduce_rms_norm_parity() {
    let rocm_dev = RocmDevice::new(0);

    let shape = Shape::new(vec![2, 4]);
    let local_data = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
    let peer_data = vec![0.5f32, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0];
    let weight_data = vec![1.0f32, 1.0, 1.0, 1.0];
    let eps = 1e-5f32;

    let loc_s = rocm_dev.from_cpu(&local_data, &shape, DType::F32).unwrap();
    let peer_s = rocm_dev.from_cpu(&peer_data, &shape, DType::F32).unwrap();
    let w_s = rocm_dev
        .from_cpu(&weight_data, &Shape::new(vec![4]), DType::F32)
        .unwrap();

    let (res_out, norm_out, handle) = rocm_dev
        .fused_add_rms_norm(loc_s.as_ref(), peer_s.as_ref(), w_s.as_ref(), eps, &shape)
        .unwrap();
    handle.synchronize().unwrap();

    let res_vec = res_out.to_cpu_vec_f32().unwrap();
    let norm_vec = norm_out.to_cpu_vec_f32().unwrap();

    assert_eq!(res_vec, vec![1.5, 2.5, 3.5, 4.5, 0.0, -1.0, -2.0, -3.0]);
    assert_eq!(norm_vec.len(), 8);
}
