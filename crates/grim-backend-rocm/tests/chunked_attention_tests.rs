//! Integration and Numerical Parity Tests for Advanced Attention Suite:
//! Extend-path chunking, LSE state merging, BatchReorderer, and Preshuffled vector-tiled KV-cache.

use grim_backend_rocm::RocmDevice;
use grim_backend_rocm::device::batch_orchestrator::{BatchReorderer, RequestCategory, SequenceMeta};
use grim_tensor::{BackendDevice, Shape, dtype::DType};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    match panic::catch_unwind(|| {
        RocmDevice::try_new(0).expect("RocmDevice::new should succeed on ROCm")
    }) {
        Ok(d) => Some(d),
        Err(_) => None,
    }
}

fn as_u8_slice<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}

#[test]
fn test_advanced_attention_kernels_compile() {
    let dev = RocmDevice::new(0);
    let src = grim_backend_rocm::kernels::source_asm::compute_kernel_source();
    assert!(src.contains("grim_extend_attention_chunk"));
    assert!(src.contains("grim_merge_attn_states"));
    assert!(src.contains("grim_reshape_and_cache_preshuffled"));
    assert!(src.contains("grim_preshuffled_paged_attention"));
    assert_eq!(dev.wavefront_size(), grim_backend_rocm::WavefrontSize::W32);
}

#[test]
fn test_batch_reordering_logic() {
    let seqs = vec![
        SequenceMeta::new(101, 1, 512),  // Decode (1 token, existing cache)
        SequenceMeta::new(102, 128, 0),  // Prefill (128 tokens, zero cache)
        SequenceMeta::new(103, 32, 256), // Extend (32 tokens, existing prefix cache)
        SequenceMeta::new(104, 1, 1024), // Decode (1 token, existing cache)
    ];

    assert_eq!(seqs[0].category(), RequestCategory::Decode);
    assert_eq!(seqs[1].category(), RequestCategory::Prefill);
    assert_eq!(seqs[2].category(), RequestCategory::Extend);
    assert_eq!(seqs[3].category(), RequestCategory::Decode);

    let plan = BatchReorderer::plan(&seqs);
    assert_eq!(plan.decode_count(), 2);
    assert_eq!(plan.extend_count(), 1);
    assert_eq!(plan.prefill_count(), 1);

    // Contiguous order should be [Decode (0, 3), Extend (2), Prefill (1)]
    assert_eq!(plan.decode_indices, vec![0, 3]);
    assert_eq!(plan.extend_indices, vec![2]);
    assert_eq!(plan.prefill_indices, vec![1]);

    let original_ids: Vec<u64> = seqs.iter().map(|s| s.seq_id).collect();
    let permuted = BatchReorderer::permute(&original_ids, &plan);
    assert_eq!(permuted, vec![101, 104, 103, 102]);

    let restored = BatchReorderer::restore(&permuted, &plan);
    assert_eq!(restored, original_ids);
}

#[test]
fn test_extend_chunk_and_lse_merge_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let num_tokens = 2usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 64usize;
    let total_context_len = 128usize;
    let chunk_size = 64usize; // 2 chunks: [0..64) and [64..128)
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

    let q_data: Vec<f32> = (0..num_tokens * num_heads * head_dim)
        .map(|i| (i as f32 * 0.05).sin())
        .collect();
    let k_data: Vec<f32> = (0..total_context_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.03).cos())
        .collect();
    let v_data: Vec<f32> = (0..total_context_len * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.04).sin())
        .collect();

    // CPU Reference Full Attention across all 128 tokens
    let mut expected_out = vec![0.0f32; num_tokens * num_heads * head_dim];
    let q_per_kv = num_heads / num_kv_heads;

    for t in 0..num_tokens {
        for h in 0..num_heads {
            let kv_h = h / q_per_kv;
            let mut running_max = -1e20f32;
            let mut running_sum = 0.0f32;
            let mut acc = vec![0.0f32; head_dim];

            for j in 0..total_context_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q_data[(t * num_heads + h) * head_dim + d];
                    let k_val = k_data[(j * num_kv_heads + kv_h) * head_dim + d];
                    dot += q_val * k_val;
                }
                let score = dot * inv_sqrt_d;

                let new_max = running_max.max(score);
                let alpha = (running_max - new_max).exp();
                let beta = (score - new_max).exp();

                running_sum = running_sum * alpha + beta;
                running_max = new_max;

                for d in 0..head_dim {
                    let v_val = v_data[(j * num_kv_heads + kv_h) * head_dim + d];
                    acc[d] = acc[d] * alpha + beta * v_val;
                }
            }

            for d in 0..head_dim {
                expected_out[(t * num_heads + h) * head_dim + d] = acc[d] / running_sum;
            }
        }
    }

    let q_shape = Shape::from_slice(&[num_tokens, num_heads, head_dim]);
    let kv_shape = Shape::from_slice(&[total_context_len, num_kv_heads, head_dim]);
    let chunk_out_shape = Shape::from_slice(&[num_tokens, num_heads, head_dim]);
    let lse_shape = Shape::from_slice(&[num_tokens, num_heads]);

    let q_dev = BackendDevice::from_cpu(&dev, &q_data, &q_shape, DType::F32)?;
    let k_dev = BackendDevice::from_cpu(&dev, &k_data, &kv_shape, DType::F32)?;
    let v_dev = BackendDevice::from_cpu(&dev, &v_data, &kv_shape, DType::F32)?;

    let out_chunk1 = BackendDevice::zeros(&dev, &chunk_out_shape, DType::F32)?;
    let lse_chunk1 = BackendDevice::zeros(&dev, &lse_shape, DType::F32)?;
    let out_chunk2 = BackendDevice::zeros(&dev, &chunk_out_shape, DType::F32)?;
    let lse_chunk2 = BackendDevice::zeros(&dev, &lse_shape, DType::F32)?;
    let out_merged = BackendDevice::zeros(&dev, &chunk_out_shape, DType::F32)?;
    let lse_merged = BackendDevice::zeros(&dev, &lse_shape, DType::F32)?;

    let q_s = grim_backend_rocm::device::util::as_rocm(q_dev.as_ref())?;
    let k_s = grim_backend_rocm::device::util::as_rocm(k_dev.as_ref())?;
    let v_s = grim_backend_rocm::device::util::as_rocm(v_dev.as_ref())?;

    let o1_s = grim_backend_rocm::device::util::as_rocm(out_chunk1.as_ref())?;
    let l1_s = grim_backend_rocm::device::util::as_rocm(lse_chunk1.as_ref())?;
    let o2_s = grim_backend_rocm::device::util::as_rocm(out_chunk2.as_ref())?;
    let l2_s = grim_backend_rocm::device::util::as_rocm(lse_chunk2.as_ref())?;
    let om_s = grim_backend_rocm::device::util::as_rocm(out_merged.as_ref())?;
    let lm_s = grim_backend_rocm::device::util::as_rocm(lse_merged.as_ref())?;

    // Step 1: Chunk 1 [0..64)
    dev.launch_extend_attention_chunk(
        q_s, k_s, v_s, o1_s, l1_s, num_tokens, num_heads, num_kv_heads, head_dim, 0, chunk_size,
    )?;
    // Step 2: Chunk 2 [64..128)
    dev.launch_extend_attention_chunk(
        q_s,
        k_s,
        v_s,
        o2_s,
        l2_s,
        num_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        chunk_size,
        total_context_len,
    )?;
    // Step 3: LSE Merge
    dev.launch_merge_attn_states(
        o1_s, l1_s, o2_s, l2_s, om_s, lm_s, num_tokens, num_heads, head_dim,
    )?;
    dev.synchronize();

    let actual_out = out_merged.to_cpu_vec_f32()?;
    assert_eq!(actual_out.len(), expected_out.len());

    for (i, (&act, &exp)) in actual_out.iter().zip(expected_out.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-4,
            "Extend Chunk + LSE Merge mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }

    Ok(())
}

#[test]
fn test_preshuffled_paged_attention_parity() -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };

    let num_seqs = 2usize;
    let num_heads = 4usize;
    let head_dim = 64usize;
    let block_size = 16usize;
    let max_blocks = 4usize;
    let context_len = 32usize; // 2 blocks per sequence
    let total_blocks = 8usize;

    let q_data: Vec<f32> = (0..num_seqs * num_heads * head_dim)
        .map(|i| (i as f32 * 0.05).sin())
        .collect();
    let k_tokens: Vec<f32> = (0..num_seqs * context_len * num_heads * head_dim)
        .map(|i| (i as f32 * 0.03).cos())
        .collect();
    let v_tokens: Vec<f32> = (0..num_seqs * context_len * num_heads * head_dim)
        .map(|i| (i as f32 * 0.04).sin())
        .collect();

    // CPU Reference Attention
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
    let mut expected_out = vec![0.0f32; num_seqs * num_heads * head_dim];

    for s in 0..num_seqs {
        for h in 0..num_heads {
            let mut running_max = -1e20f32;
            let mut running_sum = 0.0f32;
            let mut acc = vec![0.0f32; head_dim];

            for j in 0..context_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q_data[(s * num_heads + h) * head_dim + d];
                    let k_val = k_tokens[((s * context_len + j) * num_heads + h) * head_dim + d];
                    dot += q_val * k_val;
                }
                let score = dot * inv_sqrt_d;

                let new_max = running_max.max(score);
                let alpha = (running_max - new_max).exp();
                let beta = (score - new_max).exp();

                running_sum = running_sum * alpha + beta;
                running_max = new_max;

                for d in 0..head_dim {
                    let v_val = v_tokens[((s * context_len + j) * num_heads + h) * head_dim + d];
                    acc[d] = acc[d] * alpha + beta * v_val;
                }
            }

            for d in 0..head_dim {
                expected_out[(s * num_heads + h) * head_dim + d] = acc[d] / running_sum;
            }
        }
    }

    // Allocate Preshuffled Cache buffers
    let k_cache_shape = Shape::from_slice(&[total_blocks, num_heads, head_dim / 4, block_size, 4]);
    let v_cache_shape = Shape::from_slice(&[total_blocks, num_heads, block_size / 4, head_dim, 4]);
    let q_shape = Shape::from_slice(&[num_seqs, num_heads, head_dim]);
    let tokens_shape = Shape::from_slice(&[num_seqs * context_len, num_heads, head_dim]);

    let slot_mapping: Vec<i32> = (0..(num_seqs * context_len) as i32).collect();
    let slot_shape = Shape::from_slice(&[num_seqs * context_len]);

    let mut block_tables = vec![0i32; num_seqs * max_blocks];
    block_tables[0] = 0;
    block_tables[1] = 1;
    block_tables[max_blocks] = 2;
    block_tables[max_blocks + 1] = 3;
    let bt_shape = Shape::from_slice(&[num_seqs, max_blocks]);

    let context_lens = vec![context_len as i32, context_len as i32];
    let cl_shape = Shape::from_slice(&[num_seqs]);

    let q_dev = BackendDevice::from_cpu(&dev, &q_data, &q_shape, DType::F32)?;
    let k_dev = BackendDevice::from_cpu(&dev, &k_tokens, &tokens_shape, DType::F32)?;
    let v_dev = BackendDevice::from_cpu(&dev, &v_tokens, &tokens_shape, DType::F32)?;
    let kc_dev = BackendDevice::zeros(&dev, &k_cache_shape, DType::F32)?;
    let vc_dev = BackendDevice::zeros(&dev, &v_cache_shape, DType::F32)?;

    let sm_dev = BackendDevice::from_cpu_bytes(
        &dev,
        as_u8_slice(&slot_mapping),
        &slot_shape,
        DType::U32,
    )?;
    let bt_dev = BackendDevice::from_cpu_bytes(
        &dev,
        as_u8_slice(&block_tables),
        &bt_shape,
        DType::U32,
    )?;
    let cl_dev = BackendDevice::from_cpu_bytes(
        &dev,
        as_u8_slice(&context_lens),
        &cl_shape,
        DType::U32,
    )?;
    let out_dev = BackendDevice::zeros(&dev, &q_shape, DType::F32)?;

    let q_s = grim_backend_rocm::device::util::as_rocm(q_dev.as_ref())?;
    let k_s = grim_backend_rocm::device::util::as_rocm(k_dev.as_ref())?;
    let v_s = grim_backend_rocm::device::util::as_rocm(v_dev.as_ref())?;
    let kc_s = grim_backend_rocm::device::util::as_rocm(kc_dev.as_ref())?;
    let vc_s = grim_backend_rocm::device::util::as_rocm(vc_dev.as_ref())?;
    let sm_s = grim_backend_rocm::device::util::as_rocm(sm_dev.as_ref())?;
    let bt_s = grim_backend_rocm::device::util::as_rocm(bt_dev.as_ref())?;
    let cl_s = grim_backend_rocm::device::util::as_rocm(cl_dev.as_ref())?;
    let out_s = grim_backend_rocm::device::util::as_rocm(out_dev.as_ref())?;

    // Step 1: Reshape & Cache into preshuffled layout
    dev.launch_reshape_and_cache_preshuffled(
        k_s,
        v_s,
        kc_s,
        vc_s,
        sm_s,
        num_seqs * context_len,
        num_heads,
        head_dim,
        block_size,
    )?;

    // Step 2: Preshuffled Paged Attention Decode
    dev.launch_preshuffled_paged_attention(
        q_s,
        kc_s,
        vc_s,
        bt_s,
        cl_s,
        out_s,
        num_seqs,
        num_heads,
        head_dim,
        block_size,
        max_blocks,
    )?;
    dev.synchronize();

    let actual_out = out_dev.to_cpu_vec_f32()?;
    assert_eq!(actual_out.len(), expected_out.len());

    for (i, (&act, &exp)) in actual_out.iter().zip(expected_out.iter()).enumerate() {
        let err = (act - exp).abs();
        assert!(
            err < 1e-4,
            "Preshuffled Paged Attention mismatch at index {i}: actual={act}, expected={exp}, err={err}"
        );
    }

    Ok(())
}
