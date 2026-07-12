use std::sync::Arc;
use grim_backend_rocm::{
    RocmDevice, BlockTableEntry, launch_paged_attention,
};
use grim_tensor::{Shape, DType, BackendDevice, BackendStorage};

#[test]
fn test_paged_attention_gpu_matches_reference() {
    // Only run if the GPU tests gate is open
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        return;
    }

    let dev = Arc::new(RocmDevice::new(0));

    let batch = 1u32;
    let num_heads = 2u32;
    let num_kv_heads = 1u32;
    let head_dim = 8u32;
    let page_size = 4u32;
    let kv_seq_len = 8u32;
    let max_blocks = 2u32;
    let cache_offset = 7u32;

    // Define shapes
    let q_shape = Shape::new(vec![batch as usize, num_heads as usize, head_dim as usize]);
    let page_shape = Shape::new(vec![
        2usize, // num_pages
        page_size as usize,
        num_kv_heads as usize,
        head_dim as usize,
    ]);
    let out_shape = q_shape.clone();

    // Generate mock Q, K, V data on CPU
    let q_cpu: Vec<f32> = (0..16).map(|x| (x as f32 * 0.1).sin()).collect();
    let k_cpu: Vec<f32> = (0..64).map(|x| (x as f32 * 0.15).cos()).collect();
    let v_cpu: Vec<f32> = (0..64).map(|x| (x as f32 * 0.2).sin()).collect();

    // Upload to GPU
    let q_storage = dev.from_cpu(&q_cpu, &q_shape, DType::F32).unwrap();
    let k_storage = dev.from_cpu(&k_cpu, &page_shape, DType::F32).unwrap();
    let v_storage = dev.from_cpu(&v_cpu, &page_shape, DType::F32).unwrap();
    let mut out_storage = dev.zeros(&out_shape, DType::F32).unwrap();

    // Create block table
    let table_entries = vec![
        BlockTableEntry { block_id: 0, page_size: 4 },
        BlockTableEntry { block_id: 1, page_size: 4 },
    ];
    // Cast the struct slice to an f32 slice to copy via RocmDevice::from_cpu
    let table_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(
            table_entries.as_ptr() as *const f32,
            table_entries.len() * 2,
        )
    };
    let table_storage = dev.from_cpu(
        table_f32,
        &Shape::new(vec![batch as usize, max_blocks as usize, 2]),
        DType::F32,
    ).unwrap();

    // Launch
    let res = launch_paged_attention(
        &dev,
        q_storage.as_ref(),
        table_storage.as_ref(),
        k_storage.as_ref(),
        v_storage.as_ref(),
        out_storage.as_mut(),
        batch,
        num_heads,
        num_kv_heads,
        head_dim,
        max_blocks,
        page_size,
        kv_seq_len,
        cache_offset,
    );

    assert!(res.is_ok(), "launch_paged_attention failed: {:?}", res.err());

    // Read back output
    let out_cpu = out_storage.to_cpu_vec_f32().unwrap();
    assert_eq!(out_cpu.len(), 16);

    // Reconstruct flat K and V from pages
    let mut k_flat = vec![0.0f32; (kv_seq_len * num_kv_heads * head_dim) as usize];
    let mut v_flat = vec![0.0f32; (kv_seq_len * num_kv_heads * head_dim) as usize];
    
    for b in 0..max_blocks {
        let entry = table_entries[b as usize];
        for t in 0..entry.page_size {
            let j = (b * page_size + t) as usize;
            if j >= kv_seq_len as usize { break; }
            let physical_token = entry.block_id * page_size + t;
            
            for h_kv in 0..num_kv_heads as usize {
                for d in 0..head_dim as usize {
                    let page_offset = ((physical_token as usize * num_kv_heads as usize + h_kv) * head_dim as usize) + d;
                    let flat_offset = ((j * num_kv_heads as usize + h_kv) * head_dim as usize) + d;
                    k_flat[flat_offset] = k_cpu[page_offset];
                    v_flat[flat_offset] = v_cpu[page_offset];
                }
            }
        }
    }

    // CPU reference attention logic
    let mut want_out = vec![0.0f32; 16];
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
    
    for h in 0..num_heads as usize {
        let kv_head = h / (num_heads as usize / num_kv_heads as usize);
        
        let mut scores = vec![0.0f32; kv_seq_len as usize];
        let mut max_score = -1e30f32;
        
        for j in 0..kv_seq_len as usize {
            let mut dot = 0.0f32;
            for d in 0..head_dim as usize {
                let q_val = q_cpu[h * head_dim as usize + d];
                let k_val = k_flat[(j * num_kv_heads as usize + kv_head) * head_dim as usize + d];
                dot += q_val * k_val;
            }
            let score = dot * inv_sqrt_d;
            scores[j] = score;
            if score > max_score {
                max_score = score;
            }
        }
        
        let mut sum_exp = 0.0f32;
        let mut exp_scores = vec![0.0f32; kv_seq_len as usize];
        for j in 0..kv_seq_len as usize {
            let val = (scores[j] - max_score).exp();
            exp_scores[j] = val;
            sum_exp += val;
        }
        
        for d in 0..head_dim as usize {
            let mut val_acc = 0.0f32;
            for j in 0..kv_seq_len as usize {
                let v_val = v_flat[(j * num_kv_heads as usize + kv_head) * head_dim as usize + d];
                val_acc += exp_scores[j] * v_val;
            }
            want_out[h * head_dim as usize + d] = val_acc / sum_exp;
        }
    }

    // Assert relative error is within tolerance
    for i in 0..out_cpu.len() {
        let got = out_cpu[i];
        let want = want_out[i];
        let diff = (got - want).abs();
        let denom = got.abs().max(want.abs()).max(1e-6);
        assert!(
            diff / denom < 1e-3,
            "Divergence at index {}: got {}, want {} (rel diff: {})",
            i, got, want, diff / denom
        );
    }
}
