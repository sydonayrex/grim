use std::sync::Arc;
use grim_backend_rocm::{
    RocmDevice, launch_tree_attention,
};
use grim_tensor::{Shape, DType, BackendDevice, BackendStorage};

#[test]
fn test_tree_attention_gpu_matches_reference() {
    // Only run if the GPU tests gate is open
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() {
        return;
    }

    let dev = Arc::new(RocmDevice::new(0));

    let batch = 1u32;
    let num_heads = 2u32;
    let num_kv_heads = 1u32;
    let head_dim = 8u32;
    let gamma = 3u32; // tree has 1 + gamma = 4 nodes
    let kv_seq_len = 12u32; // 8 past tokens + 4 speculative tokens
    let cache_offset = 8u32;

    // Define shapes
    let q_shape = Shape::new(vec![batch as usize, (1 + gamma) as usize, num_heads as usize, head_dim as usize]);
    let kv_shape = Shape::new(vec![batch as usize, kv_seq_len as usize, num_kv_heads as usize, head_dim as usize]);
    let parents_shape = Shape::new(vec![(1 + gamma) as usize]);
    let out_shape = q_shape.clone();

    // Generate mock Q, K, V data on CPU
    let q_cpu: Vec<f32> = (0..q_shape.elem_count()).map(|x| (x as f32 * 0.1).sin()).collect();
    let k_cpu: Vec<f32> = (0..kv_shape.elem_count()).map(|x| (x as f32 * 0.15).cos()).collect();
    let v_cpu: Vec<f32> = (0..kv_shape.elem_count()).map(|x| (x as f32 * 0.2).sin()).collect();

    // Tree parents: 
    // Node 0 is root (parent 0)
    // Node 1 parent is 0
    // Node 2 parent is 1
    // Node 3 parent is 0 (branching!)
    // So:
    // path for 0: [0]
    // path for 1: [0, 1]
    // path for 2: [0, 1, 2]
    // path for 3: [0, 3]
    let parents_cpu: Vec<u32> = vec![0, 0, 1, 0];

    // Upload to GPU
    let q_storage = dev.from_cpu(&q_cpu, &q_shape, DType::F32).unwrap();
    let k_storage = dev.from_cpu(&k_cpu, &kv_shape, DType::F32).unwrap();
    let v_storage = dev.from_cpu(&v_cpu, &kv_shape, DType::F32).unwrap();
    let mut out_storage = dev.zeros(&out_shape, DType::F32).unwrap();

    // We can upload parents by casting u32 to f32 raw bits
    let parents_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(
            parents_cpu.as_ptr() as *const f32,
            parents_cpu.len(),
        )
    };
    let parents_storage = dev.from_cpu(parents_f32, &parents_shape, DType::F32).unwrap();

    // Launch
    let res = launch_tree_attention(
        &dev,
        q_storage.as_ref(),
        k_storage.as_ref(),
        v_storage.as_ref(),
        parents_storage.as_ref(),
        out_storage.as_mut(),
        batch,
        num_heads,
        num_kv_heads,
        head_dim,
        gamma,
        kv_seq_len,
        cache_offset,
    );

    assert!(res.is_ok(), "launch_tree_attention failed: {:?}", res.err());

    // Read back output
    let out_cpu = out_storage.to_cpu_vec_f32().unwrap();

    // Reconstruct path verification helper
    let is_ancestor = |j_tree: usize, i_tree: usize| -> bool {
        if j_tree == i_tree { return true; }
        let mut curr = i_tree;
        while curr > 0 {
            curr = parents_cpu[curr] as usize;
            if curr == j_tree { return true; }
        }
        false
    };

    // CPU reference tree attention logic
    let mut want_out = vec![0.0f32; out_shape.elem_count()];
    let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
    
    for b_idx in 0..batch as usize {
        for i_tree in 0..(1 + gamma) as usize {
            for h in 0..num_heads as usize {
                let kv_head = h / (num_heads as usize / num_kv_heads as usize);
                
                let mut scores = Vec::new();
                let mut max_score = -1e30f32;
                
                // Collect valid keys/values for this query tree node i_tree
                for j in 0..kv_seq_len as usize {
                    let mut attend = false;
                    if j < cache_offset as usize {
                        attend = true;
                    } else {
                        let j_tree = j - cache_offset as usize;
                        if j_tree <= i_tree && is_ancestor(j_tree, i_tree) {
                            attend = true;
                        }
                    }
                    if !attend { continue; }

                    let mut dot = 0.0f32;
                    for d in 0..head_dim as usize {
                        let q_offset = ((b_idx * (1 + gamma) as usize + i_tree) * num_heads as usize + h) * head_dim as usize + d;
                        let kv_offset = ((b_idx * kv_seq_len as usize + j) * num_kv_heads as usize + kv_head) * head_dim as usize + d;
                        dot += q_cpu[q_offset] * k_cpu[kv_offset];
                    }
                    let score = dot * inv_sqrt_d;
                    scores.push((j, score));
                    if score > max_score {
                        max_score = score;
                    }
                }
                
                let mut sum_exp = 0.0f32;
                let mut exp_scores = Vec::new();
                for &(_, score) in &scores {
                    let val = (score - max_score).exp();
                    exp_scores.push(val);
                    sum_exp += val;
                }
                
                for d in 0..head_dim as usize {
                    let mut val_acc = 0.0f32;
                    for idx in 0..scores.len() {
                        let j = scores[idx].0;
                        let w = exp_scores[idx];
                        let kv_offset = ((b_idx * kv_seq_len as usize + j) * num_kv_heads as usize + kv_head) * head_dim as usize + d;
                        val_acc += w * v_cpu[kv_offset];
                    }
                    let out_offset = ((b_idx * (1 + gamma) as usize + i_tree) * num_heads as usize + h) * head_dim as usize + d;
                    want_out[out_offset] = val_acc / sum_exp;
                }
            }
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
