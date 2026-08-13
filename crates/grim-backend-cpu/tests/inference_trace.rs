//! End-to-end CPU inference trace verification through all CpuDevice primitives,
//! attention kernels, fused normalization, MoE dispatch, and decode graph capture.

use grim_backend_cpu::{CpuDevice, moe_fused_dispatch};
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::{BackendDevice, Shape};

#[test]
fn cpu_full_inference_trace_and_graph_replay() {
    let seq_len = 1;
    let hidden_dim = 16;
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let inter_dim = 32;
    let num_experts = 2;
    let top_k = 1;
    let vocab_size = 64;

    let dev = CpuDevice::new();
    let shape_hidden = Shape::from_slice(&[seq_len, hidden_dim]);
    let shape_weight = Shape::from_slice(&[hidden_dim]);
    let shape_qkv = Shape::from_slice(&[seq_len, num_heads, head_dim]);
    let dtype = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };

    // 1. Embedding lookup
    let token_ids = vec![42u32];
    let emb_table_data: Vec<f32> = (0..(vocab_size * hidden_dim))
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let emb_table = dev
        .from_cpu(&emb_table_data, &Shape::from_slice(&[vocab_size, hidden_dim]), dtype.clone())
        .expect("emb_table");

    let (x_emb, handle_emb) = dev
        .embedding(emb_table.as_ref(), &token_ids, &shape_hidden)
        .expect("embedding");

    handle_emb.synchronize().expect("sync emb");
    let x_val = x_emb.to_cpu_vec_f32().expect("read x_emb");
    assert_eq!(x_val.len(), hidden_dim);

    // 2. Fused Add + RMSNorm
    let res_initial = vec![0.0f32; hidden_dim];
    let res_storage = dev.from_cpu(&res_initial, &shape_hidden, dtype.clone()).expect("res");
    let norm_w_data = vec![1.0f32; hidden_dim];
    let norm_w = dev.from_cpu(&norm_w_data, &shape_weight, dtype.clone()).expect("norm_w");

    let (_y_norm, _updated_res, handle_norm) = dev
        .fused_add_rms_norm(
            x_emb.as_ref(),
            res_storage.as_ref(),
            norm_w.as_ref(),
            1e-5,
            &shape_hidden,
        )
        .expect("fused_add_rms_norm");
    handle_norm.synchronize().expect("sync norm");


    // 3. QKV Attention (Self Attention)
    let q_data = vec![0.1f32; seq_len * hidden_dim];
    let k_data = vec![0.1f32; seq_len * hidden_dim];
    let v_data = vec![0.2f32; seq_len * hidden_dim];

    let q = dev.from_cpu(&q_data, &shape_qkv, dtype.clone()).expect("q");
    let k = dev.from_cpu(&k_data, &shape_qkv, dtype.clone()).expect("k");
    let v = dev.from_cpu(&v_data, &shape_qkv, dtype.clone()).expect("v");

    let shape_attn = Shape::from_slice(&[seq_len, num_heads, head_dim]);
    let (attn_out, handle_attn) = dev
        .qkv_attention(
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            num_kv_heads,
            seq_len,
            0,
            None,
            &shape_attn,
            None,
            None,
        )
        .expect("qkv_attention");

    handle_attn.synchronize().expect("sync attn");
    let attn_val = attn_out.to_cpu_vec_f32().expect("read attn");
    assert_eq!(attn_val.len(), hidden_dim);

    // 4. Fused MoE Grouped GEMM Dispatch
    let gate_logits = vec![5.0f32, 0.0f32]; // Token prefers Expert 0
    let w_gate = vec![vec![0.05f32; hidden_dim * inter_dim]; num_experts];
    let w_up = vec![vec![0.05f32; hidden_dim * inter_dim]; num_experts];
    let w_down = vec![vec![0.05f32; inter_dim * hidden_dim]; num_experts];

    let moe_out = moe_fused_dispatch(
        &attn_val,
        &gate_logits,
        &w_gate,
        &w_up,
        &w_down,
        seq_len,
        hidden_dim,
        inter_dim,
        num_experts,
        top_k,
    )
    .expect("moe_fused_dispatch");
    assert_eq!(moe_out.len(), hidden_dim);

    // 5. Decode Graph Capture and Replay Verification
    dev.begin_graph_capture("cpu_decode_token_1").expect("begin_graph_capture");
    assert!(dev.is_capturing());

    dev.record_op(|| Ok(()));
    dev.record_op(|| Ok(()));

    dev.end_graph_capture("cpu_decode_token_1").expect("end_graph_capture");
    assert!(!dev.is_capturing());

    let replayed = dev.replay_graph("cpu_decode_token_1").expect("replay_graph");
    assert!(replayed, "Decode graph replay should succeed");
}
