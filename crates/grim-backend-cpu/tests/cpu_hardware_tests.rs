use grim_backend_cpu::{CpuDevice, moe_fused_dispatch};

#[test]
fn cpu_device_returns_hardware_spec_and_cache_key() {
    let dev = CpuDevice::new();
    let spec = dev.hardware_spec();
    assert!(!spec.arch.is_empty());
    assert!(spec.logical_cores >= 1);

    let cache_key = dev.cache_key("quantized_matmul", 0x42);
    assert_eq!(cache_key.entry, "quantized_matmul");
    assert!(
        cache_key
            .to_key_string()
            .contains("grim_cpu_quantized_matmul")
    );
}

#[test]
fn cpu_numa_topology_inspection() {
    let dev = CpuDevice::new();
    let topo = dev.topology();
    assert!(topo.numa_nodes >= 1);
    assert!(topo.cores_per_node >= 1);
    assert_eq!(topo.node_for_core(0), 0);
}

#[test]
fn moe_fused_dispatch_integration() {
    let (num_tokens, hidden_dim, inter_dim, num_experts, top_k) = (2, 4, 8, 2, 2);
    let tokens = vec![1.0, 0.5, -0.5, 2.0, 0.0, 1.0, 1.0, -1.0];
    let logits = vec![2.0, 1.0, 0.5, 3.0]; // Token 0 prefers E0, Token 1 prefers E1

    let w_gate = vec![vec![0.1; 32], vec![0.2; 32]];
    let w_up = vec![vec![0.1; 32], vec![0.2; 32]];
    let w_down = vec![vec![0.1; 32], vec![0.2; 32]];

    let out = moe_fused_dispatch(
        &tokens,
        &logits,
        &w_gate,
        &w_up,
        &w_down,
        num_tokens,
        hidden_dim,
        inter_dim,
        num_experts,
        top_k,
    )
    .expect("moe_fused_dispatch");

    assert_eq!(out.len(), num_tokens * hidden_dim);
    for val in &out {
        assert!(val.is_finite());
    }
}

#[test]
fn cpu_device_graph_capture_and_replay() {
    let dev = CpuDevice::new();
    assert!(!dev.is_capturing());

    dev.begin_graph_capture("decode_layer_0").expect("begin");
    assert!(dev.is_capturing());

    dev.record_op(|| Ok(()));

    dev.end_graph_capture("decode_layer_0").expect("end");
    assert!(!dev.is_capturing());

    let replayed = dev.replay_graph("decode_layer_0").expect("replay");
    assert!(replayed);

    let not_found = dev.replay_graph("nonexistent").expect("replay nonexistent");
    assert!(!not_found);
}
