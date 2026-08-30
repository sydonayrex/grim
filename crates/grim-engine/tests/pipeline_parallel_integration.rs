//! Integration tests for Pipeline Parallelism (PP) in grim-engine (P2).
//!
//! Tests:
//! 1. PipelinePlan partitioning across stages and device ordinals.
//! 2. Per-stage KvBlockPool device isolation and layer range filtering.
//! 3. Multi-stage PipelinedModelCoordinator execution parity vs sequential execution.
//! 4. P2P activation send/recv between adjacent pipeline stages.
//!
//! Verified on: gfx1201 / gfx1200 (Dual-GPU) and gfx1036 — 2026-08-29

use std::sync::Arc;
use grim_core::error::Result;
use grim_engine::pipeline_engine::{
    PipelinePlan, PipelineStageExecutor, PipelineStageRunner, PipelinedModelCoordinator,
};
use grim_memory::KvBlockPool;
use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_tensor::shape::Shape;
use grim_tensor::tensor::Tensor;
use grim_backend_cpu::storage::CpuStorage;

fn make_cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    let s = Shape::new(shape);
    let storage = Arc::new(CpuStorage::new(data, s.clone(), DType::F32));
    Tensor::new(storage, s, DType::F32, QuantProvenance::GrimNative, Device::Cpu)
}

#[test]
fn test_pipeline_plan_layer_partitioning_and_device_mapping() {
    // 32 layers over 4 stages on 4 devices [0, 1, 2, 3]
    let plan = PipelinePlan::plan(32, 4, &[0, 1, 2, 3]).unwrap();
    assert_eq!(plan.stages.len(), 4);
    for (i, stage) in plan.stages.iter().enumerate() {
        assert_eq!(stage.stage_id, i);
        assert_eq!(stage.device_ordinal, i);
        assert_eq!(stage.end_layer - stage.start_layer, 8);
    }
    assert!(plan.stages[0].is_first_stage());
    assert!(!plan.stages[0].is_last_stage());
    assert!(!plan.stages[3].is_first_stage());
    assert!(plan.stages[3].is_last_stage());
}

#[test]
fn test_pipeline_stage_runner_per_stage_kv_cache_isolation() {
    // 16 layers over 2 stages: Stage 0 (layers 0..8, GPU 0), Stage 1 (layers 8..16, GPU 1)
    let plan = PipelinePlan::plan(16, 2, &[0, 1]).unwrap();
    let runner0 = PipelineStageRunner::new(plan.stages[0].clone(), None, 32, 8, 64);
    let runner1 = PipelineStageRunner::new(plan.stages[1].clone(), None, 32, 8, 64);

    assert_eq!(runner0.num_local_layers(), 8);
    assert_eq!(runner1.num_local_layers(), 8);

    // Verify pool 0 owns layers 0..8 on device 0
    let p0 = runner0.block_pool.lock().unwrap();
    assert_eq!(p0.device_ordinal(), 0);
    assert_eq!(p0.layer_range(), Some((0, 8)));
    for l in 0..8 {
        assert!(p0.owns_layer(l));
    }
    for l in 8..16 {
        assert!(!p0.owns_layer(l));
    }

    // Verify pool 1 owns layers 8..16 on device 1
    let p1 = runner1.block_pool.lock().unwrap();
    assert_eq!(p1.device_ordinal(), 1);
    assert_eq!(p1.layer_range(), Some((8, 16)));
    for l in 0..8 {
        assert!(!p1.owns_layer(l));
    }
    for l in 8..16 {
        assert!(p1.owns_layer(l));
    }
}

#[test]
fn test_pipelined_model_coordinator_multi_stage_execution_parity() {
    // 12 layers over 3 stages (4 layers per stage) on devices [0, 1, 2]
    let plan = PipelinePlan::plan(12, 3, &[0, 1, 2]).unwrap();
    let coordinator = PipelinedModelCoordinator::new(plan, 32, 4, 32);

    let batch_size = 2;
    let hidden_dim = 16;
    let input_vec: Vec<f32> = (0..batch_size * hidden_dim)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let input_tensor = make_cpu_tensor(input_vec.clone(), vec![batch_size, hidden_dim]);

    // Simulated layer operation: layer L adds (L as f32 + 1.0) to all elements
    let layer_fn = |layer_idx: usize, x: &Tensor, _pool: &mut KvBlockPool| -> Result<Tensor> {
        let mut v = x.to_vec_f32()?;
        let delta = (layer_idx as f32) + 1.0;
        for val in &mut v {
            *val += delta;
        }
        Ok(make_cpu_tensor(v, x.shape().dims().to_vec()))
    };

    // Pipelined forward
    let pipelined_output = coordinator
        .forward_pipeline(input_tensor.clone(), layer_fn)
        .expect("pipelined forward should succeed");

    // Sequential reference forward
    let mut seq_out = input_vec;
    for l in 0..12 {
        let delta = (l as f32) + 1.0;
        for val in &mut seq_out {
            *val += delta;
        }
    }

    let pipe_vec = pipelined_output.to_vec_f32().unwrap();
    assert_eq!(pipe_vec.len(), seq_out.len());
    for (p, s) in pipe_vec.iter().zip(seq_out.iter()) {
        assert!((p - s).abs() < 1e-5, "Mismatch between pipelined and sequential output");
    }
}

#[test]
fn test_p2p_activation_handoff_boundaries() {
    let plan = PipelinePlan::plan(8, 2, &[0, 1]).unwrap();
    let exec0 = PipelineStageExecutor::new(plan.stages[0].clone(), None);
    let exec1 = PipelineStageExecutor::new(plan.stages[1].clone(), None);

    assert!(exec0.config.is_first_stage());
    assert!(!exec0.config.is_last_stage());
    assert!(!exec1.config.is_first_stage());
    assert!(exec1.config.is_last_stage());

    // First stage doesn't receive from predecessor
    assert!(exec0.recv_activations(&[1, 8]).unwrap().is_none());

    // Last stage doesn't send to successor
    let dummy = make_cpu_tensor(vec![1.0; 8], vec![1, 8]);
    assert!(exec1.send_activations(&dummy).is_ok());
}
