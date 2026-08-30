//! 1F1B Pipeline Parallel (PP) stage execution engine.
//!
//! Partitions transformer layers across multiple pipeline stages/GPUs and schedules
//! activation transfers between adjacent stages using point-to-point communication.

use std::sync::Arc;
use grim_core::error::{Error, Result};
use grim_tensor::tensor::Tensor;
use grim_tensor::shape::Shape;
use grim_tensor::dtype::{DType, Device, QuantProvenance};
use grim_backend_cpu::storage::CpuStorage;
use grim_backend_rocm::device::parallel_comm::ParallelCommunicator;

fn tensor_from_f32_vec(vec: Vec<f32>, shape: Shape) -> Tensor {
    let storage = Arc::new(CpuStorage::new(vec, shape.clone(), DType::F32));
    Tensor::new(
        storage,
        shape,
        DType::F32,
        QuantProvenance::GrimNative,
        Device::Cpu,
    )
}

/// Pipeline parallelism stage configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStageConfig {
    /// Stage index (0 .. num_stages - 1).
    pub stage_id: usize,
    /// Total number of pipeline stages.
    pub num_stages: usize,
    /// First transformer layer index executed on this stage.
    pub start_layer: usize,
    /// Last transformer layer index (exclusive) executed on this stage.
    pub end_layer: usize,
    /// Hardware GPU ordinal assigned to this stage.
    pub device_ordinal: usize,
}

/// Computed pipeline layout: stage boundaries plus per-stage device
/// placement, ready to drive a partitioned execution. `Engine::new`
/// validates one against the loaded model depth and visible GPUs when
/// `pp_size > 1` (see `EngineConfig::pp_size` for the execution gate).
#[derive(Debug, Clone)]
pub struct PipelinePlan {
    /// One config per stage, in order.
    pub stages: Vec<PipelineStageConfig>,
}

impl PipelinePlan {
    /// Evenly partition `total_layers` across `num_stages` stages pinned to
    /// `device_ordinals[stage]`.
    ///
    /// # Contracts
    /// * `num_stages >= 1` and `total_layers >= num_stages` (a stage with no
    ///   layers is a config error, not a degenerate pass-through).
    /// * `device_ordinals.len() == num_stages`.
    pub fn plan(
        total_layers: usize,
        num_stages: usize,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        if total_layers < num_stages {
            return Err(Error::Config(format!(
                "PipelinePlan: {total_layers} layers cannot fill {num_stages} stages"
            )));
        }
        if device_ordinals.len() != num_stages {
            return Err(Error::Config(format!(
                "PipelinePlan: {} device ordinals for {num_stages} stages",
                device_ordinals.len()
            )));
        }
        Ok(Self {
            stages: PipelineStageConfig::partition_layers(
                total_layers,
                num_stages,
                device_ordinals,
            )?,
        })
    }

    /// Stage index that executes `layer`.
    pub fn stage_for_layer(&self, layer: usize) -> Option<usize> {
        self.stages
            .iter()
            .find(|s| layer >= s.start_layer && layer < s.end_layer)
            .map(|s| s.stage_id)
    }
}

impl PipelineStageConfig {
    /// Partitions `total_layers` evenly across `num_stages`.
    pub fn partition_layers(
        total_layers: usize,
        num_stages: usize,
        device_ordinals: &[usize],
    ) -> Result<Vec<Self>> {
        if num_stages == 0 {
            return Err(Error::Config("PipelineStageConfig: num_stages must be >= 1".into()));
        }
        let layers_per_stage = total_layers / num_stages;
        let remainder = total_layers % num_stages;

        let mut configs = Vec::with_capacity(num_stages);
        let mut curr_layer = 0;

        for s in 0..num_stages {
            let count = layers_per_stage + if s < remainder { 1 } else { 0 };
            let start = curr_layer;
            let end = curr_layer + count;
            curr_layer = end;

            let dev = device_ordinals.get(s).copied().unwrap_or(s);
            configs.push(Self {
                stage_id: s,
                num_stages,
                start_layer: start,
                end_layer: end,
                device_ordinal: dev,
            });
        }
        Ok(configs)
    }

    /// Whether this is the first stage (embeds tokens).
    pub fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }

    /// Whether this is the last stage (computes final logits/loss).
    pub fn is_last_stage(&self) -> bool {
        self.stage_id + 1 == self.num_stages
    }
}

/// Pipeline parallel executor for a single stage.
pub struct PipelineStageExecutor {
    /// Stage configuration.
    pub config: PipelineStageConfig,
    /// Inter-stage point-to-point communicator.
    pub comm: Option<Arc<ParallelCommunicator>>,
}

impl PipelineStageExecutor {
    /// Creates a new pipeline stage executor.
    pub fn new(config: PipelineStageConfig, comm: Option<Arc<ParallelCommunicator>>) -> Self {
        Self { config, comm }
    }

    /// Receives activation tensor from predecessor stage (if not first stage).
    pub fn recv_activations(&self, shape: &[usize]) -> Result<Option<Tensor>> {
        if self.config.is_first_stage() {
            return Ok(None);
        }
        if let Some(comm) = &self.comm {
            let elem_count: usize = shape.iter().product();
            let mut recv_buf = vec![0.0f32; elem_count];
            let prev_stage = self.config.stage_id - 1;
            comm.send_recv_p2p(None, 0, Some(&mut recv_buf), prev_stage)?;
            let t = tensor_from_f32_vec(recv_buf, Shape::from_slice(shape));
            Ok(Some(t))
        } else {
            Ok(None)
        }
    }

    /// Sends activation tensor to successor stage (if not last stage).
    pub fn send_activations(&self, activations: &Tensor) -> Result<()> {
        if self.config.is_last_stage() {
            return Ok(());
        }
        if let Some(comm) = &self.comm {
            let send_buf = activations.to_vec_f32()?;
            let next_stage = self.config.stage_id + 1;
            comm.send_recv_p2p(Some(&send_buf), next_stage, None, 0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_partitioning() {
        let partitions = PipelineStageConfig::partition_layers(32, 4, &[0, 1, 2, 3]).unwrap();
        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0].start_layer, 0);
        assert_eq!(partitions[0].end_layer, 8);
        assert!(partitions[0].is_first_stage());
        assert!(!partitions[0].is_last_stage());

        assert_eq!(partitions[3].start_layer, 24);
        assert_eq!(partitions[3].end_layer, 32);
        assert!(!partitions[3].is_first_stage());
        assert!(partitions[3].is_last_stage());
    }

    #[test]
    fn test_pipeline_plan_validation() {
        // Valid 2-stage plan over 8 layers on 2 devices.
        let plan = PipelinePlan::plan(8, 2, &[0, 1]).unwrap();
        assert_eq!(plan.stage_for_layer(0), Some(0));
        assert_eq!(plan.stage_for_layer(7), Some(1));
        assert_eq!(plan.stage_for_layer(3), Some(0));
        assert_eq!(plan.stage_for_layer(4), Some(1));
        assert_eq!(plan.stage_for_layer(8), None, "out of range layer");

        // Fewer layers than stages is a config error, not a degenerate plan.
        assert!(PipelinePlan::plan(1, 2, &[0, 1]).is_err());
        // Device ordinal count must match stage count.
        assert!(PipelinePlan::plan(8, 2, &[0]).is_err());
        assert!(PipelinePlan::plan(8, 0, &[]).is_err());

        // Stage executors bind the plan's boundary conditions.
        let exec = PipelineStageExecutor::new(plan.stages[1].clone(), None);
        assert!(exec.config.is_last_stage());
        assert!(!exec.config.is_first_stage());
        assert!(
            matches!(exec.recv_activations(&[1, 4]), Ok(None)),
            "missing comm on a non-first stage must surface as None activations,              not a fake tensor"
        );
    }

    #[test]
    fn test_uneven_layer_partitioning() {
        let partitions = PipelineStageConfig::partition_layers(30, 4, &[0, 1, 2, 3]).unwrap();
        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0].start_layer, 0);
        assert_eq!(partitions[0].end_layer, 8); // 8
        assert_eq!(partitions[1].start_layer, 8);
        assert_eq!(partitions[1].end_layer, 16); // 8
        assert_eq!(partitions[2].start_layer, 16);
        assert_eq!(partitions[2].end_layer, 23); // 7
        assert_eq!(partitions[3].start_layer, 23);
        assert_eq!(partitions[3].end_layer, 30); // 7
    }
}
