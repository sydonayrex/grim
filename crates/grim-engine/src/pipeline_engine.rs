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
