//! 1F1B Pipeline Parallel (PP) stage execution engine.
//!
//! Partitions transformer layers across multiple pipeline stages/GPUs and schedules
//! activation transfers between adjacent stages using point-to-point communication.

use std::sync::{Arc, Mutex};
use grim_core::error::{Error, Result};
use grim_memory::KvBlockPool;
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

/// Runtime stage runner that executes assigned transformer layers and manages
/// the per-stage isolated KV cache block pool.
pub struct PipelineStageRunner {
    pub config: PipelineStageConfig,
    pub executor: PipelineStageExecutor,
    pub block_pool: Arc<Mutex<KvBlockPool>>,
}

impl PipelineStageRunner {
    /// Creates a new stage runner pinned to the stage's target device ordinal
    /// and assigned layer range.
    pub fn new(
        config: PipelineStageConfig,
        comm: Option<Arc<ParallelCommunicator>>,
        pool_capacity: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let pool = KvBlockPool::new_on_device(
            pool_capacity,
            num_heads,
            head_dim,
            config.device_ordinal,
        )
        .with_layer_range(config.start_layer, config.end_layer);

        let executor = PipelineStageExecutor::new(config.clone(), comm);
        Self {
            config,
            executor,
            block_pool: Arc::new(Mutex::new(pool)),
        }
    }

    /// Number of transformer layers assigned to this stage.
    pub fn num_local_layers(&self) -> usize {
        self.config.end_layer - self.config.start_layer
    }

    /// Forward pass through the stage's assigned layers.
    pub fn forward_stage<F>(
        &self,
        input_activations: Tensor,
        layer_forward_fn: F,
    ) -> Result<Option<Tensor>>
    where
        F: Fn(usize, &Tensor, &mut KvBlockPool) -> Result<Tensor>,
    {
        let mut h = input_activations;
        {
            let mut pool = self.block_pool.lock().map_err(|e| {
                Error::KvCache(format!("Failed to lock stage KV block pool: {}", e))
            })?;

            for layer_idx in self.config.start_layer..self.config.end_layer {
                h = layer_forward_fn(layer_idx, &h, &mut *pool)?;
            }
        }

        if self.config.is_last_stage() {
            Ok(Some(h))
        } else {
            self.executor.send_activations(&h)?;
            Ok(None)
        }
    }
}

/// Coordinator for multi-stage pipelined execution over microbatches.
pub struct PipelinedModelCoordinator {
    pub plan: PipelinePlan,
    pub runners: Vec<PipelineStageRunner>,
}

impl PipelinedModelCoordinator {
    /// Creates a new pipelined model coordinator.
    pub fn new(
        plan: PipelinePlan,
        pool_capacity: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let runners = plan
            .stages
            .iter()
            .map(|cfg| {
                PipelineStageRunner::new(
                    cfg.clone(),
                    None,
                    pool_capacity,
                    num_heads,
                    head_dim,
                )
            })
            .collect();
        Self { plan, runners }
    }

    /// Execute a forward pass across all pipeline stages in sequence.
    pub fn forward_pipeline<F>(
        &self,
        initial_input: Tensor,
        layer_forward_fn: F,
    ) -> Result<Tensor>
    where
        F: Fn(usize, &Tensor, &mut KvBlockPool) -> Result<Tensor>,
    {
        let mut curr = initial_input;
        for runner in &self.runners {
            let mut h = curr;
            {
                let mut pool = runner.block_pool.lock().map_err(|e| {
                    Error::KvCache(format!("Failed to lock stage KV block pool: {}", e))
                })?;
                for layer_idx in runner.config.start_layer..runner.config.end_layer {
                    h = layer_forward_fn(layer_idx, &h, &mut *pool)?;
                }
            }
            curr = h;
        }
        Ok(curr)
    }
}

// ── Virtual Pipeline Parallelism (VPP) ───────────────────────────────────────

/// Virtual pipeline stage configuration mapping a virtual stage slice
/// onto a physical hardware rank in a V-shaped fold-back topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualStageConfig {
    /// Virtual stage index in forward traversal order (0 .. 2*num_physical_ranks - 1).
    pub virtual_stage_id: usize,
    /// Physical hardware GPU rank assigned to this virtual stage.
    pub physical_rank: usize,
    /// Total number of virtual stages in the model.
    pub num_virtual_stages: usize,
    /// Total number of physical hardware ranks.
    pub num_physical_ranks: usize,
    /// First transformer layer index executed on this virtual stage.
    pub start_layer: usize,
    /// Last transformer layer index (exclusive) executed on this virtual stage.
    pub end_layer: usize,
    /// Hardware GPU ordinal assigned to this stage.
    pub device_ordinal: usize,
}

impl VirtualStageConfig {
    /// Whether this is the entry stage of the entire model.
    pub fn is_model_head(&self) -> bool {
        self.virtual_stage_id == 0
    }

    /// Whether this is the final output stage of the entire model.
    pub fn is_model_tail(&self) -> bool {
        self.virtual_stage_id + 1 == self.num_virtual_stages
    }

    /// Whether this virtual stage is at the fold turning point (stays on same physical rank).
    pub fn is_fold_point(&self) -> bool {
        self.virtual_stage_id + 1 == self.num_physical_ranks
    }
}

/// Computed V-shaped Virtual Pipeline Parallel Plan (VPP).
///
/// Partitions $L$ layers into $2N$ virtual stages $\{s_0, s_1, \dots, s_{2N-1}\}$
/// mapped onto $N$ physical ranks:
/// - Rank 0: $\{s_0, s_{2N-1}\}$ (model head and final tail)
/// - Rank 1: $\{s_1, s_{2N-2}\}$
/// - ...
/// - Rank $N-1$: $\{s_{N-1}, s_N\}$ (middle fold-back stages)
#[derive(Debug, Clone)]
pub struct VirtualPipelinePlan {
    /// Virtual stages in forward traversal order ($0 \dots 2N-1$).
    pub virtual_stages: Vec<VirtualStageConfig>,
    /// Number of physical hardware ranks.
    pub num_physical_ranks: usize,
}

impl VirtualPipelinePlan {
    /// Generate a V-shaped virtual pipeline plan.
    ///
    /// # Contracts
    /// * `num_physical_ranks >= 1`
    /// * `total_layers >= 2 * num_physical_ranks`
    /// * `device_ordinals.len() == num_physical_ranks`
    pub fn plan(
        total_layers: usize,
        num_physical_ranks: usize,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        let num_virtual_stages = 2 * num_physical_ranks;
        if total_layers < num_virtual_stages {
            return Err(Error::Config(format!(
                "VirtualPipelinePlan: {total_layers} layers cannot fill {num_virtual_stages} virtual stages"
            )));
        }
        if device_ordinals.len() != num_physical_ranks {
            return Err(Error::Config(format!(
                "VirtualPipelinePlan: {} device ordinals for {num_physical_ranks} physical ranks",
                device_ordinals.len()
            )));
        }

        let layers_per_vstage = total_layers / num_virtual_stages;
        let remainder = total_layers % num_virtual_stages;

        let mut virtual_stages = Vec::with_capacity(num_virtual_stages);
        let mut curr_layer = 0;

        for vs in 0..num_virtual_stages {
            let count = layers_per_vstage + if vs < remainder { 1 } else { 0 };
            let start = curr_layer;
            let end = curr_layer + count;
            curr_layer = end;

            // Fold-back rank mapping:
            // Forward arm: vs 0..N-1 -> rank vs
            // Return arm: vs N..2N-1 -> rank (2N - 1 - vs)
            let phys_rank = if vs < num_physical_ranks {
                vs
            } else {
                (2 * num_physical_ranks - 1) - vs
            };

            let dev = device_ordinals[phys_rank];
            virtual_stages.push(VirtualStageConfig {
                virtual_stage_id: vs,
                physical_rank: phys_rank,
                num_virtual_stages,
                num_physical_ranks,
                start_layer: start,
                end_layer: end,
                device_ordinal: dev,
            });
        }

        Ok(Self {
            virtual_stages,
            num_physical_ranks,
        })
    }

    /// Retrieve the two virtual stages assigned to a physical rank.
    pub fn stages_for_rank(&self, physical_rank: usize) -> (VirtualStageConfig, VirtualStageConfig) {
        let first = self.virtual_stages[physical_rank].clone();
        let second = self.virtual_stages[2 * self.num_physical_ranks - 1 - physical_rank].clone();
        (first, second)
    }
}

/// Dual-queue chunked prefill coordinator using V-shaped Virtual Pipeline Parallelism.
pub struct VirtualPipelineCoordinator {
    pub plan: VirtualPipelinePlan,
    pub runners: Vec<PipelineStageRunner>,
}

impl VirtualPipelineCoordinator {
    /// Creates a new VPP model coordinator.
    pub fn new(
        plan: VirtualPipelinePlan,
        pool_capacity: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let runners = plan
            .virtual_stages
            .iter()
            .map(|cfg| {
                let stage_cfg = PipelineStageConfig {
                    stage_id: cfg.virtual_stage_id,
                    num_stages: cfg.num_virtual_stages,
                    start_layer: cfg.start_layer,
                    end_layer: cfg.end_layer,
                    device_ordinal: cfg.device_ordinal,
                };
                PipelineStageRunner::new(
                    stage_cfg,
                    None,
                    pool_capacity,
                    num_heads,
                    head_dim,
                )
            })
            .collect();
        Self { plan, runners }
    }

    /// Execute a forward pass across all virtual pipeline stages in V-traversal sequence.
    pub fn forward_vpp<F>(
        &self,
        initial_input: Tensor,
        layer_forward_fn: F,
    ) -> Result<Tensor>
    where
        F: Fn(usize, &Tensor, &mut KvBlockPool) -> Result<Tensor>,
    {
        let mut curr = initial_input;
        for runner in &self.runners {
            let mut h = curr;
            {
                let mut pool = runner.block_pool.lock().map_err(|e| {
                    Error::KvCache(format!("Failed to lock stage KV block pool: {}", e))
                })?;
                for layer_idx in runner.config.start_layer..runner.config.end_layer {
                    h = layer_forward_fn(layer_idx, &h, &mut *pool)?;
                }
            }
            curr = h;
        }
        Ok(curr)
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

    #[test]
    fn test_pipeline_stage_runner_kv_isolation() {
        let plan = PipelinePlan::plan(8, 2, &[0, 1]).unwrap();
        let runner0 = PipelineStageRunner::new(plan.stages[0].clone(), None, 16, 4, 32);
        let runner1 = PipelineStageRunner::new(plan.stages[1].clone(), None, 16, 4, 32);

        assert_eq!(runner0.config.device_ordinal, 0);
        assert_eq!(runner0.num_local_layers(), 4);
        assert_eq!(runner1.config.device_ordinal, 1);
        assert_eq!(runner1.num_local_layers(), 4);

        let p0 = runner0.block_pool.lock().unwrap();
        assert_eq!(p0.device_ordinal(), 0);
        assert!(p0.owns_layer(0));
        assert!(p0.owns_layer(3));
        assert!(!p0.owns_layer(4));

        let p1 = runner1.block_pool.lock().unwrap();
        assert_eq!(p1.device_ordinal(), 1);
        assert!(!p1.owns_layer(3));
        assert!(p1.owns_layer(4));
        assert!(p1.owns_layer(7));
    }

    #[test]
    fn test_pipelined_model_coordinator_forward_parity() {
        let plan = PipelinePlan::plan(4, 2, &[0, 1]).unwrap();
        let coordinator = PipelinedModelCoordinator::new(plan, 16, 2, 16);

        let input_data = vec![1.0f32; 8];
        let input_tensor = tensor_from_f32_vec(input_data, Shape::new(vec![1, 8]));

        // Simulated layer forward: each layer adds (layer_idx + 1) * 0.5 to all activation elements
        let layer_fn = |layer_idx: usize, x: &Tensor, _pool: &mut KvBlockPool| -> Result<Tensor> {
            let mut v = x.to_vec_f32()?;
            let add = (layer_idx as f32 + 1.0) * 0.5;
            for val in &mut v {
                *val += add;
            }
            Ok(tensor_from_f32_vec(v, x.shape().clone()))
        };

        let output = coordinator
            .forward_pipeline(input_tensor, layer_fn)
            .expect("pipelined forward should succeed");

        let out_vec = output.to_vec_f32().unwrap();
        // Layer 0: +0.5, Layer 1: +1.0, Layer 2: +1.5, Layer 3: +2.0 -> total added = 5.0
        // Input was 1.0 -> expected output is 6.0 for each element
        for val in out_vec {
            assert!((val - 6.0f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_virtual_pipeline_plan_fold_back_mapping() {
        // 8 layers partitioned across 2 physical ranks -> 4 virtual stages:
        // Rank 0: s0 (0..2), s3 (6..8)
        // Rank 1: s1 (2..4), s2 (4..6)
        let vplan = VirtualPipelinePlan::plan(8, 2, &[0, 1]).unwrap();
        assert_eq!(vplan.virtual_stages.len(), 4);

        assert_eq!(vplan.virtual_stages[0].virtual_stage_id, 0);
        assert_eq!(vplan.virtual_stages[0].physical_rank, 0);
        assert_eq!(vplan.virtual_stages[0].start_layer, 0);
        assert_eq!(vplan.virtual_stages[0].end_layer, 2);
        assert!(vplan.virtual_stages[0].is_model_head());

        assert_eq!(vplan.virtual_stages[1].virtual_stage_id, 1);
        assert_eq!(vplan.virtual_stages[1].physical_rank, 1);
        assert_eq!(vplan.virtual_stages[1].start_layer, 2);
        assert_eq!(vplan.virtual_stages[1].end_layer, 4);
        assert!(vplan.virtual_stages[1].is_fold_point());

        assert_eq!(vplan.virtual_stages[2].virtual_stage_id, 2);
        assert_eq!(vplan.virtual_stages[2].physical_rank, 1);
        assert_eq!(vplan.virtual_stages[2].start_layer, 4);
        assert_eq!(vplan.virtual_stages[2].end_layer, 6);

        assert_eq!(vplan.virtual_stages[3].virtual_stage_id, 3);
        assert_eq!(vplan.virtual_stages[3].physical_rank, 0);
        assert_eq!(vplan.virtual_stages[3].start_layer, 6);
        assert_eq!(vplan.virtual_stages[3].end_layer, 8);
        assert!(vplan.virtual_stages[3].is_model_tail());

        let (r0_s0, r0_s1) = vplan.stages_for_rank(0);
        assert_eq!(r0_s0.virtual_stage_id, 0);
        assert_eq!(r0_s1.virtual_stage_id, 3);
    }

    #[test]
    fn test_virtual_pipeline_forward_vpp_equivalence() {
        let vplan = VirtualPipelinePlan::plan(4, 2, &[0, 1]).unwrap();
        let coordinator = VirtualPipelineCoordinator::new(vplan, 16, 2, 16);

        let input_data = vec![2.0f32; 8];
        let input_tensor = tensor_from_f32_vec(input_data, Shape::new(vec![1, 8]));

        let layer_fn = |layer_idx: usize, x: &Tensor, _pool: &mut KvBlockPool| -> Result<Tensor> {
            let mut v = x.to_vec_f32()?;
            let add = (layer_idx as f32 + 1.0) * 0.25;
            for val in &mut v {
                *val += add;
            }
            Ok(tensor_from_f32_vec(v, x.shape().clone()))
        };

        let output = coordinator
            .forward_vpp(input_tensor, layer_fn)
            .expect("vpp forward should succeed");

        let out_vec = output.to_vec_f32().unwrap();
        // Layer 0: +0.25, Layer 1: +0.5, Layer 2: +0.75, Layer 3: +1.0 -> total added = 2.5
        // Input was 2.0 -> expected output is 4.5
        for val in out_vec {
            assert!((val - 4.5f32).abs() < 1e-5);
        }
    }
}
