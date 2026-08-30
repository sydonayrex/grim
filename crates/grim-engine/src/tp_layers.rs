//! Megatron-style Tensor Parallel (TP) Layers with collective synchronization.
//!
//! Provides ColumnParallelLinear, RowParallelLinear, and TPMLP
//! layers that split weight tensors across multiple GPU ranks and perform
//! AllReduce / AllGather communication steps.

use std::sync::Arc;
use grim_core::error::Result;
use grim_nn::modules::{Linear, TensorParallelConfig, silu_mul_on_device};
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

/// Column-Parallel Linear Layer.
///
/// Weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$ is partitioned along
/// the output dimension ($dim=0$): each rank $i$ holds $W_i \in \mathbb{R}^{(d_{out}/TP) \times d_{in}}$.
///
/// Forward computation: $Y_i = X \cdot W_i^T$, yielding a column shard of the output.
#[derive(Clone)]
pub struct ColumnParallelLinear {
    /// Local sharded linear layer.
    pub linear: Linear,
    /// TP configuration (rank and world_size).
    pub tp_config: TensorParallelConfig,
    /// Whether to all-gather the output across all ranks.
    pub gather_output: bool,
    /// Parallel communicator for collectives.
    pub comm: Option<Arc<ParallelCommunicator>>,
}

impl ColumnParallelLinear {
    /// Constructs a ColumnParallelLinear from an existing sharded linear layer.
    pub fn new(
        linear: Linear,
        tp_config: TensorParallelConfig,
        gather_output: bool,
        comm: Option<Arc<ParallelCommunicator>>,
    ) -> Self {
        Self {
            linear,
            tp_config,
            gather_output,
            comm,
        }
    }

    /// Performs the forward column-parallel linear projection.
    ///
    /// # Contract
    /// Input `x` is replicated across all TP ranks.
    /// Returns local shard $Y_i \in \mathbb{R}^{B \times (d_{out}/TP)}$, or full gathered tensor if `gather_output` is true.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let local_out = self.linear.forward(x)?;

        if !self.gather_output || self.tp_config.world_size <= 1 {
            return Ok(local_out);
        }

        if let Some(comm) = &self.comm {
            let local_vec = local_out.to_vec_f32()?;
            let total_len = local_vec.len() * self.tp_config.world_size;
            let mut gathered = vec![0.0f32; total_len];
            comm.all_gather_f32(&local_vec, &mut gathered)?;

            let orig_shape = local_out.shape().dims();
            let mut new_dims = orig_shape.to_vec();
            if let Some(last) = new_dims.last_mut() {
                *last *= self.tp_config.world_size;
            }
            Ok(tensor_from_f32_vec(gathered, Shape::from_slice(&new_dims)))
        } else {
            Ok(local_out)
        }
    }
}

/// Row-Parallel Linear Layer.
///
/// Weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$ is partitioned along
/// the input dimension ($dim=1$): each rank $i$ holds $W_i \in \mathbb{R}^{d_{out} \times (d_{in}/TP)}$.
///
/// Forward computation: $Y_i = X_i \cdot W_i^T$.
/// An `AllReduce(SUM)` collective accumulates partial results: $Y = \sum_{i=0}^{TP-1} Y_i$.
#[derive(Clone)]
pub struct RowParallelLinear {
    /// Local sharded linear layer.
    pub linear: Linear,
    /// TP configuration (rank and world_size).
    pub tp_config: TensorParallelConfig,
    /// Parallel communicator for collectives.
    pub comm: Option<Arc<ParallelCommunicator>>,
}

impl RowParallelLinear {
    /// Constructs a RowParallelLinear from an existing sharded linear layer.
    pub fn new(
        linear: Linear,
        tp_config: TensorParallelConfig,
        comm: Option<Arc<ParallelCommunicator>>,
    ) -> Self {
        Self {
            linear,
            tp_config,
            comm,
        }
    }

    /// Performs forward row-parallel linear projection with AllReduce reduction.
    ///
    /// # Contract
    /// Input `x` is column-sharded ($X_i \in \mathbb{R}^{B \times (d_{in}/TP)}$).
    /// Returns the full accumulated result $Y \in \mathbb{R}^{B \times d_{out}}$ on all ranks.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let local_out = self.linear.forward(x)?;

        if self.tp_config.world_size <= 1 {
            return Ok(local_out);
        }

        if let Some(comm) = &self.comm {
            let mut out_vec = local_out.to_vec_f32()?;
            comm.all_reduce_sum_f32(&mut out_vec)?;
            Ok(tensor_from_f32_vec(out_vec, local_out.shape().clone()))
        } else {
            Ok(local_out)
        }
    }
}

/// Tensor-Parallel MLP block (Gate/Up Column-Parallel -> SwiGLU -> Down Row-Parallel).
#[derive(Clone)]
pub struct TPMLP {
    /// Column-parallel gate projection ($d_{model} \to d_{ffn}/TP$).
    pub gate_proj: ColumnParallelLinear,
    /// Column-parallel up projection ($d_{model} \to d_{ffn}/TP$).
    pub up_proj: ColumnParallelLinear,
    /// Row-parallel down projection ($d_{ffn}/TP \to d_{model}$) with AllReduce.
    pub down_proj: RowParallelLinear,
}

impl TPMLP {
    /// Performs forward MLP computation sharded across TP ranks.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let intermediate = silu_mul_on_device(&gate, &up)?;
        self.down_proj.forward(&intermediate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_rocm::device::parallel_comm::HostStagingRing;

    #[test]
    fn test_column_parallel_gather() {
        let ring = Arc::new(HostStagingRing::new(2));
        let comm0 = Arc::new(ParallelCommunicator::with_shared_staging(0, 2, vec![0, 1], ring.clone()).unwrap());

        let tp0 = TensorParallelConfig { rank: 0, world_size: 2 };
        let w0 = tensor_from_f32_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::from_slice(&[2, 2]));
        let col_lin = ColumnParallelLinear::new(Linear::from_tensor(w0, None), tp0, true, Some(comm0));

        let x = tensor_from_f32_vec(vec![1.0, 2.0], Shape::from_slice(&[1, 2]));
        let out = col_lin.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[1, 4]);
    }
}
