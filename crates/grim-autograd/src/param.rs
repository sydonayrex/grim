//! Trainable parameter types and identifiers (WI-T1 item 4).
//!
//! Gradient accumulation buffers for `A`/`B` per adapter, per layer.

use crate::injection::LoRAInjectionPoint;
use grim_tensor::{BackendDevice, DType, Tensor, error::Result};
use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for a trainable parameter (adapter A or B matrix).
///
/// `(layer_idx, adapter_id, point, is_a)` — four coordinate fields so a single
/// hash lookup resolves any adapter's gradient buffer anywhere in the model.
/// `point` is required because a single (layer, adapter) pair owns a *distinct*
/// A/B matrix per injection point (Q/K/V/O/Gate/Up/Down): without it, all
/// seven points in a layer collide on the same `ParamId` and the
/// `TrainableParams` map collapses 14 distinct adapters into 2 overwriting
/// entries (last writer wins), which is both a silent shape bug and a silent
/// gradient-mixing bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ParamId {
    pub layer_idx: usize,
    pub adapter_id: u32,
    pub point: LoRAInjectionPoint,
    /// `true` = A (down projection), `false` = B (up projection).
    pub is_a: bool,
}

impl ParamId {
    pub fn new(layer_idx: usize, adapter_id: u32, point: LoRAInjectionPoint, is_a: bool) -> Self {
        Self {
            layer_idx,
            adapter_id,
            point,
            is_a,
        }
    }

    /// LoRA A matrix: `[rank, in_features]`.
    pub fn a(layer_idx: usize, adapter_id: u32, point: LoRAInjectionPoint) -> Self {
        Self::new(layer_idx, adapter_id, point, true)
    }

    /// LoRA B matrix: `[out_features, rank]`.
    pub fn b(layer_idx: usize, adapter_id: u32, point: LoRAInjectionPoint) -> Self {
        Self::new(layer_idx, adapter_id, point, false)
    }

    /// Base weight (non-LoRA) parameter ID — adapter_id is always 0;
    /// is_a is always true since there is no A/B distinction for base weights.
    pub fn base(layer_idx: usize, point: LoRAInjectionPoint) -> Self {
        Self {
            layer_idx,
            adapter_id: 0,
            point,
            is_a: true,
        }
    }
}

/// A trainable parameter tensor paired with its gradient accumulator.
///
/// `data` is the live parameter value mutated by the optimizer (WI-T4);
/// `grad` is the accumulated gradient written by `backward` (WI-T1) and
/// zeroed at the start of each step. When `frozen == true`, the parameter is
/// tracked (so it stays on the tape for downstream gradients) but `accumulate_grad`
/// is a no-op and optimizers skip it — the frozen-base QLoRA mechanism where
/// base weights are registered in `TrainableParams` but never updated.
#[derive(Debug, Clone)]
pub struct TrainableParam {
    pub id: ParamId,
    pub data: Tensor,
    grad: Tensor,
    frozen: bool,
}

impl TrainableParam {
    /// Create a new trainable parameter with a zero-initialized gradient
    /// buffer matching the parameter's shape and device.
    pub fn new(id: ParamId, data: Tensor) -> Result<Self> {
        let grad = zeros_like(&data)?;
        Ok(Self {
            id,
            data,
            grad,
            frozen: false,
        })
    }

    /// Create a new parameter and mark it frozen (base-weight tracking).
    pub fn register_base_weight(id: ParamId, data: Tensor, frozen: bool) -> Result<Self> {
        let grad = zeros_like(&data)?;
        Ok(Self {
            id,
            data,
            grad,
            frozen,
        })
    }

    /// Accumulate `grad` into this parameter's gradient buffer (`grad += g`).
    /// No-op for frozen parameters.
    pub fn accumulate_grad(&mut self, grad: &Tensor) -> Result<()> {
        if self.frozen {
            return Ok(());
        }
        let dev = crate::pick_device_for_tensor(&self.grad);
        let (sum_storage, handle) = BackendDevice::add(
            &*dev,
            self.grad.storage().as_ref(),
            grad.storage().as_ref(),
            self.grad.shape(),
        )?;
        handle.synchronize()?;
        self.grad = Tensor::new(
            Arc::from(sum_storage),
            self.grad.shape().clone(),
            DType::F32,
            self.grad.provenance().clone(),
            self.grad.device().clone(),
        );
        Ok(())
    }

    /// Zero out the gradient buffer in place.
    pub fn zero_grad(&mut self) -> Result<()> {
        self.grad = zeros_like(&self.grad)?;
        Ok(())
    }

    /// Whether this parameter is frozen (tracked but not updated).
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Mark this parameter frozen/unfrozen.
    pub fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
    }

    /// Read-only view of the accumulated gradient.
    pub fn grad(&self) -> &Tensor {
        &self.grad
    }

    /// Mutable view of the accumulated gradient (used by the optimizer).
    pub fn grad_mut(&mut self) -> &mut Tensor {
        &mut self.grad
    }
}

/// Registry of all trainable parameters in the model — the full set of
/// gradient accumulation buffers for `A`/`B` per adapter, per layer.
#[derive(Debug, Clone, Default)]
pub struct TrainableParams {
    params: HashMap<ParamId, TrainableParam>,
}

impl TrainableParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, param: TrainableParam) {
        self.params.insert(param.id, param);
    }

    pub fn get(&self, id: ParamId) -> Option<&TrainableParam> {
        self.params.get(&id)
    }

    pub fn get_mut(&mut self, id: ParamId) -> Option<&mut TrainableParam> {
        self.params.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ParamId, &TrainableParam)> {
        self.params.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&ParamId, &mut TrainableParam)> {
        self.params.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    pub fn all_reduce_grads(
        &mut self,
        dev: &dyn grim_tensor::backend::BackendDevice,
        placement: &grim_tensor::backend::ScythePlacement,
        rccl: Option<&grim_backend_rocm::RcclAllReduce>,
    ) -> Result<()> {
        let num_gpus = placement.ranks.len().max(1);

        // ── Fast path: RCCL device-pointer all-reduce ─────────────────────
        // When an RcclAllReduce handle is provided and we have >1 GPU, reduce
        // gradients on-device via ncclAllReduce to avoid the D2H round-trip.
        if let Some(rccl_handle) = rccl {
            if num_gpus > 1 {
                for (_, param) in self.params.iter_mut() {
                    let dev_ptr = param.grad.storage().device_ptr();
                    let count = param.grad.shape().elem_count();

                    if let Some(ptr) = dev_ptr {
                        // In-place device-pointer all-reduce: send and recv
                        // alias the same buffer so ncclAllReduce reduces into
                        // the gradient tensor directly.
                        let stream = 0u64; // default HIP stream
                        rccl_handle.sum_gradients_device(ptr, ptr, count, stream)?;
                        // ncclAllReduce with NCCL_SUM produces a sum; scale by
                        // 1/num_gpus to get the mean. Use the backend's
                        // mul_scalar kernel so this stays on-device — no D2H
                        // round-trip.
                        let scale = 1.0 / num_gpus as f32;
                        let (scaled_storage, handle) = dev.mul_scalar(
                            param.grad.storage().as_ref(),
                            scale,
                            param.grad.shape(),
                        )?;
                        handle.synchronize()?;
                        param.grad = Tensor::new(
                            Arc::from(scaled_storage),
                            param.grad.shape().clone(),
                            param.grad.dtype(),
                            param.grad.provenance().clone(),
                            param.grad.device().clone(),
                        );
                    } else {
                        // CPU fallback for this tensor: host round-trip.
                        let grad_vec = param.grad.to_vec_f32()?;
                        let mut scaled = grad_vec;
                        rccl_handle.scale_gradients(&mut scaled)?;
                        let grad_tensor =
                            grim_backend_cpu::cpu_tensor(scaled, param.grad.shape().clone());
                        param.accumulate_grad(&grad_tensor)?;
                    }
                }
                return Ok(());
            }
        }

        // ── Fallback: CPU-only accumulate (single-GPU or no RCCL) ──────────
        for (_, param) in self.params.iter_mut() {
            let grad_vec = param.grad.to_vec_f32()?;
            let grad_tensor = grim_backend_cpu::cpu_tensor(grad_vec, param.grad.shape().clone());
            param.accumulate_grad(&grad_tensor)?;
        }
        Ok(())
    }

    /// Zero all gradient buffers — called at the start of each training step.
    pub fn zero_all_grads(&mut self) -> Result<()> {
        for (_, param) in self.params.iter_mut() {
            param.zero_grad()?;
        }
        Ok(())
    }

    /// Collect every parameter id in the registry.
    pub fn ids(&self) -> Vec<ParamId> {
        self.params.keys().copied().collect()
    }
}

/// Allocate a zero tensor with the same shape, dtype, provenance, and device
/// as `t`. Used to initialize gradient buffers.
pub fn zeros_like(t: &Tensor) -> Result<Tensor> {
    let shape = t.shape().clone();
    let zeros = vec![0.0f32; shape.elem_count()];
    let dev = crate::pick_device_for_tensor(t);
    let storage = dev.from_cpu(&zeros, &shape, DType::F32)?;
    Ok(Tensor::new(
        Arc::from(storage),
        shape,
        DType::F32,
        t.provenance().clone(),
        t.device().clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        cpu_tensor(data, Shape::new(shape))
    }

    #[test]
    fn param_id_distinguishes_a_and_b() {
        let a = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let b = ParamId::b(0, 1, LoRAInjectionPoint::QProj);
        assert!(a.is_a);
        assert!(!b.is_a);
        assert_ne!(a, b);
        assert_eq!(a.layer_idx, 0);
        assert_eq!(b.adapter_id, 1);
    }

    #[test]
    fn trainable_param_initializes_zero_grad() {
        let id = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let data = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let param = TrainableParam::new(id, data).unwrap();
        assert_eq!(param.id, id);
        assert_eq!(param.grad().shape().dims(), &[2, 2]);
        assert!(param.grad().to_vec_f32().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn accumulate_grad_adds_to_buffer() {
        let mut param = TrainableParam::new(
            ParamId::a(0, 1, LoRAInjectionPoint::QProj),
            tensor(vec![1.0, 2.0], vec![2, 1]),
        )
        .unwrap();
        param
            .accumulate_grad(&tensor(vec![3.0, 4.0], vec![2, 1]))
            .unwrap();
        assert_eq!(param.grad().to_vec_f32().unwrap(), vec![3.0, 4.0]);
        param
            .accumulate_grad(&tensor(vec![1.0, 1.0], vec![2, 1]))
            .unwrap();
        assert_eq!(param.grad().to_vec_f32().unwrap(), vec![4.0, 5.0]);
    }

    #[test]
    fn zero_grad_resets_buffer() {
        let mut param = TrainableParam::new(
            ParamId::a(0, 1, LoRAInjectionPoint::QProj),
            tensor(vec![1.0, 2.0], vec![2, 1]),
        )
        .unwrap();
        param
            .accumulate_grad(&tensor(vec![5.0, 6.0], vec![2, 1]))
            .unwrap();
        assert_eq!(param.grad().to_vec_f32().unwrap(), vec![5.0, 6.0]);
        param.zero_grad().unwrap();
        assert!(param.grad().to_vec_f32().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn frozen_param_accumulate_grad_is_noop() {
        let mut param = TrainableParam::register_base_weight(
            ParamId::a(0, 1, LoRAInjectionPoint::QProj),
            tensor(vec![1.0, 2.0], vec![2, 1]),
            true,
        )
        .unwrap();
        assert!(param.is_frozen());
        param
            .accumulate_grad(&tensor(vec![3.0, 4.0], vec![2, 1]))
            .unwrap();
        assert_eq!(param.grad().to_vec_f32().unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    fn non_frozen_param_defaults_trainable() {
        let param = TrainableParam::new(
            ParamId::a(0, 1, LoRAInjectionPoint::QProj),
            tensor(vec![1.0, 2.0], vec![2, 1]),
        )
        .unwrap();
        assert!(!param.is_frozen());
    }

    #[test]
    fn frozen_param_is_skipped_by_optimizer() {
        let mut params = TrainableParams::new();
        params.insert(
            TrainableParam::register_base_weight(
                ParamId::a(0, 1, LoRAInjectionPoint::QProj),
                tensor(vec![1.0], vec![1]),
                true,
            )
            .unwrap(),
        );
        params.insert(
            TrainableParam::new(
                ParamId::b(0, 1, LoRAInjectionPoint::QProj),
                tensor(vec![1.0], vec![1]),
            )
            .unwrap(),
        );

        let mut adam = crate::adamw::AdamW::new(crate::adamw::AdamWConfig {
            lr: 0.5,
            ..crate::adamw::AdamWConfig::default()
        });
        // Give the trainable param a gradient so it gets updated.
        params
            .get_mut(ParamId::b(0, 1, LoRAInjectionPoint::QProj))
            .unwrap()
            .accumulate_grad(&tensor(vec![1.0], vec![1]))
            .unwrap();
        adam.step(&mut params).unwrap();

        // Frozen param unchanged; trainable param moved by lr * grad.
        let frozen_val = params
            .get(ParamId::a(0, 1, LoRAInjectionPoint::QProj))
            .unwrap()
            .data
            .to_vec_f32()
            .unwrap()[0];
        let trained_val = params
            .get(ParamId::b(0, 1, LoRAInjectionPoint::QProj))
            .unwrap()
            .data
            .to_vec_f32()
            .unwrap()[0];
        assert_eq!(frozen_val, 1.0);
        assert!((trained_val - 1.0).abs() > 0.0);
    }

    #[test]
    fn trainable_params_registry_zeroes_all() {
        let mut params = TrainableParams::new();
        params.insert(
            TrainableParam::new(
                ParamId::a(0, 1, LoRAInjectionPoint::QProj),
                tensor(vec![1.0], vec![1]),
            )
            .unwrap(),
        );
        params.insert(
            TrainableParam::new(
                ParamId::b(0, 1, LoRAInjectionPoint::QProj),
                tensor(vec![2.0], vec![1]),
            )
            .unwrap(),
        );
        params
            .get_mut(ParamId::a(0, 1, LoRAInjectionPoint::QProj))
            .unwrap()
            .accumulate_grad(&tensor(vec![7.0], vec![1]))
            .unwrap();
        params.zero_all_grads().unwrap();
        for (_, p) in params.iter() {
            assert!(p.grad().to_vec_f32().unwrap().iter().all(|&v| v == 0.0));
        }
    }
    #[test]
    fn test_all_reduce_grads_execution() {
        let mut params = TrainableParams::new();
        let pid = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let t_data = tensor(vec![1.0f32; 4], vec![2, 2]);
        let mut tp = TrainableParam::new(pid, t_data).unwrap();
        tp.accumulate_grad(&tensor(vec![0.5f32; 4], vec![2, 2]))
            .unwrap();
        params.insert(tp);

        let dev = grim_backend_cpu::CpuDevice::new();
        let placement = grim_tensor::backend::ScythePlacement {
            ranks: vec![0],
            partition: vec![1.0],
            routes: vec![grim_tensor::backend::ScytheLink::Host; 1],
        };

        params.all_reduce_grads(&dev, &placement, None).unwrap();
        assert_eq!(
            params.get(pid).unwrap().grad().to_vec_f32().unwrap(),
            vec![1.0f32; 4]
        );
    }
}
