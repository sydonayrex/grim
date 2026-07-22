//! Trainable parameter types and identifiers (WI-T1 item 4).
//!
//! Gradient accumulation buffers for `A`/`B` per adapter, per layer.

use grim_tensor::{BackendDevice, DType, Tensor, error::Result};
use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for a trainable parameter (adapter A or B matrix).
///
/// `(layer_idx, adapter_id, is_a)` — three coordinate fields so a single
/// hash lookup resolves any adapter's gradient buffer anywhere in the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ParamId {
    pub layer_idx: usize,
    pub adapter_id: u32,
    /// `true` = A (down projection), `false` = B (up projection).
    pub is_a: bool,
}

impl ParamId {
    pub fn new(layer_idx: usize, adapter_id: u32, is_a: bool) -> Self {
        Self { layer_idx, adapter_id, is_a }
    }

    /// LoRA A matrix: `[rank, in_features]`.
    pub fn a(layer_idx: usize, adapter_id: u32) -> Self {
        Self::new(layer_idx, adapter_id, true)
    }

    /// LoRA B matrix: `[out_features, rank]`.
    pub fn b(layer_idx: usize, adapter_id: u32) -> Self {
        Self::new(layer_idx, adapter_id, false)
    }
}

/// A trainable parameter tensor paired with its gradient accumulator.
///
/// `data` is the live parameter value mutated by the optimizer (WI-T4);
/// `grad` is the accumulated gradient written by `backward` (WI-T1) and
/// zeroed at the start of each step.
#[derive(Debug, Clone)]
pub struct TrainableParam {
    pub id: ParamId,
    pub data: Tensor,
    grad: Tensor,
}

impl TrainableParam {
    /// Create a new trainable parameter with a zero-initialized gradient
    /// buffer matching the parameter's shape and device.
    pub fn new(id: ParamId, data: Tensor) -> Result<Self> {
        let grad = zeros_like(&data)?;
        Ok(Self { id, data, grad })
    }

    /// Accumulate `grad` into this parameter's gradient buffer (`grad += g`).
    pub fn accumulate_grad(&mut self, grad: &Tensor) -> Result<()> {
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
        let a = ParamId::a(0, 1);
        let b = ParamId::b(0, 1);
        assert!(a.is_a);
        assert!(!b.is_a);
        assert_ne!(a, b);
        assert_eq!(a.layer_idx, 0);
        assert_eq!(b.adapter_id, 1);
    }

    #[test]
    fn trainable_param_initializes_zero_grad() {
        let id = ParamId::a(0, 1);
        let data = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let param = TrainableParam::new(id, data).unwrap();
        assert_eq!(param.id, id);
        assert_eq!(param.grad().shape().dims(), &[2, 2]);
        assert!(param.grad().to_vec_f32().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn accumulate_grad_adds_to_buffer() {
        let mut param = TrainableParam::new(
            ParamId::a(0, 1),
            tensor(vec![1.0, 2.0], vec![2, 1]),
        ).unwrap();
        param.accumulate_grad(&tensor(vec![3.0, 4.0], vec![2, 1])).unwrap();
        assert_eq!(param.grad().to_vec_f32().unwrap(), vec![3.0, 4.0]);
        param.accumulate_grad(&tensor(vec![1.0, 1.0], vec![2, 1])).unwrap();
        assert_eq!(param.grad().to_vec_f32().unwrap(), vec![4.0, 5.0]);
    }

    #[test]
    fn zero_grad_resets_buffer() {
        let mut param = TrainableParam::new(
            ParamId::a(0, 1),
            tensor(vec![1.0, 2.0], vec![2, 1]),
        ).unwrap();
        param.accumulate_grad(&tensor(vec![5.0, 6.0], vec![2, 1])).unwrap();
        assert_eq!(param.grad().to_vec_f32().unwrap(), vec![5.0, 6.0]);
        param.zero_grad().unwrap();
        assert!(param.grad().to_vec_f32().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn trainable_params_registry_zeroes_all() {
        let mut params = TrainableParams::new();
        params.insert(TrainableParam::new(ParamId::a(0, 1), tensor(vec![1.0], vec![1])).unwrap());
        params.insert(TrainableParam::new(ParamId::b(0, 1), tensor(vec![2.0], vec![1])).unwrap());
        params.get_mut(ParamId::a(0, 1)).unwrap().accumulate_grad(&tensor(vec![7.0], vec![1])).unwrap();
        params.zero_all_grads().unwrap();
        for (_, p) in params.iter() {
            assert!(p.grad().to_vec_f32().unwrap().iter().all(|&v| v == 0.0));
        }
    }
}
