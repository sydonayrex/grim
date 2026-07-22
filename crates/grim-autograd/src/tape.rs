//! Autograd tape (WI-T1 item 2).
//!
//! Records only the ops touching adapter parameters during forward —
//! matmul, elementwise add, scale, and the fused LoRA application. The
//! recorded tensor values are held by reference-style id; the inputs and
//! outputs are stored in the tape's own `TensorRegistry`. Backward walks
//! the tape in reverse (entries are popped from the back) and routes
//! gradients through the recorded ops.

use crate::param::ParamId;
use grim_tensor::Tensor;
use std::collections::HashMap;

/// Identifier for a tensor in the tape's registry. Id-only (no reference)
/// because the tape owns the tensor data and the graph rewrites itself
/// during replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u32);

/// What kind of op a tape entry represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapeKind {
    /// `output = input @ weight^T`. Arity is 2.
    MatMul,
    /// `output = lhs + rhs`. Arity is 2.
    Add,
    /// `output = input * scale`. Arity is 1.
    Scale,
    /// Fused LoRA: `output = base + scale * (x @ A^T) @ B^T`. Arity is 4.
    LoRAApply,
}

/// A single recorded operation on the tape.
///
/// Holds the data needed to replay its backward pass *without* needing the
/// input tensors to still be alive — the forward pass stores them in the
/// tape's tensor registry as it goes.
#[derive(Debug, Clone)]
pub struct TapeEntry {
    pub kind: TapeKind,
    /// Input tensor ids in declaration order.
    pub inputs: Vec<TensorId>,
    /// Output tensor id.
    pub output: TensorId,
    /// If this op touches a trainable LoRA parameter, its id (for
    /// `trainable_params.accumulate_grad`).
    pub param_id: Option<ParamId>,
    /// Operation-specific context.
    pub metadata: TapeMetadata,
}

/// Operation-specific context. Mirrors the existing forward op semantics so
/// backward can compute the right gradient.
#[derive(Debug, Clone)]
pub enum TapeMetadata {
    /// MatMul with optional transposes (the LoRA path uses transposed operands).
    MatMul {
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        k: usize,
        n: usize,
    },
    /// Add — gradient just routes through.
    Add,
    /// Scale with `alpha / rank` factor.
    Scale { factor: f32 },
    /// LoRA fused op — both A and B are trainable.
    LoRAApply {
        alpha: f32,
        rank: usize,
        a: ParamId,
        b: ParamId,
    },
}

#[derive(Debug, Default)]
pub struct Tape {
    entries: Vec<TapeEntry>,
    tensors: HashMap<TensorId, Tensor>,
    param_tensors: HashMap<ParamId, TensorId>,
    next_id: u32,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tensor and return its `TensorId`.
    pub fn register(&mut self, t: Tensor) -> TensorId {
        let id = TensorId(self.next_id);
        self.next_id += 1;
        self.tensors.insert(id, t);
        id
    }

    /// Register a trainable parameter tensor — also records the param→tensor mapping
    /// so `accumulate_grad` can route the gradient.
    pub fn register_param(&mut self, p: ParamId, t: Tensor) -> TensorId {
        let id = self.register(t);
        self.param_tensors.insert(p, id);
        id
    }

    pub fn get(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(&id)
    }

    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut Tensor> {
        self.tensors.get_mut(&id)
    }

    pub fn param_tensor(&self, p: ParamId) -> Option<TensorId> {
        self.param_tensors.get(&p).copied()
    }

    pub fn entries(&self) -> &[TapeEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the tape (call between training steps).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.tensors.clear();
        self.param_tensors.clear();
        self.next_id = 0;
    }

    /// Record a MatMul `output = a @ b` (possibly transposed).
    ///
    /// `m`, `k`, `n` are the matmul dims before any transpose so that
    /// backward can reconstruct the gradient shapes.
    pub fn record_matmul(
        &mut self,
        a: TensorId,
        b: TensorId,
        output: Tensor,
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        k: usize,
        n: usize,
        param_id: Option<ParamId>,
    ) -> TensorId {
        let out_id = self.register(output);
        self.entries.push(TapeEntry {
            kind: TapeKind::MatMul,
            inputs: vec![a, b],
            output: out_id,
            param_id,
            metadata: TapeMetadata::MatMul {
                transpose_a,
                transpose_b,
                m,
                k,
                n,
            },
        });
        out_id
    }

    pub fn record_add(
        &mut self,
        lhs: TensorId,
        rhs: TensorId,
        output: Tensor,
        param_id: Option<ParamId>,
    ) -> TensorId {
        let out_id = self.register(output);
        self.entries.push(TapeEntry {
            kind: TapeKind::Add,
            inputs: vec![lhs, rhs],
            output: out_id,
            param_id,
            metadata: TapeMetadata::Add,
        });
        out_id
    }

    pub fn record_scale(
        &mut self,
        input: TensorId,
        output: Tensor,
        factor: f32,
        param_id: Option<ParamId>,
    ) -> TensorId {
        let out_id = self.register(output);
        self.entries.push(TapeEntry {
            kind: TapeKind::Scale,
            inputs: vec![input],
            output: out_id,
            param_id,
            metadata: TapeMetadata::Scale { factor },
        });
        out_id
    }

    /// Record a fused LoRA application `output = base + scale * (x @ A^T) @ B^T`.
    pub fn record_lora_apply(
        &mut self,
        base: TensorId,
        x: TensorId,
        a: TensorId,
        b: TensorId,
        output: Tensor,
        alpha: f32,
        rank: usize,
        a_param: ParamId,
        b_param: ParamId,
    ) -> TensorId {
        let out_id = self.register(output);
        self.entries.push(TapeEntry {
            kind: TapeKind::LoRAApply,
            inputs: vec![base, x, a, b],
            output: out_id,
            // Mark as a trainable op using the A id — both A and B grads are
            // accumulated by backward() and the registry picks them up.
            param_id: Some(a_param),
            metadata: TapeMetadata::LoRAApply {
                alpha,
                rank,
                a: a_param,
                b: b_param,
            },
        });
        out_id
    }

    /// Reverse-order iteration for backward pass.
    pub fn iter_rev(&self) -> impl Iterator<Item = &TapeEntry> {
        self.entries.iter().rev()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    fn t(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        cpu_tensor(data, Shape::new(shape))
    }

    #[test]
    fn tape_records_and_returns_tensors() {
        let mut tape = Tape::new();
        let a = tape.register(t(vec![1.0, 2.0], vec![2]));
        let b = tape.register(t(vec![3.0, 4.0], vec![2]));
        assert_eq!(tape.get(a).unwrap().to_vec_f32().unwrap(), vec![1.0, 2.0]);
        assert_eq!(tape.get(b).unwrap().to_vec_f32().unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn tape_records_matmul() {
        let mut tape = Tape::new();
        let a = tape.register(t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        let b = tape.register(t(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]));
        let out = tape.record_matmul(
            a,
            b,
            t(vec![19.0, 22.0, 43.0, 50.0], vec![2, 2]),
            false,
            false,
            2,
            2,
            2,
            None,
        );
        assert_eq!(tape.len(), 1);
        let entry = &tape.entries()[0];
        assert_eq!(entry.kind, TapeKind::MatMul);
        assert_eq!(entry.outputs_check(out), true);
    }

    #[test]
    fn tape_records_lora_apply_with_trainable_params() {
        let mut tape = Tape::new();
        let base = tape.register(t(vec![1.0, 2.0], vec![1, 2]));
        let x = tape.register(t(vec![0.5, 0.5], vec![1, 2]));
        let a_param = ParamId::a(0, 1);
        let b_param = ParamId::b(0, 1);
        let a = tape.register_param(a_param, t(vec![0.1, 0.1], vec![1, 2]));
        let b = tape.register_param(b_param, t(vec![0.2, 0.3], vec![2, 1]));
        let _out = tape.record_lora_apply(
            base, x, a, b, t(vec![1.01, 2.01], vec![1, 2]),
            1.0, 1, a_param, b_param,
        );
        assert_eq!(tape.len(), 1);
        let e = &tape.entries()[0];
        assert_eq!(e.kind, TapeKind::LoRAApply);
        assert_eq!(e.param_id, Some(a_param));
        assert_eq!(tape.param_tensor(a_param), Some(a));
        assert_eq!(tape.param_tensor(b_param), Some(b));
    }

    // Helper for the test above
    impl TapeEntry {
        pub(in crate) fn outputs_check(&self, out: TensorId) -> bool {
            self.output == out
        }
    }
}
