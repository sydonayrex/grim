//! Computation Graph Intermediate Representation for Grim model fusion.
//!
//! This is a Burn-inspired IR that splits a model into typed `GraphNode`s and
//! then identifies fusable sequences (e.g. RmsNorm + MatMul -> fused_rmsnorm_matmul_rocm).

use grim_tensor::{ArithType, Shape};

/// Operation types that can be fused in the ROCm backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    MatMul,
    RmsNorm,
    Silu,
    Gelu,
    QkvProjection,
    AttentionScore,
    Linear,
}

/// A single node in the computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub op_type: OpType,
    pub input_tensors: Vec<String>,
    pub output_tensor: String,
    pub shape: Option<Shape>,
    pub dtype: ArithType,
}

/// The computation graph representing a model or layer.
#[derive(Debug, Clone, Default)]
pub struct ComputationGraph {
    pub nodes: Vec<GraphNode>,
    pub entry_points: Vec<String>,
    pub fusion_candidates: Vec<FusionSequence>,
}

/// A contiguous sequence of graph ops that can be fused into one backend operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionSequence {
    pub ops: Vec<OpType>,
    pub target_backend_op: String,
}

impl ComputationGraph {
    /// Build an empty graph. Used by callers that append nodes incrementally.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a node; assigns the next auto-incrementing id.
    pub fn push(&mut self, op_type: OpType, output_tensor: impl Into<String>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            id,
            op_type,
            input_tensors: Vec::new(),
            output_tensor: output_tensor.into(),
            shape: None,
            dtype: ArithType::F32,
        });
        id
    }

    /// Identify fusable operation sequences from the graph in node order.
    ///
    /// Detects:
    /// - `RmsNorm` followed by `MatMul` or `Linear` -> `fused_rmsnorm_matmul_rocm`
    /// - `QkvProjection` followed by `AttentionScore` -> `fused_qkv_attention_rocm`
    pub fn identify_fusion_sequences(&mut self) {
        self.fusion_candidates.clear();
        let mut current: Vec<OpType> = Vec::new();

        for node in &self.nodes {
            match node.op_type {
                OpType::RmsNorm => {
                    if !current.is_empty() {
                        current.clear();
                    }
                    current.push(OpType::RmsNorm);
                }
                OpType::MatMul | OpType::Linear
                    if matches!(current.last(), Some(&OpType::RmsNorm)) =>
                {
                    current.push(node.op_type.clone());
                    self.fusion_candidates.push(FusionSequence {
                        ops: current.clone(),
                        target_backend_op: "fused_rmsnorm_matmul_rocm".to_string(),
                    });
                    current.clear();
                }
                OpType::QkvProjection => {
                    if !current.is_empty() {
                        current.clear();
                    }
                    current.push(OpType::QkvProjection);
                }
                OpType::AttentionScore
                    if matches!(current.last(), Some(&OpType::QkvProjection)) =>
                {
                    current.push(OpType::AttentionScore);
                    self.fusion_candidates.push(FusionSequence {
                        ops: current.clone(),
                        target_backend_op: "fused_qkv_attention_rocm".to_string(),
                    });
                    current.clear();
                }
                _ => current.clear(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_has_no_fusion_candidates() {
        let mut g = ComputationGraph::new();
        g.identify_fusion_sequences();
        assert!(g.fusion_candidates.is_empty());
    }

    #[test]
    fn rmsnorm_followed_by_matmul_is_fused() {
        let mut g = ComputationGraph::new();
        g.push(OpType::RmsNorm, "normed");
        g.push(OpType::MatMul, "out");
        g.identify_fusion_sequences();
        assert_eq!(g.fusion_candidates.len(), 1);
        let cand = &g.fusion_candidates[0];
        assert_eq!(cand.ops, vec![OpType::RmsNorm, OpType::MatMul]);
        assert_eq!(cand.target_backend_op, "fused_rmsnorm_matmul_rocm");
    }

    #[test]
    fn rmsnorm_followed_by_linear_is_fused() {
        let mut g = ComputationGraph::new();
        g.push(OpType::RmsNorm, "n");
        g.push(OpType::Linear, "out");
        g.identify_fusion_sequences();
        assert_eq!(g.fusion_candidates.len(), 1);
        assert_eq!(g.fusion_candidates[0].ops[1], OpType::Linear);
    }

    #[test]
    fn qkv_projection_followed_by_attention_is_fused() {
        let mut g = ComputationGraph::new();
        g.push(OpType::QkvProjection, "qkv");
        g.push(OpType::AttentionScore, "scores");
        g.identify_fusion_sequences();
        assert_eq!(g.fusion_candidates.len(), 1);
        assert_eq!(g.fusion_candidates[0].target_backend_op, "fused_qkv_attention_rocm");
    }

    #[test]
    fn matmul_without_rmsnorm_is_not_fused() {
        let mut g = ComputationGraph::new();
        g.push(OpType::MatMul, "out");
        g.identify_fusion_sequences();
        assert!(g.fusion_candidates.is_empty());
    }

    #[test]
    fn rmsnorm_unmatched_does_not_emit_candidate() {
        let mut g = ComputationGraph::new();
        g.push(OpType::RmsNorm, "n");
        g.push(OpType::Silu, "s");
        g.identify_fusion_sequences();
        assert!(g.fusion_candidates.is_empty());
    }

    #[test]
    fn two_independent_rmsnorm_matmul_pairs_emit_two_candidates() {
        let mut g = ComputationGraph::new();
        g.push(OpType::RmsNorm, "n1");
        g.push(OpType::MatMul, "out1");
        g.push(OpType::RmsNorm, "n2");
        g.push(OpType::MatMul, "out2");
        g.identify_fusion_sequences();
        assert_eq!(g.fusion_candidates.len(), 2);
        assert_eq!(g.fusion_candidates[0].target_backend_op, "fused_rmsnorm_matmul_rocm");
        assert_eq!(g.fusion_candidates[1].target_backend_op, "fused_rmsnorm_matmul_rocm");
    }

    #[test]
    fn push_assigns_monotonic_ids() {
        let mut g = ComputationGraph::new();
        let a = g.push(OpType::RmsNorm, "n");
        let b = g.push(OpType::MatMul, "out");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }
}
