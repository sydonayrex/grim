//! Fusion-pattern detection over checkpoint tensor names (folded in from
//! the former grim-tensor-graph crate).
//!
//! Detects which [`GrimFusionOp`] fusion groups a checkpoint's tensor
//! naming suggests, so the oxidizer can recommend fusion ops for the
//! `.grim` `fusion_mask`. Detection is intentionally name-substring based
//! (`ponytail:` heuristic — upgrade to dataflow-matched detection only if
//! a consumer needs per-layer tensor-level pairing; today's consumer,
//! `recommended_fusion_ops`, is op-level only).

use super::gguf::GrimFusionOp;

/// A detected fusion group combining ops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionGroup {
    pub op: GrimFusionOp,
    pub tensors: Vec<String>,
}

/// IR graph with tensor nodes and fusion groups.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorGraphIr {
    pub nodes: Vec<String>,
    pub fusion_groups: Vec<FusionGroup>,
}

impl TensorGraphIr {
    /// Returns unique fusion ops recommended for this graph.
    pub fn recommended_fusion_ops(&self) -> Vec<GrimFusionOp> {
        let mut ops = Vec::new();
        for group in &self.fusion_groups {
            if !ops.contains(&group.op) {
                ops.push(group.op);
            }
        }
        ops
    }
}

pub fn build_transformer_ir<'a, I>(tensor_names: I) -> TensorGraphIr
where
    I: IntoIterator<Item = &'a str>,
{
    let names: Vec<String> = tensor_names.into_iter().map(str::to_string).collect();
    let nodes = names.clone();

    let mut fusion_groups = Vec::new();

    if let Some(group) = detect_rmsnorm_matmul(&names) {
        fusion_groups.push(group);
    }
    if let Some(group) = detect_qkv_attention(&names) {
        fusion_groups.push(group);
    }

    TensorGraphIr {
        nodes,
        fusion_groups,
    }
}

fn detect_rmsnorm_matmul(names: &[String]) -> Option<FusionGroup> {
    let norm = find_first(names, &["input_layernorm", "attention_norm", "rms_norm"])?;
    let linear = find_first(
        names,
        &[
            "attn_q.weight",
            "attention.wq.weight",
            "self_attn.q_proj.weight",
        ],
    )?;
    Some(FusionGroup {
        op: GrimFusionOp::RmsNormMatMul,
        tensors: vec![norm, linear],
    })
}

fn detect_qkv_attention(names: &[String]) -> Option<FusionGroup> {
    let q = find_first(
        names,
        &[
            "attn_q.weight",
            "attention.wq.weight",
            "self_attn.q_proj.weight",
        ],
    )?;
    let k = find_first(
        names,
        &[
            "attn_k.weight",
            "attention.wk.weight",
            "self_attn.k_proj.weight",
        ],
    )?;
    let v = find_first(
        names,
        &[
            "attn_v.weight",
            "attention.wv.weight",
            "self_attn.v_proj.weight",
        ],
    )?;
    Some(FusionGroup {
        op: GrimFusionOp::QkvAttention,
        tensors: vec![q, k, v],
    })
}

fn find_first(names: &[String], needles: &[&str]) -> Option<String> {
    names
        .iter()
        .find(|name| needles.iter().any(|needle| name.contains(needle)))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_transformer_fusion_patterns() {
        let ir = build_transformer_ir([
            "blk.0.attention_norm.weight",
            "blk.0.attention.wq.weight",
            "blk.0.attention.wk.weight",
            "blk.0.attention.wv.weight",
        ]);
        let ops = ir.recommended_fusion_ops();
        assert!(ops.contains(&GrimFusionOp::RmsNormMatMul));
        assert!(ops.contains(&GrimFusionOp::QkvAttention));
    }

    #[test]
    fn qkv_detection_requires_all_three_projections() {
        let ir = build_transformer_ir([
            "blk.0.attention_norm.weight",
            "blk.0.attention.wq.weight",
            "blk.0.attention.wk.weight",
            // No wv → no QkvAttention group.
        ]);
        assert!(!ir.recommended_fusion_ops().contains(&GrimFusionOp::QkvAttention));
    }

    #[test]
    fn empty_names_yield_no_fusion_groups() {
        let ir = build_transformer_ir(Vec::<&str>::new());
        assert!(ir.nodes.is_empty());
        assert!(ir.fusion_groups.is_empty());
        assert!(ir.recommended_fusion_ops().is_empty());
    }

    #[test]
    fn recommended_ops_are_deduplicated() {
        // Two layers that both match → op list must contain each op once.
        let ir = build_transformer_ir([
            "blk.0.attention_norm.weight",
            "blk.0.attention.wq.weight",
            "blk.0.attention.wk.weight",
            "blk.0.attention.wv.weight",
            "blk.1.attention_norm.weight",
            "blk.1.attention.wq.weight",
            "blk.1.attention.wk.weight",
            "blk.1.attention.wv.weight",
        ]);
        let ops = ir.recommended_fusion_ops();
        assert_eq!(ops.len(), 2, "each op must appear exactly once: {ops:?}");
    }
}
