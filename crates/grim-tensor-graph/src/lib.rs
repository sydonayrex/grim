use grim_format::gguf::GrimFusionOp;

pub mod ir;
pub use ir::{ComputationGraph, FusionSequence, GraphNode, OpType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorNode {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionGroup {
    pub op: GrimFusionOp,
    pub tensors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorGraphIr {
    pub nodes: Vec<TensorNode>,
    pub fusion_groups: Vec<FusionGroup>,
}

impl TensorGraphIr {
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
    let nodes = names
        .iter()
        .cloned()
        .map(|name| TensorNode { name })
        .collect::<Vec<_>>();

    let mut fusion_groups = Vec::new();

    if let Some(group) = detect_rmsnorm_matmul(&names) {
        fusion_groups.push(group);
    }
    if let Some(group) = detect_qkv_attention(&names) {
        fusion_groups.push(group);
    }

    TensorGraphIr { nodes, fusion_groups }
}

fn detect_rmsnorm_matmul(names: &[String]) -> Option<FusionGroup> {
    let norm = find_first(names, &["input_layernorm", "attention_norm", "rms_norm"])?;
    let linear = find_first(
        names,
        &["attn_q.weight", "attention.wq.weight", "self_attn.q_proj.weight", "feed_forward.w1.weight"],
    )?;
    Some(FusionGroup {
        op: GrimFusionOp::RmsNormMatMul,
        tensors: vec![norm, linear],
    })
}

fn detect_qkv_attention(names: &[String]) -> Option<FusionGroup> {
    let q = find_first(names, &["attn_q.weight", "attention.wq.weight", "self_attn.q_proj.weight"])?;
    let k = find_first(names, &["attn_k.weight", "attention.wk.weight", "self_attn.k_proj.weight"])?;
    let v = find_first(names, &["attn_v.weight", "attention.wv.weight", "self_attn.v_proj.weight"])?;
    Some(FusionGroup {
        op: GrimFusionOp::QkvAttention,
        tensors: vec![q, k, v],
    })
}

fn find_first(names: &[String], needles: &[&str]) -> Option<String> {
    names.iter()
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
}
