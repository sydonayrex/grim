//! Heterogeneous KV Layer Grouping for non-uniform transformer topologies.
//!
//! Repurposed from LMCache's `KernelGroupIdentity` / `kv_layer_groups.py`.
//! Maps model layers into distinct physical layout groups (e.g. standard dense attention,
//! sliding-window attention, hybrid linear attention such as Qwen3.8 GDN, or low-rank MLA)
//! so each layer group can be allocated, transferred, and spilled under independent policies.

use grim_tensor::DType;
use std::collections::HashMap;

/// Identifies the physical memory layout of a layer's KV cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LayerGroupIdentity {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub dtype: DType,
    /// Sliding window length if localized attention (0 for full causal)
    pub sliding_window: usize,
    /// Low-rank compression dimension if MLA / latent-KV (0 for standard MHA/GQA)
    pub latent_dim: usize,
}

impl LayerGroupIdentity {
    pub fn standard_gqa(
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: DType,
    ) -> Self {
        Self {
            num_kv_heads,
            head_dim,
            block_size,
            dtype,
            sliding_window: 0,
            latent_dim: 0,
        }
    }

    pub fn sliding_window_gqa(
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: DType,
        sliding_window: usize,
    ) -> Self {
        Self {
            num_kv_heads,
            head_dim,
            block_size,
            dtype,
            sliding_window,
            latent_dim: 0,
        }
    }

    pub fn mla_latent(
        num_kv_heads: usize,
        head_dim: usize,
        latent_dim: usize,
        block_size: usize,
        dtype: DType,
    ) -> Self {
        Self {
            num_kv_heads,
            head_dim,
            block_size,
            dtype,
            sliding_window: 0,
            latent_dim,
        }
    }

    /// Size in bytes of a single block for this layer group.
    pub fn block_bytes(&self) -> usize {
        let effective_dim = if self.latent_dim > 0 {
            self.latent_dim
        } else {
            self.num_kv_heads * self.head_dim * 2 // K and V
        };
        let elem_bytes = self.dtype.arith.byte_size();
        self.block_size * effective_dim * elem_bytes
    }
}

/// Registry that maps layer indices to their respective physical layer groups.
#[derive(Debug, Default, Clone)]
pub struct LayerGroupRegistry {
    /// Maps layer_idx -> group identity
    layer_to_group: HashMap<usize, LayerGroupIdentity>,
    /// Unique groups present in the model
    groups: Vec<LayerGroupIdentity>,
}

impl LayerGroupRegistry {
    pub fn new() -> Self {
        Self {
            layer_to_group: HashMap::new(),
            groups: Vec::new(),
        }
    }

    /// Register a layer with its layout identity.
    pub fn register_layer(&mut self, layer_idx: usize, identity: LayerGroupIdentity) {
        if !self.groups.contains(&identity) {
            self.groups.push(identity.clone());
        }
        self.layer_to_group.insert(layer_idx, identity);
    }

    /// Query the layout identity of a layer.
    pub fn get_group(&self, layer_idx: usize) -> Option<&LayerGroupIdentity> {
        self.layer_to_group.get(&layer_idx)
    }

    /// Total unique layer layout groups in the model.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_group_registry_hybrid_model() {
        let mut registry = LayerGroupRegistry::new();

        // Hybrid 3:1 linear attention (e.g. Qwen3.8 GDN) + standard GQA
        let gqa = LayerGroupIdentity::standard_gqa(8, 128, 16, DType::F16);
        let swa = LayerGroupIdentity::sliding_window_gqa(8, 128, 16, DType::F16, 4096);

        registry.register_layer(0, swa.clone());
        registry.register_layer(1, swa.clone());
        registry.register_layer(2, swa);
        registry.register_layer(3, gqa.clone());

        assert_eq!(registry.num_groups(), 2);
        assert_eq!(registry.get_group(0).unwrap().sliding_window, 4096);
        assert_eq!(registry.get_group(3).unwrap().sliding_window, 0);
    }
}
