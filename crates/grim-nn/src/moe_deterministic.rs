//! Deterministic token mapping and scoreboard synchronization for MoE dispatch (UniEP).
//!
//! Implements UniEP-style deterministic token ordering (Zheng et al., arXiv:2604.19241)
//! using exclusive prefix-sum offset tables and atomic-free destination addressing.
//! Guarantees bitwise numerical consistency under asynchronous comm-compute overlap
//! and strictly preserves GRIM's `routed_scaling_factor` and shared-expert conventions.

use std::sync::atomic::{AtomicU32, Ordering};
use grim_tensor::error::Error;

/// Deterministic token mapping metadata for MoE expert dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterministicTokenMap {
    /// Total number of tokens in the dispatch batch ($N_{\text{tok}}$).
    pub num_tokens: usize,
    /// Total number of experts ($E$).
    pub num_experts: usize,
    /// Number of active experts per token ($\text{top\_k}$).
    pub top_k: usize,
    /// Total routed token instances across all experts ($N_{\text{tok}} \times \text{top\_k}$).
    pub total_routed_instances: usize,
    /// Token count per expert $C_{\text{exp}}[e]$.
    pub expert_counts: Vec<usize>,
    /// Exclusive prefix-sum offsets $O_{\text{all}}[e] = \sum_{j=0}^{e-1} C_{\text{exp}}[j]$.
    pub global_offsets: Vec<usize>,
    /// Destination memory slot for each `(token_idx, top_k_slot)` instance.
    /// Shape: `[num_tokens * top_k]`.
    pub destination_slots: Vec<usize>,
    /// Reverse mapping: maps each packed destination slot back to `(token_idx, expert_id, weight_slot)`.
    pub reverse_map: Vec<(usize, usize, usize)>,
}

impl DeterministicTokenMap {
    /// Computes deterministic, conflict-free destination addressing using exclusive prefix sums.
    ///
    /// # Mathematical Guarantee
    /// Given selected expert indices $E_{\text{sel}}[i, k]$:
    /// 1. $C_{\text{exp}}[e] = \sum_{i, k} \mathbf{1}(E_{\text{sel}}[i, k] == e)$
    /// 2. $O_{\text{all}}[e] = \sum_{j=0}^{e-1} C_{\text{exp}}[j]$
    /// 3. Tokens assigned to expert $e$ are packed in deterministic arrival order:
    ///    $\text{slot}(i, k) = O_{\text{all}}[e] + \text{rank\_within\_expert}(i, e)$
    ///
    /// This eliminates race conditions, atomic write contention, and ensures bitwise
    /// identical accumulation order during reduction across parallel ranks.
    pub fn build(
        selected_experts: &[Vec<usize>],
        num_experts: usize,
    ) -> Result<Self, Error> {
        let num_tokens = selected_experts.len();
        if num_tokens == 0 {
            return Ok(Self {
                num_tokens: 0,
                num_experts,
                top_k: 0,
                total_routed_instances: 0,
                expert_counts: vec![0; num_experts],
                global_offsets: vec![0; num_experts + 1],
                destination_slots: Vec::new(),
                reverse_map: Vec::new(),
            });
        }

        let top_k = selected_experts[0].len();
        let total_routed_instances = num_tokens * top_k;

        // Step 1: Count tokens assigned to each expert
        let mut expert_counts = vec![0usize; num_experts];
        for token_exp in selected_experts {
            if token_exp.len() != top_k {
                return Err(Error::Backend(format!(
                    "DeterministicTokenMap: inconsistent top_k (expected {top_k}, got {})",
                    token_exp.len()
                )));
            }
            for &exp in token_exp {
                if exp >= num_experts {
                    return Err(Error::Backend(format!(
                        "DeterministicTokenMap: expert index {exp} out of bounds (num_experts {num_experts})"
                    )));
                }
                expert_counts[exp] += 1;
            }
        }

        // Step 2: Compute exclusive prefix sums (GlobalOffsets)
        let mut global_offsets = vec![0usize; num_experts + 1];
        let mut running_sum = 0usize;
        for e in 0..num_experts {
            global_offsets[e] = running_sum;
            running_sum += expert_counts[e];
        }
        global_offsets[num_experts] = running_sum;

        // Step 3: Assign conflict-free, deterministic destination slots
        let mut local_cursors = global_offsets[0..num_experts].to_vec();
        let mut destination_slots = vec![0usize; total_routed_instances];
        let mut reverse_map = vec![(0usize, 0usize, 0usize); total_routed_instances];

        for (token_idx, token_exp) in selected_experts.iter().enumerate() {
            for (k_idx, &exp) in token_exp.iter().enumerate() {
                let slot = local_cursors[exp];
                local_cursors[exp] += 1;

                let instance_idx = token_idx * top_k + k_idx;
                destination_slots[instance_idx] = slot;
                reverse_map[slot] = (token_idx, exp, k_idx);
            }
        }

        Ok(Self {
            num_tokens,
            num_experts,
            top_k,
            total_routed_instances,
            expert_counts,
            global_offsets,
            destination_slots,
            reverse_map,
        })
    }

    /// Pack flattened token activations $[N_{\text{tok}}, D]$ into continuous expert buffers
    /// $[N_{\text{routed}}, D]$ according to the deterministic mapping.
    pub fn pack_activations(
        &self,
        flat_activations: &[f32],
        hidden_dim: usize,
        out_packed: &mut [f32],
    ) -> Result<(), Error> {
        if flat_activations.len() != self.num_tokens * hidden_dim {
            return Err(Error::Backend(format!(
                "pack_activations: input size mismatch (expected {}, got {})",
                self.num_tokens * hidden_dim,
                flat_activations.len()
            )));
        }
        if out_packed.len() != self.total_routed_instances * hidden_dim {
            return Err(Error::Backend(format!(
                "pack_activations: output buffer size mismatch (expected {}, got {})",
                self.total_routed_instances * hidden_dim,
                out_packed.len()
            )));
        }

        for token_idx in 0..self.num_tokens {
            let src_start = token_idx * hidden_dim;
            let src_slice = &flat_activations[src_start..src_start + hidden_dim];

            for k in 0..self.top_k {
                let instance_idx = token_idx * self.top_k + k;
                let dest_slot = self.destination_slots[instance_idx];
                let dest_start = dest_slot * hidden_dim;
                out_packed[dest_start..dest_start + hidden_dim].copy_from_slice(src_slice);
            }
        }
        Ok(())
    }

    /// Combine expert outputs $[N_{\text{routed}}, D]$ back to $[N_{\text{tok}}, D]$ using
    /// deterministic summation ordering, applying `routed_scaling_factor`.
    pub fn combine_expert_outputs(
        &self,
        packed_outputs: &[f32],
        weights: &[Vec<f32>],
        hidden_dim: usize,
        routed_scaling_factor: f32,
        out_combined: &mut [f32],
    ) -> Result<(), Error> {
        if out_combined.len() != self.num_tokens * hidden_dim {
            return Err(Error::Backend(format!(
                "combine_expert_outputs: output buffer size mismatch (expected {}, got {})",
                self.num_tokens * hidden_dim,
                out_combined.len()
            )));
        }

        out_combined.fill(0.0);

        // Accumulate in strict token-index and top-k ascending order to guarantee bitwise determinism
        for token_idx in 0..self.num_tokens {
            let out_start = token_idx * hidden_dim;
            let out_slice = &mut out_combined[out_start..out_start + hidden_dim];
            let token_weights = &weights[token_idx];

            for k in 0..self.top_k {
                let weight = token_weights[k] * routed_scaling_factor;
                let instance_idx = token_idx * self.top_k + k;
                let slot = self.destination_slots[instance_idx];
                let src_start = slot * hidden_dim;
                let src_slice = &packed_outputs[src_start..src_start + hidden_dim];

                for d in 0..hidden_dim {
                    out_slice[d] += src_slice[d] * weight;
                }
            }
        }
        Ok(())
    }
}

/// Scoreboard synchronization flags for persistent SM / compute worker coordination.
#[derive(Debug)]
pub struct ScoreboardSync {
    /// Number of tokens arrived per tile.
    pub token_arrivals: Vec<AtomicU32>,
    /// Tile ready flags (1 = tile complete and ready for GroupGEMM).
    pub tile_ready: Vec<AtomicU32>,
    /// Number of tokens per tile.
    pub tile_size: usize,
    /// Total number of tiles across all experts.
    pub num_tiles: usize,
}

impl ScoreboardSync {
    /// Creates a new scoreboard tracker for `num_tiles` of size `tile_size`.
    pub fn new(num_tiles: usize, tile_size: usize) -> Self {
        let mut token_arrivals = Vec::with_capacity(num_tiles);
        let mut tile_ready = Vec::with_capacity(num_tiles);
        for _ in 0..num_tiles {
            token_arrivals.push(AtomicU32::new(0));
            tile_ready.push(AtomicU32::new(0));
        }
        Self {
            token_arrivals,
            tile_ready,
            tile_size,
            num_tiles,
        }
    }

    /// Record a token arrival into a specific tile. If tile fills, sets `tile_ready`.
    #[inline]
    pub fn record_token_arrival(&self, tile_idx: usize) -> bool {
        if tile_idx >= self.num_tiles {
            return false;
        }
        let arrived = self.token_arrivals[tile_idx].fetch_add(1, Ordering::SeqCst) + 1;
        if arrived == self.tile_size as u32 {
            self.tile_ready[tile_idx].store(1, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Check whether a tile is ready for execution.
    #[inline]
    pub fn is_tile_ready(&self, tile_idx: usize) -> bool {
        if tile_idx >= self.num_tiles {
            return false;
        }
        self.tile_ready[tile_idx].load(Ordering::SeqCst) == 1
    }

    /// Reset all scoreboard flags for the next forward pass.
    pub fn reset(&self) {
        for t in &self.token_arrivals {
            t.store(0, Ordering::SeqCst);
        }
        for r in &self.tile_ready {
            r.store(0, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_token_map_prefix_sums() {
        // 4 tokens, top-2 routing, 4 total experts
        // Token 0 -> E0, E2
        // Token 1 -> E1, E2
        // Token 2 -> E0, E3
        // Token 3 -> E2, E3
        let selected = vec![
            vec![0, 2],
            vec![1, 2],
            vec![0, 3],
            vec![2, 3],
        ];

        let map = DeterministicTokenMap::build(&selected, 4).unwrap();

        // Counts: E0: 2, E1: 1, E2: 3, E3: 2 -> Total = 8
        assert_eq!(map.expert_counts, vec![2, 1, 3, 2]);
        // Offsets: E0: 0, E1: 2, E2: 3, E3: 6, End: 8
        assert_eq!(map.global_offsets, vec![0, 2, 3, 6, 8]);

        // Destination slots:
        // Token 0: E0 -> slot 0, E2 -> slot 3
        // Token 1: E1 -> slot 2, E2 -> slot 4
        // Token 2: E0 -> slot 1, E3 -> slot 6
        // Token 3: E2 -> slot 5, E3 -> slot 7
        assert_eq!(map.destination_slots, vec![0, 3, 2, 4, 1, 6, 5, 7]);
    }

    #[test]
    fn test_pack_and_combine_numerical_exactness() {
        let selected = vec![
            vec![0, 1],
            vec![1, 2],
        ];
        let weights = vec![
            vec![0.6, 0.4],
            vec![0.7, 0.3],
        ];
        let map = DeterministicTokenMap::build(&selected, 3).unwrap();

        let hidden_dim = 2;
        let activations = vec![
            1.0, 2.0, // Token 0
            3.0, 4.0, // Token 1
        ];

        let mut packed = vec![0.0f32; 4 * hidden_dim];
        map.pack_activations(&activations, hidden_dim, &mut packed).unwrap();

        // Simulated expert GEMMs:
        // E0: doubles inputs (*2.0)
        // E1: triples inputs (*3.0)
        // E2: quadruples inputs (*4.0)
        let mut gemm_out = packed.clone();
        for slot in 0..4 {
            let (_, exp, _) = map.reverse_map[slot];
            let multiplier = (exp + 2) as f32;
            for d in 0..hidden_dim {
                gemm_out[slot * hidden_dim + d] *= multiplier;
            }
        }

        let mut combined = vec![0.0f32; 2 * hidden_dim];
        map.combine_expert_outputs(&gemm_out, &weights, hidden_dim, 1.0, &mut combined).unwrap();

        // Expected Token 0:
        // E0: [1.0, 2.0] * 2.0 = [2.0, 4.0] * 0.6 = [1.2, 2.4]
        // E1: [1.0, 2.0] * 3.0 = [3.0, 6.0] * 0.4 = [1.2, 2.4]
        // Total = [2.4, 4.8]
        assert!((combined[0] - 2.4).abs() < 1e-5);
        assert!((combined[1] - 4.8).abs() < 1e-5);

        // Expected Token 1:
        // E1: [3.0, 4.0] * 3.0 = [9.0, 12.0] * 0.7 = [6.3, 8.4]
        // E2: [3.0, 4.0] * 4.0 = [12.0, 16.0] * 0.3 = [3.6, 4.8]
        // Total = [9.9, 13.2]
        assert!((combined[2] - 9.9).abs() < 1e-5);
        assert!((combined[3] - 13.2).abs() < 1e-5);
    }

    #[test]
    fn test_scoreboard_synchronization_lifecycle() {
        let scoreboard = ScoreboardSync::new(2, 4); // 2 tiles, 4 tokens each

        assert!(!scoreboard.is_tile_ready(0));
        assert!(!scoreboard.record_token_arrival(0)); // 1
        assert!(!scoreboard.record_token_arrival(0)); // 2
        assert!(!scoreboard.record_token_arrival(0)); // 3
        assert!(scoreboard.record_token_arrival(0));  // 4 -> ready!

        assert!(scoreboard.is_tile_ready(0));
        assert!(!scoreboard.is_tile_ready(1));

        scoreboard.reset();
        assert!(!scoreboard.is_tile_ready(0));
    }
}
