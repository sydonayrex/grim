//! Expert Parallel Load Balancing (EPLB) for ROCm MoE Inference.
//!
//! Implements greedy Longest Processing Time (LPT) bin packing and dynamic
//! expert replication across multi-GPU ranks to balance skewed routing workloads.

#[derive(Debug, Clone)]
pub struct EplbPackingPlan {
    /// Mapping: expert_id -> assigned rank
    pub expert_to_rank: Vec<usize>,
    /// Accumulated total load per rank
    pub rank_loads: Vec<f32>,
    /// Experts replicated across multiple ranks for high throughput
    pub replicated_experts: Vec<(usize, Vec<usize>)>,
}

impl EplbPackingPlan {
    pub fn max_load(&self) -> f32 {
        self.rank_loads.iter().cloned().fold(0.0f32, f32::max)
    }

    pub fn min_load(&self) -> f32 {
        self.rank_loads
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }

    pub fn imbalance_ratio(&self) -> f32 {
        let mean = self.rank_loads.iter().sum::<f32>() / self.rank_loads.len().max(1) as f32;
        if mean > 0.0f32 {
            self.max_load() / mean
        } else {
            1.0f32
        }
    }
}

pub struct EplbRouter;

impl EplbRouter {
    /// Computes balanced expert placement across `num_ranks` using greedy LPT bin packing.
    pub fn balance_experts(
        expert_frequencies: &[f32],
        num_ranks: usize,
        replication_slots: usize,
    ) -> EplbPackingPlan {
        let num_experts = expert_frequencies.len();
        let mut indexed_loads: Vec<(usize, f32)> = expert_frequencies
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w))
            .collect();

        // Sort descending by load (LPT)
        indexed_loads.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut rank_loads = vec![0.0f32; num_ranks];
        let mut expert_to_rank = vec![0usize; num_experts];

        // Greedy LPT assignment
        for (expert_id, load) in &indexed_loads {
            // Find rank with minimum accumulated load
            let min_rank = rank_loads
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            expert_to_rank[*expert_id] = min_rank;
            rank_loads[min_rank] += load;
        }

        // Replicate top hot experts into underutilized ranks
        let mut replicated_experts = Vec::new();
        if replication_slots > 0 && num_ranks > 1 {
            for &(hot_expert_id, _hot_load) in &indexed_loads[..replication_slots.min(num_experts)]
            {
                let primary_rank = expert_to_rank[hot_expert_id];

                // Pick secondary rank with lowest load other than primary
                let secondary_rank = rank_loads
                    .iter()
                    .enumerate()
                    .filter(|(r, _)| *r != primary_rank)
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(r, _)| r)
                    .unwrap_or((primary_rank + 1) % num_ranks);

                replicated_experts.push((hot_expert_id, vec![primary_rank, secondary_rank]));
            }
        }

        EplbPackingPlan {
            expert_to_rank,
            rank_loads,
            replicated_experts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eplb_greedy_lpt_packing() {
        // 8 experts with skewed frequency distribution
        let frequencies = vec![100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0, 5.0];
        let num_ranks = 4;

        let plan = EplbRouter::balance_experts(&frequencies, num_ranks, 2);
        assert_eq!(plan.expert_to_rank.len(), 8);
        assert_eq!(plan.rank_loads.len(), 4);

        // Sum of rank loads must equal sum of frequencies
        let total_freq: f32 = frequencies.iter().sum();
        let total_packed: f32 = plan.rank_loads.iter().sum();
        assert!((total_freq - total_packed).abs() < 1e-4);

        // Imbalance ratio should be well controlled (< 1.35)
        assert!(plan.imbalance_ratio() < 1.35);

        // 2 hot experts replicated
        assert_eq!(plan.replicated_experts.len(), 2);
        assert_eq!(plan.replicated_experts[0].0, 0); // Expert 0 (load 100)
    }
}
