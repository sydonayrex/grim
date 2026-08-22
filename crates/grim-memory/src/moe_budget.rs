//! MoE resident-set HBM budget (`rocm_kernel_plan.md` WI-C).
//!
//! DynaExq-style budget-feasible accounting for the per-expert resident
//! set: hot experts stay fp16-resident, cold experts demote to int8 (via
//! the existing `q*k_gemm` dequant path) to fit the HBM envelope. This
//! module is the *budget envelope* half of `PlanBuilder` (which lives in
//! `grim-nn::moe`); it tracks current residency and answers "can we
//! promote expert E to fp16 without breaching the envelope?"
//!
//! Pure host logic, unit-testable without a GPU (G-C1's budget-kept check).

use std::collections::HashSet;

use grim_core::error::{Error, Result};

/// Per-expert precision tier in the resident set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentTier {
    /// Hot: full fp16 weights resident in HBM.
    Fp16,
    /// Cold: int8/quantized fallback (dequantized on load by `q*k_gemm`).
    Int8,
    /// Not resident — would need prefetch from host/NVMe on dispatch.
    Off,
}

/// HBM envelope tracker for the MoE expert resident set. Knows the
/// per-tier byte cost of one expert and the total envelope; answers
/// promotion/demotion queries without touching the device.
pub struct MoeResidentBudget {
    num_experts: usize,
    bytes_fp16: usize,
    bytes_int8: usize,
    hbm_envelope_bytes: usize,
    /// Current tier of each expert.
    tiers: Vec<ResidentTier>,
}

impl MoeResidentBudget {
    /// New budget over `num_experts`, with per-expert byte costs and the
    /// total HBM envelope. All experts start `Off`.
    pub fn new(
        num_experts: usize,
        bytes_fp16: usize,
        bytes_int8: usize,
        hbm_envelope_bytes: usize,
    ) -> Self {
        Self {
            num_experts,
            bytes_fp16,
            bytes_int8,
            hbm_envelope_bytes,
            tiers: vec![ResidentTier::Off; num_experts],
        }
    }

    /// Bytes currently consumed by the resident set.
    pub fn used_bytes(&self) -> usize {
        self.tiers
            .iter()
            .map(|t| match t {
                ResidentTier::Fp16 => self.bytes_fp16,
                ResidentTier::Int8 => self.bytes_int8,
                ResidentTier::Off => 0,
            })
            .sum()
    }

    /// Bytes remaining in the HBM envelope.
    pub fn remaining_bytes(&self) -> usize {
        self.hbm_envelope_bytes.saturating_sub(self.used_bytes())
    }

    /// True if promoting expert `e` to `tier` would stay within the envelope.
    /// Accounts for freeing the expert's current tier first.
    pub fn can_promote(&self, e: usize, tier: ResidentTier) -> bool {
        if e >= self.num_experts {
            return false;
        }
        let current_cost = match self.tiers[e] {
            ResidentTier::Fp16 => self.bytes_fp16,
            ResidentTier::Int8 => self.bytes_int8,
            ResidentTier::Off => 0,
        };
        let new_cost = match tier {
            ResidentTier::Fp16 => self.bytes_fp16,
            ResidentTier::Int8 => self.bytes_int8,
            ResidentTier::Off => 0,
        };
        // Free the current tier, then check the envelope.
        let freed = self.used_bytes().saturating_sub(current_cost);
        freed + new_cost <= self.hbm_envelope_bytes
    }

    /// Promote expert `e` to `tier`. Returns `Err` if the promotion would
    /// breach the envelope — the caller must demote another expert first
    /// (the DynaExq top-n eviction policy lives in `PlanBuilder`).
    pub fn promote(&mut self, e: usize, tier: ResidentTier) -> Result<()> {
        if e >= self.num_experts {
            return Err(Error::Config(format!(
                "MoeResidentBudget::promote: expert {e} out of range ({})",
                self.num_experts
            )));
        }
        if !self.can_promote(e, tier) {
            return Err(Error::Config(format!(
                "MoeResidentBudget::promote: expert {e} to {tier:?} would breach envelope (used {}, envelope {})",
                self.used_bytes(),
                self.hbm_envelope_bytes
            )));
        }
        self.tiers[e] = tier;
        Ok(())
    }

    /// Demote expert `e` one tier (Fp16→Int8→Off). Never errors.
    pub fn demote(&mut self, e: usize) {
        if e >= self.num_experts {
            return;
        }
        self.tiers[e] = match self.tiers[e] {
            ResidentTier::Fp16 => ResidentTier::Int8,
            ResidentTier::Int8 => ResidentTier::Off,
            ResidentTier::Off => ResidentTier::Off,
        };
    }

    /// Current tier of expert `e`.
    pub fn tier(&self, e: usize) -> ResidentTier {
        self.tiers.get(e).copied().unwrap_or(ResidentTier::Off)
    }

    /// The set of fp16-resident (hot) experts — the resident hot set the
    /// WI-B selector and WI-C predictor both feed.
    pub fn hot_set(&self) -> HashSet<usize> {
        self.tiers
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == ResidentTier::Fp16)
            .map(|(e, _)| e)
            .collect()
    }

    /// Number of experts at each tier.
    pub fn tier_counts(&self) -> (usize, usize, usize) {
        let (mut fp16, mut int8, mut off) = (0, 0, 0);
        for t in &self.tiers {
            match t {
                ResidentTier::Fp16 => fp16 += 1,
                ResidentTier::Int8 => int8 += 1,
                ResidentTier::Off => off += 1,
            }
        }
        (fp16, int8, off)
    }
}

/// Dynamic GPU memory allocator managing runtime repartitioning between KV cache and MoE expert slots.
///
/// On edge/consumer devices, total VRAM fluctuates and KV cache grows over multi-turn agent sessions.
/// This structure dynamically adjusts the envelope at scheduler step boundaries (safe points)
/// without restarting the engine.
#[derive(Debug, Clone)]
pub struct ElasticMoEAllocation {
    /// Total VRAM envelope in bytes dedicated to serving.
    pub total_vram_bytes: usize,
    /// Bytes allocated to KV cache pages.
    pub kv_budget_bytes: usize,
    /// Bytes allocated to the GPU MoE expert cache.
    pub expert_budget_bytes: usize,
    /// Size in bytes of one complete cached expert slot.
    pub slot_size_bytes: usize,
    /// Current number of active GPU expert slots.
    pub max_expert_slots: usize,
}

impl ElasticMoEAllocation {
    /// Create a new elastic allocation given total VRAM and initial KV/expert split.
    ///
    /// # Contract
    /// `slot_size_bytes` must be > 0. Total of `kv_budget_bytes + expert_budget_bytes`
    /// must not exceed `total_vram_bytes`.
    pub fn new(
        total_vram_bytes: usize,
        kv_budget_bytes: usize,
        expert_budget_bytes: usize,
        slot_size_bytes: usize,
    ) -> Result<Self> {
        if kv_budget_bytes + expert_budget_bytes > total_vram_bytes {
            return Err(Error::Config(format!(
                "ElasticMoEAllocation: KV ({kv_budget_bytes}) + Expert ({expert_budget_bytes}) exceeds total VRAM ({total_vram_bytes})"
            )));
        }
        assert!(slot_size_bytes > 0, "slot_size_bytes must be > 0");
        let max_expert_slots = expert_budget_bytes / slot_size_bytes;
        Ok(Self {
            total_vram_bytes,
            kv_budget_bytes,
            expert_budget_bytes,
            slot_size_bytes,
            max_expert_slots,
        })
    }

    /// Rebalance the split between KV cache and expert cache at a scheduler safe point.
    ///
    /// # Contract
    /// Dynamically shifts capacity. Returns the new number of available expert slots.
    pub fn rebalance(
        &mut self,
        new_kv_budget_bytes: usize,
        new_expert_budget_bytes: usize,
    ) -> Result<usize> {
        if new_kv_budget_bytes + new_expert_budget_bytes > self.total_vram_bytes {
            return Err(Error::Config(format!(
                "ElasticMoEAllocation::rebalance: request exceeds total VRAM ({})",
                self.total_vram_bytes
            )));
        }
        self.kv_budget_bytes = new_kv_budget_bytes;
        self.expert_budget_bytes = new_expert_budget_bytes;
        self.max_expert_slots = self.expert_budget_bytes / self.slot_size_bytes;
        Ok(self.max_expert_slots)
    }
}

/// GPU-resident LRU expert slot residency tracker.
///
/// Tracks the mapping of logical `(layer_idx, expert_idx)` to physical GPU cache slot indices.
#[derive(Debug, Clone)]
pub struct LruResidencyTracker {
    /// Number of physical slots available in the GPU expert cache.
    capacity: usize,
    /// Physical slot index -> optional `(layer_idx, expert_idx, access_timestamp)`.
    slots: Vec<Option<(usize, usize, u64)>>,
    /// Reverse lookup: `(layer_idx, expert_idx)` -> physical slot index.
    key_to_slot: std::collections::HashMap<(usize, usize), usize>,
    /// Monotonic logical clock for access recency.
    clock: u64,
}

impl LruResidencyTracker {
    /// Create a new tracker with `capacity` physical slots.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            slots: vec![None; capacity],
            key_to_slot: std::collections::HashMap::new(),
            clock: 0,
        }
    }

    /// Check if `(layer_idx, expert_idx)` is currently resident in the GPU cache.
    /// If hit, refreshes recency clock and returns slot index.
    pub fn lookup(&mut self, layer: usize, expert: usize) -> Option<usize> {
        let &slot = self.key_to_slot.get(&(layer, expert))?;
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = &mut self.slots[slot] {
            entry.2 = self.clock;
        }
        Some(slot)
    }

    /// Allocate or evict an LRU slot to admit `(layer_idx, expert_idx)`.
    ///
    /// # Contract
    /// If free slot exists, uses it. Otherwise evicts the least recently used slot.
    /// Returns `(allocated_slot_idx, evicted_expert_if_any)`.
    pub fn admit(&mut self, layer: usize, expert: usize) -> (usize, Option<(usize, usize)>) {
        if let Some(slot) = self.lookup(layer, expert) {
            return (slot, None);
        }

        self.clock = self.clock.wrapping_add(1);

        // Find empty slot if available
        for slot in 0..self.capacity {
            if self.slots[slot].is_none() {
                self.slots[slot] = Some((layer, expert, self.clock));
                self.key_to_slot.insert((layer, expert), slot);
                return (slot, None);
            }
        }

        // Evict LRU victim
        let mut oldest_slot = 0;
        let mut oldest_time = u64::MAX;
        for (slot, entry) in self.slots.iter().enumerate() {
            if let Some((_, _, time)) = entry {
                if *time < oldest_time {
                    oldest_time = *time;
                    oldest_slot = slot;
                }
            }
        }

        let evicted = self.slots[oldest_slot].take().map(|(l, e, _)| (l, e));
        if let Some((old_l, old_e)) = evicted {
            self.key_to_slot.remove(&(old_l, old_e));
        }

        self.slots[oldest_slot] = Some((layer, expert, self.clock));
        self.key_to_slot.insert((layer, expert), oldest_slot);

        (oldest_slot, evicted)
    }

    /// Resize slot capacity dynamically during runtime rebalancing.
    pub fn resize(&mut self, new_capacity: usize) {
        if new_capacity < self.capacity {
            // Reclaim slots starting from cold end
            while self.slots.len() > new_capacity {
                if let Some(Some((l, e, _))) = self.slots.pop() {
                    self.key_to_slot.remove(&(l, e));
                }
            }
        } else if new_capacity > self.capacity {
            self.slots.resize(new_capacity, None);
        }
        self.capacity = new_capacity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G-C1: a fresh budget has everything Off and uses zero bytes.
    #[test]
    fn fresh_budget_is_all_off() {
        let b = MoeResidentBudget::new(8, 1000, 400, 4000);
        assert_eq!(b.used_bytes(), 0);
        assert_eq!(b.remaining_bytes(), 4000);
        let (fp16, int8, off) = b.tier_counts();
        assert_eq!((fp16, int8, off), (0, 0, 8));
    }

    /// G-C1: promotion respects the envelope — promote until full, then
    /// further promotions error.
    #[test]
    fn promotion_respects_envelope() {
        let mut b = MoeResidentBudget::new(8, 1000, 400, 3000);
        // 3 fp16 promotions = 3000 bytes → envelope exactly hit.
        assert!(b.promote(0, ResidentTier::Fp16).is_ok());
        assert!(b.promote(1, ResidentTier::Fp16).is_ok());
        assert!(b.promote(2, ResidentTier::Fp16).is_ok());
        assert_eq!(b.used_bytes(), 3000);
        // A 4th fp16 promotion must error.
        assert!(b.promote(3, ResidentTier::Fp16).is_err());
        // But an int8 promotion (after demoting one) fits.
        b.demote(2); // expert 2 → Int8 (frees 600)
        assert_eq!(b.used_bytes(), 2400);
        assert!(b.promote(3, ResidentTier::Int8).is_ok());
    }

    /// G-C1: demotion walks Fp16→Int8→Off and frees bytes at each step.
    #[test]
    fn demotion_walks_tiers_and_frees_bytes() {
        let mut b = MoeResidentBudget::new(2, 1000, 400, 4000);
        b.promote(0, ResidentTier::Fp16).unwrap();
        assert_eq!(b.used_bytes(), 1000);
        b.demote(0);
        assert_eq!(b.tier(0), ResidentTier::Int8);
        assert_eq!(b.used_bytes(), 400);
        b.demote(0);
        assert_eq!(b.tier(0), ResidentTier::Off);
        assert_eq!(b.used_bytes(), 0);
        // Demoting Off is a no-op.
        b.demote(0);
        assert_eq!(b.tier(0), ResidentTier::Off);
    }

    /// G-C1: the hot set tracks fp16-resident experts.
    #[test]
    fn hot_set_tracks_fp16_experts() {
        let mut b = MoeResidentBudget::new(4, 1000, 400, 4000);
        b.promote(0, ResidentTier::Fp16).unwrap();
        b.promote(2, ResidentTier::Fp16).unwrap();
        b.promote(1, ResidentTier::Int8).unwrap();
        let hot = b.hot_set();
        assert_eq!(hot, [0, 2].into_iter().collect::<HashSet<_>>());
    }

    /// G-C1: out-of-range expert promotion is rejected.
    #[test]
    fn out_of_range_promotion_rejected() {
        let mut b = MoeResidentBudget::new(4, 1000, 400, 4000);
        assert!(b.promote(99, ResidentTier::Fp16).is_err());
        assert!(!b.can_promote(99, ResidentTier::Fp16));
    }

    #[test]
    fn test_elastic_moe_allocation_rebalance() {
        let mut alloc = ElasticMoEAllocation::new(16_000, 6_000, 10_000, 1_000).unwrap();
        assert_eq!(alloc.max_expert_slots, 10);

        // Rebalance to give more to KV
        let new_slots = alloc.rebalance(10_000, 6_000).unwrap();
        assert_eq!(new_slots, 6);
        assert_eq!(alloc.max_expert_slots, 6);

        // Exceeding total VRAM errors
        assert!(alloc.rebalance(12_000, 6_000).is_err());
    }

    #[test]
    fn test_lru_residency_tracker() {
        let mut tracker = LruResidencyTracker::new(2);
        // Fill slots
        let (s0, ev0) = tracker.admit(0, 5);
        assert_eq!(s0, 0);
        assert_eq!(ev0, None);

        let (s1, ev1) = tracker.admit(0, 12);
        assert_eq!(s1, 1);
        assert_eq!(ev1, None);

        // Hit (0, 5) -> refreshes recency
        assert_eq!(tracker.lookup(0, 5), Some(0));

        // Admit 3rd item -> evicts (0, 12) from slot 1
        let (s2, ev2) = tracker.admit(1, 3);
        assert_eq!(s2, 1);
        assert_eq!(ev2, Some((0, 12)));

        // Verify residency
        assert_eq!(tracker.lookup(0, 5), Some(0));
        assert_eq!(tracker.lookup(0, 12), None);
        assert_eq!(tracker.lookup(1, 3), Some(1));
    }
}
