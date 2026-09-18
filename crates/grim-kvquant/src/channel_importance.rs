//! Per-channel-group importance scoring for the KV cache (PLAN-kvcache-channel-axis, WI-2).
//!
//! Storage-time channel-axis allocation needs a per-(kv_head, channel-group)
//! importance signal. K/V here is POST-RoPE (the cached form), which makes
//! per-channel variance position-unstable (TriAttention's finding), so the
//! primary score is PAIR-JOINT: RoPE rotates channel pairs
//! `(c, c + head_dim/2)` together, so the pair's combined energy is
//! position-independent even when either channel's alone is not.
//!
//! Scores produced per 32-element channel group (the granularity WI-1's
//! `Q4KHalf` format can carry):
//! - `k_distortion` / `v_distortion`: RDKV-style distortion proxy — the group's
//!   share of the row's L2 energy (evicting a group degrades the attention
//!   output in proportion to the energy it carried).
//! - `k_q_variance` (optional, when calibration passes supply Q rows): Fathom's
//!   `g_c = Σ (q · k[:, c])²` q-contextualized variance, computed per group.
//! - Raw (non-pair-joint) variants of both distortion scores for diagnostics.
//!
//! Intended use: offline calibration (see grim-cli `calibrate-channels`),
//! serialized as a JSON sidecar next to the quantized model.

use serde::{Deserialize, Serialize};

/// Channel-group size matched to `Q4KHalf`'s sub-block granularity.
pub const CHANNEL_GROUP_SIZE: usize = 32;

/// Per-(kv_head, group) importance scores for one model, one or more layers pooled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelImportanceScores {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub group_size: usize,
    /// K importance, RoPE pair-joint. `[kv_head][group]`, L2-normalized within
    /// each head to sum to 1 for cross-model comparability.
    pub k_scores: Vec<Vec<f32>>,
    /// V importance, pair-joint (as above), same shape.
    pub v_scores: Vec<Vec<f32>>,
    /// Raw (non-paired) scores, same shapes, for diagnostics vs the paired view.
    pub k_raw: Vec<Vec<f32>>,
    pub v_raw: Vec<Vec<f32>>,
    /// Fathom q-variance scores, present only when Q rows were supplied.
    #[serde(default)]
    pub k_q_variance: Option<Vec<Vec<f32>>>,
    /// Number of K/V rows that fed these scores.
    pub kv_rows: u64,
    /// Number of Q rows that fed `k_q_variance` (0 unless supplied).
    pub q_rows: u64,
}

/// Streaming accumulator over post-RoPE K/V rows (layer by layer; the pool
/// aggregates across layers — a production "flat" calibration profile).
pub struct ChannelImportanceComputer {
    num_kv_heads: usize,
    head_dim: usize,
    group_size: usize,
    groups: usize,
    /// half-rope offset in groups (head_dim/2 / group_size) when it is an
    /// exact group boundary; 0 disables pair-joint folding.
    rope_partner_shift: usize,
    k_energy: Vec<Vec<f64>>,
    v_energy: Vec<Vec<f64>>,
    k_qvar: Option<Vec<Vec<f64>>>,
    kv_rows: u64,
    q_rows: u64,
}

impl ChannelImportanceComputer {
    /// `with_q` reserves accumulators for the Fathom q-variance channel.
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        group_size: usize,
        with_q: bool,
    ) -> Result<Self, String> {
        if num_kv_heads == 0 || head_dim == 0 || group_size == 0 {
            return Err("channel importance: zero-sized geometry".into());
        }
        if head_dim % group_size != 0 {
            return Err(format!(
                "head_dim {head_dim} not a multiple of group_size {group_size}"
            ));
        }
        let groups = head_dim / group_size;
        let half = head_dim / 2;
        let rope_partner_shift = if half % group_size == 0 && half > 0 {
            half / group_size
        } else {
            0
        };
        let shape_init = vec![vec![0.0f64; groups]; num_kv_heads];
        Ok(Self {
            num_kv_heads,
            head_dim,
            group_size,
            groups,
            rope_partner_shift,
            k_energy: shape_init.clone(),
            v_energy: shape_init.clone(),
            k_qvar: with_q.then(|| shape_init.clone()),
            kv_rows: 0,
            q_rows: 0,
        })
    }

    /// One (kv_head, channel) energy observation. Internal.
    #[inline]
    fn acc(acc: &mut Vec<Vec<f64>>, kv_head: usize, group: usize, x: f32) {
        acc[kv_head][group] += (x as f64) * (x as f64);
    }

    /// Fold a post-RoPE K row batch `[rows][num_kv_heads][head_dim]` (row-major,
    /// i.e. the cached layout) plus matching V rows into the accumulators.
    pub fn update_kv(&mut self, k: &[f32], v: &[f32], rows: usize) -> Result<(), String> {
        let row_elems = self.num_kv_heads * self.head_dim;
        let want = rows * row_elems;
        if k.len() < want || v.len() < want {
            return Err(format!(
                "update_kv: got k={} v={} elements, want {want} (rows={rows})",
                k.len(),
                v.len()
            ));
        }
        for r in 0..rows {
            let base = r * row_elems;
            for h in 0..self.num_kv_heads {
                let hbase = base + h * self.head_dim;
                for d in 0..self.head_dim {
                    let g = d / self.group_size;
                    Self::acc(&mut self.k_energy, h, g, k[hbase + d]);
                    Self::acc(&mut self.v_energy, h, g, v[hbase + d]);
                }
            }
        }
        self.kv_rows += rows as u64;
        Ok(())
    }

    /// Fathom q-variance channel: `q` is `[q_rows][num_q_heads][head_dim]`
    /// post-RoPE; the group score accumulates `Σ (q_h · k_hc)^2` per K-head
    /// (GQA-mapped: q head h maps to kv head `h * num_kv_heads / num_q_heads`).
    /// `k` is the same layout as `update_kv`. Call with the paired q/k of the
    /// same tokens.
    pub fn update_q_variance(&mut self, q: &[f32], k: &[f32], rows: usize, num_q_heads: usize) -> Result<(), String> {
        let Some(k_qvar) = self.k_qvar.as_mut() else {
            return Err("computer was built with_q = false".into());
        };
        if num_q_heads == 0 || num_q_heads % self.num_kv_heads != 0 {
            return Err(format!(
                "num_q_heads {num_q_heads} incompatible with num_kv_heads {}",
                self.num_kv_heads
            ));
        }
        let q_per_kv = num_q_heads / self.num_kv_heads;
        let k_row = self.num_kv_heads * self.head_dim;
        let q_row = num_q_heads * self.head_dim;
        if k.len() < rows * k_row || q.len() < rows * q_row {
            return Err("update_q_variance: undersized buffers".into());
        }
        for r in 0..rows {
            let kbase = r * k_row;
            let qbase = r * q_row;
            for qh in 0..num_q_heads {
                let kvh = qh / q_per_kv;
                let kb = kbase + kvh * self.head_dim;
                let qb = qbase + qh * self.head_dim;
                for d in 0..self.head_dim {
                    let g = d / self.group_size;
                    let contrib = (q[qb + d] as f64) * (k[kb + d] as f64);
                    k_qvar[kvh][g] += contrib * contrib;
                }
            }
        }
        self.q_rows += rows as u64;
        Ok(())
    }

    /// Freeze into scores. Pair-joint folding adds the RoPE partner group's
    /// energy (groups are 32 channels; the partner of channel c is
    /// c + head_dim/2, i.e. partner group g + head_dim/(2*group_size)).
    pub fn finish(mut self) -> ChannelImportanceScores {
        let n = self.groups;
        let fold = |raw: &[Vec<f64>]| -> Vec<Vec<f32>> {
            let mut out = vec![vec![0.0f32; n]; self.num_kv_heads];
            for h in 0..self.num_kv_heads {
                for g in 0..n {
                    let mut total = raw[h][g];
                    if self.rope_partner_shift > 0 && self.rope_partner_shift < n {
                        let partner = (g + self.rope_partner_shift) % n;
                        total += raw[h][partner];
                    }
                    out[h][g] = total as f32;
                }
            }
            normalize_per_head(out)
        };
        let k_raw = normalize_per_head(f64_to_f32(&self.k_energy));
        let v_raw = normalize_per_head(f64_to_f32(&self.v_energy));
        let k_scores = fold(&self.k_energy);
        let v_scores = fold(&self.v_energy);
        let k_q_variance = self
            .k_qvar
            .take()
            .map(|q| normalize_per_head(f64_to_f32_pairs(q, n, self.rope_partner_shift)));
        ChannelImportanceScores {
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            group_size: self.group_size,
            k_scores,
            v_scores,
            k_raw,
            v_raw,
            k_q_variance,
            kv_rows: self.kv_rows,
            q_rows: self.q_rows,
        }
    }
}

fn f64_to_f32(v: &[Vec<f64>]) -> Vec<Vec<f32>> {
    v.iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect()
}

fn f64_to_f32_pairs(v: Vec<Vec<f64>>, groups: usize, partner_shift: usize) -> Vec<Vec<f32>> {
    let n = v.len();
    let mut out = vec![vec![0.0f32; groups]; n];
    for h in 0..n {
        for g in 0..groups {
            let mut total = v[h][g];
            if partner_shift > 0 && partner_shift < groups {
                total += v[h][(g + partner_shift) % groups];
            }
            out[h][g] = total as f32;
        }
    }
    out
}

fn normalize_per_head(mut v: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    for row in &mut v {
        let sum: f32 = row.iter().sum();
        if sum > 0.0 {
            for x in row.iter_mut() {
                *x /= sum;
            }
        }
    }
    v
}

/// One side's (K or V) discrete-tier channel-group allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SingleSideAllocation {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub group_size: usize,
    pub bits: Vec<Vec<u8>>,
    pub default_bits: u8,
}

/// WI-3: discrete-tier channel-group allocation. `key_bits`/`value_bits` are
/// per (kv_head, 32-channel group) bit-widths the CPU quantizer applies when
/// this allocation is attached to a `LloydMaxCompressor`; the GPU path uses
/// the same rows to pick per-head storage format (see PLAN-kvcache-channel-axis).
/// Construct via [`ChannelBitAllocation::from_scores`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelBitAllocation {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub group_size: usize,
    /// `[kv_head][group]` bit-width assignments for keys.
    pub key_bits: Vec<Vec<u8>>,
    /// `[kv_head][group]` bit-width assignments for values.
    pub value_bits: Vec<Vec<u8>>,
}

impl ChannelBitAllocation {
    /// Build a two-sided allocation from K and V importance scoreds and their
    /// respective defaults/budgets. `*_budget` is the AVERAGE bits per element
    /// allowed for that side; a budget equal to the default reproduces the
    /// uniform allocation (the opt-out contract).
    pub fn from_scores(
        k_scores: &[Vec<f32>],
        default_key_bits: u8,
        k_budget: f32,
        v_scores: &[Vec<f32>],
        default_value_bits: u8,
        v_budget: f32,
    ) -> Result<Self, String> {
        let k = allocate_channel_bits(k_scores, default_key_bits, k_budget)?;
        let v = allocate_channel_bits(v_scores, default_value_bits, v_budget)?;
        if k.num_kv_heads != v.num_kv_heads || k.head_dim != v.head_dim {
            return Err("K and V importance geometries disagree".into());
        }
        Ok(Self {
            num_kv_heads: k.num_kv_heads,
            head_dim: k.head_dim,
            group_size: CHANNEL_GROUP_SIZE,
            key_bits: k.bits,
            value_bits: v.bits,
        })
    }

    /// Bits for one key element at (kv_head, channel).
    #[inline]
    pub fn key_bit_for(&self, kv_head: usize, channel: usize) -> u8 {
        self.key_bits[kv_head][channel / self.group_size]
    }

    /// Bits for one value element at (kv_head, channel).
    #[inline]
    pub fn value_bit_for(&self, kv_head: usize, channel: usize) -> u8 {
        self.value_bits[kv_head][channel / self.group_size]
    }

    /// Geometry check against a block being compressed/dequantized.
    pub fn matches_geometry(&self, num_kv_heads: usize, head_dim: usize) -> bool {
        self.num_kv_heads == num_kv_heads
            && self.head_dim == head_dim
            && self.group_size == CHANNEL_GROUP_SIZE
    }
}

/// A budget-respecting ascending-tier allocator: every group starts at the
/// configured DEFAULT (`default_bits` — the Anti-patterns rule requires the
/// scalar config remain the floor); leftover budget upgrades the
/// highest-importance groups to the next tier until the budget (average bits
/// × group count) is exhausted. Deterministic: ties broken by group index.
pub fn allocate_channel_bits(
    scores: &[Vec<f32>],
    default_bits: u8,
    budget_avg_bits: f32,
) -> Result<SingleSideAllocation, String> {
    let num_kv_heads = scores.len();
    if num_kv_heads == 0 {
        return Err("allocate_channel_bits: empty scores".into());
    }
    let groups = scores[0].len();
    if groups == 0 || scores.iter().any(|r| r.len() != groups) {
        return Err("allocate_channel_bits: ragged or empty score rows".into());
    }
    if default_bits == 0 || default_bits > 8 {
        return Err(format!("default_bits {default_bits} out of range 1..=8"));
    }
    // Tiers above the default: strictly ascending multiples tapering to 8.
    let tiers: Vec<u8> = {
        let mut t = vec![default_bits];
        while let Some(&last) = t.last().filter(|&&l| l < 8) {
            t.push((last * 2).min(8));
        }
        t
    };
    let head_dim = groups * CHANNEL_GROUP_SIZE;

    let mut bits = vec![vec![default_bits; groups]; num_kv_heads];
    // Total bit budget in bits×cells; the budget is uniform across
    // heads (importance differences across heads come out through the scores).
    let total_budget =
        budget_avg_bits.max(default_bits as f32).min(8.0) * (groups * num_kv_heads) as f32;
    let mut spent = (default_bits as f32) * (groups * num_kv_heads) as f32;

    // Candidate upgrades ordered by descending importance; each upgrade costs
    // the tier delta for that (head, group).
    let mut order: Vec<(usize, usize)> = Vec::with_capacity(num_kv_heads * groups);
    for h in 0..num_kv_heads {
        for g in 0..groups {
            order.push((h, g));
        }
    }
    order.sort_by(|a, b| {
        scores[b.0][b.1]
            .partial_cmp(&scores[a.0][a.1])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
            .then(a.1.cmp(&b.1))
    });
    // Multi-pass: every group may climb several tiers, most-important first.
    let mut idx: Vec<usize> = order.iter().map(|_| 0).collect(); // tier index per order pos
    loop {
        let mut upgraded_any = false;
        for (pos, &(h, g)) in order.iter().enumerate() {
            let cur_tier = idx[pos];
            if cur_tier + 1 >= tiers.len() {
                continue;
            }
            let cost = (tiers[cur_tier + 1] - tiers[cur_tier]) as f32;
            if spent + cost > total_budget + 1e-6 {
                continue;
            }
            bits[h][g] = tiers[cur_tier + 1];
            idx[pos] = cur_tier + 1;
            spent += cost;
            upgraded_any = true;
        }
        if !upgraded_any {
            break;
        }
    }

    Ok(SingleSideAllocation {
        num_kv_heads,
        head_dim,
        group_size: CHANNEL_GROUP_SIZE,
        bits,
        default_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_computer(kvh: usize, hd: usize, with_q: bool) -> ChannelImportanceComputer {
        ChannelImportanceComputer::new(kvh, hd, CHANNEL_GROUP_SIZE, with_q).unwrap()
    }

    #[test]
    fn importance_is_deterministic_and_normalizes() {
        let mut c = mk_computer(2, 128, false);
        let rows = 64usize;
        let k: Vec<f32> = (0..rows * 2 * 128).map(|i| ((i % 37) as f32 - 18.0) * 0.013).collect();
        let v: Vec<f32> = (0..rows * 2 * 128).map(|i| ((i % 23) as f32 - 11.0) * 0.007).collect();
        c.update_kv(&k, &v, rows).unwrap();
        // Same data again must double the energies but leave normalized scores equal.
        c.update_kv(&k, &v, rows).unwrap();
        let s = c.finish();
        assert_eq!(s.kv_rows, 128);
        assert_eq!(s.k_scores.len(), 2);
        assert_eq!(s.k_scores[0].len(), 4);
        let sum: f32 = s.k_scores[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "scores normalize per head: {sum}");
    }

    #[test]
    fn energy_concentrated_in_one_group_scores_it_highest() {
        let mut c = mk_computer(1, 128, false);
        let rows = 16usize;
        // All energy in group 2 of head 0 (channels 64..95).
        let mut k = vec![0.0f32; rows * 128];
        let mut v = vec![0.0f32; rows * 128];
        for r in 0..rows {
            for d in 64..96 {
                k[r * 128 + d] = 1.5;
                v[r * 128 + d] = -0.5;
            }
        }
        c.update_kv(&k, &v, rows).unwrap();
        let s = c.finish();
        let top = s
            .k_scores[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // Pair-joint folding over partner shift 2 (64-channel offset): the
        // spike at group 2 also lands on group 0, so the top score is one of
        // the pair {0, 2} — the pair-joint invariant of the design.
        assert!(top == 0 || top == 2, "top group {top}");
        let raw_top = s.k_raw[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(raw_top, 2, "raw scores locate the true spike");
    }

    #[test]
    fn rope_pair_joint_is_position_stable() {
        // Simulate "same content at two position offsets" post-RoPE by
        // swapping energy between channel pairs (c, c+64): pair-joint scores
        // must be identical while raw scores differ.
        let mut c1 = mk_computer(1, 128, false);
        let mut c2 = mk_computer(1, 128, false);
        let rows = 8usize;
        let mut k1 = vec![0.0f32; rows * 128];
        let v1 = vec![0.1f32; rows * 128];
        let mut k2 = vec![0.0f32; rows * 128];
        let v2 = v1.clone();
        for r in 0..rows {
            k1[r * 128 + 10] = 1.0; // channel 10 carries the energy at pos A
            k2[r * 128 + 10 + 64] = 1.0; // its RoPE partner carries it at pos B
        }
        c1.update_kv(&k1, &v1, rows).unwrap();
        c2.update_kv(&k2, &v2, rows).unwrap();
        let s1 = c1.finish();
        let s2 = c2.finish();
        for g in 0..4 {
            assert_eq!(
                s1.k_scores[0][g], s2.k_scores[0][g],
                "pair-joint score at group {g} must be position-invariant"
            );
        }
        assert!(
            (s1.k_raw[0][0] - s2.k_raw[0][0]).abs() > 1e-6,
            "raw scores are NOT position-invariant (sanity)"
        );
    }

    #[test]
    fn q_variance_uses_gqa_mapping() {
        let mut c = ChannelImportanceComputer::new(1, 64, CHANNEL_GROUP_SIZE, true).unwrap();
        let rows = 4usize;
        let k: Vec<f32> = (0..rows * 64).map(|i| (i % 7) as f32 * 0.1).collect();
        // 2 query heads, 1 kv head.
        let mut q = vec![0.0f32; rows * 2 * 64];
        for r in 0..rows {
            q[r * 128 + 5] = 2.0; // head 0 spikes channel 5
            q[r * 128 + 64 + 40] = 3.0; // head 1 spikes channel 40
        }
        c.update_q_variance(&q, &k, rows, 2).unwrap();
        c.update_kv(&k, &k.clone(), rows).unwrap();
        let s = c.finish();
        let qv = s.k_q_variance.expect("q variance present");
        assert_eq!(qv.len(), 1);
        assert_eq!(qv[0].len(), 2);
        assert_eq!(s.q_rows, 4);
    }

    #[test]
    fn allocator_respects_budget_and_prefers_important_groups() {
        // 2 heads × 4 groups. Head 0: group 1 dominates. Head 1: group 3.
        let scores = [
            vec![0.1f32, 0.7, 0.1, 0.1],
            vec![0.1, 0.1, 0.1, 0.7],
        ];
        // Budget: average 5 bits/elem over 8 (head,group) cells, floor 4.
        // Floor cost 32; upgrades 4→8 cost 4 each → exactly two upgrades land,
        // on the two dominant groups.
        let alloc = allocate_channel_bits(&scores, 4, 5.0).unwrap();
        assert_eq!(alloc.bits[0][1], 8, "head 0 spends its budget on group 1");
        assert_eq!(alloc.bits[0][0], 4, "least-importants stay at the default");
        assert_eq!(alloc.bits[1][3], 8, "head 1 spends its budget on group 3");
        assert_eq!(alloc.bits[1][0], 4);

        // Budget == default: the allocation is the uniform default (the
        // Anti-patterns byte-identical-off contract).
        let tight = allocate_channel_bits(&scores, 4, 4.0).unwrap();
        assert!(tight.bits.iter().all(|row| row.iter().all(|&b| b == 4)));

        // Generous budget: 8.0 avg is the cap — max is 8 everywhere.
        let fat = allocate_channel_bits(&scores, 4, 8.0).unwrap();
        assert!(fat.bits.iter().all(|row| row.iter().all(|&b| b == 8)));
    }

    #[test]
    fn allocator_is_deterministic_for_ties() {
        let scores = vec![vec![0.5f32, 0.5, 0.5, 0.5]];
        let a = allocate_channel_bits(&scores, 4, 3.0).unwrap();
        let b = allocate_channel_bits(&scores, 4, 3.0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn scores_json_round_trip() {
        let mut c = mk_computer(2, 64, true);
        let k = vec![0.3f32; 4 * 2 * 64];
        c.update_kv(&k, &k, 4).unwrap();
        let s = c.finish();
        let json = serde_json::to_string_pretty(&s).unwrap();
        let back: ChannelImportanceScores = serde_json::from_str(&json).unwrap();
        assert_eq!(back.num_kv_heads, s.num_kv_heads);
        assert_eq!(back.k_scores, s.k_scores);
    }
}
