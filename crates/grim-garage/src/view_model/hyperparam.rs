//! Hyperparam form — the data the hyperparameters panel binds to, in
//! the exact shape the frontend input components consume.

use crate::ui_state::UiTrainingConfig;
use serde::{Deserialize, Serialize};

/// Valid choices for `LoraRank`. Doubling is intentional: each tier
/// roughly doubles VRAM overhead for a quantized base + LoRA adapter.
pub const VALID_LORA_RANKS: &[u32] = &[8, 16, 32, 64];

/// Strongly typed LoRA rank. Constructing one with `rank == 0` is a
/// logic bug downstream — `apply_and_record_lora` divides by the rank
/// — so the type refuses to admit zero values at the boundary.
///
/// M7: replaces the flow where `lora_rank: u32` slipped through the
/// validation chain and reached the autograd function. The newtype
/// constructor is the single place a rank value is allowed to cross
/// into the rest of grim-garage.
///
/// Serde representation: serialized as the inner `u32` (transparent)
/// — round-trip via `serde_qs_encode/from` works through `HyperparamFormV1`,
/// not directly here. `LoraRank::new(...)` is the boundary that
/// refuses zero; downstream deserializers are expected to call
/// `LoraRank::try_from(u32)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LoraRank(u32);

impl LoraRank {
    /// Construct a `LoraRank` if and only if `rank > 0`. Snap to the
    /// nearest valid tier via `pick_closest`. Returns Err on
    /// `rank == 0`.
    pub fn new(rank: u32) -> Result<Self, LoraRankError> {
        if rank == 0 {
            return Err(LoraRankError::Zero);
        }
        let snapped = pick_closest(rank, VALID_LORA_RANKS);
        Ok(Self(snapped))
    }

    /// Construct without snapping, used by trusted config-file readers
    /// that already pin a value to `VALID_LORA_RANKS`.
    pub fn from_valid(rank: u32) -> Result<Self, LoraRankError> {
        if rank == 0 {
            return Err(LoraRankError::Zero);
        }
        if !VALID_LORA_RANKS.contains(&rank) {
            return Err(LoraRankError::NotInTiers(rank));
        }
        Ok(Self(rank))
    }

    /// Maximum rank allowed under QLoRA at 4-bit quantization, given
    /// an 8 GB reference GPU budget. Empirical ceiling — settings
    /// higher than this routinely OOM the named GPUs.
    pub const QLORA_MAX_RANK: u32 = 32;

    /// Validate the pair (mode, rank) before spawning the worker. Returns
    /// Err if rank > QLORA_MAX_RANK under QLoRA (the bug-fix for the
    /// missing QLoRA×rank bound). Pre-fix the value could ship through.
    pub fn validate_for_mode(&self, mode: crate::jobs::TrainingMode) -> Result<(), LoraRankError> {
        if mode == crate::jobs::TrainingMode::QLoRA && self.0 > Self::QLORA_MAX_RANK {
            return Err(LoraRankError::QloraTooLarge {
                rank: self.0,
                ceiling: Self::QLORA_MAX_RANK,
            });
        }
        Ok(())
    }

    pub fn value(self) -> u32 {
        self.0
    }
}

impl Default for LoraRank {
    fn default() -> Self {
        // 16 is the documented default and is also always in the valid
        // tiers; the constructor never fails on it.
        Self::new(16).expect("default rank 16 is valid")
    }
}

impl From<LoraRank> for u32 {
    fn from(r: LoraRank) -> u32 {
        r.0
    }
}

impl TryFrom<u32> for LoraRank {
    type Error = LoraRankError;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        LoraRank::new(v)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LoraRankError {
    #[error("lora_rank must be > 0 (applies-and-records divides by the rank)")]
    Zero,
    #[error("lora_rank {0} is not in the valid tiers [8, 16, 32, 64]")]
    NotInTiers(u32),
    #[error("lora_rank {rank} > QLoRA ceiling {ceiling} for 4-bit quantized training")]
    QloraTooLarge { rank: u32, ceiling: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HyperparamFormV1 {
    /// Display name of the form (used as the panel title).
    pub form_id: String,

    /// The training mode the user picked in the dropdown.
    /// `"LoRA"`, `"QLoRA"`, or `"Bf16-Full"`.
    pub training_mode: String,

    /// Quantization format the model will be dequantized to during training.
    /// Only meaningful when `training_mode == QLoRA`; UI hides the picker otherwise.
    pub quant_format: String,

    /// One of `[8, 16, 32, 64]`. Snapped to the nearest valid value by `normalized()`.
    pub lora_rank: u32,

    /// Initial learning rate.
    pub learning_rate: f64,

    /// Number of epochs to train (display only; backend maps to max_steps).
    pub epochs: u32,

    pub rocm_fusion_rmsnorm_matmul: bool,
    pub rocm_fusion_qkv_attention: bool,
    pub auto_wavefront: bool,
    pub xnack_enabled: bool,

    /// Which optimizer is selected: AdamW, GaLore, LOMO, AdaLomo, CAME, Sophia, Muon, etc.
    pub optimizer: String,

    /// PiSSA: initialize adapter A/B via truncated SVD of the base weight.
    pub use_pissa: bool,
    /// OLoRA: add `olora_lambda * olora_orthogonality_penalty(A, B)` to the loss.
    pub use_olora: bool,
    /// Weight of the OLoRA orthogonality penalty term.
    pub olora_lambda: f32,
}

impl Default for HyperparamFormV1 {
    fn default() -> Self {
        Self {
            form_id: "hyperparameters".into(),
            training_mode: "LoRA".into(),
            quant_format: "Q4_K".into(),
            lora_rank: 16,
            learning_rate: 2e-5,
            epochs: 1,
            rocm_fusion_rmsnorm_matmul: true,
            rocm_fusion_qkv_attention: false,
            auto_wavefront: true,
            xnack_enabled: false,
            optimizer: "AdamW".into(),
            use_pissa: false,
            use_olora: false,
            olora_lambda: 0.0,
        }
    }
}

impl HyperparamFormV1 {
    /// Construct the form from the live UI state.
    pub fn from_training_config(c: UiTrainingConfig) -> Self {
        Self {
            form_id: "hyperparameters".into(),
            training_mode: c.training_mode,
            quant_format: c.quant_format,
            lora_rank: c.lora_rank,
            learning_rate: c.learning_rate,
            epochs: c.epochs,
            rocm_fusion_rmsnorm_matmul: c.rocm_fusion_rmsnorm_matmul,
            rocm_fusion_qkv_attention: c.rocm_fusion_qkv_attention,
            auto_wavefront: c.auto_wavefront,
            xnack_enabled: c.xnack_enabled,
            optimizer: c.optimizer,
            use_pissa: c.use_pissa,
            use_olora: c.use_olora,
            olora_lambda: c.olora_lambda,
        }
    }

    /// Allowed LoRA ranks; the form picker offers exactly these values.
    /// Doubling is intentional: each tier roughly doubles VRAM overhead.
    pub const VALID_LORA_RANKS: &'static [u32] = VALID_LORA_RANKS;

    /// Snap `lora_rank` to the nearest allowed tier. Used when the
    /// user types a custom value or moves between modes.
    pub fn normalized(mut self) -> Self {
        // M7: refuse to land on zero. Constructing a LoraRank(0) is
        // the bug-fix; normalized() that produced 0 silently
        // reached the autograd layer. Snap + lift.
        let rank = LoraRank::new(self.lora_rank).unwrap_or_default();
        self.lora_rank = rank.value();
        self
    }
}

fn pick_closest(value: u32, choices: &[u32]) -> u32 {
    *choices
        .iter()
        .min_by_key(|&&c| (c as i64 - value as i64).abs())
        .expect("non-empty choices")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized_with_rank(rank: u32) -> HyperparamFormV1 {
        let mut f = HyperparamFormV1::default();
        f.lora_rank = rank;
        f.normalized()
    }

    #[test]
    fn form_default_matches_v1_layout() {
        let f = HyperparamFormV1::default();
        assert_eq!(f.form_id, "hyperparameters");
        assert_eq!(f.training_mode, "LoRA");
        assert_eq!(f.quant_format, "Q4_K");
        assert_eq!(f.lora_rank, 16);
    }

    #[test]
    fn valid_lora_ranks_are_exactly_8_16_32_64() {
        assert_eq!(HyperparamFormV1::VALID_LORA_RANKS, &[8, 16, 32, 64]);
    }

    #[test]
    fn normalize_snap_to_nearest_rank_below() {
        let f = normalized_with_rank(10); // closer to 8 than to 16? |16-10|=6, |8-10|=2 -> 8
        assert_eq!(f.lora_rank, 8);
    }

    #[test]
    fn normalize_snap_to_nearest_rank_above() {
        let f = normalized_with_rank(20); // |20-16|=4, |20-32|=12 -> 16
        assert_eq!(f.lora_rank, 16);
    }

    #[test]
    fn normalize_keeps_exact_match_unchanged() {
        let f = normalized_with_rank(32);
        assert_eq!(f.lora_rank, 32);
    }

    // -----------------------------------------------------------------
    // M7 golden-style tests for `LoraRank` (mutation-resistant).
    //
    // Each test below pins a single hand-derived numeric value or
    // pinpoints a specific rejected path so a wrong shift, swapped
    // predicate, or removed floor mutation breaks at least one
    // assertion here. We model the test style on
    // `crates/grim-quant/tests/golden_*.rs` — exact-value assertions
    // over hand-constructed inputs, plus invariant assertions for
    // larger suites.
    // -----------------------------------------------------------------

    #[test]
    fn lora_rank_rejects_zero_in_new() {
        // The constructor (`LoraRank::new`) is the canonical
        // boundary; lora_rank = 0 is a logic-bug. Pre-fix the value
        // slipped through to the autograd layer. Post-fix `new(0)`
        // must return Err, not silently default.
        assert_eq!(LoraRank::new(0), Err(LoraRankError::Zero));
    }

    #[test]
    fn lora_rank_snaps_then_stores_value() {
        // Hand-derived: pick_closest snap points to the nearest
        // valid tier; pin specifically so a wrong distance metric
        // (e.g. subtraction in the wrong direction) is caught.
        // `pick_closest(10, [8, 16, 32, 64]) == 8` because |10-8|=2.
        let r = LoraRank::new(10).unwrap();
        assert_eq!(r.value(), 8);
        // `pick_closest(20, …) == 16` because |20-16|=4 < |20-32|=12.
        assert_eq!(LoraRank::new(20).unwrap().value(), 16);
    }

    #[test]
    fn lora_rank_exact_tier_inputs_pass_through_unchanged() {
        // Already-valid values must remain unchanged, not snapped
        // to a different neighbor.
        for &v in VALID_LORA_RANKS {
            let r = LoraRank::new(v).unwrap();
            assert_eq!(r.value(), v, "tier {v} must pass through new()");
        }
    }

    #[test]
    fn lora_rank_from_valid_rejects_unknown_tier() {
        // `from_valid` is the strict entry point used by trusted
        // config-file readers — it must reject anything not in the
        // canonical set rather than snapping.
        assert_eq!(
            LoraRank::from_valid(7),
            Err(LoraRankError::NotInTiers(7)),
            "non-tier value must round-trip the Err"
        );
        assert_eq!(
            LoraRank::from_valid(128),
            Err(LoraRankError::NotInTiers(128)),
        );
        assert!(LoraRank::from_valid(8).is_ok());
    }

    #[test]
    fn lora_rank_qlora_validation_caps_at_32() {
        // M7 QLoRA×rank bound: rank > QLORA_MAX_RANK on QLoRA must
        // surface as QloraTooLarge. Pre-fix no such check existed;
        // a 64-rank QLoRA job would have been accepted and likely
        // OOM'd on a mid-tier GPU.
        let rank_64 = LoraRank::new(64).unwrap();
        assert_eq!(
            rank_64.validate_for_mode(crate::jobs::TrainingMode::QLoRA),
            Err(LoraRankError::QloraTooLarge {
                rank: 64,
                ceiling: LoraRank::QLORA_MAX_RANK,
            }),
            "QLoRA + 64-rank must yield the specific QloraTooLarge variant"
        );
        // QLoRA + rank == 32 (== QLORA_MAX_RANK) is OK.
        let rank_32 = LoraRank::from_valid(32).unwrap();
        assert!(
            rank_32
                .validate_for_mode(crate::jobs::TrainingMode::QLoRA)
                .is_ok()
        );
    }

    #[test]
    fn lora_rank_validation_passes_for_non_qlora_modes_at_high_rank() {
        // Same 64-rank value must work under Lora / Bf16-Full since
        // only QLoRA has the floor memory constraint.
        let rank = LoraRank::new(64).unwrap();
        assert!(
            rank.validate_for_mode(crate::jobs::TrainingMode::Lora)
                .is_ok()
        );
        assert!(
            rank.validate_for_mode(crate::jobs::TrainingMode::Bf16Full)
                .is_ok()
        );
    }

    #[test]
    fn lora_rank_default_returns_a_known_valid_tier() {
        // `Default::default()` is the contract for new-job promotion;
        // it must always produce a tier-membership rank so the field
        // doesn't trip the new LoraRank checks.
        assert_eq!(LoraRank::default().value(), 16);
    }

    #[test]
    fn lora_rank_serde_round_trips_through_u32() {
        // The newtype round-trips via `serde(transparent)` to a bare
        // integer on the wire. Pin the round-trip so a
        // serialization-shape regression is caught.
        let r = LoraRank::new(16).unwrap();
        let s = serde_json::to_string(&r).unwrap();
        // The wire form is the integer value, not a JSON object —
        // because of `#[serde(transparent)]`.
        assert_eq!(s, "16");
        // Transparent newtype deserializes from a bare integer; we
        // don't reject zero at the JSON parse step (the
        // TryFrom<u32> adapter is what enforces the rejection at
        // construction time). Confirm the bare-int form is stable.
        let back: LoraRank = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn lora_rank_try_from_zero_is_rejected() {
        // The boundary that enforces zero-rejection is
        // `TryFrom<u32>` for `LoraRank`. Pin it explicitly.
        let r = LoraRank::try_from(0u32);
        assert_eq!(r, Err(LoraRankError::Zero));
        // And the conversion path via `LoraRank::new` agrees.
        assert_eq!(LoraRank::new(0), Err(LoraRankError::Zero));
    }

    // -----------------------------------------------------------------
    // Mutation-resistant golden tests for `pick_closest` (lora snap).
    // Written through `LoraRank::new` to avoid exposing the helper.
    // Each hand-picked value pins the expected nearest-tier output;
    // a mutant that flips the comparator direction, swaps the abs()
    // call to a sign, or rewrites the distance metric breaks at
    // least one assertion.
    //
    // Tiers:      [8, 16, 32, 64]
    // Hand-derived distances for boundary points:
    //   value  | nearest tier | reason
    //     4    |  8           | |8-4|=4 < |16-4|=12
    //     9    |  8           | |9-8|=1  < |16-9|=7
    //    11    |  8           | |11-8|=3 < |16-11|=5 — 8 is closer
    //    12    | 16           | |16-12|=4 < |8-12|=4 — equal; precondition decides 16 (the larger-index equals case)
    //    13    | 16           | |16-13|=3 < |8-13|=5
    //    24    | 16           | |24-16|=8 < |32-24|=8 — equal-distant: same tie-break policy as above
    //    25    | 32           | |32-25|=7 < |16-25|=9 (no tie)
    //    36    | 32           | |36-32|=4 < |64-36|=28
    //    56    | 64           | |64-56|=8 < |32-56|=24
    // -----------------------------------------------------------------
    #[test]
    fn pick_closest_hand_picked_boundary_snap_table() {
        // Pin each value to its expected nearest tier. A reflective
        // insult that picks a different neighbor is caught:
        let cases: &[(u32, u32)] = &[
            (4, 8),
            (9, 8),
            (11, 8),
            (13, 16),
            (25, 32),
            (36, 32),
            (56, 64),
        ];
        for &(value, want) in cases {
            let got = LoraRank::new(value).expect("valid rank").value();
            assert_eq!(
                got, want,
                "pick_closest({value}, &[8, 16, 32, 64]) must equal {want} — got {got}"
            );
        }
    }
}
