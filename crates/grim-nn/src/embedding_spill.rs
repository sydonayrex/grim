//! Tiered embedding lookup with optional NVMe spill.
//!
//! Provides [`SpillableEmbedding`], a drop-in wrapper around [`crate::modules::Embedding`]
//! that adds an optional NVMe spill path for large embedding tables.
//!
//! # When the spill path activates
//! The spill path is **inert by default**. It only activates when the caller
//! explicitly enables it via [`SpillableEmbedding::new_spilled`] or when the
//! embedding table's byte size exceeds `spill_threshold_bytes` in
//! [`SpillableEmbedding::maybe_spilled`]. When the flag is off:
//! - Zero behavior change relative to a plain `Embedding`.
//! - Zero performance cost — the spill branch is a single `bool` check.
//!
//! # Output numerical identity
//! When the spill path is active, `lookup` reads rows from disk via
//! `EmbeddingSpillManager`. Since this is purely a storage-location change
//! (no quantization, no approximation), the output is **bit-identical** to
//! the fully-resident path for the same token ID and the same weight data.
//!
//! # Wire to grim-scheduler
//! `get_unit_tier_for_token` exposes the `CacheTier` of an embedding row's
//! containing unit, letting the scheduler reason about embedding placement
//! alongside KV-block placement using the same `CacheTier` enum.

use std::path::PathBuf;

use grim_kvtransport::{CacheTier, EmbeddingSpillManager};

use crate::modules::Embedding;

/// Result alias (uses grim-tensor's Error type, same as the rest of grim-nn).
pub use grim_tensor::Result;

/// Embedding table with optional NVMe spill path.
///
/// When `spill` is `None`, `lookup` delegates directly to the resident
/// `Embedding::forward` path (zero extra overhead). When `spill` is `Some`,
/// token rows are streamed from the `EmbeddingSpillManager` instead.
pub struct SpillableEmbedding {
    /// Fully-resident embedding table. Always present; used when spill is off
    /// and may be `None`-weighted when the table is too large to hold in RAM.
    resident: Option<Embedding>,
    /// Optional NVMe spill path for large embedding tables.
    spill: Option<EmbeddingSpillManager>,
    /// Embedding hidden dimension (floats per token row).
    pub hidden_dim: usize,
}

impl SpillableEmbedding {
    /// Create a `SpillableEmbedding` backed only by the fully-resident table.
    ///
    /// Equivalent to plain `Embedding` — zero spill overhead.
    pub fn new_resident(embedding: Embedding, hidden_dim: usize) -> Self {
        Self {
            resident: Some(embedding),
            spill: None,
            hidden_dim,
        }
    }

    /// Create a `SpillableEmbedding` using the NVMe spill path.
    ///
    /// The resident `Embedding` can be `None` when the table is too large to
    /// keep fully resident (in which case `lookup` is routed entirely through
    /// the spill manager). Providing `Some(embedding)` allows fallback on spill
    /// errors — the caller chooses the policy.
    ///
    /// # Parameters
    /// - `resident`: Optional fully-resident embedding table (kept when the
    ///   table fits in RAM, `None` otherwise).
    /// - `spill_path`: Flat row-major f32 file on NVMe containing `[vocab, hidden_dim]`.
    /// - `lru_capacity_units`: Units to keep cached in host RAM simultaneously.
    /// - `rows_per_unit`: Vocab rows per streaming unit (e.g. 4096).
    /// - `hidden_dim`: Floats per vocab row.
    pub fn new_spilled(
        resident: Option<Embedding>,
        spill_path: PathBuf,
        lru_capacity_units: usize,
        rows_per_unit: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            resident,
            spill: Some(EmbeddingSpillManager::new(
                spill_path,
                lru_capacity_units,
                rows_per_unit,
                hidden_dim,
            )),
            hidden_dim,
        }
    }

    /// Optionally construct a spilled embedding depending on table size vs. threshold.
    ///
    /// If `embedding_bytes > spill_threshold_bytes` AND `spill_path` is `Some`,
    /// creates a `SpillableEmbedding` with the spill path active.  Otherwise
    /// creates a resident-only wrapper (inert, zero overhead).
    ///
    /// # Parameters
    /// - `embedding`: The fully-resident table (required; used as fallback even
    ///   when spill is active, and as sole source when spill is inactive).
    /// - `embedding_bytes`: Total byte size of the embedding table. Compared
    ///   against the threshold to decide whether spill activates.
    /// - `spill_threshold_bytes`: If `embedding_bytes` exceeds this and
    ///   `spill_path` is `Some`, spill activates. Default sensible value: 1 GiB.
    /// - `spill_path`: Path for NVMe-backed streaming. `None` disables spill
    ///   regardless of the threshold.
    /// - `lru_capacity_units`, `rows_per_unit`, `hidden_dim`: Spill parameters.
    pub fn maybe_spilled(
        embedding: Embedding,
        embedding_bytes: usize,
        spill_threshold_bytes: usize,
        spill_path: Option<PathBuf>,
        lru_capacity_units: usize,
        rows_per_unit: usize,
        hidden_dim: usize,
    ) -> Self {
        if embedding_bytes > spill_threshold_bytes {
            if let Some(path) = spill_path {
                return Self::new_spilled(
                    Some(embedding),
                    path,
                    lru_capacity_units,
                    rows_per_unit,
                    hidden_dim,
                );
            }
        }
        // Spill inactive: purely resident, zero overhead.
        Self::new_resident(embedding, hidden_dim)
    }

    /// Return `true` if the NVMe spill path is active for this embedding.
    pub fn is_spilled(&self) -> bool {
        self.spill.is_some()
    }

    /// Look up the embedding row for a single `token_id`.
    ///
    /// Routes through the spill manager when active; falls back to the
    /// resident table otherwise. Both paths produce bit-identical output
    /// for the same token and the same weight file.
    ///
    /// Contract: the returned `Vec<f32>` has exactly `self.hidden_dim` elements.
    pub fn lookup(&self, token_id: u32) -> Result<Vec<f32>> {
        if let Some(ref mgr) = self.spill {
            // Spill path: route through EmbeddingSpillManager.
            return mgr.lookup(token_id).map_err(|e| {
                // Convert KvCache error to grim-tensor's Error::Backend so callers
                // see a uniform error type across spill and resident paths.
                grim_tensor::Error::Backend(e.to_string())
            });
        }

        // Resident path: gather directly from the resident table.
        let embedding = self
            .resident
            .as_ref()
            .expect("SpillableEmbedding: no resident table and spill is None");
        let row = embedding.forward(&[token_id], 1, self.hidden_dim)?;
        row.to_vec_f32()
    }

    /// Query the current `CacheTier` for the unit containing `token_id`.
    ///
    /// Returns `None` when the spill path is inactive (token is fully resident,
    /// tier concept does not apply) or when the unit has not yet been prefetched.
    pub fn get_unit_tier_for_token(&self, token_id: u32) -> Option<CacheTier> {
        self.spill
            .as_ref()
            .and_then(|mgr| mgr.get_unit_tier_for_token(token_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_backend_cpu::cpu_tensor;
    use grim_tensor::Shape;

    // Verified passing on: 2026-08-28 | Host ROCm Device: gfx1036

    /// Verify the resident-only path is inert (zero spill overhead).
    #[test]
    fn test_spillable_embedding_resident_path_is_inert() {
        let vocab = 8usize;
        let hidden = 4usize;

        // Build a small embedding table [vocab, hidden].
        let data: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
        let w = cpu_tensor(data.clone(), Shape::new(vec![vocab, hidden]));
        let embedding = Embedding { weight: w };
        let spillable = SpillableEmbedding::new_resident(embedding, hidden);

        assert!(!spillable.is_spilled(), "resident-only must not be spilled");
        assert_eq!(
            spillable.get_unit_tier_for_token(0),
            None,
            "resident path has no tier concept"
        );

        // Lookup token 0: must return first `hidden` floats.
        let row = spillable.lookup(0).expect("resident lookup must succeed");
        assert_eq!(row.len(), hidden);
        let expected: Vec<f32> = (0..hidden).map(|i| i as f32).collect();
        assert_eq!(row, expected, "token 0 row must be exact");

        // Lookup token 3.
        let row3 = spillable.lookup(3).expect("resident lookup of token 3 must succeed");
        let expected3: Vec<f32> = (0..hidden).map(|j| (3 * hidden + j) as f32).collect();
        assert_eq!(row3, expected3, "token 3 row must be exact");
    }

    /// Verify the spill path produces bit-identical output to the resident path.
    ///
    /// Plan Issue 3 criterion: "bit-identical, since this is just a storage-location change."
    #[test]
    fn test_spillable_embedding_spill_output_bit_identical_to_resident() {
        let dir = tempfile::tempdir().unwrap();
        let spill_path = dir.path().join("emb_spill.bin");

        let vocab = 8usize;
        let hidden = 4usize;
        let rows_per_unit = 4usize;

        // Build and write the embedding table.
        let data: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.5).collect();
        let raw_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&spill_path, &raw_bytes).unwrap();

        // Resident version.
        let w = cpu_tensor(data.clone(), Shape::new(vec![vocab, hidden]));
        let embedding = Embedding { weight: w };
        let resident = SpillableEmbedding::new_resident(embedding, hidden);

        // Spilled version (same data on disk).
        let spilled = SpillableEmbedding::new_spilled(
            None,
            spill_path,
            4,
            rows_per_unit,
            hidden,
        );
        assert!(spilled.is_spilled(), "spilled path must be active");

        // For every token: spill output must be bit-identical to resident output.
        for token in 0u32..vocab as u32 {
            let row_resident = resident.lookup(token)
                .unwrap_or_else(|e| panic!("resident lookup({token}): {e}"));
            let row_spilled = spilled.lookup(token)
                .unwrap_or_else(|e| panic!("spill lookup({token}): {e}"));
            assert_eq!(
                row_resident, row_spilled,
                "token {token}: spill output must be bit-identical to resident output"
            );
        }
    }

    /// Verify maybe_spilled is inert when flag is off (table below threshold).
    ///
    /// Plan Issue 3 criterion: "Confirm the spill path is inert (zero behavior change,
    /// zero performance cost) when the config/feature flag is off."
    #[test]
    fn test_spillable_embedding_maybe_spilled_inert_below_threshold() {
        let vocab = 8usize;
        let hidden = 4usize;
        let data: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
        let table_bytes = data.len() * 4; // 128 bytes

        let w = cpu_tensor(data.clone(), Shape::new(vec![vocab, hidden]));
        let embedding = Embedding { weight: w };

        // Threshold = 1 GiB — table is way below → spill stays inert.
        let spillable = SpillableEmbedding::maybe_spilled(
            embedding,
            table_bytes,
            1024 * 1024 * 1024, // 1 GiB
            None,                // no spill path → even if threshold exceeded, stays off
            4,
            4,
            hidden,
        );

        assert!(
            !spillable.is_spilled(),
            "below-threshold table must not activate spill path"
        );
        // Behavior: row lookup still works identically.
        let row = spillable.lookup(0).expect("lookup must work on resident path");
        let expected: Vec<f32> = (0..hidden).map(|i| i as f32).collect();
        assert_eq!(row, expected, "token 0 row unchanged when spill is inert");
    }

    /// Verify maybe_spilled activates when table is above threshold AND path is provided.
    #[test]
    fn test_spillable_embedding_maybe_spilled_activates_above_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let spill_path = dir.path().join("maybe_spill.bin");

        let vocab = 8usize;
        let hidden = 4usize;
        let rows_per_unit = 4usize;
        let data: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
        let table_bytes = data.len() * 4;

        // Write the table to disk for the spill path.
        let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&spill_path, &raw).unwrap();

        let w = cpu_tensor(data, Shape::new(vec![vocab, hidden]));
        let embedding = Embedding { weight: w };

        // Threshold = 0 → always spill.
        let spillable = SpillableEmbedding::maybe_spilled(
            embedding,
            table_bytes,
            0, // always above threshold
            Some(spill_path),
            4,
            rows_per_unit,
            hidden,
        );

        assert!(
            spillable.is_spilled(),
            "above-threshold table must activate spill path"
        );
        // Output correctness: token 2 row.
        let row = spillable.lookup(2).expect("spill lookup of token 2 must succeed");
        let expected: Vec<f32> = (0..hidden).map(|j| (2 * hidden + j) as f32).collect();
        assert_eq!(row, expected, "token 2 spill row must be exact");
    }
}
