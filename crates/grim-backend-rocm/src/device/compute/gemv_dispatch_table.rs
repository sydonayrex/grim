//! Measured per-arch GEMV variant dispatch table (astrum P2).
//!
//! Converts the existing GEMV variant zoo into a generated, data-driven dispatch
//! decision. Default OFF: when `GRIM_ASTRUM_GEMV_TABLE` is unset/unloadable, or on
//! any miss, dispatch returns `Plain` so existing heuristics (bit-identical
//! default behavior) run unchanged.
//!
//! Rust-only dispatch — no HIP kernel source is modified by this module.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// GEMV kernel variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GemvVariant {
    Plain,
    RowsR2,
    Prefetch,
    PrefetchR2,
    KGroupSkip,
}

/// Dispatch key for a GEMV call.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GemvKey {
    pub quant: String,
    pub arch: String,
    /// Token count (M) bucket/value used by the harness (e.g. 1, 2).
    pub tokens: u32,
    /// Row-count class: 0 = <1024 rows, 1 = >=1024 rows.
    pub rows_class: u32,
}

impl GemvKey {
    pub fn new(quant: impl Into<String>, arch: impl Into<String>, tokens: u32, n_rows: usize) -> Self {
        Self {
            quant: quant.into(),
            arch: arch.into(),
            tokens,
            rows_class: if n_rows >= 1024 { 1 } else { 0 },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TableEntry {
    quant: String,
    tokens: u32,
    rows_class: u32,
    variant: GemvVariant,
    speedup: f64,
}

/// Measured GEMV dispatch table for one arch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemvTable {
    pub arch: String,
    #[serde(default)]
    entries: Vec<TableEntry>,
}

impl GemvTable {
    /// Construct a table (convenience for tests).
    pub fn from_entries(arch: impl Into<String>, entries: Vec<(GemvKey, GemvVariant, f64)>) -> Self {
        Self {
            arch: arch.into(),
            entries: entries
                .into_iter()
                .map(|(k, variant, speedup)| TableEntry {
                    quant: k.quant,
                    tokens: k.tokens,
                    rows_class: k.rows_class,
                    variant,
                    speedup,
                })
                .collect(),
        }
    }
}

// Allow the tuple-style construction used by the plan's test sketch.
impl GemvTable {
    #[allow(clippy::type_complexity)]
    pub fn new(arch: String, entries: Vec<(GemvKey, GemvVariant, f64)>) -> Self {
        Self::from_entries(arch, entries)
    }
}

// Support `GemvTable { arch, entries: vec![(key, variant, speedup)] }` style by
// implementing From via a helper macro-free path: tests should prefer
// `GemvTable::from_entries`, but keep the struct fields public.
impl From<(GemvKey, GemvVariant, f64)> for TableEntry {
    fn from((k, variant, speedup): (GemvKey, GemvVariant, f64)) -> Self {
        Self {
            quant: k.quant,
            tokens: k.tokens,
            rows_class: k.rows_class,
            variant,
            speedup,
        }
    }
}

/// Lookup a variant in `table` for `key`. Any mismatch -> `Plain`.
pub fn lookup_variant(table: &GemvTable, key: &GemvKey) -> GemvVariant {
    if !table.arch.eq_ignore_ascii_case(&key.arch) {
        return GemvVariant::Plain;
    }
    table
        .entries
        .iter()
        .find(|e| {
            e.quant.eq_ignore_ascii_case(&key.quant)
                && e.tokens == key.tokens
                && e.rows_class == key.rows_class
        })
        .map(|e| e.variant)
        .unwrap_or(GemvVariant::Plain)
}

/// Load a GEMV dispatch table from JSON.
pub fn load_gemv_table(path: &str) -> Option<GemvTable> {
    let text = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str::<GemvTable>(&text) {
        Ok(t) => {
            eprintln!(
                "[astrum] gemv dispatch table: {path} (arch={}, {} entries)",
                t.arch,
                t.entries.len()
            );
            Some(t)
        }
        Err(e) => {
            eprintln!("[astrum] gemv dispatch table load failed ({path}): {e}");
            None
        }
    }
}

static GEMV_TABLE: OnceLock<Option<GemvTable>> = OnceLock::new();

/// Env-gated variant lookup. `GRIM_ASTRUM_GEMV_TABLE` unset/unloadable -> `Plain`.
pub fn current_variant(key: &GemvKey) -> GemvVariant {
    let table = GEMV_TABLE.get_or_init(|| {
        let path = std::env::var("GRIM_ASTRUM_GEMV_TABLE").ok()?;
        load_gemv_table(&path)
    });
    match table.as_ref() {
        Some(t) => lookup_variant(t, key),
        None => GemvVariant::Plain,
    }
}
