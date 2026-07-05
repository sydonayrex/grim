//! Top-level dashboard layout — the column/row grid that the CVKG window
//! renders into. Kept as data so renderers can read it without depending
//! on any CVKG-widget-specific knowledge (e.g. `MjolnirFrame`).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnSpec {
    /// Stable id for the column; used as the CVKG `BifrostTab` group key.
    pub id: String,
    /// Header label shown above the column.
    pub header: String,
    /// Width in cvkg `Fraction` units (0..=1).
    pub fraction: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppShellLayout {
    pub window_title: String,
    pub header_height: u32,
    pub sidebar_fraction: f32,
    pub main_fraction: f32,
    /// Columns inside the main panel — left to right.
    pub columns: Vec<ColumnSpec>,
}

impl Default for AppShellLayout {
    fn default() -> Self {
        Self {
            window_title: "Grim's Garage — Local ROCm Training Dashboard".into(),
            header_height: 56,
            sidebar_fraction: 0.28,
            main_fraction: 0.72,
            columns: vec![
                ColumnSpec {
                    id: "rocm".into(),
                    header: "Device & Fusion".into(),
                    fraction: 0.33,
                },
                ColumnSpec {
                    id: "hyperparams".into(),
                    header: "Hyperparameters".into(),
                    fraction: 0.33,
                },
                ColumnSpec {
                    id: "jobs".into(),
                    header: "Job History".into(),
                    fraction: 0.34,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_columns_sum_to_one() {
        let layout = AppShellLayout::default();
        let total: f32 = layout.columns.iter().map(|c| c.fraction).sum();
        assert!((total - 1.0).abs() < 1e-6, "columns must sum to 1.0, got {total}");
    }

    #[test]
    fn sidebar_and_main_sum_to_one() {
        let layout = AppShellLayout::default();
        let sum = layout.sidebar_fraction + layout.main_fraction;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn column_ids_are_stable() {
        let layout = AppShellLayout::default();
        assert_eq!(layout.columns[0].id, "rocm");
        assert_eq!(layout.columns[1].id, "hyperparams");
        assert_eq!(layout.columns[2].id, "jobs");
    }

    #[test]
    fn header_height_is_tall_enough_for_a_toolbar() {
        let layout = AppShellLayout::default();
        assert!(layout.header_height >= 32);
    }
}
