//! OKLCH design tokens for Grim's Garage (WI-T9).
//!
//! Provides OKLCH color structures and theme tokens for CSS styling and layout configuration.

use serde::{Deserialize, Serialize};

/// OKLCH color token representation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OklchColor {
    pub l: f32,
    pub c: f32,
    pub h: f32,
    pub a: f32,
}

impl OklchColor {
    /// Format as CSS `oklch(L C H / A)` string.
    pub fn to_css(&self) -> String {
        format!("oklch({:.2} {:.3} {:.1} / {:.2})", self.l, self.c, self.h, self.a)
    }
}

/// Pilot-blue OKLCH seed that anchors the whole theme.
pub const GARAGE_SEED_OKLCH: OklchColor = OklchColor {
    l: 0.62,
    c: 0.10,
    h: 240.0,
    a: 1.0,
};

/// Plain surface token used by the sidebar / main shell / KPI cards.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GarageSurface {
    pub oklch: OklchColor,
}

impl GarageSurface {
    pub const SIDEBAR: GarageSurface = GarageSurface {
        oklch: OklchColor { l: 0.96, c: 0.005, h: 240.0, a: 1.0 },
    };
    pub const MAIN_PANEL: GarageSurface = GarageSurface {
        oklch: OklchColor { l: 0.98, c: 0.0, h: 0.0, a: 1.0 },
    };
    pub const KPI_CARD: GarageSurface = GarageSurface {
        oklch: OklchColor { l: 0.92, c: 0.02, h: 240.0, a: 1.0 },
    };
}

/// Aggregate layout theme.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThemingLayout {
    pub base_seed: OklchColor,
    pub sidebar: GarageSurface,
    pub main_panel: GarageSurface,
    pub kpi_card: GarageSurface,
}

pub fn themed_layout() -> ThemingLayout {
    ThemingLayout {
        base_seed: GARAGE_SEED_OKLCH,
        sidebar: GarageSurface::SIDEBAR,
        main_panel: GarageSurface::MAIN_PANEL,
        kpi_card: GarageSurface::KPI_CARD,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garage_seed_is_in_unit_alpha_range() {
        assert!(GARAGE_SEED_OKLCH.a >= 0.0 && GARAGE_SEED_OKLCH.a <= 1.0);
    }

    #[test]
    fn garage_seed_lightness_in_oklch_perceptual_range() {
        assert!(GARAGE_SEED_OKLCH.l >= 0.0 && GARAGE_SEED_OKLCH.l <= 1.0);
    }

    #[test]
    fn sidebar_surface_is_lighter_than_seed() {
        assert!(GarageSurface::SIDEBAR.oklch.l > GARAGE_SEED_OKLCH.l);
    }

    #[test]
    fn main_panel_is_lightest_surface() {
        let sidebar_l = GarageSurface::SIDEBAR.oklch.l;
        let main_l = GarageSurface::MAIN_PANEL.oklch.l;
        let kpi_l = GarageSurface::KPI_CARD.oklch.l;
        assert!(main_l >= sidebar_l);
        assert!(kpi_l <= main_l);
    }

    #[test]
    fn themed_layout_carries_all_surfaces() {
        let layout = themed_layout();
        assert_eq!(layout.base_seed.l, GARAGE_SEED_OKLCH.l);
        assert_eq!(layout.sidebar.oklch.l, GarageSurface::SIDEBAR.oklch.l);
        assert_eq!(layout.main_panel.oklch.l, GarageSurface::MAIN_PANEL.oklch.l);
        assert_eq!(layout.kpi_card.oklch.l, GarageSurface::KPI_CARD.oklch.l);
    }
}
