//! OKLCH design tokens for Grim's Garage.
//!
//! CVKG-themes provides the OKLCH foundation; this module wraps the
//! `ThemeBuilder` with Grim's Garage-specific chunk choices so the
//! entire UI surface consumes one consistent token set.

use cvkg_themes::{OklchColor, Theme, ThemeBuilder};

/// Pilot-blue OKLCH seed that anchors the whole theme. Glass tints and
/// KPI cards derive from this base.
pub const GARAGE_SEED_OKLCH: OklchColor = OklchColor {
    l: 0.62,
    c: 0.10,
    h: 240.0,
    a: 1.0,
};

/// Default light theme used across Grim's Garage panels.
pub fn grim_garage_default_theme() -> Theme {
    ThemeBuilder::from_seed(GARAGE_SEED_OKLCH).build()
}

/// Plain glassmorphic surface token used by the sidebar / main shell / KPI cards.
#[derive(Debug, Clone, Copy)]
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

/// Aggregate that the CVKG `View` implementors read from.
#[derive(Debug, Clone, Copy)]
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
    fn default_theme_builds_without_panic() {
        let _theme = grim_garage_default_theme();
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
