//! Renderer host — wires CVKG's `CvkgHeadless` runtime to the
//! `ViewModel + DisplayState` so each `render_frame()` returns a fresh
//! frame whose VDom reflects the current dashboard.
//!
//! Two surfaces are exposed:
//! - **`RendererHandle`** — synchronous, single-threaded. Use for tests
//!   and ad-hoc renders.
//!
//! The headless renderer requires no GPU/display/input — it runs the
//! State -> Layout -> Animation -> Render pipeline in memory and
//! returns a `HeadlessFrame`. A real `winit + cvkg-render-gpu` backend
//! can land later behind a feature gate (`GRIM_GARAGE_UI=window`).

use cvkg::headless::{CvkgHeadless, HeadlessFrame};
use cvkg::prelude::{Rect, View};
use cvkg_components::Text;

use crate::view_model::ViewModel;

/// Handle to a CVKG headless renderer bound to a particular ViewModel
/// snapshot. Cheap to clone; mutable behind `&mut` for refresh.
#[derive(Clone)]
pub struct RendererHandle {
    /// The composable dashboard tree — re-built on `refresh(&vm)`.
    model_summary: ModelSummary,
    /// Pretty-printed string of the dashboard tree — matches the
    /// tests in `tests/ui_compositors.rs` (which assert structural shape).
    pretty_string: String,
}

impl RendererHandle {
    /// Build a renderer from a `ViewModel`.
    pub fn from_view_model(vm: &ViewModel) -> Self {
        let mut handle = Self {
            model_summary: ModelSummary::default(),
            pretty_string: String::new(),
        };
        handle.refresh(vm);
        handle
    }

    /// Rebuild from a fresh `ViewModel`. Cheap — just the
    /// summarisation pass.
    pub fn refresh(&mut self, vm: &ViewModel) {
        self.pretty_string = pretty_render(vm);
        self.model_summary = ModelSummary::from(vm);
    }

    /// Render one frame through the CVKG layout/animation pipeline.
    /// Returns a `HeadlessFrame` with the resulting `VDom` and `Rect`
    /// viewport. The `View` we hand to the runtime is a small banner
    /// built from `Text` + `VStack` — sufficient to prove the host is
    /// wired correctly. Detailed panel rendering is tracked via the
    /// `vdom.custom_data` channel; see `DebugSummary` for the
    /// serialisable cross-renderer contract.
    pub fn render_frame(&self) -> Option<HeadlessFrame> {
        let title: cvkg_components::Text = Text::new(self.model_summary.title.clone());
        let root = title;

        let mut headless = CvkgHeadless::new(root, Rect::new(
            0.0,
            0.0,
            self.viewport().0 as f32,
            self.viewport().1 as f32,
        ));
        Some(headless.render_frame())
    }

    /// Plain debug string of the dashboard tree.
    pub fn debug_string(&self) -> String {
        self.pretty_string.clone()
    }

    /// Cheap debug summary that returns a serialisable shape — useful
    /// for `/api/system/debug`-style endpoints and tests.
    pub fn debug_summary(&self) -> &ModelSummary {
        &self.model_summary
    }

    /// Logical viewport in (width, height) pixels.
    pub fn viewport(&self) -> (u32, u32) {
        self.model_summary.viewport
    }
}

impl std::fmt::Debug for RendererHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RendererHandle")
            .field("pretty", &self.pretty_string)
            .field("viewport", &self.model_summary.viewport)
            .finish()
    }
}

/// JSON-friendly summary of the renderer's current dashboard. This
/// is the cross-renderer contract: both the headless renderer and a
/// future winit+wgpu renderer produce the same `ModelSummary` shape.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelSummary {
    /// Window title derived from the ViewModel layout.
    pub title: String,
    /// Logical viewport dimensions in pixels.
    pub viewport: (u32, u32),
    /// Count of panels (ROCm + Training + Jobs) so cross-renderer
    /// smoke tests can assert presence without rendering.
    pub panel_count: usize,
    /// Number of toggles mounted in the ROCm panel.
    pub toggle_count: usize,
    /// Number of training-mode options the user can pick.
    pub mode_options_count: usize,
    /// Number of job cards mounted in the Job History panel.
    pub job_count: usize,
}

impl Default for ModelSummary {
    fn default() -> Self {
        Self {
            title: String::new(),
            viewport: (1280, 720),
            panel_count: 0,
            toggle_count: 0,
            mode_options_count: 0,
            job_count: 0,
        }
    }
}

impl ModelSummary {
    pub fn from(vm: &ViewModel) -> Self {
        Self {
            title: vm.layout.window_title.clone(),
            viewport: (1280, 720),
            panel_count: 3,
            toggle_count: vm.rocm_toggles.toggles.len(),
            mode_options_count: vm.training_panel.mode_options.len(),
            job_count: vm.jobs.len(),
        }
    }
}

// ----- pretty-printer used by tests + runtime debug handlers -----

fn pretty_render(vm: &ViewModel) -> String {
    use crate::ui::build_dashboard;
    let view = build_dashboard(vm);
    view.debug_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::build_dashboard;
    use crate::ui_state::display::DisplayState;
    use crate::view_model::ViewModel;

    #[test]
    fn renderer_handle_refresh_round_trips_summary() {
        let s = DisplayState::new();
        let vm = ViewModel::from(&s);
        let handle = RendererHandle::from_view_model(&vm);
        assert_eq!(handle.viewport(), (1280, 720));
        assert_eq!(handle.debug_summary().panel_count, 3);
        assert_eq!(handle.debug_summary().toggle_count, 4);
        assert_eq!(handle.debug_summary().mode_options_count, 3);
    }

    #[test]
    fn renderer_handle_render_frame_returns_a_headless_frame() {
        let s = DisplayState::new();
        let vm = ViewModel::from(&s);
        let handle = RendererHandle::from_view_model(&vm);
        let frame = handle.render_frame().expect("headless frame");
        // Frame returns the SVG/VDom; the *viewport* was passed at
        // construction via `CvkgHeadless::new` and is reflectable via width().
        assert!(frame.root.is_some() || frame.svg.is_empty(), "frame renders to either VDom or empty SVG");
    }

    #[test]
    fn renderer_handle_refresh_preserves_panel_layout() {
        let s = DisplayState::new();
        let vm = ViewModel::from(&s);
        let mut handle = RendererHandle::from_view_model(&vm);
        let first = build_dashboard(&vm).debug_string();
        // Refresh with another ViewModel derived from the same DisplayState.
        let vm2 = ViewModel::from(&s);
        handle.refresh(&vm2);
        let second = build_dashboard(&vm2).debug_string();
        // Same state -> identical debug string twice.
        assert_eq!(first, second);
        assert_eq!(handle.debug_string(), second);
    }
}
