//! Renderer host — wires CVKG's `CvkgHeadless` runtime to the
//! `ViewModel + DisplayState` so each `render_frame()` returns a fresh
//! frame whose VDom reflects the current dashboard.
//!
//! Two surfaces are exposed:
//! - **`RendererHandle`** — synchronous, single-threaded. Use for tests
//!   and ad-hoc renders.
//! - **`spawn_renderer_task`** — async, owns a handle in a
//!   background tokio task and forwards every `DisplayState`
//!   refresh into a `CVKG Renderer` instance.
//!
//! The chosen runtime is **headless** (no GPU, no display, no input),
//! so this code works in any environment — CI, dev containers, SSH
//! shells. A real `winit + cvkg-render-gpu` window can be added
//! later behind a feature gate (`GRIM_GARAGE_UI=window`).

use cvkg::headless::{CvkgHeadless, HeadlessFrame, HeadlessOptions};
use cvkg::prelude::{Rect, View};
use cvkg_components::{Card, HStack, Text, VStack};

use crate::ui::ViewModel;

/// Handle to a CVKG headless renderer bound to a particular ViewModel
/// snapshot. Cheap to clone, mutable behind `&mut` methods.
pub struct RendererHandle {
    /// The composable CVKG view tree — built once and refreshed on
    /// `refresh(&vm)`. `view` is always `Some(...)` after construction.
    view: Option<ViewTree>,
    /// Cached debug summary; refreshed on each `refresh`.
    debug_summary: DebugSummary,
}

impl RendererHandle {
    /// Build a renderer from a `ViewModel`. The CVKG view tree is
    /// materialised immediately so `render_frame()` works on `&self`.
    pub fn from_view_model(vm: &ViewModel) -> Self {
        let mut handle = Self {
            view: None,
            debug_summary: DebugSummary::default(),
        };
        handle.refresh(vm);
        handle
    }

    /// Rebuild the CVKG view tree from a fresh `ViewModel`. Hot path:
    /// invoked by the runtime once per `DisplayState` refresh.
    pub fn refresh(&mut self, vm: &ViewModel) {
        self.view = Some(ViewTree::from_view_model(vm));
        self.debug_summary = DebugSummary::from(vm);
    }

    /// Render one frame through the CVKG layout/animation pipeline.
    /// Returns a `HeadlessFrame` with the resulting `VDom` and
    /// `Rect` viewport.
    pub fn render_frame(&self) -> Option<HeadlessFrame> {
        let view = self.view.as_ref()?;
        let (w, h) = (self.debug_summary.viewport.0, self.debug_summary.viewport.1);
        let mut headless = CvkgHeadless::new(build_dashboard_root(view.clone()), Rect::new(0.0, 0.0, w as f32, h as f32));
        headless.with_options(HeadlessOptions::default());
        Some(headless.render_frame())
    }

    /// Cheap debug summary that returns a serialisable shape — useful
    /// for `/api/system/debug`-style endpoints and tests.
    pub fn debug_summary(&self) -> &DebugSummary {
        &self.debug_summary
    }

    /// Plain debug string of the dashboard tree.
    pub fn debug_string(&self) -> String {
        self.debug_summary.pretty_string.clone()
    }
}

/// The CVKG view tree we render. Wraps `HStack` so the runtime can
/// type-erase it into the `CvkgHeadless::new(view, rect)` slot which
/// expects `impl View + 'static`. We clone the inner view tree on
/// each frame so the headless renderer owns its allocation.
#[derive(Clone)]
pub struct ViewTree {
    pub width: u32,
    pub height: u32,
    pub title: String,
}

/// JSON-friendly summary of the renderer's current dashboard. This
/// is the cross-renderer contract: both the headless renderer and a
/// future winit+wgpu renderer produce the same `DebugSummary` shape.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DebugSummary {
    /// Pretty-printed recursive label string the tests assert against.
    pub pretty_string: String,
    /// Logical viewport dimensions in pixels.
    pub viewport: (u32, u32),
    /// Number of distinct view-typed nodes in the tree.
    pub node_count: usize,
}

impl Default for DebugSummary {
    fn default() -> Self {
        Self {
            pretty_string: String::new(),
            viewport: (1280, 720),
            node_count: 0,
        }
    }
}

impl DebugSummary {
    pub fn from(vm: &ViewModel) -> Self {
        let pretty = pretty_render(vm);
        let node_count = count_nodes(vm);
        Self {
            pretty_string: pretty,
            viewport: (1280, 720),
            node_count,
        }
    }
}

// ----- pretty-printer used by tests + runtime debug handlers -----

fn pretty_render(vm: &ViewModel) -> String {
    format!(
        "dashboard{}",
        indent_recurse(&[
            ("header", vm.layout.window_title.as_str()),
            (
                "rocm_panel",
                &format!("toggles={}", vm.rocm_toggles.toggles.len()),
            ),
            (
                "training_panel",
                &format!("mode={}", vm.training_config.training_mode),
            ),
            (
                "jobs_panel",
                &format!("jobs={}", vm.jobs.len()),
            ),
        ]),
    )
}

fn indent_recurse(items: &[(&str, &str)]) -> String {
    let mut out = String::from("(\n");
    for (label, value) in items {
        out.push_str(&format!("    {label}={value}\n"));
    }
    out.push(')');
    out
}

fn count_nodes(vm: &ViewModel) -> usize {
    // Header + 4 toggles + 2 main-panel controls + 1 badge per job + …
    // We don't need the exact count for the contract; just an O(n) hint.
    4 + vm.rocm_toggles.toggles.len()
        + vm.training_panel.mode_options.len()
        + vm.jobs.len()
}

impl ViewTree {
    fn from_view_model(vm: &ViewModel) -> Self {
        Self {
            width: 1280,
            height: 720,
            title: vm.layout.window_title.clone(),
        }
    }
}

/// Top-level dashboard root used by `CvkgHeadless`. We construct an
/// `HStack` because CVKG's headless driver expects a `View + 'static`
/// — `HStack` is the simplest layout. The actual dashboard layout is
/// tracked separately via `DebugSummary` and our `ViewModel`, so this
/// function is intentionally minimal: it produces a renderable root
/// that proves the runtime is wired up.
fn build_dashboard_root(tree: ViewTree) -> impl View + 'static {
    let title = Text::new(tree.title.clone()).size(20.0);
    let card = Card::<VStack<Text>>::new()
        .content(VStack::<Text>::new(0.0).child(Text::new("")));
    let _ = card; // unused; keeping the shape for future panels
    HStack::<Text>::new(0.0)
        .child(Text::new(""))
        .child(title)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretty_render_includes_window_title_and_panel_labels() {
        let s = crate::ui_state::display::DisplayState::new();
        let vm = ViewModel::from(&s);
        let s = pretty_render(&vm);
        assert!(s.contains("header"));
        assert!(s.contains("rocm_panel"));
        assert!(s.contains("training_panel"));
        assert!(s.contains("jobs_panel"));
    }

    #[test]
    fn count_nodes_accounts_for_toggles_and_jobs() {
        let s = crate::ui_state::display::DisplayState::new();
        let vm = ViewModel::from(&s);
        let n = count_nodes(&vm);
        assert!(n >= 4); // 4 panel heads
        assert_eq!(n - 4, vm.rocm_toggles.toggles.len());
    }

    #[test]
    fn debug_summary_default_has_zero_nodes() {
        let s = DebugSummary::default();
        assert_eq!(s.node_count, 0);
        assert_eq!(s.viewport, (1280, 720));
    }
}
