//! Introspection shim — gives every CVKG widget we render a stable
//! handle the test suite can poke at without invoking the renderer.
//!
//! `View` is plain-data: a tagged kind (so tests can compare cheaply),
//! a debug label string the tests assert against, and an optional
//! widget slot the renderer can pull out at runtime.
//!
//! `ViewKind` is the only Debug-derived enum; CVKG widgets themselves
//! aren't Debug, so we don't ask them to be.

use cvkg_components::{Badge, Button, HStack, Input, Slider, Stepper, Toggle, VStack};
use cvkg_components::Text;

/// Tagged view wrapper. Tests assert on `kind()` and `debug_string()`
/// to lock in the structural shape of the dashboard without invoking a
/// renderer. At runtime the renderer can consume `kind()` to know what
/// to draw on the GPU.
#[derive(Clone)]
pub struct View {
    pub kind: ViewKind,
    pub children: Vec<View>,
    pub debug: String,
    pub widget: Option<WidgetSlot>,
}

impl View {
    pub fn new(kind: ViewKind) -> Self {
        Self {
            kind,
            children: Vec::new(),
            debug: String::new(),
            widget: None,
        }
    }

    pub fn with_debug(kind: ViewKind, debug: impl Into<String>) -> Self {
        Self {
            kind,
            children: Vec::new(),
            debug: debug.into(),
            widget: None,
        }
    }

    pub fn with_children(mut self, children: Vec<View>) -> Self {
        self.children = children;
        self
    }

    pub fn kind(&self) -> ViewKind {
        self.kind
    }

    pub fn children(&self) -> &[View] {
        &self.children
    }

    /// Recursive debug string: own debug label + children joined with ", ".
    pub fn debug_string(&self) -> String {
        let own = if self.debug.is_empty() {
            format!("{:?}", self.kind)
        } else {
            self.debug.clone()
        };
        if self.children.is_empty() {
            own
        } else {
            let sub: Vec<String> = self.children.iter().map(View::debug_string).collect();
            format!("{own}({})", sub.join(", "))
        }
    }

    /// Builder: attach the live CVKG widget so the renderer can pull it.
    pub fn widget_of(mut self, slot: WidgetSlot) -> Self {
        self.widget = Some(slot);
        self
    }

    /// Builder: append a single child.
    pub fn child(mut self, child: View) -> Self {
        self.children.push(child);
        self
    }

    /// Builder: replace children.
    pub fn children_of(mut self, children: Vec<View>) -> Self {
        self.children = children;
        self
    }
}

/// Coarse view kind — used by the test shim to assert structural shape.
/// At render time the renderer dispatches on the inner CVKG widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewKind {
    VStack,
    HStack,
    Card,
    Text,
    Toggle,
    Button,
    Slider,
    Stepper,
    Input,
    Badge,
    Picker,
}

/// Type-erased slot for any CVKG widget the compositor produces.
///
/// CVKG widgets themselves don't implement `Debug`, so this slot is
/// `Debug`-free on purpose; the [`View`] wrapper above carries the
/// text-friendly debug label.
#[derive(Clone)]
pub struct WidgetSlot {
    /// Stable id for debugging.
    pub id: String,
    /// Inner widget kind.
    pub kind: WidgetInner,
}

#[derive(Clone)]
pub enum WidgetInner {
    /// `RunesCard` is generic over the View type — we exclude it from the
    /// widget slot for now. The `View` shim preserves the panel shape;
    /// the renderer can materialise the Card from scratch if needed.
    Card,
    Text(Text),
    Toggle(Toggle),
    Button(Button),
    Slider(Slider),
    Stepper(Stepper),
    Input(Input),
    Badge(Badge),
    VStack(VStack),
    HStack(HStack),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_kind_round_trips() {
        let v = View::with_debug(ViewKind::Card, "/Header/");
        assert_eq!(v.kind(), ViewKind::Card);
        assert_eq!(v.debug_string(), "/Header/");
    }

    #[test]
    fn view_with_children_renders_subtree() {
        let parent = View::with_debug(ViewKind::VStack, "main")
            .child(View::with_debug(ViewKind::Text, "title"))
            .child(View::with_debug(ViewKind::Button, "start"));
        let s = parent.debug_string();
        assert!(s.contains("main"));
        assert!(s.contains("title"));
        assert!(s.contains("start"));
    }

    #[test]
    fn view_kind_hash_eq_works() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(ViewKind::Card);
        s.insert(ViewKind::Card);
        assert_eq!(s.len(), 1);
    }
}
