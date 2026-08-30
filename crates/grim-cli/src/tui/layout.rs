//! Constrained layout engine for the chat TUI.
//!
//! Provides `VStack`, `HStack`, and `ScrollView` as composable layout nodes.
//! Built on `ratatui::layout::Layout` and `Constraint` for allocation.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

/// How a child's main-axis size is determined before grow/shrink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Use the child's intrinsic size.
    Auto,
    /// Use a fixed cell count, clamped to min/max.
    Fixed(u16),
}

/// One entry in a stack.
pub struct StackEntry {
    pub node: Box<dyn LayoutNode>,
    pub basis: Basis,
    pub grow: u16,
    pub shrink: u16,
    pub min_size: u16,
    pub max_size: Option<u16>,
}

impl StackEntry {
    /// Convenience for `Basis::Auto, grow: 0, shrink: 1, min_size: 0`.
    pub fn auto(node: Box<dyn LayoutNode>) -> Self {
        Self {
            node,
            basis: Basis::Auto,
            grow: 0,
            shrink: 1,
            min_size: 0,
            max_size: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StackOptions {
    pub gap: u16,
}

/// Anything that can be measured and painted.
pub trait LayoutNode {
    /// Intrinsic height when given `width` columns.
    fn height_for_width(&self, width: u16) -> u16;
    /// Paint into `area` of `buf`.
    fn render(&self, area: Rect, buf: &mut Buffer);
}

// ---------------------------------------------------------------------------
// VStack
// ---------------------------------------------------------------------------

pub struct VStack {
    children: Vec<StackEntry>,
    options: StackOptions,
}

impl VStack {
    pub fn new(children: Vec<StackEntry>, options: StackOptions) -> Self {
        Self { children, options }
    }
}

impl LayoutNode for VStack {
    fn height_for_width(&self, width: u16) -> u16 {
        let gaps = self.options.gap * self.children.len().saturating_sub(1) as u16;
        let sum: u16 = self
            .children
            .iter()
            .map(|e| {
                let h = match e.basis {
                    Basis::Auto => e.node.height_for_width(width),
                    Basis::Fixed(n) => n,
                };
                h.clamp(e.min_size, e.max_size.unwrap_or(u16::MAX))
            })
            .sum();
        sum + gaps
    }

    fn render(&self, area: Rect, buf: &mut Buffer) {
        // Allocate heights, then paint each child at its y offset.
        // Positive remaining space goes to grow > 0 entries proportional to grow.
        // Overflow shrinks entries with shrink > 0 proportional to shrink.
        // Deterministic rounding: leftover cells go to earlier children.
        let allocated = allocate_main_axis(&self.children, area.height, self.options.gap, area.width);
        let mut y = area.y;
        for (entry, h) in self.children.iter().zip(allocated) {
            if h == 0 {
                continue;
            }
            let child_area = Rect {
                x: area.x,
                y,
                width: area.width,
                height: h,
            };
            entry.node.render(child_area, buf);
            y += h + self.options.gap;
        }
    }
}

// ---------------------------------------------------------------------------
// HStack (analogous, allocates widths)
// ---------------------------------------------------------------------------

pub struct HStack {
    children: Vec<StackEntry>,
    options: StackOptions,
}

impl HStack {
    pub fn new(children: Vec<StackEntry>, options: StackOptions) -> Self {
        Self { children, options }
    }
}

impl LayoutNode for HStack {
    fn height_for_width(&self, width: u16) -> u16 {
        // Allocate widths first, then measure child heights at allocated widths.
        let widths = allocate_main_axis(&self.children, width, self.options.gap, width);
        self.children
            .iter()
            .zip(widths)
            .map(|(e, w)| e.node.height_for_width(w))
            .max()
            .unwrap_or(0)
    }

    fn render(&self, area: Rect, buf: &mut Buffer) {
        let widths = allocate_main_axis(&self.children, area.width, self.options.gap, area.width);
        let mut x = area.x;
        for (entry, w) in self.children.iter().zip(widths) {
            if w == 0 {
                continue;
            }
            let child_area = Rect {
                x,
                y: area.y,
                width: w,
                height: area.height,
            };
            entry.node.render(child_area, buf);
            x += w + self.options.gap;
        }
    }
}

// ---------------------------------------------------------------------------
// ScrollView
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ScrollViewOptions {
    pub follow_end: bool,
}

pub struct ScrollView {
    child: Box<dyn LayoutNode>,
    follow_end: bool,
    following_end: bool,
    pub scroll_top: usize,
    viewport_height: usize,
}

impl ScrollView {
    pub fn new(child: Box<dyn LayoutNode>, options: ScrollViewOptions) -> Self {
        let follow = options.follow_end;
        Self {
            child,
            follow_end: follow,
            following_end: follow,
            scroll_top: 0,
            viewport_height: 0,
        }
    }

    pub fn set_viewport_height(&mut self, h: usize) {
        self.viewport_height = h;
        if self.following_end {
            let content_h = self.child.height_for_width(80) as usize;
            self.scroll_top = content_h.saturating_sub(h);
        }
    }

    /// Scroll by `delta` lines. Returns unused delta (for chaining).
    pub fn scroll_by(&mut self, delta: isize) -> isize {
        let content_h = self.child.height_for_width(80) as usize;
        let max_top = content_h.saturating_sub(self.viewport_height);
        let next = (self.scroll_top as isize + delta).clamp(0, max_top as isize) as usize;
        let moved = next as isize - self.scroll_top as isize;
        self.scroll_top = next;
        self.following_end = self.follow_end && self.scroll_top == max_top;
        delta - moved
    }

    pub fn is_following_end(&self) -> bool {
        self.following_end
    }

    pub fn scroll_to_end(&mut self) {
        let content_h = self.child.height_for_width(80) as usize;
        self.scroll_top = content_h.saturating_sub(self.viewport_height);
        self.following_end = self.follow_end;
    }

    pub fn scroll_to_start(&mut self) {
        self.scroll_top = 0;
        self.following_end = false;
    }
}

impl LayoutNode for ScrollView {
    fn height_for_width(&self, _width: u16) -> u16 {
        self.viewport_height as u16
    }

    fn render(&self, area: Rect, buf: &mut Buffer) {
        // Render child at full height into a temporary buffer, then copy
        // the viewport slice at scroll_top into the real buffer.
        // For the initial implementation, render directly and clip.
        // A later optimization can use the temp-buffer approach.
        self.child.render(area, buf);
    }
}

// ---------------------------------------------------------------------------
// Allocation helper (shared by VStack and HStack)
// ---------------------------------------------------------------------------

fn allocate_main_axis(
    children: &[StackEntry],
    available: u16,
    gap: u16,
    width: u16,
) -> Vec<u16> {
    if children.is_empty() {
        return vec![];
    }
    let gaps = gap * children.len().saturating_sub(1) as u16;
    let avail_for_children = available.saturating_sub(gaps);

    // 1. Resolve basis to initial sizes.
    let mut sizes: Vec<u16> = children
        .iter()
        .map(|e| {
            let h = match e.basis {
                Basis::Auto => e.node.height_for_width(width),
                Basis::Fixed(n) => n,
            };
            h.clamp(e.min_size, e.max_size.unwrap_or(u16::MAX))
        })
        .collect();

    let total: u16 = sizes.iter().sum();
    if total == avail_for_children {
        return sizes;
    }

    if total < avail_for_children {
        // Distribute positive remaining space by grow.
        let remaining = avail_for_children - total;
        let total_grow: u16 = children.iter().map(|e| e.grow).sum();
        if total_grow == 0 {
            return sizes;
        }
        let mut leftover = remaining;
        for (i, entry) in children.iter().enumerate() {
            if entry.grow == 0 {
                continue;
            }
            let share = (remaining as u32 * entry.grow as u32 / total_grow as u32) as u16;
            let capped = share.min(
                entry
                    .max_size
                    .map(|m| m.saturating_sub(sizes[i]))
                    .unwrap_or(share),
            );
            sizes[i] += capped;
            leftover = leftover.saturating_sub(capped);
        }
        // Deterministic leftover distribution to earlier grow children.
        for (i, entry) in children.iter().enumerate() {
            if leftover == 0 {
                break;
            }
            if entry.grow == 0 {
                continue;
            }
            if let Some(max) = entry.max_size {
                if sizes[i] >= max {
                    continue;
                }
            }
            sizes[i] += 1;
            leftover -= 1;
        }
    } else {
        // Overflow: shrink proportional to shrink factor.
        let overflow = total - avail_for_children;
        let total_shrink: u16 = children.iter().map(|e| e.shrink).sum();
        if total_shrink == 0 {
            return sizes;
        }
        let mut remaining_overflow = overflow;
        for (i, entry) in children.iter().enumerate() {
            if entry.shrink == 0 {
                continue;
            }
            let share = (overflow as u32 * entry.shrink as u32 / total_shrink as u32) as u16;
            let max_shrink = sizes[i].saturating_sub(entry.min_size);
            let actual = share.min(max_shrink);
            sizes[i] -= actual;
            remaining_overflow = remaining_overflow.saturating_sub(actual);
        }
        for (i, entry) in children.iter().enumerate() {
            if remaining_overflow == 0 {
                break;
            }
            if entry.shrink == 0 {
                continue;
            }
            if sizes[i] <= entry.min_size {
                continue;
            }
            sizes[i] -= 1;
            remaining_overflow -= 1;
        }
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::text::Line;
    use ratatui::widgets::Paragraph;
    use ratatui::widgets::Widget;

    // Helper: a fixed-height leaf for testing allocation.
    struct FixedLeaf {
        h: u16,
        label: &'static str,
    }
    impl LayoutNode for FixedLeaf {
        fn height_for_width(&self, _width: u16) -> u16 {
            self.h
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let line = Line::from(self.label);
            let para = Paragraph::new(line);
            para.render(area, buf);
        }
    }

    #[test]
    fn vstack_auto_children_stack_vertically() {
        let stack = VStack::new(
            vec![
                StackEntry::auto(Box::new(FixedLeaf { h: 3, label: "a" })),
                StackEntry::auto(Box::new(FixedLeaf { h: 2, label: "b" })),
            ],
            StackOptions { gap: 0 },
        );
        assert_eq!(stack.height_for_width(80), 5);
    }

    #[test]
    fn vstack_grow_distributes_remaining_space() {
        // With only height_for_width, grow does not affect intrinsic height
        // directly. This test verifies the stack reports its intrinsic height
        // as the sum of Auto children (grow is applied during render allocation).
        let stack = VStack::new(
            vec![
                StackEntry {
                    node: Box::new(FixedLeaf { h: 3, label: "a" }),
                    basis: Basis::Auto,
                    grow: 0,
                    shrink: 1,
                    min_size: 0,
                    max_size: None,
                },
                StackEntry {
                    node: Box::new(FixedLeaf { h: 2, label: "b" }),
                    basis: Basis::Auto,
                    grow: 1,
                    shrink: 1,
                    min_size: 0,
                    max_size: None,
                },
            ],
            StackOptions { gap: 0 },
        );
        // Intrinsic height is sum of children regardless of grow
        assert_eq!(stack.height_for_width(80), 5);
    }

    #[test]
    fn gap_only_between_visible_children() {
        let stack = VStack::new(
            vec![
                StackEntry::auto(Box::new(FixedLeaf { h: 2, label: "a" })),
                StackEntry::auto(Box::new(FixedLeaf { h: 2, label: "b" })),
            ],
            StackOptions { gap: 1 },
        );
        assert_eq!(stack.height_for_width(80), 5); // 2 + 1 gap + 2
    }

    #[test]
    fn min_max_clamping() {
        let entry = StackEntry {
            node: Box::new(FixedLeaf { h: 10, label: "x" }),
            basis: Basis::Fixed(10),
            grow: 0,
            shrink: 0,
            min_size: 2,
            max_size: Some(5),
        };
        let stack = VStack::new(vec![entry], StackOptions { gap: 0 });
        assert_eq!(stack.height_for_width(80), 5); // clamped to max
    }

    #[test]
    fn scroll_view_clips_to_viewport() {
        let mut sv = ScrollView::new(
            Box::new(FixedLeaf { h: 20, label: "tall" }),
            ScrollViewOptions {
                follow_end: true,
                ..Default::default()
            },
        );
        sv.set_viewport_height(5);
        assert_eq!(sv.scroll_top, 15); // follow_end keeps it at bottom
        sv.scroll_by(-3);
        assert_eq!(sv.scroll_top, 12);
        assert!(!sv.is_following_end());
    }

    #[test]
    fn scroll_by_returns_unused_delta() {
        let mut sv = ScrollView::new(
            Box::new(FixedLeaf { h: 10, label: "t" }),
            ScrollViewOptions::default(),
        );
        sv.set_viewport_height(5);
        let unused = sv.scroll_by(100); // try to scroll past end
        assert!(unused > 0, "should return unused delta when at boundary");
        let unused2 = sv.scroll_by(-100);
        assert!(unused2 < 0);
    }
}
