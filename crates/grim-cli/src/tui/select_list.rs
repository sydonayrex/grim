//! Keyboard-navigable selection menu for autocomplete and model picker.
//!
//! Returns styled `Line` vectors so callers can place the menu with
//! `Paragraph::new(lines)` without coupling to ratatui's `Frame`.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::tui::fuzzy::fuzzy_filter;

/// One row in the selection menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectItem {
    /// Value inserted on confirm (e.g. "model").
    pub value: String,
    /// Display label (often same as value).
    pub label: String,
    /// Optional description shown in a second column.
    pub description: Option<String>,
}

/// Visual theme for the menu.
#[derive(Debug, Clone)]
pub struct SelectListTheme {
    pub selected_prefix: String,
    pub selected_style: Style,
    pub description_style: Style,
    pub scroll_info_style: Style,
    pub no_match_text: String,
}

impl Default for SelectListTheme {
    fn default() -> Self {
        Self {
            selected_prefix: "> ".to_string(),
            selected_style: Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
            description_style: Style::default().fg(Color::DarkGray),
            scroll_info_style: Style::default().fg(Color::DarkGray),
            no_match_text: "  No matching commands".to_string(),
        }
    }
}

/// What happened after a key press.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectAction {
    None,
    Cancel,
    Confirm(SelectItem),
    SelectionChanged(SelectItem),
}

/// Stateful single-selection menu.
#[derive(Debug, Clone)]
pub struct SelectList {
    all_items: Vec<SelectItem>,
    filtered: Vec<SelectItem>,
    selected: usize,
    max_visible: usize,
    theme: SelectListTheme,
}

impl SelectList {
    /// Create a menu from `items` with room for `max_visible` rows.
    pub fn new(items: Vec<SelectItem>, max_visible: usize, theme: SelectListTheme) -> Self {
        let filtered = items.clone();
        Self {
            all_items: items,
            filtered,
            selected: 0,
            max_visible: max_visible.max(1),
            theme,
        }
    }

    /// Filter items by `query` using fuzzy matching. Resets selection to 0.
    pub fn set_filter(&mut self, query: &str) {
        if query.is_empty() {
            self.filtered = self.all_items.clone();
        } else {
            let results = fuzzy_filter(query, &self.all_items, |item| item.label.as_str());
            self.filtered = results.into_iter().map(|(item, _)| item.clone()).collect();
        }
        self.selected = 0;
    }

    /// Move selection up, wrapping to the bottom.
    pub fn move_up(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        self.selected = if self.selected == 0 {
            self.filtered.len() - 1
        } else {
            self.selected - 1
        };
    }

    /// Move selection down, wrapping to the top.
    pub fn move_down(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        self.selected = (self.selected + 1) % self.filtered.len();
    }

    /// Number of items passing the current filter.
    pub fn filtered_len(&self) -> usize {
        self.filtered.len()
    }

    /// Currently highlighted item, if any.
    pub fn selected(&self) -> Option<&SelectItem> {
        self.filtered.get(self.selected)
    }

    /// Handle a key event. Returns what the caller should do next.
    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> SelectAction {
        use crossterm::event::KeyCode;
        match key.code {
            KeyCode::Up => {
                self.move_up();
                self.selected()
                    .cloned()
                    .map(SelectAction::SelectionChanged)
                    .unwrap_or(SelectAction::None)
            }
            KeyCode::Down => {
                self.move_down();
                self.selected()
                    .cloned()
                    .map(SelectAction::SelectionChanged)
                    .unwrap_or(SelectAction::None)
            }
            KeyCode::Enter => self
                .selected()
                .cloned()
                .map(SelectAction::Confirm)
                .unwrap_or(SelectAction::None),
            KeyCode::Esc => SelectAction::Cancel,
            _ => SelectAction::None,
        }
    }

    /// Render the visible window as styled lines.
    ///
    /// When `filtered` has more items than `max_visible`, a `(n/N)` scroll
    /// indicator is appended. The window is centered on `selected` when
    /// possible, otherwise clamped to the start or end.
    pub fn render(&self, width: u16) -> Vec<Line<'static>> {
        if self.filtered.is_empty() {
            return vec![Line::from(Span::styled(
                self.theme.no_match_text.clone(),
                self.theme.scroll_info_style,
            ))];
        }
        let w = width as usize;
        // Visible window: center on selected when possible.
        let half = self.max_visible / 2;
        let start = (self.selected.saturating_sub(half))
            .min(self.filtered.len().saturating_sub(self.max_visible));
        let end = (start + self.max_visible).min(self.filtered.len());

        let mut lines = Vec::with_capacity(end - start + 1);
        for idx in start..end {
            let item = &self.filtered[idx];
            let is_selected = idx == self.selected;
            if is_selected {
                // Selected row: prefix + label, highlighted.
                let text = format!("{}{}", self.theme.selected_prefix, item.label);
                lines.push(Line::from(Span::styled(text, self.theme.selected_style)));
            } else if let Some(desc) = &item.description {
                // Unselected with description: label + dim description.
                let label_part = format!("  {}", item.label);
                // Truncate description to remaining width.
                let remaining = w.saturating_sub(label_part.len() + 2);
                let desc_text = if desc.len() > remaining {
                    format!("  {}", &desc[..remaining.saturating_sub(3)])
                } else {
                    format!("  {}", desc)
                };
                lines.push(Line::from(vec![
                    Span::raw(label_part),
                    Span::styled(desc_text, self.theme.description_style),
                ]));
            } else {
                lines.push(Line::from(Span::raw(format!("  {}", item.label))));
            }
        }
        if self.filtered.len() > self.max_visible {
            let info = format!("  ({}/{})", self.selected + 1, self.filtered.len());
            lines.push(Line::from(Span::styled(
                info,
                self.theme.scroll_info_style,
            )));
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> SelectListTheme {
        SelectListTheme::default()
    }

    fn items() -> Vec<SelectItem> {
        vec![
            SelectItem {
                value: "model".into(),
                label: "model".into(),
                description: Some("List or load a model".into()),
            },
            SelectItem {
                value: "temp".into(),
                label: "temp".into(),
                description: Some("Set temperature".into()),
            },
            SelectItem {
                value: "clear".into(),
                label: "clear".into(),
                description: None,
            },
            SelectItem {
                value: "help".into(),
                label: "help".into(),
                description: None,
            },
            SelectItem {
                value: "ctx".into(),
                label: "ctx".into(),
                description: None,
            },
        ]
    }

    #[test]
    fn empty_items_renders_no_match_line() {
        let list = SelectList::new(vec![], 5, theme());
        let lines = list.render(40);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].width() > 0);
    }

    #[test]
    fn up_wraps_to_bottom() {
        let mut list = SelectList::new(items(), 5, theme());
        // Start at 0, press Up wraps to last
        list.move_up();
        assert_eq!(list.selected().unwrap().value, "ctx");
    }

    #[test]
    fn down_wraps_to_top() {
        let mut list = SelectList::new(items(), 5, theme());
        // Move to last, then Down wraps to 0
        for _ in 0..4 {
            list.move_down();
        }
        assert_eq!(list.selected().unwrap().value, "ctx");
        list.move_down();
        assert_eq!(list.selected().unwrap().value, "model");
    }

    #[test]
    fn filter_resets_selection() {
        let mut list = SelectList::new(items(), 5, theme());
        list.move_down();
        list.move_down();
        assert_eq!(list.selected().unwrap().value, "clear");
        list.set_filter("mod");
        // After filter, only "model" matches, selection resets to 0
        assert_eq!(list.selected().unwrap().value, "model");
    }

    #[test]
    fn scroll_indicator_appears_when_clipped() {
        let list = SelectList::new(items(), 2, theme());
        let lines = list.render(40);
        // 2 visible items + 1 scroll indicator line
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn description_column_truncates_safely() {
        let list = SelectList::new(items(), 5, theme());
        // Very narrow width should not panic
        let lines = list.render(10);
        assert!(!lines.is_empty());
    }
}
