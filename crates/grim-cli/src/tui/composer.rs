//! Input composer managing text editing, cursor navigation, and input history.
//!
//! Owns character buffer, cursor position, and history ring.
//! Ensures cursor remains within unicode scalar boundaries and survives terminal resize.

use crate::tui::kill_ring::{KillPushOpts, KillRing};
use crate::tui::undo_stack::UndoStack;

/// Snapshot of composer text and cursor for undo.
#[derive(Debug, Clone)]
struct ComposerSnapshot {
    chars: Vec<char>,
    cursor: usize,
}

/// Text composer for terminal chat input.
#[derive(Debug, Clone)]
pub struct Composer {
    /// Internal buffer as Unicode characters for correct slicing and cursor positioning.
    chars: Vec<char>,
    /// Character index of the cursor (0 <= cursor <= chars.len()).
    cursor: usize,
    /// Ring buffer of submitted lines.
    history: Vec<String>,
    /// History navigation index. None when typing a new line.
    history_idx: Option<usize>,
    /// Saved draft when user navigates into history.
    draft: String,
    /// Maximum number of lines kept in history.
    max_history: usize,
    /// Undo stack for editing actions.
    undo_stack: UndoStack<ComposerSnapshot>,
    /// Emacs kill-ring for kill and yank cycling.
    kill_ring: KillRing,
    /// Span of the last yank in chars, for yank-pop replacement.
    last_yank_span: Option<(usize, usize)>,
    /// Whether the last kill was a cut (for accumulate merging).
    last_kill_was_cut: bool,
}

impl Default for Composer {
    fn default() -> Self {
        Self::new()
    }
}

impl Composer {
    /// Create a new empty composer with a default history capacity of 100 lines.
    pub fn new() -> Self {
        Self {
            chars: Vec::new(),
            cursor: 0,
            history: Vec::new(),
            history_idx: None,
            draft: String::new(),
            max_history: 100,
            undo_stack: UndoStack::new(64),
            kill_ring: KillRing::new(),
            last_yank_span: None,
            last_kill_was_cut: false,
        }
    }

    /// Current input as a String.
    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    /// Current cursor position in characters.
    pub fn cursor_offset(&self) -> usize {
        self.cursor
    }

    /// Check if composer text is empty.
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Clear current text and reset cursor.
    pub fn clear(&mut self) {
        self.push_undo();
        self.chars.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.last_yank_span = None;
        self.last_kill_was_cut = false;
    }

    /// Set composer text explicitly and reposition cursor at end.
    pub fn set_text(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }

    /// Save the current state onto the undo stack.
    fn push_undo(&mut self) {
        let snap = ComposerSnapshot {
            chars: self.chars.clone(),
            cursor: self.cursor,
        };
        self.undo_stack.push(snap);
    }

    /// Undo the last editing action. Returns false when the stack is empty.
    pub fn undo(&mut self) -> bool {
        if let Some(snap) = self.undo_stack.pop() {
            self.chars = snap.chars;
            self.cursor = snap.cursor;
            self.last_yank_span = None;
            self.last_kill_was_cut = false;
            true
        } else {
            false
        }
    }

    /// Insert a single character at current cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.push_undo();
        self.chars.insert(self.cursor, c);
        self.cursor += 1;
        self.last_yank_span = None;
        self.last_kill_was_cut = false;
    }

    /// Delete character immediately before the cursor (Backspace).
    pub fn delete_prev_char(&mut self) {
        if self.cursor > 0 {
            self.push_undo();
            self.cursor -= 1;
            self.chars.remove(self.cursor);
            self.last_yank_span = None;
            self.last_kill_was_cut = false;
        }
    }

    /// Delete character at the cursor (Delete key).
    pub fn delete_current_char(&mut self) {
        if self.cursor < self.chars.len() {
            self.push_undo();
            self.chars.remove(self.cursor);
            self.last_yank_span = None;
            self.last_kill_was_cut = false;
        }
    }

    /// Delete word backwards from cursor (Ctrl+W).
    pub fn delete_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.push_undo();
        let mut idx = self.cursor;
        while idx > 0 && self.chars[idx - 1].is_whitespace() {
            idx -= 1;
        }
        while idx > 0 && !self.chars[idx - 1].is_whitespace() {
            idx -= 1;
        }
        self.chars.drain(idx..self.cursor);
        self.cursor = idx;
        self.last_yank_span = None;
        self.last_kill_was_cut = false;
    }

    /// Kill from cursor to end of the current logical line (Ctrl+K).
    ///
    /// Returns the killed text, if any, and pushes it to the kill-ring.
    /// Consecutive kills accumulate into one ring entry.
    pub fn kill_to_end(&mut self) -> Option<String> {
        let (row_start, row_end) = self.current_logical_line_bounds();
        let _ = row_start;
        if self.cursor >= row_end {
            return None;
        }
        self.push_undo();
        let killed: String = self.chars[self.cursor..row_end].iter().collect();
        let accumulate = self.last_kill_was_cut;
        self.kill_ring.push(
            killed.clone(),
            KillPushOpts {
                prepend: false,
                accumulate,
            },
        );
        self.chars.drain(self.cursor..row_end);
        self.last_kill_was_cut = true;
        self.last_yank_span = None;
        Some(killed)
    }

    /// Yank the most recent kill at the cursor (Ctrl+Y).
    pub fn yank(&mut self) -> bool {
        let Some(text) = self.kill_ring.peek().map(|s| s.to_string()) else {
            return false;
        };
        self.push_undo();
        let start = self.cursor;
        for ch in text.chars() {
            self.chars.insert(self.cursor, ch);
            self.cursor += 1;
        }
        self.last_yank_span = Some((start, self.cursor));
        self.last_kill_was_cut = false;
        true
    }

    /// Replace the last yank with the next kill-ring entry (Alt+Y after yank).
    pub fn yank_pop(&mut self) -> bool {
        let Some((start, end)) = self.last_yank_span else {
            return false;
        };
        if self.kill_ring.len() <= 1 {
            return false;
        }
        self.kill_ring.rotate();
        let Some(text) = self.kill_ring.peek().map(|s| s.to_string()) else {
            return false;
        };
        // Replace the previously yanked span. No undo push: yank and yank-pop
        // are one undo unit, so the push was already done at yank time.
        self.chars.drain(start..end);
        let mut insert_at = start;
        for ch in text.chars() {
            self.chars.insert(insert_at, ch);
            insert_at += 1;
        }
        // Adjust cursor and span for the new length.
        let new_end = start + text.chars().count();
        self.cursor = new_end;
        self.last_yank_span = Some((start, new_end));
        true
    }

    /// Jump forward to the next occurrence of `target` after the cursor.
    pub fn jump_forward(&mut self, target: char) -> bool {
        let target_lower = target.to_lowercase().next().unwrap_or(target);
        for (idx, &ch) in self.chars.iter().enumerate().skip(self.cursor + 1) {
            if ch.to_lowercase().next().unwrap_or(ch) == target_lower {
                self.cursor = idx;
                self.last_yank_span = None;
                self.last_kill_was_cut = false;
                return true;
            }
        }
        false
    }

    /// Jump backward to the previous occurrence of `target` before the cursor.
    pub fn jump_backward(&mut self, target: char) -> bool {
        let target_lower = target.to_lowercase().next().unwrap_or(target);
        for idx in (0..self.cursor).rev() {
            if self.chars[idx].to_lowercase().next().unwrap_or(self.chars[idx]) == target_lower {
                self.cursor = idx;
                self.last_yank_span = None;
                self.last_kill_was_cut = false;
                return true;
            }
        }
        false
    }

    /// Move cursor left by one character.
    pub fn move_cursor_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
        self.last_kill_was_cut = false;
    }

    /// Move cursor right by one character.
    pub fn move_cursor_right(&mut self) {
        if self.cursor < self.chars.len() {
            self.cursor += 1;
        }
        self.last_kill_was_cut = false;
    }

    /// Move cursor to the start of the line (Home / Ctrl+A).
    pub fn move_cursor_home(&mut self) {
        self.cursor = 0;
        self.last_kill_was_cut = false;
    }

    /// Move cursor to the end of the line (End / Ctrl+E).
    pub fn move_cursor_end(&mut self) {
        self.cursor = self.chars.len();
        self.last_kill_was_cut = false;
    }

    /// Bounds of the logical line containing the cursor, as char indices [start, end).
    fn current_logical_line_bounds(&self) -> (usize, usize) {
        let mut start = self.cursor;
        while start > 0 && self.chars[start - 1] != '\n' {
            start -= 1;
        }
        let mut end = self.cursor;
        while end < self.chars.len() && self.chars[end] != '\n' {
            end += 1;
        }
        (start, end)
    }

    /// Calculate current (row, col) position from cursor index (0-indexed).
    pub fn cursor_row_col(&self) -> (usize, usize) {
        let mut row = 0;
        let mut col = 0;
        for (i, &c) in self.chars.iter().enumerate() {
            if i == self.cursor {
                return (row, col);
            }
            if c == '\n' {
                row += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (row, col)
    }

    /// Total number of lines in current composer text.
    pub fn line_count(&self) -> usize {
        self.chars.iter().filter(|&&c| c == '\n').count() + 1
    }

    /// Move cursor up one line, or to earlier history if on the first line.
    pub fn move_cursor_up(&mut self) {
        let (row, col) = self.cursor_row_col();
        if row == 0 {
            self.history_prev();
            return;
        }
        let target_row = row - 1;
        let mut cur_row = 0;
        let mut cur_col = 0;
        let mut target_idx = 0;
        for (i, &c) in self.chars.iter().enumerate() {
            if cur_row == target_row {
                target_idx = i;
                if cur_col == col || c == '\n' {
                    break;
                }
            }
            if c == '\n' {
                cur_row += 1;
                cur_col = 0;
            } else {
                cur_col += 1;
            }
        }
        self.cursor = target_idx;
        self.last_kill_was_cut = false;
    }

    /// Move cursor down one line, or to later history if on the last line.
    pub fn move_cursor_down(&mut self) {
        let (row, col) = self.cursor_row_col();
        let total_lines = self.line_count();
        if row + 1 >= total_lines {
            self.history_next();
            return;
        }
        let target_row = row + 1;
        let mut cur_row = 0;
        let mut cur_col = 0;
        let mut target_idx = self.chars.len();
        for (i, &c) in self.chars.iter().enumerate() {
            if cur_row == target_row {
                target_idx = i;
                if cur_col == col || c == '\n' {
                    break;
                }
            }
            if c == '\n' {
                cur_row += 1;
                cur_col = 0;
            } else {
                cur_col += 1;
            }
        }
        self.cursor = target_idx;
        self.last_kill_was_cut = false;
    }

    /// Navigate to older entry in command history (Up arrow).
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        if self.history_idx.is_none() {
            self.draft = self.text();
            let last_idx = self.history.len() - 1;
            self.history_idx = Some(last_idx);
            let prev_text = self.history[last_idx].clone();
            self.set_text(&prev_text);
        } else if let Some(idx) = self.history_idx {
            if idx > 0 {
                let next_idx = idx - 1;
                self.history_idx = Some(next_idx);
                let prev_text = self.history[next_idx].clone();
                self.set_text(&prev_text);
            }
        }
    }

    /// Navigate to newer entry in command history (Down arrow).
    pub fn history_next(&mut self) {
        let Some(idx) = self.history_idx else {
            return;
        };
        if idx + 1 < self.history.len() {
            let next_idx = idx + 1;
            self.history_idx = Some(next_idx);
            let next_text = self.history[next_idx].clone();
            self.set_text(&next_text);
        } else {
            self.history_idx = None;
            let draft = std::mem::take(&mut self.draft);
            self.set_text(&draft);
        }
    }

    /// Submit current line, adding non-empty strings to history and clearing text.
    pub fn submit(&mut self) -> String {
        let text = self.text();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            if self.history.last().map(|s| s.as_str()) != Some(trimmed) {
                self.history.push(trimmed.to_string());
                if self.history.len() > self.max_history {
                    self.history.remove(0);
                }
            }
        }
        self.clear();
        self.draft.clear();
        // Clear undo on submit so the next prompt starts fresh.
        self.undo_stack.clear();
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_movement_and_insert() {
        let mut composer = Composer::new();
        composer.insert_char('h');
        composer.insert_char('l');
        composer.insert_char('o');
        composer.move_cursor_left();
        composer.move_cursor_left();
        composer.insert_char('e');
        composer.insert_char('l');
        assert_eq!(composer.text(), "hello");
        assert_eq!(composer.cursor_offset(), 3);
    }

    #[test]
    fn test_delete_word_back() {
        let mut composer = Composer::new();
        for c in "/model llama3-8b".chars() {
            composer.insert_char(c);
        }
        composer.delete_word_back();
        assert_eq!(composer.text(), "/model ");
    }

    #[test]
    fn test_history_navigation() {
        let mut composer = Composer::new();
        for c in "first prompt".chars() {
            composer.insert_char(c);
        }
        let s1 = composer.submit();
        assert_eq!(s1, "first prompt");

        for c in "second prompt".chars() {
            composer.insert_char(c);
        }
        let s2 = composer.submit();
        assert_eq!(s2, "second prompt");

        composer.history_prev();
        assert_eq!(composer.text(), "second prompt");
        composer.history_prev();
        assert_eq!(composer.text(), "first prompt");
        composer.history_next();
        assert_eq!(composer.text(), "second prompt");
    }

    #[test]
    fn test_multiline_cursor_and_navigation() {
        let mut composer = Composer::new();
        for c in "line1\nline2\nline3".chars() {
            composer.insert_char(c);
        }
        assert_eq!(composer.line_count(), 3);
        assert_eq!(composer.cursor_row_col(), (2, 5));

        composer.move_cursor_up();
        assert_eq!(composer.cursor_row_col(), (1, 5));

        composer.move_cursor_up();
        assert_eq!(composer.cursor_row_col(), (0, 5));

        composer.move_cursor_down();
        assert_eq!(composer.cursor_row_col(), (1, 5));
    }

    #[test]
    fn undo_restores_after_insert() {
        let mut c = Composer::new();
        c.insert_char('h');
        c.insert_char('i');
        assert!(c.undo());
        assert_eq!(c.text(), "h");
        assert!(c.undo());
        assert_eq!(c.text(), "");
        assert!(!c.undo());
    }

    #[test]
    fn kill_to_end_and_yank() {
        let mut c = Composer::new();
        for ch in "hello world".chars() {
            c.insert_char(ch);
        }
        // Move cursor to after "hello" (position 5)
        c.move_cursor_home();
        for _ in 0..5 {
            c.move_cursor_right();
        }
        let killed = c.kill_to_end();
        assert_eq!(killed.as_deref(), Some(" world"));
        assert_eq!(c.text(), "hello");
        assert!(c.yank());
        assert_eq!(c.text(), "hello world");
    }

    #[test]
    fn yank_pop_cycles_through_ring() {
        let mut c = Composer::new();
        // Build two kill entries by inserting, moving, and killing.
        for ch in "first".chars() {
            c.insert_char(ch);
        }
        c.move_cursor_home();
        let _ = c.kill_to_end();
        c.clear();
        // Fresh line for second kill (clear already pushed undo; start new).
        for ch in "second".chars() {
            c.insert_char(ch);
        }
        c.move_cursor_home();
        let _ = c.kill_to_end();
        // Now kill_ring has ["first", "second"] where "second" is most recent.
        c.clear();
        assert!(c.yank());
        assert_eq!(c.text(), "second");
        assert!(c.yank_pop());
        assert_eq!(c.text(), "first");
        assert!(c.yank_pop());
        assert_eq!(c.text(), "second");
    }

    #[test]
    fn jump_forward_and_backward() {
        let mut c = Composer::new();
        for ch in "hello world".chars() {
            c.insert_char(ch);
        }
        c.move_cursor_home();
        assert!(c.jump_forward('o'));
        assert_eq!(c.cursor_offset(), 4);
        assert!(c.jump_forward('o'));
        assert_eq!(c.cursor_offset(), 7);
        assert!(c.jump_backward('l'));
        assert_eq!(c.cursor_offset(), 3);
        assert!(!c.jump_forward('z'));
    }
}
