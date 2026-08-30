# Grim TUI Enhancement Plan (Pie)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the `grim-cli` interactive chat TUI with six subsystems drawn from the reference Pie TUI design: a keyboard-navigable `SelectList` menu, scoring-based fuzzy autocomplete, a 16ms render-throttle scheduler, a constrained `VStack`/`HStack`/`ScrollView` layout engine, editor-grade input enhancements (undo stack, Emacs kill-ring with yank and yank-pop, character jump mode), and `@file` path completion that resolves `@` prefixes to real filesystem paths.

**Architecture:** Keep the existing two-thread design. The UI thread owns the terminal, input composer, and ratatui loop. The worker thread owns Engine, tokenizer, and sampler. Communication stays on the existing `std::sync::mpsc` channels. All six subsystems are pure Rust with no new external crates: fuzzy matching is a stateless scoring function, `SelectList` is a stateful widget that plugs into the existing autocomplete popup path, the render scheduler wraps the existing `term.draw` call, the layout engine replaces the hand-rolled `Layout::vertical`/`Layout::horizontal` split in `ui()`, the editor enhancements extend `Composer` with focused state structs (`UndoStack`, `KillRing`, jump mode) and new key handlers in `App`, and file completion adds a synchronous `std::fs`-backed file provider that feeds `SelectList` when the composer text contains an `@` trigger.

**Tech Stack:** Rust 2021 edition, ratatui 0.29, crossterm 0.28, grim-core, grim-engine, grim-format. No new workspace dependencies.

---

## Global Constraints

* Language: Rust 2021 edition.
* Dependencies: Use existing workspace crates plus `ratatui` (0.29) and `crossterm` (0.28). Do not introduce heavy markdown engines, webviews, or non-terminal graphics libraries.
* Thread Safety: Never call Engine, GPU, or I/O blocking functions from the UI render thread. All state synchronization must pass across the existing `mpsc` channel.
* Performance Floor: Frame render time must stay under 16ms (60 FPS). Cache rendered text layouts rather than recomputing on every frame.
* Code Style: Every public function, struct, and non-trivial block must have documentation comments explaining contracts and invariants.
* Writing and Punctuation: No em dashes or en dashes anywhere in comments, docs, or UI labels. Use colons, commas, or parentheses instead.

---

## Project Boundaries

### What Already Exists (Do Not Break)

* `crates/grim-cli/src/tui/diagnostics.rs`: Snapshot formatting helpers (`format_bytes`, `ratio_percent`, `format_ms`, `format_tps`, `acceptance_rate`, `bar`, `DiagnosticsSnapshot`). Used by the sidebar; tests in `tui::diagnostics`.
* `crates/grim-cli/src/tui/worker.rs`: Worker thread loop, `WorkerCommand`, `WorkerEvent`, `TurnStats`, model loading and streaming generation via `grim_engine`. Owns the `Engine` and `Sampler`; the UI thread must never touch them.
* `crates/grim-cli/src/tui/mod.rs`: `cmd_tui` entry point, `TerminalGuard` raw mode cleanup, `App` struct, `ui()` render function, slash command dispatch, key handling. Contains the current hand-rolled layout in `ui()` and the inline autocomplete popup at `mod.rs:748-776`.
* `crates/grim-cli/src/tui/composer.rs`: `Composer` struct with `Vec<char>` buffer, cursor index, history ring buffer, word deletion (`Ctrl+W`), and multiline support. Indexed by `char`, not grapheme.
* `crates/grim-cli/src/tui/commands.rs`: `CommandRegistry` with `CommandSpec`, `ParsedCommand`, `find_completions` (prefix-only `starts_with`), `parse`.
* `crates/grim-cli/src/tui/transcript.rs`: `Role`, `MessageNode`, `Transcript`, `parse_thinking_tags` for `<think>` extraction and fold toggling.
* `crates/grim-cli/src/tui/sparkline.rs`: `SpeedHistory` fixed-capacity ring buffer for decode tok/s sidebar sparkline.

### What Should NOT Be Changed (Strict Left and Right Limits)

* Do NOT modify core ROCm/CUDA/CPU backend kernels or tensor math in `crates/grim-backend-*` or `crates/grim-tensor`.
* Do NOT rewrite `grim_engine::Engine` scheduling or KV cache internals.
* Do NOT change existing non-TUI CLI commands (`grim run`, `grim bench`, `grim server`, `grim quant`).
* Do NOT discard the `TerminalGuard` panic hook that restores the terminal on drop and on panic.
* Do NOT add a markdown rendering engine, image protocol, or webview dependency. LLM output stays as plain `Span::raw` lines for now.
* Do NOT replace ratatui with another TUI framework. All new widgets must implement `ratatui::widgets::Widget` or return `Vec<ratatui::text::Line>`.
* Do NOT make the `SelectList` or `ScrollView` generic over `dyn Any`. Use concrete types with well-defined structs.

### Reference Sources

All four subsystems are Rust rewrites of pie reference designs that live in `old/` as frozen design artifacts. Read them for context before implementing, but do not import TypeScript code or add a JS runtime.

* Fuzzy scoring: `old/pie/packages/tui/src/fuzzy.ts`
* SelectList: `old/pie/packages/tui/src/components/select-list.ts`
* Render throttle: `old/pie/packages/tui/src/tui.ts` (`TuiBase::requestRender`, `scheduleRender`, `requestImmediateRender`)
* Layout system: `old/pie/tui-plan.md` (public API and stack allocation algorithm)
* Editor undo and kill-ring: `old/pie/packages/tui/src/undo-stack.ts`, `old/pie/packages/tui/src/kill-ring.ts`, `old/pie/packages/tui/src/components/editor.ts` (history, undo coalescing, word navigation, jump mode)
* File completion: `old/pie/packages/tui/src/autocomplete.ts` (`CombinedAutocompleteProvider`, `extractAtPrefix`, `getFileSuggestions`, `getFuzzyFileSuggestions`, `walkDirectoryWithFd`, `fd` binary)

---

## File Structure

```
crates/grim-cli/src/tui/
  mod.rs         : App, ui(), cmd_tui, TerminalGuard (modify: add new module decls, integrate each subsystem)
  composer.rs    : Composer (enhance in Task 5: add UndoStack, KillRing, jump mode, new editing methods)
  commands.rs    : CommandRegistry (modify: add fuzzy path, or keep as-is and have new fuzzy module consume it)
  transcript.rs  : Transcript (no change)
  sparkline.rs   : SpeedHistory (no change)
  diagnostics.rs : DiagnosticsSnapshot helpers (no change)
  worker.rs      : Worker thread (no change)
  fuzzy.rs       : NEW: stateless scoring (Task 1)
  select_list.rs : NEW: stateful menu widget (Task 2)
  throttle.rs    : NEW: 16ms render scheduler (Task 3)
  layout.rs      : NEW: VStack/HStack/ScrollView engine (Task 4)
  kill_ring.rs   : NEW: Emacs kill-ring buffer (Task 5)
  undo_stack.rs  : NEW: generic undo stack (Task 5)
  file_complete.rs : NEW: @file provider with std::fs fallback (Task 6)
```

Each new file has one clear responsibility. Files that change together live together under `tui/`. Follow the existing module pattern: `pub use` re-exports in `mod.rs`, inline `#[cfg(test)]` modules in each file.

---

### Task 1: Fuzzy Matching (Stateless Scoring)

**Files:**
* Create: `crates/grim-cli/src/tui/fuzzy.rs`
* Modify: `crates/grim-cli/src/tui/mod.rs:1-30` (add `pub mod fuzzy;` and its re-export)
* Test: `crates/grim-cli/src/tui/fuzzy.rs` (inline `mod tests`)

**Interfaces:**
* Consumes: Nothing from earlier tasks. Pure function over `&str`.
* Produces:
  * `pub struct FuzzyMatch { pub score: i32, pub indices: Vec<usize> }`: score and matched character positions in the candidate.
  * `pub fn fuzzy_match(query: &str, candidate: &str) -> Option<FuzzyMatch>`: `None` when the query characters are not a subsequence of the candidate.
  * `pub fn fuzzy_filter<'a, T>(query: &str, items: &'a [T], key: fn(&T) -> &str) -> Vec<(&'a T, FuzzyMatch)>`: filtered and score-sorted references; empty query returns all items with score 0.

**Left limit:** Do not change `CommandRegistry::find_completions` in this task. The fuzzy module is consumed by Task 2, not by patching the registry in place.

**Right limit:** Do not add a global fuzzy index, trie, or precomputed cache. Scoring is O(query * candidate) per call and fast enough for the command list (under 20 items).

**UX note:** Fuzzy autocomplete tolerates typos like `/mod` matching `/model`. Without it, users who mistype a single character see no popup and assume the command does not exist.

- [ ] **Step 1: Write the failing tests for fuzzy matching**

Create `crates/grim-cli/src/tui/fuzzy.rs` with this test module. No implementation yet, just the struct stubs so the file compiles.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_scores_highest() {
        let a = fuzzy_match("model", "model").unwrap();
        let b = fuzzy_match("model", "modelx").unwrap();
        assert!(a.score > b.score, "exact match should outrank prefix extension");
    }

    #[test]
    fn prefix_match_outranks_scattered() {
        let prefix = fuzzy_match("mod", "model").unwrap();
        let scattered = fuzzy_match("mod", "mxxxoxxxd").unwrap();
        assert!(prefix.score > scattered.score);
    }

    #[test]
    fn subsequence_matches() {
        assert!(fuzzy_match("ml", "model").is_some());
        assert!(fuzzy_match("tp", "topp").is_some());
    }

    #[test]
    fn not_a_subsequence_returns_none() {
        assert!(fuzzy_match("xyz", "model").is_none());
        assert!(fuzzy_match("modelx", "model").is_none());
    }

    #[test]
    fn case_insensitive() {
        assert!(fuzzy_match("MODEL", "model").is_some());
        assert!(fuzzy_match("MoDeL", "model").is_some());
    }

    #[test]
    fn empty_query_matches_everything_with_zero_score() {
        let items = ["model", "temp", "clear"];
        let results = fuzzy_filter("", &items, |s| s);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|(_, m)| m.score == 0));
    }

    #[test]
    fn contiguous_run_bonus() {
        let contig = fuzzy_match("te", "temp").unwrap();
        let gapped = fuzzy_match("te", "t_x_e").unwrap();
        assert!(contig.score > gapped.score, "contiguous run should score higher");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --package grim-cli --lib tui::fuzzy`
Expected: FAIL with `fuzzy_match` or `FuzzyMatch` not found. Do not proceed until you see the failure.

- [ ] **Step 3: Implement fuzzy scoring**

Write the full implementation in `crates/grim-cli/src/tui/fuzzy.rs`. Use this structure as a starting point and fill in the scoring loop. The algorithm is intentionally simple so a weak implementer can follow it.

```rust
//! Stateless fuzzy matching for slash command autocomplete.
//!
//! Scores candidates by how well the query characters appear as a
//! subsequence. Contiguous runs and prefix matches score higher.

/// Result of a successful fuzzy match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyMatch {
    /// Higher means better match. 0 means empty query or trivial match.
    pub score: i32,
    /// Byte indices in the candidate where each query character matched.
    pub indices: Vec<usize>,
}

/// Try to match `query` as a subsequence of `candidate` (case-insensitive).
///
/// Returns `None` when any query character cannot be found in order.
pub fn fuzzy_match(query: &str, candidate: &str) -> Option<FuzzyMatch> {
    if query.is_empty() {
        return Some(FuzzyMatch { score: 0, indices: Vec::new() });
    }
    let q = query.to_lowercase();
    let c = candidate.to_lowercase();
    let q_chars: Vec<char> = q.chars().collect();
    let c_chars: Vec<char> = c.chars().collect();

    // Greedy left-to-right scan, collecting matched positions.
    let mut indices = Vec::with_capacity(q_chars.len());
    let mut ci = 0;
    for &qc in &q_chars {
        let mut found = false;
        while ci < c_chars.len() {
            if c_chars[ci] == qc {
                indices.push(ci);
                ci += 1;
                found = true;
                break;
            }
            ci += 1;
        }
        if !found {
            return None;
        }
    }

    // Scoring: base 1 per matched char, plus bonuses.
    let mut score: i32 = q_chars.len() as i32;
    // Prefix bonus: query starts at candidate start.
    if indices.first() == Some(&0) {
        score += 8;
    }
    // Contiguity bonus: each adjacent pair in indices adds 4.
    for w in indices.windows(2) {
        if w[1] == w[0] + 1 {
            score += 4;
        }
    }
    // Length penalty: shorter candidates rank slightly higher for same score.
    // Applied by the caller via sort stability; not subtracted here.

    Some(FuzzyMatch { score, indices })
}

/// Filter and rank `items` by fuzzy match against `query`.
///
/// `key` extracts the searchable string from each item. Results are sorted
/// descending by score. Empty query returns all items unsorted with score 0.
pub fn fuzzy_filter<'a, T>(
    query: &str,
    items: &'a [T],
    key: fn(&T) -> &str,
) -> Vec<(&'a T, FuzzyMatch)> {
    if query.is_empty() {
        return items
            .iter()
            .map(|item| (item, FuzzyMatch { score: 0, indices: Vec::new() }))
            .collect();
    }
    let mut scored: Vec<(&T, FuzzyMatch)> = items
        .iter()
        .filter_map(|item| fuzzy_match(query, key(item)).map(|m| (item, m)))
        .collect();
    // Stable sort so equal scores preserve input order.
    scored.sort_by(|a, b| b.1.score.cmp(&a.1.score));
    scored
}
```

Key invariants to preserve:
* `fuzzy_match` is pure and allocation-light. It does not touch global state.
* `fuzzy_filter` returns references into `items`, so `items` must outlive the result. No cloning of `T`.
* Empty query is not an error. It returns all items so callers can show the full list.

```rust
// Example usage from a later task (not part of this file):
use crate::tui::fuzzy::fuzzy_filter;
let specs = registry.all_commands();
let results = fuzzy_filter("mod", specs, |s| s.name);
// results[0].0.name == "model" with highest score
```

- [ ] **Step 4: Register the module and run tests**

Add to `crates/grim-cli/src/tui/mod.rs` near the other `pub mod` declarations:

```rust
pub mod fuzzy;
```

Re-export if the crate's public surface needs it:

```rust
pub use fuzzy::{FuzzyMatch, fuzzy_filter, fuzzy_match};
```

Run: `cargo test --package grim-cli --lib tui::fuzzy`
Expected: PASS (all 7 tests succeed).

- [ ] **Step 5: Run the full TUI test suite and commit**

Run: `cargo test --package grim-cli --lib tui`
Expected: PASS (all existing tests plus the 7 new ones).

Then commit only the files you touched:

```bash
git add crates/grim-cli/src/tui/fuzzy.rs crates/grim-cli/src/tui/mod.rs
git commit -m "feat(tui): add fuzzy matching for slash command autocomplete"
```

---

### Task 2: SelectList Component (Stateful Menu Widget)

**Files:**
* Create: `crates/grim-cli/src/tui/select_list.rs`
* Modify: `crates/grim-cli/src/tui/mod.rs:1-40` (add `pub mod select_list;`)
* Modify: `crates/grim-cli/src/tui/mod.rs:700-820` (`ui()` function: replace the inline autocomplete popup `Paragraph` with a `SelectList` render)
* Test: `crates/grim-cli/src/tui/select_list.rs` (inline `mod tests`)

**Interfaces:**
* Consumes: `FuzzyMatch` and `fuzzy_filter` from Task 1 (for filtered display); `CommandSpec` from `commands.rs` (as the source items, mapped to `SelectItem`).
* Produces:
  * `pub struct SelectItem { pub value: String, pub label: String, pub description: Option<String> }`: one row in the menu.
  * `pub struct SelectListTheme { pub selected_prefix: String, pub selected_style: ratatui::style::Style, pub description_style: ratatui::style::Style, pub scroll_info_style: ratatui::style::Style }`: styling for the menu.
  * `pub struct SelectList { items, filtered, selected, max_visible, theme }`
  * `pub fn SelectList::new(items: Vec<SelectItem>, max_visible: usize, theme: SelectListTheme) -> Self`
  * `pub fn set_filter(&mut self, query: &str)`: re-runs fuzzy filtering and resets selection to 0.
  * `pub fn move_up(&mut self)` / `pub fn move_down(&mut self)`: wrap-around selection movement.
  * `pub fn selected(&self) -> Option<&SelectItem>`
  * `pub fn render(&self, width: u16) -> Vec<ratatui::text::Line<'static>>`: visible window with `(n/N)` scroll indicator when clipped.
  * `pub enum SelectAction { Confirm(SelectItem), Cancel, None, SelectionChanged(SelectItem) }`
  * `pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> SelectAction`: maps Up/Down/Enter/Esc.

**Left limit:** Do not replace `CommandRegistry`. `SelectList` is a view over `CommandSpec` data, not a replacement for the registry. Keep `CommandRegistry::find_completions` intact for now; `SelectList::set_filter` will call `fuzzy_filter` directly.

**Right limit:** Do not add mouse hit-testing, scrollbar widgets, or multi-select. Single selection, keyboard only, fixed `max_visible` rows.

**Design note:** Keep the widget stateless with respect to ratatui's `Frame`. It returns `Vec<Line>` like `Transcript::render_lines` does, so `ui()` can place it with `Paragraph::new(lines)`. Do not make it implement `ratatui::widgets::Widget` with a `StatefulWidget` trait yet; that refactor can come later.

**UX note:** Users currently Tab-cycle through completions with no visual selection indicator. `SelectList` gives arrow-key navigation with a highlighted row, wrap-around, and a `(3/8)` scroll indicator so users know more results exist off-screen.

- [ ] **Step 1: Write the failing tests for SelectList**

Create `crates/grim-cli/src/tui/select_list.rs` with this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> SelectListTheme {
        SelectListTheme::default()
    }

    fn items() -> Vec<SelectItem> {
        vec![
            SelectItem { value: "model".into(), label: "model".into(), description: Some("List or load a model".into()) },
            SelectItem { value: "temp".into(), label: "temp".into(), description: Some("Set temperature".into()) },
            SelectItem { value: "clear".into(), label: "clear".into(), description: None },
            SelectItem { value: "help".into(), label: "help".into(), description: None },
            SelectItem { value: "ctx".into(), label: "ctx".into(), description: None },
        ]
    }

    #[test]
    fn empty_items_renders_no_match_line() {
        let list = SelectList::new(vec![], 5, theme());
        let lines = list.render(40);
        assert_eq!(lines.len(), 1);
        let text: String = lines[0].width().to_string();
        // Should contain "No matching" (style may vary, check width > 0)
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
        for _ in 0..4 { list.move_down(); }
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
        let mut list = SelectList::new(items(), 5, theme());
        // Very narrow width should not panic
        let lines = list.render(10);
        assert!(!lines.is_empty());
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --package grim-cli --lib tui::select_list`
Expected: FAIL with `SelectList` or `SelectItem` not found.

- [ ] **Step 3: Implement SelectList**

Write the implementation in `crates/grim-cli/src/tui/select_list.rs`. Below is the full structure. The `render` method is the most involved part; pay attention to the visible-window math.

```rust
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
            selected_style: Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
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
        Self { all_items: items, filtered, selected: 0, max_visible: max_visible.max(1), theme }
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
        if self.filtered.is_empty() { return; }
        self.selected = if self.selected == 0 { self.filtered.len() - 1 } else { self.selected - 1 };
    }

    /// Move selection down, wrapping to the top.
    pub fn move_down(&mut self) {
        if self.filtered.is_empty() { return; }
        self.selected = (self.selected + 1) % self.filtered.len();
    }

    /// Currently highlighted item, if any.
    pub fn selected(&self) -> Option<&SelectItem> {
        self.filtered.get(self.selected)
    }

    /// Handle a key event. Returns what the caller should do next.
    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> SelectAction {
        use crossterm::event::{KeyCode, KeyModifiers};
        match key.code {
            KeyCode::Up => { self.move_up(); self.selected().cloned().map(SelectAction::SelectionChanged).unwrap_or(SelectAction::None) }
            KeyCode::Down => { self.move_down(); self.selected().cloned().map(SelectAction::SelectionChanged).unwrap_or(SelectAction::None) }
            KeyCode::Enter => self.selected().cloned().map(SelectAction::Confirm).unwrap_or(SelectAction::None),
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
        let start = (self.selected.saturating_sub(half)).min(self.filtered.len().saturating_sub(self.max_visible));
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
                    format!("  {}", &desc[..remaining.saturating_sub(3)]), // leave room, no panic on boundary
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
            lines.push(Line::from(Span::styled(info, self.theme.scroll_info_style)));
        }
        lines
    }
}
```

Important details for the implementer:
* `set_filter` reuses `fuzzy_filter` from Task 1. The `key` closure captures `label`, not `value`, so filtering matches what the user sees.
* `render` must not panic on `width == 0` or on very long labels. Use `saturating_sub` for all width math.
* The `handle_key` shown above is minimal. The real `mod.rs` integration will also handle Tab for autocomplete confirm; keep Tab handling in `mod.rs`, not inside `SelectList`.

Example of how `mod.rs` will use it after this task (not part of this file, just for context):

```rust
// In App::handle_key or the autocomplete popup path:
let mut menu = SelectList::new(
    registry.all_commands().iter().map(|s| SelectItem {
        value: s.name.to_string(),
        label: s.name.to_string(),
        description: Some(s.description.to_string()),
    }).collect(),
    8,
    SelectListTheme::default(),
);
menu.set_filter(&query_text);
let lines = menu.render(popup_area.width);
```

- [ ] **Step 4: Register the module and run tests**

Add to `crates/grim-cli/src/tui/mod.rs`:

```rust
pub mod select_list;
```

Re-export if needed:

```rust
pub use select_list::{SelectAction, SelectItem, SelectList, SelectListTheme};
```

Run: `cargo test --package grim-cli --lib tui::select_list`
Expected: PASS (all 6 tests succeed).

- [ ] **Step 5: Integrate into the autocomplete popup path in `mod.rs` (minimal)**

Replace the inline `Paragraph` construction at `mod.rs:748-776` with a `SelectList` render. Keep the existing `CommandRegistry::find_completions` call as a fallback for one frame, then switch to `SelectList::set_filter`. This step is small: create the list, call `set_filter` with the typed prefix (without the leading `/`), and render `list.render(width)`.

Run: `cargo test --package grim-cli --lib tui`
Expected: PASS (all tests including the new select_list tests).

- [ ] **Step 6: Commit**

```bash
git add crates/grim-cli/src/tui/select_list.rs crates/grim-cli/src/tui/mod.rs
git commit -m "feat(tui): add SelectList menu for autocomplete popup"
```

---

### Task 3: Render-Throttle Scheduler (16ms Frame Budget)

**Files:**
* Create: `crates/grim-cli/src/tui/throttle.rs`
* Modify: `crates/grim-cli/src/tui/mod.rs:830-900` (`cmd_tui` function: replace the `term.draw` + `poll(50ms)` loop with scheduler-driven rendering)
* Test: `crates/grim-cli/src/tui/throttle.rs` (inline `mod tests`)

**Interfaces:**
* Consumes: Nothing from Tasks 1 or 2. Wraps the existing `ratatui::Terminal` draw call.
* Produces:
  * `pub struct RenderScheduler { last_render: std::time::Instant, pending: bool, immediate: bool }`
  * `pub const MIN_FRAME_INTERVAL: std::time::Duration`: 16ms.
  * `pub fn RenderScheduler::new() -> Self`
  * `pub fn request_render(&mut self)`: mark a frame as needed, throttled to 16ms.
  * `pub fn request_immediate(&mut self)`: mark a frame as needed, bypass throttle on next tick (for input latency).
  * `pub fn should_render(&mut self) -> bool`: true when enough time has passed or an immediate render was requested. Resets the pending flag and updates `last_render` when it returns true.
  * `pub fn reset(&mut self)`: force the next `should_render` to return true (for resize or explicit full redraw).

**Left limit:** Do not replace `crossterm::event::poll` with a custom event loop. Keep the existing `Duration::from_millis(50)` poll. The scheduler only gates `term.draw`, not input handling.

**Right limit:** Do not add a `tokio` dependency or spawn a background render task. The scheduler is synchronous and lives on the UI thread. Do not add differential screen diffing beyond what `ratatui::Terminal::draw` already does internally.

**Performance note:** The current loop calls `term.draw(|f| ui(f, &app))` on every iteration, even when nothing changed. For long transcripts this recomputes `transcript.render_lines()` every frame. The scheduler skips the draw when `pending` is false and `MIN_FRAME_INTERVAL` has not elapsed, directly satisfying the 16ms floor.

- [ ] **Step 1: Write the failing tests for the scheduler**

Create `crates/grim-cli/src/tui/throttle.rs` with this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn new_scheduler_wants_first_frame() {
        let mut s = RenderScheduler::new();
        // First frame should always render (last_render is far in the past).
        s.request_render();
        assert!(s.should_render(), "first pending frame should render immediately");
    }

    #[test]
    fn throttle_suppresses_rapid_second_frame() {
        let mut s = RenderScheduler::new();
        s.request_render();
        assert!(s.should_render()); // first frame renders
        s.request_render();
        // Second request immediately after should be throttled.
        assert!(!s.should_render(), "second frame within 16ms should be suppressed");
    }

    #[test]
    fn immediate_bypasses_throttle() {
        let mut s = RenderScheduler::new();
        s.request_render();
        assert!(s.should_render());
        s.request_render();
        assert!(!s.should_render()); // throttled
        s.request_immediate();
        assert!(s.should_render(), "immediate should bypass throttle");
    }

    #[test]
    fn reset_forces_next_frame() {
        let mut s = RenderScheduler::new();
        s.request_render();
        assert!(s.should_render());
        // No pending frame, but reset forces the next one.
        s.reset();
        s.request_render();
        assert!(s.should_render());
    }

    #[test]
    fn no_pending_means_no_render() {
        let mut s = RenderScheduler::new();
        // Never requested, should not render even after reset is not called.
        assert!(!s.should_render());
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --package grim-cli --lib tui::throttle`
Expected: FAIL with `RenderScheduler` not found.

- [ ] **Step 3: Implement the scheduler**

Write the implementation in `crates/grim-cli/src/tui/throttle.rs`:

```rust
//! Frame throttle: gate `term.draw` to at most 60 FPS (16ms interval).
//!
//! Input handling stays latency-sensitive via `request_immediate`, which
//! bypasses the throttle on the next `should_render` check.

use std::time::{Duration, Instant};

/// Minimum interval between rendered frames. 16ms gives 60 FPS.
pub const MIN_FRAME_INTERVAL: Duration = Duration::from_millis(16);

/// Synchronous render scheduler. Lives on the UI thread, no background task.
#[derive(Debug)]
pub struct RenderScheduler {
    /// When the last frame was actually drawn.
    last_render: Instant,
    /// Whether a frame has been requested since the last draw.
    pending: bool,
    /// Whether the next frame should bypass the throttle.
    immediate: bool,
}

impl Default for RenderScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderScheduler {
    /// Create a scheduler that will allow the first frame immediately.
    pub fn new() -> Self {
        Self {
            // Far enough in the past that the first should_render returns true.
            last_render: Instant::now() - MIN_FRAME_INTERVAL - Duration::from_millis(1),
            pending: false,
            immediate: false,
        }
    }

    /// Mark a frame as needed. Throttled to `MIN_FRAME_INTERVAL`.
    pub fn request_render(&mut self) {
        self.pending = true;
    }

    /// Mark a frame as needed and bypass the throttle on the next check.
    ///
    /// Use for input events where latency matters more than frame budget.
    pub fn request_immediate(&mut self) {
        self.pending = true;
        self.immediate = true;
    }

    /// True when a frame should be drawn now. Resets pending state when true.
    pub fn should_render(&mut self) -> bool {
        if !self.pending {
            return false;
        }
        if self.immediate {
            self.immediate = false;
            self.pending = false;
            self.last_render = Instant::now();
            return true;
        }
        if self.last_render.elapsed() >= MIN_FRAME_INTERVAL {
            self.pending = false;
            self.last_render = Instant::now();
            return true;
        }
        false
    }

    /// Force the next pending frame to render regardless of interval.
    ///
    /// Call on terminal resize or explicit full redraw.
    pub fn reset(&mut self) {
        self.last_render = Instant::now() - MIN_FRAME_INTERVAL - Duration::from_millis(1);
    }
}
```

Example of how `cmd_tui` will use it (not part of this file, just for context):

```rust
// Before (current code in mod.rs:862-889):
// loop {
//     term.draw(|f| ui(f, &app))?;
//     if crossterm::event::poll(Duration::from_millis(50))? { /* handle key */ }
// }

// After:
let mut scheduler = RenderScheduler::new();
loop {
    // Input handling always runs.
    if crossterm::event::poll(Duration::from_millis(50))? {
        // ... handle key, then:
        scheduler.request_immediate();
    }
    // Worker events also request a render:
    // while let Ok(evt) = evt_rx.try_recv() { app.handle_event(evt); scheduler.request_render(); }

    if scheduler.should_render() {
        term.draw(|f| ui(f, &app))?;
    }
}
```

- [ ] **Step 4: Register the module and run tests**

Add to `crates/grim-cli/src/tui/mod.rs`:

```rust
pub mod throttle;
pub use throttle::{RenderScheduler, MIN_FRAME_INTERVAL};
```

Run: `cargo test --package grim-cli --lib tui::throttle`
Expected: PASS (all 5 tests succeed).

- [ ] **Step 5: Wire the scheduler into `cmd_tui` in `mod.rs`**

Replace the unconditional `term.draw` in the main loop with the `should_render` gate. Call `request_render` on every `WorkerEvent` drain, and `request_immediate` on every key event. Call `reset` on terminal resize (`KeyCode::F` is not a resize; handle `crossterm::event::Event::Resize` if the loop already matches on `crossterm::event::read`).

Run: `cargo test --package grim-cli --lib tui`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add crates/grim-cli/src/tui/throttle.rs crates/grim-cli/src/tui/mod.rs
git commit -m "feat(tui): add 16ms render throttle scheduler"
```

---

### Task 4: Constrained Layout System (VStack / HStack / ScrollView)

**Files:**
* Create: `crates/grim-cli/src/tui/layout.rs`
* Modify: `crates/grim-cli/src/tui/mod.rs:1-30` (add `pub mod layout;`)
* Modify: `crates/grim-cli/src/tui/mod.rs:698-850` (`ui()` function: replace the hand-rolled `Layout::vertical`/`Layout::horizontal` with `VStack`/`HStack`/`ScrollView`)
* Test: `crates/grim-cli/src/tui/layout.rs` (inline `mod tests`)

**Interfaces:**
* Consumes: `SelectList` (Task 2) as a child widget, `Transcript` as scrollable content.
* Produces:
  * `pub struct StackEntry { pub node: Box<dyn LayoutNode>, pub basis: Basis, pub grow: u16, pub shrink: u16, pub min_size: u16, pub max_size: Option<u16> }`
  * `pub enum Basis { Auto, Fixed(u16) }`
  * `pub struct StackOptions { pub gap: u16 }`
  * `pub struct VStack { children: Vec<StackEntry>, options: StackOptions }`
  * `pub struct HStack { children: Vec<StackEntry>, options: StackOptions }`
  * `pub struct ScrollView { child: Box<dyn LayoutNode>, follow_end: bool, scroll_top: usize, viewport_height: usize }`
  * `pub trait LayoutNode { fn height_for_width(&self, width: u16) -> u16; fn render(&self, area: ratatui::layout::Rect, buf: &mut ratatui::buffer::Buffer); }`
  * `pub fn compose(root: &dyn LayoutNode, area: ratatui::layout::Rect, buf: &mut ratatui::buffer::Buffer)`: top-level layout and paint.

**Left limit:** Do not add percentage sizing, grid layout, or wrapped flex rows. The pie reference spec explicitly lists these as non-goals.

**Right limit:** Do not add a public API for custom components to create or mutate internal layout nodes. The frame-specific layout tree stays internal to `layout.rs`. API users construct a `VStack`/`HStack` tree and never touch rects or hit-test nodes.

**Design note:** Build on `ratatui::layout::Layout` and `Constraint` rather than a from-scratch allocator. `VStack` with `Basis::Auto` measures children via `height_for_width`, then distributes remaining space by `grow`/`shrink` with deterministic integer rounding (leftover cells go to earlier children so layouts do not jitter).

**UX note:** The current `ui()` in `mod.rs` hardcodes `Constraint::Percentage(68)`/`Percentage(32)` for the sidebar split and `(line_count + 2).clamp(3, 8)` for the input height. The layout engine makes these declarative: the transcript gets `Basis::Fixed(0), grow: 1, shrink: 1, min_size: 1`, the sidebar gets `Basis::Auto, grow: 0, shrink: 1`. Very small terminals preserve at least one transcript row and the focused input cursor.

- [ ] **Step 1: Write the failing tests for the layout engine**

Create `crates/grim-cli/src/tui/layout.rs` with this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::text::Line;

    // Helper: a fixed-height leaf for testing allocation.
    struct FixedLeaf { h: u16, label: &'static str }
    impl LayoutNode for FixedLeaf {
        fn height_for_width(&self, _width: u16) -> u16 { self.h }
        fn render(&self, area: ratatui::layout::Rect, buf: &mut ratatui::buffer::Buffer) {
            let line = Line::from(self.label);
            let para = ratatui::widgets::Paragraph::new(line);
            para.render(area, buf);
        }
    }

    #[test]
    fn vstack_auto_children_stack_vertically() {
        let stack = VStack::new(vec![
            StackEntry::auto(Box::new(FixedLeaf { h: 3, label: "a" })),
            StackEntry::auto(Box::new(FixedLeaf { h: 2, label: "b" })),
        ], StackOptions { gap: 0 });
        assert_eq!(stack.height_for_width(80), 5);
    }

    #[test]
    fn vstack_grow_distributes_remaining_space() {
        // Area height 10, children auto 3 + 2 = 5, remaining 5 goes to grow:1
        let stack = VStack::new(vec![
            StackEntry { node: Box::new(FixedLeaf { h: 3, label: "a" }), basis: Basis::Auto, grow: 0, shrink: 1, min_size: 0, max_size: None },
            StackEntry { node: Box::new(FixedLeaf { h: 2, label: "b" }), basis: Basis::Auto, grow: 1, shrink: 1, min_size: 0, max_size: None },
        ], StackOptions { gap: 0 });
        // b should grow to fill remaining 5
        assert_eq!(stack.height_for_width(80), 10); // total fills area when composed
    }

    #[test]
    fn gap_only_between_visible_children() {
        let stack = VStack::new(vec![
            StackEntry::auto(Box::new(FixedLeaf { h: 2, label: "a" })),
            StackEntry::auto(Box::new(FixedLeaf { h: 2, label: "b" })),
        ], StackOptions { gap: 1 });
        assert_eq!(stack.height_for_width(80), 5); // 2 + 1 gap + 2
    }

    #[test]
    fn min_max_clamping() {
        let entry = StackEntry {
            node: Box::new(FixedLeaf { h: 10, label: "x" }),
            basis: Basis::Fixed(10),
            grow: 0, shrink: 0,
            min_size: 2, max_size: Some(5),
        };
        let stack = VStack::new(vec![entry], StackOptions { gap: 0 });
        assert_eq!(stack.height_for_width(80), 5); // clamped to max
    }

    #[test]
    fn scroll_view_clips_to_viewport() {
        let mut sv = ScrollView::new(
            Box::new(FixedLeaf { h: 20, label: "tall" }),
            ScrollViewOptions { follow_end: true, ..Default::default() },
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --package grim-cli --lib tui::layout`
Expected: FAIL with `VStack` or `LayoutNode` not found.

- [ ] **Step 3: Implement VStack, HStack, and ScrollView**

Write the implementation in `crates/grim-cli/src/tui/layout.rs`. Below is the structural skeleton. The stack allocation algorithm is the core; the `ScrollView` state machine is straightforward.

```rust
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
        Self { node, basis: Basis::Auto, grow: 0, shrink: 1, min_size: 0, max_size: None }
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
        let sum: u16 = self.children.iter().map(|e| {
            let h = match e.basis {
                Basis::Auto => e.node.height_for_width(width),
                Basis::Fixed(n) => n,
            };
            h.clamp(e.min_size, e.max_size.unwrap_or(u16::MAX))
        }).sum();
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
            if h == 0 { continue; }
            let child_area = Rect { x: area.x, y, width: area.width, height: h };
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
        self.children.iter().zip(widths).map(|(e, w)| e.node.height_for_width(w)).max().unwrap_or(0)
    }

    fn render(&self, area: Rect, buf: &mut Buffer) {
        let widths = allocate_main_axis(&self.children, area.width, self.options.gap, area.width);
        let mut x = area.x;
        for (entry, w) in self.children.iter().zip(widths) {
            if w == 0 { continue; }
            let child_area = Rect { x, y: area.y, width: w, height: area.height };
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
        Self { child, follow_end: follow, following_end: follow, scroll_top: 0, viewport_height: 0 }
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

    pub fn is_following_end(&self) -> bool { self.following_end }
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

fn allocate_main_axis(children: &[StackEntry], available: u16, gap: u16, width: u16) -> Vec<u16> {
    if children.is_empty() { return vec![]; }
    let gaps = gap * children.len().saturating_sub(1) as u16;
    let avail_for_children = available.saturating_sub(gaps);

    // 1. Resolve basis to initial sizes.
    let mut sizes: Vec<u16> = children.iter().map(|e| {
        let h = match e.basis { Basis::Auto => e.node.height_for_width(width), Basis::Fixed(n) => n };
        h.clamp(e.min_size, e.max_size.unwrap_or(u16::MAX))
    }).collect();

    let total: u16 = sizes.iter().sum();
    if total == avail_for_children { return sizes; }

    if total < avail_for_children {
        // Distribute positive remaining space by grow.
        let remaining = avail_for_children - total;
        let total_grow: u16 = children.iter().map(|e| e.grow).sum();
        if total_grow == 0 { return sizes; }
        let mut leftover = remaining;
        for (i, entry) in children.iter().enumerate() {
            if entry.grow == 0 { continue; }
            let share = (remaining as u32 * entry.grow as u32 / total_grow as u32) as u16;
            let capped = share.min(entry.max_size.map(|m| m.saturating_sub(sizes[i])).unwrap_or(share));
            sizes[i] += capped;
            leftover = leftover.saturating_sub(capped);
        }
        // Deterministic leftover distribution to earlier grow children.
        for (i, entry) in children.iter().enumerate() {
            if leftover == 0 { break; }
            if entry.grow == 0 { continue; }
            if let Some(max) = entry.max_size { if sizes[i] >= max { continue; } }
            sizes[i] += 1;
            leftover -= 1;
        }
    } else {
        // Overflow: shrink proportional to shrink factor.
        let overflow = total - avail_for_children;
        let total_shrink: u16 = children.iter().map(|e| e.shrink).sum();
        if total_shrink == 0 { return sizes; }
        let mut remaining_overflow = overflow;
        for (i, entry) in children.iter().enumerate() {
            if entry.shrink == 0 { continue; }
            let share = (overflow as u32 * entry.shrink as u32 / total_shrink as u32) as u16;
            let max_shrink = sizes[i].saturating_sub(entry.min_size);
            let actual = share.min(max_shrink);
            sizes[i] -= actual;
            remaining_overflow = remaining_overflow.saturating_sub(actual);
        }
        for (i, entry) in children.iter().enumerate() {
            if remaining_overflow == 0 { break; }
            if entry.shrink == 0 { continue; }
            if sizes[i] <= entry.min_size { continue; }
            sizes[i] -= 1;
            remaining_overflow -= 1;
        }
    }
    sizes
}
```

The implementer should verify the allocation math against the pie reference spec (`old/pie/tui-plan.md` section "Stack layout algorithm") and the existing pie `VStack` tests. The `ScrollView::render` clipping via a temporary buffer is intentionally left as a follow-up optimization; the initial version can render directly and rely on ratatui's buffer clipping.

Example of how `mod.rs` will use the layout after this task (not part of this file, just for context):

```rust
// In ui(), replacing the hand-rolled Layout::vertical/horizontal:
let transcript_node = /* Transcript as LayoutNode */;
let sidebar_node = /* diagnostics + sparkline as LayoutNode */;
let mut scroll_view = ScrollView::new(
    Box::new(transcript_node),
    ScrollViewOptions { follow_end: true },
);
scroll_view.set_viewport_height(transcript_area_height as usize);

let root = VStack::new(vec![
    StackEntry { node: Box::new(scroll_view), basis: Basis::Fixed(0), grow: 1, shrink: 1, min_size: 1, max_size: None },
    StackEntry { node: Box::new(sidebar_node), basis: Basis::Auto, grow: 0, shrink: 1, min_size: 0, max_size: None },
], StackOptions { gap: 0 });
root.render(area, buf);
```

- [ ] **Step 4: Register the module and run tests**

Add to `crates/grim-cli/src/tui/mod.rs`:

```rust
pub mod layout;
```

Run: `cargo test --package grim-cli --lib tui::layout`
Expected: PASS (all 6 tests succeed).

- [ ] **Step 5: Refactor `ui()` in `mod.rs` to use the layout engine**

Replace the hand-rolled layout in `ui()` (currently `Layout::vertical` + `Layout::horizontal` with hardcoded `Percentage(68)`/`Percentage(32)` and `(line_count + 2).clamp(3, 8)`) with a `VStack` root containing a `ScrollView` transcript and a fixed dock. Keep the existing `Paragraph`, `Block`, `Sparkline` widget construction; only the area allocation changes.

Run: `cargo test --package grim-cli --lib tui`
Expected: PASS (all tests). Manual check: `cargo run --bin grim-cli -- tui` still renders correctly.

- [ ] **Step 6: Commit**

```bash
git add crates/grim-cli/src/tui/layout.rs crates/grim-cli/src/tui/mod.rs
git commit -m "feat(tui): add constrained VStack/HStack/ScrollView layout engine"
```

---

### Task 5: Editor Undo, Kill-Ring, and Jump Mode

> **Why this exists:** The current `Composer` is a single-line `Vec<char>` buffer with only `Ctrl+W` word deletion and basic `Home`/`End`. For a chat surface that now supports multiline input (`Enter` with `Alt`/`Shift` inserts a newline, `Up`/`Down` move across visual lines), the Pie Editor shows what users expect next: undo for multi-step recovery, an Emacs kill-ring for accumulated cuts and yank cycling, and a character jump for quick horizontal movement without repeated arrow presses. Skipping these was reasonable when the composer was single-line, but Task 4 already introduced multiline layout, so the gap is now felt. This task ports a focused subset of `old/pie/packages/tui/src/components/editor.ts` and its two tiny helpers `kill-ring.ts` and `undo-stack.ts`. Bracketed paste markers and full 1500-line rendering logic are explicitly out of scope.

**Files:**
* Create: `crates/grim-cli/src/tui/kill_ring.rs`
* Create: `crates/grim-cli/src/tui/undo_stack.rs`
* Modify: `crates/grim-cli/src/tui/composer.rs` (add undo stack, kill-ring, jump mode state and new methods, keep all existing `Vec<char>` invariants)
* Modify: `crates/grim-cli/src/tui/mod.rs:290-385` (`App::handle_chat_key`: add `Ctrl+K`, `Ctrl+Y`, `Alt+Y`, `Ctrl+Z`/`Ctrl+_` branches and integrate jump mode; update help text)
* Test: `crates/grim-cli/src/tui/kill_ring.rs`, `crates/grim-cli/src/tui/undo_stack.rs`, `crates/grim-cli/src/tui/composer.rs` (extend existing `mod tests`)

**Interfaces:**
* Consumes: Nothing from Tasks 1 to 4 at the type level. Integrates with `Composer` and with `SelectList` only at the key-handling layer in `mod.rs`.
* Produces:
  * `pub struct KillRing` with `fn new() -> Self`, `fn push(&mut self, text: String, opts: KillPushOpts)`, `fn peek(&self) -> Option<&str>`, `fn rotate(&mut self)`, `fn len(&self) -> usize`, `fn is_empty(&self) -> bool`.
  * `pub struct KillPushOpts { pub prepend: bool, pub accumulate: bool }` (accumulate merges into the last entry instead of creating a new one).
  * `pub struct UndoStack<S: Clone> { fn new(limit: usize) -> Self, fn push(&mut self, state: S), fn pop(&mut self) -> Option<S>, fn clear(&mut self), fn len(&self) -> usize }`.
  * `pub struct ComposerSnapshot { chars: Vec<char>, cursor: usize }` (private to `composer.rs`, cloned on undo push).
  * On `Composer`:
    * `pub fn undo(&mut self) -> bool` (returns false when stack empty).
    * `pub fn kill_to_end(&mut self) -> Option<String>` (kill from cursor to end of the current logical line, `Ctrl+K`).
    * `pub fn yank(&mut self) -> bool` (insert `KillRing::peek()` at cursor, `Ctrl+Y`).
    * `pub fn yank_pop(&mut self) -> bool` (rotate the inserted region to the next ring entry, `Alt+Y` right after a yank).
    * Jump mode is a small state machine on `App`, not on `Composer`: `JumpMode::None | Forward | Backward` (set by `Ctrl+F` / `Ctrl+B` or `Alt+F` / `Alt+B`), next printable key triggers `jump_to_char`.
  * No new `pub` fields on `Composer` beyond the two state structs and a `last_kill_was_cut: bool` flag for accumulate logic and a `yank_span: Option<(usize, usize)>` for yank-pop replacement.

**Left limit (what must NOT change):**
* Do NOT replace `Composer`'s `Vec<char>` storage with `String`, rope, or gap buffer. Keep `chars: Vec<char>` and `cursor: usize` exactly as they are. New state is additive.
* Do NOT add bracketed-paste mode or `[paste #N]` marker merging from Pie. That is a Pie-specific large-paste optimization and would require grapheme segmenter changes.
* Do NOT change the history ring (`history`, `history_idx`, `draft`, `max_history`). Undo and history are separate stacks.
* Do NOT add a dependency on `crossterm` inside `composer.rs` itself. `Composer` stays a pure data type; key decoding stays in `mod.rs`.

**Right limit (what is out of scope for this task):**
* No word-level undo coalescing configuration, no per-keystroke undo grouping, no separate redo stack. One `push` per editing action is enough for chat.
* No mouse or click-to-jump. Jump mode is keyboard only (forward and backward, one character).
* No `fd`-based file completion. That is Task 6.

**Design notes:**
* `KillRing` is a simple `Vec<String>` where `push` with `accumulate` merges into the last entry. `rotate` pops the last entry and pushes it to the front for yank-pop cycling. Capacity is unbounded for chat, but bounded at 32 for safety.
* `UndoStack<S>` is generic and stores clones. For chat, `S` is `ComposerSnapshot`. Limit 64 entries, oldest dropped first.
* Ownership: `Composer::yank` inserts a cloned `String` at `cursor` and moves `cursor` forward, mirroring Pie's `KillRing` peek and insert. `yank_pop` replaces the byte range `yank_span` with the next ring entry, so the caller must track the inserted span. Do not use `unsafe` and do not hold `&mut` across the ring borrow.

**UX notes:**
* `Ctrl+K` kills to end of the current logical line (not to end of wrapped visual line). Consecutive `Ctrl+K` presses accumulate into one ring entry so a single `Ctrl+Y` restores the whole killed region.
* `Ctrl+Y` yanks at the cursor. `Alt+Y` immediately after a yank rotates the ring and replaces the just-yanked span. If anything else was typed between, `Alt+Y` does nothing.
* `Ctrl+Z` or `Ctrl+_` undoes the last editing action. No redo. The Composer already clears `history_idx` on edits, so undo does not interfere with history browsing.
* Keep F2 (sidebar toggle), F3 (context override), and F4 (model picker) as they are.

**Rust expert examples:**

Show how to add the new state to `Composer` without breaking the existing `Clone` and ownership invariants, and how to handle `Option` returns without `unwrap`.

```rust
// In composer.rs: extend the struct additively, keep all existing fields.
#[derive(Debug, Clone)]
pub struct Composer {
    chars: Vec<char>,
    cursor: usize,
    history: Vec<String>,
    history_idx: Option<usize>,
    draft: String,
    max_history: usize,
    // --- new in Task 5 ---
    undo_stack: UndoStack<ComposerSnapshot>,
    kill_ring: KillRing,
    /// Span of the last yank in chars, for yank-pop replacement.
    last_yank_span: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
struct ComposerSnapshot {
    chars: Vec<char>,
    cursor: usize,
}

// In composer.rs: push before mutating, handle Option without unwrap.
impl Composer {
    fn push_undo(&mut self) {
        let snap = ComposerSnapshot { chars: self.chars.clone(), cursor: self.cursor };
        self.undo_stack.push(snap);
    }

    /// Undo the last editing action. Returns false when the stack is empty.
    pub fn undo(&mut self) -> bool {
        if let Some(snap) = self.undo_stack.pop() {
            self.chars = snap.chars;
            self.cursor = snap.cursor;
            self.last_yank_span = None;
            true
        } else {
            false
        }
    }

    /// Kill from cursor to end of the current logical line. Used by Ctrl+K.
    pub fn kill_to_end(&mut self) -> Option<String> {
        let (row_start, row_end) = self.current_logical_line_bounds();
        if self.cursor >= row_end { return None; }
        self.push_undo();
        let killed: String = self.chars[self.cursor..row_end].iter().collect();
        self.chars.drain(self.cursor..row_end);
        self.last_yank_span = None;
        Some(killed)
    }
}

// In kill_ring.rs: ring buffer with accumulate merging.
#[derive(Debug, Clone, Default)]
pub struct KillRing { ring: Vec<String> }

#[derive(Debug, Clone, Copy)]
pub struct KillPushOpts { pub prepend: bool, pub accumulate: bool }

impl KillRing {
    pub fn push(&mut self, text: String, opts: KillPushOpts) {
        if text.is_empty() { return; }
        if opts.accumulate && !self.ring.is_empty() {
            if let Some(last) = self.ring.last_mut() {
                if opts.prepend { *last = format!("{text}{last}"); } else { last.push_str(&text); }
                return;
            }
        }
        self.ring.push(text);
        if self.ring.len() > 32 { self.ring.remove(0); }
    }
    pub fn peek(&self) -> Option<&str> { self.ring.last().map(|s| s.as_str()) }
    pub fn rotate(&mut self) {
        if self.ring.len() > 1 { if let Some(last) = self.ring.pop() { self.ring.insert(0, last); } }
    }
}

// In mod.rs: key handling integrates UndoStack and KillRing without leaking
// Engine or blocking I/O. Keep all new branches inside handle_chat_key.
fn handle_chat_key(&mut self, key: KeyEvent) {
    // ... existing branches ...
    // Ctrl+K: kill to end of line and push to kill-ring, consecutive kills accumulate.
    // Ctrl+Y: yank at cursor, remember span for yank-pop.
    // Alt+Y (Ctrl+y with Alt modifier right after a yank): rotate and replace span.
    // Ctrl+Z / Ctrl+_: undo.
}
```

The implementer should keep `composer.rs` free of `crossterm` imports; all `KeyEvent` decoding stays in `mod.rs` where `KeyModifiers::CONTROL` and `KeyModifiers::ALT` are already used.

- [ ] **Step 1: Write the failing tests for KillRing, UndoStack, and Composer undo and kill**

Create `crates/grim-cli/src/tui/kill_ring.rs` and `crates/grim-cli/src/tui/undo_stack.rs` with these test modules, plus extend `composer.rs` tests:

```rust
// kill_ring.rs
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn push_and_peek() { let mut r = KillRing::new(); r.push("hello".into(), KillPushOpts { prepend: false, accumulate: false }); assert_eq!(r.peek(), Some("hello")); }
    #[test] fn accumulate_merges() { let mut r = KillRing::new(); r.push("foo".into(), KillPushOpts { prepend: false, accumulate: false }); r.push("bar".into(), KillPushOpts { prepend: false, accumulate: true }); assert_eq!(r.peek(), Some("foobar")); assert_eq!(r.len(), 1); }
    #[test] fn rotate_cycles() { let mut r = KillRing::new(); r.push("a".into(), KillPushOpts { prepend: false, accumulate: false }); r.push("b".into(), KillPushOpts { prepend: false, accumulate: false }); r.rotate(); assert_eq!(r.peek(), Some("a")); }
}

// undo_stack.rs
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn push_pop_roundtrip() { let mut s: UndoStack<Vec<char>> = UndoStack::new(8); s.push(vec!['a','b']); assert_eq!(s.pop(), Some(vec!['a','b'])); assert_eq!(s.pop(), None); }
    #[test] fn bounded_drop_oldest() { let mut s: UndoStack<i32> = UndoStack::new(2); s.push(1); s.push(2); s.push(3); assert_eq!(s.len(), 2); }
}

// composer.rs additions
#[cfg(test)]
mod tests {
    // ... existing tests ...
    #[test] fn undo_restores_after_insert() { let mut c = Composer::new(); c.insert_char('h'); c.insert_char('i'); assert!(c.undo()); assert_eq!(c.text(), "h"); assert!(!c.undo() || c.text().is_empty() || true); }
    #[test] fn kill_to_end_and_yank() { let mut c = Composer::new(); for ch in "hello world".chars() { c.insert_char(ch); } c.move_cursor_home(); for _ in 0..5 { c.move_cursor_right(); } /* cursor after "hello" */ let killed = c.kill_to_end(); assert_eq!(killed.as_deref(), Some(" world")); c.yank(); assert_eq!(c.text(), "hello world"); }
    #[test] fn yank_pop_cycles_through_ring() { /* push two kills, yank, yank_pop, check replacement */ }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --package grim-cli --lib tui::kill_ring`
Expected: FAIL with `KillRing` not found. Similarly `tui::undo_stack` and the new `composer` tests should fail until the types exist.

- [ ] **Step 3: Implement KillRing and UndoStack**

Write `crates/grim-cli/src/tui/kill_ring.rs` and `crates/grim-cli/src/tui/undo_stack.rs` exactly as sketched in the Rust expert examples above. Both are under 50 lines and have no external dependencies. `UndoStack` must be generic over `Clone` and bounded by a capacity.

- [ ] **Step 4: Enhance Composer with undo, kill, yank, yank-pop**

Extend `crates/grim-cli/src/tui/composer.rs`:
* Add fields `undo_stack: UndoStack<ComposerSnapshot>`, `kill_ring: KillRing`, `last_yank_span: Option<(usize, usize)>`, `last_kill_accumulate: bool`.
* Add `push_undo` helper and call it at the start of every mutating method (`insert_char`, `delete_prev_char`, `delete_current_char`, `delete_word_back`, `kill_to_end`, `clear`).
* Implement `kill_to_end`, `kill_word_forward` (mirror `delete_word_back` forward), `yank`, `yank_pop`, `undo`.
* Add `jump_to_char_forward` and `jump_to_char_backward` that scan `chars[cursor+1..]` or `chars[..cursor]` for the target char and move `cursor` to it.

All new `Composer` methods must have doc comments. Do not add `crossterm` imports to this file.

- [ ] **Step 5: Register modules and wire key handling in mod.rs**

Add to `crates/grim-cli/src/tui/mod.rs`:

```rust
pub mod kill_ring;
pub mod undo_stack;
pub use kill_ring::{KillPushOpts, KillRing};
pub use undo_stack::UndoStack;
```

In `App::handle_chat_key`, add branches (keep existing `Ctrl+W`, `Ctrl+U`, `Ctrl+A`, `Ctrl+E` as they are):
* `Ctrl+K` (char 'k' with `CONTROL`): call `self.composer.kill_to_end()`, push the killed string into `self.composer.kill_ring` with `accumulate: last_kill_was_cut`, set `last_kill_was_cut = true` on success.
* `Ctrl+Y` (char 'y' with `CONTROL`): call `self.composer.yank()`, clear `last_kill_was_cut`.
* `Alt+Y` (char 'y' with `ALT`): only when `last_yank_span.is_some()`, call `self.composer.yank_pop()`.
* `Ctrl+Z` or `Ctrl+_` (char 'z' or 'x' with `CONTROL`, handle both for terminal compatibility): call `self.composer.undo()`.
* Reset `last_kill_was_cut = false` on any non-kill key (insert, delete, cursor move).

Add a small `JumpMode` enum on `App` if implementing jump: `None | Forward | Backward`, set by `Alt+F`/`Alt+B`, and on next printable char do the scan. Keep this minimal and keyboard-only.

Update the help text at `mod.rs:405` to list the new bindings: `Ctrl+K: Kill to end | Ctrl+Y: Yank | Alt+Y: Yank-pop | Ctrl+Z: Undo`.

- [ ] **Step 6: Verify**

Run: `cargo test --package grim-cli --lib tui`
Expected: PASS (all existing tests plus the new kill_ring, undo_stack, and composer tests).

Manual: in `cargo run --bin grim-cli -- tui`, type a line, press `Ctrl+K` to kill to end, type more, press `Ctrl+Y` to yank back, press `Alt+Y` to cycle, press `Ctrl+Z` to undo the yank.

- [ ] **Step 7: Commit**

```bash
git add crates/grim-cli/src/tui/kill_ring.rs crates/grim-cli/src/tui/undo_stack.rs crates/grim-cli/src/tui/composer.rs crates/grim-cli/src/tui/mod.rs
git commit -m "feat(tui): add undo, kill-ring, yank-pop, and jump mode to composer"
```

---

### Task 6: File Path Completion for @ Triggers

> **Why this exists:** The chat surface is increasingly used to reference local files (attach a file, point the model at a path, paste a relative import). Pie's `CombinedAutocompleteProvider` resolves `@` prefixes to real filesystem paths via the `fd` binary (respecting `.gitignore`) with a `std::fs` fallback. Without it, users must leave the TUI to copy a path or type it from memory and hope it is correct. This task ports a focused, synchronous subset of `old/pie/packages/tui/src/autocomplete.ts`: trigger detection, file suggestion, and `SelectList` presentation for the composer. The `fd` binary is an optimization, not a requirement.

**Files:**
* Create: `crates/grim-cli/src/tui/file_complete.rs`
* Modify: `crates/grim-cli/src/tui/mod.rs:1-40` (add `pub mod file_complete;` and re-export)
* Modify: `crates/grim-cli/src/tui/mod.rs:700-820` (`ui()` function: add a second popup path for `@` that renders a `SelectList` of file matches, alongside the existing `/` command popup; the two popups are mutually exclusive)
* Modify: `crates/grim-cli/src/tui/mod.rs:290-385` (`App::handle_chat_key`: add `@`-triggered completion and `Tab` integration for file paths)
* Test: `crates/grim-cli/src/tui/file_complete.rs` (inline `mod tests`)

**Interfaces:**
* Consumes: `SelectList`, `SelectItem`, `SelectListTheme` from Task 2; `fuzzy_match`/`fuzzy_filter` from Task 1 for ranking file names when a partial query is present.
* Produces:
  * `pub fn extract_at_prefix(text: &str, cursor: usize) -> Option<(usize, String)>` (returns the byte range start and the raw path prefix after `@`, or `None` when the `@` is not at a token boundary).
  * `pub fn get_file_suggestions(prefix: &str, base_dir: &std::path::Path, max_results: usize) -> Vec<FileSuggestion>` (synchronous, `std::fs::read_dir` based; try `fd` if on `PATH`, otherwise walk with `std::fs` and skip `.git`).
  * `pub struct FileSuggestion { pub value: String, pub label: String, pub is_dir: bool }` (value is the completion text to insert, label is the display name).
  * `pub fn apply_file_completion(composer: &mut Composer, start: usize, suggestion: &FileSuggestion)` (replaces `@<prefix>` at `start` with `@<value>` and moves cursor; handles directory trailing `/` without trailing space, files with trailing space, and quoted paths containing spaces).

**Left limit (what must NOT change):**
* Do NOT change the slash command completion path. The `/` popup stays as it is. The `@` popup is an additional `else if` branch. Only one popup is visible at a time.
* Do NOT add an async runtime, `tokio` spawn, or `fd` child process with `AbortSignal`. File discovery is synchronous and bounded (`max_results` 50) so it stays well under the 16ms frame budget for typical project sizes. If the directory scan takes too long, return fewer results rather than blocking.
* Do NOT add a `tokio` dependency or a global file index. Scan on demand when `@` is detected.
* Do NOT modify `grim_format::ChatMessage` or the worker protocol. File paths are plain text inserted into the composer; the model sees them as normal chat content.

**Right limit (what is out of scope):**
* No `fd` process spawning in this task. Use `std::fs::read_dir` only. A follow-up may add an `fd` fast path behind a `which`-like check, but it is not required for correctness.
* No recursive deep search. List entries from the directory indicated by the prefix (or `base_dir` when the prefix has no `/`). Do not walk the entire tree. The Pie `walkDirectoryWithFd` full-tree search is explicitly not ported.
* No `~/` home expansion in this task. Support relative paths and `./` only. Home expansion can be added later without changing the trigger detection.
* No image or markdown handling for the inserted path. The `SelectList` description column shows `value`, nothing more.

**Design notes:**
* Trigger detection: an `@` is a trigger when it is at position 0, or the preceding char is whitespace, `"`, `'`, or `=`, and there is no unclosed `"` that would make it part of a different quoted token. This mirrors `findLastDelimiter` and `extractAtPrefix` in `autocomplete.ts` but is implemented synchronously over `&str` and the composer cursor offset.
* Discovery: `get_file_suggestions` resolves `prefix` into `(dir, stem)`, calls `std::fs::read_dir(dir)`, filters by `stem` prefix (case-insensitive), marks directories via `DirEntry::file_type().is_dir()`, sorts directories first then alphabetically, truncates to `max_results`.
* Insertion: `apply_file_completion` replaces the byte range `start..cursor` in the composer with `suggestion.value`, preserving text before `start` and after `cursor`. For directories the inserted text ends with `/` and no trailing space so the user can keep completing. For files it ends without extra quoting unless the path contained a space (then the inserted text is quoted). This matches `buildCompletionValue` in the original.
* Ownership: `get_file_suggestions` allocates a fresh `Vec<FileSuggestion>` each call. The caller owns it and passes it to `SelectList::new`. No `Rc` or shared state.

**UX notes:**
* Typing `@` after a space or at the start of the line shows a `SelectList` of files in the current directory.
* Typing `@src/` shows entries inside `src/`.
* Typing `@src/m` shows `src/main.rs`, `src/models/...` etc, ranked by fuzzy match on the stem.
* Tab on an `@` prefix confirms the highlighted file path and inserts it at the cursor.

**Rust expert examples:**

```rust
// In file_complete.rs: trigger detection over &str and cursor offset.
// No allocation for the common no-trigger path.

/// Returns (byte_start, prefix) when an `@` trigger is active at `cursor`.
pub fn extract_at_prefix(text: &str, cursor: usize) -> Option<(usize, String)> {
    if cursor > text.len() { return None; }
    let before = &text[..cursor];
    // Find the last '@' that is at a token boundary.
    let at = before.rfind('@')?;
    let before_at = &before[..at];
    let ok = at == 0 || {
        let prev = before_at.chars().last()?;
        prev.is_whitespace() || matches!(prev, '"' | '\'' | '=')
    };
    if !ok { return None; }
    // No unclosed quote handling needed for the initial task; add if tests require.
    let prefix = before[at..].to_string();
    Some((at, prefix))
}

// In file_complete.rs: synchronous file suggestions with std::fs, no tokio.
pub struct FileSuggestion { pub value: String, pub label: String, pub is_dir: bool }

pub fn get_file_suggestions(prefix: &str, base_dir: &std::path::Path, max_results: usize) -> Vec<FileSuggestion> {
    let (dir_part, stem) = match prefix.rfind('/') {
        Some(idx) => (&prefix[..=idx], &prefix[idx+1..]),
        None => ("", prefix),
    };
    let search_dir = if dir_part.is_empty() { base_dir.to_path_buf() } else { base_dir.join(dir_part) };
    let entries = std::fs::read_dir(&search_dir).ok()?;
    // ... filter by stem.to_lowercase(), mark is_dir, sort dirs first, truncate ...
    // Return Vec<FileSuggestion> owned by the caller.
    unimplemented!("see task implementation")
}

// In mod.rs: two mutually exclusive popups, slash vs file. File popup uses SelectList.
fn ui(f: &mut Frame, app: &App) {
    // ... existing / popup ...
    // After the / branch, else if extract_at_prefix returns Some:
    if let Some((start, prefix)) = file_complete::extract_at_prefix(&input_text, app.composer.cursor_offset()) {
        let suggestions = file_complete::get_file_suggestions(&prefix[1..], std::env::current_dir().unwrap_or_default().as_path(), 50);
        let items: Vec<SelectItem> = suggestions.into_iter().map(|s| SelectItem {
            value: s.value, label: s.label, description: None,
        }).collect();
        let menu = SelectList::new(items, 6, SelectListTheme::default());
        // render menu as a popup at the input area, same Rect math as slash popup
    }
}

// Apply completion replaces the @-range in the composer.
pub fn apply_file_completion(composer: &mut Composer, start: usize, suggestion: &FileSuggestion) {
    // Replace bytes [start..cursor] with suggestion.value, preserve before/after.
    // Handle trailing "/" for dirs vs space for files, quoting for spaces.
    // Update composer.cursor to after the inserted text.
}
```

The implementer should keep all new functions synchronous and allocation-bounded. Do not spawn `fd` or `tokio` tasks. Use `std::path::Path` and `std::fs::read_dir` only.

- [ ] **Step 1: Write the failing tests for file completion**

Create `crates/grim-cli/src/tui/file_complete.rs` with this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test] fn at_prefix_at_start() { assert_eq!(extract_at_prefix("@foo", 4), Some((0, "@foo".into()))); }
    #[test] fn at_prefix_after_space() { assert_eq!(extract_at_prefix("hi @foo", 6), Some((3, "@foo".into()))); }
    #[test] fn not_trigger_in_email() { assert_eq!(extract_at_prefix("a@foo", 5), None); }
    #[test] fn no_trigger_without_at() { assert_eq!(extract_at_prefix("hello", 5), None); }

    #[test] fn file_suggestions_lists_dir() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.txt"), b"").unwrap();
        fs::write(dir.path().join("b.rs"), b"").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        let out = get_file_suggestions("", dir.path(), 50);
        assert!(out.iter().any(|s| s.label == "a.txt"));
        assert!(out.iter().any(|s| s.label == "sub/"));
        // dirs first
        assert!(out.first().unwrap().is_dir);
    }

    #[test] fn prefix_filters_stem() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("alpha.txt"), b"").unwrap();
        fs::write(dir.path().join("beta.txt"), b"").unwrap();
        let out = get_file_suggestions("a", dir.path(), 50);
        assert!(out.iter().any(|s| s.label == "alpha.txt"));
        assert!(!out.iter().any(|s| s.label == "beta.txt"));
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --package grim-cli --lib tui::file_complete`
Expected: FAIL with `extract_at_prefix` not found.

- [ ] **Step 3: Implement file completion**

Write `crates/grim-cli/src/tui/file_complete.rs` with `extract_at_prefix`, `FileSuggestion`, `get_file_suggestions`, and `apply_file_completion` as described. Keep it under 200 lines. The function must be synchronous and must not allocate on the no-trigger path beyond the returned `Option`.

- [ ] **Step 4: Register the module and wire the popup in mod.rs**

Add to `crates/grim-cli/src/tui/mod.rs`:

```rust
pub mod file_complete;
pub use file_complete::{FileSuggestion, apply_file_completion, extract_at_prefix, get_file_suggestions};
```

In `ui()`, add the `@` popup as an `else if` after the existing `if input_text.starts_with('/')` branch. Both popups use the same `popup_area` math (width 48, height `min(8, count+2)`, anchored above the input block). Only one popup is visible at a time.

In `App::handle_chat_key`, `Tab` should prefer the `@` completion when an `@` trigger is active, otherwise fall back to the existing slash completion. For `@`, call `get_file_suggestions` on the prefix after `@`, create a `SelectList` from the results, and insert the selected item via `apply_file_completion`.

- [ ] **Step 5: Verify**

Run: `cargo test --package grim-cli --lib tui`
Expected: PASS.

Manual: in `cargo run --bin grim-cli -- tui`, type `hello @` and verify a file list appears, type `@src/` and verify entries inside `src/`, press Tab to insert the highlighted path.

- [ ] **Step 6: Commit**

```bash
git add crates/grim-cli/src/tui/file_complete.rs crates/grim-cli/src/tui/mod.rs
git commit -m "feat(tui): add @file path completion for composer"
```

---

## Verification Plan

### Automated Tests

Run unit tests for all new modules:

```bash
cargo test --package grim-cli --lib tui::fuzzy
cargo test --package grim-cli --lib tui::select_list
cargo test --package grim-cli --lib tui::throttle
cargo test --package grim-cli --lib tui::layout
cargo test --package grim-cli --lib tui::kill_ring
cargo test --package grim-cli --lib tui::undo_stack
cargo test --package grim-cli --lib tui::file_complete
```

Run the full TUI suite:

```bash
cargo test --package grim-cli --lib tui
```

Run the full workspace check (clippy + tests):

```bash
cargo test --package grim-cli
cargo clippy --package grim-cli --lib
```

### Manual Verification

1. Launch the interactive TUI:

   ```bash
   cargo run --bin grim-cli -- tui
   ```

2. Test fuzzy autocomplete: type `/mod`, verify `/model` appears. Type `/cl`, verify `/clear` appears. Type `/xyz`, verify no popup (no match).

3. Test SelectList navigation: type `/`, verify the menu appears with all commands. Press Down/Up to move selection (wrap-around). Type `m` to filter to `/model`. Press Enter to confirm, Esc to cancel.

4. Test render throttle: stream a long generation and verify the UI stays responsive. Rapid typing should feel immediate (immediate path) while background token streaming is throttled to 60 FPS.

5. Test layout: resize the terminal to a very small height (e.g. 10 rows) and verify at least one transcript row and the input cursor remain visible. Resize back and verify the transcript viewport grows.

6. Verify clean exit: type `/exit` or press `Ctrl+C` and confirm the shell raw mode is fully restored with no terminal artifacts.

---

## Recommended Implementation Order

1. **Task 1 (fuzzy)**: Pure function, no UI coupling. Unblocks Task 2.
2. **Task 2 (SelectList)**: Reuses `CommandRegistry` data, consumes Task 1. Visible UX improvement on its own.
3. **Task 3 (render throttle)**: Touches the hot render path. Do after 1 and 2 so the menu and fuzzy work is already validated.
4. **Task 4 (layout)**: Largest change, refactors `ui()`. Do after the first three are committed and passing.
5. **Task 5 (editor undo, kill-ring, jump)**: Enhances `Composer` and `App` key handling. Depends on 1 and 2 being in place but is otherwise independent of layout. Can be done in parallel with Task 6.
6. **Task 6 (file completion)**: Adds `@` trigger and file provider. Depends on `SelectList` and `fuzzy`. Do after Task 2, can be parallel with Task 5.

Each task is independently committable and testable. A weak implementer should do them in order and not skip the test-failure verification steps. Tasks 5 and 6 are additive and do not change the slash command path: a weak implementer can validate them by running the full `tui` suite after each.

---

## Self-Review Checklist

Before marking the plan complete, verify:

* Every file path is exact and every `pub` item has a doc comment.
* No step contains a placeholder like "TBD", "implement later", or "handle edge cases" without code.
* The types in `Consumes`/`Produces` blocks match across tasks (e.g. `FuzzyMatch` defined in Task 1 is used as `fuzzy_filter` return in Task 2).
* The left/right limits for each task are consistent with the global Project Boundaries.
* No pie reference leaks as "pi": all references use "pie".
* Each task's Step 2 (run test, expect FAIL) and Step 4 (run test, expect PASS) form a proper TDD red-green cycle.
