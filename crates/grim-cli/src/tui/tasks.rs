//! Agent task / todo list panel for the grim TUI sidebar.
//!
//! The model can emit structured task updates via a `tool` call (e.g.
//! `update_tasks`) and the UI renders them in the sidebar between the
//! diagnostics panel and the tok/s sparkline. Tasks are ephemeral — they
//! live only for the current TUI session and are cleared on `/clear`.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Status of a single task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskStatus {
    /// Not yet started.
    Pending,
    /// Currently being worked on.
    InProgress,
    /// Finished successfully.
    Completed,
    /// Failed with an error.
    Failed,
    /// Cancelled by the user or the agent.
    Cancelled,
}

impl TaskStatus {
    /// Short single-character indicator for compact display.
    pub fn indicator(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "○",
            TaskStatus::InProgress => "◐",
            TaskStatus::Completed => "●",
            TaskStatus::Failed => "✖",
            TaskStatus::Cancelled => "⊘",
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "pending",
            TaskStatus::InProgress => "in-progress",
            TaskStatus::Completed => "done",
            TaskStatus::Failed => "failed",
            TaskStatus::Cancelled => "cancelled",
        }
    }

    /// Color for the status indicator.
    pub fn color(&self) -> Color {
        match self {
            TaskStatus::Pending => Color::Rgb(136, 136, 136),    // muted
            TaskStatus::InProgress => Color::Rgb(245, 158, 11),  // amber
            TaskStatus::Completed => Color::Rgb(16, 185, 129),    // green
            TaskStatus::Failed => Color::Rgb(239, 68, 68),       // red
            TaskStatus::Cancelled => Color::Rgb(112, 50, 180),   // dim purple
        }
    }

    /// Cycle to the next logical state (for Tab key on a task).
    pub fn cycle(&self) -> Self {
        match self {
            TaskStatus::Pending => TaskStatus::InProgress,
            TaskStatus::InProgress => TaskStatus::Completed,
            TaskStatus::Completed => TaskStatus::Pending,
            TaskStatus::Failed => TaskStatus::Pending,
            TaskStatus::Cancelled => TaskStatus::Pending,
        }
    }
}

impl std::str::FromStr for TaskStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "pending" | "todo" | "planned" => Ok(TaskStatus::Pending),
            "in-progress" | "in_progress" | "inprogress" | "active" | "doing" => {
                Ok(TaskStatus::InProgress)
            }
            "completed" | "complete" | "done" | "finished" | "ok" => Ok(TaskStatus::Completed),
            "failed" | "error" | "fail" => Ok(TaskStatus::Failed),
            "cancelled" | "canceled" | "skip" | "skipped" => Ok(TaskStatus::Cancelled),
            other => Err(format!("unknown task status: {other}")),
        }
    }
}

/// A single task item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task {
    /// Stable identifier (e.g. "1", "2", or a short slug).
    pub id: String,
    /// Short title / summary.
    pub title: String,
    /// Optional longer description.
    pub description: Option<String>,
    /// Current status.
    pub status: TaskStatus,
    /// True when the task detail is expanded in the UI.
    pub expanded: bool,
}

impl Task {
    /// Create a new pending task.
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            description: None,
            status: TaskStatus::Pending,
            expanded: false,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the status.
    pub fn with_status(mut self, status: TaskStatus) -> Self {
        self.status = status;
        self
    }
}

/// A collection of tasks managed by the agent during a session.
#[derive(Debug, Clone, Default)]
pub struct TaskList {
    pub tasks: Vec<Task>,
    /// Currently selected row (for keyboard navigation).
    pub selected: usize,
    /// Scroll offset for long lists.
    pub scroll_offset: usize,
}

impl TaskList {
    /// Create an empty task list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace all tasks with the given list.
    pub fn set_tasks(&mut self, tasks: Vec<Task>) {
        self.tasks = tasks;
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Add a task. If a task with the same id exists, it is replaced.
    pub fn upsert(&mut self, task: Task) {
        if let Some(existing) = self.tasks.iter_mut().find(|t| t.id == task.id) {
            *existing = task;
        } else {
            self.tasks.push(task);
        }
    }

    /// Update the status of a task by id. Returns true if found.
    pub fn update_status(&mut self, id: &str, status: TaskStatus) -> bool {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.status = status;
            true
        } else {
            false
        }
    }

    /// Remove a task by id. Returns true if it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        let len_before = self.tasks.len();
        self.tasks.retain(|t| t.id != id);
        let removed = self.tasks.len() < len_before;
        if removed && self.selected >= self.tasks.len() && self.selected > 0 {
            self.selected = self.tasks.len() - 1;
        }
        removed
    }

    /// Clear all tasks.
    pub fn clear(&mut self) {
        self.tasks.clear();
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Number of tasks.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// True when there are no tasks.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Move selection up.
    pub fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
        // Adjust scroll to keep selection visible.
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }
    }

    /// Move selection down.
    pub fn move_down(&mut self) {
        if !self.tasks.is_empty() && self.selected < self.tasks.len() - 1 {
            self.selected += 1;
        }
    }

    /// Toggle expand on the selected task.
    pub fn toggle_expand_selected(&mut self) {
        if let Some(task) = self.tasks.get_mut(self.selected) {
            task.expanded = !task.expanded;
        }
    }

    /// Cycle the status of the selected task.
    pub fn cycle_selected_status(&mut self) {
        if let Some(task) = self.tasks.get_mut(self.selected) {
            task.status = task.status.cycle();
        }
    }

    /// Count tasks by status.
    pub fn count_by_status(&self, status: TaskStatus) -> usize {
        self.tasks.iter().filter(|t| t.status == status).count()
    }

    /// Render the task list as styled Lines for the sidebar.
    ///
    /// `max_rows` caps how many task rows are rendered; when the list
    /// exceeds this, a "(+N more)" footer is appended.
    pub fn render(&self, max_rows: usize) -> Vec<Line<'static>> {
        let c_purple_soft = Color::Rgb(192, 132, 252);
        let c_muted = Color::Rgb(136, 136, 136);
        let c_amber = Color::Rgb(245, 158, 11);

        let mut lines: Vec<Line<'static>> = Vec::new();

        if self.tasks.is_empty() {
            lines.push(Line::from(Span::styled(
                "  no tasks — agent will",
                Style::default().fg(c_muted),
            )));
            lines.push(Line::from(Span::styled(
                "  add them as it works",
                Style::default().fg(c_muted),
            )));
            return lines;
        }

        // Summary header.
        let total = self.tasks.len();
        let done = self.count_by_status(TaskStatus::Completed);
        lines.push(Line::from(vec![
            Span::styled(
                format!("╭─ tasks "),
                Style::default().fg(c_muted),
            ),
            Span::styled(
                format!("{done}/{total}"),
                Style::default().fg(if done == total {
                    Color::Rgb(16, 185, 129)
                } else {
                    c_amber
                }),
            ),
            Span::styled(" ──────", Style::default().fg(c_muted)),
        ]));

        let visible_count = max_rows.min(self.tasks.len());
        let start = self.scroll_offset;
        let end = (start + visible_count).min(self.tasks.len());

        for idx in start..end {
            let task = &self.tasks[idx];
            let is_selected = idx == self.selected;
            let status_color = task.status.color();
            let indicator = task.status.indicator();

            let title_color = if is_selected {
                Color::White
            } else {
                Color::White
            };

            // Status indicator + title.
            let prefix = if is_selected { "▶ " } else { "  " };
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), Style::default().fg(c_purple_soft)),
                Span::styled(format!("{indicator} "), Style::default().fg(status_color)),
                Span::styled(
                    truncate(&task.title, 22),
                    Style::default().fg(title_color).add_modifier(
                        if is_selected {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        },
                    ),
                ),
                Span::styled(
                    format!(" [{}]", task.status.label()),
                    Style::default().fg(if is_selected {
                        status_color
                    } else {
                        c_muted
                    }),
                ),
            ]));

            // Expanded description.
            if task.expanded {
                if let Some(desc) = &task.description {
                    for desc_line in desc.lines().take(2) {
                        lines.push(Line::from(vec![
                            Span::styled("    ╎ ".to_string(), Style::default().fg(c_muted)),
                            Span::styled(
                                truncate(desc_line, 22),
                                Style::default().fg(c_muted),
                            ),
                        ]));
                    }
                }
            }
        }

        // "+N more" footer when clamped.
        if self.tasks.len() > visible_count {
            let remaining = self.tasks.len() - end;
            lines.push(Line::from(Span::styled(
                format!("    +{remaining} more"),
                Style::default().fg(c_muted),
            )));
        }

        // Bottom border with hint.
        lines.push(Line::from(Span::styled(
            "╰ Tab:cycle →:expand",
            Style::default().fg(c_muted),
        )));

        lines
    }
}

/// Truncate a string to at most `max` characters, adding "…" if truncated.
fn truncate(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        let mut truncated: String = chars[..max.saturating_sub(1)].iter().collect();
        truncated.push('…');
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_status_parse() {
        assert_eq!("pending".parse::<TaskStatus>().unwrap(), TaskStatus::Pending);
        assert_eq!("in-progress".parse::<TaskStatus>().unwrap(), TaskStatus::InProgress);
        assert_eq!("done".parse::<TaskStatus>().unwrap(), TaskStatus::Completed);
        assert_eq!("failed".parse::<TaskStatus>().unwrap(), TaskStatus::Failed);
        assert_eq!("cancelled".parse::<TaskStatus>().unwrap(), TaskStatus::Cancelled);
        assert!("unknown".parse::<TaskStatus>().is_err());
    }

    #[test]
    fn task_status_cycle() {
        assert_eq!(TaskStatus::Pending.cycle(), TaskStatus::InProgress);
        assert_eq!(TaskStatus::InProgress.cycle(), TaskStatus::Completed);
        assert_eq!(TaskStatus::Completed.cycle(), TaskStatus::Pending);
    }

    #[test]
    fn task_status_indicators() {
        assert_eq!(TaskStatus::Pending.indicator(), "○");
        assert_eq!(TaskStatus::InProgress.indicator(), "◐");
        assert_eq!(TaskStatus::Completed.indicator(), "●");
        assert_eq!(TaskStatus::Failed.indicator(), "✖");
        assert_eq!(TaskStatus::Cancelled.indicator(), "⊘");
    }

    #[test]
    fn task_list_upsert_and_update() {
        let mut list = TaskList::new();
        list.upsert(Task::new("1", "First task"));
        list.upsert(Task::new("2", "Second task"));
        assert_eq!(list.len(), 2);

        // Update existing.
        list.upsert(Task::new("1", "First task").with_status(TaskStatus::Completed));
        assert_eq!(list.tasks[0].status, TaskStatus::Completed);
        assert_eq!(list.len(), 2);

        // Update status helper.
        assert!(list.update_status("2", TaskStatus::InProgress));
        assert_eq!(list.tasks[1].status, TaskStatus::InProgress);
        assert!(!list.update_status("nonexistent", TaskStatus::Pending));
    }

    #[test]
    fn task_list_remove() {
        let mut list = TaskList::new();
        list.upsert(Task::new("1", "A"));
        list.upsert(Task::new("2", "B"));
        assert!(list.remove("1"));
        assert_eq!(list.len(), 1);
        assert!(!list.remove("1"));
    }

    #[test]
    fn task_list_navigation() {
        let mut list = TaskList::new();
        list.upsert(Task::new("1", "A"));
        list.upsert(Task::new("2", "B"));
        list.upsert(Task::new("3", "C"));
        assert_eq!(list.selected, 0);

        list.move_down();
        assert_eq!(list.selected, 1);
        list.move_down();
        assert_eq!(list.selected, 2);
        list.move_down(); // clamp
        assert_eq!(list.selected, 2);

        list.move_up();
        assert_eq!(list.selected, 1);
        list.move_up();
        assert_eq!(list.selected, 0);
        list.move_up(); // clamp
        assert_eq!(list.selected, 0);
    }

    #[test]
    fn task_list_count_by_status() {
        let mut list = TaskList::new();
        list.upsert(Task::new("1", "A").with_status(TaskStatus::Pending));
        list.upsert(Task::new("2", "B").with_status(TaskStatus::InProgress));
        list.upsert(Task::new("3", "C").with_status(TaskStatus::Completed));
        list.upsert(Task::new("4", "D").with_status(TaskStatus::Completed));

        assert_eq!(list.count_by_status(TaskStatus::Pending), 1);
        assert_eq!(list.count_by_status(TaskStatus::InProgress), 1);
        assert_eq!(list.count_by_status(TaskStatus::Completed), 2);
        assert_eq!(list.count_by_status(TaskStatus::Failed), 0);
    }

    #[test]
    fn task_list_render_empty() {
        let list = TaskList::new();
        let lines = list.render(6);
        // Should have "no tasks" hint lines.
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(text.contains("no tasks"));
    }

    #[test]
    fn task_list_render_with_tasks() {
        let mut list = TaskList::new();
        list.upsert(Task::new("1", "Read config file"));
        list.upsert(Task::new("2", "Write tests").with_status(TaskStatus::Completed));
        let lines = list.render(6);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(text.contains("1/2")); // progress summary
        assert!(text.contains("Read config file"));
        assert!(text.contains("Write tests"));
    }

    #[test]
    fn task_list_render_truncation() {
        let mut list = TaskList::new();
        list.upsert(Task::new("1", "This is a very long task title that should be truncated"));
        let lines = list.render(6);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(text.contains('…'));
    }

    #[test]
    fn truncate_helper() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("exactlyten", 10), "exactlyten");
        // Truncated output is exactly `max` chars (9 content + ellipsis).
        assert_eq!(truncate("this is longer", 10), "this is l…");
    }
}
