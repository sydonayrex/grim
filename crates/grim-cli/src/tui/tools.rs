//! Coding tool definitions and sandboxed execution.
//!
//! Tools are exposed to the model via grim's OpenAI-compatible `ToolDef`
//! format so the chat template receives them through the `tools` Jinja
//! variable. Execution is sandboxed to a single allow-listed directory.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use grim_format::{FunctionDef, ToolCallMsg, ToolDef};

/// Max lines returned by a single read_file call before truncation.
pub const MAX_READ_LINES: usize = 400;
/// Max bytes of file content returned by read_file before truncation.
pub const MAX_READ_BYTES: usize = 32 * 1024;
/// Max bytes of combined stdout+stderr returned by run_command.
pub const MAX_CMD_BYTES: usize = 30 * 1024;
/// Default wall-clock timeout for run_command.
pub const DEFAULT_CMD_TIMEOUT_MS: u64 = 120_000;
/// Hard ceiling for the run_command timeout, even if the model asks for more.
pub const MAX_CMD_TIMEOUT_MS: u64 = 600_000;
/// Max directory entries returned by list_files.
pub const MAX_LIST_ENTRIES: usize = 1_000;
/// Max matched lines returned by search_files.
pub const MAX_SEARCH_MATCHES: usize = 200;

/// Serializable task entry carried over the UI channel for `update_tasks`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct TaskItem {
    pub id: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "default_task_status")]
    pub status: String,
}

fn default_task_status() -> String {
    "pending".to_string()
}

/// Parse and validate the arguments of an `update_tasks` tool call.
/// Returns the task list on success; the error string lists invalid
/// statuses so the model can correct and retry.
pub fn parse_update_tasks(arguments: &str) -> Result<Vec<TaskItem>, String> {
    let args: serde_json::Value =
        serde_json::from_str(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
    let tasks = args
        .get("tasks")
        .cloned()
        .ok_or_else(|| "missing tasks array".to_string())?;
    let items: Vec<TaskItem> =
        serde_json::from_value(tasks).map_err(|e| format!("invalid tasks: {e}"))?;
    let mut bad = Vec::new();
    for item in &items {
        if item.id.trim().is_empty() || item.title.trim().is_empty() {
            bad.push(format!("task with empty id or title ({:?})", item.id));
        }
        if item.status.parse::<crate::tui::tasks::TaskStatus>().is_err() {
            bad.push(format!(
                "invalid status {:?} for task {} (use pending, in-progress, completed, failed, cancelled)",
                item.status, item.id
            ));
        }
    }
    if !bad.is_empty() {
        return Err(bad.join("; "));
    }
    Ok(items)
}

/// Summarize a parsed task list for the tool-result message.
pub fn summarize_tasks(items: &[TaskItem]) -> String {
    let count = |want: crate::tui::tasks::TaskStatus| {
        items
            .iter()
            .filter(|t| t.status.parse::<crate::tui::tasks::TaskStatus>().ok() == Some(want))
            .count()
    };
    use crate::tui::tasks::TaskStatus as S;
    format!(
        "tasks updated: {} total ({} pending, {} in progress, {} completed, {} failed, {} cancelled)",
        items.len(),
        count(S::Pending),
        count(S::InProgress),
        count(S::Completed),
        count(S::Failed),
        count(S::Cancelled),
    )
}

/// Remove tool-call markup from generated text so the cleaned remainder can
/// be stored as an assistant message alongside structured `tool_calls`
/// without double-encoding the calls on the next template render. Handles
/// the two conventions grim's parser recognizes: Hermes `<tool_call>…`
/// and LFM2.5 `<|tool_call_start|>…<|tool_call_end|>`.
pub fn strip_tool_markup(text: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    let mut rest = text;
    loop {
        let hermes = rest.find("<tool_call>");
        let lfm = rest.find("<|tool_call_start|>");
        let (start, close) = match (hermes, lfm) {
            (Some(h), Some(l)) => {
                if h <= l { (h, "</tool_call>") } else { (l, "<|tool_call_end|>") }
            }
            (Some(h), None) => (h, "</tool_call>"),
            (None, Some(l)) => (l, "<|tool_call_end|>"),
            (None, None) => break,
        };
        segments.push(&rest[..start]);
        let after_open = &rest[start..];
        match after_open.find(close) {
            Some(rel_end) => rest = &after_open[rel_end + close.len()..],
            // Unterminated block runs to end of text.
            None => return segments_join(&segments),
        }
    }
    segments.push(rest);
    segments_join(&segments)
}

/// Join non-empty trimmed segments with newlines.
fn segments_join(segments: &[&str]) -> String {
    segments
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The set of coding tools exposed to the model. Reuses grim's
/// OpenAI-compatible `ToolDef` format.
pub fn coding_tools() -> Vec<ToolDef> {
    vec![
    ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "read_file".to_string(),
            description: Some(
                "Read a file and return its lines prefixed with line numbers. \
                 Long files are truncated to the first 400 lines / 32 KiB; use \
                 `offset` and `limit` to page through larger files."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file, relative to the sandbox root" },
                    "offset": { "type": "integer", "description": "1-based line number to start reading from (default: 1)" },
                    "limit": { "type": "integer", "description": "Maximum number of lines to return (default: 400)" }
                },
                "required": ["path"]
            })),
        },
    },
    ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "write_file".to_string(),
            description: Some(
                "Write content to a file at the given path. Creates the file if it \
                 does not exist, overwrites if it does. Creates parent directories \
                 as needed."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file, relative to the sandbox root" },
                    "content": { "type": "string", "description": "The text content to write" }
                },
                "required": ["path", "content"]
            })),
        },
    },
    ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "edit_file".to_string(),
            description: Some(
                "Edit a file by replacing an exact string occurrence. `old_string` \
                 must match exactly including whitespace and indentation. Use this \
                 for precise, surgical edits."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" }
                },
                "required": ["path", "old_string", "new_string"]
            })),
        },
    },
    ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "run_command".to_string(),
            description: Some(
                "Execute a shell command in the sandbox directory. Returns stdout, \
                 stderr, and exit status. Use this to run tests, builds, git, and \
                 other development commands. The command is killed after \
                 `timeout_ms` (default 120s, max 600s). Combined output is \
                 truncated past 30 KiB."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "workdir": { "type": "string", "description": "Working directory relative to sandbox (default: \".\")" },
                    "timeout_ms": { "type": "integer", "description": "Kill the command after this many milliseconds (default 120000, max 600000)" }
                },
                "required": ["command"]
            })),
        },
    },
    ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "list_files".to_string(),
            description: Some(
                "List files and directories in a given path. Directories are \
                 suffixed with \"/\". Returns one entry per line."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory path relative to sandbox (default: \".\")" }
                },
                "required": []
            })),
        },
    },
    ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "search_files".to_string(),
            description: Some(
                "Search for a regex pattern in files under a directory. Returns \
                 matching lines as path:line:content, skipping build artifacts \
                 (target/, node_modules/, .git/). Capped at 200 matches."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "pattern": { "type": "string" }
                },
                "required": ["path", "pattern"]
            })),
        },
    },
    ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "update_tasks".to_string(),
            description: Some(
                "Update the session task list shown to the user. Send the FULL \
                 list each call — it replaces the current one. Use this at the \
                 start of multi-step work to lay out the plan, and after each \
                 step to report progress. Mark exactly one task in-progress at \
                 a time."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "The complete task list; replaces the current list.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string", "description": "Stable id, e.g. \"1\", \"2\"" },
                                "title": { "type": "string", "description": "Short imperative summary" },
                                "description": { "type": "string", "description": "Optional longer detail" },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in-progress", "completed", "failed", "cancelled"],
                                    "description": "Default: pending"
                                }
                            },
                            "required": ["id", "title"]
                        }
                    }
                },
                "required": ["tasks"]
            })),
        },
    }]
}

/// Sandbox policy: all file operations are restricted to this directory.
/// Paths are canonicalized and checked to ensure they don't escape the root.
#[derive(Debug, Clone)]
pub struct Sandbox {
    pub root: PathBuf,
}

impl Sandbox {
    /// Create a new sandbox rooted at `root`.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Resolve a user-provided path against the sandbox root, verifying it
    /// does not escape the sandbox. Works for paths that don't exist yet
    /// (write operations) by canonicalizing the existing prefix.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let joined = self.root.join(path);
        // Canonicalize the longest existing prefix, then re-append the rest.
        let canonical = match joined.canonicalize() {
            Ok(c) => c,
            Err(_) => {
                // Path doesn't fully exist yet — walk up to find an existing ancestor.
                let mut components = joined.components();
                let mut existing = PathBuf::new();
                for component in &mut components {
                    let candidate = existing.join(&component);
                    if candidate.exists() {
                        existing = candidate.canonicalize().map_err(|e| {
                            format!("path error: {e}")
                        })?;
                    } else {
                        existing = existing.join(component);
                        // Remaining components are appended as-is (no canonicalize).
                        for rest in components {
                            existing = existing.join(rest);
                        }
                        break;
                    }
                }
                existing
            }
        };
        // Strip trailing symlinks/dots and compare against canonical root.
        let root_canonical = self.root.canonicalize().map_err(|e| {
            format!("sandbox root error: {e}")
        })?;
        // For escape detection, compare using the canonical prefix.
        if !canonical.starts_with(&root_canonical) {
            return Err("path escapes sandbox".to_string());
        }
        Ok(canonical)
    }
}

/// Execute a tool call within the sandbox. Returns the result as a string
/// suitable for injection as a `tool`-role message.
///
/// `update_tasks` is handled by the worker's agentic loop (it targets the UI,
/// not the filesystem) and falls through to the unknown-tool error here.
pub fn execute_tool(call: &ToolCallMsg, sandbox: &Sandbox) -> Result<String, String> {
    let args: serde_json::Value = serde_json::from_str(&call.arguments)
        .map_err(|e| format!("invalid arguments: {e}"))?;
    match call.name.as_str() {
        "read_file" => {
            let path = args["path"].as_str().ok_or("missing path")?;
            let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
            let limit = (args["limit"].as_u64().unwrap_or(MAX_READ_LINES as u64) as usize)
                .min(MAX_READ_LINES);
            let full = sandbox.resolve(path)?;
            let text = fs::read_to_string(&full).map_err(|e| format!("read error: {e}"))?;
            let total_lines = text.lines().count();
            let mut out = String::new();
            let mut bytes = 0usize;
            let mut last_shown = offset.saturating_sub(1);
            for (idx, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
                let numbered = format!("{:>6}\t{}\n", idx + 1, line);
                if bytes + numbered.len() > MAX_READ_BYTES {
                    break;
                }
                bytes += numbered.len();
                out.push_str(&numbered);
                last_shown = idx + 1;
            }
            if last_shown < total_lines {
                out.push_str(&format!(
                    "[truncated: showing lines {offset}-{last_shown} of {total_lines}; \
                     use offset/limit to read more]"
                ));
            }
            Ok(out)
        }
        "write_file" => {
            let path = args["path"].as_str().ok_or("missing path")?;
            let content = args["content"].as_str().ok_or("missing content")?;
            let full = sandbox.resolve(path)?;
            // Create parent directories if needed.
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("mkdir error: {e}"))?;
            }
            fs::write(&full, content).map_err(|e| format!("write error: {e}"))?;
            Ok(format!("wrote {} bytes", content.len()))
        }
        "edit_file" => {
            let path = args["path"].as_str().ok_or("missing path")?;
            let old = args["old_string"].as_str().ok_or("missing old_string")?;
            let new = args["new_string"].as_str().ok_or("missing new_string")?;
            let full = sandbox.resolve(path)?;
            let existing =
                fs::read_to_string(&full).map_err(|e| format!("read error: {e}"))?;
            if !existing.contains(old) {
                return Err("old_string not found in file".to_string());
            }
            let replaced = existing.replace(old, new);
            fs::write(&full, &replaced).map_err(|e| format!("write error: {e}"))?;
            Ok("edited".to_string())
        }
        "run_command" => {
            let command = args["command"].as_str().ok_or("missing command")?;
            let workdir = args["workdir"].as_str().unwrap_or(".");
            let timeout_ms = args["timeout_ms"]
                .as_u64()
                .unwrap_or(DEFAULT_CMD_TIMEOUT_MS)
                .min(MAX_CMD_TIMEOUT_MS);
            let cwd = sandbox.resolve(workdir)?;
            run_shell_command(command, &cwd, std::time::Duration::from_millis(timeout_ms))
        }
        "list_files" => {
            let path = args["path"].as_str().unwrap_or(".");
            let full = sandbox.resolve(path)?;
            let mut entries = Vec::new();
            for entry in fs::read_dir(&full).map_err(|e| format!("read dir error: {e}"))? {
                let entry = entry.map_err(|e| format!("entry error: {e}"))?;
                let name = entry.file_name().to_string_lossy().to_string();
                let suffix = if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    "/"
                } else {
                    ""
                };
                entries.push(format!("{name}{suffix}"));
            }
            entries.sort();
            let total = entries.len();
            let mut out: String = entries
                .into_iter()
                .take(MAX_LIST_ENTRIES)
                .collect::<Vec<_>>()
                .join("\n");
            if total > MAX_LIST_ENTRIES {
                out.push_str(&format!(
                    "\n[truncated: {total} entries, showing first {MAX_LIST_ENTRIES}]"
                ));
            }
            Ok(out)
        }
        "search_files" => {
            let path = args["path"].as_str().ok_or("missing path")?;
            let pattern = args["pattern"].as_str().ok_or("missing pattern")?;
            let full = sandbox.resolve(path)?;
            let output = Command::new("grep")
                .args([
                    "-r",
                    "-n",
                    "--exclude-dir=target",
                    "--exclude-dir=node_modules",
                    "--exclude-dir=.git",
                    "--exclude-dir=old",
                    pattern,
                ])
                .arg(&full)
                .output()
                .map_err(|e| format!("grep error: {e}"))?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut lines = stdout.lines();
            let mut hits: Vec<&str> = Vec::new();
            let mut truncated = false;
            for line in &mut lines {
                if hits.len() >= MAX_SEARCH_MATCHES {
                    truncated = true;
                    break;
                }
                hits.push(line);
            }
            let mut out = hits.join("\n");
            if truncated {
                out.push_str(&format!(
                    "\n[truncated at {MAX_SEARCH_MATCHES} matches; narrow the pattern]"
                ));
            }
            Ok(out)
        }
        _ => Err(format!("unknown tool: {}", call.name)),
    }
}

/// Execute a tool call with a pre-execution checkpoint of every file the call
/// will modify. The snapshot persists only when the tool succeeds, so a
/// failed call never pollutes the checkpoint list. This is the single choke
/// point for tool execution from both the worker (auto-exec) and the UI
/// (approval) paths.
pub fn execute_tool_checked(
    call: &ToolCallMsg,
    sandbox: &Sandbox,
    store: &crate::tui::checkpoints::CheckpointStore,
) -> Result<String, String> {
    let pending = store.capture(call, sandbox);
    let result = execute_tool(call, sandbox);
    if result.is_ok() {
        if let Some(p) = pending {
            store.persist(p);
        }
    }
    result
}

/// Run `sh -c <command>` with a wall-clock timeout. Stdout/stderr are drained
/// on reader threads so a chatty child can't deadlock on a full pipe, and the
/// combined result is truncated past MAX_CMD_BYTES keeping head and tail.
fn run_shell_command(
    command: &str,
    cwd: &std::path::Path,
    timeout: std::time::Duration,
) -> Result<String, String> {
    use std::io::Read;

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("exec error: {e}"))?;

    let mut out_pipe = child.stdout.take().expect("stdout piped");
    let mut err_pipe = child.stderr.take().expect("stderr piped");
    let out_reader =
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = out_pipe.read_to_end(&mut buf);
            buf
        });
    let err_reader =
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = err_pipe.read_to_end(&mut buf);
            buf
        });

    let start = std::time::Instant::now();
    let (status, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(s)) => break (Some(s), false),
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait(); // reap
                    break (None, true);
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("wait error: {e}"));
            }
        }
    };

    let stdout = String::from_utf8_lossy(&out_reader.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err_reader.join().unwrap_or_default()).into_owned();
    let combined = match (timed_out, status) {
        (true, _) => format!(
            "timed out after {}s (killed)\n{}\n{}",
            timeout.as_secs(),
            stdout,
            stderr
        ),
        (false, Some(s)) => format!(
            "exit {}\n{}\n{}",
            s.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
            stdout,
            stderr
        ),
        (false, None) => format!("{stdout}\n{stderr}"),
    };
    Ok(cap_head_tail(&combined, MAX_CMD_BYTES))
}

/// Truncate `s` to at most `max_bytes`, keeping the head and tail with a
/// marker in the middle so both the start and end of the output survive.
fn cap_head_tail(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let head = max_bytes * 3 / 4;
    let tail = max_bytes / 4;
    let head = s.floor_char_boundary(head);
    let tail = s.ceil_char_boundary(s.len() - tail);
    format!(
        "{}\n[… {} bytes omitted …]\n{}",
        &s[..head],
        s.len() - (s.len() - tail + head),
        &s[tail..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_rejects_escape() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().to_path_buf());
        assert!(sb.resolve("../etc/passwd").is_err());
    }

    #[test]
    fn sandbox_accepts_child() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().to_path_buf());
        fs::write(dir.path().join("test.txt"), b"hello").unwrap();
        assert!(sb.resolve("test.txt").is_ok());
    }

    #[test]
    fn read_write_file() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().to_path_buf());
        let write_call = ToolCallMsg {
            id: "1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "test.txt", "content": "hello world"})
                .to_string(),
        };
        assert!(execute_tool(&write_call, &sb).is_ok());
        let read_call = ToolCallMsg {
            id: "2".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "test.txt"}).to_string(),
        };
        // read_file returns cat -n style line-numbered output.
        assert_eq!(execute_tool(&read_call, &sb).unwrap(), "     1\thello world\n");
    }

    #[test]
    fn edit_file_replaces_text() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().to_path_buf());
        let write_call = ToolCallMsg {
            id: "1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "a.txt", "content": "hello world"})
                .to_string(),
        };
        execute_tool(&write_call, &sb).unwrap();
        let edit_call = ToolCallMsg {
            id: "2".into(),
            name: "edit_file".into(),
            arguments: serde_json::json!({"path": "a.txt", "old_string": "world", "new_string": "rust"})
                .to_string(),
        };
        assert_eq!(execute_tool(&edit_call, &sb).unwrap(), "edited");
        let read_call = ToolCallMsg {
            id: "3".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "a.txt"}).to_string(),
        };
        assert_eq!(execute_tool(&read_call, &sb).unwrap(), "     1\thello rust\n");
    }

    #[test]
    fn read_file_offset_and_limit() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().to_path_buf());
        let content: String = (1..=10).map(|i| format!("line{i}\n")).collect();
        fs::write(dir.path().join("big.txt"), content).unwrap();
        let call = ToolCallMsg {
            id: "1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "big.txt", "offset": 4, "limit": 2})
                .to_string(),
        };
        let out = execute_tool(&call, &sb).unwrap();
        assert!(out.contains("     4\tline4\n"));
        assert!(out.contains("     5\tline5\n"));
        assert!(!out.contains("line3"));
        // Lines remain after the window, so a truncation note is appended.
        assert!(out.contains("of 10"));
    }

    #[test]
    fn read_file_truncates_huge_file() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().to_path_buf());
        let content = "x".repeat(MAX_READ_BYTES * 2);
        fs::write(dir.path().join("huge.txt"), content).unwrap();
        let call = ToolCallMsg {
            id: "1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "huge.txt"}).to_string(),
        };
        let out = execute_tool(&call, &sb).unwrap();
        assert!(out.len() <= MAX_READ_BYTES + 128);
    }

    #[test]
    fn run_command_respects_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().to_path_buf());
        let call = ToolCallMsg {
            id: "1".into(),
            name: "run_command".into(),
            arguments: serde_json::json!({"command": "sleep 30", "timeout_ms": 200})
                .to_string(),
        };
        let start = std::time::Instant::now();
        let out = execute_tool(&call, &sb).unwrap();
        assert!(start.elapsed() < std::time::Duration::from_secs(10));
        assert!(out.contains("timed out"));
    }

    #[test]
    fn run_command_captures_output_and_status() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().to_path_buf());
        let call = ToolCallMsg {
            id: "1".into(),
            name: "run_command".into(),
            arguments: serde_json::json!({"command": "echo hello; exit 3"}).to_string(),
        };
        let out = execute_tool(&call, &sb).unwrap();
        assert!(out.starts_with("exit 3"));
        assert!(out.contains("hello"));
    }

    #[test]
    fn parse_update_tasks_validates_status() {
        let ok = parse_update_tasks(
            &serde_json::json!({
                "tasks": [
                    {"id": "1", "title": "Do thing"},
                    {"id": "2", "title": "Next", "status": "in-progress",
                     "description": "detail"}
                ]
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(ok.len(), 2);
        assert_eq!(ok[0].status, "pending");
        assert_eq!(ok[1].status, "in-progress");

        let err = parse_update_tasks(
            &serde_json::json!({"tasks": [{"id": "1", "title": "x", "status": "bogus"}]})
                .to_string(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn summarize_tasks_counts_statuses() {
        let items = vec![
            TaskItem {
                id: "1".into(),
                title: "a".into(),
                description: None,
                status: "pending".into(),
            },
            TaskItem {
                id: "2".into(),
                title: "b".into(),
                description: None,
                status: "completed".into(),
            },
        ];
        let summary = summarize_tasks(&items);
        assert!(summary.contains("2 total"));
        assert!(summary.contains("1 pending"));
        assert!(summary.contains("1 completed"));
    }

    #[test]
    fn strip_tool_markup_removes_known_tags() {
        let hermes = r#"Let me check. <tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>"#;
        assert_eq!(strip_tool_markup(hermes), "Let me check.");
        let lfm = "ok<|tool_call_start|>[read_file(path=\"a\")]<|tool_call_end|>done";
        assert_eq!(strip_tool_markup(lfm), "ok\ndone");
        // Unterminated tag: drop to end.
        assert_eq!(strip_tool_markup("text <tool_call>{broken"), "text");
        // No markup: unchanged (except trim).
        assert_eq!(strip_tool_markup("  plain answer  "), "plain answer");
    }

    #[test]
    fn cap_head_tail_keeps_both_ends() {
        let s = format!("{}{}{}", "a".repeat(100), "b".repeat(100), "c".repeat(100));
        let capped = cap_head_tail(&s, 120);
        assert!(capped.starts_with("aaa"));
        assert!(capped.ends_with("ccc"));
        assert!(capped.contains("bytes omitted"));
    }

    #[test]
    fn edit_file_rejects_missing_old_string() {
        let dir = tempfile::tempdir().unwrap();
        let sb = Sandbox::new(dir.path().to_path_buf());
        let write_call = ToolCallMsg {
            id: "1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "a.txt", "content": "hello"}).to_string(),
        };
        execute_tool(&write_call, &sb).unwrap();
        let edit_call = ToolCallMsg {
            id: "2".into(),
            name: "edit_file".into(),
            arguments: serde_json::json!({"path": "a.txt", "old_string": "missing", "new_string": "x"})
                .to_string(),
        };
        assert!(execute_tool(&edit_call, &sb).is_err());
    }
}
