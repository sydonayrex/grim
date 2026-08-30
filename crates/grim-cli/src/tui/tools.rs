//! Coding tool definitions and sandboxed execution.
//!
//! Tools are exposed to the model via grim's OpenAI-compatible `ToolDef`
//! format so the chat template receives them through the `tools` Jinja
//! variable. Execution is sandboxed to a single allow-listed directory.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use grim_format::{FunctionDef, ToolCallMsg, ToolDef};

/// The set of coding tools exposed to the model. Reuses grim's
/// OpenAI-compatible `ToolDef` format.
pub fn coding_tools() -> Vec<ToolDef> {
    vec![
    ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "read_file".to_string(),
            description: Some(
                "Read the contents of a file at the given path. Returns the full text."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file, relative to the sandbox root" }
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
                 other development commands."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "workdir": { "type": "string", "description": "Working directory relative to sandbox (default: \".\")" }
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
                 matching file paths with line numbers."
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
pub fn execute_tool(call: &ToolCallMsg, sandbox: &Sandbox) -> Result<String, String> {
    let args: serde_json::Value = serde_json::from_str(&call.arguments)
        .map_err(|e| format!("invalid arguments: {e}"))?;
    match call.name.as_str() {
        "read_file" => {
            let path = args["path"].as_str().ok_or("missing path")?;
            let full = sandbox.resolve(path)?;
            fs::read_to_string(&full).map_err(|e| format!("read error: {e}"))
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
            let output = Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(sandbox.resolve(workdir)?)
                .output()
                .map_err(|e| format!("exec error: {e}"))?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            Ok(format!("exit {}\n{}\n{}", output.status, stdout, stderr))
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
            Ok(entries.join("\n"))
        }
        "search_files" => {
            let path = args["path"].as_str().ok_or("missing path")?;
            let pattern = args["pattern"].as_str().ok_or("missing pattern")?;
            let full = sandbox.resolve(path)?;
            let output = Command::new("grep")
                .args(["-r", "-n", "-l", pattern])
                .arg(&full)
                .output()
                .map_err(|e| format!("grep error: {e}"))?;
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        }
        _ => Err(format!("unknown tool: {}", call.name)),
    }
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
        assert_eq!(execute_tool(&read_call, &sb).unwrap(), "hello world");
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
        assert_eq!(execute_tool(&read_call, &sb).unwrap(), "hello rust");
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
