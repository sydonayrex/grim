//! Persistent tool-permission rules for the agentic TUI.
//!
//! When the user answers "a" (always allow) on a tool approval prompt, a rule
//! is appended here and saved to `$XDG_CONFIG_HOME/grim/permissions.toml`
//! (falling back to `~/.config/grim/`). The worker consults the shared rules
//! before asking for approval, so an allowed tool runs without prompting.
//!
//! Rules are name-based for file tools and binary-prefix-based for
//! `run_command`: allowing `run_command` with prefix `cargo` permits any
//! command whose first shell word is `cargo`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// One always-allow rule.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AllowRule {
    /// Tool name, e.g. "write_file", "run_command".
    pub tool: String,
    /// For run_command: the first word of the shell command that is allowed
    /// (e.g. "cargo", "git"). None means any command is allowed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_prefix: Option<String>,
}

/// The set of persisted allow rules.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PermissionRules {
    #[serde(default)]
    pub allow: Vec<AllowRule>,
}

/// Shared handle passed to both the App (which adds rules) and the worker
/// (which checks them before prompting).
pub type SharedPermissions = Arc<Mutex<PermissionRules>>;

/// Wrap rules in the shared handle used across both threads.
pub fn shared(rules: PermissionRules) -> SharedPermissions {
    Arc::new(Mutex::new(rules))
}

/// Build the rule implied by approving a tool call with "always allow".
///
/// For run_command the rule is pinned to the command's first word; for other
/// tools it covers the whole tool.
pub fn rule_for_tool_call(name: &str, arguments: &str) -> AllowRule {
    let command_prefix = if name == "run_command" {
        serde_json::from_str::<serde_json::Value>(arguments)
            .ok()
            .and_then(|v| v["command"].as_str().map(|s| s.to_string()))
            .and_then(|cmd| cmd.split_whitespace().next().map(|s| s.to_string()))
    } else {
        None
    };
    AllowRule {
        tool: name.to_string(),
        command_prefix,
    }
}

impl PermissionRules {
    /// Path of the permissions file: `$XDG_CONFIG_HOME/grim/permissions.toml`,
    /// falling back to `~/.config/grim/permissions.toml`.
    pub fn config_path() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
        Some(base.join("grim").join("permissions.toml"))
    }

    /// Load rules from disk. A missing or unreadable file yields empty rules —
    /// the TUI works fine without ever having saved permissions.
    pub fn load() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&text).unwrap_or_default()
    }

    /// Persist rules to disk, creating the config directory if needed.
    pub fn save(&self) -> Result<(), String> {
        let path = Self::config_path().ok_or("no config directory available")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir error: {e}"))?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// True if a tool call is covered by an existing allow rule.
    pub fn permits(&self, tool: &str, arguments: &str) -> bool {
        let rule = rule_for_tool_call(tool, arguments);
        self.allow.contains(&rule)
            || self
                .allow
                .iter()
                .any(|r| r.tool == tool && r.command_prefix.is_none())
    }

    /// Add a rule (deduplicated).
    pub fn add(&mut self, rule: AllowRule) {
        if !self.allow.contains(&rule) {
            self.allow.push(rule);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_tool_rule_matches_by_name() {
        let mut rules = PermissionRules::default();
        rules.add(AllowRule {
            tool: "write_file".into(),
            command_prefix: None,
        });
        assert!(rules.permits("write_file", r#"{"path":"a.txt"}"#));
        assert!(!rules.permits("edit_file", r#"{"path":"a.txt"}"#));
    }

    #[test]
    fn run_command_rule_matches_first_word() {
        let mut rules = PermissionRules::default();
        rules.add(rule_for_tool_call(
            "run_command",
            r#"{"command": "cargo test --lib"}"#,
        ));
        assert!(rules.permits("run_command", r#"{"command": "cargo build"}"#));
        assert!(!rules.permits("run_command", r#"{"command": "cargo-uninstall x"}"#));
        assert!(!rules.permits("run_command", r#"{"command": "rm -rf /"}"#));
    }

    #[test]
    fn unprefixless_run_command_rule_allows_everything() {
        let mut rules = PermissionRules::default();
        rules.add(AllowRule {
            tool: "run_command".into(),
            command_prefix: None,
        });
        assert!(rules.permits("run_command", r#"{"command": "make clean"}"#));
    }

    #[test]
    fn rules_roundtrip_through_toml() {
        let mut rules = PermissionRules::default();
        rules.add(AllowRule {
            tool: "edit_file".into(),
            command_prefix: None,
        });
        rules.add(AllowRule {
            tool: "run_command".into(),
            command_prefix: Some("cargo".into()),
        });
        let text = toml::to_string_pretty(&rules).unwrap();
        let parsed: PermissionRules = toml::from_str(&text).unwrap();
        assert_eq!(parsed, rules);
    }

    #[test]
    fn add_deduplicates() {
        let mut rules = PermissionRules::default();
        rules.add(AllowRule {
            tool: "write_file".into(),
            command_prefix: None,
        });
        rules.add(AllowRule {
            tool: "write_file".into(),
            command_prefix: None,
        });
        assert_eq!(rules.allow.len(), 1);
    }
}
