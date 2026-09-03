//! MCP server registry: config from `$XDG_CONFIG_HOME/grim/mcp.toml`,
//! one `McpClient` per enabled server, tools exposed to the model with
//! `mcp_<server>_<tool>` names. All MCP output is untrusted: truncated hard.
//!
//! Config format:
//! ```toml
//! [servers.filesystem]
//! command = "mcp-server-filesystem"
//! args = ["/home/me/project"]
//! enabled = true
//! ```

use crate::tui::mcp::client::McpClient;
use crate::tui::mcp::types::McpTool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const MAX_TOOL_OUTPUT_CHARS: usize = 20_000;

#[derive(serde::Deserialize, Clone, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: HashMap<String, ServerConfig>,
}

#[derive(serde::Deserialize, Clone)]
pub struct ServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

pub struct McpManager {
    pub config: McpConfig,
    clients: HashMap<String, McpClient>,
    tools: Vec<(String, McpTool)>, // (server name, tool)
    statuses: Vec<String>,
}

pub type SharedMcp = Arc<Mutex<McpManager>>;

/// Model-facing tool name for an MCP server tool: `mcp_<server>_<tool>`,
/// lowercased, non-alphanumerics collapsed to `_`.
pub fn sanitize(server: &str, tool: &str) -> String {
    let clean: String = format!("{server}_{tool}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("mcp_{clean}")
}

impl McpManager {
    pub fn load() -> Self {
        let config = crate::tui::paths::config_dir()
            .and_then(|d| std::fs::read_to_string(d.join("mcp.toml")).ok())
            .and_then(|t| toml::from_str::<McpConfig>(&t).ok())
            .unwrap_or_default();
        Self {
            config,
            clients: HashMap::new(),
            tools: Vec::new(),
            statuses: Vec::new(),
        }
    }

    /// Spawn + initialize every enabled server and cache its tool list.
    /// Failures are reported in status lines, never fatal.
    pub fn connect_all(&mut self) -> Vec<String> {
        self.statuses.clear();
        self.tools.clear();
        let servers: Vec<(String, ServerConfig)> = self
            .config
            .servers
            .iter()
            .filter(|(_, c)| c.enabled)
            .map(|(n, c)| (n.clone(), c.clone()))
            .collect();
        for (name, cfg) in servers {
            match McpClient::spawn(&cfg.command, &cfg.args).and_then(|mut c| {
                c.initialize()?;
                let tools = c.list_tools()?;
                Ok((c, tools))
            }) {
                Ok((client, tools)) => {
                    self.statuses
                        .push(format!("{name}: connected, {} tool(s)", tools.len()));
                    for t in tools {
                        self.tools.push((name.clone(), t));
                    }
                    self.clients.insert(name, client);
                }
                Err(e) => self.statuses.push(format!("{name}: FAILED — {e}")),
            }
        }
        self.statuses.clone()
    }

    /// grim `ToolDef`s for every connected MCP tool, with prefixed names.
    pub fn tool_defs(&self) -> Vec<grim_format::ToolDef> {
        self.tools
            .iter()
            .map(|(server, t)| grim_format::ToolDef {
                r#type: "function".to_string(),
                function: grim_format::FunctionDef {
                    name: sanitize(server, &t.name),
                    description: Some(format!("(mcp:{server}) {}", t.description)),
                    parameters: if t.input_schema.is_null() {
                        None
                    } else {
                        Some(t.input_schema.clone())
                    },
                },
            })
            .collect()
    }

    /// Route a model tool call to its MCP server. Output is truncated hard —
    /// it is untrusted input destined for a small context window.
    pub fn call(&mut self, prefixed: &str, arguments_json: &str) -> Result<String, String> {
        let (server, tool) = self
            .tools
            .iter()
            .find(|(s, t)| sanitize(s, &t.name) == prefixed)
            .map(|(s, t)| (s.clone(), t.name.clone()))
            .ok_or_else(|| format!("unknown mcp tool: {prefixed}"))?;
        let args: serde_json::Value = serde_json::from_str(arguments_json)
            .unwrap_or(serde_json::Value::Object(Default::default()));
        let client = self
            .clients
            .get_mut(&server)
            .ok_or_else(|| format!("mcp server {server} not connected"))?;
        let out = client.call_tool(&tool, args)?;
        let mut truncated: String = out.chars().take(MAX_TOOL_OUTPUT_CHARS).collect();
        if out.chars().count() > MAX_TOOL_OUTPUT_CHARS {
            truncated.push_str("\n[truncated by grim]");
        }
        Ok(truncated)
    }

    pub fn status_lines(&self) -> Vec<String> {
        self.statuses.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_is_prefixed_and_clean() {
        assert_eq!(sanitize("File System", "read_file"), "mcp_file_system_read_file");
        assert_eq!(sanitize("git", "commit!"), "mcp_git_commit_");
    }

    #[test]
    fn call_unknown_tool_errors() {
        let mut m = McpManager::load();
        assert!(m.call("mcp_nope_x", "{}").is_err());
    }

    #[test]
    fn load_without_config_is_empty() {
        let _guard = crate::tui::paths::env_lock();
        let dir = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("XDG_CONFIG_HOME", dir.path()) };
        let m = McpManager::load();
        unsafe { std::env::remove_var("XDG_CONFIG_HOME") };
        assert!(m.config.servers.is_empty());
    }

    #[test]
    fn parse_mcp_toml() {
        let cfg: McpConfig = toml::from_str(
            "[servers.filesystem]\ncommand = \"mcp-server-fs\"\nargs = [\"/tmp\"]\nenabled = false\n",
        )
        .unwrap();
        assert_eq!(cfg.servers["filesystem"].command, "mcp-server-fs");
        assert!(!cfg.servers["filesystem"].enabled);
    }

    #[test]
    fn tool_defs_use_prefixed_function_shape() {
        let mut m = McpManager::load();
        m.tools.push((
            "fs".into(),
            McpTool {
                name: "read".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({"type": "object"}),
            },
        ));
        let defs = m.tool_defs();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].r#type, "function");
        assert_eq!(defs[0].function.name, "mcp_fs_read");
        assert_eq!(
            defs[0].function.description.as_deref(),
            Some("(mcp:fs) Read a file")
        );
        assert_eq!(
            defs[0].function.parameters.as_ref().unwrap()["type"],
            "object"
        );
    }
}
