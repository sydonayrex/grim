//! Minimal MCP client over stdio: one child process per server, JSON-RPC
//! lines on stdin/stdout, reader thread feeding a timeout-guarded channel.
//! Everything a server writes is untrusted input: non-JSON lines are dropped,
//! results are parsed defensively.

use crate::tui::mcp::types;
use std::io::{BufRead, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::Duration;

const PROTOCOL_VERSION: &str = "2024-11-05";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

pub struct McpClient {
    child: Child,
    stdin: ChildStdin,
    responses: std::sync::mpsc::Receiver<serde_json::Value>,
    next_id: u64,
}

impl McpClient {
    pub fn spawn(command: &str, args: &[String]) -> Result<Self, String> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {command}: {e}"))?;
        let stdin = child.stdin.take().ok_or("child stdin missing")?;
        let stdout = child.stdout.take().ok_or("child stdout missing")?;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<serde_json::Value>(&line) {
                    Ok(v) => {
                        if tx.send(v).is_err() {
                            break;
                        }
                    }
                    Err(_) => continue, // untrusted output: drop non-JSON lines
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            responses: rx,
            next_id: 1,
        })
    }

    fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let req = serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut line = serde_json::to_string(&req).map_err(|e| e.to_string())?;
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .map_err(|e| format!("mcp write: {e}"))?;
        self.stdin
            .flush()
            .map_err(|e| format!("mcp flush: {e}"))?;
        loop {
            let v = self
                .responses
                .recv_timeout(REQUEST_TIMEOUT)
                .map_err(|_| "mcp timeout waiting for response".to_string())?;
            if v.get("id").and_then(|i| i.as_u64()) == Some(id) {
                if let Some(err) = v.get("error") {
                    return Err(format!("mcp error: {err}"));
                }
                return Ok(v.get("result").cloned().unwrap_or(serde_json::Value::Null));
            }
            // Notification or stale response: ignore, keep waiting.
        }
    }

    pub fn initialize(&mut self) -> Result<(), String> {
        self.request(
            "initialize",
            serde_json::json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "grim-tui", "version": env!("CARGO_PKG_VERSION")}
            }),
        )
        .map(|_| ())
    }

    pub fn list_tools(&mut self) -> Result<Vec<types::McpTool>, String> {
        let result = self.request("tools/list", serde_json::json!({}))?;
        serde_json::from_value::<types::ToolsListResult>(result)
            .map(|r| r.tools)
            .map_err(|e| e.to_string())
    }

    pub fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, String> {
        let result = self.request(
            "tools/call",
            serde_json::json!({"name": name, "arguments": arguments}),
        )?;
        let parsed: types::CallToolResult =
            serde_json::from_value(result).map_err(|e| e.to_string())?;
        let text: String = parsed
            .content
            .iter()
            .filter(|c| c.kind == "text" || c.kind.is_empty())
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if parsed.is_error {
            Err(if text.is_empty() {
                "mcp tool error".into()
            } else {
                text
            })
        } else {
            Ok(text)
        }
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canned JSON-RPC server: answers initialize, tools/list, tools/call in
    /// order, then blocks on a final read so stdin stays open.
    fn fake_server_script(dir: &std::path::Path) -> String {
        let script = dir.join("fake_mcp.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\n\
             read _init_req\n\
             printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"serverInfo\":{\"name\":\"fake\",\"version\":\"0\"}}}'\n\
             read _list_req\n\
             printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo back\",\"inputSchema\":{\"type\":\"object\"}}]}}'\n\
             read _call_req\n\
             printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"hello from mcp\"}]}}'\n\
             read _hold_open\n",
        )
        .unwrap();
        script.to_string_lossy().to_string()
    }

    #[test]
    fn initialize_list_and_call_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let script = fake_server_script(dir.path());
        let mut c = McpClient::spawn("sh", &[script]).unwrap();
        c.initialize().unwrap();
        let tools = c.list_tools().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        let out = c.call_tool("echo", serde_json::json!({})).unwrap();
        assert_eq!(out, "hello from mcp");
    }

    #[test]
    fn spawn_failure_is_reported() {
        assert!(McpClient::spawn("/nonexistent/grim-fake-cmd-9x", &[]).is_err());
    }
}
