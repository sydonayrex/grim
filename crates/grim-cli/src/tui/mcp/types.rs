//! MCP JSON-RPC 2.0 wire types (subset: initialize, tools/list, tools/call).

use serde::{Deserialize, Serialize};

/// A tool advertised by an MCP server via `tools/list`.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

#[derive(Deserialize)]
pub struct ToolsListResult {
    #[serde(default)]
    pub tools: Vec<McpTool>,
}

#[derive(Deserialize)]
pub struct ContentItem {
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Deserialize)]
pub struct CallToolResult {
    #[serde(default)]
    pub content: Vec<ContentItem>,
    #[serde(default)]
    pub is_error: bool,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tools_list_result() {
        let v: ToolsListResult = serde_json::from_value(serde_json::json!({
            "tools": [{"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}}]
        }))
        .unwrap();
        assert_eq!(v.tools[0].name, "echo");
        assert_eq!(v.tools[0].input_schema["type"], "object");
    }

    #[test]
    fn parses_call_result_with_missing_fields() {
        let v: CallToolResult = serde_json::from_value(serde_json::json!({
            "content": [{"type": "text", "text": "hi"}]
        }))
        .unwrap();
        assert!(!v.is_error);
        assert_eq!(v.content[0].text, "hi");
    }

    #[test]
    fn mcp_tool_tolerates_missing_description_and_schema() {
        let v: McpTool = serde_json::from_value(serde_json::json!({"name": "x"})).unwrap();
        assert_eq!(v.name, "x");
        assert!(v.description.is_empty());
        assert!(v.input_schema.is_null());
    }
}
