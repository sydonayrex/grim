//! `grim adapter` — runtime LoRA adapter management against a live server.
//!
//! Wraps `POST /v1/adapters/load`, `GET /v1/adapters`, and
//! `DELETE /v1/adapters/:name` so zero-downtime adapter swaps are scriptable
//! without curl. The server-side loader is honest about what it can apply:
//! logits-projection pairs load at runtime; per-layer projections (Q/K/V/O/
//! Gate/Up/Down) report 409 with the `grim merge` bake path.

use clap::Subcommand;
use grim_core::error::{Error, Result};

#[derive(Subcommand, Debug)]
pub enum AdapterCmd {
    /// Load a `grim train` sidecar (`*.grim.train`) without an engine restart.
    Load {
        /// Path to the sidecar written by `grim train`.
        path: String,
        /// Name used in per-request `"adapters": [..]` routing.
        #[arg(short, long)]
        name: String,
        /// Base model to attach to (defaults to the server's default model).
        #[arg(short, long)]
        base_model: Option<String>,
        /// Server address.
        #[arg(short, long, default_value = "127.0.0.1:11434")]
        addr: String,
    },
    /// List adapters currently loaded on the server.
    List {
        #[arg(short, long, default_value = "127.0.0.1:11434")]
        addr: String,
    },
    /// Unload an adapter by name (engine keeps running).
    Unload {
        name: String,
        #[arg(short, long, default_value = "127.0.0.1:11434")]
        addr: String,
    },
}

async fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| Error::Config(format!("failed to build client: {e}")))
}

fn map_err(e: reqwest::Error) -> Error {
    Error::Config(format!(
        "server request failed (is grim serve running?): {e}"
    ))
}

pub async fn cmd_adapter(cmd: AdapterCmd) -> Result<()> {
    match cmd {
        AdapterCmd::Load {
            path,
            name,
            base_model,
            addr,
        } => {
            let mut body = serde_json::json!({ "path": path, "name": name });
            if let Some(b) = base_model {
                body["base_model"] = serde_json::json!(b);
            }
            let res = client()
                .await?
                .post(format!("http://{addr}/v1/adapters/load"))
                .json(&body)
                .send()
                .await
                .map_err(map_err)?;
            let status = res.status();
            let json: serde_json::Value = res.json().await.map_err(map_err)?;
            if status.is_success() {
                println!(
                    "loaded '{}' ({}): applied {:?}",
                    name,
                    json.get("base_model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?"),
                    json.get("applied_tensors")
                        .and_then(|v| v.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0)
                );
                if let Some(skipped) = json.get("skipped_tensors").and_then(|v| v.as_array()) {
                    for s in skipped {
                        eprintln!(
                            "  skipped {}: {}",
                            s.get("tensor").and_then(|t| t.as_str()).unwrap_or("?"),
                            s.get("reason").and_then(|r| r.as_str()).unwrap_or("?")
                        );
                    }
                }
            } else {
                return Err(Error::Config(format!(
                    "load failed ({}): {} | bake with: {}",
                    status.as_u16(),
                    json["error"]["message"].as_str().unwrap_or("?"),
                    json["error"]["bake_command"]
                        .as_str()
                        .unwrap_or("grim merge"),
                )));
            }
        }
        AdapterCmd::List { addr } => {
            let res = client()
                .await?
                .get(format!("http://{addr}/v1/adapters"))
                .send()
                .await
                .map_err(map_err)?;
            let json: serde_json::Value = res.json().await.map_err(map_err)?;
            println!("=== Adapters on {addr} ===");
            match json.get("data").and_then(|v| v.as_array()) {
                Some(list) if !list.is_empty() => {
                    for a in list {
                        println!(
                            "  {} (base {})",
                            a.get("name").and_then(|n| n.as_str()).unwrap_or("?"),
                            a.get("base_model").and_then(|b| b.as_str()).unwrap_or("?"),
                        );
                    }
                }
                _ => println!("  (none loaded)"),
            }
        }
        AdapterCmd::Unload { name, addr } => {
            let res = client()
                .await?
                .delete(format!("http://{addr}/v1/adapters/{name}"))
                .send()
                .await
                .map_err(map_err)?;
            let status = res.status();
            let json: serde_json::Value = res.json().await.unwrap_or_default();
            if status.is_success() {
                println!("unloaded '{name}'");
            } else {
                return Err(Error::Config(format!(
                    "unload failed ({status}): {}",
                    json["error"]["message"].as_str().unwrap_or("?")
                )));
            }
        }
    }
    Ok(())
}
