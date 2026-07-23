//! grim stop - Stop a currently running model (unload from memory).

use grim_core::error::{Error, Result};
use reqwest::Client;
use serde_json::Value;

/// Stop a currently running model by unloading it from the server.
pub async fn cmd_stop(model: &str, addr: &str) -> Result<()> {
    let client = Client::new();
    let url = format!("http://{}/v1/models/unload", addr);

    let req = serde_json::json!({ "name": model });
    let resp = client.post(&url)
        .json(&req)
        .send()
        .await
        .map_err(|e| Error::Config(format!("Failed to send unload request: {e}")))?;

    if resp.status().is_success() {
        let body: Value = resp.json().await
            .map_err(|e| Error::Config(format!("Failed to parse response: {e}")))?;
        println!("{}", body.get("message").and_then(|v| v.as_str()).unwrap_or("Model stopped"));
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Config(format!("Failed to stop model '{}': {} - {}", model, status, body)));
    }

    Ok(())
}