//! grim server - Alias for serve, starts the HTTP server daemon.

use grim_core::error::Result;
use grim_engine::Engine;

/// Start the server (alias for serve).
pub async fn cmd_server(address: &str, _config: &str, _plugins: &str) -> Result<()> {
    let engine = Engine::new(grim_engine::EngineConfig::default());
    eprintln!("[grim] server: binding to {} (Ollama-compatible)", address);
    grim_server::serve(address, engine, None).await
}