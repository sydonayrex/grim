//! grim server — Alias for serve, starts the HTTP server daemon.

use grim_core::error::Result;
use grim_engine::Engine;

/// Start the server (alias for serve).
pub async fn cmd_server(address: &str, _config: &str, plugins: &str) -> Result<()> {
    let engine = Engine::new(grim_engine::EngineConfig::default());
    let plugin_registry = if !plugins.is_empty() {
        let mut registry = grim_plugin::PluginRegistry::new();
        match crate::plugin::load_plugins(plugins, &mut registry) {
            Ok(n) => eprintln!("[grim] server: loaded {n} plugin(s) from {plugins}"),
            Err(e) => eprintln!("[grim] server: failed to load plugins from {plugins}: {e}"),
        }
        Some(std::sync::Arc::new(registry))
    } else {
        None
    };
    eprintln!("[grim] server: binding to {} (Ollama-compatible)", address);
    grim_server::serve(address, engine, None, plugin_registry).await
}
