//! grim-garage — local-first training dashboard web application (WI-T9 & WI-T10).
//!
//! Serves the browser web UI and JSON API on `GRIM_GARAGE_BIND_ADDR`
//! (default `127.0.0.1:8741`).
//!
//! WI-4: the default bind is loopback, matching `grim serve`. `grim-garage`
//! exposes unauthenticated write endpoints (`/api/train/start`), so a
//! LAN-reachable default would put a training control plane on the network
//! without the operator asking for it. Binding publicly stays available via
//! `--bind 0.0.0.0:8741` or `GRIM_GARAGE_BIND_ADDR`, and warns when used.

use clap::Parser;
use grim_garage::{
    jobs::JobRegistry, routes, ui_state::DisplayState, ui_state::GarageClient, ui_state::Poller,
};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "grim-garage",
    about = "Grim's Garage — local-first training dashboard",
    version
)]
struct Args {
    /// Bind address (overrides `GRIM_GARAGE_BIND_ADDR`).
    #[arg(long, env = "GRIM_GARAGE_BIND_ADDR", default_value = "127.0.0.1:8741")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let state = routes::AppState {
        registry: Arc::new(JobRegistry::new()),
        engine: Arc::new(std::sync::Mutex::new(grim_engine::Engine::new(
            grim_engine::EngineConfig::default(),
        ))),
        tokenizer: Arc::new(std::sync::Mutex::new(None)),
        model_path: None,
    };
    let router = routes::build_router(state);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    let local = listener.local_addr()?;
    // WI-4: warn only when the operator explicitly opted into a non-loopback
    // bind, so the message never fires spuriously on the safe default.
    if !local.ip().is_loopback() {
        tracing::warn!(
            "grim-garage is bound to {local}, which is reachable from the network. \
             Its training-control endpoints are unauthenticated — bind to \
             127.0.0.1:8741 unless you intend to expose them."
        );
    }
    tracing::info!("grim-garage web server listening on http://{local}");
    tracing::info!("  api routes:    /api/*  /sse/metrics/*");
    tracing::info!("  web dashboard: http://{local}/");

    let display_state = Arc::new(tokio::sync::Mutex::new(DisplayState::new()));
    let client = GarageClient::new(format!("http://{local}"));
    let mut poller = Poller::new(client, Arc::clone(&display_state));
    let _ = poller.with_interval(std::time::Duration::from_secs(5));
    poller.spawn();
    tracing::info!("display-state poller spawned (interval 5s)");

    axum::serve(listener, router).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// WI-4 regression guard: the default bind must stay on loopback.
    /// `grim-garage` serves unauthenticated training-control write endpoints,
    /// so a `0.0.0.0` default would silently publish them to the LAN.
    #[test]
    fn test_default_bind_is_loopback() {
        let args = Args::parse_from(["grim-garage"]);
        let addr: std::net::SocketAddr = args.bind.parse().expect("default bind must parse");
        assert!(
            addr.ip().is_loopback(),
            "default bind must be loopback, got {addr}"
        );
        assert_eq!(addr.port(), 8741);
    }

    /// An explicit non-loopback bind stays supported — the change tightens the
    /// default only, it does not remove the LAN-exposure capability.
    #[test]
    fn test_explicit_public_bind_still_accepted() {
        let args = Args::parse_from(["grim-garage", "--bind", "0.0.0.0:8741"]);
        let addr: std::net::SocketAddr = args.bind.parse().unwrap();
        assert!(!addr.ip().is_loopback());
    }

    /// The clap default string is what actually ships in `--help`; assert it
    /// directly so a future edit to the attribute cannot drift from the docs.
    #[test]
    fn test_clap_declared_default_value_is_loopback() {
        let cmd = Args::command();
        let arg = cmd.get_arguments().find(|a| a.get_id() == "bind").unwrap();
        let defaults: Vec<_> = arg
            .get_default_values()
            .iter()
            .map(|v| v.to_string_lossy().into_owned())
            .collect();
        assert_eq!(defaults, vec!["127.0.0.1:8741".to_string()]);
    }
}
