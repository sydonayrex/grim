//! grim-garage — local-first training dashboard backend.
//!
//! Listens on `GRIM_GARAGE_BIND_ADDR` (default `0.0.0.0:8741`). One axum
//! listener serves both grim-garage's `/api/*` + `/sse/metrics/*` AND the
//! CVKG dev-server routes from `cvkg_webkit_server::router::create_router`
//! (`/`, `/snapshot`, `/build`, `/health/{liveness,readiness}`, `/metrics`,
//! `/api/system/time`, `/cvkg-ws`, `/hmr`, static dirs).
//!
//! CVKG's hostname, package dir, rate limit, etc. are still controlled
//! via its own env vars (`CVKG_BIND_ADDR`, `CVKG_PKG_DIR`, etc.) — only
//! the bind port is fixed by grim-garage.
//!
//! Run:
//!   cargo run -p grim-garage --release
//!   # → open http://localhost:8741/

use std::sync::Arc;

use clap::Parser;
use grim_garage::{jobs::JobRegistry, routes, ui_state::DisplayState, ui_state::GarageClient,
    ui_state::Poller};

#[derive(Parser, Debug)]
#[command(name = "grim-garage", about = "Grim's Garage — local-first training dashboard", version)]
struct Args {
    /// Bind address (overrides `GRIM_GARAGE_BIND_ADDR`).
    #[arg(long, env = "GRIM_GARAGE_BIND_ADDR", default_value = "0.0.0.0:8741")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let state = routes::AppState { registry: Arc::new(JobRegistry::new()) };
    // One axum `Router` for both grim-garage's API and CVKG's dev-server.
    let router = routes::build_combined_router(state);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    let local = listener.local_addr()?;
    tracing::info!("grim-garage listening on http://{local}");
    tracing::info!("  grim routes:  /api/*  /sse/metrics/*");
    tracing::info!("  cvkg routes:  /  /snapshot  /build  /health/liveness");
    tracing::info!("                /health/readiness  /metrics  /api/system/time");
    tracing::info!("                /cvkg-ws  /hmr");

    // Spawn the runtime poller that keeps the UI's DisplayState in sync
    // with this very server. URL is `http://<local>` so the poller hits
    // the listener we just bound.
    let display_state = Arc::new(tokio::sync::Mutex::new(DisplayState::new()));
    let client = GarageClient::new(format!("http://{local}"));
    let mut poller = Poller::new(client, Arc::clone(&display_state));
    let _ = poller.with_interval(std::time::Duration::from_secs(5));
    poller.spawn();
    tracing::info!("display-state poller spawned (interval 5s, immediate initial refresh)");

    axum::serve(listener, router).await?;

    Ok(())
}
