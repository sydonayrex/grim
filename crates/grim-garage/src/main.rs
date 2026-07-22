//! grim-garage — local-first training dashboard web application (WI-T9 & WI-T10).
//!
//! Serves the browser web UI and JSON API on `GRIM_GARAGE_BIND_ADDR` (default 8741).

use std::sync::Arc;
use clap::Parser;
use grim_garage::{
    jobs::JobRegistry, routes,
    ui_state::DisplayState, ui_state::GarageClient, ui_state::Poller,
};

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
    let router = routes::build_router(state);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    let local = listener.local_addr()?;
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
