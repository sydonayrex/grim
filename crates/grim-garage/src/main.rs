//! grim-garage — local-first training dashboard backend.
//!
//! Three concurrent tasks at startup:
//! - **axum listener** on `GRIM_GARAGE_BIND_ADDR` (default 8741). Serves
//!   both grim-garage's `/api/*` + `/sse/metrics/*` AND the CVKG
//!   dev-server routes from `cvkg_webkit_server::router::create_router`.
//! - **runtime poller** that keeps the displayed `DisplayState` in
//!   sync with the local backend at the configured interval.
//! - **renderer host** that owns a `CvkgHeadless` and re-renders one
//!   frame per `DisplayState` mutation. Cheap, no display required —
//!   produces SVG/VDom each tick for any future /api/system/debug-style
//!   introspection. The mutex/print out of `debug_string()` proves
//!   the runtime is alive in CI just as well as in a real GUI session.
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
use grim_garage::{jobs::JobRegistry, renderer_host::RendererHandle, routes,
    ui_state::DisplayState, ui_state::GarageClient, ui_state::Poller,
    view_model::ViewModel};

#[derive(Parser, Debug)]
#[command(name = "grim-garage", about = "Grim's Garage — local-first training dashboard", version)]
struct Args {
    /// Bind address (overrides `GRIM_GARAGE_BIND_ADDR`).
    #[arg(long, env = "GRIM_GARAGE_BIND_ADDR", default_value = "0.0.0.0:8741")]
    bind: String,
    /// Renderer frame interval in milliseconds (0 = no renderer).
    #[arg(long, env = "GRIM_GARAGE_RENDERER_INTERVAL_MS", default_value = "1000")]
    renderer_ms: u64,
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

    // Spawn the headless renderer task. It pulls the latest DisplayState
    // every `renderer_ms` and renders one CVKG frame. The result is
    // logged at debug level — production would forward it to a winit
    // window or another consumer via a channel.
    if args.renderer_ms > 0 {
        spawn_renderer_task(Arc::clone(&display_state), args.renderer_ms);
        tracing::info!("renderer host spawned (interval {}ms)", args.renderer_ms);
    } else {
        tracing::info!("renderer host disabled (interval=0)");
    }

    axum::serve(listener, router).await?;

    Ok(())
}

/// Background tokio task that owns a `CvkgHeadless` instance and re-renders
/// it on every significant `DisplayState` mutation. Cheap: a no-display
/// CVKG headless frame is ~1µs, so an interval of 1s is fine for CI.
fn spawn_renderer_task(display_state: Arc<tokio::sync::Mutex<DisplayState>>, interval_ms: u64) {
    tokio::spawn(async move {
        let mut handle: Option<RendererHandle> = None;
        let mut last_refresh_signature: u64 = 0;
        let mut frame_count: u64 = 0;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(interval_ms)).await;
            // Snapshot the current DisplayState cheaply (Vec<usize for sizes).
            let snapshot = {
                let s = display_state.lock().await;
                let snap = s.snapshot();
                (
                    snap.models.len(),
                    snap.datasets.len(),
                    snap.devices.len(),
                    snap.jobs.len(),
                    snap.config.lora_rank,
                    snap.config.training_mode.clone(),
                )
            };
            // 64-bit cheap hash so we re-render only when something changed.
            let signature = {
                let mut h: u64 = 1469598103934665603;
                for v in [
                    snapshot.0 as u64,
                    snapshot.1 as u64,
                    snapshot.2 as u64,
                    snapshot.3 as u64,
                    snapshot.4 as u64,
                ] {
                    h ^= v;
                    h = h.wrapping_mul(1099511628211);
                }
                h ^= seq_hash_str(&snapshot.5);
                h
            };
            if signature != last_refresh_signature || handle.is_none() {
                last_refresh_signature = signature;
                let vm = {
                    let s = display_state.lock().await;
                    ViewModel::from(&*s)
                };
                if handle.is_none() {
                    handle = Some(RendererHandle::from_view_model(&vm));
                } else {
                    handle.as_mut().unwrap().refresh(&vm);
                }
                frame_count += 1;
                if let Some(h) = handle.as_ref() {
                    let frame = h.render_frame().expect("frame");
                    if frame_count <= 3 {
                        let dbg = h.debug_string();
                        tracing::debug!(
                            "renderer frame {}: svg_len={} root={:?} preview={:?}",
                            frame_count,
                            frame.svg.len(),
                            frame.root.is_some(),
                            dbg.chars().take(80).collect::<String>()
                        );
                    }
                }
                if frame_count % 60 == 0 {
                    tracing::info!("renderer heartbeat: {} frames rendered", frame_count);
                }
            }
        }
    });
}

/// Cheap unstable hash for a `str` — used as part of the renderer's
/// "has the state changed?" signature.
fn seq_hash_str(s: &str) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}
