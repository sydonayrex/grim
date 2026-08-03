//! Runtime environment configuration for grim.
//!
//! A single source of truth for the `GRIM_*` environment variables that the
//! CLI, server, and install script agree on. Values are read lazily and
//! re-read on every access so a systemd `Environment=` drop-in or a shell
//! `export GRIM_*` takes effect without restarting an already-running daemon
//! (the next request picks it up).

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

/// Parsed backend selector. `Auto` re-probes every time it is consulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Rocm,
    Cuda,
    Vulkan,
    Metal,
    Cpu,
}

impl Default for Backend {
    fn default() -> Self {
        match std::env::var("GRIM_BACKEND") {
            Ok(s) => match s.to_ascii_lowercase().as_str() {
                "rocm" => Backend::Rocm,
                "cuda" => Backend::Cuda,
                "vulkan" => Backend::Vulkan,
                "metal" => Backend::Metal,
                "cpu" => Backend::Cpu,
                "auto" | _ => Backend::Auto,
            },
            Err(_) => Backend::Auto,
        }
    }
}

/// Runtime knobs surfaced as environment variables.
#[derive(Debug, Clone)]
pub struct RuntimeEnv {
    /// Override for the server bind address. Only consulted when the caller
    /// does not pass an explicit address.
    pub host: Option<String>,
    /// Override for the server port. Only consulted when the caller does not
    /// pass an explicit address.
    pub port: Option<u16>,
    /// Override for the model context window (KV cache length). When set,
    /// takes precedence over the GGUF `max_position_embeddings`, capped
    /// downwards if the model advertises a smaller hard limit.
    pub context: Option<usize>,
    /// Backend selection (rocm/cuda/vulkan/metal/cpu/auto).
    pub backend: Backend,
    /// Comma-separated ordinal list of GPUs to use (e.g. `0,1`). Empty means
    /// "all visible devices".
    pub gpus: Vec<usize>,
    /// Tensor-parallel world size. Empty/0 means single-device (no TP); a
    /// value > 1 requires `world_size` GPUs and the backend's collective
    /// implementation (RCCL on ROCm, NCCL on CUDA). Honored by `Engine::new`.
    pub tp_size: usize,
    /// `Yes`/`No` — advisory multi-GPU parallelism hint read by the install
    /// script and surfaced to the scheduler. The engine honors it only on
    /// multi-GPU backends (RCCL / NCCL); on a single-GPU or CPU host it is
    /// a no-op.
    pub parallel: Option<bool>,
    /// Per-device GPU memory budget cap in MiB. `None` = use default.
    pub mem_budget_mib: Option<usize>,
    /// Soft GPU kernel timeout in seconds before the host aborts a launch.
    pub kernel_timeout: Duration,
}

impl Default for RuntimeEnv {
    fn default() -> Self {
        Self::from_env()
    }
}

impl RuntimeEnv {
    /// Read the current `GRIM_*` environment.
    pub fn from_env() -> Self {
        let host = std::env::var("GRIM_HOST").ok();
        let port = std::env::var("GRIM_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok());
        let context = std::env::var("GRIM_CONTEXT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let gpus: Vec<usize> = std::env::var("GRIM_GPUS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_default();
        let tp_size = std::env::var("GRIM_TP_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let parallel = std::env::var("GRIM_PARALLEL")
            .ok()
            .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
                "yes" | "true" | "1" => Some(true),
                "no" | "false" | "0" => Some(false),
                _ => None,
            });
        let mem_budget_mib = std::env::var("GRIM_MEM_BUDGET_MIB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let kernel_timeout = std::env::var("GRIM_KERNEL_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(300));

        Self {
            host,
            port,
            context,
            backend: Backend::default(),
            gpus,
            tp_size,
            parallel,
            mem_budget_mib,
            kernel_timeout,
        }
    }

    /// Resolve the server bind address. `cli_addr` wins; otherwise the
    /// `GRIM_HOST` + `GRIM_PORT` env vars; otherwise `127.0.0.1:11434`.
    pub fn resolve_bind(cli_addr: Option<&str>) -> String {
        if let Some(a) = cli_addr {
            if !a.is_empty() {
                return a.to_string();
            }
        }
        let env = RuntimeEnv::from_env();
        let host = env.host.as_deref().unwrap_or("127.0.0.1");
        let port = env.port.unwrap_or(11434);
        // Validate the resolved socket address parses; fall back to the
        // safe default if a user-supplied host/port is malformed.
        let addr = format!("{host}:{port}");
        match addr.parse::<SocketAddr>() {
            Ok(_) => addr,
            Err(_) => "127.0.0.1:11434".to_string(),
        }
    }

    /// True when the runtime explicitly requested a loopback / private bind.
    pub fn binds_private() -> bool {
        let addr = Self::resolve_bind(None);
        match addr.parse::<SocketAddr>() {
            Ok(sa) => match sa.ip() {
                IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_unspecified(),
                IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified(),
            },
            Err(_) => true,
        }
    }
}
