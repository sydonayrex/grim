//! Runtime environment configuration for grim.
//!
//! A single source of truth for configuration flags loaded from `grim.toml`
//! (first) and overridden by `GRIM_*` environment variables (second).

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Parsed backend selector. `Auto` re-probes every time it is consulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum Backend {
    #[default]
    Auto,
    Rocm,
    Cuda,
    Vulkan,
    Metal,
    Cpu,
}

static WARNED_UNKNOWN: AtomicBool = AtomicBool::new(false);

/// Runtime knobs surfaced as configuration / environment variables.
#[derive(Debug, Clone)]
pub struct RuntimeEnv {
    /// Override for the server bind address. Only consulted when the caller
    /// does not pass an explicit address.
    pub host: Option<String>,
    pub host_src: &'static str,

    /// Override for the server port. Only consulted when the caller does not
    /// pass an explicit address.
    pub port: Option<u16>,
    pub port_src: &'static str,

    /// Override for the model context window (KV cache length).
    pub context: Option<usize>,
    pub context_src: &'static str,

    /// Backend selection (rocm/cuda/vulkan/metal/cpu/auto).
    pub backend: Backend,
    pub backend_src: &'static str,

    /// Comma-separated ordinal list of GPUs to use (e.g. `0,1`). Empty means
    /// "all visible devices".
    pub gpus: Vec<usize>,
    pub gpus_src: &'static str,

    /// Tensor-parallel world size.
    pub tp_size: usize,
    pub tp_size_src: &'static str,

    /// Advisory multi-GPU parallelism hint.
    pub parallel: Option<bool>,
    pub parallel_src: &'static str,

    /// Per-device GPU memory budget cap in MiB.
    pub mem_budget_mib: Option<usize>,
    pub mem_budget_mib_src: &'static str,

    /// Soft GPU kernel timeout in seconds before the host aborts a launch.
    pub kernel_timeout: Duration,
    pub kernel_timeout_src: &'static str,
}

impl Default for RuntimeEnv {
    fn default() -> Self {
        Self::from_env()
    }
}

impl RuntimeEnv {
    /// Find potential path to `grim.toml`.
    pub fn locate_config_file() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("GRIM_CONFIG") {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }
        let local = Path::new("grim.toml");
        if local.exists() {
            return Some(local.to_path_buf());
        }
        if let Ok(home) = std::env::var("HOME") {
            let user_conf = PathBuf::from(home).join(".grim").join("grim.toml");
            if user_conf.exists() {
                return Some(user_conf);
            }
        }
        None
    }

    /// Read configuration: `grim.toml` (file-first) -> `GRIM_*` env (override second) -> defaults.
    pub fn from_env() -> Self {
        let mut host = None;
        let mut host_src = "default";
        let mut port = None;
        let mut port_src = "default";
        let mut context = None;
        let mut context_src = "default";
        let mut backend = Backend::Auto;
        let mut backend_src = "default";
        let mut gpus = Vec::new();
        let mut gpus_src = "default";
        let mut tp_size = 0usize;
        let mut tp_size_src = "default";
        let mut parallel = None;
        let mut parallel_src = "default";
        let mut mem_budget_mib = None;
        let mut mem_budget_mib_src = "default";
        let mut kernel_timeout = Duration::from_secs(300);
        let mut kernel_timeout_src = "default";

        // 1. Load from grim.toml if present
        if let Some(cfg_path) = Self::locate_config_file() {
            if let Ok(content) = std::fs::read_to_string(&cfg_path) {
                if let Ok(val) = content.parse::<toml::Value>() {
                    if let Some(table) = val.as_table() {
                        let known_keys: HashSet<&str> = [
                            "host",
                            "port",
                            "context",
                            "backend",
                            "gpus",
                            "tp_size",
                            "parallel",
                            "mem_budget_mib",
                            "kernel_timeout",
                        ]
                        .into_iter()
                        .collect();

                        for k in table.keys() {
                            if !known_keys.contains(k.as_str())
                                && !WARNED_UNKNOWN.swap(true, Ordering::Relaxed)
                            {
                                eprintln!(
                                    "[grim config] Warning: unknown key '{}' in {}",
                                    k,
                                    cfg_path.display()
                                );
                            }
                        }

                        if let Some(h) = table.get("host").and_then(|v| v.as_str()) {
                            host = Some(h.to_string());
                            host_src = "toml";
                        }
                        if let Some(p) = table.get("port").and_then(|v| v.as_integer()) {
                            port = Some(p as u16);
                            port_src = "toml";
                        }
                        if let Some(c) = table.get("context").and_then(|v| v.as_integer()) {
                            context = Some(c as usize);
                            context_src = "toml";
                        }
                        if let Some(b) = table.get("backend").and_then(|v| v.as_str()) {
                            backend = match b.to_ascii_lowercase().as_str() {
                                "rocm" => Backend::Rocm,
                                "cuda" => Backend::Cuda,
                                "vulkan" => Backend::Vulkan,
                                "metal" => Backend::Metal,
                                "cpu" => Backend::Cpu,
                                _ => Backend::Auto,
                            };
                            backend_src = "toml";
                        }
                        if let Some(arr) = table.get("gpus").and_then(|v| v.as_array()) {
                            gpus = arr
                                .iter()
                                .filter_map(|x| x.as_integer().map(|i| i as usize))
                                .collect();
                            gpus_src = "toml";
                        }
                        if let Some(tp) = table.get("tp_size").and_then(|v| v.as_integer()) {
                            tp_size = tp as usize;
                            tp_size_src = "toml";
                        }
                        if let Some(par) = table.get("parallel").and_then(|v| v.as_bool()) {
                            parallel = Some(par);
                            parallel_src = "toml";
                        }
                        if let Some(mb) = table.get("mem_budget_mib").and_then(|v| v.as_integer()) {
                            mem_budget_mib = Some(mb as usize);
                            mem_budget_mib_src = "toml";
                        }
                        if let Some(kt) = table.get("kernel_timeout").and_then(|v| v.as_integer()) {
                            kernel_timeout = Duration::from_secs(kt as u64);
                            kernel_timeout_src = "toml";
                        }
                    }
                }
            }
        }

        // 2. Env vars override toml values
        if let Ok(h) = std::env::var("GRIM_HOST") {
            host = Some(h);
            host_src = "env";
        }
        if let Some(p) = std::env::var("GRIM_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
        {
            port = Some(p);
            port_src = "env";
        }
        if let Some(c) = std::env::var("GRIM_CONTEXT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            context = Some(c);
            context_src = "env";
        }
        if let Ok(b) = std::env::var("GRIM_BACKEND") {
            backend = match b.to_ascii_lowercase().as_str() {
                "rocm" => Backend::Rocm,
                "cuda" => Backend::Cuda,
                "vulkan" => Backend::Vulkan,
                "metal" => Backend::Metal,
                "cpu" => Backend::Cpu,
                _ => Backend::Auto,
            };
            backend_src = "env";
        }
        if let Ok(s) = std::env::var("GRIM_GPUS") {
            gpus = s
                .split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect();
            gpus_src = "env";
        }
        if let Some(tp) = std::env::var("GRIM_TP_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            tp_size = tp;
            tp_size_src = "env";
        }
        if let Ok(p) = std::env::var("GRIM_PARALLEL") {
            match p.trim().to_ascii_lowercase().as_str() {
                "yes" | "true" | "1" => {
                    parallel = Some(true);
                    parallel_src = "env";
                }
                "no" | "false" | "0" => {
                    parallel = Some(false);
                    parallel_src = "env";
                }
                _ => {}
            }
        }
        if let Some(mb) = std::env::var("GRIM_MEM_BUDGET_MIB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            mem_budget_mib = Some(mb);
            mem_budget_mib_src = "env";
        }
        if let Some(kt) = std::env::var("GRIM_KERNEL_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            kernel_timeout = Duration::from_secs(kt);
            kernel_timeout_src = "env";
        }

        Self {
            host,
            host_src,
            port,
            port_src,
            context,
            context_src,
            backend,
            backend_src,
            gpus,
            gpus_src,
            tp_size,
            tp_size_src,
            parallel,
            parallel_src,
            mem_budget_mib,
            mem_budget_mib_src,
            kernel_timeout,
            kernel_timeout_src,
        }
    }

    /// Return summary of effective configuration values with their sources.
    pub fn effective_config_summary(&self) -> Vec<(String, String, &'static str)> {
        vec![
            (
                "host".into(),
                self.host.clone().unwrap_or_else(|| "127.0.0.1".into()),
                self.host_src,
            ),
            (
                "port".into(),
                self.port
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "11434".into()),
                self.port_src,
            ),
            (
                "context".into(),
                self.context
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "auto".into()),
                self.context_src,
            ),
            (
                "backend".into(),
                format!("{:?}", self.backend).to_lowercase(),
                self.backend_src,
            ),
            ("gpus".into(), format!("{:?}", self.gpus), self.gpus_src),
            ("tp_size".into(), self.tp_size.to_string(), self.tp_size_src),
            (
                "parallel".into(),
                self.parallel
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "none".into()),
                self.parallel_src,
            ),
            (
                "mem_budget_mib".into(),
                self.mem_budget_mib
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "unlimited".into()),
                self.mem_budget_mib_src,
            ),
            (
                "kernel_timeout".into(),
                format!("{}s", self.kernel_timeout.as_secs()),
                self.kernel_timeout_src,
            ),
        ]
    }

    /// Resolve the server bind address. `cli_addr` wins; otherwise the
    /// `GRIM_HOST` + `GRIM_PORT` env vars / config; otherwise `127.0.0.1:11434`.
    pub fn resolve_bind(cli_addr: Option<&str>) -> String {
        if let Some(a) = cli_addr {
            if !a.is_empty() {
                return a.to_string();
            }
        }
        let env = RuntimeEnv::from_env();
        let host = env.host.as_deref().unwrap_or("127.0.0.1");
        let port = env.port.unwrap_or(11434);
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

#[cfg(test)]
mod runtime_env_tests {
    use super::*;

    /// Env vars are process-global: serialize every test that mutates them.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    const GRIM_VARS: &[&str] = &[
        "GRIM_CONFIG",
        "GRIM_HOST",
        "GRIM_PORT",
        "GRIM_CONTEXT",
        "GRIM_BACKEND",
        "GRIM_GPUS",
        "GRIM_TP_SIZE",
        "GRIM_PARALLEL",
        "GRIM_MEM_BUDGET_MIB",
        "GRIM_KERNEL_TIMEOUT",
    ];

    /// SAFETY: single-threaded with respect to the other env-mutating tests
    /// via ENV_LOCK; no other thread reads these GRIM_* vars concurrently.
    fn clear_grim_vars() {
        unsafe {
            for v in GRIM_VARS {
                std::env::remove_var(v);
            }
        }
    }

    /// SAFETY: same ENV_LOCK discipline as `clear_grim_vars`.
    fn set_var(key: &str, value: &str) {
        unsafe {
            std::env::set_var(key, value);
        }
    }

    #[test]
    fn defaults_when_no_env_or_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_grim_vars();
        // CWD is the crate dir during tests; ensure no grim.toml leak.
        let saved_cwd = std::env::current_dir().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_current_dir(&tmp).unwrap();
        let env = RuntimeEnv::from_env();
        std::env::set_current_dir(saved_cwd).unwrap();

        assert_eq!(env.host_src, "default");
        assert!(env.host.is_none());
        assert!(env.port.is_none());
        assert!(env.context.is_none());
        assert!(matches!(env.backend, Backend::Auto));
        assert_eq!(env.backend_src, "default");
        assert!(env.gpus.is_empty());
        assert_eq!(env.tp_size, 0);
        assert_eq!(env.parallel, None);
        assert!(env.mem_budget_mib.is_none());
        assert_eq!(env.kernel_timeout, Duration::from_secs(300));
        assert_eq!(env.kernel_timeout_src, "default");
    }

    #[test]
    fn env_vars_override_defaults() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_grim_vars();
        set_var("GRIM_HOST", "0.0.0.0");
        set_var("GRIM_PORT", "8123");
        set_var("GRIM_CONTEXT", "131072");
        set_var("GRIM_BACKEND", "ROCM"); // case-insensitive
        set_var("GRIM_GPUS", "0, 2, 5");
        set_var("GRIM_TP_SIZE", "2");
        set_var("GRIM_PARALLEL", "yes");
        set_var("GRIM_MEM_BUDGET_MIB", "65536");
        set_var("GRIM_KERNEL_TIMEOUT", "45");

        let env = RuntimeEnv::from_env();
        assert_eq!(env.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(env.host_src, "env");
        assert_eq!(env.port, Some(8123));
        assert_eq!(env.context, Some(131072));
        assert!(matches!(env.backend, Backend::Rocm));
        assert_eq!(env.gpus, vec![0, 2, 5]);
        assert_eq!(env.tp_size, 2);
        assert_eq!(env.parallel, Some(true));
        assert_eq!(env.mem_budget_mib, Some(65536));
        assert_eq!(env.kernel_timeout, Duration::from_secs(45));

        clear_grim_vars();
    }

    #[test]
    fn malformed_env_values_fall_back_to_defaults() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_grim_vars();
        set_var("GRIM_PORT", "not-a-port");
        set_var("GRIM_CONTEXT", "-5");
        set_var("GRIM_KERNEL_TIMEOUT", "soon");
        set_var("GRIM_PARALLEL", "banana");

        let env = RuntimeEnv::from_env();
        assert_eq!(env.port_src, "default");
        assert!(env.port.is_none());
        assert!(env.context.is_none());
        assert_eq!(env.kernel_timeout, Duration::from_secs(300));
        assert_eq!(env.parallel, None);
        clear_grim_vars();
    }

    #[test]
    fn toml_config_is_parsed_and_env_wins() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_grim_vars();
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join("grim.toml");
        std::fs::write(
            &cfg,
            "host = \"127.0.0.1\"\nport = 9000\nbackend = \"cpu\"\ngpus = [1, 3]\n",
        )
        .unwrap();
        set_var("GRIM_CONFIG", cfg.to_str().unwrap());

        // toml alone.
        let env = RuntimeEnv::from_env();
        assert_eq!(env.host.as_deref(), Some("127.0.0.1"));
        assert_eq!(env.host_src, "toml");
        assert_eq!(env.port, Some(9000));
        assert!(matches!(env.backend, Backend::Cpu));
        assert_eq!(env.gpus, vec![1, 3]);

        // Env overrides toml for the same key.
        set_var("GRIM_PORT", "7777");
        let env = RuntimeEnv::from_env();
        assert_eq!(env.port, Some(7777));
        assert_eq!(env.port_src, "env");
        // toml value for host is untouched.
        assert_eq!(env.host_src, "toml");

        // locate_config_file honors GRIM_CONFIG when the path exists.
        assert_eq!(
            RuntimeEnv::locate_config_file().as_deref(),
            Some(cfg.as_path())
        );

        clear_grim_vars();
    }

    #[test]
    fn effective_config_summary_lists_keys_and_sources() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_grim_vars();
        set_var("GRIM_BACKEND", "cuda");
        let env = RuntimeEnv::from_env();
        let summary = env.effective_config_summary();
        assert!(!summary.is_empty());
        // Every row is (key, value, source).
        for (k, v, src) in &summary {
            assert!(!k.is_empty());
            assert!(!v.is_empty());
            assert!(!src.is_empty());
        }
        let backend_row = summary.iter().find(|(k, _, _)| k == "backend");
        assert!(backend_row.is_some(), "backend must appear in the summary");
        let (_, v, src) = backend_row.unwrap();
        assert_eq!(*src, "env");
        assert!(v.eq_ignore_ascii_case("cuda"));
        clear_grim_vars();
    }
}
