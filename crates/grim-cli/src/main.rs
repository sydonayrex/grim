//! Grim CLI — main entry point for all subcommands.

use clap::{Parser, Subcommand};
use grim_core::error::Result;

pub mod arch_plugin;
pub mod bench;
pub mod catalog;
pub mod client;
pub mod cp;
pub mod doctor;
pub mod echo;
pub mod oxidizer;
pub mod plugin;
pub mod progress;
pub mod reap;
pub mod rm;
pub mod run;
pub mod server;
pub mod service;
pub mod show;
pub mod spec;
pub mod start;
pub mod stop;
pub mod train;
pub mod tui;
pub mod verify;

pub use service::ServiceManager;

/// Grim inference engine CLI.
#[derive(Parser)]
#[command(name = "grim", version, about = "Rust inference engine — ROCm-first")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Client integrations for `grim start`.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ClientIntegration {
    /// Hermes — local chat UI
    Hermes,
    /// OpenClaw — code generation
    Openclaw,
    /// Claude Code — Anthropic's coding agent
    Claw,
    /// Codex — OpenAI's coding agent
    Codex,
    /// Antigravity — workflow automation
    Antigravity,
    /// ZCode — zero-config coding
    Zcode,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference HTTP server (Ollama-compatible, default port 11434). Used by systemd/launchd.
    Serve {
        /// Address to bind the server (overrides --host/--port and GRIM_HOST/GRIM_PORT).
        #[arg(short, long, default_value = "")]
        address: String,
        /// HTTP bind host (overrides GRIM_HOST). Defaults to 127.0.0.1.
        #[arg(long)]
        host: Option<String>,
        /// HTTP bind port (overrides GRIM_PORT). Defaults to 11434.
        #[arg(short, long)]
        port: Option<u16>,
        /// Path to grim config file.
        #[arg(short, long, default_value = "grim.toml")]
        config: String,
        /// Path to plugins directory.
        #[arg(long, default_value = "plugins")]
        plugins: String,
        /// Disaggregation role: prefill, decode, or colocated (default: colocated).
        #[arg(long, default_value = "colocated")]
        disagg_role: String,
        /// Prefill node address for decode mode (where to fetch KV from).
        #[arg(long, default_value = "")]
        prefill_addr: String,
        /// Decode node address for prefill mode (where to push KV to).
        #[arg(long, default_value = "")]
        decode_addr: String,
        /// Allow binding to 0.0.0.0 / :: (all interfaces). Without this flag,
        /// binding to a wildcard address is refused for safety.
        #[arg(long)]
        allow_public: bool,
    },
    /// One-shot inference or HTTP serving.
    Run {
        /// Name or path of the model.
        model: Option<String>,
        /// Prompt string (runs one-shot mode instead of interactive chat).
        prompt: Option<String>,
        /// Start the HTTP server (Ollama-compatible) on the specified port.
        #[arg(long)]
        serve: bool,
        /// Address to bind (only used with --serve).
        #[arg(short, long, default_value = "127.0.0.1:11434")]
        address: String,
        /// Path to config file.
        #[arg(short, long, default_value = "grim.toml")]
        config: String,
        /// Path to plugins directory.
        #[arg(short, long, default_value = "plugins")]
        plugins: String,
        /// Preferred ROCm profile (cdna2/cdna3/rdna2/rdna3/rdna4/auto). Never forces conversion on its own.
        #[arg(long)]
        rocml_profile: Option<String>,
        /// Sampling temperature (0 = greedy).
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Top-p (nucleus) sampling threshold.
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Top-k sampling limit (0 = disabled).
        #[arg(long, default_value = "40")]
        top_k: u32,
        /// Maximum tokens to generate.
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// RNG seed (0 = random).
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Target compute device (e.g. cpu, cuda, rocm, vulkan, metal).
        #[arg(long)]
        device: Option<String>,
        /// Repetition penalty (1.0 = disabled). Default 1.10 matches Ollama.
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,
    },
    /// Delete a model from local cache.
    Rm {
        /// Model name or path to delete.
        model: String,
        /// Skip confirmation prompt.
        #[arg(short, long)]
        force: bool,
    },
    /// Stop a currently running model (unload from memory).
    Stop {
        /// Name of the model to stop.
        model: String,
    },
    /// Download a model from Hugging Face or Ollama.
    Dl {
        /// Registry model path or URL (e.g. hf.co/user/model or ollama.com/library/llama3).
        model: String,
        /// Optional destination path.
        #[arg(short, long)]
        output: Option<String>,
        /// Preferred ROCm profile to suggest for ROCm-tuned conversion after
        /// the pull (cdna2, cdna3, rdna2, rdna3, rdna4, or "auto"). See
        /// `Pull` for semantics; `dl` shares the same flag.
        #[arg(long)]
        rocml_profile: Option<String>,
    },
    /// Pull (download) a model from Hugging Face or Ollama. Alias for `dl`.
    Pull {
        /// Registry model path or URL (e.g. hf.co/user/model, ollama.com/library/llama3).
        model: String,
        /// Optional destination path.
        #[arg(short, long)]
        output: Option<String>,
        /// Preferred ROCm profile (cdna2/cdna3/rdna2/rdna3/rdna4/auto). Suggestion only, never auto-executed.
        #[arg(long)]
        rocml_profile: Option<String>,
    },
    /// Start a client integration (hermes, openclaw, claude-code, codex, antigravity, zcode).
    Start {
        /// Client to start.
        #[arg(value_enum)]
        client: ClientIntegration,
        /// Model to use (defaults to context default).
        model: Option<String>,
        /// Additional arguments passed to the client.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Launch an external app with a grim-tracked model baked in.
    Reap {
        /// Client to launch.
        #[arg(value_enum)]
        client: ClientIntegration,
        /// Grim-tracked model name (validated against local catalog; defaults to "default").
        #[arg(long)]
        model: Option<String>,
        /// Extra arguments passed through to the launched program after `--`.
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Copy a model to a new name in the local cache.
    Cp {
        /// Source model name or path.
        src: String,
        /// Destination model name.
        dst: String,
    },
    /// Show active loaded models (alias for status)
    Ps,
    /// List local cached models (alias for check)
    List,
    /// Show loaded models, memory usage, and execution backend.
    Status,
    /// Check the local model cache and report completed and partial downloads.
    Check,
    /// Show available models organized by format (GRIM, GGUF, others).
    Show {
        /// Verbose output with details.
        #[arg(short, long)]
        verbose: bool,
    },
    /// Set a model (local or cloud-routed) as the default model point for a client context.
    Use {
        /// Context to bind (e.g. 'default', 'claude-code', 'hermes').
        context: String,
        /// Target model name (e.g. 'llama3', 'ollama:cloud').
        model: String,
    },
    /// Log in to a registry or cloud provider.
    Login {
        /// Provider name (e.g. 'hf.co', 'ollama', 'openai').
        provider: Option<String>,
        /// API key or Token.
        #[arg(short, long)]
        token: Option<String>,
        /// List all saved provider credentials.
        #[arg(short, long)]
        list: bool,
    },
    /// Benchmark / smoke test.
    Bench {
        /// Number of tokens to generate per run.
        #[arg(long, default_value = "128")]
        tokens: usize,
        /// Number of concurrent requests.
        #[arg(long, default_value = "1")]
        concurrency: usize,
        /// Path to a model file (.gguf, .grim, .safetensors). If omitted,
        /// a small random Llama is used for smoke testing.
        #[arg(short, long)]
        model: Option<String>,
    },
    /// Quantize a model.
    Quantize,
    /// Train / fine-tune LoRA adapters on a dataset (SFT QLoRA).
    Train {
        /// Base model path or catalog name.
        #[arg(short, long)]
        model: String,
        /// Dataset path.
        #[arg(short, long)]
        dataset: String,
        /// Output .grim.train sidecar path.
        #[arg(short, long, default_value = "adapter.grim.train")]
        output: String,
        /// Number of training epochs.
        #[arg(long, default_value_t = 3)]
        epochs: usize,
        /// Learning rate.
        #[arg(long, default_value_t = 2e-4)]
        lr: f32,
        /// LoRA rank.
        #[arg(long, default_value_t = 16)]
        rank: usize,
        /// LoRA alpha.
        #[arg(long, default_value_t = 32.0)]
        alpha: f32,
        /// Maximum tokens per packed batch (micro-batch size in tokens).
        #[arg(long, default_value_t = 2048)]
        batch_size: usize,
        /// Number of micro-batches to accumulate gradients over before an optimizer step.
        #[arg(long, default_value_t = 1)]
        gradient_accumulation_steps: usize,
        /// Number of optimizer steps for linear LR warmup at the start of training.
        #[arg(long, default_value_t = 0)]
        warmup_steps: usize,
        /// Log loss every N optimizer steps. 0 disables step-level logging.
        #[arg(long, default_value_t = 0)]
        logging_steps: usize,
        /// Maximum gradient norm for global gradient clipping. 0 disables clipping.
        #[arg(long, default_value_t = 1.0)]
        max_grad_norm: f32,
        /// Stop training if loss does not improve for this many epochs. 0 disables early stopping.
        #[arg(long, default_value_t = 0)]
        early_stopping_patience: usize,
        /// Number of GPUs for data-parallel training. >1 enables RCCL all-reduce.
        #[arg(long, default_value_t = 1)]
        num_gpus: usize,
        /// Target compute device (e.g. "cpu", "rocm", "rocm:0").
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Training mode (e.g. "qlora", "soul-eater").
        #[arg(long, default_value = "qlora")]
        mode: String,
        /// Enable SCALE-ECHO echo training mode. Bypasses the autograd tape
        /// and uses subspace echo state + FP4 updates instead.
        #[arg(long)]
        echo_mode: bool,
        /// Optimizer (adamw, adamw-8bit, paged-adamw, paged-adamw-8bit, lion,
        /// lion-8bit, adafactor, adamw-bnb, qgalore, galore, galore-8bit,
        /// lomo, adalomo, came, sophia).
        #[arg(long, default_value = "adamw")]
        optimizer: grim_autograd::OptimizerKind,
        /// LR scheduler (cosine-warmup, linear, polynomial, constant,
        /// inverse-sqrt, yolo, one-cycle, reduce-on-plateau).
        #[arg(long, default_value = "cosine-warmup")]
        scheduler: grim_autograd::LRScheduler,
        /// Initialize adapters via PiSSA (SVD-based) rather than random LoRA.
        #[arg(long)]
        use_pissa: bool,
        /// Apply the OLoRA orthogonality penalty to the scalar loss.
        #[arg(long)]
        use_olora: bool,
        /// Weight of the OLoRA orthogonality penalty.
        #[arg(long, default_value_t = 1.0)]
        olora_lambda: f32,
    },
    /// Convert a model file to ROCm-optimized .grim format using Oxidizer.
    /// Supports GGUF (.gguf), GGML (.ggml), safetensors (.safetensors), and PyTorch (.bin).
    Convert {
        /// Path to input model file (.gguf, .ggml, .safetensors, or .bin).
        #[arg(short, long)]
        input: String,
        /// Path to output .grim model file.
        #[arg(short, long)]
        output: String,
        /// Target GPU GCN architecture (e.g. gfx1100, gfx1201), or "auto" to detect the host GPU.
        #[arg(short, long, default_value = "auto")]
        target: String,
        /// Target average bits-per-weight.
        #[arg(long, default_value = "4.0")]
        target_bpw: f32,
        /// Number of EvoPress generations.
        #[arg(long, default_value = "50")]
        generations: usize,
        /// Calibration dataset name.
        #[arg(long)]
        dataset: Option<String>,
        /// Wavefront size to build for: "auto" (derive from --target GCN,
        /// RDNA → Wave32), "w32", or "w64" (explicit CDNA opt-in).
        #[arg(long, default_value = "auto")]
        wave: String,
        /// Offload tensor dequantization to host ROCm GPU during conversion (GPU-first with CPU fallback).
        #[arg(long, default_value_t = true)]
        gpu: bool,
    },
    /// Bake a trained LoRA/QLoRA adapter sidecar permanently into a base .grim model file.
    Merge {
        /// Path to base .grim model file.
        #[arg(short, long)]
        model: String,
        /// Path to trained adapter file (.grim.train or sidecar).
        #[arg(short, long)]
        adapter: String,
        /// Optional path to save merged output file (overwrites base model if omitted).
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Speculative decoding commands.
    Spec {
        #[command(subcommand)]
        subcommand: SpecCommands,
    },
    /// Plugin management.
    Plugin {
        #[command(subcommand)]
        subcommand: PluginCommands,
    },
    /// Generate and install an architecture compatibility plugin (.grimplugin) from a
    /// HuggingFace model repo. Fetches config.json via the HF Hub API, validates the
    /// required fields, and installs the plugin into `grim_plugins_dir()` where
    /// `model_loader.rs` can discover it at model-load time.
    ///
    /// Example: `grim arch-plugin generate hf:Qwen/Qwen3.8-27B`
    ArchPlugin {
        #[command(subcommand)]
        subcommand: ArchPluginCommands,
    },
    /// Service management.
    Service {
        #[command(subcommand)]
        subcommand: ServiceCommands,
    },
    /// Re-verify every claim Grim makes about itself (§13.5).
    /// Checks: unit on disk, OS service visibility, HTTP health, GPU backend,
    /// WASM grant enforcement, and ExecStart consistency.
    Doctor {
        /// Address the server is expected to be reachable on.
        #[arg(long, default_value = "127.0.0.1:11434")]
        addr: String,
        /// Service name registered with the OS service manager.
        #[arg(long, default_value = "grim")]
        service_name: String,
        /// Absolute path to the grim binary (used for ExecStart check).
        #[arg(long, default_value = "/usr/local/bin/grim")]
        exec_path: String,
        /// Absolute path to grim.toml (used for ExecStart check).
        #[arg(long, default_value = "/etc/grim/grim.toml")]
        config_path: String,
    },
    /// ROCm-optimized GGUF conversion tool — calibrate, search, and convert.
    Oxidizer {
        #[command(subcommand)]
        subcommand: OxidizerCommands,
    },
    /// Verify a .grim file: structure, compression, payload readability,
    /// and QLoRA adapter presence in backup2 slots.
    Verify {
        /// Path to .grim file to verify.
        path: String,
        /// Verbose output (show per-tensor details).
        #[arg(short, long)]
        verbose: bool,
    },
}

#[derive(Subcommand)]
enum ServiceCommands {
    /// Install platform-native background daemon.
    Install {
        /// Service name registered with the OS service manager.
        #[arg(short, long, default_value = service::DEFAULT_SERVICE_NAME)]
        name: String,
        #[arg(short, long, default_value = "grim.toml")]
        config: String,
    },
    /// Uninstall platform-native background daemon.
    Uninstall {
        /// Service name registered with the OS service manager.
        #[arg(short, long, default_value = service::DEFAULT_SERVICE_NAME)]
        name: String,
        #[arg(short, long)]
        purge: bool,
    },
    /// Start service daemon.
    Start {
        /// Service name registered with the OS service manager.
        #[arg(short, long, default_value = service::DEFAULT_SERVICE_NAME)]
        name: String,
    },
    /// Stop service daemon.
    Stop {
        /// Service name registered with the OS service manager.
        #[arg(short, long, default_value = service::DEFAULT_SERVICE_NAME)]
        name: String,
    },
    /// Query current service status.
    Status {
        /// Service name registered with the OS service manager.
        #[arg(short, long, default_value = service::DEFAULT_SERVICE_NAME)]
        name: String,
    },
    /// Run the service process (invoked by Windows SCM/service manager).
    Run {
        #[arg(short, long, default_value = "grim.toml")]
        config: String,
        /// Plugin directory to load samplers/processors from at startup — the
        /// same `--plugins <dir>` surface the interactive `serve` and
        /// `run --serve` commands honor. Empty (default) means no plugins.
        #[arg(long, default_value = "")]
        plugins: String,
    },
}

#[derive(Subcommand)]
enum SpecCommands {
    /// Distill / train a draft model.
    Train {
        /// Path to target model.
        #[arg(short, long)]
        target: String,
        /// Path to output draft model.
        #[arg(short, long)]
        output: String,
        /// Training dataset path.
        #[arg(short, long)]
        dataset: String,
    },
}

#[derive(Subcommand)]
enum PluginCommands {
    /// List loaded plugins.
    List,
    /// Load plugins from a directory.
    Load {
        /// Path to plugins directory.
        #[arg(short, long, default_value = "plugins")]
        path: String,
    },
}

/// Subcommands for `grim arch-plugin`.
#[derive(Subcommand)]
enum ArchPluginCommands {
    /// Generate and install a .grimplugin from a HuggingFace model repo.
    ///
    /// `model_id` is an `hf:org/repo` reference (e.g. `hf:Qwen/Qwen3.8-27B`).
    /// The command fetches config.json via the HF Hub API, validates required fields,
    /// and installs the plugin into `grim_plugins_dir()`.
    Generate {
        /// HuggingFace model reference (hf:org/repo).
        model_id: String,
        /// Optional output path override. Defaults to `{model_type}.grimplugin`
        /// in `grim_plugins_dir()`.
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[derive(Subcommand)]
enum OxidizerCommands {
    /// Display grim metadata from a GGUF/.grim file.
    Info {
        /// Path to GGUF or .grim file.
        path: String,
    },
    /// Run importance-matrix calibration and cache results.
    Calibrate {
        /// Path to input GGUF model.
        model: String,
        /// Path for output (importance scores written alongside).
        output: String,
        /// Optional calibration dataset name.
        #[arg(long)]
        dataset: Option<String>,
    },
    /// Run EvoPress evolutionary search on pre-computed importance scores.
    Search {
        /// Path to importance scores JSON (from `calibrate`).
        scores_path: String,
        /// Comma-separated list of tensor sizes.
        tensor_sizes: String,
        /// Target average bits-per-weight.
        #[arg(long, default_value = "4.0")]
        target_bpw: f32,
        /// Number of EvoPress generations.
        #[arg(long, default_value = "50")]
        generations: usize,
    },
    /// Full convert pipeline: calibrate → search → write .grim.
    Convert {
        /// Path to input GGUF model.
        model: String,
        /// Path for output .grim file.
        output: String,
        /// Target average bits-per-weight.
        #[arg(long, default_value = "4.0")]
        target_bpw: f32,
        /// Number of EvoPress generations.
        #[arg(long, default_value = "50")]
        generations: usize,
        /// Target ROCm profile (cdna2, rdna3, mi300x).
        #[arg(long)]
        profile: Option<String>,
        /// Calibration dataset name.
        #[arg(long)]
        dataset: Option<String>,
        /// Wavefront size: "auto" (derive from profile; RDNA → Wave32),
        /// "w32", or "w64" (CDNA opt-in).
        #[arg(long, default_value = "auto")]
        wave: String,
        /// Offload tensor dequantization to host ROCm GPU during conversion (GPU-first with CPU fallback).
        #[arg(long, default_value_t = true)]
        gpu: bool,
    },
    /// Raven FP8/MXFP4 repack pipeline: rewrite model tensors into FP8 format.
    Raven {
        /// Path to input GGUF model.
        model: String,
        /// Path for output .grim file.
        output: String,
        /// Target bits-per-weight.
        #[arg(long, default_value = "8.0")]
        target_bpw: Option<f32>,
        /// Optional calibration dataset path.
        #[arg(long)]
        dataset: Option<String>,
    },
    /// Prepare a training-capable `.grim` artifact from a base checkpoint.
    Prepare {
        /// Path to input GGUF or `.grim` file.
        input: String,
        /// Path for output `.grim` file.
        output: String,
        /// Enable training metadata materialization.
        #[arg(long, default_value_t = true)]
        train: bool,
        /// Preferred training materialization format.
        #[arg(long, default_value = "bf16")]
        format: String,
        /// Target ROCm profile (cdna2, cdna3, rdna3, mi300x).
        #[arg(long)]
        profile: Option<String>,
        /// Calibration dataset name recorded in metadata.
        #[arg(long)]
        dataset: Option<String>,
    },
    /// Analyze a checkpoint and bake ROCm fusion hints into the output artifact.
    Fuse {
        /// Path to input GGUF or `.grim` file.
        input: String,
        /// Path for output `.grim` file.
        output: String,
        /// Target ROCm profile (cdna2, cdna3, rdna3, mi300x).
        #[arg(long)]
        profile: Option<String>,
        /// Mark the output as ROCm KV-layout optimized.
        #[arg(long, default_value_t = true)]
        rocm: bool,
    },
}

/// WI-S6: offer ROCm-tuned conversion after pull (opt-in, never auto-executed).
/// Maps GPU gfx target to profile; no suggestion on non-ROCm hosts.
fn offer_rocml_conversion(model_ref: &str, preferred: Option<&str>) {
    let profile = match preferred {
        Some("auto") | None => detect_host_rocml_profile(),
        Some(p) => {
            // Validate against known profile names (convert re-parses via GrimRocmlProfile::from_str).
            let valid = matches!(
                p.to_lowercase().as_str(),
                "cdna2" | "cdna3" | "rdna2" | "rdna3" | "rdna4" | "all"
            );
            if valid {
                Some(p.to_string())
            } else {
                eprintln!(
                    "[grim] WARNING: unknown --rocml-profile '{p}'; falling back to auto-detection."
                );
                detect_host_rocml_profile()
            }
        }
    };

    if let Some(profile) = profile {
        println!();
        println!(
            "[grim] Tip: convert '{model_ref}' to a ROCm-tuned .grim for better performance on this GPU:"
        );
        println!("       grim oxidize convert {model_ref} --rocml-profile {profile}");
        println!("       Or run 'grim run {model_ref}' now to use the unconverted GGUF.");
    }
}

/// Detect host GPU's ROCm profile, or `None` if no ROCm GPU present.
fn detect_host_rocml_profile() -> Option<String> {
    match grim_backend_rocm::probe_host_gpu(0) {
        Ok(caps) => Some(gcn_to_rocml_profile_str(&caps.gcn)),
        Err(_) => None,
    }
}

/// Map GCN `gfx` target to ROCm profile name. Mirrors `grim convert --target auto` mapping.
fn gcn_to_rocml_profile_str(gcn: &str) -> String {
    if gcn.starts_with("gfx103") {
        "rdna2".to_string()
    } else if gcn.starts_with("gfx12") {
        "rdna4".to_string()
    } else if gcn.starts_with("gfx11") {
        "rdna3".to_string()
    } else if gcn.starts_with("gfx90") {
        "cdna3".to_string()
    } else if gcn.starts_with("gfx9") {
        "cdna2".to_string()
    } else {
        "rdna3".to_string()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Serve {
            address,
            host,
            port,
            config: _,
            plugins,
            disagg_role,
            prefill_addr,
            decode_addr,
            allow_public,
        } => {
            // Build EngineConfig with optional disaggregation wiring.
            let mut engine_config = grim_engine::EngineConfig::default();
            let role_lower = disagg_role.to_ascii_lowercase();
            let pool_role = match role_lower.as_str() {
                "prefill" => grim_disagg::PoolRole::Prefill,
                "decode" => grim_disagg::PoolRole::Decode,
                "colocated" => grim_disagg::PoolRole::Colocated,
                other => {
                    eprintln!(
                        "[grim] serve: unknown --disagg-role '{other}' (expected prefill|decode|colocated), defaulting to colocated"
                    );
                    grim_disagg::PoolRole::Colocated
                }
            };

            if pool_role != grim_disagg::PoolRole::Colocated {
                let dc = grim_disagg::DisaggConfig {
                    role: pool_role,
                    prefill_addr: prefill_addr.clone(),
                    decode_addr: decode_addr.clone(),
                };
                // Build a router for cross-node KV transfers.  The engine
                // supplies its own shared KvBlockPool to transfer_kv_cache_real
                // directly; the router's `pool` field is left None for
                // standalone trait-method use (the engine always passes the
                // pool as a parameter).
                let router = std::sync::Arc::new(grim_disagg::DisaggRouter::new(
                    if prefill_addr.is_empty() {
                        &decode_addr
                    } else {
                        &prefill_addr
                    },
                    if decode_addr.is_empty() {
                        &prefill_addr
                    } else {
                        &decode_addr
                    },
                    pool_role,
                ));
                engine_config.disagg_router = Some(router);
                engine_config.disagg_config = Some(dc);
            }

            let engine = grim_engine::Engine::new(engine_config);
            // Load plugins into a registry that is *kept* and threaded into
            // `serve()` so request handlers can look up registered samplers by
            // name. Prior behavior loaded then dropped the registry before a
            // single request was served — the whole pipeline ran for nothing.
            let plugin_registry = if !plugins.is_empty() {
                let mut registry = grim_plugin::PluginRegistry::new();
                match plugin::load_plugins(&plugins, &mut registry) {
                    Ok(n) => eprintln!("[grim] serve: loaded {n} plugin(s) from {plugins}"),
                    Err(e) => eprintln!("[grim] serve: failed to load plugins from {plugins}: {e}"),
                }
                Some(std::sync::Arc::new(registry))
            } else {
                None
            };
            // Precedence: explicit --address > --host/--port > GRIM_HOST/GRIM_PORT > default.
            // If --port is given without --host, default host to 127.0.0.1 so the
            // port is not silently ignored (a missing --host was the most common
            // cause of "port ignored" reports).
            let effective = if !address.is_empty() {
                let host_part = address.split(':').next().unwrap_or("");
                if host_part == "0.0.0.0" || host_part == "::" {
                    eprintln!(
                        "[grim] WARNING: binding to {host_part} exposes the server on all network \
                         interfaces. This is a security risk on untrusted networks. Use \
                         --allow-public to suppress this warning."
                    );
                    if !allow_public {
                        eprintln!(
                            "[grim] ERROR: refusing to bind to {host_part} without --allow-public flag."
                        );
                        std::process::exit(1);
                    }
                }
                address.clone()
            } else if let (Some(h), Some(p)) = (&host, &port) {
                format!("{h}:{p}")
            } else if let (Some(h), None) = (&host, &port) {
                let env = grim_core::RuntimeEnv::from_env();
                format!("{h}:{}", env.port.unwrap_or(11434))
            } else if let (None, Some(p)) = (&host, &port) {
                format!("127.0.0.1:{p}")
            } else {
                grim_core::RuntimeEnv::resolve_bind(None)
            };
            eprintln!("[grim] serve: binding to {effective} (Ollama-compatible)");
            grim_server::serve(&effective, engine, None, plugin_registry).await?;
        }
        Commands::Run {
            model,
            prompt,
            serve,
            address,
            config: _,
            plugins,
            rocml_profile,
            temperature,
            top_p,
            top_k,
            max_tokens,
            seed,
            device,
            repeat_penalty,
        } => {
            if let Some(ref dev) = device {
                // SAFETY: env::set_var is UB if other threads concurrently read the
                // environment. At this point no background worker tasks exist yet
                // (engine hasn't been constructed), so no concurrent reader of
                // GRIM_BACKEND is possible within this process.
                unsafe {
                    std::env::set_var("GRIM_BACKEND", dev);
                }
            }
            // --rocml-profile is a hint only; existing .grim conversions are auto-preferred (WI-S6).
            if let Some(ref profile) = rocml_profile {
                eprintln!(
                    "[grim] ROCm profile preference noted: {profile} (used automatically if a .grim conversion exists)."
                );
            }
            if serve {
                let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
                // Resolve model name and load into engine.
                let model_path = if let Some(ref m) = model {
                    let p = catalog::resolve_model_preferring_grim(m).or_else(|| {
                        // Direct file path fallback.
                        let dp = std::path::Path::new(m);
                        if dp.exists() {
                            Some(dp.to_path_buf())
                        } else {
                            None
                        }
                    });
                    if let Some(ref path) = p {
                        match grim_engine::model_loader::load_from_path(&path.display().to_string())
                        {
                            Ok(loaded) => engine.register_model(m, loaded),
                            Err(e) => eprintln!("[grim] WARNING: could not load '{}': {e}", m),
                        }
                    } else {
                        eprintln!(
                            "[grim] WARNING: model '{}' not found in catalog. \
                             Starting server without a preloaded model. \
                             Run 'grim pull {}' to download it.",
                            m, m
                        );
                    }
                    p
                } else {
                    None
                };
                let r_addr = grim_core::RuntimeEnv::resolve_bind(Some(&address));
                // Symmetric to the `Serve` arm: honor `--plugins <dir>` here too.
                // Prior behavior ignored `plugins` entirely in the `run --serve`
                // path, so a plugin directory was silently dropped.
                let plugin_registry = if !plugins.is_empty() {
                    let mut registry = grim_plugin::PluginRegistry::new();
                    match plugin::load_plugins(&plugins, &mut registry) {
                        Ok(n) => eprintln!("[grim] serve: loaded {n} plugin(s) from {plugins}"),
                        Err(e) => {
                            eprintln!("[grim] serve: failed to load plugins from {plugins}: {e}")
                        }
                    }
                    Some(std::sync::Arc::new(registry))
                } else {
                    None
                };
                eprintln!("[grim] serve: binding to {r_addr} (Ollama-compatible)");
                grim_server::serve(&r_addr, engine, model_path, plugin_registry).await?;
            } else {
                let model_name = model.unwrap_or_else(|| "default".to_string());
                // Bypass cache for local GGUF paths; security boundary still applies to named models.
                let resolved = if model_name.to_lowercase().ends_with(".gguf")
                    && std::path::Path::new(&model_name).is_file()
                {
                    model_name.clone()
                } else {
                    // Resolve from catalog, prefer .grim over .gguf (WI-S6).
                    let model_path = catalog::resolve_model_preferring_grim(&model_name)
                        .ok_or_else(|| {
                            grim_core::error::Error::Config(format!(
                                "Model '{}' not found. Run 'grim pull {}' to download it.",
                                model_name, model_name
                            ))
                        })?;
                    model_path.to_string_lossy().into_owned()
                };
                if let Some(p) = prompt {
                    println!("[grim run] Running prompt on: {}", resolved);
                    run::cmd_run(
                        resolved,
                        Some(p),
                        false,
                        address,
                        &plugins,
                        temperature,
                        top_p,
                        top_k,
                        max_tokens,
                        seed,
                        repeat_penalty,
                    )
                    .await?;
                } else {
                    println!(
                        "[grim run] Starting interactive session with: {}",
                        model_name
                    );
                    println!("Type your prompt below (Ctrl+C to exit):");
                    // B.4: model loaded once, REPL loop runs per-turn.
                    if let Err(e) = run::cmd_run_interactive(
                        resolved.clone(),
                        address.clone(),
                        temperature,
                        top_p,
                        top_k,
                        max_tokens,
                        seed,
                        repeat_penalty,
                    )
                    .await
                    {
                        eprintln!("[grim run] Command failed: {e}");
                    }
                }
            }
        }
        Commands::Rm { model, force } => {
            if let Err(e) = rm::cmd_rm(&model, force).await {
                eprintln!("Remove failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Stop { model } => {
            if let Err(e) = stop::cmd_stop(&model, "127.0.0.1:11434").await {
                eprintln!("Stop failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Dl {
            model,
            output,
            rocml_profile,
        }
        | Commands::Pull {
            model,
            output,
            rocml_profile,
        } => {
            client::download_model(&model, output).await?;
            // WI-S6: offer ROCm-tuned conversion after pull. Detection respects --rocml-profile or auto-detect.
            offer_rocml_conversion(&model, rocml_profile.as_deref());
        }
        Commands::Status | Commands::Ps => {
            client::query_server_status("127.0.0.1:11434").await?;
        }
        Commands::Check | Commands::List => {
            client::check_model_cache()?;
        }
        Commands::Use { context, model } => {
            client::set_default_model(&context, &model)?;
        }
        Commands::Login {
            provider,
            token,
            list,
        } => {
            if list {
                let saved = client::list_login_tokens()?;
                if saved.is_empty() {
                    println!("[grim] No stored credentials found in ~/.grim/credentials.toml");
                } else {
                    println!("[grim] Stored Provider Credentials:");
                    for (prov, tok) in saved {
                        let masked = if tok.len() > 8 {
                            format!("{}...{}", &tok[..4], &tok[tok.len() - 4..])
                        } else {
                            "********".to_string()
                        };
                        println!("  - {:<15} {}", prov, masked);
                    }
                }
            } else if let Some(p) = provider {
                let t = match token {
                    Some(tk) => tk,
                    None => {
                        print!("Enter API token for {}: ", p);
                        use std::io::Write;
                        std::io::stdout().flush().unwrap();
                        let mut input = String::new();
                        std::io::stdin().read_line(&mut input).unwrap();
                        input.trim().to_string()
                    }
                };
                client::save_login_token(&p, &t)?;
            } else {
                println!(
                    "Please specify a provider (e.g. 'grim login hf.co') or run 'grim login --list'."
                );
            }
        }
        Commands::Bench {
            tokens,
            concurrency,
            model,
        } => {
            bench::cmd_bench(tokens, concurrency, model.as_deref()).await?;
        }
        Commands::Quantize => {
            // WI-5: the redirect previously named `grim oxidize`, which is not
            // a real subcommand — following it produced a clap "unrecognized
            // subcommand" error, so the stub sent users to a dead end. The
            // actual commands are `oxidizer` (calibrate/search/convert
            // pipeline) and `convert` (one-shot GGUF -> .grim).
            println!(
                "`grim quantize` is a stub. Quantization is available via:\n  \
                 grim convert -i <input.gguf> -o <output.grim> --target-bpw 4.0\n  \
                 grim oxidizer convert --help    # full calibrate -> search -> write pipeline\n\
                 Run `grim oxidizer --help` for all conversion and quantization options."
            );
        }
        Commands::Train {
            model,
            dataset,
            output,
            epochs,
            lr,
            rank,
            alpha,
            batch_size,
            gradient_accumulation_steps,
            warmup_steps,
            logging_steps,
            max_grad_norm,
            early_stopping_patience,
            num_gpus,
            device,
            mode,
            optimizer,
            scheduler,
            use_pissa,
            use_olora,
            olora_lambda,
            echo_mode,
        } => {
            let opts = train::TrainOptions {
                model_path: model,
                dataset_path: dataset,
                output_sidecar: output,
                epochs,
                lr,
                rank,
                alpha,
                batch_size,
                gradient_accumulation_steps,
                warmup_steps,
                logging_steps,
                max_grad_norm,
                early_stopping_patience,
                num_gpus,
                device,
                mode,
                optimizer,
                scheduler,
                use_pissa,
                use_olora,
                olora_lambda,
                echo_mode,
                use_spectral_qlora: false,
            };
            if let Err(e) = train::cmd_train(opts) {
                eprintln!("[grim train] Failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Merge {
            model,
            adapter,
            output,
        } => {
            let out_path = output.unwrap_or_else(|| model.clone());
            println!(
                "[grim] Merging adapter '{}' into model '{}'...",
                adapter, out_path
            );
            if model != out_path {
                std::fs::copy(&model, &out_path).map_err(|e| {
                    grim_tensor::error::Error::Backend(format!("failed to copy base model: {e}"))
                })?;
            }
            // Sidecar parsing & merge invocation
            let state = grim_format::train::TrainState::read(std::path::Path::new(&adapter))?
                .ok_or_else(|| {
                    grim_tensor::error::Error::Backend(format!(
                        "sidecar file '{}' not found",
                        adapter
                    ))
                })?;

            for tensor_name in state.lora_tensor_names() {
                if let Some((a_data, a_shape, b_data, b_shape)) =
                    state.lora_weights_for(&tensor_name)
                {
                    let shape_a = grim_tensor::shape::Shape::from_slice(a_shape);
                    let shape_b = grim_tensor::shape::Shape::from_slice(b_shape);
                    let a_tensor = grim_backend_cpu::cpu_tensor(a_data, shape_a);
                    let b_tensor = grim_backend_cpu::cpu_tensor(b_data, shape_b);
                    // Standard scaling: scale = alpha / rank (alpha=32.0, rank=b_shape[1])
                    let rank = if b_shape.len() > 1 { b_shape[1] } else { 1 };
                    let scale = 32.0 / (rank as f32);
                    grim_format::bolt_on::merge_bolt_on(
                        std::path::Path::new(&out_path),
                        &tensor_name,
                        &a_tensor,
                        &b_tensor,
                        scale,
                    )?;
                    println!("[grim merge] Merged tensor: {}", tensor_name);
                }
            }
            println!("[grim] Permanently merged adapter into '{}'.", out_path);
        }
        Commands::Spec { subcommand } => match subcommand {
            SpecCommands::Train {
                target,
                output,
                dataset,
            } => {
                spec::cmd_spec_train(target, output, dataset)?;
            }
        },
        Commands::Plugin { subcommand } => match subcommand {
            PluginCommands::List => {
                println!("Loaded plugins: (none loaded in this mode)");
            }
            PluginCommands::Load { path } => {
                let mut registry = grim_plugin::PluginRegistry::new();
                match plugin::load_plugins(&path, &mut registry) {
                    Ok(n) => println!("Loaded {n} plugins from {path}"),
                    Err(e) => eprintln!("Failed to load plugins: {e}"),
                }
            }
        },
        Commands::ArchPlugin { subcommand } => match subcommand {
            ArchPluginCommands::Generate { model_id, output } => {
                arch_plugin::cmd_arch_plugin_generate(&model_id, output).await?;
            }
        },
        Commands::Service { subcommand } => {
            // Build platform service manager with resolved service name (single source of truth).
            let build_manager = |name: String| -> Box<dyn service::ServiceManager> {
                if cfg!(target_os = "windows") {
                    Box::new(service::WindowsScmManager::new(name))
                } else if cfg!(target_os = "macos") {
                    Box::new(service::LaunchdManager::new(name))
                } else {
                    Box::new(service::SystemdManager::new(name))
                }
            };

            match subcommand {
                ServiceCommands::Install { name, config } => {
                    let manager = build_manager(name);
                    let cfg = service::ServiceConfig {
                        exec_path: std::env::current_exe()
                            .unwrap_or_else(|_| std::path::PathBuf::from("grim")),
                        config_path: std::path::PathBuf::from(config),
                        restart_policy: service::RestartPolicy::OnFailure,
                        run_as_user: Some("grim".to_string()),
                        health_check: service::HealthCheckConfig {
                            endpoint: "/healthz".to_string(),
                            interval_secs: 10,
                            timeout_secs: 3,
                            failure_threshold: 3,
                        },
                        log_path: None,
                        tls_subject_alt_names: Vec::new(),
                    };
                    manager.install(&cfg)?;
                    println!("Service installation finished successfully.");
                }
                ServiceCommands::Uninstall { name, purge } => {
                    build_manager(name).uninstall(purge)?;
                    println!("Service uninstall finished successfully.");
                }
                ServiceCommands::Start { name } => {
                    build_manager(name).start()?;
                }
                ServiceCommands::Stop { name } => {
                    build_manager(name).stop()?;
                }
                ServiceCommands::Status { name } => {
                    let manager = build_manager(name.clone());
                    match manager.status()? {
                        service::ServiceStatus::Running => println!("{name} service: running"),
                        service::ServiceStatus::Stopped => println!("{name} service: stopped"),
                        service::ServiceStatus::Failed(msg) => {
                            println!("{name} service: FAILED — {msg}")
                        }
                        service::ServiceStatus::Unknown(s) => {
                            println!("{name} service: unknown ({s})")
                        }
                    }
                }
                ServiceCommands::Run { config, plugins } => {
                    #[cfg(target_os = "windows")]
                    {
                        let _ = plugins;
                        run_windows_service_dispatcher(&config)?;
                    }
                    #[cfg(not(target_os = "windows"))]
                    {
                        let _ = config;
                        let _engine =
                            grim_engine::Engine::new(grim_engine::EngineConfig::default());
                        println!("[Service] Running background daemon on port 11434");
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        rt.block_on(async {
                            if let Err(e) =
                                server::cmd_server("127.0.0.1:11434", &config, &plugins).await
                            {
                                eprintln!("[Service] Server failed: {e}");
                            }
                        });
                    }
                }
            }
        }
        Commands::Doctor {
            addr,
            service_name,
            exec_path,
            config_path,
        } => {
            let healthy = doctor::run_doctor(&addr, &service_name, &exec_path, &config_path);
            match healthy {
                Ok(ok) => {
                    if !ok {
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Doctor check failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::Convert {
            input,
            output,
            target,
            target_bpw,
            generations,
            dataset,
            wave,
            gpu,
        } => {
            // Detect input format and warn the user.
            let ext = std::path::Path::new(&input)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            match ext.as_str() {
                "gguf" => {
                    println!("[grim convert] Detected GGUF format — using Oxidizer pipeline.");
                }
                "safetensors" | "bin" => {
                    println!(
                        "[grim convert] Detected safetensors/PyTorch format — using SafetensorsProvider pipeline."
                    );
                }
                "ggml" => {
                    println!(
                        "[grim convert] Detected GGML format — using GGUF/GGML compatibility reader."
                    );
                }
                other => {
                    eprintln!(
                        "[grim convert] WARNING: Unknown extension '.{other}' — attempting GGUF reader."
                    );
                }
            }

            let resolved_gcn = if target == "auto" {
                println!("[grim convert] Auto-detecting host GPU target architecture...");
                match grim_backend_rocm::probe_system_rocm() {
                    Ok(rocm) => {
                        println!(
                            "[grim convert] ROCm installation detected: {} (version {})",
                            rocm.path.display(),
                            rocm.version
                        );
                        match grim_backend_rocm::probe_host_gpu(0) {
                            Ok(caps) => {
                                println!(
                                    "[grim convert] Host GPU detected GCN architecture: {}",
                                    caps.gcn
                                );
                                caps.gcn
                            }
                            Err(e) => {
                                eprintln!("Error querying host GPU properties: {e}");
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("ROCm environment dynamic discovery failed: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                target
            };

            let profile_str = if resolved_gcn.starts_with("gfx103") {
                "rdna2"
            } else if resolved_gcn.starts_with("gfx12") {
                "rdna4"
            } else if resolved_gcn.starts_with("gfx11") {
                "rdna3"
            } else if resolved_gcn.starts_with("gfx90") {
                "cdna3"
            } else if resolved_gcn.starts_with("gfx9") {
                "cdna2"
            } else {
                "rdna3"
            };

            // "auto" → let the GCN/profile resolution pick the wave (RDNA → W32);
            // explicit w32/w64 → hard override.
            let wave_override = match wave.to_ascii_lowercase().as_str() {
                "w32" => Some(grim_format::WaveSize::W32),
                "w64" => Some(grim_format::WaveSize::W64),
                "auto" => None,
                other => {
                    eprintln!(
                        "[grim convert] WARNING: unknown wave '{other}' — expected auto, w32, or w64; using auto."
                    );
                    None
                }
            };

            let mut prog = progress::Progress::new();
            let mut cb = |stage: &str, done: usize, total: usize| {
                prog.render(stage, done, total);
            };
            if let Err(e) = oxidizer::cmd_oxidizer_convert(
                &input,
                &output,
                target_bpw,
                generations,
                Some(profile_str),
                dataset,
                wave_override,
                gpu,
                Some(&mut cb),
            ) {
                prog.finish();
                eprintln!("Conversion failed: {e}");
                std::process::exit(1);
            }
            prog.finish();
        }
        Commands::Oxidizer { subcommand } => {
            match subcommand {
                OxidizerCommands::Info { path } => {
                    if let Err(e) = oxidizer::cmd_oxidizer_info(&path) {
                        eprintln!("oxidizer info failed: {e}");
                        std::process::exit(1);
                    }
                }
                OxidizerCommands::Calibrate {
                    model,
                    output,
                    dataset,
                } => {
                    let mut prog = progress::Progress::new();
                    let mut cb = |stage: &str, done: usize, total: usize| {
                        prog.render(stage, done, total);
                    };
                    let mut progress: Option<&mut (dyn FnMut(&str, usize, usize) + Send + Sync)> =
                        Some(&mut cb);
                    match oxidizer::cmd_oxidizer_calibrate(
                        &model,
                        &output,
                        dataset.as_deref(),
                        &mut progress,
                    ) {
                        Ok(_scores) => {
                            prog.finish();
                            println!("[oxidizer] calibration complete")
                        }
                        Err(e) => {
                            prog.finish();
                            eprintln!("oxidizer calibrate failed: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                OxidizerCommands::Search {
                    scores_path,
                    tensor_sizes,
                    target_bpw,
                    generations,
                } => {
                    let content = std::fs::read_to_string(&scores_path).unwrap_or_else(|e| {
                        eprintln!("failed to read {}: {e}", scores_path);
                        std::process::exit(1);
                    });
                    let v: serde_json::Value = serde_json::from_str(&content).unwrap_or_else(|e| {
                        eprintln!("failed to parse {}: {e}", scores_path);
                        std::process::exit(1);
                    });
                    let tensors = match v["tensors"].as_array() {
                        Some(arr) => arr,
                        None => {
                            eprintln!(
                                "invalid scores format: missing 'tensors' array in {}",
                                scores_path
                            );
                            std::process::exit(1);
                        }
                    };
                    let names: Vec<String> = tensors.iter().map(|t| {
                        t["name"].as_str().unwrap_or_else(|| {
                            eprintln!("invalid scores format: missing 'name' string in tensor entry");
                            std::process::exit(1);
                        }).to_string()
                    }).collect();
                    let scores: Vec<f32> = tensors.iter().map(|t| {
                        t["importance_score"].as_f64().unwrap_or_else(|| {
                            eprintln!("invalid scores format: missing 'importance_score' number in tensor entry");
                            std::process::exit(1);
                        }) as f32
                    }).collect();
                    let sizes: Vec<usize> = tensor_sizes
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    let imp_scores = grim_quant::ImportanceScores::new(names, scores);
                    let mut prog = progress::Progress::new();
                    let mut cb = |done: usize, total: usize| {
                        prog.render("evopress", done, total);
                    };
                    let bitwidths = oxidizer::cmd_oxidizer_search(
                        &imp_scores,
                        &sizes,
                        target_bpw,
                        generations,
                        Some(&mut cb),
                    );
                    prog.finish();
                    println!("EvoPress result (per-tensor bitwidths):");
                    for (i, bw) in bitwidths.iter().enumerate() {
                        let name = imp_scores
                            .tensor_names
                            .get(i)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        println!("  {name}: {bw}");
                    }
                }
                OxidizerCommands::Convert {
                    model,
                    output,
                    target_bpw,
                    generations,
                    profile,
                    dataset,
                    wave,
                    gpu,
                } => {
                    let wave_override = match wave.to_ascii_lowercase().as_str() {
                        "w32" => Some(grim_format::WaveSize::W32),
                        "w64" => Some(grim_format::WaveSize::W64),
                        "auto" => None,
                        other => {
                            eprintln!(
                                "[oxidizer convert] WARNING: unknown wave '{other}' — expected auto, w32, or w64; using auto."
                            );
                            None
                        }
                    };
                    let mut prog = progress::Progress::new();
                    let mut cb = |stage: &str, done: usize, total: usize| {
                        prog.render(stage, done, total);
                    };
                    if let Err(e) = oxidizer::cmd_oxidizer_convert(
                        &model,
                        &output,
                        target_bpw,
                        generations,
                        profile.as_deref(),
                        dataset,
                        wave_override,
                        gpu,
                        Some(&mut cb),
                    ) {
                        prog.finish();
                        eprintln!("oxidizer convert failed: {e}");
                        std::process::exit(1);
                    }
                    prog.finish();
                }
                OxidizerCommands::Raven {
                    model,
                    output,
                    target_bpw,
                    dataset,
                } => {
                    let mut prog = progress::Progress::new();
                    let mut cb = |stage: &str, done: usize, total: usize| {
                        prog.render(stage, done, total);
                    };
                    if let Err(e) = oxidizer::cmd_oxidizer_raven(
                        &model,
                        &output,
                        target_bpw.unwrap_or(8.0),
                        dataset.as_deref(),
                        Some(&mut cb),
                    ) {
                        prog.finish();
                        eprintln!("oxidizer raven failed: {e}");
                        std::process::exit(1);
                    }
                    prog.finish();
                }
                OxidizerCommands::Prepare {
                    input,
                    output,
                    train,
                    format,
                    profile,
                    dataset,
                } => {
                    if let Err(e) = oxidizer::cmd_oxidizer_prepare(
                        &input,
                        &output,
                        train,
                        &format,
                        profile.as_deref(),
                        dataset,
                    ) {
                        eprintln!("oxidizer prepare failed: {e}");
                        std::process::exit(1);
                    }
                }
                OxidizerCommands::Fuse {
                    input,
                    output,
                    profile,
                    rocm,
                } => {
                    if let Err(e) =
                        oxidizer::cmd_oxidizer_fuse(&input, &output, profile.as_deref(), rocm)
                    {
                        eprintln!("oxidizer fuse failed: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
        Commands::Verify { path, verbose: _ } => {
            if let Err(e) = verify::cmd_verify(&path) {
                eprintln!("Verification failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Cp { src, dst } => {
            if let Err(e) = cp::cmd_cp(&src, &dst).await {
                eprintln!("Copy failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Start {
            client,
            model,
            args,
        } => {
            if let Err(e) = start::cmd_start(client, model.as_deref(), &args).await {
                eprintln!("Start failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Reap {
            client,
            model,
            args,
        } => {
            if let Err(e) = reap::cmd_reap(client, model.as_deref(), &args) {
                eprintln!("Reap failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Show { verbose } => {
            if let Err(e) = show::cmd_show(verbose).await {
                eprintln!("Show failed: {e}");
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "windows")]
windows_service::define_windows_service!(ffi_service_main, win_service_main);

#[cfg(target_os = "windows")]
fn win_service_main(_arguments: Vec<std::ffi::OsString>) {
    if let Err(e) = run_service_loop() {
        eprintln!("[Service] Windows service execution error: {e}");
    }
}

#[cfg(target_os = "windows")]
fn run_service_loop() -> Result<()> {
    use std::sync::mpsc;
    use std::time::Duration;
    use windows_service::service::{
        ServiceControlAccept, ServiceState, ServiceStatus, ServiceType,
    };
    use windows_service::service_control_handler::{self, ServiceControlHandlerResult};

    let (shutdown_tx, shutdown_rx) = mpsc::channel();

    let event_handler = move |control_event| -> ServiceControlHandlerResult {
        match control_event {
            windows_service::service::ServiceControl::Stop => {
                let _ = shutdown_tx.send(());
                ServiceControlHandlerResult::NoError
            }
            windows_service::service::ServiceControl::Interrogate => {
                ServiceControlHandlerResult::NoError
            }
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    };

    let status_handle = service_control_handler::register(service::SERVICE_NAME, event_handler)
        .map_err(|e| {
            grim_core::error::Error::Backend(format!("Failed to register SCM handler: {e}"))
        })?;

    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::Running,
            controls_accepted: ServiceControlAccept::STOP,
            exit_code: 0,
            checkpoint: 0,
            wait_hint: Duration::from_secs(5),
            process_id: None,
        })
        .map_err(|e| grim_core::error::Error::Backend(format!("Failed to set SCM status: {e}")))?;

    // Spin up tokio runtime and HTTP server, keeping runtime alive for service lifetime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let shutdown = shutdown_rx.recv();
    rt.block_on(async {
        let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let server = grim_server::serve("127.0.0.1:11434", engine, None, None);
        tokio::select! {
            _ = server => {}
            _ = shutdown => {
                let _ = status_handle.set_service_status(ServiceStatus {
                    service_type: ServiceType::OWN_PROCESS,
                    current_state: ServiceState::Stopped,
                    controls_accepted: ServiceControlAccept::empty(),
                    exit_code: 0,
                    checkpoint: 0,
                    wait_hint: Duration::from_secs(1),
                    process_id: None,
                });
            }
        }
    });
    Ok(())
}

#[cfg(target_os = "windows")]
fn run_windows_service_dispatcher(_config: &str) -> Result<()> {
    use windows_service::service_dispatcher;
    service_dispatcher::start(service::SERVICE_NAME, ffi_service_main).map_err(|e| {
        grim_core::error::Error::Backend(format!("Failed to start service dispatcher: {e}"))
    })?;
    Ok(())
}
