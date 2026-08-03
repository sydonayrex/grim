//! Grim CLI — main entry point for all subcommands.

use clap::{Parser, Subcommand};
use grim_core::error::Result;

pub mod accept;
pub mod bench;
pub mod catalog;
pub mod client;
pub mod compat;
pub mod cp;
pub mod doctor;
pub mod echo;
pub mod oxidizer;
pub mod plugin;
pub mod rm;
pub mod run;
pub mod server;
pub mod service;
pub mod show;
pub mod spec;
pub mod start;
pub mod stop;
pub mod train;
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
        /// Address to bind the server.
        #[arg(short, long, default_value = "127.0.0.1:11434")]
        address: String,
        /// Path to grim config file.
        #[arg(short, long, default_value = "grim.toml")]
        config: String,
        /// Path to plugins directory.
        #[arg(short, long, default_value = "plugins")]
        plugins: String,
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
        /// Repetition penalty (1.0 = disabled). Default 1.10 matches Ollama.
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,
    },
    /// Delete a model from local cache.
    Rm {
        /// Model name or path to delete.
        model: String,
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
        /// Provider name (e.g. 'hf.co', 'ollama').
        provider: String,
        /// API key or Token.
        #[arg(short, long)]
        token: Option<String>,
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
    /// Validate and install a model architecture plugin into system plugin directory.
    Accept {
        /// Path to the plugin file (e.g., ling-2.6.grimplugin).
        plugin_path: String,
    },
    /// Generate a model architecture compatibility plugin (.grimplugin) from a HuggingFace config.json.
    Compat {
        /// Path to config.json file.
        config_path: String,
        /// Optional output path for the generated .grimplugin file.
        #[arg(short, long)]
        output: Option<String>,
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
            config: _,
            plugins,
        } => {
            // Starts the HTTP server with first available model and tokenizer.
            let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
            if !plugins.is_empty() {
                let mut registry = grim_plugin::PluginRegistry::new();
                match plugin::load_plugins(&plugins, &mut registry) {
                    Ok(n) => eprintln!("[grim] serve: loaded {n} plugin(s) from {plugins}"),
                    Err(e) => eprintln!("[grim] serve: failed to load plugins from {plugins}: {e}"),
                }
            }
            eprintln!("[grim] serve: binding to {address} (Ollama-compatible)");
            grim_server::serve(&address, engine, None).await?;
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
            repeat_penalty,
        } => {
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
                eprintln!("[grim] serve: binding to {address} (Ollama-compatible)");
                grim_server::serve(&address, engine, model_path).await?;
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
        Commands::Rm { model } => {
            if let Err(e) = rm::cmd_rm(&model).await {
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
        Commands::Login { provider, token } => {
            let t = match token {
                Some(tk) => tk,
                None => {
                    print!("Enter API token for {}: ", provider);
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input).unwrap();
                    input.trim().to_string()
                }
            };
            client::save_login_token(&provider, &t)?;
        }
        Commands::Bench {
            tokens,
            concurrency,
            model,
        } => {
            bench::cmd_bench(tokens, concurrency, model.as_deref()).await?;
        }
        Commands::Quantize => {
            println!(
                "Quantization is available via 'grim oxidize'. Run 'grim oxidize --help' for conversion and quantization options."
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
            };
            if let Err(e) = train::cmd_train(opts) {
                eprintln!("[grim train] Failed: {e}");
                std::process::exit(1);
            }
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
                ServiceCommands::Run { config } => {
                    #[cfg(target_os = "windows")]
                    {
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
                            if let Err(e) = server::cmd_server("127.0.0.1:11434", &config, "").await
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

            if let Err(e) = oxidizer::cmd_oxidizer_convert(
                &input,
                &output,
                target_bpw,
                generations,
                Some(profile_str),
                dataset,
            ) {
                eprintln!("Conversion failed: {e}");
                std::process::exit(1);
            }
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
                } => match oxidizer::cmd_oxidizer_calibrate(&model, &output, dataset.as_deref()) {
                    Ok(_scores) => println!("[oxidizer] calibration complete"),
                    Err(e) => {
                        eprintln!("oxidizer calibrate failed: {e}");
                        std::process::exit(1);
                    }
                },
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
                    let bitwidths =
                        oxidizer::cmd_oxidizer_search(&imp_scores, &sizes, target_bpw, generations);
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
                } => {
                    if let Err(e) = oxidizer::cmd_oxidizer_convert(
                        &model,
                        &output,
                        target_bpw,
                        generations,
                        profile.as_deref(),
                        dataset,
                    ) {
                        eprintln!("oxidizer convert failed: {e}");
                        std::process::exit(1);
                    }
                }
                OxidizerCommands::Raven {
                    model,
                    output,
                    target_bpw,
                    dataset,
                } => {
                    if let Err(e) = oxidizer::cmd_oxidizer_raven(
                        &model,
                        &output,
                        target_bpw.unwrap_or(8.0),
                        dataset.as_deref(),
                    ) {
                        eprintln!("oxidizer raven failed: {e}");
                        std::process::exit(1);
                    }
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
        Commands::Show { verbose } => {
            if let Err(e) = show::cmd_show(verbose).await {
                eprintln!("Show failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Accept { plugin_path } => {
            if let Err(e) = accept::cmd_accept(&plugin_path).await {
                eprintln!("Accept failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Compat {
            config_path,
            output,
        } => {
            if let Err(e) = compat::cmd_compat(&config_path, output).await {
                eprintln!("Compat generation failed: {e}");
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
        let server = grim_server::serve("127.0.0.1:11434", engine, None);
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
