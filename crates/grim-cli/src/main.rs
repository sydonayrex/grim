//! Grim CLI — the main entry point for `grim run`, `grim bench`, etc.

use clap::{Parser, Subcommand};
use grim_core::error::Result;

mod run;
mod bench;
mod spec;
mod plugin;
mod service;
mod doctor;
mod oxidizer;
mod client;
mod catalog;
mod train;
mod verify;
mod cp;
mod rm;
mod stop;
mod server;
mod accept;
mod compat;
mod start;
mod show;

/// Grim inference engine CLI.
#[derive(Parser)]
#[command(name = "grim", version, about = "Rust inference engine — ROCm-first")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Client integrations supported by `grim start`.
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
    /// Start the inference HTTP server (Ollama-compatible, default port 11434).
    /// This is the subcommand used by the systemd/launchd service unit.
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
    /// One-shot inference or HTTP serving for a model.
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
        /// Preferred ROCm profile to use when a `.grim` conversion is
        /// available (cdna2, cdna3, rdna2, rdna3, rdna4, or "auto" to detect
        /// the host GPU). Only affects which sibling `grim run` loads once a
        /// conversion exists; it never forces a conversion on its own.
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
        /// Preferred ROCm profile to suggest for ROCm-tuned conversion after
        /// the pull (cdna2, cdna3, rdna2, rdna3, rdna4, or "auto"). When
        /// "auto" (or unset on a ROCm host), the local GPU's `gfx` target is
        /// detected and the matching profile is suggested. The suggestion is
        /// offered, never executed automatically.
        #[arg(long)]
        rocml_profile: Option<String>,
    },
    /// Start the inference HTTP server (alias for serve).
    Server {
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
        #[arg(short, long, default_value = "grim")]
        name: String,
        #[arg(short, long, default_value = "grim.toml")]
        config: String,
    },
    /// Uninstall platform-native background daemon.
    Uninstall {
        #[arg(short, long)]
        purge: bool,
    },
    /// Start service daemon.
    Start,
    /// Stop service daemon.
    Stop,
    /// Query current service status.
    Status,
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

/// WI-S6: after a successful `grim pull`, offer the ROCm-tuned conversion
/// instead of silently running it. This keeps the behaviour opt-in (the
/// user decides) while making the capability reachable by default.
///
/// - If `preferred` is `Some`, it is used as the profile suggestion verbatim
///   (e.g. `cdna3`, `rdna2`). `Some("auto")` falls through to detection.
/// - Otherwise, on a ROCm host we probe the local GPU's `gfx` target and map
///   it to a profile (`gfx1036`→`rdna2`, `gfx11xx`→`rdna3`, `gfx12xx`→`rdna4`,
///   `gfx90a`/`gfx908`→`cdna3`). If no ROCm GPU is present, no suggestion is
///   printed (there is no target to tune for).
fn offer_rocml_conversion(model_ref: &str, preferred: Option<&str>) {
    let profile = match preferred {
        Some("auto") | None => detect_host_rocml_profile(),
        Some(p) => {
            // Validate against known profile names; the convert command will
            // parse this again via GrimRocmlProfile::from_str.
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
        println!(
            "       grim oxidize convert {model_ref} --rocml-profile {profile}"
        );
        println!(
            "       Or run 'grim run {model_ref}' now to use the unconverted GGUF."
        );
    }
}

/// Detect the local host GPU's ROCm profile string, or `None` if no ROCm GPU
/// is present.
fn detect_host_rocml_profile() -> Option<String> {
    match grim_backend_rocm::device::probe::probe_host_gpu(0) {
        Ok(caps) => Some(gcn_to_rocml_profile_str(&caps.gcn)),
        Err(_) => None,
    }
}

/// Map a GCN `gfx` target string to a ROCm profile name (WI-S5 adds `rdna2`
/// for `gfx1036`/`gfx103x`). Mirrors the mapping in `Convert` so the suggested
/// profile matches what `grim convert --target auto` would pick.
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
        Commands::Serve { address, config: _, plugins } => {
            // `grim serve` — starts the HTTP server. Scans the models directory
            // for the first available model and loads its tokenizer automatically.
            let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
            eprintln!("[grim] serve: binding to {address} (Ollama-compatible)");
            grim_server::serve(&address, engine, None).await?;
            let _ = plugins;
        }
        Commands::Run { model, prompt, serve, address, config: _, plugins, rocml_profile, temperature, top_p, top_k, max_tokens, seed, repeat_penalty } => {
            // The --rocml-profile flag documents an explicit ROCm tuning
            // preference. `resolve_model_preferring_grim` already honours an
            // existing .grim conversion automatically; if the user passed a
            // profile we surface a hint so they know the preference is noted
            // (it never forces a conversion on its own — WI-S6).
            if let Some(ref profile) = rocml_profile {
                eprintln!("[grim] ROCm profile preference noted: {profile} (used automatically if a .grim conversion exists).");
            }
            if serve {
                let mut engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
                // Resolve model name → file path and load it into the engine.
                let model_path = if let Some(ref m) = model {
                    let p = catalog::resolve_model_preferring_grim(m)
                        .or_else(|| {
                            // Direct file path fallback.
                            let dp = std::path::Path::new(m);
                            if dp.exists() { Some(dp.to_path_buf()) } else { None }
                        });
                    if let Some(ref path) = p {
                        match grim_engine::model_loader::load_from_path(
                            &path.display().to_string()
                        ) {
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
                // Local GGUF file path: bypass the cache/download gate and
                // hand the path straight to `cmd_run`, which loads it via
                // `load_model_from_gguf`. This is the documented escape hatch
                // for running a model you already have on disk (e.g.
                // `grim run ./models/sleipnir.gguf`). The download security
                // boundary still applies to named/cached models below.
                let resolved = if model_name.to_lowercase().ends_with(".gguf")
                    && std::path::Path::new(&model_name).is_file()
                {
                    model_name.clone()
                } else {
                    // Resolve from catalog, preferring an existing ROCm-tuned
                    // `.grim` conversion over a sibling `.gguf` (WI-S6).
                    let model_path = catalog::resolve_model_preferring_grim(&model_name)
                        .ok_or_else(|| grim_core::error::Error::Config(
                            format!("Model '{}' not found. Run 'grim pull {}' to download it.",
                                model_name, model_name)
                        ))?;
                    model_path.to_string_lossy().into_owned()
                };
                if let Some(p) = prompt {
                    println!("[grim run] Running prompt on: {}", resolved);
                    run::cmd_run(resolved, Some(p), false, address, &plugins, temperature, top_p, top_k, max_tokens, seed, repeat_penalty).await?;
                } else {
                    println!("[grim run] Starting interactive session with: {}", model_name);
                    println!("Type your prompt below (Ctrl+C to exit):");
                    loop {
                        print!(">>> ");
                        use std::io::Write;
                        std::io::stdout().flush().unwrap();
                        let mut line = String::new();
                        std::io::stdin().read_line(&mut line).unwrap();
                        let trimmed = line.trim();
                        if trimmed.is_empty() { continue; }
                        let _ = run::cmd_run(resolved.clone(), Some(trimmed.to_string()), false, address.clone(), &plugins, temperature, top_p, top_k, max_tokens, seed, repeat_penalty).await;
                        println!();
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
        Commands::Dl { model, output, rocml_profile } | Commands::Pull { model, output, rocml_profile } => {
            client::download_model(&model, output).await?;
            // WI-S6: after a successful pull, offer (but never silently run)
            // the ROCm-tuned conversion. Detection respects an explicit
            // --rocml-profile; otherwise on a ROCm host we detect the local
            // GPU's gfx target; non-ROCm hosts get no suggestion.
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
        Commands::Bench { tokens, concurrency } => {
            bench::cmd_bench(tokens, concurrency).await?;
        }
        Commands::Quantize => {
            println!("Quantize command — not yet implemented (phase 2).");
        }
        Commands::Train { model, dataset, output, epochs, lr, rank, alpha } => {
            let opts = train::TrainOptions {
                model_path: model,
                dataset_path: dataset,
                output_sidecar: output,
                epochs,
                lr,
                rank,
                alpha,
            };
            if let Err(e) = train::cmd_train(opts) {
                eprintln!("[grim train] Failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Spec { subcommand } => match subcommand {
            SpecCommands::Train { target, output, dataset } => {
                spec::cmd_spec_train(target, output, dataset)?;
            }
        }
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
        }
        Commands::Service { subcommand } => {
            // Select appropriate platform manager
            let manager: Box<dyn service::ServiceManager> = if cfg!(target_os = "windows") {
                Box::new(service::WindowsScmManager)
            } else if cfg!(target_os = "macos") {
                Box::new(service::LaunchdManager)
            } else {
                Box::new(service::SystemdManager)
            };

            match subcommand {
                ServiceCommands::Install { name, config } => {
                    let cfg = service::ServiceConfig {
                        name,
                        exec_path: std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("grim")),
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
                    };
                    manager.install(&cfg)?;
                    println!("Service installation finished successfully.");
                }
                ServiceCommands::Uninstall { purge } => {
                    manager.uninstall(purge)?;
                    println!("Service uninstall finished successfully.");
                }
                ServiceCommands::Start => {
                    manager.start()?;
                }
                ServiceCommands::Stop => {
                    manager.stop()?;
                }
                ServiceCommands::Status => {
                    match manager.status()? {
                        service::ServiceStatus::Running => println!("grim service: running"),
                        service::ServiceStatus::Stopped => println!("grim service: stopped"),
                        service::ServiceStatus::Failed(msg) => println!("grim service: FAILED — {msg}"),
                        service::ServiceStatus::Unknown(s) => println!("grim service: unknown ({s})"),
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
                        let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
                        println!("[Service] Running background daemon on port 11434");
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        rt.block_on(async {
                            let _ = grim_server::serve("127.0.0.1:11434", engine, None).await;
                        });
                    }
                }
            }
        }
        Commands::Doctor { addr, service_name, exec_path, config_path } => {
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
        Commands::Convert { input, output, target, target_bpw, generations, dataset } => {
            // Detect input format from file extension and warn the user.
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
                    println!("[grim convert] Detected safetensors/PyTorch format — using SafetensorsProvider pipeline.");
                }
                "ggml" => {
                    println!("[grim convert] Detected GGML format — using GGUF/GGML compatibility reader.");
                }
                other => {
                    eprintln!("[grim convert] WARNING: Unknown extension '.{other}' — attempting GGUF reader.");
                }
            }

            let resolved_gcn = if target == "auto" {
                println!("[grim convert] Auto-detecting host GPU target architecture...");
                match grim_backend_rocm::device::probe::probe_system_rocm() {
                    Ok(rocm) => {
                        println!("[grim convert] ROCm installation detected: {} (version {})", rocm.path.display(), rocm.version);
                        match grim_backend_rocm::device::probe::probe_host_gpu(0) {
                            Ok(caps) => {
                                println!("[grim convert] Host GPU detected GCN architecture: {}", caps.gcn);
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
                OxidizerCommands::Calibrate { model, output, dataset } => {
                    match oxidizer::cmd_oxidizer_calibrate(&model, &output, dataset.as_deref()) {
                        Ok(_scores) => println!("[oxidizer] calibration complete"),
                        Err(e) => {
                            eprintln!("oxidizer calibrate failed: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                OxidizerCommands::Search { scores_path, tensor_sizes, target_bpw, generations } => {
                    let content = std::fs::read_to_string(&scores_path).unwrap_or_else(|e| {
                        eprintln!("failed to read {}: {e}", scores_path);
                        std::process::exit(1);
                    });
                    let v: serde_json::Value = serde_json::from_str(&content).unwrap_or_else(|e| {
                        eprintln!("failed to parse {}: {e}", scores_path);
                        std::process::exit(1);
                    });
                    let tensors = v["tensors"].as_array().expect("invalid scores format");
                    let names: Vec<String> = tensors.iter().map(|t| t["name"].as_str().unwrap().to_string()).collect();
                    let scores: Vec<f32> = tensors.iter().map(|t| t["importance_score"].as_f64().unwrap() as f32).collect();
                    let sizes: Vec<usize> = tensor_sizes
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    let imp_scores = grim_quant::ImportanceScores::new(names, scores);
                    let bitwidths = oxidizer::cmd_oxidizer_search(&imp_scores, &sizes, target_bpw, generations);
                    println!("EvoPress result (per-tensor bitwidths):");
                    for (i, bw) in bitwidths.iter().enumerate() {
                        let name = imp_scores.tensor_names.get(i).map(|s| s.as_str()).unwrap_or("?");
                        println!("  {name}: {bw}");
                    }
                }
                OxidizerCommands::Convert { model, output, target_bpw, generations, profile, dataset } => {
                    if let Err(e) = oxidizer::cmd_oxidizer_convert(
                        &model, &output, target_bpw, generations,
                        profile.as_deref(), dataset,
                    ) {
                        eprintln!("oxidizer convert failed: {e}");
                        std::process::exit(1);
                    }
                }
                OxidizerCommands::Prepare { input, output, train, format, profile, dataset } => {
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
                OxidizerCommands::Fuse { input, output, profile, rocm } => {
                    if let Err(e) = oxidizer::cmd_oxidizer_fuse(&input, &output, profile.as_deref(), rocm) {
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
        Commands::Server { address, config, plugins } => {
            server::cmd_server(&address, &config, &plugins).await?;
        }
        Commands::Start { client, model, args } => {
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
        Commands::Compat { config_path, output } => {
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
    use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
    use windows_service::service::{ServiceStatus, ServiceType, ServiceState, ServiceControlAccept};
    use std::sync::mpsc;
    use std::time::Duration;

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

    let status_handle = service_control_handler::register("grim", event_handler)
        .map_err(|e| grim_core::error::Error::Backend(format!("Failed to register SCM handler: {e}")))?;

    status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Running,
        controls_accepted: ServiceControlAccept::STOP,
        exit_code: 0,
        checkpoint: 0,
        wait_hint: Duration::from_secs(5),
        process_id: None,
    }).map_err(|e| grim_core::error::Error::Backend(format!("Failed to set SCM status: {e}")))?;

    // Spin up tokio runtime and HTTP server
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.spawn(async {
        let engine = grim_engine::Engine::new(grim_engine::EngineConfig::default());
        let _ = grim_server::serve("127.0.0.1:11434", engine, None).await;
    });

    let _ = shutdown_rx.recv();

    let _ = status_handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code: 0,
        checkpoint: 0,
        wait_hint: Duration::from_secs(1),
        process_id: None,
    });

    rt.shutdown_background();
    Ok(())
}

#[cfg(target_os = "windows")]
fn run_windows_service_dispatcher(_config: &str) -> Result<()> {
    use windows_service::service_dispatcher;
    service_dispatcher::start("grim", ffi_service_main)
        .map_err(|e| grim_core::error::Error::Backend(format!("Failed to start service dispatcher: {e}")))?;
    Ok(())
}

