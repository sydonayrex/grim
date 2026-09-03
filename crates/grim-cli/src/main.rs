//! Grim CLI — main entry point for all subcommands.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_map_or,
    clippy::useless_format,
    clippy::redundant_closure,
    clippy::print_literal,
    clippy::field_reassign_with_default,
    clippy::unnecessary_sort_by,
    clippy::manual_repeat_n,
    clippy::if_same_then_else,
    clippy::manual_checked_ops,
    clippy::while_let_loop,
    clippy::new_without_default,
    clippy::let_unit_value,
    clippy::should_implement_trait,
    clippy::needless_range_loop,
    clippy::manual_ignore_case_cmp
)]

use clap::{Parser, Subcommand};
use grim_core::error::Result;

pub mod adapter;
pub mod arch_plugin;
pub mod bench;
pub mod catalog;
pub mod client;
pub mod config;
pub mod cp;
pub mod doctor;
pub mod echo;
pub mod eval;
pub mod multimodal;
pub mod oxidizer;
pub mod plugin;
pub mod progress;
pub mod provenance;
pub mod reap;
pub mod rm;
pub mod run;
pub mod scheduler;
pub mod server;
pub mod service;
pub mod show;
pub mod spec;
pub mod start;
pub mod stop;
pub mod template_registry;
pub mod train;
pub mod tui;
pub mod tune;
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

/// Kill TP peer children when rank 0's serve scope exits. A surviving peer
/// rank is worse than a dead one: it would hang on collectives forever.
struct TpChildGuard(Vec<std::process::Child>);

impl Drop for TpChildGuard {
    fn drop(&mut self) {
        for child in self.0.iter_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Liveness probe for a TP peer pid. Linux-only by way of /proc; other
/// platforms conservatively report alive (the peer's own HTTP port going
/// silent is then the operator's signal).
fn tp_pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/proc").join(pid.to_string()).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        true
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference HTTP server (Ollama-compatible, default port 11434). Used by systemd/launchd.
    Serve {
        /// Optional model name or path to preload upon server startup.
        #[arg(short, long)]
        model: Option<String>,
        /// Optional speculative draft model (EAGLE3 or small draft checkpoint).
        #[arg(long)]
        draft_model: Option<String>,
        /// Enable lookahead speculative decoding.
        #[arg(long)]
        lookahead: bool,
        /// Target compute device/backend (e.g. rocm, cuda, vulkan, metal, cpu).
        #[arg(short, long, alias = "device")]
        backend: Option<String>,
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
        /// Tensor-parallel size (Design A: one OS process per rank). This
        /// process is rank 0; ranks 1..N spawn as children with
        /// GRIM_TP_SIZE/GRIM_TP_RANK stamped and HTTP ports offset by rank.
        /// All ranks must receive the same request stream in the same order —
        /// rank 0's collectives rendezvous with its peers on every forward.
        #[arg(long, default_value = "1")]
        tp_size: usize,
    },
    /// One-shot inference or HTTP serving.
    Run {
        /// Name or path of the model.
        model: Option<String>,
        /// Optional speculative draft model (EAGLE3 or small draft checkpoint).
        #[arg(long)]
        draft_model: Option<String>,
        /// Enable lookahead speculative decoding.
        #[arg(long)]
        lookahead: bool,
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
        #[arg(long, default_value = "plugins")]
        plugins: String,
        /// Preferred ROCm profile (cdna2/cdna3/rdna2/rdna3/rdna4/auto). Never forces conversion on its own.
        #[arg(long)]
        rocml_profile: Option<String>,
        /// Sampling temperature (0 = greedy). Sets server default; overridable per request.
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Top-p (nucleus) sampling threshold. Sets server default; overridable per request.
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Top-k sampling limit (0 = disabled). Sets server default; overridable per request.
        #[arg(long, default_value = "40")]
        top_k: u32,
        /// Maximum tokens to generate.
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// RNG seed (0 = random).
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Target compute device / backend (e.g. cpu, cuda, rocm, vulkan, metal).
        #[arg(long, alias = "backend")]
        device: Option<String>,
        /// Repetition penalty (1.0 = disabled). Default 1.10 matches Ollama. Sets server default; overridable per request.
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,
    },
    /// Diagnostics TUI chat interface.
    Tui {
        /// Name or path of the model (optional; `/model` loads later).
        model: Option<String>,
        /// Sampling temperature (0 = greedy).
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Top-p (nucleus) sampling threshold.
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Top-k sampling limit (0 = disabled).
        #[arg(long, default_value = "40")]
        top_k: u32,
        /// Maximum tokens per turn.
        #[arg(long, default_value = "512")]
        max_tokens: usize,
        /// RNG seed (0 = random).
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Repetition penalty (1.0 = disabled). Default 1.10 matches Ollama.
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,
        /// Resume a saved session (JSONL transcript, as written by /save).
        #[arg(long, value_name = "PATH")]
        resume: Option<String>,
        /// Continue the most recently saved session (*.jsonl in the current directory).
        #[arg(short = 'c', long = "continue")]
        continue_last: bool,
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
        /// Bench mode: 'local' (default) or 'serve' (load-test a running
        /// server's /v1/chat/completions endpoint; §WI-E2).
        #[arg(long, default_value = "local")]
        mode: String,
        /// Server port for serve mode.
        #[arg(long, default_value_t = 11434)]
        port: u16,
        /// Duration in seconds for serve mode.
        #[arg(long, default_value_t = 60)]
        duration: u64,
    },
    /// Evaluate model on benchmark datasets (PPL, GSM8k).
    Eval {
        /// Model name or path to evaluate.
        #[arg(short, long)]
        model: Option<String>,
        /// Tasks to run (e.g. 'ppl', 'gsm8k', or 'ppl,gsm8k').
        #[arg(short, long, default_value = "ppl")]
        task: String,
        /// Path to write output evaluation metrics JSON.
        #[arg(short, long)]
        output: Option<String>,
        /// Port if evaluating against a running server (for gsm8k).
        #[arg(short, long, default_value_t = 11434)]
        port: u16,
    },
    /// Launch the Grim Garage telemetry and fine-tuning dashboard web service.
    Garage {
        /// Address to bind the garage service (default 127.0.0.1:8741).
        #[arg(short, long, default_value = "127.0.0.1:8741")]
        bind: String,
    },
    /// Run hardware-adaptive JIT kernel tuning and persist optimized tile configurations.
    Tune {
        /// GPU device ordinal to tune (default 0).
        #[arg(long, default_value_t = 0)]
        device: usize,
        /// Output directory for persisted .json and .hsaco caches.
        #[arg(short, long)]
        output_dir: Option<String>,
    },
    /// Train / fine-tune LoRA adapters on a dataset (SFT QLoRA).
    Train {
        /// Quick preset: sets low-rank LoRA defaults for rapid experimentation.
        #[arg(long)]
        quick: bool,
        /// Base model path or catalog name (empty allowed with --recipe).
        #[arg(short, long, default_value = "")]
        model: String,
        /// Dataset path (empty allowed with --recipe).
        #[arg(short, long, default_value = "")]
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
        /// WI-E5: MXFP4 quantization-aware training (STE fake-quant in forward).
        #[arg(long)]
        qat_mxfp4: bool,
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
        /// Number of compute nodes in multi-node training cluster.
        #[arg(long, default_value_t = 1)]
        num_nodes: usize,
        /// Rank of this node in multi-node training (0..num_nodes).
        #[arg(long, default_value_t = 0)]
        node_rank: usize,
        /// Master coordinator address for multi-node RCCL rendezvous.
        #[arg(long, default_value = "127.0.0.1")]
        master_addr: String,
        /// Master coordinator port for multi-node RCCL rendezvous.
        #[arg(long, default_value_t = 29500)]
        master_port: u16,
        /// Target compute device (e.g. "cpu", "rocm", "rocm:0").
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Training mode:
        ///  - qlora (default): 4-bit base weights + LoRA adapter (lowest VRAM)
        ///  - lora: 16-bit LoRA adapter on unquantized base
        ///  - full-bf16: full parameter fine-tuning in bfloat16
        ///  - full-fp16: full parameter fine-tuning in float16
        ///  - soul-eater: orthogonal weight matrix evolution
        ///  - oft: orthogonal fine-tuning preserving representation norms
        ///  - dpo: direct preference optimization with paired chosen/rejected targets
        ///  - orpo: odds ratio preference optimization
        ///  - simpo: simple reference-free preference optimization
        ///  - kto: kahneman-tversky optimization
        ///  - grpo: group relative policy optimization
        #[arg(long, default_value = "qlora")]
        mode: String,
        /// Enable SCALE-ECHO echo training mode. Bypasses the autograd tape
        /// and uses subspace echo state + FP4 updates instead.
        #[arg(long)]
        echo_mode: bool,
        /// RNG seed for deterministic adapter init (0 = random).
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Master-parameter compute precision: f32, bf16, or fp16.
        /// bf16/fp16 halve VRAM vs f32 on consumer RDNA (salamander.md P1).
        #[arg(long, value_enum, default_value = "f32")]
        train_dtype: train::TrainDtype,
        /// Optimizer (adamw, adamw-8bit, paged-adamw, paged-adamw-8bit, lion,
        /// lion-8bit, adafactor, qgalore-8bit, galore-8bit, muon, madam,
        /// lion-vote, lomo, adalomo, came, sophia, scythe, sickle).
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
        /// LoRA+: differential learning rate ratio for B matrix (default: 1.0).
        #[arg(long, default_value_t = 1.0)]
        lora_plus_ratio: f32,
        /// ReLoRA: merge adapters into base weights and reset momentum every N steps (0 = disabled).
        #[arg(long, default_value_t = 0)]
        relora_reset_steps: usize,
        /// Use OFT (Orthogonal Fine-Tuning) instead of standard LoRA.
        #[arg(long)]
        use_oft: bool,
        /// OFT rank. Lower = more parameter efficient.
        #[arg(long, default_value_t = 8)]
        oft_rank: usize,
        /// Held-out evaluation dataset path.
        #[arg(long)]
        eval_dataset: Option<String>,
        /// Run evaluation every N steps. 0 = disabled.
        #[arg(long, default_value_t = 0)]
        eval_every_steps: usize,
        /// Warmup steps before starting evaluation.
        #[arg(long, default_value_t = 0)]
        eval_warmup_steps: usize,
        /// Additional training dataset paths for multi-file weighted mixing.
        #[arg(long = "dataset-path")]
        dataset_paths: Vec<String>,
        /// Mixing weights per dataset path, comma-separated (e.g. "1.0,2.0").
        #[arg(long)]
        mix_weights: Option<String>,
        /// Deduplicate identical token sequences across mixed datasets.
        #[arg(long)]
        dedup: bool,
        /// Training recipe YAML (docs/recipes/*.yaml, WI-E6). Values apply to
        /// arguments left at their defaults; explicit flags win.
        #[arg(long)]
        recipe: Option<String>,
        /// Number of gradient checkpointing segments across layers (0 = disabled).
        #[arg(long, default_value_t = 0)]
        checkpoint_segs: usize,
    },
    /// Manage and inspect chat templates.
    Templates {
        #[command(subcommand)]
        cmd: TemplatesCmd,
    },
    /// Multimodal commands (Vision, Audio, Diffusion).
    Multimodal {
        #[command(subcommand)]
        cmd: multimodal::MultimodalCmd,
    },
    /// Query live continuous batching scheduler queues and KV cache memory tiers.
    Scheduler {
        /// Server address to query. Defaults to GRIM_HOST/GRIM_PORT env (or
        /// 127.0.0.1:11434) so a server started on a non-default port is found
        /// without re-typing the address (F-2).
        #[arg(short, long, default_value = "")]
        addr: String,
    },
    /// Runtime LoRA adapter management against a live server (load/list/unload
    /// without engine restart).
    Adapter {
        #[command(subcommand)]
        cmd: adapter::AdapterCmd,
    },
    /// Verify model integrity, checksums, and catalog provenance.
    Provenance {
        /// Path to model file to inspect.
        path: std::path::PathBuf,
    },
    /// Convert a model file to ROCm-optimized .grim format using Oxidizer.
    /// Supports GGUF (.gguf), GGML (.ggml), safetensors (.safetensors), and PyTorch (.bin).
    /// Tip: Use `grim oxidizer convert` for the full calibrate -> search -> write evolutionary pipeline.
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
    ///
    /// WI-2: `--model <path>` additionally runs a pre-flight model/hardware
    /// compatibility check *before* the user attempts a load — predicts
    /// fit (fits / tight / doesn't fit) and native/fallback/unsupported
    /// verdicts from the existing `resolve_quant_mode` arch gate.
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
        /// WI-2: model file to pre-flight-check (.gguf or .grim). Header-only
        /// parse — no tensor data is loaded.
        #[arg(long)]
        model: Option<std::path::PathBuf>,
    },
    /// ROCm-optimized GGUF conversion tool — calibrate, search, and convert.
    /// Tip: For one-shot GGUF -> .grim conversion, use `grim convert`.
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

#[derive(Subcommand, Debug)]
pub enum TemplatesCmd {
    /// List all known built-in chat template families.
    List,
    /// Inspect the Jinja template and description of a chat template family.
    Inspect {
        /// Name of the template family (e.g. "chatml", "llama3", "qwen", "mistral", "gemma").
        family: String,
    },
    /// Render a template family against a JSON file containing a messages array.
    Render {
        /// Name of the template family.
        family: String,
        /// Path to JSON file containing a list of {role, content} objects.
        #[arg(short, long)]
        input: String,
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
    /// Run RCO (Riemannian Constrained Optimization) bitwidth search on pre-computed importance scores.
    Search {
        /// Path to importance scores JSON (from `calibrate`).
        scores_path: String,
        /// Comma-separated list of tensor sizes.
        tensor_sizes: String,
        /// Target average bits-per-weight.
        #[arg(long, default_value = "4.0")]
        target_bpw: f32,
        /// Number of RCO optimization steps (legacy flag: generations).
        #[arg(long, default_value = "40", alias = "steps")]
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
        /// Number of RCO optimization steps (legacy flag: generations).
        #[arg(long, default_value = "40", alias = "steps")]
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
            // Validate against known profile names (convert re-parses via GrimRocmlProfile::parse).
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
            model,
            draft_model,
            lookahead,
            backend,
            address,
            host,
            port,
            config,
            plugins,
            disagg_role,
            prefill_addr,
            decode_addr,
            allow_public,
            tp_size,
        } => {
            if !config.is_empty() {
                unsafe {
                    std::env::set_var("GRIM_CONFIG_PATH", &config);
                }
            }
            if let Some(ref b) = backend {
                unsafe {
                    std::env::set_var("GRIM_BACKEND", b);
                }
            }
            // Tensor-parallel launcher (Design A, one OS process per rank):
            // this process is rank 0; peers 1..tp_size-1 run as children with
            // their rank and HTTP port stamped. Children never re-spawn —
            // launching is gated on GRIM_TP_RANK being unset. Peers die with
            // this process (ChildGuard) and this process dies with any peer
            // (fail-stop monitor): a TP rank set missing one rank deadlocks
            // the survivors' collectives on the next forward.
            let mut tp_children: Vec<std::process::Child> = Vec::new();
            let mut tp_peer_pids: Vec<u32> = Vec::new();
            if tp_size > 1 {
                if !address.is_empty() {
                    eprintln!(
                        "[grim] serve: --address cannot be combined with --tp-size; use \
                         --host/--port so per-rank ports can be derived"
                    );
                    std::process::exit(2);
                }
                let my_rank: usize = std::env::var("GRIM_TP_RANK")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                if my_rank == 0 {
                    unsafe {
                        std::env::set_var("GRIM_TP_SIZE", tp_size.to_string());
                        std::env::set_var("GRIM_TP_RANK", "0");
                    }
                    let base_port = port.unwrap_or_else(|| {
                        std::env::var("GRIM_PORT")
                            .ok()
                            .and_then(|p| p.parse().ok())
                            .unwrap_or(11434u16)
                    });
                    let exe = std::env::current_exe().unwrap_or_else(|e| {
                        eprintln!("[grim] serve: cannot resolve current exe for TP peers: {e}");
                        std::process::exit(2);
                    });
                    let mut pass_args: Vec<String> = std::env::args().skip(1).collect();
                    let port_flag = pass_args
                        .iter()
                        .position(|a| a == "--port" || a == "-p");
                    match port_flag {
                        Some(pos) => {
                            if let Some(next) = pass_args.get_mut(pos + 1) {
                                *next = base_port.to_string();
                            }
                        }
                        None => {
                            pass_args.push("--port".into());
                            pass_args.push(base_port.to_string());
                        }
                    }
                    for rank in 1..tp_size {
                        let mut args = pass_args.clone();
                        if let Some(pos) = args.iter().position(|a| a == "--port" || a == "-p") {
                            if let Some(next) = args.get_mut(pos + 1) {
                                *next = (base_port + rank as u16).to_string();
                            }
                        }
                        let result = std::process::Command::new(&exe)
                            .args(&args)
                            .env("GRIM_TP_SIZE", tp_size.to_string())
                            .env("GRIM_TP_RANK", rank.to_string())
                            .spawn();
                        match result {
                            Ok(child) => {
                                eprintln!(
                                    "[grim] TP rank {rank}/{tp_size} spawned (pid {}, port {})",
                                    child.id(),
                                    base_port as usize + rank
                                );
                                tp_peer_pids.push(child.id());
                                tp_children.push(child);
                            }
                            Err(e) => {
                                eprintln!("[grim] serve: failed to spawn TP rank {rank}: {e}");
                                for c in tp_children.iter_mut() {
                                    let _ = c.kill();
                                }
                                std::process::exit(2);
                            }
                        }
                    }
                    // Fail-stop monitor: any peer dying breaks rank 0's
                    // collectives on the next all-reduce, so take rank 0
                    // down instead of hanging the next forward.
                    let watched = tp_peer_pids.clone();
                    std::thread::spawn(move || loop {
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        for pid in &watched {
                            if !tp_pid_alive(*pid) {
                                eprintln!(
                                    "[grim] TP peer pid {pid} died; rank set is broken,                                      rank 0 exiting instead of hanging on collectives"
                                );
                                std::process::exit(3);
                            }
                        }
                    });
                }
            }
            // Children are killed whenever this arm's scope exits (server
            // shutdown, error return) — a surviving rank is worse than a
            // dead one.
            let _tp_child_guard = TpChildGuard(tp_children);
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

            let mut engine = grim_engine::Engine::new(engine_config);
            if let Some(ref m) = model {
                let p = catalog::resolve_model_preferring_grim(m).or_else(|| {
                    let dp = std::path::Path::new(m);
                    if dp.exists() {
                        Some(dp.to_path_buf())
                    } else {
                        None
                    }
                });
                if let Some(ref path) = p {
                    if let Err(e) = engine.load_and_register_scythe_farm_speculative(
                        m,
                        &path.display().to_string(),
                        draft_model.as_deref(),
                        lookahead,
                    ) {
                        eprintln!("[grim] WARNING: could not preload '{m}': {e}");
                    }
                }
            }
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
            let model_path = model.map(std::path::PathBuf::from);
            grim_server::serve(&effective, engine, model_path, plugin_registry).await?;
        }
        Commands::Garage { bind } => {
            eprintln!("[grim] starting Garage dashboard on http://{bind}");
            let state = grim_garage::routes::AppState {
                registry: std::sync::Arc::new(grim_garage::jobs::JobRegistry::new()),
                engine: std::sync::Arc::new(std::sync::Mutex::new(grim_engine::Engine::new(
                    grim_engine::EngineConfig::default(),
                ))),
                tokenizer: std::sync::Arc::new(std::sync::Mutex::new(None)),
                model_path: None,
            };
            let router = grim_garage::routes::build_router(state);
            let listener = tokio::net::TcpListener::bind(&bind)
                .await
                .map_err(|e| grim_core::Error::Config(format!("failed to bind garage: {e}")))?;
            let local = listener
                .local_addr()
                .map_err(|e| grim_core::Error::Config(format!("failed to get local addr: {e}")))?;
            if !local.ip().is_loopback() {
                eprintln!(
                    "[grim] WARNING: garage bound to public address {local}. Training endpoints are unauthenticated."
                );
            }
            eprintln!("[grim] Garage dashboard live at http://{local}/");
            axum::serve(listener, router)
                .await
                .map_err(|e| grim_core::Error::Config(format!("garage server error: {e}")))?;
        }
        Commands::Tui {
            model,
            temperature,
            top_p,
            top_k,
            max_tokens,
            seed,
            repeat_penalty,
            resume,
            continue_last,
        } => {
            tui::cmd_tui(
                model,
                temperature,
                top_p,
                top_k,
                max_tokens,
                seed,
                repeat_penalty,
                resume,
                continue_last,
            )
            .await?;
        }
        Commands::Run {
            model,
            draft_model,
            lookahead,
            prompt,
            serve,
            address,
            config,
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
            if !config.is_empty() {
                unsafe {
                    std::env::set_var("GRIM_CONFIG_PATH", &config);
                }
            }
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
                        if let Err(e) = engine.load_and_register_scythe_farm_speculative(
                            m,
                            &path.display().to_string(),
                            draft_model.as_deref(),
                            lookahead,
                        ) {
                            eprintln!("[grim] WARNING: could not load '{m}': {e}");
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
            // Auto-install architecture plugin if model reference is Hugging Face
            if model.starts_with("hf:")
                || (model.contains('/')
                    && !model.starts_with("http://")
                    && !model.starts_with("https://"))
            {
                let clean_ref = model.trim_start_matches("hf:");
                let parts: Vec<&str> = clean_ref.split('/').collect();
                if parts.len() >= 2 {
                    let org_repo = format!("{}/{}", parts[0], parts[1]);
                    if let Err(e) = arch_plugin::cmd_arch_plugin_generate(&org_repo, None).await {
                        eprintln!(
                            "[grim] note: could not auto-generate arch plugin for {org_repo}: {e}"
                        );
                    }
                }
            }
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
            mode,
            port,
            duration,
        } => {
            bench::cmd_bench(tokens, concurrency, model.as_deref(), &mode, port, duration).await?;
        }
        Commands::Eval {
            model,
            task,
            output,
            port,
        } => {
            eval::cmd_eval(model, task, output, port).await?;
        }
        Commands::Tune { device, output_dir } => {
            if let Err(e) = tune::cmd_tune(device, output_dir) {
                eprintln!("[grim tune] Error during hardware kernel tuning: {e}");
                std::process::exit(1);
            }
        }
        Commands::Multimodal { cmd } => {
            if let Err(e) = multimodal::cmd_multimodal(cmd) {
                eprintln!("[grim multimodal] Error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Scheduler { addr } => {
            // F-2: empty --addr resolves through GRIM_HOST/GRIM_PORT so a
            // server on a non-default port is found without re-typing it.
            let resolved = if addr.is_empty() {
                grim_core::RuntimeEnv::resolve_bind(None)
            } else {
                addr
            };
            scheduler::cmd_scheduler(&resolved).await?;
        }
        Commands::Adapter { cmd } => {
            adapter::cmd_adapter(cmd).await?;
        }
        Commands::Provenance { path } => {
            provenance::cmd_provenance(&path)?;
        }
        Commands::Train {
            quick,
            model,
            dataset,
            output,
            epochs,
            lr,
            rank,
            alpha,
            qat_mxfp4,
            batch_size,
            gradient_accumulation_steps,
            warmup_steps,
            logging_steps,
            max_grad_norm,
            early_stopping_patience,
            num_gpus,
            num_nodes,
            node_rank,
            master_addr,
            master_port,
            device,
            mode,
            optimizer,
            scheduler,
            use_pissa,
            use_olora,
            olora_lambda,
            echo_mode,
            seed,
            train_dtype,
            lora_plus_ratio,
            relora_reset_steps,
            use_oft,
            oft_rank,
            eval_dataset,
            eval_every_steps,
            eval_warmup_steps,
            dataset_paths,
            mix_weights,
            dedup,
            recipe,
            checkpoint_segs,
        } => {
            // Load grim.toml defaults if available
            let cfg_toml = grim_cli::config::GrimToml::from_path("grim.toml").unwrap_or_default();

            // WI-E6: load the training recipe and resolve its dataset registry
            // entry (sha256-verified when pinned). Recipe values apply only to
            // arguments still at their defaults; explicit flags win.
            let loaded_recipe = match recipe.as_deref() {
                Some(path) => match grim_cli::recipe::load_recipe(std::path::Path::new(path)) {
                    Ok(r) => {
                        println!(
                            "[grim train] Loaded recipe '{}' (v{})",
                            r.name, r.recipe_version
                        );
                        Some(r)
                    }
                    Err(e) => {
                        eprintln!("[grim train] Failed to load recipe {path}: {e}");
                        std::process::exit(1);
                    }
                },
                None => None,
            };
            if let Some(r) = &loaded_recipe {
                match grim_cli::recipe::resolve_dataset(&r.dataset.registry_id) {
                    Ok(p) => println!(
                        "[grim train] Recipe dataset '{}' resolved to {}",
                        r.dataset.registry_id,
                        p.display()
                    ),
                    Err(e) => {
                        eprintln!("[grim train] Dataset resolution failed: {e}");
                        std::process::exit(1);
                    }
                }
            }
            let rt = loaded_recipe.as_ref().map(|r| &r.training);
            let recipe_model = loaded_recipe.as_ref().map(|r| r.model.clone());
            let recipe_dataset = loaded_recipe
                .as_ref()
                .map(|r| grim_cli::recipe::resolve_dataset(&r.dataset.registry_id))
                .map(|p| p.map(|p| p.to_string_lossy().into_owned()));

            let effective_model = match (model.is_empty(), recipe_model) {
                (true, Some(m)) => m,
                _ => model,
            };
            let effective_dataset = match (dataset.is_empty(), recipe_dataset) {
                (true, Some(Ok(d))) => d,
                _ => dataset,
            };
            let effective_epochs = if epochs == 3 {
                rt.map(|t| t.epochs).unwrap_or(epochs)
            } else {
                epochs
            };
            let effective_lr = if (lr - 2e-4).abs() < f32::EPSILON {
                rt.map(|t| t.lr).unwrap_or(lr)
            } else {
                lr
            };
            let effective_rank = if rank == 16 {
                rt.map(|t| t.rank).unwrap_or(rank)
            } else {
                rank
            };
            let effective_alpha = if (alpha - 32.0).abs() < f32::EPSILON {
                rt.map(|t| t.alpha).unwrap_or(alpha)
            } else {
                alpha
            };
            let effective_batch_size = if batch_size == 2048 {
                rt.map(|t| t.batch_size).unwrap_or(batch_size)
            } else {
                batch_size
            };
            let effective_grad_accum = if gradient_accumulation_steps == 1 {
                rt.map(|t| t.gradient_accumulation_steps)
                    .unwrap_or(gradient_accumulation_steps)
            } else {
                gradient_accumulation_steps
            };
            let effective_warmup = if warmup_steps == 0 {
                rt.map(|t| t.warmup_steps).unwrap_or(warmup_steps)
            } else {
                warmup_steps
            };
            let effective_logging = if logging_steps == 0 {
                rt.map(|t| t.logging_steps).unwrap_or(logging_steps)
            } else {
                logging_steps
            };
            let effective_max_grad_norm = if (max_grad_norm - 1.0).abs() < f32::EPSILON {
                rt.map(|t| t.max_grad_norm).unwrap_or(max_grad_norm)
            } else {
                max_grad_norm
            };
            let effective_patience = if early_stopping_patience == 0 {
                rt.map(|t| t.early_stopping_patience)
                    .unwrap_or(early_stopping_patience)
            } else {
                early_stopping_patience
            };
            let effective_mode = if mode == "qlora" {
                rt.map(|t| t.mode.clone()).unwrap_or(mode)
            } else {
                mode
            };
            let effective_optimizer = match rt.map(|t| t.optimizer.as_str()) {
                Some(name) if optimizer == grim_autograd::OptimizerKind::AdamW => {
                    name.parse().unwrap_or(grim_autograd::OptimizerKind::AdamW)
                }
                _ => optimizer,
            };
            let effective_scheduler = match rt.map(|t| t.scheduler.as_str()) {
                Some(name) if scheduler == grim_autograd::LRScheduler::Cosine => {
                    name.parse().unwrap_or(grim_autograd::LRScheduler::Cosine)
                }
                _ => scheduler,
            };
            let effective_lora_plus = if (lora_plus_ratio - 1.0).abs() > 1e-5 {
                lora_plus_ratio
            } else {
                cfg_toml.train.lora_plus_ratio
            };
            let effective_relora_steps = if relora_reset_steps != 0 {
                relora_reset_steps
            } else {
                cfg_toml.train.relora_reset_steps
            };
            let effective_use_oft = use_oft || cfg_toml.train.use_oft;
            let effective_oft_rank = if oft_rank != 8 {
                oft_rank
            } else {
                cfg_toml.train.oft_rank
            };
            let effective_eval_ds = eval_dataset.or(cfg_toml.train.eval);
            let effective_eval_every = if eval_every_steps != 0 {
                eval_every_steps
            } else {
                cfg_toml.train.eval_every_steps
            };
            let effective_eval_warmup = if eval_warmup_steps != 0 {
                eval_warmup_steps
            } else {
                cfg_toml.train.eval_warmup_steps
            };
            let mut effective_paths = dataset_paths;
            if effective_paths.is_empty() && !cfg_toml.train.dataset.is_empty() {
                effective_paths = cfg_toml.train.dataset;
            }
            let effective_mix_weights = mix_weights
                .map(|s| {
                    s.split(',')
                        .filter_map(|w| w.trim().parse::<f32>().ok())
                        .collect()
                })
                .unwrap_or(cfg_toml.train.mix_weights);
            let effective_dedup = dedup || cfg_toml.train.dedup;

            // Apply --quick preset defaults if requested
            let (final_epochs, final_rank, final_alpha, final_device, final_mode) = if quick {
                println!(
                    "[grim train] Using --quick LoRA preset (1 epoch, rank 8, alpha 16.0, cpu device)"
                );
                (1, 8, 16.0, "cpu".to_string(), "lora".to_string())
            } else {
                (
                    effective_epochs,
                    effective_rank,
                    effective_alpha,
                    device,
                    effective_mode,
                )
            };

            let opts = train::TrainOptions {
                model_path: effective_model,
                dataset_path: effective_dataset,
                output_sidecar: output,
                epochs: final_epochs,
                lr: effective_lr,
                rank: final_rank,
                alpha: final_alpha,
                batch_size: effective_batch_size,
                gradient_accumulation_steps: effective_grad_accum,
                warmup_steps: effective_warmup,
                logging_steps: effective_logging,
                max_grad_norm: effective_max_grad_norm,
                early_stopping_patience: effective_patience,
                num_gpus,
                num_nodes,
                node_rank,
                master_addr,
                master_port,
                device: final_device,
                mode: final_mode,
                optimizer: effective_optimizer,
                scheduler: effective_scheduler,
                use_pissa,
                use_olora,
                olora_lambda,
                echo_mode,
                seed,
                train_dtype,
                use_spectral_qlora: false,
                qat_mxfp4,
                checkpoint_segs,
                lora_plus_ratio: effective_lora_plus,
                relora_reset_steps: effective_relora_steps,
                use_oft: effective_use_oft,
                oft_rank: effective_oft_rank,
                eval_dataset: effective_eval_ds,
                eval_every_steps: effective_eval_every,
                eval_warmup_steps: effective_eval_warmup,
                dataset_paths: effective_paths,
                mix_weights: effective_mix_weights,
                dedup: effective_dedup,
                quick,
            };
            if let Err(e) = train::cmd_train(opts) {
                eprintln!("[grim train] Failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Templates { cmd } => match cmd {
            TemplatesCmd::List => {
                println!("{:<12} {}", "FAMILY", "DESCRIPTION");
                println!("{:<12} {}", "------", "-----------");
                for f in grim_cli::template_registry::TemplateRegistry::default() {
                    println!("{:<12} {}", f.name, f.description);
                }
            }
            TemplatesCmd::Inspect { family } => {
                if let Some(f) = grim_cli::template_registry::TemplateRegistry::lookup(&family) {
                    println!(
                        "--- {} ---\n{}\n\nJinja Template:\n{}",
                        f.name, f.description, f.jinja
                    );
                } else {
                    eprintln!(
                        "Unknown template family: '{family}'. Run 'grim templates list' for available families."
                    );
                    std::process::exit(1);
                }
            }
            TemplatesCmd::Render { family, input } => {
                let text = match std::fs::read_to_string(&input) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("Failed to read input JSON file '{}': {}", input, e);
                        std::process::exit(1);
                    }
                };
                let msgs: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Failed to parse JSON in '{}': {}", input, e);
                        std::process::exit(1);
                    }
                };
                match grim_cli::template_registry::render_family(&family, msgs) {
                    Ok(out) => println!("{out}"),
                    Err(e) => {
                        eprintln!("Render error: {e}");
                        std::process::exit(1);
                    }
                }
            }
        },
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

            // F2b: full-parameter sidecars carry base-weight blobs
            // (`param_{layer}_0_{point}_a`, no `_b` partner). Overwrite the
            // matching GGUF tensors so the merged checkpoint IS the trained
            // model, not just an adapter delta.
            let base_blobs = state.base_weight_blobs();
            if !base_blobs.is_empty() {
                println!(
                    "[grim merge] Full-parameter checkpoint: {} base weight(s).",
                    base_blobs.len()
                );
            }
            for (layer, point_suffix, blob) in &base_blobs {
                let Some(fmt) = state.dtypes.get(&blob.name).copied() else {
                    continue;
                };
                let Some(vals) = grim_format::train::decode_f32s_from(&blob.data, fmt) else {
                    continue;
                };
                let tensor_name = match point_suffix.as_str() {
                    "logits" => "output.weight".to_string(),
                    "qproj" => format!("layers.{layer}.attn.wq.weight"),
                    "kproj" => format!("layers.{layer}.attn.wk.weight"),
                    "vproj" => format!("layers.{layer}.attn.wv.weight"),
                    "oproj" => format!("layers.{layer}.attn.wo.weight"),
                    "gateproj" => format!("layers.{layer}.ffn.w_gate.weight"),
                    "upproj" => format!("layers.{layer}.ffn.w_up.weight"),
                    "downproj" => format!("layers.{layer}.ffn.w_down.weight"),
                    other => {
                        eprintln!("[grim merge] Unknown point suffix '{other}', skipping");
                        continue;
                    }
                };
                match grim_format::bolt_on::overwrite_tensor_f32(
                    std::path::Path::new(&out_path),
                    &tensor_name,
                    &vals,
                ) {
                    Ok(()) => println!("[grim merge] Overwrote base tensor: {tensor_name}"),
                    Err(e) => eprintln!("[grim merge] Skipped {tensor_name}: {e}"),
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
            model,
        } => {
            let healthy = doctor::run_doctor(
                &addr,
                &service_name,
                &exec_path,
                &config_path,
                model.as_deref(),
            );
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
