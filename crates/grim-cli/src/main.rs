//! Grim CLI — the main entry point for `grim run`, `grim bench`, etc.

use clap::{Parser, Subcommand};
use grim_core::error::Result;

mod run;
mod bench;

/// Grim inference engine CLI.
#[derive(Parser)]
#[command(name = "grim", version, about = "Rust inference engine — ROCm-first")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a model (single-request or server mode).
    Run {
        /// Path to model file (GGUF or safetensors).
        #[arg(short, long)]
        model: String,
        /// Prompt string.
        prompt: Option<String>,
        /// Start HTTP server instead of one-shot inference.
        #[arg(short, long)]
        serve: bool,
        /// Address to bind the server (default 127.0.0.1:8080).
        #[arg(short, long, default_value = "127.0.0.1:8080")]
        address: String,
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { model, prompt, serve, address } => {
            run::cmd_run(model, prompt, serve, address).await?;
        }
        Commands::Bench { tokens, concurrency } => {
            bench::cmd_bench(tokens, concurrency).await?;
        }
        Commands::Quantize => {
            println!("Quantize command — not yet implemented (phase 2).");
        }
    }
    Ok(())
}