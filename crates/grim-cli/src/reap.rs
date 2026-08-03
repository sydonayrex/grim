//! grim reap — launch an external app with a grim-tracked model baked in.
//!
//! Mirrors ollama's `ollama launch` flow: resolve a tracked model, verify the
//! binary exists, inject env vars so the child routes through the local grim
//! serve, and exec with passthrough args after `--`.

use crate::ClientIntegration;
use crate::catalog::resolve_model_preferring_grim;
use grim_core::error::{Error, Result};
use std::process::Command;

/// Launch an external integration with a grim-tracked model.
pub fn cmd_reap(
    client: ClientIntegration,
    model: Option<&str>,
    extra_args: &[String],
) -> Result<()> {
    // Resolve model name through the local catalog. Falls back to "default"
    // when no --model is provided.
    let model_name = model
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "default".to_string());

    let resolved_path = resolve_model_preferring_grim(&model_name).or_else(|| {
        // If the user passed a bare file path, accept it.
        let p = std::path::Path::new(&model_name);
        if p.exists() {
            Some(p.to_path_buf())
        } else {
            None
        }
    });

    // We pass the *name* to the child, but warn if grim doesn't track it.
    // The child itself will resolve via the ollama-compatible API.
    let effective_model = if resolved_path.is_some() {
        &model_name
    } else if model.is_some() {
        eprintln!(
            "[grim reap] WARNING: model '{}' is not in the local catalog; \
             the launched app may fail to resolve it. Run 'grim pull {}' first.",
            model_name, model_name
        );
        &model_name
    } else {
        "default"
    };

    let (program, mut cmd_args) = match client {
        ClientIntegration::Hermes => ("hermes", vec!["--model", effective_model]),
        ClientIntegration::Openclaw => ("openclaw", vec!["--model", effective_model]),
        ClientIntegration::Claw => ("claude-code", vec!["--model", effective_model]),
        ClientIntegration::Codex => ("codex", vec!["--model", effective_model]),
        ClientIntegration::Antigravity => ("antigravity", vec!["--model", effective_model]),
        ClientIntegration::Zcode => ("zcode", vec!["--model", effective_model]),
    };

    cmd_args.extend(extra_args.iter().map(|s| s.as_str()));

    eprintln!(
        "[grim reap] Launching {} with model '{}'",
        client_name(client),
        effective_model
    );

    // Resolve the grim serve endpoint so child processes can route through it.
    let grim_host = std::env::var("GRIM_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let grim_port = std::env::var("GRIM_PORT").unwrap_or_else(|_| "11434".to_string());
    let ollama_host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| {
        if grim_port == "11434" {
            format!("{grim_host}:{grim_port}")
        } else {
            format!("{grim_host}:{grim_port}")
        }
    });

    let status = Command::new(program)
        .args(&cmd_args)
        .env("OLLAMA_HOST", &ollama_host)
        .env("GRIM_HOST", &grim_host)
        .env("GRIM_PORT", &grim_port)
        .status()
        .map_err(|e| {
            Error::Config(format!(
                "Failed to start {}: {}. Is it installed?",
                client_name(client),
                e
            ))
        })?;

    if !status.success() {
        return Err(Error::Config(format!(
            "{} exited with status: {}",
            client_name(client),
            status
        )));
    }

    Ok(())
}

fn client_name(client: ClientIntegration) -> &'static str {
    match client {
        ClientIntegration::Hermes => "Hermes",
        ClientIntegration::Openclaw => "OpenClaw",
        ClientIntegration::Claw => "Claude Code",
        ClientIntegration::Codex => "Codex",
        ClientIntegration::Antigravity => "Antigravity",
        ClientIntegration::Zcode => "ZCode",
    }
}
