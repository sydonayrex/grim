//! grim start - Start a client integration with a model.

use grim_core::error::{Error, Result};
use crate::ClientIntegration;
use std::process::Command;

/// Start a client integration with a model.
pub async fn cmd_start(
    client: ClientIntegration,
    model: Option<&str>,
    args: &[String],
) -> Result<()> {
    // Resolve model if provided
    let model_name = model.unwrap_or("default").to_string();

    let (program, mut cmd_args) = match client {
        ClientIntegration::Hermes => {
            ("hermes", vec!["--model", &model_name])
        }
        ClientIntegration::Openclaw => {
            ("openclaw", vec!["--model", &model_name])
        }
        ClientIntegration::Claw => {
            ("claude-code", vec!["--model", &model_name])
        }
        ClientIntegration::Codex => {
            ("codex", vec!["--model", &model_name])
        }
        ClientIntegration::Antigravity => {
            ("antigravity", vec!["--model", &model_name])
        }
        ClientIntegration::Zcode => {
            ("zcode", vec!["--model", &model_name])
        }
    };

    cmd_args.extend(args.iter().map(|s| s.as_str()));

    println!("Starting {} with model '{}'...", client_name(client), model_name);

    let status = Command::new(program)
        .args(&cmd_args)
        .status()
        .map_err(|e| Error::Config(format!("Failed to start {}: {}. Is it installed?", client_name(client), e)))?;

    if !status.success() {
        return Err(Error::Config(format!("{} exited with status: {}", client_name(client), status)));
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