//! grim show - Show available models organized by format (GRIM > GGUF > others).

use grim_core::catalog::{list_local_models, ModelEntry};
use grim_core::error::Result;

/// Show available models organized by extension priority.
pub async fn cmd_show(verbose: bool) -> Result<()> {
    let entries = list_local_models();

    if entries.is_empty() {
        println!("No models found in cache.");
        println!("Run 'grim pull <model>' to download models.");
        return Ok(());
    }

    // Group by format
    let mut grim_models = Vec::new();
    let mut gguf_models = Vec::new();
    let mut other_models = Vec::new();

    for entry in entries {
        let path = std::path::PathBuf::from(&entry.path);
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_lowercase();

        match ext.as_str() {
            "grim" => grim_models.push(entry),
            "gguf" => gguf_models.push(entry),
            _ => other_models.push(entry),
        }
    }

    // Print in priority order
    print_section("GRIM (ROCm-optimized)", &grim_models, verbose);
    print_section("GGUF (General format)", &gguf_models, verbose);
    if !other_models.is_empty() {
        print_section("Other formats", &other_models, verbose);
    }

    // Summary
    println!("\nTotal: {} GRIM, {} GGUF, {} Other",
        grim_models.len(), gguf_models.len(), other_models.len());

    Ok(())
}

fn print_section(title: &str, models: &[ModelEntry], verbose: bool) {
    if models.is_empty() {
        return;
    }

    println!("\n=== {} ({}) ===", title, models.len());

    for entry in models {

        if verbose {
            println!("  {}", entry.name);
            println!("    Path:      {}", entry.path);
            if !entry.arch.is_empty() { println!("    Arch:      {}", entry.arch); }
            if !entry.params.is_empty() { println!("    Params:    {}", entry.params); }
            if !entry.quant.is_empty() { println!("    Quant:     {}", entry.quant); }
            if entry.context_length > 0 { println!("    Context:   {}", entry.context_length); }
            if entry.size_bytes > 0 {
                println!("    Size:      {}", format_bytes(entry.size_bytes));
            }
            if !entry.sha256.is_empty() { println!("    SHA256:    {}", &entry.sha256[..16.min(entry.sha256.len())]); }
            if !entry.pulled_at.is_empty() { println!("    Pulled:    {}", entry.pulled_at); }
            if !entry.source.is_empty() { println!("    Source:    {}", entry.source); }
        } else {
            let details = [
                if !entry.params.is_empty() { Some(format!("{}", entry.params)) } else { None },
                if !entry.quant.is_empty() { Some(entry.quant.clone()) } else { None },
                if entry.context_length > 0 { Some(format!("ctx{}", entry.context_length)) } else { None },
                if entry.size_bytes > 0 { Some(format_bytes(entry.size_bytes)) } else { None },
            ].into_iter().flatten().collect::<Vec<_>>().join(" | ");

            println!("  {}  [{}]", entry.name, details);
        }
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}