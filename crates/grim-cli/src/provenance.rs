//! `grim provenance` — Model integrity verification and provenance summary.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use grim_core::catalog::list_local_models;
use grim_core::error::{Error, Result};
use grim_format::{GrimFile, read_gguf};
use sha2::{Digest, Sha256};

/// Compute streaming SHA-256 digest of a file.
fn compute_sha256(path: &Path) -> Result<String> {
    let file = File::open(path).map_err(|e| {
        Error::Config(format!(
            "failed to open file for hashing '{}': {e}",
            path.display()
        ))
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];

    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|e| Error::Config(format!("read error during hashing: {e}")))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Print comprehensive model provenance, hash, format, and catalog trust details.
pub fn cmd_provenance(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(Error::Config(format!(
            "model file not found: {}",
            path.display()
        )));
    }

    let meta = std::fs::metadata(path)
        .map_err(|e| Error::Config(format!("cannot read file metadata: {e}")))?;
    let size_bytes = meta.len();
    let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

    println!("=== Grim Model Provenance & Trust ===");
    println!("File Path         : {}", path.display());
    println!(
        "File Size         : {:.2} MB ({} bytes)",
        size_mb, size_bytes
    );

    print!("Calculating SHA256: ");
    let sha256 = compute_sha256(path)?;
    println!("{sha256}");

    // Format and architecture detection
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let path_str = path.to_string_lossy();

    if ext == "grim" {
        println!("Format            : Native GRIM (.grim)");
        if let Ok(mut f) = File::open(path) {
            match GrimFile::read(&mut f) {
                Ok(gf) => {
                    println!("Tensors           : {}", gf.tensors.len());
                    println!(
                        "Target GCN        : {}",
                        gf.metadata.target_gcn.as_deref().unwrap_or("auto")
                    );
                    println!("Wavefront Target  : Wave{}", gf.metadata.wavefront_size);
                    println!(
                        "Preferred Dtype   : {}",
                        gf.metadata.preferred_dtype.as_deref().unwrap_or("bf16")
                    );
                    println!(
                        "Quantization Method: {}",
                        gf.metadata.quant_method.as_deref().unwrap_or("standard")
                    );
                }
                Err(e) => {
                    println!("Header Validation : FAILED ({e})");
                }
            }
        }
    } else if ext == "gguf" || ext == "bin" {
        println!("Format            : GGUF Checkpoint (.gguf)");
        if let Ok(mut f) = File::open(path) {
            match read_gguf(&mut f) {
                Ok(gg) => {
                    let arch = gg
                        .metadata
                        .get("general.architecture")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    println!("Architecture      : {arch}");
                    println!("Tensors           : {}", gg.tensors.len());
                    println!("Metadata Fields   : {}", gg.metadata.len());
                }
                Err(e) => {
                    println!("Header Validation : FAILED ({e})");
                }
            }
        }
    } else {
        println!("Format            : Raw / Unrecognized ({ext})");
    }

    // Check catalog status
    let catalog = list_local_models();
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let in_catalog = catalog.iter().find(|m| {
        let p = Path::new(&m.path);
        let cp = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        cp == canonical || m.path == path_str || m.sha256 == sha256
    });

    println!("\n--- Catalog Trust Status ---");
    if let Some(entry) = in_catalog {
        println!("Catalog Registered: YES (alias: '{}')", entry.name);
        println!("Catalog Source    : {}", entry.source);
        println!("Catalog Pulled At : {}", entry.pulled_at);
    } else {
        println!("Catalog Registered: NO (local untracked file)");
    }
    println!();

    Ok(())
}
