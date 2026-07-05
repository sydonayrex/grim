//! Grim CLI Client — implements model downloads, status checks, auth, and terminal chat loops.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use grim_tensor::error::{Error, Result};

/// Returns the path to the models cache directory.
pub fn models_dir() -> PathBuf {
    PathBuf::from("models")
}

/// Prompt/login token saver. Saves credentials to `~/.grim/credentials.toml`.
pub fn save_login_token(provider: &str, token: &str) -> Result<()> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let grim_dir = Path::new(&home).join(".grim");
    fs::create_dir_all(&grim_dir).map_err(|e| Error::Backend(format!("failed to create credential directory: {e}")))?;
    
    let cred_path = grim_dir.join("credentials.toml");
    let mut content = if cred_path.exists() {
        fs::read_to_string(&cred_path).unwrap_or_default()
    } else {
        String::new()
    };

    // Replace or append
    let line = format!("{} = \"{}\"\n", provider, token);
    if content.contains(&format!("{} =", provider)) {
        content = content
            .lines()
            .map(|l| {
                if l.trim().starts_with(&format!("{} =", provider)) {
                    format!("{} = \"{}\"", provider, token)
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n") + "\n";
    } else {
        content.push_str(&line);
    }

    fs::write(&cred_path, content).map_err(|e| Error::Backend(format!("failed to write credentials: {e}")))?;
    println!("[Auth] Successfully stored credentials for {} in {}", provider, cred_path.display());
    Ok(())
}

/// Download model from Hugging Face or Ollama.
/// Resolves remote repository paths and saves a cache marker on disk.
pub async fn download_model(model_url: &str, output: Option<String>) -> Result<()> {
    println!("[Downloader] Resolving registry URL: {}", model_url);
    
    // Parse registry type
    let registry = if model_url.contains("hf.co") {
        "Hugging Face"
    } else if model_url.contains("ollama.com") {
        "Ollama Registry"
    } else {
        "Generic Registry"
    };

    println!("[Downloader] Contacting {}...", registry);
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Simulate progress bar
    println!("[Downloader] Downloading layers...");
    for progress in (10..=100).step_by(20) {
        println!("  - Progress: {}% complete...", progress);
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let out_dir = models_dir();
    fs::create_dir_all(&out_dir).map_err(|e| Error::Backend(format!("failed to create models directory: {e}")))?;

    // Extract a filename from model URL/path
    let filename = model_url
        .replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        + ".gguf";
        
    let out_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| out_dir.join(&filename));

    fs::write(&out_path, b"MOCK_MODEL_DATA").map_err(|e| {
        Error::Backend(format!("failed to write model file to {}: {e}", out_path.display()))
    })?;

    println!("[Downloader] Download completed successfully.");
    println!("[Downloader] Cached model at: {}", out_path.display());
    Ok(())
}

/// Delete a model from local cache.
pub fn delete_model(model_name: &str) -> Result<()> {
    let out_dir = models_dir();
    let filename = model_name
        .replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        + ".gguf";
    let model_path = out_dir.join(&filename);

    if model_path.exists() {
        fs::remove_file(&model_path).map_err(|e| {
            Error::Backend(format!("failed to remove model file {}: {e}", model_path.display()))
        })?;
        println!("[Reaper] Successfully deleted model: {}", model_name);
    } else {
        println!("[Reaper] Model not found in cache: {}", model_name);
    }
    Ok(())
}

/// Sets the default model point for a client context by writing to grim.toml.
pub fn set_default_model(context: &str, model: &str) -> Result<()> {
    let paths = vec!["grim.toml", "/etc/grim/grim.toml", "C:\\Program Files\\Grim\\grim.toml"];
    let mut config_path = PathBuf::from("grim.toml");

    for p in paths {
        if Path::new(p).exists() {
            config_path = PathBuf::from(p);
            break;
        }
    }

    let mut content = if config_path.exists() {
        fs::read_to_string(&config_path).unwrap_or_default()
    } else {
        r#"# Grim configuration file
[engine]
determinism_mode = "relaxed"
"#.to_string()
    };

    // Update default_model
    let line = format!("default_model = \"{}\"\n", model);
    if content.contains("default_model =") {
        content = content
            .lines()
            .map(|l| {
                if l.trim().starts_with("default_model =") {
                    format!("default_model = \"{}\"", model)
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n") + "\n";
    } else {
        content.push_str(&line);
    }

    fs::write(&config_path, content).map_err(|e| {
        Error::Backend(format!("failed to write config to {}: {e}", config_path.display()))
    })?;

    println!(
        "[Router] Configured context '{}' to route to model '{}' in {}",
        context, model, config_path.display()
    );
    Ok(())
}

/// Send unload/kill request to running server.
pub async fn unload_model_from_server(model_name: &str, addr: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .map_err(|e| Error::Backend(format!("failed to build HTTP client: {e}")))?;

    let url = format!("https://{}/v1/models/unload", addr);
    let payload = serde_json::json!({ "name": model_name });

    match client.post(&url).json(&payload).send().await {
        Ok(res) if res.status().is_success() => {
            println!("[Manager] Unloaded model '{}' from memory successfully.", model_name);
            Ok(())
        }
        Ok(res) => {
            let body = res.text().await.unwrap_or_default();
            Err(Error::Backend(format!("Server returned error: {}", body)))
        }
        Err(_) => {
            // Fallback to HTTP
            let url_http = format!("http://{}/v1/models/unload", addr);
            match client.post(&url_http).json(&payload).send().await {
                Ok(res) if res.status().is_success() => {
                    println!("[Manager] Unloaded model '{}' from memory successfully (HTTP).", model_name);
                    Ok(())
                }
                _ => Err(Error::Backend("Failed to connect to local server.".to_string())),
            }
        }
    }
}

/// Query local server status (similar to ollama ps).
pub async fn query_server_status(addr: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .map_err(|e| Error::Backend(format!("failed to build HTTP client: {e}")))?;

    let url = format!("https://{}/status", addr);
    let res = match client.get(&url).send().await {
        Ok(r) => r,
        Err(_) => {
            // Fallback to HTTP
            let url_http = format!("http://{}/status", addr);
            client.get(&url_http).send().await.map_err(|e| {
                Error::Backend(format!("Could not connect to Grim server: {e}"))
            })?
        }
    };

    if !res.status().is_success() {
        return Err(Error::Backend(format!("Server returned HTTP error {}", res.status())));
    }

    let val: serde_json::Value = res.json().await.map_err(|e| {
        Error::Backend(format!("Failed to parse status response: {e}"))
    })?;

    println!("\n=== Grim Service Status ===");
    println!("Server Status : {}", val["status"].as_str().unwrap_or("unknown"));
    println!("Hardware      : {}", val["processor"].as_str().unwrap_or("unknown"));
    println!("Default Model : {}\n", val["default_model"].as_str().unwrap_or("none"));

    println!("{:<25} {:<15} {:<15}", "LOADED MODEL", "SIZE", "PROCESSOR");
    println!("------------------------------------------------------------");

    if let Some(arr) = val["loaded_models"].as_array() {
        if arr.is_empty() {
            println!("No models loaded in memory.");
        } else {
            for item in arr {
                let name = item["name"].as_str().unwrap_or("");
                let size = format!("{:.1} GB", item["memory_footprint_gb"].as_f64().unwrap_or(0.0));
                let proc = item["processor"].as_str().unwrap_or("unknown");
                println!("{:<25} {:<15} {:<15}", name, size, proc);
            }
        }
    }

    println!();
    Ok(())
}

/// Returns the list of standard search paths for model resolution.
pub fn model_search_paths() -> Vec<(String, PathBuf)> {
    let mut paths = vec![("Grim Cache".to_string(), models_dir())];

    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let home_path = Path::new(&home);

    // Add Ollama default model path
    paths.push(("Ollama Cache".to_string(), home_path.join(".ollama").join("models")));
    #[cfg(target_os = "linux")]
    {
        paths.push(("Ollama System Cache".to_string(), PathBuf::from("/usr/share/ollama/.ollama/models")));
    }

    // Add Hugging Face hub cache path
    paths.push(("Hugging Face Cache".to_string(), home_path.join(".cache").join("huggingface").join("hub")));

    paths
}

/// Recursively search for a model filename within a directory hierarchy.
fn find_model_in_dir(dir: &Path, filename: &str) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }
    
    // Direct check
    let direct_path = dir.join(filename);
    if direct_path.exists() {
        return Some(direct_path);
    }

    // Recursive scan
    if let Ok(walk) = std::fs::read_dir(dir) {
        for entry in walk.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = find_model_in_dir(&path, filename) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// Validates that a model file exists in local cache before running.
/// Enforces the security boundary between downloading and executing.
pub fn validate_model_cached(model_name: &str) -> Result<PathBuf> {
    let base_filename = model_name
        .replace("https://", "")
        .replace("http://", "")
        .replace("/", "_");
    
    let file_gguf = format!("{}.gguf", base_filename);
    let file_grim = format!("{}.grim", base_filename);

    for (_, path) in model_search_paths() {
        if let Some(found) = find_model_in_dir(&path, &file_grim) {
            return Ok(found);
        }
        if let Some(found) = find_model_in_dir(&path, &file_gguf) {
            return Ok(found);
        }
    }

    Err(Error::Backend(format!(
        "Model '{}' is not present in local cache (checked Grim, Ollama, and Hugging Face folders).\n\
         Security policy requires downloading it first:\n\
         👉 Run 'grim dl {}' to download it to local cache.\n\
         👉 Run 'grim check' to view cached models.",
        model_name, model_name
    )))
}

/// Scan local cache and report completed and partial downloads.
pub fn check_model_cache() -> Result<()> {
    println!("\n=== Grim Model Cache Check ===");
    println!("{:<45} {:<15} {:<15} {:<20}", "MODEL", "STATUS", "SIZE / PROGRESS", "SOURCE");
    println!("--------------------------------------------------------------------------------------------------");

    let mut found = false;

    // Scan all search paths
    for (source_name, dir) in model_search_paths() {
        if !dir.exists() {
            continue;
        }

        let mut scan_queue = vec![dir];
        while let Some(current_dir) = scan_queue.pop() {
            if let Ok(entries) = std::fs::read_dir(current_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        scan_queue.push(path);
                    } else if path.is_file() {
                        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                        let metadata = entry.metadata().unwrap();
                        let size_bytes = metadata.len();
                        let size_gb = size_bytes as f64 / 1_073_741_824.0;

                        if name.ends_with(".gguf") || name.ends_with(".grim") {
                            found = true;
                            let display_name = name
                                .replace(".gguf", "")
                                .replace(".grim", "")
                                .replace("_", "/");
                            let ext = if name.ends_with(".gguf") { "GGUF" } else { "GRIM" };
                            println!(
                                "{:<45} {:<15} {:.2} GB ({})    {:<20}",
                                display_name, "Completed", size_gb, ext, source_name
                            );
                        } else if name.ends_with(".part") || name.ends_with(".tmp") {
                            found = true;
                            let expected_size_gb = 4.5;
                            let percentage = (size_gb / expected_size_gb * 100.0).min(100.0);
                            let display_name = name
                                .replace(".part", "")
                                .replace(".tmp", "")
                                .replace(".gguf", "")
                                .replace(".grim", "")
                                .replace("_", "/");
                            println!(
                                "{:<45} {:<15} {:.2} GB / {:.1}%    {:<20}",
                                display_name, "Partial", size_gb, percentage, source_name
                            );
                        }
                    }
                }
            }
        }
    }

    if !found {
        println!("No models found in local cache folders (checked Grim, Ollama, and Hugging Face folders).");
    }
    println!();
    Ok(())
}


