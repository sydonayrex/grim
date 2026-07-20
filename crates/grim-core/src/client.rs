//! Grim CLI Client — model downloads, status checks, auth, and terminal interactions.
//!
//! `download_model` is the primary entry point:
//!
//! - Accepts Ollama-style short names (`llama3`, `mistral:7b-q4_k_m`).
//! - Accepts Hugging Face refs (`hf:org/repo/file.gguf` or full HTTPS URLs to
//!   `.gguf` / `.safetensors` files).
//! - Accepts plain HTTPS URLs to any model file.
//!
//! After a successful download the function writes a JSON sidecar via
//! `catalog::ModelEntry::save` so `grim run <name>` can resolve the file
//! without a filesystem scan.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use crate::grim_models_dir;
use grim_tensor::error::{Error, Result};
use sha2::{Digest, Sha256};

use crate::catalog::ModelEntry;

// ---------------------------------------------------------------------------
// Ollama registry constants
// ---------------------------------------------------------------------------

const GRIM_REGISTRY: &str = "https://registry.ollama.ai";
const GRIM_LIBRARY_NS: &str = "library";

/// Progress update message sent during download.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DownloadProgress {
    /// Status description ("pulling manifest", "downloading", "verifying sha256 digest", "success").
    pub status: String,
    /// Digest of the file being downloaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    /// Total size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    /// Bytes completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Download a model with default console progress output.
///
/// `model_ref` is resolved as follows (first matching rule wins):
/// 1. `hf:<org>/<repo>/<file>` — Hugging Face direct file download.
/// 2. Any string containing `huggingface.co` or `hf.co` — treated as a
///    plain HTTPS Hugging Face URL.
/// 3. A plain `https://` or `http://` URL — downloaded as-is.
/// 4. `<name>` or `<name>:<tag>` — looked up in the Ollama registry.
///
/// `output` overrides the destination file path (defaults to
/// `grim_models_dir()/<derived_name>.gguf`).
pub async fn download_model(model_ref: &str, output: Option<String>) -> Result<()> {
    download_model_with_progress(model_ref, output, |p| {
        if p.status == "downloading" {
            let total = p.total.unwrap_or(0);
            let completed = p.completed.unwrap_or(0);
            if total > 0 {
                let pct = completed * 100 / total;
                print!(
                    "\r  [{:>3}%] {:.2} / {:.2} GB",
                    pct,
                    completed as f64 / 1_073_741_824.0,
                    total as f64 / 1_073_741_824.0,
                );
            } else {
                print!("\r  {:.2} MB downloaded", completed as f64 / 1_048_576.0);
            }
            let _ = std::io::stdout().flush();
        } else {
            if p.status == "success" {
                println!();
            }
            println!("[grim] {}", p.status);
        }
    }).await
}

/// Download a model while invoking the specified progress callback for updates.
///
/// See `download_model` for details on how `model_ref` is resolved.
pub async fn download_model_with_progress<F>(
    model_ref: &str,
    output: Option<String>,
    progress_fn: F,
) -> Result<()>
where
    F: Fn(DownloadProgress) + Send + Sync + Clone + 'static,
{
    // Ensure the destination directory exists.
    let models_dir = grim_models_dir();
    fs::create_dir_all(&models_dir)
        .map_err(|e| Error::Backend(format!("cannot create models dir {}: {e}", models_dir.display())))?;

    // Dispatch based on the reference format.
    if model_ref.starts_with("hf:") {
        download_huggingface(model_ref.trim_start_matches("hf:"), &models_dir, output, progress_fn).await
    } else if model_ref.contains("huggingface.co") || model_ref.contains("hf.co") {
        download_url(model_ref, derive_filename_from_url(model_ref), &models_dir, output, "huggingface", progress_fn).await
    } else if model_ref.starts_with("https://") || model_ref.starts_with("http://") {
        let fname = derive_filename_from_url(model_ref);
        download_url(model_ref, fname, &models_dir, output, "url", progress_fn).await
    } else {
        download_grim_registry(model_ref, &models_dir, output, progress_fn).await
    }
}

// ---------------------------------------------------------------------------
// Ollama registry download
// ---------------------------------------------------------------------------

/// Resolve and download a model from the Ollama registry.
async fn download_grim_registry<F>(
    model_ref: &str,
    models_dir: &Path,
    output: Option<String>,
    progress_fn: F,
) -> Result<()>
where
    F: Fn(DownloadProgress) + Send + Sync + Clone + 'static,
{
    let (ns, name, tag) = parse_grim_registry_ref(model_ref);

    progress_fn(DownloadProgress {
        status: format!("Pulling {} from Ollama registry (tag: {})...", model_ref, tag),
        digest: None,
        total: None,
        completed: None,
    });

    let client = build_http_client()?;

    // 1. Fetch manifest.
    let manifest_url = format!("{GRIM_REGISTRY}/v2/{ns}/{name}/manifests/{tag}");
    progress_fn(DownloadProgress {
        status: "pulling manifest".to_string(),
        digest: None,
        total: None,
        completed: None,
    });

    let manifest_resp = client
        .get(&manifest_url)
        .header("Accept", "application/vnd.docker.distribution.manifest.v2+json")
        .send()
        .await
        .map_err(|e| Error::Backend(format!("manifest fetch failed: {e}")))?;

    if !manifest_resp.status().is_success() {
        let status = manifest_resp.status();
        let body = manifest_resp.text().await.unwrap_or_default();
        return Err(Error::Backend(format!(
            "Ollama registry returned {status} for '{model_ref}'.\n\
             Hint: verify the model name with https://ollama.com/library\n\
             Registry response: {body}"
        )));
    }

    let manifest: serde_json::Value = manifest_resp
        .json()
        .await
        .map_err(|e| Error::Backend(format!("manifest parse failed: {e}")))?;

    // 2. Locate model layer(s).
    let layers = manifest["layers"]
        .as_array()
        .ok_or_else(|| Error::Backend("manifest has no 'layers' array".into()))?;

    let model_layer = layers
        .iter()
        .find(|l| {
            l["mediaType"]
                .as_str()
                .map(|t| t.contains("model") || t.ends_with(".gguf"))
                .unwrap_or(false)
        })
        .or_else(|| layers.first()) // Fallback: first layer.
        .ok_or_else(|| Error::Backend("manifest has no downloadable layers".into()))?;

    let digest = model_layer["digest"]
        .as_str()
        .ok_or_else(|| Error::Backend("layer has no digest".into()))?;

    let size = model_layer["size"].as_u64().unwrap_or(0);

    // 3. Build blob URL and destination path.
    let blob_url = format!("{GRIM_REGISTRY}/v2/{ns}/{name}/blobs/{digest}");
    let dest_name = format!("{}_{}.gguf", name, tag.replace(':', "_"));
    let dest_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| models_dir.join(&dest_name));

    // 4. Stream download with progress + digest.
    let sha256_hex = stream_download(&client, &blob_url, &dest_path, size, digest.to_string(), &progress_fn).await?;

    // 5. Write catalog sidecar.
    let entry = ModelEntry {
        name: model_ref.to_string(),
        path: dest_path.display().to_string(),
        arch: String::new(), // Will be enriched when loaded by GgufProvider.
        params: String::new(),
        quant: extract_quant_hint_from_tag(&tag),
        context_length: 0,
        size_bytes: size,
        sha256: sha256_hex,
        pulled_at: utc_now_rfc3339(),
        source: "ollama".to_string(),
    };
    entry.save(&dest_path)?;

    progress_fn(DownloadProgress {
        status: "success".to_string(),
        digest: None,
        total: None,
        completed: None,
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Hugging Face download
// ---------------------------------------------------------------------------

/// Download a specific file from a Hugging Face repository.
async fn download_huggingface<F>(
    hf_path: &str,
    models_dir: &Path,
    output: Option<String>,
    progress_fn: F,
) -> Result<()>
where
    F: Fn(DownloadProgress) + Send + Sync + Clone + 'static,
{
    let parts: Vec<&str> = hf_path.splitn(3, '/').collect();
    if parts.len() < 2 {
        return Err(Error::Backend(format!(
            "invalid Hugging Face reference '{hf_path}'. \
             Expected format: hf:org/repo/file.gguf  or  hf:org/repo"
        )));
    }

    let (org, repo) = (parts[0], parts[1]);
    let filename = if parts.len() == 3 {
        parts[2].to_string()
    } else {
        resolve_hf_gguf_filename(org, repo).await?
    };

    let url = format!("https://huggingface.co/{org}/{repo}/resolve/main/{filename}");
    let friendly_name = format!("{org}/{repo}");

    progress_fn(DownloadProgress {
        status: format!("Pulling {friendly_name}/{filename} from Hugging Face..."),
        digest: None,
        total: None,
        completed: None,
    });

    let client = build_http_client()?;
    let size = get_content_length(&client, &url).await.unwrap_or(0);

    let stem = Path::new(&filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&filename)
        .replace([' ', '/'], "_");
    let ext = Path::new(&filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("gguf");
    let dest_name = format!("{stem}.{ext}");
    let dest_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| models_dir.join(&dest_name));

    let sha256_hex = stream_download(&client, &url, &dest_path, size, filename.clone(), &progress_fn).await?;

    let entry = ModelEntry {
        name: format!("{org}/{repo}/{filename}"),
        path: dest_path.display().to_string(),
        arch: String::new(),
        params: String::new(),
        quant: extract_quant_hint_from_filename(&filename),
        context_length: 0,
        size_bytes: size,
        sha256: sha256_hex,
        pulled_at: utc_now_rfc3339(),
        source: "huggingface".to_string(),
    };
    entry.save(&dest_path)?;

    progress_fn(DownloadProgress {
        status: "success".to_string(),
        digest: None,
        total: None,
        completed: None,
    });

    Ok(())
}

/// Query the HF API to find the first GGUF file in a repository.
async fn resolve_hf_gguf_filename(org: &str, repo: &str) -> Result<String> {
    let client = build_http_client()?;
    let api_url = format!("https://huggingface.co/api/models/{org}/{repo}");
    let resp = client
        .get(&api_url)
        .send()
        .await
        .map_err(|e| Error::Backend(format!("HF API request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(Error::Backend(format!(
            "HF API returned {} for {org}/{repo}", resp.status()
        )));
    }

    let meta: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| Error::Backend(format!("HF API parse failed: {e}")))?;

    if let Some(siblings) = meta["siblings"].as_array() {
        let filenames: Vec<&str> = siblings
            .iter()
            .filter_map(|s| s["rfilename"].as_str())
            .collect();

        for pref in &["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q8_0", "F16"] {
            if let Some(f) = filenames.iter().find(|n| n.contains(pref) && n.ends_with(".gguf")) {
                return Ok(f.to_string());
            }
        }
        if let Some(f) = filenames.iter().find(|n| n.ends_with(".gguf")) {
            return Ok(f.to_string());
        }
    }

    Err(Error::Backend(format!(
        "No GGUF file found in {org}/{repo}. \
         Specify a filename: hf:{org}/{repo}/model.gguf"
    )))
}

// ---------------------------------------------------------------------------
// Plain URL download
// ---------------------------------------------------------------------------

async fn download_url<F>(
    url: &str,
    fname: String,
    models_dir: &Path,
    output: Option<String>,
    source: &str,
    progress_fn: F,
) -> Result<()>
where
    F: Fn(DownloadProgress) + Send + Sync + Clone + 'static,
{
    progress_fn(DownloadProgress {
        status: format!("Downloading {}...", url),
        digest: None,
        total: None,
        completed: None,
    });

    let client = build_http_client()?;
    let size = get_content_length(&client, url).await.unwrap_or(0);
    let dest_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| models_dir.join(&fname));

    let sha256_hex = stream_download(&client, url, &dest_path, size, fname.clone(), &progress_fn).await?;

    let entry = ModelEntry {
        name: fname,
        path: dest_path.display().to_string(),
        arch: String::new(),
        params: String::new(),
        quant: String::new(),
        context_length: 0,
        size_bytes: size,
        sha256: sha256_hex,
        pulled_at: utc_now_rfc3339(),
        source: source.to_string(),
    };
    entry.save(&dest_path)?;

    progress_fn(DownloadProgress {
        status: "success".to_string(),
        digest: None,
        total: None,
        completed: None,
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Streaming download with progress + SHA-256
// ---------------------------------------------------------------------------

/// Stream `url` to `dest`, calling `progress_fn` periodically.
async fn stream_download<F>(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    total_bytes: u64,
    digest: String,
    progress_fn: &F,
) -> Result<String>
where
    F: Fn(DownloadProgress) + Send + Sync,
{
    let part = dest.with_extension(
        format!(
            "{}.part",
            dest.extension().and_then(|e| e.to_str()).unwrap_or("tmp")
        )
    );

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| Error::Backend(format!("download GET failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(Error::Backend(format!(
            "download server returned {}: {}",
            resp.status(),
            url
        )));
    }

    let mut file = fs::File::create(&part)
        .map_err(|e| Error::Backend(format!("cannot create {}: {e}", part.display())))?;

    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut last_report: u64 = 0;
    const REPORT_INTERVAL: u64 = 1024 * 1024; // 1 MB updates for smooth stream

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| Error::Backend(format!("stream read error: {e}")))?;
        hasher.update(&bytes);
        file.write_all(&bytes)
            .map_err(|e| Error::Backend(format!("disk write error: {e}")))?;
        downloaded += bytes.len() as u64;

        if downloaded - last_report >= REPORT_INTERVAL || downloaded == total_bytes {
            progress_fn(DownloadProgress {
                status: "downloading".to_string(),
                digest: Some(digest.clone()),
                total: Some(total_bytes),
                completed: Some(downloaded),
            });
            last_report = downloaded;
        }
    }

    drop(file);

    progress_fn(DownloadProgress {
        status: "verifying sha256 digest".to_string(),
        digest: Some(digest.clone()),
        total: None,
        completed: None,
    });

    fs::rename(&part, dest)
        .map_err(|e| Error::Backend(format!("rename failed: {e}")))?;

    let sha256_hex = format!("{:x}", hasher.finalize());
    Ok(sha256_hex)
}

// ---------------------------------------------------------------------------
// Credential management
// ---------------------------------------------------------------------------

/// Save a provider API token to `~/.grim/credentials.toml`.
pub fn save_login_token(provider: &str, token: &str) -> Result<()> {
    let grim_dir = crate::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".grim");
    fs::create_dir_all(&grim_dir)
        .map_err(|e| Error::Backend(format!("failed to create credential directory: {e}")))?;

    let cred_path = grim_dir.join("credentials.toml");
    let mut content = if cred_path.exists() {
        fs::read_to_string(&cred_path).unwrap_or_default()
    } else {
        String::new()
    };

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
            .join("\n")
            + "\n";
    } else {
        content.push_str(&line);
    }

    fs::write(&cred_path, content)
        .map_err(|e| Error::Backend(format!("failed to write credentials: {e}")))?;
    println!("[grim] Stored credentials for {} in {}", provider, cred_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Model management helpers
// ---------------------------------------------------------------------------

/// Delete a model and its sidecar from the models directory.
pub fn delete_model(model_name: &str) -> Result<()> {
    use crate::catalog::resolve_model_path;
    match resolve_model_path(model_name) {
        Some(p) => {
            fs::remove_file(&p)
                .map_err(|e| Error::Backend(format!("failed to remove {}: {e}", p.display())))?;
            let sidecar = ModelEntry::sidecar_path_for(&p);
            let _ = fs::remove_file(sidecar); // Ignore if missing.
            println!("[grim] Deleted model: {}", model_name);
            Ok(())
        }
        None => {
            println!("[grim] Model not found in cache: {}", model_name);
            Ok(())
        }
    }
}

/// Sets the default model in `grim.toml`.
pub fn set_default_model(context: &str, model: &str) -> Result<()> {
    let config_paths = [
        "grim.toml",
        "/etc/grim/grim.toml",
        "C:\\Program Files\\Grim\\grim.toml",
    ];
    let mut config_path = PathBuf::from("grim.toml");
    for p in &config_paths {
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
"#
        .to_string()
    };

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
            .join("\n")
            + "\n";
    } else {
        content.push_str(&line);
    }

    fs::write(&config_path, content)
        .map_err(|e| Error::Backend(format!("failed to write config: {e}")))?;
    println!(
        "[grim] Context '{}' → model '{}' in {}",
        context,
        model,
        config_path.display()
    );
    Ok(())
}

/// Send unload/kill request to running server.
pub async fn unload_model_from_server(model_name: &str, addr: &str) -> Result<()> {
    let client = build_http_client()?;
    let payload = serde_json::json!({ "name": model_name });

    for scheme in &["https", "http"] {
        let url = format!("{scheme}://{addr}/v1/models/unload");
        if let Ok(res) = client.post(&url).json(&payload).send().await {
            if res.status().is_success() {
                println!("[grim] Unloaded '{}' from server.", model_name);
                return Ok(());
            }
        }
    }
    Err(Error::Backend("Failed to connect to local server.".to_string()))
}

/// Query local server status.
pub async fn query_server_status(addr: &str) -> Result<()> {
    let client = build_http_client()?;
    let mut val: Option<serde_json::Value> = None;
    for scheme in &["https", "http"] {
        let url = format!("{scheme}://{addr}/status");
        if let Ok(res) = client.get(&url).send().await {
            if res.status().is_success() {
                val = res.json().await.ok();
                break;
            }
        }
    }

    let val = val.ok_or_else(|| Error::Backend(format!("Could not connect to {addr}")))?;
    println!("\n=== Grim Service Status ===");
    println!("Server Status : {}", val["status"].as_str().unwrap_or("unknown"));
    println!("Hardware      : {}", val["processor"].as_str().unwrap_or("unknown"));
    println!("Default Model : {}\n", val["default_model"].as_str().unwrap_or("none"));

    println!("{:<25} {:<15} {:<15}", "LOADED MODEL", "SIZE", "PROCESSOR");
    println!("{}", "-".repeat(60));
    if let Some(arr) = val["loaded_models"].as_array() {
        if arr.is_empty() {
            println!("No models loaded.");
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

/// Returns the list of standard model search paths (Grim + Ollama + HF).
pub fn model_search_paths() -> Vec<(String, PathBuf)> {
    let mut paths = vec![("Grim".to_string(), grim_models_dir())];
    if let Some(home) = crate::home_dir() {
        paths.push(("Ollama".to_string(), home.join(".ollama").join("models")));
        paths.push(("HuggingFace".to_string(), home.join(".cache").join("huggingface").join("hub")));
        #[cfg(target_os = "linux")]
        paths.push(("Ollama System".to_string(), PathBuf::from("/usr/share/ollama/.ollama/models")));
    }
    paths
}

/// Scan the local cache and print a summary table.
pub fn check_model_cache() -> Result<()> {
    println!("\n=== Grim Model Cache ===");
    println!("{:<40} {:<10} {:<12} {:<20}", "MODEL", "STATUS", "SIZE", "SOURCE");
    println!("{}", "-".repeat(85));

    let mut found = false;
    for (source_name, dir) in model_search_paths() {
        if !dir.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if !matches!(ext, "gguf" | "grim") {
                    continue;
                }
                found = true;
                let size_gb = entry.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1_073_741_824.0;
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").replace('_', "/");
                println!("{:<40} {:<10} {:<12} {:<20}", name, "OK", format!("{size_gb:.2} GB"), source_name);
            }
        }
    }

    if !found {
        println!("No models found. Run 'grim pull llama3' to get started.");
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| Error::Backend(format!("failed to build HTTP client: {e}")))
}

async fn get_content_length(client: &reqwest::Client, url: &str) -> Option<u64> {
    let resp = client.head(url).send().await.ok()?;
    resp.headers()
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

fn derive_filename_from_url(url: &str) -> String {
    url.split('/')
        .last()
        .unwrap_or("model.gguf")
        .split('?')
        .next()
        .unwrap_or("model.gguf")
        .to_string()
}

fn parse_grim_registry_ref(model_ref: &str) -> (String, String, String) {
    let (ns_name, tag) = if let Some(pos) = model_ref.rfind(':') {
        let (left, right) = model_ref.split_at(pos);
        (left, right.trim_start_matches(':').to_string())
    } else {
        (model_ref, "latest".to_string())
    };

    if let Some(pos) = ns_name.find('/') {
        let (ns, name) = ns_name.split_at(pos);
        (ns.to_string(), name.trim_start_matches('/').to_string(), tag)
    } else {
        (GRIM_LIBRARY_NS.to_string(), ns_name.to_string(), tag)
    }
}

fn extract_quant_hint_from_tag(tag: &str) -> String {
    for q in &["q4_k_m", "q5_k_m", "q4_k_s", "q8_0", "q4_0", "f16", "bf16"] {
        if tag.to_lowercase().contains(q) {
            return q.to_uppercase();
        }
    }
    tag.to_string()
}

fn extract_quant_hint_from_filename(filename: &str) -> String {
    let lower = filename.to_lowercase();
    for q in &["q4_k_m", "q5_k_m", "q4_k_s", "q8_0", "q4_0", "f16", "bf16"] {
        if lower.contains(q) {
            return q.to_uppercase();
        }
    }
    String::new()
}

fn utc_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (year, month, day, hour, minute, second) = epoch_to_parts(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn epoch_to_parts(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let second = secs % 60;
    let minutes = secs / 60;
    let minute = minutes % 60;
    let hours = minutes / 60;
    let hour = hours % 24;
    let days = hours / 24;

    let mut year = 1970u64;
    let mut remaining = days;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    let month_days = [31u64, if is_leap(year) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u64;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        month += 1;
    }
    let day = remaining + 1;
    (year, month, day, hour, minute, second)
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn find_model_in_dir(dir: &Path, filename: &str) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }
    let direct = dir.join(filename);
    if direct.exists() {
        return Some(direct);
    }
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

/// Check that a model file is present in the cache before running it.
pub fn validate_model_cached(model_name: &str) -> Result<PathBuf> {
    if let Some(p) = crate::catalog::resolve_model_path(model_name) {
        return Ok(p);
    }

    let base = model_name
        .replace("https://", "")
        .replace("http://", "")
        .replace('/', "_");
    for (_, dir) in model_search_paths() {
        for ext in &["gguf", "grim"] {
            if let Some(p) = find_model_in_dir(&dir, &format!("{base}.{ext}")) {
                return Ok(p);
            }
        }
    }

    Err(Error::Backend(format!(
        "Model '{model_name}' not found in local cache.\n\
         Run 'grim pull {model_name}' to download it."
    )))
}
