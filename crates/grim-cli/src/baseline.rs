//! Model-agnostic baseline preparation and measurement.
//!
//! This is intentionally a checkpoint-driven harness. It discovers all model
//! files in a directory, or consumes a newline-delimited manifest containing
//! paths that may not exist yet. Existing checkpoints can be loaded through
//! the normal model loader and exercised with one fixed prefill workload; the
//! same command therefore works for dense, MoE, recurrent, and future models
//! without adding architecture-specific benchmark code.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use grim_core::error::{Error, Result};
use grim_core::model::CausalLm;
use grim_tensor::{Device, Shape};
use serde::Serialize;

const MODEL_EXTENSIONS: &[&str] = &["gguf", "grim", "safetensors", "bin"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineStatus {
    /// The manifest entry is not present on disk yet.
    Pending,
    /// A checkpoint exists but execution was not requested.
    Ready,
    /// A checkpoint loaded and all requested baseline iterations completed.
    Passed,
    /// A checkpoint was present but could not be loaded or executed.
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaselineRecord {
    pub path: String,
    pub status: BaselineStatus,
    pub architecture: Option<String>,
    pub device: String,
    pub prompt_tokens: usize,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub forward_ms: Option<f64>,
    pub samples_ms: Vec<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaselineReport {
    pub model_dir: String,
    pub device: String,
    pub execute: bool,
    pub records: Vec<BaselineRecord>,
}

#[derive(Debug, Clone)]
pub struct BaselineOptions {
    pub model_dir: PathBuf,
    pub manifest: Option<PathBuf>,
    pub execute: bool,
    pub prompt_tokens: usize,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub device: Option<String>,
}

impl BaselineOptions {
    pub fn validate(&self) -> Result<()> {
        if self.prompt_tokens == 0 {
            return Err(Error::Config(
                "baseline prompt_tokens must be greater than zero".into(),
            ));
        }
        if self.measured_iterations == 0 {
            return Err(Error::Config(
                "baseline measured_iterations must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

pub fn is_model_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_ascii_lowercase();
            MODEL_EXTENSIONS.contains(&ext.as_str())
        })
        .unwrap_or(false)
}

fn collect_model_files(root: &Path, depth: usize, output: &mut Vec<PathBuf>) -> Result<()> {
    if depth == 0 {
        return Ok(());
    }
    if !root.exists() {
        return Err(Error::Config(format!(
            "baseline model directory '{}' does not exist",
            root.display()
        )));
    }
    let entries = fs::read_dir(root).map_err(|error| {
        Error::Config(format!(
            "failed to scan baseline directory '{}': {error}",
            root.display()
        ))
    })?;
    for entry in entries {
        let path = entry
            .map_err(|error| {
                Error::Config(format!(
                    "failed to read baseline directory entry under '{}': {error}",
                    root.display()
                ))
            })?
            .path();
        if path.is_dir() {
            collect_model_files(&path, depth - 1, output)?;
        } else if is_model_path(&path) {
            output.push(path);
        }
    }
    Ok(())
}

fn manifest_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let text = fs::read_to_string(path).map_err(|error| {
        Error::Config(format!(
            "failed to read baseline manifest '{}': {error}",
            path.display()
        ))
    })?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(PathBuf::from)
        .collect())
}

pub fn discover_baseline_paths(options: &BaselineOptions) -> Result<Vec<PathBuf>> {
    let mut paths = match &options.manifest {
        Some(path) => manifest_paths(path)?,
        None => {
            let mut found = Vec::new();
            collect_model_files(&options.model_dir, 4, &mut found)?;
            found
        }
    };
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn load_checkpoint(path: &Path, device: &Device) -> Result<Box<dyn CausalLm>> {
    let path_string = path.to_string_lossy().to_string();
    let lower = path_string.to_ascii_lowercase();
    if lower.ends_with(".gguf") {
        grim_engine::model_loader::load_model_from_gguf(&path_string, device.clone())
    } else if lower.ends_with(".grim") {
        grim_engine::model_loader::load_model_from_grim(&path_string, device.clone())
    } else if lower.ends_with(".safetensors") || lower.ends_with(".bin") {
        grim_engine::model_loader::load_model_from_safetensors(&path_string, device.clone())
    } else {
        Err(Error::Config(format!(
            "unsupported baseline model extension: {}",
            path.display()
        )))
    }
}

fn run_checkpoint(
    model: &dyn CausalLm,
    device: &Device,
    options: &BaselineOptions,
) -> Result<(Vec<f64>,)> {
    let input_data = vec![0.0f32; options.prompt_tokens];
    let position_data: Vec<f32> = (0..options.prompt_tokens)
        .map(|position| position as f32)
        .collect();
    let shape = Shape::new(vec![1, options.prompt_tokens]);
    let input = crate::run::build_tensor(&input_data, &shape, device)?;
    let positions = crate::run::build_tensor(&position_data, &shape, device)?;

    for _ in 0..options.warmup_iterations {
        let mut session = model.new_session();
        let output = model.forward(&mut *session, &input, &positions, &[])?;
        // Reading the output forces lazy device work to complete before the
        // next iteration; it also makes the baseline comparable across CPU,
        // ROCm, and future backends.
        let _ = output.to_vec_f32()?;
    }

    let mut samples = Vec::with_capacity(options.measured_iterations);
    for _ in 0..options.measured_iterations {
        let mut session = model.new_session();
        let start = Instant::now();
        let output = model.forward(&mut *session, &input, &positions, &[])?;
        let _ = output.to_vec_f32()?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    Ok((samples,))
}

fn base_record(
    path: &Path,
    status: BaselineStatus,
    device: &str,
    options: &BaselineOptions,
) -> BaselineRecord {
    BaselineRecord {
        path: path.display().to_string(),
        status,
        architecture: None,
        device: device.to_string(),
        prompt_tokens: options.prompt_tokens,
        warmup_iterations: options.warmup_iterations,
        measured_iterations: options.measured_iterations,
        forward_ms: None,
        samples_ms: Vec::new(),
        error: None,
    }
}

pub fn run_baseline(options: BaselineOptions) -> Result<BaselineReport> {
    options.validate()?;
    let (device, device_label) = crate::run::probe_device_with(options.device.as_deref())?;
    let paths = discover_baseline_paths(&options)?;
    let mut records = Vec::with_capacity(paths.len());

    for path in paths {
        if !path.is_file() {
            let mut record = base_record(&path, BaselineStatus::Pending, &device_label, &options);
            record.error = Some("checkpoint not available yet".into());
            records.push(record);
            continue;
        }
        if !options.execute {
            records.push(base_record(
                &path,
                BaselineStatus::Ready,
                &device_label,
                &options,
            ));
            continue;
        }

        let mut record = base_record(&path, BaselineStatus::Failed, &device_label, &options);
        match load_checkpoint(&path, &device) {
            Ok(model) => {
                record.architecture = Some(model.config().name().to_string());
                match run_checkpoint(model.as_ref(), &device, &options) {
                    Ok((samples,)) => {
                        record.status = BaselineStatus::Passed;
                        record.forward_ms =
                            Some(samples.iter().sum::<f64>() / samples.len() as f64);
                        record.samples_ms = samples;
                    }
                    Err(error) => record.error = Some(error.to_string()),
                }
            }
            Err(error) => record.error = Some(error.to_string()),
        }
        records.push(record);
    }

    Ok(BaselineReport {
        model_dir: options.model_dir.display().to_string(),
        device: device_label,
        execute: options.execute,
        records,
    })
}

pub fn cmd_baseline(options: BaselineOptions) -> Result<()> {
    let report = run_baseline(options)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| Error::Backend(format!("serialize baseline report: {error}")))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_extensions_are_case_insensitive() {
        assert!(is_model_path(Path::new("model.gguf")));
        assert!(is_model_path(Path::new("model.GGUF")));
        assert!(is_model_path(Path::new("model.safetensors")));
        assert!(!is_model_path(Path::new("README.md")));
    }

    #[test]
    fn manifest_keeps_pending_paths() {
        let dir = std::env::temp_dir().join(format!(
            "grim-baseline-manifest-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let present = dir.join("present.gguf");
        fs::write(&present, b"not a real checkpoint").unwrap();
        let manifest = dir.join("models.manifest");
        fs::write(
            &manifest,
            format!(
                "# pending model\n{}\n{}\n",
                present.display(),
                dir.join("future.gguf").display()
            ),
        )
        .unwrap();

        let options = BaselineOptions {
            model_dir: dir.clone(),
            manifest: Some(manifest),
            execute: false,
            prompt_tokens: 4,
            warmup_iterations: 0,
            measured_iterations: 1,
            device: None,
        };
        let paths = discover_baseline_paths(&options).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().any(|path| path.ends_with("future.gguf")));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_workload_is_rejected() {
        let options = BaselineOptions {
            model_dir: PathBuf::from("models"),
            manifest: None,
            execute: false,
            prompt_tokens: 0,
            warmup_iterations: 0,
            measured_iterations: 1,
            device: None,
        };
        assert!(options.validate().is_err());
    }
}
