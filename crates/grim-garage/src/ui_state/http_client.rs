//! HTTP client — fetches models, datasets, devices, and starts jobs from
//! the local grim-garage axum API. The CVKG runtime invokes this on
//! startup and on refresh; tests use the in-process mock to confirm wire
//! formats match what the server returns.

use serde::Deserialize;

use crate::discovery::{DatasetEntry, ModelEntry};
use crate::jobs::TrainingMode;
use crate::rocm::RocmDeviceInfo;

#[derive(Debug, Deserialize)]
struct ModelsEnvelope {
    models: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct DatasetsEnvelope {
    datasets: Vec<DatasetEntry>,
}

#[derive(Debug, Deserialize)]
struct DevicesEnvelope {
    devices: Vec<RocmDeviceInfo>,
}

/// Job summary mirroring the wire format returned by `/api/train/jobs`.
/// Held as a private DTO here so neither the UI nor the server depends
/// on the other for this type.
#[derive(Debug, Deserialize, Clone)]
pub struct JobSummaryDto {
    pub job_id: String,
    pub status: String,
    pub model_path: String,
    pub dataset_path: String,
    pub training_mode: TrainingMode,
}

#[derive(Debug, Deserialize)]
struct JobsEnvelope {
    jobs: Vec<JobSummaryDto>,
}

/// One-stop shop for the CVKG runtime to call against the local backend.
#[derive(Debug, Clone)]
pub struct GarageClient {
    base_url: String,
}

impl GarageClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { base_url: base_url.into() }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    // ----- live GETs against the local backend -----

    /// `GET /api/models`. Returns the parsed list, or an error on
    /// network / parse failure so the poller can skip that endpoint
    /// for this round.
    pub async fn get_models(&self) -> Result<Vec<ModelEntry>, String> {
        let body = self.get_json("/api/models").await?;
        self.parse_models(&body).map_err(|e| e.to_string())
    }

    /// `GET /api/datasets`.
    pub async fn get_datasets(&self) -> Result<Vec<DatasetEntry>, String> {
        let body = self.get_json("/api/datasets").await?;
        self.parse_datasets(&body).map_err(|e| e.to_string())
    }

    /// `GET /api/rocm/devices`.
    pub async fn get_devices(&self) -> Result<Vec<RocmDeviceInfo>, String> {
        let body = self.get_json("/api/rocm/devices").await?;
        self.parse_devices(&body).map_err(|e| e.to_string())
    }

    /// `GET /api/train/jobs`.
    pub async fn get_jobs(&self) -> Result<Vec<JobSummaryDto>, String> {
        let body = self.get_json("/api/train/jobs").await?;
        let env: JobsEnvelope =
            serde_json::from_str(&body).map_err(|e| e.to_string())?;
        Ok(env.jobs)
    }

    /// Tiny async GET using tokio's TcpStream — no heavyweight reqwest
    /// dep needed. Trims the response body at the first empty line of
    /// the chunked encoding or at connection close.
    async fn get_json(&self, path: &str) -> Result<String, String> {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        let host_port_path = url
            .trim_start_matches("http://")
            .trim_start_matches("https://");
        let (host_port, req_path) = match host_port_path.find('/') {
            Some(i) => (&host_port_path[..i], &host_port_path[i..]),
            None => (host_port_path, "/"),
        };
        let request = format!(
            "GET {path} HTTP/1.1\r\n\
             Host: {host}\r\n\
             Connection: close\r\n\
             \r\n",
            path = req_path,
            host = host_port,
        );

        let mut stream = tokio::net::TcpStream::connect(host_port.to_string())
            .await
            .map_err(|e| format!("connect {host_port}: {e}"))?;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        stream
            .write_all(request.as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        stream.flush().await.map_err(|e| e.to_string())?;

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await.map_err(|e| e.to_string())?;

        let head_end = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| "no CRLF CRLF in response".to_string())?;
        let (head, body) = raw.split_at(head_end + 4);
        let head_str = String::from_utf8_lossy(head);
        let status_line = head_str
            .lines()
            .next()
            .ok_or_else(|| "missing status line".to_string())?;
        let status = status_line
            .split_ascii_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or_else(|| "bad status code".to_string())?;
        if !(200..300).contains(&status) {
            return Err(format!(
                "GET {} returned {}: {}",
                path,
                status,
                String::from_utf8_lossy(body)
            ));
        }
        String::from_utf8(body.to_vec()).map_err(|e| e.to_string())
    }

    /// Parse the JSON body returned by `GET /api/models` into a typed list.
    pub fn parse_models(&self, body: &str) -> Result<Vec<ModelEntry>, serde_json::Error> {
        let env: ModelsEnvelope = serde_json::from_str(body)?;
        Ok(env.models)
    }

    pub fn parse_datasets(&self, body: &str) -> Result<Vec<DatasetEntry>, serde_json::Error> {
        let env: DatasetsEnvelope = serde_json::from_str(body)?;
        Ok(env.datasets)
    }

    pub fn parse_devices(&self, body: &str) -> Result<Vec<RocmDeviceInfo>, serde_json::Error> {
        let env: DevicesEnvelope = serde_json::from_str(body)?;
        Ok(env.devices)
    }

    /// Build the JSON body for `POST /api/train/start`.
    pub fn build_start_training_request(
        model_path: &str,
        dataset_path: &str,
        training_mode: TrainingMode,
        lora_rank: u32,
        learning_rate: f64,
        epochs: u32,
        rocm_rmsnorm_matmul: bool,
        rocm_qkv_attention: bool,
    ) -> Result<String, serde_json::Error> {
        let body = serde_json::json!({
            "model_path": model_path,
            "dataset_path": dataset_path,
            "training_mode": match training_mode {
                TrainingMode::Lora => "Lora",
                TrainingMode::QLoRA => "QLoRA",
                TrainingMode::Bf16Full => "Bf16Full",
                TrainingMode::Orpo => "Orpo",
                TrainingMode::Dpo => "Dpo",
                TrainingMode::Grpo => "Grpo",
            },
            "lora_rank": lora_rank,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "rocm_fusion_rmsnorm_matmul": rocm_rmsnorm_matmul,
            "rocm_fusion_qkv_attention": rocm_qkv_attention,
        });
        serde_json::to_string(&body)
    }

    /// Parse an SSE metric stream body into `(step, loss)` tuples.
    pub fn parse_sse_data_line(&self, line: &str) -> Option<(u64, f64)> {
        // Format: `data: { "job_id":..., "metric":{ "step":N, "loss":F, "tokens":T } }`
        let payload = line.strip_prefix("data: ").unwrap_or(line);
        let v: serde_json::Value = serde_json::from_str(payload).ok()?;
        let step = v.get("metric")?.get("step")?.as_u64()?;
        let loss = v.get("metric")?.get("loss")?.as_f64()?;
        Some((step, loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_models_response() {
        let c = GarageClient::new("http://localhost:8741");
        let body = r#"{"models":[{"id":"tiny.gguf","path":"/tmp/tiny.gguf","format":"gguf","is_grim":false}]}"#;
        let models = c.parse_models(body).expect("parse");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "tiny.gguf");
    }

    #[test]
    fn parses_datasets_response() {
        let c = GarageClient::new("http://localhost:8741");
        let body = r#"{"datasets":[{"id":"train.jsonl","path":"/tmp/train.jsonl","format":"jsonl","size_bytes":1024}]}"#;
        let datasets = c.parse_datasets(body).expect("parse");
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].format, "jsonl");
    }

    #[test]
    fn parses_devices_response() {
        let c = GarageClient::new("http://localhost:8741");
        let body = r#"{"devices":[{"ordinal":0,"gcn_arch":"gfx1100","vram_bytes":0,"wavefront_size":32,"xnack_enabled":false}]}"#;
        let devices = c.parse_devices(body).expect("parse");
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].wavefront_size, 32);
    }

    #[test]
    fn builds_start_training_payload_for_lora() {
        let body = GarageClient::build_start_training_request(
            "/m.gguf",
            "/d.jsonl",
            TrainingMode::Lora,
            16,
            2e-5,
            1,
            true,
            false,
        )
        .expect("build");
        let v: serde_json::Value = serde_json::from_str(&body).expect("parse back");
        assert_eq!(v["model_path"], "/m.gguf");
        assert_eq!(v["training_mode"], "Lora");
        assert_eq!(v["lora_rank"], 16);
        assert!(v["rocm_fusion_rmsnorm_matmul"].as_bool().unwrap());
    }

    #[test]
    fn builds_start_training_payload_for_qlora() {
        let body = GarageClient::build_start_training_request(
            "/m.gguf",
            "/d.jsonl",
            TrainingMode::QLoRA,
            32,
            5e-5,
            3,
            true,
            true,
        )
        .expect("build");
        let v: serde_json::Value = serde_json::from_str(&body).expect("parse back");
        assert_eq!(v["training_mode"], "QLoRA");
        assert_eq!(v["epochs"], 3);
    }

    #[test]
    fn parses_sse_metric_line() {
        let c = GarageClient::new("http://localhost:8741");
        let line = r#"data: {"job_id":"abc","metric":{"step":12,"loss":1.42,"tokens":4096},"status":"running"}"#;
        let (step, loss) = c.parse_sse_data_line(line).expect("parse");
        assert_eq!(step, 12);
        assert!((loss - 1.42).abs() < 1e-9);
    }

    #[test]
    fn sse_line_without_data_prefix_is_rejected() {
        let c = GarageClient::new("http://localhost:8741");
        assert!(c.parse_sse_data_line("not-json").is_none());
    }
}
