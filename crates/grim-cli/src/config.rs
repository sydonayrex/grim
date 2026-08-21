//! Strongly-typed schema for `grim.toml`. CLI flags override these; TOML provides defaults.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct GrimToml {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub train: TrainConfig,
    #[serde(default)]
    pub template: TemplateConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ServerConfig {
    pub default_model: Option<String>,
    #[serde(default = "default_max_batched")]
    pub max_batched_tokens: usize,
    #[serde(default = "default_max_seqs")]
    pub max_num_seqs: usize,
    pub target_ttft_ms: Option<u64>,
    pub target_itl_ms: Option<u64>,
}

fn default_max_batched() -> usize {
    2048
}

fn default_max_seqs() -> usize {
    32
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TrainConfig {
    #[serde(default)]
    pub dataset: Vec<String>,
    #[serde(default)]
    pub mix_weights: Vec<f32>,
    #[serde(default)]
    pub dedup: bool,
    #[serde(default = "one_f32")]
    pub lora_plus_ratio: f32,
    #[serde(default)]
    pub relora_reset_steps: usize,
    #[serde(default)]
    pub use_oft: bool,
    #[serde(default = "default_rank")]
    pub oft_rank: usize,
    pub eval: Option<String>,
    #[serde(default)]
    pub eval_every_steps: usize,
    #[serde(default)]
    pub eval_warmup_steps: usize,
}

fn one_f32() -> f32 {
    1.0
}

fn default_rank() -> usize {
    8
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TemplateConfig {
    pub family: Option<String>,
    pub override_path: Option<String>,
}

impl GrimToml {
    pub fn from_path(path: &str) -> Result<Self, std::io::Error> {
        let text = std::fs::read_to_string(path)?;
        Ok(Self::from_str(&text).unwrap_or_default())
    }

    pub fn from_str(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_example() {
        let toml_text = r#"
[server]
default_model = "my-model.grim"
max_batched_tokens = 4096
max_num_seqs = 64
target_ttft_ms = 200
target_itl_ms = 30

[train]
dataset = ["./data/a.jsonl", "./data/b.jsonl"]
lora_plus_ratio = 4.0
relora_reset_steps = 100
use_oft = true
oft_rank = 8
eval = "./data/eval.jsonl"
eval_every_steps = 50

[template]
family = "chatml"
override_path = ""
"#;
        let cfg = GrimToml::from_str(toml_text).unwrap();
        assert_eq!(cfg.server.default_model.as_deref(), Some("my-model.grim"));
        assert_eq!(cfg.server.max_batched_tokens, 4096);
        assert_eq!(cfg.train.dataset.len(), 2);
        assert!((cfg.train.lora_plus_ratio - 4.0).abs() < 1e-6);
        assert_eq!(cfg.train.relora_reset_steps, 100);
        assert!(cfg.train.use_oft);
        assert_eq!(cfg.template.family.as_deref(), Some("chatml"));
    }
}
