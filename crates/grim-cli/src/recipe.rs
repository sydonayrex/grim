//! WI-E6 UX half: training recipe (YAML) loader and dataset registry with
//! sha256 verification.
//!
//! A recipe file maps onto `TrainOptions` fields; the dataset registry
//! (`data/dataset_info.json`) resolves a registry id to a real path and
//! verifies its content hash before training consumes it.

use grim_core::error::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;

/// Parsed recipe document. Field names match the YAML in docs/recipes/.
#[derive(Debug, Deserialize)]
pub struct Recipe {
    #[allow(dead_code)]
    pub recipe_version: u32,
    pub name: String,
    pub model: String,
    pub dataset: RecipeDataset,
    pub training: RecipeTraining,
    pub adapter_output: String,
}

#[derive(Debug, Deserialize)]
pub struct RecipeDataset {
    pub registry_id: String,
}

#[derive(Debug, Deserialize)]
pub struct RecipeTraining {
    pub mode: String,
    pub epochs: usize,
    pub lr: f32,
    pub rank: usize,
    pub alpha: f32,
    pub batch_size: usize,
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,
    #[serde(default)]
    pub warmup_steps: usize,
    #[serde(default = "default_logging_steps")]
    pub logging_steps: usize,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,
    pub optimizer: String,
    #[serde(default)]
    pub scheduler: String,
    #[serde(default)]
    pub early_stopping_patience: usize,
}

fn default_grad_accum() -> usize {
    1
}
fn default_logging_steps() -> usize {
    10
}
fn default_max_grad_norm() -> f32 {
    1.0
}

/// Minimal flat-YAML scalar map parser. Recipes are 2-level documents of
/// `key: value` lines with 2-space nesting and optional `>-` folded strings;
/// that subset is all this loader supports (no anchors, no flow styles, no
/// multi-doc).
fn parse_flat_yaml(text: &str) -> HashMap<String, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    let mut section = String::new();
    let mut folding = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if folding {
            // Folded continuation lines append with a space until dedent.
            if line.starts_with("  ") && !trimmed.ends_with(':') {
                if let Some(k) = map.get_mut(&section.clone()) {
                    k.push(' ');
                    k.push_str(trimmed.trim_end_matches('-').trim());
                }
                if trimmed.ends_with('-') {
                    folding = false;
                }
                continue;
            }
            folding = false;
        }
        let indent = line.len() - line.trim_start().len();
        let (key, value) = match trimmed.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => continue,
        };
        // Strip inline comments ("value  # explanation").
        let value = match value.find('#') {
            Some(pos) => value[..pos].trim(),
            None => value,
        };
        if indent == 0 {
            section = key.to_string();
            map.insert(key.to_string(), value.to_string());
            continue;
        }
        let full_key = format!("{section}.{key}");
        if value == ">-" || value == "|" {
            folding = true;
            map.insert(full_key.clone(), String::new());
        } else {
            let v = value.trim_matches('"').trim_matches('\'').to_string();
            map.insert(full_key, v);
        }
    }
    map
}

fn get<'a>(m: &'a HashMap<String, String>, key: &str) -> Result<&'a str> {
    m.get(key)
        .map(|s| s.as_str())
        .ok_or_else(|| Error::Config(format!("recipe missing field '{key}'")))
}

fn get_parse<T: std::str::FromStr>(m: &HashMap<String, String>, key: &str) -> Result<T> {
    get(m, key)?
        .parse::<T>()
        .map_err(|_| Error::Config(format!("recipe field '{key}': bad value")))
}

fn get_or<T: std::str::FromStr>(m: &HashMap<String, String>, key: &str, default: T) -> T {
    m.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Load and validate a recipe file.
pub fn load_recipe(path: &std::path::Path) -> Result<Recipe> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Config(format!("recipe {}: {e}", path.display())))?;
    let m = parse_flat_yaml(&text);
    let recipe = Recipe {
        recipe_version: get_parse(&m, "recipe_version")?,
        name: get(&m, "name")?.to_string(),
        model: get(&m, "model")?.to_string(),
        dataset: RecipeDataset {
            registry_id: get(&m, "dataset.registry_id")?.to_string(),
        },
        training: RecipeTraining {
            mode: get(&m, "training.mode")?.to_string(),
            epochs: get_parse(&m, "training.epochs")?,
            lr: get_parse(&m, "training.lr")?,
            rank: get_parse(&m, "training.rank")?,
            alpha: get_parse(&m, "training.alpha")?,
            batch_size: get_parse(&m, "training.batch_size")?,
            gradient_accumulation_steps: get_or(
                &m,
                "training.gradient_accumulation_steps",
                default_grad_accum(),
            ),
            warmup_steps: get_or(&m, "training.warmup_steps", 0),
            logging_steps: get_or(&m, "training.logging_steps", default_logging_steps()),
            max_grad_norm: get_or(&m, "training.max_grad_norm", default_max_grad_norm()),
            optimizer: get(&m, "training.optimizer")?.to_string(),
            scheduler: m.get("training.scheduler").cloned().unwrap_or_default(),
            early_stopping_patience: get_or(&m, "training.early_stopping_patience", 0),
        },
        adapter_output: get(&m, "adapter_output")?.to_string(),
    };
    if recipe.recipe_version != 1 {
        return Err(Error::Config(format!(
            "recipe {}: unsupported version {}",
            path.display(),
            recipe.recipe_version
        )));
    }
    Ok(recipe)
}

/// Resolve a dataset registry id via `data/dataset_info.json`, verify sha256
/// when the entry pins one, return the file path.
pub fn resolve_dataset(registry_id: &str) -> Result<std::path::PathBuf> {
    let registry_path = workspace_root().join("data/dataset_info.json");
    let text = std::fs::read_to_string(&registry_path)
        .map_err(|e| Error::Config(format!("dataset registry {}: {e}", registry_path.display())))?;
    let reg: std::collections::HashMap<String, DatasetEntry> =
        serde_json::from_str(&text).map_err(|e| Error::Config(format!("registry json: {e}")))?;
    let entry = reg
        .get(registry_id)
        .ok_or_else(|| Error::Config(format!("dataset '{registry_id}' not in registry")))?;

    let path = if std::path::Path::new(&entry.file).is_absolute() {
        std::path::PathBuf::from(&entry.file)
    } else {
        workspace_root().join(&entry.file)
    };
    if !path.exists() {
        return Err(Error::Config(format!(
            "dataset '{registry_id}' points at missing file {}",
            path.display()
        )));
    }

    if let Some(expected) = &entry.sha256 {
        let actual = sha256_hex_file(&path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(Error::Config(format!(
                "dataset '{registry_id}' sha256 mismatch: expected {expected}, got {actual}"
            )));
        }
    }
    Ok(path)
}

#[derive(Debug, Deserialize)]
struct DatasetEntry {
    file: String,
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    description: Option<String>,
}

/// Workspace root: crates/grim-cli → ../../. Anchored so tests and CLI
/// invocations from any cwd agree.
fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// SHA-256 of a file as lowercase hex. Uses a small pure-Rust implementation
/// to avoid adding a crypto dependency for this one hash.
fn sha256_hex_file(path: &std::path::Path) -> Result<String> {
    let bytes =
        std::fs::read(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    Ok(sha256_hex(&bytes))
}

// Minimal SHA-256 (FIPS 180-4). Only used for dataset integrity checks.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut w = [0u32; 64];
    for block in msg.chunks(64) {
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_answer_vectors() {
        // FIPS 180-4 test vectors.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn load_recipe_parses_lfm25_example() {
        // The committed example must always parse — it is the contract doc.
        let recipe = load_recipe(&workspace_root().join("docs/recipes/lora-lfm25.yaml"))
            .expect("committed recipe must parse");
        assert_eq!(recipe.name, "lora-lfm25-consumer");
        assert_eq!(recipe.training.mode, "lora");
        assert_eq!(recipe.training.rank, 16);
    }

    #[test]
    fn resolve_dataset_unknown_id_errors() {
        assert!(resolve_dataset("no-such-dataset-xyz").is_err());
    }

    #[test]
    fn resolve_dataset_wikitext_sample_resolves() {
        let p = resolve_dataset("wikitext2-sample").expect("registry sample resolves");
        assert!(p.exists(), "{} should exist", p.display());
    }
}
