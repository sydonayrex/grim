use std::collections::HashMap;
use grim_tensor::error::{Result, Error};
use crate::gguf::GgufValue;

#[derive(Clone)]
pub struct GgufTokenizer {
    pub tokens: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub scores: Option<Vec<f32>>,
    pub model_type: String,
}

impl GgufTokenizer {
    pub fn from_metadata(metadata: &HashMap<String, GgufValue>) -> Result<Self> {
        let model_type = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let tokens_val = metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| Error::Backend("tokenizer.ggml.tokens not found in GGUF metadata".into()))?;

        let array_tokens = tokens_val
            .as_array()
            .ok_or_else(|| Error::Backend("tokenizer.ggml.tokens is not an array".into()))?;

        let mut tokens = Vec::with_capacity(array_tokens.len());
        let mut token_to_id = HashMap::with_capacity(array_tokens.len());

        for (id, val) in array_tokens.iter().enumerate() {
            let t = val
                .as_str()
                .ok_or_else(|| Error::Backend("tokenizer.ggml.tokens contains non-string element".into()))?
                .to_string();
            token_to_id.insert(t.clone(), id as u32);
            tokens.push(t);
        }

        let scores = metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_f32().unwrap_or(0.0))
                    .collect::<Vec<f32>>()
            });

        Ok(Self {
            tokens,
            token_to_id,
            scores,
            model_type,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let processed = if self.model_type == "llama" || self.model_type == "lfm2" {
            let mut p = text.replace(" ", "\u{2581}");
            if !p.starts_with('\u{2581}') {
                p.insert(0, '\u{2581}');
            }
            p
        } else {
            text.to_string()
        };

        let mut ids: Vec<u32> = Vec::new();
        let chars: Vec<char> = processed.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c_str: String = chars[i].to_string();
            if let Some(&id) = self.token_to_id.get(&c_str) {
                ids.push(id);
                i += 1;
            } else {
                let byte_val = chars[i] as u32;
                let hex_str = format!("<0x{:02X}>", byte_val);
                if let Some(&id) = self.token_to_id.get(&hex_str) {
                    ids.push(id);
                    i += 1;
                } else {
                    ids.push(0); // fallback
                    i += 1;
                }
            }
        }

        loop {
            let mut best_pair: Option<(u32, u32)> = None;
            let mut best_score: f32 = f32::MIN;
            let mut best_merged_id: Option<u32> = None;

            for pair in ids.windows(2) {
                let t1 = pair[0];
                let t2 = pair[1];
                let t1_str = &self.tokens[t1 as usize];
                let t2_str = &self.tokens[t2 as usize];
                let merged_str = format!("{}{}", t1_str, t2_str);
                if let Some(&merged_id) = self.token_to_id.get(&merged_str) {
                    let score = self.scores.as_ref().map(|s| s[merged_id as usize]).unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_pair = Some((t1, t2));
                        best_merged_id = Some(merged_id);
                    }
                }
            }

            if let (Some(pair), Some(merged_id)) = (best_pair, best_merged_id) {
                let mut next_ids = Vec::with_capacity(ids.len());
                let mut idx = 0;
                while idx < ids.len() {
                    if idx + 1 < ids.len() && ids[idx] == pair.0 && ids[idx+1] == pair.1 {
                        next_ids.push(merged_id);
                        idx += 2;
                    } else {
                        next_ids.push(ids[idx]);
                        idx += 1;
                    }
                }
                if ids == next_ids {
                    break;
                }
                ids = next_ids;
            } else {
                break;
            }
        }

        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id < self.tokens.len() as u32 {
                let t = &self.tokens[id as usize];
                out.push_str(t);
            }
        }
        if self.model_type == "llama" || self.model_type == "lfm2" {
            out = out.replace("\u{2581}", " ");
        }
        out
    }
}
