use std::collections::HashMap;
use grim_tensor::error::{Result, Error};
use crate::gguf::GgufValue;

#[derive(Clone)]
pub struct GgufTokenizer {
    pub tokens: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub scores: Option<Vec<f32>>,
    pub model_type: String,
    /// BPE merge ranks: maps a "tokenA tokenB" pair string to its rank
    /// (lower = higher priority). Populated from HF tokenizer.json merges.
    pub bpe_merges: Option<HashMap<String, u32>>,
    /// GPT-2 byte-to-unicode lookup table for byte-level BPE decode.
    /// Maps unicode char code → original byte value.
    pub byte_decoder: Option<HashMap<char, u8>>,
}

impl GgufTokenizer {
    /// Load a tokenizer from a HuggingFace `tokenizer.json` file.
    ///
    /// Supports BPE and WordLevel model types. Reads the vocab from
    /// `model.vocab` (a JSON object mapping token strings to IDs) and
    /// constructs the same `tokens`/`token_to_id` structures that the
    /// GGUF metadata path produces, so downstream encode/decode works
    /// identically regardless of source format.
    pub fn from_hf_json(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Backend(format!("failed to read tokenizer.json: {e}")))?;
        let root: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| Error::Backend(format!("failed to parse tokenizer.json: {e}")))?;

        let model = root.get("model")
            .ok_or_else(|| Error::Backend("tokenizer.json missing 'model' key".into()))?;
        let model_type = model.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("bpe")
            .to_lowercase();

        let vocab_obj = model.get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Backend("tokenizer.json missing model.vocab".into()))?;

        let vocab_size = vocab_obj.len();
        let mut id_to_token = vec![String::new(); vocab_size];
        let mut token_to_id = HashMap::with_capacity(vocab_size);

        for (token, id_val) in vocab_obj {
            let id = id_val.as_u64()
                .ok_or_else(|| Error::Backend("vocab ID is not an integer".into()))? as usize;
            if id < vocab_size {
                id_to_token[id] = token.clone();
            }
            token_to_id.insert(token.clone(), id as u32);
        }

        // Merge added_tokens (special tokens) that may not be in model.vocab
        if let Some(added) = root.get("added_tokens").and_then(|v| v.as_array()) {
            for entry in added {
                let content = entry.get("content").and_then(|v| v.as_str());
                let id = entry.get("id").and_then(|v| v.as_u64());
                if let (Some(t), Some(id)) = (content, id) {
                    let id = id as usize;
                    if id >= id_to_token.len() {
                        id_to_token.resize(id + 1, String::new());
                    }
                    id_to_token[id] = t.to_string();
                    token_to_id.insert(t.to_string(), id as u32);
                }
            }
        }

        // Trim trailing empty entries
        while id_to_token.last().map_or(false, |s| s.is_empty()) {
            id_to_token.pop();
        }

        // Parse BPE merges from HF tokenizer.json (for BPE model type).
        // Merges are stored as an array of [tokenA, tokenB] pairs.
        let bpe_merges = if model_type == "bpe" {
            model.get("merges")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    let mut map = HashMap::with_capacity(arr.len());
                    for (rank, entry) in arr.iter().enumerate() {
                        // Merges can be either ["tokenA", "tokenB"] arrays
                        // or "tokenA tokenB" strings.
                        let pair = if let Some(pair_arr) = entry.as_array() {
                            if pair_arr.len() == 2 {
                                format!("{} {}",
                                    pair_arr[0].as_str().unwrap_or(""),
                                    pair_arr[1].as_str().unwrap_or(""))
                            } else { continue }
                        } else if let Some(s) = entry.as_str() {
                            s.to_string()
                        } else { continue };
                        map.insert(pair, rank as u32);
                    }
                    map
                })
        } else { None };

        Ok(Self {
            tokens: id_to_token,
            token_to_id,
            scores: None,
            model_type: model_type.clone(),
            bpe_merges,
            byte_decoder: if model_type == "bpe" { Some(gpt2_byte_decoder()) } else { None },
        })
    }

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
            bpe_merges: None,
            byte_decoder: None,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Special tokens that should pass through directly
        let special_tokens = [
            "<|startoftext|>", "<|endoftext|>", "<|pad|>",
            "<|im_start|>", "<|im_end|>",
            "<|system|>", "<|user|>", "<|assistant|>",
        ];

        // For byte-level BPE tokenizers (model_type == "bpe"), use the GPT-2
        // byte encoder to map text → byte-level unicode chars, then apply BPE
        // merges rank-by-rank.
        if self.model_type == "bpe" {
            return self.encode_bpe(text, &special_tokens);
        }

        // Legacy path: SentencePiece / llama-style tokenizers
        let uses_gpt2_bpe = self.token_to_id.keys().any(|k| k.contains('\u{0120}'));
        let uses_sentencepiece = self.token_to_id.keys().any(|k| k.contains('\u{2581}'));

        let processed = if uses_gpt2_bpe {
            let mut p = text.replace(" ", "\u{0120}").replace("\n", "\u{010A}");
            if !p.starts_with('\u{0120}') && !p.starts_with('\u{010A}') && !p.starts_with('<') {
                p.insert(0, '\u{0120}');
            }
            p
        } else if uses_sentencepiece || self.model_type == "llama" {
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
            let rest: String = chars[i..].iter().collect();
            let mut matched_special = false;
            for special in ["<|im_start|>", "<|im_end|>", "<|startoftext|>"] {
                if rest.starts_with(special) {
                    if let Some(&id) = self.token_to_id.get(special) {
                        ids.push(id);
                        i += special.chars().count();
                        matched_special = true;
                        break;
                    }
                }
            }
            if matched_special {
                continue;
            }

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
                    let score = self.scores.as_ref().map(|s| s[merged_id as usize]).unwrap_or(-(merged_id as f32));
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

    /// BPE encode using merge ranks. Handles special tokens, byte-level
    /// encoding, and applies merges greedily by rank priority.
    fn encode_bpe(&self, text: &str, special_tokens: &[&str]) -> Vec<u32> {
        let encoder = gpt2_byte_encoder();
        let mut result: Vec<u32> = Vec::new();
        let mut remaining = text;

        loop {
            // Try to match the longest special token at the current position
            let mut found_special: Option<(&str, usize)> = None;
            for st in special_tokens {
                if let Some(pos) = remaining.find(st) {
                    if found_special.is_none() || pos < found_special.unwrap().1 {
                        found_special = Some((st, pos));
                    }
                }
            }

            let (chunk, rest) = match found_special {
                Some((st, pos)) => {
                    let (before, after) = remaining.split_at(pos);
                    let rest = &after[st.len()..];
                    (before, Some((st, rest)))
                }
                None => (remaining, None),
            };

            // Encode the non-special chunk via BPE
            if !chunk.is_empty() {
                result.extend(self.bpe_encode_chunk(chunk, &encoder));
            }

            match rest {
                Some((st, r)) => {
                    if let Some(&id) = self.token_to_id.get(st) {
                        result.push(id);
                    }
                    remaining = r;
                }
                None => break,
            }
        }
        result
    }

    /// Apply BPE merges to a single chunk of text (no special tokens).
    fn bpe_encode_chunk(&self, text: &str, encoder: &HashMap<u8, char>) -> Vec<u32> {
        let merges = match &self.bpe_merges {
            Some(m) => m,
            None => {
                // No merges — just look up whole words/substrings in vocab
                return self.encode_fallback(text);
            }
        };

        // Split into words, encode each via BPE.
        // GPT-2 pretokenizer splits on whitespace but keeps the space prefix
        // attached to the following word. For simplicity here we split on
        // word boundaries using a regex-like approach.
        let mut ids: Vec<u32> = Vec::new();

        // Byte-encode the entire text: map each byte to its unicode char
        let byte_str: String = text.bytes()
            .map(|b| encoder.get(&b).copied().unwrap_or(b as char))
            .collect();

        // Split into "words" at spaces (Ġ in byte-level encoding).
        // GPT-2 splits as: ' word1 word2 ...' → ['Ġword1', 'Ġword2', ...]
        // We pre-tokenize by splitting after each Ġ (space).
        let words: Vec<&str> = split_on_gpt2_pretokenize(&byte_str);

        for word in words {
            if word.is_empty() { continue; }

            // Check if the whole word is a single token
            if let Some(&id) = self.token_to_id.get(word) {
                ids.push(id);
                continue;
            }

            // Apply BPE: start with individual chars, merge by rank
            let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            loop {
                // Find the best pair (lowest merge rank)
                let mut best_rank: Option<u32> = None;
                let mut best_idx: Option<usize> = None;

                for i in 0..symbols.len().saturating_sub(1) {
                    let pair = format!("{} {}", symbols[i], symbols[i + 1]);
                    if let Some(&rank) = merges.get(&pair) {
                        if best_rank.is_none() || rank < best_rank.unwrap() {
                            best_rank = Some(rank);
                            best_idx = Some(i);
                        }
                    }
                }

                match best_idx {
                    Some(idx) => {
                        // Merge the pair at idx
                        let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                        symbols[idx] = merged;
                        symbols.remove(idx + 1);
                    }
                    None => break,
                }
            }

            // Look up each symbol in vocab
            for sym in &symbols {
                if let Some(&id) = self.token_to_id.get(sym) {
                    ids.push(id);
                } else {
                    // Fallback: try individual chars
                    for c in sym.chars() {
                        let cs = c.to_string();
                        if let Some(&id) = self.token_to_id.get(&cs) {
                            ids.push(id);
                        } else {
                            ids.push(0); // unknown → pad
                        }
                    }
                }
            }
        }

        ids
    }

    /// Fallback encoding for when no BPE merges are available.
    fn encode_fallback(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for c in text.chars() {
            let cs = c.to_string();
            if let Some(&id) = self.token_to_id.get(&cs) {
                ids.push(id);
            } else {
                ids.push(0);
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        // For byte-level BPE tokenizers, we need to:
        // 1. Concatenate token strings (which contain byte-level unicode chars)
        // 2. Map each unicode char back to its original byte via byte_decoder
        // 3. Decode the resulting byte sequence as UTF-8
        if let Some(ref decoder) = self.byte_decoder {
            let mut text = String::new();
            for &id in ids {
                if id < self.tokens.len() as u32 {
                    text.push_str(&self.tokens[id as usize]);
                }
            }
            // Map byte-level unicode chars back to actual bytes
            let bytes: Vec<u8> = text.chars()
                .filter_map(|c| decoder.get(&c).copied())
                .collect();
            return String::from_utf8_lossy(&bytes).into_owned();
        }

        // Non-BPE path (GGUF/SentencePiece): concatenate and replace space markers
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
        out = out.replace('\u{0120}', " ").replace('\u{010A}', "\n");
        out
    }
}

/// GPT-2 byte-to-unicode mapping. Maps each of 256 byte values to a
/// specific unicode character. This is the standard `bytes_to_unicode()`
/// function from the GPT-2 implementation.
///
/// Printable ASCII + Latin-1 (33-126, 161-172, 174-255) map to themselves.
/// Everything else maps to U+0100 + offset.
fn gpt2_byte_encoder() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = Vec::new();
    for b in 33..=126 { bs.push(b); }
    for b in 161..=172 { bs.push(b); }
    for b in 174..=255 { bs.push(b); }
    let mut map = HashMap::new();
    let mut c: u32 = 0;
    for b in 0..=255u8 {
        if bs.contains(&b) {
            map.insert(b, b as char);
        } else {
            map.insert(b, char::from_u32(256 + c).unwrap());
            c += 1;
        }
    }
    map
}

/// Reverse of `gpt2_byte_encoder`: maps unicode chars back to byte values.
fn gpt2_byte_decoder() -> HashMap<char, u8> {
    gpt2_byte_encoder().into_iter().map(|(k, v)| (v, k)).collect()
}

/// GPT-2-style pre-tokenization. Splits the byte-level encoded string into
/// word units where a space (Ġ) starts a new word. This is a simplified
/// version of the GPT-2 regex `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
/// that handles the common cases.
fn split_on_gpt2_pretokenize(s: &str) -> Vec<&str> {
    let chars: Vec<char> = s.chars().collect();
    let mut words = Vec::new();
    let mut start = 0;

    for i in 0..chars.len() {
        // Split before Ġ (space char in GPT-2 byte encoding) unless we're at the start
        if chars[i] == '\u{0120}' && i > start {
            words.push(&s[start..char_byte_offset(&chars, i)]);
            start = char_byte_offset(&chars, i);
        }
    }
    if start < s.len() {
        words.push(&s[start..]);
    }
    words
}

/// Calculate the byte offset of char index `idx` in the original string
/// described by `chars`.
fn char_byte_offset(chars: &[char], idx: usize) -> usize {
    chars[..idx].iter().map(|c| c.len_utf8()).sum()
}
