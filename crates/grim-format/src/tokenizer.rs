use crate::gguf::GgufValue;
use grim_tensor::error::{Error, Result};
use std::collections::HashMap;

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
    /// EOS token ID if available from tokenizer metadata
    pub eos_token_id: Option<u32>,
    /// BOS token ID if available from tokenizer metadata (`tokenizer.ggml.bos_token_id`).
    pub bos_token_id: Option<u32>,
    /// Whether the tokenizer should prepend a BOS token (`tokenizer.ggml.add_bos_token`).
    pub add_bos_token: bool,
    /// UNK token ID if available from tokenizer metadata
    pub unk_token_id: Option<u32>,
    /// Jinja chat template extracted from GGUF `tokenizer.chat_template`, if present.
    /// Drives instruction-tuned prompt formatting in the CLI/server prompt path.
    pub chat_template: Option<String>,
}

impl Default for GgufTokenizer {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
            token_to_id: HashMap::new(),
            scores: None,
            model_type: "bpe".into(),
            bpe_merges: None,
            byte_decoder: None,
            eos_token_id: None,
            bos_token_id: None,
            add_bos_token: false,
            unk_token_id: None,
            chat_template: None,
        }
    }
}

impl GgufTokenizer {
    /// Return pad token ID if found, otherwise default to 0.
    pub fn pad_token_id(&self) -> u32 {
        self.token_to_id
            .get("<|pad|>")
            .or_else(|| self.token_to_id.get("<pad>"))
            .copied()
            .unwrap_or(0)
    }

    /// Return UNK token ID if configured or present in vocabulary, defaulting to 0.
    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
            .or_else(|| self.token_to_id.get("<unk>").copied())
            .or_else(|| self.token_to_id.get("<|unk|>").copied())
            .or_else(|| self.token_to_id.get("<|unknown|>").copied())
            .unwrap_or(0)
    }

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

        let model = root
            .get("model")
            .ok_or_else(|| Error::Backend("tokenizer.json missing 'model' key".into()))?;
        let model_type = model
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("bpe")
            .to_lowercase();

        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Backend("tokenizer.json missing model.vocab".into()))?;

        let vocab_size = vocab_obj.len();
        let mut id_to_token = vec![String::new(); vocab_size];
        let mut token_to_id = HashMap::with_capacity(vocab_size);

        for (token, id_val) in vocab_obj {
            let id = id_val
                .as_u64()
                .ok_or_else(|| Error::Backend("vocab ID is not an integer".into()))?
                as usize;
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
            model.get("merges").and_then(|v| v.as_array()).map(|arr| {
                let mut map = HashMap::with_capacity(arr.len());
                for (rank, entry) in arr.iter().enumerate() {
                    // Merges can be either ["tokenA", "tokenB"] arrays
                    // or "tokenA tokenB" strings.
                    let pair = if let Some(pair_arr) = entry.as_array() {
                        if pair_arr.len() == 2 {
                            format!(
                                "{} {}",
                                pair_arr[0].as_str().unwrap_or(""),
                                pair_arr[1].as_str().unwrap_or("")
                            )
                        } else {
                            continue;
                        }
                    } else if let Some(s) = entry.as_str() {
                        s.to_string()
                    } else {
                        continue;
                    };
                    map.insert(pair, rank as u32);
                }
                map
            })
        } else {
            None
        };

        let unk_token_id = token_to_id
            .get("<unk>")
            .or_else(|| token_to_id.get("<|unk|>"))
            .or_else(|| token_to_id.get("<|unknown|>"))
            .copied();

        Ok(Self {
            tokens: id_to_token,
            token_to_id,
            scores: None,
            model_type: model_type.clone(),
            bpe_merges,
            byte_decoder: if model_type == "bpe" {
                Some(gpt2_byte_decoder())
            } else {
                None
            },
            eos_token_id: None, // HF tokenizer.json doesn't have explicit EOS token ID in a standard way
            bos_token_id: None,
            add_bos_token: false,
            unk_token_id,
            chat_template: None,
        })
    }

    pub fn from_metadata(metadata: &HashMap<String, GgufValue>) -> Result<Self> {
        let model_type = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let tokens_val = metadata.get("tokenizer.ggml.tokens").ok_or_else(|| {
            Error::Backend("tokenizer.ggml.tokens not found in GGUF metadata".into())
        })?;

        let array_tokens = tokens_val
            .as_array()
            .ok_or_else(|| Error::Backend("tokenizer.ggml.tokens is not an array".into()))?;

        let mut tokens = Vec::with_capacity(array_tokens.len());
        let mut token_to_id = HashMap::with_capacity(array_tokens.len());

        for (id, val) in array_tokens.iter().enumerate() {
            let t = val
                .as_str()
                .ok_or_else(|| {
                    Error::Backend("tokenizer.ggml.tokens contains non-string element".into())
                })?
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

        // Extract EOS token ID from metadata (typically tokenizer.ggml.eos_token_id)
        let eos_token_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());

        // Extract BOS token ID and add_bos flag from metadata.
        let bos_token_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let add_bos_token = metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Extract UNK token ID from metadata (typically tokenizer.ggml.unknown_token_id)
        let unk_token_id = metadata
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u32());

        // Extract the Jinja chat template if the GGUF embeds one.
        let chat_template = metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Load BPE merges from GGUF metadata if the model type is BPE.
        // HF tokenizer.json-style merges may be stored as
        // tokenizer.ggml.merges (array of ["tokenA", "tokenB"] or "tokenA tokenB").
        let bpe_merges = if model_type == "bpe" || model_type == "gpt2" {
            metadata
                .get("tokenizer.ggml.merges")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    let mut map = HashMap::with_capacity(arr.len());
                    for (rank, entry) in arr.iter().enumerate() {
                        let pair = if let Some(pair_arr) = entry.as_array() {
                            if pair_arr.len() == 2 {
                                format!(
                                    "{} {}",
                                    pair_arr[0].as_str().unwrap_or(""),
                                    pair_arr[1].as_str().unwrap_or("")
                                )
                            } else {
                                continue;
                            }
                        } else if let Some(s) = entry.as_str() {
                            s.to_string()
                        } else {
                            continue;
                        };
                        map.insert(pair, rank as u32);
                    }
                    map
                })
        } else {
            None
        };

        let byte_decoder = if model_type == "bpe" || model_type == "gpt2" {
            Some(gpt2_byte_decoder())
        } else {
            None
        };

        Ok(Self {
            tokens,
            token_to_id,
            scores,
            model_type,
            bpe_merges,
            byte_decoder,
            eos_token_id,
            bos_token_id,
            add_bos_token,
            unk_token_id,
            chat_template,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Special tokens that should pass through directly
        let special_tokens = [
            "<|startoftext|>",
            "<|endoftext|>",
            "<|pad|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<s>",
            "</s>",
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
                    ids.push(self.unk_token_id()); // unknown token fallback
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
                // Bounds-check token lookup — panic on OOB is a DoS risk with
                // untrusted/malformed tokenizer configs. [P1-25 fix.]
                let t1_str = match self.tokens.get(t1 as usize) {
                    Some(s) => s,
                    None => continue,
                };
                let t2_str = match self.tokens.get(t2 as usize) {
                    Some(s) => s,
                    None => continue,
                };
                let merged_str = format!("{}{}", t1_str, t2_str);
                if let Some(&merged_id) = self.token_to_id.get(&merged_str) {
                    let score = self
                        .scores
                        .as_ref()
                        .and_then(|s| s.get(merged_id as usize).copied())
                        .unwrap_or(-(merged_id as f32));
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
                    if idx + 1 < ids.len() && ids[idx] == pair.0 && ids[idx + 1] == pair.1 {
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
        let byte_str: String = text
            .bytes()
            .map(|b| encoder.get(&b).copied().unwrap_or(b as char))
            .collect();

        // Split into "words" at spaces (Ġ in byte-level encoding).
        // GPT-2 splits as: ' word1 word2 ...' → ['Ġword1', 'Ġword2', ...]
        // We pre-tokenize by splitting after each Ġ (space).
        let words: Vec<&str> = split_on_gpt2_pretokenize(&byte_str);

        for word in words {
            if word.is_empty() {
                continue;
            }

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
                            ids.push(self.unk_token_id()); // unknown token fallback
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
                ids.push(self.unk_token_id());
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
            let mut bytes: Vec<u8> = Vec::with_capacity(text.len());
            for c in text.chars() {
                if let Some(&b) = decoder.get(&c) {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
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

fn deserialize_chat_content<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct ContentVisitor;

    impl<'de> serde::de::Visitor<'de> for ContentVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or an array of content parts")
        }

        fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v.to_string())
        }

        fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut text_parts = Vec::new();
            while let Some(elem) = seq.next_element::<serde_json::Value>()? {
                if let Some(s) = elem.as_str() {
                    text_parts.push(s.to_string());
                } else if let Some(obj) = elem.as_object() {
                    if let Some(t) = obj.get("text").and_then(|t| t.as_str()) {
                        text_parts.push(t.to_string());
                    }
                }
            }
            Ok(text_parts.join("\n"))
        }
    }

    deserializer.deserialize_any(ContentVisitor)
}

/// A single chat message in an OpenAI-style `messages` array.
///
/// `tool_calls` carries one or more tool invocations an assistant message
/// produced (assistant-role only). `tool_call_id` / `name` carry a tool result
/// back to the model (tool-role only). Both are `Option`al so the common
/// user/assistant text-only messages stay the simple two-field construction
/// callers already use — the new fields default to `None`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(deserialize_with = "deserialize_chat_content")]
    pub content: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallMsg>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

/// One tool call embedded in an assistant `ChatMessage`. `arguments` is a
/// JSON-encoded *string* matching OpenAI's wire format (the arguments are a
/// string containing JSON, not a nested object), so it can be re-parsed by the
/// caller without ambiguity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ToolCallMsg {
    pub id: String,
    pub name: String,
    /// JSON-encoded string of the tool's arguments, per OpenAI wire format.
    pub arguments: String,
}

/// A tool definition accepted in a `/v1/chat/completions` request body.
/// Serializes to the OpenAI tool-definition wire shape (`{"type":"function",
/// "function": {...}}`) that tool-capable embedded chat templates (Hermes,
/// Llama 3.1, Qwen2.5, …) consume directly via the `tools` Jinja variable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ToolDef {
    /// Always `"function"` today; OpenAI's schema carries the discriminator.
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// A JSON Schema object describing the tool's parameters. Kept as an
    /// opaque `serde_json::Value` so we never have to model the full JSON-Schema
    /// surface — we round-trip the schema the client gave us.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// `tool_choice` controls whether tool calling happens at all for a request,
/// matching OpenAI's semantics exactly. The custom `Serialize` impl emits the
/// OpenAI wire forms — `"auto"` / `"none"` / `"required"` / a specific-tool
/// object — so the enum can be injected directly into the Jinja `tool_choice`
/// context variable and compared against the string literals real model
/// templates test for.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    /// `"auto"` (the default): the model decides whether to call a tool.
    Auto,
    /// `"none"`: suppress tool-calling behavior entirely for this request.
    None,
    /// `"required"`: the model must call a tool. (Per the spec, grammar
    /// enforcement of this is WI-TOOLS-6; in the MVP we surface it to the
    /// template and rely on the model's own convention.)
    Required,
    /// A specific named tool is forced.
    Specific {
        r#type: String,
        function: FunctionName,
    },
}

impl serde::Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        match self {
            ToolChoice::Auto => serializer.serialize_str("auto"),
            ToolChoice::None => serializer.serialize_str("none"),
            ToolChoice::Required => serializer.serialize_str("required"),
            ToolChoice::Specific { r#type, function } => {
                let mut m = serializer.serialize_map(Some(2))?;
                m.serialize_entry("type", r#type)?;
                m.serialize_entry("function", function)?;
                m.end()
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = serde_json::Value::deserialize(deserializer)?;
        match v {
            serde_json::Value::String(s) => match s.as_str() {
                "auto" => Ok(ToolChoice::Auto),
                "none" => Ok(ToolChoice::None),
                "required" => Ok(ToolChoice::Required),
                other => Err(serde::de::Error::unknown_variant(
                    other,
                    &["auto", "none", "required"],
                )),
            },
            serde_json::Value::Object(obj) => {
                let v = serde_json::Value::Object(obj);
                let r#type = v
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("type"))?
                    .to_string();
                let name = v
                    .pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("function.name"))?
                    .to_string();
                Ok(ToolChoice::Specific {
                    r#type,
                    function: FunctionName { name },
                })
            }
            other => Err(serde::de::Error::custom(format!(
                "expected string or object for tool_choice, got {}",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct FunctionName {
    pub name: String,
}

/// Strip unsupported Jinja block directives from a chat template string.
///
/// minijinja supports the standard block tags (`if`, `for`, `set`, `block`,
/// `extends`, `macro`) but not arbitrary custom tags that some model finetuners
/// embed in GGUF metadata (e.g. `{% generation %}…{% endgeneration %}`). These
/// cause `template_from_str` to return a parse error, which triggers a silent
/// fallback to the last message's content. This function removes the unsupported
/// outer tags while preserving their inner content so the remaining template is
/// valid minijinja.
pub fn sanitize_jinja_template(template: &str) -> String {
    // Match `{% tag … %}` and `{% endtag %}`. We strip tags whose outer name
    // minijinja doesn't recognise, keeping the text between opening and closing.
    // The recognised set is deliberately conservative — minijinja's built-in tags.
    let recognised: &[&str] = &[
        "if",
        "elif",
        "else",
        "endif",
        "for",
        "endfor",
        "endcase",
        "case",
        "set",
        "block",
        "endblock",
        "extends",
        "macro",
        "endmacro",
        "call",
        "endcall",
        "filter",
        "endfilter",
        "raw",
        "endraw",
        "with",
        "endwith",
        "trans",
        "endtrans",
        "spaceless",
        "endspaceless",
        "autoescape",
        "endautoescape",
        "do",
        "recursive",
        "endif",
    ];

    // Simple line-by-line pass: collect tag names and strip unrecognised
    // opening/closing tags while preserving their inner text.
    let mut result = String::with_capacity(template.len());
    let mut buf = template.chars().peekable();
    while let Some(ch) = buf.next() {
        if ch == '{' && buf.peek() == Some(&'%') {
            // Capture the full tag `{% ... %}`.
            let mut tag = String::from("{%");
            let mut inner = String::new();
            if buf.next() == Some('%') {
                // Read until closing `%}`.
                while let Some(c) = buf.next() {
                    tag.push(c);
                    inner.push(c);
                    if c == '%' && buf.peek() == Some(&'}') {
                        tag.push(buf.next().unwrap());
                        break;
                    }
                }
            }
            // Extract the directive name (first whitespace-delimited token
            // after stripping whitespace and Jinja whitespace-control markers
            // (`-`). Without this, `{%- set ... -%}` would extract `-` as the
            // tag name, which is not in the recognised list, causing all
            // whitespace-controlled tags to be silently stripped.
            let name = inner
                .trim()
                .trim_start_matches('-')
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_start_matches("end");
            if recognised.contains(&name) || name.is_empty() {
                result.push_str(&tag);
            }
            // Unrecognised tags are silently removed (content preserved).
            continue;
        }
        result.push(ch);
    }

    // ---- minijinja compatibility transforms ----
    //
    // Many HuggingFace Jinja templates use Python-dict methods that minijinja
    // does not implement. These cause runtime render errors (`unknown method:
    // map has no method named get/items`), which trigger a silent fallback to
    // the last message. We patch the two most common patterns:
    //
    // 1. `.get('key')` / `.get("key")` → `["key"]`
    //    minijinja returns Undefined for missing bracket keys (which is
    //    falsy in `{% if %}`), matching the Python `.get()` semantics for
    //    absent keys (which returns None, also falsy).
    //
    // 2. `.items()` → `| items`
    //    minijinja provides an `items` filter (pipe form) but not the `.items()`
    //    method. This is used by tool-call templates: `{% for k, v in d.items() %}`.

    // Transform `.get('key')` and `.get("key")` → `["key"]`.
    while let Some(pos) = result.find(".get('") {
        if let Some(end) = result[pos..].find("')") {
            let key = result[pos + 6..pos + end].to_string();
            result.replace_range(pos..pos + end + 2, &format!("[\"{key}\"]"));
        } else {
            break;
        }
    }
    while let Some(pos) = result.find(".get(\"") {
        if let Some(end) = result[pos..].find("\")") {
            let key = result[pos + 6..pos + end].to_string();
            result.replace_range(pos..pos + end + 2, &format!("[\"{key}\"]"));
        } else {
            break;
        }
    }

    // Transform `.startswith('prefix')` and `.startswith("prefix")` → slice comparison
    while let Some(pos) = result.find(".startswith('") {
        if let Some(end) = result[pos..].find("')") {
            let prefix = result[pos + 13..pos + end].to_string();
            let len = prefix.len();
            result.replace_range(pos..pos + end + 2, &format!("[0:{len}] == '{prefix}'"));
        } else {
            break;
        }
    }
    while let Some(pos) = result.find(".startswith(\"") {
        if let Some(end) = result[pos..].find("\")") {
            let prefix = result[pos + 13..pos + end].to_string();
            let len = prefix.len();
            result.replace_range(pos..pos + end + 2, &format!("[0:{len}] == \"{prefix}\""));
        } else {
            break;
        }
    }

    // Transform `.endswith('suffix')` and `.endswith("suffix")` → slice comparison.
    // `.endswith('Y')` → `[-N:] == 'Y'` where N = len('Y').
    while let Some(pos) = result.find(".endswith('") {
        if let Some(end) = result[pos..].find("')") {
            let suffix = result[pos + 11..pos + end].to_string();
            let len = suffix.len();
            result.replace_range(pos..pos + end + 2, &format!("[-{len}:] == '{suffix}'"));
        } else {
            break;
        }
    }
    while let Some(pos) = result.find(".endswith(\"") {
        if let Some(end) = result[pos..].find("\")") {
            let suffix = result[pos + 11..pos + end].to_string();
            let len = suffix.len();
            result.replace_range(pos..pos + end + 2, &format!("[-{len}:] == \"{suffix}\""));
        } else {
            break;
        }
    }

    // Transform `.items()` → `| items`.
    // Only matches the method-call form `.items()`, not the filter `| items`.
    let result = result.replace(".items()", " | items");

    // In Jinja, `+` fails on string + undefined. `~` is Jinja's string concatenation
    // operator which automatically coerces undefined variables to empty strings.
    let result = result.replace(" + ", " ~ ");

    result
}

/// Renders an OpenAI-style `messages` array through a model's Jinja chat
/// template, producing the final prompt string ready for tokenization.
///
/// Covers the common HF/GGUF template variable subset (`messages`,
/// `add_generation_prompt`, plus `bos_token`/`eos_token` when supplied). If a
/// specific model's template references an unsupplied variable, minijinja
/// surfaces the exact name — widen `ctx` as needed. Falls back to the raw
/// last-message content if the template fails to render.
///
/// `tools` (when `Some`) exposes the OpenAI tool-definition array to the
/// template's own Jinja logic under the standard `tools` variable name. The
/// caller passes the *already-shaped* `&[ToolDef]`; we do not attempt a
/// unified per-family templating shim — each tool-capable model's embedded
/// template was written by its finetuner to match its own tool-call output
/// convention, so per-family quirks are a parsing (WI-TOOLS-4) concern, not a
/// rendering concern.
pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
    tools: Option<&[ToolDef]>,
    tool_choice: Option<&ToolChoice>,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    // Most GGUF chat templates are self-contained; disable autoescaping and treat undefined variables gracefully.
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
    env.add_function(
        "raise_exception",
        |_msg: Option<&str>| -> std::result::Result<String, minijinja::Error> { Ok(String::new()) },
    );
    env.add_filter(
        "tojson",
        |v: minijinja::Value| -> std::result::Result<String, minijinja::Error> {
            serde_json::to_string(&v).map_err(|e| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
            })
        },
    );
    // P1-3.3: Some GGUF-embedded Jinja templates use block directives that
    // minijinja does not support (e.g. `{% generation %}…{% endgeneration %}`).
    // Strip those unsupported tags (keeping their inner content) so the
    // remaining template parses and renders correctly instead of failing
    // silently and falling back to the last message.
    let sanitized = sanitize_jinja_template(template);
    let tmpl = env
        .template_from_str(&sanitized)
        .map_err(|e| Error::Backend(format!("chat template parse error: {e}")))?;
    // minijinja's `context!` macro requires every referenced variable to be
    // named, so we build the context conditionally: `tools`/`tool_choice` are
    // only emitted when provided. A template that never references `tools`
    // simply ignores them; a template that does reference them expects the
    // caller to have supplied real definitions.
    let empty_str = "";
    let ctx = minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token,
        eos_token => eos_token,
        tools => tools,
        tool_choice => tool_choice,
        system_prompt => empty_str,
        system_message => empty_str,
        system => empty_str,
        extra_context => empty_str,
    };
    let rendered = tmpl
        .render(ctx)
        .map_err(|e| Error::Backend(format!("chat template render error: {e}")))?;
    // If the caller supplied tools but the rendered output made no use of
    // them (i.e. the model's template isn't tool-capable), surface a loud
    // diagnostic so operators understand why no tool calls will be produced,
    // rather than silently returning an ordinary completion.
    if let Some(ts) = tools {
        if !ts.is_empty() {
            // minijinja silently ignores unreferenced context variables, so we
            // detect "tool-capable template" structurally: a tool-aware template
            // references `tools` somewhere in its source text.
            if !template.contains("tools") {
                eprintln!(
                    "[grim-format] WARNING: {n} tool(s) supplied but the model's chat \
                     template does not reference a `tools` variable — this model was \
                     not fine-tuned for tool calling; tool calls will not be produced.",
                    n = ts.len()
                );
            }
        }
    }
    Ok(rendered)
}

/// Convenience: render `messages` through a tokenizer's embedded template when
/// present, otherwise return the last message's content (raw fallback). No
/// tools are exposed to the template.
pub fn render_messages_or_last(tokenizer: &GgufTokenizer, messages: &[ChatMessage]) -> String {
    render_messages_or_last_with_tools(tokenizer, messages, None, None)
}

/// Convenience: as [`render_messages_or_last`] but additionally exposes the
/// provided `tools` / `tool_choice` to the template's own Jinja logic. Use this
/// path for tool-calling requests; the tool-less overload keeps existing call
/// sites unchanged.
pub fn render_messages_or_last_with_tools(
    tokenizer: &GgufTokenizer,
    messages: &[ChatMessage],
    tools: Option<&[ToolDef]>,
    tool_choice: Option<&ToolChoice>,
) -> String {
    match &tokenizer.chat_template {
        Some(tpl) => render_chat_template(
            tpl,
            messages,
            true,
            tokenizer
                .bos_token_id
                .and_then(|id| tokenizer.tokens.get(id as usize))
                .map(|s| s.as_str())
                .unwrap_or(""),
            tokenizer
                .eos_token_id
                .and_then(|id| tokenizer.tokens.get(id as usize))
                .map(|s| s.as_str())
                .unwrap_or(""),
            tools,
            tool_choice,
        )
        .unwrap_or_else(|e| {
            eprintln!(
                "[grim-format] chat template render failed, falling back to last message: {e}"
            );
            messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default()
        }),
        None => messages
            .last()
            .map(|m| m.content.clone())
            .unwrap_or_default(),
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
    for b in 33..=126 {
        bs.push(b);
    }
    for b in 161..=172 {
        bs.push(b);
    }
    for b in 174..=255 {
        bs.push(b);
    }
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
    gpt2_byte_encoder()
        .into_iter()
        .map(|(k, v)| (v, k))
        .collect()
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

#[cfg(test)]
mod chat_template_tests {
    use super::*;

    #[test]
    fn renders_chatml_template_with_messages() {
        let tpl = "{{ bos_token }}{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}".to_string();
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are grim.".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: "Hi.".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];
        let rendered = render_chat_template(&tpl, &msgs, false, "", "", None, None)
            .expect("render must succeed");
        assert!(
            rendered.contains("<|im_start|>system"),
            "missing system role"
        );
        assert!(rendered.contains("You are grim."), "missing system content");
        assert!(rendered.contains("<|im_start|>user"), "missing user role");
        assert!(rendered.contains("Hi."), "missing user content");
        assert!(rendered.contains("<|im_end|>"), "missing im_end marker");
    }

    #[test]
    fn renders_single_user_turn() {
        let tpl = "{{'<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>'}}".to_string();
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "translate: hi".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let rendered = render_chat_template(&tpl, &msgs, false, "", "", None, None)
            .expect("render must succeed");
        assert_eq!(rendered, "<|im_start|>user\ntranslate: hi<|im_end|>");
    }

    #[test]
    fn unparseable_template_errors() {
        // A syntactically broken template should surface a parse error, not panic.
        assert!(
            render_chat_template("{% if %}", &[], false, "", "", None, None).is_err(),
            "malformed template must error"
        );
    }

    /// A tool-aware template (Hermes-2-Pro style) must receive the `tools`
    /// array and render the function definitions into the prompt.
    #[test]
    fn renders_tools_for_hermes_style_template() {
        // Simplified Hermes-2-Pro tool section: emits an XML-ish block listing
        // each tool's function name and reads the city property from the
        // parameters schema (minijinja has no `tojson` filter, so we drill
        // into the nested value directly instead).
        let tpl = "{% if tools %}<tools>{% for t in tools %}<tool>{{ t['function']['name'] }} {{ t['function']['parameters']['properties']['city']['type'] }}</tool>{% endfor %}</tools>{% endif %}{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}".to_string();
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "What's the weather?".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let tools = vec![ToolDef {
            r#type: "function".into(),
            function: FunctionDef {
                name: "get_weather".into(),
                description: Some("Get the current weather".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                })),
            },
        }];
        let rendered = render_chat_template(&tpl, &msgs, false, "", "", Some(&tools), None)
            .expect("render must succeed");
        assert!(rendered.contains("<tools>"), "missing tools block");
        assert!(rendered.contains("get_weather"), "missing function name");
        assert!(
            rendered.contains("string"),
            "parameters schema not rendered"
        );
    }

    /// `tool_choice == "none"` must suppress tool definitions in the prompt. A
    /// template that honors the directive omits the tools block.
    #[test]
    fn tool_choice_none_suppresses_in_template() {
        let tpl = "{% if tool_choice and tool_choice != 'none' %}<tools>{% for t in tools %}{{ t['function']['name'] }}{% endfor %}</tools>{% endif %}{{ messages[0]['content'] }}".to_string();
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let tools = vec![ToolDef {
            r#type: "function".into(),
            function: FunctionDef {
                name: "f".into(),
                description: None,
                parameters: None,
            },
        }];
        let rendered = render_chat_template(
            &tpl,
            &msgs,
            false,
            "",
            "",
            Some(&tools),
            Some(&ToolChoice::None),
        )
        .expect("render must succeed");
        assert!(!rendered.contains("<tools>"), "tools should be suppressed");
        assert_eq!(rendered, "hi");
    }

    /// A message carrying an assistant tool call must serialize `tool_calls`
    /// so a tool-aware template can render prior tool invocations.
    #[test]
    fn renders_assistant_tool_calls_in_message() {
        let tpl = "{% for m in messages %}{% if m['tool_calls'] %}{% for tc in m['tool_calls'] %}<call>{{ tc['name'] }}({{ tc['arguments'] }})</call>{% endfor %}{% else %}{{ m['role'] }}: {{ m['content'] }}{% endif %}\n{% endfor %}".to_string();
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: "Get weather for Paris".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "".into(),
                tool_calls: Some(vec![ToolCallMsg {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: "{\"city\":\"Paris\"}".into(),
                }]),
                tool_call_id: None,
                name: None,
            },
        ];
        let rendered = render_chat_template(&tpl, &msgs, false, "", "", None, None)
            .expect("render must succeed");
        assert!(
            rendered.contains("<call>get_weather({\"city\":\"Paris\"})</call>"),
            "assistant tool call not rendered; got: {rendered}"
        );
    }

    /// A tool-role message carrying a `tool_call_id` and `name` must round-trip
    /// through the template's normal message-rendering path.
    #[test]
    fn renders_tool_role_message() {
        let tpl =
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}".to_string();
        let msgs = vec![ChatMessage {
            role: "tool".into(),
            content: "72°F".into(),
            tool_calls: None,
            tool_call_id: Some("call_1".into()),
            name: Some("get_weather".into()),
        }];
        let rendered = render_chat_template(&tpl, &msgs, false, "", "", None, None)
            .expect("render must succeed");
        assert!(rendered.contains("tool: 72°F"), "tool message not rendered");
    }

    /// LFM2 / HuggingFace templates use Python-dict methods (`.get()`,
    /// `.items()`), whitespace-controlled tags (`{%- ... -%}`), and the `+`
    /// operator for string concatenation — all of which need sanitizer
    /// transforms to work under minijinja. This test exercises a template that
    /// combines all three patterns (mirrors the real LFM2 chat template).
    #[test]
    fn renders_lfm2_style_template_with_compat_transforms() {
        let tpl = r#"{{- bos_token -}}
{%- set ns = namespace(s="") -%}
{%- if messages[0]["role"] == "system" -%}
    {%- set ns.s = messages[0]["content"] -%}
{%- endif -%}
{%- if ns.s -%}
    {{- "<|im_start|>system\n" + ns.s + "<|im_end|>\n" -}}
{%- endif -%}
{%- for m in messages -%}
    {{- "<|im_start|>" + m["role"] + "\n" -}}
    {%- if m.get('tool_calls') -%}
        {{- m["tool_calls"] | tojson -}}
    {%- endif -%}
    {{- m["content"] + "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n" -}}
{%- endif -%}"#
            .to_string();
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "Hello!".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let rendered = render_chat_template(&tpl, &msgs, true, "<bos>", "<eos>", None, None)
            .expect("LFM2-style template must render without falling back");
        assert!(rendered.contains("<bos>"), "missing bos_token");
        assert!(
            rendered.contains("<|im_start|>user"),
            "missing user role marker"
        );
        assert!(rendered.contains("Hello!"), "missing user content");
        assert!(
            rendered.contains("<|im_start|>assistant"),
            "missing generation prompt"
        );
        // No system message was provided, so the system block should be absent.
        assert!(
            !rendered.contains("<|im_start|>system"),
            "system block should not appear without a system message"
        );
    }

    /// HuggingFace templates (e.g. MiniCPM5, Ternary-Bonsai) use Python string
    /// methods `.startswith()` and `.endswith()` which minijinja does not
    /// implement. The sanitizer must transform them into slice comparisons.
    #[test]
    fn renders_template_with_startswith_endswith() {
        let tpl = r#"{%- set s = "hello world" -%}
{%- if s.startswith("hello") -%}START{% endif -%}
{%- if s.endswith("world") -%}END{% endif -%}
{%- if not s.startswith("xyz") -%}NOTXYZ{% endif -%}"#
            .to_string();
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let rendered = render_chat_template(&tpl, &msgs, false, "", "", None, None)
            .expect("startswith/endswith template must render");
        assert!(rendered.contains("START"), "startswith match failed");
        assert!(rendered.contains("END"), "endswith match failed");
        assert!(rendered.contains("NOTXYZ"), "startswith non-match failed");
    }
}
