use grim_format::tokenizer::GgufTokenizer;
use grim_tensor::{Shape, Tensor};
use grim_tensor::error::Result;
use std::io::{BufRead, BufReader};
use std::fs::File;

/// Reads a `.jsonl` file where each line is `{"text": "..."}`,
/// tokenizes each line, packs tokens into fixed-length sequences,
/// and yields `(input_ids, labels)` tensor pairs.
///
/// Labels are input_ids shifted left by 1 position (next-token prediction).
/// The last token in each sequence is set to the pad token ID.
/// Shorter sequences are padded with the pad token ID.
pub struct JsonlBatchIterator {
    token_buffer: Vec<u32>,
    seq_len: usize,
    batch_size: usize,
    tokenizer: GgufTokenizer,
    reader: BufReader<File>,
    exhausted: bool,
}

impl JsonlBatchIterator {
    pub fn new(
        path: &str,
        tokenizer: GgufTokenizer,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            grim_tensor::error::Error::Backend(format!("failed to open dataloader path {path:?}: {e}"))
        })?;
        Ok(Self {
            token_buffer: Vec::new(),
            seq_len,
            batch_size,
            tokenizer,
            reader: BufReader::new(file),
            exhausted: false,
        })
    }

    /// Returns the next batch as `(input_ids, labels)` tensors of shape
    /// `[batch_size, seq_len]`. Returns `Err` when the file is exhausted.
    pub fn next_batch(&mut self) -> Result<(Tensor, Tensor)> {
        let needed = self.batch_size * self.seq_len;
        while self.token_buffer.len() < needed && !self.exhausted {
            self.fill_buffer()?;
        }
        if self.token_buffer.is_empty() {
            return Err(grim_tensor::error::Error::Backend("dataloader exhausted".into()));
        }
        let pad_id = self.tokenizer.pad_token_id();
        while self.token_buffer.len() < needed {
            self.token_buffer.push(pad_id);
        }
        let flat: Vec<u32> = self.token_buffer.drain(..needed).collect();
        let data_f32: Vec<f32> = flat.iter().map(|&x| x as f32).collect();
        let input_ids = grim_backend_cpu::cpu_tensor(data_f32, Shape::from_slice(&[self.batch_size, self.seq_len]));
        let labels = Self::build_labels(&flat, self.batch_size, self.seq_len, pad_id);
        Ok((input_ids, labels))
    }
    
    /// Load the next preference optimization batch (chosen/rejected pairs).
    /// Returns `Ok(Some((chosen, rejected, is_preferred)))` when a batch is ready,
    /// `Ok(None)` when the file is exhausted, or `Err` on I/O errors.
    pub fn next_preference_batch(&mut self) -> Result<Option<(Tensor, Tensor, Vec<bool>)>> {
        let needed = self.batch_size * self.seq_len;
        while self.token_buffer.len() < needed * 2 && !self.exhausted {
            self.fill_preference_buffer()?;
        }
        
        if self.token_buffer.is_empty() {
            return Ok(None);
        }
        
        let pad_id = self.tokenizer.pad_token_id();
        let total_needed = needed * 2;
        while self.token_buffer.len() < total_needed {
            self.token_buffer.push(pad_id);
        }
        
        let flat: Vec<u32> = self.token_buffer.drain(..total_needed).collect();
        let chosen_flat: Vec<u32> = flat[0..needed].to_vec();
        let rejected_flat: Vec<u32> = flat[needed..total_needed].to_vec();
        
        let chosen_f32: Vec<f32> = chosen_flat.iter().map(|&x| x as f32).collect();
        let rejected_f32: Vec<f32> = rejected_flat.iter().map(|&x| x as f32).collect();
        
        let chosen_ids = grim_backend_cpu::cpu_tensor(chosen_f32, Shape::from_slice(&[self.batch_size, self.seq_len]));
        let rejected_ids = grim_backend_cpu::cpu_tensor(rejected_f32, Shape::from_slice(&[self.batch_size, self.seq_len]));
        
        // For now, all are "preferred" by default
        let is_preferred = vec![true; self.batch_size];
        
        Ok(Some((chosen_ids, rejected_ids, is_preferred)))
    }
    
    fn fill_preference_buffer(&mut self) -> Result<()> {
        // Load preference pairs from JSONL file
        // Format: {"prompt": "...", "chosen": "...", "rejected": "...", "preferred": true/false}
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => {
                self.exhausted = true;
            }
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return Ok(());
                }
                let v: serde_json::Value = serde_json::from_str(trimmed)
                    .map_err(|e| grim_tensor::error::Error::Backend(e.to_string()))?;
                
                // Load chosen and rejected texts
                let chosen_text = v["chosen"].as_str().unwrap_or("");
                let rejected_text = v["rejected"].as_str().unwrap_or("");
                
                let chosen_tokens = self.tokenizer.encode(chosen_text);
                let rejected_tokens = self.tokenizer.encode(rejected_text);
                
                self.token_buffer.extend(chosen_tokens);
                self.token_buffer.extend(rejected_tokens);
            }
            Err(e) => {
                return Err(grim_tensor::error::Error::Backend(e.to_string()));
            }
        }
        Ok(())
    }

    fn fill_buffer(&mut self) -> Result<()> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => {
                self.exhausted = true;
            }
            Ok(_) => {
                let v: serde_json::Value = serde_json::from_str(line.trim())
                    .map_err(|e| grim_tensor::error::Error::Backend(e.to_string()))?;
                let text = v["text"].as_str().unwrap_or("");
                let tokens = self.tokenizer.encode(text);
                self.token_buffer.extend(tokens);
            }
            Err(e) => {
                return Err(grim_tensor::error::Error::Backend(e.to_string()));
            }
        }
        Ok(())
    }

    fn build_labels(
        flat: &[u32],
        batch_size: usize,
        seq_len: usize,
        pad_token_id: u32,
    ) -> Tensor {
        let mut labels_flat = flat.to_vec();
        for row in 0..batch_size {
            let start = row * seq_len;
            for col in 0..(seq_len - 1) {
                labels_flat[start + col] = flat[start + col + 1];
            }
            labels_flat[start + seq_len - 1] = pad_token_id;
        }
        let data_f32: Vec<f32> = labels_flat.iter().map(|&x| x as f32).collect();
        grim_backend_cpu::cpu_tensor(data_f32, Shape::from_slice(&[batch_size, seq_len]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_batch_iterator_returns_correct_shapes() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("toy.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        for _ in 0..10 {
            writeln!(f, r#"{{"text": "hello world"}}"#).unwrap();
        }
        let mut tokenizer = GgufTokenizer::default();
        tokenizer.tokens = vec!["hello".into(), "world".into()];
        tokenizer.token_to_id.insert("hello".into(), 1);
        tokenizer.token_to_id.insert("world".into(), 2);
        let mut loader = JsonlBatchIterator::new(
            path.to_str().unwrap(),
            tokenizer,
            64,
            2,
        ).expect("datloader should construct");
        let (inputs, labels) = loader.next_batch().expect("first batch");
        assert_eq!(inputs.shape().dims(), &[2, 64]);
        assert_eq!(labels.shape().dims(), &[2, 64]);
    }
}