use grim_format::tokenizer::GgufTokenizer;
use grim_tensor::{DType, Tensor};
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
        if self.token_buffer.len() < needed {
            return Err(grim_tensor::error::Error::Backend("dataloader exhausted".into()));
        }
        let flat: Vec<u32> = self.token_buffer.drain(..needed).collect();
        let input_ids = Tensor::from_slice_u32(&flat, &[self.batch_size, self.seq_len])?;
        let labels = Self::build_labels(&flat, self.batch_size, self.seq_len, self.tokenizer.pad_token_id());
        Ok((input_ids, labels))
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
                let tokens = self.tokenizer.encode(text)?;
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
        Tensor::from_slice_u32(&labels_flat, &[batch_size, seq_len]).expect("labels shape matches input")
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
        let tokenizer = GgufTokenizer::default();
        let mut loader = JsonlBatchIterator::new(
            path.to_str().unwrap(),
            tokenizer,
            64,
            2,
        ).expect("datloader should construct");
        let (inputs, labels) = loader.next_batch().expect("first batch");
        assert_eq!(inputs.shape(), &[2, 64]);
        assert_eq!(labels.shape(), &[2, 64]);
    }
}