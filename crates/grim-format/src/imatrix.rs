//! llama.cpp imatrix GGUF consumption — turns a collected importance matrix
//! into per-tensor importance scores the oxidizer can quantize against.
//!
//! Layout (verified against a real Xing4.0 imatrix dump): one GGUF metadata
//! cluster (`general.type = "imatrix"`, `imatrix.datasets`,
//! `imatrix.chunk_count`, `imatrix.chunk_size`) plus, per measured tensor, an
//! `<name>.in_sum2` F32 array holding the per-column accumulated sum of
//! squared activations and an `<name>.counts` record holding how many chunks
//! each column accumulated. Depending on the writer these live either as GGUF
//! **tensor entries** or as metadata KV arrays — both are handled here, and a
//! file carrying neither fails loud instead of yielding empty scores.
//!
//! Column importance = `in_sum2[col] / counts[col]` (the mean square of that
//! column's activation across chunks); a tensor's score is the mean over its
//! columns. Scores keep the raw activation scale: consumers weight them
//! relatively within one model, and every score in one call comes from the
//! same source file, so scales cannot mix.

use grim_tensor::error::{Error, Result};
use grim_tensor::provider::TensorProvider;

use crate::gguf::{GgufDType, GgufValue};
use crate::tprov::GgufProvider;

/// Per-tensor importance parsed out of an imatrix GGUF.
#[derive(Debug, Clone)]
pub struct ImatrixScores {
    /// `(tensor_name, importance)` sorted by tensor name. Names are the
    /// imatrix's base names with the `.in_sum2` suffix stripped, which match
    /// the source checkpoint's tensor names (e.g. `blk.0.attn_q_a.weight`).
    pub scores: Vec<(String, f32)>,
    /// Chunks the collection ran over, when the file declares it.
    pub chunk_count: Option<u32>,
    /// Tokens per chunk, when the file declares it.
    pub chunk_size: Option<u32>,
}

/// Parse an imatrix GGUF from disk.
pub fn read_imatrix_scores(path: &str) -> Result<ImatrixScores> {
    let provider = GgufProvider::open(path)?;
    read_imatrix_from_provider(&provider, path)
}

fn read_imatrix_from_provider(provider: &GgufProvider, path: &str) -> Result<ImatrixScores> {
    let chunk_count = metadata_u32(provider, "imatrix.chunk_count");
    let chunk_size = metadata_u32(provider, "imatrix.chunk_size");

    // The in_sum2 records are the structural marker of an imatrix file; where
    // they live decides how every lookup below proceeds.
    let mut tensor_bases: Vec<String> = provider
        .tensors()
        .keys()
        .filter(|n| n.ends_with(".in_sum2"))
        .map(|n| n.strip_suffix(".in_sum2").unwrap_or(n).to_string())
        .collect();
    let mut metadata_layout = false;
    if tensor_bases.is_empty() {
        tensor_bases = provider
            .metadata_map()
            .keys()
            .filter(|k| k.ends_with(".in_sum2"))
            .map(|k| k.strip_suffix(".in_sum2").unwrap_or(k).to_string())
            .collect();
        metadata_layout = true;
    }
    if tensor_bases.is_empty() {
        return Err(Error::Backend(format!(
            "'{path}' carries no .in_sum2 entries in either its tensor table or \
             metadata — it is not an imatrix GGUF"
        )));
    }
    tensor_bases.sort();

    let mut scores = Vec::with_capacity(tensor_bases.len());
    for base in &tensor_bases {
        let values = if metadata_layout {
            let v = provider
                .metadata(&format!("{base}.in_sum2"))
                .ok_or_else(|| {
                    Error::Backend(format!("imatrix '{path}': '{base}.in_sum2' vanished"))
                })?;
            gguf_value_f32_vec(v).ok_or_else(|| {
                Error::Backend(format!(
                    "imatrix '{path}': '{base}.in_sum2' metadata is not a numeric array"
                ))
            })?
        } else {
            read_f32_tensor(provider, &format!("{base}.in_sum2"))?
        };
        if values.is_empty() {
            return Err(Error::Backend(format!(
                "imatrix '{path}': '{base}.in_sum2' is empty"
            )));
        }
        let counts = read_counts(provider, base, values.len())?;
        // Mean over columns of (sum2 / count); columns that never saw a chunk
        // contribute nothing rather than poisoning the mean with a division
        // by zero.
        let mut acc = 0.0f32;
        let mut used = 0usize;
        for (s2, c) in values.iter().zip(counts.iter()) {
            if *c > 0.0 {
                acc += s2 / c;
                used += 1;
            }
        }
        let score = if used == 0 {
            0.0
        } else {
            acc / used as f32
        };
        scores.push((base.clone(), score));
    }

    Ok(ImatrixScores {
        scores,
        chunk_count,
        chunk_size,
    })
}

/// `<base>.counts` — a metadata KV (scalar or array) on some writers, a GGUF
/// tensor entry (I32 per llama.cpp, F32/I64 tolerated) on others.
fn read_counts(provider: &GgufProvider, base: &str, n_values: usize) -> Result<Vec<f32>> {
    let counts_key = format!("{base}.counts");
    if let Some(v) = provider.metadata(&counts_key) {
        if let Some(mut counts) = gguf_value_f32_vec(v) {
            align_counts(&mut counts, &counts_key, n_values)?;
            return Ok(counts);
        }
        return Err(Error::Backend(format!(
            "imatrix: '{counts_key}' metadata is not numeric"
        )));
    }
    if let Some(info) = provider.tensors().get(&counts_key) {
        let raw = provider.get(&counts_key)?;
        let counts: Vec<f32> = match info.dtype {
            GgufDType::I32 => raw
                .bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect(),
            GgufDType::I64 => raw
                .bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect(),
            GgufDType::F32 => raw
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            other => {
                return Err(Error::Backend(format!(
                    "imatrix: '{counts_key}' tensor has unsupported dtype {other:?}"
                )))
            }
        };
        let mut counts = counts;
        align_counts(&mut counts, &counts_key, n_values)?;
        return Ok(counts);
    }
    // A writer that recorded no counts measured one chunk per column by
    // definition of `in_sum2` — treat every count as 1 rather than failing.
    Ok(vec![1.0; n_values])
}

fn align_counts(counts: &mut Vec<f32>, key: &str, n_values: usize) -> Result<()> {
    if counts.len() == n_values {
        return Ok(());
    }
    if counts.len() == 1 {
        counts.resize(n_values, counts[0]);
        return Ok(());
    }
    Err(Error::Backend(format!(
        "imatrix: '{key}' length {} does not match its .in_sum2 length {n_values}",
        counts.len()
    )))
}

fn read_f32_tensor(provider: &GgufProvider, name: &str) -> Result<Vec<f32>> {
    let info = provider.tensors().get(name).ok_or_else(|| {
        Error::Backend(format!("imatrix: tensor '{name}' missing from the file"))
    })?;
    if info.dtype != GgufDType::F32 {
        return Err(Error::Backend(format!(
            "imatrix: '{name}' has dtype {:?}, expected F32",
            info.dtype
        )));
    }
    let raw = provider.get(name)?;
    Ok(raw
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// Numeric GGUF metadata value -> f32 vector (scalar becomes a 1-element vec).
fn gguf_value_f32_vec(v: &GgufValue) -> Option<Vec<f32>> {
    let one = |f: f32| Some(vec![f]);
    match v {
        GgufValue::Float32(x) => one(*x),
        GgufValue::Float64(x) => one(*x as f32),
        GgufValue::Uint8(x) => one(*x as f32),
        GgufValue::Int8(x) => one(*x as f32),
        GgufValue::Uint16(x) => one(*x as f32),
        GgufValue::Int16(x) => one(*x as f32),
        GgufValue::Uint32(x) => one(*x as f32),
        GgufValue::Int32(x) => one(*x as f32),
        GgufValue::Uint64(x) => one(*x as f32),
        GgufValue::Int64(x) => one(*x as f32),
        GgufValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(match item {
                    GgufValue::Float32(x) => *x,
                    GgufValue::Float64(x) => *x as f32,
                    GgufValue::Uint32(x) => *x as f32,
                    GgufValue::Int32(x) => *x as f32,
                    GgufValue::Uint64(x) => *x as f32,
                    GgufValue::Int64(x) => *x as f32,
                    GgufValue::Uint16(x) => *x as f32,
                    GgufValue::Int16(x) => *x as f32,
                    GgufValue::Uint8(x) => *x as f32,
                    GgufValue::Int8(x) => *x as f32,
                    _ => return None,
                });
            }
            Some(out)
        }
        _ => None,
    }
}

fn metadata_u32(provider: &GgufProvider, key: &str) -> Option<u32> {
    match provider.metadata(key)? {
        GgufValue::Uint32(v) => Some(*v),
        GgufValue::Int32(v) => (*v >= 0).then_some(*v as u32),
        GgufValue::Uint64(v) => u32::try_from(*v).ok(),
        GgufValue::Int64(v) => u32::try_from((*v).max(0)).ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Hand-rolled GGUF v3 writer covering exactly what the synthetic tests
    /// need: u32/u64/f32 metadata scalars, f32/i32 metadata arrays, strings,
    /// and 1-D F32/I32 tensor entries with 32-byte-aligned data.
    struct GgufBuilder {
        metadata: Vec<(String, GgufValue)>,
        tensors: Vec<(String, GgufDType, Vec<u64>, Vec<u8>)>,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self {
                metadata: Vec::new(),
                tensors: Vec::new(),
            }
        }

        fn kv(mut self, k: &str, v: GgufValue) -> Self {
            self.metadata.push((k.into(), v));
            self
        }

        fn tensor(mut self, name: &str, dtype: GgufDType, dims: &[u64], bytes: Vec<u8>) -> Self {
            self.tensors.push((name.into(), dtype, dims.to_vec(), bytes));
            self
        }

        fn build(self) -> Vec<u8> {
            fn put_str(b: &mut Vec<u8>, s: &str) {
                b.extend_from_slice(&(s.len() as u64).to_le_bytes());
                b.extend_from_slice(s.as_bytes());
            }
            fn put_val(b: &mut Vec<u8>, v: &GgufValue) {
                match v {
                    GgufValue::Uint32(x) => {
                        b.extend_from_slice(&4u32.to_le_bytes());
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                    GgufValue::Uint64(x) => {
                        b.extend_from_slice(&10u32.to_le_bytes());
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                    GgufValue::Float32(x) => {
                        b.extend_from_slice(&6u32.to_le_bytes());
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                    GgufValue::Array(items) => {
                        let tag = match items.first() {
                            Some(GgufValue::Float32(_)) => 6u32,
                            _ => 4u32,
                        };
                        b.extend_from_slice(&9u32.to_le_bytes());
                        b.extend_from_slice(&tag.to_le_bytes());
                        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                        // GGUF arrays carry the element tag ONCE; elements
                        // are raw payloads (a repeated tag per element is
                        // what every real writer omits and read_gguf_value
                        // does not expect).
                        for item in items {
                            match item {
                                GgufValue::Float32(x) => b.extend_from_slice(&x.to_le_bytes()),
                                GgufValue::Uint32(x) => b.extend_from_slice(&x.to_le_bytes()),
                                other => panic!("test builder lacks raw elem write for {other:?}"),
                            }
                        }
                    }
                    GgufValue::String(s) => {
                        b.extend_from_slice(&8u32.to_le_bytes());
                        put_str(b, s);
                    }
                    other => panic!("test builder lacks tag for {other:?}"),
                }
            }

            let mut out = Vec::new();
            out.extend_from_slice(&crate::gguf::GGUF_MAGIC.to_le_bytes());
            out.extend_from_slice(&crate::gguf::GGUF_VERSION.to_le_bytes());
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
            for (k, v) in &self.metadata {
                put_str(&mut out, k);
                put_val(&mut out, v);
            }
            // Tensor infos: offsets are relative to the data region start,
            // which itself begins 32-aligned after the header (computed AFTER
            // the infos are written — resize below would truncate otherwise).
            let mut infos = Vec::with_capacity(self.tensors.len());
            let mut running = 0u64;
            for (name, dtype, dims, bytes) in &self.tensors {
                let offset = running;
                running = (running + bytes.len() as u64).div_ceil(32) * 32;
                put_str(&mut out, name);
                out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&(*dtype as u32).to_le_bytes());
                out.extend_from_slice(&offset.to_le_bytes());
                infos.push(bytes);
            }
            // Pad to the aligned data region start, then the payloads.
            let data_region_start = out.len().div_ceil(32) * 32;
            out.resize(data_region_start, 0);
            for bytes in &infos {
                out.extend_from_slice(bytes);
                while out.len() % 32 != 0 {
                    out.push(0);
                }
            }
            out
        }
    }

    fn write_temp(path: &std::path::Path, bytes: Vec<u8>) -> String {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&bytes).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn tensor_layout_in_sum2_and_counts_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(
            &dir.path().join("im.tensor.gguf"),
            GgufBuilder::new()
                .kv("general.type", GgufValue::String("imatrix".into()))
                .kv("imatrix.chunk_count", GgufValue::Uint32(7))
                .kv("imatrix.chunk_size", GgufValue::Uint32(512))
                .tensor(
                    "blk.0.attn_q.weight.in_sum2",
                    GgufDType::F32,
                    &[4],
                    [1.0f32, 3.0, 0.5, 2.0]
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect(),
                )
                .tensor(
                    "blk.0.attn_q.weight.counts",
                    GgufDType::I32,
                    &[4],
                    [2i32, 2, 1, 1].iter().flat_map(|f| f.to_le_bytes()).collect(),
                )
                .build(),
        );
        let im = read_imatrix_scores(&path).unwrap();
        assert_eq!(im.chunk_count, Some(7));
        assert_eq!(im.chunk_size, Some(512));
        assert_eq!(im.scores.len(), 1);
        assert_eq!(im.scores[0].0, "blk.0.attn_q.weight");
        // mean([1/2, 3/2, 1/2, 2/1]) = mean([0.5, 1.5, 0.5, 2.0]) = 1.125
        assert!((im.scores[0].1 - 1.125).abs() < 1e-6, "{}", im.scores[0].1);
    }

    #[test]
    fn metadata_layout_scalar_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(
            &dir.path().join("im.meta.gguf"),
            GgufBuilder::new()
                .kv("general.type", GgufValue::String("imatrix".into()))
                .kv(
                    "blk.3.ffn_up.weight.in_sum2",
                    GgufValue::Array(vec![
                        GgufValue::Float32(1.0),
                        GgufValue::Float32(3.0),
                        GgufValue::Float32(0.5),
                        GgufValue::Float32(2.0),
                    ]),
                )
                .kv("blk.3.ffn_up.weight.counts", GgufValue::Uint32(2))
                .build(),
        );
        let im = read_imatrix_scores(&path).unwrap();
        assert_eq!(im.scores.len(), 1);
        // mean([1,3,0.5,2]) / 2 = 1.625 / 2 = 0.8125
        assert!((im.scores[0].1 - 0.8125).abs() < 1e-6, "{}", im.scores[0].1);
    }

    #[test]
    fn zero_count_columns_do_not_poison_the_mean() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(
            &dir.path().join("im.zero.gguf"),
            GgufBuilder::new()
                .tensor(
                    "blk.0.w.in_sum2",
                    GgufDType::F32,
                    &[3],
                    [4.0f32, 100.0, 8.0]
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect(),
                )
                .tensor(
                    "blk.0.w.counts",
                    GgufDType::I32,
                    &[3],
                    [2i32, 0, 4].iter().flat_map(|f| f.to_le_bytes()).collect(),
                )
                .build(),
        );
        let im = read_imatrix_scores(&path).unwrap();
        // dead column skipped: mean([4/2, 8/4]) = mean([2, 2]) = 2
        assert!((im.scores[0].1 - 2.0).abs() < 1e-6, "{}", im.scores[0].1);
    }

    #[test]
    fn counts_length_mismatch_fails_loud() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(
            &dir.path().join("im.mismatch.gguf"),
            GgufBuilder::new()
                .tensor(
                    "blk.0.w.in_sum2",
                    GgufDType::F32,
                    &[4],
                    [1.0f32, 1.0, 1.0, 1.0]
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect(),
                )
                .tensor(
                    "blk.0.w.counts",
                    GgufDType::I32,
                    &[2],
                    [1i32, 1].iter().flat_map(|f| f.to_le_bytes()).collect(),
                )
                .build(),
        );
        let err = read_imatrix_scores(&path).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("blk.0.w.counts"), "{msg}");
        assert!(msg.contains("length"), "{msg}");
    }

}
