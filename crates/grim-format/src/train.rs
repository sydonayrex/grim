//! Training-state sidecar format: `model.grim.train` (WI-R6).
//!
//! The V3 `.grim` wire format is weight/inference-only; it has no slot for
//! optimizer state, LoRA/DoRA adapters, or SERQ low-rank error matrices.
//! Research shows consumer fine-tune is viable (LoRA Edge 26× peak-mem cut
//! on Llama-3.2-3B; DoRA ~24% train-mem reduction vs LoRA; SERQ saliency
//! low-rank error for 4-bit GEMM). This module defines a **companion
//! sidecar** — `model.grim.train` written next to `model.grim` — so the
//! inference reader is never touched and legacy files ignore it.
//!
//! Layout (little-endian):
//!
//! ```text
//! [ magic: 8 bytes "GRIMTRN\x01" ]
//! [ header_len: u32 LE ][ header JSON ]
//! [ per-blob: name_len:u16 | name | ndim:u8 | dims:u32×ndim | nbytes:u64 | bytes ]
//! ```
//!
//! Each blob (adapter A/B, optimizer m/v, error matrix, …) is a self-describing
//! byte region. The header JSON records the `fp_format` numeric descriptor and
//! which named blobs belong to which logical slot, so a resumed fine-tune can
//! reconstruct step-N state bit-for-bit.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use serde_json::Value;

use grim_tensor::error::{Error, Result};

/// Magic bytes for the `.grim.train` sidecar.
pub const TRAIN_MAGIC: [u8; 8] = [0x47, 0x52, 0x49, 0x4d, 0x54, 0x52, 0x4e, 0x01]; // "GRIMTRN\x01"

/// FP format descriptor for training-state tensors (WI-R6).
///
/// The numeric set RDNA3/4 training targets (Dual-Precision MAC paper:
/// FP8/FP4 rising in inference, FP16/FP32 still dominate training).
///
/// `Bf16` and `Fp16` encode the param blob bytes in that format while
/// optimizer moments remain f32 (sidecar header `dtypes` map disambiguates).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrainFpFormat {
    Fp16 = 0,
    Fp32 = 1,
    Fp8E4M3 = 2,
    Fp8E5M2 = 3,
    Fp4 = 4,
    Bf16 = 5,
    Fp16Param = 6,
}

impl TrainFpFormat {
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Fp16),
            1 => Some(Self::Fp32),
            2 => Some(Self::Fp8E4M3),
            3 => Some(Self::Fp8E5M2),
            4 => Some(Self::Fp4),
            5 => Some(Self::Bf16),
            6 => Some(Self::Fp16Param),
            _ => None,
        }
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
    /// Whether this format encodes 2-byte elements (param blobs).
    pub fn is_half(self) -> bool {
        matches!(self, Self::Bf16 | Self::Fp16 | Self::Fp16Param)
    }
}

/// Convert raw bytes to f32 slice based on the fp_format.
fn bytes_to_f32s(data: &[u8], fmt: TrainFpFormat) -> Option<Vec<f32>> {
    match fmt {
        TrainFpFormat::Fp32 => {
            if data.len() % 4 != 0 {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        }
        TrainFpFormat::Bf16 => {
            if data.len() % 2 != 0 {
                return None;
            }
            Some(
                data.chunks_exact(2)
                    .map(|c| {
                        let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
                        f32::from_bits(bits << 16)
                    })
                    .collect(),
            )
        }
        TrainFpFormat::Fp16 | TrainFpFormat::Fp16Param => {
            if data.len() % 2 != 0 {
                return None;
            }
            Some(data.chunks_exact(2).map(|c| f16_to_f32_le(c)).collect())
        }
        TrainFpFormat::Fp8E4M3 | TrainFpFormat::Fp8E5M2 | TrainFpFormat::Fp4 => None,
    }
}

/// Round-to-nearest-even F32 -> little-endian BF16 bytes.
pub fn f32_to_bf16_bytes(v: f32) -> [u8; 2] {
    let bits = f32::to_bits(v);
    let discard = (bits & 0xFFFF) as u32;
    let mut bf16 = (bits >> 16) as u16;
    if discard > 0x8000 || (discard == 0x8000 && (bf16 & 1) == 1) {
        bf16 = bf16.wrapping_add(1);
    }
    bf16.to_le_bytes()
}

/// F32 -> little-endian IEEE F16 bytes (subnormals, Inf/NaN, overflow handled).
pub fn f32_to_f16_bytes(v: f32) -> [u8; 2] {
    let bits = f32::to_bits(v);
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as u32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf / NaN
        let result = sign << 15 | 0x7C00 | ((mant >> 13) as u16);
        return result.to_le_bytes();
    }

    if exp == 0 {
        // Zero / subnormal f32 -> zero f16
        return (sign << 15).to_le_bytes();
    }

    let new_exp = exp as i32 - 127 + 15;

    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15).to_le_bytes();
        }
        // Subnormal f16
        let mant_shift = (-new_exp + 1) as u32;
        let new_mant = (mant | 0x800000) >> mant_shift;
        let result = sign << 15 | (new_mant >> 10) as u16;
        return result.to_le_bytes();
    }

    if new_exp >= 31 {
        return (sign << 15 | 0x7C00).to_le_bytes();
    }

    let new_mant = mant >> 13;
    let result = (sign << 15) | ((new_exp as u16) << 10) | (new_mant as u16);
    result.to_le_bytes()
}

/// Little-endian F16 (IEEE half) byte pair to F32.
fn f16_to_f32_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        let value = (mant as f32) * 2f32.powi(-24);
        if sign != 0 { -value } else { value }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Encode a slice of f32 values into raw little-endian bytes in `fmt`.
///
/// Fp32 emits 4-byte words; Bf16/Fp16 emit 2-byte words. FP8/FP4 formats are
/// not encodable here and fall back to Fp32 (callers must not request them).
pub fn encode_f32s_as(vals: &[f32], fmt: TrainFpFormat) -> Vec<u8> {
    match fmt {
        TrainFpFormat::Bf16 => vals.iter().flat_map(|v| f32_to_bf16_bytes(*v)).collect(),
        TrainFpFormat::Fp16 | TrainFpFormat::Fp16Param => {
            vals.iter().flat_map(|v| f32_to_f16_bytes(*v)).collect()
        }
        _ => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// Decode raw little-endian bytes in `fmt` back to f32 values.
pub fn decode_f32s_from(data: &[u8], fmt: TrainFpFormat) -> Option<Vec<f32>> {
    bytes_to_f32s(data, fmt)
}

/// Map a tensor `DType` to the matching sidecar `TrainFpFormat`.
pub fn train_format_for_dtype(dt: &grim_tensor::DType) -> TrainFpFormat {
    use grim_tensor::ArithType;
    match dt.arith {
        ArithType::BF16 => TrainFpFormat::Bf16,
        ArithType::F16 => TrainFpFormat::Fp16Param,
        _ => TrainFpFormat::Fp32,
    }
}

/// One named training-state blob (adapter weight, optimizer moment, error matrix).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainBlob {
    pub name: String,
    pub shape: Vec<usize>,
    /// Raw little-endian bytes of the blob (caller owns the numeric encoding;
    /// typically f32/f16/quantized per `TrainState::fp_format`).
    pub data: Vec<u8>,
}

impl TrainBlob {
    fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        let name_bytes = self.name.as_bytes();
        w.write_all(&(name_bytes.len() as u16).to_le_bytes())
            .map_err(|e| Error::Backend(format!("train blob name write failed: {e}")))?;
        w.write_all(name_bytes)
            .map_err(|e| Error::Backend(format!("train blob name write failed: {e}")))?;
        w.write_all(&(self.shape.len() as u8).to_le_bytes())
            .map_err(|e| Error::Backend(format!("train blob shape write failed: {e}")))?;
        for dim in &self.shape {
            w.write_all(&(*dim as u32).to_le_bytes())
                .map_err(|e| Error::Backend(format!("train blob dim write failed: {e}")))?;
        }
        w.write_all(&(self.data.len() as u64).to_le_bytes())
            .map_err(|e| Error::Backend(format!("train blob len write failed: {e}")))?;
        w.write_all(&self.data)
            .map_err(|e| Error::Backend(format!("train blob data write failed: {e}")))?;
        Ok(())
    }

    fn read<R: Read>(r: &mut R) -> Result<Self> {
        let mut name_len = [0u8; 2];
        r.read_exact(&mut name_len)
            .map_err(|e| Error::Backend(format!("train blob name read failed: {e}")))?;
        let name_len = u16::from_le_bytes(name_len) as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes)
            .map_err(|e| Error::Backend(format!("train blob name read failed: {e}")))?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| Error::Backend(format!("invalid UTF-8 in train blob name: {e}")))?;

        let mut ndim_b = [0u8; 1];
        r.read_exact(&mut ndim_b)
            .map_err(|e| Error::Backend(format!("train blob shape read failed: {e}")))?;
        let ndim = ndim_b[0] as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut dim_b = [0u8; 4];
            r.read_exact(&mut dim_b)
                .map_err(|e| Error::Backend(format!("train blob dim read failed: {e}")))?;
            shape.push(u32::from_le_bytes(dim_b) as usize);
        }

        let mut len_b = [0u8; 8];
        r.read_exact(&mut len_b)
            .map_err(|e| Error::Backend(format!("train blob len read failed: {e}")))?;
        let len = u64::from_le_bytes(len_b) as usize;
        // Cap blob length to a sane maximum (1 GiB) to avoid allocating up to 4 GB
        // from an untrusted length field. [P1-28 fix.]
        const MAX_TRAIN_BLOB: usize = 1 << 30;
        if len > MAX_TRAIN_BLOB {
            return Err(Error::Backend(format!(
                "train blob '{name}': length {} exceeds maximum {}",
                len, MAX_TRAIN_BLOB
            )));
        }
        let mut data = vec![0u8; len];
        r.read_exact(&mut data)
            .map_err(|e| Error::Backend(format!("train blob data read failed: {e}")))?;

        Ok(Self { name, shape, data })
    }
}

/// A training-state sidecar: adapters, optimizer moments, error matrices.
///
/// Optional companion to a `.grim` inference file. The inference reader never
/// requires it; a resumed fine-tune reproduces step-N state from it.
#[derive(Debug, Clone)]
pub struct TrainState {
    /// Current optimizer step number. Persisted so resumed training
    /// picks up exactly where it left off.
    pub step: u64,
    /// Numeric format the training-state tensors are encoded in.
    pub fp_format: TrainFpFormat,
    /// Per-blob dtype map. Disambiguates param blobs from optimizer moment blobs
    /// when `fp_format` is a half-precision type (bf16/fp16) and moments stay f32.
    pub dtypes: HashMap<String, TrainFpFormat>,
    /// Named training-state blobs keyed by logical slot name
    /// (e.g. `lora_a`, `lora_b`, `opt_m`, `opt_v`, `error_matrix`).
    pub blobs: HashMap<String, TrainBlob>,
}

impl Default for TrainState {
    fn default() -> Self {
        Self {
            step: 0,
            fp_format: TrainFpFormat::Fp32,
            dtypes: HashMap::new(),
            blobs: HashMap::new(),
        }
    }
}

impl TrainState {
    /// Insert a blob under `name`.
    pub fn add_blob(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<u8>) {
        let name = name.into();
        self.blobs
            .insert(name.clone(), TrainBlob { name, shape, data });
    }

    /// Extract lora A and B raw f32 data for a given base tensor name.
    ///
    /// Expects blobs named `{tensor_name}.lora_A.weight` and `{tensor_name}.lora_B.weight`.
    /// Returns `(a_data, a_shape, b_data, b_shape)` or `None` if either blob is missing.
    pub fn lora_weights_for(
        &self,
        tensor_name: &str,
    ) -> Option<(Vec<f32>, &[usize], Vec<f32>, &[usize])> {
        let a_key = format!("{}.lora_A.weight", tensor_name);
        let b_key = format!("{}.lora_B.weight", tensor_name);
        let a_blob = self.blobs.get(&a_key)?;
        let b_blob = self.blobs.get(&b_key)?;

        let a_data = bytes_to_f32s(&a_blob.data, self.fp_format)?;
        let b_data = bytes_to_f32s(&b_blob.data, self.fp_format)?;
        Some((a_data, &a_blob.shape, b_data, &b_blob.shape))
    }

    /// List all base tensor names that have lora adapters in this sidecar.
    pub fn lora_tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .blobs
            .keys()
            .filter(|k| k.ends_with(".lora_A.weight"))
            .map(|k| k.strip_suffix(".lora_A.weight").unwrap().to_string())
            .collect();
        names.sort();
        names
    }

    /// Write the sidecar to `path` (conventionally `model.grim.train`).
    pub fn write<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let mut buf = Vec::new();
        buf.write_all(&TRAIN_MAGIC)
            .map_err(|e| Error::Backend(format!("train magic write failed: {e}")))?;

        let dtypes_tags: std::collections::BTreeMap<&str, u8> = self
            .dtypes
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_u8()))
            .collect();
        let header = serde_json::json!({
            "step": self.step,
            "fp_format": self.fp_format.as_u8(),
            "dtypes": dtypes_tags,
            "blobs": self.blobs.keys().collect::<Vec<_>>(),
        });
        let header_bytes = serde_json::to_vec(&header)
            .map_err(|e| Error::Backend(format!("train header serialize failed: {e}")))?;
        buf.write_all(&(header_bytes.len() as u32).to_le_bytes())
            .map_err(|e| Error::Backend(format!("train header len write failed: {e}")))?;
        buf.write_all(&header_bytes)
            .map_err(|e| Error::Backend(format!("train header write failed: {e}")))?;

        for blob in self.blobs.values() {
            blob.write(&mut buf)?;
        }

        std::fs::write(path, &buf).map_err(|e| Error::Backend(format!("train write failed: {e}")))
    }

    /// Read a sidecar from `path`. Returns `None` (not an error) when the
    /// file is absent, so inference readers can ignore a missing sidecar.
    pub fn read<P: AsRef<std::path::Path>>(path: P) -> Result<Option<Self>> {
        let path = path.as_ref();
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
        let mut reader = std::io::BufReader::new(file);

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| Error::Backend(format!("train magic read failed: {e}")))?;
        if magic != TRAIN_MAGIC {
            return Err(Error::Backend(format!(
                "Invalid train sidecar magic: expected {:?}, got {:?}",
                TRAIN_MAGIC, magic
            )));
        }

        let mut header_len_b = [0u8; 4];
        reader
            .read_exact(&mut header_len_b)
            .map_err(|e| Error::Backend(format!("train header len read failed: {e}")))?;
        let header_len = u32::from_le_bytes(header_len_b) as usize;
        // Cap header length to a sane maximum (64 MiB) to avoid allocating up to
        // 4 GB from an untrusted length field. [P1-28 fix.]
        const MAX_TRAIN_HEADER: usize = 1 << 26;
        if header_len > MAX_TRAIN_HEADER {
            return Err(Error::Backend(format!(
                "train sidecar: header_len {} exceeds maximum {}",
                header_len, MAX_TRAIN_HEADER
            )));
        }
        let mut header_bytes = vec![0u8; header_len];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| Error::Backend(format!("train header read failed: {e}")))?;
        let header: Value = serde_json::from_slice(&header_bytes)
            .map_err(|e| Error::Backend(format!("train header JSON invalid: {e}")))?;
        let step = header.get("step").and_then(|v| v.as_u64()).unwrap_or(0);
        let fp_format = header
            .get("fp_format")
            .and_then(|v| v.as_u64())
            .and_then(|v| TrainFpFormat::from_u8(v as u8))
            .unwrap_or(TrainFpFormat::Fp32);
        let dtypes = header
            .get("dtypes")
            .and_then(|v| v.as_object())
            .into_iter()
            .flatten()
            .filter_map(|(k, v)| {
                let tag = v.as_u64()?;
                TrainFpFormat::from_u8(tag as u8).map(|fmt| (k.clone(), fmt))
            })
            .collect();

        let mut blobs = HashMap::new();
        // The blob stream ends at EOF; read until exhausted.
        loop {
            // Peek: a short read means EOF — stop cleanly.
            let mut peek = [0u8; 2];
            let pos = reader
                .stream_position()
                .map_err(|e| Error::Backend(e.to_string()))?;
            let n = reader
                .read(&mut peek)
                .map_err(|e| Error::Backend(format!("train blob peek failed: {e}")))?;
            if n == 0 {
                break; // clean EOF
            }
            // Rewind the 2 peek bytes and read the full blob.
            reader
                .seek(SeekFrom::Start(pos))
                .map_err(|e| Error::Backend(e.to_string()))?;
            let blob = TrainBlob::read(&mut reader)?;
            blobs.insert(blob.name.clone(), blob);
        }

        Ok(Some(Self {
            step,
            fp_format,
            dtypes,
            blobs,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_state_round_trips_byte_identical() {
        let mut state = TrainState {
            step: 42,
            fp_format: TrainFpFormat::Fp8E4M3,
            dtypes: HashMap::new(),
            blobs: HashMap::new(),
        };
        state.add_blob("lora_a", vec![64, 128], (0u8..=127).collect());
        state.add_blob("lora_b", vec![128, 64], (128u8..=255).collect());
        state.add_blob("opt_m", vec![4096], vec![7u8; 4096]);
        state.add_blob(
            "error_matrix",
            vec![32, 32],
            (0u8..32).cycle().take(1024).collect(),
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.grim.train");
        state.write(&path).unwrap();

        let restored = TrainState::read(&path).unwrap().expect("should read");
        assert_eq!(restored.step, 42);
        assert_eq!(restored.fp_format, TrainFpFormat::Fp8E4M3);
        assert_eq!(restored.blobs.len(), 4);
        assert_eq!(
            restored.blobs["lora_a"].data,
            (0u8..=127).collect::<Vec<_>>()
        );
        assert_eq!(
            restored.blobs["lora_b"].data,
            (128u8..=255).collect::<Vec<_>>()
        );
        assert_eq!(restored.blobs["opt_m"].data, vec![7u8; 4096]);
        assert_eq!(restored.blobs["error_matrix"].shape, vec![32, 32]);
    }

    #[test]
    fn train_state_step_round_trips() {
        for step in [0u64, 1, 100, 1000, u64::MAX] {
            let state = TrainState {
                step,
                fp_format: TrainFpFormat::Fp32,
                dtypes: HashMap::new(),
                blobs: HashMap::new(),
            };
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(format!("step_{step}.grim.train"));
            state.write(&path).unwrap();
            let restored = TrainState::read(&path).unwrap().expect("should read");
            assert_eq!(restored.step, step);
        }
    }

    #[test]
    fn train_state_read_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.grim.train");
        let res = TrainState::read(&path).unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn train_fp_format_round_trips() {
        for fmt in [
            TrainFpFormat::Fp16,
            TrainFpFormat::Fp32,
            TrainFpFormat::Fp8E4M3,
            TrainFpFormat::Fp8E5M2,
            TrainFpFormat::Fp4,
            TrainFpFormat::Bf16,
            TrainFpFormat::Fp16Param,
        ] {
            assert_eq!(TrainFpFormat::from_u8(fmt.as_u8()), Some(fmt));
        }
        assert_eq!(TrainFpFormat::from_u8(99), None);
    }

    #[test]
    fn bf16_sidecar_round_trips() {
        let vals: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 0.5, 3.14159, -2.71828, 1e10, 1e-10];
        let bytes = encode_f32s_as(&vals, TrainFpFormat::Bf16);
        assert_eq!(bytes.len(), vals.len() * 2);
        let decoded = decode_f32s_from(&bytes, TrainFpFormat::Bf16).expect("decode");
        for (orig, dec) in vals.iter().zip(decoded.iter()) {
            let rel = (orig - dec).abs() / orig.abs().max(1e-30);
            assert!(rel < 1e-2, "bf16 round-trip: {orig} -> {dec}");
        }

        // Sidecar-level: dtypes map + fp_format survive write/read.
        let mut state = TrainState {
            step: 7,
            fp_format: TrainFpFormat::Bf16,
            dtypes: HashMap::new(),
            blobs: HashMap::new(),
        };
        state
            .dtypes
            .insert("param_0_0_a".to_string(), TrainFpFormat::Bf16);
        state.add_blob("param_0_0_a", vec![9], bytes.clone());
        state.add_blob(
            "opt_m_0_0_a",
            vec![9],
            vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bf16.grim.train");
        state.write(&path).unwrap();
        let restored = TrainState::read(&path).unwrap().expect("should read");
        assert_eq!(restored.fp_format, TrainFpFormat::Bf16);
        assert_eq!(
            restored.dtypes.get("param_0_0_a"),
            Some(&TrainFpFormat::Bf16)
        );
        assert_eq!(restored.blobs["param_0_0_a"].data, bytes);
    }

    #[test]
    fn fp16_sidecar_round_trips() {
        let vals: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0, 1e-4, 42.0];
        let bytes = encode_f32s_as(&vals, TrainFpFormat::Fp16);
        assert_eq!(bytes.len(), vals.len() * 2);
        let decoded = decode_f32s_from(&bytes, TrainFpFormat::Fp16).expect("decode");
        for (orig, dec) in vals.iter().zip(decoded.iter()) {
            let rel = (orig - dec).abs() / orig.abs().max(1e-30);
            assert!(rel < 1e-3, "fp16 round-trip: {orig} -> {dec}");
        }
        // Saturation beyond f16 max clamps to inf, not garbage.
        let over = encode_f32s_as(&[1e30], TrainFpFormat::Fp16);
        let dec = decode_f32s_from(&over, TrainFpFormat::Fp16).unwrap();
        assert!(dec[0].is_infinite());
    }

    /// Verifies `TrainState::read` rejects invalid magic bytes.
    #[test]
    fn test_train_state_read_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.grim.train");
        std::fs::write(&path, b"BADMAGIC00000000").unwrap();
        let res = TrainState::read(&path);
        assert!(res.is_err());
    }
}
