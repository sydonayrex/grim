//! A unified compressed tensor container format compatible with all
//! production-ready compressed tensor types in grim.
//!
//! This module defines a single container (.gcct — Grim Compact Tensor)
//! that can hold `W8A8Mxfp8`, `WNA16`, and `EmbeddingWNA16Int` tensors
//! in one file, with type-tagged per-tensor metadata and data sections.
//!
//! The container is a binary format with:
//! - A 4-byte magic + u32 version + u32 tensor_count header
//! - Per-tensor: name (u32-prefixed), type tag (u32), metadata (u32-prefixed
//!   length + opaque bytes), data (u64-prefixed length + raw bytes)
//!
//! [`read_gcct`] deserializes into [`CompressedTensor`] values that expose
//! the type tag, metadata bytes, and raw data bytes. [`dequantize_w8a8`]
//! dispatches the two W8A8 variants whose payload layouts this crate
//! DEFINES (see below); the remaining variants are container pass-through
//! because their payload layouts are owned by their producers (grim-quant /
//! the model loaders) — requesting their dequantization here is an explicit
//! [`GcctError::UnsupportedLayout`], never a silent guess.
//!
//! # Defined payload layouts
//!
//! - **`CompressedTensorsW8A8Int8`** — metadata: little-endian
//!   `(num_channels: u32, hidden: u32)`; data: `num_channels * hidden`
//!   int8 codes followed by `num_channels` f32 per-channel scales.
//!   Dequantized: `code * scale[channel]`.
//! - **`CompressedTensorsW8A8Fp8`** — metadata: little-endian
//!   `(num_channels: u32, hidden: u32)`; data: OCP E4M3 fp8 codes
//!   (`num_channels * hidden`) followed by `num_channels` f32 scales.
//!   Dequantized: `fp8_to_f32(code) * scale[channel]`.
//! - **`W8A8Mxfp8` / `WNA16` / `EmbeddingWNA16Int`** — container-only here:
//!   layout is producer-defined; [`dequantize_w8a8`] rejects them loudly.

use std::fmt;
use std::io::{Read, Write};

/// Magic bytes for the GCcT (Grim Compact Tensor) container: `GCT\x01`.
pub const GCCT_MAGIC: &[u8; 4] = b"GCT\x01";
/// Current container format version.
pub const GCCT_VERSION: u32 = 1;

/// Container parse / IO error. (Audit fix: `from_tag` previously returned
/// `fmt::Error` — the error type of std::fmt formatting traits — for a
/// data-format parse failure. All format errors now flow through this type.)
#[derive(Debug)]
pub enum GcctError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// The stream does not start with [`GCCT_MAGIC`].
    BadMagic,
    /// The stream declares a version this crate cannot read.
    UnsupportedVersion(u32),
    /// An unknown type tag was encountered.
    BadTag(u32),
    /// The stream ended before the declared contents were fully read.
    Truncated(String),
    /// A tensor name was not valid UTF-8.
    InvalidName,
    /// A tensor name or section length is implausible (0-length name,
    /// absurd allocation request).
    InvalidLength(u64),
    /// A dequantization request for a variant whose payload layout this
    /// crate does not define.
    UnsupportedLayout(&'static str),
}

impl fmt::Display for GcctError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GcctError::Io(e) => write!(f, "gcct i/o error: {e}"),
            GcctError::BadMagic => write!(f, "gcct: bad magic bytes (expected GCT\\x01)"),
            GcctError::UnsupportedVersion(v) => {
                write!(f, "gcct: unsupported container version {v} (this crate reads {GCCT_VERSION})")
            }
            GcctError::BadTag(t) => write!(f, "gcct: unknown compressed tensor type tag {t}"),
            GcctError::Truncated(what) => write!(f, "gcct: truncated stream while reading {what}"),
            GcctError::InvalidName => write!(f, "gcct: tensor name is not valid UTF-8"),
            GcctError::InvalidLength(n) => write!(f, "gcct: implausible section length {n}"),
            GcctError::UnsupportedLayout(what) => {
                write!(f, "gcct: dequantization layout not defined in this crate: {what}")
            }
        }
    }
}

impl std::error::Error for GcctError {}

impl From<std::io::Error> for GcctError {
    fn from(e: std::io::Error) -> Self {
        GcctError::Io(e)
    }
}

/// Tag identifying which compressed tensor type a section holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressedTensorType {
    /// W8A8MXFP8: 8-bit MXFP8 activations + weights in some quantized storage.
    W8A8Mxfp8 = 1,
    /// WNA16: weight-only N-bit with 16-bit precision.
    WNA16 = 2,
    /// EmbeddingWNA16Int: embedding weights stored as N-bit integers.
    EmbeddingWNA16Int = 3,
    /// CompressedTensors W8A8 with INT8 weights (SmoothQuant-style, per-channel
    /// int8 codes + per-channel scales).
    CompressedTensorsW8A8Int8 = 4,
    /// CompressedTensors W8A8 with FP8 (OCP E4M3) weights (per-channel
    /// fp8 codes + per-channel f32 scale).
    CompressedTensorsW8A8Fp8 = 5,
}

impl CompressedTensorType {
    /// Parse the u32 tag back into a `CompressedTensorType`.
    pub fn from_tag(tag: u32) -> Result<Self, GcctError> {
        match tag {
            1 => Ok(Self::W8A8Mxfp8),
            2 => Ok(Self::WNA16),
            3 => Ok(Self::EmbeddingWNA16Int),
            4 => Ok(Self::CompressedTensorsW8A8Int8),
            5 => Ok(Self::CompressedTensorsW8A8Fp8),
            _ => Err(GcctError::BadTag(tag)),
        }
    }

    /// The u32 tag written to the `.gcct` container header for this type.
    pub fn to_tag(self) -> u32 {
        match self {
            Self::W8A8Mxfp8 => 1,
            Self::WNA16 => 2,
            Self::EmbeddingWNA16Int => 3,
            Self::CompressedTensorsW8A8Int8 => 4,
            Self::CompressedTensorsW8A8Fp8 => 5,
        }
    }
}

/// One tensor section read from (or queued for) a `.gcct` container.
#[derive(Debug, Clone)]
pub struct CompressedTensor {
    pub name: String,
    pub tensor_type: CompressedTensorType,
    /// Opaque per-writer metadata bytes (dims, group size, …). The W8A8
    /// layouts consumed by [`dequantize_w8a8`] are defined in this crate's
    /// module docs; every other writer defines its own.
    pub metadata: Vec<u8>,
    pub data: Vec<u8>,
}

/// Write `tensors` as a complete `.gcct` container to `w`.
pub fn write_gcct<W: Write>(w: &mut W, tensors: &[CompressedTensor]) -> Result<(), GcctError> {
    w.write_all(GCCT_MAGIC)?;
    w.write_all(&GCCT_VERSION.to_le_bytes())?;
    w.write_all(&(tensors.len() as u32).to_le_bytes())?;
    for t in tensors {
        let name_bytes = t.name.as_bytes();
        if name_bytes.is_empty() {
            return Err(GcctError::InvalidName);
        }
        w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        w.write_all(name_bytes)?;
        w.write_all(&t.tensor_type.to_tag().to_le_bytes())?;
        w.write_all(&(t.metadata.len() as u32).to_le_bytes())?;
        w.write_all(&t.metadata)?;
        w.write_all(&(t.data.len() as u64).to_le_bytes())?;
        w.write_all(&t.data)?;
    }
    Ok(())
}

/// Read a complete `.gcct` container from `r`.
pub fn read_gcct<R: Read>(r: &mut R) -> Result<Vec<CompressedTensor>, GcctError> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|_| GcctError::Truncated("magic".into()))?;
    if &magic != GCCT_MAGIC {
        return Err(GcctError::BadMagic);
    }
    let mut version_buf = [0u8; 4];
    r.read_exact(&mut version_buf)
        .map_err(|_| GcctError::Truncated("version".into()))?;
    let version = u32::from_le_bytes(version_buf);
    if version != GCCT_VERSION {
        return Err(GcctError::UnsupportedVersion(version));
    }
    let mut count_buf = [0u8; 4];
    r.read_exact(&mut count_buf)
        .map_err(|_| GcctError::Truncated("tensor count".into()))?;
    let count = u32::from_le_bytes(count_buf) as usize;

    let mut out = Vec::with_capacity(count.min(1 << 20));
    for _ in 0..count {
        let mut name_len_buf = [0u8; 4];
        r.read_exact(&mut name_len_buf)
            .map_err(|_| GcctError::Truncated("name length".into()))?;
        let name_len = u32::from_le_bytes(name_len_buf) as usize;
        if name_len == 0 || name_len > (1 << 20) {
            return Err(GcctError::InvalidLength(name_len as u64));
        }
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes)
            .map_err(|_| GcctError::Truncated("name".into()))?;
        let name = String::from_utf8(name_bytes).map_err(|_| GcctError::InvalidName)?;

        let mut tag_buf = [0u8; 4];
        r.read_exact(&mut tag_buf)
            .map_err(|_| GcctError::Truncated("type tag".into()))?;
        let tensor_type = CompressedTensorType::from_tag(u32::from_le_bytes(tag_buf))?;

        let mut meta_len_buf = [0u8; 4];
        r.read_exact(&mut meta_len_buf)
            .map_err(|_| GcctError::Truncated("metadata length".into()))?;
        let meta_len = u32::from_le_bytes(meta_len_buf) as usize;
        if meta_len > (1 << 28) {
            return Err(GcctError::InvalidLength(meta_len as u64));
        }
        let mut metadata = vec![0u8; meta_len];
        r.read_exact(&mut metadata)
            .map_err(|_| GcctError::Truncated("metadata".into()))?;

        let mut data_len_buf = [0u8; 8];
        r.read_exact(&mut data_len_buf)
            .map_err(|_| GcctError::Truncated("data length".into()))?;
        let data_len = u64::from_le_bytes(data_len_buf);
        if data_len > (1 << 40) {
            return Err(GcctError::InvalidLength(data_len));
        }
        let mut data = vec![0u8; data_len as usize];
        r.read_exact(&mut data)
            .map_err(|_| GcctError::Truncated("data".into()))?;

        out.push(CompressedTensor {
            name,
            tensor_type,
            metadata,
            data,
        });
    }
    Ok(out)
}

/// Decode an OCP E4M3 fp8 byte to f32 (no NaN encodings beyond the standard:
/// exp=15/mantissa=7 is NaN per the E4M3 spec).
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as u32;
    if exp == 15 && mant == 7 {
        return f32::NAN * sign;
    }
    if exp == 0 {
        // Subnormal: mant / 8 * 2^(1-7)
        return sign * (mant as f32 / 8.0) * 2.0f32.powi(1 - 7);
    }
    sign * (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp - 7)
}

/// Dispatch dequantization for the W8A8 variants whose payload layouts this
/// crate defines (see the module docs). Returns
/// [`GcctError::UnsupportedLayout`] for the producer-owned variants —
/// loudly, never as a guessed decode.
pub fn dequantize_w8a8(t: &CompressedTensor) -> Result<Vec<f32>, GcctError> {
    match t.tensor_type {
        CompressedTensorType::CompressedTensorsW8A8Int8
        | CompressedTensorType::CompressedTensorsW8A8Fp8 => {}
        CompressedTensorType::W8A8Mxfp8 => {
            return Err(GcctError::UnsupportedLayout("W8A8Mxfp8 (layout owned by its producer)"))
        }
        CompressedTensorType::WNA16 => {
            return Err(GcctError::UnsupportedLayout("WNA16 (layout owned by its producer)"))
        }
        CompressedTensorType::EmbeddingWNA16Int => {
            return Err(GcctError::UnsupportedLayout(
                "EmbeddingWNA16Int (layout owned by its producer)",
            ))
        }
    }
    if t.metadata.len() < 8 {
        return Err(GcctError::Truncated("metadata shorter than (channels, hidden)".into()));
    }
    let num_channels = u32::from_le_bytes(t.metadata[0..4].try_into().unwrap()) as usize;
    let hidden = u32::from_le_bytes(t.metadata[4..8].try_into().unwrap()) as usize;
    let n = num_channels * hidden;
    let scales_bytes = num_channels * 4;
    if t.data.len() != n + scales_bytes {
        return Err(GcctError::Truncated(format!(
            "data length {} does not match {} codes + {} scale bytes",
            t.data.len(),
            n,
            scales_bytes
        )));
    }
    let mut out = Vec::with_capacity(n);
    match t.tensor_type {
        CompressedTensorType::CompressedTensorsW8A8Int8 => {
            for ch in 0..num_channels {
                let scale = f32::from_le_bytes(
                    t.data[n + ch * 4..n + ch * 4 + 4].try_into().unwrap(),
                );
                for &code in &t.data[ch * hidden..(ch + 1) * hidden] {
                    out.push((code as i8) as f32 * scale);
                }
            }
        }
        CompressedTensorType::CompressedTensorsW8A8Fp8 => {
            for ch in 0..num_channels {
                let scale = f32::from_le_bytes(
                    t.data[n + ch * 4..n + ch * 4 + 4].try_into().unwrap(),
                );
                for &code in &t.data[ch * hidden..(ch + 1) * hidden] {
                    out.push(fp8_e4m3_to_f32(code) * scale);
                }
            }
        }
        _ => unreachable!("dispatched above"),
    }
    Ok(out)
}

/// Build the metadata bytes for the W8A8 layouts defined in this crate.
pub fn w8a8_metadata(num_channels: usize, hidden: usize) -> Vec<u8> {
    let mut m = Vec::with_capacity(8);
    m.extend_from_slice(&(num_channels as u32).to_le_bytes());
    m.extend_from_slice(&(hidden as u32).to_le_bytes());
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le_f32s(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn tag_round_trip_and_discriminants() {
        for tag in 1..=5u32 {
            let t = CompressedTensorType::from_tag(tag).unwrap();
            assert_eq!(t.to_tag(), tag, "discriminant {tag} must round-trip");
        }
        assert!(matches!(
            CompressedTensorType::from_tag(0),
            Err(GcctError::BadTag(0))
        ));
        assert!(matches!(
            CompressedTensorType::from_tag(6),
            Err(GcctError::BadTag(6))
        ));
        assert!(matches!(
            CompressedTensorType::from_tag(u32::MAX),
            Err(GcctError::BadTag(u32::MAX))
        ));
    }

    #[test]
    fn container_round_trip_preserves_every_section() {
        let tensors = vec![
            CompressedTensor {
                name: "blk.0.attn_q.weight".into(),
                tensor_type: CompressedTensorType::CompressedTensorsW8A8Int8,
                metadata: w8a8_metadata(2, 4),
                data: {
                    let mut d = vec![1i8 as u8, 2, 3, 4, 200u8 as u8 /* wraps to -56 */, 0, 127, 128u8 as u8];
                    d.extend_from_slice(&le_f32s(&[0.5, 2.0]));
                    d
                },
            },
            CompressedTensor {
                name: "visual.encoder.λ.weight".into(), // non-ASCII name
                tensor_type: CompressedTensorType::WNA16,
                metadata: vec![],
                data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            },
            CompressedTensor {
                name: "empty-metadata-and-data".into(),
                tensor_type: CompressedTensorType::CompressedTensorsW8A8Fp8,
                metadata: w8a8_metadata(1, 2),
                data: {
                    let mut d = vec![0x38u8, 0xBC]; // E4M3: 1.0, -0.09375
                    d.extend_from_slice(&le_f32s(&[4.0]));
                    d
                },
            },
        ];
        let mut buf = Vec::new();
        write_gcct(&mut buf, &tensors).unwrap();
        assert_eq!(&buf[..4], GCCT_MAGIC);
        let read_back = read_gcct(&mut &buf[..]).unwrap();
        assert_eq!(read_back.len(), 3);
        for (orig, read) in tensors.iter().zip(&read_back) {
            assert_eq!(orig.name, read.name);
            assert_eq!(orig.tensor_type, read.tensor_type);
            assert_eq!(orig.metadata, read.metadata);
            assert_eq!(orig.data, read.data);
        }
    }

    #[test]
    fn container_corruption_is_rejected() {
        let tensors = vec![CompressedTensor {
            name: "t".into(),
            tensor_type: CompressedTensorType::CompressedTensorsW8A8Int8,
            metadata: w8a8_metadata(1, 1),
            data: vec![7, 0, 0, 0, 64],
        }];
        let mut buf = Vec::new();
        write_gcct(&mut buf, &tensors).unwrap();

        // Bad magic.
        let mut bad = buf.clone();
        bad[0] = b'X';
        assert!(matches!(read_gcct(&mut &bad[..]), Err(GcctError::BadMagic)));

        // Future version.
        let mut bad = buf.clone();
        bad[4] = 99; // low byte of the little-endian version field
        assert!(matches!(
            read_gcct(&mut &bad[..]),
            Err(GcctError::UnsupportedVersion(99))
        ));

        // Unknown tag inside the stream.
        let mut bad = buf.clone();
        let tag_pos = 4 + 4 + 4 + 4 + 1; // magic+ver+count + name_len + name
        bad[tag_pos] = 200;
        assert!(matches!(
            read_gcct(&mut &bad[..]),
            Err(GcctError::BadTag(200))
        ));

        // Every truncation prefix must error.
        for cut in 0..buf.len() {
            assert!(
                read_gcct(&mut &buf[..cut]).is_err(),
                "truncated container of {} bytes must not parse",
                cut
            );
        }
    }

    #[test]
    fn w8a8_int8_dequant_is_exact_per_channel() {
        let channels = 3usize;
        let hidden = 4usize;
        let codes: Vec<u8> = vec![10, 20, 30, 40, 200, 100, 0, 255, 128, 64, 32, 16];
        let scales = [0.5f32, -2.0, 0.0];
        let mut data = codes.clone();
        data.extend_from_slice(&le_f32s(&scales));
        let t = CompressedTensor {
            name: "w".into(),
            tensor_type: CompressedTensorType::CompressedTensorsW8A8Int8,
            metadata: w8a8_metadata(channels, hidden),
            data,
        };
        let got = dequantize_w8a8(&t).unwrap();
        for ch in 0..channels {
            for i in 0..hidden {
                let code = codes[ch * hidden + i] as i8;
                let want = code as f32 * scales[ch];
                assert!(
                    (got[ch * hidden + i] - want).abs() < 1e-6,
                    "ch{ch}[{i}]: got {} want {want}",
                    got[ch * hidden + i]
                );
            }
        }
    }

    #[test]
    fn fp8_e4m3_decode_matches_bit_patterns() {
        // Zero, negatives, subnormals, the canonical 1.0.
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x80), -0.0);
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0); // exp 7, mantissa 0
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);
        // 0x40: exp=8 → 2^(8-7)·1.0 = 2.0
        assert_eq!(fp8_e4m3_to_f32(0x40), 2.0);
        assert!((fp8_e4m3_to_f32(0x30) - 0.5).abs() < 1e-6); // exp 6 → 2^-1
        // Largest finite: exp 15, mantissa 6 → 1.75 · 2^8 = 448.
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        // NaN: exp 15, mantissa 7.
        assert!(fp8_e4m3_to_f32(0x7F).is_nan());
        // Subnormal: exp 0, mantissa 1 → 1/8 · 2^-6 = 2^-9.
        assert!((fp8_e4m3_to_f32(0x01) - 2.0f32.powi(-9)).abs() < 1e-12);
    }

    #[test]
    fn fp8_w8a8_dequant_applies_per_channel_scale() {
        let t = CompressedTensor {
            name: "fp8".into(),
            tensor_type: CompressedTensorType::CompressedTensorsW8A8Fp8,
            metadata: w8a8_metadata(2, 2),
            data: {
                let mut d = vec![0x38u8, 0x38, 0xB8, 0xBC]; // 1.0, 1.0, -1.0, -0.09375
                d.extend_from_slice(&le_f32s(&[3.0, 4.0]));
                d
            },
        };
        let got = dequantize_w8a8(&t).unwrap();
        // 0xBC = -1.5 (s=1, exp=7, mant=4) → -1.5 · 4.0 = -6.0
        assert_eq!(got, vec![3.0, 3.0, -4.0, -6.0]);
    }

    #[test]
    fn dequantize_rejects_producer_owned_layouts_and_bad_geometry() {
        let owned = CompressedTensor {
            name: "mxfp8".into(),
            tensor_type: CompressedTensorType::W8A8Mxfp8,
            metadata: vec![],
            data: vec![1, 2, 3],
        };
        assert!(matches!(
            dequantize_w8a8(&owned),
            Err(GcctError::UnsupportedLayout(_))
        ));
        // Data length inconsistent with declared geometry.
        let bad_geom = CompressedTensor {
            name: "w".into(),
            tensor_type: CompressedTensorType::CompressedTensorsW8A8Int8,
            metadata: w8a8_metadata(2, 4),
            data: vec![0u8; 3],
        };
        assert!(matches!(
            dequantize_w8a8(&bad_geom),
            Err(GcctError::Truncated(_))
        ));
    }
}
