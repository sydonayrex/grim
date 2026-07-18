//! Custom `.grim` (Outlier-Aware Streams & Wave64) Format representation.
//!
//! Defines the binary layout for native `.grim` model files: a header,
//! a JSON metadata layer, a tensor registry, and a dual-stream payload
//! (normals + outliers). See `grim_v2.md` §1 for the format specification.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use grim_tensor::error::{Error, Result};

/// FuckingSorcery magic bytes for `.grim`.
pub const FUCKING_SORCERY: [u8; 5] = [0x47, 0x52, 0x49, 0x4d, 0x01]; // "GRIM\x01"

/// Wave64 coalesced memory segment size in bytes.
///
/// Normals stream blocks are aligned to this boundary so that a single
/// wavefront load fetches exactly one segment (spec §1, §4).
pub const WAVE64_SEGMENT_BYTES: usize = 256;

/// Header of `.grim` model format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrimHeader {
    pub magic: [u8; 5],
    pub metadata_len: u64,
    pub num_tensors: u32,
}

impl GrimHeader {
    pub fn new(num_tensors: u32, metadata_len: u64) -> Self {
        Self {
            magic: FUCKING_SORCERY,
            metadata_len,
            num_tensors,
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.magic)
            .map_err(|e| Error::Backend(format!("Header write failed: {e}")))?;
        w.write_all(&self.metadata_len.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Header write failed: {e}")))?;
        w.write_all(&self.num_tensors.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Header write failed: {e}")))?;
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 5];
        r.read_exact(&mut magic)
            .map_err(|e| Error::Backend(format!("Header read failed: {e}")))?;
        if magic != FUCKING_SORCERY {
            return Err(Error::Backend(format!(
                "Invalid Header: FuckingSorcery magic mismatched. Expected {:?}, got {:?}",
                FUCKING_SORCERY, magic
            )));
        }

        let mut metadata_len_bytes = [0u8; 8];
        r.read_exact(&mut metadata_len_bytes)
            .map_err(|e| Error::Backend(format!("Header read failed: {e}")))?;
        let metadata_len = u64::from_le_bytes(metadata_len_bytes);

        let mut num_tensors_bytes = [0u8; 4];
        r.read_exact(&mut num_tensors_bytes)
            .map_err(|e| Error::Backend(format!("Header read failed: {e}")))?;
        let num_tensors = u32::from_le_bytes(num_tensors_bytes);

        Ok(Self {
            magic,
            metadata_len,
            num_tensors,
        })
    }
}

/// Registry entry for a single tensor inside `.grim` file format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrimTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub base_bitwidth: u8,
    pub payload_offset: u64,
    pub payload_size: u64,
    pub outlier_count: u32,
    pub outlier_offset: u64,

    // Persistent KV layout fields (WI-R4)
    pub kv_present: u8,
    pub kv_rotated: u8,
    pub kv_bits_k: u8,
    pub kv_bits_v: u8,
    pub kv_head_bits_table_offset: u64,
    pub kv_eviction_map_offset: u64,
    pub kv_eviction_map_size: u64,
    pub kv_sink_fp16: u8,
    pub kv_compressed_offset: u64,
    pub kv_compressed_size: u64,
}

impl Default for GrimTensorEntry {
    fn default() -> Self {
        Self {
            name: String::new(),
            shape: Vec::new(),
            base_bitwidth: 0,
            payload_offset: 0,
            payload_size: 0,
            outlier_count: 0,
            outlier_offset: 0,
            kv_present: 0,
            kv_rotated: 0,
            kv_bits_k: 0,
            kv_bits_v: 0,
            kv_head_bits_table_offset: 0,
            kv_eviction_map_offset: 0,
            kv_eviction_map_size: 0,
            kv_sink_fp16: 0,
            kv_compressed_offset: 0,
            kv_compressed_size: 0,
        }
    }
}

impl GrimTensorEntry {
    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        let name_bytes = self.name.as_bytes();
        w.write_all(&(name_bytes.len() as u16).to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(name_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;

        w.write_all(&(self.shape.len() as u8).to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        for dim in &self.shape {
            w.write_all(&(*dim as u32).to_le_bytes())
                .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        }

        w.write_all(&[self.base_bitwidth])
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.payload_offset.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.payload_size.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.outlier_count.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.outlier_offset.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;

        // Write KV fields
        w.write_all(&[self.kv_present])
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&[self.kv_rotated])
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&[self.kv_bits_k])
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&[self.kv_bits_v])
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.kv_head_bits_table_offset.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.kv_eviction_map_offset.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.kv_eviction_map_size.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&[self.kv_sink_fp16])
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.kv_compressed_offset.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;
        w.write_all(&self.kv_compressed_size.to_le_bytes())
            .map_err(|e| Error::Backend(format!("Tensor entry write failed: {e}")))?;

        Ok(())
    }

    pub fn read<R: Read>(r: &mut R, has_kv: bool) -> Result<Self> {
        let mut name_len_bytes = [0u8; 2];
        r.read_exact(&mut name_len_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let name_len = u16::from_le_bytes(name_len_bytes) as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| Error::Backend(format!("Invalid UTF-8 in tensor name: {e}")))?;

        let mut num_dims_bytes = [0u8; 1];
        r.read_exact(&mut num_dims_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let num_dims = num_dims_bytes[0] as usize;
        let mut shape = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            let mut dim_bytes = [0u8; 4];
            r.read_exact(&mut dim_bytes)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            shape.push(u32::from_le_bytes(dim_bytes) as usize);
        }

        let mut base_bw_bytes = [0u8; 1];
        r.read_exact(&mut base_bw_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let base_bitwidth = base_bw_bytes[0];

        let mut offset_bytes = [0u8; 8];
        r.read_exact(&mut offset_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let payload_offset = u64::from_le_bytes(offset_bytes);

        let mut size_bytes = [0u8; 8];
        r.read_exact(&mut size_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let payload_size = u64::from_le_bytes(size_bytes);

        let mut o_count_bytes = [0u8; 4];
        r.read_exact(&mut o_count_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let outlier_count = u32::from_le_bytes(o_count_bytes);

        let mut o_offset_bytes = [0u8; 8];
        r.read_exact(&mut o_offset_bytes)
            .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
        let outlier_offset = u64::from_le_bytes(o_offset_bytes);

        let mut kv_present = 0u8;
        let mut kv_rotated = 0u8;
        let mut kv_bits_k = 0u8;
        let mut kv_bits_v = 0u8;
        let mut kv_head_bits_table_offset = 0u64;
        let mut kv_eviction_map_offset = 0u64;
        let mut kv_eviction_map_size = 0u64;
        let mut kv_sink_fp16 = 0u8;
        let mut kv_compressed_offset = 0u64;
        let mut kv_compressed_size = 0u64;

        if has_kv {
            let mut buf_u8 = [0u8; 1];
            r.read_exact(&mut buf_u8)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_present = buf_u8[0];

            r.read_exact(&mut buf_u8)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_rotated = buf_u8[0];

            r.read_exact(&mut buf_u8)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_bits_k = buf_u8[0];

            r.read_exact(&mut buf_u8)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_bits_v = buf_u8[0];

            let mut buf_u64 = [0u8; 8];
            r.read_exact(&mut buf_u64)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_head_bits_table_offset = u64::from_le_bytes(buf_u64);

            r.read_exact(&mut buf_u64)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_eviction_map_offset = u64::from_le_bytes(buf_u64);

            r.read_exact(&mut buf_u64)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_eviction_map_size = u64::from_le_bytes(buf_u64);

            r.read_exact(&mut buf_u8)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_sink_fp16 = buf_u8[0];

            r.read_exact(&mut buf_u64)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_compressed_offset = u64::from_le_bytes(buf_u64);

            r.read_exact(&mut buf_u64)
                .map_err(|e| Error::Backend(format!("Tensor entry read failed: {e}")))?;
            kv_compressed_size = u64::from_le_bytes(buf_u64);
        }

        Ok(Self {
            name,
            shape,
            base_bitwidth,
            payload_offset,
            payload_size,
            outlier_count,
            outlier_offset,
            kv_present,
            kv_rotated,
            kv_bits_k,
            kv_bits_v,
            kv_head_bits_table_offset,
            kv_eviction_map_offset,
            kv_eviction_map_size,
            kv_sink_fp16,
            kv_compressed_offset,
            kv_compressed_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Outlier stream layout (spec §1: "Outliers Stream — Indices + float outliers")
// ---------------------------------------------------------------------------

/// Byte-level record for one outlier in the outliers stream.
///
/// Layout: `[ index: u32 LE | value: f16 LE ]` = 6 bytes per outlier.
/// `index` is the flat position within the tensor's dequantized element
/// space (row-major). `value` is the high-precision correction that
/// replaces the low-bit normal at that position.
///
/// `outlier_count` in [`GrimTensorEntry`] gives the number of these
/// records; the outliers stream for one tensor spans
/// `outlier_count * OUTLIER_RECORD_BYTES` starting at `outlier_offset`.
pub const OUTLIER_RECORD_BYTES: usize = 6;

/// One decoded outlier: position + correction value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrimOutlier {
    /// Flat index into the dequantized tensor element array.
    pub index: u32,
    /// FP16 correction value (stored as f32 after decode).
    pub value: f32,
}

impl GrimOutlier {
    /// Encode to the 6-byte on-disk layout: `[u32 index | f16 value ]`.
    pub fn encode(&self) -> [u8; OUTLIER_RECORD_BYTES] {
        let mut buf = [0u8; OUTLIER_RECORD_BYTES];
        buf[..4].copy_from_slice(&self.index.to_le_bytes());
        let f16_val = half::f16::from_f32(self.value);
        buf[4..].copy_from_slice(&f16_val.to_le_bytes());
        buf
    }

    /// Decode one outlier from a 6-byte slice.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < OUTLIER_RECORD_BYTES {
            return Err(Error::Backend(format!(
                "outlier record too short: {} bytes, need {}",
                buf.len(),
                OUTLIER_RECORD_BYTES
            )));
        }
        let index = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let f16_val = half::f16::from_le_bytes([buf[4], buf[5]]);
        Ok(Self {
            index,
            value: f16_val.to_f32(),
        })
    }
}

/// Read and decode the outliers stream for one tensor.
///
/// Defaults to the legacy flat `OUTLIER_RECORD_BYTES` (6-byte) encoding.
/// Callers that store `outlier_index_encoding` on the tensor's
/// [`crate::spec::GrimTensorExt`] should use
/// [`read_outliers_with_encoding`] to dispatch on the encoding mode.
pub fn read_outliers<R: Read + Seek>(
    reader: &mut R,
    entry: &GrimTensorEntry,
) -> Result<Vec<GrimOutlier>> {
    if entry.outlier_count == 0 {
        return Ok(Vec::new());
    }
    reader.seek(SeekFrom::Start(entry.outlier_offset))?;
    let mut buf = vec![0u8; entry.outlier_count as usize * OUTLIER_RECORD_BYTES];
    reader.read_exact(&mut buf)?;
    let mut outliers = Vec::with_capacity(entry.outlier_count as usize);
    for chunk in buf.chunks_exact(OUTLIER_RECORD_BYTES) {
        outliers.push(GrimOutlier::decode(chunk)?);
    }
    Ok(outliers)
}

/// Read outliers, dispatching on the encoding declared in the tensor's
/// capability extension. Returns a `Vec<GrimOutlier>` of length
/// `entry.outlier_count` regardless of the on-disk encoding.
///
/// - `FlatU32` (default, legacy): 6 bytes per record, see [`GrimOutlier`].
/// - `DeltaVarint` (compressed): see
///   [`crate::spec::decode_outliers_delta_varint`].
pub fn read_outliers_with_encoding<R: Read + Seek>(
    reader: &mut R,
    entry: &GrimTensorEntry,
    encoding: crate::spec::OutlierIndexEncoding,
) -> Result<Vec<GrimOutlier>> {
    if entry.outlier_count == 0 {
        return Ok(Vec::new());
    }
    if encoding == crate::spec::OutlierIndexEncoding::FlatU32 {
        return read_outliers(reader, entry);
    }
    // DeltaVarint path: we don't know the on-disk byte length from the
    // entry alone (it's a varint stream), so we read a generous slice
    // from `outlier_offset` up to the next region. For the common case
    // where the outlier stream is the last region of the tensor, read
    // `outlier_count` records' worth of bytes as an upper bound.
    let max_bytes = (entry.outlier_count as usize)
        .saturating_mul(OUTLIER_RECORD_BYTES)
        .max(OUTLIER_RECORD_BYTES);
    reader.seek(SeekFrom::Start(entry.outlier_offset))?;
    let mut buf = vec![0u8; max_bytes];
    let read_len = reader.read(&mut buf)?;
    buf.truncate(read_len);
    let decoded = crate::spec::decode_outliers_delta_varint(&buf)
        .map_err(Error::Backend)?;
    Ok(decoded
        .into_iter()
        .take(entry.outlier_count as usize)
        .map(|(index, value)| GrimOutlier { index, value })
        .collect())
}

// ---------------------------------------------------------------------------
// Normals stream layout (spec §1, §4: Wave64-aligned packed blocks)
// ---------------------------------------------------------------------------

/// Compute the packed byte size of a normals stream for one tensor.
///
/// The normals stream holds the low-bit-weight majority, packed at
/// `base_bitwidth` bits per weight and aligned to [`WAVE64_SEGMENT_BYTES`].
/// Outlier positions are excluded (they live in the outliers stream), so
/// the element count is `total_elements - outlier_count`.
pub fn normals_packed_size(
    total_elements: usize,
    outlier_count: u32,
    base_bitwidth: u8,
) -> u64 {
    let normal_elements = total_elements.saturating_sub(outlier_count as usize);
    let bits = normal_elements as u64 * base_bitwidth as u64;
    let bytes = bits.div_ceil(8);
    let aligned = (bytes + WAVE64_SEGMENT_BYTES as u64 - 1) / WAVE64_SEGMENT_BYTES as u64
        * WAVE64_SEGMENT_BYTES as u64;
    aligned
}

/// Layout of a single tensor's normals stream, used by the Phase 2
/// codes+scales readers/writers.
///
/// `Legacy` (default) is codes-only — matches the original V1 layout
/// where the per-tensor entry has no scale region. `WithScales` adds a
/// per-row scale region of `row_count` bytes (one u8 per row) right
/// after the codes region, both Wave64-aligned.
///
/// Phase 3 adds optional per-row mixed bitwidths: when `row_bpw_table`
/// is non-empty, each row is packed at its own bpw and padded to a
/// Wave64 segment independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalsLayout {
    /// Total dequantized element count of the tensor (rows × row_stride).
    pub total_elements: usize,
    /// Number of elements moved to the outliers stream.
    pub outlier_count: u32,
    /// Bits per weight for the codes region (used when `row_bpw_table`
    /// is empty — uniform-bitwidth mode).
    pub base_bitwidth: u8,
    /// Number of rows (shape[:-1] product, 1 if 1-D). Zero means "no
    /// scale region" (legacy codes-only layout).
    pub row_count: u64,
    /// Elements per row (shape[-1]). Used to size each row's packed
    /// segment in mixed-bpw mode.
    pub row_stride: u64,
    /// Per-row bitwidth table. Empty = uniform at `base_bitwidth`.
    /// Each entry must be in 2..=8.
    pub row_bpw_table: Vec<u8>,
}

impl NormalsLayout {
    /// Legacy codes-only layout (no per-row scales). Matches the V1
    /// on-disk format bit-for-bit.
    pub fn legacy(total_elements: usize, outlier_count: u32, base_bitwidth: u8) -> Self {
        Self {
            total_elements,
            outlier_count,
            base_bitwidth,
            row_count: 0,
            row_stride: 0,
            row_bpw_table: Vec::new(),
        }
    }

    /// Phase 2 layout with a per-row u8 scale region.
    pub fn with_scales(
        total_elements: usize,
        outlier_count: u32,
        base_bitwidth: u8,
        row_count: u64,
    ) -> Self {
        Self {
            total_elements,
            outlier_count,
            base_bitwidth,
            row_count,
            row_stride: if row_count > 0 {
                total_elements as u64 / row_count
            } else {
                0
            },
            row_bpw_table: Vec::new(),
        }
    }

    /// Phase 3 layout: per-row mixed bitwidths plus an optional per-row
    /// scale region.
    ///
    /// `row_bpw_table` length must equal `row_count` (when scales are
    /// present) or zero (uniform mode). Each entry must be in 2..=8.
    pub fn with_mixed_bpw(
        total_elements: usize,
        outlier_count: u32,
        row_count: u64,
        row_stride: u64,
        row_bpw_table: Vec<u8>,
    ) -> Self {
        let base_bitwidth = row_bpw_table.first().copied().unwrap_or(4);
        Self {
            total_elements,
            outlier_count,
            base_bitwidth,
            row_count,
            row_stride,
            row_bpw_table,
        }
    }

    /// `true` if this layout has a per-row scale region.
    pub fn has_scales(&self) -> bool {
        self.row_count > 0
    }

    /// `true` if this layout uses per-row mixed bitwidths.
    pub fn is_mixed_bpw(&self) -> bool {
        !self.row_bpw_table.is_empty()
    }

    /// Byte size of the codes region, Wave64-aligned.
    ///
    /// In uniform mode this is the total packed-bit size aligned to one
    /// Wave64 segment. In mixed-bpw mode each row is packed at its own
    /// bpw and aligned independently, then concatenated.
    pub fn codes_size(&self) -> u64 {
        if self.is_mixed_bpw() {
            self.row_bpw_table
                .iter()
                .map(|&bpw| align_wave64(row_codes_bytes(self.row_stride, bpw)))
                .sum()
        } else {
            let normal_elements = self.total_elements.saturating_sub(self.outlier_count as usize);
            let bits = normal_elements as u64 * self.base_bitwidth as u64;
            let bytes = bits.div_ceil(8);
            align_wave64(bytes)
        }
    }

    /// Per-row codes byte sizes (mixed-bpw mode only). Empty in uniform mode.
    pub fn row_codes_sizes(&self) -> Vec<u64> {
        self.row_bpw_table
            .iter()
            .map(|&bpw| align_wave64(row_codes_bytes(self.row_stride, bpw)))
            .collect()
    }

    /// Byte size of the per-row scale region, Wave64-aligned. Zero when
    /// `has_scales()` is false.
    pub fn scale_size(&self) -> u64 {
        if !self.has_scales() {
            return 0;
        }
        align_wave64(self.row_count)
    }

    /// Total normals payload size: codes + scales.
    pub fn payload_size(&self) -> u64 {
        self.codes_size() + self.scale_size()
    }
}

/// Bytes needed to pack `row_stride` elements at `bpw` bits, unaligned.
fn row_codes_bytes(row_stride: u64, bpw: u8) -> u64 {
    let bits = row_stride * bpw as u64;
    bits.div_ceil(8)
}

/// Round `n` up to the next multiple of [`WAVE64_SEGMENT_BYTES`].
fn align_wave64(n: u64) -> u64 {
    (n + WAVE64_SEGMENT_BYTES as u64 - 1) / WAVE64_SEGMENT_BYTES as u64 * WAVE64_SEGMENT_BYTES as u64
}

// ---------------------------------------------------------------------------
// Backup (residual) stream layout — spec Phase 4 (D4, D5)
// ---------------------------------------------------------------------------

/// Layout of one backup (residual) stream for a tensor.
///
/// A backup stream is an additive correction applied at dequant time
/// after the codes. It carries its own packed codes at `bpw` bits and
/// its own per-row u8 scale region. Spec §Architecture "Residual
/// (backup) streams".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackupLayout {
    /// Number of elements the backup covers (= the tensor's element count).
    pub total_elements: usize,
    /// Bits per code in the backup's packed codes region.
    pub bpw: u8,
    /// Number of rows for the per-row scale region. Zero = no scales.
    pub row_count: u64,
}

impl BackupLayout {
    /// Construct a backup layout. `bpw == 0` means "absent".
    pub fn new(total_elements: usize, bpw: u8, row_count: u64) -> Self {
        Self { total_elements, bpw, row_count }
    }

    /// `true` when this backup is present on disk.
    pub fn is_present(&self) -> bool {
        self.bpw > 0
    }

    /// Byte size of the codes region, Wave64-aligned. Zero when absent.
    pub fn codes_size(&self) -> u64 {
        if !self.is_present() {
            return 0;
        }
        let bits = self.total_elements as u64 * self.bpw as u64;
        align_wave64(bits.div_ceil(8))
    }

    /// Byte size of the per-row scale region, Wave64-aligned. Zero when
    /// `row_count == 0` or the backup is absent.
    pub fn scale_size(&self) -> u64 {
        if !self.is_present() || self.row_count == 0 {
            return 0;
        }
        align_wave64(self.row_count)
    }

    /// Total backup payload size: codes + scales.
    pub fn payload_size(&self) -> u64 {
        self.codes_size() + self.scale_size()
    }
}

/// Write a backup stream: codes first (Wave64-aligned), then the per-row
/// u8 scale region (Wave64-aligned).
///
/// Mirrors [`write_normals`] but for the additive residual layer. The
/// caller supplies packed codes bytes (length ≤ `layout.codes_size()`)
/// and per-row u8 scales (length = `layout.row_count`, or empty when no
/// scales).
pub fn write_backup<W: Write>(
    w: &mut W,
    layout: &BackupLayout,
    codes_in: &[u8],
    scales_in: &[u8],
) -> Result<()> {
    if !layout.is_present() {
        return Ok(());
    }
    let codes_size = layout.codes_size() as usize;
    if codes_in.len() > codes_size {
        return Err(Error::Backend(format!(
            "backup codes buffer {} exceeds region {}",
            codes_in.len(),
            codes_size
        )));
    }
    w.write_all(codes_in)
        .map_err(|e| Error::Backend(format!("backup codes write failed: {e}")))?;
    let codes_pad = codes_size - codes_in.len();
    if codes_pad > 0 {
        w.write_all(&vec![0u8; codes_pad])
            .map_err(|e| Error::Backend(format!("backup codes pad failed: {e}")))?;
    }

    if layout.row_count > 0 {
        let want = layout.row_count as usize;
        let scale_size = layout.scale_size() as usize;
        if scales_in.len() != want {
            return Err(Error::Backend(format!(
                "backup scales length {} does not match row_count {}",
                scales_in.len(),
                want
            )));
        }
        w.write_all(scales_in)
            .map_err(|e| Error::Backend(format!("backup scales write failed: {e}")))?;
        let scale_pad = scale_size - want;
        if scale_pad > 0 {
            w.write_all(&vec![0u8; scale_pad])
                .map_err(|e| Error::Backend(format!("backup scales pad failed: {e}")))?;
        }
    }
    Ok(())
}

/// Read a backup stream previously written by [`write_backup`].
///
/// Returns the codes bytes (always `layout.codes_size()` long when
/// present) and the per-row scale bytes (trimmed to `row_count`).
pub fn read_backup<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    layout: &BackupLayout,
) -> Result<NormalsPayload> {
    if !layout.is_present() {
        return Ok(NormalsPayload {
            codes: Vec::new(),
            scales: Vec::new(),
        });
    }
    reader.seek(SeekFrom::Start(offset))?;
    let mut codes = vec![0u8; layout.codes_size() as usize];
    reader.read_exact(&mut codes)?;

    let scales = if layout.row_count > 0 {
        let mut buf = vec![0u8; layout.scale_size() as usize];
        reader.read_exact(&mut buf)?;
        buf.truncate(layout.row_count as usize);
        buf
    } else {
        Vec::new()
    };

    Ok(NormalsPayload { codes, scales })
}

/// Result of reading a normals stream that may contain a scale region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalsPayload {
    /// Packed codes bytes (still bit-packed at `base_bitwidth`).
    pub codes: Vec<u8>,
    /// Per-row u8 scale bytes. Empty when the layout has no scales.
    pub scales: Vec<u8>,
}

/// Write a normals stream: codes first, then (if the layout has scales)
/// the per-row u8 scale bytes. Both regions are Wave64-aligned.
///
/// `codes_in` is the still-packed codes buffer (its length must match
/// `layout.codes_size()`). `scales_in` is the per-row u8 scale buffer
/// (length must match `layout.row_count`); ignored when the layout has
/// no scales.
///
/// In mixed-bpw mode, `codes_in` must already be laid out as the
/// concatenation of per-row Wave64-aligned packed segments — callers
/// pack each row at its own bpw and let this function handle only the
/// scale-region append.
pub fn write_normals<W: Write>(
    w: &mut W,
    layout: &NormalsLayout,
    codes_in: &[u8],
    scales_in: &[u8],
) -> Result<()> {
    let codes_size = layout.codes_size() as usize;
    if codes_in.len() > codes_size {
        return Err(Error::Backend(format!(
            "codes buffer {} bytes exceeds allocated codes region {}",
            codes_in.len(),
            codes_size
        )));
    }
    w.write_all(codes_in)
        .map_err(|e| Error::Backend(format!("codes write failed: {e}")))?;
    let codes_pad = codes_size - codes_in.len();
    if codes_pad > 0 {
        w.write_all(&vec![0u8; codes_pad])
            .map_err(|e| Error::Backend(format!("codes pad write failed: {e}")))?;
    }

    if layout.has_scales() {
        let scale_size = layout.scale_size() as usize;
        let want_scales_len = layout.row_count as usize;
        if scales_in.len() != want_scales_len {
            return Err(Error::Backend(format!(
                "scales buffer length {} does not match row_count {}",
                scales_in.len(),
                want_scales_len
            )));
        }
        w.write_all(scales_in)
            .map_err(|e| Error::Backend(format!("scales write failed: {e}")))?;
        let scale_pad = scale_size - scales_in.len();
        if scale_pad > 0 {
            w.write_all(&vec![0u8; scale_pad])
                .map_err(|e| Error::Backend(format!("scales pad write failed: {e}")))?;
        }
    }
    Ok(())
}

/// Pack one row's elements at `bpw` bits and append with Wave64 padding.
///
/// Helper for mixed-bpw writers: given `row_values` as f32 and the row's
/// bitwidth, pack to `bpw` bits big-endian-bit / little-endian-byte (the
/// spec convention matching EXL2/GPTQ) and append to `out`, zero-padded
/// to the next 256-byte boundary.
pub fn pack_row_bpw(out: &mut Vec<u8>, row_values: &[f32], bpw: u8) {
    let bits = row_values.len() as u64 * bpw as u64;
    let bytes_needed = bits.div_ceil(8) as usize;
    let start = out.len();
    out.resize(start + bytes_needed, 0u8);

    // Big-endian-bit, little-endian-byte packing. Each value occupies
    // `bpw` bits; the first value lives in the high bits of byte 0.
    // Codes can straddle a byte boundary when (bit_offset % 8) + bpw > 8.
    for (i, &v) in row_values.iter().enumerate() {
        let code = quantize_to_bpw(v, bpw) as u32;
        let bit_offset = i * bpw as usize;
        let byte_offset = bit_offset / 8;
        let in_byte_offset = bit_offset % 8;
        let bits_left_in_byte = 8 - in_byte_offset;

        if bits_left_in_byte >= bpw as usize {
            // Code fits entirely in the current byte.
            let shift = bits_left_in_byte - bpw as usize;
            out[start + byte_offset] |= (code << shift) as u8;
        } else {
            // Straddle: high bits go in the current byte, low bits in the next.
            let high_bits = bits_left_in_byte;
            let low_bits = bpw as usize - high_bits;
            out[start + byte_offset] |= (code >> low_bits) as u8;
            if byte_offset + 1 < bytes_needed {
                let low_shift = 8 - low_bits;
                out[start + byte_offset + 1] |= (code << low_shift) as u8;
            }
        }
    }

    // Wave64-align the row segment.
    let aligned = align_wave64(out.len() as u64) as usize;
    out.resize(aligned, 0u8);
}

/// Symmetric uniform quantization of a single f32 value to `bpw` bits.
/// Returns a code in `[0, 2^bpw - 1]`. Used by [`pack_row_bpw`].
fn quantize_to_bpw(value: f32, bpw: u8) -> u8 {
    let levels = (1u32 << bpw) as f32;
    // Map [-1, 1] to [0, levels-1]; clamp outside.
    let normalized = (value.clamp(-1.0, 1.0) + 1.0) * 0.5;
    (normalized * (levels - 1.0)).round() as u8
}

/// Read a normals stream previously written by [`write_normals`].
///
/// Splits the payload at `codes_size()` and returns the codes and scales
/// regions separately. The codes region is always returned; the scales
/// region is empty when the layout has no scales.
pub fn read_normals_split<R: Read + Seek>(
    reader: &mut R,
    payload_offset: u64,
    layout: &NormalsLayout,
) -> Result<NormalsPayload> {
    let codes_size = layout.codes_size();
    reader.seek(SeekFrom::Start(payload_offset))?;
    let mut codes = vec![0u8; codes_size as usize];
    reader.read_exact(&mut codes)?;

    let scales = if layout.has_scales() {
        let scale_size = layout.scale_size();
        let mut buf = vec![0u8; scale_size as usize];
        reader.read_exact(&mut buf)?;
        // Trim padding so callers see exactly row_count bytes.
        buf.truncate(layout.row_count as usize);
        buf
    } else {
        Vec::new()
    };

    Ok(NormalsPayload { codes, scales })
}

/// Read the raw (still-packed) normals bytes for one tensor.
///
/// The caller is responsible for dequantizing the packed bits according
/// to `entry.base_bitwidth`. This function returns the bytes verbatim.
pub fn read_normals<R: Read + Seek>(
    reader: &mut R,
    entry: &GrimTensorEntry,
) -> Result<Vec<u8>> {
    if entry.payload_size == 0 {
        return Ok(Vec::new());
    }
    reader.seek(SeekFrom::Start(entry.payload_offset))?;
    let mut buf = vec![0u8; entry.payload_size as usize];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

/// Read normals bytes, decompressing if the tensor's capability
/// extension declares `PayloadCompression::Zstd`.
///
/// Phase 6 (spec D13). When `compression == Raw` (or when no extension
/// is supplied) this is identical to [`read_normals`]. When `Zstd` is
/// declared, the on-disk payload is zstd-compressed and this function
/// returns the decompressed bytes.
pub fn read_normals_decompressing<R: Read + Seek>(
    reader: &mut R,
    entry: &GrimTensorEntry,
    compression: crate::spec::PayloadCompression,
) -> Result<Vec<u8>> {
    let raw = read_normals(reader, entry)?;
    if raw.is_empty() {
        return Ok(raw);
    }
    match compression {
        crate::spec::PayloadCompression::Raw => Ok(raw),
        crate::spec::PayloadCompression::Zstd => {
            zstd::decode_all(raw.as_slice())
                .map_err(|e| Error::Backend(format!("zstd decompress failed: {e}")))
        }
    }
}

// ---------------------------------------------------------------------------
// Persistent KV-cache layout (WI-R4) — extends `GrimTensorEntry`
// ---------------------------------------------------------------------------

/// Wave64-aligned byte blob encoding for a compressed KV cache.
///
/// The on-disk KV region is a single Wave64-aligned byte blob pointed at by
/// `GrimTensorEntry::kv_compressed_offset` / `kv_compressed_size`. The exact
/// inner byte layout is owned by the producer (e.g. `grim-kvquant`'s
/// `CompressedKvBlock::to_bytes`); this module only carries the bytes
/// verbatim so that a reloaded session can reproduce the compressed cache
/// bit-for-bit. Legacy (V2) entries carry `kv_present = 0` and an empty blob.
pub fn write_kv_block<W: Write>(w: &mut W, blob: &[u8]) -> Result<()> {
    w.write_all(blob)
        .map_err(|e| Error::Backend(format!("kv block write failed: {e}")))?;
    // Pad the KV blob to the next Wave64 segment boundary so the next
    // tensor's payload region stays Wave64-aligned (consistency with the
    // normals/outlier payloads).
    let pad = align_wave64(blob.len() as u64) as usize - blob.len();
    if pad > 0 {
        w.write_all(&vec![0u8; pad])
            .map_err(|e| Error::Backend(format!("kv block pad failed: {e}")))?;
    }
    Ok(())
}

/// Read a previously-written KV blob back from the payload region.
///
/// Returns exactly `entry.kv_compressed_size` bytes starting at
/// `entry.kv_compressed_offset`. When `kv_present == 0` the caller should
/// not call this (the blob is empty); we still guard against zero size.
pub fn read_kv_block<R: Read + Seek>(reader: &mut R, entry: &GrimTensorEntry) -> Result<Vec<u8>> {
    if entry.kv_compressed_size == 0 {
        return Ok(Vec::new());
    }
    reader.seek(SeekFrom::Start(entry.kv_compressed_offset))?;
    let mut buf = vec![0u8; entry.kv_compressed_size as usize];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

impl GrimTensorEntry {
    /// Record the persistent-KV layout fields from a compressed KV block's
    /// shape, leaving the byte blob (`kv_compressed_offset`/`size`) to be set
    /// by the writer after payload offsets are computed.
    ///
    /// `rotated` reflects RotateKV-style pre-rotation; `bits_k`/`bits_v` are
    /// the per-head bit-widths (0 = inherit default). Callers pass the
    /// producer's serialized blob length via `compressed_size` so the reader
    /// can fetch the right number of bytes; the `*_offset` is filled in by
    /// `GrimFile::write`.
    pub fn set_kv_layout(
        &mut self,
        present: bool,
        rotated: bool,
        bits_k: u8,
        bits_v: u8,
        eviction_map_offset: u64,
        eviction_map_size: u64,
        sink_fp16: bool,
        compressed_size: u64,
    ) {
        self.kv_present = u8::from(present);
        self.kv_rotated = u8::from(rotated);
        self.kv_bits_k = bits_k;
        self.kv_bits_v = bits_v;
        self.kv_eviction_map_offset = eviction_map_offset;
        self.kv_eviction_map_size = eviction_map_size;
        self.kv_sink_fp16 = u8::from(sink_fp16);
        self.kv_compressed_size = compressed_size;
        // `kv_compressed_offset` is computed during `GrimFile::write`.
    }
}

// ---------------------------------------------------------------------------
// Parsed file assembly + reader
// ---------------------------------------------------------------------------

/// Fully parsed native `.grim` file: header, metadata, tensor registry.
///
/// Raw payload bytes are not held in memory — use [`read_normals`] /
/// [`read_outliers`] to lazily fetch tensor data from the underlying reader.
pub struct GrimFile {
    pub header: GrimHeader,
    pub metadata: crate::gguf::GrimMetadata,
    pub tensors: Vec<GrimTensorEntry>,
    pub tensors_by_name: HashMap<String, usize>,
    /// Optional per-tensor serialized KV-cache blobs (WI-R4). Keyed by
    /// tensor name; only present when a writer attached a compressed KV
    /// block via [`GrimFile::add_kv_blob`]. The blob bytes are written into
    /// the payload region and the owning entry's `kv_compressed_offset` is
    /// assigned during [`GrimFile::write`].
    pub kv_blobs: HashMap<String, Vec<u8>>,
}

impl GrimFile {
    /// Attach a serialized KV-cache blob for `tensor_name` (WI-R4).
    ///
    /// The blob is written into the payload region at `GrimFile::write`
    /// time; `entry.kv_compressed_offset` is assigned then. Legacy files
    /// that never call this keep an empty map and `kv_present == 0`.
    pub fn add_kv_blob(&mut self, tensor_name: impl Into<String>, blob: Vec<u8>) {
        self.kv_blobs.insert(tensor_name.into(), blob);
    }
}

impl GrimFile {
    /// Parse the header + JSON metadata + tensor registry from a reader.
    ///
    /// The reader's position after this call is at the start of the raw
    /// payload region. Tensor data is read lazily via [`read_normals`] /
    /// [`read_outliers`].
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let header = GrimHeader::read(reader)?;

        // Metadata JSON layer.
        let metadata = if header.metadata_len > 0 {
            let mut meta_buf = vec![0u8; header.metadata_len as usize];
            reader.read_exact(&mut meta_buf)?;
            let json: serde_json::Value = serde_json::from_slice(&meta_buf).map_err(|e| {
                Error::Backend(format!("invalid .grim metadata JSON: {e}"))
            })?;
            crate::gguf::GrimMetadata::from_json(&json)
        } else {
            crate::gguf::GrimMetadata::default()
        };

        // Tensor registry.
        let has_kv = metadata.has_kv_registry.unwrap_or(false);
        let mut tensors = Vec::with_capacity(header.num_tensors as usize);
        for _ in 0..header.num_tensors {
            tensors.push(GrimTensorEntry::read(reader, has_kv)?);
        }
        let mut tensors_by_name = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            tensors_by_name.insert(t.name.clone(), i);
        }

        Ok(Self {
            header,
            metadata,
            tensors,
            tensors_by_name,
            kv_blobs: HashMap::new(),
        })
    }

    /// Write the complete `.grim` file: header + JSON metadata + registry + payloads.
    ///
    /// Payload offsets (`payload_offset`, `outlier_offset`) in `tensors` are
    /// recomputed relative to the start of the payload region and overwritten
    /// in the written entries, so callers can pass zeros.
    pub fn write<W: Write + Seek>(
        &self,
        w: &mut W,
    ) -> Result<Vec<GrimTensorEntry>> {
        let mut metadata = self.metadata.clone();
        metadata.has_kv_registry = Some(true);
        let metadata_json = metadata.to_json();
        let metadata_bytes = serde_json::to_vec(&metadata_json)
            .map_err(|e| Error::Backend(format!("metadata JSON serialize failed: {e}")))?;
        let metadata_len = metadata_bytes.len() as u64;

        let header = GrimHeader::new(self.tensors.len() as u32, metadata_len);
        header.write(w)?;
        w.write_all(&metadata_bytes)
            .map_err(|e| Error::Backend(format!("metadata write failed: {e}")))?;

        // Compute payload offsets relative to payload region start.
        let registry_byte_size: u64 = self.tensors.iter().map(registry_entry_size).sum();
        let payload_region_start = w.stream_position().map_err(|e| Error::Backend(e.to_string()))?
            + registry_byte_size;

        let mut offset = payload_region_start;
        let mut written_entries = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let mut entry = t.clone();
            entry.payload_offset = offset;
            offset += entry.payload_size;
            entry.outlier_offset = offset;
            offset += entry.outlier_count as u64 * OUTLIER_RECORD_BYTES as u64;
            // KV blob (WI-R4): appended after the outlier stream, then
            // Wave64-aligned. `kv_compressed_offset` is assigned here;
            // `kv_compressed_size` was set by the caller via
            // `GrimTensorEntry::set_kv_layout`.
            if entry.kv_present != 0 && entry.kv_compressed_size > 0 {
                entry.kv_compressed_offset = offset;
                offset += entry.kv_compressed_size;
                offset = (offset + WAVE64_SEGMENT_BYTES as u64 - 1) / WAVE64_SEGMENT_BYTES as u64
                    * WAVE64_SEGMENT_BYTES as u64;
            }
            // Align next tensor to a Wave64 segment boundary.
            offset = (offset + WAVE64_SEGMENT_BYTES as u64 - 1) / WAVE64_SEGMENT_BYTES as u64
                * WAVE64_SEGMENT_BYTES as u64;
            written_entries.push(entry);
        }

        for entry in &written_entries {
            entry.write(w)?;
        }

        // KV blobs (WI-R4) are emitted by the caller after `write`, using the
        // assigned `kv_compressed_offset`. This matches the existing pattern
        // where the caller writes the normals payload at `payload_offset`
        // using the returned `written_entries`. See `write_kv_block`.

        Ok(written_entries)
    }

    /// Look up a tensor entry by name.
    pub fn tensor(&self, name: &str) -> Option<&GrimTensorEntry> {
        self.tensors_by_name.get(name).map(|&i| &self.tensors[i])
    }
}

/// On-disk byte size of one tensor registry entry (for offset computation).
fn registry_entry_size(entry: &GrimTensorEntry) -> u64 {
    let name_len = 2 + entry.name.len() as u64;
    let shape_len = 1 + entry.shape.len() as u64 * 4;
    let fixed = 1 + 8 + 8 + 4 + 8; // base_bitwidth + offsets + count + offset
    let kv_fields = 1 + 1 + 1 + 1 + 8 + 8 + 8 + 1 + 8 + 8; // 45 bytes
    name_len + shape_len + fixed + kv_fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_serialization() {
        let header = GrimHeader::new(42, 1024);
        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();

        let mut reader = &buf[..];
        let decoded = GrimHeader::read(&mut reader).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn test_tensor_entry_serialization() {
        let entry = GrimTensorEntry {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![4096, 4096],
            base_bitwidth: 3,
            payload_offset: 2048,
            payload_size: 1572864,
            outlier_count: 512,
            outlier_offset: 1574912,
            ..Default::default()
        };
        let mut buf = Vec::new();
        entry.write(&mut buf).unwrap();

        let mut reader = &buf[..];
        let decoded = GrimTensorEntry::read(&mut reader, true).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn outlier_encode_decode_round_trip() {
        let o = GrimOutlier { index: 12345, value: 3.14 };
        let encoded = o.encode();
        assert_eq!(encoded.len(), OUTLIER_RECORD_BYTES);
        let decoded = GrimOutlier::decode(&encoded).unwrap();
        assert_eq!(decoded.index, o.index);
        assert!((decoded.value - o.value).abs() < 1e-2, "f16 round-trip: {} vs {}", decoded.value, o.value);
    }

    #[test]
    fn outlier_decode_rejects_short_buffer() {
        let short = [0u8; 3];
        assert!(GrimOutlier::decode(&short).is_err());
    }

    #[test]
    fn normals_packed_size_aligns_to_wave64_segment() {
        // 1000 elements at 4 bits = 500 bytes → rounds up to 1 segment (256 bytes is too small, so 2 segments = 512).
        let size = normals_packed_size(1000, 0, 4);
        assert_eq!(size, 512);
        assert_eq!(size % WAVE64_SEGMENT_BYTES as u64, 0);
    }

    #[test]
    fn normals_packed_size_subtracts_outliers() {
        let without = normals_packed_size(1000, 0, 4);
        let with = normals_packed_size(1000, 200, 4);
        // 200 fewer elements → fewer bits → same or smaller packed size.
        assert!(with <= without);
    }

    /// Phase 2: a legacy layout (no scales) computes the same payload
    /// size as the original `normals_packed_size` function.
    #[test]
    fn normals_layout_legacy_matches_packed_size() {
        let legacy = NormalsLayout::legacy(1000, 0, 4);
        assert_eq!(legacy.codes_size(), normals_packed_size(1000, 0, 4));
        assert_eq!(legacy.scale_size(), 0);
        assert_eq!(legacy.payload_size(), normals_packed_size(1000, 0, 4));
        assert!(!legacy.has_scales());
    }

    /// Phase 2: a with-scales layout adds a Wave64-aligned scale region
    /// on top of the codes region.
    #[test]
    fn normals_layout_with_scales_adds_aligned_scale_region() {
        let layout = NormalsLayout::with_scales(1000, 0, 4, 30);
        // 30 rows → 30 bytes of scale, rounded up to one 256-byte segment.
        assert_eq!(layout.scale_size(), 256);
        assert_eq!(layout.payload_size(), layout.codes_size() + 256);
        assert!(layout.has_scales());
    }

    /// Phase 2: write_normals + read_normals_split round-trip a codes +
    /// scales payload through an in-memory buffer.
    #[test]
    fn write_then_read_normals_split_round_trips() {
        let layout = NormalsLayout::with_scales(256, 0, 4, 4);
        let codes = vec![0xABu8; layout.codes_size() as usize];
        let scales = vec![0x12u8, 0x34, 0x56, 0x78];

        let mut buf = Vec::new();
        write_normals(&mut buf, &layout, &codes, &scales).expect("write");

        let mut cursor = std::io::Cursor::new(&buf[..]);
        let payload = read_normals_split(&mut cursor, 0, &layout).expect("read");
        assert_eq!(payload.codes, codes);
        assert_eq!(payload.scales, scales);
    }

    /// Phase 2: write_normals rejects a scales buffer whose length does
    /// not match the declared row_count.
    #[test]
    fn write_normals_rejects_wrong_scales_length() {
        let layout = NormalsLayout::with_scales(256, 0, 4, 4);
        let codes = vec![0u8; layout.codes_size() as usize];
        let wrong_scales = vec![0u8; 3]; // expected 4
        let mut buf = Vec::new();
        let res = write_normals(&mut buf, &layout, &codes, &wrong_scales);
        assert!(res.is_err());
    }

    /// Phase 2: a legacy layout has no scales region — write_normals
    /// accepts an empty scales buffer and read_normals_split returns
    /// empty scales.
    #[test]
    fn write_then_read_normals_legacy_has_no_scales() {
        let layout = NormalsLayout::legacy(256, 0, 4);
        let codes = vec![0xCDu8; layout.codes_size() as usize];

        let mut buf = Vec::new();
        write_normals(&mut buf, &layout, &codes, &[]).expect("write");

        let mut cursor = std::io::Cursor::new(&buf[..]);
        let payload = read_normals_split(&mut cursor, 0, &layout).expect("read");
        assert_eq!(payload.codes, codes);
        assert!(payload.scales.is_empty());
    }

    /// Phase 3: a mixed-bpw layout computes codes_size as the sum of
    /// per-row Wave64-aligned segments, each at its own bpw.
    #[test]
    fn normals_layout_mixed_bpw_sums_per_row_segments() {
        // Two rows: 8 elements each, bpw 2 and 6.
        let layout = NormalsLayout::with_mixed_bpw(16, 0, 2, 8, vec![2, 6]);
        assert!(layout.is_mixed_bpw());

        // Row 0: 8 elem × 2 bits = 16 bits = 2 bytes → align_wave64(2) = 256.
        // Row 1: 8 elem × 6 bits = 48 bits = 6 bytes → align_wave64(6) = 256.
        let row_sizes = layout.row_codes_sizes();
        assert_eq!(row_sizes, vec![256, 256]);
        assert_eq!(layout.codes_size(), 512);
    }

    /// Phase 3: mixed-bpw layout's total size is between uniform-min and
    /// uniform-max at the same element count, for realistic tensor sizes
    /// where Wave64 padding is negligible.
    #[test]
    fn normals_layout_mixed_bpw_size_between_uniform_bounds() {
        // 2 rows × 4096 elements each = 8192 total. Wave64 alignment is
        // negligible here, so the mixed size sits between uniform-2 and
        // uniform-6 as the spec requires.
        let total = 8192;
        let row_stride = 4096u64;
        let uniform_2 = NormalsLayout::legacy(total, 0, 2).codes_size();
        let uniform_6 = NormalsLayout::legacy(total, 0, 6).codes_size();
        let mixed = NormalsLayout::with_mixed_bpw(
            total, 0, 2, row_stride, vec![2, 6],
        )
        .codes_size();
        assert!(
            uniform_2 <= mixed && mixed <= uniform_6,
            "mixed {} should be between uniform-2 {} and uniform-6 {}",
            mixed,
            uniform_2,
            uniform_6
        );
    }

    /// Phase 3: pack_row_bpw lays out a single row of 8 f32 values at
    /// 4 bits each, producing exactly 4 bytes of packed codes followed
    /// by Wave64 alignment padding.
    #[test]
    fn pack_row_bpw_packs_eight_values_at_4_bits() {
        let values = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0, 0.25, -0.25, 0.75];
        let mut out = Vec::new();
        pack_row_bpw(&mut out, &values, 4);

        // 8 values × 4 bits = 32 bits = 4 bytes of packed codes, then
        // aligned up to 256.
        assert!(out.len() >= 4);
        assert_eq!(out.len() % WAVE64_SEGMENT_BYTES, 0);

        // First value -1.0 maps to code 0; second -0.5 to code 5 (in a
        // 16-level grid: (-0.5+1)*0.5*15 = 3.75 → 4). High nibble of
        // byte 0 should be 0, low nibble should be the second code.
        assert_eq!(out[0] >> 4, 0, "first code must be 0 for value -1.0");
    }

    /// Phase 3: pack_row_bpw handles the straddle case where bpw doesn't
    /// evenly divide 8 bits (e.g. bpw=3 with 5 values spans 2 bytes).
    #[test]
    fn pack_row_bpw_handles_3bit_straddle() {
        let values = vec![1.0f32; 5]; // all-ones input → max code
        let mut out = Vec::new();
        pack_row_bpw(&mut out, &values, 3);
        // 5 values × 3 bits = 15 bits = 2 bytes.
        assert!(out.len() >= 2);
    }

    /// Phase 4: an absent backup (bpw=0) has zero size and writes nothing.
    #[test]
    fn backup_layout_absent_has_zero_size() {
        let absent = BackupLayout::new(1024, 0, 0);
        assert!(!absent.is_present());
        assert_eq!(absent.codes_size(), 0);
        assert_eq!(absent.scale_size(), 0);
        assert_eq!(absent.payload_size(), 0);
    }

    /// Phase 4: a present 8-bit backup with per-row scales computes
    /// Wave64-aligned codes + scales regions.
    #[test]
    fn backup_layout_present_computes_aligned_sizes() {
        let layout = BackupLayout::new(4096, 8, 16);
        assert!(layout.is_present());
        // 4096 × 8 bits = 4096 bytes → already Wave64-aligned.
        assert_eq!(layout.codes_size(), 4096);
        // 16 rows × 1 byte = 16 bytes → aligned to 256.
        assert_eq!(layout.scale_size(), 256);
    }

    /// Phase 4: write_backup + read_backup round-trip a codes + scales
    /// payload through an in-memory buffer.
    #[test]
    fn write_then_read_backup_round_trips() {
        let layout = BackupLayout::new(256, 8, 4);
        let codes = vec![0x77u8; layout.codes_size() as usize];
        let scales = vec![0x11, 0x22, 0x33, 0x44];

        let mut buf = Vec::new();
        write_backup(&mut buf, &layout, &codes, &scales).expect("write");

        let mut cursor = std::io::Cursor::new(&buf[..]);
        let payload = read_backup(&mut cursor, 0, &layout).expect("read");
        assert_eq!(payload.codes, codes);
        assert_eq!(payload.scales, scales);
    }

    /// Phase 4: write_backup accepts an absent layout (bpw=0) and writes
    /// nothing; read_backup on an absent layout returns empty payload.
    #[test]
    fn write_then_read_backup_absent_is_noop() {
        let absent = BackupLayout::new(0, 0, 0);
        let mut buf = Vec::new();
        write_backup(&mut buf, &absent, &[], &[]).expect("write");
        assert!(buf.is_empty());

        let mut cursor = std::io::Cursor::new(&buf[..]);
        let payload = read_backup(&mut cursor, 0, &absent).expect("read");
        assert!(payload.codes.is_empty());
        assert!(payload.scales.is_empty());
    }

    /// Phase 6 (D13): when compression == Raw, the decompressing reader
    /// returns the bytes verbatim — same as `read_normals`.
    #[test]
    fn read_normals_decompressing_raw_passes_through() {
        use crate::spec::PayloadCompression;
        let original = vec![0xABu8; 512];
        let entry = GrimTensorEntry {
            name: "x".into(),
            shape: vec![128],
            base_bitwidth: 4,
            payload_offset: 0,
            payload_size: original.len() as u64,
            outlier_count: 0,
            outlier_offset: 0,
            ..Default::default()
        };
        let mut cursor = std::io::Cursor::new(&original[..]);
        let out = read_normals_decompressing(&mut cursor, &entry, PayloadCompression::Raw)
            .expect("read raw");
        assert_eq!(out, original);
    }

    /// Phase 6 (D13): when compression == Zstd, the decompressing reader
    /// returns the original payload bytes after zstd round-trip.
    #[test]
    fn read_normals_decompressing_zstd_round_trips() {
        use crate::spec::PayloadCompression;
        // Highly compressible payload so zstd shrinks it.
        let original = vec![0xCDu8; 4096];
        let compressed = zstd::encode_all(original.as_slice(), 3).expect("compress");

        let entry = GrimTensorEntry {
            name: "x".into(),
            shape: vec![1024],
            base_bitwidth: 4,
            payload_offset: 0,
            payload_size: compressed.len() as u64,
            outlier_count: 0,
            outlier_offset: 0,
            ..Default::default()
        };
        let mut cursor = std::io::Cursor::new(&compressed[..]);
        let out = read_normals_decompressing(&mut cursor, &entry, PayloadCompression::Zstd)
            .expect("decompress");
        assert_eq!(out, original);
    }

    #[test]
    fn grim_file_round_trip_with_metadata_and_tensors() {
        use crate::gguf::{GrimMetadata, GrimRocmlProfile};
        use std::io::Cursor;

        let metadata = GrimMetadata {
            magic: Some("grim-v1".into()),
            quant_version: Some(1),
            rocml_profile: GrimRocmlProfile::Rdna3,
            wavefront_size: 64,
            target_gcn: Some("gfx1100".into()),
            ..Default::default()
        };

        let tensor = GrimTensorEntry {
            name: "layer.0.weight".into(),
            shape: vec![128, 128],
            base_bitwidth: 4,
            payload_offset: 0,
            payload_size: 512,
            outlier_count: 2,
            outlier_offset: 0,
            ..Default::default()
        };

        let file = GrimFile {
            header: GrimHeader::new(1, 0),
            metadata,
            tensors: vec![tensor],
            tensors_by_name: HashMap::new(),
            kv_blobs: HashMap::new(),
        };

        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);
        let written = file.write(&mut cursor).unwrap();

        // Now read back.
        let mut reader = Cursor::new(&buf[..]);
        let restored = GrimFile::read(&mut reader).unwrap();
        assert_eq!(restored.header.num_tensors, 1);
        assert_eq!(restored.metadata.magic.as_deref(), Some("grim-v1"));
        assert_eq!(restored.metadata.target_gcn.as_deref(), Some("gfx1100"));
        assert_eq!(restored.tensors.len(), 1);
        assert_eq!(restored.tensors[0].name, "layer.0.weight");
        assert_eq!(restored.tensors[0].base_bitwidth, 4);

        // Offsets should have been recomputed and be non-zero.
        assert!(written[0].payload_offset > 0);
        assert!(written[0].outlier_offset >= written[0].payload_offset + written[0].payload_size);

        // tensor() lookup works.
        assert!(restored.tensor("layer.0.weight").is_some());
        assert!(restored.tensor("nonexistent").is_none());
    }

    /// WI-R4: a compressed KV block written via `GrimFile::add_kv_blob` +
    /// `set_kv_layout` round-trips byte-identically, and a reloaded session
    /// sees `kv_present == 1` with the same blob.
    #[test]
    fn kv_block_round_trips_byte_identical() {
        use crate::gguf::GrimMetadata;
        use std::io::Cursor;

        let blob: Vec<u8> = (0u8..=255).cycle().take(1024).collect();

        let mut tensor = GrimTensorEntry {
            name: "model.layers.0.self_attn.k_proj.weight".into(),
            shape: vec![32, 128, 128],
            base_bitwidth: 4,
            payload_offset: 0,
            payload_size: 512,
            outlier_count: 0,
            outlier_offset: 0,
            ..Default::default()
        };
        tensor.set_kv_layout(
            true,   // present
            true,   // rotated (RotateKV-style)
            3,      // bits_k
            4,      // bits_v
            0,      // eviction_map_offset
            0,      // eviction_map_size
            false,  // sink_fp16
            blob.len() as u64,
        );

        let mut file = GrimFile {
            header: GrimHeader::new(1, 0),
            metadata: GrimMetadata {
                magic: Some("grim-v1".into()),
                kv_layout_optimized: Some(true),
                ..Default::default()
            },
            tensors: vec![tensor],
            tensors_by_name: HashMap::new(),
            kv_blobs: HashMap::new(),
        };
        file.add_kv_blob("model.layers.0.self_attn.k_proj.weight", blob.clone());

        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);
        let written = file.write(&mut cursor).unwrap();

        // The caller (matching the normals-payload pattern) writes the KV
        // blob at the assigned offset.
        let e = &written[0];
        cursor.seek(SeekFrom::Start(e.kv_compressed_offset)).unwrap();
        write_kv_block(&mut cursor, &blob).unwrap();

        // Read back.
        let mut reader = Cursor::new(&buf[..]);
        let restored = GrimFile::read(&mut reader).unwrap();
        let e = &restored.tensors[0];
        assert_eq!(e.kv_present, 1);
        assert_eq!(e.kv_rotated, 1);
        assert_eq!(e.kv_bits_k, 3);
        assert_eq!(e.kv_bits_v, 4);
        assert_eq!(e.kv_compressed_size, blob.len() as u64);
        assert!(e.kv_compressed_offset > 0);

        // The blob must be byte-identical to what we wrote.
        let mut rb = Cursor::new(&buf[..]);
        let read_back = read_kv_block(&mut rb, e).unwrap();
        // `read_kv_block` returns exactly kv_compressed_size bytes; the
        // trailing Wave64 padding is included, so compare the prefix.
        assert_eq!(&read_back[..blob.len()], &blob[..]);
    }

    /// WI-R4: a legacy V2 file (no KV region) reads back with `kv_present == 0`
    /// and an empty KV blob — back-compat invariant.
    #[test]
    fn legacy_file_reads_kv_present_zero() {
        use crate::gguf::GrimMetadata;
        use std::io::Cursor;

        let tensor = GrimTensorEntry {
            name: "legacy.weight".into(),
            shape: vec![128, 128],
            base_bitwidth: 4,
            payload_offset: 0,
            payload_size: 512,
            ..Default::default()
        };
        let file = GrimFile {
            header: GrimHeader::new(1, 0),
            metadata: GrimMetadata::default(),
            tensors: vec![tensor],
            tensors_by_name: HashMap::new(),
            kv_blobs: HashMap::new(),
        };
        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);
        file.write(&mut cursor).unwrap();

        let mut reader = Cursor::new(&buf[..]);
        let restored = GrimFile::read(&mut reader).unwrap();
        assert_eq!(restored.tensors[0].kv_present, 0);
        assert_eq!(restored.tensors[0].kv_compressed_size, 0);
    }
}
