//! Per-tensor capability descriptors for the `.grim` spec.
//!
//! The on-disk wire format stays at version 1 (`GRIM\x01`, fixed header +
//! JSON metadata + tensor registry). All advanced capabilities from the
//! spec (per-row scales, mixed-bitwidth rows, two-level residual backups,
//! GPTQ ordering, outlier compression, fusion mask, optional payload
//! compression) ride the JSON metadata layer as declarations on a
//! per-tensor extension struct. Backends read these declarations to pick
//! dequant kernels; the bytes on disk never change.
//!
//! Field naming and enum values track `docs/grim-file.md` decisions
//! D2–D15. Each field defaults to its legacy meaning when zeroed so a
//! reader that ignores the extension struct behaves exactly like a plain
//! V1 reader.

use serde_json::Value;

// ---------------------------------------------------------------------------
// Enums — kept small and explicit so the on-JSON representation is stable.
// ---------------------------------------------------------------------------

/// Per-row scale dtype. Spec D2 + D15.
///
/// `U8` is the default symmetric mode; `F16` is reserved for a future
/// asymmetric-quant fallback and is documented in the spec as "writers
/// MUST set `row_scale_dtype = 0`" until asymmetric lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowScaleDtype {
    /// Symmetric u8 scale (default).
    U8 = 0,
    /// Reserved asymmetric f16 scale; not emitted by current writers.
    F16 = 1,
}

impl RowScaleDtype {
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::U8),
            1 => Some(Self::F16),
            _ => None,
        }
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Per-row bitwidth assignment mode. Spec D3 + Phase 3.
///
/// `Uniform` is the legacy whole-tensor mode: every row uses
/// `default_bpw` bits. `PerRowTable` reads per-row bitwidths from a
/// `[u8; row_count]` table stored at `bpw_table_offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerRowBpwMode {
    /// Whole-tensor uniform bitwidth at `default_bpw`.
    Uniform = 0,
    /// Per-row table at `bpw_table_offset` (each entry in {2..8}).
    PerRowTable = 1,
}

impl PerRowBpwMode {
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Uniform),
            1 => Some(Self::PerRowTable),
            _ => None,
        }
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Outlier index encoding. Spec D6 + Phase 5.
///
/// `FlatU32` is the legacy 6-byte record (`u32 index | f16 value`) that
/// matches `crate::format::GrimOutlier`. `DeltaVarint` is the compressed
/// path from Phase 5: delta-varint sorted indices plus delta-u8 residual
/// values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierIndexEncoding {
    /// Legacy flat u32 index + f16 value (6 bytes/outlier).
    FlatU32 = 1,
    /// Phase 5 compressed path: delta-varint indices + delta-u8 values.
    DeltaVarint = 0,
}

impl OutlierIndexEncoding {
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::DeltaVarint),
            1 => Some(Self::FlatU32),
            _ => None,
        }
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Per-tensor payload compression. Spec D13 + Phase 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadCompression {
    /// Uncompressed (default).
    Raw = 0,
    /// zstd per-tensor payload.
    Zstd = 1,
}

impl PayloadCompression {
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Raw),
            1 => Some(Self::Zstd),
            _ => None,
        }
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Kernel-side layout hint. Spec field `layout_hint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutHintTag {
    Default = 0,
    WavefrontTiled = 1,
    BlockSparse = 2,
}

impl LayoutHintTag {
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Default),
            1 => Some(Self::WavefrontTiled),
            2 => Some(Self::BlockSparse),
            _ => None,
        }
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// One backup (residual) layer descriptor. Spec D4 + D5 + Phase 4.
///
/// Up to two of these attach per tensor. Each describes an additive
/// correction stream packed at `bpw` bits with its own per-row u8 scale
/// region.
#[derive(Debug, Clone, PartialEq)]
pub struct BackupLayer {
    /// Offset of the packed backup codes inside the payload region,
    /// Wave64-aligned. 0 = absent.
    pub codes_offset: u64,
    /// Size of the packed backup codes in bytes.
    pub codes_size: u64,
    /// Bitwidth the backup codes are packed at (typically 8).
    pub bpw: u8,
    /// Offset of the per-row scale bytes for this backup. 0 = absent.
    pub scale_offset: u64,
    /// Size of the per-row scale bytes for this backup.
    pub scale_size: u64,
}

impl Default for BackupLayer {
    fn default() -> Self {
        Self {
            codes_offset: 0,
            codes_size: 0,
            bpw: 0,
            scale_offset: 0,
            scale_size: 0,
        }
    }
}

impl BackupLayer {
    /// `true` if this backup layer is present on disk.
    pub fn is_present(&self) -> bool {
        self.bpw > 0 && self.codes_size > 0
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "codes_offset": self.codes_offset,
            "codes_size": self.codes_size,
            "bpw": self.bpw,
            "scale_offset": self.scale_offset,
            "scale_size": self.scale_size,
        })
    }

    pub fn from_json(value: &Value) -> Option<Self> {
        let obj = value.as_object()?;
        Some(Self {
            codes_offset: obj.get("codes_offset")?.as_u64()?,
            codes_size: obj.get("codes_size")?.as_u64()?,
            bpw: obj.get("bpw")?.as_u64()? as u8,
            scale_offset: obj.get("scale_offset").and_then(|v| v.as_u64()).unwrap_or(0),
            scale_size: obj.get("scale_size").and_then(|v| v.as_u64()).unwrap_or(0),
        })
    }
}

/// Four-word opaque kernel dispatch hint. Spec field `layout_descriptor`.
///
/// The reader does not interpret these; they are passed through to the
/// backend kernel verbatim. Defaults to all zeros (= "let the kernel
/// decide").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutDescriptor(pub [u32; 4]);

impl Default for LayoutDescriptor {
    fn default() -> Self {
        Self([0, 0, 0, 0])
    }
}

// ---------------------------------------------------------------------------
// GrimTensorExt — per-tensor capability declaration
// ---------------------------------------------------------------------------

/// Per-tensor capability extension.
///
/// One of these attaches to each tensor in the file via the JSON metadata
/// key `grim.ext.entries` (an array indexed by tensor name in
/// `tensor_name`). Every field defaults to its legacy meaning when zeroed,
/// so a reader that never reads extensions sees a plain version-1 file.
///
/// Field set covers spec Phases 2–7:
/// - Phase 2 per-row scales: `row_count`, `row_stride`, `block_size`,
///   `scale_offset`, `scale_size`, `row_scale_dtype`
/// - Phase 3 per-row mixed bitwidth: `per_row_bpw_mode`, `default_bpw`,
///   `bpw_table_offset`, `bpw_table_count`, `own_bpw_table`
/// - Phase 4 backups + GPTQ ordering: `backup1`, `backup2`, `gptq_ordered`
/// - Phase 5 outlier compression: `outlier_index_encoding`,
///   `outlier_residual_bpw`
/// - Phase 6 payload compression: `compression`
/// - Phase 7 fusion dispatch: `fusion_mask`, `layout_hint`,
///   `layout_descriptor`
#[derive(Debug, Clone, PartialEq)]
pub struct GrimTensorExt {
    /// Tensor name this extension applies to. Matches the V1 registry
    /// entry's `name` field 1:1.
    pub tensor_name: String,

    /// Π(shape[:-1]); 1 if the tensor is 1-D. Used to size the per-row
    /// scale region and the per-row bitwidth table.
    pub row_count: u64,
    /// shape[-1] — number of elements per row.
    pub row_stride: u64,
    /// 0 = per-row scale mode (EXL2-style); 32/64/128/256 = block mode.
    pub block_size: u16,

    /// Uniform vs per-row mixed bitwidth.
    pub per_row_bpw_mode: PerRowBpwMode,
    /// Whether the bpw table is owned by this entry (1) or shared in
    /// metadata (0).
    pub own_bpw_table: u8,
    /// Bits per weight when `per_row_bpw_mode = Uniform`. Range 2..=8.
    pub default_bpw: u8,

    /// Per-row scale dtype.
    pub row_scale_dtype: RowScaleDtype,
    /// Offset of the per-row scale bytes inside the payload region.
    pub scale_offset: u64,
    /// Size of the per-row scale bytes.
    pub scale_size: u64,

    /// 1 if the codes were quantized through GPTQ with inverse-Hessian
    /// ordering. Kernels apply `gptq_inv_r` to backups only when this is
    /// set; spec D8 + Q3.
    pub gptq_ordered: u8,

    /// Outlier index/value encoding.
    pub outlier_index_encoding: OutlierIndexEncoding,
    /// 8 or 16; 0 = no residual encoding (raw f16 values, legacy path).
    pub outlier_residual_bpw: u8,

    /// Per-tensor payload compression.
    pub compression: PayloadCompression,

    /// bit0 = RmsNormMatMul, bit1 = QkvAttention. Kernel dispatch hint.
    /// Populate via [`fusion_mask_from_ops`].
    pub fusion_mask: u8,
    /// Layout hint tag.
    pub layout_hint: LayoutHintTag,
    /// Opaque 4-word kernel dispatch descriptor.
    pub layout_descriptor: LayoutDescriptor,

    /// Up to two residual backup layers. Absent layers have `bpw == 0`.
    pub backup1: BackupLayer,
    pub backup2: BackupLayer,
}

/// bit0 = RmsNormMatMul, bit1 = QkvAttention (spec §GrimTensorEntry V3
/// `fusion_mask`).
pub const FUSION_MASK_RMSNORM_MATMUL: u8 = 0b01;
pub const FUSION_MASK_QKV_ATTENTION: u8 = 0b10;

/// Translate a list of `GrimFusionOp`s into the spec's `fusion_mask`
/// bitfield (Phase 7.1).
///
/// - `RmsNormMatMul` → bit0
/// - `QkvAttention` → bit1
///
/// Unknown ops are ignored. Duplicate ops OR into the same bit (idempotent).
pub fn fusion_mask_from_ops(ops: &[crate::gguf::GrimFusionOp]) -> u8 {
    let mut mask = 0u8;
    for op in ops {
        match op {
            crate::gguf::GrimFusionOp::RmsNormMatMul => mask |= FUSION_MASK_RMSNORM_MATMUL,
            crate::gguf::GrimFusionOp::QkvAttention => mask |= FUSION_MASK_QKV_ATTENTION,
        }
    }
    mask
}

impl Default for GrimTensorExt {
    fn default() -> Self {
        Self {
            tensor_name: String::new(),
            row_count: 0,
            row_stride: 0,
            block_size: 0,
            per_row_bpw_mode: PerRowBpwMode::Uniform,
            own_bpw_table: 0,
            default_bpw: 0,
            row_scale_dtype: RowScaleDtype::U8,
            scale_offset: 0,
            scale_size: 0,
            gptq_ordered: 0,
            outlier_index_encoding: OutlierIndexEncoding::FlatU32,
            outlier_residual_bpw: 0,
            compression: PayloadCompression::Raw,
            fusion_mask: 0,
            layout_hint: LayoutHintTag::Default,
            layout_descriptor: LayoutDescriptor::default(),
            backup1: BackupLayer::default(),
            backup2: BackupLayer::default(),
        }
    }
}

impl GrimTensorExt {
    /// Convenience: does this tensor declare anything beyond the legacy
    /// version-1 surface? A reader can use this to short-circuit the
    /// extension lookup table for plain tensors.
    pub fn is_legacy(&self) -> bool {
        self.block_size == 0
            && self.per_row_bpw_mode == PerRowBpwMode::Uniform
            && self.scale_size == 0
            && self.gptq_ordered == 0
            && self.outlier_index_encoding == OutlierIndexEncoding::FlatU32
            && self.outlier_residual_bpw == 0
            && self.compression == PayloadCompression::Raw
            && self.fusion_mask == 0
            && self.layout_hint == LayoutHintTag::Default
            && !self.backup1.is_present()
            && !self.backup2.is_present()
    }

    /// Encode this extension to a JSON object for the metadata layer.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "tensor_name": self.tensor_name,
            "row_count": self.row_count,
            "row_stride": self.row_stride,
            "block_size": self.block_size,
            "per_row_bpw_mode": self.per_row_bpw_mode.as_u8(),
            "own_bpw_table": self.own_bpw_table,
            "default_bpw": self.default_bpw,
            "row_scale_dtype": self.row_scale_dtype.as_u8(),
            "scale_offset": self.scale_offset,
            "scale_size": self.scale_size,
            "gptq_ordered": self.gptq_ordered,
            "outlier_index_encoding": self.outlier_index_encoding.as_u8(),
            "outlier_residual_bpw": self.outlier_residual_bpw,
            "compression": self.compression.as_u8(),
            "fusion_mask": self.fusion_mask,
            "layout_hint": self.layout_hint.as_u8(),
            "layout_descriptor": [
                self.layout_descriptor.0[0],
                self.layout_descriptor.0[1],
                self.layout_descriptor.0[2],
                self.layout_descriptor.0[3],
            ],
            "backup1": self.backup1.to_json(),
            "backup2": self.backup2.to_json(),
        })
    }

    /// Decode an extension from a JSON object. Returns `None` if the value
    /// is not an object or the `tensor_name` field is missing.
    pub fn from_json(value: &Value) -> Option<Self> {
        let obj = value.as_object()?;
        let tensor_name = obj.get("tensor_name")?.as_str()?.to_string();

        let pick_u8 = |k: &str, default: u8| {
            obj.get(k).and_then(|v| v.as_u64()).map(|v| v as u8).unwrap_or(default)
        };
        let pick_u64 = |k: &str| obj.get(k).and_then(|v| v.as_u64()).unwrap_or(0);

        let layout_descriptor = if let Some(arr) = obj.get("layout_descriptor").and_then(|v| v.as_array()) {
            let mut buf = [0u32; 4];
            for (i, slot) in arr.iter().enumerate().take(4) {
                buf[i] = slot.as_u64().unwrap_or(0) as u32;
            }
            LayoutDescriptor(buf)
        } else {
            LayoutDescriptor::default()
        };

        Some(Self {
            tensor_name,
            row_count: pick_u64("row_count"),
            row_stride: pick_u64("row_stride"),
            block_size: pick_u64("block_size") as u16,
            per_row_bpw_mode: PerRowBpwMode::from_u8(pick_u8("per_row_bpw_mode", 0))
                .unwrap_or(PerRowBpwMode::Uniform),
            own_bpw_table: pick_u8("own_bpw_table", 0),
            default_bpw: pick_u8("default_bpw", 0),
            row_scale_dtype: RowScaleDtype::from_u8(pick_u8("row_scale_dtype", 0))
                .unwrap_or(RowScaleDtype::U8),
            scale_offset: pick_u64("scale_offset"),
            scale_size: pick_u64("scale_size"),
            gptq_ordered: pick_u8("gptq_ordered", 0),
            outlier_index_encoding: OutlierIndexEncoding::from_u8(
                pick_u8("outlier_index_encoding", 1),
            )
            .unwrap_or(OutlierIndexEncoding::FlatU32),
            outlier_residual_bpw: pick_u8("outlier_residual_bpw", 0),
            compression: PayloadCompression::from_u8(pick_u8("compression", 0))
                .unwrap_or(PayloadCompression::Raw),
            fusion_mask: pick_u8("fusion_mask", 0),
            layout_hint: LayoutHintTag::from_u8(pick_u8("layout_hint", 0))
                .unwrap_or(LayoutHintTag::Default),
            layout_descriptor,
            backup1: obj
                .get("backup1")
                .and_then(BackupLayer::from_json)
                .unwrap_or_default(),
            backup2: obj
                .get("backup2")
                .and_then(BackupLayer::from_json)
                .unwrap_or_default(),
        })
    }
}

// ---------------------------------------------------------------------------
// Delta-varint outlier codec (spec D6 + Phase 5.2/5.3)
// ---------------------------------------------------------------------------

/// Encode a sorted slice of `(index, value)` outliers using the spec's
/// delta-varint + delta-u8 residual encoding.
///
/// Layout:
/// ```text
/// [ value_dtype : u8 ]                    // 0 = u8 residual, 1 = f16 raw
/// [ count_varint ]                        // number of records
/// [ delta_varint(idx_0) | delta_varint(idx_i - idx_{i-1}) | ... ]
/// [ delta_value_0 | delta_value_1 | ... ] // u8 if dtype=0, else f16
/// ```
///
/// Indices MUST be sorted ascending. The leading `count_varint` lets the
/// decoder know where the index region ends and the value region begins
/// without an out-of-band length.
pub fn encode_outliers_delta_varint(outliers: &[(u32, f32)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(outliers.len() * 2 + 8);

    // Always emit u8 residual mode (the spec's preferred path). If a
    // caller needs raw f16 mode it should use the legacy FlatU32 encoder.
    buf.push(0u8);
    encode_varint(&mut buf, outliers.len() as u64);

    let mut prev_idx: u64 = 0;
    for (idx, _) in outliers.iter() {
        let cur = *idx as u64;
        let delta = cur.wrapping_sub(prev_idx);
        encode_varint(&mut buf, delta);
        prev_idx = cur;
    }

    // Delta-u8 values: each value is the residual between consecutive
    // dequant-base reconstructions. We don't have the base here, so we
    // emit the raw value quantized to u8 in [-128, 127] when it fits,
    // falling back to clamping. The reader reconstructs by adding the
    // delta to the running reconstruction.
    let mut prev_val: f32 = 0.0;
    for (_, value) in outliers.iter() {
        let delta = value - prev_val;
        let clamped = delta.round().clamp(-128.0, 127.0) as i8;
        buf.push(clamped as u8);
        prev_val = *value;
    }

    buf
}

/// Decode a buffer produced by [`encode_outliers_delta_varint`].
///
/// Returns the reconstructed `(index, value)` pairs. Errors out if the
/// buffer is malformed.
pub fn decode_outliers_delta_varint(buf: &[u8]) -> Result<Vec<(u32, f32)>, String> {
    if buf.is_empty() {
        return Err("empty outlier buffer".into());
    }
    let value_dtype = buf[0];
    if value_dtype != 0 {
        return Err(format!(
            "unsupported outlier value dtype {value_dtype}; only u8 residual (0) is implemented"
        ));
    }
    let mut cursor = 1usize;

    let (count, consumed) = decode_varint(&buf[cursor..])?;
    cursor += consumed;
    let count = count as usize;

    let mut indices: Vec<u32> = Vec::with_capacity(count);
    let mut prev_idx: u64 = 0;
    for _ in 0..count {
        if cursor >= buf.len() {
            return Err("outlier index stream truncated".into());
        }
        let (delta, consumed) = decode_varint(&buf[cursor..])?;
        cursor += consumed;
        prev_idx = prev_idx.wrapping_add(delta);
        if prev_idx > u32::MAX as u64 {
            return Err("outlier index overflow".into());
        }
        indices.push(prev_idx as u32);
    }

    let mut result = Vec::with_capacity(count);
    let mut prev_val: f32 = 0.0;
    for idx in indices {
        if cursor >= buf.len() {
            return Err("outlier value stream truncated".into());
        }
        let delta = (buf[cursor] as i8) as f32;
        cursor += 1;
        prev_val += delta;
        result.push((idx, prev_val));
    }
    Ok(result)
}

/// LEB128-style varint encoder. Writes 7 bits per byte with continuation
/// in the high bit.
fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// LEB128 varint decoder. Returns `(value, bytes_consumed)`.
fn decode_varint(buf: &[u8]) -> Result<(u64, usize), String> {
    let mut value: u64 = 0;
    let mut shift = 0;
    for (i, byte) in buf.iter().enumerate() {
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint overflow".into());
        }
    }
    Err("truncated varint".into())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A default-constructed extension is the legacy surface — readers
    /// that never inspect it behave identically to a version-1 reader.
    #[test]
    fn default_extension_is_legacy() {
        let ext = GrimTensorExt::default();
        assert!(ext.is_legacy());
        assert_eq!(ext.outlier_index_encoding, OutlierIndexEncoding::FlatU32);
        assert_eq!(ext.compression, PayloadCompression::Raw);
    }

    /// Phase 2: per-row u8 scales round-trip through JSON.
    #[test]
    fn per_row_scale_ext_round_trips_through_json() {
        let ext = GrimTensorExt {
            tensor_name: "layer.0.weight".into(),
            row_count: 128,
            row_stride: 4096,
            block_size: 0,
            per_row_bpw_mode: PerRowBpwMode::Uniform,
            default_bpw: 4,
            row_scale_dtype: RowScaleDtype::U8,
            scale_offset: 8192,
            scale_size: 128,
            ..Default::default()
        };
        let json = ext.to_json();
        let restored = GrimTensorExt::from_json(&json).expect("round-trip");
        assert_eq!(restored.tensor_name, ext.tensor_name);
        assert_eq!(restored.row_count, 128);
        assert_eq!(restored.row_stride, 4096);
        assert_eq!(restored.block_size, 0);
        assert_eq!(restored.default_bpw, 4);
        assert_eq!(restored.scale_offset, 8192);
        assert_eq!(restored.scale_size, 128);
        assert!(!restored.is_legacy());
    }

    /// Phase 3: per-row mixed-bitwidth mode round-trips.
    #[test]
    fn per_row_bpw_table_ext_round_trips() {
        let ext = GrimTensorExt {
            tensor_name: "layer.1.weight".into(),
            row_count: 64,
            row_stride: 1024,
            per_row_bpw_mode: PerRowBpwMode::PerRowTable,
            default_bpw: 4,
            own_bpw_table: 1,
            ..Default::default()
        };
        let restored = GrimTensorExt::from_json(&ext.to_json()).expect("round-trip");
        assert_eq!(restored.per_row_bpw_mode, PerRowBpwMode::PerRowTable);
        assert_eq!(restored.own_bpw_table, 1);
        assert!(!restored.is_legacy());
    }

    /// Phase 4: backup layers + GPTQ ordering round-trip.
    #[test]
    fn backup_layers_and_gptq_flag_round_trip() {
        let ext = GrimTensorExt {
            tensor_name: "layer.2.weight".into(),
            gptq_ordered: 1,
            backup1: BackupLayer {
                codes_offset: 16384,
                codes_size: 4096,
                bpw: 8,
                scale_offset: 20480,
                scale_size: 64,
            },
            backup2: BackupLayer::default(),
            ..Default::default()
        };
        let restored = GrimTensorExt::from_json(&ext.to_json()).expect("round-trip");
        assert_eq!(restored.gptq_ordered, 1);
        assert!(restored.backup1.is_present());
        assert!(!restored.backup2.is_present());
        assert_eq!(restored.backup1.codes_offset, 16384);
        assert_eq!(restored.backup1.bpw, 8);
        assert!(!restored.is_legacy());
    }

    /// Phase 5: outlier delta-varint codec round-trips a multi-record
    /// sequence including a contiguous run (where delta compression wins).
    #[test]
    fn outlier_delta_varint_round_trips() {
        let outliers = vec![(5u32, 1.0f32), (10, 2.0), (11, 3.0), (12, 4.0)];
        let encoded = encode_outliers_delta_varint(&outliers);
        let decoded = decode_outliers_delta_varint(&encoded).expect("decode");
        assert_eq!(decoded.len(), outliers.len());
        for (got, want) in decoded.iter().zip(outliers.iter()) {
            assert_eq!(got.0, want.0, "index mismatch");
            assert!((got.1 - want.1).abs() < 1.5, "value {} vs {}", got.1, want.1);
        }
    }

    /// Phase 5: single-record round-trip (boundary case).
    #[test]
    fn outlier_delta_varint_single_record() {
        let one = vec![(42u32, 7.0f32)];
        let enc = encode_outliers_delta_varint(&one);
        let dec = decode_outliers_delta_varint(&enc).expect("decode");
        assert_eq!(dec.len(), 1);
        assert_eq!(dec[0].0, 42);
    }

    /// Phase 5: empty input is rejected by the decoder.
    #[test]
    fn outlier_delta_varint_rejects_empty() {
        assert!(decode_outliers_delta_varint(&[]).is_err());
    }

    /// Phase 5: varint primitive round-trips small and large values.
    #[test]
    fn varint_round_trips() {
        for &v in &[0u64, 1, 127, 128, 255, 16384, 1 << 21, u32::MAX as u64, u64::MAX] {
            let mut buf = Vec::new();
            encode_varint(&mut buf, v);
            let (decoded, consumed) = decode_varint(&buf).expect("decode");
            assert_eq!(decoded, v, "varint round-trip failed for {}", v);
            assert_eq!(consumed, buf.len());
        }
    }

    /// Phase 6: payload compression tag round-trips.
    #[test]
    fn compression_tag_round_trips() {
        let ext = GrimTensorExt {
            tensor_name: "layer.3.weight".into(),
            compression: PayloadCompression::Zstd,
            ..Default::default()
        };
        let restored = GrimTensorExt::from_json(&ext.to_json()).expect("round-trip");
        assert_eq!(restored.compression, PayloadCompression::Zstd);
        assert!(!restored.is_legacy());
    }

    /// Phase 7: fusion mask + layout descriptor round-trip.
    #[test]
    fn fusion_mask_and_descriptor_round_trip() {
        let ext = GrimTensorExt {
            tensor_name: "layer.4.weight".into(),
            fusion_mask: 0b11,
            layout_hint: LayoutHintTag::WavefrontTiled,
            layout_descriptor: LayoutDescriptor([1, 2, 3, 4]),
            ..Default::default()
        };
        let restored = GrimTensorExt::from_json(&ext.to_json()).expect("round-trip");
        assert_eq!(restored.fusion_mask, 0b11);
        assert_eq!(restored.layout_hint, LayoutHintTag::WavefrontTiled);
        assert_eq!(restored.layout_descriptor.0, [1, 2, 3, 4]);
    }

    /// Phase 7.1: empty fusion-op list produces mask 0 (no fusion).
    #[test]
    fn fusion_mask_from_empty_ops_is_zero() {
        let mask = fusion_mask_from_ops(&[]);
        assert_eq!(mask, 0);
    }

    /// Phase 7.1: a single QkvAttention op sets only bit1.
    #[test]
    fn fusion_mask_from_qkv_attention_only() {
        let mask = fusion_mask_from_ops(&[crate::gguf::GrimFusionOp::QkvAttention]);
        assert_eq!(mask, FUSION_MASK_QKV_ATTENTION);
        assert_eq!(mask, 0b10);
    }

    /// Phase 7.1: a single RmsNormMatMul op sets only bit0.
    #[test]
    fn fusion_mask_from_rmsnorm_matmul_only() {
        let mask = fusion_mask_from_ops(&[crate::gguf::GrimFusionOp::RmsNormMatMul]);
        assert_eq!(mask, FUSION_MASK_RMSNORM_MATMUL);
        assert_eq!(mask, 0b01);
    }

    /// Phase 7.1: both ops together OR into the full mask 0b11. Duplicate
    /// ops are idempotent.
    #[test]
    fn fusion_mask_from_both_ops_ors_correctly() {
        let mask = fusion_mask_from_ops(&[
            crate::gguf::GrimFusionOp::RmsNormMatMul,
            crate::gguf::GrimFusionOp::QkvAttention,
            crate::gguf::GrimFusionOp::RmsNormMatMul, // duplicate
        ]);
        assert_eq!(mask, 0b11);
    }
}
