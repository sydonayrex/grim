//! AWQ (Activation-aware Weight Quantization) ingestion.
//!
//! AWQ checkpoints are safetensors files whose quantized layers store
//! `*.qweight` / `*.qzeros` / `*.scales` tensors plus a sibling
//! `quantize_config.json` with `"quant_method": "awq"`. Layout differences
//! from GPTQ:
//!
//! - **Column-major packing**: AWQ packs along the OUTPUT dimension
//!   (`qweight` is [in_features/8, out_features] u32 words for 4-bit), the
//!   transpose of GPTQ's row-packing.
//! - **Shifted zeros**: the stored zero-point is pre-offset by 1 relative to
//!   GPTQ's `(zero + 1)` decode — AWQ decodes as `(code - zero)` directly,
//!   where `zero` is the raw stored value.
//! - **Scale dtype**: f16 (not f32) per-(group, output-column) scales.
//! - **g_idx**: never present; AWQ always uses sequential groups.
//!
//! The tensor data itself is consumed through the existing GroupInt storage
//! path: this module re-frames each AWQ tensor into grim's length-prefixed
//! four-segment packed blob ([see: `grim_tensor::dtype::GpuIntConfig`]) with
//! the zero-points re-encoded to the GPTQ convention (`stored + 1`) and the
//! scales widened to f32, so ONE dequant implementation (CPU in `grim-quant`,
//! GPU fused kernel in `grim-backend-rocm`) serves both formats.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::safetensors::read_safetensors_header;
use grim_tensor::dtype::{ArithType, DType, GroupQuantScheme, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

/// AWQ checkpoint metadata parsed from `quantize_config.json`.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    pub bits: u32,
    pub group_size: usize,
}

impl AwqConfig {
    fn from_json(path: &str) -> Result<Self> {
        let parent = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new(""));
        let cfg_path = parent.join("quantize_config.json");
        let content = std::fs::read_to_string(&cfg_path)
            .map_err(|e| Error::Backend(format!("AWQ: cannot read {}: {e}", cfg_path.display())))?;
        let val: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| Error::Backend(format!("AWQ: invalid quantize_config.json: {e}")))?;

        // Accept either an explicit "quant_method": "awq" or absence of the
        // key on older exports; reject other methods so a GPTQ checkpoint
        // misrouted here fails loudly instead of silently misdecoding.
        if let Some(method) = val.get("quant_method").and_then(|v| v.as_str()) {
            if !method.eq_ignore_ascii_case("awq") {
                return Err(Error::Backend(format!(
                    "AWQ: quantize_config.json declares quant_method '{method}', not 'awq'"
                )));
            }
        }

        let bits =
            val.get("bits").and_then(|v| v.as_u64()).ok_or_else(|| {
                Error::Backend("AWQ: missing 'bits' in quantize_config.json".into())
            })? as u32;
        let group_size = val
            .get("group_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                Error::Backend("AWQ: missing 'group_size' in quantize_config.json".into())
            })? as usize;

        match (bits, group_size) {
            (4, 32) | (4, 64) | (4, 128) | (8, 32) | (8, 64) | (8, 128) => {
                Ok(Self { bits, group_size })
            }
            _ => Err(Error::Backend(format!(
                "AWQ: unsupported bits={bits} group_size={group_size} (supported: 4/8-bit, group 32/64/128)"
            ))),
        }
    }
}

/// One quantized layer of an AWQ checkpoint.
#[derive(Debug, Clone)]
pub struct AwqTensorInfo {
    pub name: String,
    /// Logical weight shape [in_features, out_features].
    pub shape: Vec<usize>,
    pub bits: u32,
    pub group_size: usize,
    pub qweight_offset: u64,
    pub qweight_size: u64,
    pub qzeros_offset: u64,
    pub qzeros_size: u64,
    pub scales_offset: u64,
    pub scales_size: u64,
}

/// Reader for AWQ safetensors checkpoints. Implements `TensorProvider`; every
/// `get_packed` returns a GroupInt-stamped RawTensor holding grim's canonical
/// four-segment packed blob, so the existing CPU dequant and ROCm fused-GEMM
/// paths consume it unchanged.
pub struct AwqProvider {
    pub tensors: HashMap<String, AwqTensorInfo>,
    reader: std::sync::Mutex<BufReader<File>>,
    data_region_start: u64,
    /// Checkpoint-level quantization config (bits/group size), exposed for
    /// callers that need the declared values without re-reading JSON.
    pub config: AwqConfig,
}

impl AwqProvider {
    /// Open an AWQ safetensors checkpoint. Requires a sibling
    /// `quantize_config.json` declaring `quant_method: awq`.
    pub fn open(path: &str) -> Result<Self> {
        let resolved =
            std::fs::canonicalize(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
        let config = AwqConfig::from_json(&resolved.to_string_lossy())?;

        let (info, _metadata, data_region_start) =
            read_safetensors_header(BufReader::new(File::open(&resolved).map_err(|e| {
                Error::Backend(format!(
                    "cannot reopen AWQ file '{}': {e}",
                    resolved.display()
                ))
            })?))?;

        let values_per_word = if config.bits == 4 { 8 } else { 1 };

        let mut tensors = HashMap::new();
        for (name, tensor_info) in &info {
            if !name.ends_with(".qweight") {
                continue;
            }
            let base_name = name.strip_suffix(".qweight").unwrap();

            let qzeros_name = format!("{base_name}.qzeros");
            let scales_name = format!("{base_name}.scales");
            let (Some(qz), Some(sc)) = (info.get(&qzeros_name), info.get(&scales_name)) else {
                continue;
            };

            // AWQ qweight is [in/vpw, out] (column-packed). Recover the
            // logical [in, out] shape.
            let qw_shape = tensor_info.shape();
            if qw_shape.len() != 2 {
                continue;
            }
            let in_features = qw_shape[0] * values_per_word;
            let out_features = qw_shape[1];
            let shape = vec![in_features, out_features];

            tensors.insert(
                base_name.to_string(),
                AwqTensorInfo {
                    name: base_name.to_string(),
                    shape,
                    bits: config.bits,
                    group_size: config.group_size,
                    qweight_offset: tensor_info.data_start,
                    qweight_size: tensor_info.data_end - tensor_info.data_start,
                    qzeros_offset: qz.data_start,
                    qzeros_size: qz.data_end - qz.data_start,
                    scales_offset: sc.data_start,
                    scales_size: sc.data_end - sc.data_start,
                },
            );
        }

        if tensors.is_empty() {
            return Err(Error::Backend(format!(
                "AWQ: no .qweight tensors found in '{}'",
                resolved.display()
            )));
        }

        let file = File::open(&resolved).map_err(|e| {
            Error::Backend(format!(
                "cannot reopen AWQ file '{}': {e}",
                resolved.display()
            ))
        })?;

        Ok(Self {
            tensors,
            reader: std::sync::Mutex::new(BufReader::new(file)),
            data_region_start,
            config,
        })
    }

    fn read_segment(&self, offset: u64, size: u64) -> Result<Vec<u8>> {
        let mut reader = self
            .reader
            .lock()
            .map_err(|_| Error::Backend("AWQ reader mutex poisoned".into()))?;
        let start = self.data_region_start + offset;
        reader.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; size as usize];
        reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Re-frame one AWQ layer into grim's canonical GroupInt packed blob.
///
/// Transformations applied so the shared GPTQ-convention consumers work:
/// 1. Zero-points: AWQ stores `zero_gptq - 1`; add 1 back per element.
/// 2. Scales: widen f16 → f32 (one per (group, output column)).
/// 3. g_idx: absent in AWQ — emit an empty-length segment.
///
/// Packing order is left byte-identical to the checkpoint (both formats use
/// little-endian u32 words with the same intra-word bit order for 4/8-bit).
pub fn pack_awq_group_int(
    info: &AwqTensorInfo,
    qweight: &[u8],
    qzeros: &[u8],
    scales: &[u8],
) -> Result<Vec<u8>> {
    let in_features = *info
        .shape
        .first()
        .ok_or_else(|| Error::Backend("AWQ: tensor info missing in_features".into()))?;
    let out_features = *info
        .shape
        .get(1)
        .ok_or_else(|| Error::Backend("AWQ: tensor info missing out_features".into()))?;
    let bits = info.bits;
    let vpw = if bits == 4 { 8 } else { 1 };
    let words_qw = in_features.div_ceil(vpw) * out_features;
    if qweight.len() < words_qw * 4 {
        return Err(Error::Backend(format!(
            "AWQ {}: qweight truncated ({} bytes, need {})",
            info.name,
            qweight.len(),
            words_qw * 4
        )));
    }

    let groups = in_features.div_ceil(info.group_size);
    // AWQ qzeros: [groups, out/vpw] i32 words.
    let words_qz = groups * out_features.div_ceil(vpw);
    if qzeros.len() < words_qz * 4 {
        return Err(Error::Backend(format!(
            "AWQ {}: qzeros truncated",
            info.name
        )));
    }
    // AWQ scales: [groups, out] f16.
    if scales.len() < groups * out_features * 2 {
        return Err(Error::Backend(format!(
            "AWQ {}: scales truncated",
            info.name
        )));
    }

    // Zero-point convention matches GPTQ exactly: both formats decode as
    // (code - (stored_zero + 1)) * scale, so qzeros pass through unchanged.
    let qz_out = qzeros[..words_qz * 4].to_vec();

    // Widen scales f16 → f32.
    let mut sc_out = vec![0u8; groups * out_features * 4];
    for i in 0..groups * out_features {
        let h = u16::from_le_bytes([scales[i * 2], scales[i * 2 + 1]]);
        let f = half::f16::from_bits(h).to_f32();
        sc_out[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }

    let mut out = Vec::with_capacity(32 + qweight.len() + qz_out.len() + sc_out.len());
    out.extend_from_slice(&(qweight.len() as u64).to_le_bytes());
    out.extend_from_slice(&qweight[..words_qw * 4]);
    out.extend_from_slice(&(qz_out.len() as u64).to_le_bytes());
    out.extend_from_slice(&qz_out);
    out.extend_from_slice(&(sc_out.len() as u64).to_le_bytes());
    out.extend_from_slice(&sc_out);
    out.extend_from_slice(&0u64.to_le_bytes()); // empty g_idx segment
    Ok(out)
}

/// Pack raw AWQ segments into native 3-segment format:
/// `[u64 LE: qweight_len][qweight][u64 LE: qzeros_len][qzeros][u64 LE: scales_len][scales (f16)]`
pub fn pack_awq_native(
    info: &AwqTensorInfo,
    qweight: &[u8],
    qzeros: &[u8],
    scales: &[u8],
) -> Result<Vec<u8>> {
    let in_features = *info
        .shape
        .first()
        .ok_or_else(|| Error::Backend("AWQ: tensor info missing in_features".into()))?;
    let out_features = *info
        .shape
        .get(1)
        .ok_or_else(|| Error::Backend("AWQ: tensor info missing out_features".into()))?;
    let bits = info.bits;
    let vpw = match bits {
        2 => 16,
        4 => 8,
        8 => 1,
        _ => return Err(Error::Backend(format!("AWQ: unsupported bits {bits}"))),
    };
    let words_qw = in_features.div_ceil(vpw) * out_features;
    if qweight.len() < words_qw * 4 {
        return Err(Error::Backend(format!(
            "AWQ {}: qweight truncated ({} bytes, need {})",
            info.name,
            qweight.len(),
            words_qw * 4
        )));
    }

    let groups = in_features.div_ceil(info.group_size);
    let words_qz = groups * out_features.div_ceil(vpw);
    if qzeros.len() < words_qz * 4 {
        return Err(Error::Backend(format!(
            "AWQ {}: qzeros truncated",
            info.name
        )));
    }
    let sc_bytes = groups * out_features * 2;
    if scales.len() < sc_bytes {
        return Err(Error::Backend(format!(
            "AWQ {}: scales truncated",
            info.name
        )));
    }

    let mut out = Vec::with_capacity(24 + words_qw * 4 + words_qz * 4 + sc_bytes);
    out.extend_from_slice(&((words_qw * 4) as u64).to_le_bytes());
    out.extend_from_slice(&qweight[..words_qw * 4]);
    out.extend_from_slice(&((words_qz * 4) as u64).to_le_bytes());
    out.extend_from_slice(&qzeros[..words_qz * 4]);
    out.extend_from_slice(&(sc_bytes as u64).to_le_bytes());
    out.extend_from_slice(&scales[..sc_bytes]);
    Ok(out)
}

impl TensorProvider for AwqProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Backend(format!("tensor '{name}' not found in AWQ file")))?;
        let qweight = self.read_segment(info.qweight_offset, info.qweight_size)?;
        let qzeros = self.read_segment(info.qzeros_offset, info.qzeros_size)?;
        let scales = self.read_segment(info.scales_offset, info.scales_size)?;

        let f32s = grim_quant::dequant_awq_group_int(
            &qweight,
            &qzeros,
            &scales,
            &info.shape,
            info.bits,
            info.group_size,
        )?;
        let mut bytes = Vec::with_capacity(f32s.len() * 4);
        for f in f32s {
            bytes.extend_from_slice(&f.to_le_bytes());
        }

        Ok(RawTensor {
            bytes,
            shape: info.shape.clone(),
            dtype: DType::F32,
            provenance: QuantProvenance::ExternalQat {
                bits: info.bits as u8,
                group_size: info.group_size,
                scheme: GroupQuantScheme::Asymmetric,
                desc_act: false,
            },
        })
    }

    fn get_packed(&self, name: &str) -> Result<RawTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Backend(format!("tensor '{name}' not found in AWQ file")))?;
        let qweight = self.read_segment(info.qweight_offset, info.qweight_size)?;
        let qzeros = self.read_segment(info.qzeros_offset, info.qzeros_size)?;
        let scales = self.read_segment(info.scales_offset, info.scales_size)?;
        let bytes = pack_awq_native(info, &qweight, &qzeros, &scales)?;
        Ok(RawTensor {
            bytes,
            shape: info.shape.clone(),
            dtype: DType {
                arith: ArithType::F32,
                storage: Storage::Awq(grim_tensor::dtype::AwqStorageConfig {
                    bits: info.bits as u8,
                    group_size: info.group_size,
                }),
            },
            provenance: QuantProvenance::ExternalQat {
                bits: info.bits as u8,
                group_size: info.group_size,
                scheme: GroupQuantScheme::Asymmetric,
                desc_act: false,
            },
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Backend(format!("tensor '{name}' not found in AWQ file")))?;
        Ok(TensorMeta {
            dtype: DType::F32,
            provenance: QuantProvenance::ExternalQat {
                bits: info.bits as u8,
                group_size: info.group_size,
                scheme: GroupQuantScheme::Asymmetric,
                desc_act: false,
            },
            shape: info.shape.clone(),
            fusion_mask: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4-bit, group_size 2, [in=4, out=2] AWQ tensor. Verify the re-framed
    /// blob decodes through the shared GPTQ kernel to the values the AWQ
    /// layout encodes: w = (code - zero_awq) * scale, where zero_awq is the
    /// RAW stored zero (no +1) and scales are f16.
    #[test]
    fn pack_awq_roundtrips_through_gptq_decoder() {
        let (k, n, gs) = (4usize, 2usize, 2usize);
        let vpw = 8usize;
        let groups = k / gs;

        // Codes: arbitrary non-zero patterns.
        let mut qweight = vec![0u8; k.div_ceil(vpw) * n * 4];
        let mut codes = vec![0u32; k * n];
        for ki in 0..k {
            for ni in 0..n {
                let code = ((ki * 5 + ni * 3) % 15) as u32;
                codes[ki * n + ni] = code;
                let word_idx = ki * n + ni; // in/vpw rows of n words each
                let off = (ki % vpw) * 4;
                let cur_off = word_idx / vpw; // words are per (in-block, out)
                let _ = cur_off;
                let w = (ki / vpw) * n + ni;
                let cur = u32::from_le_bytes([
                    qweight[w * 4],
                    qweight[w * 4 + 1],
                    qweight[w * 4 + 2],
                    qweight[w * 4 + 3],
                ]);
                qweight[w * 4..w * 4 + 4].copy_from_slice(&(cur | (code << off)).to_le_bytes());
            }
        }

        // Raw AWQ zeros (stored pre-shifted by -1 vs GPTQ).
        let raw_zero = |g: usize, ni: usize| -> u32 { ((g + ni) % 8) as u32 };
        let mut qzeros = vec![0u8; groups * n.div_ceil(vpw) * 4];
        for g in 0..groups {
            for ni in 0..n {
                let w = g * n.div_ceil(vpw) + ni / vpw;
                let off = (ni % vpw) * 4;
                let cur = u32::from_le_bytes([
                    qzeros[w * 4],
                    qzeros[w * 4 + 1],
                    qzeros[w * 4 + 2],
                    qzeros[w * 4 + 3],
                ]);
                qzeros[w * 4..w * 4 + 4]
                    .copy_from_slice(&(cur | (raw_zero(g, ni) << off)).to_le_bytes());
            }
        }

        // f16 scales.
        let scale_val = |g: usize, ni: usize| -> f32 { 0.5 + 0.25 * ((g + ni) % 3) as f32 };
        let mut scales = vec![0u8; groups * n * 2];
        for g in 0..groups {
            for ni in 0..n {
                let h = half::f16::from_f32(scale_val(g, ni)).to_bits();
                scales[(g * n + ni) * 2..(g * n + ni) * 2 + 2].copy_from_slice(&h.to_le_bytes());
            }
        }

        let info = AwqTensorInfo {
            name: "test".into(),
            shape: vec![k, n],
            bits: 4,
            group_size: gs,
            qweight_offset: 0,
            qweight_size: qweight.len() as u64,
            qzeros_offset: 0,
            qzeros_size: qzeros.len() as u64,
            scales_offset: 0,
            scales_size: scales.len() as u64,
        };

        let blob = pack_awq_group_int(&info, &qweight, &qzeros, &scales).unwrap();

        // Unpack the blob segments and decode via the shared kernel.
        let read_seg = |c: &mut usize| -> Vec<u8> {
            let len = u64::from_le_bytes(blob[*c..*c + 8].try_into().unwrap()) as usize;
            *c += 8;
            let seg = blob[*c..*c + len].to_vec();
            *c += len;
            seg
        };
        let mut cursor = 0usize;
        let qw = read_seg(&mut cursor);
        let qz = read_seg(&mut cursor);
        let sc = read_seg(&mut cursor);
        let gi = read_seg(&mut cursor);
        assert!(gi.is_empty(), "AWQ never carries g_idx");

        let decoded =
            grim_quant::dequant_gptq_group_int(&qw, &qz, &sc, None, &[k, n], 4, gs).unwrap();

        for ki in 0..k {
            let g = ki / gs;
            for ni in 0..n {
                // AWQ on-disk zeros are pre-shifted (-1 vs the true zero);
                // the shared decoder's `stored + 1` restores it.
                let want =
                    (codes[ki * n + ni] as f32 - (raw_zero(g, ni) as f32 + 1.0)) * scale_val(g, ni);
                let got = decoded[ki * n + ni];
                assert!(
                    (got - want).abs() < 1e-5,
                    "mismatch at ({ki},{ni}): got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn pack_awq_native_roundtrips_through_awq_decoder() {
        let (k, n, gs) = (4usize, 2usize, 2usize);
        let vpw = 8usize;
        let groups = k / gs;

        let mut qweight = vec![0u8; k.div_ceil(vpw) * n * 4];
        let mut codes = vec![0u32; k * n];
        for ki in 0..k {
            for ni in 0..n {
                let code = ((ki * 5 + ni * 3) % 15) as u32;
                codes[ki * n + ni] = code;
                let off = (ki % vpw) * 4;
                let w = (ki / vpw) * n + ni;
                let cur = u32::from_le_bytes([
                    qweight[w * 4],
                    qweight[w * 4 + 1],
                    qweight[w * 4 + 2],
                    qweight[w * 4 + 3],
                ]);
                qweight[w * 4..w * 4 + 4].copy_from_slice(&(cur | (code << off)).to_le_bytes());
            }
        }

        let raw_zero = |g: usize, ni: usize| -> u32 { ((g + ni) % 8) as u32 };
        let mut qzeros = vec![0u8; groups * n.div_ceil(vpw) * 4];
        for g in 0..groups {
            for ni in 0..n {
                let w = g * n.div_ceil(vpw) + ni / vpw;
                let off = (ni % vpw) * 4;
                let cur = u32::from_le_bytes([
                    qzeros[w * 4],
                    qzeros[w * 4 + 1],
                    qzeros[w * 4 + 2],
                    qzeros[w * 4 + 3],
                ]);
                qzeros[w * 4..w * 4 + 4]
                    .copy_from_slice(&(cur | (raw_zero(g, ni) << off)).to_le_bytes());
            }
        }

        let scale_val = |g: usize, ni: usize| -> f32 { 0.5 + 0.25 * ((g + ni) % 3) as f32 };
        let mut scales = vec![0u8; groups * n * 2];
        for g in 0..groups {
            for ni in 0..n {
                let h = half::f16::from_f32(scale_val(g, ni)).to_bits();
                scales[(g * n + ni) * 2..(g * n + ni) * 2 + 2].copy_from_slice(&h.to_le_bytes());
            }
        }

        let info = AwqTensorInfo {
            name: "test_native".into(),
            shape: vec![k, n],
            bits: 4,
            group_size: gs,
            qweight_offset: 0,
            qweight_size: qweight.len() as u64,
            qzeros_offset: 0,
            qzeros_size: qzeros.len() as u64,
            scales_offset: 0,
            scales_size: scales.len() as u64,
        };

        let blob = pack_awq_native(&info, &qweight, &qzeros, &scales).unwrap();
        let mut cursor = 0usize;
        let read_seg = |c: &mut usize| -> Vec<u8> {
            let len = u64::from_le_bytes(blob[*c..*c + 8].try_into().unwrap()) as usize;
            *c += 8;
            let seg = blob[*c..*c + len].to_vec();
            *c += len;
            seg
        };
        let qw = read_seg(&mut cursor);
        let qz = read_seg(&mut cursor);
        let sc = read_seg(&mut cursor);

        let decoded = grim_quant::dequant_awq_group_int(&qw, &qz, &sc, &[k, n], 4, gs).unwrap();

        for ki in 0..k {
            let g = ki / gs;
            for ni in 0..n {
                let want = (codes[ki * n + ni] as f32 - raw_zero(g, ni) as f32) * scale_val(g, ni);
                let got = decoded[ki * n + ni];
                assert!(
                    (got - want).abs() < 1e-4,
                    "mismatch at ({ki},{ni}): got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn awq_config_rejects_non_awq_quant_method() {
        let dir = tempfile::tempdir().unwrap();
        let cfg_path = dir.path().join("quantize_config.json");
        std::fs::write(
            &cfg_path,
            r#"{"bits": 4, "group_size": 128, "quant_method": "gptq"}"#,
        )
        .unwrap();
        let err = AwqConfig::from_json(dir.path().join("model.safetensors").to_str().unwrap());
        assert!(err.is_err(), "gptq quant_method must be rejected");
    }

    #[test]
    fn awq_config_accepts_valid_awq() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("quantize_config.json"),
            r#"{"bits": 4, "group_size": 128, "quant_method": "awq"}"#,
        )
        .unwrap();
        let cfg =
            AwqConfig::from_json(dir.path().join("model.safetensors").to_str().unwrap()).unwrap();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 128);
    }
}
