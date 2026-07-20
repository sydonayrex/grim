//! GPTQ v2 tensor layout reader for EfficientQAT/GPTQ checkpoints.
//!
//! §7.2: Reads grouped INT weights with asymmetric quantization:
//! - `qweight`: packed low-bit weights
//! - `qzeros`: per-group zero-points
//! - `scales`: per-group scales
//! - `g_idx`: group assignment or permutation indices

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::sync::Mutex;

use grim_tensor::dtype::{DType, GroupQuantScheme, QuantProvenance, Storage, ArithType};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};
use crate::safetensors::read_safetensor_bytes;

/// GPTQ tensor info with quantization metadata.
#[derive(Debug, Clone)]
pub struct GptqTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    /// Quantization bit width (2, 3, 4, or 8)
    pub bits: u32,
    /// Group size (64 or 128 for EfficientQAT)
    pub group_size: usize,
    /// Whether desc_act (activation ordering) is enabled
    pub desc_act: bool,
    /// Offset to qweight tensor in bytes (optional if not yet loaded)
    pub qweight_offset: Option<u64>,
    pub qweight_size: u64,
    /// Offset to qzeros tensor in bytes (optional if not yet loaded)
    pub qzeros_offset: Option<u64>,
    pub qzeros_size: u64,
    /// Offset to scales tensor in bytes (optional if not yet loaded)
    pub scales_offset: Option<u64>,
    pub scales_size: u64,
    /// Offset to g_idx tensor in bytes (may be absent)
    pub g_idx_offset: Option<u64>,
    pub g_idx_size: Option<u64>,
}

impl GptqTensorInfo {
    /// Returns the Grim Storage configuration for this GPTQ tensor.
    pub fn storage(&self) -> Storage {
        Storage::GroupInt(grim_tensor::dtype::GpuIntConfig {
            bits: self.bits as u8,
            group_size: self.group_size,
            scheme: GroupQuantScheme::Asymmetric, // EfficientQAT is always asymmetric
            desc_act: self.desc_act,
        })
    }
}

/// GPTQ v2 provider for EfficientQAT checkpoints.
/// Reads the (qweight, qzeros, scales, g_idx) tensor layout.
pub struct GptqProvider {
    pub tensors: HashMap<String, GptqTensorInfo>,
    reader: Mutex<BufReader<File>>,
    data_region_start: u64,
}

/// Reads the quantization parameters from quantize_config.json, config.json, or __metadata__.
fn read_quant_params(path: &str, metadata: &Option<HashMap<String, String>>) -> Result<(u32, usize, bool)> {
    let parent = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new(""));
    let quantize_config_path = parent.join("quantize_config.json");
    let config_path = parent.join("config.json");

    let mut bits = None;
    let mut group_size = None;
    let mut desc_act = None;

    if quantize_config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(quantize_config_path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(b) = val.get("bits").and_then(|v| v.as_u64()).map(|v| v as u32) {
                    bits = Some(b);
                }
                if let Some(g) = val.get("group_size").and_then(|v| v.as_u64()).map(|v| v as usize) {
                    group_size = Some(g);
                }
                if let Some(d) = val.get("desc_act").and_then(|v| v.as_bool()) {
                    desc_act = Some(d);
                }
            }
        }
    }

    if bits.is_none() || group_size.is_none() {
        if config_path.exists() {
            if let Ok(content) = std::fs::read_to_string(config_path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(qcfg) = val.get("quantization_config") {
                        if let Some(b) = qcfg.get("bits").and_then(|v| v.as_u64()).map(|v| v as u32) {
                            bits = Some(b);
                        }
                        if let Some(g) = qcfg.get("group_size").and_then(|v| v.as_u64()).map(|v| v as usize) {
                            group_size = Some(g);
                        }
                        if let Some(d) = qcfg.get("desc_act").and_then(|v| v.as_bool()) {
                            desc_act = Some(d);
                        }
                    }
                }
            }
        }
    }

    if bits.is_none() || group_size.is_none() {
        if let Some(meta) = metadata {
            if let Some(b) = meta.get("bits").and_then(|s| s.parse::<u32>().ok()) {
                bits = Some(b);
            }
            if let Some(g) = meta.get("group_size").and_then(|s| s.parse::<usize>().ok()) {
                group_size = Some(g);
            }
            if let Some(d) = meta.get("desc_act").and_then(|s| s.parse::<bool>().ok()) {
                desc_act = Some(d);
            }
        }
    }

    let b = bits.ok_or_else(|| Error::Backend("Missing 'bits' in quantization config metadata".into()))?;
    let g = group_size.ok_or_else(|| Error::Backend("Missing 'group_size' in quantization config metadata".into()))?;
    let d = desc_act.unwrap_or(false);

    Ok((b, g, d))
}

impl GptqProvider {
    /// Open a safetensors file containing GPTQ tensors.
    pub fn open(path: &str) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot open GPTQ file '{}': {e}", path)))?;
        let reader = BufReader::new(file);
        
        // Read safetensors header to get tensor names and metadata
        let (info, metadata, data_region_start) = crate::safetensors::read_safetensors_header(reader)?;
        
        let (bits, group_size, default_desc_act) = match read_quant_params(path, &metadata) {
            Ok(params) => params,
            Err(e) => return Err(e),
        };

        let mut tensors = HashMap::new();
        for (name, tensor_info) in &info {
            // Check for GPTQ tensor naming pattern ending in .qweight
            if !name.ends_with(".qweight") {
                continue;
            }
            
            // Get base name by removing .qweight suffix
            let base_name = name.strip_suffix(".qweight").unwrap();
            
            // Try to find companion tensors
            let qzeros_name = format!("{}.qzeros", base_name);
            let scales_name = format!("{}.scales", base_name);
            let g_idx_name = format!("{}.g_idx", base_name);
            
            // Get base tensor shape - infer from qweight shape for now
            let qw = tensor_info.shape();
            let shape: Vec<usize> = if qw.len() >= 2 {
                // Approximate shape reconstruction: [in_features, out_features / bits * 32]
                vec![qw[0], qw[1].saturating_mul(32 / bits as usize).max(1)]
            } else {
                qw.clone()
            };
            
            let qzeros_offset = info.get(&qzeros_name).map(|i| i.data_start);
            let qzeros_size = info.get(&qzeros_name).map(|i| i.data_end - i.data_start).unwrap_or(0);
            let scales_offset = info.get(&scales_name).map(|i| i.data_start);
            let scales_size = info.get(&scales_name).map(|i| i.data_end - i.data_start).unwrap_or(0);
            let g_idx_offset = info.get(&g_idx_name).map(|i| i.data_start);
            let g_idx_size = info.get(&g_idx_name).map(|i| i.data_end - i.data_start);
            
            let mut desc_act = default_desc_act;
            if let Some(_g_idx_off) = g_idx_offset {
                // Read g_idx to verify monotonicity
                let g_idx_info = info.get(&g_idx_name).unwrap();
                let mut local_reader = BufReader::new(File::open(path).map_err(|e| Error::Backend(e.to_string()))?);
                if let Ok(g_idx_bytes) = read_safetensor_bytes(&mut local_reader, g_idx_info, data_region_start) {
                    let dtype = g_idx_info.dtype_tag.as_str();
                    let mut prev = -1i64;
                    if dtype == "I32" || dtype == "U32" {
                        let elems = g_idx_bytes.len() / 4;
                        for i in 0..elems {
                            let val = u32::from_le_bytes([g_idx_bytes[i*4], g_idx_bytes[i*4+1], g_idx_bytes[i*4+2], g_idx_bytes[i*4+3]]) as i64;
                            if val < prev {
                                desc_act = true;
                                break;
                            }
                            prev = val;
                        }
                    } else if dtype == "I64" || dtype == "U64" {
                        let elems = g_idx_bytes.len() / 8;
                        for i in 0..elems {
                            let val = u64::from_le_bytes([
                                g_idx_bytes[i*8], g_idx_bytes[i*8+1], g_idx_bytes[i*8+2], g_idx_bytes[i*8+3],
                                g_idx_bytes[i*8+4], g_idx_bytes[i*8+5], g_idx_bytes[i*8+6], g_idx_bytes[i*8+7]
                            ]) as i64;
                            if val < prev {
                                desc_act = true;
                                break;
                            }
                            prev = val;
                        }
                    }
                }
            }

            if qzeros_offset.is_some() && scales_offset.is_some() {
                tensors.insert(base_name.to_string(), GptqTensorInfo {
                    name: base_name.to_string(),
                    shape,
                    bits,
                    group_size,
                    desc_act,
                    qweight_offset: Some(tensor_info.data_start),
                    qweight_size: tensor_info.data_end - tensor_info.data_start,
                    qzeros_offset,
                    qzeros_size,
                    scales_offset,
                    scales_size,
                    g_idx_offset,
                    g_idx_size,
                });
            }
        }
        
        let file = File::open(path)
            .map_err(|e| Error::Backend(format!("cannot reopen GPTQ file '{}': {e}", path)))?;
        let reader = Mutex::new(BufReader::new(file));

        Ok(Self { tensors, reader, data_region_start })
    }
}

impl TensorProvider for GptqProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        let info = self.tensors.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in GPTQ file"))
        })?;
        
        let mut reader = self.reader.lock().unwrap();
        
        // Helper to read raw bytes from offsets
        let mut read_bytes = |offset: Option<u64>, size: u64| -> Result<Vec<u8>> {
            let off = offset.ok_or_else(|| Error::Backend("Missing companion offset".into()))?;
            let start = self.data_region_start + off;
            reader.seek(SeekFrom::Start(start))?;
            let mut buf = vec![0u8; size as usize];
            reader.read_exact(&mut buf)?;
            Ok(buf)
        };

        let qweight = read_bytes(info.qweight_offset, info.qweight_size)?;
        let qzeros = read_bytes(info.qzeros_offset, info.qzeros_size)?;
        let scales = read_bytes(info.scales_offset, info.scales_size)?;
        let g_idx = if let Some(off) = info.g_idx_offset {
            let sz = info.g_idx_size.unwrap_or(0);
            if sz > 0 {
                Some(read_bytes(Some(off), sz)?)
            } else {
                None
            }
        } else {
            None
        };

        let dequanted = dequant_gptq_tensor(info, &qweight, &qzeros, &scales, g_idx.as_deref())?;

        // Convert f32 vector to raw bytes safely
        let bytes = unsafe {
            let ptr = dequanted.as_ptr() as *const u8;
            let len = dequanted.len() * std::mem::size_of::<f32>();
            std::slice::from_raw_parts(ptr, len).to_vec()
        };

        Ok(RawTensor {
            bytes,
            shape: info.shape.clone(),
            dtype: DType::F32,
            provenance: QuantProvenance::ExternalQat {
                bits: info.bits as u8,
                group_size: info.group_size,
                scheme: GroupQuantScheme::Asymmetric,
                desc_act: info.desc_act,
            },
        })
    }

    fn get_packed(&self, name: &str) -> Result<RawTensor> {
        let info = self.tensors.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in GPTQ file"))
        })?;
        
        let mut reader = self.reader.lock().unwrap();
        
        // Helper to read raw bytes from offsets
        let mut read_bytes = |offset: Option<u64>, size: u64| -> Result<Vec<u8>> {
            let off = offset.ok_or_else(|| Error::Backend("Missing companion offset".into()))?;
            let start = self.data_region_start + off;
            reader.seek(SeekFrom::Start(start))?;
            let mut buf = vec![0u8; size as usize];
            reader.read_exact(&mut buf)?;
            Ok(buf)
        };

        let qweight = read_bytes(info.qweight_offset, info.qweight_size)?;
        let qzeros = read_bytes(info.qzeros_offset, info.qzeros_size)?;
        let scales = read_bytes(info.scales_offset, info.scales_size)?;
        let g_idx = if let Some(off) = info.g_idx_offset {
            let sz = info.g_idx_size.unwrap_or(0);
            if sz > 0 {
                Some(read_bytes(Some(off), sz)?)
            } else {
                None
            }
        } else {
            None
        };

        // Concatenate following the documented length-prefixed convention:
        // [u64 LE: qweight_len] [qweight_bytes...]
        // [u64 LE: qzeros_len]  [qzeros_bytes...]
        // [u64 LE: scales_len]  [scales_bytes...]
        // [u64 LE: g_idx_len]   [g_idx_bytes...]
        let qweight_len = qweight.len() as u64;
        let qzeros_len = qzeros.len() as u64;
        let scales_len = scales.len() as u64;
        let g_idx_len = g_idx.as_ref().map(|v| v.len()).unwrap_or(0) as u64;

        let total_size = 32 + qweight_len as usize + qzeros_len as usize + scales_len as usize + g_idx_len as usize;
        let mut bytes = Vec::with_capacity(total_size);

        bytes.extend_from_slice(&qweight_len.to_le_bytes());
        bytes.extend_from_slice(&qweight);

        bytes.extend_from_slice(&qzeros_len.to_le_bytes());
        bytes.extend_from_slice(&qzeros);

        bytes.extend_from_slice(&scales_len.to_le_bytes());
        bytes.extend_from_slice(&scales);

        bytes.extend_from_slice(&g_idx_len.to_le_bytes());
        if let Some(ref g) = g_idx {
            bytes.extend_from_slice(g);
        }

        let prov = QuantProvenance::ExternalQat {
            bits: info.bits as u8,
            group_size: info.group_size,
            scheme: GroupQuantScheme::Asymmetric,
            desc_act: info.desc_act,
        };

        let storage_cfg = grim_tensor::dtype::GpuIntConfig {
            bits: info.bits as u8,
            group_size: info.group_size,
            scheme: GroupQuantScheme::Asymmetric,
            desc_act: info.desc_act,
        };

        Ok(RawTensor {
            bytes,
            shape: info.shape.clone(),
            dtype: DType {
                arith: ArithType::F32,
                storage: Storage::GroupInt(storage_cfg),
            },
            provenance: prov,
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        let info = self.tensors.get(name).ok_or_else(|| {
            Error::Backend(format!("tensor '{name}' not found in GPTQ file"))
        })?;
        Ok(TensorMeta {
            dtype: DType::F32, // Dequantized output dtype
            provenance: QuantProvenance::ExternalQat {
                bits: info.bits as u8,
                group_size: info.group_size,
                scheme: GroupQuantScheme::Asymmetric,
                desc_act: info.desc_act,
            },
            shape: info.shape.clone(),
            fusion_mask: 0,
        })
    }
}

/// Dequantize a GPTQ tensor using the grouped INT kernel.
/// Returns f32 values ready for GPU upload.
pub fn dequant_gptq_tensor(
    info: &GptqTensorInfo,
    qweight: &[u8],
    qzeros: &[u8],
    scales: &[u8],
    g_idx: Option<&[u8]>,
) -> Result<Vec<f32>> {
    grim_quant::dequant_gptq_group_int(
        qweight,
        qzeros,
        scales,
        g_idx,
        &info.shape,
        info.bits,
        info.group_size,
    )
}

/// Compute the packed u32 word count for a given bit width.
/// For 3-bit weights, elements span three consecutive u32 words.
pub fn packed_elem_count(shape: &[usize], bits: u32) -> usize {
    let elem = shape.iter().product::<usize>();
    match bits {
        2 => (elem + 15) / 16, // 16 values per u32
        3 => (elem + 31) / 32 * 3, // 32 values across 3 u32 words
        4 => (elem + 7) / 8, // 8 values per u32
        8 => elem, // 1 value per u32
        _ => elem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_elem_count_2bit() {
        // 128x128 = 16384 elements
        // 2-bit packed: 16 values per u32 => 16384 / 16 = 1024 u32s
        assert_eq!(packed_elem_count(&[128, 128], 2), 128 * 128 / 16);
    }

    #[test]
    fn test_packed_elem_count_3bit_cross_word() {
        // 32 values packed across 3 u32 words (96 bits)
        // 32 / 32 * 3 = 3 u32s
        assert_eq!(packed_elem_count(&[32, 1], 3), 3);
        // For 128x128: 16384 / 32 * 3 = 1536 u32s
        assert_eq!(packed_elem_count(&[128, 128], 3), 128 * 128 / 32 * 3);
    }

    #[test]
    fn test_packed_elem_count_4bit() {
        // 128x128 = 16384 elements
        // 4-bit packed: 8 values per u32 => 16384 / 8 = 2048 u32s
        assert_eq!(packed_elem_count(&[128, 128], 4), 128 * 128 / 8);
    }

    #[test]
    fn test_gptq_packed_vs_eager_correctness() {
        // Build mock inputs representing a GPTQ tensor layout
        let in_features = 32;
        let out_features = 32;
        let group_size = 16;
        let bits = 4;
        let values_per_word = 8;
        
        let mut qweight = vec![0u8; (in_features / values_per_word) * out_features * 4];
        let mut qzeros = vec![0u8; (in_features / group_size) * (out_features / values_per_word) * 4];
        let mut scales = vec![0u8; (in_features / group_size) * out_features * 4];
        
        let zero_val = 7u32;
        let scale_val = 0.5f32;
        
        let num_groups = in_features / group_size;
        for g in 0..num_groups {
            for col in 0..out_features {
                let scale_idx = g * out_features + col;
                let sb = scale_val.to_le_bytes();
                scales[scale_idx * 4..scale_idx * 4 + 4].copy_from_slice(&sb);
                
                let zero_word_idx = g * (out_features / values_per_word) + col / values_per_word;
                let bit_offset = (col % values_per_word) * bits;
                let offset = zero_word_idx * 4;
                let mut word = u32::from_le_bytes([qzeros[offset], qzeros[offset+1], qzeros[offset+2], qzeros[offset+3]]);
                word |= zero_val << bit_offset;
                qzeros[offset..offset+4].copy_from_slice(&word.to_le_bytes());
            }
        }
        
        for in_idx in 0..in_features {
            for out_idx in 0..out_features {
                let code = ((in_idx + out_idx) % 16) as u32;
                let word_idx = (in_idx / values_per_word) * out_features + out_idx;
                let bit_offset = (in_idx % values_per_word) * bits;
                let offset = word_idx * 4;
                let mut word = u32::from_le_bytes([qweight[offset], qweight[offset+1], qweight[offset+2], qweight[offset+3]]);
                word |= code << bit_offset;
                qweight[offset..offset+4].copy_from_slice(&word.to_le_bytes());
            }
        }

        let info = GptqTensorInfo {
            name: "test_tensor".to_string(),
            shape: vec![in_features, out_features],
            bits: bits as u32,
            group_size,
            desc_act: false,
            qweight_offset: Some(0),
            qweight_size: qweight.len() as u64,
            qzeros_offset: Some(0),
            qzeros_size: qzeros.len() as u64,
            scales_offset: Some(0),
            scales_size: scales.len() as u64,
            g_idx_offset: None,
            g_idx_size: None,
        };

        // 1. Eager dequant path (old GptqProvider::get behavior)
        let eager_dequant = dequant_gptq_tensor(&info, &qweight, &qzeros, &scales, None).unwrap();

        // 2. Packed path (new GptqProvider::get_packed behavior)
        let qweight_len = qweight.len() as u64;
        let qzeros_len = qzeros.len() as u64;
        let scales_len = scales.len() as u64;
        let g_idx_len = 0u64;

        let total_size = 32 + qweight_len as usize + qzeros_len as usize + scales_len as usize;
        let mut packed_bytes = Vec::with_capacity(total_size);
        packed_bytes.extend_from_slice(&qweight_len.to_le_bytes());
        packed_bytes.extend_from_slice(&qweight);
        packed_bytes.extend_from_slice(&qzeros_len.to_le_bytes());
        packed_bytes.extend_from_slice(&qzeros);
        packed_bytes.extend_from_slice(&scales_len.to_le_bytes());
        packed_bytes.extend_from_slice(&scales);
        packed_bytes.extend_from_slice(&g_idx_len.to_le_bytes());

        let storage_cfg = grim_tensor::dtype::GpuIntConfig {
            bits: bits as u8,
            group_size,
            scheme: GroupQuantScheme::Asymmetric,
            desc_act: false,
        };
        let raw_tensor = RawTensor {
            bytes: packed_bytes,
            shape: info.shape.clone(),
            dtype: DType {
                arith: ArithType::F32,
                storage: Storage::GroupInt(storage_cfg),
            },
            provenance: QuantProvenance::ExternalQat {
                bits: bits as u8,
                group_size,
                scheme: GroupQuantScheme::Asymmetric,
                desc_act: false,
            },
        };

        // Dequantize via the new Storage::GroupInt arm logic
        let mut cursor = 0;
        let read_segment = |bytes: &[u8], cursor: &mut usize| -> Result<Vec<u8>> {
            let len = u64::from_le_bytes(bytes[*cursor..*cursor+8].try_into().unwrap()) as usize;
            *cursor += 8;
            let segment = bytes[*cursor..*cursor+len].to_vec();
            *cursor += len;
            Ok(segment)
        };

        let unpacked_qweight = read_segment(&raw_tensor.bytes, &mut cursor).unwrap();
        let unpacked_qzeros = read_segment(&raw_tensor.bytes, &mut cursor).unwrap();
        let unpacked_scales = read_segment(&raw_tensor.bytes, &mut cursor).unwrap();
        let unpacked_g_idx = read_segment(&raw_tensor.bytes, &mut cursor).unwrap();

        let unpacked_g_idx_opt = if unpacked_g_idx.is_empty() { None } else { Some(&unpacked_g_idx[..]) };

        let varbuilder_dequant = dequant_gptq_tensor(
            &info,
            &unpacked_qweight,
            &unpacked_qzeros,
            &unpacked_scales,
            unpacked_g_idx_opt,
        ).unwrap();

        // Assert identical float outputs
        assert_eq!(eager_dequant, varbuilder_dequant);
    }
}