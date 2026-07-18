//! GPTQ v2 tensor layout reader for EfficientQAT checkpoints.
//!
//! §7.2: Reads grouped INT weights with asymmetric quantization:
//! - `qweight`: packed low-bit weights
//! - `qzeros`: per-group zero-points
//! - `scales`: per-group scales
//! - `g_idx`: group assignment or permutation indices

use grim_tensor::dtype::{DType, GroupQuantScheme, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

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
    /// Offset to qzeros tensor in bytes (optional if not yet loaded)
    pub qzeros_offset: Option<u64>,
    /// Offset to scales tensor in bytes (optional if not yet loaded)
    pub scales_offset: Option<u64>,
    /// Offset to g_idx tensor in bytes (may be absent)
    pub g_idx_offset: Option<u64>,
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
    tensors: std::collections::HashMap<String, GptqTensorInfo>,
}

impl GptqProvider {
    /// Open a safetensors file containing GPTQ tensors.
    pub fn open(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Backend(format!("cannot open GPTQ file '{}': {e}", path)))?;
        let mut reader = std::io::BufReader::new(file);
        
        // Read safetensors header to get tensor names
        let header_json = crate::safetensors::read_safetensors_header(&mut reader)?;
        let (info, _data_start) = header_json;
        
        let mut tensors = std::collections::HashMap::new();
        for (name, tensor_info) in &info {
            // Check for GPTQ tensor naming pattern ending in .qweight
            if !name.ends_with(".qweight") {
                continue;
            }
            
            // Get base name by removing .qweight suffix
            let base_name = name.strip_suffix(".qweight").unwrap();
            
            // Try to find companion tensors
            let qzeros_name = format!("{}.qzeros", name);
            let scales_name = format!("{}.scales", name);
            let g_idx_name = format!("{}.g_idx", name);
            
            // Get base tensor shape - infer from qweight shape for now
            let qw = tensor_info.shape();
            let shape: Vec<usize> = if qw.len() >= 2 {
                // Approximate shape reconstruction: [in_features, out_features / bits * 32]
                vec![qw[0], qw[1].saturating_mul(32).max(1)]
            } else {
                qw.clone()
            };
            
            // Default quantization params (4-bit, 128 group size)
            // Real implementation would parse from name pattern like "w4g128"
            let bits: u32 = 4;
            let group_size: usize = 128;
            
            let qzeros_offset = info.get(&qzeros_name).map(|i| i.data_start);
            let scales_offset = info.get(&scales_name).map(|i| i.data_start);
            let g_idx_offset = info.get(&g_idx_name).map(|i| i.data_start);
            
            if qzeros_offset.is_some() && scales_offset.is_some() {
                tensors.insert(name.clone(), GptqTensorInfo {
                    name: base_name.to_string(),
                    shape,
                    bits,
                    group_size,
                    desc_act: false, // Will check g_idx later
                    qweight_offset: Some(tensor_info.data_start),
                    qzeros_offset,
                    scales_offset,
                    g_idx_offset,
                });
            }
        }
        
        Ok(Self { tensors })
    }
}

impl TensorProvider for GptqProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        // GPTQ tensors store multiple parallel arrays
        // This is a placeholder that indicates the format is recognized
        // Full implementation would read and dequantize the weights using grim-quant
        Err(Error::Unimplemented(
            "GPTQ tensor get() requires full dequant kernel with weight reading".into()
        ))
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
pub fn dequant_gptq_tensor(info: &GptqTensorInfo, qweight: &[u8], qzeros: &[u8], scales: &[u8]) -> Result<Vec<f32>> {
    grim_quant::dequant_gptq_group_int(qweight, qzeros, scales, &info.shape, info.bits, info.group_size)
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
}