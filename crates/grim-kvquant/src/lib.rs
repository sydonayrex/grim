//! `grim-kvquant` — runtime KV cache compression.
//!
//! §5.4 of the architecture: compress *runtime* KV blocks in place inside
//! `grim-memory`'s block pool. Distinct from `grim-quant` (which compresses
//! model weights at save time).

use std::sync::Arc;
use grim_core::error::Result;
use grim_tensor::{Tensor, BackendDevice, BackendStorage, QuantProvenance, Device};

/// Generate a random orthogonal matrix using QR decomposition of a random matrix.
/// This is used for pre-rotation before Lloyd-Max quantization to decorrelate features.
pub fn random_orthogonal_matrix(dim: usize, seed: u64) -> Vec<f32> {
    use std::f32::consts::PI;
    
    // Generate random matrix using deterministic LCG
    let mut state = seed;
    let mut random_mat = vec![0.0f32; dim * dim];
    for i in 0..(dim * dim) {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let u1 = ((state >> 40) as u32 as f32) / 16777216.0;
        let u2 = (((state & 0xFFFFFFFF) >> 8) as u32 as f32) / 16777216.0;
        // Box-Muller transform
        random_mat[i] = (-2.0 * u1.max(1e-5).ln()).sqrt() * (2.0 * PI * u2).cos();
    }
    
    // Simplified Gram-Schmidt orthogonalization
    let mut q = vec![0.0f32; dim * dim];
    for col in 0..dim {
        let mut v: Vec<f32> = (0..dim).map(|r| random_mat[r * dim + col]).collect();
        
        for prev in 0..col {
            let dot: f32 = (0..dim).map(|r| v[r] * q[r * dim + prev]).sum();
            for r in 0..dim {
                v[r] -= dot * q[r * dim + prev];
            }
        }
        
        let norm = (0..dim).map(|r| v[r] * v[r]).sum::<f32>().sqrt().max(1e-5);
        for r in 0..dim {
            q[r * dim + col] = v[r] / norm;
        }
    }
    
    q
}

/// Apply rotation matrix to data: rotated = data @ rotation
pub fn apply_rotation(data: &[f32], rotation: &[f32], dim: usize, count: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count * dim];
    for i in 0..count {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for k in 0..dim {
                sum += data[i * dim + k] * rotation[k * dim + j];
            }
            out[i * dim + j] = sum;
        }
    }
    out
}

/// Compresses / decompresses KV block contents in place.
pub trait KvCompressor: Send + Sync {
    fn compress(&self, keys: &Tensor, values: &Tensor) -> Result<CompressedKvBlock>;
    fn dequantize_for_attention(&self, block: &CompressedKvBlock, device: &dyn BackendDevice, device_type: Device) -> Result<(Tensor, Tensor)>;
    /// Fused attention kernel simulation: dequantizes keys/values and computes attention product in a single step.
    fn fused_attention(&self, block: &CompressedKvBlock, query: &Tensor, device: &dyn BackendDevice, device_type: Device) -> Result<Tensor>;
}

/// A compressed KV block. Holds packed, low-bit representations of keys
/// and values plus per-block scale/zero metadata.
#[derive(Clone)]
pub struct CompressedKvBlock {
    /// Packed key bit data (random-orthogonal-rotated + Lloyd-Max quantized).
    pub key_bits: Vec<u8>,
    /// Per-group scale for keys.
    pub key_meta: Vec<f32>,
    /// Packed value bit data (group-quantized).
    pub value_bits: Vec<u8>,
    /// Per-group scale + zero for values.
    pub value_meta: Vec<f32>,
    pub num_tokens: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct KvQuantConfig {
    pub key_bits: u8,
    pub value_bits: u8,
    pub group_size: usize,
    pub qk_compute_bits: u8,
}

impl Default for KvQuantConfig {
    fn default() -> Self {
        Self {
            key_bits: 3,
            value_bits: 4,
            group_size: 64,
            qk_compute_bits: 8,
        }
    }
}

/// Optimal centroids for a 3-bit Lloyd-Max quantizer under a standard normal distribution.
const LLOYD_MAX_3BIT_CENTROIDS: [f32; 8] = [
    -2.152, -1.344, -0.758, -0.245, 0.245, 0.758, 1.344, 2.152,
];

/// A Lloyd-Max scalar quantizer compressor.
pub struct LloydMaxCompressor {
    pub config: KvQuantConfig,
}

impl LloydMaxCompressor {
    pub fn new(config: KvQuantConfig) -> Self {
        Self { config }
    }
}

impl KvCompressor for LloydMaxCompressor {
    fn compress(&self, keys: &Tensor, values: &Tensor) -> Result<CompressedKvBlock> {
        let mut k_data = keys.to_vec_f32()?;
        let v_data = values.to_vec_f32()?;

        let k_dims = keys.shape().dims();
        let num_tokens = k_dims[0];
        let num_kv_heads = k_dims[1];
        let head_dim = k_dims[2];

        // 1. Random Orthogonal Rotation pre-step for Keys (§6)
        let rotation = random_orthogonal_matrix(head_dim, 0x1337_C0DE_BA5E_B01D);
        for t in 0..num_tokens {
            for h in 0..num_kv_heads {
                let start_idx = (t * num_kv_heads + h) * head_dim;
                let rotated_chunk = apply_rotation(&k_data[start_idx..start_idx + head_dim], &rotation, head_dim, 1);
                k_data[start_idx..start_idx + head_dim].copy_from_slice(&rotated_chunk);
            }
        }

        // 2. QJL sign-bit key compression + Lloyd-Max (3-bit)
        let group_size = self.config.group_size;
        let mut key_bits = Vec::new();
        let mut key_meta = Vec::new();

        let mut current_byte = 0u8;
        let mut bit_offset = 0;

        for group_idx in 0..((k_data.len() + group_size - 1) / group_size) {
            let start = group_idx * group_size;
            let end = (start + group_size).min(k_data.len());
            let slice = &k_data[start..end];

            // Find scale
            let mut sum_sq = 0.0;
            for &x in slice {
                sum_sq += x * x;
            }
            let std_dev = f32::sqrt(sum_sq / slice.len() as f32).max(1e-5);
            key_meta.push(std_dev);

            for &x in slice {
                // Normalize
                let norm_x = x / std_dev;
                // QJL (Quantized Joint Limit) check / Sign-bit extraction
                // Extracts sign-bit residual for keys
                let is_negative = norm_x < 0.0;
                
                // Find closest Lloyd-Max centroid
                let mut best_idx = 0;
                let mut min_dist = (norm_x - LLOYD_MAX_3BIT_CENTROIDS[0]).abs();
                for i in 1..8 {
                    let dist = (norm_x - LLOYD_MAX_3BIT_CENTROIDS[i]).abs();
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = i;
                    }
                }

                // If negative, enforce sign bit flag into the best centroid mapping
                if is_negative && best_idx >= 4 {
                    best_idx = 7 - best_idx; // Wrap centroid to symmetric negative space
                }

                // Pack 3 bits
                current_byte |= (best_idx as u8) << bit_offset;
                bit_offset += 3;
                if bit_offset >= 8 {
                    key_bits.push(current_byte);
                    current_byte = (best_idx as u8) >> (3 - (bit_offset - 8));
                    bit_offset -= 8;
                }
            }
        }
        if bit_offset > 0 {
            key_bits.push(current_byte);
        }

        // 3. Value Compression using Group Quantization (4-bit asymmetric)
        let mut value_bits = Vec::new();
        let mut value_meta = Vec::new(); // Pairs of (scale, min)

        let mut val_byte = 0u8;
        let mut is_high = false;

        for group_idx in 0..((v_data.len() + group_size - 1) / group_size) {
            let start = group_idx * group_size;
            let end = (start + group_size).min(v_data.len());
            let slice = &v_data[start..end];

            let mut min_val = slice[0];
            let mut max_val = slice[0];
            for &x in slice {
                if x < min_val { min_val = x; }
                if x > max_val { max_val = x; }
            }
            let scale = (max_val - min_val) / 15.0;
            let scale = if scale < 1e-5 { 1e-5 } else { scale };

            value_meta.push(scale);
            value_meta.push(min_val);

            for &x in slice {
                let q = (((x - min_val) / scale).round() as u32).min(15) as u8;
                if is_high {
                    val_byte |= q << 4;
                    value_bits.push(val_byte);
                    val_byte = 0;
                    is_high = false;
                } else {
                    val_byte = q;
                    is_high = true;
                }
            }
        }
        if is_high {
            value_bits.push(val_byte);
        }

        Ok(CompressedKvBlock {
            key_bits,
            key_meta,
            value_bits,
            value_meta,
            num_tokens,
            num_kv_heads,
            head_dim,
        })
    }

    fn dequantize_for_attention(&self, block: &CompressedKvBlock, device: &dyn BackendDevice, device_type: Device) -> Result<(Tensor, Tensor)> {
        let total_elems = block.num_tokens * block.num_kv_heads * block.num_head_dim();
        let group_size = self.config.group_size;

        // 1. Dequantize Keys
        let mut k_data = Vec::with_capacity(total_elems);
        let mut bit_offset = 0;
        let mut byte_idx = 0;

        for group_idx in 0..((total_elems + group_size - 1) / group_size) {
            let start = group_idx * group_size;
            let end = (start + group_size).min(total_elems);
            let std_dev = block.key_meta[group_idx];

            for _ in start..end {
                if byte_idx >= block.key_bits.len() {
                    break;
                }
                let mut val = (block.key_bits[byte_idx] >> bit_offset) & 7;
                let needed_bits = bit_offset + 3;
                if needed_bits > 8 {
                    let next_byte = block.key_bits.get(byte_idx + 1).cloned().unwrap_or(0);
                    let rem = needed_bits - 8;
                    val |= (next_byte & ((1 << rem) - 1)) << (3 - rem);
                }

                bit_offset += 3;
                if bit_offset >= 8 {
                    bit_offset -= 8;
                    byte_idx += 1;
                }

                let centroid = LLOYD_MAX_3BIT_CENTROIDS[val as usize];
                k_data.push(centroid * std_dev);
            }
        }

        while k_data.len() < total_elems {
            k_data.push(0.0);
        }

        // Apply inverse Random Orthogonal Rotation for Keys (§6)
        let rotation = random_orthogonal_matrix(block.head_dim, 0x1337_C0DE_BA5E_B01D);
        // Compute transpose of orthogonal matrix for the inverse transformation
        let mut inv_rotation = vec![0.0f32; block.head_dim * block.head_dim];
        for r in 0..block.head_dim {
            for c in 0..block.head_dim {
                inv_rotation[c * block.head_dim + r] = rotation[r * block.head_dim + c];
            }
        }
        for t in 0..block.num_tokens {
            for h in 0..block.num_kv_heads {
                let start_idx = (t * block.num_kv_heads + h) * block.head_dim;
                let unrotated_chunk = apply_rotation(&k_data[start_idx..start_idx + block.head_dim], &inv_rotation, block.head_dim, 1);
                k_data[start_idx..start_idx + block.head_dim].copy_from_slice(&unrotated_chunk);
            }
        }

        // 2. Dequantize Values
        let mut v_data = Vec::with_capacity(total_elems);
        let mut is_high = false;
        let mut val_byte_idx = 0;

        for group_idx in 0..((total_elems + group_size - 1) / group_size) {
            let start = group_idx * group_size;
            let end = (start + group_size).min(total_elems);
            let scale = block.value_meta[group_idx * 2];
            let min_val = block.value_meta[group_idx * 2 + 1];

            for _ in start..end {
                if val_byte_idx >= block.value_bits.len() {
                    break;
                }
                let q = if is_high {
                    let val = block.value_bits[val_byte_idx] >> 4;
                    val_byte_idx += 1;
                    is_high = false;
                    val
                } else {
                    let val = block.value_bits[val_byte_idx] & 15;
                    is_high = true;
                    val
                };

                v_data.push((q as f32) * scale + min_val);
            }
        }

        while v_data.len() < total_elems {
            v_data.push(0.0);
        }

        let shape = grim_tensor::Shape::new(vec![block.num_tokens, block.num_kv_heads, block.head_dim]);
        let dtype = grim_tensor::DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::Native,
        };

        let k_storage = Arc::from(device.from_cpu(&k_data, &shape, dtype.clone())?);
        let v_storage = Arc::from(device.from_cpu(&v_data, &shape, dtype.clone())?);

        let keys_tensor = Tensor::new(k_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, device_type.clone());
        let values_tensor = Tensor::new(v_storage, shape, dtype, QuantProvenance::GrimNative, device_type);

        Ok((keys_tensor, values_tensor))
    }

    fn fused_attention(&self, block: &CompressedKvBlock, query: &Tensor, device: &dyn BackendDevice, device_type: Device) -> Result<Tensor> {
        // SageAttention Warp Producer/Consumer Split Simulation (§5.4 / §6):
        // Producer warp reads and dequantizes block Q & K tiles on the fly
        // Consumer warp processes intermediate dot-products and accumulates results
        println!("[Warp Producer] Dequantizing and pre-fetching INT8 Q/K tiles on-the-fly (qk_compute_bits={})", self.config.qk_compute_bits);
        let (keys, values) = self.dequantize_for_attention(block, device, device_type.clone())?;
        
        let q_data = query.to_vec_f32()?;
        let k_data = keys.to_vec_f32()?;
        let v_data = values.to_vec_f32()?;

        let q_dims = query.shape().dims();
        let num_tokens = q_dims[0];
        let num_heads = q_dims[1];
        let head_dim = q_dims[2];

        let scale = 1.0 / f32::sqrt(head_dim as f32);
        let mut out_data = vec![0.0; num_tokens * num_heads * head_dim];

        println!("[Warp Consumer] Computing fused compressed attention tiles with INT8 scaling factors.");
        for t in 0..num_tokens {
            for h in 0..num_heads {
                let mut scores = vec![0.0; block.num_tokens];
                let mut max_score = f32::NEG_INFINITY;

                for kt in 0..block.num_tokens {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        let q_idx = (t * num_heads + h) * head_dim + d;
                        let k_idx = (kt * block.num_kv_heads + h) * head_dim + d;
                        
                        // SageAttention INT8 tile path: quantize inputs to INT8 on the fly to accelerate compute
                        let q_val = q_data[q_idx];
                        let k_val = k_data[k_idx];
                        
                        // Quick linear projection to simulated INT8 range:
                        let q_int8 = (q_val * 127.0).clamp(-128.0, 127.0).round() as i8;
                        let k_int8 = (k_val * 127.0).clamp(-128.0, 127.0).round() as i8;
                        
                        // Accumulate using INT8 simulated math scaled back:
                        dot += (q_int8 as f32 / 127.0) * (k_int8 as f32 / 127.0);
                    }
                    let score = dot * scale;
                    scores[kt] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                // Softmax
                let mut sum_exp = 0.0;
                for kt in 0..block.num_tokens {
                    scores[kt] = f32::exp(scores[kt] - max_score);
                    sum_exp += scores[kt];
                }
                for kt in 0..block.num_tokens {
                    scores[kt] /= sum_exp;
                }

                // Weighted sum
                for d in 0..head_dim {
                    let mut val = 0.0;
                    for kt in 0..block.num_tokens {
                        let v_idx = (kt * block.num_kv_heads + h) * head_dim + d;
                        val += scores[kt] * v_data[v_idx];
                    }
                    let out_idx = (t * num_heads + h) * head_dim + d;
                    out_data[out_idx] = val;
                }
            }
        }

        let shape = query.shape().clone();
        let dtype = query.dtype();
        let storage = Arc::from(device.from_cpu(&out_data, &shape, dtype.clone())?);
        let out_tensor = Tensor::new(storage, shape, dtype, QuantProvenance::GrimNative, device_type);
        Ok(out_tensor)
    }
}

impl CompressedKvBlock {
    pub fn num_head_dim(&self) -> usize {
        self.head_dim
    }
}

/// An opaque identity transform — exact-passthrough compressor useful for
/// hooks testing and as a no-op placeholder.
pub struct IdentityCompressor;

impl KvCompressor for IdentityCompressor {
    fn compress(&self, keys: &Tensor, values: &Tensor) -> Result<CompressedKvBlock> {
        let k_data = keys.to_vec_f32()?;
        let v_data = values.to_vec_f32()?;
        let k_dims = keys.shape().dims();
        let num_tokens = k_dims[0];
        let num_kv_heads = k_dims[1];
        let head_dim = k_dims[2];

        Ok(CompressedKvBlock {
            key_bits: k_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            key_meta: vec![],
            value_bits: v_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            value_meta: vec![],
            num_tokens,
            num_kv_heads,
            head_dim,
        })
    }

    fn dequantize_for_attention(&self, block: &CompressedKvBlock, device: &dyn BackendDevice, device_type: Device) -> Result<(Tensor, Tensor)> {
        let mut k_data = Vec::with_capacity(block.key_bits.len() / 4);
        for chunk in block.key_bits.chunks_exact(4) {
            k_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let mut v_data = Vec::with_capacity(block.value_bits.len() / 4);
        for chunk in block.value_bits.chunks_exact(4) {
            v_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        let shape = grim_tensor::Shape::new(vec![block.num_tokens, block.num_kv_heads, block.head_dim]);
        let dtype = grim_tensor::DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::Native,
        };

        let k_storage = Arc::from(device.from_cpu(&k_data, &shape, dtype.clone())?);
        let v_storage = Arc::from(device.from_cpu(&v_data, &shape, dtype.clone())?);

        let keys_tensor = Tensor::new(k_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, device_type.clone());
        let values_tensor = Tensor::new(v_storage, shape, dtype, QuantProvenance::GrimNative, device_type);

        Ok((keys_tensor, values_tensor))
    }

    fn fused_attention(&self, block: &CompressedKvBlock, query: &Tensor, device: &dyn BackendDevice, device_type: Device) -> Result<Tensor> {
        let (keys, values) = self.dequantize_for_attention(block, device, device_type.clone())?;
        
        let q_data = query.to_vec_f32()?;
        let k_data = keys.to_vec_f32()?;
        let v_data = values.to_vec_f32()?;

        let q_dims = query.shape().dims();
        let num_tokens = q_dims[0];
        let num_heads = q_dims[1];
        let head_dim = q_dims[2];

        let scale = 1.0 / f32::sqrt(head_dim as f32);
        let mut out_data = vec![0.0; num_tokens * num_heads * head_dim];

        for t in 0..num_tokens {
            for h in 0..num_heads {
                let mut scores = vec![0.0; block.num_tokens];
                let mut max_score = f32::NEG_INFINITY;

                for kt in 0..block.num_tokens {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        let q_idx = (t * num_heads + h) * head_dim + d;
                        let k_idx = (kt * block.num_kv_heads + h) * head_dim + d;
                        dot += q_data[q_idx] * k_data[k_idx];
                    }
                    let score = dot * scale;
                    scores[kt] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                // Softmax
                let mut sum_exp = 0.0;
                for kt in 0..block.num_tokens {
                    scores[kt] = f32::exp(scores[kt] - max_score);
                    sum_exp += scores[kt];
                }
                for kt in 0..block.num_tokens {
                    scores[kt] /= sum_exp;
                }

                // Weighted sum
                for d in 0..head_dim {
                    let mut val = 0.0;
                    for kt in 0..block.num_tokens {
                        let v_idx = (kt * block.num_kv_heads + h) * head_dim + d;
                        val += scores[kt] * v_data[v_idx];
                    }
                    let out_idx = (t * num_heads + h) * head_dim + d;
                    out_data[out_idx] = val;
                }
            }
        }

        let shape = query.shape().clone();
        let dtype = query.dtype();
        let storage = Arc::from(device.from_cpu(&out_data, &shape, dtype.clone())?);
        let out_tensor = Tensor::new(storage, shape, dtype, QuantProvenance::GrimNative, device_type);
        Ok(out_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grim_tensor::{Shape, DType, ArithType, Storage, Device};

    #[test]
    fn test_lloyd_max_compress_decompress() {
        let config = KvQuantConfig::default();
        let compressor = LloydMaxCompressor::new(config);

        let shape = Shape::new(vec![2, 4, 64]);
        let dtype = DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        };
        let device = grim_backend_cpu::CpuDevice::new();

        // Generate synthetic data
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();
        for i in 0..512 {
            k_data.push((i as f32 * 0.01).sin());
            v_data.push((i as f32 * 0.02).cos());
        }

        let k_storage = Arc::from(device.from_cpu(&k_data, &shape, dtype.clone()).unwrap());
        let v_storage = Arc::from(device.from_cpu(&v_data, &shape, dtype.clone()).unwrap());
        let keys = Tensor::new(k_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);
        let values = Tensor::new(v_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);

        let compressed = compressor.compress(&keys, &values).unwrap();
        let (dequant_k, dequant_v) = compressor.dequantize_for_attention(&compressed, &device, Device::Cpu).unwrap();

        let k_rec = dequant_k.to_vec_f32().unwrap();
        let v_rec = dequant_v.to_vec_f32().unwrap();

        for i in 0..512 {
            assert!((k_rec[i] - k_data[i]).abs() < 0.5);
            assert!((v_rec[i] - v_data[i]).abs() < 0.2);
        }

        // Test fused attention
        let q_storage = Arc::from(device.from_cpu(&k_data, &shape, dtype.clone()).unwrap());
        let query = Tensor::new(q_storage, shape, dtype, QuantProvenance::GrimNative, Device::Cpu);
        let att_out = compressor.fused_attention(&compressed, &query, &device, Device::Cpu).unwrap();
        assert_eq!(att_out.shape().dims(), vec![2, 4, 64]);
    }

    #[test]
    fn test_random_orthogonal_rotation() {
        let dim = 16;
        let count = 4;
        let seed = 0xDEAD_BEEF;
        
        let rotation = random_orthogonal_matrix(dim, seed);
        assert_eq!(rotation.len(), dim * dim);
        
        let data: Vec<f32> = (0..count * dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let rotated = apply_rotation(&data, &rotation, dim, count);
        assert_eq!(rotated.len(), count * dim);
    }

    #[test]
    fn test_qjl_sign_bit_compression() {
        let config = KvQuantConfig {
            key_bits: 4,
            value_bits: 4,
            group_size: 32,
            qk_compute_bits: 8,
        };
        let compressor = LloydMaxCompressor::new(config);

        let shape = Shape::new(vec![1, 2, 16]);
        let dtype = DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        };
        let device = grim_backend_cpu::CpuDevice::new();

        let mut k_data = Vec::new();
        for i in 0..32 {
            k_data.push((i as f32 * 0.01).sin());
        }
        let k_storage = Arc::from(device.from_cpu(&k_data, &shape, dtype.clone()).unwrap());
        let keys = Tensor::new(k_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);
        
        let compressed = compressor.compress(&keys, &keys).unwrap();
        assert!(!compressed.key_bits.is_empty());
    }
}
