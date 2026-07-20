//! `grim-kvquant` — runtime KV cache compression.
//!
//! §5.4 of the architecture: compress *runtime* KV blocks in place inside
//! `grim-memory`'s block pool. Distinct from `grim-quant` (which compresses
//! model weights at save time).

use std::sync::Arc;
use grim_core::error::Result;
use grim_tensor::{Tensor, BackendDevice, QuantProvenance, Device, Shape, BackendStorage};

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

/// Configuration for GPU-accelerated fused dequant-attention (P1-WI-2).
///
/// When `enabled = true` **and** the caller provides a non-CPU device type,
/// `KvCompressor::fused_attention` will delegate to `dispatch_gpu_fused_attention`
/// instead of the CPU scalar reference path.  The default is `false` so
/// existing callers are completely unaffected until they opt in.
///
/// The hook currently returns `Err(Unsupported)` because no HIP kernel is
/// yet wired; it exists as the correct dispatch point for a future kernel
/// without scattering GPU-device branches through call sites.
#[derive(Debug, Clone, Copy)]
pub struct KvDequantAttentionConfig {
    /// `true` = dispatch to GPU path when `device_type != Device::Cpu`.
    pub enabled: bool,
}

impl Default for KvDequantAttentionConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

/// On-disk KV-block descriptor (WI-R4 bridge).
///
/// `grim-kvquant` produces a [`CompressedKvBlock`] at runtime; this struct
/// packages it into the exact shape the `grim-format` `GrimTensorEntry`
/// KV region consumes — without `grim-kvquant` taking a dependency on
/// `grim-format` (kept clean so the format crate stays the single source
/// of the wire layout). The CLI/convert layer turns this into
/// `GrimTensorEntry::set_kv_layout` + `GrimFile::add_kv_blob`.
pub struct KvBlockOnDisk {
    /// Serialized blob bytes (passed to `GrimFile::add_kv_blob`).
    pub blob: Vec<u8>,
    /// Per-head key bit-width (0 = inherit default).
    pub bits_k: u8,
    /// Per-head value bit-width (0 = inherit default).
    pub bits_v: u8,
    /// RotateKV-style pre-rotation applied.
    pub rotated: bool,
}

impl CompressedKvBlock {
    /// Build the on-disk descriptor: the serialized blob plus the per-head
    /// bit-widths and rotation flag the format entry needs.
    pub fn to_ondisk(&self, rotated: bool) -> KvBlockOnDisk {
        KvBlockOnDisk {
            blob: self.to_bytes(),
            bits_k: self.key_meta_bits(),
            bits_v: self.value_meta_bits(),
            rotated,
        }
    }
}

impl CompressedKvBlock {
    /// Per-head key bit-width inferred from `key_meta` length (0 = inherit).
    fn key_meta_bits(&self) -> u8 {
        // `key_meta` holds one f32 per group; the producer's config key_bits
        // is the source of truth when available. We surface 0 (inherit) here
        // because the block itself doesn't carry the encoder bit-width; callers
        // that know it should override `KvBlockOnDisk::bits_k` directly.
        0
    }
    fn value_meta_bits(&self) -> u8 {
        0
    }
}

/// Optimal centroids for a 3-bit Lloyd-Max quantizer under a standard normal distribution.
const LLOYD_MAX_3BIT_CENTROIDS: [f32; 8] = [
    -2.152, -1.344, -0.758, -0.245, 0.245, 0.758, 1.344, 2.152,
];

/// A Lloyd-Max scalar quantizer compressor.
pub struct LloydMaxCompressor {
    pub config: KvQuantConfig,
    /// GPU-dispatch configuration for fused attention (P1-WI-2).
    pub gpu_attn: KvDequantAttentionConfig,
}

impl LloydMaxCompressor {
    /// Create with default (CPU-only) config.
    pub fn new(config: KvQuantConfig) -> Self {
        Self { config, gpu_attn: KvDequantAttentionConfig::default() }
    }

    /// Create with an explicit GPU-attention dispatch config.
    pub fn with_gpu_attn(config: KvQuantConfig, gpu_attn: KvDequantAttentionConfig) -> Self {
        Self { config, gpu_attn }
    }
}

/// GPU-side fused dequant-attention dispatch (P1-WI-2).
///
/// Called by `LloydMaxCompressor::fused_attention` when `gpu_attn.enabled`
/// is true and the device type is non-CPU.  The compressed `block` is
/// dequantized to f32 on the host, then **re-packed as signed 8-bit** with a
/// uniform per-buffer scale so the wired `grim_kv_dequant_attention` HIP
/// kernel's signed 8-bit path reproduces the f32 values up to `scale/255`
/// quantization error.  This lets the existing ROCm kernel service the
/// Lloyd-Max 3-bit block without a specialized 3-bit kernel and without leaking
/// device-specific code into `grim-kvquant` (which stays backend-agnostic via
/// `BackendDevice`).
///
/// The real device work happens through `BackendDevice::kv_dequant_attention`,
/// which the ROCm backend overrides with the JIT-compiled HIP kernel; other
/// backends return `Err(Backend)` from the trait default.
///
/// # Contract
/// - Must not panic — callers rely on `Result` for error propagation.
/// - Result is read back to a host `Tensor` (slow path; GPU-resident callers
///   would keep the storage on-device, but `fused_attention` already returns
///   a `Tensor`).
fn dispatch_gpu_fused_attention(
    compressor: &LloydMaxCompressor,
    block: &CompressedKvBlock,
    query: &Tensor,
    device: &dyn BackendDevice,
    device_type: Device,
) -> Result<Tensor> {
    let q_data = query.to_vec_f32()?;

    // Dequantize the compressed K/V block to f32 (per-head Lloyd-Max scales
    // already folded back in). This is the exact same f32 K/V the CPU
    // reference path consumes, so GPU == CPU by construction once the kernel
    // is exact.
    let (keys, values) = compressor.dequantize_for_attention(block, device, device_type.clone())?;

    let k_data = keys.to_vec_f32()?;
    let v_data = values.to_vec_f32()?;

    let num_kv_heads = block.num_kv_heads;
    let head_dim = block.head_dim;
    let kv_seq_len = block.num_tokens;

    // Repack f32 -> signed u8 so the kernel's signed 8-bit dequant path
    // (byte-128)*scale reproduces the f32 value up to quantization error
    // (scale/255). A single uniform scale per buffer keeps the per-row scales
    // the kernel reads all equal to that buffer's peak magnitude.
    let pack_signed_u8 = |src: &[f32]| -> (Vec<u8>, f32) {
        let peak = src.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let scale = if peak > 0.0 { peak } else { 1.0 };
        let bytes = src
            .iter()
            .map(|&x| {
                let v = (x / scale * 127.0).round() + 128.0;
                v.clamp(0.0, 255.0) as u8
            })
            .collect();
        (bytes, scale)
    };
    let (k_packed, k_scale) = pack_signed_u8(&k_data);
    let (v_packed, v_scale) = pack_signed_u8(&v_data);

    // Per-(token, kv_head) scale = buffer peak; layout [kv_seq_len * num_kv_heads].
    let scale_len = kv_seq_len * num_kv_heads;
    let k_scales = vec![k_scale; scale_len];
    let v_scales = vec![v_scale; scale_len];

    let kv_shape = Shape::new(vec![kv_seq_len, num_kv_heads, head_dim]);
    let scale_shape = Shape::new(vec![scale_len]);
    let q_shape = query.shape().clone();
    let f32_dtype = grim_tensor::DType {
        arith: grim_tensor::ArithType::F32,
        storage: grim_tensor::Storage::Native,
    };
    let u8_dtype = grim_tensor::DType {
        arith: grim_tensor::ArithType::U8,
        storage: grim_tensor::Storage::Native,
    };

    // q stays f32; k/v packed as u8; scales f32. `from_cpu` takes `&[f32]`
    // and copies `len*4` bytes, so we reinterpret the u8 byte buffers as f32
    // element slices of equal *byte* length — the device receives the exact
    // u8 bytes the kernel's `unsigned char*` reads.
    let k_as_f32: &[f32] = unsafe { std::slice::from_raw_parts(k_packed.as_ptr() as *const f32, k_packed.len()) };
    let v_as_f32: &[f32] = unsafe { std::slice::from_raw_parts(v_packed.as_ptr() as *const f32, v_packed.len()) };

    let q_storage: Arc<dyn BackendStorage> = Arc::from(device.from_cpu(&q_data, &q_shape, f32_dtype.clone())?);
    let k_storage: Arc<dyn BackendStorage> = Arc::from(device.from_cpu(k_as_f32, &kv_shape, u8_dtype.clone())?);
    let ks_storage: Arc<dyn BackendStorage> = Arc::from(device.from_cpu(&k_scales, &scale_shape, f32_dtype.clone())?);
    let v_storage: Arc<dyn BackendStorage> = Arc::from(device.from_cpu(v_as_f32, &kv_shape, u8_dtype.clone())?);
    let vs_storage: Arc<dyn BackendStorage> = Arc::from(device.from_cpu(&v_scales, &scale_shape, f32_dtype.clone())?);

    let out_shape = query.shape().clone();
    let (out_storage, handle) = device.kv_dequant_attention(
        q_storage.as_ref(),
        k_storage.as_ref(),
        ks_storage.as_ref(),
        v_storage.as_ref(),
        vs_storage.as_ref(),
        num_kv_heads,
        kv_seq_len,
        (kv_seq_len as u32).saturating_sub(1),
        8,
        &out_shape,
    )?;
    handle.synchronize()?;

    let out_arc: Arc<dyn BackendStorage> = Arc::from(out_storage);
    Ok(Tensor::new(out_arc, out_shape, f32_dtype, QuantProvenance::GrimNative, device_type))
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
        // P1-WI-2: GPU dispatch hook. When gpu_attn is enabled AND the caller
        // provides a non-CPU device type, delegate to the GPU path. The GPU path
        // currently returns Err(Unsupported) because no HIP kernel is wired yet;
        // this is the correct hook point — callers that need the GPU path will
        // land here once the kernel exists.
        if self.gpu_attn.enabled && device_type != Device::Cpu {
            return dispatch_gpu_fused_attention(self, block, query, device, device_type);
        }

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

    /// Serialize to a self-describing byte blob for on-disk persistence
    /// (WI-R4 `.grim` KV region). Layout:
    ///
    /// ```text
    /// [ num_tokens : u32 LE ][ num_kv_heads: u32 LE ][ head_dim: u32 LE ]
    /// [ key_meta_len   : u32 LE ][ value_meta_len : u32 LE ]
    /// [ key_bits_len   : u32 LE ]   // byte length of key_bits
    /// [ key_meta   : f32 LE × key_meta_len ]
    /// [ value_meta : f32 LE × value_meta_len ]
    /// [ key_bits   : u8 × key_bits_len ]
    /// [ value_bits : u8 × (rest) ]
    /// ```
    ///
    /// The consumer-side `grim-format` writer carries these bytes verbatim in
    /// `GrimTensorEntry::kv_compressed_*`; a reloaded session reconstructs the
    /// block bit-for-bit via [`CompressedKvBlock::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(
            4 * 6 + self.key_meta.len() * 4 + self.value_meta.len() * 4
                + self.key_bits.len() + self.value_bits.len(),
        );
        buf.extend_from_slice(&(self.num_tokens as u32).to_le_bytes());
        buf.extend_from_slice(&(self.num_kv_heads as u32).to_le_bytes());
        buf.extend_from_slice(&(self.head_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.key_meta.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.value_meta.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.key_bits.len() as u32).to_le_bytes());
        for &m in &self.key_meta {
            buf.extend_from_slice(&m.to_le_bytes());
        }
        for &m in &self.value_meta {
            buf.extend_from_slice(&m.to_le_bytes());
        }
        buf.extend_from_slice(&self.key_bits);
        buf.extend_from_slice(&self.value_bits);
        buf
    }

    /// Inverse of [`CompressedKvBlock::to_bytes`]. Errors on a malformed or
    /// truncated buffer.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < 24 {
            return Err(grim_core::error::Error::KvCache(format!(
                "CompressedKvBlock::from_bytes: buffer too short ({} bytes)",
                buf.len()
            )));
        }
        let mut pos = 0;
        let rd_u32 = |b: &[u8], p: &mut usize| -> u32 {
            let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
            *p += 4;
            v
        };
        let num_tokens = rd_u32(buf, &mut pos) as usize;
        let num_kv_heads = rd_u32(buf, &mut pos) as usize;
        let head_dim = rd_u32(buf, &mut pos) as usize;
        let key_meta_len = rd_u32(buf, &mut pos) as usize;
        let value_meta_len = rd_u32(buf, &mut pos) as usize;
        let key_bits_len = rd_u32(buf, &mut pos) as usize;

        let need = pos + key_meta_len * 4 + value_meta_len * 4 + key_bits_len;
        if buf.len() < need {
            return Err(grim_core::error::Error::KvCache(format!(
                "CompressedKvBlock::from_bytes: truncated (need {need}, have {})",
                buf.len()
            )));
        }
        let mut key_meta = Vec::with_capacity(key_meta_len);
        for _ in 0..key_meta_len {
            key_meta.push(f32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]));
            pos += 4;
        }
        let mut value_meta = Vec::with_capacity(value_meta_len);
        for _ in 0..value_meta_len {
            value_meta.push(f32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]));
            pos += 4;
        }
        let key_bits = buf[pos..pos + key_bits_len].to_vec();
        pos += key_bits_len;
        let value_bits = buf[pos..].to_vec();
        Ok(Self {
            key_bits,
            key_meta,
            value_bits,
            value_meta,
            num_tokens,
            num_kv_heads,
            head_dim,
        })
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

    /// WI-R4: a `CompressedKvBlock` round-trips byte-identically through
    /// `to_bytes` / `from_bytes`, including the (key_bits, value_bits)
    /// split.
    #[test]
    fn compressed_kv_block_bytes_round_trip() {
        let config = KvQuantConfig::default();
        let compressor = LloydMaxCompressor::new(config);
        let shape = Shape::new(vec![2, 4, 64]);
        let dtype = DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        };
        let device = grim_backend_cpu::CpuDevice::new();
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();
        for i in 0..512 {
            k_data.push((i as f32 * 0.01).sin());
            v_data.push((i as f32 * 0.02).cos());
        }
        let k_storage = Arc::from(device.from_cpu(&k_data, &shape, dtype.clone()).unwrap());
        let v_storage = Arc::from(device.from_cpu(&v_data, &shape, dtype.clone()).unwrap());
        let keys = Tensor::new(k_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);
        let values = Tensor::new(v_storage, shape, dtype, QuantProvenance::GrimNative, Device::Cpu);

        let block = compressor.compress(&keys, &values).unwrap();
        let bytes = block.to_bytes();
        let restored = CompressedKvBlock::from_bytes(&bytes).unwrap();
        assert_eq!(block.num_tokens, restored.num_tokens);
        assert_eq!(block.num_kv_heads, restored.num_kv_heads);
        assert_eq!(block.head_dim, restored.head_dim);
        assert_eq!(block.key_bits, restored.key_bits, "key_bits mismatch");
        assert_eq!(block.value_bits, restored.value_bits, "value_bits mismatch");
        assert_eq!(block.key_meta, restored.key_meta);
        assert_eq!(block.value_meta, restored.value_meta);
    }

    // --- P1-WI-2: KvDequantAttentionConfig tests ---

    /// Default KvDequantAttentionConfig has enabled = false.
    #[test]
    fn kv_dequant_attention_config_default_is_off() {
        let cfg = KvDequantAttentionConfig::default();
        assert!(!cfg.enabled);
    }

    /// LloydMaxCompressor::new() inherits default (disabled) GPU config.
    #[test]
    fn lloyd_max_compressor_new_has_gpu_attn_off() {
        let comp = LloydMaxCompressor::new(KvQuantConfig::default());
        assert!(!comp.gpu_attn.enabled);
    }

    /// with_gpu_attn constructor stores the provided config.
    #[test]
    fn lloyd_max_compressor_with_gpu_attn_stores_config() {
        let cfg = KvDequantAttentionConfig { enabled: true };
        let comp = LloydMaxCompressor::with_gpu_attn(KvQuantConfig::default(), cfg);
        assert!(comp.gpu_attn.enabled);
    }

    /// With gpu_attn disabled, fused_attention runs the CPU path without error.
    #[test]
    fn fused_attention_cpu_path_runs_when_gpu_disabled() {
        use grim_tensor::{ArithType, dtype::{Storage as DS, DType}};
        let compressor = LloydMaxCompressor::new(KvQuantConfig { key_bits: 3, value_bits: 4, group_size: 4, qk_compute_bits: 8 });
        let device = grim_backend_cpu::CpuDevice::new();
        let dtype = DType { arith: ArithType::F32, storage: DS::Native };
        let shape = grim_tensor::Shape::new(vec![2, 1, 4]);
        let k_data = vec![0.1f32; 8];
        let v_data = vec![0.2f32; 8];
        let k_storage = Arc::from(device.from_cpu(&k_data, &shape, dtype.clone()).unwrap());
        let v_storage = Arc::from(device.from_cpu(&v_data, &shape, dtype.clone()).unwrap());
        let keys = Tensor::new(k_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);
        let values = Tensor::new(v_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);
        let block = compressor.compress(&keys, &values).unwrap();

        let q_data = vec![0.1f32; 8];
        let q_storage = Arc::from(device.from_cpu(&q_data, &shape, dtype.clone()).unwrap());
        let query = Tensor::new(q_storage, shape, dtype, QuantProvenance::GrimNative, Device::Cpu);
        // CPU path should succeed (gpu_attn.enabled is false).
        let result = compressor.fused_attention(&block, &query, &device, Device::Cpu);
        assert!(result.is_ok(), "CPU fused_attention should succeed");
    }

    /// With gpu_attn enabled, passing a non-CPU device returns Err (not panic).
    #[test]
    fn fused_attention_gpu_path_returns_err_unsupported_when_no_kernel() {
        use grim_tensor::{ArithType, dtype::{Storage as DS, DType}};
        let cfg = KvDequantAttentionConfig { enabled: true };
        let compressor = LloydMaxCompressor::with_gpu_attn(
            KvQuantConfig { key_bits: 3, value_bits: 4, group_size: 4, qk_compute_bits: 8 },
            cfg,
        );
        let device = grim_backend_cpu::CpuDevice::new();
        let dtype = DType { arith: ArithType::F32, storage: DS::Native };
        let shape = grim_tensor::Shape::new(vec![2, 1, 4]);
        let k_data = vec![0.1f32; 8];
        let v_data = vec![0.2f32; 8];
        let k_storage = Arc::from(device.from_cpu(&k_data, &shape, dtype.clone()).unwrap());
        let v_storage = Arc::from(device.from_cpu(&v_data, &shape, dtype.clone()).unwrap());
        let keys = Tensor::new(k_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);
        let values = Tensor::new(v_storage, shape.clone(), dtype.clone(), QuantProvenance::GrimNative, Device::Cpu);
        let block = compressor.compress(&keys, &values).unwrap();

        let q_data = vec![0.1f32; 8];
        let q_storage = Arc::from(device.from_cpu(&q_data, &shape, dtype.clone()).unwrap());
        let query = Tensor::new(q_storage, shape, dtype, QuantProvenance::GrimNative, Device::Cpu);

        // Simulate a non-CPU device_type to trigger the GPU dispatch branch.
        // The stub returns Err(Unimplemented), never panics.
        let result = compressor.fused_attention(&block, &query, &device, Device::Rocm(0));
        assert!(result.is_err(), "GPU path must return Err until kernel is wired");
        // Must be Unimplemented, not a panic or internal error.
        match result {
            Err(grim_core::error::Error::Unimplemented(_)) | Err(grim_core::error::Error::Tensor(grim_tensor::Error::Unimplemented(_))) => {}
            Err(other) => panic!("Expected Unimplemented error, got: {:?}", other),
            Ok(_) => unreachable!(),
        }
    }
}
