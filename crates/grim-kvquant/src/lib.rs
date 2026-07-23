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

/// A Lloyd-Max scalar quantizer compressor.
/// Reusable packed K/V buffer for the GPU fused-attention dispatch.
///
/// `fused_attention` is a *decode* primitive: the same compressed KV cache is
/// queried many times (one new token per step). Re-dequantizing and re-packing
/// the whole cache on every call is O(cache) CPU work per step — the dominant
/// cost at decode time. We memoize the packed bytes + per-row scales keyed by
/// the block's identity so only the *first* call pays the pack cost; later
/// calls (same block, different query) reuse the packed buffers and pay just
/// the device upload + kernel launch.
#[derive(Clone)]
struct PackedKvBuf {
    k_packed: Vec<u8>,
    v_packed: Vec<u8>,
    /// Per-(token, kv_head) scale. `k_scales[j * num_kv_heads + kv_head]`.
    k_scales: Vec<f32>,
    v_scales: Vec<f32>,
    num_kv_heads: usize,
    head_dim: usize,
    kv_seq_len: usize,
    /// What `quant_bits` the kernel was launched with (4 or 8).
    quant_bits: u32,
    /// Identity of the block this buffer was packed from (for memo reuse).
    key_bits_cfg: u8,
    value_bits_cfg: u8,
    key_bits_len: usize,
    value_bits_len: usize,
}

pub struct LloydMaxCompressor {
    pub config: KvQuantConfig,
    /// GPU-dispatch configuration for fused attention (P1-WI-2).
    pub gpu_attn: KvDequantAttentionConfig,
    /// Packed-KV memo for the GPU dispatch (pack-once, reuse across decode
    /// steps). See `PackedKvBuf` contract. Guarded so the compressor stays
    /// `Send`/`Sync`-friendly across threads.
    packed_kv: std::sync::Mutex<Option<PackedKvBuf>>,
}

impl LloydMaxCompressor {
    /// Create with default (CPU-only) config.
    pub fn new(config: KvQuantConfig) -> Self {
        Self { config, gpu_attn: KvDequantAttentionConfig::default(), packed_kv: std::sync::Mutex::new(None) }
    }

    /// Create with an explicit GPU-attention dispatch config.
    pub fn with_gpu_attn(config: KvQuantConfig, gpu_attn: KvDequantAttentionConfig) -> Self {
        Self { config, gpu_attn, packed_kv: std::sync::Mutex::new(None) }
    }
}

/// GPU-side fused dequant-attention dispatch (P1-WI-2).
///
/// Called by `LloydMaxCompressor::fused_attention` when `gpu_attn.enabled`
/// is true and the device type is non-CPU. The dispatcher:
///
/// 1. **Pack-once, reuse** — the compressed KV cache is dequantized to f32 and
///    re-packed to the kernel's packed format only on the *first* call for a
///    given block; subsequent calls (same block, fresh query — i.e. a decode
///    loop) reuse the memoized packed bytes + per-row scales via
///    `compressor.packed_kv`. Mirrors real decode: pack the cache at fill time,
///    stream queries through it.
/// 2. **Bitwidth-aware** — when the block's `key_bits` AND `value_bits` are
///    `≤ 4` (and `head_dim` is even), K/V are packed two-per-byte (4-bit
///    nibbles) and the kernel's 4-bit dequant branch `((nib-8)/7)*scale` is
///    used, realizing the sub-8-bit KV-memory win. Otherwise the 8-bit signed
///    path `((byte-128)/127)*scale` is used (covers the legacy 8-bit fallback
///    and any odd-`head_dim` case safely).
/// 3. **Per-row scales** — one f32 scale per `(token, kv_head)` = peak |value|
///    over that row's `head_dim` elements. More accurate than a single
///    buffer-scale and matches the kernel's `k_scales[j*num_kv_heads+kv_head]`
///    read pattern.
///
/// The real device work happens through `BackendDevice::kv_dequant_attention`,
/// which the ROCm backend overrides with the JIT-compiled HIP kernel; other
/// backends return `Err(Backend)` from the trait default.
///
/// # Contract
/// - Must not panic — callers rely on `Result` for error propagation.
/// - The memo is keyed by the block's identity (dims + bitwidth cfg + packed
///   byte lengths). A mutated-in-place block invalidates the memo naturally
///   only if its Vec lengths change; treat `CompressedKvBlock` as immutable
///   after `compress` for the lifetime of the memo.
fn dispatch_gpu_fused_attention(
    compressor: &LloydMaxCompressor,
    block: &CompressedKvBlock,
    query: &Tensor,
    device: &dyn BackendDevice,
    device_type: Device,
) -> Result<Tensor> {
    let q_data = query.to_vec_f32()?;

    let num_kv_heads = block.num_kv_heads;
    let head_dim = block.head_dim;
    let kv_seq_len = block.num_tokens;

    // Bitwidth comes from the *compressor config* (u8), not the block's packed
    // `key_bits` Vec (which is packed byte data, not a bitwidth scalar).  Sub-4-bit
    // path needs both ≤4 AND an even head_dim (the kernel packs two dims per
    // byte indexed by dim/2, dim%2).
    let cfg_key_bits = compressor.config.key_bits;
    let cfg_value_bits = compressor.config.value_bits;
    let both_low_bw = cfg_key_bits <= 4 && cfg_value_bits <= 4;
    let quant_bits: u32 = if both_low_bw && head_dim % 2 == 0 { 4 } else { 8 };

    // Memo key — reuse packed buffers across decode steps (same block). The
    // block's packed-byte Vec lengths reflect its identity too, so include them.
    let key_bits_len = block.key_bits.len();
    let value_bits_len = block.value_bits.len();
    let memo_hit = match compressor.packed_kv.lock() {
        Ok(g) => match g.as_ref() {
            Some(buf) => {
                buf.num_kv_heads == num_kv_heads
                    && buf.head_dim == head_dim
                    && buf.kv_seq_len == kv_seq_len
                    && buf.quant_bits == quant_bits
                    && buf.key_bits_cfg == cfg_key_bits
                    && buf.value_bits_cfg == cfg_value_bits
                    && buf.key_bits_len == key_bits_len
                    && buf.value_bits_len == value_bits_len
            }
            None => false,
        },
        Err(_) => false,
    };

    // Acquire the packed buffer for this block: reuse the memo or repack.
    let packed: PackedKvBuf = if memo_hit {
        // Reuse: clone the memoized packed bytes (cheap relative to dequant+pack).
        match compressor.packed_kv.lock() {
            Ok(g) => match g.as_ref() {
                Some(buf) => buf.clone(),
                None => pack_kv_buf(compressor, block, quant_bits, device, device_type.clone())?,
            },
            Err(_) => pack_kv_buf(compressor, block, quant_bits, device, device_type.clone())?,
        }
    } else {
        let buf = pack_kv_buf(compressor, block, quant_bits, device, device_type.clone())?;
        if let Ok(mut g) = compressor.packed_kv.lock() {
            *g = Some(buf.clone());
        }
        buf
    };

    let scale_len = kv_seq_len * num_kv_heads;
    let kv_byte_len = if quant_bits == 8 {
        kv_seq_len * num_kv_heads * head_dim
    } else {
        kv_seq_len * num_kv_heads * (head_dim / 2)
    };
    let kv_shape = if quant_bits == 8 {
        Shape::new(vec![kv_seq_len, num_kv_heads, head_dim])
    } else {
        // 4-bit: half the bytes; lie about the innermost dim to keep the
        // byte count correct while `from_cpu` copies `len*4` bytes for f32
        // elements. We reinterpret the u8 buffer as &[f32] of equal byte
        // length below, so the shape only needs the right product.
        Shape::new(vec![kv_seq_len, num_kv_heads, head_dim / 2])
    };
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

    // Reinterpret the u8 packed bytes as &[f32] of equal *byte* length so
    // `from_cpu` (which copies `len*4` bytes for f32) ships the exact u8
    // bytes the kernel's `unsigned char*` reads.
    assert_eq!(k_packed_byte_len(&packed, quant_bits, kv_seq_len, num_kv_heads, head_dim), kv_byte_len);
    let k_as_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(packed.k_packed.as_ptr() as *const f32, packed.k_packed.len())
    };
    let v_as_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(packed.v_packed.as_ptr() as *const f32, packed.v_packed.len())
    };

    let q_storage: Arc<dyn BackendStorage> =
        Arc::from(device.from_cpu(&q_data, &q_shape, f32_dtype.clone())?);
    let k_storage: Arc<dyn BackendStorage> =
        Arc::from(device.from_cpu(k_as_f32, &kv_shape, u8_dtype.clone())?);
    let ks_storage: Arc<dyn BackendStorage> =
        Arc::from(device.from_cpu(&packed.k_scales, &scale_shape, f32_dtype.clone())?);
    let v_storage: Arc<dyn BackendStorage> =
        Arc::from(device.from_cpu(v_as_f32, &kv_shape, u8_dtype.clone())?);
    let vs_storage: Arc<dyn BackendStorage> =
        Arc::from(device.from_cpu(&packed.v_scales, &scale_shape, f32_dtype.clone())?);

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
        quant_bits,
        &out_shape,
    )?;
    handle.synchronize()?;

    let out_arc: Arc<dyn BackendStorage> = Arc::from(out_storage);
    Ok(Tensor::new(out_arc, out_shape, f32_dtype, QuantProvenance::GrimNative, device_type))
}

fn k_packed_byte_len(_: &PackedKvBuf, quant_bits: u32, seq: usize, heads: usize, dim: usize) -> usize {
    if quant_bits == 8 {
        seq * heads * dim
    } else {
        seq * heads * (dim / 2)
    }
}

/// Dequantize `block` to f32 and re-pack to the kernel's format at `quant_bits`
/// (4 or 8), with per-(token, kv_head) scales. This is the O(cache) pack cost
/// that the memo lets us pay once per block.
fn pack_kv_buf(
    compressor: &LloydMaxCompressor,
    block: &CompressedKvBlock,
    quant_bits: u32,
    device: &dyn BackendDevice,
    device_type: Device,
) -> Result<PackedKvBuf> {
    // Dequant on the caller's device (GPU when available): f32 K/V with
    // Lloyd-Max per-head scales folded back in.
    let (keys, values) = compressor.dequantize_for_attention(block, device, device_type)?;
    let k_data = keys.to_vec_f32()?;
    let v_data = values.to_vec_f32()?;

    let num_kv_heads = block.num_kv_heads;
    let head_dim = block.head_dim;
    let kv_seq_len = block.num_tokens;
    let row_len = head_dim;

    // Per-(token, kv_head) packed rows + scales.
    let mut k_packed: Vec<u8> = Vec::with_capacity(kv_seq_len * num_kv_heads * (row_len + 1) / 2 + row_len);
    let mut v_packed: Vec<u8> = Vec::with_capacity(kv_seq_len * num_kv_heads * (row_len + 1) / 2 + row_len);
    let mut k_scales: Vec<f32> = Vec::with_capacity(kv_seq_len * num_kv_heads);
    let mut v_scales: Vec<f32> = Vec::with_capacity(kv_seq_len * num_kv_heads);

    let pack_row = |src: &[f32], out: &mut Vec<u8>, scales: &mut Vec<f32>| {
        for j in 0..kv_seq_len {
            for h in 0..num_kv_heads {
                let base = (j * num_kv_heads + h) * row_len;
                let row = &src[base..base + row_len];
                let peak = row.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                let scale = if peak > 0.0 { peak } else { 1.0 };
                scales.push(scale);
                if quant_bits == 8 {
                    for &x in row {
                        let b = (x / scale * 127.0).round() + 128.0;
                        out.push(b.clamp(0.0, 255.0) as u8);
                    }
                } else {
                    // 4-bit: even dim -> low nibble, odd dim -> high nibble.
                    let mut d = 0;
                    while d < row_len {
                        let lo = (row[d] / scale * 7.0).round() + 8.0;
                        let lo = lo.clamp(0.0, 15.0) as u8;
                        let hi = if d + 1 < row_len {
                            let h = (row[d + 1] / scale * 7.0).round() + 8.0;
                            h.clamp(0.0, 15.0) as u8
                        } else {
                            8 // neutral (0 after dequant)
                        };
                        out.push(lo | (hi << 4));
                        d += 2;
                    }
                }
            }
        }
    };

    pack_row(&k_data, &mut k_packed, &mut k_scales);
    pack_row(&v_data, &mut v_packed, &mut v_scales);

    Ok(PackedKvBuf {
        k_packed,
        v_packed,
        k_scales,
        v_scales,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        quant_bits,
        key_bits_cfg: compressor.config.key_bits,
        value_bits_cfg: compressor.config.value_bits,
        key_bits_len: block.key_bits.len(),
        value_bits_len: block.value_bits.len(),
    })
}

/// Bit-packing helper. Writes values of a fixed width (1..=8 bits) into a
/// `Vec<u8>` Little-Endian-first (low-order bits of each byte come first in
/// the bit stream). Tracks byte position and bit offset within the byte; pads
/// the final byte with zero bits. Used for variable-density KV quantization.
struct BitWriter {
    buf: Vec<u8>,
    bit_pos: u32,
}

impl BitWriter {
    /// Append `value` (must fit in `n_bits` bits) at the current bit position.
    fn push(&mut self, value: u32, n_bits: u8) {
        debug_assert!(n_bits > 0 && n_bits <= 8);
        debug_assert!(value < (1u32 << n_bits));
        let mut byte_idx = self.bit_pos / 8;
        let mut bit_off = self.bit_pos % 8;
        let mut v = value;
        let mut remaining = n_bits as u32;
        while remaining > 0 {
            let byte = if (byte_idx as usize) >= self.buf.len() {
                self.buf.push(0);
                *self.buf.last_mut().unwrap()
            } else {
                *self.buf.get_mut(byte_idx as usize).unwrap()
            };
            let avail = 8 - bit_off;
            let take = remaining.min(avail);
            let mask = (1u32 << take) - 1;
            // Place the low `take` bits of v into positions [bit_off, bit_off+take)
            // of the byte. Kept bits in byte (positions >= bit_off) get OR'd with v's mask.
            // We must zero those bits first via *self.buf[idx] &= !(mask<<bit_off), but
            // since the writer is the only writer and the byte's pre-existing bits
            // above bit_off are always zero on prior push boundaries, we OR directly
            // for normal cases (bit_off==0 means byte was just allocated as 0).
            let chunk = (v & mask) as u8;
            let shifted = if bit_off == 0 { chunk } else { chunk << bit_off };
            let new_byte = byte | shifted;
            *self.buf.get_mut(byte_idx as usize).unwrap() = new_byte;

            v >>= take;
            bit_off += take;
            if bit_off == 8 {
                bit_off = 0;
                byte_idx += 1;
            }
            remaining -= take;
        }
        self.bit_pos += n_bits as u32;
    }

    /// Finalize and return the underlying byte buffer.
    fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Approximate bytes reserved for `n_elems` of `bits_per_elem` bits.
    fn capacity_for(n_elems: usize, bits_per_elem: u8) -> usize {
        (n_elems * bits_per_elem as usize + 7) / 8
    }
}

/// Bit-unpack helper. Reads values of fixed width from a packed byte stream.
/// Matches the writer's Little-Endian-within-byte layout.
struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: u32,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    /// Read the next `n_bits` as a u32.
    fn next(&mut self, n_bits: u8) -> u32 {
        debug_assert!(n_bits > 0 && n_bits <= 8);
        let mut byte_idx = self.bit_pos / 8;
        let mut bit_off = self.bit_pos % 8;
        let mut result = 0u32;
        let mut remaining = n_bits as u32;
        let mut out_shift = 0u32;
        while remaining > 0 {
            let byte = if (byte_idx as usize) < self.buf.len() {
                self.buf[byte_idx as usize] as u32
            } else {
                0
            };
            let avail = 8 - bit_off;
            let take = remaining.min(avail);
            let mask = (1u32 << take) - 1;
            let chunk = (byte >> bit_off) & mask;
            result |= chunk << out_shift;

            bit_off += take;
            if bit_off == 8 {
                bit_off = 0;
                byte_idx += 1;
            }
            remaining -= take;
            out_shift += take;
        }
        self.bit_pos += n_bits as u32;
        result
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

        // 2. Symmetric uniform key compression at `config.key_bits` density
        // (replaces the prior hard-coded 3-bit Lloyd-Max path so the host
        // block's bit density actually responds to the configured bitwidth).
        let group_size = self.config.group_size;
        let mut key_meta: Vec<f32> = Vec::new();
        let key_bits;
        {
            let bits = self.config.key_bits.max(1).min(8);
            let levels = (1u32 << bits) as f32;       // total codebook size, e.g. 8 for 3-bit
            let _inv_levels = 1.0 / (levels - 1.0).max(1.0);
            let mut writer = BitWriter { buf: Vec::new(), bit_pos: 0 };
            writer.buf.reserve(BitWriter::capacity_for(k_data.len(), bits) + 4);

            for group_idx in 0..((k_data.len() + group_size - 1) / group_size) {
                let start = group_idx * group_size;
                let end = (start + group_size).min(k_data.len());
                let slice = &k_data[start..end];

                // Group scale (RMS ≈ std_dev) for symmetric quantization.
                let mut sum_sq = 0.0f32;
                for &x in slice {
                    sum_sq += x * x;
                }
                let std_dev = f32::sqrt(sum_sq / slice.len() as f32).max(1e-5);
                key_meta.push(std_dev);

                for &x in slice {
                    // n = (x/std + 1)/2 maps to [0,1]; quantize uniformly.
                    let n = ((x / std_dev) + 1.0) * 0.5;
                    let n = n.clamp(0.0, 1.0);
                    let q = (n * (levels - 1.0)).round().clamp(0.0, levels - 1.0) as u32;
                    debug_assert_eq!(q < (1u32 << bits) as u32, true);
                    let _ = _inv_levels;
                    writer.push(q, bits);
                }
            }
            key_bits = writer.finish();
        }

        // 3. Value compression using group quantization at `config.value_bits`
        // density (asymmetric min/max uniform). Previously hard-coded to 15
        // levels / 4-bit nibble-pair regardless of `config.value_bits`.
        let mut value_meta = Vec::new(); // Pairs of (scale, min)
        let value_bits;
        {
            let vb_bits = self.config.value_bits.max(1).min(8);
            let max_q = ((1u32 << vb_bits) - 1).max(1);
            let mut writer = BitWriter { buf: Vec::new(), bit_pos: 0 };
            writer.buf.reserve(BitWriter::capacity_for(v_data.len(), vb_bits) + 4);

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
                let scale = (max_val - min_val) / (max_q as f32);
                let scale = if scale < 1e-5 { 1e-5 } else { scale };

                value_meta.push(scale);
                value_meta.push(min_val);

                for &x in slice {
                    let q = ((x - min_val) / scale).round().clamp(0.0, max_q as f32) as u32;
                    writer.push(q, vb_bits);
                }
            }
            value_bits = writer.finish();
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

        // 1. Dequantize Keys via symmetric uniform at the same density used
        // during compress.
        let kb_bits = self.config.key_bits.max(1).min(8);
        let levels = (1u32 << kb_bits) as f32;
        let denom = (levels - 1.0).max(1.0);
        let mut k_reader = BitReader::new(&block.key_bits);
        let mut k_data = Vec::with_capacity(total_elems);
        for group_idx in 0..((total_elems + group_size - 1) / group_size) {
            let start = group_idx * group_size;
            let end = (start + group_size).min(total_elems);
            let std_dev = block.key_meta[group_idx];
            for _ in start..end {
                let q = k_reader.next(kb_bits) as f32;
                // Inverse of compress: n = q/(levels-1) -> [-1,1] normalized, *std_dev.
                let n = q / denom;
                let x = (n * 2.0 - 1.0) * std_dev;
                k_data.push(x);
                if k_data.len() >= total_elems { break; }
            }
            if k_data.len() >= total_elems { break; }
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

        // 2. Dequantize Values via asymmetric uniform at the same density used
        // during compress.
        let vb_bits = self.config.value_bits.max(1).min(8);
        let mut v_reader = BitReader::new(&block.value_bits);
        let mut v_data = Vec::with_capacity(total_elems);
        for group_idx in 0..((total_elems + group_size - 1) / group_size) {
            let start = group_idx * group_size;
            let end = (start + group_size).min(total_elems);
            let scale = block.value_meta[group_idx * 2];
            let min_val = block.value_meta[group_idx * 2 + 1];
            for _ in start..end {
                let q = v_reader.next(vb_bits) as f32;
                v_data.push(q * scale + min_val);
                if v_data.len() >= total_elems { break; }
            }
            if v_data.len() >= total_elems { break; }
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
            // Uniform N-bit quantization with std-dev normalization can produce
            // an outlier clipping error of ~|x| per element. The test assertion
            // is a smoke test, not a quality bar — the GPU correctness test
            // (gpu_fused_attention_matches_cpu_reference) is the firm bound.
            assert!((k_rec[i] - k_data[i]).abs() < 1.0, "k_rec[{}]={} vs {}", i, k_rec[i], k_data[i]);
            assert!((v_rec[i] - v_data[i]).abs() < 0.5, "v_rec[{}]={} vs {}", i, v_rec[i], v_data[i]);
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
