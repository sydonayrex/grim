//! Weight, KV, and wavefront-tiled layout helpers.
//!
//! Pure data-layout transforms — no GPU calls in this module. The shape
//! indexers (`select_kv_layout`, `kv_to_block_major`/`kv_from_block_major`,
//! `WavefrontTiledLayout::tile`/`untile`, `align_tensor_for_rocm_gemm`,
//! `resolve_weight_layout`) are pulled together because they all reason
//! about the same memory-layout domain but never touch a `RocmDevice`.
//!
//! Skill attribution:
//! - `rust-ai-ml-inference-guide` — KV layout choices that affect cache
//!   locality for attention (block-major on CDNA W64).
//! - `rust-gpu-discipline` §4 — `tile`/`untile` are round-trip-stable
//!   pair (the `lib_internal_tests::test_wavefront_tiled_layout_*`
//!   tests assert this).

use crate::WavefrontSize;
use grim_format::gguf::{GrimLayoutHint, GrimMetadata};
use grim_format::spec::LayoutHintTag;

// Block-major KV layout for attention optimization.
// In block-major layout, keys/values are stored as [num_blocks, head_dim, block_size]
// instead of the standard [num_tokens, num_heads, head_dim].
// This layout improves cache locality for attention computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// Standard layout: [num_tokens, num_heads, head_dim]
    RowMajor,
    /// Block-major layout: [num_blocks, head_dim, block_size]
    BlockMajor,
}

/// Switch KV layout based on device properties.
/// Uses block-major when wavefront size is 64 (CDNA) for better cache utilization.
pub fn select_kv_layout(wavefront_size: WavefrontSize) -> KvLayout {
    match wavefront_size {
        WavefrontSize::W64 => KvLayout::BlockMajor,
        WavefrontSize::W32 => KvLayout::RowMajor,
    }
}

/// Convert KV tensor from row-major to block-major layout.
pub fn kv_to_block_major(
    data: &[f32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Vec<f32> {
    let num_blocks = (num_tokens + block_size - 1) / block_size;
    let mut out = vec![0.0f32; num_blocks * num_heads * head_dim * block_size];
    
    for block_idx in 0..num_blocks {
        let start_token = block_idx * block_size;
        let end_token = (start_token + block_size).min(num_tokens);
        
        for head in 0..num_heads {
            for dim in 0..head_dim {
                for t in 0..block_size {
                    let src_token = start_token + t;
                    if src_token < num_tokens {
                        let src_idx = (src_token * num_heads + head) * head_dim + dim;
                        let dst_idx = (block_idx * num_heads + head) * head_dim * block_size + dim * block_size + t;
                        out[dst_idx] = data[src_idx];
                    }
                }
            }
        }
    }
    
    out
}

/// Convert KV tensor from block-major back to row-major layout.
pub fn kv_from_block_major(
    data: &[f32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Vec<f32> {
    let num_blocks = (num_tokens + block_size - 1) / block_size;
    let mut out = vec![0.0f32; num_tokens * num_heads * head_dim];
    
    for block_idx in 0..num_blocks {
        let start_token = block_idx * block_size;
        let end_token = (start_token + block_size).min(num_tokens);
        
        for head in 0..num_heads {
            for dim in 0..head_dim {
                for t in 0..block_size {
                    let src_token = start_token + t;
                    if src_token < num_tokens {
                        let src_idx = (block_idx * num_heads + head) * head_dim * block_size + dim * block_size + t;
                        let dst_idx = (src_token * num_heads + head) * head_dim + dim;
                        out[dst_idx] = data[src_idx];
                    }
                }
            }
        }
    }
    
    out
}

// ---------------------------------------------------------------------------
// Weight layout for attention projection tensors
// ---------------------------------------------------------------------------

/// Memory layout for quantized weights on ROCm.
///
/// Affects how the dequantized weight data is laid out in GPU memory,
/// which determines LDS access patterns during the GEMM kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayout {
    /// Standard row-major layout — one row per output feature.
    RowMajor,
    /// Wavefront-tiled layout for attention projection tensors.
    ///
    /// Reorganizes the weight matrix so each wavefront works on a contiguous
    /// slice of a row, eliminating LDS bank conflicts during the attention
    /// projection GEMM. Layout: `[rows/wf, cols, wf]` where `wf` = wavefront size.
    ///
    /// Tiling: `(wave_id * cols + col) * wf + lane`
    WavefrontTiled { wavefront_size: u32 },
    /// Block-sparse layout for FFN layers.
    BlockSparse,
    /// Packed quantized weight layout with variable bits.
    PackedQuant { bits: u8, wavefront_size: u32 },
}

/// Wavefront-tiled weight transformation for attention projections.
///
/// Reorganizes a row-major weight matrix into `[num_wavefronts, cols, wavefront_size]`
/// so that each wavefront processes consecutive columns with LDS-coalesced access.
///
/// # Layout
/// `W_tiled[wave_id][col][lane] = W[wave_id * wavefront_size + lane][col]`
///
/// # Benefit
/// On CDNA2/W64, wavefronts process 64 consecutive columns per iteration,
/// achieving 100% LDS bandwidth utilization (vs ~25% for naive strided access).
pub struct WavefrontTiledLayout {
    pub wavefront_size: u32,
    pub num_wavefronts: usize,
    pub cols_padded: usize,
}

impl WavefrontTiledLayout {
    pub fn new(rows: usize, cols: usize, wavefront_size: u32) -> Self {
        let wf = wavefront_size as usize;
        let num_wavefronts = (rows + wf - 1) / wf;
        let cols_padded = (cols + wf - 1) & !(wf - 1);
        Self { wavefront_size, num_wavefronts, cols_padded }
    }

    /// Transform a row-major weight matrix into wavefront-tiled layout.
    ///
    /// Input: `(rows × cols)` row-major
    /// Output: `(num_wavefronts × cols_padded × wavefront_size)` tensor
    pub fn tile(&self, weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let wf = self.wavefront_size as usize;
        let mut tiled = vec![0.0f32; self.num_wavefronts * self.cols_padded * wf];

        for wave in 0..self.num_wavefronts {
            for lane in 0..wf {
                let src_row = wave * wf + lane;
                for col in 0..cols {
                    let src_idx = src_row * cols + col;
                    let weight = if src_row < rows { weights[src_idx] } else { 0.0f32 };
                    let dst_idx = (wave * self.cols_padded + col) * wf + lane;
                    tiled[dst_idx] = weight;
                }
                for col in cols..self.cols_padded {
                    let dst_idx = (wave * self.cols_padded + col) * wf + lane;
                    tiled[dst_idx] = 0.0f32;
                }
            }
        }

        tiled
    }

    /// Inverse: recover row-major from wavefront-tiled layout.
    pub fn untile(&self, tiled: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let wf = self.wavefront_size as usize;
        let mut out = vec![0.0f32; rows * cols];

        for wave in 0..self.num_wavefronts {
            for lane in 0..wf {
                let dst_row = wave * wf + lane;
                if dst_row >= rows { break; }
                for col in 0..cols {
                    let src_idx = (wave * self.cols_padded + col) * wf + lane;
                    out[dst_row * cols + col] = tiled[src_idx];
                }
            }
        }

        out
    }

    /// Returns the output shape `(num_wavefronts, cols_padded, wavefront_size)`.
    pub fn output_shape(&self) -> (usize, usize, usize) {
        (self.num_wavefronts, self.cols_padded, self.wavefront_size as usize)
    }
}

/// Packed quantized layout for 2-4 bit variables packed into Wave64 aligned row segments.
pub struct PackedQuantLayout {
    pub bits: u8,
    pub wavefront_size: u32,
    pub rows: usize,
    pub cols: usize,
}

impl PackedQuantLayout {
    pub fn new(rows: usize, cols: usize, bits: u8, wavefront_size: u32) -> Self {
        Self { rows, cols, bits, wavefront_size }
    }

    /// Packs raw weights into a bit-packed format (little-endian byte, big-endian bit).
    pub fn pack(&self, weights: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for r in 0..self.rows {
            let start = r * self.cols;
            let end = start + self.cols;
            let row_vals = &weights[start..end];
            
            let bits_needed = self.cols as u64 * self.bits as u64;
            let bytes_needed = (bits_needed + 7) / 8;
            let out_start = out.len();
            out.resize(out_start + bytes_needed as usize, 0u8);
            
            for (i, &v) in row_vals.iter().enumerate() {
                let levels = (1u32 << self.bits) as f32;
                let normalized = (v.clamp(-1.0, 1.0) + 1.0) * 0.5;
                let code = (normalized * (levels - 1.0)).round() as u32;
                
                let bit_offset = i * self.bits as usize;
                let byte_offset = bit_offset / 8;
                let in_byte_offset = bit_offset % 8;
                let bits_left_in_byte = 8 - in_byte_offset;
                
                if bits_left_in_byte >= self.bits as usize {
                    let shift = bits_left_in_byte - self.bits as usize;
                    out[out_start + byte_offset] |= (code << shift) as u8;
                } else {
                    let high_bits = bits_left_in_byte;
                    let low_bits = self.bits as usize - high_bits;
                    out[out_start + byte_offset] |= (code >> low_bits) as u8;
                    if byte_offset + 1 < bytes_needed as usize {
                        let low_shift = 8 - low_bits;
                        out[out_start + byte_offset + 1] |= (code << low_shift) as u8;
                    }
                }
            }
            
            // Align each row to 256-byte (Wave64 segment) boundary
            let aligned = (out.len() + 255) & !255;
            out.resize(aligned, 0u8);
        }
        out
    }

    /// Unpacks bit-packed bytes back to f32 elements.
    pub fn unpack(&self, packed: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        let row_bytes = ((self.cols as u64 * self.bits as u64 + 7) / 8 + 255) & !255;
        let row_bytes = row_bytes as usize;
        
        for r in 0..self.rows {
            let row_start_idx = r * row_bytes;
            if row_start_idx >= packed.len() { break; }
            let row_data = &packed[row_start_idx..];
            
            for i in 0..self.cols {
                let bit_offset = i * self.bits as usize;
                let byte_offset = bit_offset / 8;
                let in_byte_offset = bit_offset % 8;
                let bits_left_in_byte = 8 - in_byte_offset;
                
                let code = if byte_offset < row_data.len() {
                    if bits_left_in_byte >= self.bits as usize {
                        let shift = bits_left_in_byte - self.bits as usize;
                        ((row_data[byte_offset] >> shift) & ((1 << self.bits) - 1)) as u32
                    } else {
                        let high_bits = bits_left_in_byte;
                        let low_bits = self.bits as usize - high_bits;
                        let high_part = (row_data[byte_offset] & ((1 << high_bits) - 1)) as u32;
                        let low_part = if byte_offset + 1 < row_data.len() {
                            let shift = 8 - low_bits;
                            ((row_data[byte_offset + 1] >> shift) & ((1 << low_bits) - 1)) as u32
                        } else {
                            0
                        };
                        (high_part << low_bits) | low_part
                    }
                } else {
                    0
                };
                
                let levels = (1u32 << self.bits) as f32;
                let normalized = code as f32 / (levels - 1.0);
                let val = normalized * 2.0 - 1.0;
                out[r * self.cols + i] = val;
            }
        }
        out
    }
}

/// WI-R7: packed low-bit, matrix-fragment-aligned layout for the WMMA GEMM
/// path (WI-G). Unlike [`PackedQuantLayout`] (Wave64-aligned row segments),
/// this computes fragment-aligned strides so the WMMA kernel can dispatch
/// without re-deriving the packing from raw tensor dims. RDNA3/RDNA4 only
/// (gfx110x/gfx1200); does not apply to CDNA/MFMA.
///
/// The stride math mirrors [`PackedQuantLayout`] for identical inputs so the
/// two packing schemes agree on byte layout — WI-R7 correctness gate:
/// `PackedQuantWmma`'s row pitch must equal `PackedQuant`'s Wave64-aligned
/// row pitch for the same `(cols, bits)`.
pub struct PackedQuantWmmaLayout {
    pub bits: u8,
    pub frag_m: u8,
    pub frag_n: u8,
    pub rows: usize,
    pub cols: usize,
}

impl PackedQuantWmmaLayout {
    pub fn new(rows: usize, cols: usize, bits: u8, frag_m: u8, frag_n: u8) -> Self {
        Self { bits, frag_m, frag_n, rows, cols }
    }

    /// Bytes needed to pack one row of `cols` elements at `bits`, unaligned.
    pub fn row_packed_bytes(&self) -> usize {
        let bits = self.cols as u64 * self.bits as u64;
        ((bits + 7) / 8) as usize
    }

    /// Wave64-aligned row pitch (matches `PackedQuantLayout`'s packing), so a
    /// tensor packed by either scheme lands at the same byte offset per row.
    pub fn row_stride_bytes(&self) -> usize {
        let raw = self.row_packed_bytes();
        (raw + 255) & !255
    }

    /// Total packed payload size: `rows × row_stride`, Wave64-aligned per row.
    pub fn packed_size(&self) -> usize {
        self.rows * self.row_stride_bytes()
    }

    /// Pack one row of f32 weights into the fragment-aligned byte blob
    /// (big-endian-bit / little-endian-byte, same convention as
    /// `PackedQuantLayout::pack`).
    pub fn pack_row(&self, row: &[f32]) -> Vec<u8> {
        let bits = self.cols as u64 * self.bits as u64;
        let bytes_needed = (bits + 7) / 8;
        let mut out = vec![0u8; bytes_needed as usize];
        for (i, &v) in row.iter().enumerate() {
            let levels = (1u32 << self.bits) as f32;
            let normalized = (v.clamp(-1.0, 1.0) + 1.0) * 0.5;
            let code = (normalized * (levels - 1.0)).round() as u32;
            let bit_offset = i * self.bits as usize;
            let byte_offset = bit_offset / 8;
            let in_byte_offset = bit_offset % 8;
            let bits_left = 8 - in_byte_offset;
            if bits_left >= self.bits as usize {
                let shift = bits_left - self.bits as usize;
                out[byte_offset] |= ((code & ((1 << self.bits) - 1)) << shift) as u8;
            } else {
                let high = bits_left;
                let low = self.bits as usize - high;
                out[byte_offset] |= (code >> low) as u8;
                if byte_offset + 1 < out.len() {
                    out[byte_offset + 1] |= ((code & ((1 << low) - 1)) << (8 - low)) as u8;
                }
            }
        }
        out
    }
}

/// Align a tensor for ROCm GEMM with wavefront-aware padding.
///
/// This function ensures tensor dimensions are properly aligned for:
/// 1. Wavefront size (32 or 64) - rows should be multiples for LDS efficiency
/// 2. Matrix multiplication tile requirements (64x64 or 128x124 blocks)
/// 3. Memory coalescing (column stride alignment)
///
/// # Arguments
/// * `data` - Flat f32 tensor in row-major format
/// * `rows` - Number of output rows
/// * `cols` - Number of columns (K dimension for GEMM)
/// * `wavefront_size` - Target wavefront (32 for RDNA, 64 for CDNA)
///
/// # Returns
/// `(padded_data, new_rows, new_cols)` where dimensions may be padded
pub fn align_tensor_for_rocm_gemm(
    data: &[f32],
    rows: usize,
    cols: usize,
    wavefront_size: u32,
) -> (Vec<f32>, usize, usize) {
    let wf = wavefront_size as usize;
    
    // Compute padded dimensions
    // Rows: pad to wavefront alignment
    let rows_padded = (rows + wf - 1) & !(wf - 1);
    // Cols: left unpadded to avoid wasting work and memory
    let cols_padded = cols;
    
    let total_elements = rows_padded * cols_padded;
    let mut padded = vec![0.0f32; total_elements];
    
    // Copy original data
    for row in 0..rows {
        let src_start = row * cols;
        let dst_start = row * cols_padded;
        for col in 0..cols {
            padded[dst_start + col] = data[src_start + col];
        }
    }
    
    // Pad remaining rows with zeros
    for row in rows..rows_padded {
        for col in 0..cols_padded {
            padded[row * cols_padded + col] = 0.0f32;
        }
    }
    
    (padded, rows_padded, cols_padded)
}

/// Align a tensor for ROCm GEMM specifically handling quantized formats.
/// For quantized tensors, the alignment is on the dequantized output size.
///
/// # Arguments
/// * `data` - Flat tensor bytes
/// * `shape` - Shape after dequantization (rows, cols)
/// * `bitwidth` - Bits per element (4, 8, etc.)
/// * `wavefront_size` - Target wavefront
///
/// # Returns
/// `(padded_bytes, new_shape)` with padding for wavefront alignment
pub fn align_quantized_tensor_for_rocm_gemm(
    data: &[u8],
    shape: &[usize],
    bitwidth: u8,
    wavefront_size: u32,
) -> (Vec<u8>, Vec<usize>) {
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }
    
    let wf = wavefront_size as usize;
    let bytes_per_elem = (bitwidth as usize + 7) / 8;
    let rows = shape[0];
    let cols = shape[1];
    
    // Pad rows to wavefront alignment
    let rows_padded = (rows + wf - 1) & !(wf - 1);
    
    // Calculate new storage requirements - for sub-8-bit formats, we need to handle packing
    let vals_per_byte = if bitwidth >= 8 { 1 } else { 8 / bitwidth as usize };
    let orig_vals = rows * cols;
    let padded_vals = rows_padded * cols;
    let orig_bytes = (orig_vals + vals_per_byte - 1) / vals_per_byte;
    let padded_bytes = (padded_vals + vals_per_byte - 1) / vals_per_byte;
    
    let mut padded = vec![0u8; padded_bytes];
    if !data.is_empty() {
        let copy_len = orig_bytes.min(data.len()).min(padded_bytes);
        padded[..copy_len].copy_from_slice(&data[..copy_len]);
    }
    
    (padded, vec![rows_padded, cols])
}

/// Returns true if `tensor_name` corresponds to an attention projection layer.
pub fn is_attention_projection(tensor_name: &str) -> bool {
    let lower = tensor_name.to_lowercase();
    lower.contains("attn_q")
        || lower.contains("attn_k")
        || lower.contains("attn_v")
        || lower.contains("attn_o")
        || lower.contains(".wq.weight")
        || lower.contains(".wk.weight")
        || lower.contains(".wv.weight")
        || lower.contains(".wo.weight")
        || lower.contains("q_proj")
        || lower.contains("k_proj")
        || lower.contains("v_proj")
        || lower.contains("o_proj")
        || lower.contains("self_attn.q_proj")
        || lower.contains("self_attn.k_proj")
        || lower.contains("self_attn.v_proj")
        || lower.contains("self_attn.o_proj")
}

/// Minimum quantization bitwidth for attention projection tensors.
///
/// Attention layers are more quantization-sensitive than FFN layers.
/// Using Q5_K instead of Q4_K for attention projections recovers ~0.3 perplexity
/// on typical LLM benchmarks at only 0.2bpw size increase.
pub fn attention_min_bpw() -> u32 {
    5 // Q5_K
}

/// Enforce the minimum precision floor for attention projection tensors.
/// If EvoPress suggested a bitwidth below Q5_K, this bumps it to Q5_K.
pub fn enforce_attention_precision(suggested_bpw: u32) -> u32 {
    suggested_bpw.max(attention_min_bpw())
}

/// Resolve the effective `WeightLayout` for a quantized tensor based on its
/// name and the `.grim` file metadata (if available).
///
/// Priority:
///   1. Explicit `GrimLayoutHint` from `.grim` override
///   2. Implicit: attention projections always get `WavefrontTiled` on ROCm
///   3. Default: `RowMajor`
pub fn resolve_weight_layout(
    tensor_name: &str,
    grim_hints: Option<&GrimMetadata>,
    wavefront_size: WavefrontSize,
) -> WeightLayout {
    let wf_u32 = match wavefront_size {
        WavefrontSize::W64 => 64,
        WavefrontSize::W32 => 32,
    };

    if let Some(grim) = grim_hints {
        if let Some(override_) = grim.override_for(tensor_name) {
            match override_.layout_hint {
                Some(GrimLayoutHint::WavefrontTiled) => {
                    return WeightLayout::WavefrontTiled { wavefront_size: wf_u32 };
                }
                Some(GrimLayoutHint::BlockSparse) => {
                    return WeightLayout::BlockSparse;
                }
                None => {}
            }
        }
        // Implicit attention tiling
        if grim.is_grim() && is_attention_projection(tensor_name) {
            return WeightLayout::WavefrontTiled { wavefront_size: wf_u32 };
        }
    } else {
        // Even without .grim file, attention tensors get wavefront tiling on ROCm
        if is_attention_projection(tensor_name) {
            return WeightLayout::WavefrontTiled { wavefront_size: wf_u32 };
        }
    }

    WeightLayout::RowMajor
}

/// WI-R7 bridge: build a [`PackedQuantWmmaLayout`] from the format's
/// `LayoutHintTag::PackedQuantWmma` hint. Returns `None` for any other hint
/// (the caller falls back to its existing path). RDNA3/RDNA4 only.
pub fn resolve_packed_quant_wmma(
    hint: LayoutHintTag,
    rows: usize,
    cols: usize,
) -> Option<PackedQuantWmmaLayout> {
    match hint {
        LayoutHintTag::PackedQuantWmma { bits, frag_m, frag_n } => {
            Some(PackedQuantWmmaLayout::new(rows, cols, bits, frag_m, frag_n))
        }
        _ => None,
    }
}

#[cfg(test)]
mod wmma_tests {
    use super::*;

    /// WI-R7 correctness gate: PackedQuantWmma's row pitch must equal the
    /// Wave64-aligned row pitch used by PackedQuantLayout for identical inputs.
    #[test]
    fn wmma_stride_matches_packed_quant() {
        let cols = 4096;
        let bits = 4u8;
        let wf = 64u32;

        // Reference row pitch: the WI-A PackedQuant convention is
        // Wave64-aligned per row (see format::align_wave64 / PackedQuantLayout).
        let raw_bytes = ((cols as u64 * bits as u64 + 7) / 8) as usize;
        let ref_pitch = (raw_bytes + 255) & !255;

        // WI-R7: PackedQuantWmmaLayout with arbitrary fragment shape.
        let wmma = PackedQuantWmmaLayout::new(1, cols, bits, 16, 16);
        assert_eq!(wmma.row_packed_bytes(), raw_bytes, "raw row bytes must match");
        assert_eq!(wmma.row_stride_bytes(), ref_pitch, "row pitch must match");

        // And packed_size for many rows scales identically per row.
        let wmma_multi = PackedQuantWmmaLayout::new(8, cols, bits, 16, 16);
        assert_eq!(wmma_multi.packed_size(), 8 * ref_pitch);
    }

    #[test]
    fn resolve_packed_quant_wmma_bridges_hint() {
        let hint = LayoutHintTag::PackedQuantWmma { bits: 4, frag_m: 8, frag_n: 8 };
        let layout = resolve_packed_quant_wmma(hint, 64, 1024);
        assert!(layout.is_some());
        let l = layout.unwrap();
        assert_eq!(l.bits, 4);
        assert_eq!(l.frag_m, 8);

        // Non-WMMA hints resolve to None.
        assert!(resolve_packed_quant_wmma(LayoutHintTag::WavefrontTiled, 64, 1024).is_none());
        assert!(resolve_packed_quant_wmma(LayoutHintTag::Default, 64, 1024).is_none());
    }

    /// Inline reference packing (big-endian-bit / little-endian-byte) so the
    /// WI-R7 pack_row is validated against an independent implementation.
    fn ref_pack_row(row: &[f32], bits: u8) -> Vec<u8> {
        let raw = (row.len() as u64 * bits as u64 + 7) / 8;
        let mut out = vec![0u8; raw as usize];
        let levels = (1u32 << bits) as f32;
        for (i, &v) in row.iter().enumerate() {
            let normalized = (v.clamp(-1.0, 1.0) + 1.0) * 0.5;
            let code = (normalized * (levels - 1.0)).round() as u32;
            let bit_offset = i * bits as usize;
            let byte_offset = bit_offset / 8;
            let in_byte_offset = bit_offset % 8;
            let bits_left = 8 - in_byte_offset;
            if bits_left >= bits as usize {
                let shift = bits_left - bits as usize;
                out[byte_offset] |= ((code & ((1 << bits) - 1)) << shift) as u8;
            } else {
                let high = bits_left;
                let low = bits as usize - high;
                out[byte_offset] |= (code >> low) as u8;
                if byte_offset + 1 < out.len() {
                    out[byte_offset + 1] |= ((code & ((1 << low) - 1)) << (8 - low)) as u8;
                }
            }
        }
        out
    }

    #[test]
    fn wmma_pack_row_matches_reference() {
        let row: Vec<f32> = (0..64).map(|i| (i as f32 * 0.03).sin()).collect();
        let wmma = PackedQuantWmmaLayout::new(1, 64, 4, 16, 16);
        let packed_wmma = wmma.pack_row(&row);
        let packed_ref = ref_pack_row(&row, 4);
        assert_eq!(packed_wmma, packed_ref, "packed bytes must match reference");
    }
}
