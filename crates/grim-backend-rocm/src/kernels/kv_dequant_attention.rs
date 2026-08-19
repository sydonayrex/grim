//! Fused KV-dequantized attention HIP kernel (WI-R5).
//!
//! WRECK-5: added KvQuantFormat-based dequant paths for Q8_0 block-quantized
//! and Q4K super-block-quantized KV caches. Legacy quant_bits paths (4=nibble,
//! 8=int8) are preserved for backward compatibility; the new paths are gated by
//! the `quant_format` kernel argument (0=Fp16, 1=Q8_0, 2=Q4K).
//!
//! Layout (WRECK-5):
//! - Fp16: row-major fp16, row bytes = head_dim * 2.
//! - Q8_0: row-major, each row is ceil(head_dim/32) Q8_0 blocks (34 bytes each:
//!   2-byte fp16 delta + 32× int8 codes). Per-block scale is the fp16 delta,
//!   not a separate k_scales entry; k_scales[] is unused for Q8_0 (backward-compat
//!   stub, kept for signature stability).
//! - Q4K: row-major, each row is ceil(head_dim/256) Q4K super-blocks (144 bytes
//!   each: 2-byte fp16 d + 2-byte fp16 min + 12-byte packed scales + 128 bytes
//!   nibbles). Per-super-block scale is the d/min/scales embedded in the block;
//!   k_scales[] is unused for Q4K (backward-compat stub).
//!
//! Device helpers: `fp16_to_float_device` (software fp16→f32, no HIP fp16 header
//! dependency) and `dequant_q4k_element` (Q4K super-block element dequant, mirrors
//! `kernels::q4k_dequant::dequant_q4k_grim_element`).

pub const KERNEL_SOURCE: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void grim_kv_dequant_attention(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_tensor,
    const float* __restrict__ k_scales,
    const unsigned char* __restrict__ v_tensor,
    const float* __restrict__ v_scales,
    float* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_seq_len,
    int cache_offset,
    float inv_sqrt_d,
    int quant_bits,
    int quant_format
) {
    // grid_x enumerates seq_len * num_heads (one block per (seq, head)),
    // grid_y == 1. Derive both indices from blockIdx.x.
    const int flat = blockIdx.x;
    const int i = flat / num_heads; // query position (0..seq_len)
    const int h = flat % num_heads; // head index
    if (i >= seq_len || h >= num_heads) return;

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = h / q_per_kv;
    const int q_offset = (i * num_heads + h) * head_dim;
    const int abs_i = cache_offset + i;

    const int tid = threadIdx.x;
    const int wave_size = warpSize;
    const int wave_id = tid / wave_size;
    const int lane_id = tid % wave_size;
    const int num_waves = blockDim.x / wave_size;

    const int d = lane_id;
    const bool thread_active = d < head_dim;

    if (head_dim > 256) {
        for (int chunk = 0; chunk < 4; ++chunk) {
            int dd = lane_id + chunk * wave_size;
            if (dd < head_dim) out[q_offset + dd] = nanf("");
        }
        return;
    }

    __shared__ float s_max[8];
    __shared__ float s_sum[8];
    __shared__ float s_acc[8][256];

    const int hi = (abs_i < kv_seq_len) ? (abs_i + 1) : kv_seq_len;
    const int range_len = hi;
    const int base = range_len / num_waves;
    const int rem  = range_len % num_waves;
    int j_start = wave_id * base + (wave_id < rem ? wave_id : rem);
    int j_end   = j_start + base + (wave_id < rem ? 1 : 0);

    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -1e30f;
    float running_sum = 0.0f;

    // ---- WRECK-5 device helpers: fp16→f32 + Q4K element dequant ----
    // fp16_to_float_device: software conversion, no HIP fp16 header dependency.
    // Handles normal, subnormal, zero, inf, nan.
    // dequant_q4k_element: Q4K super-block (144 bytes, 256 elements) element dequant.
    // Formula: d * sc * q - min * m (mirrors kernels::q4k_dequant::dequant_q4k_grim_element).

    // ---- WRECK-5 pre-compute row byte strides for each quant_format ----
    // Fp16: row_bytes = head_dim * 2.
    // Q8_0: row_bytes = ceil(head_dim/32) * 34  (34 = 2 fp16 delta + 32 int8 codes).
    // Q4K: row_bytes = ceil(head_dim/256) * 144 (144 = 2 d + 2 min + 12 scales + 128 nibbles).
    const int row_bytes_fp16 = head_dim * 2;
    const int row_bytes_q8_0 = ((head_dim + 31) / 32) * 34;
    const int row_bytes_q4k = ((head_dim + 255) / 256) * 144;

    for (int j = j_start; j < j_end; ++j) {
        float score = 0.0f;

        // ---- K dequant + dot product ----
        if (quant_format == 0) {
            // Fp16: read K row as fp16, convert to f32.
            const int k_row_byte_offset = (j * num_kv_heads + kv_head) * row_bytes_fp16;
            const unsigned char* __restrict__ k_row_bytes = k_tensor + k_row_byte_offset;
            for (int dim = 0; dim < 256; ++dim) {
                if (dim < head_dim) {
                    const unsigned short* k_elem = (const unsigned short*)(k_row_bytes + dim * 2);
                    float k_val;
                    {
                        unsigned short bits = k_elem[0];
                        unsigned exp = (bits >> 10) & 0x1F;
                        unsigned mant = bits & 0x3FF;
                        if (exp == 0) {
                            k_val = (mant == 0) ? 0.0f : ((float)mant * 5.9604644775390625e-8f);
                        } else if (exp == 31) {
                            k_val = ((int)bits >> 31) ? -1e30f : 1e30f;
                        } else {
                            float sign = (bits >> 15) & 1 ? -1.0f : 1.0f;
                            k_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)(exp - 15));
                        }
                    }
                    score += q[q_offset + dim] * k_val;
                }
            }
        } else if (quant_format == 1) {
            // Q8_0 block dequant: each 32-element block is 34 bytes (2-byte fp16 delta + 32× int8).
            // Per-block scale = fp16 delta (not k_scales[]). k_scales[] unused for Q8_0.
            const int k_row_byte_offset = (j * num_kv_heads + kv_head) * row_bytes_q8_0;
            const unsigned char* __restrict__ k_row_bytes = k_tensor + k_row_byte_offset;
            for (int dim = 0; dim < 256; ++dim) {
                if (dim < head_dim) {
                    const int block_idx = dim / 32;
                    const int elem_idx = dim % 32;
                    const unsigned char* __restrict__ block = k_row_bytes + block_idx * 34;
                    // Read fp16 delta from block header (first 2 bytes, little-endian).
                    const unsigned short* delta_bits = (const unsigned short*)(block);
                    float delta;
                    {
                        unsigned short bits = delta_bits[0];
                        unsigned exp = (bits >> 10) & 0x1F;
                        unsigned mant = bits & 0x3FF;
                        if (exp == 0) {
                            delta = (mant == 0) ? 0.0f : ((float)mant * 5.9604644775390625e-8f);
                        } else if (exp == 31) {
                            delta = ((int)bits >> 31) ? -1e30f : 1e30f;
                        } else {
                            float sign = (bits >> 15) & 1 ? -1.0f : 1.0f;
                            delta = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)(exp - 15));
                        }
                    }
                    // Read int8 code from block body (bytes 2..34).
                    const signed char* codes = (const signed char*)(block + 2);
                    float k_val = delta * (float)codes[elem_idx];
                    score += q[q_offset + dim] * k_val;
                }
            }
        } else if (quant_format == 2) {
            // Q4K super-block dequant: each 256-element super-block is 144 bytes.
            // Per-super-block scale embedded in block (d, min, scales). k_scales[] unused for Q4K.
            const int k_row_byte_offset = (j * num_kv_heads + kv_head) * row_bytes_q4k;
            const unsigned char* __restrict__ k_row_bytes = k_tensor + k_row_byte_offset;
            for (int dim = 0; dim < 256; ++dim) {
                if (dim < head_dim) {
                    const int sb_idx = dim / 256;
                    const int elem_idx = dim % 256;
                    const unsigned char* __restrict__ sb = k_row_bytes + sb_idx * 144;
                    // dequant_q4k_element: d * sc * q - min * m.
                    const unsigned short* h_ptr = (const unsigned short*)(sb);
                    float d_val;
                    {
                        unsigned short bits = h_ptr[0];
                        unsigned exp = (bits >> 10) & 0x1F;
                        unsigned mant = bits & 0x3FF;
                        if (exp == 0) {
                            d_val = (mant == 0) ? 0.0f : ((float)mant * 5.9604644775390625e-8f);
                        } else if (exp == 31) {
                            d_val = ((int)bits >> 31) ? -1e30f : 1e30f;
                        } else {
                            float sign = (bits >> 15) & 1 ? -1.0f : 1.0f;
                            d_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)(exp - 15));
                        }
                    }
                    float dmin_val;
                    {
                        unsigned short bits = h_ptr[1];
                        unsigned exp = (bits >> 10) & 0x1F;
                        unsigned mant = bits & 0x3FF;
                        if (exp == 0) {
                            dmin_val = (mant == 0) ? 0.0f : ((float)mant * 5.9604644775390625e-8f);
                        } else if (exp == 31) {
                            dmin_val = ((int)bits >> 31) ? -1e30f : 1e30f;
                        } else {
                            float sign = (bits >> 15) & 1 ? -1.0f : 1.0f;
                            dmin_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)(exp - 15));
                        }
                    }
                    const unsigned char* scales = sb + 4;
                    const unsigned char* qs = sb + 16;
                    const int k = elem_idx / 64;
                    const int off = elem_idx % 64;
                    const int s = 2 * k + (off >= 32 ? 1 : 0);
                    const int j_idx = off & 31;
                    unsigned char sc, m;
                    if (s < 4) {
                        sc = scales[s] & 63;
                        m  = scales[s + 4] & 63;
                    } else {
                        sc = (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4);
                        m  = (scales[s + 4] >> 4)  | ((scales[s - 4] >> 6) << 4);
                    }
                    const int qs_byte = 32 * k + j_idx;
                    unsigned char q_nib = (off < 32) ? (qs[qs_byte] & 0x0F) : (qs[qs_byte] >> 4);
                    float k_val = d_val * (float)sc * (float)q_nib - dmin_val * (float)m;
                    score += q[q_offset + dim] * k_val;
                }
            }
        } else if (quant_bits == 8) {
            const int k_row_offset = (j * num_kv_heads + kv_head) * head_dim;
            const float scale = k_scales[j * num_kv_heads + kv_head];
            for (int dim = 0; dim < 256; ++dim) {
                if (dim < head_dim) {
                    float k_val = (((float)((int)k_tensor[k_row_offset + dim]) - 128.0f) / 127.0f) * scale;
                    score += q[q_offset + dim] * k_val;
                }
            }
        } else {
            // Legacy nibble path (quant_bits == 4): 4-bit per nibble, 2 nibbles per byte.
            const int k_row_offset = ((j * num_kv_heads + kv_head) * head_dim) / 2;
            const float scale = k_scales[j * num_kv_heads + kv_head];
            for (int dim = 0; dim < 256; ++dim) {
                if (dim < head_dim) {
                    unsigned char byte = k_tensor[k_row_offset + dim / 2];
                    float nib = (float)((dim % 2 == 0) ? (byte & 0xF) : (byte >> 4));
                    float k_val = (nib - 8.0f) / 7.0f * scale;
                    score += q[q_offset + dim] * k_val;
                }
            }
        }

        score *= inv_sqrt_d;

        const float old_max = running_max;
        running_max = fmaxf(running_max, score);
        const float scale_old = expf(old_max - running_max);
        const float scale_new = expf(score - running_max);
        running_sum = running_sum * scale_old + scale_new;

        // ---- V dequant + accumulate ----
        if (quant_format == 0) {
            // Fp16: read V row as fp16, convert to f32.
            const int v_row_byte_offset = (j * num_kv_heads + kv_head) * row_bytes_fp16;
            const unsigned char* __restrict__ v_row_bytes = v_tensor + v_row_byte_offset;
            for (int chunk = 0; chunk < 4; ++chunk) {
                int dd = lane_id + chunk * wave_size;
                if (dd < head_dim) {
                    const unsigned short* v_elem = (const unsigned short*)(v_row_bytes + dd * 2);
                    float v_val;
                    {
                        unsigned short bits = v_elem[0];
                        unsigned exp = (bits >> 10) & 0x1F;
                        unsigned mant = bits & 0x3FF;
                        if (exp == 0) {
                            v_val = (mant == 0) ? 0.0f : ((float)mant * 5.9604644775390625e-8f);
                        } else if (exp == 31) {
                            v_val = ((int)bits >> 31) ? -1e30f : 1e30f;
                        } else {
                            float sign = (bits >> 15) & 1 ? -1.0f : 1.0f;
                            v_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)(exp - 15));
                        }
                    }
                    out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_val;
                }
            }
        } else if (quant_format == 1) {
            // Q8_0 V dequant: per-block fp16 delta * int8 code.
            const int v_row_byte_offset = (j * num_kv_heads + kv_head) * row_bytes_q8_0;
            const unsigned char* __restrict__ v_row_bytes = v_tensor + v_row_byte_offset;
            for (int chunk = 0; chunk < 4; ++chunk) {
                int dd = lane_id + chunk * wave_size;
                if (dd < head_dim) {
                    const int block_idx = dd / 32;
                    const int elem_idx = dd % 32;
                    const unsigned char* __restrict__ block = v_row_bytes + block_idx * 34;
                    const unsigned short* delta_bits = (const unsigned short*)(block);
                    float delta;
                    {
                        unsigned short bits = delta_bits[0];
                        unsigned exp = (bits >> 10) & 0x1F;
                        unsigned mant = bits & 0x3FF;
                        if (exp == 0) {
                            delta = (mant == 0) ? 0.0f : ((float)mant * 5.9604644775390625e-8f);
                        } else if (exp == 31) {
                            delta = ((int)bits >> 31) ? -1e30f : 1e30f;
                        } else {
                            float sign = (bits >> 15) & 1 ? -1.0f : 1.0f;
                            delta = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)(exp - 15));
                        }
                    }
                    const signed char* codes = (const signed char*)(block + 2);
                    float v_val = delta * (float)codes[elem_idx];
                    out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_val;
                }
            }
        } else if (quant_format == 2) {
            // Q4K V dequant: per-super-block d * sc * q - min * m.
            const int v_row_byte_offset = (j * num_kv_heads + kv_head) * row_bytes_q4k;
            const unsigned char* __restrict__ v_row_bytes = v_tensor + v_row_byte_offset;
            for (int chunk = 0; chunk < 4; ++chunk) {
                int dd = lane_id + chunk * wave_size;
                if (dd < head_dim) {
                    const int sb_idx = dd / 256;
                    const int elem_idx = dd % 256;
                    const unsigned char* __restrict__ sb = v_row_bytes + sb_idx * 144;
                    const unsigned short* h_ptr = (const unsigned short*)(sb);
                    float d_val;
                    {
                        unsigned short bits = h_ptr[0];
                        unsigned exp = (bits >> 10) & 0x1F;
                        unsigned mant = bits & 0x3FF;
                        if (exp == 0) {
                            d_val = (mant == 0) ? 0.0f : ((float)mant * 5.9604644775390625e-8f);
                        } else if (exp == 31) {
                            d_val = ((int)bits >> 31) ? -1e30f : 1e30f;
                        } else {
                            float sign = (bits >> 15) & 1 ? -1.0f : 1.0f;
                            d_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)(exp - 15));
                        }
                    }
                    float dmin_val;
                    {
                        unsigned short bits = h_ptr[1];
                        unsigned exp = (bits >> 10) & 0x1F;
                        unsigned mant = bits & 0x3FF;
                        if (exp == 0) {
                            dmin_val = (mant == 0) ? 0.0f : ((float)mant * 5.9604644775390625e-8f);
                        } else if (exp == 31) {
                            dmin_val = ((int)bits >> 31) ? -1e30f : 1e30f;
                        } else {
                            float sign = (bits >> 15) & 1 ? -1.0f : 1.0f;
                            dmin_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)(exp - 15));
                        }
                    }
                    const unsigned char* scales = sb + 4;
                    const unsigned char* qs = sb + 16;
                    const int k = elem_idx / 64;
                    const int off = elem_idx % 64;
                    const int s = 2 * k + (off >= 32 ? 1 : 0);
                    const int j_idx = off & 31;
                    unsigned char sc, m;
                    if (s < 4) {
                        sc = scales[s] & 63;
                        m  = scales[s + 4] & 63;
                    } else {
                        sc = (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4);
                        m  = (scales[s + 4] >> 4)  | ((scales[s - 4] >> 6) << 4);
                    }
                    const int qs_byte = 32 * k + j_idx;
                    unsigned char q_nib = (off < 32) ? (qs[qs_byte] & 0x0F) : (qs[qs_byte] >> 4);
                    float v_val = d_val * (float)sc * (float)q_nib - dmin_val * (float)m;
                    out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_val;
                }
            }
        } else if (quant_bits == 8) {
            const int v_row_offset = (j * num_kv_heads + kv_head) * head_dim;
            const float scale = v_scales[j * num_kv_heads + kv_head];
            for (int chunk = 0; chunk < 4; ++chunk) {
                int dd = lane_id + chunk * wave_size;
                if (dd < head_dim) {
                    float v_val = (((float)((int)v_tensor[v_row_offset + dd]) - 128.0f) / 127.0f) * scale;
                    out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_val;
                }
            }
        } else {
            // Legacy nibble path (quant_bits == 4).
            const int v_row_offset = ((j * num_kv_heads + kv_head) * head_dim) / 2;
            const float scale = v_scales[j * num_kv_heads + kv_head];
            for (int chunk = 0; chunk < 4; ++chunk) {
                int dd = lane_id + chunk * wave_size;
                if (dd < head_dim) {
                    unsigned char byte = v_tensor[v_row_offset + dd / 2];
                    float nib = (float)((dd % 2 == 0) ? (byte & 0xF) : (byte >> 4));
                    float v_val = (nib - 8.0f) / 7.0f * scale;
                    out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_val;
                }
            }
        }
    }

    if (lane_id == 0) {
        s_max[wave_id] = running_max;
        s_sum[wave_id] = running_sum;
    }
    for (int chunk = 0; chunk < 4; ++chunk) {
        int dd = lane_id + chunk * wave_size;
        if (dd < head_dim) {
            s_acc[wave_id][dd] = out_acc[chunk];
        } else if (dd < 256) {
            s_acc[wave_id][dd] = 0.0f;
        }
    }
    __syncthreads();

    if (wave_id != 0) return;

    float m_final = s_max[0];
    float sum_final = s_sum[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) {
        if (w >= num_waves) break;
        const float mw = s_max[w];
        const float uw = s_sum[w];
        const float m_new = fmaxf(m_final, mw);
        const float scale_a = expf(m_final - m_new);
        const float scale_b = expf(mw - m_new);
        sum_final = sum_final * scale_a + uw * scale_b;
        m_final = m_new;
    }

    for (int chunk = 0; chunk < 4; ++chunk) {
        int dd = lane_id + chunk * wave_size;
        if (dd < head_dim) {
            float acc_final = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w >= num_waves) break;
                acc_final += s_acc[w][dd] * expf(s_max[w] - m_final);
            }
            const float inv_sum = (sum_final > 0.0f) ? (1.0f / sum_final) : 0.0f;
            out[q_offset + dd] = acc_final * inv_sum;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_dequant_attention_source_contains_quant_format_param() {
        assert!(KERNEL_SOURCE.contains("int quant_format"));
    }

    #[test]
    fn kv_dequant_attention_source_contains_fp16_path() {
        assert!(KERNEL_SOURCE.contains("quant_format == 0"));
        assert!(KERNEL_SOURCE.contains("row_bytes_fp16"));
    }

    #[test]
    fn kv_dequant_attention_source_contains_q8_0_path() {
        assert!(KERNEL_SOURCE.contains("quant_format == 1"));
        assert!(KERNEL_SOURCE.contains("row_bytes_q8_0"));
        // Q8_0 block header fp16 delta read.
        assert!(KERNEL_SOURCE.contains("delta_bits"));
        assert!(KERNEL_SOURCE.contains("codes[elem_idx]"));
    }

    #[test]
    fn kv_dequant_attention_source_contains_q4k_path() {
        assert!(KERNEL_SOURCE.contains("quant_format == 2"));
        assert!(KERNEL_SOURCE.contains("row_bytes_q4k"));
        // Q4K super-block element dequant: d * sc * q - min * m.
        assert!(KERNEL_SOURCE.contains("d_val"));
        assert!(KERNEL_SOURCE.contains("dmin_val"));
        assert!(KERNEL_SOURCE.contains("sc"));
        assert!(KERNEL_SOURCE.contains("q_nib"));
    }

    #[test]
    fn kv_dequant_attention_source_contains_legacy_paths() {
        assert!(KERNEL_SOURCE.contains("quant_bits == 8"));
        assert!(KERNEL_SOURCE.contains("else {"));
    }

    #[test]
    fn kv_dequant_attention_source_fp16_path_reads_k_and_v() {
        // Fp16 path must read both K and V rows (not just one).
        let k_reads = KERNEL_SOURCE.matches("quant_format == 0").count();
        // There are two quant_format == 0 blocks: one for K, one for V.
        assert_eq!(k_reads, 2, "Fp16 path must have K and V dequant blocks");
    }

    #[test]
    fn kv_dequant_attention_source_q8_0_path_reads_k_and_v() {
        let q8_reads = KERNEL_SOURCE.matches("quant_format == 1").count();
        assert_eq!(q8_reads, 2, "Q8_0 path must have K and V dequant blocks");
    }

    #[test]
    fn kv_dequant_attention_source_q4k_path_reads_k_and_v() {
        let q4k_reads = KERNEL_SOURCE.matches("quant_format == 2").count();
        assert_eq!(q4k_reads, 2, "Q4K path must have K and V dequant blocks");
    }

    #[test]
    fn kv_dequant_attention_source_legacy_path_reads_k_and_v() {
        // Legacy paths: quant_bits == 8 and the else (nibble) branch.
        // Count the K dequant blocks for legacy paths.
        let legacy_k_blocks = KERNEL_SOURCE.matches("quant_bits == 8").count()
            + KERNEL_SOURCE.matches("else {").count();
        // There's one "else {" for the K dequant branch and one for V.
        assert!(legacy_k_blocks >= 2, "Legacy paths must have K and V dequant blocks");
    }

    #[test]
    fn kv_dequant_attention_source_has_row_byte_strides() {
        assert!(KERNEL_SOURCE.contains("row_bytes_fp16 = head_dim * 2"));
        assert!(KERNEL_SOURCE.contains("row_bytes_q8_0 = ((head_dim + 31) / 32) * 34"));
        assert!(KERNEL_SOURCE.contains("row_bytes_q4k = ((head_dim + 255) / 256) * 144"));
    }
}
