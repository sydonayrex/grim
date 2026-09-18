//! Fused KV-dequantized attention HIP kernel (WI-R5).
//! WRECK-5: added KvQuantFormat-based dequant paths for Q8_0 block-quantized and Q4K super-block-quantized KV caches.

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

    // ---- WRECK-5 device helpers: fp16→f32 + Q4K element dequant ---- fp16_to_float_device: software conversion, no HIP fp16 header dependency.
    // Handles normal, subnormal, zero, inf, nan.

    // ---- WRECK-5 pre-compute row byte strides for each quant_format ---- Fp16: row_bytes = head_dim * 2.
    // Q8_0: row_bytes = ceil(head_dim/32) * 34 (34 = 2 fp16 delta + 32 int8 codes).
    const int row_bytes_fp16 = head_dim * 2;
    const int row_bytes_q8_0 = ((head_dim + 31) / 32) * 34;
    const int row_bytes_q4k = ((head_dim + 255) / 256) * 144;
    const int num_sub_blocks_q4khalf = (head_dim + 31) / 32;
    const int row_bytes_q4khalf = 4 + 2 * num_sub_blocks_q4khalf + head_dim / 2;

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
                            k_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                            delta = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                            d_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                            dmin_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                        // ggml get_scale_min_k4: m's top bits come from byte s itself.
                        m  = (scales[s + 4] >> 4)  | ((scales[s] >> 6) << 4);
                    }
                    const int qs_byte = 32 * k + j_idx;
                    unsigned char q_nib = (off < 32) ? (qs[qs_byte] & 0x0F) : (qs[qs_byte] >> 4);
                    float k_val = d_val * (float)sc * (float)q_nib - dmin_val * (float)m;
                    score += q[q_offset + dim] * k_val;
                }
            }
        } else if (quant_format == 3) {
            // Q4KHalf (PLAN-kvcache-channel-axis WI-1): sub-block-quantized KV per row.
            // 4 bytes: fp16 d, fp16 min. Then s scales (6-bit in u8), s mins (6-bit in u8), then head_dim/2 nibbles.
            const int k_row_byte_offset = (j * num_kv_heads + kv_head) * row_bytes_q4khalf;
            const unsigned char* __restrict__ k_row_bytes = k_tensor + k_row_byte_offset;
            const unsigned short* h_ptr = (const unsigned short*)(k_row_bytes);
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
                    d_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                    dmin_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
                }
            }
            const unsigned char* scales = k_row_bytes + 4;
            const unsigned char* mins   = scales + num_sub_blocks_q4khalf;
            const unsigned char* qs     = mins + num_sub_blocks_q4khalf;
            for (int dim = 0; dim < 256; ++dim) {
                if (dim < head_dim) {
                    const int s = dim / 32;
                    const int off = dim % 32;
                    const unsigned char sc = scales[s] & 63;
                    const unsigned char m  = mins[s] & 63;
                    const int byte_idx = (s * 32 + off) / 2;
                    const unsigned char byte_val = qs[byte_idx];
                    const unsigned char q_nib = ((off & 1) == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
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
                            v_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                            delta = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                            d_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                            dmin_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                        // ggml get_scale_min_k4: m's top bits come from byte s itself.
                        m  = (scales[s + 4] >> 4)  | ((scales[s] >> 6) << 4);
                    }
                    const int qs_byte = 32 * k + j_idx;
                    unsigned char q_nib = (off < 32) ? (qs[qs_byte] & 0x0F) : (qs[qs_byte] >> 4);
                    float v_val = d_val * (float)sc * (float)q_nib - dmin_val * (float)m;
                    out_acc[chunk] = out_acc[chunk] * scale_old + scale_new * v_val;
                }
            }
        } else if (quant_format == 3) {
            // Q4KHalf V dequant: per-sub-block d * sc * q - min * m.
            const int v_row_byte_offset = (j * num_kv_heads + kv_head) * row_bytes_q4khalf;
            const unsigned char* __restrict__ v_row_bytes = v_tensor + v_row_byte_offset;
            const unsigned short* h_ptr = (const unsigned short*)(v_row_bytes);
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
                    d_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
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
                    dmin_val = sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
                }
            }
            const unsigned char* scales = v_row_bytes + 4;
            const unsigned char* mins   = scales + num_sub_blocks_q4khalf;
            const unsigned char* qs     = mins + num_sub_blocks_q4khalf;
            for (int chunk = 0; chunk < 4; ++chunk) {
                int dd = lane_id + chunk * wave_size;
                if (dd < head_dim) {
                    const int s = dd / 32;
                    const int off = dd % 32;
                    const unsigned char sc = scales[s] & 63;
                    const unsigned char m  = mins[s] & 63;
                    const int byte_idx = (s * 32 + off) / 2;
                    const unsigned char byte_val = qs[byte_idx];
                    const unsigned char q_nib = ((off & 1) == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
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

// ─── M4: dequantize a quantized KV cache to f32 so decode attention can take
// the split-KV FlashDecoding path. One block per (token, kv_head) row; the
// output layout [kv_seq_len, num_kv_heads, head_dim] matches the packed row
// order, so grim_flash_decode_stage1 consumes it directly.
__device__ __forceinline__ float grim_kvrow_h2f(unsigned short bits) {
    unsigned exp = (bits >> 10) & 0x1F;
    unsigned mant = bits & 0x3FF;
    if (exp == 0) return (mant == 0) ? 0.0f : ((float)mant * 5.9604644775390625e-8f);
    if (exp == 31) return ((int)bits >> 31) ? -1e30f : 1e30f;
    float sign = (bits >> 15) & 1 ? -1.0f : 1.0f;
    return sign * (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)((int)exp - 15));
}

__device__ __forceinline__ float grim_kvrow_dequant_elem(
    const unsigned char* __restrict__ row,  // byte offset of this (j, kv_head) row
    const float* __restrict__ scales,       // legacy per-row scale table
    int dim,
    int head_dim,
    int quant_bits,
    int quant_format
) {
    if (quant_format == 0) {
        return grim_kvrow_h2f(((const unsigned short*)row)[dim]);
    }
    if (quant_format == 1) {
        const unsigned char* block = row + (dim / 32) * 34;
        float delta = grim_kvrow_h2f(((const unsigned short*)block)[0]);
        return delta * (float)((const signed char*)(block + 2))[dim % 32];
    }
    if (quant_format == 2) {
        const unsigned char* sb = row + (dim / 256) * 144;
        const unsigned short* h_ptr = (const unsigned short*)sb;
        float d_val = grim_kvrow_h2f(h_ptr[0]);
        float dmin_val = grim_kvrow_h2f(h_ptr[1]);
        const unsigned char* sbytes = sb + 4;
        const unsigned char* qs = sb + 16;
        const int k = (dim % 256) / 64;
        const int off = dim % 64;
        const int s = 2 * k + (off >= 32 ? 1 : 0);
        unsigned char sc, m;
        if (s < 4) {
            sc = sbytes[s] & 63;
            m  = sbytes[s + 4] & 63;
        } else {
            sc = (sbytes[s + 4] & 0x0F) | ((sbytes[s - 4] >> 6) << 4);
            m  = (sbytes[s + 4] >> 4)  | ((sbytes[s] >> 6) << 4);
        }
        unsigned char q_nib = (off < 32) ? (qs[32 * k + (off & 31)] & 0x0F)
                                         : (qs[32 * k + (off & 31)] >> 4);
        return d_val * (float)sc * (float)q_nib - dmin_val * (float)m;
    }
    if (quant_format == 3) {
        // Q4KHalf: 4B fp16 header (d, min) + s scale bytes + s min bytes
        // + plain-order nibbles (s = head_dim/32).
        const unsigned short* h_ptr2 = (const unsigned short*)row;
        float d2 = grim_kvrow_h2f(h_ptr2[0]);
        float dmin2 = grim_kvrow_h2f(h_ptr2[1]);
        const int nsub = (head_dim + 31) / 32;
        const unsigned char* sbytes2 = row + 4;
        const unsigned char* mbytes2 = sbytes2 + nsub;
        const unsigned char* qs2 = mbytes2 + nsub;
        const int sb2 = dim / 32;
        const int off2 = dim % 32;
        const unsigned char sc2 = sbytes2[sb2] & 63;
        const unsigned char m2 = mbytes2[sb2] & 63;
        const unsigned char qb = qs2[(sb2 * 32 + off2) / 2];
        const float q_nib2 = ((off2 & 1) == 0) ? (float)(qb & 0x0F) : (float)(qb >> 4);
        return d2 * (float)sc2 * q_nib2 - dmin2 * (float)m2;
    }
    if (quant_bits == 8) {
        return (((float)((int)row[dim]) - 128.0f) / 127.0f) * scales[0];
    }
    // Legacy nibble (quant_bits == 4)
    unsigned char byte = row[dim / 2];
    float nib = (float)((dim % 2 == 0) ? (byte & 0xF) : (byte >> 4));
    return (nib - 8.0f) / 7.0f * scales[0];
}

extern "C" __global__ void grim_kv_dequant_to_f32(
    const unsigned char* __restrict__ tensor,
    const float* __restrict__ scales,
    float* __restrict__ out,   // [kv_seq_len, num_kv_heads, head_dim]
    int num_kv_heads,
    int head_dim,
    int kv_seq_len,
    int quant_bits,
    int quant_format
) {
    const int row = blockIdx.x;               // j * num_kv_heads + kv_head
    const int dim = threadIdx.x;
    if (row >= kv_seq_len * num_kv_heads || dim >= head_dim) return;

    const int row_bytes_fp16 = head_dim * 2;
    const int row_bytes_q8_0 = ((head_dim + 31) / 32) * 34;
    const int row_bytes_q4k = ((head_dim + 255) / 256) * 144;
    const int nsub_elems = (head_dim + 31) / 32;
    const int row_bytes_q4kh = 4 + 2 * nsub_elems + head_dim / 2;

    int row_off;
    if (quant_format == 0)      row_off = row * row_bytes_fp16;
    else if (quant_format == 1) row_off = row * row_bytes_q8_0;
    else if (quant_format == 2) row_off = row * row_bytes_q4k;
    else if (quant_format == 3) row_off = row * row_bytes_q4kh;
    else if (quant_bits == 8)   row_off = row * head_dim;
    else                        row_off = (row * head_dim) / 2;

    float val;
    if (quant_format == 0 || quant_format == 1 || quant_format == 2 || quant_format == 3) {
        val = grim_kvrow_dequant_elem(tensor + row_off, scales, dim, head_dim, quant_bits, quant_format);
    } else {
        // Legacy paths index the per-row scale from the flat table.
        val = grim_kvrow_dequant_elem(tensor + row_off,
                                      scales + row,
                                      dim, head_dim, quant_bits, quant_format);
    }
    out[row * head_dim + dim] = val;
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
    fn kv_dequant_attention_source_contains_q4khalf_paths() {
        // WI-3 (Q4KHalf): main kernel K + V branches AND the split-KV
        // FlashDecode helper must all accept quant_format == 3 using the
        // 4+2s+hd/2 row stride.
        let (k_sec, v_sec) = kv_sections();
        assert!(k_sec.contains("row_bytes_q4khalf"));
        assert!(v_sec.contains("row_bytes_q4khalf"));
        assert!(KERNEL_SOURCE.contains("row_bytes_q4kh"));
        // Host contract mirror: 4 + 2*s + hd/2.
        assert!(KERNEL_SOURCE.contains("4 + 2 * num_sub_blocks_q4khalf + head_dim / 2"));
        assert!(KERNEL_SOURCE.contains("4 + 2 * nsub_elems + head_dim / 2"));
        // Plain-order nibbles: (s*32 + off)/2, low nibble on even offsets.
        assert!(KERNEL_SOURCE.contains("(off2 & 1) == 0"));
    }

    #[test]
    fn kv_dequant_attention_source_contains_legacy_paths() {
        assert!(KERNEL_SOURCE.contains("quant_bits == 8"));
        assert!(KERNEL_SOURCE.contains("else {"));
    }

    /// Split the main attention kernel into its K-dequant and V-dequant
    /// sections. Raw substring counts across the whole source are stale:
    /// helpers (`grim_kvrow_dequant_elem`, `grim_kv_dequant_to_f32`) repeat
    /// the same `quant_format == N` discriminants outside the K/V loop.
    fn kv_sections() -> (String, String) {
        let src = KERNEL_SOURCE;
        let k_mark = "K dequant + dot product";
        let v_mark = "V dequant + accumulate";
        // Main-kernel end: the first device helper after the kernel.
        let helpers_mark = "grim_kvrow_h2f";
        let k_start = src.find(k_mark).expect("K section marker");
        let v_start = src.find(v_mark).expect("V section marker");
        let end = src.find(helpers_mark).expect("helpers marker");
        assert!(k_start < v_start && v_start < end);
        (
            src[k_start..v_start].to_string(),
            src[v_start..end].to_string(),
        )
    }

    #[test]
    fn kv_dequant_attention_source_fp16_path_reads_k_and_v() {
        // Fp16 path must read both K and V rows (not just one).
        let (k_sec, v_sec) = kv_sections();
        assert!(
            k_sec.contains("quant_format == 0"),
            "Fp16 path must have a K dequant block"
        );
        assert!(
            v_sec.contains("quant_format == 0"),
            "Fp16 path must have a V dequant block"
        );
    }

    #[test]
    fn kv_dequant_attention_source_q8_0_path_reads_k_and_v() {
        let (k_sec, v_sec) = kv_sections();
        assert!(
            k_sec.contains("quant_format == 1"),
            "Q8_0 path must have a K dequant block"
        );
        assert!(
            v_sec.contains("quant_format == 1"),
            "Q8_0 path must have a V dequant block"
        );
    }

    #[test]
    fn kv_dequant_attention_source_q4k_path_reads_k_and_v() {
        let (k_sec, v_sec) = kv_sections();
        assert!(
            k_sec.contains("quant_format == 2"),
            "Q4K path must have a K dequant block"
        );
        assert!(
            v_sec.contains("quant_format == 2"),
            "Q4K path must have a V dequant block"
        );
    }

    #[test]
    fn kv_dequant_attention_source_legacy_path_reads_k_and_v() {
        // Legacy paths: quant_bits == 8 and the else (nibble) branch.
        // Count the K dequant blocks for legacy paths.
        let legacy_k_blocks = KERNEL_SOURCE.matches("quant_bits == 8").count()
            + KERNEL_SOURCE.matches("else {").count();
        // There's one "else {" for the K dequant branch and one for V.
        assert!(
            legacy_k_blocks >= 2,
            "Legacy paths must have K and V dequant blocks"
        );
    }

    #[test]
    fn kv_dequant_attention_source_pins_the_q4k_min_byte() {
        // ggml get_scale_min_k4 ("q[j-0]"): m's top 2 bits come from scales[s] itself.
        // The K and V dequant blocks each embed the formula; both must use scales[s], never.
        let good = KERNEL_SOURCE
            .matches("m  = (scales[s + 4] >> 4)  | ((scales[s] >> 6) << 4);")
            .count();
        assert_eq!(
            good, 2,
            "K and V q4k dequant blocks must both take m's top bits from scales[s]"
        );
        assert!(
            !KERNEL_SOURCE.contains(">> 4)  | ((scales[s - 4] >> 6)"),
            "kv q4k dequant must NOT read m's top bits from scales[s-4]"
        );
    }

    #[test]
    fn kv_dequant_attention_source_has_row_byte_strides() {
        assert!(KERNEL_SOURCE.contains("row_bytes_fp16 = head_dim * 2"));
        assert!(KERNEL_SOURCE.contains("row_bytes_q8_0 = ((head_dim + 31) / 32) * 34"));
        assert!(KERNEL_SOURCE.contains("row_bytes_q4k = ((head_dim + 255) / 256) * 144"));
    }
}
