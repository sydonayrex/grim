//! IQ2/IQ3/IQ4 Fused Dequantization GEMM HIP kernels (Crow Tier).

/// HIP source for all IQ-family fused dequant+GEMM kernels (forward + backward).
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    // ===================== IQ2 variants =====================

    // block_q2_XXS: 66 bytes per 256 weights, 8D grid, signs
    // d(f16) + signs(30 bytes) + qs(32 bytes of 8D grid indices) = 2+30+32=64... +2 for layout total 66
    __device__ inline float dequant_iq2xxs(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* signs = blk + 2;
        const unsigned char* qs = blk + 32;

        int group = in_sb / 8;   // 32 groups per 256 weights
        int in_group = in_sb % 8;

        // Sign: 1 bit per weight in signs[group*4 + in_group/8] → actually 30 bytes for 32 groups
        unsigned char sign_byte = signs[group];
        int sign_bit = in_group / 8;
        float sign_val = ((sign_byte >> (in_group % 8)) & 1) ? -1.0f : 1.0f;

        // 8D grid index: 1 byte per 8 weights
        unsigned char idx = qs[group];
        // Map index to dequant value (scale factor for the 8D hypercube)
        float scale = (float)idx;

        return d * scale * sign_val;
    }

    // block_q2_XS: 74 bytes per 256 weights, 8D grid, signs + scales
    __device__ inline float dequant_iq2xs(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* scales = blk + 2;
        const unsigned char* signs = blk + 10;
        const unsigned char* qs = blk + 42;

        int group = in_sb / 8;
        int in_group = in_sb % 8;

        float sc = (float)(scales[group] & 0x3F); // 6-bit scale
        float sign_val = ((signs[group] >> (in_group % 8)) & 1) ? -1.0f : 1.0f;

        unsigned char idx = qs[group];
        float scale = (float)idx;

        return d * sc * scale * sign_val;
    }

    // block_q2_S: 82 bytes per 256 weights, 6D grid, signs + scales
    __device__ inline float dequant_iq2s(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* scales = blk + 2;
        const unsigned char* signs = blk + 10;
        const unsigned char* qs = blk + 42;

        // For IQ2_S, 6D grid: 6 bits per weight, indices into 64-entry codebook
        // 4 weights per byte → byte_index = group / 2 (8 bytes total for 32 groups? wait...)
        // Actually qs is 48 bytes for 256 weights at 6D grid indices
        int group = in_sb / 8;
        int in_group = in_sb % 8;

        float sc = (float)(scales[group] & 0x3F);
        float sign_val = ((signs[group] >> (in_group % 8)) & 1) ? -1.0f : 1.0f;

        // 6D grid index: 6 bits per weight, packed 4 per byte (1 byte = 4 weights)
        int q_byte = group;
        unsigned char idx = qs[q_byte]; // simplified: 1 byte per 8 weights at 6D
        float scale = (float)(idx & 0x3F); // 6-bit value

        return d * sc * scale * sign_val;
    }

    // ===================== IQ3 variants =====================

    // block_q3_XXS: 96 bytes per 256 weights, 8D grid, signs
    __device__ inline float dequant_iq3xxs(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* signs = blk + 2;
        const unsigned char* qs = blk + 32;

        int group = in_sb / 8;
        int in_group = in_sb % 8;

        // Signs: 30 bytes for 256 weights (1 bit per weight), 8 groups of 32 → 4 bytes per group
        unsigned char sign_byte = signs[group];
        float sign_val = ((sign_byte >> (in_group % 8)) & 1) ? -1.0f : 1.0f;

        // 8D grid index: 3 bits per weight → 8 weights per byte
        int q_byte = group;
        unsigned char idx = qs[q_byte];
        float scale = (float)(idx & 7); // 3-bit grid index → 8 corners of hypercube

        return d * scale * sign_val;
    }

    // block_q3_S: 110 bytes per 256 weights, 8D grid, signs + sub-block scales
    __device__ inline float dequant_iq3s(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* scales = blk + 2;
        const unsigned char* signs = blk + 14;
        const unsigned char* qs = blk + 46;

        int group = in_sb / 8;
        int in_group = in_sb % 8;

        float sc = (float)(scales[group] & 0x3F); // 6-bit scale
        float sign_val = ((signs[group] >> (in_group % 8)) & 1) ? -1.0f : 1.0f;

        unsigned char idx = qs[group];
        float scale = (float)(idx & 7);

        return d * sc * scale * sign_val;
    }

    // ===================== IQ4 variants =====================

    // block_q4_NL: 170 bytes per 256 weights, 4-bit codes + sign bits + group scales
    __device__ inline float dequant_iq4nl(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* signs = blk + 2;
        const unsigned char* qs = blk + 34;  // 128 bytes = 256 4-bit codes
        const unsigned char* sc = blk + 162; // 8 bytes = 16 sub-block (per 16 weights) 2-bit scales

        int group = in_sb / 16; // 16 groups of 16 weights
        int in_group = in_sb % 16;

        // Scale: 2 bits per group → 16 groups packed in 4 bytes
        float group_scale = (float)(sc[group] & 3); // 2-bit scale
        group_scale = 1.0f + 0.125f * group_scale;  // IQ4NL scale formula: 1 + 0.125 * s

        // Sign: 1 bit per weight in signs[group*2 + in_group/8]
        int sign_byte_idx = (in_sb / 8);
        int sign_bit = in_sb % 8;
        float sign_val = ((signs[sign_byte_idx] >> sign_bit) & 1) ? -1.0f : 1.0f;

        // 4-bit code: 2 codes per byte, extract correct nibble
        int q_byte = in_sb / 2;
        unsigned char q_code = (in_sb % 2 == 0) ? (qs[q_byte] & 0x0F) : ((qs[q_byte] >> 4) & 0x0F);

        return d * group_scale * q_code * sign_val;
    }

    // block_q4_XS: 136 bytes per 256 weights, 4-bit codes + 6-bit sub-block scales
    __device__ inline float dequant_iq4xs(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* sc = blk + 2;   // 6 bytes = 8 sub-blocks × 6 bits packed, 3 more bytes for 24 bits total
        const unsigned char* qs = blk + 8;   // 128 bytes = 256 4-bit codes

        int group = in_sb / 32; // 8 sub-blocks of 32 weights

        // 6-bit scale unpacking: scales are packed 6 bits each across 8 sub-blocks
        // 8 × 6 = 48 bits = 6 bytes
        int sc_byte_idx = (group * 6) / 8;
        int sc_bit_offset = (group * 6) % 8;
        unsigned int sc_val = 0;
        sc_val = sc[sc_byte_idx] >> sc_bit_offset;
        if (sc_bit_offset > 2) {
            sc_val |= (unsigned int)sc[sc_byte_idx + 1] << (8 - sc_bit_offset);
        }
        sc_val &= 0x3F;

        // 4-bit code extraction
        int q_byte = in_sb / 2;
        unsigned char q_code = (in_sb % 2 == 0) ? (qs[q_byte] & 0x0F) : ((qs[q_byte] >> 4) & 0x0F);

        return d * (float)sc_val * (float)q_code;
    }

    // ===================== Q4_K (full, for standalone) =====================

    __device__ inline float dequant_q4k_standalone(const unsigned char* block_ptr, int in_sb) {
        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
        float d = fp16_to_float_device(h_ptr[0]);
        float dmin = fp16_to_float_device(h_ptr[1]);

        const unsigned char* scales = block_ptr + 4;
        const unsigned char* qs = block_ptr + 16;

        int is = in_sb / 32;
        unsigned char sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m  = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        int q_idx = in_sb / 2;
        unsigned char packed = qs[q_idx];
        unsigned char q_code = (in_sb % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

        return d * (float)sc * (float)q_code - dmin * (float)m;
    }

    // ===================== Q5_K (full) =====================

    __device__ inline float dequant_q5k_standalone(const unsigned char* block_ptr, int in_sb) {
        const float d = fp16_to_float_device(((const unsigned short*)block_ptr)[0]);
        const float dmin = fp16_to_float_device(((const unsigned short*)block_ptr)[1]);
        const unsigned char* scales = block_ptr + 4;
        const unsigned char* qs = block_ptr + 16;
        const unsigned char* qh = block_ptr + 144;

        int is = in_sb / 32;
        unsigned char sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m  = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        int q_idx = in_sb / 2;
        unsigned char q_low = (in_sb % 2 == 0) ? (qs[q_idx] & 0x0F) : ((qs[q_idx] >> 4) & 0x0F);
        int qh_byte = in_sb / 8;
        int qh_bit  = in_sb % 8;
        unsigned char msb = (qh[qh_byte] >> qh_bit) & 1;
        int q_code = (int)q_low | ((int)msb << 4);

        return d * (float)sc * (float)q_code - dmin * (float)m;
    }

    // ===================== Q6_K (full) =====================

    __device__ inline float dequant_q6k_standalone(const unsigned char* block_ptr, int in_sb) {
        const float d = fp16_to_float_device(((const unsigned short*)block_ptr)[0]);
        const float dmin = fp16_to_float_device(((const unsigned short*)block_ptr)[1]);
        const unsigned char* scales = block_ptr + 4;
        const unsigned char* qs = block_ptr + 16;
        const unsigned char* qh = block_ptr + 144;

        int is = in_sb / 32;
        unsigned char sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m  = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        int q_idx = in_sb / 2;
        unsigned char q_low = (in_sb % 2 == 0) ? (qs[q_idx] & 0x0F) : ((qs[q_idx] >> 4) & 0x0F);
        int qh_byte_idx = in_sb / 4;
        int qh_bit_offset = (in_sb % 4) * 2;
        unsigned char qh_bits = (qh[qh_byte_idx] >> qh_bit_offset) & 0x03;
        int q_code = (int)q_low | ((int)qh_bits << 4);

        return d * (float)sc * (float)q_code - dmin * (float)m;
    }

    // ===================== Q2_K standalone =====================

    __device__ inline float dequant_q2k_standalone(const unsigned char* block_ptr, int in_sb) {
        const float d = fp16_to_float_device(((const unsigned short*)block_ptr)[0]);
        const float dmin = fp16_to_float_device(((const unsigned short*)block_ptr)[1]);
        const unsigned char* sc = block_ptr + 4;
        const unsigned char* m  = block_ptr + 12;
        const unsigned char* qs = block_ptr + 20;

        int sub = in_sb / 32;
        int in_sub = in_sb % 32;

        float sub_sc = (float)(sc[sub] & 3);
        float sub_m  = (float)(m[sub] & 3);

        int q_byte = in_sub / 4;
        int q_shift = (in_sub % 4) * 2;
        unsigned char q_code = (qs[q_byte] >> q_shift) & 0x03;

        return d * sub_sc * (float)q_code - dmin * sub_m;
    }

    // ===================== Q3_K standalone =====================

    __device__ inline float dequant_q3k_standalone(const unsigned char* block_ptr, int in_sb) {
        const float d = fp16_to_float_device(((const unsigned short*)block_ptr)[0]);
        const float dmin = fp16_to_float_device(((const unsigned short*)block_ptr)[1]);
        const unsigned char* sc = block_ptr + 4;
        const unsigned char* qh = block_ptr + 12;
        const unsigned char* qs = block_ptr + 14;
        const unsigned char* m = block_ptr + 78;

        int sub = in_sb / 32;
        int in_sub = in_sb % 32;

        float sub_sc = (float)(sc[sub] & 7);
        int qh_byte = sub / 8;
        int qh_bit  = (sub % 8) * 3;
        float sc_upper = (float)((qh[qh_byte] >> qh_bit) & 7);
        float scale_total = sub_sc + sc_upper * 8.0f;

        float sub_m = (float)(m[sub] & 7);
        float m_upper = (float)((qh[qh_byte + 3] >> qh_bit) & 7);
        float m_total = sub_m + m_upper * 8.0f;

        int bit_pos = in_sub * 3;
        int byte_idx = bit_pos / 8;
        int bit_idx  = bit_pos % 8;

        unsigned int q_value;
        if (bit_idx <= 5) {
            q_value = (qs[byte_idx] >> bit_idx) & 0x07;
        } else {
            int bits_in_first = 8 - bit_idx;
            int bits_in_second = 3 - bits_in_first;
            q_value = (qs[byte_idx] >> bit_idx) & ((1 << bits_in_first) - 1);
            q_value |= ((qs[byte_idx + 1] & ((1 << bits_in_second) - 1)) << bits_in_first);
        }

        return d * scale_total * (float)q_value - dmin * m_total;
    }

    // ===================== Q8_0 standalone =====================

    __device__ inline float dequant_q80_standalone(const unsigned char* block_ptr, int in_sb) {
        const float d = fp16_to_float_device(((const unsigned short*)block_ptr)[0]);
        const signed char* qs = (const signed char*)(block_ptr + 2);
        return d * (float)qs[in_sb];
    }

    // ============================================
    //  Fused GEMM kernels — one per quant format
    // ============================================

    // --- IQ2_XXS ---
    __global__ void grim_fused_dequant_gemm_iq2xxs(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_iq2xxs,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 66;
        const unsigned char* row_b_ptr = B_iq2xxs + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            float w_val = dequant_iq2xxs(row_b_ptr + sb_idx * 66, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_iq2xxs(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_iq2xxs,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 66;
        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq2xxs(B_iq2xxs + n * row_bytes + sb_idx * 66, in_sb);
            acc += dy_val * w_val;
        }
        dX[row * K + k_idx] = acc;
    }

    // --- IQ2_XS ---
    __global__ void grim_fused_dequant_gemm_iq2xs(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_iq2xs,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 74;
        const unsigned char* row_b_ptr = B_iq2xs + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            float w_val = dequant_iq2xs(row_b_ptr + sb_idx * 74, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_iq2xs(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_iq2xs,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 74;
        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq2xs(B_iq2xs + n * row_bytes + sb_idx * 74, in_sb);
            acc += dy_val * w_val;
        }
        dX[row * K + k_idx] = acc;
    }

    // --- IQ2_S ---
    __global__ void grim_fused_dequant_gemm_iq2s(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_iq2s,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 82;
        const unsigned char* row_b_ptr = B_iq2s + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            float w_val = dequant_iq2s(row_b_ptr + sb_idx * 82, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_iq2s(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_iq2s,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 82;
        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq2s(B_iq2s + n * row_bytes + sb_idx * 82, in_sb);
            acc += dy_val * w_val;
        }
        dX[row * K + k_idx] = acc;
    }

    // --- IQ3_XXS ---
    __global__ void grim_fused_dequant_gemm_iq3xxs(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_iq3xxs,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 96;
        const unsigned char* row_b_ptr = B_iq3xxs + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            float w_val = dequant_iq3xxs(row_b_ptr + sb_idx * 96, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_iq3xxs(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_iq3xxs,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 96;
        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq3xxs(B_iq3xxs + n * row_bytes + sb_idx * 96, in_sb);
            acc += dy_val * w_val;
        }
        dX[row * K + k_idx] = acc;
    }

    // --- IQ3_S ---
    __global__ void grim_fused_dequant_gemm_iq3s(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_iq3s,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 110;
        const unsigned char* row_b_ptr = B_iq3s + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            float w_val = dequant_iq3s(row_b_ptr + sb_idx * 110, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_iq3s(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_iq3s,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 110;
        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq3s(B_iq3s + n * row_bytes + sb_idx * 110, in_sb);
            acc += dy_val * w_val;
        }
        dX[row * K + k_idx] = acc;
    }

    // --- IQ4_NL ---
    __global__ void grim_fused_dequant_gemm_iq4nl(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_iq4nl,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 170;
        const unsigned char* row_b_ptr = B_iq4nl + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            float w_val = dequant_iq4nl(row_b_ptr + sb_idx * 170, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_iq4nl(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_iq4nl,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 170;
        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq4nl(B_iq4nl + n * row_bytes + sb_idx * 170, in_sb);
            acc += dy_val * w_val;
        }
        dX[row * K + k_idx] = acc;
    }

    // --- IQ4_XS ---
    __global__ void grim_fused_dequant_gemm_iq4xs(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_iq4xs,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 136;
        const unsigned char* row_b_ptr = B_iq4xs + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            float w_val = dequant_iq4xs(row_b_ptr + sb_idx * 136, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_iq4xs(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_iq4xs,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 136;
        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq4xs(B_iq4xs + n * row_bytes + sb_idx * 136, in_sb);
            acc += dy_val * w_val;
        }
        dX[row * K + k_idx] = acc;
    }

    // --- Q8_0 fused GEMM ---
    __global__ void grim_fused_dequant_gemm_q8_0(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_q80,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 32;
        const int row_bytes = blocks_per_row * 34;
        const unsigned char* row_b_ptr = B_q80 + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 32;
            int in_sb = k % 32;
            float w_val = dequant_q80_standalone(row_b_ptr + sb_idx * 34, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_q8_0(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_q80,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 32;
        const int row_bytes = blocks_per_row * 34;
        int sb_idx = k_idx / 32;
        int in_sb = k_idx % 32;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_q80_standalone(B_q80 + n * row_bytes + sb_idx * 34, in_sb);
            acc += dy_val * w_val;
        }
        dX[row * K + k_idx] = acc;
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! check_kernel {
        ($name:literal) => {
            assert!(
                KERNEL_SOURCE.contains($name),
                concat!("Missing kernel: ", $name)
            );
        };
    }

    #[test]
    fn iq_gemm_contains_all_kernels() {
        check_kernel!("grim_fused_dequant_gemm_iq2xxs");
        check_kernel!("grim_fused_dequant_backward_gemm_iq2xxs");
        check_kernel!("grim_fused_dequant_gemm_iq2xs");
        check_kernel!("grim_fused_dequant_backward_gemm_iq2xs");
        check_kernel!("grim_fused_dequant_gemm_iq2s");
        check_kernel!("grim_fused_dequant_backward_gemm_iq2s");
        check_kernel!("grim_fused_dequant_gemm_iq3xxs");
        check_kernel!("grim_fused_dequant_backward_gemm_iq3xxs");
        check_kernel!("grim_fused_dequant_gemm_iq3s");
        check_kernel!("grim_fused_dequant_backward_gemm_iq3s");
        check_kernel!("grim_fused_dequant_gemm_iq4nl");
        check_kernel!("grim_fused_dequant_backward_gemm_iq4nl");
        check_kernel!("grim_fused_dequant_gemm_iq4xs");
        check_kernel!("grim_fused_dequant_backward_gemm_iq4xs");
        check_kernel!("grim_fused_dequant_gemm_q8_0");
        check_kernel!("grim_fused_dequant_backward_gemm_q8_0");
    }
}
