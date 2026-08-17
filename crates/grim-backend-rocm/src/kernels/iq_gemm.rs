//! IQ2/IQ3/IQ4 Fused Dequantization GEMM HIP kernels (Crow Tier).

/// HIP source for all IQ-family fused dequant+GEMM kernels (forward + backward).
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    // ===================== IQ2 variants =====================

    // block_q2_XXS: 66 bytes per 256 weights.
    // Layout: d(f16,2) + qs(32) + signs(32) = 66.
    __device__ inline float dequant_iq2xxs(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* signs = blk + 34;
        int grid_idx = qs[in_sb / 8];
        float val = (float)((grid_idx + (in_sb % 8)) % 4) - 1.5f;
        float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return d * val * sign_val;
    }

    // block_q2_XS: 74 bytes per 256 weights.
    // Layout: d(f16,2) + qs(32) + scales(8) + signs(32) = 74.
    __device__ inline float dequant_iq2xs(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* scales = blk + 34;
        const unsigned char* signs = blk + 42;
        int sb = in_sb / 16;
        float sc = ((float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F)) * 0.125f + 0.5f;
        float scale = d * sc;
        int grid_idx = qs[in_sb / 8];
        float val = (float)((grid_idx + (in_sb % 8)) % 4) - 1.5f;
        float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return scale * val * sign_val;
    }

    // block_q2_S: 82 bytes per 256 weights.
    // Layout: d(f16,2) + qs(48) + scales(8) + signs(24) = 82.
    __device__ inline float dequant_iq2s(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* scales = blk + 50;
        const unsigned char* signs = blk + 58;
        int sb = in_sb / 16;
        float sc = ((float)((scales[sb / 2] >> ((sb % 2) * 4)) & 0x0F)) * 0.125f + 0.5f;
        float scale = d * sc;
        int grid_idx = qs[in_sb / 8];
        float val = (float)((grid_idx + (in_sb % 8)) % 4) - 1.5f;
        float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return scale * val * sign_val;
    }

    // ===================== IQ3 variants =====================

    // block_q3_XXS: 96 bytes per 256 weights.
    // Layout: d(f16,2) + qs(64) + signs(30) = 96.
    __device__ inline float dequant_iq3xxs(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* signs = blk + 66;
        int grid_idx = qs[in_sb / 8];
        int sub_idx = in_sb % 8;
        float base_val = (float)((grid_idx + sub_idx * 17) % 7) - 3.0f;
        int sign_byte_idx = (in_sb / 8);
        if (sign_byte_idx >= 30) sign_byte_idx = 29;
        float sign_val = ((signs[sign_byte_idx] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return d * base_val * 0.25f * sign_val;
    }

    // block_q3_S: 110 bytes per 256 weights.
    // Layout: d(f16,2) + qs(64) + scales(12) + signs(32) = 110.
    __device__ inline float dequant_iq3s(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* scales = blk + 66;
        const unsigned char* signs = blk + 78;
        int sb = in_sb / 32;
        float sc = ((float)(scales[sb * 12 / 8]) + 1.0f) * 0.125f;
        float scale = d * sc;
        float grid_val = (float)((qs[in_sb / 8] + in_sb) % 7) - 3.0f;
        float sign_val = ((signs[in_sb / 8] >> (in_sb % 8)) & 1) ? -1.0f : 1.0f;
        return scale * grid_val * sign_val;
    }

    // ===================== IQ4 variants =====================

    // block_q4_NL: 170 bytes per 256 weights, 4-bit codes + sign bits + group scales
    // Uses KVALUES_IQ4NL codebook (same as iq_dequant.rs).
    __device__ inline float dequant_iq4nl(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* signs = blk + 2;
        const unsigned char* qs = blk + 34;
        const unsigned char* sc = blk + 162;

        int group = in_sb / 16;
        float group_scale = (float)(sc[group] & 3);
        group_scale = 1.0f + 0.125f * group_scale;

        int sign_byte_idx = (in_sb / 8);
        int sign_bit = in_sb % 8;
        float sign_val = ((signs[sign_byte_idx] >> sign_bit) & 1) ? -1.0f : 1.0f;

        int q_byte = in_sb / 2;
        unsigned char q_code = (in_sb % 2 == 0) ? (qs[q_byte] & 0x0F) : ((qs[q_byte] >> 4) & 0x0F);

        static const float kvalues_iq4nl[16] = {
            -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
            1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 87.0f, 107.0f
        };
        float code_abs = kvalues_iq4nl[q_code];
        code_abs = code_abs < 0.0f ? -code_abs : code_abs;
        return d * group_scale * code_abs * sign_val;
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
    // Mirrors the corrected `dequant_q3k_element` in q3k_gemm.rs and the
    // authoritative CPU reference `grim_quant::dequant_q3k`. block_q3_K is
    // 110 bytes / 256 weights with NO `dmin` and NO `m` array; the value is
    // x = d * sc_i * q with the high bit of each 4-bit q taken from hmask.
    __device__ inline float dequant_q3k_standalone(const unsigned char* block_ptr, int in_sb) {
        const unsigned char* hmask  = block_ptr + 0;
        const unsigned char* qs     = block_ptr + 32;
        const unsigned char* scales = block_ptr + 96;
        float d = fp16_to_float_device(((const unsigned short*)(block_ptr + 108))[0]);

        int sub    = in_sb / 32;
        int is     = (in_sb % 32) / 16;
        int sc_idx = 2 * sub + is;
        int j      = sc_idx & 7;
        signed char sc_byte;
        if (sc_idx < 8) {
            sc_byte = (signed char)((scales[j] & 0x0F) | ((scales[j + 8] & 0x03) << 4));
        } else {
            sc_byte = (signed char)((scales[j] >> 4)   | ((scales[j + 8] & 0x0C) << 2));
        }
        float sc = (float)sc_byte - 32.0f;

        int in_sub = in_sb % 32;
        unsigned char hm_bit = (hmask[in_sub / 8] >> (in_sub % 8)) & 0x01;
        int col      = in_sb / 32;
        int byte_off = (col & 1) * 32;
        int shift    = (col & 6) >> 1;
        unsigned char qbits = (qs[in_sub + byte_off] >> (2 * shift)) & 0x03;
        int q_with_high = (int)qbits | (hm_bit ? 0 : 4);
        float q = (float)q_with_high - 4.0f;

        return d * sc * q;
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

        // P4: hoist every loop-invariant decode index out of the per-MAC N
        // loop. dX[row][k] = sum_n dY[row][n] * B[n][k] walks one packed
        // superblock per output row n (B varies with n), so the work that CAN
        // be hoisted per thread is the superblock/sub-block/nibble index math
        // — the old kernel recomputed `k/256`, `k%256`, `in_sb/2`, `%2`,
        // `(group*6)/8`, `%8` inside the N loop via a full `dequant_iq4xs`
        // call per MAC. The decode below is byte-for-byte the same as
        // `dequant_iq4xs`, with those indices precomputed once.
        const int superblock_idx = k_idx >> 8;      // k_idx / 256
        const int k_in_superblock = k_idx & 255;    // k_idx % 256
        const int group = k_in_superblock >> 5;     // 32-weight sub-block
        const int sc_byte_idx = (group * 6) >> 3;   // 6-bit scale byte
        const int sc_bit_offset = (group * 6) & 7;  // 6-bit scale shift
        const int q_byte = k_in_superblock >> 1;    // nibble byte
        const bool low_nibble = (k_in_superblock & 1) == 0;

        const int blocks_per_row = K / 256;
        const unsigned long long row_bytes = (unsigned long long)blocks_per_row * 136;

        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const unsigned char* blk = B_iq4xs + (unsigned long long)n * row_bytes
                                              + (unsigned long long)superblock_idx * 136;
            const float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
            const unsigned char* sc = blk + 2;
            const unsigned char* qs = blk + 8;
            unsigned int sc_val = (unsigned int)sc[sc_byte_idx] >> sc_bit_offset;
            if (sc_bit_offset > 2) {
                sc_val |= (unsigned int)sc[sc_byte_idx + 1] << (8 - sc_bit_offset);
            }
            sc_val &= 0x3F;
            const unsigned char q_code = low_nibble
                ? (unsigned char)(qs[q_byte] & 0x0F)
                : (unsigned char)((qs[q_byte] >> 4) & 0x0F);
            acc += dY[row * N + n] * (d * (float)sc_val * (float)q_code);
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
        const int sb_idx = k_idx / 32;
        const int in_sb = k_idx % 32;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_q80_standalone(B_q80 + (unsigned long long)n * row_bytes + sb_idx * 34, in_sb);
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
