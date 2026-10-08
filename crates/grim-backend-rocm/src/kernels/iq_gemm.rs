//! IQ2/IQ3/IQ4 Fused Dequantization GEMM HIP kernels (Crow Tier).

/// HIP source for all IQ-family fused dequant+GEMM kernels (forward + backward).
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

static const unsigned char IQ3S_GRID_B[512][4] = {
    {1,1,1,1},
    {3,1,1,1},
    {5,1,1,1},
    {11,1,1,1},
    {15,1,1,1},
    {1,3,1,1},
    {3,3,1,1},
    {5,3,1,1},
    {9,3,1,1},
    {13,3,1,1},
    {1,5,1,1},
    {3,5,1,1},
    {11,5,1,1},
    {7,7,1,1},
    {1,9,1,1},
    {5,9,1,1},
    {11,9,1,1},
    {15,9,1,1},
    {3,11,1,1},
    {7,11,1,1},
    {1,13,1,1},
    {5,13,1,1},
    {3,15,1,1},
    {9,15,1,1},
    {15,15,1,1},
    {1,1,3,1},
    {3,1,3,1},
    {5,1,3,1},
    {9,1,3,1},
    {1,3,3,1},
    {3,3,3,1},
    {11,3,3,1},
    {1,5,3,1},
    {7,5,3,1},
    {15,5,3,1},
    {3,7,3,1},
    {11,7,3,1},
    {9,9,3,1},
    {3,13,3,1},
    {11,13,3,1},
    {5,15,3,1},
    {1,1,5,1},
    {3,1,5,1},
    {11,1,5,1},
    {15,1,5,1},
    {1,3,5,1},
    {7,3,5,1},
    {13,3,5,1},
    {3,5,5,1},
    {11,5,5,1},
    {1,7,5,1},
    {9,7,5,1},
    {5,9,5,1},
    {11,9,5,1},
    {15,9,5,1},
    {3,11,5,1},
    {7,11,5,1},
    {1,15,5,1},
    {7,15,5,1},
    {7,1,7,1},
    {3,3,7,1},
    {11,3,7,1},
    {1,5,7,1},
    {5,5,7,1},
    {3,7,7,1},
    {7,7,7,1},
    {13,7,7,1},
    {9,9,7,1},
    {1,11,7,1},
    {5,11,7,1},
    {15,13,7,1},
    {3,15,7,1},
    {11,15,7,1},
    {1,1,9,1},
    {7,3,9,1},
    {15,3,9,1},
    {3,5,9,1},
    {9,5,9,1},
    {5,7,9,1},
    {1,9,9,1},
    {7,9,9,1},
    {3,11,9,1},
    {1,15,9,1},
    {5,1,11,1},
    {9,1,11,1},
    {1,5,11,1},
    {5,5,11,1},
    {13,5,11,1},
    {7,7,11,1},
    {3,9,11,1},
    {11,9,11,1},
    {15,9,11,1},
    {13,13,11,1},
    {7,15,11,1},
    {13,1,13,1},
    {3,3,13,1},
    {7,3,13,1},
    {3,7,13,1},
    {5,11,13,1},
    {3,15,13,1},
    {1,1,15,1},
    {5,1,15,1},
    {9,1,15,1},
    {1,5,15,1},
    {5,5,15,1},
    {13,5,15,1},
    {7,7,15,1},
    {1,11,15,1},
    {9,11,15,1},
    {1,1,1,3},
    {3,1,1,3},
    {5,1,1,3},
    {9,1,1,3},
    {1,3,1,3},
    {3,3,1,3},
    {7,3,1,3},
    {11,3,1,3},
    {15,3,1,3},
    {1,5,1,3},
    {5,5,1,3},
    {3,7,1,3},
    {9,7,1,3},
    {13,7,1,3},
    {9,11,1,3},
    {13,11,1,3},
    {3,13,1,3},
    {5,15,1,3},
    {1,1,3,3},
    {3,1,3,3},
    {7,1,3,3},
    {13,1,3,3},
    {1,3,3,3},
    {9,3,3,3},
    {3,5,3,3},
    {1,7,3,3},
    {7,7,3,3},
    {3,9,3,3},
    {1,11,3,3},
    {5,11,3,3},
    {1,15,3,3},
    {13,15,3,3},
    {1,1,5,3},
    {5,3,5,3},
    {11,3,5,3},
    {15,3,5,3},
    {1,5,5,3},
    {9,5,5,3},
    {5,7,5,3},
    {1,9,5,3},
    {7,9,5,3},
    {11,11,5,3},
    {1,13,5,3},
    {5,15,5,3},
    {3,1,7,3},
    {9,1,7,3},
    {15,1,7,3},
    {1,3,7,3},
    {7,3,7,3},
    {3,5,7,3},
    {15,5,7,3},
    {1,7,7,3},
    {9,7,7,3},
    {3,9,7,3},
    {5,13,7,3},
    {1,15,7,3},
    {7,1,9,3},
    {11,1,9,3},
    {5,3,9,3},
    {9,3,9,3},
    {3,7,9,3},
    {7,7,9,3},
    {5,9,9,3},
    {13,9,9,3},
    {1,11,9,3},
    {9,11,9,3},
    {3,1,11,3},
    {1,3,11,3},
    {7,3,11,3},
    {3,5,11,3},
    {1,7,11,3},
    {5,7,11,3},
    {3,11,11,3},
    {1,5,13,3},
    {9,5,13,3},
    {15,5,13,3},
    {9,9,13,3},
    {13,9,13,3},
    {3,1,15,3},
    {7,1,15,3},
    {1,3,15,3},
    {5,3,15,3},
    {3,5,15,3},
    {11,7,15,3},
    {3,9,15,3},
    {5,13,15,3},
    {1,15,15,3},
    {1,1,1,5},
    {3,1,1,5},
    {7,1,1,5},
    {11,1,1,5},
    {15,1,1,5},
    {1,3,1,5},
    {5,3,1,5},
    {9,3,1,5},
    {13,3,1,5},
    {3,5,1,5},
    {7,5,1,5},
    {15,5,1,5},
    {1,7,1,5},
    {5,7,1,5},
    {3,9,1,5},
    {7,9,1,5},
    {11,9,1,5},
    {1,11,1,5},
    {5,11,1,5},
    {15,13,1,5},
    {1,15,1,5},
    {7,15,1,5},
    {11,15,1,5},
    {1,1,3,5},
    {5,1,3,5},
    {1,3,3,5},
    {7,3,3,5},
    {15,3,3,5},
    {5,5,3,5},
    {11,5,3,5},
    {3,7,3,5},
    {9,7,3,5},
    {5,9,3,5},
    {3,11,3,5},
    {3,1,5,5},
    {9,1,5,5},
    {15,1,5,5},
    {3,5,5,5},
    {7,5,5,5},
    {1,7,5,5},
    {15,7,5,5},
    {3,9,5,5},
    {7,11,5,5},
    {15,11,5,5},
    {3,15,5,5},
    {9,15,5,5},
    {1,1,7,5},
    {5,1,7,5},
    {11,1,7,5},
    {3,3,7,5},
    {5,5,7,5},
    {9,5,7,5},
    {3,7,7,5},
    {7,7,7,5},
    {5,9,7,5},
    {1,11,7,5},
    {13,13,7,5},
    {3,1,9,5},
    {15,1,9,5},
    {1,5,9,5},
    {7,5,9,5},
    {5,7,9,5},
    {11,7,9,5},
    {3,9,9,5},
    {5,15,9,5},
    {11,15,9,5},
    {9,1,11,5},
    {3,3,11,5},
    {5,5,11,5},
    {15,7,11,5},
    {1,9,11,5},
    {7,11,11,5},
    {1,15,11,5},
    {1,1,13,5},
    {5,1,13,5},
    {15,1,13,5},
    {3,5,13,5},
    {11,11,13,5},
    {3,13,13,5},
    {11,1,15,5},
    {3,3,15,5},
    {13,5,15,5},
    {1,7,15,5},
    {7,9,15,5},
    {1,11,15,5},
    {5,1,1,7},
    {3,3,1,7},
    {7,3,1,7},
    {11,3,1,7},
    {15,3,1,7},
    {5,5,1,7},
    {3,7,1,7},
    {7,7,1,7},
    {11,7,1,7},
    {5,9,1,7},
    {9,9,1,7},
    {15,9,1,7},
    {3,11,1,7},
    {7,13,1,7},
    {3,15,1,7},
    {3,1,3,7},
    {7,1,3,7},
    {11,1,3,7},
    {9,3,3,7},
    {3,5,3,7},
    {7,5,3,7},
    {1,9,3,7},
    {1,13,3,7},
    {5,15,3,7},
    {13,15,3,7},
    {1,1,5,7},
    {5,3,5,7},
    {1,5,5,7},
    {5,7,5,7},
    {9,7,5,7},
    {1,11,5,7},
    {3,1,7,7},
    {1,3,7,7},
    {9,3,7,7},
    {3,5,7,7},
    {7,5,7,7},
    {15,5,7,7},
    {1,7,7,7},
    {3,9,7,7},
    {7,9,7,7},
    {15,9,7,7},
    {11,11,7,7},
    {7,15,7,7},
    {7,1,9,7},
    {3,3,9,7},
    {13,3,9,7},
    {5,5,9,7},
    {3,7,9,7},
    {5,11,9,7},
    {1,13,9,7},
    {9,13,9,7},
    {3,1,11,7},
    {1,3,11,7},
    {5,3,11,7},
    {11,5,11,7},
    {5,7,11,7},
    {9,9,11,7},
    {13,11,11,7},
    {7,15,11,7},
    {13,3,13,7},
    {3,9,13,7},
    {3,1,15,7},
    {7,1,15,7},
    {1,5,15,7},
    {5,5,15,7},
    {11,7,15,7},
    {1,1,1,9},
    {9,1,1,9},
    {5,3,1,9},
    {1,5,1,9},
    {9,5,1,9},
    {15,5,1,9},
    {5,7,1,9},
    {3,9,1,9},
    {1,11,1,9},
    {1,15,1,9},
    {5,1,3,9},
    {15,1,3,9},
    {3,3,3,9},
    {7,3,3,9},
    {5,5,3,9},
    {1,7,3,9},
    {11,7,3,9},
    {7,9,3,9},
    {3,11,3,9},
    {11,11,3,9},
    {3,1,5,9},
    {7,1,5,9},
    {1,3,5,9},
    {11,3,5,9},
    {3,5,5,9},
    {7,7,5,9},
    {1,9,5,9},
    {15,11,5,9},
    {5,13,5,9},
    {1,15,5,9},
    {9,1,7,9},
    {3,3,7,9},
    {7,3,7,9},
    {1,5,7,9},
    {5,5,7,9},
    {3,7,7,9},
    {11,7,7,9},
    {1,1,9,9},
    {5,1,9,9},
    {9,5,9,9},
    {15,7,9,9},
    {1,9,9,9},
    {3,15,9,9},
    {11,1,11,9},
    {15,1,11,9},
    {3,5,11,9},
    {5,13,11,9},
    {7,3,13,9},
    {9,7,13,9},
    {1,13,13,9},
    {1,3,15,9},
    {11,3,15,9},
    {1,7,15,9},
    {7,9,15,9},
    {3,11,15,9},
    {5,1,1,11},
    {1,3,1,11},
    {9,3,1,11},
    {5,5,1,11},
    {1,9,1,11},
    {9,9,1,11},
    {15,9,1,11},
    {5,11,1,11},
    {13,13,1,11},
    {9,15,1,11},
    {3,1,3,11},
    {7,1,3,11},
    {11,1,3,11},
    {5,3,3,11},
    {3,5,3,11},
    {5,7,3,11},
    {5,15,3,11},
    {1,1,5,11},
    {3,3,5,11},
    {7,5,5,11},
    {1,7,5,11},
    {13,7,5,11},
    {7,11,5,11},
    {5,1,7,11},
    {15,1,7,11},
    {1,3,7,11},
    {15,5,7,11},
    {9,9,7,11},
    {3,11,7,11},
    {11,13,7,11},
    {7,15,7,11},
    {3,1,9,11},
    {9,1,9,11},
    {1,5,9,11},
    {5,7,9,11},
    {13,9,9,11},
    {5,3,11,11},
    {13,5,11,11},
    {3,11,11,11},
    {7,11,11,11},
    {5,9,13,11},
    {5,1,15,11},
    {9,1,15,11},
    {5,5,15,11},
    {3,3,1,13},
    {7,3,1,13},
    {11,3,1,13},
    {3,7,1,13},
    {7,7,1,13},
    {1,13,1,13},
    {1,1,3,13},
    {1,5,3,13},
    {15,5,3,13},
    {9,13,3,13},
    {5,3,5,13},
    {9,7,5,13},
    {5,9,5,13},
    {11,11,5,13},
    {5,13,5,13},
    {1,15,5,13},
    {1,1,7,13},
    {9,3,7,13},
    {3,5,7,13},
    {1,9,7,13},
    {11,5,9,13},
    {7,9,9,13},
    {5,13,9,13},
    {1,1,11,13},
    {7,1,11,13},
    {9,7,11,13},
    {1,13,11,13},
    {11,1,13,13},
    {1,9,13,13},
    {3,3,15,13},
    {7,3,15,13},
    {1,1,1,15},
    {9,1,1,15},
    {15,1,1,15},
    {1,5,1,15},
    {5,5,1,15},
    {13,7,1,15},
    {1,9,1,15},
    {9,11,1,15},
    {5,13,1,15},
    {5,1,3,15},
    {3,3,3,15},
    {9,5,3,15},
    {7,9,3,15},
    {11,9,3,15},
    {3,1,5,15},
    {9,1,5,15},
    {1,3,5,15},
    {13,3,5,15},
    {3,5,5,15},
    {1,7,5,15},
    {3,11,5,15},
    {5,1,7,15},
    {5,7,7,15},
    {11,7,7,15},
    {7,11,7,15},
    {3,1,9,15},
    {11,1,9,15},
    {7,3,9,15},
    {1,5,9,15},
    {1,11,9,15},
    {5,5,11,15},
    {5,9,11,15},
    {5,1,13,15},
    {3,7,13,15},
    {1,1,15,15},
};

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

    // block_q3_S layout, mirroring grim_quant::dequant_iq3s exactly:
    //   d(f16,2) | qs(64) | qh(8) | signs(32) | scales(4)
    // 512-entry IQ3S grid of four signed bytes per entry; the 9th index bit
    // rides qh. One scale byte per 64 weights: lo nibble scales the first 32,
    // hi nibble the second 32 (db = d * (1 + 2*nibble)).
    __device__ inline float dequant_iq3s(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* qh = blk + 66;
        const unsigned char* signs = blk + 74;
        const unsigned char* scales = blk + 106;
        int k = in_sb / 64;           // scale byte (0..3)
        int t = in_sb / 32;           // 32-weight block (0..7)
        int e = in_sb % 32;
        int l = e / 8;                // grid pair row (0..3)
        int m = e % 8;                // element within the row's 8
        unsigned char sc = scales[k];
        unsigned char nib = ((in_sb % 64) < 32) ? (sc & 0xF) : (sc >> 4);
        float db = d * (1.0f + 2.0f * (float)nib);
        int qh_bit = 2 * l + (m >= 4 ? 1 : 0);
        int idx = qs[t * 8 + 2 * l + (m >= 4 ? 1 : 0)]
                | ((((int)qh[t] >> qh_bit) & 1) << 8);
        float g = (float)(signed char)IQ3S_GRID_B[idx][(m >= 4) ? (m - 4) : m];
        float sign = ((signs[t * 4 + l] >> m) & 1) ? -1.0f : 1.0f;
        return db * g * sign;
    }

    // 4 consecutive IQ3_S elements (in_sb % 4 == 0, within one 32-block).
    // They share the scale byte, quadrant (l / m>=4), grid row (idx) and
    // sign byte, so the per-element scattered IQ3S_GRID_B global fetches
    // collapse to ONE contiguous 4-byte row read. Bit-identical to four
    // dequant_iq3s calls.
    __device__ inline void dequant_iq3s_x4(const unsigned char* blk, int in_sb, float* out) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        const unsigned char* qs = blk + 2;
        const unsigned char* qh = blk + 66;
        const unsigned char* signs = blk + 74;
        const unsigned char* scales = blk + 106;
        int k = in_sb / 64;
        int t = in_sb / 32;
        int e0 = in_sb % 32;
        int l = e0 / 8;
        int m0 = e0 % 8;
        unsigned char sc = scales[k];
        unsigned char nib = (((unsigned int)(in_sb % 64)) < 32u) ? (sc & 0xF) : (sc >> 4);
        float db = d * (1.0f + 2.0f * (float)nib);
        int qh_bit = 2 * l + (m0 >= 4 ? 1 : 0);
        int idx = qs[t * 8 + 2 * l + (m0 >= 4 ? 1 : 0)]
                | ((((int)qh[t] >> qh_bit) & 1) << 8);
        unsigned char sign_byte = signs[t * 4 + l];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int m = m0 + j;
            int col = (m >= 4) ? (m - 4) : m;
            float g = (float)(signed char)IQ3S_GRID_B[idx][col];
            float sign = ((sign_byte >> m) & 1) ? -1.0f : 1.0f;
            out[j] = db * g * sign;
        }
    }

    // ===================== IQ4 variants =====================

    // block_iq4_nl: 18 bytes per 32 weights (QK4_NL = 32), 4-bit signed codes.
    // Layout: ggml_half d, then qs[QK4_NL/2] = 16 bytes. The sign is carried by
    // the codebook entry, so there is no sign plane and no per-group scale --
    // the previous version read a 32-byte sign plane at blk+2 and 8 sub-block
    // scales at blk+162, neither of which exists in this format.
    __device__ inline float dequant_iq4nl(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        // Reference order: element j takes the low nibble of qs[j] and
        // element j+16 takes the high nibble (dequantize_row_iq4_nl).
        int j = in_sb & 15;
        unsigned char qb = blk[2 + j];
        unsigned char q_code = (in_sb < 16) ? (qb & 0x0F) : ((qb >> 4) & 0x0F);

        // Canonical signed codebook (ggml kvalues_iq4nl), verbatim.
        static const float kvalues_iq4nl[16] = {
            -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
            1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f
        };
        return d * kvalues_iq4nl[q_code];
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
        const unsigned char* scales = block_ptr;
        const unsigned char* qs = block_ptr + 16;
        float d    = fp16_to_float_device(((const unsigned short*)(block_ptr + 80))[0]);
        float dmin = fp16_to_float_device(((const unsigned short*)(block_ptr + 82))[0]);

        int sub = in_sb / 16;
        int w   = in_sb % 16;

        float sc = (float)(scales[sub] & 0x0F);
        float m  = (float)(scales[sub] >> 4);

        // llama.cpp interleaved codes (ggml-quants.c:959): field
        // 2*((sub%8)/2) of byte qs[(sub/8)*32 + w + (sub%2)*16]. The
        // sequential 4-bytes-per-sub-block form exists in no released GGUF.
        int q_byte = (sub / 8) * 32 + w + (sub % 2) * 16;
        int q_shift = 2 * ((sub % 8) / 2);
        unsigned char q_code = (qs[q_byte] >> q_shift) & 0x03;

        return d * sc * (float)q_code - dmin * m;
    }

    // ===================== Q3_K standalone ===================== Mirrors the corrected `dequant_q3k_element` in q3k_gemm.rs and the authoritative CPU reference `grim_quant::dequant_q3k`.
    // block_q3_K is 110 bytes / 256 weights with NO `dmin` and NO `m` array; the.
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

    // Fused GEMM kernels - one per quant format

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
    static const unsigned long long IQ1S_GRID[2048] = {
0xffffffffffffffff,0xffffffffffffff01,0xffffffffffff0000,0xffffffffffff01ff,0xffffffffffff0101,0xffffffffff00ff00,0xffffffffff000000,0xffffffffff01ffff,0xffffffffff01ff01,0xffffffffff0101ff,0xffffffffff010101,0xffffffff00ff0000,0xffffffff0000ff00,0xffffffff000000ff,0xffffffff00000001,0xffffffff00010000,0xffffffff01ffffff,0xffffffff01ffff01,0xffffffff01ff01ff,0xffffffff01ff0101,0xffffffff01000000,0xffffffff0101ffff,0xffffffff0101ff01,0xffffffff010101ff,0xffffffff01010101,0xffffff00ffff00ff,0xffffff00ffff0000,0xffffff00ff00ff00,0xffffff00ff0000ff,0xffffff00ff000001,0xffffff00ff000100,0xffffff00ff000101,0xffffff00ff010000,0xffffff0000ffff00,0xffffff0000ff0001,0xffffff0000ff0100,0xffffff000000ff01,0xffffff0000000000,0xffffff0000000101,0xffffff000001ff00,0xffffff00000100ff,0xffffff0000010001,0xffffff00000101ff,0xffffff0001ff0000,0xffffff000100ff00,0xffffff00010000ff,0xffffff0001000001,0xffffff0001010000,0xffffff01ffffffff,0xffffff01ffffff01,0xffffff01ffff01ff,0xffffff01ffff0101,0xffffff01ff000000,0xffffff01ff01ffff,0xffffff01ff01ff01,0xffffff01ff0101ff,0xffffff01ff010101,0xffffff0100ff0000,0xffffff010000ff00,0xffffff0100000100,0xffffff01000100ff,0xffffff0100010100,0xffffff0101ffffff,0xffffff0101ffff01,0xffffff0101ff01ff,0xffffff0101ff0101,0xffffff010100ff00,0xffffff0101000000,0xffffff0101000100,0xffffff010101ffff,0xffffff010101ff01,0xffffff01010101ff,0xffffff0101010101,0xffff00ffff00ff00,0xffff00ffff0000ff,0xffff00ffff000001,0xffff00ffff010000,0xffff00ff00ffff00,0xffff00ff00ff0100,0xffff00ff00000000,0xffff00ff00000101,0xffff00ff000100ff,0xffff00ff00010000,0xffff00ff0100ff00,0xffff00ff01000100,0xffff00ff01010000,0xffff0000ffffff00,0xffff0000ffff00ff,0xffff0000ffff0000,0xffff0000ffff0001,0xffff0000ff000000,0xffff0000ff0001ff,0xffff0000ff000101,0xffff0000ff010100,0xffff000000ffffff,0xffff000000ff0000,0xffff000000ff0101,0xffff00000000ffff,0xffff00000000ff00,0xffff0000000000ff,0xffff000000000000,0xffff000000000001,0xffff000000000100,0xffff00000001ffff,0xffff00000001ff01,0xffff000000010000,0xffff0000000101ff,0xffff000000010101,0xffff000001ffff00,0xffff00000100ff00,0xffff000001000000,0xffff0000010001ff,0xffff000001000101,0xffff00000101ff00,0xffff0000010100ff,0xffff000001010000,0xffff000001010001,0xffff000001010100,0xffff0001ff0000ff,0xffff0001ff000100,0xffff000100ffff00,0xffff000100ff00ff,0xffff00010000ffff,0xffff00010000ff01,0xffff000100000000,0xffff0001000001ff,0xffff00010001ffff,0xffff00010001ff00,0xffff000100010001,0xffff000100010100,0xffff000101ff0000,0xffff00010100ff00,0xffff0001010000ff,0xffff000101000100,0xffff01ffffffffff,0xffff01ffffffff01,0xffff01ffffff01ff,0xffff01ffffff0101,0xffff01ffff000000,0xffff01ffff01ffff,0xffff01ffff01ff01,0xffff01ffff0101ff,0xffff01ffff010101,0xffff01ff00ff0000,0xffff01ff0000ff00,0xffff01ff00000001,0xffff01ff00010000,0xffff01ff01ffffff,0xffff01ff01ffff01,0xffff01ff01ff01ff,0xffff01ff01ff0101,0xffff01ff01000000,0xffff01ff0101ffff,0xffff01ff0101ff01,0xffff01ff010101ff,0xffff01ff01010101,0xffff0100ffff0000,0xffff0100ff00ff00,0xffff0100ff0000ff,0xffff0100ff000100,0xffff0100ff0100ff,0xffff0100ff010000,0xffff010000ffff00,0xffff01000000ffff,0xffff01000000ff00,0xffff010000000000,0xffff01000001ff00,0xffff0100000100ff,0xffff010000010100,0xffff01000100ff00,0xffff0100010000ff,0xffff010001000001,0xffff010001000100,0xffff010001010000,0xffff0101ffffffff,0xffff0101ffffff01,0xffff0101ffff01ff,0xffff0101ffff0101,0xffff0101ff000000,0xffff0101ff01ffff,0xffff0101ff01ff01,0xffff0101ff0101ff,0xffff0101ff010101,0xffff010100ff0000,0xffff01010000ff00,0xffff010100000100,0xffff01010001ff00,0xffff010100010000,0xffff010101ffffff,0xffff010101ffff01,0xffff010101ff0000,0xffff010101ff01ff,0xffff010101ff0101,0xffff010101000000,0xffff01010101ffff,0xffff01010101ff01,0xffff0101010101ff,0xffff010101010101,0xff00ffffff00ffff,0xff00ffffff00ff00,0xff00ffffff0000ff,0xff00ffffff000100,0xff00ffffff0100ff,0xff00ffffff010000,0xff00ffff00ffff00,0xff00ffff00ff00ff,0xff00ffff0000ffff,0xff00ffff00000000,0xff00ffff000001ff,0xff00ffff0001ff00,0xff00ffff000100ff,0xff00ffff00010000,0xff00ffff00010100,0xff00ffff0100ff00,0xff00ffff010000ff,0xff00ffff01000001,0xff00ffff0101ff00,0xff00ffff01010000,0xff00ff00ffffff00,0xff00ff00ffff00ff,0xff00ff00ffff0001,0xff00ff00ffff0100,0xff00ff00ff00ffff,0xff00ff00ff00ff01,0xff00ff00ff000000,0xff00ff00ff0001ff,0xff00ff00ff01ff00,0xff00ff00ff0100ff,0xff00ff00ff010100,0xff00ff0000ff0000,0xff00ff0000ff0101,0xff00ff000000ffff,0xff00ff000000ff00,0xff00ff000000ff01,0xff00ff00000000ff,0xff00ff0000000000,0xff00ff0000000001,0xff00ff0000000100,0xff00ff000001ffff,0xff00ff0000010000,0xff00ff0001ff00ff,0xff00ff000100ff01,0xff00ff0001000000,0xff00ff000101ff00,0xff00ff00010100ff,0xff00ff01ff00ff00,0xff00ff01ff0000ff,0xff00ff01ff000001,0xff00ff01ff010000,0xff00ff0100ffffff,0xff00ff0100ff0001,0xff00ff0100ff0100,0xff00ff010000ff01,0xff00ff0100000000,0xff00ff01000001ff,0xff00ff0100000101,0xff00ff01000100ff,0xff00ff0100010001,0xff00ff0101ff0000,0xff00ff010100ff00,0xff00ff01010000ff,0xff00ff0101000001,0xff00ff0101010000,0xff0000ffffffff00,0xff0000ffffff0001,0xff0000ffffff0100,0xff0000ffff0000ff,0xff0000ffff000000,0xff0000ffff0001ff,0xff0000ffff000100,0xff0000ffff01ff00,0xff0000ffff010001,0xff0000ff00ffff00,0xff0000ff00ff0000,0xff0000ff00ff0001,0xff0000ff00ff01ff,0xff0000ff00ff0101,0xff0000ff0000ff00,0xff0000ff000000ff,0xff0000ff00000000,0xff0000ff00000001,0xff0000ff00000100,0xff0000ff0001ff01,0xff0000ff00010000,0xff0000ff000101ff,0xff0000ff01ff00ff,0xff0000ff01ff0100,0xff0000ff0100ffff,0xff0000ff010000ff,0xff0000ff01000000,0xff0000ff010001ff,0xff0000ff01000100,0xff0000ff01000101,0xff0000ff0101ff00,0xff0000ff010100ff,0xff0000ff01010000,0xff0000ff01010100,0xff000000ffffff01,0xff000000ffff0000,0xff000000ffff0101,0xff000000ff00ff00,0xff000000ff0000ff,0xff000000ff000000,0xff000000ff000001,0xff000000ff000100,0xff000000ff01ffff,0xff000000ff01ff01,0xff000000ff010000,0xff000000ff0101ff,0xff000000ff010101,0xff00000000ffff00,0xff00000000ff00ff,0xff00000000ff0000,0xff00000000ff0001,0xff0000000000ff00,0xff0000000000ff01,0xff000000000000ff,0xff00000000000000,0xff00000000000001,0xff00000000000100,0xff00000000000101,0xff0000000001ff00,0xff000000000100ff,0xff00000000010000,0xff00000000010001,0xff00000000010100,0xff00000001ffffff,0xff00000001ffff01,0xff00000001ff00ff,0xff00000001ff0000,0xff00000001ff01ff,0xff00000001ff0101,0xff0000000100ffff,0xff0000000100ff00,0xff000000010000ff,0xff00000001000000,0xff00000001000001,0xff00000001000100,0xff00000001000101,0xff0000000101ffff,0xff0000000101ff01,0xff00000001010000,0xff000001ffffff00,0xff000001ffff00ff,0xff000001ffff0000,0xff000001ffff0001,0xff000001ff000000,0xff000001ff000001,0xff000001ff0001ff,0xff000001ff000101,0xff000001ff01ff00,0xff000001ff010001,0xff00000100ffffff,0xff00000100ffff01,0xff00000100ff00ff,0xff00000100ff0000,0xff00000100ff01ff,0xff00000100ff0101,0xff0000010000ff00,0xff00000100000000,0xff00000100000001,0xff000001000001ff,0xff00000100000100,0xff0000010001ff00,0xff000001000100ff,0xff00000100010000,0xff000001000101ff,0xff00000100010100,0xff00000100010101,0xff00000101ff0001,0xff00000101ff0101,0xff0000010100ff01,0xff00000101000000,0xff000001010100ff,0xff00000101010100,0xff0001ffff00ff00,0xff0001ffff000001,0xff0001ffff010000,0xff0001ff00ffff00,0xff0001ff00ff00ff,0xff0001ff00ff0001,0xff0001ff00ff0100,0xff0001ff0000ffff,0xff0001ff00000000,0xff0001ff000001ff,0xff0001ff00000101,0xff0001ff0001ffff,0xff0001ff0001ff00,0xff0001ff000100ff,0xff0001ff00010001,0xff0001ff00010100,0xff0001ff01ff0000,0xff0001ff0100ff00,0xff0001ff010000ff,0xff0001ff01010000,0xff000100ff00ffff,0xff000100ff00ff01,0xff000100ff000000,0xff000100ff000101,0xff000100ff01ff00,0xff000100ff010000,0xff00010000ffff01,0xff00010000ff00ff,0xff00010000ff0000,0xff00010000ff01ff,0xff0001000000ff00,0xff000100000000ff,0xff00010000000000,0xff00010000000001,0xff00010000000100,0xff00010000000101,0xff0001000001ffff,0xff00010000010000,0xff00010000010101,0xff00010001ff0100,0xff0001000100ff00,0xff0001000100ff01,0xff00010001000000,0xff000100010001ff,0xff0001000101ff00,0xff00010001010001,0xff00010001010100,0xff000101ffff0100,0xff000101ff000001,0xff000101ff0100ff,0xff000101ff010001,0xff00010100ff00ff,0xff00010100ff0001,0xff00010100ff0100,0xff0001010000ffff,0xff0001010000ff01,0xff00010100000000,0xff000101000001ff,0xff0001010001ff00,0xff00010100010001,0xff00010100010100,0xff00010101ff0000,0xff0001010100ff00,0xff00010101000001,0xff00010101000101,0xff01ffffffffffff,0xff01ffffffffff01,0xff01ffffffff01ff,0xff01ffffffff0101,0xff01ffffff000000,0xff01ffffff01ffff,0xff01ffffff01ff01,0xff01ffffff010000,0xff01ffffff0101ff,0xff01ffffff010101,0xff01ffff00ff0000,0xff01ffff0000ff00,0xff01ffff00000100,0xff01ffff0001ff00,0xff01ffff00010000,0xff01ffff01ffffff,0xff01ffff01ffff01,0xff01ffff01ff01ff,0xff01ffff01ff0101,0xff01ffff01000000,0xff01ffff0101ffff,0xff01ffff0101ff01,0xff01ffff01010000,0xff01ffff010101ff,0xff01ffff01010101,0xff01ff00ffff0000,0xff01ff00ff00ff00,0xff01ff00ff0000ff,0xff01ff00ff000100,0xff01ff00ff010000,0xff01ff0000ffff01,0xff01ff0000ff00ff,0xff01ff0000ff0100,0xff01ff0000000000,0xff01ff00000001ff,0xff01ff0000000101,0xff01ff000001ff00,0xff01ff00000100ff,0xff01ff0000010000,0xff01ff0000010001,0xff01ff0001ff0000,0xff01ff000100ffff,0xff01ff0001000001,0xff01ff0001000100,0xff01ff0001010000,0xff01ff01ffffff00,0xff01ff01ffff01ff,0xff01ff01ffff0101,0xff01ff01ff00ff00,0xff01ff01ff000000,0xff01ff01ff01ffff,0xff01ff01ff01ff01,0xff01ff01ff0101ff,0xff01ff01ff010101,0xff01ff0100ff0000,0xff01ff010000ff00,0xff01ff0100000001,0xff01ff0100000100,0xff01ff0100010000,0xff01ff0101ffff00,0xff01ff0101ff01ff,0xff01ff0101ff0101,0xff01ff010100ff00,0xff01ff0101000000,0xff01ff010101ffff,0xff01ff010101ff01,0xff01ff01010101ff,0xff01ff0101010101,0xff0100ffffff0000,0xff0100ffff0000ff,0xff0100ffff000001,0xff0100ffff000100,0xff0100ffff010000,0xff0100ff00ff00ff,0xff0100ff00ff0000,0xff0100ff00ff0001,0xff0100ff00ff0100,0xff0100ff0000ff01,0xff0100ff00000000,0xff0100ff000001ff,0xff0100ff00000101,0xff0100ff00010001,0xff0100ff01ff0000,0xff0100ff0100ff00,0xff0100ff010000ff,0xff0100ff01000100,0xff0100ff0101ff00,0xff0100ff01010000,0xff010000ffff0100,0xff010000ff000000,0xff010000ff01ff00,0xff010000ff010100,0xff01000000ffffff,0xff01000000ff0000,0xff01000000ff01ff,0xff0100000000ff00,0xff010000000000ff,0xff01000000000000,0xff01000000000100,0xff0100000001ff01,0xff01000000010000,0xff010000000101ff,0xff01000001ff0100,0xff0100000100ffff,0xff010000010000ff,0xff01000001000000,0xff010000010001ff,0xff01000001000101,0xff0100000101ff00,0xff010000010100ff,0xff01000001010001,0xff01000001010100,0xff010001ffff0000,0xff010001ff00ffff,0xff010001ff00ff01,0xff010001ff000100,0xff010001ff010000,0xff01000100ffff00,0xff01000100ff0100,0xff01000100000000,0xff0100010001ffff,0xff0100010001ff00,0xff01000100010100,0xff01000101ff00ff,0xff01000101ff0001,0xff0100010100ffff,0xff01000101000101,0xff0101ffffffffff,0xff0101ffffffff01,0xff0101ffffff01ff,0xff0101ffffff0101,0xff0101ffff000000,0xff0101ffff01ffff,0xff0101ffff01ff01,0xff0101ffff0101ff,0xff0101ffff010101,0xff0101ff00ff0000,0xff0101ff0000ff00,0xff0101ff000000ff,0xff0101ff00010000,0xff0101ff01ffffff,0xff0101ff01ffff01,0xff0101ff01ff01ff,0xff0101ff01ff0101,0xff0101ff0101ffff,0xff0101ff0101ff01,0xff0101ff010101ff,0xff0101ff01010101,0xff010100ffff0100,0xff010100ff00ff00,0xff010100ff0000ff,0xff010100ff000100,0xff010100ff010000,0xff01010000ff0001,0xff01010000ff0100,0xff0101000000ff01,0xff01010000000000,0xff0101000001ff00,0xff010100000100ff,0xff01010000010001,0xff01010000010100,0xff01010001ff0000,0xff0101000100ffff,0xff01010001000001,0xff01010001000100,0xff010100010100ff,0xff01010001010000,0xff010101ffffffff,0xff010101ffffff01,0xff010101ffff01ff,0xff010101ffff0101,0xff010101ff01ffff,0xff010101ff01ff01,0xff010101ff0101ff,0xff010101ff010101,0xff01010100ff0000,0xff0101010000ff00,0xff01010100000001,0xff01010100000100,0xff01010100010000,0xff01010101ffffff,0xff01010101ffff01,0xff01010101ff01ff,0xff01010101ff0101,0xff01010101000000,0xff0101010101ffff,0xff0101010101ff01,0xff010101010101ff,0xff01010101010101,0x00ffffffffff0000,0x00ffffffff00ff00,0x00ffffffff000001,0x00ffffffff010000,0x00ffffff00ff0100,0x00ffffff0000ff01,0x00ffffff00000000,0x00ffffff000001ff,0x00ffffff00000101,0x00ffffff0001ff00,0x00ffffff000100ff,0x00ffffff00010001,0x00ffffff010000ff,0x00ffffff01000100,0x00ffffff0101ff00,0x00ffffff01010001,0x00ffff00ffffffff,0x00ffff00ffffff00,0x00ffff00ffff00ff,0x00ffff00ffff0001,0x00ffff00ffff0100,0x00ffff00ff00ff01,0x00ffff00ff000000,0x00ffff00ff000001,0x00ffff00ff0001ff,0x00ffff00ff000101,0x00ffff00ff01ff00,0x00ffff00ff010001,0x00ffff00ff010100,0x00ffff0000ff0000,0x00ffff0000ff01ff,0x00ffff0000ff0101,0x00ffff000000ff00,0x00ffff00000000ff,0x00ffff0000000000,0x00ffff0000000001,0x00ffff0000000100,0x00ffff0000000101,0x00ffff0000010000,0x00ffff00000101ff,0x00ffff0000010101,0x00ffff0001ffff00,0x00ffff0001ff00ff,0x00ffff0001ff0001,0x00ffff000100ffff,0x00ffff000100ff01,0x00ffff0001000000,0x00ffff000101ffff,0x00ffff000101ff00,0x00ffff000101ff01,0x00ffff01ffff0000,0x00ffff01ff00ff00,0x00ffff01ff0000ff,0x00ffff01ff000001,0x00ffff01ff010000,0x00ffff0100ffff00,0x00ffff010000ff01,0x00ffff0100000000,0x00ffff0100000101,0x00ffff01000100ff,0x00ffff0100010100,0x00ffff0101ff0100,0x00ffff01010000ff,0x00ffff0101010000,0x00ff00ffffffff00,0x00ff00ffff000000,0x00ff00ffff000100,0x00ff00ffff010100,0x00ff00ff00ff0000,0x00ff00ff00ff01ff,0x00ff00ff00ff0101,0x00ff00ff0000ff00,0x00ff00ff000000ff,0x00ff00ff00000000,0x00ff00ff00000001,0x00ff00ff0001ff00,0x00ff00ff0001ff01,0x00ff00ff00010000,0x00ff00ff000101ff,0x00ff00ff00010101,0x00ff00ff01ffff00,0x00ff00ff01ff0001,0x00ff00ff01ff0100,0x00ff00ff0100ffff,0x00ff00ff0100ff01,0x00ff00ff01000000,0x00ff00ff0101ffff,0x00ff00ff0101ff00,0x00ff00ff01010100,0x00ff0000ffffff00,0x00ff0000ffffff01,0x00ff0000ffff0000,0x00ff0000ffff0101,0x00ff0000ff00ff00,0x00ff0000ff0000ff,0x00ff0000ff000000,0x00ff0000ff000001,0x00ff0000ff000100,0x00ff0000ff01ffff,0x00ff0000ff010000,0x00ff0000ff010101,0x00ff000000ffff00,0x00ff000000ff00ff,0x00ff000000ff0000,0x00ff000000ff0001,0x00ff000000ff0100,0x00ff00000000ffff,0x00ff00000000ff00,0x00ff0000000000ff,0x00ff000000000000,0x00ff000000000001,0x00ff0000000001ff,0x00ff000000000100,0x00ff00000001ff00,0x00ff0000000100ff,0x00ff000000010000,0x00ff000000010001,0x00ff000000010100,0x00ff000001ffff01,0x00ff000001ff00ff,0x00ff000001ff0000,0x00ff000001ff01ff,0x00ff00000100ff00,0x00ff0000010000ff,0x00ff000001000000,0x00ff000001000001,0x00ff000001000100,0x00ff000001000101,0x00ff000001010000,0x00ff0000010101ff,0x00ff000001010101,0x00ff0001ffffff00,0x00ff0001ffff0000,0x00ff0001ffff0100,0x00ff0001ff0000ff,0x00ff0001ff000000,0x00ff0001ff0001ff,0x00ff0001ff000101,0x00ff0001ff01ff00,0x00ff0001ff0100ff,0x00ff0001ff010100,0x00ff000100ffffff,0x00ff000100ffff01,0x00ff000100ff0000,0x00ff000100ff01ff,0x00ff00010000ffff,0x00ff00010000ff00,0x00ff00010000ff01,0x00ff000100000000,0x00ff000100000001,0x00ff000100000100,0x00ff00010001ff01,0x00ff000100010000,0x00ff0001000101ff,0x00ff000101ffff00,0x00ff000101ff0000,0x00ff000101ff0101,0x00ff0001010000ff,0x00ff000101000000,0x00ff00010101ff00,0x00ff0001010100ff,0x00ff000101010001,0x00ff01ffffff0000,0x00ff01ffff00ff00,0x00ff01ffff000000,0x00ff01ffff000101,0x00ff01ffff010000,0x00ff01ff00ffff01,0x00ff01ff00ff0100,0x00ff01ff0000ffff,0x00ff01ff00000000,0x00ff01ff000001ff,0x00ff01ff0001ff00,0x00ff01ff000100ff,0x00ff01ff00010001,0x00ff01ff00010100,0x00ff01ff01ff0000,0x00ff01ff0100ff00,0x00ff01ff010000ff,0x00ff01ff01000001,0x00ff01ff01000100,0x00ff01ff01010000,0x00ff0100ffffff00,0x00ff0100ffff0000,0x00ff0100ffff0001,0x00ff0100ffff0101,0x00ff0100ff00ffff,0x00ff0100ff0000ff,0x00ff0100ff000000,0x00ff0100ff0001ff,0x00ff0100ff01ff00,0x00ff0100ff0100ff,0x00ff0100ff010001,0x00ff010000ffffff,0x00ff010000ff0000,0x00ff010000ff0101,0x00ff01000000ff00,0x00ff01000000ff01,0x00ff0100000000ff,0x00ff010000000000,0x00ff010000000001,0x00ff010000000100,0x00ff01000001ffff,0x00ff01000001ff01,0x00ff010000010000,0x00ff010000010001,0x00ff010000010101,0x00ff010001ff0001,0x00ff010001ff0100,0x00ff01000100ff01,0x00ff010001000000,0x00ff010001000001,0x00ff0100010001ff,0x00ff01000101ff00,0x00ff0100010100ff,0x00ff010001010001,0x00ff010001010100,0x00ff0101ff000001,0x00ff010100ff00ff,0x00ff010100ff0001,0x00ff010100ff0100,0x00ff010100000000,0x00ff0101000001ff,0x00ff010100000101,0x00ff0101000100ff,0x00ff010100010100,0x00ff0101010000ff,0x00ff010101010000,0x0000ffffffffff00,0x0000ffffffff00ff,0x0000ffffffff0000,0x0000ffffffff0001,0x0000ffffffff0100,0x0000ffffff00ff01,0x0000ffffff000000,0x0000ffffff000101,0x0000ffffff01ff00,0x0000ffffff0100ff,0x0000ffffff010100,0x0000ffff00ffffff,0x0000ffff00ff0000,0x0000ffff00ff01ff,0x0000ffff0000ff00,0x0000ffff000000ff,0x0000ffff00000000,0x0000ffff00000001,0x0000ffff00000100,0x0000ffff00010000,0x0000ffff000101ff,0x0000ffff01ff0001,0x0000ffff01ff0100,0x0000ffff01000000,0x0000ffff010001ff,0x0000ffff0101ffff,0x0000ffff0101ff00,0x0000ffff01010001,0x0000ffff01010100,0x0000ff00ffff0000,0x0000ff00ffff01ff,0x0000ff00ffff0100,0x0000ff00ffff0101,0x0000ff00ff00ff00,0x0000ff00ff0000ff,0x0000ff00ff000000,0x0000ff00ff000001,0x0000ff00ff0001ff,0x0000ff00ff000100,0x0000ff00ff01ffff,0x0000ff00ff010000,0x0000ff00ff010001,0x0000ff00ff0101ff,0x0000ff00ff010101,0x0000ff0000ffff00,0x0000ff0000ff00ff,0x0000ff0000ff0000,0x0000ff0000ff0001,0x0000ff0000ff0100,0x0000ff000000ffff,0x0000ff000000ff00,0x0000ff000000ff01,0x0000ff00000000ff,0x0000ff0000000000,0x0000ff0000000001,0x0000ff00000001ff,0x0000ff0000000100,0x0000ff0000000101,0x0000ff000001ff00,0x0000ff00000100ff,0x0000ff0000010000,0x0000ff0000010001,0x0000ff0000010100,0x0000ff0001ffff01,0x0000ff0001ff0000,0x0000ff000100ff00,0x0000ff00010000ff,0x0000ff0001000000,0x0000ff0001000001,0x0000ff0001000100,0x0000ff000101ffff,0x0000ff0001010000,0x0000ff0001010101,0x0000ff01ffffff00,0x0000ff01ffff0001,0x0000ff01ff00ff01,0x0000ff01ff000000,0x0000ff01ff000101,0x0000ff01ff01ff00,0x0000ff01ff0100ff,0x0000ff0100ffff01,0x0000ff0100ff0000,0x0000ff0100ff0101,0x0000ff010000ff00,0x0000ff01000000ff,0x0000ff0100000000,0x0000ff0100000001,0x0000ff0100000100,0x0000ff010001ff01,0x0000ff0100010000,0x0000ff0101ff0000,0x0000ff010100ffff,0x0000ff010100ff01,0x0000ff0101000000,0x0000ff0101000100,0x0000ff0101000101,0x0000ff01010100ff,0x000000ffffff00ff,0x000000ffffff0000,0x000000ffff00ff00,0x000000ffff0000ff,0x000000ffff000000,0x000000ffff000001,0x000000ffff0001ff,0x000000ffff000100,0x000000ffff01ff00,0x000000ffff010000,0x000000ffff0101ff,0x000000ffff010101,0x000000ff00ffff00,0x000000ff00ff00ff,0x000000ff00ff0000,0x000000ff00ff0001,0x000000ff00ff0100,0x000000ff00ff0101,0x000000ff0000ffff,0x000000ff0000ff00,0x000000ff000000ff,0x000000ff00000000,0x000000ff00000001,0x000000ff000001ff,0x000000ff00000100,0x000000ff00000101,0x000000ff0001ff00,0x000000ff0001ff01,0x000000ff000100ff,0x000000ff00010000,0x000000ff00010001,0x000000ff00010100,0x000000ff01ffffff,0x000000ff01ff01ff,0x000000ff01ff0101,0x000000ff0100ff00,0x000000ff010000ff,0x000000ff01000000,0x000000ff01000001,0x000000ff01000100,0x000000ff0101ff00,0x000000ff010100ff,0x000000ff01010000,0x000000ff01010101,0x00000000ffffff00,0x00000000ffffff01,0x00000000ffff00ff,0x00000000ffff0000,0x00000000ffff0001,0x00000000ffff0100,0x00000000ff00ffff,0x00000000ff00ff00,0x00000000ff00ff01,0x00000000ff0000ff,0x00000000ff000000,0x00000000ff000001,0x00000000ff000100,0x00000000ff000101,0x00000000ff01ff00,0x00000000ff0100ff,0x00000000ff010000,0x00000000ff010001,0x00000000ff010100,0x0000000000ffffff,0x0000000000ffff00,0x0000000000ffff01,0x0000000000ff00ff,0x0000000000ff0000,0x0000000000ff0001,0x0000000000ff01ff,0x0000000000ff0100,0x000000000000ffff,0x000000000000ff00,0x000000000000ff01,0x00000000000000ff,0x0000000000000000,0x0000000000000001,0x00000000000001ff,0x0000000000000100,0x0000000000000101,0x000000000001ffff,0x000000000001ff00,0x00000000000100ff,0x0000000000010000,0x0000000000010001,0x00000000000101ff,0x0000000000010100,0x0000000000010101,0x0000000001ffff00,0x0000000001ff00ff,0x0000000001ff0000,0x0000000001ff0100,0x0000000001ff0101,0x000000000100ffff,0x000000000100ff00,0x00000000010000ff,0x0000000001000000,0x0000000001000001,0x00000000010001ff,0x0000000001000100,0x000000000101ff00,0x00000000010100ff,0x0000000001010000,0x0000000001010001,0x0000000001010100,0x00000001ffffffff,0x00000001ffffff00,0x00000001ffffff01,0x00000001ffff00ff,0x00000001ffff0001,0x00000001ffff01ff,0x00000001ffff0100,0x00000001ff00ff00,0x00000001ff0000ff,0x00000001ff000000,0x00000001ff0001ff,0x00000001ff000100,0x00000001ff01ffff,0x00000001ff01ff00,0x00000001ff01ff01,0x00000001ff0100ff,0x00000001ff010000,0x00000001ff010001,0x00000001ff0101ff,0x00000001ff010100,0x0000000100ffff00,0x0000000100ff0000,0x0000000100ff0001,0x0000000100ff01ff,0x0000000100ff0100,0x0000000100ff0101,0x000000010000ffff,0x000000010000ff00,0x000000010000ff01,0x00000001000000ff,0x0000000100000000,0x0000000100000001,0x00000001000001ff,0x0000000100000100,0x0000000100000101,0x000000010001ff00,0x00000001000100ff,0x0000000100010000,0x0000000100010100,0x0000000101ffff01,0x0000000101ff0000,0x0000000101ff0001,0x0000000101ff01ff,0x0000000101ff0100,0x0000000101ff0101,0x000000010100ff00,0x0000000101000000,0x0000000101000101,0x000000010101ff01,0x0000000101010000,0x0000000101010001,0x00000001010101ff,0x0000000101010100,0x000001ffffff00ff,0x000001ffffff0000,0x000001ffffff0001,0x000001ffffff0100,0x000001ffff00ffff,0x000001ffff000000,0x000001ffff0001ff,0x000001ffff01ff00,0x000001ffff010101,0x000001ff00ff0000,0x000001ff00ff01ff,0x000001ff00ff0101,0x000001ff0000ff00,0x000001ff000000ff,0x000001ff00000000,0x000001ff00000001,0x000001ff000001ff,0x000001ff00000100,0x000001ff0001ffff,0x000001ff0001ff01,0x000001ff000100ff,0x000001ff00010000,0x000001ff01ffff01,0x000001ff01ff0100,0x000001ff0100ffff,0x000001ff0100ff01,0x000001ff01000000,0x000001ff010001ff,0x000001ff0101ff00,0x000001ff01010100,0x00000100ffffff00,0x00000100ffffff01,0x00000100ffff0000,0x00000100ffff0101,0x00000100ff00ff00,0x00000100ff0000ff,0x00000100ff000000,0x00000100ff000001,0x00000100ff000100,0x00000100ff010000,0x0000010000ffff00,0x0000010000ff00ff,0x0000010000ff0000,0x0000010000ff0001,0x0000010000ff0100,0x000001000000ffff,0x000001000000ff00,0x000001000000ff01,0x00000100000000ff,0x0000010000000000,0x0000010000000001,0x00000100000001ff,0x0000010000000100,0x0000010000000101,0x000001000001ff00,0x00000100000100ff,0x0000010000010000,0x0000010000010001,0x0000010000010100,0x0000010001ffff00,0x0000010001ff0000,0x0000010001ff0100,0x000001000100ff00,0x00000100010000ff,0x0000010001000000,0x0000010001000001,0x00000100010001ff,0x0000010001000100,0x0000010001010000,0x00000101ffff00ff,0x00000101ffff01ff,0x00000101ff000000,0x00000101ff000101,0x00000101ff01ffff,0x00000101ff010000,0x00000101ff010001,0x00000101ff010100,0x0000010100ff0000,0x0000010100ff01ff,0x0000010100ff0100,0x000001010000ff00,0x0000010100000000,0x0000010100000001,0x00000101000001ff,0x0000010100000100,0x000001010001ff01,0x0000010100010000,0x00000101000101ff,0x0000010100010101,0x0000010101ffff00,0x0000010101ff0101,0x000001010100ff01,0x0000010101000000,0x0000010101000001,0x00000101010001ff,0x0000010101000101,0x000001010101ff00,0x0001ffffffff0000,0x0001ffffff0000ff,0x0001ffffff000001,0x0001ffffff000100,0x0001ffffff010000,0x0001ffff00ff00ff,0x0001ffff0000ffff,0x0001ffff00000000,0x0001ffff00000001,0x0001ffff000001ff,0x0001ffff00000101,0x0001ffff0001ff00,0x0001ffff000100ff,0x0001ffff00010001,0x0001ffff00010100,0x0001ffff01ffff00,0x0001ffff01000001,0x0001ffff01010000,0x0001ff00ffffff00,0x0001ff00ffff00ff,0x0001ff00ffff0001,0x0001ff00ffff0100,0x0001ff00ff00ff01,0x0001ff00ff000000,0x0001ff00ff01ff00,0x0001ff00ff01ff01,0x0001ff00ff010001,0x0001ff00ff010100,0x0001ff0000ff0000,0x0001ff0000ff0100,0x0001ff000000ff00,0x0001ff0000000000,0x0001ff0000000001,0x0001ff0000000100,0x0001ff0000010000,0x0001ff0000010001,0x0001ff0000010101,0x0001ff0001ff00ff,0x0001ff0001ff0101,0x0001ff000100ff01,0x0001ff0001000000,0x0001ff000101ff00,0x0001ff0001010001,0x0001ff0001010100,0x0001ff01ff00ff00,0x0001ff01ff000001,0x0001ff01ff000100,0x0001ff0100ffffff,0x0001ff0100ffff00,0x0001ff0100ff0001,0x0001ff0100000000,0x0001ff0100000001,0x0001ff01000001ff,0x0001ff010001ffff,0x0001ff0101ff0000,0x0001ff010100ff00,0x0001ff0101000001,0x0001ff0101010000,0x000100ffff00ff00,0x000100ffff00ff01,0x000100ffff000000,0x000100ffff000001,0x000100ffff000101,0x000100ffff01ff00,0x000100ffff010001,0x000100ffff010100,0x000100ff00ffffff,0x000100ff00ffff01,0x000100ff00ff0000,0x000100ff00ff01ff,0x000100ff00ff0101,0x000100ff0000ff00,0x000100ff000000ff,0x000100ff00000000,0x000100ff00000001,0x000100ff00000100,0x000100ff00000101,0x000100ff0001ffff,0x000100ff0001ff01,0x000100ff00010000,0x000100ff01ff00ff,0x000100ff01ff0000,0x000100ff01ff0100,0x000100ff0100ffff,0x000100ff0100ff01,0x000100ff010000ff,0x000100ff01000000,0x000100ff01000001,0x000100ff010001ff,0x000100ff01000101,0x000100ff0101ff00,0x000100ff010100ff,0x000100ff01010100,0x00010000ffff0000,0x00010000ffff01ff,0x00010000ffff0101,0x00010000ff00ff00,0x00010000ff000000,0x00010000ff000001,0x00010000ff000100,0x0001000000ff00ff,0x0001000000ff0000,0x0001000000ff0001,0x0001000000ff0100,0x000100000000ffff,0x000100000000ff00,0x00010000000000ff,0x0001000000000000,0x0001000000000001,0x0001000000000100,0x000100000001ff00,0x00010000000100ff,0x0001000000010000,0x0001000000010001,0x0001000000010100,0x0001000001ff0001,0x0001000001ff0100,0x0001000001ff0101,0x000100000100ff00,0x0001000001000000,0x0001000001000001,0x0001000001000100,0x0001000001000101,0x000100000101ff01,0x0001000001010000,0x0001000001010001,0x00010000010101ff,0x00010001ffffff01,0x00010001ffff0100,0x00010001ff000000,0x00010001ff01ffff,0x00010001ff010001,0x00010001ff0101ff,0x00010001ff010100,0x0001000100ffffff,0x0001000100ff0000,0x0001000100ff01ff,0x0001000100ff0101,0x000100010000ff00,0x00010001000000ff,0x0001000100000000,0x0001000100000001,0x00010001000001ff,0x0001000100000101,0x000100010001ffff,0x0001000100010000,0x00010001000101ff,0x0001000101ffffff,0x0001000101ffff01,0x0001000101ff0000,0x0001000101ff0101,0x00010001010000ff,0x0001000101000001,0x00010001010001ff,0x0001000101000100,0x000100010101ffff,0x00010001010100ff,0x0001000101010001,0x0001000101010101,0x000101ffff000001,0x000101ffff000100,0x000101ffff010000,0x000101ff00ffff00,0x000101ff0000ff01,0x000101ff00000000,0x000101ff00000101,0x000101ff0001ff00,0x000101ff00010100,0x000101ff01ff0000,0x000101ff0100ff00,0x000101ff010001ff,0x000101ff01010001,0x00010100ffffff00,0x00010100ffff00ff,0x00010100ff00ffff,0x00010100ff000000,0x00010100ff01ff00,0x00010100ff0100ff,0x00010100ff010001,0x00010100ff010100,0x0001010000ffffff,0x0001010000ffff00,0x0001010000ff0000,0x0001010000ff0001,0x0001010000ff01ff,0x000101000000ff00,0x00010100000000ff,0x0001010000000000,0x0001010000000001,0x0001010000000100,0x000101000001ffff,0x0001010000010000,0x0001010000010101,0x0001010001ffff01,0x0001010001ff00ff,0x0001010001ff0101,0x0001010001000000,0x000101000101ff00,0x00010100010100ff,0x0001010001010000,0x0001010001010100,0x00010101ff00ff00,0x00010101ff000001,0x00010101ff0001ff,0x0001010100ffff00,0x0001010100ff00ff,0x0001010100ff0100,0x000101010000ffff,0x0001010100000000,0x00010101000001ff,0x0001010100000101,0x00010101000100ff,0x0001010100010000,0x0001010100010100,0x0001010101ff0001,0x00010101010000ff,0x00010101010001ff,0x0001010101000101,0x0001010101010001,0x01ffffffffffffff,0x01ffffffffffff01,0x01ffffffffff01ff,0x01ffffffffff0101,0x01ffffffff01ffff,0x01ffffffff01ff01,0x01ffffffff0101ff,0x01ffffffff010101,0x01ffffff00ff0000,0x01ffffff0000ffff,0x01ffffff0000ff00,0x01ffffff000000ff,0x01ffffff00000001,0x01ffffff00000100,0x01ffffff00010000,0x01ffffff01ffffff,0x01ffffff01ffff01,0x01ffffff01ff01ff,0x01ffffff01ff0101,0x01ffffff01000000,0x01ffffff0101ffff,0x01ffffff0101ff01,0x01ffffff010101ff,0x01ffffff01010101,0x01ffff00ffff0000,0x01ffff00ff00ff00,0x01ffff00ff0000ff,0x01ffff00ff000001,0x01ffff00ff000100,0x01ffff00ff010000,0x01ffff0000ffff00,0x01ffff0000ff00ff,0x01ffff0000ff0100,0x01ffff000000ffff,0x01ffff000000ff01,0x01ffff0000000000,0x01ffff0000000001,0x01ffff00000001ff,0x01ffff0000000100,0x01ffff00000100ff,0x01ffff0000010001,0x01ffff0000010100,0x01ffff0001ff0000,0x01ffff0001ff0100,0x01ffff00010000ff,0x01ffff0001000001,0x01ffff0001000100,0x01ffff0001010000,0x01ffff01ffffffff,0x01ffff01ffffff01,0x01ffff01ffff01ff,0x01ffff01ffff0101,0x01ffff01ff000000,0x01ffff01ff01ffff,0x01ffff01ff01ff01,0x01ffff01ff0101ff,0x01ffff01ff010101,0x01ffff010000ff00,0x01ffff01000000ff,0x01ffff0100000100,0x01ffff0100010000,0x01ffff0101ffffff,0x01ffff0101ffff01,0x01ffff0101ff01ff,0x01ffff0101ff0101,0x01ffff0101000000,0x01ffff010101ffff,0x01ffff010101ff01,0x01ffff01010101ff,0x01ffff0101010101,0x01ff00ffff0000ff,0x01ff00ffff000100,0x01ff00ff00ffff00,0x01ff00ff00ff00ff,0x01ff00ff0000ff00,0x01ff00ff00000000,0x01ff00ff00000101,0x01ff00ff0001ff00,0x01ff00ff000100ff,0x01ff00ff00010100,0x01ff00ff010000ff,0x01ff00ff01000100,0x01ff0000ffffff00,0x01ff0000ffff0100,0x01ff0000ff00ff01,0x01ff0000ff000000,0x01ff0000ff000101,0x01ff0000ff010001,0x01ff0000ff010100,0x01ff000000ffffff,0x01ff000000ffff00,0x01ff000000ff0000,0x01ff000000ff01ff,0x01ff00000000ff00,0x01ff0000000000ff,0x01ff000000000000,0x01ff000000000001,0x01ff000000000100,0x01ff000000000101,0x01ff000000010000,0x01ff000000010001,0x01ff0000000101ff,0x01ff000000010101,0x01ff000001ffff00,0x01ff000001ff00ff,0x01ff000001ff0001,0x01ff000001ff0100,0x01ff00000100ffff,0x01ff00000100ff01,0x01ff000001000000,0x01ff0000010001ff,0x01ff000001010001,0x01ff0001ff00ff00,0x01ff0001ff000001,0x01ff0001ff000100,0x01ff0001ff010000,0x01ff000100ffff00,0x01ff000100ff00ff,0x01ff000100ff0100,0x01ff000100ff0101,0x01ff00010000ffff,0x01ff000100000000,0x01ff000100000100,0x01ff000100000101,0x01ff00010001ff00,0x01ff000100010001,0x01ff000100010101,0x01ff000101ff0000,0x01ff00010100ff00,0x01ff000101000101,0x01ff0001010100ff,0x01ff01ffffffffff,0x01ff01ffffffff01,0x01ff01ffffff01ff,0x01ff01ffffff0101,0x01ff01ffff000000,0x01ff01ffff01ffff,0x01ff01ffff01ff01,0x01ff01ffff0101ff,0x01ff01ffff010101,0x01ff01ff00ffff00,0x01ff01ff00ff0000,0x01ff01ff0000ff00,0x01ff01ff000000ff,0x01ff01ff00000100,0x01ff01ff00010000,0x01ff01ff00010100,0x01ff01ff01ffffff,0x01ff01ff01ffff01,0x01ff01ff01ff01ff,0x01ff01ff01ff0101,0x01ff01ff01000000,0x01ff01ff0101ffff,0x01ff01ff0101ff01,0x01ff01ff010101ff,0x01ff01ff01010101,0x01ff0100ffff0000,0x01ff0100ffff0001,0x01ff0100ff00ff00,0x01ff0100ff0000ff,0x01ff0100ff000001,0x01ff0100ff010000,0x01ff010000ffff00,0x01ff010000ff00ff,0x01ff010000ff0001,0x01ff010000ff0100,0x01ff01000000ffff,0x01ff01000000ff01,0x01ff010000000000,0x01ff010000000101,0x01ff01000001ff00,0x01ff0100000100ff,0x01ff010001ff0000,0x01ff010001000001,0x01ff010001000100,0x01ff010001010000,0x01ff0101ffffffff,0x01ff0101ffffff01,0x01ff0101ffff01ff,0x01ff0101ffff0101,0x01ff0101ff000000,0x01ff0101ff01ffff,0x01ff0101ff01ff01,0x01ff0101ff0101ff,0x01ff0101ff010101,0x01ff010100ff0000,0x01ff01010000ff00,0x01ff0101000000ff,0x01ff010100000001,0x01ff010101ffffff,0x01ff010101ffff01,0x01ff010101ff01ff,0x01ff010101ff0101,0x01ff010101000000,0x01ff01010101ffff,0x01ff01010101ff01,0x01ff0101010101ff,0x01ff010101010101,0x0100ffffffff0000,0x0100ffffff00ff00,0x0100ffffff000001,0x0100ffffff0001ff,0x0100ffffff000100,0x0100ffffff010000,0x0100ffff00ffff00,0x0100ffff00ff0001,0x0100ffff00ff0100,0x0100ffff00000000,0x0100ffff000001ff,0x0100ffff00000101,0x0100ffff00010100,0x0100ffff00010101,0x0100ffff01ff0000,0x0100ffff0100ff00,0x0100ffff010000ff,0x0100ffff01000001,0x0100ffff01000100,0x0100ffff01010000,0x0100ff00ffffff00,0x0100ff00ffff00ff,0x0100ff00ffff0001,0x0100ff00ffff0100,0x0100ff00ff00ffff,0x0100ff00ff000000,0x0100ff00ff0001ff,0x0100ff00ff000101,0x0100ff00ff01ff00,0x0100ff00ff0100ff,0x0100ff00ff010001,0x0100ff00ff010100,0x0100ff0000ffffff,0x0100ff0000ff0000,0x0100ff000000ffff,0x0100ff000000ff00,0x0100ff00000000ff,0x0100ff0000000000,0x0100ff0000000001,0x0100ff0000000100,0x0100ff000001ff01,0x0100ff0000010000,0x0100ff0001ff00ff,0x0100ff0001ff0001,0x0100ff000100ff01,0x0100ff0001000000,0x0100ff00010001ff,0x0100ff000101ff00,0x0100ff00010100ff,0x0100ff0001010001,0x0100ff0001010100,0x0100ff01ffff0000,0x0100ff01ff00ff00,0x0100ff01ff0000ff,0x0100ff01ff000100,0x0100ff01ff010000,0x0100ff0100ff00ff,0x0100ff0100ff0001,0x0100ff0100ff0100,0x0100ff010000ffff,0x0100ff010000ff01,0x0100ff0100000000,0x0100ff01000001ff,0x0100ff0100010001,0x0100ff0100010100,0x0100ff0101ff0000,0x0100ff01010000ff,0x0100ff0101000001,0x0100ff0101010100,0x010000ffffffff00,0x010000ffffff00ff,0x010000ffffff0001,0x010000ffff00ffff,0x010000ffff000000,0x010000ffff0001ff,0x010000ffff010001,0x010000ff00ffffff,0x010000ff00ff0101,0x010000ff0000ff00,0x010000ff000000ff,0x010000ff00000000,0x010000ff00000001,0x010000ff000001ff,0x010000ff00000100,0x010000ff0001ffff,0x010000ff0001ff00,0x010000ff0001ff01,0x010000ff00010000,0x010000ff01ff00ff,0x010000ff01ff0001,0x010000ff0100ff01,0x010000ff010000ff,0x010000ff01000000,0x010000ff010001ff,0x010000ff0101ff00,0x010000ff01010100,0x01000000ffffffff,0x01000000ffff0000,0x01000000ffff01ff,0x01000000ffff0101,0x01000000ff00ffff,0x01000000ff00ff00,0x01000000ff0000ff,0x01000000ff000000,0x01000000ff000001,0x01000000ff000100,0x01000000ff01ff00,0x01000000ff010000,0x01000000ff010100,0x01000000ff010101,0x0100000000ffff00,0x0100000000ff00ff,0x0100000000ff0000,0x0100000000ff0001,0x0100000000ff0100,0x010000000000ffff,0x010000000000ff00,0x010000000000ff01,0x01000000000000ff,0x0100000000000000,0x0100000000000001,0x01000000000001ff,0x0100000000000100,0x0100000000000101,0x010000000001ff00,0x01000000000100ff,0x0100000000010000,0x0100000000010001,0x0100000000010100,0x0100000001ffff00,0x0100000001ff0000,0x0100000001ff01ff,0x010000000100ff00,0x010000000100ff01,0x01000000010000ff,0x0100000001000000,0x0100000001000001,0x0100000001000100,0x0100000001000101,0x010000000101ffff,0x010000000101ff01,0x0100000001010000,0x01000000010101ff,0x0100000001010101,0x01000001ffffff00,0x01000001ffff00ff,0x01000001ff00ffff,0x01000001ff000000,0x01000001ff000100,0x01000001ff01ffff,0x01000001ff010001,0x01000001ff010100,0x0100000100ff0000,0x0100000100ff01ff,0x0100000100ff0100,0x010000010000ff00,0x010000010000ff01,0x0100000100000000,0x0100000100000001,0x0100000100000100,0x0100000100010000,0x01000001000101ff,0x0100000101ffff01,0x0100000101ff00ff,0x0100000101ff0100,0x0100000101ff0101,0x010000010100ff01,0x01000001010000ff,0x0100000101000000,0x01000001010100ff,0x0100000101010001,0x0100000101010100,0x010001ffffff0000,0x010001ffff000001,0x010001ffff000100,0x010001ffff010000,0x010001ff00ffff00,0x010001ff00ff0001,0x010001ff0000ffff,0x010001ff0000ff01,0x010001ff00000000,0x010001ff00000001,0x010001ff00000101,0x010001ff000100ff,0x010001ff00010000,0x010001ff01ff0000,0x010001ff0100ff00,0x010001ff01000001,0x010001ff01000100,0x010001ff01010000,0x01000100ffff00ff,0x01000100ffff0001,0x01000100ffff0100,0x01000100ff00ffff,0x01000100ff00ff01,0x01000100ff000000,0x01000100ff0001ff,0x01000100ff000101,0x01000100ff01ffff,0x01000100ff01ff00,0x01000100ff0100ff,0x01000100ff010001,0x0100010000ffffff,0x0100010000ffff01,0x0100010000ff0000,0x0100010000ff01ff,0x0100010000ff0101,0x010001000000ff00,0x01000100000000ff,0x0100010000000000,0x0100010000000001,0x0100010000000100,0x010001000001ff01,0x0100010000010000,0x0100010000010001,0x0100010000010101,0x0100010001ffff00,0x0100010001ff00ff,0x010001000100ffff,0x010001000100ff01,0x0100010001000000,0x0100010001000101,0x010001000101ff00,0x0100010001010001,0x01000101ffff0000,0x01000101ff000000,0x01000101ff010000,0x0100010100ff00ff,0x0100010100ff0001,0x0100010100ff0100,0x010001010000ffff,0x0100010100000000,0x01000101000001ff,0x010001010001ff00,0x0100010101ff0000,0x010001010100ff00,0x01000101010000ff,0x0100010101000000,0x0100010101000001,0x0101ffffffffffff,0x0101ffffffffff01,0x0101ffffffff01ff,0x0101ffffffff0101,0x0101ffffff000000,0x0101ffffff01ffff,0x0101ffffff01ff01,0x0101ffffff0101ff,0x0101ffffff010101,0x0101ffff00ff0000,0x0101ffff0000ff00,0x0101ffff000000ff,0x0101ffff00000001,0x0101ffff00000100,0x0101ffff01ffffff,0x0101ffff01ffff01,0x0101ffff01ff01ff,0x0101ffff01ff0101,0x0101ffff01000000,0x0101ffff0101ffff,0x0101ffff0101ff01,0x0101ffff010101ff,0x0101ffff01010101,0x0101ff00ffff0000,0x0101ff00ffff0100,0x0101ff00ff00ff00,0x0101ff00ff0000ff,0x0101ff00ff000001,0x0101ff00ff000100,0x0101ff00ff000101,0x0101ff0000ff0001,0x0101ff0000ff0100,0x0101ff000000ff00,0x0101ff0000000000,0x0101ff00000001ff,0x0101ff0000000101,0x0101ff000001ff00,0x0101ff00000100ff,0x0101ff0001ff0000,0x0101ff000100ffff,0x0101ff000100ff01,0x0101ff0001000001,0x0101ff0001000100,0x0101ff01ffffff01,0x0101ff01ffff01ff,0x0101ff01ffff0101,0x0101ff01ff00ffff,0x0101ff01ff000100,0x0101ff01ff01ff01,0x0101ff01ff0101ff,0x0101ff01ff010101,0x0101ff0100ff0000,0x0101ff010000ff00,0x0101ff0100000001,0x0101ff0100000100,0x0101ff0100010000,0x0101ff0101ffffff,0x0101ff0101ffff01,0x0101ff0101ff01ff,0x0101ff0101ff0101,0x0101ff0101000000,0x0101ff010101ffff,0x0101ff010101ff01,0x0101ff01010101ff,0x0101ff0101010101,0x010100ffff000100,0x010100ffff010000,0x010100ff00ffff00,0x010100ff00ff00ff,0x010100ff0000ffff,0x010100ff000000ff,0x010100ff00000000,0x010100ff000001ff,0x010100ff00000101,0x010100ff0001ff00,0x010100ff00010000,0x010100ff00010001,0x010100ff000101ff,0x010100ff00010100,0x010100ff01ff0000,0x01010000ffff0001,0x01010000ffff0100,0x01010000ff00ffff,0x01010000ff00ff01,0x01010000ff000000,0x01010000ff0001ff,0x01010000ff010001,0x01010000ff010100,0x0101000000ffff01,0x0101000000ff0000,0x010100000000ff00,0x01010000000000ff,0x0101000000000000,0x0101000000000001,0x0101000000000100,0x0101000000010000,0x0101000000010101,0x0101000001ffff00,0x0101000001ff00ff,0x0101000001ff0000,0x0101000001ff0001,0x0101000001ff0100,0x010100000100ff01,0x0101000001000000,0x01010000010001ff,0x01010001ffff0000,0x01010001ff00ff00,0x01010001ff000001,0x01010001ff000101,0x01010001ff01ff00,0x01010001ff010000,0x0101000100ff00ff,0x0101000100ff0001,0x0101000100ff0101,0x010100010000ff01,0x0101000100000000,0x0101000100000001,0x01010001000001ff,0x010100010001ffff,0x010100010001ff01,0x0101000101ff0001,0x010100010100ffff,0x0101000101000000,0x0101000101000001,0x0101000101000100,0x010100010101ff00,0x01010001010100ff,0x0101000101010001,0x010101ffffffffff,0x010101ffffffff01,0x010101ffffff01ff,0x010101ffffff0101,0x010101ffff01ffff,0x010101ffff01ff01,0x010101ffff0101ff,0x010101ffff010101,0x010101ff0000ff00,0x010101ff000000ff,0x010101ff00000001,0x010101ff00000100,0x010101ff01ffffff,0x010101ff01ffff01,0x010101ff01ff01ff,0x010101ff01ff0101,0x010101ff01000000,0x010101ff0101ffff,0x010101ff0101ff01,0x010101ff010101ff,0x010101ff01010101,0x01010100ffff0000,0x01010100ff0000ff,0x01010100ff000100,0x01010100ff01ff00,0x01010100ff010000,0x0101010000ffff00,0x010101000000ffff,0x0101010000000000,0x0101010000000101,0x010101000001ff00,0x0101010000010001,0x0101010000010100,0x010101000100ffff,0x0101010001000001,0x01010101ffffffff,0x01010101ffffff01,0x01010101ffff01ff,0x01010101ffff0101,0x01010101ff01ffff,0x01010101ff01ff01,0x01010101ff0101ff,0x01010101ff010101,0x010101010000ff00,0x01010101000000ff,0x0101010100000001,0x0101010101ffffff,0x0101010101ffff01,0x0101010101ff01ff,0x0101010101ff0101,0x0101010101000000,0x010101010101ffff,0x010101010101ff01,0x01010101010101ff,0x0101010101010101
    };

    // ===================== IQ1_S / IQ1_M =====================
    // Reference: ggml-quants.c:2650 (dequantize_row_iq1_s) and :2675
    // (dequantize_row_iq1_m). Block geometry per ggml-common.h:429/:437.
    // IQ1S block (50 B / 256 w): fp16 d | qs[32] (4 B per 32-w sub-block)
    //   | qh u16[8] (one per sub-block).
    // IQ1M block (56 B / 256 w): qs[32] | qh[16] (2 B per sub-block)
    //   | scales[8] -> 4 u16 LE; fp16 scale bit-packed across them.
    // Each 32-w sub-block: dl = d*(2*scale_code+1); 4 lattice rows of 8;
    // w = dl * (grid[j] + delta), delta = +/-0.125 from a qh sign bit.

    __device__ inline float dequant_iq1s(const unsigned char* blk, int in_sb) {
        float d = fp16_to_float_device(((const unsigned short*)blk)[0]);
        int ib = in_sb / 32;
        int l = (in_sb % 32) / 8;
        int j = in_sb % 8;
        unsigned short qh = ((const unsigned short*)(blk + 34))[ib];
        float dl = d * (2.0f * (float)((qh >> 12) & 7) + 1.0f);
        float delta = (qh & 0x8000) ? -0.125f : 0.125f;
        int idx = blk[2 + ib * 4 + l] | ((((int)(qh >> (3 * l)) & 7)) << 8);
        unsigned long long g = IQ1S_GRID[idx];
        signed char grid_j = (signed char)((g >> (8 * j)) & 0xff);
        return dl * ((float)grid_j + delta);
    }

    __device__ inline float dequant_iq1m(const unsigned char* blk, int in_sb) {
        int ib = in_sb / 32;
        int l = (in_sb % 32) / 8;
        int j = in_sb % 8;
        const unsigned char* qs = blk;
        const unsigned char* qh = blk + 32;
        const unsigned char* sc = blk + 48;
        unsigned short sc0 = ((const unsigned short*)sc)[0];
        unsigned short sc1 = ((const unsigned short*)sc)[1];
        unsigned short sc2 = ((const unsigned short*)sc)[2];
        unsigned short sc3 = ((const unsigned short*)sc)[3];
        unsigned short scale_bits = (unsigned short)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0)
            | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
        float d = fp16_to_float_device(scale_bits);
        int half = ib % 2;
        unsigned short scv = ((const unsigned short*)sc)[ib / 2];
        float dl1 = d * (2.0f * (float)((scv >> (6 * half)) & 7) + 1.0f);
        float dl2 = d * (2.0f * (float)((scv >> (6 * half + 3)) & 7) + 1.0f);
        unsigned char qh0 = qh[ib * 2];
        unsigned char qh1 = qh[ib * 2 + 1];
        int idx0 = qs[ib * 4 + 0] | (((int)qh0 << 8) & 0x700);
        int idx1 = qs[ib * 4 + 1] | (((int)qh0 << 4) & 0x700);
        int idx2 = qs[ib * 4 + 2] | (((int)qh1 << 8) & 0x700);
        int idx3 = qs[ib * 4 + 3] | (((int)qh1 << 4) & 0x700);
        float delta[4] = {
            (qh0 & 0x08) ? -0.125f : 0.125f,
            (qh0 & 0x80) ? -0.125f : 0.125f,
            (qh1 & 0x08) ? -0.125f : 0.125f,
            (qh1 & 0x80) ? -0.125f : 0.125f,
        };
        int idxs[4] = { idx0, idx1, idx2, idx3 };
        int l_idx = idxs[l];
        float dl = (l < 2) ? dl1 : dl2;
        unsigned long long g = IQ1S_GRID[l_idx];
        signed char grid_j = (signed char)((g >> (8 * j)) & 0xff);
        return dl * ((float)grid_j + delta[l]);
    }

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

    __global__ void grim_fused_dequant_gemm_iq1s(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_iq1s,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 50;
        const unsigned char* row_b_ptr = B_iq1s + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            float w_val = dequant_iq1s(row_b_ptr + sb_idx * 50, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_iq1s(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_iq1s,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 50;
        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq1s(B_iq1s + n * row_bytes + sb_idx * 50, in_sb);
            acc += dy_val * w_val;
        }
        dX[row * K + k_idx] = acc;
    }

    __global__ void grim_fused_dequant_gemm_iq1m(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_iq1m,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * N;
        if (idx >= total) return;
        const int row = (int)(idx / N);
        const int col = (int)(idx % N);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 56;
        const unsigned char* row_b_ptr = B_iq1m + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 256;
            int in_sb = k % 256;
            float w_val = dequant_iq1m(row_b_ptr + sb_idx * 56, in_sb);
            acc += a_val * w_val;
        }
        C[row * N + col] = acc;
    }

    __global__ void grim_fused_dequant_backward_gemm_iq1m(
        const float* __restrict__ dY,
        const unsigned char* __restrict__ B_iq1m,
        float* __restrict__ dX,
        int M, int N, int K)
    {
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long total = (unsigned long long)M * K;
        if (idx >= total) return;
        const int row = (int)(idx / K);
        const int k_idx = (int)(idx % K);
        const int blocks_per_row = K / 256;
        const int row_bytes = blocks_per_row * 56;
        int sb_idx = k_idx / 256;
        int in_sb = k_idx % 256;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq1m(B_iq1m + n * row_bytes + sb_idx * 56, in_sb);
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
        // IQ4_NL blocks hold QK4_NL = 32 weights in 18 bytes.
        const int blocks_per_row = K / 32;
        const int row_bytes = blocks_per_row * 18;
        const unsigned char* row_b_ptr = B_iq4nl + col * row_bytes;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[row * K + k];
            int sb_idx = k / 32;
            int in_sb = k % 32;
            float w_val = dequant_iq4nl(row_b_ptr + sb_idx * 18, in_sb);
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
        // IQ4_NL blocks hold QK4_NL = 32 weights in 18 bytes.
        const int blocks_per_row = K / 32;
        const int row_bytes = blocks_per_row * 18;
        int sb_idx = k_idx / 32;
        int in_sb = k_idx % 32;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float dy_val = dY[row * N + n];
            float w_val = dequant_iq4nl(B_iq4nl + n * row_bytes + sb_idx * 18, in_sb);
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

        // P4: hoist every loop-invariant decode index out of the per-MAC N loop.
        // dX[row][k] = sum_n dY[row][n] * B[n][k] walks one packed superblock per output row n (B.
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

    // --- Q8_0 fused GEMM, NUM_ROWS=4 (row-count-aware) ---
    // Each thread computes 4 consecutive output columns for the same activation
    // row, sharing the A[row, k] read across 4 weight dequants.  When the
    // activation vector is large (e.g. down projection: ~68 KB at K=intermediate),
    // this halves L2 traffic vs. the 1-col-per-thread variant.  Grid covers
    // M * (N/4) thread-slots; N must be a multiple of 4 (caller guards this).
    __global__ void grim_fused_dequant_gemm_q8_0_rows4(
        const float* __restrict__ A,
        const unsigned char* __restrict__ B_q80,
        float* __restrict__ C,
        int M, int N, int K)
    {
        const int cols_per_thread = 4;
        const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long slots = (unsigned long long)M * (N / cols_per_thread);
        if (idx >= slots) return;
        const int row = (int)(idx / (N / cols_per_thread));
        const int col_base = (int)(idx % (N / cols_per_thread)) * cols_per_thread;
        const int blocks_per_row = K / 32;
        const int row_bytes = blocks_per_row * 34;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        const unsigned char* b0 = B_q80 + col_base * row_bytes;
        const unsigned char* b1 = B_q80 + (col_base + 1) * row_bytes;
        const unsigned char* b2 = B_q80 + (col_base + 2) * row_bytes;
        const unsigned char* b3 = B_q80 + (col_base + 3) * row_bytes;
        for (int k = 0; k < K; ++k) {
            const float a_val = A[row * K + k];
            const int sb_idx = k / 32;
            const int in_sb = k % 32;
            acc0 += a_val * dequant_q80_standalone(b0 + sb_idx * 34, in_sb);
            acc1 += a_val * dequant_q80_standalone(b1 + sb_idx * 34, in_sb);
            acc2 += a_val * dequant_q80_standalone(b2 + sb_idx * 34, in_sb);
            acc3 += a_val * dequant_q80_standalone(b3 + sb_idx * 34, in_sb);
        }
        C[row * N + col_base]     = acc0;
        C[row * N + col_base + 1] = acc1;
        C[row * N + col_base + 2] = acc2;
        C[row * N + col_base + 3] = acc3;
    }



}

// --- Native K-quant grouped dispatch (xing4.0: IQ3_S gate/up + Q4_K down) ---
// One block per (token, expert) pair; 256 threads. Reads the ALREADY-RESIDENT
// packed per-expert banks through device pointer arrays (the banks ship as
// part of the model, so this arm costs ZERO extra weight VRAM - unlike the
// WhiteCrow stacks, it needs no budget). Per-expert blobs are contiguous
// [row][super-block] packed rows: gate/up IQ3_S (110 B per 256 weights,
// hidden/256 blocks per row), down Q4_K (144 B per 256, inter/256 blocks per
// row - the runtime storages were measured, not assumed: gate/up 1576960 B,
// down 2064384 B per expert). The decoders are the SAME leaf functions the
// rest of the tree trusts (dequant_iq3s + IQ3S_GRID_B above in this TU;
// dequant_q4k_element from shared_device_fns). Routing comes from device
// buffers: decode-graph capture-safe.
// ===================== PREFILL dequant-once arm =====================
// Dequantizes ONE expert's three banks to f32 (gate/up [inter, hidden],
// down [hidden, inter]) so a prefill window can run per-expert rocBLAS
// SGEMMs instead of the per-pair in-register decode. Launched once per
// expert with a 44 MB rolling scratch — see `moe_dequant_gemm_prefill_into`.
// blockIdx.y: 0 gate, 1 up, 2 down.
extern "C" __global__ void grim_dequant_expert_banks_f32(
    const unsigned long long* __restrict__ gate_ptrs,
    const unsigned long long* __restrict__ up_ptrs,
    const unsigned long long* __restrict__ down_ptrs,
    float* __restrict__ out_gate,   // [inter, hidden]
    float* __restrict__ out_up,     // [inter, hidden]
    float* __restrict__ out_down,   // [hidden, inter]
    int inter, int hidden, int expert,
    int gate_row_bytes, int down_row_bytes, int down_fmt)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bank = blockIdx.y;
    // NATURAL layout, no transpose: matmul_op_into computes A x B^T with B
    // stored [out, in] — which is exactly how the banks already lie
    // (gate/up [inter, hidden], down [hidden, inter]).
    if (bank == 0 || bank == 1) {
        if (row >= inter) return;
        const unsigned char* src = (bank == 0
            ? (const unsigned char*)(size_t)gate_ptrs[expert]
            : (const unsigned char*)(size_t)up_ptrs[expert])
            + (unsigned long long)row * gate_row_bytes;
        float* dst = (bank == 0 ? out_gate : out_up) + (unsigned long long)row * hidden;
        for (int i = tid * 4; i < hidden; i += blockDim.x * 4) {
            float w4[4];
            // The x4 helper indexes WITHIN a 256-weight super-block: advance
            // the base by the block stride. Passing the whole-row index read
            // block 0's bytes for every element past 255 — heads matched, the
            // rest was wrong, and a same-scratch "manual" check certified it.
            dequant_iq3s_x4(src + (size_t)(i / 256) * 110, i % 256, w4);
            dst[i] = w4[0]; dst[i+1] = w4[1]; dst[i+2] = w4[2]; dst[i+3] = w4[3];
        }
    } else {
        if (row >= hidden) return;
        const unsigned char* src = (const unsigned char*)(size_t)down_ptrs[expert]
            + (unsigned long long)row * down_row_bytes;
        float* dst = out_down + (unsigned long long)row * inter;
        for (int i = tid * 4; i < inter; i += blockDim.x * 4) {
            if (down_fmt) {
                // The loop strides by 4 — write all four lanes. Writing only
                // dst[i] left 3/4 of every Q4_K down row unwritten.
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int ii = i + k;
                    dst[ii] = dequant_q4k_element(src + (size_t)(ii / 256) * 144, ii % 256);
                }
            } else {
                float w4[4];
                dequant_iq3s_x4(src + (size_t)(i / 256) * 110, i % 256, w4);
                dst[i] = w4[0]; dst[i+1] = w4[1]; dst[i+2] = w4[2]; dst[i+3] = w4[3];
            }
        }
    }
}

// out[r] = x[toks[r]] for r in 0..count (pair-compact gather).
extern "C" __global__ void grim_moe_gather_rows(
    const float* __restrict__ x, const int* __restrict__ toks,
    float* __restrict__ out, int hidden)
{
    const int r = blockIdx.x;
    const int tok = toks[r];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        out[(long long)r * hidden + i] = x[(long long)tok * hidden + i];
}

// Fixed-order reduction of per-pair rows into a single-token output
// (decode: batch 1). Deterministic — pairs summed in index order.
extern "C" __global__ void grim_moe_pairs_reduce(
    const float* __restrict__ pair_out, float* __restrict__ out,
    int hidden, int num_pairs)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    float acc = 0.0f;
    for (int p = 0; p < num_pairs; ++p)
        acc += pair_out[(unsigned long long)p * hidden + h];
    out[h] = acc;
}

// out[toks[r]] += wts[r] * in[r] (pair-compact scaled scatter-add).
extern "C" __global__ void grim_moe_scatter_add_rows(
    float* __restrict__ out, const int* __restrict__ toks,
    const float* __restrict__ wts, const float* __restrict__ in, int hidden)
{
    const int r = blockIdx.x;
    const int tok = toks[r];
    const float w = wts[r];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        atomicAdd(out + (long long)tok * hidden + i, w * in[(long long)r * hidden + i]);
}

// DETERMINISTIC variant: write w*in[r] to out[idx[r]] (plain store, no
// atomics). idx[r] is the pair's position in TOKEN-sorted order, so the
// subsequent per-token reduction sums in a fixed order.
extern "C" __global__ void grim_moe_scatter_rows_idx(
    float* __restrict__ out, const int* __restrict__ idx,
    const float* __restrict__ wts, const float* __restrict__ in, int hidden)
{
    const int r = blockIdx.x;
    const int dst = idx[r];
    const float w = wts[r];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        out[(long long)dst * hidden + i] = w * in[(long long)r * hidden + i];
}

// Per-token fixed-order reduction over pair rows: out[t] = sum of
// full[off[t]..off[t+1]) in index order. Grid = num_tokens; offsets are the
// CSR over the token-sorted pairs. Plain adds in a fixed order — no atomics.
extern "C" __global__ void grim_moe_token_reduce(
    const float* __restrict__ full, const int* __restrict__ offsets,
    float* __restrict__ out, int hidden)
{
    const int t = blockIdx.x;
    const int lo = offsets[t];
    const int hi = offsets[t + 1];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = lo; j < hi; ++j)
            acc += full[(long long)j * hidden + i];
        out[(long long)t * hidden + i] = acc;
    }
}

extern "C" __global__ void grim_moe_fused_dispatch_kq_native(
    const float* __restrict__ activations,            // [batch, hidden]
    const unsigned long long* __restrict__ gate_ptrs, // [num_experts]
    const unsigned long long* __restrict__ up_ptrs,   // [num_experts]
    const unsigned long long* __restrict__ down_ptrs, // [num_experts]
    const unsigned int* __restrict__ router_tokens,   // [num_pairs]
    const unsigned int* __restrict__ router_experts,  // [num_pairs]
    const float* __restrict__ router_weights,         // [num_pairs]
    float* __restrict__ out,                          // [batch, hidden]
    int hidden, int inter, int num_pairs,
    float routed_scaling_factor, int num_experts, int batch, int phase_mask,
    int gate_row_bytes, int down_row_bytes, int jlimit_arg, int down_fmt,
    float* __restrict__ pair_out, int deterministic)
{
    const int pair = blockIdx.x;
    if (pair >= num_pairs) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int tok = (int)router_tokens[pair];
    const int exp = (int)router_experts[pair];
    // A routing id outside the bank set (or a token outside the batch) would
    // read the pointer array / activations out of bounds and then dereference
    // garbage - return instead of faulting.
    if (exp < 0 || exp >= num_experts) return;
    if (tok < 0 || tok >= batch) return;
    const float w = router_weights[pair];

    extern __shared__ float s_act[]; // [inter]

    const unsigned char* gb = (const unsigned char*)gate_ptrs[exp];
    const unsigned char* ub = (const unsigned char*)up_ptrs[exp];
    const unsigned char* db = (const unsigned char*)down_ptrs[exp];
    // Row strides are HOST-MEASURED from the actual per-expert storages
    // (storage.bytes() / rows) — the loader may re-tag/re-encode the down
    // bank, so the kernel must not assume the raw-file geometry.
    const int g_sb = gate_row_bytes / 110;  // IQ3_S super-blocks per gate/up row
    // down bank format per layer: 1 = Q4_K (144 B/256), 0 = IQ3_S (110 B/256).
    // The checkpoint's down banks are NOT uniform across layers (blk.2 is
    // Q4_K, others IQ3_S), so the leg switches on the measured scheme.
    const int down_block_bytes = down_fmt ? 144 : 110;
    const int d_sb = down_row_bytes / down_block_bytes;  // super-blocks per down row

    const float* a = activations + (unsigned long long)tok * hidden;

    // Phase 1: fused gate|up with in-register SiLU combine -> shared act.
    // phase_mask bit 0 gates this phase (fault bisection).
    if (phase_mask & 1) {
    for (int j = tid; j < jlimit_arg; j += nthreads) {
        const unsigned char* gblk = gb + (unsigned long long)j * gate_row_bytes;
        const unsigned char* ublk = ub + (unsigned long long)j * gate_row_bytes;
        float g = 0.0f;
        float u = 0.0f;
        for (int b = 0; b < g_sb; ++b) {
            const int kb = b * 256;
            #pragma unroll 2
            for (int i = 0; i < 256; i += 4) {
                float g4[4];
                float u4[4];
                dequant_iq3s_x4(gblk + b * 110, i, g4);
                dequant_iq3s_x4(ublk + b * 110, i, u4);
                g += a[kb + i + 0] * g4[0] + a[kb + i + 1] * g4[1]
                   + a[kb + i + 2] * g4[2] + a[kb + i + 3] * g4[3];
                u += a[kb + i + 0] * u4[0] + a[kb + i + 1] * u4[1]
                   + a[kb + i + 2] * u4[2] + a[kb + i + 3] * u4[3];
            }
        }
        s_act[j] = (g / (1.0f + expf(-g))) * u;
    }
    __syncthreads();
    }

    // Phase 2: down projection, atomicAdd accumulation with the routing
    // weight. phase_mask bit 1 gates this phase (fault bisection).
    if (phase_mask & 2) {
    for (int h = tid; h < hidden; h += nthreads) {
        const unsigned char* dblk = db + (unsigned long long)h * down_row_bytes;
        float acc = 0.0f;
        for (int b = 0; b < d_sb; ++b) {
            const int kb = b * 256;
            // Down decode with the SAME leaf functions the weight path and the
            // embedding gather use, selected by the layer's measured format.
            if (down_fmt) {
                #pragma unroll 8
                for (int i = 0; i < 256; ++i) {
                    acc += s_act[kb + i] * dequant_q4k_element(dblk + b * 144, i);
                }
            } else {
                for (int i = 0; i < 256; i += 4) {
                    float w4[4];
                    dequant_iq3s_x4(dblk + b * 110, i, w4);
                    acc += s_act[kb + i + 0] * w4[0] + s_act[kb + i + 1] * w4[1]
                         + s_act[kb + i + 2] * w4[2] + s_act[kb + i + 3] * w4[3];
                }
            }
        }
        if (deterministic) {
            // Per-pair rows + a fixed-order reduce: atomicAdd float ordering
            // made decode tails vary run to run.
            pair_out[(unsigned long long)pair * hidden + h] =
                routed_scaling_factor * w * acc;
        } else {
            atomicAdd(out + (unsigned long long)tok * hidden + h,
                      routed_scaling_factor * w * acc);
        }
    }
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
        check_kernel!("grim_fused_dequant_gemm_q8_0_rows4");
        check_kernel!("grim_fused_dequant_backward_gemm_q8_0");
    }
}
