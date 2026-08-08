//! Shared HIP device helper functions for all ROCm kernel translation units. [see: `KERNEL_SOURCE`, `__device__`]

/// Shared HIP device helper source code prepended to all compute kernel assemblies.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    __device__ inline float fp16_to_float_device(unsigned short h) {
        unsigned int sign = (h >> 15) & 1;
        unsigned int exp  = (h >> 10) & 0x1f;
        unsigned int mant = h & 0x3ff;
        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            float res = (float)mant / 1024.0f * 0.00006103515625f;
            return sign ? -res : res;
        } else if (exp == 31) {
            return sign ? -1.0f/0.0f : 1.0f/0.0f;
        }
        float res = (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
        return sign ? -res : res;
    }

    __device__ inline float fp8_e4m3_to_float_hip(unsigned char val) {
        int sign = (val >> 7) & 1;
        int exp = (val >> 3) & 0x0F;
        int mant = val & 0x07;
        if (exp == 0xF) {
            if (mant == 7) return 0.0f / 0.0f; // NaN
            float v = 448.0f;
            return sign ? -v : v;
        }
        float res;
        if (exp != 0) {
            res = (1.0f + (float)mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
        } else {
            res = (float)mant / 512.0f;
        }
        return sign ? -res : res;
    }

    __device__ inline float mxfp4_to_float_hip(unsigned char code, unsigned char shared_exp) {
        int sign = (code >> 3) & 1;
        int exp = (code >> 1) & 3;
        int mant = code & 1;
        float base_val = 0.0f;
        if (exp == 0) {
            base_val = (float)mant * 0.5f;
        } else {
            base_val = (1.0f + (float)mant * 0.5f) * powf(2.0f, (float)exp - 1.0f);
        }
        if (sign) base_val = -base_val;
        float scale = powf(2.0f, (float)shared_exp - 127.0f);
        return base_val * scale;
    }

    __device__ inline float dequant_q4k_element(const unsigned char* block_ptr, int in_sb) {
        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
        float d = fp16_to_float_device(h_ptr[0]);
        float dmin = fp16_to_float_device(h_ptr[1]);

        const unsigned char* scales = block_ptr + 4;
        const unsigned char* qs = block_ptr + 16;

        // ggml `dequantize_row_q4_K`: four 64-weight groups. Within group g,
        // the first 32 outputs take low nibbles (q[l] & 0xF) and the next 32
        // take high nibbles (q[l] >> 4) of the *same* 32-byte `qs` window
        // (q advances by 32 bytes per 64-output group). The low group uses
        // scale sub-block `is = 2*g`, the high group uses `2*g + 1`.
        //
        // The previous per-element formula here read `qs[is*16 + il]`, which
        // for the high nibble (is odd) shifted up by 16 bytes into the next
        // group's window and so crossed group boundaries — wrong bytes,
        // wrong results.
        int group = in_sb / 64;           // 0..3
        int half  = (in_sb % 64) / 32;     // 0 = low nibble, 1 = high nibble
        int l     = in_sb % 32;           // 0..31 within the 32-byte window
        int is    = 2 * group + half;     // 0..7 scale sub-block index

        unsigned char sc, m;
        if (is < 4) {
            sc = scales[is] & 63;
            m  = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        unsigned char byte = qs[group * 32 + l];
        unsigned char q = half ? (byte >> 4) : (byte & 0xF);

        return d * (float)sc * (float)q - dmin * (float)m;
    }

}
"#;
