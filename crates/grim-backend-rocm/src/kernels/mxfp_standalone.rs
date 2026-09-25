//! Standalone MXFP4 / MXFP8 dequantization HIP kernels for ROCm.

/// HIP source for `grim_dequant_mxfp4` and `grim_dequant_mxfp8`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// Dequantize MXFP4 codes + shared exponents to F32.
    /// Layout: codes[] (2 4-bit values per byte) + exps[] (1 FP8 per 32 elements).
    __global__ void grim_dequant_mxfp4(
        const unsigned char* __restrict__ codes,
        const unsigned char* __restrict__ exps,
        float* __restrict__ out,
        int n_weights)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_weights) return;

        int group_idx = idx / 32;
        unsigned char shared_exp = exps[group_idx];
        int code_byte_idx = idx / 2;
        unsigned char packed_byte = codes[code_byte_idx];
        unsigned char code = (idx % 2 == 0) ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);

        out[idx] = mxfp4_to_float_hip(code, shared_exp);
    }

    /// Dequantize MXFP8 codes + shared exponents to F32.
    /// Layout: codes[] (1 FP8 per element) + exps[] (1 FP8 per 32 elements).
    __global__ void grim_dequant_mxfp8(
        const unsigned char* __restrict__ codes,
        const unsigned char* __restrict__ exps,
        float* __restrict__ out,
        int n_weights)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_weights) return;

        int group_idx = idx / 32;
        unsigned char shared_exp = exps[group_idx];
        unsigned char fp8_code = codes[idx];
        float fp8_val = fp8_e4m3_to_float_hip(fp8_code);
        float exp_scale = exp2f((float)shared_exp - 127.0f);

        out[idx] = fp8_val * exp_scale;
    }

    /// Dequantize Nutcracker codes + per-16 block scale to F32.
    /// Interleaved layout: 9 bytes per 16 weights (1 `[exp:6|sel:2]` scale byte
    /// + 8 packed E2M1 code bytes). The zero code emits the block's special
    /// value; see `nutcracker_to_float_hip` in shared_device_fns.
    __global__ void grim_dequant_nutcracker(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_weights)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_weights) return;

        int sub_block_idx = idx / 16;
        int in_sub_block = idx % 16;
        int blk_offset = sub_block_idx * 9;
        unsigned char scale_byte = packed[blk_offset];
        unsigned char code_byte = packed[blk_offset + 1 + (in_sub_block / 2)];
        unsigned char code = (in_sub_block % 2 == 0) ? (code_byte & 0x0F) : ((code_byte >> 4) & 0x0F);

        out[idx] = nutcracker_to_float_hip(code, scale_byte);
    }

    /// Dequantize NVFP4 codes + per-16 E4M3 block scale to F32.
    /// Interleaved layout: 9 bytes per 16 weights (1 E4M3 scale byte + 8
    /// packed E2M1 code bytes). Byte-identical to Nutcracker; the zero code
    /// decodes to a real zero.
    __global__ void grim_dequant_nvfp4(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_weights)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_weights) return;

        int sub_block_idx = idx / 16;
        int in_sub_block = idx % 16;
        int blk_offset = sub_block_idx * 9;
        unsigned char scale_byte = packed[blk_offset];
        unsigned char code_byte = packed[blk_offset + 1 + (in_sub_block / 2)];
        unsigned char code = (in_sub_block % 2 == 0) ? (code_byte & 0x0F) : ((code_byte >> 4) & 0x0F);

        out[idx] = mxfp4_to_float_hip(code, 127) * fp8_e4m3_to_float_hip(scale_byte);
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp_standalone_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("grim_dequant_mxfp4"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_mxfp8"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_nutcracker"));
        assert!(KERNEL_SOURCE.contains("grim_dequant_nvfp4"));
        assert!(KERNEL_SOURCE.contains("mxfp4_to_float_hip"));
    }

    /// The two 9-byte schemes must not share a decode. This guards the exact
    /// bug where `grim_dequant_nutcracker` was still calling the E8M0 helper
    /// after the split: the rename alone did not catch it.
    #[test]
    fn the_two_9byte_schemes_use_different_scale_decoders() {
        let nut = KERNEL_SOURCE
            .split("void grim_dequant_nutcracker")
            .nth(1)
            .expect("nutcracker kernel");
        let nut_body = nut.split("}").next().unwrap();
        assert!(
            nut_body.contains("nutcracker_to_float_hip"),
            "Nutcracker must use the selector-aware decoder, not the E8M0 one"
        );
        assert!(
            !nut_body.contains("fp8_e4m3_to_float_hip"),
            "Nutcracker must not use the NVFP4 E4M3 decoder"
        );

        let nv = KERNEL_SOURCE
            .split("void grim_dequant_nvfp4")
            .nth(1)
            .expect("nvfp4 kernel");
        let nv_body = nv.split("}").next().unwrap();
        assert!(
            nv_body.contains("fp8_e4m3_to_float_hip"),
            "NVFP4 must use the E4M3 decoder"
        );
        assert!(
            !nv_body.contains("nutcracker_to_float_hip"),
            "NVFP4 must not use the Nutcracker selector decoder"
        );
    }
}
