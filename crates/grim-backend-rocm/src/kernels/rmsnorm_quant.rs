//! SPEED-DOT-OPFUSE (space-balls.md op-count reduction): fused RMSNorm →
//! int8 activation quantization for the decode path.
//!
//! The Q8_0 decode Linear currently costs two tiny launches before the GEMV —
//! rmsnorm, then q8_1 activation quantization. This kernel fuses both into
//! ONE launch: normalize (weighted RMSNorm), then per-32-block symmetric
//! int8 quantization, writing the same packed q8_1 block layout that
//! `grim_dot4_q80_q81_gemv` consumes (per block: fp16 `d` LE, fp16 `sum` LE,
//! 32 int8 codes). Decode-shaped: single row, one thread block; replaces
//! norm + quant launches (2→1) for every Linear input on the hot path.
//!
//! Multi-row (prefill) inputs keep the separate-kernel path.

/// Fused RMSNorm + q8_1-quantization kernel source. Rides the aggregate JIT
/// translation unit (reuses nothing external — self-contained).
pub const KERNEL_SOURCE: &str = r#"
// SPEED-DOT-OPFUSE: fused RMSNorm + int8 q8_1 quantization, one launch.
// out_q81: packed per-32-block layout [d fp16 LE][sum fp16 LE][32 int8 codes]
// — bit-identical to grim_quantize_q8_1's output, so grim_dot4_q80_q81_gemv
// consumes it unchanged. Decode-shaped: M=1, grid (1,1,1), block (32,1,1).
extern "C" __global__ void grim_rmsnorm_quant_i8(
    const float* __restrict__ x, const float* __restrict__ weight,
    float eps, int K,
    unsigned char* __restrict__ out_q81,
    float* __restrict__ norm_cache)   // optional normalized fp32 row (nullable)
{
    const int tid = threadIdx.x;   // 0..31
    float ss = 0.0f;
    for (int k = tid; k < K; k += 32)
        ss += x[k] * x[k];
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor(ss, off);
    const float inv_rms = 1.0f / sqrtf(ss / (float)K + eps);   // exact (not rsqrtf) for bit-exact parity with host reference

    const int n_blocks = K / 32;
    for (int blk = tid; blk < n_blocks; blk += 32) {
        const int base = blk * 32;
        float normed[32];
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float v = x[base + j] * inv_rms * weight[base + j];
            normed[j] = v;
            amax = fmaxf(amax, fabsf(v));
        }
        const float d = amax / 127.0f;
        const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
        unsigned char* dst = out_q81 + (long long)blk * 36;
        float fsum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            int q = (int)roundf(normed[j] * inv_d);
            q = max(-127, min(127, q));
            dst[4 + j] = (unsigned char)(signed char)q;
            fsum += (float)q;
        }
        _Float16 hd = (_Float16)d;
        _Float16 hs = (_Float16)(fsum * d);
        unsigned short wd, ws;
        __builtin_memcpy(&wd, &hd, 2);
        __builtin_memcpy(&ws, &hs, 2);
        dst[0] = (unsigned char)(wd & 0xFF);
        dst[1] = (unsigned char)(wd >> 8);
        dst[2] = (unsigned char)(ws & 0xFF);
        dst[3] = (unsigned char)(ws >> 8);
        if (norm_cache != 0) {
            #pragma unroll
            for (int j = 0; j < 32; ++j)
                norm_cache[base + j] = normed[j];
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_fused_kernel() {
        assert!(
            KERNEL_SOURCE.contains("grim_rmsnorm_quant_i8"),
            "missing fused rmsnorm+quant kernel"
        );
    }

    #[test]
    fn output_layout_matches_dot4_gemv_input() {
        // grim_dot4_q80_q81_gemv reads A_q81 rows as 36-byte packed blocks.
        assert!(
            KERNEL_SOURCE.contains("blk * 36"),
            "q8_1 packed row stride must be 36 bytes per block"
        );
        assert!(
            KERNEL_SOURCE.contains("dst[4 + j]"),
            "codes must start at byte 4 (after d + sum fp16 pair)"
        );
    }

    #[test]
    fn writes_scale_and_sum_pair() {
        // fp16 d and fp16 (sum*d) pair at bytes 0..4 — q8_1 layout contract.
        assert!(KERNEL_SOURCE.contains("wd & 0xFF") && KERNEL_SOURCE.contains("ws & 0xFF"));
    }
}
