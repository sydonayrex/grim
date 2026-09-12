//! SPEED-DOT: vector dot-product GEMV kernels for M=1..4 decode on RDNA3/4.
//!
//! Uses __builtin_amdgcn_sudot4 (V_DOT4_I32_IU8, signed x signed) to execute
//! fast vec_dot_q8_0_q8_1 mat-vec multiplication with Q8_1 activation quantization.

/// Full JIT source for Q8_0 x Q8_1 dot-product GEMV kernels.
pub const KERNEL_SOURCE: &str = r#"
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)

__device__ __forceinline__ int grim_sdot4(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

#define GRIM_Q8_1_BLOCK_SIZE 32
#define GRIM_Q8_1_BYTES      36
#define GRIM_Q8_0_BLOCK_SIZE 32
#define GRIM_Q8_0_BYTES      34

extern "C" __global__ void grim_quantize_q8_1(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst,
    int K,
    int n_rows)
{
    const int global_blk = blockIdx.x;
    const int n_q_blocks = K / GRIM_Q8_1_BLOCK_SIZE;
    const int row = global_blk / n_q_blocks;
    const int blk = global_blk % n_q_blocks;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const float* row_src = src + (long long)row * K + blk * GRIM_Q8_1_BLOCK_SIZE;
    unsigned char* row_dst = dst + (long long)row * n_q_blocks * GRIM_Q8_1_BYTES
                                 + blk * GRIM_Q8_1_BYTES;

    float val = (tid < GRIM_Q8_1_BLOCK_SIZE) ? row_src[tid] : 0.0f;

    float amax = __builtin_fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, mask));

    const float d     = amax / 127.0f;
    const float inv_d = (amax > 1e-9f) ? (127.0f / amax) : 0.0f;
    int8_t qi = (int8_t)__builtin_roundf(val * inv_d);

    float fsum = (tid < GRIM_Q8_1_BLOCK_SIZE) ? (float)qi : 0.0f;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        fsum += __shfl_xor(fsum, mask);

    if (tid == 0) {
        _Float16 hd = (_Float16)d;
        _Float16 hs = (_Float16)(fsum * d);
        unsigned short wd, ws;
        __builtin_memcpy(&wd, &hd, 2);
        __builtin_memcpy(&ws, &hs, 2);
        row_dst[0] = (unsigned char)(wd & 0xFF);
        row_dst[1] = (unsigned char)(wd >> 8);
        row_dst[2] = (unsigned char)(ws & 0xFF);
        row_dst[3] = (unsigned char)(ws >> 8);
    }

    if (tid < GRIM_Q8_1_BLOCK_SIZE) {
        ((int8_t*)(row_dst + 4))[tid] = qi;
    }
}

extern "C" __global__ void grim_dot4_q80_q81_gemv(
    const unsigned char* __restrict__ A_q81,
    const unsigned char* __restrict__ B_q80,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int col_base = blockIdx.x * 4;
    const int row      = blockIdx.y;
    const int lane     = threadIdx.x;

    if (row >= M || col_base >= N) return;

    const int n_q_blocks = K / GRIM_Q8_0_BLOCK_SIZE;
    const unsigned char* a_row = A_q81 + (long long)row * n_q_blocks * GRIM_Q8_1_BYTES;

    const unsigned char* b_col[4];
    const int active_cols = (col_base + 4 <= N) ? 4 : (N - col_base);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b_col[j] = (j < active_cols)
                 ? B_q80 + (long long)(col_base + j) * n_q_blocks * GRIM_Q8_0_BYTES
                 : nullptr;
    }

    float facc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < n_q_blocks; blk += 32) {
        const unsigned char* a_blk = a_row + blk * GRIM_Q8_1_BYTES;
        float d_a = fp16_to_float_device(((const unsigned short*)a_blk)[0]);
        const int8_t* a_codes = (const int8_t*)(a_blk + 4);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= active_cols) break;
            const unsigned char* b_blk = b_col[j] + blk * GRIM_Q8_0_BYTES;
            float d_b = fp16_to_float_device(((const unsigned short*)b_blk)[0]);
            const int8_t* b_codes = (const int8_t*)(b_blk + 2);

            int iacc = 0;
            #pragma unroll
            for (int e = 0; e < GRIM_Q8_0_BLOCK_SIZE; e += 4) {
                int a4;
                __builtin_memcpy(&a4, a_codes + e, 4);
                int b4;
                __builtin_memcpy(&b4, b_codes + e, 4);
                iacc = grim_sdot4(a4, b4, iacc);
            }
            facc[j] += (float)iacc * d_a * d_b;
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            facc[j] += __shfl_xor(facc[j], off);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < active_cols; j++) {
            C[(long long)row * N + col_base + j] = facc[j];
        }
    }
}

__device__ __forceinline__ unsigned grim_pk_f16(float lo, float hi) {
    _Float16 a = (_Float16)lo;
    _Float16 b = (_Float16)hi;
    unsigned short ba, bb;
    __builtin_memcpy(&ba, &a, 2);
    __builtin_memcpy(&bb, &b, 2);
    return (unsigned)ba | ((unsigned)bb << 16);
}

extern "C" __global__ void grim_dot2_q80_gemv(
    const unsigned short* __restrict__ act_f16,
    const unsigned char*  __restrict__ B_q80,
    float* __restrict__ C,
    int N, int K)
{
    const int col  = blockIdx.x;
    const int lane = threadIdx.x;
    if (col >= N) return;
    const int blocks_per_row = K / 32;
    const int row_bytes = blocks_per_row * 34;
    const unsigned char* row_ptr = B_q80 + (long long)col * row_bytes;
    const unsigned* act2 = (const unsigned*)act_f16;
    float facc = 0.0f;
    for (int blk = lane; blk < blocks_per_row; blk += 32) {
        float d = fp16_to_float_device(((const unsigned short*)(row_ptr + blk * 34))[0]);
        const unsigned char* cp = row_ptr + blk * 34 + 2;
        float bacc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            unsigned b4 = (unsigned)cp[j] | ((unsigned)cp[j+1] << 8)
                        | ((unsigned)cp[j+2] << 16) | ((unsigned)cp[j+3] << 24);
            float c0 = (float)(int)(signed char)(b4 & 0xFF);
            float c1 = (float)(int)(signed char)((b4 >>  8) & 0xFF);
            float c2 = (float)(int)(signed char)((b4 >> 16) & 0xFF);
            float c3 = (float)(int)(signed char)((b4 >> 24) & 0xFF);
            unsigned a01 = act2[(blk * 32 + j) >> 1];
            unsigned w01 = grim_pk_f16(c0, c1);
            asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(bacc) : "v"(a01), "v"(w01));
            unsigned a23 = act2[(blk * 32 + j + 2) >> 1];
            unsigned w23 = grim_pk_f16(c2, c3);
            asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(bacc) : "v"(a23), "v"(w23));
        }
        facc += bacc * d;
    }
    for (int off = 16; off > 0; off >>= 1)
        facc += __shfl_xor(facc, off);
    if (lane == 0)
        C[col] = facc;
}

#endif // RDNA3/RDNA4
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_dot4_gemv_kernel() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot4_q80_q81_gemv"),
            "missing dot4 Q8_0xQ8_1 GEMV kernel"
        );
    }

    #[test]
    fn source_contains_quantize_q8_1() {
        assert!(
            KERNEL_SOURCE.contains("grim_quantize_q8_1"),
            "missing Q8_1 activation quantizer"
        );
    }

    #[test]
    fn source_uses_signed_dot4() {
        assert!(
            KERNEL_SOURCE.contains("amdgcn_sudot4"),
            "must use __builtin_amdgcn_sudot4 for signed int8 dot (V_DOT4_I32_IU8)"
        );
    }

    #[test]
    fn source_retains_legacy_dot2_gemv() {
        assert!(
            KERNEL_SOURCE.contains("grim_dot2_q80_gemv"),
            "legacy dot2 GEMV retained for A/B testing"
        );
    }
}

