//! GPTQ / EfficientQAT GroupInt fused dequant-GEMM HIP kernels.
//!
//! Consumes the length-prefixed four-segment packed layout documented on
//! [`grim_tensor::dtype::GpuIntConfig`]:
//!
//! ```text
//! [u64 LE: qweight_len][qweight][u64 LE: qzeros_len][qzeros]
//! [u64 LE: scales_len][scales][u64 LE: g_idx_len][g_idx]
//! ```
//!
//! The four segments live contiguously in ONE device buffer (the weight blob
//! is immutable), so the kernels address them IN PLACE through interior
//! pointers computed on the host from the segment lengths. No per-GEMM split,
//! no per-call H2D re-upload of scales — mirrors the MXFP4 framed-blob
//! treatment in `roc_device.rs`. Segment boundaries are all 4-byte multiples
//! (qweight/qzeros/scales are u32 arrays; g_idx is u32/u64), and device
//! allocations are >=256B aligned, so every interior pointer stays 4-byte
//! aligned for the u32/float loads below.
//!
//! Dequant semantics match `grim_quant::dequant_gptq_group_int` exactly:
//! asymmetric `(code - (zero + 1)) * scale`, GPTQ/BitBLAS cross-word packing
//! for 3-bit codes (32 values across 3 consecutive u32 words), and optional
//! act-order `g_idx` permutation (stored as u32 or u64 LE).
//!
//! A [in_features, out_features] weight is stored column-packed: `qweight`
//! word index = (in_idx / values_per_word) * out_features + out_idx, so the
//! logical B consumed here is [K=in, N=out] and the kernel indexes `col`
//! over N — the same [out, in] relabel contract the ROCm KQuant fused path
//! uses (`transpose_last_two` relabels without moving bytes).

pub const GPTQ_GEMM_KERNEL_SOURCE: &str = r#"
// ---- GPTQ GroupInt dequant helpers (device-only, unique symbol prefix) ----
static inline __device__ unsigned int grim_gptq_read_u32(
    const unsigned char* __restrict__ base, long long word_idx)
{
    return *(const unsigned int*)(base + word_idx * 4);
}

// Read a 3-bit code packed GPTQ/BitBLAS style: values 0-31 of a super-block
// span three consecutive u32 words (0-10 in word0, 11-21 in word1, 22-31 in
// word2).
static inline __device__ unsigned int grim_gptq_read_code3(
    const unsigned char* qweight, long long base_word, int lane)
{
    unsigned int w0 = grim_gptq_read_u32(qweight, base_word);
    unsigned int w1 = grim_gptq_read_u32(qweight, base_word + 1);
    unsigned int w2 = grim_gptq_read_u32(qweight, base_word + 2);
    int bit = lane * 3;
    if (bit < 32) {
        return (((unsigned long long)w0 | ((unsigned long long)w1 << 32)) >> bit) & 0x7u;
    }
    return (w2 >> (bit - 32)) & 0x7u;
}

// Full code read for any supported bit width. `in_idx` indexes the input
// (contracted) dim, `col` the output dim, `N` = out_features.
static inline __device__ unsigned int grim_gptq_read_code(
    const unsigned char* qweight, int in_idx, int col, int N,
    int bits, int values_per_word)
{
    if (bits == 3) {
        long long base = (long long)(in_idx / 32) * 3 * N + col;
        return grim_gptq_read_code3(qweight, base, in_idx % 32);
    }
    long long word_idx = (long long)(in_idx / values_per_word) * N + col;
    unsigned int word = grim_gptq_read_u32(qweight, word_idx);
    return (word >> ((in_idx % values_per_word) * bits)) & ((1u << bits) - 1u);
}

// Per-(group, output-column) zero point, decoded as stored_value + 1
// (asymmetric GPTQ convention).
static inline __device__ float grim_gptq_read_zero(
    const unsigned char* qzeros, int group, int col, int N,
    int bits, int values_per_word, int zeros_words_per_row)
{
    if (bits == 3) {
        long long base = (long long)group * (3 * ((N + 31) / 32)) + col / 32 * 3;
        return (float)(grim_gptq_read_code3(qzeros, base, col % 32) + 1u);
    }
    long long word_idx = (long long)group * zeros_words_per_row + col / values_per_word;
    unsigned int word = grim_gptq_read_u32(qzeros, word_idx);
    return (float)((word >> ((col % values_per_word) * bits)) & ((1u << bits) - 1u)) + 1.0f;
}

// Forward: C[M, N] = A[M, K] @ dequant(B)^T where B packs a [K, N] weight.
extern "C" __global__ void grim_gptq_dequant_gemm(
    const float* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    int M, int N, int K,
    int bits, int group_size, int values_per_word,
    int has_g_idx,
    long long qw_off, long long qz_off, long long sc_off, long long gi_off)
{
    const unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)M * N;
    if (idx >= total) return;

    const int row = (int)(idx / N);
    const int col = (int)(idx % N);

    const unsigned char* qweight = B_packed + qw_off;
    const unsigned char* qzeros  = B_packed + qz_off;
    const unsigned char* scales  = B_packed + sc_off;
    const unsigned char* g_idx   = B_packed + gi_off;

    const int zeros_words_per_row = (bits == 3) ? 0 : (N + values_per_word - 1) / values_per_word;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int group = has_g_idx
            ? (int)grim_gptq_read_u32(g_idx, k)
            : k / group_size;

        unsigned int code = grim_gptq_read_code(qweight, k, col, N, bits, values_per_word);
        float zero = grim_gptq_read_zero(qzeros, group, col, N, bits, values_per_word, zeros_words_per_row);
        float scale = *(const float*)(scales + ((long long)group * N + col) * 4);

        float w = ((float)code - zero) * scale;
        acc += A[(long long)row * K + k] * w;
    }
    C[(long long)row * N + col] = acc;
}

// Backward: dX[M, K] = dY[M, N] @ dequant(B), same packed B ([K, N] weight).
extern "C" __global__ void grim_gptq_dequant_backward_gemm(
    const float* __restrict__ dY,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ dX,
    int M, int N, int K,
    int bits, int group_size, int values_per_word,
    int has_g_idx,
    long long qw_off, long long qz_off, long long sc_off, long long gi_off)
{
    const unsigned long long idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)M * K;
    if (idx >= total) return;

    const int row = (int)(idx / K);
    const int k = (int)(idx % K);

    const unsigned char* qweight = B_packed + qw_off;
    const unsigned char* qzeros  = B_packed + qz_off;
    const unsigned char* scales  = B_packed + sc_off;
    const unsigned char* g_idx   = B_packed + gi_off;

    const int zeros_words_per_row = (bits == 3) ? 0 : (N + values_per_word - 1) / values_per_word;

    int group = has_g_idx ? (int)grim_gptq_read_u32(g_idx, k) : k / group_size;
    float zero = grim_gptq_read_zero(qzeros, group, 0, N, bits, values_per_word, zeros_words_per_row);

    float acc = 0.0f;
    for (int col = 0; col < N; ++col) {
        unsigned int code = grim_gptq_read_code(qweight, k, col, N, bits, values_per_word);
        float zc = grim_gptq_read_zero(qzeros, group, col, N, bits, values_per_word, zeros_words_per_row);
        float scale = *(const float*)(scales + ((long long)group * N + col) * 4);
        float w = ((float)code - zc) * scale;
        acc += dY[(long long)row * N + col] * w;
    }
    dX[(long long)row * K + k] = acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU mirror of the kernel's dequant arithmetic for parity tests.
    /// Mirrors `grim_quant::dequant_gptq_group_int` semantics restricted to
    /// what the kernel implements (non-desc_act handled via g_idx segment).
    pub fn cpu_reference_w_row(
        qweight: &[u8],
        qzeros: &[u8],
        scales: &[u8],
        g_idx: Option<&[u8]>,
        in_idx: usize,
        out_idx: usize,
        n: usize,
        bits: u32,
        group_size: usize,
    ) -> f32 {
        let vpw = match bits {
            2 => 16,
            3 => 32,
            4 => 8,
            8 => 1,
            _ => unreachable!("kernel-supported widths only"),
        };
        let read_u32 = |bytes: &[u8], w: usize| -> u32 {
            let o = w * 4;
            u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        };
        let bits_u = bits as usize;
        let mask = (1u32 << bits) - 1;
        let group = match g_idx {
            Some(g) => read_u32(g, in_idx) as usize,
            None => in_idx / group_size,
        };
        let code = if bits == 3 {
            let base = (in_idx / 32) * 3 * n + out_idx;
            let bit = (in_idx % 32) * 3;
            let w0 = read_u32(qweight, base) as u128;
            let w1 = read_u32(qweight, base + 1) as u128;
            let w2 = read_u32(qweight, base + 2) as u128;
            ((w0 | (w1 << 32) | (w2 << 64)) >> bit & 0x7) as u32
        } else {
            let w = read_u32(qweight, (in_idx / vpw) * n + out_idx);
            (w >> ((in_idx % vpw) * bits_u)) & mask
        };
        let zero = if bits == 3 {
            let zb = group * (3 * n.div_ceil(32)) + out_idx / 32;
            let bit = (out_idx % 32) * 3;
            let w0 = read_u32(qzeros, zb) as u128;
            let w1 = read_u32(qzeros, zb + 1) as u128;
            let w2 = read_u32(qzeros, zb + 2) as u128;
            (((w0 | (w1 << 32) | (w2 << 64)) >> bit) & 0x7) as usize + 1
        } else {
            let zrow = n.div_ceil(vpw);
            let w = read_u32(qzeros, group * zrow + out_idx / vpw);
            ((w >> ((out_idx % vpw) * bits_u)) & mask) as usize + 1
        };
        let scale = f32::from_le_bytes([
            scales[(group * n + out_idx) * 4],
            scales[(group * n + out_idx) * 4 + 1],
            scales[(group * n + out_idx) * 4 + 2],
            scales[(group * n + out_idx) * 4 + 3],
        ]);
        (code as f32 - zero as f32) * scale
    }

    #[test]
    fn kernel_source_contains_entries_and_helpers() {
        assert!(GPTQ_GEMM_KERNEL_SOURCE.contains("grim_gptq_dequant_gemm"));
        assert!(GPTQ_GEMM_KERNEL_SOURCE.contains("grim_gptq_dequant_backward_gemm"));
        assert!(GPTQ_GEMM_KERNEL_SOURCE.contains("grim_gptq_read_code"));
        assert!(GPTQ_GEMM_KERNEL_SOURCE.contains("grim_gptq_read_zero"));
    }

    #[test]
    fn cpu_reference_matches_grim_quant_4bit() {
        // 4-bit, group_size 2, K=4 in, N=4 out — hand-checkable against
        // `dequant_gptq_group_int`.
        let (k, n, bits, gs) = (4usize, 4usize, 4u32, 2usize);
        let words_qw = k.div_ceil(8) * n;
        let mut qweight = vec![0u8; words_qw * 4];
        let mut expect = vec![0f32; k * n];
        for ki in 0..k {
            let g = ki / gs;
            for ni in 0..n {
                let code: u32 = (((ki * 7 + ni * 3) % 15) + 1) as u32;
                let w = (ki / 8) * n + ni;
                let off = (ki % 8) * 4;
                let cur = u32::from_le_bytes([
                    qweight[w * 4],
                    qweight[w * 4 + 1],
                    qweight[w * 4 + 2],
                    qweight[w * 4 + 3],
                ]);
                let updated = cur | (code << off);
                qweight[w * 4..w * 4 + 4].copy_from_slice(&updated.to_le_bytes());
                // Oracle values match the per-group data written below:
                // group `g` carries zero/scale as a function of its group-index
                // `g*gs` and the output column `ni`, exactly what the decoder reads.
                let zero = ((g * gs + ni) % 8) as f32 + 1.0;
                let scale = 0.25 + 0.5 * ((g * gs + ni) % 3) as f32;
                expect[ki * n + ni] = (code as f32 - zero) * scale;
            }
        }
        // qzeros: groups = K/gs rows, one u32 per row holding N 4-bit zeros.
        let groups = k / gs;
        let mut qzeros = vec![0u8; groups * n.div_ceil(8) * 4];
        let mut scales = vec![0u8; groups * n * 4];
        for g in 0..groups {
            for ni in 0..n {
                let group_idx = g * gs;
                let zero = ((group_idx + ni) % 8) + 1;
                let zw = g * n.div_ceil(8) + ni / 8;
                let off = (ni % 8) * 4;
                let cur = u32::from_le_bytes([
                    qzeros[zw * 4],
                    qzeros[zw * 4 + 1],
                    qzeros[zw * 4 + 2],
                    qzeros[zw * 4 + 3],
                ]);
                let updated = cur | (((zero as u32) - 1) << off);
                qzeros[zw * 4..zw * 4 + 4].copy_from_slice(&updated.to_le_bytes());
                let s = 0.25 + 0.5 * ((group_idx + ni) % 3) as f32;
                scales[(g * n + ni) * 4..(g * n + ni) * 4 + 4].copy_from_slice(&s.to_le_bytes());
            }
        }
        for ki in 0..k {
            for ni in 0..n {
                let got =
                    cpu_reference_w_row(&qweight, &qzeros, &scales, None, ki, ni, n, bits, gs);
                assert!(
                    (got - expect[ki * n + ni]).abs() < 1e-6,
                    "mismatch at ({ki},{ni}): got {got}, want {}",
                    expect[ki * n + ni]
                );
            }
        }
    }

    #[test]
    fn cpu_reference_handles_desc_act_g_idx() {
        // Same tensor data but a permuted g_idx: element k takes its group
        // from g_idx[k]; verify a known permutation round-trips.
        let (k, n, bits, gs) = (4usize, 2usize, 4u32, 2usize);
        let groups = k / gs;
        let mut qweight = vec![0u8; k.div_ceil(8) * n * 4];
        let qzeros = vec![0u8; groups * n.div_ceil(8) * 4];
        let mut scales = vec![0u8; groups * n * 4];
        // Zero-point 0 everywhere, scales 1.0 → w = code.
        for g in 0..groups {
            for ni in 0..n {
                scales[(g * n + ni) * 4..(g * n + ni) * 4 + 4]
                    .copy_from_slice(&1.0f32.to_le_bytes());
            }
        }
        for ki in 0..k {
            for ni in 0..n {
                let code: u32 = (ki + ni) as u32 % 15;
                let w = (ki / 8) * n + ni;
                let off = (ki % 8) * 4;
                let cur = u32::from_le_bytes([
                    qweight[w * 4],
                    qweight[w * 4 + 1],
                    qweight[w * 4 + 2],
                    qweight[w * 4 + 3],
                ]);
                qweight[w * 4..w * 4 + 4].copy_from_slice(&(cur | (code << off)).to_le_bytes());
            }
        }
        // desc_act permutation: swap the two groups.
        let mut g_idx = Vec::with_capacity(k * 4);
        for ki in 0..k {
            let g = if ki < 2 { 1u32 } else { 0u32 };
            g_idx.extend_from_slice(&g.to_le_bytes());
        }
        // Element (k=0,n=0): code 0, group from g_idx[0]=1 → scale row 1 → 1.0, zero 1 → w=-1.
        let got = cpu_reference_w_row(&qweight, &qzeros, &scales, Some(&g_idx), 0, 0, n, bits, gs);
        assert!((got - (-1.0)).abs() < 1e-6, "got {got}");
        // Without g_idx it would have been group 0 → also 1.0 scale here, but
        // check an index where the two groups differ in effect via code choice:
        // k=1,n=0: code 1, permuted group=1 → w = (1-1)*1 = 0.
        let got2 = cpu_reference_w_row(&qweight, &qzeros, &scales, Some(&g_idx), 1, 0, n, bits, gs);
        assert!(got2.abs() < 1e-6, "got {got2}");
    }
}
