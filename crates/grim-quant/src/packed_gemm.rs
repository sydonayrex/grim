//! SIMD fast paths for the packed quantized GEMMs (`gemm_q8_0_packed`,
//! `gemm_q4k_packed`) defined in the crate root.
//!
//! Block layouts (authoritative source: the scalar reference implementations in
//! `lib.rs`, which these kernels mirror instruction-for-instruction where it
//! matters):
//!
//! * Q8_0 — 34-byte block: 2-byte little-endian f16 scale, then 32 i8 quants.
//!   Requires `k % 32 == 0`.
//! * Q4_K — 144-byte super-block per 256 weights: f16 `d`, f16 `min`,
//!   `scales[12]`, then `qs[128]`. There are eight 32-weight sub-blocks; byte
//!   `t` of `qs` carries weight `t` in its low nibble (sub-block `t / 32`) and
//!   weight `t + 128` in its high nibble (sub-block `t / 32 + 1`); the per
//!   sub-block scale/min pair comes from [`crate::get_scale_min_k4`]. Requires
//!   `k % 256 == 0`.
//!
//! Math contract: every kernel computes, per sub-block, the same sums the
//! scalar loop accumulates — `sum(a[l] * q[l])` with the quantized value used
//! exactly (sign-extension preserved) — then applies the block scale/min in
//! f32 and folds into the running dot product. Results therefore match the
//! scalar path (and dequantize-then-GEMM) up to ordinary floating-point
//! reassociation of the summation order, which is inherent to any vectorized
//! reduction.
//!
//! Note on instruction choice: the left operand `A` is arbitrary f32 in these
//! GEMMs, so pure integer byte-dots (`_mm256_maddubs_epi16`, `vdotq_s32`) have
//! no exact application on the A side. The kernels instead widen the quantized
//! bytes exactly to f32 lanes and multiply-accumulate there — simpler and
//! bit-exact on the quantized values, at some cost versus peak integer throughput
//! (correctness over peak perf).

// ---------------------------------------------------------------------------
// x86-64 / AVX2
// ---------------------------------------------------------------------------

/// Runtime AVX2 availability (gates the x86-64 fast paths).
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_detected() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use std::arch::x86_64::*;

    /// Sum of the 8 f32 lanes of a 256-bit vector.
    macro_rules! hsum256 {
        ($v:expr) => {{
            let v = $v;
            // s = [L0+L4, L1+L5, L2+L6, L3+L7]
            let s = _mm_add_ps(
                _mm256_castps256_ps128(v),
                _mm256_extractf128_ps::<1>(v),
            );
            // unpacklo(a,a) = [a0,a0,a1,a1], unpackhi(a,a) = [a2,a2,a3,a3]
            let lo = _mm_unpacklo_ps(s, s);
            let hi = _mm_unpackhi_ps(s, s);
            // t = [L0+L2+L4+L6, _, L1+L3+L5+L7, _]
            let t = _mm_add_ps(lo, hi);
            let t = _mm_add_ss(t, _mm_unpackhi_ps(t, t));
            _mm_cvtss_f32(t)
        }};
    }

    /// `dsc * sum(a[i] * q[i]) - mm * sum(a[i])` over the 32 consecutive
    /// weights starting at `a[a_off]`, where `bytes` is a 256-bit vector whose
    /// 32 bytes hold the quantized values (i8 codes or unpacked nibbles, one
    /// per weight, order preserved). `dsc`/`mm` are the sub-block's f32 scale
    /// and offset.
    macro_rules! sub_block_contrib {
        ($a:expr, $a_off:expr, $bytes:expr, $dsc:expr, $mm:expr) => {{
            let raw = $bytes;
            let w16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(raw));
            let w16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(raw));
            let weights = [
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w16_lo))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(w16_lo))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(w16_hi))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(w16_hi))),
            ];
            let aptr = ($a).as_ptr().add($a_off);
            let mut dq = _mm256_setzero_ps();
            let mut da = _mm256_setzero_ps();
            for (lane, w) in weights.iter().enumerate() {
                let av = _mm256_loadu_ps(aptr.add(lane * 8));
                dq = _mm256_add_ps(dq, _mm256_mul_ps(av, *w));
                da = _mm256_add_ps(da, av);
            }
            ($dsc) * hsum256!(dq) - ($mm) * hsum256!(da)
        }};
    }

    /// AVX2 kernel for [`crate::gemm_q8_0_packed`].
    ///
    /// # Safety
    /// The caller must have verified AVX2 support at runtime, and the inputs
    /// must satisfy the validation contract of the public function
    /// (`k % 32 == 0`, buffer lengths sufficient for `[m, k]` x `[n, k]`).
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn gemm_q8_0_packed_avx2(
        a: &[f32],
        b_q80_bytes: &[u8],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        unsafe {
            let blocks_per_row = k / 32;
            let stride_b = blocks_per_row * 34;
            let mut c = vec![0.0f32; m * n];

            for row_m in 0..m {
                for col_n in 0..n {
                    let b_row = &b_q80_bytes[col_n * stride_b..(col_n + 1) * stride_b];
                    let mut dot = 0.0f32;
                    let mut b_pos = 0usize;
                    let mut a_pos = row_m * k;

                    for _blk in 0..blocks_per_row {
                        let scale = crate::f16_to_f32(b_row[b_pos], b_row[b_pos + 1]);

                        // Widen the 32 i8 quants to four f32 vectors (exact
                        // sign extension: i8 -> i16 -> i32 -> f32) and
                        // multiply-accumulate against the matching A lanes.
                        let q = _mm256_loadu_si256(b_row.as_ptr().add(b_pos + 2) as *const __m256i);
                        let halves = [
                            _mm256_castsi256_si128(q),
                            _mm256_extracti128_si256::<1>(q),
                        ];
                        let mut acc = _mm256_setzero_ps();
                        for (half, q_half) in halves.iter().enumerate() {
                            let w16 = _mm256_cvtepi8_epi16(*q_half);
                            let quarters = [
                                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w16)),
                                _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(w16)),
                            ];
                            for (quarter, wq) in quarters.iter().enumerate() {
                                let av =
                                    _mm256_loadu_ps(a.as_ptr().add(a_pos + half * 16 + quarter * 8));
                                acc = _mm256_add_ps(acc, _mm256_mul_ps(av, _mm256_cvtepi32_ps(*wq)));
                            }
                        }

                        dot += scale * hsum256!(acc);
                        b_pos += 34;
                        a_pos += 32;
                    }

                    c[row_m * n + col_n] = dot;
                }
            }

            c
        }
    }

    /// AVX2 kernel for [`crate::gemm_q4k_packed`].
    ///
    /// Nibble unpacking and scale/min lookup stay close to the scalar
    /// reference; the per-sub-block dot products are vectorized.
    ///
    /// # Safety
    /// The caller must have verified AVX2 support at runtime, and the inputs
    /// must satisfy the validation contract of the public function
    /// (`k % 256 == 0`, buffer lengths sufficient for `[m, k]` x `[n, k]`).
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn gemm_q4k_packed_avx2(
        a: &[f32],
        b_q4k_bytes: &[u8],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        unsafe {
            let blocks_per_row = k / 256;
            let stride_b = blocks_per_row * 144;
            let mut c = vec![0.0f32; m * n];
            let mask_0f = _mm256_set1_epi8(0x0F);

            for row_m in 0..m {
                for col_n in 0..n {
                    let b_row = &b_q4k_bytes[col_n * stride_b..(col_n + 1) * stride_b];
                    let mut dot = 0.0f32;
                    let mut pos = 0usize;
                    let mut a_base = row_m * k;

                    for _blk in 0..blocks_per_row {
                        let d = crate::f16_to_f32(b_row[pos], b_row[pos + 1]);
                        let min = crate::f16_to_f32(b_row[pos + 2], b_row[pos + 3]);
                        let scales = &b_row[pos + 4..pos + 16];
                        let qs = &b_row[pos + 16..pos + 144];

                        // Same iteration shape as the scalar loop: 4 groups of
                        // 64 weights; low nibbles feed even sub-blocks, high
                        // nibbles feed odd ones.
                        let mut is = 0usize;
                        for group in 0..4 {
                            let raw =
                                _mm256_loadu_si256(qs.as_ptr().add(group * 32) as *const __m256i);
                            let lo_bytes = _mm256_and_si256(raw, mask_0f);
                            let hi_bytes =
                                _mm256_and_si256(_mm256_srli_epi16::<4>(raw), mask_0f);

                            let (sc, mi) = crate::get_scale_min_k4(is, scales);
                            dot += sub_block_contrib!(
                                a,
                                a_base,
                                lo_bytes,
                                d * sc,
                                min * mi
                            );
                            let (sc, mi) = crate::get_scale_min_k4(is + 1, scales);
                            dot += sub_block_contrib!(
                                a,
                                a_base + 32,
                                hi_bytes,
                                d * sc,
                                min * mi
                            );

                            a_base += 64;
                            is += 2;
                        }
                        pos += 144;
                    }

                    c[row_m * n + col_n] = dot;
                }
            }

            c
        }
    }
}

// ---------------------------------------------------------------------------
// aarch64 / NEON (NEON is baseline on aarch64 — no runtime detection needed)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use std::arch::aarch64::*;

    /// NEON kernel for [`crate::gemm_q8_0_packed`] (exact i8 -> f32 widening,
    /// widening accumulate in f32; mirrors the AVX2 kernel above).
    ///
    /// # Safety
    /// Inputs must satisfy the validation contract of the public function
    /// (`k % 32 == 0`, buffer lengths sufficient for `[m, k]` x `[n, k]`).
    pub(crate) unsafe fn gemm_q8_0_packed_neon(
        a: &[f32],
        b_q80_bytes: &[u8],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        unsafe {
            let blocks_per_row = k / 32;
            let stride_b = blocks_per_row * 34;
            let mut c = vec![0.0f32; m * n];

            for row_m in 0..m {
                for col_n in 0..n {
                    let b_row = &b_q80_bytes[col_n * stride_b..(col_n + 1) * stride_b];
                    let mut dot = 0.0f32;
                    let mut b_pos = 0usize;
                    let mut a_pos = row_m * k;

                    for _blk in 0..blocks_per_row {
                        let scale = crate::f16_to_f32(b_row[b_pos], b_row[b_pos + 1]);

                        let qs8 = vreinterpretq_s8_u8(vld1q_u8(b_row.as_ptr().add(b_pos + 2)));
                        let halves = [
                            vmovl_s8(vget_low_s8(qs8)),
                            vmovl_s8(vget_high_s8(qs8)),
                        ];
                        let mut acc = vdupq_n_f32(0.0);
                        for (half, q16) in halves.iter().enumerate() {
                            let quarters = [
                                vmovl_s16(vget_low_s16(*q16)),
                                vmovl_s16(vget_high_s16(*q16)),
                            ];
                            for (quarter, q32) in quarters.iter().enumerate() {
                                let av =
                                    vld1q_f32(a.as_ptr().add(a_pos + half * 16 + quarter * 4));
                                acc = vmlaq_f32(acc, av, vcvtq_f32_s32(*q32));
                            }
                        }

                        dot += scale * vaddvq_f32(acc);
                        b_pos += 34;
                        a_pos += 32;
                    }

                    c[row_m * n + col_n] = dot;
                }
            }

            c
        }
    }
}
