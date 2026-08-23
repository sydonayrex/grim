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
            // High sub-blocks (4..7), per upstream ggml get_scale_min_k4
            // (ggml-quants.c): sc takes its top 2 bits from scales[is-4],
            // but m takes its top 2 bits from scales[is] itself
            // ("q[j-0]" upstream). scales[is-4] here is a DIFFERENT byte;
            // using it corrupts min values for sub-blocks 4..7 (weights
            // 128..255 per super-block). Keep in lockstep with
            // grim_quant::get_scale_min_k4 and q5k/iq_gemm.
            sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);
        }

        unsigned char byte = qs[group * 32 + l];
        unsigned char q = half ? (byte >> 4) : (byte & 0xF);

        return d * (float)sc * (float)q - dmin * (float)m;
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::KERNEL_SOURCE;

    fn fp16_to_f32_host(lo: u8, hi: u8) -> f32 {
        let bits = u16::from_le_bytes([lo, hi]);
        let sign: u32 = (bits >> 15) as u32;
        let exp: u32 = ((bits >> 10) & 0x1F) as u32;
        let mant: u32 = (bits & 0x3FF) as u32;
        if exp == 0 {
            (mant as f32) * 2f32.powi(-24)
        } else if exp == 31 {
            f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
        } else {
            f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
        }
    }

    /// Host mirror of the DEVICE fn `dequant_q4k_element` above. The
    /// standalone-dequant kernel has its own mirror in `q4k_dequant.rs`; the
    /// GEMM-fused path used by every Q4_K matmul had NO oracle coverage
    /// before these tests — exactly how the `scales[is-4]` min-byte bug
    /// survived there.
    fn dequant_q4k_element_host(blk: &[u8], in_sb: usize) -> f32 {
        let d = fp16_to_f32_host(blk[0], blk[1]);
        let dmin = fp16_to_f32_host(blk[2], blk[3]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];

        let group = in_sb / 64;
        let half = (in_sb % 64) / 32;
        let l = in_sb % 32;
        let is = 2 * group + half;

        let (sc, m) = if is < 4 {
            (scales[is] & 63, scales[is + 4] & 63)
        } else {
            let sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
            let m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
            (sc, m)
        };

        let byte = qs[group * 32 + l];
        let q = if half != 0 { byte >> 4 } else { byte & 0xF };

        d * (sc as f32) * (q as f32) - dmin * (m as f32)
    }

    fn assert_close(a: f32, b: f32, msg: &str) {
        assert!(
            (a - b).abs() <= 1e-5 || (a - b).abs() / b.abs().max(1e-6) <= 1e-5,
            "{msg}: gemm-element mirror {a} != cpu oracle {b}"
        );
    }

    #[test]
    fn q4k_element_mirror_matches_cpu_oracle_high_bit_scales_golden() {
        // Golden block built so the DIVERGENT bits are non-zero AND differ:
        // for sub-block s=4 the correct m takes its top 2 bits from
        // scales[4] (top2 = 0b10), while the historical bug read them from
        // scales[0] (top2 = 0b01). Every weight 128..255 therefore
        // discriminates the two formulas — the older fixtures masked all
        // scale bytes to 6 bits and could not.
        let mut buf = vec![0u8; 144];
        buf[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d   = 1.0
        buf[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // min = 0.25
        let mut scales = [0u8; 12];
        scales[0] = 0x40; // sc0 low6=0, top2=0b01 (bug bait)
        scales[4] = 0x80; // m0 low6=0, top2=0b10 (true source of m4's high bits)
        scales[8] = 0x35; // sc4 lo nibble=5, m4 hi nibble=3
        buf[4..16].copy_from_slice(&scales);
        buf[16] = 4 | (0 << 4);
        buf[80] = 10 | (7 << 4); // out[128]: s=4 lo nibble path
        buf[81] = 6 | (2 << 4); // out[129]: s=4 lo nibble 6

        let oracle = grim_quant::dequant_q4k(&buf, 256).expect("cpu dequant");

        // Hand-derived expectation via ggml get_scale_min_k4(j=4):
        //   sc4 = (scales[8]&0xF)=5 | ((scales[0]>>6)<<4)=1<<4 -> sc4 = 21
        //   m4  = (scales[8]>>4)=3 | ((scales[4]>>6)<<4)=2<<4 -> m4  = 35
        // out[128] = d*sc4*q - min*m4 = 21*10 - 0.25*35 = 201.25.
        // Under the scales[is-4] bug, m4 would be 3|(1<<4)=19 → 205.25.
        assert_close(
            oracle[128],
            21.0 * 10.0 - 0.25 * 35.0,
            "oracle itself must follow ggml q[j-0]",
        );
        // Second LO-nibble weight of the same s=4 sub-block: out[128+l],
        // l in 0..32 reads the lo nibble of qs byte 64+l. buf[81] lo = 6:
        assert_close(
            oracle[129],
            21.0 * 6.0 - 0.25 * 35.0,
            "oracle second s=4 weight",
        );
        // The hi nibble of byte 80 belongs to sub-block s=5 (all-zero
        // scales here) — documents that hi weights of a 64-group use the
        // ODD scale index, i.e. out[160..192) are s=5 weights.
        assert_close(oracle[160], 0.0, "s=5 weights are zeroed by fixture");

        let mirror: Vec<f32> = (0..256)
            .map(|w| dequant_q4k_element_host(&buf, w))
            .collect();
        for (i, (&a, &b)) in mirror.iter().zip(oracle.iter()).enumerate() {
            assert_close(a, b, &format!("q4k element golden w={i}"));
        }
    }

    #[test]
    fn q4k_element_mirror_matches_cpu_oracle_full_range_random() {
        // Unlike the standalone-kernel fixtures, scale bytes here span the
        // FULL byte range so both cross-byte terms carry live high bits.
        let mut rng_state: u64 = 0xC0FF_EE12_3456_789A;
        let next = |st: &mut u64| {
            *st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*st >> 33) as u8
        };
        for _ in 0..64 {
            let mut buf = vec![0u8; 144];
            // Normalized f16 d/min, exponents bounded so the f16 value can
            // never overflow to inf (exp field ≤ 15+13 < 31).
            let d_exp = next(&mut rng_state) % 14;
            let d_bits = 0x3C00u16 + (d_exp as u16) * 0x400;
            buf[0..2].copy_from_slice(&d_bits.to_le_bytes());
            let min_exp = next(&mut rng_state) % 10;
            let min_bits = 0x3400u16 + (min_exp as u16) * 0x400;
            buf[2..4].copy_from_slice(&min_bits.to_le_bytes());
            for v in &mut buf[4..16] {
                *v = next(&mut rng_state); // FULL RANGE: high bits live
            }
            for v in &mut buf[16..144] {
                let lo = next(&mut rng_state) & 0xF;
                let hi = next(&mut rng_state) & 0xF;
                *v = lo | (hi << 4);
            }

            let oracle = grim_quant::dequant_q4k(&buf, 256).expect("cpu dequant");
            let mirror: Vec<f32> = (0..256)
                .map(|w| dequant_q4k_element_host(&buf, w))
                .collect();
            for (i, (&a, &b)) in mirror.iter().zip(oracle.iter()).enumerate() {
                assert_close(a, b, &format!("q4k element random mirror w={i}"));
            }
        }
    }

    #[test]
    fn q4k_element_device_source_pins_the_min_byte() {
        // Structural pin: the fused-GEMM device source must take m's top
        // bits from scales[is] (ggml "q[j-0]") — the exact line the
        // scales[is-4] mutation changes. Positive and negative assertions.
        assert!(
            KERNEL_SOURCE.contains("m  = (scales[is + 4] >> 4)  | ((scales[is] >> 6) << 4);"),
            "dequant_q4k_element m-line drifted from grim_quant::get_scale_min_k4"
        );
        assert!(
            !KERNEL_SOURCE.contains(">> 4)  | ((scales[is - 4] >> 6)"),
            "dequant_q4k_element must NOT read m's top bits from scales[is-4]"
        );
    }
}
