//! Standalone Q4_K dequantization HIP kernel for ROCm (Crow Tier).
//! [see: `block_q4_K`]
//!
//! Layout (verified against `grim_quant::dequant_q4k`): each 144-byte
//! super-block holds 256 weights. Bytes:
//!   [0..2]   d    = f16 super-block scale
//!   [2..4]   min  = f16 super-block minimum scale
//!   [4..16]  scales (12 bytes, packed 6-bit sub-block scales/mins)
//!   [16..144] 128 bytes = 256 × 4-bit nibbles
//!
//! Nibble interleaving (grim-specific, NOT the GGML per-sub-block layout):
//! for pair index `k` in 0..4, byte `qs[32*k+j]` (j in 0..32) holds
//!   lo nibble → sub-block `2k`,   weight j
//!   hi nibble → sub-block `2k+1`, weight j
//! scale/min for sub-block `s`:
//!   s<4:  sc = scales[s]&63,             m = scales[s+4]&63
//!   s>=4: sc = (scales[s+4]&0x0F)|((scales[s-4]>>6)<<4),
//!         m  = (scales[s+4]>>4) |((scales[s-4]>>6)<<4) [= scales[s]>>6<<4]
//! value = d * sc * q - min * m

/// HIP source for `grim_dequant_q4k`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    __device__ inline float dequant_q4k_grim_element(const unsigned char* block_ptr, int w) {
        const unsigned short* h_ptr = (const unsigned short*)block_ptr;
        float d    = fp16_to_float_device(h_ptr[0]);
        float dmin = fp16_to_float_device(h_ptr[1]);

        const unsigned char* scales = block_ptr + 4;
        const unsigned char* qs     = block_ptr + 16;

        int k  = w / 64;
        int off = w % 64;
        int s  = 2 * k + (off >= 32 ? 1 : 0);
        int j  = off & 31;

        unsigned char sc, m;
        if (s < 4) {
            sc = scales[s] & 63;
            m  = scales[s + 4] & 63;
        } else {
            sc = (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4);
            m  = (scales[s + 4] >> 4)  | ((scales[s - 4] >> 6) << 4);
        }

        int qs_byte = 32 * k + j;
        unsigned char q = (off < 32) ? (qs[qs_byte] & 0x0F) : (qs[qs_byte] >> 4);

        return d * (float)sc * (float)q - dmin * (float)m;
    }

    /// Dequantize one Q4_K super-block into 256 F32 values. Each output thread
    /// handles one super-block (144 bytes).
    __global__ void grim_dequant_q4k(
        const unsigned char* __restrict__ packed,
        float* __restrict__ out,
        int n_blocks)
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= n_blocks) return;

        const unsigned char* blk = packed + b * 144;
        for (int w = 0; w < 256; ++w) {
            out[b * 256 + w] = dequant_q4k_grim_element(blk, w);
        }
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

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

    // Host mirror of `dequant_q4k_grim_element` in KERNEL_SOURCE. Kept in lockstep
    // with the HIP device function so the kernel's bit-level arithmetic can be
    // validated against the CPU oracle `grim_quant::dequant_q4k` without a ROCm
    // device (the dispatch trusts this parity).
    fn dequant_q4k_grim_element_host(blk: &[u8], w: usize) -> f32 {
        let d = fp16_to_f32_host(blk[0], blk[1]);
        let dmin = fp16_to_f32_host(blk[2], blk[3]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];

        let k = w / 64;
        let off = w % 64;
        let s = 2 * k + if off >= 32 { 1 } else { 0 };
        let j = off & 31;

        let (sc, m) = if s < 4 {
            (scales[s] & 63, scales[s + 4] & 63)
        } else {
            let sc = (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4);
            let m = (scales[s + 4] >> 4) | ((scales[s - 4] >> 6) << 4);
            (sc, m)
        };

        let qs_byte = 32 * k + j;
        let q = if off < 32 {
            qs[qs_byte] & 0x0F
        } else {
            qs[qs_byte] >> 4
        };

        d * (sc as f32) * (q as f32) - dmin * (m as f32)
    }

    fn assert_close(a: f32, b: f32, msg: &str) {
        assert!(
            (a - b).abs() <= 1e-5 || (a - b).abs() / b.abs().max(1e-6) <= 1e-5,
            "{msg}: gpu mirror {a} != cpu oracle {b}"
        );
    }

    #[test]
    fn q4k_dequant_source_contains_entry() {
        assert!(KERNEL_SOURCE.contains("grim_dequant_q4k"));
        assert!(KERNEL_SOURCE.contains("dequant_q4k_grim_element"));
    }

    #[test]
    fn q4k_dequant_kernel_matches_cpu_oracle_golden() {
        // Reproduce the golden Q4K super-block from
        // grim_quant::tests::q4k_golden_cross_byte_scale_min_subblock_offset.
        let mut buf = vec![0u8; 144];
        buf[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d   = 1.0
        buf[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // min = 0.25
        let mut scales = [0u8; 12];
        scales[0] = 0x01; // sc0 = 1
        scales[4] = 0x00; // m0  = 0
        scales[8] = 0x35; // sc4_lo=5, m4_lo=3
        buf[4..16].copy_from_slice(&scales);
        buf[16] = 4 | (0 << 4); // out[0]: d*sc0*4 - min*m0 = 4.0
        buf[80] = 10 | (7 << 4); // out[128] lo nibble=10

        let oracle = grim_quant::dequant_q4k(&buf, 256).expect("cpu dequant");
        assert_eq!(oracle.len(), 256);

        let mirror: Vec<f32> = (0..256)
            .map(|w| dequant_q4k_grim_element_host(&buf, w))
            .collect();

        for (i, (&a, &b)) in mirror.iter().zip(oracle.iter()).enumerate() {
            assert_close(a, b, &format!("q4k mirror vs oracle at w={i}"));
        }
        assert_close(mirror[0], 4.0, "q4k mirror out[0]");
        assert_close(mirror[128], 5.0 * 10.0 - 0.75, "q4k mirror out[128]");
    }

    #[test]
    fn q4k_dequant_kernel_matches_cpu_oracle_random() {
        let mut rng_state: u64 = 0x9E37_79B9_5BF0_C465;
        let next = |st: &mut u64| {
            *st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*st >> 33) as u8
        };
        for _ in 0..32 {
            let mut buf = vec![0u8; 144];
            // Normalized f16 d / min (avoids the subnormal edge path).
            let d_exp = next(&mut rng_state) % 30;
            let d_bits = 0x3C00u16 + (d_exp as u16) * 0x400;
            buf[0..2].copy_from_slice(&d_bits.to_le_bytes());
            let min_exp = next(&mut rng_state) % 16;
            let min_bits = 0x3400u16 + (min_exp as u16) * 0x400;
            buf[2..4].copy_from_slice(&min_bits.to_le_bytes());
            for v in &mut buf[4..16] {
                *v = next(&mut rng_state) & 0x3F;
            }
            for v in &mut buf[16..144] {
                let lo = next(&mut rng_state) & 0xF;
                let hi = next(&mut rng_state) & 0xF;
                *v = lo | (hi << 4);
            }

            let oracle = grim_quant::dequant_q4k(&buf, 256).expect("cpu dequant");
            let mirror: Vec<f32> = (0..256)
                .map(|w| dequant_q4k_grim_element_host(&buf, w))
                .collect();

            for (i, (&a, &b)) in mirror.iter().zip(oracle.iter()).enumerate() {
                assert_close(a, b, &format!("q4k random mirror vs oracle w={i}"));
            }
        }
    }
}
