//! Standalone verification of the WMMA inline Q3_K dequant formula
//! (`grim_deq_q3k` in grim_wmma_gemm_q3k) against canonical dequant_q3k.

#[test]
fn q3k_wmma_inline_dequant_formula_vs_canonical() {
    for seed in 0..32u32 {
        let mut block = [0u8; 110];
        for (i, v) in block[..32].iter_mut().enumerate() {
            *v = (i.wrapping_mul(7).wrapping_add(seed as usize)) as u8;
        }
        for (i, v) in block[32..96].iter_mut().enumerate() {
            *v = (i.wrapping_mul(13).wrapping_add(seed as usize)) as u8;
        }
        for (i, v) in block[96..108].iter_mut().enumerate() {
            *v = (i.wrapping_mul(17).wrapping_add(seed as usize)) as u8;
        }
        block[108] = 0x00;
        block[109] = 0x3C; // d = 1.0
        let canon = grim_quant::dequant_q3k(&block, 256).unwrap();

        let d = half::f16::from_le_bytes([block[108], block[109]]).to_f32();
        let sr = &block[96..108];
        let au = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let av = [au(&sr[0..4]), au(&sr[4..8]), au(&sr[8..12])];
        let tmp = av[2];
        let km1 = 0x0303_0303u32;
        let km2 = 0x0F0F_0F0Fu32;
        let aux = [
            (av[0] & km2) | ((tmp & km1) << 4),
            (av[1] & km2) | (((tmp >> 2) & km1) << 4),
            ((av[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4),
            ((av[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4),
        ];
        let mut sc = [0i32; 16];
        let mut si = 0;
        for &av in aux.iter() {
            for b in 0..4 {
                let v = (av >> (8 * b)) & 0xFF;
                sc[si] = if v < 128 { v as i32 } else { v as i32 - 256 }; si += 1;
            }
        }
        let qs = &block[32..96];
        let hmask = &block[0..32];
        let mut worst = 0.0f32;
        for w in (0..256).step_by(8) {
            for e in 0..8 {
                let idx = w + e;
                let half = idx >> 7;
                let sub = (idx & 0x7F) >> 5;
                let p = idx & 0x1F;
                let sc_idx = (half << 3) + (sub << 1) + (p >> 4);
                let shift = sub << 1;
                let q_code = ((qs[(half << 5) + p]) >> shift) & 3;
                let hm = if (hmask[p] as u32 & (1u32 << (half * 4 + sub))) != 0 { 0 } else { 4 };
                let val = d * (sc[sc_idx] as f32 - 32.0) * (q_code as f32 - hm as f32);
                worst = worst.max((val - canon[idx]).abs());
            }
        }
        assert!(
            worst < 1e-5,
            "seed={seed}: WMMA q3k dequant formula diverges from canonical: {worst}"
        );
    }
}
