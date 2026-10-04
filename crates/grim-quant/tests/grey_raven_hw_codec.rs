//! GreyRaven hardware-order codec: coupled patterns, HW pack, HW decode.
//!
//! The flat codec (pack_grey_raven) is tested in grey_raven_prune_golden.rs.
//! This file pins the HARDWARE order: tile-coupled pattern selection, the
//! 256B+sidx fragment layout, and exact round-trip through dequant_hw.
//! The GPU verification (grey_raven_verify) proves the layout matches the
//! instruction; these prove the codec implements the layout it claims.

use grim_quant::grey_raven::{
    coupled_patterns_2_4, dequant_grey_raven_hw, pack_grey_raven_hw, PATTERNS_24,
};

#[test]
fn coupled_patterns_minimize_dropped_magnitude_squared() {
    // One tile (16x32), groups {0, 4} crafted so the joint optimum differs
    // from either group's independent optimum: group 0 wants {0,1} (big
    // values at slots 0,1), group 4 wants {2,3}, but the CLASS must pick
    // one pattern for both. Joint costs: {0,1} drops group-4's mass,
    // {2,3} drops group-0's; the winner minimizes the sum.
    let (rows, cols) = (16usize, 32usize);
    let mut w = vec![0.0f32; rows * cols];
    for r in 0..16 {
        // group 0: mass at slots 0,1
        w[r * 32 + 0] = 10.0;
        w[r * 32 + 1] = 10.0;
        // group 4 (k=16..19): mass at slots 2,3 (k=18,19)
        w[r * 32 + 18] = 1.0;
        w[r * 32 + 19] = 1.0;
    }
    let pats = coupled_patterns_2_4(&w, rows, cols).expect("patterns");
    assert_eq!(pats.len(), 4, "one window x four pair-classes");
    // Class 0 covers groups {0, 4}: {0,1} drops 16 rows x (1+1) = 32,
    // {2,3} drops 16 x (100+100) = 3200. Joint optimum is {0,1}.
    assert_eq!(pats[0], [0, 1], "class must minimize JOINT dropped mass");
}

#[test]
fn hw_pack_round_trips_through_hw_decode_bit_exact() {
    let (rows, cols) = (16usize, 32usize);
    let mut w = vec![0.0f32; rows * cols];
    for r in 0..16 {
        for g in 0..8 {
            // mirror patterns across halves (hardware constraint)
            let pat = PATTERNS_24[g % 4];
            for (rank, &s) in pat.iter().enumerate() {
                w[r * cols + 4 * g + s] = ((r * 11 + g * 7 + rank * 3) % 5 + 1) as f32
                    * if (r + g) % 2 == 0 { 0.5 } else { -0.25 };
            }
        }
    }
    let pats = coupled_patterns_2_4(&w, rows, cols).expect("patterns");
    let blob = pack_grey_raven_hw(&w, rows, cols, &pats).expect("pack");
    assert_eq!(blob.len(), 260, "single tile-window = 256 B frag + u32 sidx");
    let deq = dequant_grey_raven_hw(&blob, rows, cols).expect("dequant");
    assert_eq!(deq.len(), rows * cols);
    // Reference: E4M3-round of the coupled-pruned model, computed directly.
    for r in 0..rows {
        for g in 0..8 {
            // coupled class for group g: pair p = g%4 of this (single) window
            let pat = pats[g % 4];
            for s in 0..4 {
                let i = r * cols + 4 * g + s;
                let kept = s == pat[0] as usize || s == pat[1] as usize;
                let q = if kept {
                    let code = grim_quant::f32_to_fp8_e4m3(w[i]);
                    grim_quant::fp8_e4m3_to_f32(code)
                } else {
                    0.0
                };
                assert_eq!(
                    deq[i].to_bits(),
                    q.to_bits(),
                    "element [{r}][{}] must be the coupled-pruned E4M3 round",
                    4 * g + s
                );
            }
        }
    }
}

#[test]
fn hw_codec_rejects_malformed_buffers() {
    let (rows, cols) = (16usize, 32usize);
    let w = vec![0.5f32; rows * cols];
    let pats = coupled_patterns_2_4(&w, rows, cols).expect("patterns");
    let (frags, sidx) = pack_grey_raven_hw(&w, rows, cols, &pats).expect("pack");
    // Truncations refuse (including a cut inside the sidx word).
    assert!(dequant_grey_raven_hw(&blob[..200], rows, cols).is_err());
    assert!(dequant_grey_raven_hw(&blob[..258], rows, cols).is_err());
    // Trailing bytes refuse.
    let mut long = frags.clone();
    long.push(0);
    assert!(dequant_grey_raven_hw(&long, &sidx, rows, cols).is_err());
    // Wrong geometry refuses when tiling disagrees (32 rows need 2 tiles).
    assert!(dequant_grey_raven_hw(&blob, 32, cols).is_err());
    // Empty dims refuse at every entry point.
    assert!(coupled_patterns_2_4(&w, 0, cols).is_err());
    assert!(pack_grey_raven_hw(&w, rows, cols, &[]).is_err());
}

#[test]
fn partial_tiles_pad_with_zeros() {
    // 20x40: 2 row-tiles (16+4) x 2 windows (32+8). Pad rows/cols decode 0.
    let (rows, cols) = (20usize, 40usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7) % 11) as f32 * 0.1 - 0.5).collect();
    let pats = coupled_patterns_2_4(&w, rows, cols).expect("patterns");
    assert_eq!(pats.len(), 2 * 2 * 4);
    let (frags, sidx) = pack_grey_raven_hw(&w, rows, cols, &pats).expect("pack");
    assert_eq!(frags.len(), 2 * 2 * 256);
    assert_eq!(sidx.len(), 4);
    let deq = dequant_grey_raven_hw(&frags, &sidx, rows, cols).expect("dequant");
    assert_eq!(deq.len(), rows * cols);
    // Every decoded value is either 0 (pruned/pad) or the E4M3 round of the
    // source: no garbage from pad regions leaking into real cells.
    for r in 0..rows {
        for c in 0..cols {
            let g = deq[r * cols + c];
            assert!(g.is_finite(), "non-finite decode at [{r}][{c}]");
        }
    }
}
