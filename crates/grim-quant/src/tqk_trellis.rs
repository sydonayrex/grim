//! TQK Trellis 2-bit / 3-bit / 4-bit vector quantization helpers.

#[inline]
pub fn tqk_trellis_enabled() -> bool {
    std::env::var("GRIM_STRATA_TQK_TRELLIS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// 2-bit trellis codebook: {-1.5, -0.5, 0.5, 1.5} * scale.
pub const TQK_2BIT_CODEBOOK: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];

/// Pack 16 2-bit indices (0..=3) into a single `u32`.
#[inline]
pub fn pack_tqk_2bit(indices: &[u8; 16]) -> u32 {
    let mut word = 0u32;
    for (i, &idx) in indices.iter().enumerate() {
        word |= ((idx & 0x3) as u32) << (i * 2);
    }
    word
}

/// Dequantize 16 2-bit trellis weights from a packed `u32` and block `scale`.
#[inline]
pub fn dequant_tqk_2bit(word: u32, scale: f32) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        let idx = ((word >> (i * 2)) & 0x3) as usize;
        out[i] = TQK_2BIT_CODEBOOK[idx] * scale;
    }
    out
}

/// Pack 8 4-bit signed-offset indices (0..=15, representing -8..=+7) into a `u32`.
#[inline]
pub fn pack_tqk_4bit(indices: &[u8; 8]) -> u32 {
    let mut word = 0u32;
    for (i, &idx) in indices.iter().enumerate() {
        word |= ((idx & 0xF) as u32) << (i * 4);
    }
    word
}

pub const TQK_LCG_A0: u32 = 81_276_397;
pub const TQK_LCG_B0: u32 = 2_471_635_446;
pub const TQK_LCG_A1: u32 = 59_162_845;
pub const TQK_LCG_B1: u32 = 1_382_947_113;

/// Compute two packed IEEE-754 FP16 values from 14-bit trellis state `s` in 2 integer ops.
#[inline]
pub fn tq_computed_pair_bits_with(s: u32, a: u32, b: u32) -> u32 {
    ((s.wrapping_mul(a).wrapping_add(b)) & 0x8FFF_8FFFu32) ^ 0x3B60_3B60u32
}

#[inline]
pub fn tq_computed_pair_bits(s: u32) -> u32 {
    tq_computed_pair_bits_with(s, TQK_LCG_A0, TQK_LCG_B0)
}

#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;
    if exp == 0 {
        f32::from_bits(sign)
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Generate a 4-tuple (`V=4` vector dimension) of FP32 codebook values from 14-bit state `state14`.
#[inline]
pub fn tq_computed_quad(state14: u16) -> [f32; 4] {
    let s = (state14 & 0x3FFF) as u32;
    let p0 = tq_computed_pair_bits_with(s, TQK_LCG_A0, TQK_LCG_B0);
    let p1 = tq_computed_pair_bits_with(s, TQK_LCG_A1, TQK_LCG_B1);
    [
        f16_bits_to_f32((p0 & 0xFFFF) as u16),
        f16_bits_to_f32((p0 >> 16) as u16),
        f16_bits_to_f32((p1 & 0xFFFF) as u16),
        f16_bits_to_f32((p1 >> 16) as u16),
    ]
}

/// Decode a 32-element block (`8` steps of `V=4` elements) from its 8 state indices and block scale.
pub fn dequant_tqk_block32(states: &[u16; 8], scale: f32) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    for step in 0..8 {
        let quad = tq_computed_quad(states[step]);
        for j in 0..4 {
            out[step * 4 + j] = quad[j] * scale;
        }
    }
    out
}

/// Contiguous-Group tail-biting Viterbi encoder.
pub fn encode_tqk_tail_biting_viterbi(target: &[f32; 32], k_bits: u8) -> ([u16; 8], f32) {
    let k = k_bits.clamp(2, 5) as u32;
    let n_u = 1usize << k;
    let amax = target.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if amax > 1e-12 { amax / 1.25 } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let mut norm_t = [0.0f32; 32];
    for i in 0..32 {
        norm_t[i] = target[i] * inv_scale;
    }

    let mut best_states = [0u16; 8];
    let mut best_err = f32::INFINITY;

    for seed_idx in 0..64u16 {
        let s0 = (seed_idx.wrapping_mul(251)) & 0x3FFF;
        let mut beam: Vec<(u16, f32, [u16; 8])> = vec![(s0, 0.0, [0u16; 8])];
        for step in 0..8 {
            let tgt = &norm_t[step * 4..step * 4 + 4];
            let mut next_beam = Vec::with_capacity(beam.len() * n_u);
            for &(st, cost_so_far, path) in &beam {
                for u in 0..n_u {
                    let next_st = ((st >> k) | ((u as u16) << (14 - k))) & 0x3FFF;
                    let quad = tq_computed_quad(next_st);
                    let mut step_err = 0.0f32;
                    for j in 0..4 {
                        let d = tgt[j] - quad[j];
                        step_err += d * d;
                    }
                    let mut next_path = path;
                    next_path[step] = next_st;
                    next_beam.push((next_st, cost_so_far + step_err, next_path));
                }
            }
            next_beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            next_beam.truncate(8);
            beam = next_beam;
        }
        if let Some(&(_final_st, total_err, path)) = beam.first() {
            if total_err < best_err {
                best_err = total_err;
                best_states = path;
            }
        }
    }

    let raw = dequant_tqk_block32(&best_states, 1.0);
    let mut dot_tq = 0.0f32;
    let mut dot_qq = 0.0f32;
    for i in 0..32 {
        dot_tq += target[i] * raw[i];
        dot_qq += raw[i] * raw[i];
    }
    let opt_scale = if dot_qq > 1e-12 { dot_tq / dot_qq } else { scale };
    (best_states, opt_scale)
}

/// Dequantize 8 4-bit trellis weights from a packed `u32` and block `scale`.
#[inline]
pub fn dequant_tqk_4bit(word: u32, scale: f32) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        let q = ((word >> (i * 4)) & 0xF) as i32 - 8;
        out[i] = (q as f32) * scale;
    }
    out
}
