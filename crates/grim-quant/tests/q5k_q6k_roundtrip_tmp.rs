use grim_quant::{dequant_q4k, dequant_q5k, dequant_q6k, quant_q4k, quant_q5k, quant_q6k};

fn rt(name: &str, quant: fn(&[f32]) -> Result<Vec<u8>, grim_tensor::error::Error>,
      deq: fn(&[u8], usize) -> Result<Vec<f32>, grim_tensor::error::Error>, n: usize) {
    let mut state = 0xDEAD_BEEF_CAFEu64;
    let mut next = || {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        ((state >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
    };
    let data: Vec<f32> = (0..n).map(|_| next()).collect();
    let packed = quant(&data).expect("quant");
    let back = deq(&packed, n).expect("deq");
    let max_d: f32 = back.iter().zip(&data).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let amax = data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mean_d: f32 = back.iter().zip(&data).map(|(a, b)| (a - b).abs()).sum::<f32>() / n as f32;
    println!("{name}: max_d={max_d:.5} mean_d={mean_d:.5} amax={amax:.5} ratio_max={:.3}", max_d / amax);
}

#[test]
fn roundtrips() {
    rt("Q4K", quant_q4k, dequant_q4k, 256 * 32);
    rt("Q5K", quant_q5k, dequant_q5k, 256 * 32);
    rt("Q6K", quant_q6k, dequant_q6k, 256 * 32);
    {
        let mut state2 = 0xDEAD_BEEF_CAFEu64;
        let mut next2 = || {
            state2 ^= state2 << 13; state2 ^= state2 >> 7; state2 ^= state2 << 17;
            ((state2 >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let data: Vec<f32> = (0..256).map(|_| next2()).collect();
        let packed = quant_q6k(&data).unwrap();
        // decode with EXPLICIT scale bookkeeping
        let ql = &packed[0..128];
        let qh = &packed[128..192];
        let scales: Vec<i8> = packed[192..208].iter().map(|&b| b as i8).collect();
        let _d = f32::from(half::f16::from_le_bytes([packed[208], packed[209]]));
        let mut sc_idx = 0usize;
        for sg in 0..2 {
            let ql_idx = sg * 64; let qh_idx = sg * 32;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_idx + l] & 0x0F) | ((qh[qh_idx + l] & 0x03) << 4)) as f32 - 32.0;
                let q2 = ((ql[ql_idx + l + 32] & 0x0F) | ((qh[qh_idx + l] & 0x0C) << 2)) as f32 - 32.0;
                let q3 = ((ql[ql_idx + l] >> 4) | (qh[qh_idx + l] & 0x30)) as f32 - 32.0;
                let q4 = ((ql[ql_idx + l + 32] >> 4) | ((qh[qh_idx + l] & 0xC0) >> 2)) as f32 - 32.0;
                let sc1 = scales[sc_idx + is];
                let sc2 = scales[sc_idx + is + 2];
                let sc3 = scales[sc_idx + is + 4];
                let sc4 = scales[sc_idx + is + 6];
                let w1 = sg * 128 + l; let w2 = sg * 128 + 32 + l;
                let w3 = sg * 128 + 16 + l; let w4 = sg * 128 + 48 + l;
                println!("Q6KX sg={sg} l={l}: w{w1} q1={q1} sc1={sc1} | w{w2} q2={q2} sc2={sc2} | w{w3} q3={q3} sc3={sc3} | w{w4} q4={q4} sc4={sc4}");
            }
            sc_idx += 8;
        }
    }
    {
        let data: Vec<f32> = (0..256).map(|i| ((i % 16) as f32) * 0.031 - 0.25).collect();
        let packed = quant_q6k(&data).unwrap();
        let d = f32::from(half::f16::from_le_bytes([packed[208], packed[209]]));
        let scales: Vec<i8> = packed[192..208].iter().map(|&b| b as i8).collect();
        let dec = dequant_q6k(&packed, 256).unwrap();
        println!("Q6K dbg: d={d} scales={scales:?}");
        for w in [0usize, 1, 15, 16, 31, 32, 33, 64, 96, 128, 144] {
            println!("  w={w}: src={:.4} dec={:.4}", data[w], dec[w]);
        }
    }
    // debug: single block, print d/scales/first codes
    {
        let mut state2 = 0xDEAD_BEEF_CAFEu64;
        let mut next2 = || {
            state2 ^= state2 << 13; state2 ^= state2 >> 7; state2 ^= state2 << 17;
            ((state2 >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let data: Vec<f32> = (0..256).map(|_| next2()).collect();
        let packed = quant_q6k(&data).unwrap();
        let d = f32::from(half::f16::from_le_bytes([packed[208], packed[209]]));
        let scales: Vec<i8> = packed[192..208].iter().map(|&b| b as i8).collect();
        println!("Q6K debug: d={d} scales={scales:?}");
        let dec = dequant_q6k(&packed, 256).unwrap();
        println!("Q6K debug: src[0..4]={:?} dec[0..4]={:?}", &data[..4], &dec[..4]);
        println!("Q6K debug: src[16..20]={:?} dec[16..20]={:?}", &data[16..20], &dec[16..20]);
        // Worst element + its sub-block scale
        let max_w = data.iter().zip(&dec).enumerate().map(|(i, (a, b))| (i, (a - b).abs())).fold((0usize, 0.0f32), |m, x| if x.1 > m.1 { (x.0, x.1) } else { m });
        println!("Q6K debug: worst w={} src={:.4} dec={:.4} (sub-block {} of 16)", max_w.0, data[max_w.0], dec[max_w.0], max_w.0 / 16);
        // PER-SUB-BLOCK scale assignment check: decode sub-block j with
        // scale i and see which (j, i) pair minimizes error. If the encoder
        // and decoder disagree on scale indexing, argmin(i) != j.
        for j in 0..16usize {
            let src_sub = &data[16 * j..16 * j + 16];
            let _dec_sub = &dec[16 * j..16 * j + 16];
            let mut best_i = 0usize;
            let mut best_e = f32::INFINITY;
            for i in 0..16usize {
                let d_sc = d * scales[i] as f32;
                if d_sc == 0.0 { continue; }
                let e: f32 = src_sub.iter().map(|v| {
                    let q = ((v / d_sc).round() + 32.0).clamp(0.0, 63.0) as i32;
                    (d_sc * (q as f32 - 32.0) - v).abs()
                }).sum();
                if e < best_e { best_e = e; best_i = i; }
            }
            if best_i != j {
                println!("Q6K dbg: sub{j} best-matching scale index = {best_i} (expected {j})");
            }
        }
        println!("Q6K dbg: scale-index check done");
    }
}
