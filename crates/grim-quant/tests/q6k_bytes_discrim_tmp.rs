//! One-tensor discriminator: is the .grim Q6K payload byte-identical to
//! quant_q6k(dequant(GGUF source))? Env: XING_GRIM + XING_GGUF + TENSOR_NAME.
use grim_format::tprov::GrimProvider;
use grim_tensor::provider::TensorProvider;
use std::io::{Read, Seek};

#[test]
fn q6k_bytes_discrim() {
    let Ok(grim_path) = std::env::var("XING_GRIM") else {
        eprintln!("[SKIP] XING_GRIM unset");
        return;
    };
    let gguf_path = std::env::var("XING_GGUF").unwrap();
    let name = std::env::var("TENSOR_NAME").unwrap();

    let gp = GrimProvider::open(&grim_path).unwrap();
    let gguf_f = std::fs::File::open(&gguf_path).unwrap();
    let gg = grim_format::gguf::read_gguf(std::io::BufReader::new(&gguf_f)).unwrap();
    let t = gg.tensors.iter().find(|t| t.name == name).unwrap();
    let n: usize = t.shape().iter().product();

    let mut f2 = std::fs::File::open(&gguf_path).unwrap();
    f2.seek(std::io::SeekFrom::Start(gg.data_start + t.offset)).unwrap();
    let mut src_bytes = vec![0u8; t.size_bytes as usize];
    f2.read_exact(&mut src_bytes).unwrap();
    let s = grim_quant::dequant_iq3s(&src_bytes, n).unwrap();

    let raw = gp.get_packed(&name).unwrap();
    eprintln!("[disc] grim dtype={:?} bytes_len={}", raw.dtype.storage, raw.bytes.len());
    let expected_q6k = n / 256 * 210;
    eprintln!("[disc] expected Q6K len={expected_q6k}");
    let g = grim_quant::dequant_q6k(&raw.bytes, n).unwrap();

    // Fresh encode of the source f32 through the fixed encoder.
    let fresh = grim_quant::quant_q6k(&s).unwrap();
    let byte_eq = fresh == raw.bytes;
    eprintln!("[disc] quant_q6k(src) == grim bytes: {byte_eq} (fresh_len={})", fresh.len());
    if !byte_eq {
        let first_diff = fresh.iter().zip(raw.bytes.iter()).position(|(a, b)| a != b);
        eprintln!("[disc] first byte diff at {first_diff:?}");
    }

    let r = grim_quant::dequant_q6k(&fresh, n).unwrap();
    let amax = |v: &Vec<f32>| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let md = |a: &Vec<f32>, b: &Vec<f32>| {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    };
    eprintln!("[disc] amax src={:.5} grim={:.5} fresh={:.5}", amax(&s), amax(&g), amax(&r));
    eprintln!("[disc] maxdiff grim-vs-src={} fresh-vs-src={}", md(&g, &s), md(&r, &s));
    eprintln!("[disc] maxdiff grim-vs-fresh={}", md(&g, &r));
    // Worst 16 elements: where in the 256-block do they sit?
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        let da = (g[a] - s[a]).abs(); let db = (g[b] - s[b]).abs();
        db.partial_cmp(&da).unwrap()
    });
    for &i in idx.iter().take(16) {
        eprintln!("[disc] worst i={i} mod256={} qslot={} ih={} src={:.5} dec={:.5}",
            i % 256, (i % 128) / 32, ((i % 32) / 16), s[i], g[i]);
    }
    let worst = idx[0];
    let blk = worst / 256;
    // Hand-replicate the encoder for the failing block; print element 237's path.
    {
        let sblk = &s[blk * 256..blk * 256 + 256];
        let _l = [0i32; 256];
        let mut scales_f = [0.0f32; 16];
        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;
        for sg in 0..2 {
            for qs in 0..4 {
                for ih in 0..2 {
                    let gidx = sg * 8 + ih + qs * 2;
                    let base = sg * 128 + qs * 32 + ih * 16;
                    let sub = &sblk[base..base + 16];
                    let mut amax = 0.0f32;
                    let mut mx = 0.0f32;
                    for &v in sub {
                        let a = v.abs();
                        if a > amax { amax = a; mx = v; }
                    }
                    let sc_fit = if amax < 1e-15 { 0.0 } else {
                        let iscale = -32.0 / mx;
                        let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
                        for &v in sub {
                            let li = (iscale * v).round().clamp(-32.0, 31.0);
                            let w = v * v;
                            sumlx += w * v * li;
                            suml2 += w * li * li;
                        }
                        if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
                    };
                    scales_f[gidx] = sc_fit;
                    if sc_fit.abs() > max_abs_scale { max_abs_scale = sc_fit.abs(); max_scale = sc_fit; }
                }
            }
        }
        let _ = max_abs_scale;
        let d = -max_scale / 128.0;
        let gidx = 1 * 8 + 0 + 3 * 2; // sg=1, ih=0, qs=3 -> group 14
        let stored = (-128.0 / max_scale * scales_f[gidx]).round().clamp(-127.0, 127.0);
        let dd = d * stored;
        let v = sblk[237];
        let q = ((v / dd).round() + 32.0).clamp(0.0, 63.0) as u8;
        eprintln!("[disc] hand: fit14={:.6} maxsc={:.6} d={:.8} stored={:.1} dd={:.8} v={:.6} v/dd={:.3} q={} low={:x} high={:x}",
            scales_f[gidx], max_scale, d, stored, dd, v, v / dd, q, q & 0xF, (q >> 4) & 3);
        // Actual bytes from the fresh payload for this block:
        let fb = &fresh[blk * 210..blk * 210 + 210];
        eprintln!("[disc] actual: ql[109]={:#04x} (high nibble={}) qh[45]={:#010b} (bits6-7={})",
            fb[109], fb[109] >> 4, fb[128 + 45], (fb[128 + 45] >> 6) & 3);
    }

    // Encode-side algebra for the failing block: fit scale vs stored dd.
    {
        use grim_quant::quant_q6k;
        let sblk = &s[blk * 256..blk * 256 + 256];
        let fblk = quant_q6k(sblk).unwrap();
        let d = f32::from(half::f16::from_le_bytes([fblk[208], fblk[209]]));
        let sc: Vec<i8> = fblk[192..208].iter().map(|&b| b as i8).collect();
        eprintln!("[disc] single-block d={d} scales={sc:?}");
        // For each group (sg, qslot, ih): dd = d*sc[sg*8+ih+qslot*2],
        // max needed = |v|max/31.
        for sg in 0..2 {
            for qs in 0..4 {
                for ih in 0..2 {
                    let gidx = sg * 8 + ih + qs * 2;
                    let base = sg * 128 + qs * 32 + ih * 16;
                    let dd = d * sc[gidx] as f32;
                    let gmax = sblk[base..base + 16].iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let need = gmax / 31.0;
                    if need > 0.0 && dd.abs() < need * 0.9 {
                        eprintln!("[disc] TIGHT sg={sg} qs={qs} ih={ih} gidx={gidx}: dd={dd:.6} needed={need:.6} ratio={:.3}", dd.abs() / need);
                    }
                }
            }
        }
    }

    // Arbiter: full-payload reference decode (llama.cpp ggml-quants.c:1939).
    let nb = n / 256;
    let mut refd = Vec::with_capacity(n);
    for b in 0..nb {
        let base = b * 210;
        let ql = &fresh[base..base + 128];
        let qh = &fresh[base + 128..base + 192];
        let sc = &fresh[base + 192..base + 208];
        let bits = u16::from_le_bytes([fresh[base + 208], fresh[base + 209]]);
        let dd = f32::from(half::f16::from_le_bytes([fresh[base + 208], fresh[base + 209]]));
        let _ = bits;
        for nn in (0..256).step_by(128) {
            let qlb = &ql[nn / 2..nn / 2 + 64];
            let qhb = &qh[nn / 4..nn / 4 + 32];
            let scb = &sc[(nn / 128) * 8..(nn / 128) * 8 + 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((qlb[l] & 0x0F) | ((qhb[l] & 3) << 4)) as i32 - 32;
                let q2 = ((qlb[l + 32] & 0x0F) | ((qhb[l] >> 2 & 3) << 4)) as i32 - 32;
                let q3 = ((qlb[l] >> 4) | ((qhb[l] >> 4 & 3) << 4)) as i32 - 32;
                let q4 = ((qlb[l + 32] >> 4) | ((qhb[l] >> 6 & 3) << 4)) as i32 - 32;
                refd.push(dd * scb[is] as i8 as f32 * q1 as f32);
                refd.push(dd * scb[is + 2] as i8 as f32 * q2 as f32);
                refd.push(dd * scb[is + 4] as i8 as f32 * q3 as f32);
                refd.push(dd * scb[is + 6] as i8 as f32 * q4 as f32);
            }
        }
    }
    let md_ref = s.iter().zip(&refd).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("[disc] maxdiff reference-vs-src={md_ref}");
    let md_gr = g.iter().zip(&refd).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let n_dis = g.iter().zip(&refd).filter(|(a, b)| (**a - **b).abs() > 1e-3).count();
    eprintln!("[disc] maxdiff grim-vs-ref={md_gr} disagreeing_elems={n_dis}");
    {
        let mut di: Vec<usize> = (0..n).collect();
        di.sort_by(|&a, &b| {
            let da = (g[a] - refd[a]).abs(); let db = (g[b] - refd[b]).abs();
            db.partial_cmp(&da).unwrap()
        });
        for &i in di.iter().take(8) {
            eprintln!("[disc] dec-disagree i={i} mod256={} src={:.5} grim={:.5} ref={:.5}",
                i % 256, s[i], g[i], refd[i]);
        }
    }
    eprintln!("[disc] at worst element: src={:.5} grim_dec={:.5} ref_dec={:.5}",
        s[worst], g[worst], refd[worst]);

    // Arbiter: decode the failing block with the llama.cpp-faithful reference.

    let b0 = blk * 210;
    let fb = &fresh[b0..b0 + 210];
    let ql = &fb[0..128];
    let qh = &fb[128..192];
    let sc = &fb[192..208];
    let local = worst % 256;
    eprintln!("[disc] worst blk={blk} local={local}: ql[{}..]={:?} qh={:?} sc={:?}",
        (local % 64), &ql[local % 64..(local % 64) + 3], qh[local % 32], sc);
    eprintln!("[disc] src block amax={:.5} at local {}",
        s[blk * 256..blk * 256 + 256].iter().fold(0.0f32, |m, v| m.max(v.abs())),
        s[blk * 256..blk * 256 + 256].iter().enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap().0);
    // Per-group amax profile of block 0: does the source have heavy-tailed groups?
    for j in 0..16usize {
        let a: f32 = s[j * 16..j * 16 + 16].iter().fold(0.0, |m, v| m.max(v.abs()));
        let b: f32 = g[j * 16..j * 16 + 16].iter().fold(0.0, |m, v| m.max(v.abs()));
        eprintln!("[disc] blk0 group{j:2}: src_amax={a:.5} dec_amax={b:.5}");
    }
    // Scale cross-check on block 0: decode d + scales of both payloads.
    let d = |bytes: &[u8]| {
        let bits = u16::from_le_bytes([bytes[208], bytes[209]]);
        f32::from_bits(((bits as u32) & 0x8000) << 16
            | (((((bits >> 10) & 0x1F) as i32 - 15 + 127) as u32) << 23)
            | ((bits & 0x3FF) as u32) << 13)
    };
    eprintln!("[disc] blk0: src[0..4]={:?} scales_grim={:?} scales_fresh={:?} d_grim={} d_fresh={}",
        &s[0..4],
        &raw.bytes[192..208], &fresh[192..208],
        d(&raw.bytes), d(&fresh));
}
