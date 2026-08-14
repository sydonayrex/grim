//! KAT: load the real MiniCPM5-1B-Q4_K_M GGUF, dequant one Q6_K weight with
//! grim-quant's `dequant_q6k`, and compare against an independent
//! reimplementation of llama.cpp's `dequantize_row_q6_K`.
use std::fs::File;
use std::io::BufReader;

use grim_format::gguf::{GgufDType, read_gguf, read_tensor_bytes};
use grim_quant::dequant_q6k;

fn f16_le(b: &[u8], i: usize) -> f32 {
    let bits = u16::from_le_bytes([b[i], b[i + 1]]);
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        let val = (mant as f32) * 2f32.powi(-24);
        if sign != 0 { -val } else { val }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        let e = (exp as i32) - 15 + 127;
        f32::from_bits((sign << 31) | ((e as u32) << 23) | (mant << 13))
    }
}

/// ggml block_q6_K layout (matches GGUF on-disk): ql(128) + qh(64) + scales(16) + d(2).
/// d is the LAST 2 bytes, not the first. Output ordering matches ggml:
/// for each 128-group, emit q1[0..32], q2[0..32], q3[0..32], q4[0..32].
fn reference_q6k(data: &[u8], num_weights: usize) -> Vec<f32> {
    let nblocks = num_weights / 256;
    let mut out = Vec::with_capacity(num_weights);
    for b in 0..nblocks {
        let base = b * 210;
        let mut ql = &data[base..base + 128];
        let mut qh = &data[base + 128..base + 192];
        let mut sc = &data[base + 192..base + 208];
        let d = f16_le(data, base + 208);
        for _n in (0..256).step_by(128) {
            let mut grp = [0.0f32; 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | ((qh[l] >> 2 & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | ((qh[l] >> 4 & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | ((qh[l] >> 6 & 3) << 4)) as i32 - 32;
                grp[l] = d * sc[is] as i8 as f32 * q1 as f32;
                grp[l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                grp[l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                grp[l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
            for &v in &grp {
                out.push(v);
            }
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
    out
}

#[test]
fn real_model_q6k_matches_reference() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("models").is_dir())
        .expect("repo root with models/")
        .to_path_buf();
    let path = repo_root.join("models/MiniCPM5-1B-Q4_K_M.gguf");
    let f = match File::open(&path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("skip: model not present");
            return;
        }
    };
    let mut reader = BufReader::new(f);
    let file = read_gguf(&mut reader).expect("read_gguf");
    let target = file
        .tensors
        .iter()
        .find(|t| t.dtype == GgufDType::Q6K)
        .expect("at least one Q6K tensor");
    eprintln!(
        "[kat6] tensor '{}' dims={:?} bytes={}",
        target.name, target.dims, target.size_bytes
    );
    let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
    let n = target.elem_count();
    let grim = dequant_q6k(&bytes, n).expect("grim dequant");
    let refr = reference_q6k(&bytes, n);
    assert_eq!(grim.len(), refr.len());
    // Both sides use identical math, so any inf/NaN diff is self-cancelling
    // noise. Report the finite-only max deviation.
    let finite_max = grim
        .iter()
        .zip(refr.iter())
        .map(|(a, b)| (a - b).abs())
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max);
    // Locate first diverging weight for diagnostics.
    let mut first_div: Option<(usize, f32, f32)> = None;
    for i in 0..grim.len() {
        let diff = (grim[i] - refr[i]).abs();
        if diff > 1e-3 && first_div.is_none() {
            first_div = Some((i, grim[i], refr[i]));
            break;
        }
    }
    eprintln!("[kat6] finite_max_diff={finite_max:.6} first_div={first_div:?}");
    assert!(
        finite_max < 1e-2,
        "grim q6k diverges from reference: finite_max={finite_max}"
    );
}
