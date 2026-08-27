//! KAT: load the real MiniCPM5-1B-Q4_K_M GGUF, dequant one Q4_K weight with
//! grim-quant's `dequant_q4k`, and compare against an independent
//! reimplementation of llama.cpp's `dequantize_row_q4_K`. Isolation test for the
//! Q4_K garbage bug.

use std::fs::File;
use std::io::BufReader;

use grim_format::gguf::{GgufDType, read_gguf, read_tensor_bytes};
use grim_quant::dequant_q4k;

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

fn reference_q4k(data: &[u8], num_weights: usize) -> Vec<f32> {
    fn get_scale_min(scales: &[u8], j: usize) -> (f32, f32) {
        let (sc, m) = if j < 4 {
            (scales[j] & 63, scales[j + 4] & 63)
        } else {
            (
                (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
            )
        };
        (sc as f32, m as f32)
    }
    let mut out = Vec::with_capacity(num_weights);
    let mut pos = 0;
    let nblocks = num_weights.div_ceil(256);
    for _ in 0..nblocks {
        let d = f16_le(data, pos);
        let dmin = f16_le(data, pos + 2);
        let scales = &data[pos + 4..pos + 16];
        let qs = &data[pos + 16..pos + 144];
        let mut q = 0usize;
        let mut is = 0usize;
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min(scales, is);
            let (sc2, m2) = get_scale_min(scales, is + 1);
            let d1 = d * sc1;
            let d2 = d * sc2;
            let m1v = dmin * m1;
            let m2v = dmin * m2;
            for l in 0..32 {
                if out.len() < num_weights {
                    out.push(d1 * (qs[q + l] & 0x0F) as f32 - m1v);
                }
            }
            for l in 0..32 {
                if out.len() < num_weights {
                    out.push(d2 * (qs[q + l] >> 4) as f32 - m2v);
                }
            }
            q += 32;
            is += 2;
        }
        pos += 144;
    }
    out
}

#[test]
fn real_model_q4k_matches_reference() {
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
        .find(|t| t.dtype == GgufDType::Q4K)
        .expect("at least one Q4K tensor");

    eprintln!(
        "[kat] tensor '{}' dims={:?} dtype={:?} bytes={}",
        target.name, target.dims, target.dtype, target.size_bytes
    );

    let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
    let n = target.elem_count();
    eprintln!("[kat] elem_count={n} packed_bytes={}", bytes.len());

    let expected_blocks = n.div_ceil(256);
    assert_eq!(
        expected_blocks * 144,
        bytes.len(),
        "block-count/size mismatch: n={n} bytes={}",
        bytes.len()
    );

    let grim = dequant_q4k(&bytes, n).expect("grim dequant");
    let refr = reference_q4k(&bytes, n);

    assert_eq!(grim.len(), refr.len());
    let mut max_abs = 0.0f32;
    let mut first_mismatch = None;
    for i in 0..grim.len() {
        let diff = (grim[i] - refr[i]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if diff > 1e-3 && first_mismatch.is_none() {
            first_mismatch = Some((i, grim[i], refr[i]));
        }
    }
    eprintln!("[kat] max_abs_diff={max_abs:.6} first_mismatch={first_mismatch:?}",);
    eprintln!(
        "[kat] grim first8={:?}",
        grim.iter().take(8).collect::<Vec<_>>()
    );
    eprintln!(
        "[kat] ref  first8={:?}",
        refr.iter().take(8).collect::<Vec<_>>()
    );
    assert!(
        max_abs < 1e-2,
        "grim q4k diverges from reference: max_abs={max_abs}"
    );
}
