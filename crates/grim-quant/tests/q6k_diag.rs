//! Diagnostic: compare grim dequant_q6k vs the ggml-faithful reference on a
//! single real Q6_K block to localize the residual divergence.
use std::fs::File;
use std::io::BufReader;

use grim_format::gguf::{read_gguf, read_tensor_bytes, GgufDType};
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

fn reference_q6k(data: &[u8], num_weights: usize) -> Vec<f32> {
    let nblocks = num_weights / 256;
    let mut out = Vec::with_capacity(num_weights);
    for b in 0..nblocks {
        let base = b * 210;
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 192];
        let sc = &data[base + 192..base + 208];
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
        }
    }
    out
}

#[test]
fn diag_block0() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("models").is_dir())
        .expect("repo root with models/")
        .to_path_buf();
    let path = repo_root.join("models/MiniCPM5-1B-Q4_K_M.gguf");
    let f = match File::open(&path) {
        Ok(f) => f,
        Err(_) => { eprintln!("skip: model not present"); return; }
    };
    let mut reader = BufReader::new(f);
    let file = read_gguf(&mut reader).expect("read_gguf");
    let target = file
        .tensors
        .iter()
        .find(|t| t.dtype == GgufDType::Q6K)
        .expect("at least one Q6K tensor");
    let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
    let n = target.elem_count();
    eprintln!("[diag] elem_count={n} bytes.len={}", bytes.len());

    // Block 0 only.
    let block = &bytes[0..210.min(bytes.len())];
    let a = dequant_q6k(block, 256).expect("grim");
    let b = reference_q6k(block, 256);
    eprintln!("[diag] len a={} b={}", a.len(), b.len());
    let mut max_i = 0;
    let mut max_d = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > max_d { max_d = d; max_i = i; }
    }
    eprintln!("[diag] block0 max_diff={max_d} at {max_i}");
    eprintln!("[diag] a[128..140]={:?}", &a[128..140]);
    eprintln!("[diag] b[128..140]={:?}", &b[128..140]);
    eprintln!("[diag] a[236..245]={:?}", &a[236..245.min(a.len())]);
    eprintln!("[diag] b[236..245]={:?}", &b[236..245.min(b.len())]);
}
