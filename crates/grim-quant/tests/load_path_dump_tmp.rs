//! LOAD-PATH dump: for each tensor, what f32 does GrimProvider serve vs
//! what the same dequant of GgufProvider bytes gives? The FIRST diverging
//! tensor names the .grim serving defect. Env: XING_GRIM + XING_GGUF.

use grim_format::tprov::GrimProvider;
use grim_tensor::provider::TensorProvider;
use std::io::{Read, Seek, Write};

fn deq(scheme: &str, bytes: &[u8], n: usize) -> Option<Vec<f32>> {
    match scheme {
        "Q4K" => grim_quant::dequant_q4k(bytes, n).ok(),
        "Q6K" => grim_quant::dequant_q6k(bytes, n).ok(),
        "Q5K" => grim_quant::dequant_q5k(bytes, n).ok(),
        "Q2K" => grim_quant::dequant_q2k(bytes, n).ok(),
        "Q3K" => grim_quant::dequant_q3k(bytes, n).ok(),
        "Q8_0" => grim_quant::dequant_q80(bytes, n).ok(),
        "IQ3S" => grim_quant::dequant_iq3s(bytes, n).ok(),
        "IQ4NL" => grim_quant::dequant_iq4nl(bytes, n).ok(),
        _ => None,
    }
}

#[test]
fn load_path_dump() {
    let Ok(grim_path) = std::env::var("XING_GRIM") else {
        eprintln!("[SKIP] XING_GRIM unset");
        return;
    };
    let gguf_path = std::env::var("XING_GGUF").unwrap();

    let gp = GrimProvider::open(&grim_path).unwrap();
    let gguf_f = std::fs::File::open(&gguf_path).unwrap();
    let gg = grim_format::gguf::read_gguf(std::io::BufReader::new(&gguf_f)).unwrap();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut checked = 0usize;
    let mut mismatched = 0usize;

    let mut sorted: Vec<String> = gg.tensors.iter().map(|t| t.name.clone()).collect();
    sorted.sort();

    for name in &sorted {
        let t = gg.tensors.iter().find(|t| t.name == *name).unwrap();
        let n: usize = t.shape().iter().product();
        // Source GGUF bytes + dequant
        let mut f2 = std::fs::File::open(&gguf_path).unwrap();
        f2.seek(std::io::SeekFrom::Start(gg.data_start + t.offset)).unwrap();
        let mut src_bytes = vec![0u8; t.size_bytes as usize];
        f2.read_exact(&mut src_bytes).unwrap();
        let src_name = format!("{:?}", t.dtype);
        let scheme = src_name.trim_start_matches("Q").to_string();
        let _ = scheme;
        let src_scheme = match t.dtype {
            grim_format::gguf::GgufDType::Q4K => "Q4K",
            grim_format::gguf::GgufDType::Q6K => "Q6K",
            grim_format::gguf::GgufDType::Q5K => "Q5K",
            grim_format::gguf::GgufDType::Q2K => "Q2K",
            grim_format::gguf::GgufDType::Q3K => "Q3K",
            grim_format::gguf::GgufDType::Q8_0 => "Q8_0",
            grim_format::gguf::GgufDType::IQ3_S => "IQ3S",
            grim_format::gguf::GgufDType::IQ4_NL => "IQ4NL",
            _ => "",
        };
        let src = deq(src_scheme, &src_bytes, n);

        // .grim side: what does GrimProvider serve?
        let raw = gp.get_packed(name).unwrap();
        let grim_name = match &raw.dtype.storage {
            grim_tensor::dtype::Storage::KQuant(k) => format!("{k:?}"),
            grim_tensor::dtype::Storage::Native => "F32".to_string(),
            other => format!("{other:?}"),
        };
        let grim = deq(&grim_name, &raw.bytes, n);

        match (&src, &grim) {
            (Some(s), Some(g)) if s.len() == g.len() => {
                let max_d: f32 = g.iter().zip(s).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                let amax: f32 = s.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let nan = g.iter().any(|v| v.is_nan());
                checked += 1;
                let rel = max_d / amax.max(1e-9);
                if nan || rel > 0.2 {
                    mismatched += 1;
                    let _ = writeln!(out, "MISMATCH {name}: src={src_name} grim={grim_name} max_d={max_d:.4} amax={amax:.4} rel={rel:.3} nan={nan}");
                }
            }
            _ => {
                let _ = writeln!(out, "SKIP {name}: src_scheme={src_scheme} grim_scheme={grim_name}");
            }
        }
    }
    let _ = writeln!(out, "checked={checked} mismatched={mismatched}");
    let _ = stdout;
}
