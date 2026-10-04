//! Xing4.0 localization gate: the per-forward `grim_dequant_iq3s` /
//! `grim_dequant_q4k` kernels must match the CPU oracle bit-for-bit on REAL
//! checkpoint bytes (blk.2 of the Xing4.0-29B IQ3_M GGUF). gfx1200 has a
//! hipRTC miscompile precedent (LFM2 scan kernel), so synthetic-byte tests
//! are not enough.

use grim_backend_rocm::RocmDevice;
use std::io::{Read, Seek, SeekFrom};

fn load_tensor_bytes(path: &str, name: &str) -> (Vec<u8>, Vec<usize>, String) {
    let file = std::fs::File::open(path).unwrap();
    let gg = grim_format::gguf::read_gguf(&file).expect("read_gguf");
    let t = gg
        .tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("tensor {name} not found"));
    let mut f = std::fs::File::open(path).unwrap();
    f.seek(SeekFrom::Start(gg.data_start + t.offset)).unwrap();
    let mut raw = vec![0u8; t.size_bytes as usize];
    f.read_exact(&mut raw).unwrap();
    (raw, t.shape().to_vec(), format!("{:?}", t.dtype))
}

#[test]
#[ignore = "requires real ROCm device + Xing4.0 GGUF; GRIM_RUN_GPU_TESTS=1 -- --ignored"]
fn iq3s_and_q4k_device_dequant_match_cpu_oracle_on_checkpoint_bytes() {
    let path = match std::env::var("XING_GGUF") {
        Ok(p) => p,
        Err(_) => return, // gate: only runs when pointed at the checkpoint
    };
    let dev = RocmDevice::shared(0);

    // attn_q_a: IQ3_S [768, 3584]
    let (bytes, shape, dtype) = load_tensor_bytes(&path, "blk.2.attn_q_a.weight");
    assert!(dtype.contains("IQ3_S"), "got {dtype}");
    let n = shape.iter().product::<usize>();
    let gpu = dev.dequantize_iq3s_host(&bytes, n).expect("device iq3s");
    let cpu = grim_quant::dequant_iq3s(&bytes, n).expect("cpu iq3s");
    let diff: f32 = gpu
        .iter()
        .zip(&cpu)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0, f32::max);
    eprintln!("[xing-dequant] IQ3_S n={n}: max|gpu-cpu| = {diff:.3e}");
    assert!(diff < 1e-4, "IQ3_S device dequant diverges: max {diff:.3e}");

    // THE GIANTS — the tensors the converter's first run faulted on
    // (hipErrorLaunchFailure 719 at ~20-30/977 completions): every tensor
    // that could have been in flight at the fault is >= 234M elements, and
    // the small-tensor gate above passed. Three schemes at giant size.
    for (name, expect, dequant) in [
        ("token_embd.weight", "IQ3_S", 0),
        ("blk.2.ffn_gate_exps.weight", "IQ3_S", 0),
        ("blk.2.ffn_down_exps.weight", "Q4K", 1),
    ] {
        let (bytes, shape, dtype) = load_tensor_bytes(&path, name);
        assert!(dtype.contains(expect), "{name}: got {dtype}");
        let n = shape.iter().product::<usize>();
        eprintln!(
            "[xing-dequant] GIANT {name}: n={n} bytes={}",
            bytes.len()
        );
        let gpu = match dequant {
            0 => dev.dequantize_iq3s_host(&bytes, n).expect("device iq3s giant"),
            _ => dev.dequantize_q4k_host(&bytes, n).expect("device q4k giant"),
        };
        let cpu = match dequant {
            0 => grim_quant::dequant_iq3s(&bytes, n).expect("cpu iq3s giant"),
            _ => grim_quant::dequant_q4k(&bytes, n).expect("cpu q4k giant"),
        };
        let diff: f32 = gpu
            .iter()
            .zip(&cpu)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0, f32::max);
        eprintln!("[xing-dequant] GIANT {name}: max|gpu-cpu| = {diff:.3e}");
        assert!(diff < 1e-4, "{name} device dequant diverges: max {diff:.3e}");
    }

    // CUMULATION discriminator: serially dequant ~40 varied tensors in this
    // one process. If the conversion fault is cumulative (per-launch resource
    // growth), this faults near the conversion's count; if it stays green,
    // the fault is specific to the conversion's context/stream usage.
    let names: Vec<(String, grim_format::gguf::GgufDType, Vec<usize>)> = {
        let file = std::fs::File::open(&path).unwrap();
        let gg = grim_format::gguf::read_gguf(&file).unwrap();
        gg.tensors
            .iter()
            .take(40)
            .map(|t| (t.name.clone(), t.dtype, t.shape().to_vec()))
            .collect()
    };
    for (i, (name, gdtype, shape)) in names.iter().enumerate() {
        let (bytes, _s, _d) = load_tensor_bytes(&path, name);
        let n: usize = shape.iter().product();
        let r = match gdtype {
            grim_format::gguf::GgufDType::IQ3_S => {
                dev.dequantize_iq3s_host(&bytes, n).map(|v| v.len())
            }
            grim_format::gguf::GgufDType::Q4K => {
                dev.dequantize_q4k_host(&bytes, n).map(|v| v.len())
            }
            grim_format::gguf::GgufDType::Q6K => Ok(0), // not on the GPU path
            _ => Ok(n),                                 // F32/F16 - not dispatched here
        };
        match &r {
            Ok(len) => eprintln!("[xing-cumulate] #{:02} {} {:?} n={} -> ok len={}", i, name, gdtype, n, len),
            Err(e) => eprintln!("[xing-cumulate] #{:02} {} {:?} n={} -> ERR {}", i, name, gdtype, n, e),
        }
        r.unwrap_or_else(|e| panic!("#{} {} faulted: {}", i, name, e));
    }

    // attn_output: Q4_K [3584, 4096]
    let (bytes, shape, dtype) = load_tensor_bytes(&path, "blk.2.attn_output.weight");
    assert!(dtype.contains("Q4K"), "got {dtype}");
    let n = shape.iter().product::<usize>();
    let gpu = dev.dequantize_q4k_host(&bytes, n).expect("device q4k");
    let cpu = grim_quant::dequant_q4k(&bytes, n).expect("cpu q4k");
    let diff: f32 = gpu
        .iter()
        .zip(&cpu)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0, f32::max);
    eprintln!("[xing-dequant] Q4K n={n}: max|gpu-cpu| = {diff:.3e}");
    assert!(diff < 1e-4, "Q4K device dequant diverges: max {diff:.3e}");
}
