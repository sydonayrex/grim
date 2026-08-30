//! Root-cause probe (scythe2 plan validation log 2026-08-23e): does a
//! farm-style replica loaded onto Rocm(1) with both GPUs visible carry
//! *correct weight bytes*? Compares device-side dequantized values against
//! host-side `grim_quant::dequant_q4k` of the same GGUF tensor.
//!
//! Device-gated (needs ≥2 HIP devices): GRIM_GPU_TEST=1.

use grim_engine::model_loader::{load_from_path_on_device, load_model_from_gguf};
use grim_models_transformer::Lfm2;
use grim_tensor::TensorProvider;

fn host_dequant(raw: &grim_tensor::provider::RawTensor) -> Vec<f32> {
    use grim_quant::*;
    let n: usize = raw.shape.iter().product();
    let bytes = &raw.bytes;
    match &raw.dtype.storage {
        grim_tensor::Storage::KQuant(grim_tensor::KQuantScheme::Q4K) => {
            dequant_q4k(bytes, n).expect("q4k")
        }
        grim_tensor::Storage::KQuant(grim_tensor::KQuantScheme::Q6K) => {
            dequant_q6k(bytes, n).expect("q6k")
        }
        grim_tensor::Storage::KQuant(grim_tensor::KQuantScheme::Q80) => {
            dequant_q80(bytes, n).expect("q8")
        }
        other => panic!("unexpected storage {other:?}"),
    }
}

const MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/LFM2.5-230M-Q4_K_M.gguf"
);

#[test]
fn replica1_weights_match_host_dequant() {
    if std::env::var("GRIM_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }

    // Host reference for the embedding table.
    let provider = grim_format::tprov::GgufProvider::open(MODEL).expect("gguf open");
    let raw = provider.get_packed("token_embd.weight").expect("packed");
    let host = host_dequant(&raw);
    println!("host dequant head={:?} len={}", &host[..6], host.len());

    // Farm-style replica load on the SECOND visible device.
    let model = load_from_path_on_device(MODEL, grim_tensor::Device::Rocm(1))
        .expect("replica load on Rocm(1)");
    let llama = model
        .as_any()
        .downcast_ref::<Lfm2>()
        .expect("llama downcast");

    let w = &llama.tok_embeddings.weight;
    let dev_vals = w.to_vec_f32().expect("device readback");
    println!(
        "device readback head={:?} len={} nonzero={}",
        &dev_vals[..6],
        dev_vals.len(),
        dev_vals.iter().filter(|&&v| v != 0.0).count()
    );
    assert_eq!(dev_vals.len(), host.len(), "element count mismatch");

    let mut bad = 0usize;
    for (i, (&a, &b)) in dev_vals.iter().zip(host.iter()).enumerate() {
        if (a - b).abs() > 1e-3 {
            if bad < 8 {
                eprintln!("MISMATCH at {i}: device={a} host={b}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} mismatching elements in replica1 embed table");
    // The LM head is the only stage producing zeros in the rank-1 forward
    // (fwd-trace evidence); verify its weight bytes too.
    let raw_out = provider
        .get_packed("output.weight")
        .or_else(|_| provider.get_packed("token_embd.weight"))
        .expect("packed out");
    let host_out = host_dequant(&raw_out);
    let dev_out = llama.output.weight().to_vec_f32().expect("out readback");
    println!(
        "output.weight: host head={:?} device head={:?} len={}",
        &host_out[..4],
        &dev_out[..dev_out.len().min(4)],
        dev_out.len()
    );
    let mut bad_out = 0usize;
    for (i, (&a, &b)) in dev_out.iter().zip(host_out.iter()).enumerate() {
        if (a - b).abs() > 1e-3 {
            if bad_out < 8 {
                eprintln!("OUT MISMATCH at {i}: device={a} host={b}");
            }
            bad_out += 1;
        }
    }
    assert_eq!(
        bad_out, 0,
        "{bad_out} mismatching elements in output.weight"
    );
}

#[test]
fn control0_weights_match_host_dequant() {
    if std::env::var("GRIM_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("[skipped: GRIM_GPU_TEST not set]");
        return;
    }
    let provider = grim_format::tprov::GgufProvider::open(MODEL).expect("gguf open");
    let raw = provider.get_packed("token_embd.weight").expect("packed");
    let host = host_dequant(&raw);

    let model =
        load_model_from_gguf(MODEL, grim_tensor::Device::Rocm(0)).expect("control load on Rocm(0)");
    let llama = model
        .as_any()
        .downcast_ref::<Lfm2>()
        .expect("llama downcast");
    let dev_vals = llama.tok_embeddings.weight.to_vec_f32().expect("readback");
    let mut bad = 0usize;
    for (i, (&a, &b)) in dev_vals.iter().zip(host.iter()).enumerate() {
        if (a - b).abs() > 1e-3 {
            if bad < 8 {
                eprintln!("MISMATCH at {i}: device={a} host={b}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0);
}
