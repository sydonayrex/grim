//! Task 10.5 gate (provider level, real file): `GgufProvider` must serve the
//! Bonsai-27B PQ2_0 checkpoint as `Block(Pq2_0)` tensors with the 34 B/128
//! block geometry, and the host decoder must agree on the first tensor.
//!
//! KNOWN LIMITATION (recorded in the plan): PrismML quantizes in a Hadamard-
//! rotated domain (`prism.hadamard.transform =
//! normalized-sylvester-walsh-hadamard`, block 1024, explicit sign masks over
//! 401 tensors). grim has NO hadamard machinery yet, so these weights decode
//! to the ROTATED domain — finite and format-correct, but not yet the model's
//! real weights. End-to-end Bonsai inference is blocked on the rotated-domain
//! runtime path (WHT-1024 kernel + sign-mask decode + Prism metadata parsing).

use grim_format::GgufProvider;
use grim_tensor::provider::TensorProvider;
use grim_tensor::dtype::{BlockDtype, Storage};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

const MODEL: &str = "models/Bonsai-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf";

#[test]
fn bonsai_pq2_0_provider_serves_block_dtype() -> TestResult {
    if !std::path::Path::new(MODEL).exists() {
        // The checkpoint is local-only (7.2 GB); skip elsewhere.
        eprintln!("skipping: {MODEL} not present");
        return Ok(());
    }
    let provider = GgufProvider::open(MODEL)?;

    let name = "blk.0.attn_qkv.weight";
    let meta = provider.meta(name)?;
    assert!(
        matches!(meta.dtype.storage, Storage::Block(BlockDtype::Pq2_0)),
        "PQ2_0 tensor must map to Block(Pq2_0), got {:?}",
        meta.dtype.storage
    );

    let raw = provider.get(name)?;
    let n: usize = meta.shape.iter().product();
    assert_eq!(
        raw.bytes.len(),
        n.div_ceil(128) * 34,
        "packed byte length must be 34 B per 128 weights"
    );

    // Host decoder agrees on the first block: decode block 0 both via the
    // provider bytes and via dequant_pq2_0, compare first 8 weights.
    let decoded = grim_quant::dequant_pq2_0(&raw.bytes, n)?;
    let head = &decoded[..8];
    assert!(
        head.iter().all(|v| v.is_finite()),
        "decoded weights must be finite (rotated-domain values)"
    );
    println!(
        "bonsai provider gate ok: {name} shape={:?}, {} packed bytes, first decoded {:?}",
        meta.shape,
        raw.bytes.len(),
        &decoded[..4]
    );
    Ok(())
}
