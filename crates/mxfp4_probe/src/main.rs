use grim_nn::WeightSource;
use grim_tensor::dtype::{FloatPackScheme, Storage};
use grim_tensor::{DType, QuantProvenance, Shape, TensorProvider};
use std::path::Path;

fn main() {
    let model = Path::new("/drive/bigfast/grim/models/Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf");
    match probe(model) {
        Ok(()) => println!("[probe] done"),
        Err(e) => {
            eprintln!("[probe] failed: {e:#}");
            std::process::exit(1);
        }
    }
}

/// Load the Mellum2 MXFP4 GGUF and exercise the exact same code path
/// `ExpertBank::load_quantized` takes for the gate projection:
///   1. `WeightSource::get_raw_packed(name)`  -> RawTensor with provider
///      framing (MXFP4 is reframed by `GgufProvider::get` before it reaches
///      this point).
///   2. `WeightSource::get(shape, name)` -> `materialize` in varbuilder.rs,
///      which on Device::Rocm with dtype.is_quantized()==true AND
///      !Storage::GroupInt => `dev.from_cpu_bytes(&raw.bytes, ...)`
///      (resident path, NOT dequant_to_f32 host F32).
///   3. Report the resulting tensor's CPU-visible dtype + storage to confirm
///      the path actually went resident (Storage::FloatPack(MxFp4)) rather
///      than falling through to F32/Native.
fn probe(model: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if !model.is_file() {
        return Err(format!("model not found: {}", model.display()).into());
    }

    // Force GPU 1 (gfx1200, RX 9060 XT) — the same device the inference runs on.
    unsafe {
        std::env::set_var("HIP_VISIBLE_DEVICES", "1");
    }

    // Open the GGUF provider, then wrap it in a WeightSource (the same
    // construction path `Mellum::load_tp` uses).
    let prov = grim_format::tprov::GgufProvider::open(model.to_str().unwrap())?;
    // Confirm which name the provider's tensor registry actually keys by.
    match prov.meta("blk.0.ffn_gate_exps.weight") {
        Ok(_) => eprintln!("[probe] provider map keyed by raw name 'blk.0.ffn_gate_exps.weight'"),
        Err(_) => eprintln!("[probe] provider map NOT keyed by raw name 'blk.0.ffn_gate_exps.weight'"),
    }
    match prov.meta("model.layers.0.ffn_gate_exps.weight") {
        Ok(_) => eprintln!("[probe] provider map keyed by HF name 'model.layers.0.ffn_gate_exps.weight'"),
        Err(e) => eprintln!("[probe] provider map NOT keyed by HF name 'model.layers.0.ffn_gate_exps.weight' (err={e:#})"),
    }
    let ws = WeightSource::new(
        &prov,
        DType::F32,
        QuantProvenance::GrimNative,
        grim_tensor::Device::Rocm(0),
    );

    // Mellum2 non-transposed expert layout:
    //   disk rounds to [ne=64, out=inter=896, in=hidden=2304]
    //   (GGUF reader reports dims BEFORE reversal; provider reverses them,
    //   so shape() == [64, 896, 2304]).
    // ExpertBank::load (transposed_expert_layout=false) uses
    //   projections = [(gate, 896, 2304), (up, 896, 2304), (down, 2304, 896)]
    // where (..., out, in) are the per-expert dims.
    //
    // Provider's tensor registry is keyed by raw GGUF names (blk.{i}....).
    let gate_name = "blk.0.ffn_gate_exps.weight";
    let raw = ws.get_raw_packed(gate_name)?;
    eprintln!(
        "[probe] gate get_raw_packed: len(bytes)={}, shape={:?}, storage={:?}",
        raw.bytes.len(),
        raw.shape,
        raw.dtype.storage,
    );
    let per_expert_elems = 896 * 2304; // out*in per expert
    let per_expert_framed = 16 + per_expert_elems / 2 + (per_expert_elems + 31) / 32;
    eprintln!(
        "[probe] per-expert expect: elems={per_expert_elems}, framed bytes={per_expert_framed}; 64 experts = {total_expected} (raw bytes = {actual})",
        total_expected = per_expert_framed * 64,
        actual = raw.bytes.len(),
    );

    // Materialize on GPU through the same residency decision point ExpertBank
    // uses. ws.get() enforces shape equality, so we pass the FULL bank shape
    // [ne=64, out=896, in=2304] — this exercises `materialize` with
    // dtype=FloatPack(MxFp4), device=Rocm(0), which is exactly where the
    // residency branch (from_cpu_bytes) vs fallthrough (dequant_to_f32 F32)
    // decision is made. The tensor's dtype afterward tells us which path ran.
    let bank_shape = Shape::new(vec![64, 896, 2304]); // [ne, out, in] full bank
    let t = ws.get(bank_shape, gate_name)?;

    eprintln!(
        "[probe] tensor after get() on ROCm: shape={:?}, dtype={:?}, storage={:?}, elems={}",
        t.shape().dims(),
        t.dtype(),
        t.storage().dtype(),
        t.shape().elem_count(),
    );

    // == Confirm resident path vs fallthrough ==
    match t.dtype().storage {
        Storage::FloatPack(FloatPackScheme::MxFp4) => {
            eprintln!("[probe] OK: tensor dtype IS FloatPack(MxFp4) => Linear::forward -> quantized_matmul MxFp4 arm -> launch_mxfp4_gemm_tiled");
        }
        Storage::Native => {
            eprintln!("[probe] WARNING: tensor dtype is F32/Native => residency path NOT taken; expert GEMM will be f32 (release build without rocm-mem, or GroupInt exclusion, or is_quantized gating failed)");
        }
        s => {
            eprintln!("[probe] UNEXPECTED storage for expert tensor: {:?}", s);
        }
    }
    Ok(())
}
