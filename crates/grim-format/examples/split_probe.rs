// Direct probe: what does SplitGgufProvider actually resolve for this model?
use grim_format::tprov::{GgufProvider, SplitGgufProvider};
use grim_tensor::provider::TensorProvider;

fn main() {
    let main_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/QWen38-Flash/Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf"
    );
    let p = GgufProvider::open(main_path).expect("primary opens");
    println!(
        "primary split.count = {:?}",
        p.metadata("split.count").and_then(|v| v.as_u32())
    );
    println!(
        "primary split.no    = {:?}",
        p.metadata("split.no").and_then(|v| v.as_u32())
    );
    println!("primary arch        = {:?}", p.architecture());
    println!("primary n_tensors   = {}", p.tensor_names().len());
    println!(
        "has token_embd      = {}",
        p.tensor_names().iter().any(|n| n == "token_embd")
    );

    // Which shard holds what?
    let shard2 = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/QWen38-Flash/Qwen3.8-Flash-Next-ngram-embeddings-Q4_0.gguf"
    );
    let s2 = GgufProvider::open(shard2).expect("shard2 opens");
    println!("shard2 n_tensors = {}", s2.tensor_names().len());
    for n in s2.tensor_names().iter() {
        println!("   shard2 tensor: {n}");
    }
    println!(
        "shard2 arch = {:?} split.no = {:?}",
        s2.architecture(),
        s2.metadata("split.no").and_then(|v| v.as_u32())
    );
    println!("--- primary tensor names sample:");
    for n in p.tensor_names().iter().take(3) {
        println!("   {n}");
    }

    // What shape does the split report for the PLE table?
    let s3 = SplitGgufProvider::open(main_path).expect("split opens");
    for n in ["per_layer_token_embd.weight", "token_embd.weight"] {
        match s3.meta(n) {
            Ok(m) => println!("  split meta {n} = shape {:?} dtype {:?}", m.shape, m.dtype),
            Err(e) => println!("  split meta {n} FAILED: {e}"),
        }
    }

    match SplitGgufProvider::open(main_path) {
        Ok(s) => {
            println!("split resolved: {} shards", s.tensor_names().len());
            for n in [
                "token_embd.weight",
                "per_layer_token_embd.weight",
                "token_embd",
                "output",
                "blk.0.ssm_a",
                "per_layer_token_embd",
            ] {
                println!(
                    "  split has {n:24} = {}",
                    s.tensor_names().iter().any(|x| x == n)
                );
            }
        }
        Err(e) => println!("split FAILED: {e}"),
    }
}
