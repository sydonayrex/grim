use grim_format::gguf::read_gguf;
use std::fs::File;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let mut f = File::open(p).unwrap();
    let g = read_gguf(&mut f).unwrap();
    // Group by layer and print the recurrent-layer tensor signature.
    for t in &g.tensors {
        let n = &t.name;
        if !(n.ends_with("attn_qkv.weight")
            || n.ends_with("ssm_conv1d.weight")
            || n.ends_with("ssm_out.weight")
            || n.ends_with("ssm_norm.weight")
            || n.ends_with("attn_gate.weight"))
        {
            continue;
        }
        let parts: Vec<&str> = n.split('.').collect();
        let layer: usize = parts[1].parse().unwrap_or(usize::MAX);
        // recurrence applies when (layer+1) % 4 != 0
        if (layer + 1) % 4 == 0 {
            continue;
        }
        println!(
            "layer {:>2} {:<24} {:?}",
            layer,
            parts[2..].join("."),
            t.shape()
        );
    }
}
