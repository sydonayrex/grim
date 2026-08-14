use grim_format::gguf::read_gguf;
use std::fs::File;
fn main() {
    let mut f = File::open(std::env::args().nth(1).unwrap()).unwrap();
    let g = read_gguf(&mut f).unwrap();
    for (k, v) in &g.metadata {
        if k.contains("ssm")
            || k.contains("d_state")
            || k.contains("d_inner")
            || k.contains("d_conv")
            || k.contains("dt_rank")
            || k.contains("n_group")
            || k.contains("intermediate")
            || k.contains("hidden")
            || k.contains("head")
        {
            println!("{} = {:?}", k, v);
        }
    }
}
