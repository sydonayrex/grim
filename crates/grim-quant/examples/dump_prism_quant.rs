// Dump Rust decoder output in the same format as the C reference, so the two
// can be diffed mechanically instead of by eye.
use grim_quant::{dequant_pq2_0, dequant_ptq1_0};

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let hx = args.get(2).cloned().unwrap_or_default();
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let blk = hex(&hx);
    let v = match which {
        "pq2_0" => dequant_pq2_0(&blk, n).unwrap(),
        "ptq1_0" => dequant_ptq1_0(&blk, n).unwrap(),
        _ => panic!("codec?"),
    };
    for x in v {
        print!("{x:.6} ");
    }
    println!();
}
