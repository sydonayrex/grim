//! WI-E3 measure-first micro-bench: how long does `encode` take on the eval
//! corpus? Gate: if >= 50 ms, the parallel fast path is worth building.


fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/LFM2.5-350M-Q8_0.gguf".to_string());
    let corpus = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "docs/eval/wikitext2.sample.txt".to_string());

    let prov = grim_format::GgufProvider::open(&path).expect("open gguf");
    let tok = prov.tokenizer().expect("tokenizer");
    let text = std::fs::read_to_string(&corpus).expect("corpus");
    println!(
        "corpus: {} bytes, tokenizer model_type={}",
        text.len(),
        tok.model_type
    );

    // Warmup (page in maps).
    let _ = tok.encode(&text[..text.len() / 10]);

    // Scaling probe: time prefixes to expose quadratic behavior.
    for frac in [0.05f64, 0.1, 0.25, 0.5, 1.0] {
        let n = (text.len() as f64 * frac) as usize;
        let t = std::time::Instant::now();
        let _ = tok.encode(&text[..n]);
        println!("  prefix {frac:.2} ({} bytes): {:?}", n, t.elapsed());
    }

    let t0 = std::time::Instant::now();
    let ids = tok.encode(&text);
    let dt = t0.elapsed();
    println!(
        "encode: {:?} for {} tokens = {:.1} MB/s",
        dt,
        ids.len(),
        (text.len() as f64 / 1e6) / dt.as_secs_f64()
    );
    if dt.as_millis() >= 50 {
        println!("VERDICT: HOT (>= 50 ms) — parallel path justified");
    } else {
        println!("VERDICT: NOT HOT (< 50 ms) — skip WI-E3 fast path");
    }
}
