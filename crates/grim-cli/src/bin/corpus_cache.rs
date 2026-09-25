//! S4 helper: build the `<corpus>.tokens.u32` cache for a training corpus
//! without loading any models. Uses the SAME `load_corpus_tokens` path as
//! the trainers, so the cache is byte-identical to what a training run
//! would build. Usage: corpus_cache <corpus.txt> <tokenizer.gguf>

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: corpus_cache <corpus.txt> <tokenizer.gguf>");
        std::process::exit(2);
    }
    let tokens = grim_cli::distill::load_corpus_tokens(&args[1], &args[2]).unwrap_or_else(|e| {
        eprintln!("[corpus_cache] error: {e}");
        std::process::exit(1);
    });
    println!(
        "[corpus_cache] {} tokens cached for {}",
        tokens.len(),
        args[1]
    );
}
