use grim_format::gguf::read_gguf;
use std::fs::File;
use std::io::BufReader;

#[test]
fn scan_q4k_model_dtypes() {
    let path = "/drive/bigfast/grim/models/Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf";
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("skip: model file {path} not present");
            return;
        }
    };
    let mut reader = BufReader::new(file);
    let file = read_gguf(&mut reader).expect("failed to parse GGUF");
    println!("Metadata:");
    for (k, v) in &file.metadata {
        if !k.contains("tokens") && !k.contains("merges") {
            println!("  {k} = {:?}", v);
        }
    }
    println!("Total tensors: {}", file.tensors.len());
    let total_bytes: u64 = file.tensors.iter().map(|t| t.size_bytes).sum();
    let file_len = std::fs::metadata(path).unwrap().len();
    println!(
        "data_start={}, sum(size_bytes)={}, sum+data_start={}, actual file len={}, diff={}",
        file.data_start,
        total_bytes,
        file.data_start + total_bytes,
        file_len,
        file_len as i64 - (file.data_start + total_bytes) as i64
    );
    for t in &file.tensors {
        if t.name.contains("blk.0.") || t.name.contains("token_embd") {
            let n: usize = t.dims.iter().map(|&d| d as usize).product();
            println!(
                "  {}: dims={:?}, dtype={:?}, size_bytes={}, elems={} (bytes/elem={:.3})",
                t.name,
                t.dims,
                t.dtype,
                t.size_bytes,
                n,
                t.size_bytes as f64 / n as f64
            );
        }
    }
}
