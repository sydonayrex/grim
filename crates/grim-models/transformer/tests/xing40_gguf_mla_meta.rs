//! Metadata probe for the real Xing4.0 GGUF: reads *only* the header.
//!
//! `GgufProvider::open` memory-maps the file read-only and slices tensors on
//! demand, so this never materializes the 20 GB of tensor payload. It exists to
//! pin the authoritative shape/dtype of the split MLA up-projections that
//! `Xing40Mla` reassembles into a kernel-facing `kv_b_proj`.
//!
//! Run explicitly — it touches a path outside the repo:
//! `cargo test -p grim-models-transformer --test xing40_gguf_mla_meta -- --ignored --nocapture`

use grim_format::tprov::GgufProvider;

/// Checkpoint to probe. Override with `XING40_GGUF=/path/to.gguf`.
fn gguf_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("XING40_GGUF") {
        return p.into();
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models/xing4_0-29b-IQ4_NL.gguf")
}

#[test]
#[ignore = "reads a real checkpoint header; run explicitly"]
fn probe_split_mla_up_projection_metadata() {
    let path = gguf_path();
    // The provider now REFUSES this file (see the guard test below), so read the
    // header through the low-level parser for the shape/dtype census.
    use grim_format::gguf::read_gguf;
    use std::io::BufReader;
    let file = std::fs::File::open(&path).expect("open gguf");
    let g = read_gguf(BufReader::new(file)).expect("read gguf header");
    println!("probing {}", path.display());
    // GGUF stores `dims` in the reverse of the provider's `shape()` order; the
    // provider reverses them on read, so do the same here to keep this probe's
    // expectations in the same frame the loader uses.
    let meta_of = |name: &str| -> Option<(Vec<usize>, String)> {
        g.tensors.iter().find(|t| t.name == name).map(|t| {
            let mut dims: Vec<usize> = t.dims.iter().map(|d| *d as usize).collect();
            dims.reverse();
            (dims, format!("{:?}", t.dtype))
        })
    };

    for name in [
        "blk.0.attn_k_b.weight",
        "blk.0.attn_v_b.weight",
        "blk.0.attn_kv_a_mqa.weight",
        "blk.0.attn_q_b.weight",
        "blk.0.attn_q_a.weight",
        "blk.0.attn_output.weight",
        "blk.2.ffn_gate_exps.weight",
        "blk.2.ffn_up_exps.weight",
        "blk.2.ffn_down_exps.weight",
        "blk.2.ffn_gate_inp.weight",
        "token_embd.weight",
    ] {
        match meta_of(name) {
            Some((dims, dt)) => println!("{name}: shape={dims:?} dtype={dt}"),
            None => println!("{name}: MISSING"),
        }
    }

    // Full dtype census from the raw header: what is actually resident.
    let mut census: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
    for t in &g.tensors {
        let n: u64 = t.dims.iter().map(|d| *d as u64).product();
        let e = census.entry(format!("{:?}", t.dtype)).or_insert((0, 0));
        e.0 += 1;
        e.1 += n;
    }
    println!("--- {} tensors ---", g.tensors.len());
    for (k, (cnt, elems)) in &census {
        println!("  {cnt:>5} tensors {elems:>14} elems  {k}");
    }

    // The shipped residency guard: opening this checkpoint must be REFUSED
    // because its dtype tags contradict the payload, and the error must name the
    // bulk FFN tensors.
    let open = GgufProvider::open(path.to_str().expect("utf-8 gguf path"));
    match open {
        Ok(_) => panic!(
            "checkpoint opened, but its bulk FFN tensors are tagged F64 with 4-bit \
             payloads — loading would reinterpret packed bytes as floats"
        ),
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("dtype/payload mismatch"), "msg: {msg}");
            assert!(msg.contains("ffn_gate.weight"), "msg: {msg}");
            assert!(msg.contains("IQ4_NL"), "msg: {msg} must name candidate formats");
            println!("refusal (first lines):\n{}", msg.lines().take(4).collect::<Vec<_>>().join("\n"));
        }
    }

    // The reassembly in `Xing40Mla::build_split_kv_b_linear` assumes:
    //   attn_k_b [num_heads, kv_lora_rank, qk_nope_head_dim]
    //   attn_v_b [num_heads, v_head_dim,      kv_lora_rank]
    // and produces the kernel-facing [num_heads * (nope + v_head), rank].
    let (k_shape, k_dt) = meta_of("blk.0.attn_k_b.weight").expect("attn_k_b present");
    let (v_shape, v_dt) = meta_of("blk.0.attn_v_b.weight").expect("attn_v_b present");
    let (k, v) = (k_shape, v_shape);
    println!("attn_k_b dtype={k_dt} attn_v_b dtype={v_dt}");
    assert_eq!(k.len(), 3, "attn_k_b must be 3-D [heads, rank, nope]");
    assert_eq!(v.len(), 3, "attn_v_b must be 3-D [heads, v_head, rank]");

    let (heads, rank, nope) = (k[0], k[1], k[2]);
    let (v_heads, v_dim, v_rank) = (v[0], v[1], v[2]);
    assert_eq!(v_heads, heads, "attn_k_b / attn_v_b head count mismatch");
    assert_eq!(v_rank, rank, "attn_k_b / attn_v_b rank mismatch");
    assert_eq!(heads * rank * nope, heads * v_dim * rank, "bank sizes disagree");

    println!(
        "reassembled kv_b_proj.weight = [{heads} * ({nope} + {v_dim}), {rank}] \
         = [{}, {rank}] ({} floats)",
        heads * (nope + v_dim),
        heads * (nope + v_dim) * rank
    );
    // If the two banks disagree in per-head layout, the concatenation above is
    // wrong and the kernel would read transposed weights.
    assert_eq!(
        nope, v_dim,
        "unexpected: attn_k_b last dim ({nope}) != attn_v_b first dim ({v_dim}); \
         verify the per-head transpose in build_split_kv_b_linear"
    );
}
