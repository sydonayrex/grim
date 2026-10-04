import subprocess, pathlib

W = pathlib.Path("crates/grim-models/vision/src/qwen3vl_clip/weights.rs")
T = pathlib.Path("crates/grim-models/vision/tests/qwen3vl_clip_load.rs")
ow, ot = W.read_text(), T.read_text()

# (label, file, old, new)
muts = [
    ("patch kernels: sum -> take first only", W,
     "let patch_kernel_summed: Vec<f32> = k0.iter().zip(k1.iter()).map(|(a, b)| a + b).collect();",
     "let patch_kernel_summed: Vec<f32> = k0.clone();"),
    ("patch kernels: sum -> take second only", W,
     "let patch_kernel_summed: Vec<f32> = k0.iter().zip(k1.iter()).map(|(a, b)| a + b).collect();",
     "let patch_kernel_summed: Vec<f32> = k1.clone();"),
    ("patch kernels: sum -> difference", W,
     "let patch_kernel_summed: Vec<f32> = k0.iter().zip(k1.iter()).map(|(a, b)| a + b).collect();",
     "let patch_kernel_summed: Vec<f32> = k0.iter().zip(k1.iter()).map(|(a, b)| a - b).collect();"),
    ("drop the mm.0.weight variance gate", W,
     'assert_non_degenerate("mm.0.weight", &mm_0_weight)?;',
     ""),
    ("drop the mm.2.weight variance gate", W,
     'assert_non_degenerate("mm.2.weight", &mm_2_weight)?;',
     ""),
    ("drop the position_embd variance gate", W,
     'assert_non_degenerate("v.position_embd.weight", &position_embd)?;',
     ""),
    ("drop the patch_bias variance gate", W,
     'assert_non_degenerate("v.patch_embd.bias", &patch_bias)?;',
     ""),
    ("drop a block attn_qkv variance gate", W,
     'assert_non_degenerate(&format!("{p}.attn_qkv.weight"), &block.attn_qkv_weight)?;',
     ""),
    ("drop a block ffn_down variance gate", W,
     'assert_non_degenerate(&format!("{p}.ffn_down.weight"), &block.ffn_down_weight)?;',
     ""),
    ("degenerate gate threshold -> infinity", W,
     "var.is_finite() && var > 1e-12",
     "var.is_finite()"),
    ("degenerate gate: never fires", W,
     "if !(var.is_finite() && var > 1e-12) {",
     "if false {"),
    ("degenerate gate: rejects non-empty too", W,
     "if v.is_empty() {",
     "if !v.is_empty() {"),
    ("load skip the unsupported-config gate", W,
     "cfg.ensure_supported()?;\n        let ws = WeightSource::root(provider, device);",
     "let ws = WeightSource::root(provider, device);"),
    ("merger_out_cols: projection_dim -> merger_in", W,
     "merger_out_cols: cfg.projection_dim,",
     "merger_out_cols: mi,"),
    ("block count: use 1 instead of cfg", W,
     "for i in 0..cfg.block_count {\n            let p = format!(\"v.blk.{i}\");",
     "for i in 0..1 {\n            let p = format!(\"v.blk.{i}\");"),
    ("test: weaken kernel-sum assertion to only != k0", T,
     "assert_ne!(sum, only1, \"must not be kernel B alone\");",
     "// removed"),
    ("test: degenerate gate accepts zeros", T,
     'assert!(assert_non_degenerate("fake.zero", &zeros).is_err());',
     'assert!(assert_non_degenerate("fake.zero", &zeros).is_ok());'),
]

for label, path, a, b in muts:
    orig = path.read_text()
    if a not in orig:
        print("SKIP     " + label)
        continue
    path.write_text(orig.replace(a, b, 1))
    r = subprocess.run(
        ["cargo", "test", "-p", "grim-models-vision", "--test", "qwen3vl_clip_load"],
        capture_output=True, text=True, timeout=900)
    killed = "FAILED" in r.stdout or r.returncode != 0
    print(("KILLED   " if killed else "SURVIVED ") + label)
    path.write_text(orig)

assert W.read_text() == ow and T.read_text() == ot, "restoration mismatch!"
print("files restored")