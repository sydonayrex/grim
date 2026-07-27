GRIM_RUN_GPU_TESTS=1 RUST_BACKTRACE=1 cargo test -p grim-engine --test sleipnir_rocm_inference -- --nocapture
   Compiling grim-nn v0.1.0 (/D/rex/projects/grim/crates/grim-nn)
   Compiling grim-core v0.1.0 (/D/rex/projects/grim/crates/grim-core)
   Compiling grim-kvtransport v0.1.0 (/D/rex/projects/grim/crates/grim-kvtransport)
   Compiling grim-kvquant v0.1.0 (/D/rex/projects/grim/crates/grim-kvquant)
   Compiling grim-models-transformer v0.1.0 (/D/rex/projects/grim/crates/grim-models/transformer)
   Compiling grim-plugin v0.1.0 (/D/rex/projects/grim/crates/grim-plugin)
   Compiling grim-models-vision v0.1.0 (/D/rex/projects/grim/crates/grim-models/vision)
   Compiling grim-memory v0.1.0 (/D/rex/projects/grim/crates/grim-memory)
   Compiling grim-scheduler v0.1.0 (/D/rex/projects/grim/crates/grim-scheduler)
warning: unused import: `std::sync::Arc`
 --> crates/grim-models/transformer/src/gpt2.rs:3:5
  |
3 | use std::sync::Arc;
  |     ^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

warning: unused import: `std::sync::Arc`
 --> crates/grim-models/transformer/src/gemma.rs:3:5
  |
3 | use std::sync::Arc;
  |     ^^^^^^^^^^^^^^

warning: unused import: `std::sync::Arc`
 --> crates/grim-models/transformer/src/deepseek.rs:3:5
  |
3 | use std::sync::Arc;
  |     ^^^^^^^^^^^^^^

warning: unused import: `std::sync::Arc`
 --> crates/grim-models/transformer/src/t5.rs:3:5
  |
3 | use std::sync::Arc;
  |     ^^^^^^^^^^^^^^

warning: unused import: `DType`
 --> crates/grim-models/transformer/src/t5.rs:9:38
  |
9 | use grim_tensor::{ArithType, Device, DType, Tensor};
  |                                      ^^^^^

   Compiling grim-models-mamba v0.1.0 (/D/rex/projects/grim/crates/grim-models/mamba)
warning: unused import: `std::sync::Arc`
 --> crates/grim-models/mamba/src/rwkv.rs:4:5
  |
4 | use std::sync::Arc;
  |     ^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

warning: unused import: `DType`
 --> crates/grim-models/mamba/src/rwkv.rs:9:38
  |
9 | use grim_tensor::{ArithType, Device, DType, Tensor};
  |                                      ^^^^^

   Compiling grim-speculative v0.1.0 (/D/rex/projects/grim/crates/grim-speculative)
warning: `grim-models-mamba` (lib) generated 2 warnings (run `cargo fix --lib -p grim-models-mamba` to apply 2 suggestions)
warning: `grim-models-transformer` (lib) generated 5 warnings (run `cargo fix --lib -p grim-models-transformer` to apply 5 suggestions)
   Compiling grim-engine v0.1.0 (/D/rex/projects/grim/crates/grim-engine)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.68s
     Running tests/sleipnir_rocm_inference.rs (target/debug/deps/sleipnir_rocm_inference-04877edece50e547)

running 5 tests
test sleipnir_gguf_metadata_contract ... ok
test sleipnir_gguf_tokenizer_output_clean ... ok
[grim] Loading config: architecture=Lfm2, layers=16, hidden=1024, vocab=65536
[grim] LFM2 layer-type map (T=shortconv): [true, true, false, true, true, false, true, true, false, true, false, true, false, true, false, true]
[grim] Loading config: architecture=Lfm2, layers=16, hidden=1024, vocab=65536
[grim] LFM2 layer-type map (T=shortconv): [true, true, false, true, true, false, true, true, false, true, false, true, false, true, false, true]
[grim] Loading config: architecture=Lfm2, layers=16, hidden=1024, vocab=65536
[grim] LFM2 layer-type map (T=shortconv): [true, true, false, true, true, false, true, true, false, true, false, true, false, true, false, true]
[Embedding::load] weight is quantized: dtype=DType { arith: F32, storage: KQuant(Q80) }, device=Rocm(0)
[Embedding::load] weight is quantized: dtype=DType { arith: F32, storage: KQuant(Q80) }, device=Rocm(0)
[Embedding::load] dequantized to f32, len=67108864
[Embedding::load] dequantized to f32, len=67108864
[Embedding::load] weight is quantized: dtype=DType { arith: F32, storage: KQuant(Q80) }, device=Rocm(0)
[Embedding::load] dequantized to f32, len=67108864
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[512,1024] got weight.shape=[512, 1024] w_t.shape=[512, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[3072,1024] got weight.shape=[3072, 1024] w_t.shape=[3072, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,1024] got weight.shape=[1024, 1024] w_t.shape=[1024, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[4608,1024] got weight.shape=[4608, 1024] w_t.shape=[4608, 1024] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
[Linear::load TEMP] requested=[1024,4608] got weight.shape=[1024, 4608] w_t.shape=[1024, 4608] dtype=DType { arith: F32, storage: KQuant(Q80) }
test sleipnir_gguf_loads_on_target_device ... ok
[lfm2-forward] seq_len=12, input_ids.device=Rocm(0)
[roc-embedding] weight.device_ptr=Some(140239251177472), weight.device_ptr_is_valid=true, indices.len=12, out_shape=[12, 1024]
[roc-embedding] allocating output: total=12288, shape=[12, 1024]
[roc-embedding] launching kernel: grid=HipDim3 { x: 48, y: 1, z: 1 }, block=HipDim3 { x: 256, y: 1, z: 1 }, w_ptr=140239251177472, out_ptr=140238881034240, idx_ptr=0x7f8be8b11000
[lfm2-forward] seq_len=12, input_ids.device=Rocm(0)
[roc-embedding] weight.device_ptr=Some(140239534292992), weight.device_ptr_is_valid=true, indices.len=12, out_shape=[12, 1024]
[roc-embedding] allocating output: total=12288, shape=[12, 1024]
[roc-embedding] launching kernel: grid=HipDim3 { x: 48, y: 1, z: 1 }, block=HipDim3 { x: 256, y: 1, z: 1 }, w_ptr=140239534292992, out_ptr=140238881112064, idx_ptr=0x7f8be8b24000
[lfm2-forward] after tok_embeddings, h.device=Rocm(0), h.shape=[12, 1024]
[lfm2-forward] before layer loop
[lfm2-forward] layer 0
[lfm2-forward] after tok_embeddings, h.device=Rocm(0), h.shape=[12, 1024]
[lfm2-forward] before layer loop
[lfm2-forward] layer 0
[lfm2-block] after attn_norm, x.device=Rocm(0)
[lfm2-block] shortconv branch
[Linear::forward] x.shape=[12, 1024], in_dim=1024, out_dim=3072, batch=12
[Linear::forward] out_shape=Shape { dims: [12, 3072] }, w_t.shape=[3072, 1024]

thread 'sleipnir_gguf_decode_golden_token_sequence' (2168538) panicked at crates/grim-engine/tests/sleipnir_rocm_inference.rs:161:10:
model.forward prefill failed: Tensor(ShapeMismatch { expected: [12, 1024], got: [3072, 1024] })
stack backtrace:
[lfm2-block] after attn_norm, x.device=Rocm(0)
[lfm2-block] shortconv branch
[Linear::forward] x.shape=[12, 1024], in_dim=1024, out_dim=3072, batch=12
[Linear::forward] out_shape=Shape { dims: [12, 3072] }, w_t.shape=[3072, 1024]
   0: __rustc::rust_begin_unwind
             at /rustc/c756124775121dea0e640652c5ee3c89e3dd0eb4/library/std/src/panicking.rs:689:5
   1: core::panicking::panic_fmt
             at /rustc/c756124775121dea0e640652c5ee3c89e3dd0eb4/library/core/src/panicking.rs:80:14
   2: core::result::unwrap_failed
             at /rustc/c756124775121dea0e640652c5ee3c89e3dd0eb4/library/core/src/result.rs:1867:5
   3: <core::result::Result<grim_tensor::tensor::Tensor, grim_core::error::Error>>::expect
             at /home/nelson/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/result.rs:1185:23
   4: sleipnir_rocm_inference::generate
             at ./tests/sleipnir_rocm_inference.rs:161:10
   5: sleipnir_rocm_inference::sleipnir_gguf_decode_golden_token_sequence
             at ./tests/sleipnir_rocm_inference.rs:341:15
   6: sleipnir_rocm_inference::sleipnir_gguf_decode_golden_token_sequence::{closure#0}
             at ./tests/sleipnir_rocm_inference.rs:326:48
   7: <sleipnir_rocm_inference::sleipnir_gguf_decode_golden_token_sequence::{closure#0} as core::ops::function::FnOnce<()>>::call_once
             at /home/nelson/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/function.rs:250:5
   8: <fn() -> core::result::Result<(), alloc::string::String> as core::ops::function::FnOnce<()>>::call_once
             at /rustc/c756124775121dea0e640652c5ee3c89e3dd0eb4/library/core/src/ops/function.rs:250:5
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.

thread 'sleipnir_gguf_prefill_logits_shape' (2168541) panicked at crates/grim-engine/tests/sleipnir_rocm_inference.rs:308:10:
prefill forward failed: Tensor(ShapeMismatch { expected: [12, 1024], got: [3072, 1024] })
stack backtrace:
   0: __rustc::rust_begin_unwind
             at /rustc/c756124775121dea0e640652c5ee3c89e3dd0eb4/library/std/src/panicking.rs:689:5
   1: core::panicking::panic_fmt
             at /rustc/c756124775121dea0e640652c5ee3c89e3dd0eb4/library/core/src/panicking.rs:80:14
   2: core::result::unwrap_failed
             at /rustc/c756124775121dea0e640652c5ee3c89e3dd0eb4/library/core/src/result.rs:1867:5
   3: <core::result::Result<grim_tensor::tensor::Tensor, grim_core::error::Error>>::expect
             at /home/nelson/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/result.rs:1185:23
   4: sleipnir_rocm_inference::sleipnir_gguf_prefill_logits_shape
             at ./tests/sleipnir_rocm_inference.rs:308:10
   5: sleipnir_rocm_inference::sleipnir_gguf_prefill_logits_shape::{closure#0}
             at ./tests/sleipnir_rocm_inference.rs:265:40
   6: <sleipnir_rocm_inference::sleipnir_gguf_prefill_logits_shape::{closure#0} as core::ops::function::FnOnce<()>>::call_once
             at /home/nelson/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/function.rs:250:5
   7: <fn() -> core::result::Result<(), alloc::string::String> as core::ops::function::FnOnce<()>>::call_once
             at /rustc/c756124775121dea0e640652c5ee3c89e3dd0eb4/library/core/src/ops/function.rs:250:5
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
test sleipnir_gguf_prefill_logits_shape ... FAILED
test sleipnir_gguf_decode_golden_token_sequence ... FAILED

failures:

failures:
    sleipnir_gguf_decode_golden_token_sequence
    sleipnir_gguf_prefill_logits_shape

test result: FAILED. 3 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 26.35s

error: test failed, to rerun pass `-p grim-engine --test sleipnir_rocm_inference`
