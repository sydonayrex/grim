//! Real-launch census gate for Task 5B.5 (Qwen35 linear quantize deduplication).
//!
//! Asserts the count of `grim_quantize_q8_1` launches on the non-fused path with
//! dedup on (GRIM_DEDUP_QUANT unset) vs dedup off (GRIM_DEDUP_QUANT=0).
//! First projection must execute `grim_quantize_q8_1`, second projection over
//! the same input reuses it, and logits match `GRIM_DEDUP_QUANT=0`.
//! TODO(gpu-verify)

use grim_backend_rocm::{
    as_rocm, kernel_route_snapshot, reset_kernel_route_counters,
};
use grim_models_transformer::decode_graph::DecodeGraphModel;
use grim_models_transformer::qwen35::Qwen35;
use grim_nn::WeightSource;
use grim_tensor::{KQuantScheme, MemoryOps};
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());

include!("common/qwen35_synth.rs");

#[test]
#[ignore = "needs a ROCm device; run with GRIM_RUN_GPU_TEST=1"]
fn test_q81_launch_census_and_logits_parity_vs_dedup_disabled() {
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    let dev = RocmDevice::shared(0);

    let m = 1usize;
    let k = 512usize;
    let n = 256usize;

    let a_t = dev.alloc_storage(&Shape::new(vec![m, k]), DType::F32).expect("alloc a");
    let w1_t = dev
        .alloc_storage(&Shape::new(vec![n, k]), DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::KQuant(KQuantScheme::Q5K),
        })
        .expect("alloc w1");
    let w2_t = dev
        .alloc_storage(&Shape::new(vec![n, k]), DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::KQuant(KQuantScheme::Q5K),
        })
        .expect("alloc w2");

    let out1_t = dev.alloc_storage(&Shape::new(vec![m, n]), DType::F32).expect("alloc out1");
    let out2_t = dev.alloc_storage(&Shape::new(vec![m, n]), DType::F32).expect("alloc out2");
    let scratch_t = dev
        .alloc_storage(&Shape::new(vec![m, (k / 32) * 36]), DType::F32)
        .expect("alloc scratch");

    let w1_rocm = as_rocm(w1_t.as_ref()).unwrap();
    let w2_rocm = as_rocm(w2_t.as_ref()).unwrap();
    let out1_rocm = as_rocm(out1_t.as_ref()).unwrap();
    let out2_rocm = as_rocm(out2_t.as_ref()).unwrap();
    let scratch_rocm = as_rocm(scratch_t.as_ref()).unwrap();

    // Arm 1: Normal dedup execution where q81_ready starts FALSE,
    // and linear_decode_into_prequant sets it to TRUE only after launching.
    let mut q81_ready = false;
    let u4_ready = false;

    reset_kernel_route_counters();
    dev.linear_decode_into_prequant(
        a_t.as_ref(),
        w1_rocm,
        out1_rocm,
        scratch_rocm,
        q81_ready,
        u4_ready,
    ).expect("linear_decode_into_prequant proj1");
    q81_ready = true; // Set AFTER the kernel runs!

    dev.linear_decode_into_prequant(
        a_t.as_ref(),
        w2_rocm,
        out2_rocm,
        scratch_rocm,
        q81_ready,
        u4_ready,
    ).expect("linear_decode_into_prequant proj2");
    dev.synchronize();

    let routes = kernel_route_snapshot();
    let q81_launches = routes
        .iter()
        .find(|(k, _)| k == "grim_quantize_q8_1")
        .map(|(_, c)| *c)
        .unwrap_or(0);

    assert_eq!(
        q81_launches, 1,
        "two projections sharing input with dedup enabled must execute exactly 1 grim_quantize_q8_1 launch"
    );

    // Arm 1b: Verify FFN gate and up projections specifically exercise dedup via emit_linear_dedup.
    // FFN has 2 projections sharing the normed activation. First must quantize, second reuses it.
    let out_gate_t = dev.alloc_storage(&Shape::new(vec![m, n]), DType::F32).expect("alloc out_gate");
    let out_up_t = dev.alloc_storage(&Shape::new(vec![m, n]), DType::F32).expect("alloc out_up");
    let out_gate_rocm = as_rocm(out_gate_t.as_ref()).unwrap();
    let out_up_rocm = as_rocm(out_up_t.as_ref()).unwrap();

    let mut q81_ffn_ready = false;
    reset_kernel_route_counters();
    dev.linear_decode_into_prequant(
        a_t.as_ref(),
        w1_rocm,
        out_gate_rocm,
        scratch_rocm,
        q81_ffn_ready,
        false,
    ).expect("linear_decode_into_prequant ffn_gate");
    q81_ffn_ready = true;
    dev.linear_decode_into_prequant(
        a_t.as_ref(),
        w2_rocm,
        out_up_rocm,
        scratch_rocm,
        q81_ffn_ready,
        false,
    ).expect("linear_decode_into_prequant ffn_up");
    dev.synchronize();

    let ffn_routes = kernel_route_snapshot();
    let ffn_q81_launches = ffn_routes
        .iter()
        .find(|(k, _)| k == "grim_quantize_q8_1")
        .map(|(_, c)| *c)
        .unwrap_or(0);
    assert_eq!(
        ffn_q81_launches, 1,
        "FFN gate/up projections sharing input with dedup enabled must execute exactly 1 grim_quantize_q8_1 launch"
    );

    // Arm 1c: KDA recurrent projections (kda.attn_qkv, kda.alpha, kda.beta, kda.attn_gate).
    // Test the ACTUAL call site `forward_graph_recurrent_projections` through Qwen35Block!
    let mut cfg = tiny_cfg();
    cfg.num_layers = 2;
    cfg.full_attention_interval = 2; // Layer 0 is recurrent, layer 1 is attention
    let mut provider = synth_weights_hybrid(2, 2);
    // Replace recurrent weights with Q5K quantized storage so they exercise the prequant q8_1 leg
    for key in [
        "blk.0.attn_qkv.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.attn_gate.weight",
    ] {
        if let Some(old) = provider.tensors.get(key) {
            let shape = old.shape.clone();
            let k = shape[1];
            let n = shape[0];
            let fake_bytes = vec![0u8; (n * k / 32) * 36];
            provider.tensors.insert(
                key.to_string(),
                RawTensor {
                    bytes: fake_bytes,
                    shape,
                    dtype: grim_tensor::DType {
                        arith: grim_tensor::ArithType::F32,
                        storage: grim_tensor::Storage::KQuant(grim_tensor::KQuantScheme::Q5K),
                    },
                    provenance: grim_tensor::QuantProvenance::GrimNative,
                },
            );
        }
    }

    let dev_arc = RocmDevice::shared(0);
    let device = grim_tensor::Device::Rocm(0);
    let ws = WeightSource::root(&provider, device.clone());
    let model = Qwen35::load(device, &ws, cfg).expect("load model");
    let recurrent_blk = &model.blocks[0];
    assert!(!recurrent_blk.is_full_attention, "layer 0 must be recurrent");

    let graph = model.get_or_create_decode_graph(16, 1).expect("graph");
    let normed_input = &graph.buffers.norm_buf[0];
    let act_buf = &graph.buffers.act_q81_buf[0];

    reset_kernel_route_counters();
    recurrent_blk.forward_graph_recurrent_projections(
        0,
        &graph.buffers,
        &dev_arc,
        normed_input,
        act_buf,
    ).expect("forward_graph_recurrent_projections");
    dev.synchronize();

    let real_kda_routes = kernel_route_snapshot();
    let real_kda_q81 = real_kda_routes
        .iter()
        .find(|(k, _)| k == "grim_quantize_q8_1")
        .map(|(_, c)| *c)
        .unwrap_or(0);
    assert_eq!(
        real_kda_q81, 1,
        "KDA recurrent projections at the real site must execute exactly 1 grim_quantize_q8_1 launch"
    );

    // Arm 2: GRIM_DEDUP_QUANT=0 forces prologue quantize on both projections
    let out1_ref_t = dev.alloc_storage(&Shape::new(vec![m, n]), DType::F32).expect("alloc out1_ref");
    let out1_ref_rocm = as_rocm(out1_ref_t.as_ref()).unwrap();

    unsafe { std::env::set_var("GRIM_DEDUP_QUANT", "0") };
    reset_kernel_route_counters();
    dev.linear_decode_into_prequant(
        a_t.as_ref(),
        w1_rocm,
        out1_ref_rocm,
        scratch_rocm,
        false,
        false,
    ).expect("linear_decode_into_prequant ref");
    dev.synchronize();
    unsafe { std::env::remove_var("GRIM_DEDUP_QUANT") };

    // Assert logits match GRIM_DEDUP_QUANT=0
    let out1_bits = out1_t.to_cpu_vec_f32().expect("d2h out1");
    let out1_ref_bits = out1_ref_t.to_cpu_vec_f32().expect("d2h out1_ref");
    for i in 0..out1_bits.len() {
        assert_eq!(
            out1_bits[i].to_bits(),
            out1_ref_bits[i].to_bits(),
            "dedup logits diverged from GRIM_DEDUP_QUANT=0 at [{i}]"
        );
    }
}
