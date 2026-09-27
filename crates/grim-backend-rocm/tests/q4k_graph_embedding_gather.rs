//! The graph-capturable embedding gather must handle a PACKED Q4_K vocab.
//!
//! `launch_embedding_gather_dev_idx` used to accept F32 native tables only, so
//! every quantized model refused graph capture and decoded eagerly on every
//! token. The Q4_K branch gathers and dequantizes inside the same launch
//! instead of inflating the table to f32.
//!
//! Two things this pins that a shape check would not:
//!
//! * the gathered row equals the host `grim_quant::dequant_q4k` of the same
//!   packed bytes, so the graph path and the eager path cannot drift apart;
//! * an out-of-range token id writes 0.0 instead of reading outside the table.
//!   The host cannot range-check here — the ids are in device memory and
//!   reading them back would sync in the middle of a capture — so the guard
//!   has to be in the kernel, and this is the only thing that observes it.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::{RocmDevice, RocmStorage};
use grim_tensor::dtype::{ArithType, DType, KQuantScheme, Storage};
use grim_tensor::{CoreTensorOps, MemoryOps, Shape};

const QK_BLOCK: usize = 256;
const QK_BLOCK_BYTES: usize = 144;

fn q4k_dtype() -> DType {
    DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    }
}

fn i32_dtype() -> DType {
    DType {
        arith: ArithType::U32,
        storage: Storage::Native,
    }
}

/// Deterministic, non-degenerate values, including exact zeros so a row that
/// is silently left unwritten cannot masquerade as a correct gather.
fn synth_table(vocab: usize, dim: usize) -> Vec<f32> {
    (0..vocab * dim)
        .map(|i| {
            if i % 613 == 0 {
                0.0
            } else {
                ((i as f32) * 0.017).sin() * 2.5 + ((i as f32) * 0.0031).cos() * 0.5
            }
        })
        .collect()
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn graph_gather_q4k_matches_host_dequant() {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let Some(dev) = std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("try_new(0)")).ok()
    else {
        return;
    };

    // The real Qwen3.8 geometry: 5120 = 20 Q4_K super-blocks per row.
    let vocab = 8usize;
    let dim = 5120usize;
    let host = synth_table(vocab, dim);
    let packed = grim_quant::quant_q4k(&host).expect("quant_q4k");
    assert_eq!(packed.len(), (vocab * dim / QK_BLOCK) * QK_BLOCK_BYTES);

    let table = dev
        .from_cpu_bytes(&packed, &Shape::new(vec![packed.len()]), q4k_dtype())
        .expect("upload packed table");
    let table = grim_backend_rocm::as_rocm(table.as_ref()).expect("RocmStorage");

    let ids: Vec<u32> = vec![0, 3, vocab as u32 - 1, 5];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let indices = dev
        .from_cpu_bytes(&id_bytes, &Shape::new(vec![ids.len()]), i32_dtype())
        .expect("upload ids");
    let indices = grim_backend_rocm::as_rocm(indices.as_ref()).expect("RocmStorage");

    let out_shape = Shape::new(vec![ids.len(), dim]);
    let out_s = dev
        .alloc_storage(&out_shape, DType::F32)
        .expect("out buffer");
    let out = grim_backend_rocm::as_rocm(out_s.as_ref()).expect("RocmStorage out");

    dev.launch_embedding_gather_dev_idx(table, out, indices, dim, ids.len() * dim)
        .expect("graph gather q4k");
    let got = <RocmStorage as grim_tensor::BackendStorage>::to_cpu_vec_f32(out).expect("read out");

    let full = grim_quant::dequant_q4k(&packed, vocab * dim).expect("host dequant");
    assert_eq!(got.len(), ids.len() * dim);
    for (n, &tok) in ids.iter().enumerate() {
        let row = tok as usize;
        let mut worst = 0.0f32;
        for j in 0..dim {
            let want = full[row * dim + j];
            let d = (got[n * dim + j] - want).abs();
            let r = d / want.abs().max(1e-3);
            worst = worst.max(d.min(r));
        }
        assert!(
            worst <= 1e-5,
            "token {tok} row differs from the host dequant: worst |err| {worst:.3e}"
        );
    }
    eprintln!(
        "[graph-gather] {} Q4_K rows match the host dequant",
        ids.len()
    );
}

#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn graph_gather_q4k_out_of_range_id_writes_zero() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    let Some(dev) = std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("try_new(0)")).ok()
    else {
        return;
    };

    let vocab = 4usize;
    let dim = 5120usize;
    let host = synth_table(vocab, dim);
    let packed = grim_quant::quant_q4k(&host).expect("quant_q4k");
    let table = dev
        .from_cpu_bytes(&packed, &Shape::new(vec![packed.len()]), q4k_dtype())
        .expect("upload");
    let table = grim_backend_rocm::as_rocm(table.as_ref()).expect("RocmStorage");

    // One past the end, and a negative id, as u32/i32 bit patterns.
    let ids: Vec<i32> = vec![vocab as i32, -1];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let indices = dev
        .from_cpu_bytes(&id_bytes, &Shape::new(vec![ids.len()]), i32_dtype())
        .expect("upload ids");
    let indices = grim_backend_rocm::as_rocm(indices.as_ref()).expect("RocmStorage");

    let out_shape = Shape::new(vec![ids.len(), dim]);
    let out_s = dev.alloc_storage(&out_shape, DType::F32).expect("out");
    let out = grim_backend_rocm::as_rocm(out_s.as_ref()).expect("RocmStorage out");
    dev.launch_embedding_gather_dev_idx(table, out, indices, dim, ids.len() * dim)
        .expect("graph gather");
    let got = <RocmStorage as grim_tensor::BackendStorage>::to_cpu_vec_f32(out).expect("read");

    assert!(
        got.iter().all(|&v| v == 0.0),
        "an out-of-range token must write 0.0, not read outside the table ({} non-zero)",
        got.iter().filter(|&&v| v != 0.0).count()
    );
    eprintln!("[graph-gather] out-of-range ids clamped to 0.0");
}

/// The F32 path must still work — the Q4_K branch is additive, not a
/// replacement, and a model with a native vocab must not regress.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn graph_gather_f32_still_supported() {
    if !grim_backend_rocm::gpu_test_enabled() {
        return;
    }
    let Some(dev) = std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("try_new(0)")).ok()
    else {
        return;
    };

    let vocab = 8usize;
    let dim = 64usize;
    let host = synth_table(vocab, dim);
    let table = CoreTensorOps::from_cpu(&dev, &host, &Shape::new(vec![vocab, dim]), DType::F32)
        .expect("upload f32 table");
    let table = grim_backend_rocm::as_rocm(table.as_ref()).expect("RocmStorage");

    let ids: Vec<u32> = vec![0, 7];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let indices = dev
        .from_cpu_bytes(&id_bytes, &Shape::new(vec![ids.len()]), i32_dtype())
        .expect("ids");
    let indices = grim_backend_rocm::as_rocm(indices.as_ref()).expect("RocmStorage");

    let out_shape = Shape::new(vec![ids.len(), dim]);
    let out_s = dev.alloc_storage(&out_shape, DType::F32).expect("out");
    let out = grim_backend_rocm::as_rocm(out_s.as_ref()).expect("RocmStorage out");
    dev.launch_embedding_gather_dev_idx(table, out, indices, dim, ids.len() * dim)
        .expect("f32 gather");
    let got = <RocmStorage as grim_tensor::BackendStorage>::to_cpu_vec_f32(out).expect("read");

    for (n, &tok) in ids.iter().enumerate() {
        for j in 0..dim {
            assert_eq!(
                got[n * dim + j],
                host[tok as usize * dim + j],
                "f32 path regressed at token {tok} elem {j}"
            );
        }
    }
    eprintln!("[graph-gather] f32 native path unchanged");
}
