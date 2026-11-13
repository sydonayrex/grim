//! Real-checkpoint-weight window parity: run ACTUAL model weights through the
//! eager `quantized_matmul` at m=1 (twice: row A, row B) and m=2 ([A; B]) and
//! compare rows BITWISE. The synthetic-weight micro-test
//! (`grim-backend-rocm/tests/mtp_window_kernel_parity.rs`) used fresh
//! `quant_*` packing; real checkpoint tensors may have different layouts
//! (superblock alignment, padding) that change the kernel arm.
//!
//! Also compares each route against a CPU dequant oracle to characterize
//! WHICH side carries the drift (act-quant noise vs exact dequant).
//!
//! RUN: GRIM_GPU_TEST=1 cargo test --release -p grim-models-transformer \
//!      --test real_weight_window_parity -- --ignored --nocapture

use grim_backend_rocm::device::util::take_gpu_test_lock_logged;
use grim_tensor::{CoreTensorOps, Device, QuantOps, Shape};

#[ignore = "needs the 27B model + ROCm device; run with GRIM_GPU_TEST=1"]
#[test]
fn real_checkpoint_weights_m2_bitwise_match_m1() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
{
    let _lock = take_gpu_test_lock_logged();
        // Probe model: GRIM_PROBE_MODEL overrides (default = the 27B). A small
    // extracted checkpoint (scripts/extract_probe_model.py) lets the
    // correctness probes run entirely in VRAM on ONE GPU instead of
    // forcing a 16 GB model onto a 16 GB card through host-backed memory.
    let model_path = std::env::var("GRIM_PROBE_MODEL")
        .unwrap_or_else(|_| "/drive/bigfast/grim/models/qwen38-27b/Qwen3.8-27B-Q4_K_M.gguf".to_string());
    #[allow(non_snake_case)]
    let MODEL: &str = &model_path;
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("skipping: no model");
        return Ok(());
    }
    let device = Device::Rocm(0);
    let model = grim_engine::model_loader::load_model_from_gguf(MODEL, device.clone())
        .map_err(|e| format!("load: {e}"))?;
    let qwen = model
        .as_any()
        .downcast_ref::<grim_models_transformer::Qwen35>()
        .ok_or("not Qwen35")?;

    let mut seed = 0x51DEu64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((seed >> 33) as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let k_max = 17408usize;
    let a_row: Vec<f32> = (0..k_max).map(|_| rand()).collect();
    let b_row: Vec<f32> = (0..k_max).map(|_| rand()).collect();
    let dev = grim_backend_rocm::RocmDevice::shared(0);

    // Per-row dequant: grim_quant::dequant_* take ONE row's packed bytes and
    // the row's element count (see dot_gemv_parity usage).
    let deq_row = |w: &grim_tensor::Tensor, row_bytes: &[u8], k: usize| -> Vec<f32> {
        use grim_tensor::KQuantScheme;
        match &w.dtype().storage {
            grim_tensor::Storage::KQuant(KQuantScheme::IQ4XS) => {
                grim_quant::dequant_iq4xs(row_bytes, k).expect("dequant_iq4xs")
            }
            grim_tensor::Storage::KQuant(KQuantScheme::Q5K) => {
                grim_quant::dequant_q5k(row_bytes, k).expect("dequant_q5k")
            }
            grim_tensor::Storage::KQuant(KQuantScheme::Q4K) => {
                grim_quant::dequant_q4k(row_bytes, k).expect("dequant_q4k")
            }
            grim_tensor::Storage::KQuant(KQuantScheme::Q3K) => {
                grim_quant::dequant_q3k(row_bytes, k).expect("dequant_q3k")
            }
            other => {
                eprintln!("[real-w] no dequant for {other:?}; skipping oracle");
                return Vec::new();
            }
        }
    };

    let blk = &qwen.blocks[0];
    let cases: Vec<(&str, &grim_tensor::Tensor)> = vec![
        ("blk0.ffn_gate (IQ4XS n=17408 k=5120)", blk.ffn_gate.weight()),
        ("blk0.ffn_down (n=5120 k=17408)", blk.ffn_down.weight()),
        ("blk0.ffn_up (n=17408 k=5120)", blk.ffn_up.weight()),
        (
            "blk3.wq (attention layer)",
            qwen.blocks[3]
                .wq
                .as_ref()
                .expect("blk3 has separated wq")
                .weight(),
        ),
    ];

    for (label, w) in cases {
        let dims = w.shape().dims().to_vec();
        let (n, k) = (dims[0], dims[1]);
        let b_storage = w.storage().as_ref();
        let b_bytes = grim_backend_rocm::device::util::as_rocm(b_storage)
            .expect("rocm weight")
            .copy_to_host()
            .expect("weight d2h");
        // Persist device bytes for the raw-GGUF-vs-device comparison.
        let tag: String = label
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let _ = std::fs::write(format!("/tmp/real_w_{}.bin", tag), &b_bytes);
        eprintln!("[real-w] {label}: device bytes {} -> /tmp/real_w_{}.bin", b_bytes.len(), tag);

        let upload = |row: &[f32]| -> Box<dyn grim_tensor::BackendStorage> {
            CoreTensorOps::from_cpu(&dev, row, &Shape::new(vec![1, k]), grim_tensor::DType::F32)
                .expect("act upload")
        };
        let one = |row: &[f32]| -> Vec<f32> {
            let a = upload(row);
            let (s, h) = dev
                .quantized_matmul(
                    a.as_ref(),
                    b_storage,
                    &[],
                    grim_tensor::QuantFormat::try_from(&w.dtype().storage)
                        .expect("format"),
                    &Shape::new(vec![1, n]),
                )
                .expect("m=1");
            h.synchronize().expect("sync");
            s.to_cpu_vec_f32().expect("d2h")
        };
        let (a0, b0) = (one(&a_row[..k]), one(&b_row[..k]));

        let both = [a_row[..k].to_vec(), b_row[..k].to_vec()].concat();
        let a2 = CoreTensorOps::from_cpu(&dev, &both, &Shape::new(vec![2, k]), grim_tensor::DType::F32)
            .expect("act2 upload");
        let (s2, h2) = dev
            .quantized_matmul(
                a2.as_ref(),
                b_storage,
                &[],
                grim_tensor::QuantFormat::try_from(&w.dtype().storage).expect("format"),
                &Shape::new(vec![2, n]),
            )
            .expect("m=2");
        h2.synchronize().expect("sync");
        let c2 = s2.to_cpu_vec_f32().expect("d2h");
        let (c_a, c_b) = (c2[..n].to_vec(), c2[n..].to_vec());

        let eq_a = a0.iter().zip(&c_a).all(|(x, y)| x.to_bits() == y.to_bits());
        let eq_b = b0.iter().zip(&c_b).all(|(x, y)| x.to_bits() == y.to_bits());
        let d_a = a0
            .iter()
            .zip(&c_a)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let d_b = b0
            .iter()
            .zip(&c_b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);

        // CPU oracle: per-row exact dequant + f32 matmul on the same rows.
        // Row byte count from the packed tensor: total bytes / n.
        let row_bytes = b_bytes.len() / n;
        let oracle = |row: &[f32]| -> Vec<f32> {
            (0..n)
                .map(|col| {
                    let w_row = deq_row(w, &b_bytes[col * row_bytes..(col + 1) * row_bytes], k);
                    row.iter().zip(&w_row).map(|(x, y)| x * y).sum::<f32>()
                })
                .collect()
        };
        let o_a = oracle(&a_row[..k]);
        let m1_vs_o = a0
            .iter()
            .zip(&o_a)
            .map(|(x, y)| ((x - y).abs()) / y.abs().max(1e-3))
            .fold(0f32, f32::max);
        let m2_vs_o = c_a
            .iter()
            .zip(&o_a)
            .map(|(x, y)| ((x - y).abs()) / y.abs().max(1e-3))
            .fold(0f32, f32::max);

        eprintln!(
            "[real-w] {label}: dtype={:?} m=2-vs-m=1 rowA={} (max {d_a:.3e}) rowB={} (max {d_b:.3e}) | rel-vs-oracle m1={m1_vs_o:.3e} m2={m2_vs_o:.3e}",
            w.dtype().storage,
            eq_a,
            eq_b
        );
    }
    Ok(())
}
