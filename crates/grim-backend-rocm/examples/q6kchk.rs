//! Controlled check: is the Q6_K output-head error in the device DEQUANT, or in
//! how the test builds activations? Uses a small, hand-checkable activation.
use grim_tensor::{CoreTensorOps, DType, MemoryOps, QuantOps, Shape};

const R: usize = 8;
const D: usize = 5120;

fn main() {
    let Ok(dev) = grim_backend_rocm::RocmDevice::try_new(0) else {
        eprintln!("[CHK] no device");
        return;
    };
    let w: Vec<f32> = (0..R * D).map(|i| (i as f32 * 0.0007).sin()).collect();
    let packed = grim_quant::quant_q6k(&w).unwrap();
    let deq = grim_quant::dequant_q6k(&packed, R * D).unwrap();

    // Alternating +/-1 so each product is trivially checkable.
    let a: Vec<f32> = (0..D).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

    let ps = dev
        .from_cpu_bytes(
            &packed,
            &Shape::new(vec![R, D]),
            DType { arith: grim_tensor::ArithType::U8, storage: grim_tensor::Storage::Native },
        )
        .unwrap();
    let av = dev.from_cpu(&a, &Shape::new(vec![1, D]), DType::F32).unwrap();
    let out_shape = Shape::new(vec![1, R]);

    match dev.fused_quant_gemm(av.as_ref(), ps.as_ref(), grim_tensor::QuantFormat::Q6K, &out_shape) {
        Ok((o, _)) => {
            let got = o.to_cpu_vec_f32().unwrap();
            for r in 0..R {
                let mut want = 0.0f64;
                for c in 0..D {
                    want += deq[r * D + c] as f64 * a[c] as f64;
                }
                eprintln!(
                    "[CHK] row {r}: got {:>14.4}  want {:>14.4}  ratio {:.4}",
                    got[r], want, got[r] as f64 / want
                );
            }
        }
        Err(e) => eprintln!("[CHK] fused_quant_gemm err: {e}"),
    }
}
