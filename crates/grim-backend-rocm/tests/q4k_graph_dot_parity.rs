//! Q4_K dot4 path through the GRAPH decode pipeline (linear_decode_into).
//!
//! The low-level `dot4_q4k_gemv_parity` test verifies the launcher against a CPU
//! reference. This test verifies the SAME kernel fires correctly when invoked
//! through the graph path's `linear_decode_into` dispatch — the function the
//! captured decode graph actually calls per projection.
//!
//! Key detail: `linear_decode_into` validates `w.shape().elem_count() / k == n`,
//! so Q4K weight storage must carry the LOGICAL shape [n, k] (not the packed
//! byte count) while `bytes` holds the compressed Q4_K data.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{
    ArithType, BackendStorage, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage,
};

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn q4k_linear_decode_into_matches_cpu() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("[SKIP] no ROCm device");
            return;
        }
    };
    unsafe {
        std::env::set_var("GRIM_DOT_GEMV", "1");
    }
    let m = 1usize;
    let n = 128usize;
    let k = 256usize; // Q4_K requires k % 256 == 0 for the dot4 path

    let a_f32 = rand_vec(m * k, 0x1337);
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::new(vec![m, k]), DType::F32).unwrap();

    // Build Q4_K weight storage. `from_cpu_bytes` returns Box<dyn BackendStorage>;
    // linear_decode_into needs &RocmStorage, so keep the box alive and downcast-ref.
    let b_f32: Vec<f32> = rand_vec(n * k, 0x2222);
    let b_bytes = grim_quant::quant_q4k(&b_f32).unwrap();
    let q4k_dtype = DType {
        arith: ArithType::F32,
        storage: Storage::KQuant(KQuantScheme::Q4K),
    };
    let w_box: Box<dyn BackendStorage> =
        MemoryOps::from_cpu_bytes(&dev, &b_bytes, &Shape::new(vec![n, k]), q4k_dtype).unwrap();
    let w_dev = w_box
        .as_any()
        .downcast_ref::<grim_backend_rocm::RocmStorage>()
        .expect("downcast");

    let alloc = dev.allocator_handle();
    let ord = dev.ordinal();
    let out_dev =
        grim_backend_rocm::RocmStorage::alloc_gpu(&Shape::new(vec![m, n]), DType::F32, &alloc, ord)
            .unwrap();

    let q81_elems = m * (k / 32) * 36;
    let act_q81 = grim_backend_rocm::RocmStorage::alloc_gpu(
        &Shape::new(vec![q81_elems]),
        DType {
            arith: ArithType::U8,
            storage: Storage::Native,
        },
        &alloc,
        ord,
    )
    .unwrap();

    dev.linear_decode_into(a_dev.as_ref(), w_dev, &out_dev, &act_q81)
        .expect("linear_decode_into Q4K");
    dev.synchronize();
    let c_dev = BackendStorage::to_cpu_vec_f32(&out_dev).expect("d2h output");

    // CPU reference: float matmul with the original weights.
    let mut c_cpu = vec![0.0f32; m * n];
    for col in 0..n {
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += a_f32[kk] * b_f32[col * k + kk];
        }
        c_cpu[col] = acc;
    }

    let diff = max_diff(&c_cpu, &c_dev);
    eprintln!("[q4k-graph-parity] m={m} n={n} k={k} max_diff={diff:.6}");
    assert!(
        diff < 0.5,
        "Q4_K graph-path GEMV diverges from CPU reference: {diff}"
    );
}
