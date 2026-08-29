use grim_backend_rocm::{CoreTensorOps, RocmDevice};
use grim_tensor::{DType, Shape};

fn main() {
    let mut args = std::env::args().skip(1);
    let ordinal = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    let m = args.next().and_then(|a| a.parse().ok()).unwrap_or(8);
    let n = args.next().and_then(|a| a.parse().ok()).unwrap_or(65536);
    let dev = RocmDevice::try_new(ordinal).expect("try_new");
    let k = 1024usize;
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.06)
        .collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.01 - 0.03).collect();

    let shape_a = Shape::from_slice(&[m, k]);
    let shape_b = Shape::from_slice(&[k, n]);
    let a = dev
        .from_cpu(&a_data, &shape_a, DType::F32)
        .expect("upload A");
    let b = dev
        .from_cpu(&b_data, &shape_b, DType::F32)
        .expect("upload B");

    let out_shape = Shape::from_slice(&[m, n]);
    let (out, handle) = dev
        .matmul(a.as_ref(), b.as_ref(), &out_shape)
        .expect("matmul");
    handle.synchronize().expect("sync");
    let got = out.to_cpu_vec_f32().expect("readback");

    // Host reference for row 0, first 4 columns.
    let mut want = vec![0f32; 4];
    for (j, w) in want.iter_mut().enumerate() {
        let mut acc = 0f32;
        for p in 0..k {
            acc += a_data[p] * b_data[p * n + j];
        }
        *w = acc;
    }
    println!(
        "ordinal {ordinal}: out head={:?} want={want:?} nonzero_frac={:.3}",
        &got[..4],
        got.iter().filter(|&&v| v != 0.0).count() as f32 / got.len() as f32
    );
}
