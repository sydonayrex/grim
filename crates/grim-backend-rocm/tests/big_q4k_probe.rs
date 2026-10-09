//! Minimal probe: big-tile kernel with a SYNTHETIC packed block.
//!
//! Block crafted analytically: d=1.0, dmin=0.0, all scale bytes = 4 (sc=4, m=0),
//! all qs nibbles = 4 -> every dequantized weight is exactly 16.0. Then
//! C[i,j] = 16 * sum(A[i]), trivially checkable, and any NaN/garbage localizes.
use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage};

#[test]
#[ignore]
fn big_q4k_minimal() {
    if std::env::var("GRIM_RUN_GPU_TESTS").is_err() { return; }
    let dev = RocmDevice::try_new(0).unwrap();
    let n = 64usize;

    for k in [32usize, 64usize, 256usize] {
        let m = 64usize;
        let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin() * 0.5 + 1.0).collect();
        // Synthetic Q4_K block (144 B): d=1.0 (0x3C00), dmin=0.0 (0x0000),
        // scales[12] = 4, qs[128] = 0x44 (nibbles 4,4) -> weight = 1*4*4 = 16.
        let mut blk = vec![0u8; 144];
        blk[0] = 0x00; blk[1] = 0x3C; // d = 1.0
        blk[2] = 0x00; blk[3] = 0x00; // dmin = 0.0
        for b in blk[4..16].iter_mut() { *b = 4; } // sc = 4, m = 0
        for b in blk[16..144].iter_mut() { *b = 0x44; } // nibbles 4,4
        let b_packed: Vec<u8> = (0..n).flat_map(|_| blk.clone()).collect();

        let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32).unwrap();
        let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q4K) };
        let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype).unwrap();
        let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![m, n]), DType::F32).unwrap();
        let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
        let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
        let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();

        dev.launch_wmma_big_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
        let got = out.to_cpu_vec_f32().unwrap();

        // oracle: weight = 16 everywhere -> C[i,j] = 16 * sum(a_host[i][..k])
        let mut nans = 0usize; let mut max_err = 0f32; let mut max_rel = 0f32;
        for row in 0..m {
            let sum: f32 = a_host[row * k..(row + 1) * k].iter().sum();
            for col in 0..n {
                let wantv = 16.0 * sum;
                let g = got[row * n + col];
                if g.is_nan() { nans += 1; continue; }
                max_err = max_err.max((g - wantv).abs());
                max_rel = max_rel.max((g - wantv).abs() / wantv.abs().max(1e-6));
            }
        }
        eprintln!("[big-probe] k={k}: nans={nans}/{} max_abs_err={max_err:.4} max_rel={max_rel:.4} want[0]={:.1}",
            m * n, 16.0 * a_host[0..k].iter().sum::<f32>());
        eprintln!("[big-probe]   got[0..8][:4] = {:?}", (0..4).map(|r| (0..4).map(|c| format!("{:.1}", got[r * n + c])).collect::<Vec<_>>()).collect::<Vec<_>>());
    }
}
