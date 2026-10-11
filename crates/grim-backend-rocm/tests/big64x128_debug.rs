//! DEBUG: 16x16 diff map for the 64x128 experiment tile.
use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape, Storage};

#[test]
#[ignore]
fn big64x128_diff_map() {
    let dev = RocmDevice::try_new(0).unwrap();
    let (m, n, k) = (128usize, 128usize, 256usize);
    let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
    let b_host: Vec<f32> = (0..k * n).map(|i| 1.0 + (i as f32 * 0.013).cos().abs() * 6.0).collect();
    let b_packed = grim_quant::quant_q4k(&b_host).unwrap();
    let a_dev = CoreTensorOps::from_cpu(&dev, &a_host, &Shape::new(vec![m, k]), DType::F32).unwrap();
    let q_dtype = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q4K) };
    let b_dev = MemoryOps::from_cpu_bytes(&dev, &b_packed, &Shape::new(vec![b_packed.len()]), q_dtype).unwrap();
    let out = CoreTensorOps::zeros(&dev, &Shape::new(vec![m * n + 21000]), DType::F32).unwrap();
    let a_r = grim_backend_rocm::as_rocm(a_dev.as_ref()).unwrap();
    let b_r = grim_backend_rocm::as_rocm(b_dev.as_ref()).unwrap();
    let o_r = grim_backend_rocm::as_rocm(out.as_ref()).unwrap();
    dev.launch_wmma_big64x128_q4k_for_ab(a_r, b_r, o_r, m, n, k).unwrap();
    let got = out.to_cpu_vec_f32().unwrap();

    // If GRIM_DUMP_SB is set the kernel dumped both sB stages instead of
    // computing; compare every live column against the host dequant.
    if std::env::var("GRIM_DUMP_SB").is_ok() {
        let ld = 40usize;
        let mut worst = [0.0f32; 2];
        let mut worst_at = [(0usize, 0usize); 2];
        for stage in 0..2usize {
            for cc in 0..n {
                for kk in 0..32 {
                    let v = got[stage * n * ld + cc * ld + kk];
                    let want = b_host[kk + stage * 32 + cc * k]; // col-major B, col `cc`, k = stage*32+kk
                    let d = (v - want).abs();
                    if d > worst[stage] { worst[stage] = d; worst_at[stage] = (cc, kk); }
                }
            }
        }
        eprintln!("[sb-dump] stage0 worst {:+.4} at {:?}; stage1 worst {:+.4} at {:?}",
                  worst[0], worst_at[0], worst[1], worst_at[1]);
        eprintln!("[sb-dump] COMPILE-TIME: BIG_BN={} BIG_THREADS={} BIG_BM={} BIG_LD={}",
                  got[20100] as i32, got[20101] as i32, got[20102] as i32, got[20103] as i32);
        for cc in [62usize, 63, 64, 65, 126, 127] {
            eprintln!("[sb-dump] col {cc}: stage0 k0..4 = {:?}  (want {:?})",
                      &got[cc * ld..cc * ld + 4],
                      &b_host[0..4].iter().zip(b_host[cc * k..cc * k + 4].iter()).map(|(_, w)| *w).collect::<Vec<_>>());
        }
        return;
    }
    let ndeq = grim_quant::dequant_q4k(&b_packed, k * n).unwrap();
    let mut total_worst = 0.0f32;
    for bi in 0..m / 16 {
        let mut row = String::new();
        for bj in 0..n / 16 {
            let mut md = 0.0f32;
            for i in 0..16 {
                for j in 0..16 {
                    let (r, c) = (bi * 16 + i, bj * 16 + j);
                    let mut acc = 0.0f32;
                    for kk in 0..k { acc += a_host[r * k + kk] * ndeq[c * k + kk]; }
                    md = md.max((got[r * n + c] - acc).abs());
                }
            }
            row.push_str(&format!("{md:8.2}"));
            total_worst = total_worst.max(md);
        }
        eprintln!("[map] fragrow {bi}: {row}");
    }
    assert!(
        total_worst < 0.6,
        "64x128 tile diverges: worst fragment {total_worst}"
    );
}
