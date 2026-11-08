// Quick A/B: production launcher (v2) vs skip kernel at tau=0, m=1 n=1024 k=1024.
use grim_backend_rocm::as_rocm;
use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::{CoreTensorOps, KQuantScheme, MemoryOps, Shape};

#[test]
#[ignore]
fn skip_tau0_vs_production_v2_bit_identity() {
    if !grim_backend_rocm::gpu_test_enabled() { return; }
    let dev = RocmDevice::try_new(0).unwrap();
    let (m, n, k) = (1usize, 1024usize, 1024usize);
    let mut seed = 42u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let a: Vec<f32> = (0..m * k).map(|_| rand()).collect();
    let mut b_f32 = vec![0.0f32; n * k];
    for v in b_f32.iter_mut() { *v = rand(); }
    let bq = grim_quant::quant_q80(&b_f32).unwrap();
    let q80ty = DType { arith: ArithType::F32, storage: Storage::KQuant(KQuantScheme::Q80) };
    let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
    let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), q80ty).unwrap();
    let a_st = CoreTensorOps::from_cpu(&dev, &a, &Shape::new(vec![m, k]), DType::F32).unwrap();
    fn mk_out(dev: &RocmDevice, m: usize, n: usize, f32ty: DType) -> Box<dyn grim_tensor::backend::BackendStorage> { MemoryOps::alloc_storage(dev, &Shape::new(vec![m * n]), f32ty).unwrap() }
    let b_r = as_rocm(b_st.as_ref()).unwrap();
    let a_r = as_rocm(a_st.as_ref()).unwrap();
    let o1 = mk_out(&dev, m, n, f32ty.clone());
    let o2 = mk_out(&dev, m, n, f32ty);
    unsafe { std::env::remove_var("GRIM_KGROUP_SKIP_TAU"); }
    // production dispatch: default V2 on -> grim_dot4_q80_f32act_gemv_v2
    dev.launch_dot4_q80_f32act_gemv(a_r, b_r, as_rocm(o1.as_ref()).unwrap(), m, n, k).unwrap();
    dev.launch_dot4_q80_f32act_gemv_skip(a_r, b_r, as_rocm(o2.as_ref()).unwrap(), m, n, k, 0.0).unwrap();
    dev.synchronize();
    let c1 = o1.to_cpu_vec_f32().unwrap();
    let c2 = o2.to_cpu_vec_f32().unwrap();
    let diffs = c1.iter().zip(&c2).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let maxd = c1.iter().zip(&c2).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()));
    println!("[v2-check] bit diffs = {diffs}/{} max|delta| = {maxd:.6e}", c1.len());
}
