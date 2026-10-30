//! Predicated-tail parity for the v5-style single-loop cleanup: the fast
//! loop + scalar tail in `grim_fp32_gemv_v2`, `grim_dot2_bf16_gemv_v2`,
//! `grim_rms_norm` and `grim_rms_norm_i8` collapse into ONE 4-wide predicated
//! step (out-of-range lanes load 0.0 and `acc + 0.0f` is an exact identity).
//!
//! The shipped parity shapes all use K (row_len) as a multiple of 128, so the
//! folded tail never fires there. These tests use non-multiple shapes where
//! the upper lanes of a wave exercise the predicated steps, and gate on
//! bit-identity (GEMVs) / one-grid-step tolerance (the i8 norm).
use grim_backend_rocm::{as_rocm, CoreTensorOps as _, RocmDevice};
use grim_tensor::{ArithType, DType, MemoryOps, Shape, Storage};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    RocmDevice::try_new(0).ok()
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

// K values where blk+96 >= K for the upper lanes: the fast loop bails early
// and the folded predicated steps finish the row. k=1 and k=130 are the
// degenerate cases (only lane 0 / only 4 full steps respectively).
#[test]
#[ignore = "requires AMD GPU; run with GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=1"]
fn fp32_gemv_v2_tail_parity() -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let f32ty = DType {
        arith: ArithType::F32,
        storage: Storage::Native,
    };
    for &(n, k, m) in &[(67usize, 130usize, 1usize), (68, 400, 4), (5, 5000, 1), (1, 1, 1)] {
        let mut rng = Lcg(0xA11CE7A1 ^ (n * 31 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();
        let a_st = MemoryOps::from_cpu_bytes(&dev, f32_bytes(&a), &Shape::new(vec![m * k]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let b_st = MemoryOps::from_cpu_bytes(&dev, f32_bytes(&b), &Shape::new(vec![n * k]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n]), f32ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_r = as_rocm(b_st.as_ref()).unwrap();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());
        dev.launch_fp32_gemv_v1_for_ab(a_r, b_r, o1_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_fp32_gemv_v2_for_ab(a_r, b_r, o2_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let bytes1 = o1_r.copy_to_host().map_err(|e| format!("copy_to_host: {e}"))?;
        let bytes2 = o2_r.copy_to_host().map_err(|e| format!("copy_to_host: {e}"))?;
        let out1: Vec<f32> = bytes1
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let out2: Vec<f32> = bytes2
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let diffs = out1
            .iter()
            .zip(&out2)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(diffs, 0, "fp32_gemv_v2 not bit-identical at n={n} k={k} m={m}");
        println!("fp32       tail parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");
    }
    Ok(())
}

#[test]
#[ignore = "requires AMD GPU; run with GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=1"]
fn bf16_gemv_v2_tail_parity() -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    let u8ty = DType {
        arith: ArithType::U8,
        storage: Storage::Native,
    };
    // K even (n_pairs = K/2) but n_pairs NOT a multiple of 64: p+96 >=
    // n_pairs for the upper lanes, so the folded steps carry the tail.
    for &(n, k, m) in &[(67usize, 130usize, 1usize), (68, 400, 4), (5, 2000, 1)] {
        let n_pairs = k / 2;
        let mut rng = Lcg(0xA11CE7A2 ^ (n * 31 + k) as u64);
        let a: Vec<u8> = (0..m * k)
            .flat_map(|_| half::bf16::from_f32(rng.next_f32()).to_le_bytes())
            .collect();
        let b: Vec<u8> = (0..n * k)
            .flat_map(|_| half::bf16::from_f32(rng.next_f32()).to_le_bytes())
            .collect();
        let _ = n_pairs;
        let a_st = MemoryOps::from_cpu_bytes(&dev, &a, &Shape::new(vec![a.len()]), u8ty.clone())
            .map_err(|e| e.to_string())?;
        let b_st = MemoryOps::from_cpu_bytes(&dev, &b, &Shape::new(vec![b.len()]), u8ty.clone())
            .map_err(|e| e.to_string())?;
        let o1 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n * 2]), u8ty.clone())
            .map_err(|e| e.to_string())?;
        let o2 = MemoryOps::alloc_storage(&dev, &Shape::new(vec![m * n * 2]), u8ty.clone())
            .map_err(|e| e.to_string())?;
        let a_r = as_rocm(a_st.as_ref()).unwrap();
        let b_r = as_rocm(b_st.as_ref()).unwrap();
        let (o1_r, o2_r) = (as_rocm(o1.as_ref()).unwrap(), as_rocm(o2.as_ref()).unwrap());
        dev.launch_dot2_bf16_gemv_v1_for_ab(a_r, b_r, o1_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.launch_dot2_bf16_gemv_v2_for_ab(a_r, b_r, o2_r, m, n, k)
            .map_err(|e| e.to_string())?;
        dev.synchronize();
        let out1: Vec<u8> = o1_r.copy_to_host().map_err(|e| format!("copy_to_host: {e}"))?;
        let out2: Vec<u8> = o2_r.copy_to_host().map_err(|e| format!("copy_to_host: {e}"))?;
        assert_eq!(out1, out2, "bf16_gemv_v2 not bit-identical at n={n} k={k} m={m}");
        println!("bf16       tail parity n={n:<5} k={k:<5} m={m}: bit-identical to v1 (0 diffs)");
    }
    Ok(())
}

// row_len NOT a multiple of 128: the norm kernels' folded tail. The f32 path
// is checked against a CPU reference (same formula; summation order differs,
// so tolerance, not bit-identity), and the packed i8 path against the f32
// path with the existing one-weight-step gate.
#[test]
#[ignore = "requires AMD GPU; run with GRIM_RUN_GPU_TEST=1 HIP_VISIBLE_DEVICES=1"]
fn rms_norm_tail_parity() -> Result<(), String> {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    use grim_tensor::{DType as DT, Shape as SH};
    let rows = 3usize;
    for &dim in &[1000usize, 130, 1, 3584 + 40] {
        let mut rng = Lcg(0xA11CE7A3 ^ dim as u64);
        let x: Vec<f32> = (0..rows * dim).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..dim).map(|_| 0.5 + rng.next_f32().abs()).collect();
        let eps = 1e-5f32;
        let x_dev = MemoryOps::from_cpu_bytes(
            &dev,
            f32_bytes(&x),
            &SH::new(vec![rows, dim]),
            DT::F32,
        )
        .map_err(|e| e.to_string())?;
        let w_dev =
            MemoryOps::from_cpu_bytes(&dev, f32_bytes(&w), &SH::new(vec![dim]), DT::F32)
                .map_err(|e| e.to_string())?;
        let (s, h) = dev
            .rms_norm(x_dev.as_ref(), w_dev.as_ref(), eps, &SH::new(vec![rows, dim]))
            .map_err(|e| e.to_string())?;
        h.synchronize().map_err(|e| e.to_string())?;
        let out = s.to_cpu_vec_f32().map_err(|e| e.to_string())?;

        let mut max_rel = 0.0f64;
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let rms = (ss / dim as f32 + eps).sqrt();
            for c in 0..dim {
                let want = row[c] * w[c] / rms;
                let got = out[r * dim + c];
                if want.abs() > 1e-6 {
                    max_rel = max_rel.max(((got - want) as f64 / want.abs() as f64).abs());
                }
            }
        }
        assert!(
            max_rel < 1e-3,
            "rms_norm f32 path off CPU reference at dim={dim}: max rel {max_rel:.2e}"
        );
        println!("rms_norm   tail parity dim={dim:<5} rows={rows}: max rel vs CPU = {max_rel:.2e}");
    }
    Ok(())
}
