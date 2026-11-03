//! SPEED-DOT-OPFUSE: GPU parity for the fused RMSNorm + int8 q8_1 quantizer
//! (`grim_rmsnorm_quant_i8`) against a CPU reference of the same two steps.
//! Output layout is the packed 36-byte-per-block q8_1 format consumed by
//! `grim_dot4_q80_q81_gemv`.
//!
//! RUN ON THIS SYSTEM: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm --test rmsnorm_quant_parity

use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, Shape};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

#[test]
#[ignore]
fn rmsnorm_quant_fused_parity() {
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let eps = 1e-5_f32;
    for k in [1024usize, 4608usize] {
        let x: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.07).collect();
        let w: Vec<f32> = (0..k).map(|i| 1.0 + (i % 7) as f32 * 0.03).collect();

        let x_dev =
            CoreTensorOps::from_cpu(&dev, &x, &Shape::new(vec![k]), DType::F32).expect("upload x");
        let w_dev =
            CoreTensorOps::from_cpu(&dev, &w, &Shape::new(vec![k]), DType::F32).expect("upload w");
        let n_blocks = k / 32;
        let out_dev = dev
            .zeros(&Shape::new(vec![n_blocks * 36]), DType::U8)
            .expect("alloc out");

        let x_ref = x_dev
            .as_ref()
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("x is RocmStorage");
        let w_ref = w_dev
            .as_ref()
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("w is RocmStorage");
        let out_ref = out_dev
            .as_ref()
            .as_any()
            .downcast_ref::<grim_backend_rocm::RocmStorage>()
            .expect("out is RocmStorage");

        dev.launch_rmsnorm_quant_i8(x_ref, w_ref, eps, k, out_ref, None)
            .expect("launch fused rmsnorm+quant");
        dev.synchronize();

        let got = out_dev.to_cpu_vec_f32().expect("readback");
        let got_bytes: Vec<u8> = got.iter().map(|v| *v as u8).collect();

        // The GPU reduces the sum-of-squares with a 32-lane tree (different
        // fp32 associativity than the host's sequential sum), so `d` can land
        // 1 fp16 ULP away from the host value and codes shift ±1 near
        // quantization boundaries. Correct contract: dequantized values match
        // the CPU-normalized row within one code step.
        let n_blocks = k / 32;
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (ss / k as f32 + eps).sqrt();
        let mut max_err = 0.0f32;
        for blk in 0..n_blocks {
            let base = blk * 32;
            let d =
                half::f16::from_le_bytes([got_bytes[blk * 36], got_bytes[blk * 36 + 1]]).to_f32();
            for j in 0..32 {
                let code = got_bytes[blk * 36 + 4 + j] as i8;
                let deq = code as f32 * d;
                let want_v = x[base + j] * inv_rms * w[base + j];
                max_err = max_err.max((deq - want_v).abs());
            }
        }
        // One code step is d; rounding error is 0.5d; allow 1.0d for shifts.
        let tol = {
            let d0 = half::f16::from_le_bytes([got_bytes[0], got_bytes[1]]).to_f32();
            d0.max(1e-6)
        };
        eprintln!("[rmsnorm_quant] k={k} max_dequant_err={max_err:.3e} (tol {tol:.3e})");
        assert!(
            max_err <= tol,
            "fused rmsnorm+quant dequant error {max_err} exceeds one code step {tol} at k={k}"
        );
    }
}

/// Task 2.6 Step 3 gate: the fused RMSNorm/add+RMSNorm -> q8_1 dual-output
/// epilogues must (a) produce `out_f32`/`norm_out` BYTE-IDENTICAL to the
/// standalone `grim_rms_norm_decode` / `grim_add_rms_norm_decode` kernels
/// (same stride-128 reduction, same `x*w/rms` expression) and (b) produce
/// `out_q81` matching `grim_quantize_q8_1(out_f32)` within ±1 int8 code per
/// element (the amax butterfly is order-free, so `d` must match bit-for-bit;
/// only `round(v * inv_d)` may differ by one code from tie-breaking).
#[test]
#[ignore]
fn rmsnorm_quant_q81_dual_parity() {
    use grim_backend_rocm::as_rocm;
    use grim_tensor::{ArithType, MemoryOps};

    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
        return;
    };
    let f32ty = DType { arith: ArithType::F32, storage: grim_tensor::Storage::Native };
    let u8ty = DType { arith: ArithType::U8, storage: grim_tensor::Storage::Native };

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / u32::MAX as f32 - 0.5) * 4.0
        }
    }
    let f16_to_f32 = |bits: u16| -> f32 {
        half::f16::from_le_bytes([bits as u8, (bits >> 8) as u8]).to_f32()
    };

    let eps = 1e-5_f32;
    for k in [2048usize, 3584, 4096, 5120] {
        let mut rng = Lcg(0x2611_u64 ^ k as u64);
        let x: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
        let res: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..k).map(|_| 0.5 + rng.next_f32().abs() * 0.5).collect();

        let mk_f32 = |data: &[f32]| -> Box<dyn grim_tensor::BackendStorage> {
            CoreTensorOps::from_cpu(&dev, data, &Shape::new(vec![k]), DType::F32)
                .expect("from_cpu")
        };
        let x_st = mk_f32(&x);
        let res_st = mk_f32(&res);
        let w_st = mk_f32(&w);
        let out_ref = MemoryOps::alloc_storage(&dev, &Shape::new(vec![k]), f32ty.clone()).unwrap();
        let sum_ref = MemoryOps::alloc_storage(&dev, &Shape::new(vec![k]), f32ty.clone()).unwrap();
        let norm_ref = MemoryOps::alloc_storage(&dev, &Shape::new(vec![k]), f32ty.clone()).unwrap();
        let out_dual = MemoryOps::alloc_storage(&dev, &Shape::new(vec![k]), f32ty.clone()).unwrap();
        let sum_dual = MemoryOps::alloc_storage(&dev, &Shape::new(vec![k]), f32ty.clone()).unwrap();
        let norm_dual = MemoryOps::alloc_storage(&dev, &Shape::new(vec![k]), f32ty.clone()).unwrap();

        let q81_bytes = (k / 32) * 36;
        let mk_u8 = || {
            MemoryOps::from_cpu_bytes(
                &dev,
                &vec![0u8; q81_bytes],
                &Shape::new(vec![q81_bytes]),
                u8ty.clone(),
            )
            .unwrap()
        };
        let q81_ref = mk_u8();
        let q81_dual = mk_u8();

        // ── RMSNorm site ────────────────────────────────────────────────
        // Standalone (routes to grim_rms_norm_decode at rows==1, k>=1024).
        dev.rms_norm_into(
            x_st.as_ref(), w_st.as_ref(), eps,
            as_rocm(out_ref.as_ref()).unwrap(), &Shape::new(vec![k]),
        ).expect("rms_norm_into");
        dev.launch_rms_norm_quant_q81_dual(
            as_rocm(x_st.as_ref()).unwrap(),
            as_rocm(w_st.as_ref()).unwrap(),
            as_rocm(out_dual.as_ref()).unwrap(),
            as_rocm(q81_dual.as_ref()).unwrap(),
            k, eps,
        ).expect("rms dual");
        // Reference q81 from the STANDALONE norm's own output.
        dev.launch_quantize_q8_1(
            as_rocm(out_ref.as_ref()).unwrap(),
            as_rocm(q81_ref.as_ref()).unwrap(),
            1, k,
        ).expect("quantize_q8_1");
        dev.synchronize();

        let bits = |s: &Box<dyn grim_tensor::BackendStorage>| -> Vec<u32> {
            as_rocm(s.as_ref()).unwrap().to_cpu_vec_f32().unwrap()
                .iter().map(|v| v.to_bits()).collect()
        };
        for (i, (&r, &d)) in bits(&out_ref).iter().zip(bits(&out_dual)).enumerate() {
            assert_eq!(r, d, "rms dual out_f32 diverged at [{i}] (k={k})");
        }
        let rq = as_rocm(q81_ref.as_ref()).unwrap().copy_to_host().unwrap();
        let dq = as_rocm(q81_dual.as_ref()).unwrap().copy_to_host().unwrap();
        for b in 0..k / 32 {
            let (rb, db) = (&rq[b * 36..(b + 1) * 36], &dq[b * 36..(b + 1) * 36]);
            assert_eq!(
                f16_to_f32(u16::from_le_bytes([rb[0], rb[1]])).to_bits(),
                f16_to_f32(u16::from_le_bytes([db[0], db[1]])).to_bits(),
                "rms dual q81 d-scale diverged at block {b} (k={k})"
            );
            for i in 0..32 {
                let diff = (db[4 + i] as i8 as i16 - rb[4 + i] as i8 as i16).abs();
                assert!(diff <= 1,
                    "rms dual q81 code diverged >1 at block {b} elem {i} (k={k})");
            }
        }
        eprintln!("[dual-parity] rms site k={k}: out_f32 byte-identical, q81 within ±1 code");

        // ── Add+RMSNorm site ────────────────────────────────────────────
        dev.fused_add_rms_norm_into(
            x_st.as_ref(), res_st.as_ref(), w_st.as_ref(), eps,
            as_rocm(sum_ref.as_ref()).unwrap(),
            as_rocm(norm_ref.as_ref()).unwrap(),
            &Shape::new(vec![k]),
        ).expect("fused_add_rms_norm_into");
        dev.launch_add_rms_norm_quant_q81_dual(
            as_rocm(x_st.as_ref()).unwrap(),
            as_rocm(res_st.as_ref()).unwrap(),
            as_rocm(w_st.as_ref()).unwrap(),
            as_rocm(sum_dual.as_ref()).unwrap(),
            as_rocm(norm_dual.as_ref()).unwrap(),
            as_rocm(q81_dual.as_ref()).unwrap(),
            k, eps,
        ).expect("add dual");
        dev.launch_quantize_q8_1(
            as_rocm(norm_ref.as_ref()).unwrap(),
            as_rocm(q81_ref.as_ref()).unwrap(),
            1, k,
        ).expect("quantize_q8_1 (add)");
        dev.synchronize();

        for (i, (&r, &d)) in bits(&norm_ref).iter().zip(bits(&norm_dual)).enumerate() {
            assert_eq!(r, d, "add dual norm_out diverged at [{i}] (k={k})");
        }
        for (i, (&r, &d)) in bits(&sum_ref).iter().zip(bits(&sum_dual)).enumerate() {
            assert_eq!(r, d, "add dual y_out diverged at [{i}] (k={k})");
        }
        let rq = as_rocm(q81_ref.as_ref()).unwrap().copy_to_host().unwrap();
        let dq = as_rocm(q81_dual.as_ref()).unwrap().copy_to_host().unwrap();
        for b in 0..k / 32 {
            let (rb, db) = (&rq[b * 36..(b + 1) * 36], &dq[b * 36..(b + 1) * 36]);
            assert_eq!(
                f16_to_f32(u16::from_le_bytes([rb[0], rb[1]])).to_bits(),
                f16_to_f32(u16::from_le_bytes([db[0], db[1]])).to_bits(),
                "add dual q81 d-scale diverged at block {b} (k={k})"
            );
            for i in 0..32 {
                let diff = (db[4 + i] as i8 as i16 - rb[4 + i] as i8 as i16).abs();
                assert!(diff <= 1,
                    "add dual q81 code diverged >1 at block {b} elem {i} (k={k})");
            }
        }
        eprintln!("[dual-parity] add site k={k}: norm_out + y_out byte-identical, q81 within ±1 code");
    }
}
