//! Task 1.2 parity gate for activation-quantization deduplication.
//!
//! The dedup contract (`d780bd54`): when a linear's activation is already
//! quantized (`q81_ready=true`), `linear_decode_into_prequant` must consume
//! `act_q81` AS-IS; when false, it quantizes the f32 activation itself with
//! the same `grim_quantize_q8_1` kernel. All arms must therefore produce
//! BIT-IDENTICAL outputs:
//!   A: prologue quantize (q81_ready=false)
//!   B: pre-quantize with `launch_quantize_q8_1`, then consume (q81_ready=true)
//!   C: `GRIM_DEDUP_QUANT=0` (forces the prologue even with q81_ready=true)
//!
//! This is the kernel-level half of the gate. The model-level half (27B census:
//! quantize_q8_1 285→217, u4 103→82, 1327 nodes, 0.000e0 logits) lives in the
//! `qwen35_production_27b_route_wall_and_utilization` harness test.
//!
//! RUN: GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-tests --features rocm \
//!        --test dedup_quant_parity -- --ignored --nocapture

#[cfg(feature = "rocm")]
mod rocm_gate {
    use grim_backend_rocm::{as_rocm, RocmDevice};
    use grim_tensor::{BackendStorage, CoreTensorOps, DType, KQuantScheme, MemoryOps, Shape};
    use std::sync::Mutex;

    static DEDUP_MUTEX: Mutex<()> = Mutex::new(());

    fn gpu_device() -> Option<RocmDevice> {
        if !grim_backend_rocm::gpu_test_enabled() {
            return None;
        }
        std::panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
    }

    fn q5k_dtype() -> DType {
        DType {
            arith: grim_tensor::ArithType::F32,
            storage: grim_tensor::Storage::KQuant(KQuantScheme::Q5K),
        }
    }

    /// The dedup contract at the kernel level: prequantized-consume ==
    /// prologue-quantize == DEDUP=0 rollback, bit for bit, on the Q5_K leg
    /// (the family the 27B dedup shares q8_1 across: 285→217 quantize launches).
    #[test]
    #[ignore = "needs a ROCm device; run with GRIM_RUN_GPU_TEST=1"]
    fn dedup_prequant_consumption_bit_identical_to_prologue_quantize() {
        let _lock = DEDUP_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dev) = gpu_device() else {
            eprintln!("[SKIP] requires GRIM_RUN_GPU_TEST=1 + GPU");
            return;
        };

        let m = 1usize;
        let k = 512usize; // multiple of 256 (Q5_K row) and of 32 (q8_1 block)
        let n = 256usize;

        let mut seed = 0x1dea_d00du64;
        let mut rand = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let a_f32: Vec<f32> = (0..m * k).map(|_| rand()).collect();
        let b_f32: Vec<f32> = (0..n * k).map(|_| rand()).collect();
        let b_bytes = grim_quant::quant_q5k(&b_f32).expect("quant_q5k");

        let w_dev = MemoryOps::from_cpu_bytes(
            &dev,
            &b_bytes,
            &Shape::new(vec![n, k]),
            q5k_dtype(),
        )
        .expect("upload q5k weights");
        let w_rocm = as_rocm(w_dev.as_ref()).expect("w is RocmStorage");
        let a_dev = CoreTensorOps::from_cpu(&dev, &a_f32, &Shape::new(vec![m, k]), DType::F32)
            .expect("upload activation");
        let a_rocm = as_rocm(a_dev.as_ref()).expect("a is RocmStorage");

        // q8_1 scratch: 36 B per 32 elements per row.
        let q81_bytes = m * (k / 32) * 36;
        let scratch = MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; q81_bytes],
            &Shape::new(vec![q81_bytes]),
            DType::U8,
        )
        .expect("alloc q81 scratch");
        let scratch_r = as_rocm(scratch.as_ref()).expect("scratch RocmStorage");

        let mk_out = || -> Box<dyn BackendStorage> {
            CoreTensorOps::from_cpu(
                &dev,
                &vec![0f32; m * n],
                &Shape::new(vec![m, n]),
                DType::F32,
            )
            .expect("alloc out")
        };
        let run_linear = |out: &mut Box<dyn BackendStorage>, q81_ready: bool| {
            let out_rocm = as_rocm(out.as_ref()).expect("out RocmStorage");
            dev.linear_decode_into_prequant(a_dev.as_ref(), w_rocm, out_rocm, scratch_r, q81_ready, false)
                .expect("linear_decode_into_prequant");
            dev.synchronize();
        };
        let d2h = |s: &Box<dyn BackendStorage>| -> Vec<f32> { s.to_cpu_vec_f32().expect("d2h") };

        // Arm A (prologue): q81_ready=false -> the launcher quantizes a itself.
        let mut out_a = mk_out();
        run_linear(&mut out_a, false);
        let a_out = d2h(&out_a);

        // Arm B (dedup consume): quantize FIRST with the same kernel the
        // prologue would run, then hand q81_ready=true to the linear.
        dev.launch_quantize_q8_1(a_rocm, scratch_r, m, k)
            .expect("grim_quantize_q8_1");
        dev.synchronize();
        let mut out_b = mk_out();
        run_linear(&mut out_b, true);
        let b_out = d2h(&out_b);

        assert_eq!(a_out.len(), m * n);
        for i in 0..m * n {
            assert_eq!(
                a_out[i].to_bits(),
                b_out[i].to_bits(),
                "dedup consume diverged from prologue quantize at [{i}]: {} vs {}",
                a_out[i],
                b_out[i]
            );
        }

        // Arm C: GRIM_DEDUP_QUANT=0 forces the prologue even when
        // q81_ready=true — the documented rollback arm. It must re-quantize
        // onto the same bits as arm A.
        let saved = std::env::var("GRIM_DEDUP_QUANT").ok();
        unsafe {
            std::env::set_var("GRIM_DEDUP_QUANT", "0");
        }
        let mut out_c = mk_out();
        run_linear(&mut out_c, true);
        unsafe {
            match saved {
                Some(v) => std::env::set_var("GRIM_DEDUP_QUANT", v),
                None => std::env::remove_var("GRIM_DEDUP_QUANT"),
            }
        }
        let c_out = d2h(&out_c);
        for i in 0..m * n {
            assert_eq!(
                a_out[i].to_bits(),
                c_out[i].to_bits(),
                "GRIM_DEDUP_QUANT=0 rollback arm diverged at [{i}]"
            );
        }

        eprintln!(
            "[dedup-parity] Q5_K m={m} n={n} k={k}: consume==prologue==DEDUP=0, {} logits bit-match",
            m * n
        );
    }
}
