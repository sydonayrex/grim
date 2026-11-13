//! MTP verify-window support: is a 2-row GEMV bitwise-identical to two
//! 1-row GEMVs through the EAGER dispatch (`quantized_matmul`)?
//!
//! The MTP accept path commits state from a seq_len==2 verify forward; plain
//! decode is seq_len==1 per step. Greedy `auto` vs `off` byte-identity
//! requires each M=2 projection to equal the two corresponding M=1
//! projections bitwise. This test calls the production eager dispatch at
//! M=1 (row A alone, row B alone) and at M=2 ([A; B]) on identical weights
//! and compares rows BITWISE — no tolerance.
//!
//! Also checks the activation quantizer `grim_quantize_q8_1` for
//! row-independence at m=2 (its output feeds the dot4 GEMVs).
//!
//! RUN: GRIM_GPU_TEST=1 GRIM_RUN_GPU_TEST=1 cargo test -p grim-backend-rocm \
//!      --test mtp_window_kernel_parity -- --nocapture

use grim_backend_rocm::device::util::take_gpu_test_lock_logged;
use grim_backend_rocm::RocmDevice;
use grim_tensor::{CoreTensorOps, DType, MemoryOps, QuantFormat, QuantOps, RecurrentOps, Shape, Storage};

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    grim_backend_rocm::require_gpu_device()
}

fn bit_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn rand_f32(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (((*seed >> 33) as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// One GEMM case: quantize `w` (n rows of k) with `quant`, run eager
/// `quantized_matmul` at M=1 for row a, row b, and at M=2 for [a;b].
/// Returns (row_a_m1, row_b_m1, row_a_m2, row_b_m2).
fn run_case(
    dev: &RocmDevice,
    label: &str,
    qfmt: QuantFormat,
    b_bytes: &[u8],
    a_row: &[f32],
    b_row: &[f32],
    n: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_dtype = DType {
        arith: grim_tensor::ArithType::F32,
        storage: Storage::from(qfmt),
    };
    let b_dev = MemoryOps::from_cpu_bytes(dev, b_bytes, &Shape::new(vec![b_bytes.len()]), q_dtype)
        .expect("upload weights");

    let one = |row: &[f32]| -> Vec<f32> {
        let a_dev = CoreTensorOps::from_cpu(dev, row, &Shape::new(vec![1, k]), DType::F32)
            .expect("upload act");
        let (c, handle) = dev
            .quantized_matmul(a_dev.as_ref(), b_dev.as_ref(), &[], qfmt, &Shape::new(vec![1, n]))
            .expect("m=1 matmul");
        handle.synchronize().expect("sync");
        c.to_cpu_vec_f32().expect("d2h")
    };

    let (a0, b0) = (one(a_row), one(b_row));

    let mut both = a_row.to_vec();
    both.extend_from_slice(b_row);
    let a_dev =
        CoreTensorOps::from_cpu(dev, &both, &Shape::new(vec![2, k]), DType::F32).expect("upload 2");
    let (c, handle) = dev
        .quantized_matmul(
            a_dev.as_ref(),
            b_dev.as_ref(),
            &[],
            qfmt,
            &Shape::new(vec![2, n]),
        )
        .expect("m=2 matmul");
    handle.synchronize().expect("sync");
    let c = c.to_cpu_vec_f32().expect("d2h");
    let (c_a, c_b) = (c[..n].to_vec(), c[n..].to_vec());

    let eq_a = bit_equal(&a0, &c_a);
    let eq_b = bit_equal(&b0, &c_b);
    let d_a = a0.iter().zip(&c_a).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let d_b = b0.iter().zip(&c_b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    eprintln!(
        "[{label}] m=2 rows bitwise vs m=1: rowA={eq_a} (max {d_a:.3e}) rowB={eq_b} (max {d_b:.3e})"
    );
    (a0, b0, c_a, c_b)
}

#[test]
fn m_equals_two_bitwise_matches_two_single_rows() {
    let _lock = take_gpu_test_lock_logged();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GPU");
        return;
    };
    let n = 256usize;
    let k = 512usize;
    let mut seed = 0xC0FFEEu64;
    let a_row: Vec<f32> = (0..k).map(|_| rand_f32(&mut seed)).collect();
    let b_row: Vec<f32> = (0..k).map(|_| rand_f32(&mut seed)).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|_| rand_f32(&mut seed)).collect();

    let q4k = grim_quant::quant_q4k(&b_f32).expect("q4k");
    let (a0, _, c_a, _) = run_case(&dev, "q4k", QuantFormat::Q4K, &q4k, &a_row, &b_row, n, k);
    assert!(bit_equal(&a0, &c_a), "Q4K m=2 rowA != m=1");

    let q5k = grim_quant::quant_q5k(&b_f32).expect("q5k");
    let (a0, _, c_a, _) = run_case(&dev, "q5k", QuantFormat::Q5K, &q5k, &a_row, &b_row, n, k);
    assert!(bit_equal(&a0, &c_a), "Q5K m=2 rowA != m=1");

    // IQ4XS m=1 (dot4-q81) vs per-row dequant ORACLE, at several k. Isolates
    // kernel brokenness from act quantization: if m=1 matches the oracle
    // computed with the QUANTIZED act, the dot4 kernel is fine and the act
    // quantizer is the problem; if m=1 is off from both oracles, the kernel
    // mis-computes. (2026-11-12: real-weight probe found m=1 3-7x off the
    // exact-dequant oracle at k=5120/17408 on gfx1201.)
    for kk in [256usize, 1024, 5120] {
        let nn = 256usize;
        let w_f32: Vec<f32> = (0..nn * kk).map(|_| rand_f32(&mut seed)).collect();
        let iq4 = grim_quant::quant_iq4xs(&w_f32).expect("iq4xs");
        let row_bytes = iq4.len() / nn;
        let act: Vec<f32> = (0..kk).map(|_| rand_f32(&mut seed)).collect();
        let a_dev = CoreTensorOps::from_cpu(&dev, &act, &Shape::new(vec![1, kk]), DType::F32)
            .expect("act");
        let w_dev = MemoryOps::from_cpu_bytes(
            &dev,
            &iq4,
            &Shape::new(vec![iq4.len()]),
            DType {
                arith: grim_tensor::ArithType::F32,
                storage: Storage::KQuant(grim_tensor::KQuantScheme::IQ4XS),
            },
        )
        .expect("w");
        let (o1, h1) = dev
            .quantized_matmul(
                a_dev.as_ref(),
                w_dev.as_ref(),
                &[],
                QuantFormat::Iq4Xs,
                &Shape::new(vec![1, nn]),
            )
            .expect("m1");
        h1.synchronize().expect("sync");
        let got = o1.to_cpu_vec_f32().expect("d2h");
        let mut worst_exact = 0f32;
        let mut worst_col = 0usize;
        for col in 0..nn {
            let w_row = grim_quant::dequant_iq4xs(&iq4[col * row_bytes..(col + 1) * row_bytes], kk)
                .expect("deq");
            let oracle: f32 = act.iter().zip(&w_row).map(|(x, y)| x * y).sum();
            let rel = ((got[col] - oracle).abs()) / oracle.abs().max(1e-3);
            if rel > worst_exact {
                worst_exact = rel;
                worst_col = col;
            }
        }
        eprintln!("[iq4xs-oracle] synthetic k={kk}: m=1 dot4-q81 worst rel vs exact-dequant oracle = {worst_exact:.3e} (col {worst_col})");
        assert!(worst_exact < 5e-2, "IQ4XS m=1 dot4 route mis-computes vs oracle at k={kk}: {worst_exact:.3e}");
    }
}

#[test]
fn quantize_q8_1_is_row_independent() {
    let _lock = take_gpu_test_lock_logged();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GPU");
        return;
    };
    let k = 512usize;
    let mut seed = 0xBEEFu64;
    let a_row: Vec<f32> = (0..k).map(|_| rand_f32(&mut seed)).collect();
    let b_row: Vec<f32> = (0..k).map(|_| rand_f32(&mut seed)).collect();

    let q81_bytes = |m: usize| (k / 32) * 36 * m;
    let q_dev = |m: usize| {
        MemoryOps::from_cpu_bytes(
            &dev,
            &vec![0u8; q81_bytes(m)],
            &Shape::new(vec![q81_bytes(m)]),
            DType {
                arith: grim_tensor::ArithType::U8,
                storage: Storage::Native,
            },
        )
        .expect("q81 buf")
    };

    let both = [a_row.clone(), b_row.clone()].concat();
    let src = CoreTensorOps::from_cpu(&dev, &both, &Shape::new(vec![2, k]), DType::F32)
        .expect("upload 2");
    let src_a = CoreTensorOps::from_cpu(&dev, &a_row, &Shape::new(vec![1, k]), DType::F32)
        .expect("upload a");
    let src_b = CoreTensorOps::from_cpu(&dev, &b_row, &Shape::new(vec![1, k]), DType::F32)
        .expect("upload b");

    let out2 = q_dev(2);
    let out_a = q_dev(1);
    let out_b = q_dev(1);
    let st2 = grim_backend_rocm::device::util::as_rocm(out2.as_ref()).expect("rocm out2");
    let st_a = grim_backend_rocm::device::util::as_rocm(out_a.as_ref()).expect("rocm outa");
    let st_b = grim_backend_rocm::device::util::as_rocm(out_b.as_ref()).expect("rocm outb");
    let s2 = grim_backend_rocm::device::util::as_rocm(src.as_ref()).expect("rocm src2");
    let s_a = grim_backend_rocm::device::util::as_rocm(src_a.as_ref()).expect("rocm srca");
    let s_b = grim_backend_rocm::device::util::as_rocm(src_b.as_ref()).expect("rocm srcb");
    dev.launch_quantize_q8_1(s2, st2, 2, k).expect("q2");
    dev.launch_quantize_q8_1(s_a, st_a, 1, k).expect("qa");
    dev.launch_quantize_q8_1(s_b, st_b, 1, k).expect("qb");
    dev.synchronize();

    let o2 = st2.copy_to_host().expect("d2h 2");
    let oa = st_a.copy_to_host().expect("d2h a");
    let ob = st_b.copy_to_host().expect("d2h b");
    let row_bytes = (k / 32) * 36;
    let r0 = &o2[..row_bytes];
    let r1 = &o2[row_bytes..2 * row_bytes];
    eprintln!(
        "[q81] row0 vs m=1 rowA: {}  row1 vs m=1 rowB: {}",
        r0 == &oa[..],
        r1 == &ob[..]
    );
    assert_eq!(r0, &oa[..], "quantize_q8_1 m=2 row0 != m=1");
    assert_eq!(r1, &ob[..], "quantize_q8_1 m=2 row1 != m=1");
}

#[test]
fn rms_norm_batch_two_row_bitwise_matches_batch_one() {
    let _lock = take_gpu_test_lock_logged();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GPU");
        return;
    };
    let dim = 5120usize;
    let mut seed = 0xABCDu64;
    let a_row: Vec<f32> = (0..dim).map(|_| rand_f32(&mut seed)).collect();
    let b_row: Vec<f32> = (0..dim).map(|_| rand_f32(&mut seed)).collect();
    let w: Vec<f32> = (0..dim).map(|_| 1.0 + rand_f32(&mut seed) * 0.1).collect();

    let w_dev = CoreTensorOps::from_cpu(&dev, &w, &Shape::new(vec![dim]), DType::F32).expect("w");

    let one = |row: &[f32]| -> Vec<f32> {
        let x = CoreTensorOps::from_cpu(&dev, row, &Shape::new(vec![1, dim]), DType::F32).expect("x");
        let (s, h) = CoreTensorOps::rms_norm(
            &dev,
            x.as_ref(),
            w_dev.as_ref(),
            1e-6,
            &Shape::new(vec![1, dim]),
        )
        .expect("rms m=1");
        h.synchronize().expect("sync");
        s.to_cpu_vec_f32().expect("d2h")
    };
    let (a0, b0) = (one(&a_row), one(&b_row));

    let both = [a_row.clone(), b_row.clone()].concat();
    let x = CoreTensorOps::from_cpu(&dev, &both, &Shape::new(vec![2, dim]), DType::F32).expect("x2");
    let (s, h) = CoreTensorOps::rms_norm(
        &dev,
        x.as_ref(),
        w_dev.as_ref(),
        1e-6,
        &Shape::new(vec![2, dim]),
    )
    .expect("rms m=2");
    h.synchronize().expect("sync");
    let c = s.to_cpu_vec_f32().expect("d2h");
    let (c_a, c_b) = (c[..dim].to_vec(), c[dim..].to_vec());

    let eq_a = bit_equal(&a0, &c_a);
    let eq_b = bit_equal(&b0, &c_b);
    let d_a = a0.iter().zip(&c_a).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let d_b = b0.iter().zip(&c_b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    eprintln!(
        "[rms] batch=2 rows bitwise vs batch=1: rowA={eq_a} (max {d_a:.3e}) rowB={eq_b} (max {d_b:.3e})"
    );
    assert!(eq_a, "rms_norm batch=2 rowA != batch=1 (max {d_a:.3e})");
    assert!(eq_b, "rms_norm batch=2 rowB != batch=1 (max {d_b:.3e})");
}

#[test]
fn q6k_m_equals_two_bitwise_matches_two_single_rows() {
    let _lock = take_gpu_test_lock_logged();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GPU");
        return;
    };
    let n = 256usize;
    let k = 512usize;
    let mut seed = 0x6789u64;
    let a_row: Vec<f32> = (0..k).map(|_| rand_f32(&mut seed)).collect();
    let b_row: Vec<f32> = (0..k).map(|_| rand_f32(&mut seed)).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|_| rand_f32(&mut seed)).collect();
    let q6k = grim_quant::quant_q6k(&b_f32).expect("q6k");
    let (a0, _, c_a, _) = run_case(&dev, "q6k", QuantFormat::Q6K, &q6k, &a_row, &b_row, n, k);
    assert!(bit_equal(&a0, &c_a), "Q6K m=2 rowA != m=1");
}

#[test]
fn f32_matmul_batch_two_bitwise_matches_batch_one() {
    let _lock = take_gpu_test_lock_logged();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GPU");
        return;
    };
    let n = 512usize;
    let k = 512usize;
    let mut seed = 0xF00Du64;
    let a_row: Vec<f32> = (0..k).map(|_| rand_f32(&mut seed)).collect();
    let b_row: Vec<f32> = (0..k).map(|_| rand_f32(&mut seed)).collect();
    let w: Vec<f32> = (0..n * k).map(|_| rand_f32(&mut seed)).collect();
    let w_dev = CoreTensorOps::from_cpu(&dev, &w, &Shape::new(vec![n, k]), DType::F32).expect("w");

    let one = |row: &[f32]| -> Vec<f32> {
        let a = CoreTensorOps::from_cpu(&dev, row, &Shape::new(vec![1, k]), DType::F32).expect("a");
        let (s, h) = dev.matmul(a.as_ref(), w_dev.as_ref(), &Shape::new(vec![1, n])).expect("mm1");
        h.synchronize().expect("sync");
        s.to_cpu_vec_f32().expect("d2h")
    };
    let (a0, b0) = (one(&a_row), one(&b_row));

    let both = [a_row.clone(), b_row.clone()].concat();
    let a = CoreTensorOps::from_cpu(&dev, &both, &Shape::new(vec![2, k]), DType::F32).expect("a2");
    let (s, h) = dev.matmul(a.as_ref(), w_dev.as_ref(), &Shape::new(vec![2, n])).expect("mm2");
    h.synchronize().expect("sync");
    let c = s.to_cpu_vec_f32().expect("d2h");
    let (c_a, c_b) = (c[..n].to_vec(), c[n..].to_vec());

    let eq_a = bit_equal(&a0, &c_a);
    let eq_b = bit_equal(&b0, &c_b);
    let d_a = a0.iter().zip(&c_a).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    eprintln!(
        "[f32-mm] batch=2 rows bitwise vs batch=1: rowA={eq_a} (max {d_a:.3e}) rowB={eq_b}"
    );
    assert!(eq_a && eq_b, "f32 matmul batch=2 != batch=1");
}

#[test]
fn conv1d_causal_seq_two_matches_two_single_steps() {
    let _lock = take_gpu_test_lock_logged();
    let Some(dev) = gpu_device() else {
        eprintln!("[SKIP] requires GPU");
        return;
    };
    // conv_dim channels, taps=4
    let conv_dim = 64usize;
    let taps = 4usize;
    let mut seed = 0xCAFEu64;
    let a_tok: Vec<f32> = (0..conv_dim).map(|_| rand_f32(&mut seed)).collect();
    let b_tok: Vec<f32> = (0..conv_dim).map(|_| rand_f32(&mut seed)).collect();
    let w: Vec<f32> = (0..conv_dim * taps).map(|_| rand_f32(&mut seed)).collect();
    let w_dev = CoreTensorOps::from_cpu(&dev, &w, &Shape::new(vec![conv_dim, taps]), DType::F32).expect("w");

    let state_shape = Shape::new(vec![1, taps - 1, conv_dim]);

    // Reference: two seq_len==1 calls with a carried ring.
    let st0 = CoreTensorOps::from_cpu(&dev, &vec![0.0f32; (taps - 1) * conv_dim], &state_shape, DType::F32).expect("st0");
    let (o_a, h) = dev
        .short_conv1d_causal_step(a_tok_dev(&dev, &a_tok).as_ref(), w_dev.as_ref(), None, st0.as_ref(), &Shape::new(vec![1, conv_dim]))
        .expect("conv a");
    h.synchronize().expect("sync");
    // Ring after step A is inside st0 (in place) — read it back for step B.
    let st1 = st0.to_cpu_vec_f32().expect("d2h ring");
    let st1_dev = CoreTensorOps::from_cpu(&dev, &st1, &state_shape, DType::F32).expect("st1");
    let (o_b, h) = dev
        .short_conv1d_causal_step(a_tok_dev(&dev, &b_tok).as_ref(), w_dev.as_ref(), None, st1_dev.as_ref(), &Shape::new(vec![1, conv_dim]))
        .expect("conv b");
    h.synchronize().expect("sync");
    let b0 = o_b.to_cpu_vec_f32().expect("d2h b");
    let a0 = o_a.to_cpu_vec_f32().expect("d2h a");

    // Window: one seq_len==2 call on a fresh ring.
    let st2 = CoreTensorOps::from_cpu(&dev, &vec![0.0f32; (taps - 1) * conv_dim], &state_shape, DType::F32).expect("st2");
    let both = [a_tok.clone(), b_tok.clone()].concat();
    let x2 = CoreTensorOps::from_cpu(&dev, &both, &Shape::new(vec![2, conv_dim]), DType::F32).expect("x2");
    let (c, h) = dev
        .short_conv1d_causal_step(x2.as_ref(), w_dev.as_ref(), None, st2.as_ref(), &Shape::new(vec![2, conv_dim]))
        .expect("conv 2");
    h.synchronize().expect("sync");
    let c = c.to_cpu_vec_f32().expect("d2h c");
    let (c_a, c_b) = (c[..conv_dim].to_vec(), c[conv_dim..].to_vec());

    let eq_a = bit_equal(&a0, &c_a);
    let eq_b = bit_equal(&b0, &c_b);
    let d_a = a0.iter().zip(&c_a).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let d_b = b0.iter().zip(&c_b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    eprintln!(
        "[conv1d] seq=2 rows bitwise vs 2x seq=1: rowA={eq_a} (max {d_a:.3e}) rowB={eq_b} (max {d_b:.3e})"
    );
    assert!(eq_a && eq_b, "short_conv1d seq=2 != 2x seq=1");
}

fn a_tok_dev(dev: &RocmDevice, row: &[f32]) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev, row, &Shape::new(vec![1, row.len()]), DType::F32).expect("tok")
}
