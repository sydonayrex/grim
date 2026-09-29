//! Does `grim_dot4_q4k_q81_gemv` actually mis-compute on RDNA3/4?
//!
//! The production dispatch gates this kernel to RDNA2:
//!
//! ```ignore
//! // grim_dot4_q4k_q81_gemv is written and verified for the RDNA2 APU
//! // (gfx103x) sdot4 path only -- on RDNA3/4 it mis-computes (scale-
//! shuffle skew). Other arches take the WMMA / scalar fused-dequant path.
//! if is_rdna2 && m == 1 && ... { this kernel }
//! ```
//!
//! That claim has never been tested, because the parity tests for this kernel
//! (`dot_gemv_parity.rs`) are themselves gated `if !dev.gpu_target_str().starts_with("gfx103")`
//! and skip everywhere else. So the belief is self-reinforcing: the only tests
//! that could refute it refuse to run on the hardware in question.
//!
//! This test runs it on whatever device is present. If it is correct on
//! gfx1200, the gate is stale and the Q4_K decode path on RDNA3/4 is leaving a
//! working vector-dot kernel unused in favour of a scalar kernel that measures
//! 0.2-3.0 GB/s -- which is what makes the B7 kill criterion a foregone
//! conclusion rather than a measurement.
//!
//! The oracle is built from the *same quantized activations the kernel
//! consumes*, not the original f32: the kernel multiplies Q8_1 codes against
//! Q4_K codes, so scoring it against f32 activations would measure the
//! activation quantizer instead.
//!
//! RUN: HIP_VISIBLE_DEVICES=1 GRIM_GPU_TEST=1 \
//!      cargo test -p grim-backend-rocm --test dot4_q4k_arch_probe -- --nocapture

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, BackendStorage, DType, MemoryOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / (1u32 << 24) as f32;
        (unit - 0.5) * 2.0 * 3.0
    }
}

fn f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

#[test]
fn dot4_q4k_gemv_on_this_arch() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    println!("target: {}", dev.gpu_target_str());

    // K % 256 == 0 (Q4_K superblock). M = 1, the decode shape the gate applies to.
    const M: usize = 1;
    let mut rng = Lcg(0xA11CE);

    for &k in &[256usize, 1024, 4096] {
        let n = 64usize;
        let a: Vec<f32> = (0..M * k).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();

        // --- A: use the real GPU q8_1 quantizer, as production does --------
        let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
        let u8ty = DType { arith: ArithType::U8, storage: Storage::Native };
        let a_st = MemoryOps::from_cpu_bytes(&dev, &f32_bytes(&a), &Shape::new(vec![M * k]), f32ty)
            .map_err(|e| format!("a h2d: {e}"))?;
        let q81_len = M * (k / 32) * 36;
        let q81_st = MemoryOps::from_cpu_bytes(&dev, &vec![0u8; q81_len], &Shape::new(vec![q81_len]), u8ty.clone())
            .map_err(|e| format!("q81 alloc: {e}"))?;
        dev.launch_quantize_q8_1(
            grim_backend_rocm::as_rocm(a_st.as_ref()).unwrap(),
            grim_backend_rocm::as_rocm(q81_st.as_ref()).unwrap(),
            M,
            k,
        )
        .map_err(|e| format!("quantize: {e}"))?;
        dev.synchronize();
        let q81 = bytes_of(&q81_st)?;

        // --- B: Q4_K via grim's own packer --------------------------------
        let row_bytes = (k / 256) * 144;
        let mut bq = vec![0u8; n * row_bytes];
        for col in 0..n {
            let packed = grim_quant::quant_q4k(&b[col * k..(col + 1) * k])
                .map_err(|e| format!("quant_q4k: {e}"))?;
            bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
        }
        let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
            .map_err(|e| format!("b h2d: {e}"))?;
        let out_st = MemoryOps::alloc_storage(
            &dev,
            &Shape::new(vec![M * n]),
            DType { arith: ArithType::F32, storage: Storage::Native },
        )
        .map_err(|e| format!("out alloc: {e}"))?;

        dev.launch_dot4_q4k_q81_gemv_for_ab(
            grim_backend_rocm::as_rocm(q81_st.as_ref()).unwrap(),
            grim_backend_rocm::as_rocm(b_st.as_ref()).unwrap(),
            grim_backend_rocm::as_rocm(out_st.as_ref()).unwrap(),
            M,
            n,
            k,
        )
        .map_err(|e| format!("launch: {e}"))?;
        dev.synchronize();
        let got = f32_vec(&out_st)?;

        // --- oracle from the SAME q8_1 codes and Q4_K bytes ---------------
        let a_deq = dequant_q81(&q81, M * k);
        let mut worst = 0.0f32;
        for col in 0..n {
            let w = grim_quant::dequant_q4k(&bq[col * row_bytes..(col + 1) * row_bytes], k)
                .map_err(|e| format!("dequant_q4k: {e}"))?;
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_deq[kk] * w[kk];
            }
            let d = (got[col] - acc).abs();
            worst = worst.max(d / acc.abs().max(1e-3));
        }
        println!("k={k:<6} worst relative error {worst:.3e}");
        assert!(
            worst <= 2e-2,
            "k={k}: grim_dot4_q4k_q81_gemv disagrees with the oracle by {worst:.3e} on {}",
            dev.gpu_target_str()
        );
    }
    Ok(())
}

/// Decode the packed q8_1 activation layout the kernel consumes: per 32-value
/// block, an f16 scale, an f16 sum (unused here), then 32 i8 codes.
fn dequant_q81(packed: &[u8], len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    for blk in 0..len / 32 {
        let b = &packed[blk * 36..blk * 36 + 36];
        let d = f16(f16_from_le(&b[0..2]));
        for i in 0..32 {
            out[blk * 32 + i] = d * (b[4 + i] as i8) as f32;
        }
    }
    out
}

fn f16_from_le(b: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

fn bytes_of(t: &Box<dyn BackendStorage>) -> TestResult<Vec<u8>> {
    Ok(grim_backend_rocm::as_rocm(t.as_ref())
        .map_err(|e| e.to_string())?
        .copy_to_host()?)
}
fn f32_vec(t: &Box<dyn BackendStorage>) -> TestResult<Vec<f32>> {
    Ok(bytes_of(t)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// The same question for the other four k-quant schemes whose dispatch carries
/// the identical unfalsified "RDNA2 only" gate: Q2_K, Q3_K, Q5_K, Q6_K.
///
/// Q5_K and Q6_K have real packers in grim_quant, so they are tested against
/// data the production quantizer produced. Q2_K and Q3_K have *dequantizers
/// but no packers* -- those formats arrive from GGUF, not from grim -- so their
/// fixtures are synthesized: random-but-structurally-valid superblocks with the
/// f16 scale fields pinned to sane magnitudes. That is checked against the host
/// dequantizer producing finite values before the kernel is run, so a fixture
/// that is not a valid block fails loudly rather than silently comparing
/// garbage. Random codes also cover more of the kernel's input space than
/// quantized-realistic data would.
#[test]
fn dot4_kquant_arch_probe() -> TestResult {
    let Some(dev) = gpu_device() else {
        eprintln!("SKIP: GRIM_GPU_TEST unset");
        return Ok(());
    };
    println!("target: {}", dev.gpu_target_str());

    const M: usize = 1;
    const N: usize = 64;
    let mut rng = Lcg(0xB0BA_FEED);

    // (name, superblock bytes, packer, kernel entry)
    let schemes: [(&str, usize, PackKind); 4] = [
        ("Q2_K", 84, PackKind::Synth(0x2A)),
        ("Q3_K", 110, PackKind::Synth(0x3A)),
        ("Q5_K", 176, PackKind::Real(quant_q5k)),
        ("Q6_K", 210, PackKind::Real(quant_q6k)),
    ];

    for (name, sb_bytes, kind) in schemes {
        for &k in &[256usize, 1024] {
            let a: Vec<f32> = (0..M * k).map(|_| rng.next_f32()).collect();
            let b: Vec<f32> = (0..N * k).map(|_| rng.next_f32()).collect();

            let row_bytes = (k / 256) * sb_bytes;
            let mut bq = vec![0u8; N * row_bytes];
            for col in 0..N {
                let packed = match kind {
                    PackKind::Real(f) => f(&b[col * k..(col + 1) * k])
                        .map_err(|e| format!("{name} pack col {col}: {e}"))?,
                    PackKind::Synth(seed) => {
                        // One superblock per 256 weights, not one per row.
                        let row = &b[col * k..(col + 1) * k];
                        let mut out = Vec::with_capacity(row_bytes);
                        for sb_i in 0..k / 256 {
                            let lo = sb_i * 256;
                            out.extend_from_slice(&synth_superblock(
                                &row[lo..lo + 256],
                                sb_bytes,
                                seed ^ (sb_i as u32).wrapping_mul(0x9E37_79B9),
                            ));
                        }
                        out
                    }
                };
                assert_eq!(packed.len(), row_bytes, "{name} row length");
                bq[col * row_bytes..(col + 1) * row_bytes].copy_from_slice(&packed);
            }

            // Sanity: the host dequantizer must read our fixture as finite
            // weights, else we would be comparing two kinds of garbage.
            let w = dequant_scheme(name, &bq[..row_bytes], k)?;
            assert!(
                w.iter().all(|v| v.is_finite()),
                "{name}: synthesized fixture does not dequantize to finite weights"
            );

            // A: real GPU q8_1 quantizer, as production uses.
            let f32ty = DType { arith: ArithType::F32, storage: Storage::Native };
            let u8ty = DType { arith: ArithType::U8, storage: Storage::Native };
            let a_st = MemoryOps::from_cpu_bytes(&dev, &f32_bytes(&a), &Shape::new(vec![M * k]), f32ty)
                .map_err(|e| format!("a h2d: {e}"))?;
            let q81_len = M * (k / 32) * 36;
            let q81_st = MemoryOps::from_cpu_bytes(&dev, &vec![0u8; q81_len], &Shape::new(vec![q81_len]), u8ty.clone())
                .map_err(|e| format!("q81: {e}"))?;
            dev.launch_quantize_q8_1(
                grim_backend_rocm::as_rocm(a_st.as_ref()).unwrap(),
                grim_backend_rocm::as_rocm(q81_st.as_ref()).unwrap(),
                M, k,
            ).map_err(|e| format!("quantize: {e}"))?;
            dev.synchronize();
            let q81 = bytes_of(&q81_st)?;
            let a_deq = dequant_q81(&q81, M * k);

            let b_st = MemoryOps::from_cpu_bytes(&dev, &bq, &Shape::new(vec![bq.len()]), u8ty)
                .map_err(|e| format!("b h2d: {e}"))?;
            let out_st = MemoryOps::alloc_storage(
                &dev, &Shape::new(vec![M * N]),
                DType { arith: ArithType::F32, storage: Storage::Native },
            ).map_err(|e| format!("out: {e}"))?;

            fn r(t: &Box<dyn BackendStorage>) -> &grim_backend_rocm::RocmStorage {
                grim_backend_rocm::as_rocm(t.as_ref()).unwrap()
            }
            let got = match name {
                "Q2_K" => { dev.launch_dot4_q2k_q81_gemv_for_ab(r(&q81_st), r(&b_st), r(&out_st), M, N, k).map_err(|e| format!("q2k: {e}"))?; }
                "Q3_K" => { dev.launch_dot4_q3k_q81_gemv_for_ab(r(&q81_st), r(&b_st), r(&out_st), M, N, k).map_err(|e| format!("q3k: {e}"))?; }
                "Q5_K" => { dev.launch_dot4_q5k_q81_gemv_for_ab(r(&q81_st), r(&b_st), r(&out_st), M, N, k).map_err(|e| format!("q5k: {e}"))?; }
                _ => { dev.launch_dot4_q6k_q81_gemv_for_ab(r(&q81_st), r(&b_st), r(&out_st), M, N, k).map_err(|e| format!("q6k: {e}"))?; }
            };
            let _ = got;
            dev.synchronize();
            let out = f32_vec(&out_st)?;

            let mut worst = 0.0f32;
            for col in 0..N {
                let w = dequant_scheme(name, &bq[col * row_bytes..(col + 1) * row_bytes], k)?;
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_deq[kk] * w[kk];
                }
                worst = worst.max((out[col] - acc).abs() / acc.abs().max(1e-3));
            }
            println!("{name} k={k:<6} worst relative error {worst:.3e}");
            assert!(worst <= 2e-2, "{name} k={k}: disagrees with the oracle by {worst:.3e} on {}", dev.gpu_target_str());
        }
    }
    Ok(())
}

type PackFn = fn(&[f32]) -> Result<Vec<u8>, String>;
enum PackKind {
    Real(PackFn),
    Synth(u32),
}
fn quant_q5k(v: &[f32]) -> Result<Vec<u8>, String> {
    grim_quant::quant_q5k(v).map_err(|e| e.to_string())
}
fn quant_q6k(v: &[f32]) -> Result<Vec<u8>, String> {
    grim_quant::quant_q6k(v).map_err(|e| e.to_string())
}

/// Build structurally-valid superblocks: deterministic pseudo-random code
/// planes, with the f16 scale fields pinned so the dequantizer yields finite
/// weights. `scale`/`dmin` placement follows ggml's block layout per scheme.
fn synth_superblock(row: &[f32], sb_bytes: usize, seed: u32) -> Vec<u8> {
    let mut st = seed ^ 0x9E37_79B9;
    let mut next = move || {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        st
    };
    let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = if amax > 1e-6 { (amax / 31.0).min(60.0) } else { 0.01 };
    let dmin = d * 0.25;
    let mut sb = vec![0u8; sb_bytes];
    for b in sb.iter_mut() {
        *b = (next() >> 13) as u8;
    }
    match sb_bytes {
        // Q2_K: scales[16] + qs[64] + d(2) + dmin(2)
        84 => {
            for b in sb.iter_mut().take(16) { *b &= 0x0F; }
            sb[82..84].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            sb[80..82].copy_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        }
        // Q3_K: hmask[32] + qs[64] + scales[12] + d(2)
        110 => {
            sb[108..110].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        }
        other => panic!("synth_superblock: unhandled superblock size {other}"),
    }
    sb
}

fn dequant_scheme(name: &str, bytes: &[u8], k: usize) -> TestResult<Vec<f32>> {
    Ok(match name {
        "Q2_K" => grim_quant::dequant_q2k(bytes, k).map_err(|e| e.to_string())?,
        "Q3_K" => grim_quant::dequant_q3k(bytes, k).map_err(|e| e.to_string())?,
        "Q5_K" => grim_quant::dequant_q5k(bytes, k).map_err(|e| e.to_string())?,
        _ => grim_quant::dequant_q6k(bytes, k).map_err(|e| e.to_string())?,
    })
}
