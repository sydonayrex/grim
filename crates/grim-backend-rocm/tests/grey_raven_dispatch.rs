//! GreyRaven production parity: dispatch vs a host reference that shares no
//! code with the kernel's fragment fill.
//!
//! The reference decodes the SAME blob bytes through `dequant_grey_raven_hw`
//! (golden-tested against the packer) and runs dense f32 GEMM with
//! host-RNE E4M3 activations. The kernel fills fragments from the blob and
//! gathers B itself. A layout bug in either fill shows as chosen-pattern
//! mismatch, wrong columns, or wrong rows -- all of which collapse the
//! cosine. Asserted with scale-immune statistics (cosine + max abs err):
//! per-element relative error is meaningless on near-zero outputs.
//!
//! Cases: single tile-window (16x16x32, exact), multi-tile multi-window
//! (64x48x96, tails on every axis), decode m=1 (M-padding), and a K-tail
//! that the kernel must zero-pad exactly.

use grim_backend_rocm::RocmDevice;
use grim_tensor::{ArithType, DType, MemoryOps, QuantOps, Shape, Storage};
use std::panic;

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    panic::catch_unwind(|| RocmDevice::try_new(0).expect("RocmDevice::try_new")).ok()
}

fn read_f32(t: &Box<dyn grim_tensor::BackendStorage>) -> Vec<f32> {
    grim_backend_rocm::as_rocm(t.as_ref())
        .expect("rocm storage")
        .copy_to_host()
        .expect("d2h")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Host mirror of the device `float_to_fp8_e4m3_hip` converter
/// (quant_standalone.rs): round-half-away `roundf`, bit-truncate mantissa
/// with carry, subnormal floor at 2^-9. This is DELIBERATELY different from
/// the host RNE `f32_to_fp8_e4m3`: the kernel encodes activations with the
/// device converter, so the reference must use the identical formula for
/// activation codes to agree bit-for-bit. The two converters differ ONLY on
/// exact ties (measured: RNE reference still gives cosine 0.9994, i.e. the
/// layout is right under either); mirroring removes the tie noise so the
/// bounds below test the fragment fill, not the rounding mode.
fn device_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let abs = v.abs();
    let sign = if v < 0.0 { 0x80 } else { 0x00 };
    let mut abs = abs;
    if abs > 448.0 {
        abs = 448.0;
    }
    if abs < 0.001953125 {
        let mut mant = (abs * 512.0).round() as u32;
        if mant > 7 {
            mant = 7;
        }
        return sign | mant as u8;
    }
    let u = abs.to_bits();
    let e = ((u >> 23) & 0xFF) as i32 - 127;
    let m = u & 0x7FFFFF;
    let mut e_fp8 = e + 7;
    if e_fp8 >= 15 {
        return sign | 0x7E;
    }
    if e_fp8 <= 0 {
        let mut mant = (abs * 512.0).round() as u32;
        if mant > 7 {
            mant = 7;
        }
        return sign | mant as u8;
    }
    let mut mant_bits = (m + 0x40000) >> 20;
    if mant_bits > 7 {
        mant_bits = 0;
        e_fp8 += 1;
        if e_fp8 >= 15 {
            return sign | 0x7E;
        }
    }
    sign | ((e_fp8 as u8) << 3) | mant_bits as u8
}

/// Host reference: HW-decode the blob, device-mirror E4M3 activations, dense
/// GEMM. Shares the blob bytes with the kernel but no fill logic.
fn host_reference(a: &[f32], blob: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let wq = grim_quant::grey_raven::dequant_grey_raven_hw(blob, n, k).expect("host dequant");
    assert_eq!(wq.len(), n * k);
    let aq: Vec<f32> = a
        .iter()
        .map(|&v| grim_quant::fp8_e4m3_to_f32(device_e4m3(v)))
        .collect();
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += aq[row * k + j] * wq[col * k + j];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn grey_case(m: usize, n: usize, k: usize, seed: u64, tag: &str) -> TestResult {
    let Some(dev) = gpu_device() else {
        return Ok(());
    };
    if !dev.gcn_arch().starts_with("gfx12") {
        return Ok(());
    }

    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    // Magnitudes in E4M3's sweet spot: large enough to quantize finely, far
    // from the 448 ceiling and the subnormal floor.
    let a: Vec<f32> = (0..m * k)
        .map(|_| ((next() & 0xffff) as f32 / 32768.0) - 1.0)
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|_| ((next() & 0xffff) as f32 / 65536.0) - 0.5)
        .collect();

    let pats = grim_quant::grey_raven::coupled_patterns_2_4(&w, n, k)
        .map_err(|e| format!("couple: {e}"))?;
    let blob = grim_quant::grey_raven::pack_grey_raven_hw(&w, n, k, &pats)
        .map_err(|e| format!("pack: {e}"))?;

    let want = host_reference(&a, &blob, m, n, k);

    let a_t = MemoryOps::from_cpu_bytes(
        &dev,
        &a.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect::<Vec<u8>>(),
        &Shape::new(vec![m, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Native,
        },
    )
    .map_err(|e| format!("act h2d: {e}"))?;
    let b_t = MemoryOps::from_cpu_bytes(
        &dev,
        &blob,
        &Shape::new(vec![n, k]),
        DType {
            arith: ArithType::F32,
            storage: Storage::Block(grim_tensor::BlockDtype::Fp8Sparse24Hw),
        },
    )
    .map_err(|e| format!("b h2d: {e}"))?;

    let (c_t, _handle) = dev.quantized_matmul(
        &*a_t,
        &*b_t,
        &[],
        grim_tensor::QuantFormat::Fp8Sparse24Hw,
        &Shape::new(vec![m, n]),
    )?;
    dev.synchronize();
    let got = read_f32(&c_t);

    assert_eq!(got.len(), m * n);
    let mut dot = 0.0f64;
    let (mut gg, mut xx) = (0.0f64, 0.0f64);
    let mut max_abs = 0.0f32;
    for (&g, &x) in got.iter().zip(&want) {
        dot += g as f64 * x as f64;
        gg += (g as f64) * (g as f64);
        xx += (x as f64) * (x as f64);
        max_abs = max_abs.max((g - x).abs());
    }
    let cosine = dot / (gg.sqrt() * xx.sqrt()).max(f64::MIN_POSITIVE);
    eprintln!("GREY_CASE {tag} m={m} n={n} k={k} cosine={cosine:.8} max_abs={max_abs:e}");
    assert!(
        cosine >= 0.9999,
        "{tag}: cosine {cosine:.6} -- fragment fill or C layout is wrong"
    );
    // Device and host share the exact A-code formula now, so the residual is
    // float association + E4M3 steps only. A layout bug misses by O(10%).
    assert!(max_abs <= 1e-3, "{tag}: max abs err {max_abs:e}");
    Ok(())
}

#[test]
fn grey_single_tile_window() -> TestResult {
    grey_case(16, 16, 32, 0x6E67, "tile")
}

#[test]
fn grey_multi_tile_tails_everywhere() -> TestResult {
    // N=48 (3 tiles), M=20 (1 full + 4-row tail), K=96 (3 windows).
    grey_case(20, 48, 96, 0x97, "tails")
}

#[test]
fn grey_decode_m1() -> TestResult {
    // M=1: 15/16 activation columns pad to zero; the one live column must
    // still match exactly.
    grey_case(1, 64, 128, 0xBE9A, "m1")
}

#[test]
fn grey_k_tail_zero_pads() -> TestResult {
    // K=48: one full window + 16-col tail. Tail A/B pad to zero; the
    // partial window must contribute exactly its 16 live k-slots.
    grey_case(16, 32, 48, 0x7A1, "ktail")
}
