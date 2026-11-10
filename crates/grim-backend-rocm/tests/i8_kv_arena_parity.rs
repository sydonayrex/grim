//! INT8 KV arena parity at Bonsai-27B geometry (16 heads / 4 kv heads /
//! head_dim 256), in the strata `kv_q8_parity.cpp` three-part methodology:
//!   1. codes BITWISE equal to the host reference (same f32 rounding);
//!   2. scales BITWISE equal to the host reference (f32, exact);
//!   3. dequantized value within half a step of the original f32
//!      (dequant = (code-128)/127*scale, step = scale/127).
//!
//! The int8 arena convention (grim_kv_quantize_rows_i8): per (row, kv_head)
//! vector, scale = maxabs (f32), codes biased +128, one denominator per
//! vector. This is the format that matches the reference token-for-token
//! (Gate D int8 control), so its bitwise behavior is worth pinning.
use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_backend_rocm::device::util::as_rocm;
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::{CoreTensorOps, Shape};

type TestResult<R = ()> = Result<R, Box<dyn std::error::Error + Send + Sync>>;

fn gpu_device() -> Option<RocmDevice> {
    if !grim_backend_rocm::gpu_test_enabled() {
        return None;
    }
    RocmDevice::try_new(0).ok()
}

fn f32ty() -> DType {
    DType { arith: ArithType::F32, storage: Storage::Native }
}
fn u8ty() -> DType {
    DType { arith: ArithType::U8, storage: Storage::Native }
}

struct Lcg(u64);
impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

#[test]
#[ignore = "needs a ROCm device; run with GRIM_GPU_TEST=1"]
fn i8_kv_arena_bitwise_parity_at_27b_geometry() -> TestResult {
    let dev = match gpu_device() {
        Some(d) => d,
        None => return Ok(()),
    };
    let _lock = grim_backend_rocm::device::util::take_gpu_test_lock_logged();

    let (nh, nkv, hd) = (16_usize, 4_usize, 256_usize);
    let rows = 8_usize;
    let stride = nkv * hd;
    let _ = nh; // query geometry is irrelevant for the append gate

    let mut rng = Lcg(0x18A8);
    let mut k_f32 = vec![0.0f32; rows * stride];
    for v in k_f32.iter_mut() {
        *v = rng.f32();
    }

    let k_codes = dev.zeros(&Shape::from_slice(&[rows * stride]), u8ty())?;
    let k_scales = dev.zeros(&Shape::from_slice(&[rows * nkv]), f32ty())?;
    for r in 0..rows {
        let row = dev.from_cpu(
            &k_f32[r * stride..(r + 1) * stride],
            &Shape::from_slice(&[1, stride]),
            f32ty(),
        )?;
        let pos = dev.from_cpu(
            &[f32::from_bits(r as i32 as u32)],
            &Shape::from_slice(&[1]),
            f32ty(),
        )?;
        grim_backend_rocm::launch_kv_append_i8(
            &dev,
            row.as_ref(),
            pos.as_ref(),
            k_codes.as_ref(),
            k_scales.as_ref(),
            stride,
            nkv,
            hd,
            1,
            1,
            stride,
            nkv,
            grim_backend_rocm::KvRingPolicy::UNLIMITED,
        )?;
    }
    dev.synchronize();

    let codes = as_rocm(k_codes.as_ref()).unwrap().copy_to_host().unwrap();
    let scales_b = as_rocm(k_scales.as_ref()).unwrap().copy_to_host().unwrap();
    let scales: Vec<f32> = scales_b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // ── three-part gate.
    let mut code_mismatch = 0usize;
    let mut scale_mismatch = 0usize;
    let mut step_worst: f32 = 0.0;
    let mut step_worst_at = 0usize;
    for r in 0..rows {
        for kh in 0..nkv {
            // 2. scale bitwise: kernel stores maxabs as f32.
            let amax = (0..hd)
                .map(|d| k_f32[r * stride + kh * hd + d].abs())
                .fold(0.0f32, f32::max);
            let want_scale = if amax > 1e-30 { amax } else { 1.0 };
            if scales[r * nkv + kh].to_bits() != want_scale.to_bits() {
                scale_mismatch += 1;
            }
            let inv = if want_scale > 1e-30 { 127.0 / want_scale } else { 0.0 };
            for d in 0..hd {
                let x = k_f32[r * stride + kh * hd + d];
                let q = (x * inv).round().clamp(-127.0, 127.0) as i16;
                let want_code = (q + 128) as u8;
                let got = codes[r * stride + kh * hd + d];
                if got != want_code {
                    code_mismatch += 1;
                }
                // 3. dequant within half a step (step = scale/127).
                let xprime = (got as i16 - 128) as f32 / 127.0 * scales[r * nkv + kh];
                let step = want_scale / 127.0;
                let err = (xprime - x).abs();
                if err > step_worst {
                    step_worst = err;
                    step_worst_at = r * stride + kh * hd + d;
                }
                assert!(
                    err <= step * 0.5 + 1e-6,
                    "dequant error {err:.3e} exceeds half a step ({:.3e}) at r={r} kh={kh} d={d}",
                    step * 0.5
                );
            }
        }
    }
    eprintln!(
        "[i8-arena-parity] strata gate: code mismatches {code_mismatch}/{} (bitwise), \
         scale mismatches {scale_mismatch}/{} (bitwise), \
         dequant worst {step_worst:.3e} at {step_worst_at} (bound half-step)",
        rows * stride,
        rows * nkv
    );
    assert_eq!(code_mismatch, 0, "codes must be bitwise equal to the host reference");
    assert_eq!(scale_mismatch, 0, "scales must be bitwise equal to the host reference");
    Ok(())
}
