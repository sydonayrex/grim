//! QSA dequant-on-gather gate (plans/PLAN-qsa-dequant-on-gather.md):
//! `grim_kv_gather_q81` must reproduce the host dequantization BITWISE
//! (same f16 conversion) and stay within half a step + fp16 rounding of
//! the original f32 — the strata `kv_q8_parity.cpp` three-part methodology
//! applied to the gather.
use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_backend_rocm::device::util::as_rocm;
use grim_tensor::dtype::{ArithType, DType, Storage};
use grim_tensor::{CoreTensorOps, MemoryOps, Shape};

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
fn f16ty() -> DType {
    DType { arith: ArithType::F16, storage: Storage::Native }
}
fn u8ty() -> DType {
    DType { arith: ArithType::U8, storage: Storage::Native }
}
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x800000;
        let shift = (14 - e) as u32;
        let mut half = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        if rem > (1 << (shift - 1)) || (rem == (1 << (shift - 1)) && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half = ((e as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1;
        if half & 0x7c00 == 0x7c00 {
            half = 0x7c00;
        }
    }
    sign | half
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        let v = (mant as f32) * 2f32.powi(-24);
        return if sign != 0 { -v } else { v };
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7F800000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
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
fn qsa_gather_q81_bitwise_parity() -> TestResult {
    let dev = match gpu_device() {
        Some(d) => d,
        None => return Ok(()),
    };
    let _lock = grim_backend_rocm::device::util::take_gpu_test_lock_logged();

    let (nkv, hd, n_rows) = (2_usize, 64_usize, 8_usize);
    let stride = nkv * hd;
    let num_blocks = hd / 32;
    let row_bytes = nkv * num_blocks * 36;
    let slot_stride = n_rows * stride; // bytes per slot arena region (row-major)

    // Build the arena on the host per the q8_1 convention.
    let mut rng = Lcg(0x0A6);
    let mut arena = vec![0u8; n_rows * row_bytes];
    let mut k_orig = vec![0.0f32; n_rows * stride];
    for r in 0..n_rows {
        for kh in 0..nkv {
            for blk in 0..num_blocks {
                let base = r * stride + kh * hd + blk * 32;
                for e in 0..32 {
                    k_orig[base + e] = rng.f32();
                }
                let amax = k_orig[base..base + 32]
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax / 127.0 > 1e-30 { amax / 127.0 } else { 1e-30 };
                let d16 = f32_to_f16_bits(d);
                let off = r * row_bytes + kh * num_blocks * 36 + blk * 36;
                arena[off..off + 2].copy_from_slice(&d16.to_le_bytes());
                // s field stays zero (grim's convention)
                for e in 0..32 {
                    let q = (k_orig[base + e] / d).round().clamp(-127.0, 127.0) as i8;
                    arena[off + 4 + e] = q as u8;
                }
            }
        }
    }
    let v_arena = arena.clone();

    let row_ids = vec![5_i32, 2, 7, 0, 3];
    let n_ids = row_ids.len();

    let arena_t = MemoryOps::from_cpu_bytes(&dev, &arena, &Shape::new(vec![arena.len()]), u8ty())?;
    let v_t = MemoryOps::from_cpu_bytes(&dev, &v_arena, &Shape::new(vec![v_arena.len()]), u8ty())?;
    // row_ids are read as `const int*`; the established pattern (i8 arena
    // gate) carries the int bit pattern in an f32 tensor.
    let ids_bits: Vec<f32> = row_ids
        .iter()
        .map(|&v| f32::from_bits(v as u32))
        .collect();
    let ids_t = CoreTensorOps::from_cpu(&dev, &ids_bits, &Shape::new(vec![n_ids]), f32ty())?;
    let dst_stride = stride;
    let k_dst = dev.zeros(&Shape::new(vec![n_ids * dst_stride]), f16ty())?;
    let v_dst = dev.zeros(&Shape::new(vec![n_ids * dst_stride]), f16ty())?;

    grim_backend_rocm::launch_kv_gather_q81(
        &dev,
        arena_t.as_ref(),
        v_t.as_ref(),
        ids_t.as_ref(),
        n_ids,
        nkv,
        hd,
        slot_stride,
        as_rocm(k_dst.as_ref()).unwrap(),
        as_rocm(v_dst.as_ref()).unwrap(),
    )?;
    dev.synchronize();

    let k_got = as_rocm(k_dst.as_ref()).unwrap().copy_to_host().unwrap();
    let _v_got = as_rocm(v_dst.as_ref()).unwrap().copy_to_host().unwrap();
    assert_eq!(k_got.len(), n_ids * dst_stride * 2);

    // Three-part gate:
    //  1. gathered fp16 BITWISE equals host dequant (code * f16(d));
    //  2. (scales are read from the arena — no separate scale op here);
    //  3. dequant within half a step + fp16 rounding of the original f32.
    let mut bit_mismatch = 0usize;
    let mut step_worst: f32 = 0.0;
    let mut step_worst_at = 0usize;
    for (gi, &arow) in row_ids.iter().enumerate() {
        let arow = arow as usize;
        for kh in 0..nkv {
            for blk in 0..num_blocks {
                let base = arow * stride + kh * hd + blk * 32;
                let amax = k_orig[base..base + 32]
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax / 127.0 > 1e-30 { amax / 127.0 } else { 1e-30 };
                let off = arow * row_bytes + kh * num_blocks * 36 + blk * 36;
                let d16 = u16::from_le_bytes([arena[off], arena[off + 1]]);
                for e in 0..32 {
                    let code = arena[off + 4 + e] as i8;
                    let want_f = code as f32 * f16_bits_to_f32(d16);
                    let want_bits = f32_to_f16_bits(want_f);
                    let gi_off = (gi * dst_stride + kh * hd + blk * 32 + e) * 2;
                    let got_bits = u16::from_le_bytes([k_got[gi_off], k_got[gi_off + 1]]);
                    if got_bits != want_bits {
                        bit_mismatch += 1;
                    }
                    let err = (want_f - k_orig[base + e]).abs();
                    if err > step_worst {
                        step_worst = err;
                        step_worst_at = base + e;
                    }
                    let bound = 0.5 * d + want_f.abs() * 2.0f32.powi(-11) + 1e-7;
                    assert!(
                        err <= bound,
                        "gather dequant error {err:.3e} exceeds bound {bound:.3e} at {base}+{e}"
                    );
                }
            }
        }
    }
    eprintln!(
        "[qsa-gather] bitwise mismatches {bit_mismatch}/{} (K), dequant worst {step_worst:.3e} at {step_worst_at} (bound half-step + fp16)",
        n_ids * stride
    );
    assert_eq!(bit_mismatch, 0, "gather must be bitwise equal to host dequant");
    Ok(())
}

#[test]
#[ignore = "needs a ROCm device; run with GRIM_GPU_TEST=1"]
fn qsa_index_score_matches_host() -> TestResult {
    let dev = match gpu_device() {
        Some(d) => d,
        None => return Ok(()),
    };
    let _lock = grim_backend_rocm::device::util::take_gpu_test_lock_logged();

    let (n_cells, idx_dim, n_idx_h) = (64_usize, 128_usize, 4_usize);
    let mut rng = Lcg(0x5C01);
    let mut keys = vec![0.0f32; n_cells * idx_dim];
    let mut q = vec![0.0f32; n_idx_h * idx_dim];
    for v in keys.iter_mut() {
        *v = rng.f32();
    }
    for v in q.iter_mut() {
        *v = rng.f32();
    }
    let keys_t = CoreTensorOps::from_cpu(&dev, &keys, &Shape::new(vec![n_cells, idx_dim]), f32ty())?;
    let q_t = CoreTensorOps::from_cpu(&dev, &q, &Shape::new(vec![n_idx_h, idx_dim]), f32ty())?;
    let scores = dev.zeros(&Shape::new(vec![n_cells]), f32ty())?;

    grim_backend_rocm::launch_qsa_index_score(
        &dev,
        keys_t.as_ref(),
        q_t.as_ref(),
        as_rocm(scores.as_ref()).unwrap(),
        n_cells,
        idx_dim,
        n_idx_h,
    )?;
    dev.synchronize();
    let got = scores.to_cpu_vec_f32()?;

    // host reference: score = sum_h relu(q_h . k_cell)
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for c in 0..n_cells {
        let mut acc = 0.0f32;
        for h in 0..n_idx_h {
            let mut dot = 0.0f32;
            for d in 0..idx_dim {
                dot += q[h * idx_dim + d] * keys[c * idx_dim + d];
            }
            acc += dot.max(0.0);
        }
        let diff = (acc - got[c]).abs();
        if diff > worst {
            worst = diff;
            at = c;
        }
    }
    eprintln!("[qsa-score] worst |host-device| {worst:.3e} at cell {at}");
    assert!(worst < 1e-4, "score kernel diverges: {worst:.3e} at cell {at}");
    Ok(())
}
