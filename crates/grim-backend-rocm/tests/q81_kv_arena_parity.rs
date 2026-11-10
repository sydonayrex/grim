//! Q8_1 KV arena parity at Bonsai-27B geometry (16 heads / 4 kv heads /
//! head_dim 256). The arena path — `launch_kv_append_q81` (quantize) +
//! `launch_qkv_attention_dev_batch_q81` (attention) — must reproduce an
//! oracle that dequantizes the SAME blocks on the host and runs the f32
//! attention kernel. A layout/stride bug in either kernel shows up as a
//! large, uncorrelated diff; correct quantization shows only ~1/127 noise.
//!
//! Origin: the Q8_1 default arena (2226f520) flipped Gate D token 3 while
//! f32/int8 arenas matched the reference 32/32 — this gate isolates whether
//! the Q8_1 arena path itself is numerically faithful.
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
fn q81_kv_arena_matches_f32_oracle_at_27b_geometry() -> TestResult {
    let dev = match gpu_device() {
        Some(d) => d,
        None => return Ok(()),
    };
    let _lock = grim_backend_rocm::device::util::take_gpu_test_lock_logged();

    let (nh, nkv, hd) = (16_usize, 4_usize, 256_usize);
    let rows = 40_usize;
    let stride = nkv * hd;
    let slot_stride = rows * stride; // one slot, unlimited ring

    let mut rng = Lcg(0x81A1);
    // K/V rows in arena layout: [row, kh*head_dim + d]
    let mut k_f32 = vec![0.0f32; rows * stride];
    let mut v_f32 = vec![0.0f32; rows * stride];
    for r in 0..rows {
        for kh in 0..nkv {
            for d in 0..hd {
                k_f32[r * stride + kh * hd + d] = rng.f32();
                v_f32[r * stride + kh * hd + d] = rng.f32();
            }
        }
    }
    // Q: one query per head at the last position.
    let mut q_data = vec![0.0f32; nh * hd];
    for v in q_data.iter_mut() {
        *v = rng.f32();
    }

    // ── Quantize on the host per the kernel convention (d = amax/127 per
    //    32-block, codes = round(v/d), s field = 0) to build the oracle.
    let mut k_dq = vec![0.0f32; rows * stride];
    let mut v_dq = vec![0.0f32; rows * stride];
    for r in 0..rows {
        for kh in 0..nkv {
            for blk in 0..(hd / 32) {
                let base = r * stride + kh * hd + blk * 32;
                let amax = k_f32[base..base + 32]
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax / 127.0 > 1e-30 { amax / 127.0 } else { 1e-30 };
                for e in 0..32 {
                    let q = (k_f32[base + e] / d).round().clamp(-127.0, 127.0);
                    k_dq[base + e] = q as f32 * d;
                }
                let amax = v_f32[base..base + 32]
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                let d = if amax / 127.0 > 1e-30 { amax / 127.0 } else { 1e-30 };
                for e in 0..32 {
                    let q = (v_f32[base + e] / d).round().clamp(-127.0, 127.0);
                    v_dq[base + e] = q as f32 * d;
                }
            }
        }
    }

    // ── Host softmax oracle over the dequantized values (same math as the
    //    i8 arena gate): softmax(q.k/sqrt(hd)) . v with GQA mapping.
    let mut want = vec![0.0f32; nh * hd];
    for h in 0..nh {
        let kvh = h * nkv / nh;
        let mut acc = vec![0.0f32; hd];
        let mut mx = f32::NEG_INFINITY;
        let mut scores = vec![0.0f32; rows];
        for r in 0..rows {
            let mut sc = 0.0f32;
            for d in 0..hd {
                sc += q_data[h * hd + d] * k_dq[r * stride + kvh * hd + d];
            }
            sc /= (hd as f32).sqrt();
            scores[r] = sc;
            mx = mx.max(sc);
        }
        let mut sum = 0.0f32;
        for r in 0..rows {
            let w = (scores[r] - mx).exp();
            sum += w;
            for d in 0..hd {
                acc[d] += w * v_dq[r * stride + kvh * hd + d];
            }
        }
        for d in 0..hd {
            want[h * hd + d] = acc[d] / sum;
        }
    }

    // ── Q8_1 arena path: append each row, then the q81 attention kernel.
    let arena_elems = rows * stride * 36 / 32;
    let k_arena = dev.zeros(&Shape::from_slice(&[arena_elems]), u8ty())?;
    let v_arena = dev.zeros(&Shape::from_slice(&[arena_elems]), u8ty())?;
    for r in 0..rows {
        let krow = dev.from_cpu(
            &k_f32[r * stride..(r + 1) * stride],
            &Shape::from_slice(&[1, stride]),
            f32ty(),
        )?;
        let vrow = dev.from_cpu(
            &v_f32[r * stride..(r + 1) * stride],
            &Shape::from_slice(&[1, stride]),
            f32ty(),
        )?;
        let pos_r = dev.from_cpu(
            &[f32::from_bits(r as i32 as u32)],
            &Shape::from_slice(&[1]),
            f32ty(),
        )?;
        grim_backend_rocm::launch_kv_append_q81(
            &dev,
            krow.as_ref(),
            pos_r.as_ref(),
            k_arena.as_ref(),
            stride,
            nkv,
            hd,
            1,
            1,
            slot_stride,
            grim_backend_rocm::KvRingPolicy::UNLIMITED,
        )?;
        grim_backend_rocm::launch_kv_append_q81(
            &dev,
            vrow.as_ref(),
            pos_r.as_ref(),
            v_arena.as_ref(),
            stride,
            nkv,
            hd,
            1,
            1,
            slot_stride,
            grim_backend_rocm::KvRingPolicy::UNLIMITED,
        )?;
    }
    let q_dev = dev.from_cpu(&q_data, &Shape::from_slice(&[1, nh * hd]), f32ty())?;
    let out_shape = Shape::from_slice(&[1, nh * hd]);
    let past_last = dev.from_cpu(
        &[f32::from_bits((rows - 1) as i32 as u32)],
        &Shape::from_slice(&[1]),
        f32ty(),
    )?;
    let out_q = dev.zeros(&out_shape, f32ty())?;
    let max_q = dev.zeros(&Shape::from_slice(&[1, nh]), f32ty())?;
    let sum_q = dev.zeros(&Shape::from_slice(&[1, nh]), f32ty())?;
    grim_backend_rocm::launch_qkv_attention_dev_batch_q81(
        &dev,
        q_dev.as_ref(),
        k_arena.as_ref(),
        v_arena.as_ref(),
        as_rocm(out_q.as_ref()).unwrap(),
        as_rocm(max_q.as_ref()).unwrap(),
        as_rocm(sum_q.as_ref()).unwrap(),
        past_last.as_ref(),
        nh as u32,
        nkv as u32,
        hd as u32,
        1,
        1,
        1.0 / (hd as f32).sqrt(),
        0,
        0.0,
        q_dev.as_ref(),
        0,
        0,
        q_dev.as_ref(),
        0,
        1,
        slot_stride,
        grim_backend_rocm::KvRingPolicy::UNLIMITED,
    )?;
    dev.synchronize();
    let got = out_q.to_cpu_vec_f32()?;

    // ── Localize: read the arena back, dequantize per the documented
    //    convention, and compare against the host oracle values.
    {
        let kb = as_rocm(k_arena.as_ref()).unwrap().copy_to_host().unwrap();
        let mut worst_arena = 0.0f32;
        let mut worst_arena_at = 0usize;
        for r in 0..rows {
            for kh in 0..nkv {
                for blk in 0..(hd / 32) {
                    let off = (r * stride + kh * hd + blk * 32) * 36 / 32;
                    let d = {
                        // fp16 -> f32
                        let h = u16::from_le_bytes([kb[off], kb[off + 1]]);
                        let sign = (h >> 15) & 1;
                        let exp = ((h >> 10) & 0x1f) as i32;
                        let mant = (h & 0x3ff) as f32;
                        let v = if exp == 0 {
                            mant * 2.0f32.powi(-24)
                        } else if exp == 0x1f {
                            f32::NAN
                        } else {
                            (1.0 + mant / 1024.0) * 2.0f32.powi(exp - 15)
                        };
                        if sign == 1 { -v } else { v }
                    };
                    let _ = d;
                    for e in 0..32 {
                        let code = kb[off + 4 + e] as i8;
                        let deq = code as f32 * d;
                        let want = k_dq[r * stride + kh * hd + blk * 32 + e];
                        let dd = (deq - want).abs();
                        if dd > worst_arena {
                            worst_arena = dd;
                            worst_arena_at = r * stride + kh * hd + blk * 32 + e;
                        }
                    }
                }
            }
        }
        eprintln!(
            "[q81-arena-parity] arena content vs host oracle: worst {worst_arena:.3e} at {worst_arena_at}"
        );
    }

    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    eprintln!(
        "[q81-arena-parity] oracle scale {scale:.4}, worst |diff| {worst:.3e} at {worst_at} \
         (rel {:.3e}); oracle[{}]={:.5} got={:.5}",
        worst / scale, worst_at, want[worst_at], got[worst_at]
    );
    // Correct q8_1 quantization noise is ~1/127 of the per-block amax; the
    // attention output mixes ~rows scores, so a generous 5% relative bar
    // separates "faithful" from "layout/stride bug" (which gives O(1) diffs).
    assert!(
        worst / scale < 0.05,
        "q8_1 arena path diverges from the f32 oracle: worst {worst:.3e} rel {:.3e} at {worst_at}",
        worst / scale
    );
    Ok(())
}
