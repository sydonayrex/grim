//! `kda_gated_delta_rule_scan`: the MULTI-TOKEN KDA kernel, against an f64
//! reference. It had no parity gate at all.
//!
//! `kda_batched_parity.rs` covers `kda_gated_delta_rule_step` and
//! `kda_gated_delta_rule_batched` — the ops used when `seq_len == 1`. The
//! `seq_len > 1` path in `gated_delta_net_forward_d2d` dispatches to
//! `kda_gated_delta_rule_scan`, and nothing compared it to anything. Since the
//! 5-token prefill takes exactly that path, a defect here corrupts the logits
//! for the very first generated token while every one-token gate stays green.
//!
//! The reference is the same f64 one `kda_batched_parity.rs` uses, written from
//! the published update (ICLR 2025, Eq. 10) rather than by calling the CPU
//! backend, so it is an oracle and not "CPU == ROCm" with a shared mistake.
//! Here it is driven as a SEQUENCE: each token's `reference_step` threads the
//! state forward, so a scan that resets or mis-orders the recurrence diverges.
//!
//! THE EXPERIMENTAL DESIGN IS THE POINT. `kda_gated_delta_rule_scan` reduces
//! the head norm / `ssm_norm` / `silu(z)` gate to `if (j == 0)` after a
//! `__syncthreads()`, with one thread per (value head, state row) and a grid of
//! `linear_launch(num_v * head_dim)`. So the number of BLOCKS decides whether
//! that reduction is reachable:
//!
//!   * `num_v * head_dim <= 256` -> ONE block -> every head's `j == 0` thread
//!     runs, and `__syncthreads()` means something;
//!   * `num_v * head_dim > 256`  -> SEVERAL blocks -> `j == 0` exists only in
//!     block 0, so only head 0 is ever normalised or gated.
//!
//! Both geometries are therefore run, with identical maths and a single
//! variable changed. The 9B is `num_v=32, head_dim=128` -> 4096 threads -> 16
//! blocks, i.e. squarely in the broken regime.
//!
//! Gated: `GRIM_GPU_TEST=1` + a real ROCm device.

use grim_backend_rocm::{RecurrentOps, RocmDevice};
use grim_tensor::CoreTensorOps;
use grim_tensor::{DType, Shape};
use std::sync::Arc;

const EPS: f32 = 1e-6;

fn gpu_device() -> Option<Arc<RocmDevice>> {
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return None;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    Some(Arc::new(RocmDevice::try_new(0).expect(
        "GRIM_GPU_TEST=1 is set but RocmDevice::try_new(0) failed. Failing loudly rather \
         than catch_unwind().ok(): a swallowed init failure turns this gate GREEN with \
         zero assertions run, which is how a real defect hides behind a passing test.",
    )))
}

fn linspace(n: usize, lo: f64, hi: f64) -> Vec<f32> {
    if n == 1 {
        return vec![lo as f32];
    }
    (0..n)
        .map(|i| (lo + (hi - lo) * (i as f64) / ((n - 1) as f64)) as f32)
        .collect()
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn upload(
    dev: &Arc<RocmDevice>,
    data: &[f32],
    shape: Shape,
) -> Box<dyn grim_tensor::BackendStorage> {
    CoreTensorOps::from_cpu(dev.as_ref(), data, &shape, DType::F32).expect("from_cpu")
}

/// Run the whole gate at one geometry. `seq_len` distinct streams per token so
/// a scan that resets or reorders the recurrence diverges immediately.
fn check_geometry(dev: &Arc<RocmDevice>, nv: usize, nk: usize, d: usize, seq_len: usize) -> f32 {
    let conv_dim = 2 * nk * d + nv * d;
    eprintln!(
        "[kda-scan] geometry nv={nv} nk={nk} d={d} seq_len={seq_len}: {} threads = {} block(s) of 256",
        nv * d,
        (nv * d).div_ceil(256)
    );

    // Per-token inputs. Each token gets its own stream so an out-of-order or
    // state-resetting scan cannot pass.
    let mut conv = Vec::with_capacity(seq_len * conv_dim);
    let mut alpha = Vec::with_capacity(seq_len * nv);
    let mut beta = Vec::with_capacity(seq_len * nv);
    let mut z = Vec::with_capacity(seq_len * nv * d);
    for t in 0..seq_len {
        let s = t as f64;
        conv.extend(linspace(conv_dim, -1.5 + 0.3 * s, 2.0 - 0.4 * s));
        alpha.extend(linspace(nv, -0.4 + 0.1 * s, 0.6));
        beta.extend(linspace(nv, -1.0, 1.0));
        z.extend(linspace(nv * d, -0.7 + 0.05 * s, 0.7));
    }
    let dt_bias = linspace(nv, 0.1, 0.9);
    let ssm_a = linspace(nv, 0.5, 1.5);
    // Per-HEAD width (d), not per-value-stream (nv*d): the kernel reads
    // `norm_weight[i]` for i in 0..d, and the checkpoint stores ssm_norm at
    // ssm_d_state. `z` IS nv*d, so the two differ and keeping them apart is
    // the point.
    let norm_w = linspace(d, 0.8, 1.2);

    // Non-zero seed: with a zero state the decayed key dot and the carried
    // state are indistinguishable, so a stale-state variant survives.
    let seed = linspace(nv * d * d, -0.3, 0.3);
    let state = upload(dev, &seed, Shape::new(vec![nv, d, d]));
    let out_shape = Shape::new(vec![seq_len, nv * d]);

    let (got, handle) = dev
        .kda_gated_delta_rule_scan(
            upload(dev, &conv, Shape::new(vec![seq_len, conv_dim])).as_ref(),
            upload(dev, &alpha, Shape::new(vec![seq_len, nv])).as_ref(),
            upload(dev, &beta, Shape::new(vec![seq_len, nv])).as_ref(),
            upload(dev, &dt_bias, Shape::new(vec![nv])).as_ref(),
            upload(dev, &ssm_a, Shape::new(vec![nv])).as_ref(),
            upload(dev, &norm_w, Shape::new(vec![d])).as_ref(),
            Some(upload(dev, &z, Shape::new(vec![seq_len, nv * d])).as_ref()),
            state.as_ref(),
            seq_len,
            nv,
            nk,
            d,
            EPS,
            &out_shape,
        )
        .expect("kda_gated_delta_rule_scan");
    handle.synchronize().expect("sync");
    let got = got.to_cpu_vec_f32().expect("read output");
    assert_eq!(
        got.len(),
        seq_len * nv * d,
        "output must be [seq_len, nv*d]"
    );

    // f64 oracle, token by token, threading the state.
    let mut want_state: Vec<f64> = seed.iter().map(|&v| v as f64).collect();
    let mut worst: f32 = 0.0;
    let mut worst_at = (0usize, 0usize, 0usize); // (token, head, dim)
    for t in 0..seq_len {
        let mut per_head_worst = 0.0f32;
        for h in 0..nv {
            let mut acc = vec![0.0f64; d];
            reference_step_head(
                &conv[t * conv_dim..(t + 1) * conv_dim],
                alpha[t * nv + h],
                beta[t * nv + h],
                h,
                &dt_bias,
                &ssm_a,
                &norm_w,
                Some(&z[t * nv * d..(t + 1) * nv * d]),
                nk,
                d,
                &mut want_state,
                &mut acc,
            );
            for i in 0..d {
                let g = got[t * nv * d + h * d + i] as f64 - acc[i];
                let rel = (g / acc[i].abs().max(1e-3)) as f32;
                if rel.abs() > per_head_worst {
                    per_head_worst = rel.abs();
                    worst_at = (t, h, i);
                }
            }
        }
        eprintln!(
            "[kda-scan] token {t}: worst rel err {per_head_worst:.3e} (head {} dim {})",
            worst_at.1, worst_at.2
        );
        if per_head_worst > worst {
            worst = per_head_worst;
        }
    }
    worst
}

/// Per-head variant of `reference_step` so alpha/beta can vary with `h`.
#[allow(clippy::too_many_arguments)]
fn reference_step_head(
    conv: &[f32],
    alpha: f32,
    beta: f32,
    h: usize,
    dt_bias: &[f32],
    ssm_a: &[f32],
    norm_w: &[f32],
    z: Option<&[f32]>,
    nk: usize,
    d: usize,
    state: &mut [f64],
    acc_out: &mut [f64],
) {
    let conv: Vec<f64> = conv
        .iter()
        .map(|&x| {
            let x = x as f64;
            x / (1.0 + (-x).exp())
        })
        .collect();
    let key_dim = nk * d;
    let eps = EPS as f64;
    let l2 = |v: &[f64]| {
        let ss: f64 = v.iter().map(|x| x * x).sum();
        let den = (ss + eps).sqrt();
        if den <= 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / den).collect()
        }
    };
    let gate = softplus(alpha as f64 + dt_bias[h] as f64) * ssm_a[h] as f64;
    let beta_val = 1.0 / (1.0 + (-(beta as f64)).exp());
    let decay = gate.exp();
    let q_raw: Vec<f64> = conv[(h % nk) * d..(h % nk) * d + d]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let k_raw: Vec<f64> = conv[key_dim + (h % nk) * d..key_dim + (h % nk) * d + d]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let qv: Vec<f64> = conv[2 * key_dim + h * d..2 * key_dim + h * d + d]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let k_l2 = l2(&k_raw);
    let q_l2 = l2(&q_raw);
    let head = &mut state[h * d * d..(h + 1) * d * d];
    let mut acc = vec![0.0f64; d];
    for j in 0..d {
        let row = &mut head[j * d..(j + 1) * d];
        let pred: f64 = k_l2
            .iter()
            .zip(row.iter())
            .map(|(k, s)| k * (decay * s))
            .sum();
        let delta = beta_val * (qv[j] - pred);
        for i in 0..d {
            row[i] = decay * row[i] + k_l2[i] * delta;
            acc[j] += q_l2[i] * row[i];
        }
        acc[j] *= 1.0 / (d as f64).sqrt();
    }
    let ss: f64 = acc.iter().map(|a| a * a).sum();
    let inv = 1.0 / (ss / d as f64 + eps).sqrt();
    for i in 0..d {
        let g = match z {
            Some(zz) => {
                let zv = zz[h * d + i] as f64;
                zv / (1.0 + (-zv).exp())
            }
            None => 1.0,
        };
        acc_out[i] = acc[i] * inv * norm_w[i] as f64 * g;
    }
}

/// One block: every head's `j == 0` thread exists and `__syncthreads()` is a
/// real barrier, so this geometry is the CONTROL and must pass.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn scan_matches_f64_when_the_geometry_fits_one_block() {
    let Some(dev) = gpu_device() else { return };
    let worst = check_geometry(&dev, 6, 2, 16, 5);
    eprintln!("[kda-scan] single-block geometry: worst {worst:.3e}");
    assert!(
        worst < 1e-3,
        "the single-block control FAILED at {worst:.3e} — the recurrence itself is wrong, \
         which is a different bug from the multi-block one this gate exists to find"
    );
}

/// Several blocks: `j == 0` exists only in block 0, so only head 0 can be
/// normalised and gated. This is the 9B's regime (32*128 = 4096 threads).
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn scan_matches_f64_when_the_geometry_spans_blocks() {
    let Some(dev) = gpu_device() else { return };
    let worst = check_geometry(&dev, 8, 2, 64, 5);
    eprintln!("[kda-scan] multi-block geometry: worst {worst:.3e}");
    assert!(
        worst < 1e-3,
        "the MULTI-BLOCK geometry failed at {worst:.3e}. `grim_kda_gated_delta_rule_scan` \
         reduces the head RMS norm / ssm_norm / silu(z) gate to `if (j == 0)` behind a \
         `__syncthreads()`, but the grid is linear_launch(num_v * head_dim) — one thread per \
         (head, state row). `j == 0` therefore exists ONLY in block 0, so every other head is \
         left un-normalised and un-gated, and the barrier does not span blocks. The single-block \
         control passes with identical maths, so the block count is the only variable. This is \
         the 9B's geometry class (32x128 = 4096 threads = 16 blocks) and it corrupts the logits \
         of the first token generated by any multi-token prefill."
    );
}

/// The 9B's ACTUAL KDA geometry: `ssm_dt_rank`=32, `ssm_n_group`=16,
/// `ssm_d_state`=128 -> 32 value heads x 128, 16 key heads.
///
/// 32 * 128 = 4096 threads = 16 blocks of 256, i.e. two whole heads per block.
/// The two smaller geometries both pass, and neither is this one, so it is
/// checked directly rather than inferred: a prefill on the 9B is exactly this
/// call, and the device chain breaks at its first KDA layer.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn scan_matches_f64_at_the_9b_geometry() {
    let Some(dev) = gpu_device() else { return };
    let worst = check_geometry(&dev, 32, 16, 128, 5);
    eprintln!("[kda-scan] 9B geometry (32x128): worst {worst:.3e}");
    assert!(
        worst < 1e-3,
        "kda_gated_delta_rule_scan is wrong at the 9B's own geometry (32 value heads x 128, \
         16 key heads; {worst:.3e}). The smaller geometries pass, so this is specific to this \
         shape — and it is the shape every 9B prefill uses."
    );
}

// ── Q8_0: the one K-quant with no reference gate at all ─────────────────────

use grim_format::gguf::{GgufDType, read_gguf, read_tensor_bytes};
use grim_tensor::{KQuantScheme, QuantOps};

const Q8_0_BLOCK_BYTES: usize = 34;
const QK8_0: usize = 32;

/// Transcription of `dequantize_row_q8_0` — `ggml/src/ggml-quants.c:553-567`.
///
/// Block is `ggml_half d` + `int8_t qs[32]` (`ggml-common.h:251-255`), 34 bytes,
/// and `y[i*32 + j] = qs[j] * d`. There is no min term and no 2-bit high part —
/// that is the whole format, and it is the reason Q8_0 is worth gating: it is
/// simple enough to get wrong in a way the K-quants with a scale/min pair
/// would hide.
///
/// Written from the reference, not from grim's kernel.
fn llama_cpp_dequantize_row_q8_0(x: &[u8], y: &mut [f32], k: usize) {
    assert_eq!(k % QK8_0, 0);
    assert_eq!(x.len(), (k / QK8_0) * Q8_0_BLOCK_BYTES);
    for i in 0..k / QK8_0 {
        let blk = &x[i * Q8_0_BLOCK_BYTES..(i + 1) * Q8_0_BLOCK_BYTES];
        let d = fp16_to_f32(blk[0], blk[1]);
        for j in 0..QK8_0 {
            y[i * QK8_0 + j] = (blk[2 + j] as i8) as f32 * d;
        }
    }
}

/// f16 -> f32, sign applied on the SUBNORMAL path (see the Q4_K note: real
/// weights carry negative subnormal scales, and a helper that drops the sign
/// silently disagrees with the device).
fn fp16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        let v = (mant as f32) * 2f32.powi(-24);
        if sign == 1 { -v } else { v }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

fn checkpoint() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("GRIM_CHECKPOINT") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for up in ["../../..", "../..", ".."] {
        let p = root
            .join(up)
            .join("models/qwen35-9b/Qwen3.5-9B-Q4_K_M.gguf");
        if p.exists() {
            return Some(p);
        }
    }
    eprintln!("[SKIP] 9B checkpoint not found (set GRIM_CHECKPOINT)");
    None
}

/// Is grim's Q8_0 DECODER right, on the real weight? Gate before blaming any
/// kernel: if the decoder is wrong then a host/device comparison is measuring
/// the decoder, not the GEMM.
#[test]
fn q8_0_dequant_matches_llama_cpp_on_the_real_alpha_weight() {
    let Some(path) = checkpoint() else { return };
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
    let file = read_gguf(&mut reader).expect("read_gguf");
    let target = file
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ssm_alpha.weight" && t.dtype == GgufDType::Q8_0)
        .expect("blk.0.ssm_alpha.weight is Q8_0");
    let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");
    let (out_n, in_n) = (target.dims[0] as usize, target.dims[1] as usize);
    let elems = out_n * in_n;
    assert_eq!(
        bytes.len(),
        (elems / QK8_0) * Q8_0_BLOCK_BYTES,
        "Q8_0 packed size"
    );

    let mut oracle = vec![0.0f32; elems];
    llama_cpp_dequantize_row_q8_0(&bytes, &mut oracle, elems);

    // grim's own decoder, reached through the storage layer the way production
    // does it (RocmStorage::to_vec_f32 on a KQuant(Q80) tensor).
    let dev = Arc::new(RocmDevice::try_new(0).expect("device"));
    let st = grim_tensor::MemoryOps::from_cpu_bytes(
        dev.as_ref(),
        &bytes,
        &Shape::new(vec![out_n, in_n]),
        DType {
            arith: grim_tensor::ArithType::U8,
            storage: grim_tensor::Storage::KQuant(KQuantScheme::Q80),
        },
    )
    .expect("upload packed Q8_0");
    let got = st.to_cpu_vec_f32().expect("dequant via the storage layer");

    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &o)) in got.iter().zip(oracle.iter()).enumerate() {
        let d = (g - o).abs();
        let rel = d / o.abs().max(1e-3);
        if rel > worst {
            worst = rel;
            at = i;
        }
    }
    eprintln!(
        "[q8_0] ssm_alpha [{out_n},{in_n}]: storage decode vs llama.cpp worst {worst:.3e} at {at}"
    );
    assert!(
        worst < 1e-5,
        "grim's Q8_0 DECODER disagrees with llama.cpp by {worst:.3e} at {at} on the real \
         ssm_alpha weight. Fix the decoder before concluding anything about the GEMM — a host \
         vs device comparison of a bad decoder measures the decoder."
    );
    eprintln!("[q8_0] decoder matches llama.cpp bit-for-bit on the real weight");
}

/// The Q8_0 GEMM, with `arith` as the last variable.
///
/// Two things this replaces a worse version of:
///   * the previous sweep read `(out_n, in_n)` from `read_gguf`'s dims, which
///     report this tensor as [4096, 32] while the BLOCK loads it as
///     [out=32, in=4096]. That transposed the problem, and since the oracle was
///     indexed the same way the two agreed and the sweep "passed" while
///     testing a computation production never performs. The geometry is now
///     asserted, not inferred.
///   * `dtype().arith` is a free variable the loader does not fix: a quantized
///     tensor is `arith = F32` with the K-quant in `storage`, while a hand
///     upload often uses U8. If the Q8_0 dispatch branches on arith, one of
///     them takes a different path — and `Linear::forward` passes the LOADER's
///     arith, so this is the one input that differs from a hand-rolled call.
#[test]
#[ignore = "device-gated: run with GRIM_GPU_TEST=1"]
fn q8_0_gemm_arith_and_m() {
    use grim_tensor::{CoreTensorOps, QuantFormat, Storage as DTypeStorage};
    if !grim_backend_rocm::gpu_test_enabled() {
        eprintln!("[SKIP] set GRIM_GPU_TEST=1 to run");
        return;
    }
    let _lock = grim_backend_rocm::device::util::gpu_test_lock();
    let Some(path) = checkpoint() else { return };
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).expect("open 9B"));
    let file = read_gguf(&mut reader).expect("read_gguf");
    let target = file
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ssm_alpha.weight" && t.dtype == GgufDType::Q8_0)
        .expect("ssm_alpha is Q8_0");
    let bytes = read_tensor_bytes(&mut reader, &file, target).expect("read tensor");

    // Asserted geometry: the block loads this as [out=32, in=4096].
    const OUT_N: usize = 32;
    const IN_N: usize = 4096;
    let elems = OUT_N * IN_N;
    assert_eq!(
        elems,
        (target.dims[0] as usize) * (target.dims[1] as usize),
        "element count is order-independent, so it still matches the file"
    );
    let mut w = vec![0.0f32; elems];
    llama_cpp_dequantize_row_q8_0(&bytes, &mut w, elems);

    let dev = Arc::new(RocmDevice::try_new(0).expect("device"));
    for (tag, arith) in [
        ("arith=U8 ", grim_tensor::ArithType::U8),
        ("arith=F32", grim_tensor::ArithType::F32),
    ] {
        let b = grim_tensor::MemoryOps::from_cpu_bytes(
            dev.as_ref(),
            &bytes,
            &Shape::new(vec![OUT_N, IN_N]),
            DType {
                arith,
                storage: DTypeStorage::KQuant(KQuantScheme::Q80),
            },
        )
        .expect("upload packed Q8_0");
        for m in [1usize, 5] {
            let a_src: Vec<f32> = (0..m * IN_N).map(|i| (i as f32 * 0.013).sin()).collect();
            let a = CoreTensorOps::from_cpu(
                dev.as_ref(),
                &a_src,
                &Shape::new(vec![m, IN_N]),
                DType::F32,
            )
            .expect("A");
            let (got, handle) = dev
                .fused_quant_gemm(
                    a.as_ref(),
                    b.as_ref(),
                    QuantFormat::Q8_0,
                    &Shape::new(vec![m, OUT_N]),
                )
                .unwrap_or_else(|e| panic!("{tag} m={m}: {e}"));
            handle.synchronize().expect("sync");
            let got = got.to_cpu_vec_f32().expect("read");
            let mut worst = 0.0f32;
            for i in 0..m {
                for j in 0..OUT_N {
                    let mut acc = 0.0f64;
                    for t in 0..IN_N {
                        acc += a_src[i * IN_N + t] as f64 * w[j * IN_N + t] as f64;
                    }
                    let d = (got[i * OUT_N + j] - acc as f32).abs() / (acc as f32).abs().max(1.0);
                    if d > worst {
                        worst = d;
                    }
                }
            }
            eprintln!("[q8_0-gemm] {tag} m={m}: max rel err {worst:.3e}");
        }
    }
}

// ── THE PRODUCTION CHAIN: scan (prefill) then batched (decode) ──────────────
//
// Every per-kernel gate here was green while 9B decode diverged from eager
// after exactly one token. The untested shape was the CHAIN: the 5-token
// prefill runs `kda_gated_delta_rule_scan` against `ssm_state_dev`, then every
// decode step runs `kda_gated_delta_rule_batched` against the state THE SCAN
// LEFT. This test runs both orders on identical inputs and states, plus the
// f64 oracle chain:
//
//   arm A: scan(t=0..5)  -> batched(t=5)   (production)
//   arm B: batched(t=0..5)                  (batched-only chain)
//   oracle: reference_step_head x 6
//
// A!=oracle localizes the break to whichever kernel wrote the state the other
// misreads; A==B!=oracle indicts both; all-equal exonerates the kernels and
// moves the search to the conv chain or the wrapper's per-token layouts.
#[test]
#[ignore]
fn scan_then_batched_chain_matches_oracle_at_9b_geometry() {
    let Some(dev) = gpu_device() else { return };
    let (nv, nk, d) = (32usize, 16usize, 128usize);
    let conv_dim = 2 * nk * d + nv * d;
    let t_scan = 5usize;
    let total = t_scan + 1usize;

    // Distinct per-token streams (same generator style as check_geometry).
    let mut conv = Vec::with_capacity(total * conv_dim);
    let mut alpha = Vec::with_capacity(total * nv);
    let mut beta = Vec::with_capacity(total * nv);
    let mut z = Vec::with_capacity(total * nv * d);
    for t in 0..total {
        let s = t as f64;
        conv.extend(linspace(conv_dim, -1.5 + 0.3 * s, 2.0 - 0.4 * s));
        alpha.extend(linspace(nv, -0.4 + 0.1 * s, 0.6));
        beta.extend(linspace(nv, -1.0, 1.0));
        z.extend(linspace(nv * d, -0.7 + 0.05 * s, 0.7));
    }
    let dt_bias = linspace(nv, 0.1, 0.9);
    let ssm_a = linspace(nv, 0.5, 1.5);
    let norm_w = linspace(d, 0.8, 1.2);
    let seed = linspace(nv * d * d, -0.3, 0.3);

    // ── oracle ──
    let mut oracle_state: Vec<f64> = seed.iter().map(|&v| v as f64).collect();
    let mut oracle_last = vec![0.0f64; nv * d];
    for t in 0..total {
        for h in 0..nv {
            let mut head_acc = vec![0.0f64; d];
            reference_step_head(
                &conv[t * conv_dim..(t + 1) * conv_dim],
                alpha[t * nv + h],
                beta[t * nv + h],
                h,
                &dt_bias,
                &ssm_a,
                &norm_w,
                Some(&z[t * nv * d..(t + 1) * nv * d]),
                nk,
                d,
                &mut oracle_state,
                &mut head_acc,
            );
            oracle_last[h * d..(h + 1) * d].copy_from_slice(&head_acc);
        }
    }

    // ── arm B: batched x 6 on a fresh copy of the seed ──
    // Uploads happen ONCE (whole sequence) and are held for the arm's
    // lifetime: an upload temporary dropped at statement end returns its block
    // to the caching allocator while the async kernel may still be queued, and
    // the next iteration's upload hands the same block out — a race the
    // per-kernel gates never hit because they upload before a single launch.
    let state_b = upload(&dev, &seed, Shape::new(vec![nv, d, d]));
    let dt_b = upload(&dev, &dt_bias, Shape::new(vec![nv]));
    let a_b = upload(&dev, &ssm_a, Shape::new(vec![nv]));
    let nw_b = upload(&dev, &norm_w, Shape::new(vec![d]));
    let mut b = vec![0.0f32; nv * d];
    for t in 0..total {
        let (out, h) = dev
            .kda_gated_delta_rule_batched(
                upload(&dev, &conv[t * conv_dim..(t + 1) * conv_dim], Shape::new(vec![conv_dim])).as_ref(),
                upload(&dev, &alpha[t * nv..(t + 1) * nv], Shape::new(vec![nv])).as_ref(),
                upload(&dev, &beta[t * nv..(t + 1) * nv], Shape::new(vec![nv])).as_ref(),
                dt_b.as_ref(),
                a_b.as_ref(),
                nw_b.as_ref(),
                Some(upload(&dev, &z[t * nv * d..(t + 1) * nv * d], Shape::new(vec![nv * d])).as_ref()),
                state_b.as_ref(),
                nv,
                nk,
                d,
                EPS,
                &Shape::new(vec![nv * d]),
            )
            .expect("batched chain");
        h.synchronize().unwrap();
        b = out.to_cpu_vec_f32().expect("read B");
    }

    // ── arm A: scan(5) -> batched(1) on one device state ──
    let state_a = upload(&dev, &seed, Shape::new(vec![nv, d, d]));
    let out_shape_scan = Shape::new(vec![t_scan, nv * d]);
    let (scan_out, h) = dev
        .kda_gated_delta_rule_scan(
            upload(&dev, &conv[..t_scan * conv_dim], Shape::new(vec![t_scan, conv_dim])).as_ref(),
            upload(&dev, &alpha[..t_scan * nv], Shape::new(vec![t_scan, nv])).as_ref(),
            upload(&dev, &beta[..t_scan * nv], Shape::new(vec![t_scan, nv])).as_ref(),
            upload(&dev, &dt_bias, Shape::new(vec![nv])).as_ref(),
            upload(&dev, &ssm_a, Shape::new(vec![nv])).as_ref(),
            upload(&dev, &norm_w, Shape::new(vec![d])).as_ref(),
            Some(upload(&dev, &z[..t_scan * nv * d], Shape::new(vec![t_scan, nv * d])).as_ref()),
            state_a.as_ref(),
            t_scan,
            nv,
            nk,
            d,
            EPS,
            &out_shape_scan,
        )
        .expect("scan in chain");
    h.synchronize().unwrap();
    drop(scan_out);
    let (a_out, h) = dev
        .kda_gated_delta_rule_batched(
            upload(&dev, &conv[t_scan * conv_dim..], Shape::new(vec![conv_dim])).as_ref(),
            upload(&dev, &alpha[t_scan * nv..], Shape::new(vec![nv])).as_ref(),
            upload(&dev, &beta[t_scan * nv..], Shape::new(vec![nv])).as_ref(),
            upload(&dev, &dt_bias, Shape::new(vec![nv])).as_ref(),
            upload(&dev, &ssm_a, Shape::new(vec![nv])).as_ref(),
            upload(&dev, &norm_w, Shape::new(vec![d])).as_ref(),
            Some(upload(&dev, &z[t_scan * nv * d..], Shape::new(vec![nv * d])).as_ref()),
            state_a.as_ref(),
            nv,
            nk,
            d,
            EPS,
            &Shape::new(vec![nv * d]),
        )
        .expect("batched after scan");
    h.synchronize().unwrap();
    let a = a_out.to_cpu_vec_f32().expect("read A");

    let rel = |got: &[f32]| -> f32 {
        got.iter()
            .zip(&oracle_last)
            .map(|(&g, &w)| ((g as f64 - w) / w.abs().max(1e-3)).abs() as f32)
            .fold(0.0f32, f32::max)
    };
    let a_err = rel(&a);
    let b_err = rel(&b);
    eprintln!("[kda-chain] scan->batched vs oracle: {a_err:.3e}; batched-only vs oracle: {b_err:.3e}");
    assert!(
        b_err < 1e-3,
        "batched-only chain diverges from the oracle ({b_err:.3e}) — the chain test's own baseline is broken"
    );
    assert!(
        a_err < 1e-3,
        "scan->batched chain diverges from the oracle ({a_err:.3e}) while batched-only matches \
         ({b_err:.3e}): the scan leaves the state in a form the batched decode step misreads — \
         this is the 9B decode divergence"
    );
}
