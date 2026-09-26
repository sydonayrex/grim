//! GPU parity for the latent-absorbed MLA **prefill** kernel.
//!
//! `grim_mla_absorbed_prefill` is the causal, query-blocked sibling of
//! `grim_mla_absorbed_decode`. Both attend in latent space against the packed
//! `[kv_len, kv_lora_rank + qk_rope_dim]` cache and emit the *normalized latent*
//! (prefill) or the value-projected output (decode). The prefill kernel's
//! correctness rests on three things that are easy to get wrong and impossible
//! to eyeball in a disassembly:
//!
//!   1. the causal bound — query `qi` must see exactly `q_abs_offset + qi + 1`
//!      cache rows, not the whole cache;
//!   2. the per-(query, head) indexing into `q_absorbed` / `q_rope` / `out`;
//!   3. the online-softmax accumulator rescaling (`alpha` on the running sum and
//!      on the latent accumulator) staying in step with the running max.
//!
// So this checks the kernel against a direct host implementation of the same
//! definition, across a range of `(q_len, kv_len, q_abs_offset, num_heads)`.

use grim_backend_rocm::{CoreTensorOps, RocmDevice};
use grim_tensor::dtype::DType;
use grim_tensor::{AttentionOps, Shape};

fn pseudo_random(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 9) as f32 / 8_388_608.0) - 0.5
        })
        .collect()
}

/// Direct transcription of the kernel's semantics, in plain host code.
#[allow(clippy::too_many_arguments)]
fn ref_prefill(
    q_absorbed: &[f32],
    q_rope: &[f32],
    kv_cache: &[f32],
    q_len: usize,
    num_heads: usize,
    rank: usize,
    rope_d: usize,
    q_abs_offset: usize,
    kv_len: usize,
    inv_sqrt_d: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; q_len * num_heads * rank];
    for qi in 0..q_len {
        for h in 0..num_heads {
            let qh = qi * num_heads + h;
            let q_c = &q_absorbed[qh * rank..(qh + 1) * rank];
            let q_r = &q_rope[qh * rope_d..(qh + 1) * rope_d];
            let last = (q_abs_offset + qi + 1).min(kv_len);

            let mut scores: Vec<f32> = Vec::with_capacity(last);
            for j in 0..last {
                let base = j * (rank + rope_d);
                let mut s = 0.0f32;
                for c in 0..rank {
                    s += q_c[c] * kv_cache[base + c];
                }
                for r in 0..rope_d {
                    s += q_r[r] * kv_cache[base + rank + r];
                }
                scores.push(s * inv_sqrt_d);
            }
            // Plain (non-streaming) softmax reference for the same value.
            let max = scores.iter().copied().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for c in 0..rank {
                let mut acc = 0.0f32;
                for (j, e) in exps.iter().enumerate() {
                    acc += e * kv_cache[j * (rank + rope_d) + c];
                }
                out[qh * rank + c] = acc * inv;
            }
        }
    }
    out
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
        }
        assert!(d < tol, "{what}[{i}]: {x} vs {y} (delta {d} >= {tol})");
    }
    println!("{what}: max abs delta = {worst:e}");
}

/// Xing4.0's real head/rank geometry, plus smaller shapes to vary the block map.
#[test]
#[ignore = "requires a visible ROCm device"]
fn mla_absorbed_prefill_gpu_matches_host() {
    let dev = RocmDevice::new(0);

    // (q_len, num_heads, rank, rope_d, q_abs_offset, kv_len)
    // The first entry is Xing4.0's real per-layer geometry (32 heads, rank 512,
    // rope 64) at a modest query block; the rest vary the causal geometry.
    let cases: [(usize, usize, usize, usize, usize, usize); 5] = [
        (7, 32, 512, 64, 0, 7),    // cold prefill, full causal triangle
        (3, 4, 512, 64, 0, 3),     // small head count
        (5, 4, 128, 16, 4, 9),     // warm cache: queries sit mid-cache
        (1, 8, 64, 8, 6, 7),       // single query against a warm cache
        (4, 2, 256, 32, 0, 11),    // query block shorter than the cache
    ];

    for (case, &(q_len, num_heads, rank, rope_d, q_abs_offset, kv_len)) in
        cases.iter().enumerate()
    {
        let q_absorbed = pseudo_random(q_len * num_heads * rank, 11 + case as u32);
        let q_rope = pseudo_random(q_len * num_heads * rope_d, 23 + case as u32);
        let kv_cache = pseudo_random(kv_len * (rank + rope_d), 37 + case as u32);
        let inv_sqrt_d = 1.0f32 / ((rank + rope_d) as f32).sqrt();

        let qa = dev
            .from_cpu(&q_absorbed, &Shape::new(vec![q_len, num_heads, rank]), DType::F32)
            .unwrap();
        let qr = dev
            .from_cpu(&q_rope, &Shape::new(vec![q_len, num_heads, rope_d]), DType::F32)
            .unwrap();
        let kv = dev
            .from_cpu(&kv_cache, &Shape::new(vec![kv_len, rank + rope_d]), DType::F32)
            .unwrap();
        let mut out = dev
            .zeros(&Shape::new(vec![q_len, num_heads, rank]), DType::F32)
            .unwrap();

        let handle = dev
            .mla_absorbed_prefill(
                qa.as_ref(),
                qr.as_ref(),
                kv.as_ref(),
                out.as_mut(),
                q_len,
                num_heads,
                rank,
                rope_d,
                q_abs_offset,
                kv_len,
                inv_sqrt_d,
            )
            .expect("mla_absorbed_prefill");
        handle.synchronize().unwrap();

        let expected = ref_prefill(
            &q_absorbed,
            &q_rope,
            &kv_cache,
            q_len,
            num_heads,
            rank,
            rope_d,
            q_abs_offset,
            kv_len,
            inv_sqrt_d,
        );
        close(
            &out.to_cpu_vec_f32().unwrap(),
            &expected,
            2e-5,
            &format!(
                "prefill q_len={q_len} heads={num_heads} rank={rank} off={q_abs_offset} kv={kv_len}"
            ),
        );
    }
}

/// A query must never see cache rows beyond its own causal position: feeding a
/// cache whose *future* rows are enormous must not change an early query's output.
#[test]
#[ignore = "requires a visible ROCm device"]
fn prefill_respects_the_causal_bound() {
    let dev = RocmDevice::new(0);
    let (q_len, num_heads, rank, rope_d, kv_len) = (2usize, 2usize, 64usize, 8usize, 6usize);

    let q_absorbed = pseudo_random(q_len * num_heads * rank, 101);
    let q_rope = pseudo_random(q_len * num_heads * rope_d, 103);
    let inv_sqrt_d = 1.0f32 / ((rank + rope_d) as f32).sqrt();

    // Rows [0, q_len) are the shared "past"; everything after is per-variant
    // filler that no query is allowed to read.
    let past = pseudo_random(q_len * (rank + rope_d), 107);

    let run = |filler_seed: u32| -> Vec<f32> {
        let mut kv_cache = vec![0.0f32; kv_len * (rank + rope_d)];
        kv_cache[..q_len * (rank + rope_d)].copy_from_slice(&past);
        let tail = pseudo_random((kv_len - q_len) * (rank + rope_d), filler_seed);
        kv_cache[q_len * (rank + rope_d)..].copy_from_slice(&tail);

        let qa = dev
            .from_cpu(&q_absorbed, &Shape::new(vec![q_len, num_heads, rank]), DType::F32)
            .unwrap();
        let qr = dev
            .from_cpu(&q_rope, &Shape::new(vec![q_len, num_heads, rope_d]), DType::F32)
            .unwrap();
        let kv = dev
            .from_cpu(&kv_cache, &Shape::new(vec![kv_len, rank + rope_d]), DType::F32)
            .unwrap();
        let mut out = dev
            .zeros(&Shape::new(vec![q_len, num_heads, rank]), DType::F32)
            .unwrap();
        dev.mla_absorbed_prefill(
            qa.as_ref(),
            qr.as_ref(),
            kv.as_ref(),
            out.as_mut(),
            q_len,
            num_heads,
            rank,
            rope_d,
            0,
            kv_len,
            inv_sqrt_d,
        )
        .unwrap()
        .synchronize()
        .unwrap();
        out.to_cpu_vec_f32().unwrap()
    };

    // Identical past, wildly different futures: query 0 only ever sees row 0, so
    // it must be bit-identical between the two runs. Query 1 sees rows 0..2,
    // which are also identical past, so it must match too.
    let a = run(9001);
    let b = run(9002);
    // Every query here sits at or before the shared prefix, so both runs must
    // agree exactly; the tolerance only absorbs a potential FMA reassociation.
    close(&a, &b, 1e-7, "causal: future cache rows must not affect any query");
}
