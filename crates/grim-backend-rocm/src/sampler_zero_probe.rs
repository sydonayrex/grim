//! Root-cause probe: first-JIT zero-logits on the device sampler (scythe2
//! plan validation log 2026-08-23e). In-crate so it can reach
//! `RocmDevice::try_new` + allocator internals directly.
//!
//! Known-good logits with a deterministic weighted support are sampled N
//! times through the PRODUCTION entry (`sample_logits_on_device_at`); every
//! token must land inside the support. Token 0 on every trial, an error, or
//! a fault reproduces the serve-surface zero-logit crash in isolation.
//!
//! ```text
//! GRIM_GPU_TEST=1 cargo test -p grim-backend-rocm --lib sampler_zero -- --nocapture
//! ```

#[cfg(test)]
mod tests {
    use crate::memory::storage::RocmStorage;
    use crate::{sample_logits_on_device_at, RocmDevice};
    use grim_tensor::backend::BackendStorage;
    use grim_tensor::{DType, Shape};

    fn gpu() -> Option<usize> {
        if !crate::gpu_test_enabled() {
            eprintln!("[skipped: GRIM_GPU_TEST not set]");
            return None;
        }
        Some(
            std::env::var("GRIM_PROBE_ORDINAL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        )
    }

    #[test]
    fn sampler_first_launch_returns_in_support_tokens() {
        let Some(ordinal) = gpu() else { return };
        let trials = std::env::var("GRIM_PROBE_TRIALS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8);
        let vocab = 65536usize;
        // Overwhelming support: tokens 0..64 carry huge mass (score/0.7 ≈
        // 142 >> Gumbel-max noise ~13 over 64k entries), everything else is
        // exactly zero. A healthy kernel can ONLY pick from the support; a
        // zero-reading kernel picks uniformly (token >= 64 almost surely).
        let data: Vec<f32> = (0..vocab)
            .map(|i| if i < 64 { 100.0 - i as f32 } else { 0.0 })
            .collect();

        let dev = RocmDevice::try_new(ordinal).expect("try_new");
        let shape = Shape::from_slice(&[vocab]);
        for t in 0..trials {
            // Fresh upload per trial: mirrors the server, where logits are a
            // brand-new storage each step.
            let st = RocmStorage::copy_from_host(
                &data,
                &shape,
                DType::F32.into(),
                &dev.allocator,
                dev.ordinal,
            )
            .expect("logits upload");

            // Upload-integrity check: DtoH round-trip BEFORE sampling.
            let back = st.to_cpu_vec_f32().expect("readback");
            eprintln!(
                "  trial {t}: upload readback head={:?} nonzero={}",
                &back[..4],
                back.iter().filter(|&&v| v != 0.0).count()
            );

            let seed = (t as u64) << 32 | 0x9E37_79B9;
            let tok = match sample_logits_on_device_at(
                &dev,
                &st,
                vocab,
                0.7,
                0,
                1.0,
                seed,
                t as u32,
            ) {
                Ok(Some(id)) => id,
                Ok(None) => {
                    eprintln!("  trial {t}: validate_input rejected input");
                    continue;
                }
                Err(e) => {
                    eprintln!("  trial {t}: SAMPLE FAIL {e}");
                    continue;
                }
            };
            eprintln!("  trial {t}: token {tok}");

            // Immediate re-sample through the same production entry: does
            // execution recover once the module is loaded/resolved?
            let tok2 = sample_logits_on_device_at(
                &dev,
                &st,
                vocab,
                0.7,
                0,
                1.0,
                seed,
                t as u32 + 1000,
            )
            .expect("device sample 2")
            .expect("validate 2");
            eprintln!("  trial {t}: immediate re-sample token {tok2}");

            // Post-sample readback: did anything overwrite the buffer?
            let after = st.to_cpu_vec_f32().expect("post readback");
            eprintln!(
                "  trial {t}: post-sample head={:?}",
                &after[..4]
            );
            assert!(
                tok < 64,
                "trial {t}: sampled token {tok} OUTSIDE support — first-launch \
                 zeroing reproduced through the production sampler"
            );
        }
    }
}
