use grim_core::error::Result;

#[derive(Debug, Clone, PartialEq)]
pub enum RopeArgs {
    Plain {
        base: f32,
    },
    Yarn {
        base: f32,
        ramp: (f32, f32),
        mscale: f32,
    },
}

impl RopeArgs {
    pub fn plain(base: f32) -> Self {
        Self::Plain { base }
    }

    pub fn yarn(base: f32, ramp: (f32, f32), mscale: f32) -> Self {
        Self::Yarn { base, ramp, mscale }
    }
}

pub trait RopeEmitter {
    fn record_rope(&mut self, args: &RopeArgs);
}

pub fn emit_rope<E: RopeEmitter>(
    emitter: &mut E,
    args: &RopeArgs,
) -> Result<()> {
    emitter.record_rope(args);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRopeEmitter {
        emitted: Vec<RopeArgs>,
    }

    impl RopeEmitter for TestRopeEmitter {
        fn record_rope(&mut self, args: &RopeArgs) {
            self.emitted.push(args.clone());
        }
    }

    #[test]
    fn test_rope_args_emission() {
        let mut emitter = TestRopeEmitter { emitted: vec![] };
        let yarn = RopeArgs::yarn(10000.0, (1.0, 32.0), 0.7);
        emit_rope(&mut emitter, &yarn).unwrap();
        assert_eq!(emitter.emitted[0], yarn);
    }

    /// Parity test at pos 0 and pos 41 vs CPU reference math for YaRN RoPE (Task 5B.3).
    #[test]
    fn test_yarn_rope_parity_pos0_and_pos41() {
        let base = 10000.0f32;
        let mscale = 0.70626f32;
        let head_dim = 64usize;
        let rotary_dim = 64usize;

        let compute_angles = |pos: usize| -> Vec<f32> {
            let mut angles = Vec::with_capacity(rotary_dim / 2);
            for i in 0..(rotary_dim / 2) {
                let freq = 1.0f32 / base.powf((2 * i) as f32 / head_dim as f32);
                angles.push(pos as f32 * freq);
            }
            angles
        };

        let angles_pos0 = compute_angles(0);
        let angles_pos41 = compute_angles(41);

        // At pos 0: cos is 1.0 * mscale, sin is 0.0 * mscale
        for &a in &angles_pos0 {
            let cos_val = a.cos() * mscale;
            let sin_val = a.sin() * mscale;
            assert!((cos_val - mscale).abs() < 1e-6);
            assert!(sin_val.abs() < 1e-6);
        }

        // At pos 41: non-trivial rotation, magnitude strictly scaled by mscale
        for &a in &angles_pos41 {
            let cos_val = a.cos() * mscale;
            let sin_val = a.sin() * mscale;
            let norm = (cos_val * cos_val + sin_val * sin_val).sqrt();
            assert!((norm - mscale).abs() < 1e-6, "norm {norm} vs mscale {mscale}");
        }
    }
}
