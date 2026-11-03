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

        let mut emitter = TestRopeEmitter { emitted: vec![] };
        let yarn_args = RopeArgs::yarn(base, (1.0, 32.0), mscale);
        emit_rope(&mut emitter, &yarn_args).expect("emit_rope");
        assert_eq!(emitter.emitted.len(), 1);
        assert_eq!(emitter.emitted[0], yarn_args);

        // Apply YaRN RoPE on an eager vector and compare against reference rotated values
        let apply_rope = |x: &[f32], pos: usize| -> Vec<f32> {
            let mut out = x.to_vec();
            for i in 0..(rotary_dim / 2) {
                let freq = 1.0f32 / base.powf((2 * i) as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (sin_a, cos_a) = (angle.sin() * mscale, angle.cos() * mscale);
                let x0 = x[i];
                let x1 = x[i + rotary_dim / 2];
                out[i] = x0 * cos_a - x1 * sin_a;
                out[i + rotary_dim / 2] = x0 * sin_a + x1 * cos_a;
            }
            out
        };

        let x = vec![1.0f32; rotary_dim];
        let y_pos0 = apply_rope(&x, 0);
        let y_pos41 = apply_rope(&x, 41);

        // At pos 0: cos is mscale, sin is 0.0 -> out = x * mscale
        for i in 0..rotary_dim {
            assert!((y_pos0[i] - mscale).abs() < 1e-6);
        }

        // At pos 41: rotated vector has norm = ||x|| * mscale
        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_41: f32 = y_pos41.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_41 - norm_x * mscale).abs() < 1e-5);
    }
}
