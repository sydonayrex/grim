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
}
