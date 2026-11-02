use grim_core::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    Norm,
    Fallback,
}

pub struct NormSpec {
    pub eps: f32,
}

pub trait NormEmitter {
    fn record_norm(&mut self, kind: NormKind, eps: f32);
}

/// The eps -> NORM / FALLBACK rule (|eps - 1e-5| < f32::EPSILON).
pub fn emit_norm<E: NormEmitter>(
    emitter: &mut E,
    spec: &NormSpec,
    _m: usize,
    _k: usize,
) -> Result<NormKind> {
    let kind = if (spec.eps - 1e-5f32).abs() <= f32::EPSILON {
        NormKind::Norm
    } else {
        NormKind::Fallback
    };
    emitter.record_norm(kind, spec.eps);
    Ok(kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestEmitter {
        recorded: Vec<(NormKind, f32)>,
    }

    impl NormEmitter for TestEmitter {
        fn record_norm(&mut self, kind: NormKind, eps: f32) {
            self.recorded.push((kind, eps));
        }
    }

    #[test]
    fn test_emit_norm_eps_routing() {
        let mut emitter = TestEmitter { recorded: vec![] };

        let spec_exact = NormSpec { eps: 1e-5 };
        let res = emit_norm(&mut emitter, &spec_exact, 1, 1).unwrap();
        assert_eq!(res, NormKind::Norm);
        assert_eq!(emitter.recorded[0], (NormKind::Norm, 1e-5));

        let spec_fallback = NormSpec { eps: 9e-6 };
        let res2 = emit_norm(&mut emitter, &spec_fallback, 1, 1).unwrap();
        assert_eq!(res2, NormKind::Fallback);
        assert_eq!(emitter.recorded[1], (NormKind::Fallback, 9e-6));
    }
}
