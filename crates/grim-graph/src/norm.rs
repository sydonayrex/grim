use grim_core::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    Norm,
    Fallback,
}

pub struct NormSpec<'a, W> {
    pub weight: Option<&'a W>,
    pub eps: f32,
}

pub trait NormEmitter<A, W, Out> {
    fn emit_norm(
        &mut self,
        input: &A,
        weight: &W,
        out: &mut Out,
        eps: f32,
        kind: NormKind,
        m: usize,
        k: usize,
    ) -> Result<()>;
}

/// The eps -> NORM / FALLBACK rule (|eps - 1e-5| < f32::EPSILON).
/// Evaluates the specification, determines NORM vs FALLBACK, and executes
/// both the launch and IR recording through `emitter`.
pub fn emit_norm<A, W, Out, E: NormEmitter<A, W, Out>>(
    emitter: &mut E,
    input: &A,
    weight: &W,
    out: &mut Out,
    eps: f32,
    m: usize,
    k: usize,
) -> Result<NormKind> {
    let kind = if (eps - 1e-5f32).abs() < f32::EPSILON {
        NormKind::Norm
    } else {
        NormKind::Fallback
    };
    emitter.emit_norm(input, weight, out, eps, kind, m, k)?;
    Ok(kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTensor(pub usize);

    struct TestEmitter {
        recorded: Vec<(NormKind, f32, usize, usize)>,
    }

    impl NormEmitter<MockTensor, MockTensor, MockTensor> for TestEmitter {
        fn emit_norm(
            &mut self,
            _input: &MockTensor,
            _weight: &MockTensor,
            _out: &mut MockTensor,
            eps: f32,
            kind: NormKind,
            m: usize,
            k: usize,
        ) -> Result<()> {
            self.recorded.push((kind, eps, m, k));
            Ok(())
        }
    }

    #[test]
    fn test_emit_norm_eps_routing() {
        let mut emitter = TestEmitter { recorded: vec![] };
        let mut out = MockTensor(0);
        let inp = MockTensor(1);
        let w = MockTensor(2);

        let res = emit_norm(&mut emitter, &inp, &w, &mut out, 1e-5, 1, 256).unwrap();
        assert_eq!(res, NormKind::Norm);
        assert_eq!(emitter.recorded[0], (NormKind::Norm, 1e-5, 1, 256));

        let res2 = emit_norm(&mut emitter, &inp, &w, &mut out, 9e-6, 1, 256).unwrap();
        assert_eq!(res2, NormKind::Fallback);
        assert_eq!(emitter.recorded[1], (NormKind::Fallback, 9e-6, 1, 256));
    }
}
