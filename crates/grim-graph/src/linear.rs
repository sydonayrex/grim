use grim_core::error::Result;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantFlags {
    pub q81_ready: bool,
    pub u4_ready: bool,
}

pub trait LinearEmitter<A, W, Out, Scratch> {
    fn emit_linear(
        &mut self,
        input: &A,
        weight: &W,
        out: &mut Out,
        scratch_q81: &Scratch,
        what: &str,
        layer: usize,
        flags: QuantFlags,
    ) -> Result<QuantFlags>;
}

/// Executes device linear projection and IR emission via `emitter`.
/// Takes `flags` by value (immutable shared copy) and returns the updated `QuantFlags`
/// after the launch, preventing premature state mutation that skips quantize kernels.
pub fn emit_linear_dedup<A, W, Out, Scratch, E: LinearEmitter<A, W, Out, Scratch>>(
    emitter: &mut E,
    input: &A,
    weight: &W,
    out: &mut Out,
    scratch_q81: &Scratch,
    what: &str,
    layer: usize,
    flags: QuantFlags,
) -> Result<QuantFlags> {
    emitter.emit_linear(input, weight, out, scratch_q81, what, layer, flags)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTensor;

    struct TestLinearEmitter {
        quant_launches: usize,
        gemv_launches: usize,
    }

    impl LinearEmitter<MockTensor, MockTensor, MockTensor, MockTensor> for TestLinearEmitter {
        fn emit_linear(
            &mut self,
            _input: &MockTensor,
            _weight: &MockTensor,
            _out: &mut MockTensor,
            _scratch_q81: &MockTensor,
            _what: &str,
            _layer: usize,
            flags: QuantFlags,
        ) -> Result<QuantFlags> {
            let mut next = flags;
            if !flags.q81_ready {
                self.quant_launches += 1;
                next.q81_ready = true;
            }
            self.gemv_launches += 1;
            Ok(next)
        }
    }

    #[test]
    fn test_two_linears_share_quant() {
        let mut emitter = TestLinearEmitter {
            quant_launches: 0,
            gemv_launches: 0,
        };
        let flags = QuantFlags::default();
        let inp = MockTensor;
        let w = MockTensor;
        let mut out = MockTensor;
        let scratch = MockTensor;

        // Flags are taken by value and returned updated, so the caller cannot
        // observe flags.q81_ready == true before the first launch occurs.
        let flags1 = emit_linear_dedup(
            &mut emitter,
            &inp,
            &w,
            &mut out,
            &scratch,
            "linear_1",
            0,
            flags,
        )
        .unwrap();
        assert_eq!(emitter.quant_launches, 1);
        assert_eq!(emitter.gemv_launches, 1);
        assert!(flags1.q81_ready);

        let flags2 = emit_linear_dedup(
            &mut emitter,
            &inp,
            &w,
            &mut out,
            &scratch,
            "linear_2",
            0,
            flags1,
        )
        .unwrap();
        assert_eq!(emitter.quant_launches, 1); // skipped!
        assert_eq!(emitter.gemv_launches, 2);
        assert!(flags2.q81_ready);
    }
}
