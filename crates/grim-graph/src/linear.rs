use grim_core::error::Result;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantFlags {
    pub q81_ready: bool,
    pub u4_ready: bool,
}

pub trait LinearEmitter<A: ?Sized, W: ?Sized, Out: ?Sized, Scratch: ?Sized> {
    fn emit_linear(
        &mut self,
        input: &A,
        weight: &W,
        out: &Out,
        scratch_q81: &Scratch,
        what: &str,
        layer: usize,
        flags: QuantFlags,
        record_ir: bool,
    ) -> Result<QuantFlags>;
}

/// Executes device linear projection and IR emission via `emitter`.
/// Takes `flags` by value (immutable copy) and returns the updated `QuantFlags`
/// post-launch, preventing premature state mutation that skips quantize kernels.
/// `record_ir` explicitly controls whether an IR node is emitted into the graph contract.
pub fn emit_linear_dedup<
    A: ?Sized,
    W: ?Sized,
    Out: ?Sized,
    Scratch: ?Sized,
    E: LinearEmitter<A, W, Out, Scratch>,
>(
    emitter: &mut E,
    input: &A,
    weight: &W,
    out: &Out,
    scratch_q81: &Scratch,
    what: &str,
    layer: usize,
    flags: QuantFlags,
    record_ir: bool,
) -> Result<QuantFlags> {
    emitter.emit_linear(input, weight, out, scratch_q81, what, layer, flags, record_ir)
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
            _out: &MockTensor,
            _scratch_q81: &MockTensor,
            _what: &str,
            _layer: usize,
            flags: QuantFlags,
            _record_ir: bool,
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
        let out = MockTensor;
        let scratch = MockTensor;

        let flags1 = emit_linear_dedup(
            &mut emitter,
            &inp,
            &w,
            &out,
            &scratch,
            "linear_1",
            0,
            flags,
            true,
        )
        .unwrap();
        assert_eq!(emitter.quant_launches, 1);
        assert_eq!(emitter.gemv_launches, 1);
        assert!(flags1.q81_ready);

        let flags2 = emit_linear_dedup(
            &mut emitter,
            &inp,
            &w,
            &out,
            &scratch,
            "linear_2",
            0,
            flags1,
            true,
        )
        .unwrap();
        assert_eq!(emitter.quant_launches, 1); // skipped on second launch!
        assert_eq!(emitter.gemv_launches, 2);
        assert!(flags2.q81_ready);
    }
}
