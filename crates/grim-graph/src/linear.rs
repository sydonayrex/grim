use grim_core::error::Result;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantFlags {
    pub q81_ready: bool,
    pub u4_ready: bool,
}

pub trait LinearEmitter {
    fn record_quant_q81(&mut self);
    fn record_gemv(&mut self, what: &str);
}

pub fn emit_linear_dedup<E: LinearEmitter>(
    emitter: &mut E,
    what: &str,
    flags: &mut QuantFlags,
) -> Result<()> {
    if !flags.q81_ready {
        emitter.record_quant_q81();
        flags.q81_ready = true;
    }
    emitter.record_gemv(what);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestLinearEmitter {
        quant_launches: usize,
        gemv_launches: usize,
    }

    impl LinearEmitter for TestLinearEmitter {
        fn record_quant_q81(&mut self) {
            self.quant_launches += 1;
        }
        fn record_gemv(&mut self, _what: &str) {
            self.gemv_launches += 1;
        }
    }

    #[test]
    fn test_two_linears_share_quant() {
        let mut emitter = TestLinearEmitter {
            quant_launches: 0,
            gemv_launches: 0,
        };
        let mut flags = QuantFlags::default();

        emit_linear_dedup(&mut emitter, "linear_1", &mut flags).unwrap();
        assert_eq!(emitter.quant_launches, 1);
        assert_eq!(emitter.gemv_launches, 1);
        assert!(flags.q81_ready);

        emit_linear_dedup(&mut emitter, "linear_2", &mut flags).unwrap();
        assert_eq!(emitter.quant_launches, 1); // skipped!
        assert_eq!(emitter.gemv_launches, 2);
    }
}
