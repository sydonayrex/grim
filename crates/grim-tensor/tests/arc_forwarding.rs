//! Regression guard for the `Arc<T>` blanket `BackendDevice` forwarding.
//!
//! Every default-overridable trait method MUST be forwarded through the
//! blanket impl — a missing forward silently routes `Box<dyn
//! BackendDevice>` built from `Arc<Concrete>` to the trait default, dead
//! on arrival for any backend override (audit finding B1: 11 methods were
//! missing). The probe overrides each method with a distinctive error;
//! dispatching through the boxed Arc must reach it.

use grim_tensor::backend::{
    BackendDevice, BackendStorage, ComputeHandle, MemAdvice, RopeConfig,
    CoreTensorOps, ElementwiseOps, SamplingOps, AttentionOps, FusionOps, AutogradOps, OptimizerOps, QuantOps, RecurrentOps, CollectiveOps, MemoryOps, GraphCaptureOps,
};
use grim_tensor::dtype::{ArithType, DType, QuantProvenance};
use grim_tensor::shape::Shape;

#[derive(Debug)]
struct ProbeStorage {
    shape: Shape,
}

impl BackendStorage for ProbeStorage {
    fn dtype(&self) -> DType {
        DType {
            arith: ArithType::F32,
            storage: grim_tensor::Storage::Native,
        }
    }
    fn provenance(&self) -> QuantProvenance {
        QuantProvenance::GrimNative
    }
    fn shape(&self) -> &Shape {
        &self.shape
    }
    fn to_cpu_vec_f32(&self) -> grim_tensor::Result<Vec<f32>> {
        Ok(vec![0.0; self.shape.elem_count()])
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Minimal device: required methods return a marker error; the overrides
/// under test return `probe::<method>` errors. If the Arc blanket impl
/// stops forwarding any of them, the trait default fires instead and the
/// marker assertion fails.
#[derive(Debug, Default)]
struct ProbeDevice;

fn probe_err(method: &str) -> grim_tensor::Result<()> {
    Err(grim_tensor::Error::Backend(format!("probe::{method}")))
}

// The macro above is awkward for the varied return types; write the
// overrides directly instead — explicit and grep-friendly.
impl CoreTensorOps for ProbeDevice {

    fn zeros(&self, shape: &Shape, _dtype: DType) -> grim_tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(ProbeStorage {
            shape: shape.clone(),
        }))
    }


    fn matmul(
        &self,
        _a: &dyn BackendStorage,
        _b: &dyn BackendStorage,
        _out: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("matmul").map(|()| {
            (
                Box::new(ProbeStorage {
                    shape: Shape::new(vec![0]),
                }) as Box<dyn BackendStorage>,
                Box::new(grim_tensor::ReadyHandle) as Box<dyn ComputeHandle>,
            )
        })
    }


    fn add(
        &self,
        _a: &dyn BackendStorage,
        _b: &dyn BackendStorage,
        _out: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("add").map(|()| {
            (
                Box::new(ProbeStorage {
                    shape: Shape::new(vec![0]),
                }) as Box<dyn BackendStorage>,
                Box::new(grim_tensor::ReadyHandle) as Box<dyn ComputeHandle>,
            )
        })
    }


    fn mul(
        &self,
        _a: &dyn BackendStorage,
        _b: &dyn BackendStorage,
        _out: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("mul").map(|()| {
            (
                Box::new(ProbeStorage {
                    shape: Shape::new(vec![0]),
                }) as Box<dyn BackendStorage>,
                Box::new(grim_tensor::ReadyHandle) as Box<dyn ComputeHandle>,
            )
        })
    }


    fn silu_mul(
        &self,
        _gate: &dyn BackendStorage,
        _up: &dyn BackendStorage,
        _out: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("silu_mul").map(|()| {
            (
                Box::new(ProbeStorage {
                    shape: Shape::new(vec![0]),
                }) as Box<dyn BackendStorage>,
                Box::new(grim_tensor::ReadyHandle) as Box<dyn ComputeHandle>,
            )
        })
    }


    fn rms_norm(
        &self,
        _x: &dyn BackendStorage,
        _weight: &dyn BackendStorage,
        _eps: f32,
        _out: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("rms_norm").map(|()| {
            (
                Box::new(ProbeStorage {
                    shape: Shape::new(vec![0]),
                }) as Box<dyn BackendStorage>,
                Box::new(grim_tensor::ReadyHandle) as Box<dyn ComputeHandle>,
            )
        })
    }


    fn softmax(
        &self,
        _x: &dyn BackendStorage,
        _out: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("softmax").map(|()| {
            (
                Box::new(ProbeStorage {
                    shape: Shape::new(vec![0]),
                }) as Box<dyn BackendStorage>,
                Box::new(grim_tensor::ReadyHandle) as Box<dyn ComputeHandle>,
            )
        })
    }


    fn embedding(
        &self,
        _weight: &dyn BackendStorage,
        _indices: &[u32],
        _out: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("embedding").map(|()| {
            (
                Box::new(ProbeStorage {
                    shape: Shape::new(vec![0]),
                }) as Box<dyn BackendStorage>,
                Box::new(grim_tensor::ReadyHandle) as Box<dyn ComputeHandle>,
            )
        })
    }


    fn from_cpu(
        &self,
        data: &[f32],
        shape: &Shape,
        _dtype: DType,
    ) -> grim_tensor::Result<Box<dyn BackendStorage>> {
        let _ = data;
        Ok(Box::new(ProbeStorage {
            shape: shape.clone(),
        }))
    }


    fn advise(&self, _storage: &dyn BackendStorage, _advice: MemAdvice) -> grim_tensor::Result<()> {
        probe_err("advise")
    }
}

impl ElementwiseOps for ProbeDevice {


    #[allow(clippy::too_many_arguments)]
    fn sub(
        &self,
        _a: &dyn BackendStorage,
        _b: &dyn BackendStorage,
        _out: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("sub")?;
        unreachable!()
    }


    fn reduce_sum(&self, _x: &dyn BackendStorage) -> grim_tensor::Result<f32> {
        probe_err("reduce_sum")?;
        unreachable!()
    }


    fn reduce_max(&self, _x: &dyn BackendStorage) -> grim_tensor::Result<f32> {
        probe_err("reduce_max")?;
        unreachable!()
    }


    fn argmax(&self, _x: &dyn BackendStorage) -> grim_tensor::Result<u32> {
        probe_err("argmax")?;
        unreachable!()
    }
}

impl SamplingOps for ProbeDevice {


    // ── Overrides under test: previously NOT forwarded by the Arc impl ──

    fn sample_on_device(
        &self,
        _logits: &dyn BackendStorage,
        _temperature: f32,
        _top_p: f32,
        _top_k: u32,
        _seed: u64,
    ) -> grim_tensor::Result<u32> {
        probe_err("sample_on_device")?;
        unreachable!()
    }
}

impl AttentionOps for ProbeDevice {


    fn qkv_attention_alibi(
        &self,
        _q: &dyn BackendStorage,
        _k: &dyn BackendStorage,
        _v: &dyn BackendStorage,
        _num_kv_heads: usize,
        _kv_seq_len: usize,
        _cache_offset: u32,
        _window: Option<usize>,
        _alibi_slopes: &dyn BackendStorage,
        _out_shape: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("qkv_attention_alibi")?;
        unreachable!()
    }


    fn rerope(
        &self,
        _k: &dyn BackendStorage,
        _old_positions: &[u32],
        _new_positions: &[u32],
        _cfg: &RopeConfig,
        _out_shape: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("rerope")?;
        unreachable!()
    }


    #[allow(clippy::too_many_arguments)]
    fn mla_q_kv_norm_split(
        &self,
        _q_raw: &dyn BackendStorage,
        _kv_raw: &dyn BackendStorage,
        _q_norm_w: &dyn BackendStorage,
        _kv_norm_w: &dyn BackendStorage,
        _qk_nope_dim: usize,
        _qk_rope_dim: usize,
        _v_dim: usize,
        _eps: f32,
    ) -> grim_tensor::Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        probe_err("mla_q_kv_norm_split")?;
        unreachable!()
    }


    fn mla_absorbed_decode(
        &self,
        _q_absorbed: &dyn BackendStorage,
        _q_rope: &dyn BackendStorage,
        _kv_cache: &dyn BackendStorage,
        _w_uv: Option<&dyn BackendStorage>,
        _out: &dyn BackendStorage,
        _num_heads: usize,
        _kv_lora_rank: usize,
        _qk_rope_dim: usize,
        _v_head_dim: usize,
        _seq_len: usize,
    ) -> grim_tensor::Result<Box<dyn ComputeHandle>> {
        probe_err("mla_absorbed_decode")?;
        unreachable!()
    }
}

impl FusionOps for ProbeDevice {


    fn fused_add_rms_norm(
        &self,
        _x: &dyn BackendStorage,
        _residual: &dyn BackendStorage,
        _weight: &dyn BackendStorage,
        _eps: f32,
        _out_shape: &Shape,
    ) -> grim_tensor::Result<(
        Box<dyn BackendStorage>,
        Box<dyn BackendStorage>,
        Box<dyn ComputeHandle>,
    )> {
        probe_err("fused_add_rms_norm")?;
        unreachable!()
    }


    #[allow(clippy::too_many_arguments)]
    fn fused_mxfp4_gemm_qk_norm_rope_kv(
        &self,
        _x: &dyn BackendStorage,
        _gamma_q: &dyn BackendStorage,
        _gamma_k: &dyn BackendStorage,
        _w_codes: &dyn BackendStorage,
        _w_exps: &dyn BackendStorage,
        _q_out: Option<&dyn BackendStorage>,
        _k_cache: Option<&dyn BackendStorage>,
        _v_cache: Option<&dyn BackendStorage>,
        _out_all: Option<&dyn BackendStorage>,
        _positions: Option<&dyn BackendStorage>,
        _m: usize,
        _k: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rotary_dim: usize,
        _rope_theta: f32,
        _inv_freq: Option<&dyn BackendStorage>,
        _mscale: f32,
        _eps: f32,
        _max_seq_len: usize,
    ) -> grim_tensor::Result<Box<dyn ComputeHandle>> {
        probe_err("fused_mxfp4_gemm_qk_norm_rope_kv")?;
        unreachable!()
    }


    fn broadcast_bias(
        &self,
        _bias: &dyn BackendStorage,
        _batch: usize,
        _out_dim: usize,
        _out_shape: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("broadcast_bias")?;
        unreachable!()
    }


    fn scale_bias_epilogue(
        &self,
        _out: &dyn BackendStorage,
        _a_scale: Option<&dyn BackendStorage>,
        _b_scale: Option<&dyn BackendStorage>,
        _bias: Option<&dyn BackendStorage>,
        _batch: usize,
        _out_dim: usize,
    ) -> grim_tensor::Result<Box<dyn ComputeHandle>> {
        probe_err("scale_bias_epilogue")?;
        unreachable!()
    }
}

impl AutogradOps for ProbeDevice {
}

impl OptimizerOps for ProbeDevice {
    fn fused_adamw_step(
        &self,
        _p: &dyn BackendStorage,
        _g: &dyn BackendStorage,
        _m: &dyn BackendStorage,
        _v: &dyn BackendStorage,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _eps: f32,
        _weight_decay: f32,
        _bc1: f32,
        _bc2: f32,
        _total: usize,
    ) -> grim_tensor::Result<Box<dyn ComputeHandle>> {
        probe_err("fused_adamw_step")?;
        unreachable!()
    }

    fn fused_lion_step(
        &self,
        _p: &dyn BackendStorage,
        _g: &dyn BackendStorage,
        _exp_avg: &dyn BackendStorage,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _weight_decay: f32,
        _total: usize,
    ) -> grim_tensor::Result<Box<dyn ComputeHandle>> {
        probe_err("fused_lion_step")?;
        unreachable!()
    }

    fn fused_madam_step(
        &self,
        _p: &dyn BackendStorage,
        _g: &dyn BackendStorage,
        _m: &dyn BackendStorage,
        _v: &dyn BackendStorage,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _eps: f32,
        _gamma: f32,
        _weight_decay: f32,
        _bc1: f32,
        _bc2: f32,
        _total: usize,
    ) -> grim_tensor::Result<Box<dyn ComputeHandle>> {
        probe_err("fused_madam_step")?;
        unreachable!()
    }
}

impl QuantOps for ProbeDevice {
}

impl RecurrentOps for ProbeDevice {


    fn short_conv1d_causal_step(
        &self,
        _x: &dyn BackendStorage,
        _weight: &dyn BackendStorage,
        _bias: Option<&dyn BackendStorage>,
        _state: &dyn BackendStorage,
        _out_shape: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("short_conv1d_causal_step")?;
        unreachable!()
    }


    #[allow(clippy::too_many_arguments)]
    fn kda_gated_delta_rule_step(
        &self,
        _q: &dyn BackendStorage,
        _k: &dyn BackendStorage,
        _v: &dyn BackendStorage,
        _beta: &dyn BackendStorage,
        _a_gate: &dyn BackendStorage,
        _recurrent_state: &dyn BackendStorage,
        _d_k: usize,
        _d_v: usize,
        _out_shape: &Shape,
    ) -> grim_tensor::Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        probe_err("kda_gated_delta_rule_step")?;
        unreachable!()
    }
}

impl CollectiveOps for ProbeDevice {
}

impl MemoryOps for ProbeDevice {
}

impl GraphCaptureOps for ProbeDevice {
}

impl BackendDevice for ProbeDevice {}


/// Dispatch each previously-unforwarded method through `Box<dyn
/// BackendDevice>` built from `Arc<ProbeDevice>` and require the probe's
/// marker error — proof the override (not the trait default) ran.
#[test]
fn arc_blanket_impl_forwards_all_overridable_methods() {
    let dev: Box<dyn BackendDevice> = Box::new(std::sync::Arc::new(ProbeDevice));
    let shape = Shape::new(vec![4]);
    let s = dev.zeros(&shape, DType::F32).expect("zeros");
    let s2 = dev.from_cpu(&[1.0], &shape, DType::F32).expect("from_cpu");

    macro_rules! assert_probe {
        ($call:expr, $name:literal) => {
            match $call {
                Ok(_) => panic!("probe::{} must return its marker error", $name),
                Err(err) => assert!(
                    err.to_string().contains(concat!("probe::", $name)),
                    "expected probe::{} override to fire (Arc forward missing?) — got: {err}",
                    $name
                ),
            }
        };
    }

    assert_probe!(
        dev.sample_on_device(s.as_ref(), 0.7, 0.9, 0, 1),
        "sample_on_device"
    );
    assert_probe!(dev.sub(s.as_ref(), s.as_ref(), &shape), "sub");
    assert_probe!(dev.reduce_sum(s.as_ref()), "reduce_sum");
    assert_probe!(dev.reduce_max(s.as_ref()), "reduce_max");
    assert_probe!(dev.argmax(s.as_ref()), "argmax");
    assert_probe!(
        dev.qkv_attention_alibi(
            s.as_ref(),
            s.as_ref(),
            s.as_ref(),
            1,
            1,
            0,
            None,
            s.as_ref(),
            &shape
        ),
        "qkv_attention_alibi"
    );
    assert_probe!(
        dev.rerope(s.as_ref(), &[0], &[1], &RopeConfig::new(4, 10000.0), &shape),
        "rerope"
    );
    assert_probe!(
        dev.fused_add_rms_norm(s.as_ref(), s.as_ref(), s.as_ref(), 1e-6, &shape),
        "fused_add_rms_norm"
    );
    assert_probe!(
        dev.fused_mxfp4_gemm_qk_norm_rope_kv(
            s.as_ref(),
            s.as_ref(),
            s.as_ref(),
            s.as_ref(),
            s.as_ref(),
            None,
            None,
            None,
            None,
            None,
            1,
            1,
            1,
            1,
            4,
            4,
            10000.0,
            None,
            1.0,
            1e-6,
            16
        ),
        "fused_mxfp4_gemm_qk_norm_rope_kv"
    );
    assert_probe!(dev.broadcast_bias(s.as_ref(), 1, 4, &shape), "broadcast_bias");
    assert_probe!(
        dev.scale_bias_epilogue(s2.as_ref(), None, None, None, 1, 4),
        "scale_bias_epilogue"
    );
    assert_probe!(
        dev.short_conv1d_causal_step(s.as_ref(), s.as_ref(), None, s.as_ref(), &shape),
        "short_conv1d_causal_step"
    );
    assert_probe!(
        dev.kda_gated_delta_rule_step(
            s.as_ref(),
            s.as_ref(),
            s.as_ref(),
            s.as_ref(),
            s.as_ref(),
            s.as_ref(),
            1,
            1,
            &shape
        ),
        "kda_gated_delta_rule_step"
    );
    assert_probe!(
        dev.mla_q_kv_norm_split(s.as_ref(), s.as_ref(), s.as_ref(), s.as_ref(), 1, 1, 1, 1e-6),
        "mla_q_kv_norm_split"
    );
    assert_probe!(
        dev.mla_absorbed_decode(s.as_ref(), s.as_ref(), s.as_ref(), None, s2.as_ref(), 1, 1, 1, 1, 1),
        "mla_absorbed_decode"
    );
    // Optimizer steps: these were the methods whose forwards went missing
    // a second time after the B1 fix (they postdate the original probe) —
    // they must stay forwarded through the Arc blanket impls.
    assert_probe!(
        dev.fused_adamw_step(s.as_ref(), s.as_ref(), s.as_ref(), s.as_ref(), 1e-3, 0.9, 0.999, 1e-8, 0.0, 1.0, 1.0, 4),
        "fused_adamw_step"
    );
    assert_probe!(
        dev.fused_lion_step(s.as_ref(), s.as_ref(), s.as_ref(), 1e-3, 0.9, 0.99, 0.0, 4),
        "fused_lion_step"
    );
    assert_probe!(
        dev.fused_madam_step(s.as_ref(), s.as_ref(), s.as_ref(), s.as_ref(), 1e-3, 0.9, 0.999, 1e-8, 1.0, 0.0, 1.0, 1.0, 4),
        "fused_madam_step"
    );
}
