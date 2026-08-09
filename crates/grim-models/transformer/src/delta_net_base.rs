//! Stub for delta_net_base — custom chunked-attention architecture.
//!
//! delta-net-base uses entirely custom chunked Q/K/V/G/B/S tensors and a
//! bespoke `build_delta_net_chunking` forward pass. A full implementation is
//! tracked separately; this stub preserves the architecture entry so the
//! loader can recognize it.

use grim_core::error::{Error, Result};
use grim_core::model::{AdapterHandle, CausalLm, Model, ModelConfig, ModalityHint};
use grim_core::session::SessionT;
use grim_nn::TensorParallelConfig;
use grim_tensor::{ArithType, Device, Tensor};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DeltaNetBaseConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
}

impl ModelConfig for DeltaNetBaseConfig {
    fn name(&self) -> &str {
        "delta-net-base"
    }
    fn modality(&self) -> ModalityHint {
        ModalityHint::TextInTextOut
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Model — stub
// ---------------------------------------------------------------------------

pub struct DeltaNetBase {
    pub cfg: DeltaNetBaseConfig,
    pub device: Device,
}

impl DeltaNetBase {
    pub fn load(device: Device, _ws: &grim_nn::WeightSource<'_>, cfg: DeltaNetBaseConfig) -> Result<Self> {
        Err(Error::Unimplemented(
            "delta-net-base architecture requires a custom forward pass; not yet implemented".into(),
        ))
    }

    pub fn load_tp(
        device: Device,
        _ws: &grim_nn::WeightSource<'_>,
        cfg: DeltaNetBaseConfig,
        _tp: TensorParallelConfig,
    ) -> Result<Self> {
        Err(Error::Unimplemented(
            "delta-net-base architecture requires a custom forward pass; not yet implemented".into(),
        ))
    }
}

impl Model for DeltaNetBase {
    fn config(&self) -> &dyn ModelConfig {
        &self.cfg
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn param_arith(&self) -> ArithType {
        ArithType::F32
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CausalLm for DeltaNetBase {
    fn new_session(&self) -> Box<dyn SessionT> {
        panic!("delta-net-base not yet implemented")
    }

    fn forward(
        &self,
        _session: &mut dyn SessionT,
        _input_ids: &Tensor,
        _positions: &Tensor,
        _adapters: &[AdapterHandle],
    ) -> Result<Tensor> {
        Err(Error::Unimplemented(
            "delta-net-base forward pass not yet implemented".into(),
        ))
    }
}
