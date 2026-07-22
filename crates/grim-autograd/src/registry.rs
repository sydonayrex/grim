//! Autograd registry holding trainable parameters and layer injection points (WI-T1).
//!
//! Integrates model geometry configs, LoRA injection point registries, and active parameter sets.

use crate::injection::{InjectionConfig, LoRAInjectionRegistry};
use crate::param::{TrainableParam, TrainableParams};
use grim_backend_cpu::cpu_tensor;
use grim_tensor::{error::Result, Shape};

/// Master registry managing autograd trainable parameters across all layers.
#[derive(Debug, Clone)]
pub struct AutogradRegistry {
    pub model_config: InjectionConfig,
    pub injection_registry: LoRAInjectionRegistry,
    pub params: TrainableParams,
}

impl AutogradRegistry {
    /// Create a new `AutogradRegistry` with initialized zero/Kaiming weights for all enabled adapters.
    pub fn new(model_config: InjectionConfig, injection_registry: LoRAInjectionRegistry) -> Result<Self> {
        let mut params = TrainableParams::new();

        for config in injection_registry.enabled() {
            let (a_rows, a_cols) = config.injection_point.lora_a_shape(&model_config, config.rank);
            let (b_rows, b_cols) = config.injection_point.lora_b_shape(&model_config, config.rank);

            let stddev = (1.0 / a_cols as f32).sqrt();
            let a_data: Vec<f32> = (0..(a_rows * a_cols))
                .map(|i| (((i % 17) as f32 / 17.0) - 0.5) * stddev)
                .collect();
            let a_tensor = cpu_tensor(a_data, Shape::new(vec![a_rows, a_cols]));
            let b_tensor = cpu_tensor(vec![0.0f32; b_rows * b_cols], Shape::new(vec![b_rows, b_cols]));

            let param_a = TrainableParam::new(config.param_id_a(), a_tensor)?;
            let param_b = TrainableParam::new(config.param_id_b(), b_tensor)?;

            params.insert(param_a);
            params.insert(param_b);
        }

        Ok(Self {
            model_config,
            injection_registry,
            params,
        })
    }

    /// Zero out all parameter gradients before starting a new step.
    pub fn zero_grads(&mut self) -> Result<()> {
        self.params.zero_all_grads()
    }
}
