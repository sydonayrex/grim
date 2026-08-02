//! Autograd registry holding trainable parameters and layer injection points (WI-T1).
//!
//! Integrates model geometry configs, LoRA injection point registries, and active parameter sets.

use crate::AutogradScope;
use crate::injection::{InjectionConfig, LoRAInjectionPoint, LoRAInjectionRegistry};
use crate::param::{ParamId, TrainableParam, TrainableParams};
use grim_backend_cpu::cpu_tensor;
use grim_tensor::{Shape, error::Result};

/// Base weights keyed by `(layer_idx, injection_point)`, in
/// `[out_features * in_features]` row-major order. Supplied by the
/// training worker once real weights exist; PiSSA reads from this map.
pub type BaseWeightMap = std::collections::HashMap<(usize, LoRAInjectionPoint), Vec<f32>>;

/// Master registry managing autograd trainable parameters across all layers.
#[derive(Debug, Clone)]
pub struct AutogradRegistry {
    pub model_config: InjectionConfig,
    pub injection_registry: LoRAInjectionRegistry,
    pub params: TrainableParams,
    pub scope: AutogradScope,
}

impl AutogradRegistry {
    /// Create a new `AutogradRegistry` with initialized zero/Kaiming weights for all enabled adapters.
    pub fn new(
        model_config: InjectionConfig,
        injection_registry: LoRAInjectionRegistry,
    ) -> Result<Self> {
        Self::with_scope(model_config, injection_registry, AutogradScope::default())
    }

    /// Create a new `AutogradRegistry` with an explicit autograd scope.
    ///
    /// Defaults to `LoRAOnly` for QLoRA modes; `FullParameter` is used for
    /// BF16Full fine-tuning so gradients reach every base weight.
    pub fn with_scope(
        model_config: InjectionConfig,
        injection_registry: LoRAInjectionRegistry,
        scope: AutogradScope,
    ) -> Result<Self> {
        Self::with_scope_and_base_weights(model_config, injection_registry, scope, None)
    }

    /// Create a new `AutogradRegistry` with an explicit scope and optional
    /// base weights.
    ///
    /// When `base_weights` supplies the weight for an adapter whose config
    /// has `use_pissa`, the A/B matrices are initialized from the principal
    /// singular components of that weight via `pissa_initialize` instead of
    /// the default Kaiming A / zero B. Adapters without a base-weight entry
    /// (or without `use_pissa`) fall back to the default initialization.
    pub fn with_scope_and_base_weights(
        model_config: InjectionConfig,
        injection_registry: LoRAInjectionRegistry,
        scope: AutogradScope,
        base_weights: Option<&BaseWeightMap>,
    ) -> Result<Self> {
        let mut params = TrainableParams::new();

        for config in injection_registry.enabled() {
            let (a_rows, a_cols) = config
                .injection_point
                .lora_a_shape(&model_config, config.rank);
            let (b_rows, b_cols) = config
                .injection_point
                .lora_b_shape(&model_config, config.rank);

            let stddev = (1.0 / a_cols as f32).sqrt();
            let default_a: Vec<f32> = (0..(a_rows * a_cols))
                .map(|i| (((i % 17) as f32 / 17.0) - 0.5) * stddev)
                .collect();
            let zero_b: Vec<f32> = vec![0.0f32; b_rows * b_cols];

            // PiSSA: initialize A/B from the base weight's principal
            // singular components. The base weight is [out, in] =
            // [b_rows, a_cols]; pissa returns a = [rank, in],
            // b = [out, rank], matching the A/B layout above.
            let (a_data, b_data) = if config.use_pissa {
                match base_weights.and_then(|m| m.get(&(config.layer_idx, config.injection_point)))
                {
                    Some(w) => {
                        let (a, b, _quantized) =
                            crate::injection::pissa_initialize(w, b_rows, a_cols, config.rank)?;
                        (a, b)
                    }
                    // No base weight yet (pre-WI-T8 worker): fall back to
                    // the default Kaiming A / zero B rather than failing,
                    // so training still starts with plain LoRA init.
                    None => (default_a, zero_b),
                }
            } else {
                (default_a, zero_b)
            };

            let a_tensor = cpu_tensor(a_data, Shape::new(vec![a_rows, a_cols]));
            let b_tensor = cpu_tensor(b_data, Shape::new(vec![b_rows, b_cols]));

            let param_a = TrainableParam::new(config.param_id_a(), a_tensor)?;
            let param_b = TrainableParam::new(config.param_id_b(), b_tensor)?;

            params.insert(param_a);
            params.insert(param_b);
        }

        // WI-T8: when scope is FullParameter, register all base weights as
        // frozen TrainableParams so gradients flow through them during backward
        // but the optimizer skips them (base weights stay frozen in WI-T8/QLoRA).
        if scope == AutogradScope::FullParameter {
            for point in LoRAInjectionPoint::all_points() {
                let (rows, cols) = point.base_weight_shape(&model_config);
                let base_id = ParamId::base(0, *point);
                let base_tensor =
                    cpu_tensor(vec![0.0f32; rows * cols], Shape::new(vec![rows, cols]));
                let base_param = TrainableParam::register_base_weight(
                    base_id,
                    base_tensor,
                    true, // frozen — optimizer skips base weights
                )?;
                params.insert(base_param);
            }
        }

        Ok(Self {
            model_config,
            injection_registry,
            params,
            scope,
        })
    }

    /// Zero out all parameter gradients before starting a new step.
    pub fn zero_grads(&mut self) -> Result<()> {
        self.params.zero_all_grads()
    }

    /// Clone the initialized adapter registry for another data-parallel rank.
    /// Parameter tensors are cloned by value, so a rank can update its copy
    /// independently while starting from identical weights.
    pub fn fork_for_rank(&self) -> Self {
        self.clone()
    }

    /// Verify that two rank registries have identical trainable parameter
    /// topology and values before entering a synchronized step.
    pub fn assert_rank_compatible(&self, other: &Self) -> Result<()> {
        if self.params.len() != other.params.len()
            || self.params.weight_checksum()? != other.params.weight_checksum()?
        {
            return Err(grim_tensor::error::Error::Backend(
                "data-parallel rank adapter registries diverged".into(),
            ));
        }
        Ok(())
    }
}
