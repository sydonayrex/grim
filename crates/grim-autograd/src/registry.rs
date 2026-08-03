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

            // SPECTRAL-QLORA override: use the well-conditioned 2D-dependent
            // seed formula from SoulEaterAdapter instead of the flat-index
            // default_a / zero_b, so that Newton-Schulz and Gram-Schmidt have
            // a full-rank matrix to orthogonalize. The standard LoRA defaults
            // are kept for non-SpectralQLoRA paths (A is Kaiming random,
            // B is zero so the adapter starts as identity).
            let (spectral_a, spectral_b) = if config.use_spectral_qlora {
                let s_a: Vec<f32> = (0..(a_rows * a_cols))
                    .map(|idx| {
                        let row = idx / a_cols;
                        let col = idx % a_cols;
                        ((((row + 1) * 17 + (col + 1) * 31) % 100) as f32 / 100.0) - 0.5
                    })
                    .collect();
                let s_b: Vec<f32> = (0..(b_rows * b_cols))
                    .map(|idx| {
                        let row = idx / b_cols;
                        let col = idx % b_cols;
                        ((((row + 1) * 13 + (col + 1) * 29) % 100) as f32 / 100.0) - 0.5
                    })
                    .collect();
                (s_a, s_b)
            } else {
                (default_a.clone(), zero_b.clone())
            };

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
                    // the well-conditioned spectral seed (if SpectralQLoRA) or
                    // the default Kaiming A / zero B otherwise.
                    None => {
                        if config.use_spectral_qlora {
                            (spectral_a, spectral_b)
                        } else {
                            (default_a, zero_b)
                        }
                    }
                }
            } else if config.use_spectral_qlora {
                // SpectralQLoRA: start from the well-conditioned 2D-dependent
                // seed so Newton-Schulz has a full-rank matrix to orthogonalize.
                (spectral_a, spectral_b)
            } else {
                (default_a, zero_b)
            };

            // SPECTRAL-QLORA: orthogonal adapter initialization.
            // Apply subspace Newton-Schulz orthogonalization once at adapter
            // creation so that AB is semi-orthogonal in the dominant subspace.
            // This reuses `grim-quant::soul_eater::subspace_newton_schulz_step`
            // for the Gram-matrix-based orthogonality check, matching
            // SoulEaterAdapter's init pattern. When Newton-Schulz cannot
            // converge (ill-conditioned seed or iteration cap reached), fall
            // back to modified Gram-Schmidt which always yields orthonormal
            // columns.
            let (a_data, b_data) = if config.use_spectral_qlora {
                let mut a_data = a_data;
                let mut b_data = b_data;

                // B [b_rows, b_cols] = [out, rank] is tall/thin → Newton-Schulz
                // directly to make columns orthonormal (B^T * B ≈ I).
                let ns_ok = grim_quant::soul_eater::subspace_newton_schulz_step(
                    &mut b_data,
                    b_rows,
                    b_cols,
                    10,
                );
                if ns_ok.map_or(true, |iters| iters >= 10) {
                    crate::injection::orthogonalize_columns(&mut b_data, b_rows, b_cols);
                }

                // A [a_rows, a_cols] = [rank, in] is wide/thin. Transpose to
                // [in, rank] (tall/thin), orthogonalize, then transpose back so
                // rows of A become orthonormal: A * A^T ≈ I.
                let mut a_t = vec![0.0f32; a_cols * a_rows];
                for row in 0..a_cols {
                    for col in 0..a_rows {
                        a_t[row * a_rows + col] = a_data[col * a_cols + row];
                    }
                }
                let ns_ok = grim_quant::soul_eater::subspace_newton_schulz_step(
                    &mut a_t, a_cols, a_rows, 10,
                );
                if ns_ok.map_or(true, |iters| iters >= 10) {
                    crate::injection::orthogonalize_columns(&mut a_t, a_cols, a_rows);
                }
                // Transpose back into a_data.
                for row in 0..a_cols {
                    for col in 0..a_rows {
                        a_data[col * a_cols + row] = a_t[row * a_rows + col];
                    }
                }

                (a_data, b_data)
            } else {
                (a_data, b_data)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::injection::{
        InjectionConfig, LoRAInjectionConfig, LoRAInjectionPoint, LoRAInjectionRegistry,
    };

    fn cfg() -> InjectionConfig {
        InjectionConfig {
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 32,
        }
    }

    #[test]
    fn spectral_qlora_init_makes_b_semi_orthogonal() {
        // When use_spectral_qlora is true, B [out, rank] should be initialized
        // with semi-orthogonal columns (B^T * B ≈ I).
        let cfg = cfg();
        let mut inj_reg = LoRAInjectionRegistry::new();
        let mut ic = LoRAInjectionConfig::new(
            LoRAInjectionPoint::QProj,
            0,
            1,
            4, // rank
            16.0,
        );
        ic.use_spectral_qlora = true;
        inj_reg.add(ic);

        let reg = AutogradRegistry::new(cfg, inj_reg).unwrap();

        // Get the B param for QProj layer 0, adapter 1.
        let pid_b = ParamId::b(0, 1, LoRAInjectionPoint::QProj);
        let b_param = reg.params.get(pid_b).expect("B param must exist");
        let b_data = b_param.data.to_vec_f32().unwrap();
        let b_rows = 8; // hidden_size for QProj
        let b_cols = 4; // rank

        // Compute B^T * B (r x r Gram matrix) and check it's approximately I.
        let mut gram = vec![0.0f32; b_cols * b_cols];
        for i in 0..b_cols {
            for j in 0..b_cols {
                let mut sum = 0.0f32;
                for k in 0..b_rows {
                    sum += b_data[k * b_cols + i] * b_data[k * b_cols + j];
                }
                gram[i * b_cols + j] = sum;
            }
        }

        // Check diagonal ≈ 1 and off-diagonal ≈ 0.
        for i in 0..b_cols {
            for j in 0..b_cols {
                if i == j {
                    assert!(
                        (gram[i * b_cols + j] - 1.0).abs() < 0.2,
                        "B^T*B diagonal at [{i},{j}] = {} (expected ≈1.0)",
                        gram[i * b_cols + j]
                    );
                } else {
                    assert!(
                        gram[i * b_cols + j].abs() < 0.2,
                        "B^T*B off-diagonal at [{i},{j}] = {} (expected ≈0.0)",
                        gram[i * b_cols + j]
                    );
                }
            }
        }
    }

    #[test]
    fn spectral_qlora_init_makes_a_semi_orthogonal() {
        // When use_spectral_qlora is true, A [rank, in] should have
        // orthonormal rows (A * A^T ≈ I) because A^T was orthogonalized.
        let cfg = cfg();
        let mut inj_reg = LoRAInjectionRegistry::new();
        let mut ic = LoRAInjectionConfig::new(
            LoRAInjectionPoint::QProj,
            0,
            1,
            4, // rank
            16.0,
        );
        ic.use_spectral_qlora = true;
        inj_reg.add(ic);

        let reg = AutogradRegistry::new(cfg, inj_reg).unwrap();

        let pid_a = ParamId::a(0, 1, LoRAInjectionPoint::QProj);
        let a_param = reg.params.get(pid_a).expect("A param must exist");
        let a_data = a_param.data.to_vec_f32().unwrap();
        let a_rows = 4; // rank
        let a_cols = 8; // hidden_size for QProj

        // Compute A * A^T (r x r) — should be ≈ I for orthonormal rows.
        let mut gram = vec![0.0f32; a_rows * a_rows];
        for i in 0..a_rows {
            for j in 0..a_rows {
                let mut sum = 0.0f32;
                for k in 0..a_cols {
                    sum += a_data[i * a_cols + k] * a_data[j * a_cols + k];
                }
                gram[i * a_rows + j] = sum;
            }
        }

        for i in 0..a_rows {
            for j in 0..a_rows {
                if i == j {
                    assert!(
                        (gram[i * a_rows + j] - 1.0).abs() < 0.2,
                        "A*A^T diagonal at [{i},{j}] = {} (expected ≈1.0)",
                        gram[i * a_rows + j]
                    );
                } else {
                    assert!(
                        gram[i * a_rows + j].abs() < 0.2,
                        "A*A^T off-diagonal at [{i},{j}] = {} (expected ≈0.0)",
                        gram[i * a_rows + j]
                    );
                }
            }
        }
    }

    #[test]
    fn spectral_qlora_flag_defaults_to_false() {
        let ic = LoRAInjectionConfig::new(LoRAInjectionPoint::QProj, 0, 1, 4, 16.0);
        assert!(
            !ic.use_spectral_qlora,
            "spectral_qlora should default to false"
        );
    }
}
