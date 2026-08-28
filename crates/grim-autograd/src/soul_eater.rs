//! SOUL EATER Adapter & Optimizer module for `grim-autograd`.
//!
//! Provides the `SoulEaterAdapter` structural parameterization:
//! ΔW = U * Σ * V^T, with forward pass Y = X * W0^T + (α/r) * (X * V) * Σ * U^T.
//! Also provides `SoulEaterOptimizer` using 1-bit Sign-SGD for Σ and
//! momentum-accelerated pre-normalized cubic Newton-Schulz for U and V.
//!
//! SCYTHE1 extends the optimizer with Natural GaLore-style inverse-FIM
//! preconditioning in the r-dimensional adapter subspace: the r×r Fisher
//! information matrix is estimated from projected gradients (EMA-smoothed
//! with diagonal damping ε), inverted, and applied to precondition the
//! projected update before Newton-Schulz orthogonalization and Σ descent.

use grim_backend_cpu::cpu_tensor;
use grim_quant::soul_eater::subspace_newton_schulz_step;
use grim_tensor::{Result, Shape, Tensor};
use std::collections::HashMap;

/// Parameter representation for SOUL EATER adapter (U, V, Σ).
pub struct SoulEaterAdapter {
    /// Output basis matrix U [d_out, r], semi-orthogonal U^T * U = I_r.
    pub u: Tensor,
    /// Input basis matrix V [d_in, r], semi-orthogonal V^T * V = I_r.
    pub v: Tensor,
    /// Diagonal singular values Σ [r].
    pub sigma: Tensor,
    /// Scaling alpha / r.
    pub scale: f32,
    pub rank: usize,
}

/// SICKLE alias for legacy SOUL EATER adapter.
pub type SickleAdapter = SoulEaterAdapter;

impl SoulEaterAdapter {
    /// Instantiate a new SOUL EATER adapter for linear layer dimensions [d_out, d_in] and rank `r`.
    pub fn new(d_out: usize, d_in: usize, r: usize, alpha: f32) -> Result<Self> {
        let mut u_data = vec![0.0f32; d_out * r];
        let mut v_data = vec![0.0f32; d_in * r];
        let u_std = (2.0f32 / (d_out + r) as f32).sqrt();
        let v_std = (2.0f32 / (d_in + r) as f32).sqrt();

        for i in 0..d_out {
            for j in 0..r {
                let pseudo_norm = (((i + 1) * 17 + (j + 1) * 31) % 997) as f32 / 997.0 - 0.5;
                u_data[i * r + j] = pseudo_norm * u_std * 2.0;
            }
        }
        for i in 0..d_in {
            for j in 0..r {
                let pseudo_norm = (((i + 1) * 13 + (j + 1) * 29) % 997) as f32 / 997.0 - 0.5;
                v_data[i * r + j] = pseudo_norm * v_std * 2.0;
            }
        }

        // Perform initial orthogonalization
        let _ = subspace_newton_schulz_step(&mut u_data, d_out, r, 10);
        let _ = subspace_newton_schulz_step(&mut v_data, d_in, r, 10);

        let u = cpu_tensor(u_data, Shape::new(vec![d_out, r]));
        let v = cpu_tensor(v_data, Shape::new(vec![d_in, r]));
        // Initialize singular values to 1.0
        let sigma = cpu_tensor(vec![1.0f32; r], Shape::new(vec![r]));

        let scale = alpha / (r as f32);

        Ok(Self {
            u,
            v,
            sigma,
            scale,
            rank: r,
        })
    }

    /// Compute forward adapter output: Y_adapter = (α/r) * (X * V) * Σ * U^T.
    /// Returns output tensor of shape [B, d_out] for input X of shape [B, d_in].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.shape().dims();
        let b = x_shape[0];
        let d_in = x_shape[1];
        let d_out = self.u.shape().dims()[0];
        let r = self.rank;

        let x_vec = x.to_vec_f32()?;
        let u_vec = self.u.to_vec_f32()?;
        let v_vec = self.v.to_vec_f32()?;
        let sig_vec = self.sigma.to_vec_f32()?;

        // 1. Compute X_V = X * V [B, d_in] * [d_in, r] = [B, r]
        let mut x_v = vec![0.0f32; b * r];
        for i in 0..b {
            for j in 0..r {
                let mut sum = 0.0f32;
                for k in 0..d_in {
                    sum += x_vec[i * d_in + k] * v_vec[k * r + j];
                }
                x_v[i * r + j] = sum;
            }
        }

        // 2. Scale by Σ: X_V_Sig[i, j] = X_V[i, j] * Σ[j]
        let mut x_v_sig = vec![0.0f32; b * r];
        for i in 0..b {
            for j in 0..r {
                x_v_sig[i * r + j] = x_v[i * r + j] * sig_vec[j];
            }
        }

        // 3. Multiply by U^T: Out = (X_V_Sig * U^T) * (alpha / r) [B, r] * [r, d_out] = [B, d_out]
        let mut out = vec![0.0f32; b * d_out];
        for i in 0..b {
            for j in 0..d_out {
                let mut sum = 0.0f32;
                for k in 0..r {
                    sum += x_v_sig[i * r + k] * u_vec[j * r + k];
                }
                out[i * d_out + j] = sum * self.scale;
            }
        }

        Ok(cpu_tensor(out, Shape::new(vec![b, d_out])))
    }

    /// Compute backward gradients for 3-factor SoulEater adapter:
    /// Returns (g_x, g_u, g_v, g_sigma).
    #[allow(clippy::type_complexity)]
    pub fn backward(
        &self,
        out_grad: &Tensor,
        x: &Tensor,
    ) -> Result<(Tensor, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let x_shape = x.shape().dims();
        let b = x_shape[0];
        let d_in = x_shape[1];
        let d_out = self.u.shape().dims()[0];
        let r = self.rank;

        let g_vec = out_grad.to_vec_f32()?;
        let x_vec = x.to_vec_f32()?;
        let u_vec = self.u.to_vec_f32()?;
        let v_vec = self.v.to_vec_f32()?;
        let sig_vec = self.sigma.to_vec_f32()?;

        // 1. Forward activations needed:
        // X_V = X * V [B, r]
        let mut x_v = vec![0.0f32; b * r];
        for i in 0..b {
            for j in 0..r {
                let mut sum = 0.0f32;
                for k in 0..d_in {
                    sum += x_vec[i * d_in + k] * v_vec[k * r + j];
                }
                x_v[i * r + j] = sum;
            }
        }
        // X_V_Sig = X_V * Σ [B, r]
        let mut x_v_sig = vec![0.0f32; b * r];
        for i in 0..b {
            for j in 0..r {
                x_v_sig[i * r + j] = x_v[i * r + j] * sig_vec[j];
            }
        }

        // 2. Compute g_U [d_out, r] = (G^T * X_V_Sig) * scale
        let mut g_u = vec![0.0f32; d_out * r];
        for j in 0..d_out {
            for k in 0..r {
                let mut sum = 0.0f32;
                for i in 0..b {
                    sum += g_vec[i * d_out + j] * x_v_sig[i * r + k];
                }
                g_u[j * r + k] = sum * self.scale;
            }
        }

        // 3. Backprop through U^T: G_X_V_Sig = (G * U) * scale [B, r]
        let mut g_x_v_sig = vec![0.0f32; b * r];
        for i in 0..b {
            for k in 0..r {
                let mut sum = 0.0f32;
                for j in 0..d_out {
                    sum += g_vec[i * d_out + j] * u_vec[j * r + k];
                }
                g_x_v_sig[i * r + k] = sum * self.scale;
            }
        }

        // 4. Compute g_sigma [r] = sum_i (G_X_V_Sig[i, k] * X_V[i, k])
        let mut g_sigma = vec![0.0f32; r];
        for k in 0..r {
            let mut sum = 0.0f32;
            for i in 0..b {
                sum += g_x_v_sig[i * r + k] * x_v[i * r + k];
            }
            g_sigma[k] = sum;
        }

        // 5. Backprop through Σ: G_X_V = G_X_V_Sig * Σ [B, r]
        let mut g_x_v = vec![0.0f32; b * r];
        for i in 0..b {
            for k in 0..r {
                g_x_v[i * r + k] = g_x_v_sig[i * r + k] * sig_vec[k];
            }
        }

        // 6. Compute g_V [d_in, r] = X^T * G_X_V
        let mut g_v = vec![0.0f32; d_in * r];
        for k in 0..d_in {
            for j in 0..r {
                let mut sum = 0.0f32;
                for i in 0..b {
                    sum += x_vec[i * d_in + k] * g_x_v[i * r + j];
                }
                g_v[k * r + j] = sum;
            }
        }

        // 7. Compute g_X [B, d_in] = G_X_V * V^T
        let mut g_x = vec![0.0f32; b * d_in];
        for i in 0..b {
            for k in 0..d_in {
                let mut sum = 0.0f32;
                for j in 0..r {
                    sum += g_x_v[i * r + j] * v_vec[k * r + j];
                }
                g_x[i * d_in + k] = sum;
            }
        }

        Ok((
            cpu_tensor(g_x, Shape::new(vec![b, d_in])),
            g_u,
            g_v,
            g_sigma,
        ))
    }
}

/// SOUL EATER Optimizer (SCYTHE1 variant): Momentum + Newton-Schulz for U, V;
/// inverse-FIM-preconditioned descent for Σ.
///
/// SCYTHE1 adds Natural GaLore-style inverse-FIM preconditioning: the r×r
/// Fisher information matrix is estimated from the subspace-projected gradients,
/// EMA-smoothed, diagonal-damped, inverted, and applied to precondition the
/// projected U/V updates before Newton-Schulz orthogonalization. Σ is updated
/// with the preconditioned direction instead of 1-bit Sign-SGD.
///
/// Per-adapter state: O(d·r) for U, V, and their momenta, plus O(r²) for the
/// FIM EMA per basis (U, V) and O(r) for the Σ FIM EMA.
pub struct SoulEaterOptimizer {
    pub lr_basis: f32,
    pub lr_sigma: f32,
    pub beta: f32,
    pub m_u: HashMap<String, Vec<f32>>,
    pub m_v: HashMap<String, Vec<f32>>,
    /// SCYTHE1: EMA-smoothed r×r FIM for U basis.
    pub fim_u: HashMap<String, Vec<f32>>,
    /// SCYTHE1: EMA-smoothed r×r FIM for V basis.
    pub fim_v: HashMap<String, Vec<f32>>,
    /// SCYTHE1: EMA-smoothed diagonal FIM for Σ (size r).
    pub fim_sigma: HashMap<String, Vec<f32>>,
    /// SCYTHE1: EMA decay for FIM estimates.
    pub fim_ema_decay: f32,
    /// SCYTHE1: diagonal damping ε added to FIM before inversion.
    pub fim_damping: f32,
}

/// SICKLE alias for legacy SOUL EATER optimizer.
pub type SickleOptimizer = SoulEaterOptimizer;

impl SoulEaterOptimizer {
    pub fn new(lr_basis: f32, lr_sigma: f32, beta: f32) -> Self {
        Self {
            lr_basis,
            lr_sigma,
            beta,
            m_u: HashMap::new(),
            m_v: HashMap::new(),
            fim_u: HashMap::new(),
            fim_v: HashMap::new(),
            fim_sigma: HashMap::new(),
            fim_ema_decay: 0.95,
            fim_damping: 1e-3,
        }
    }

    /// Create an optimizer with custom SCYTHE1 FIM parameters.
    pub fn with_fim(
        lr_basis: f32,
        lr_sigma: f32,
        beta: f32,
        fim_ema_decay: f32,
        fim_damping: f32,
    ) -> Self {
        Self {
            lr_basis,
            lr_sigma,
            beta,
            m_u: HashMap::new(),
            m_v: HashMap::new(),
            fim_u: HashMap::new(),
            fim_v: HashMap::new(),
            fim_sigma: HashMap::new(),
            fim_ema_decay,
            fim_damping,
        }
    }

    /// Perform SCYTHE1 optimizer update step on adapter parameter tensors U, V, Σ.
    ///
    /// Steps (matching new_methods.md SCYTHE1 formulation):
    /// 1. Compute gradients g_U, g_V, g_Σ (provided as input).
    /// 2. Project g_U, g_V into the r-dim subspace via U^T, V^T.
    /// 3. Estimate r×r FIM from projected gradients (EMA-smoothed + damped).
    /// 4. Apply inverse-FIM preconditioning (with diagonal damping).
    /// 5. Update U, V with momentum + Newton-Schulz orthogonalization.
    /// 6. Update Σ with preconditioned direction (not 1-bit Sign-SGD).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        name: &str,
        u: &mut Tensor,
        v: &mut Tensor,
        sigma: &mut Tensor,
        g_u: &[f32],
        g_v: &[f32],
        g_sigma: &[f32],
    ) -> Result<()> {
        let d_out = u.shape().dims()[0];
        let r = u.shape().dims()[1];
        let d_in = v.shape().dims()[0];

        let u_vec = u.to_vec_f32()?;
        let v_vec = v.to_vec_f32()?;

        // --- Step 2: Project gradients into r-dim subspace ---
        // g_u_proj = U^T @ g_U  → [r, r]
        let g_u_proj = matmul_transpose_left(&u_vec, g_u, d_out, r);
        // g_v_proj = V^T @ g_V  → [r, r]
        let g_v_proj = matmul_transpose_left(&v_vec, g_v, d_in, r);

        // --- Step 3: Estimate FIM from projected gradients (EMA + damping) ---
        let key_u_fim = format!("{name}_fim_u");
        let key_v_fim = format!("{name}_fim_v");
        self.update_fim_2x2(&key_u_fim, &g_u_proj, r);
        self.update_fim_2x2(&key_v_fim, &g_v_proj, r);

        // --- Step 4: Inverse-FIM preconditioning ---
        // g_precond = F_inv @ g_proj, then project back: g_corrected = U @ g_precond
        let g_u_precond = {
            let fim_u = self
                .fim_u
                .get(&key_u_fim)
                .expect("FIM U was just initialized");
            let fim_u_inv = invert_r_by_r(fim_u, r);
            apply_fim_inverse(&fim_u_inv, &g_u_proj, r)
        };
        let g_u_corrected = project_back(&u_vec, &g_u_precond, d_out, r);

        let g_v_precond = {
            let fim_v = self
                .fim_v
                .get(&key_v_fim)
                .expect("FIM V was just initialized");
            let fim_v_inv = invert_r_by_r(fim_v, r);
            apply_fim_inverse(&fim_v_inv, &g_v_proj, r)
        };
        let g_v_corrected = project_back(&v_vec, &g_v_precond, d_in, r);

        // --- Step 5: Update U, V with momentum + Newton-Schulz ---
        let key_u = format!("{name}_u");
        let m_u_entry = self
            .m_u
            .entry(key_u)
            .or_insert_with(|| vec![0.0f32; d_out * r]);
        for i in 0..(d_out * r) {
            m_u_entry[i] = self.beta * m_u_entry[i] + (1.0 - self.beta) * g_u_corrected[i];
        }

        let mut o_u = m_u_entry.clone();
        if subspace_newton_schulz_step(&mut o_u, d_out, r, 10).is_ok() {
            let mut u_vec = u.to_vec_f32()?;
            for i in 0..(d_out * r) {
                u_vec[i] -= self.lr_basis * o_u[i];
            }
            *u = cpu_tensor(u_vec, Shape::new(vec![d_out, r]));
        }

        let key_v = format!("{name}_v");
        let m_v_entry = self
            .m_v
            .entry(key_v)
            .or_insert_with(|| vec![0.0f32; d_in * r]);
        for i in 0..(d_in * r) {
            m_v_entry[i] = self.beta * m_v_entry[i] + (1.0 - self.beta) * g_v_corrected[i];
        }

        let mut o_v = m_v_entry.clone();
        if subspace_newton_schulz_step(&mut o_v, d_in, r, 10).is_ok() {
            let mut v_vec = v.to_vec_f32()?;
            for i in 0..(d_in * r) {
                v_vec[i] -= self.lr_basis * o_v[i];
            }
            *v = cpu_tensor(v_vec, Shape::new(vec![d_in, r]));
        }

        // --- Step 6: Update Σ with preconditioned direction ---
        // Diagonal FIM for Σ: F_σ = EMA(g_σ²) + ε
        // Precond direction: g_σ / F_σ
        let key_sig = format!("{name}_sigma");
        let fim_sig = self
            .fim_sigma
            .entry(key_sig)
            .or_insert_with(|| vec![self.fim_damping; g_sigma.len()]);
        let decay = self.fim_ema_decay;
        for i in 0..r {
            fim_sig[i] = decay * fim_sig[i] + (1.0 - decay) * (g_sigma[i] * g_sigma[i]);
            fim_sig[i] = fim_sig[i].max(self.fim_damping);
        }
        let mut sig_vec = sigma.to_vec_f32()?;
        for i in 0..r {
            let precond = g_sigma[i] / fim_sig[i];
            sig_vec[i] -= self.lr_sigma * precond;
        }
        *sigma = cpu_tensor(sig_vec, Shape::new(vec![r]));

        Ok(())
    }

    /// Step over standard `TrainableParams` registry (generic LoRA training mode).
    pub fn step_params(&mut self, params: &mut crate::param::TrainableParams) -> Result<()> {
        for (&id, param) in params.iter_mut() {
            self.step_param(id, param)?;
        }
        Ok(())
    }

    /// Perform a preconditioned FIM update on a single `TrainableParam`.
    pub fn step_param(
        &mut self,
        id: crate::param::ParamId,
        param: &mut crate::param::TrainableParam,
    ) -> Result<()> {
        if param.is_frozen() {
            param.zero_grad()?;
            return Ok(());
        }
        let g_vec = param.grad().to_vec_f32()?;
        let mut d_vec = param.data.to_vec_f32()?;
        let d = d_vec.len();

        let key = format!("{:?}", id);
        let fim_entry = self
            .fim_u
            .entry(key.clone())
            .or_insert_with(|| vec![self.fim_damping; d]);

        let m_entry = self.m_u.entry(key).or_insert_with(|| vec![0.0f32; d]);

        let decay = self.fim_ema_decay;
        for i in 0..d {
            let g = g_vec[i];
            fim_entry[i] = decay * fim_entry[i] + (1.0 - decay) * (g * g);
            let damped_fim = fim_entry[i].max(self.fim_damping);
            let precond_g = g / damped_fim;

            m_entry[i] = self.beta * m_entry[i] + (1.0 - self.beta) * precond_g;
            d_vec[i] -= self.lr_basis * m_entry[i];
        }

        let dev = crate::pick_device_for_tensor(&param.data);
        let shape = param.data.shape().clone();
        let new_storage = dev.from_cpu(&d_vec, &shape, grim_tensor::DType::F32)?;
        param.data = grim_tensor::Tensor::new(
            std::sync::Arc::from(new_storage),
            shape,
            grim_tensor::DType::F32,
            param.data.provenance().clone(),
            param.data.device().clone(),
        );
        param.zero_grad()?;
        Ok(())
    }

    pub fn save_to_train_state(
        &self,
        params: &crate::param::TrainableParams,
    ) -> grim_format::train::TrainState {
        crate::adamw::save_param_data_only(params, 0)
    }

    pub fn load_from_train_state(
        &mut self,
        params: &mut crate::param::TrainableParams,
        state: &grim_format::train::TrainState,
    ) -> Result<()> {
        crate::adamw::load_param_data_only(params, state)
    }
}

/// Compute A = M^T @ B where M is [d × r] (row-major flat), B is [d × r], result [r × r].
/// Each element: A[i,j] = sum_k M[k*r+i] * B[k*r+j]
fn matmul_transpose_left(m: &[f32], b: &[f32], d: usize, r: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut sum = 0.0f32;
            for k in 0..d {
                sum += m[k * r + i] * b[k * r + j];
            }
            result[i * r + j] = sum;
        }
    }
    result
}

/// Invert an r×r symmetric positive semi-definite matrix (FIM) with adaptive ridge regularization.
/// Attempts exact inversion first (λ = 0.0); on near-singular pivots, adds progressive ridge damping (F + λI)⁻¹.
fn invert_r_by_r(mat: &[f32], r: usize) -> Vec<f32> {
    let lambdas = [0.0f32, 1e-5, 1e-4, 1e-3, 1e-2];
    for &lambda in &lambdas {
        let mut a = mat.to_vec();
        if lambda > 0.0 {
            for i in 0..r {
                a[i * r + i] += lambda;
            }
        }

        let mut inv = vec![0.0f32; r * r];
        for i in 0..r {
            inv[i * r + i] = 1.0;
        }

        let mut failed = false;
        for col in 0..r {
            let mut pivot = col;
            let mut max_val = a[col * r + col].abs();
            for row in (col + 1)..r {
                if a[row * r + col].abs() > max_val {
                    max_val = a[row * r + col].abs();
                    pivot = row;
                }
            }
            if max_val < 1e-7 {
                failed = true;
                break;
            }
            if pivot != col {
                for j in 0..r {
                    a.swap(col * r + j, pivot * r + j);
                    inv.swap(col * r + j, pivot * r + j);
                }
            }
            let pivot_val = a[col * r + col];
            let inv_pivot = 1.0 / pivot_val;
            for j in 0..r {
                a[col * r + j] *= inv_pivot;
                inv[col * r + j] *= inv_pivot;
            }
            for row in 0..r {
                if row == col {
                    continue;
                }
                let factor = a[row * r + col];
                if factor.abs() > 1e-12 {
                    for j in 0..r {
                        a[row * r + j] -= factor * a[col * r + j];
                        inv[row * r + j] -= factor * inv[col * r + j];
                    }
                }
            }
        }

        if !failed {
            return inv;
        }
    }

    // Fallback: damped diagonal inverse if complete decomposition fails
    let mut fallback = vec![0.0f32; r * r];
    for i in 0..r {
        let d = mat[i * r + i].max(1e-4);
        fallback[i * r + i] = 1.0 / d;
    }
    fallback
}

/// Apply FIM inverse to a projected gradient: result = F_inv @ g_proj  [r × r]
fn apply_fim_inverse(fim_inv: &[f32], g_proj: &[f32], r: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut sum = 0.0f32;
            for k in 0..r {
                sum += fim_inv[i * r + k] * g_proj[k * r + j];
            }
            result[i * r + j] = sum;
        }
    }
    result
}

/// Project preconditioned gradient back to full space: result = U @ g_precond  [d × r]
fn project_back(u: &[f32], g_precond: &[f32], d: usize, r: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; d * r];
    for k in 0..d {
        for j in 0..r {
            let mut sum = 0.0f32;
            for i in 0..r {
                sum += u[k * r + i] * g_precond[i * r + j];
            }
            result[k * r + j] = sum;
        }
    }
    result
}

impl SoulEaterOptimizer {
    /// SCYTHE1: EMA-update the r×r FIM estimate from projected gradients.
    /// F_new = g_proj @ g_proj^T + εI, EMA-smoothed.
    fn update_fim_2x2(&mut self, key: &str, g_proj: &[f32], r: usize) {
        let is_u = key.ends_with("_fim_u");
        let fim_map = if is_u {
            &mut self.fim_u
        } else {
            &mut self.fim_v
        };

        let fim = fim_map.entry(key.to_string()).or_insert_with(|| {
            let mut v = vec![0.0f32; r * r];
            for i in 0..r {
                v[i * r + i] = 1.0 + self.fim_damping;
            }
            v
        });

        for i in 0..r {
            for j in 0..r {
                let mut val = 0.0f32;
                for k in 0..r {
                    val += g_proj[k * r + i] * g_proj[k * r + j];
                }
                if i == j {
                    val += self.fim_damping;
                }
                fim[i * r + j] =
                    self.fim_ema_decay * fim[i * r + j] + (1.0 - self.fim_ema_decay) * val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soul_eater_adapter_shape_algebra() {
        let b = 4;
        let d_in = 32;
        let d_out = 64;
        let r = 16;
        let adapter = SoulEaterAdapter::new(d_out, d_in, r, 1.0).unwrap();

        let x = cpu_tensor(vec![1.0f32; b * d_in], Shape::new(vec![b, d_in]));
        let y = adapter.forward(&x).unwrap();

        assert_eq!(y.shape().dims(), vec![b, d_out]);
    }

    #[test]
    fn test_soul_eater_forward_backward_loss_reduction() {
        let d_in = 16;
        let d_out = 16;
        let r = 8;
        let mut adapter = SoulEaterAdapter::new(d_out, d_in, r, 1.0).unwrap();
        let mut opt = SoulEaterOptimizer::new(0.01, 0.01, 0.0); // beta=0 for direct gradient step

        let x = cpu_tensor(vec![0.5f32; d_in], Shape::new(vec![1, d_in]));
        let target = cpu_tensor(vec![1.0f32; d_out], Shape::new(vec![1, d_out]));

        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;

        for step in 0..50 {
            let y = adapter.forward(&x).unwrap();
            let y_vec = y.to_vec_f32().unwrap();
            let target_vec = target.to_vec_f32().unwrap();

            let mut loss = 0.0f32;
            let mut dy = vec![0.0f32; d_out];
            for i in 0..d_out {
                let diff = y_vec[i] - target_vec[i];
                loss += diff * diff;
                dy[i] = 2.0 * diff; // L = sum( (y - target)^2 )
            }
            if step == 0 {
                initial_loss = loss;
            }
            final_loss = loss;

            // Analytical gradients:
            // Y = (alpha/r) * (X V) * Σ * U^T
            let x_vec = x.to_vec_f32().unwrap();
            let u_vec = adapter.u.to_vec_f32().unwrap();
            let v_vec = adapter.v.to_vec_f32().unwrap();
            let sig_vec = adapter.sigma.to_vec_f32().unwrap();
            let scale = adapter.scale;

            // dL/dU [d_out, r]: dL/dU[j, k] = dy[j] * scale * (X V)[k] * Σ[k]
            let mut x_v = vec![0.0f32; r];
            for k in 0..r {
                let mut sum = 0.0f32;
                for i in 0..d_in {
                    sum += x_vec[i] * v_vec[i * r + k];
                }
                x_v[k] = sum;
            }

            let mut g_u = vec![0.0f32; d_out * r];
            for j in 0..d_out {
                for k in 0..r {
                    g_u[j * r + k] = dy[j] * scale * x_v[k] * sig_vec[k];
                }
            }

            // dL/dΣ [r]: dL/dΣ[k] = sum_j ( dy[j] * scale * (X V)[k] * U[j, k] )
            let mut g_sigma = vec![0.0f32; r];
            for k in 0..r {
                let mut sum = 0.0f32;
                for j in 0..d_out {
                    sum += dy[j] * scale * x_v[k] * u_vec[j * r + k];
                }
                g_sigma[k] = sum;
            }

            // dL/dV [d_in, r]: dL/dV[i, k] = x_vec[i] * sum_j ( dy[j] * scale * Σ[k] * U[j, k] )
            let mut g_v = vec![0.0f32; d_in * r];
            for i in 0..d_in {
                for k in 0..r {
                    let mut sum = 0.0f32;
                    for j in 0..d_out {
                        sum += dy[j] * scale * sig_vec[k] * u_vec[j * r + k];
                    }
                    g_v[i * r + k] = x_vec[i] * sum;
                }
            }

            opt.step(
                "layer0",
                &mut adapter.u,
                &mut adapter.v,
                &mut adapter.sigma,
                &g_u,
                &g_v,
                &g_sigma,
            )
            .unwrap();
        }

        assert!(
            final_loss <= initial_loss,
            "Loss must not increase: initial {initial_loss}, final {final_loss}"
        );
    }

    #[test]
    fn test_scythe1_fim_damping_prevents_singularity() {
        // When g_u and g_v are zero (flat gradient), the FIM would be zero.
        // With damping ε, the FIM EMA should remain positive-definite.
        let d_in = 8;
        let d_out = 8;
        let r = 4;
        let mut adapter = SoulEaterAdapter::new(d_out, d_in, r, 1.0).unwrap();
        let mut opt = SoulEaterOptimizer::new(0.01, 0.01, 0.0);

        let zero_grad = vec![0.0f32; d_out * r];
        // Step with zero gradients — FIM should be identity + damping, not singular.
        opt.step(
            "layer0",
            &mut adapter.u,
            &mut adapter.v,
            &mut adapter.sigma,
            &zero_grad,
            &zero_grad,
            &vec![0.0f32; r],
        )
        .unwrap();

        // FIM should exist and be well-conditioned.
        let fim_key = "layer0_fim_u";
        let fim = opt.fim_u.get(fim_key).expect("FIM should exist after step");
        assert_eq!(fim.len(), r * r);

        // With zero gradients, F = εI (damping only). Diagonal should be ~1.0 + ε.
        for i in 0..r {
            for j in 0..r {
                if i == j {
                    assert!(fim[i * r + i] > 0.0, "FIM diagonal must be positive");
                } else {
                    assert!(fim[i * r + j].abs() < 1e-3, "FIM off-diagonal should be ~0");
                }
            }
        }
    }

    #[test]
    fn test_scythe1_sigma_not_sign_sgd() {
        // SCYTHE1 replaces 1-bit Sign-SGD with inverse-FIM preconditioned descent.
        // Verify that the Σ update uses the gradient direction (not just ±1).
        let d_in = 16;
        let d_out = 16;
        let r = 8;
        let mut adapter = SoulEaterAdapter::new(d_out, d_in, r, 1.0).unwrap();
        let mut opt = SoulEaterOptimizer::new(0.01, 0.01, 0.0);

        let x = cpu_tensor(vec![0.5f32; d_in], Shape::new(vec![1, d_in]));
        let target = cpu_tensor(vec![1.0f32; d_out], Shape::new(vec![1, d_out]));

        // Run one step and check Σ changed.
        let sigma_before = adapter.sigma.to_vec_f32().unwrap();
        let y = adapter.forward(&x).unwrap();
        let y_vec = y.to_vec_f32().unwrap();
        let target_vec = target.to_vec_f32().unwrap();
        let scale = adapter.scale;
        let x_vec = x.to_vec_f32().unwrap();
        let u_vec = adapter.u.to_vec_f32().unwrap();
        let v_vec = adapter.v.to_vec_f32().unwrap();
        let mut x_v = vec![0.0f32; r];
        for k in 0..r {
            for i in 0..d_in {
                x_v[k] += x_vec[i] * v_vec[i * r + k];
            }
        }
        let mut g_sigma = vec![0.0f32; r];
        for k in 0..r {
            let mut sum = 0.0f32;
            for j in 0..d_out {
                sum += (y_vec[j] - target_vec[j]) * 2.0 * scale * x_v[k] * u_vec[j * r + k];
            }
            g_sigma[k] = sum;
        }
        let g_u = vec![0.0f32; d_out * r];
        let g_v = vec![0.0f32; d_in * r];
        opt.step(
            "layer0",
            &mut adapter.u,
            &mut adapter.v,
            &mut adapter.sigma,
            &g_u,
            &g_v,
            &g_sigma,
        )
        .unwrap();
        let sigma_after = adapter.sigma.to_vec_f32().unwrap();

        // Sigma should have changed (not frozen by sign-SGD or zero gradient issues).
        let changed = sigma_before
            .iter()
            .zip(&sigma_after)
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(
            changed,
            "Sigma should change under non-zero gradient with FIM preconditioning"
        );
    }

    #[test]
    fn test_soul_eater_adapter_backward_parity() {
        let adapter = SoulEaterAdapter::new(8, 16, 4, 8.0).unwrap();
        let x = cpu_tensor(vec![0.5f32; 2 * 16], Shape::new(vec![2, 16]));
        let out = adapter.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[2, 8]);

        let grad_out = cpu_tensor(vec![1.0f32; 2 * 8], Shape::new(vec![2, 8]));
        let (g_x, g_u, g_v, g_sigma) = adapter.backward(&grad_out, &x).unwrap();
        assert_eq!(g_x.shape().dims(), &[2, 16]);
        assert_eq!(g_u.len(), 8 * 4);
        assert_eq!(g_v.len(), 16 * 4);
        assert_eq!(g_sigma.len(), 4);
    }

    #[test]
    fn test_scythe1_loss_decreases_over_50_steps() {
        // Full end-to-end: SCYTHE1 optimizer should reduce loss over multiple steps.
        let d_in = 16;
        let d_out = 16;
        let r = 8;
        let mut adapter = SoulEaterAdapter::new(d_out, d_in, r, 1.0).unwrap();
        let mut opt = SoulEaterOptimizer::with_fim(0.01, 0.01, 0.0, 0.9, 1e-3);

        let x = cpu_tensor(vec![0.5f32; d_in], Shape::new(vec![1, d_in]));
        let target = cpu_tensor(vec![1.0f32; d_out], Shape::new(vec![1, d_out]));

        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;

        for step in 0..50 {
            let y = adapter.forward(&x).unwrap();
            let y_vec = y.to_vec_f32().unwrap();
            let target_vec = target.to_vec_f32().unwrap();
            let mut loss = 0.0f32;
            let mut dy = vec![0.0f32; d_out];
            for i in 0..d_out {
                let diff = y_vec[i] - target_vec[i];
                loss += diff * diff;
                dy[i] = 2.0 * diff;
            }
            if step == 0 {
                initial_loss = loss;
            }
            final_loss = loss;
            let x_vec = x.to_vec_f32().unwrap();
            let u_vec = adapter.u.to_vec_f32().unwrap();
            let v_vec = adapter.v.to_vec_f32().unwrap();
            let sig_vec = adapter.sigma.to_vec_f32().unwrap();
            let scale = adapter.scale;
            let mut x_v = vec![0.0f32; r];
            for k in 0..r {
                for i in 0..d_in {
                    x_v[k] += x_vec[i] * v_vec[i * r + k];
                }
            }
            let mut g_u = vec![0.0f32; d_out * r];
            for j in 0..d_out {
                for k in 0..r {
                    g_u[j * r + k] = dy[j] * scale * x_v[k] * sig_vec[k];
                }
            }
            let mut g_sigma = vec![0.0f32; r];
            for k in 0..r {
                let mut sum = 0.0f32;
                for j in 0..d_out {
                    sum += dy[j] * scale * x_v[k] * u_vec[j * r + k];
                }
                g_sigma[k] = sum;
            }
            let mut g_v = vec![0.0f32; d_in * r];
            for i in 0..d_in {
                for k in 0..r {
                    let mut sum = 0.0f32;
                    for j in 0..d_out {
                        sum += dy[j] * scale * sig_vec[k] * u_vec[j * r + k];
                    }
                    g_v[i * r + k] = x_vec[i] * sum;
                }
            }
            opt.step(
                "layer0",
                &mut adapter.u,
                &mut adapter.v,
                &mut adapter.sigma,
                &g_u,
                &g_v,
                &g_sigma,
            )
            .unwrap();
        }

        assert!(
            final_loss <= initial_loss,
            "SCYTHE1 loss must not increase: initial {initial_loss}, final {final_loss}"
        );
    }

    #[test]
    fn test_invert_r_by_r_identity() {
        let r = 4;
        let mut identity = vec![0.0f32; r * r];
        for i in 0..r {
            identity[i * r + i] = 1.0;
        }
        let inv = invert_r_by_r(&identity, r);
        for i in 0..r {
            for j in 0..r {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i * r + j] - expected).abs() < 1e-5,
                    "Identity inverse at [{i},{j}] expected {expected}, got {}",
                    inv[i * r + j]
                );
            }
        }
    }
}
