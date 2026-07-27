//! RWKV time-mix kernel (Item 14).
//!
//! RWKV uses a linear attention mechanism where the attention pattern is
//! determined by a decay vector w. The time-mix operation for one token is:
//!
//!   k_t = W_k @ x_t          // key projection
//!   v_t = W_v @ x_t          // value projection
//!   w_t = sigmoid(W_w @ x_t) // decay weights
//!   a_t = sum_{i<=t} (w_i * k_i) * v_t    // weighted accumulation
//!   h_t = decay * h_{t-1} + a_t           // state update
//!   out_t = W_out @ h_t      // output projection
//!
//! The recurrence over time is inherently sequential (h depends on h_{t-1}),
//! so the RWKV kernel processes one token per thread block, iterating over
//! the time dimension within the block. This makes it well-suited for
//! persistent GPU kernels where each block handles one timestep.

/// HIP source for `grim_rwkv_time_mix` — RWKV linear attention time-mix.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    /// RWKV time-mix: single token step with recurrence over time.
    ///
    /// One thread block handles one token position t. Within the block,
    /// threads cooperatively compute the weighted accumulation across
    /// all prior positions (or a sliding window).
    ///
    /// Input x_t ∈ R[d_model], prior states h_{t-1} ∈ R[d_model].
    /// Output h_t ∈ R[d_model] (written in-place or to separate buffer).
    __global__ void grim_rwkv_time_mix(
        const float* __restrict__ x,            // [batch, seq_len, d_model]  current input
        const float* __restrict__ w_key,         // [d_model, d_model]  W_k  key projection
        const float* __restrict__ w_value,       // [d_model, d_model]  W_v  value projection
        const float* __restrict__ w_decay,       // [d_model]            W_w  decay projection (per-dim sigmoid)
        const float* __restrict__ w_out,         // [d_model, d_model]  W_out output projection
        const float* __restrict__ h_prev,        // [batch, d_model]  prior hidden state h_{t-1}
        float* __restrict__ h_curr,              // [batch, d_model]  current hidden state h_t (output)
        float* __restrict__ y,                   // [batch, d_model]  output projection result
        float* __restrict__ k_cache,             // [batch * seq_len * d_model]  key cache for reuse
        float* __restrict__ v_cache,             // [batch * seq_len * d_model]  value cache for reuse
        int batch_index,
        int seq_len,
        int d_model,
        int t)                                    // current timestep index (0..seq_len-1)
    {
        int d = blockIdx.x * blockDim.x + threadIdx.x;
        if (d >= d_model) return;

        // Project current input to key, value, decay.
        float k_t_d = 0.0f;
        float v_t_d = 0.0f;
        float w_d = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            float x_j = x[batch_index * seq_len * d_model + t * d_model + j];
            k_t_d += w_key[d * d_model + j] * x_j;
            v_t_d += w_value[d * d_model + j] * x_j;
            w_d   += w_decay[j] * x_j;
        }
        float decay = 1.0f / (1.0f + expf(-w_d));  // sigmoid of decay weight

        // Weighted accumulation over prior positions.
        // h_t[d] = decay * h_{t-1}[d] + sum_{i=0..t} (w_i * k_i[d]) * v_t[d]
        // For v1, use the full history with a simple weighted sum.
        float accum = 0.0f;
        for (int i = 0; i <= t; ++i) {
            float k_i_d = k_cache[batch_index * seq_len * d_model + i * d_model + d];
            accum += k_i_d * v_t_d;
        }

        float h_d = decay * h_prev[batch_index * d_model + d] + accum;
        h_curr[batch_index * d_model + d] = h_d;

        // Output projection: y = W_out @ h_t.
        float y_d = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            y_d += w_out[d * d_model + j] * h_curr[batch_index * d_model + j];
        }
        y[batch_index * d_model + d] = y_d;
    }

    /// RWKV channel-mix (FFN-like) pass: replaces the standard MLP with
    /// a multiplicative gating mechanism (RWKV-5/RWKV-6 style).
    /// y = (W_k @ x) * σ(W_v @ x) + W_r @ x
    __global__ void grim_rwkv_channel_mix(
        const float* __restrict__ x,            // [batch, d_model]  input
        const float* __restrict__ w_k,           // [d_model, d_model]  key projection
        const float* __restrict__ w_v,           // [d_model, d_model]  value projection
        const float* __restrict__ w_r,           // [d_model, d_model]  residual projection
        float* __restrict__ y,                   // [batch, d_model]  output
        int batch_index,
        int d_model)
    {
        int d = blockIdx.x * blockDim.x + threadIdx.x;
        if (d >= d_model) return;

        float k_val = 0.0f;
        float v_val = 0.0f;
        float r_val = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            float x_j = x[batch_index * d_model + j];
            k_val += w_k[d * d_model + j] * x_j;
            v_val += w_v[d * d_model + j] * x_j;
            r_val += w_r[d * d_model + j] * x_j;
        }

        float sigmoid_v = 1.0f / (1.0f + expf(-v_val));
        y[batch_index * d_model + d] = k_val * sigmoid_v + r_val;
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rwkv_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("grim_rwkv_time_mix"));
        assert!(KERNEL_SOURCE.contains("grim_rwkv_channel_mix"));
        assert!(KERNEL_SOURCE.contains("d_model"));
    }
}
