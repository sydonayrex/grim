//! On-device fused linear cross-entropy kernels.

/// Design A: one thread owns one batch row and walks the vocabulary serially.
/// This keeps the kernel free of inter-thread synchronization while avoiding the
/// host-side `[batch, vocab]` logits materialization.
pub const FUSED_LINEAR_CE_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void grim_fused_linear_ce_forward(
    const float* hidden, const float* lm_head, const unsigned int* targets,
    float* loss_out, float* lse_out, int hidden_dim, int vocab_size, int v_tile_size, int batch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || row < 0) return;
    float max_logit = -INFINITY;
    float sum_exp = 0.0f;
    float target_logit = 0.0f;
    int target = targets[row];
    for (int tile = 0; tile < vocab_size; tile += v_tile_size) {
        int end = min(tile + v_tile_size, vocab_size);
        for (int v = tile; v < end; ++v) {
            float logit = 0.0f;
            for (int d = 0; d < hidden_dim; ++d)
                logit += hidden[row * hidden_dim + d] * lm_head[v * hidden_dim + d];
            if (v == target) target_logit = logit;
            if (logit > max_logit) {
                sum_exp = sum_exp * expf(max_logit - logit) + 1.0f;
                max_logit = logit;
            } else {
                sum_exp += expf(logit - max_logit);
            }
        }
    }
    float lse = max_logit + logf(sum_exp);
    lse_out[row] = lse;
    loss_out[row] = lse - target_logit;
}

extern "C" __global__ void grim_fused_linear_ce_backward(
    const float* hidden, const float* lm_head, const unsigned int* targets, const float* lse,
    float* grad_h, int hidden_dim, int vocab_size, int v_tile_size, float inv_batch, int batch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || row < 0) return;
    int target = targets[row];
    for (int d = 0; d < hidden_dim; ++d) grad_h[row * hidden_dim + d] = 0.0f;
    for (int tile = 0; tile < vocab_size; tile += v_tile_size) {
        int end = min(tile + v_tile_size, vocab_size);
        for (int v = tile; v < end; ++v) {
            float logit = 0.0f;
            for (int d = 0; d < hidden_dim; ++d)
                logit += hidden[row * hidden_dim + d] * lm_head[v * hidden_dim + d];
            float dl = (expf(logit - lse[row]) - (v == target ? 1.0f : 0.0f)) * inv_batch;
            for (int d = 0; d < hidden_dim; ++d)
                grad_h[row * hidden_dim + d] += dl * lm_head[v * hidden_dim + d];
        }
    }
}
"#;
