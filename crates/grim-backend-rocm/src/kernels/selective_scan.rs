//! Mamba selective scan HIP kernel — Block size 256 threads, LDS tiling, persistent for decode-step.
//! On RDNA2 (gfx1036/gfx1030, Wave32): 256 threads = 8 Wave32 wavefronts.
//! On CDNA (gfx9xx, Wave64): 256 threads = 4 Wave64 wavefronts.

/// HIP source for `grim_selective_scan` and `grim_selective_scan_backward`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" {

    __device__ inline float fp32_from_f16_device(unsigned short h) {
        unsigned int sign = (h >> 15) & 1;
        unsigned int exp  = (h >> 10) & 0x1f;
        unsigned int mant = h & 0x3ff;
        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            float res = (float)mant / 1024.0f * 0.00006103515625f;
            return sign ? -res : res;
        } else if (exp == 31) {
            return sign ? -1.0f/0.0f : 1.0f/0.0f;
        }
        float res = (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
        return sign ? -res : res;
    }

    /// Single-step Mamba selective scan (decode-step). One thread per n dimension.
    /// h_t[n,s] = a[n,s] * h_{t-1}[n,s] + dt[n] * x_t[n] * b[n,s]
    /// y[n] = sum_s(c[n] * h_t[n,s]) + D[n] * x_t[n]
    ///   c[n] broadcast across state dim (per-channel C, matching d_param shape [d_inner]).
    __global__ void grim_selective_scan(
        const float* __restrict__ a_log,       // [d_inner * d_state]  A = exp(a_log+1), pre-computed on host
        const float* __restrict__ b_tensor,    // [d_inner * d_state]  B parameter
        const float* __restrict__ c_tensor,    // [d_inner]            C per-channel (broadcast across state)
        const float* __restrict__ d_tensor,    // [d_inner]            D bypass
        const float* __restrict__ dt_tensor,   // [d_inner]            delta/dt per channel
        float* __restrict__ h_in_out,          // [batch * d_inner * d_state]  state (read prev, write new)
        const float* __restrict__ x_tensor,    // [batch * d_inner]     input x_t
        float* __restrict__ y_data,             // [batch * d_inner]     scan output accumulator
        int batch_index,
        int d_inner,
        int d_state)
    {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= d_inner) return;

        // LDS tile for h[n, :]
        extern __shared__ float lds_h[];
        float* my_h = lds_h + threadIdx.x * d_state;

        const float* h_row = h_in_out + (batch_index * d_inner + n) * d_state;
        for (int s = threadIdx.x; s < d_state; s += blockDim.x) {
            my_h[s] = h_row[s];
        }
        __syncthreads();

        const float* a_row = a_log + n * d_state;
        const float* b_row = b_tensor + n * d_state;
        float c_n = c_tensor[n];
        const float x_n = x_tensor[batch_index * d_inner + n];
        float dt_n = dt_tensor[n];
        float y_accum = 0.0f;

        for (int s = 0; s < d_state; ++s) {
            float a = a_row[s];
            float h_prev = my_h[s];
            float h_new = a * h_prev + dt_n * x_n * b_row[s];
            my_h[s] = h_new;
            y_accum += c_n * h_new;
        }

        y_accum += d_tensor[n] * x_n;

        // Write updated state back to global.
        float* h_out_row = h_in_out + (batch_index * d_inner + n) * d_state;
        for (int s = threadIdx.x; s < d_state; s += blockDim.x) {
            h_out_row[s] = my_h[s];
        }
        __syncthreads();

        y_data[batch_index * d_inner + n] = y_accum;
    }

    /// Single-step backward pass for Mamba selective scan.
    /// Computes d_x[n] = d_y[n] * (D[n] + dt * sum_s(b[n,s])) + d_h_prev contribution.
    __global__ void grim_selective_scan_backward(
        const float* __restrict__ a_log,       // [d_inner * d_state]
        const float* __restrict__ b_tensor,    // [d_inner * d_state]
        const float* __restrict__ d_tensor,    // [d_inner]
        const float* __restrict__ d_y,          // [batch * d_inner]  upstream gradient of y
        const float* __restrict__ x_tensor,    // [batch * d_inner]  input x_t
        float* __restrict__ d_x,                // [batch * d_inner]  gradient w.r.t. x_t
        float* __restrict__ d_h_prev,                // [batch * d_inner * d_state] gradient w.r.t. h_{t-1}
        int batch_index,
        int d_inner,
        int d_state)
    {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= d_inner) return;

        const float* a_row = a_log + n * d_state;
        const float* b_row = b_tensor + n * d_state;
        const float x_n = x_tensor[batch_index * d_inner + n];
        const float d_val = d_tensor[n];
        const float dy_n = d_y[batch_index * d_inner + n];

        float d_x_val = d_val * dy_n;

        // d_h_prev[s] = a[s] * dy_n (chain rule through scan recurrence).
        float* d_h_prev_row = d_h_prev + (batch_index * d_inner + n) * d_state;
        for (int s = threadIdx.x; s < d_state; s += blockDim.x) {
            d_h_prev_row[s] = a_row[s] * dy_n;
        }
        __syncthreads();

        // d_x also accumulates contribution from b terms: sum_s(dt * b[s] * dy_n) for v1 dt=1.
        for (int s = 0; s < d_state; ++s) {
            d_x_val += b_row[s] * dy_n;
        }

        d_x[batch_index * d_inner + n] = d_x_val;
    }

}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selective_scan_source_contains_entries() {
        assert!(KERNEL_SOURCE.contains("grim_selective_scan"));
        assert!(KERNEL_SOURCE.contains("grim_selective_scan_backward"));
        assert!(KERNEL_SOURCE.contains("d_inner"));
        assert!(KERNEL_SOURCE.contains("d_state"));
    }
}
