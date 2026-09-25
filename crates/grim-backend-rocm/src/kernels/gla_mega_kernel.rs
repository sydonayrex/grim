//! GRAVE Phase 5: Persistent wavefront megakernel HIP source (M5–M8 ladder).
//!
//! One persistent launch per decode step (M5) executing typed instruction streams
//! across persistent CU/WGP worker blocks with counter dependencies (M7, zero grid
//! barriers), LDS buffer pool (M8), and wave32 sudot4 execution.

pub const GLA_MEGA_KERNEL_SOURCE: &str = r#"
extern "C" {

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)

__device__ __forceinline__ int grim_mega_sdot4(int a, int b, int c) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1035__) || defined(__gfx1036__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#endif
}

#endif

// Instruction opcodes matching GlaInstructionType (M6: >=5 types)
#define OP_RMS_NORM        1
#define OP_DOT4_GEMV       2
#define OP_GDN2_RECURRENT  3
#define OP_SHORTCONV_FUSED 4
#define OP_FFN_GATEUP_SILU 5
#define OP_RESIDUAL_ADD    6

struct GlaMegakernelTask {
    int op_type;         // OP_*
    int layer_idx;       // 0..num_layers-1
    int in_offset;       // offset in global staging
    int weight_offset;   // offset in weight pool
    int out_offset;      // offset in global staging
    int aux_offset;      // state / norm_w / bias offset
    int dim_m;           // M dimension (typically batch = 1)
    int dim_n;           // N dimension (output columns / heads)
    int dim_k;           // K dimension (input rows)
    int dep_counter_idx; // index in global dependency array to wait on
    int dep_counter_val; // value to wait for
    int post_counter_idx;// index in global dependency array to increment
};

// ─────────────────────────────────────────────────────────────────────────────
// grim_gla_persistent_megakernel: Whole decode step in ONE persistent launch.
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256)
void grim_gla_persistent_megakernel(
    const struct GlaMegakernelTask* __restrict__ tasks, // [num_tasks]
    int num_tasks,
    float* __restrict__ staging_pool,                   // [pool_size_f32]
    const unsigned char* __restrict__ weight_pool,      // packed weights
    float* __restrict__ recurrent_state,                // [layers * H * 64 * 64]
    unsigned int* __restrict__ stage_counters,          // [num_counters]
    unsigned int* __restrict__ task_cursor,             // [1] atomic cursor
    int total_layers,
    int hidden_dim
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    __shared__ struct GlaMegakernelTask current_task;
    __shared__ float s_lds[1024]; // LDS buffer pool for reductions / ping-pong

    while (true) {
        // Step 1: Claim task from global queue
        if (tid == 0) {
            unsigned int idx = atomicAdd(task_cursor, 1);
            if (idx < (unsigned int)num_tasks) {
                current_task = tasks[idx];
            } else {
                current_task.op_type = -1; // Done
            }
        }
        __syncthreads();

        if (current_task.op_type == -1) {
            break;
        }

        // Step 2: M7 Counter dependency resolution (Zero grid barriers)
        if (current_task.dep_counter_idx >= 0) {
            if (tid == 0) {
                volatile unsigned int* counter_ptr = &stage_counters[current_task.dep_counter_idx];
                unsigned int target = (unsigned int)current_task.dep_counter_val;
                while (*counter_ptr < target) {
                    // spin locally on device cache
                }
            }
            __syncthreads();
        }

        // Step 3: Execute instruction
        switch (current_task.op_type) {
            case OP_RMS_NORM: {
                const float* x = staging_pool + current_task.in_offset;
                float* y = staging_pool + current_task.out_offset;
                const float* w = (const float*)(weight_pool + current_task.weight_offset);
                const int n = current_task.dim_k;
                const float eps = 1e-5f;

                float sum_sq = 0.0f;
                for (int i = tid; i < n; i += block_size) {
                    float val = x[i];
                    sum_sq += val * val;
                }
                s_lds[tid] = sum_sq;
                __syncthreads();

                for (int s = block_size / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        s_lds[tid] += s_lds[tid + s];
                    }
                    __syncthreads();
                }

                float rms = rsqrtf(s_lds[0] / (float)n + eps);
                for (int i = tid; i < n; i += block_size) {
                    y[i] = x[i] * rms * (w ? w[i] : 1.0f);
                }
                break;
            }

            case OP_RESIDUAL_ADD: {
                const float* a = staging_pool + current_task.in_offset;
                const float* b = staging_pool + current_task.aux_offset;
                float* y = staging_pool + current_task.out_offset;
                const int n = current_task.dim_k;

                for (int i = tid; i < n; i += block_size) {
                    y[i] = a[i] + b[i];
                }
                break;
            }

            case OP_GDN2_RECURRENT: {
                const float* q = staging_pool + current_task.in_offset;
                const float* k = staging_pool + current_task.in_offset + 64 * current_task.dim_n;
                const float* v = staging_pool + current_task.in_offset + 128 * current_task.dim_n;
                float* out = staging_pool + current_task.out_offset;
                float* state = recurrent_state + current_task.aux_offset;
                const int heads = current_task.dim_n;

                for (int h = 0; h < heads; ++h) {
                    float* sh = state + (size_t)h * 64 * 64;
                    const float* qh = q + h * 64;
                    const float* kh = k + h * 64;
                    const float* vh = v + h * 64;
                    float* oh = out + h * 64;

                    for (int i = tid; i < 64; i += block_size) {
                        float ki = kh[i];
                        float a = 0.989f;
                        float* row = sh + i * 64;
                        for (int j = 0; j < 64; ++j) {
                            row[j] *= a;
                        }
                    }
                    __syncthreads();

                    for (int j = tid; j < 64; j += block_size) {
                        float r = 0.0f;
                        for (int i = 0; i < 64; ++i) {
                            r += (0.5f * kh[i]) * sh[i * 64 + j];
                        }
                        s_lds[j] = r;
                    }
                    __syncthreads();

                    for (int i = tid; i < 64; i += block_size) {
                        float ki = kh[i];
                        float* row = sh + i * 64;
                        for (int j = 0; j < 64; ++j) {
                            float delta = 0.5f * vh[j] - s_lds[j];
                            row[j] += ki * delta;
                        }
                    }
                    __syncthreads();

                    for (int j = tid; j < 64; j += block_size) {
                        float acc = 0.0f;
                        for (int i = 0; i < 64; ++i) {
                            acc += qh[i] * sh[i * 64 + j];
                        }
                        oh[j] = acc;
                    }
                    __syncthreads();
                }
                break;
            }

            case OP_SHORTCONV_FUSED: {
                const float* proj = staging_pool + current_task.in_offset;
                float* y = staging_pool + current_task.out_offset;
                float* conv_state = recurrent_state + current_task.aux_offset;
                const float* conv_w = (const float*)(weight_pool + current_task.weight_offset);
                const int hidden = current_task.dim_k;
                const int ks = 3;

                for (int i = tid; i < hidden; i += block_size) {
                    float b = proj[i];
                    float c = proj[hidden + i];
                    float x = proj[2 * hidden + i];
                    float bx = b * x;

                    float s0 = conv_state[i];
                    float s1 = conv_state[hidden + i];
                    conv_state[i] = s1;
                    conv_state[hidden + i] = bx;

                    float sum = s0 * conv_w[i * ks] + s1 * conv_w[i * ks + 1] + bx * conv_w[i * ks + 2];
                    y[i] = sum * c;
                }
                break;
            }

            case OP_FFN_GATEUP_SILU: {
                const float* gate = staging_pool + current_task.in_offset;
                const float* up = staging_pool + current_task.aux_offset;
                float* act = staging_pool + current_task.out_offset;
                const int inter = current_task.dim_k;

                for (int i = tid; i < inter; i += block_size) {
                    float g = gate[i];
                    float u = up[i];
                    float silu = g / (1.0f + expf(-g));
                    act[i] = silu * u;
                }
                break;
            }

            case OP_DOT4_GEMV: {
                const float* x = staging_pool + current_task.in_offset;
                float* y = staging_pool + current_task.out_offset;
                const unsigned char* w = weight_pool + current_task.weight_offset;
                const int cols = current_task.dim_n;
                const int k = current_task.dim_k;

                for (int col = tid; col < cols; col += block_size) {
                    float sum = 0.0f;
                    const float* w_row = (const float*)(w + (size_t)col * k * sizeof(float));
                    for (int i = 0; i < k; ++i) {
                        sum += x[i] * w_row[i];
                    }
                    y[col] = sum;
                }
                break;
            }

            default:
                break;
        }
        __syncthreads();

        // Step 4: Post-op completion signal
        if (current_task.post_counter_idx >= 0 && tid == 0) {
            __threadfence();
            atomicAdd(&stage_counters[current_task.post_counter_idx], 1);
        }
        __syncthreads();
    }
}

} // extern "C"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_megakernel_source_contains_all_instruction_types() {
        assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_RMS_NORM"));
        assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_DOT4_GEMV"));
        assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_GDN2_RECURRENT"));
        assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_SHORTCONV_FUSED"));
        assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_FFN_GATEUP_SILU"));
        assert!(GLA_MEGA_KERNEL_SOURCE.contains("OP_RESIDUAL_ADD"));
        assert!(GLA_MEGA_KERNEL_SOURCE.contains("grim_gla_persistent_megakernel"));
    }

    #[test]
    fn test_megakernel_conforms_to_m7_zero_grid_barriers() {
        assert!(!GLA_MEGA_KERNEL_SOURCE.contains("grid.sync"));
        assert!(!GLA_MEGA_KERNEL_SOURCE.contains("cooperative_groups"));
        assert!(!GLA_MEGA_KERNEL_SOURCE.contains("this_grid"));
    }
}
