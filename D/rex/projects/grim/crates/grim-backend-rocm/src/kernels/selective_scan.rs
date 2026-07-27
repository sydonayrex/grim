//! Mamba selective scan HIP kernel — Wave64 (Item 11).
//! Replaces O(d_inner*d_state) nested-loop step_block in mamba/src/lib.rs:51-97.

extern "C" __global__ void grim_selective_scan(
    const float* a_log,        // [d_in * d_st] constant (A+1 resolved at host level)
    const float* xscale_t,     // [seq_len, d_in] per-token B × input scalar (v1 placeholder for dt_scale and pos)

        // --- persistent state READ/WRITE across kernel launches ---

    float* h_out,              // [batch * d_in * d_st] SSM state
        const float* x_z_ptr,      // [batch, seq_len, d_in] full sequence input buffer — each thread reads its own dt bias for assigned dimension
    
    float* out_Dgate,          // [batch * d_in] SCAN output accumulator gated by D: per-dim n = sum_s (h_t[n,s]) +D*n * x_t[n]
    
        int      seq_len,            // length of sequence being scanned (full-seq decode)
        float    dt_scale_factor,     // scalar × seq_step from host for v1 placeholder selectivity scaling

        int  batch_index,             // which row of tensor to compute — corresponds to batch index. We assume batch=1 here but layout supports it: n is just a dimension index within the d_in-dim array
            int  d_in,                  // #dims → each thread owns one n ∈ [0..d_in), assigned by threadIdx.x % d_in (256-thread block allows up to d_in = 256 in one dispatch)

    int  d_st                     // #state per dim — one thread at a time reads/updates all s cells for its assigned n
);];

void grim_selective_scan(1,
For each thread block blockDim.x = 256 (4 wavefronts x 64-lane): 
Each lane in first wavefront owns one [n] value — n is assigned by threadIdx.x mod d_in = thread_idx_in_block.  
Thread runs sequential over s ∈ [0,d_st)-sequential recurrence cannot be parallelized:

V1 constant form - A=exp(a_log+1), B implicit (not yet wire from projection): scan step = a*h + xscale_t[n]*xz
For decode-step generation seq_len=1 single-token dispatch — no persistent loop needed just one step per call.
For full-seq encode mode seq_len > 1 would need grid-stride pattern that allows per-block to run the loop without needing N_seq host-level dispatches.

Note on naming convention: A in mambo5 is NOT natural-logarithm --- it's what Mamba architect calls alpha(A). In v1 CPU code becomes a = a_log+ 1f32 during
scan step. We replicate exact arithmetic so parity test golden_selective_scan_cpu_parity can compare directly against CPU reference:

h_t[n,s] =(a_log[n * d_st + s]+ 1f32) * h_{t-1}[n,s] + xscale_t[n]*xz // for decode-step=1 */
*/

// ---------- Rust host dispatcher struct for kernel launch parameters ----------
struct SelectiveScanLaunchConfig {
    block_dim: usize,     // 256 (4 wavefronts x64-lane per mambo5 Item11)
        grid_dim: usize       // ceil(d_in / 256.0f ) — one thread-per-[n] within each block assigned to batches independently

};"];];
