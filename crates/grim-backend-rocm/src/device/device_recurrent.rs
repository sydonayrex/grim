//! Recurrent operations for `RocmDevice`.

use std::ffi::c_void;

use grim_tensor::backend::ComputeHandle;
use grim_tensor::error::{Error, Result};
use grim_tensor::{BackendStorage, RecurrentOps, Shape};

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{
    arg, as_rocm, dev_ptr, dtype_f32, linear_launch, HipDim3, RocmHandle,
};

impl RecurrentOps for RocmDevice {


    fn short_conv1d_causal_step(
        &self,
        x: &dyn BackendStorage,
        weight: &dyn BackendStorage,
        bias: Option<&dyn BackendStorage>,
        conv_state: &dyn BackendStorage,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let w_s = as_rocm(weight)?;
        let st_s = as_rocm(conv_state)?;
        if !x_s.device_ptr_is_valid() || !w_s.device_ptr_is_valid() || !st_s.device_ptr_is_valid() {
            return Err(Error::Backend(
                "short_conv1d: inputs lack valid device ptr".into(),
            ));
        }
        let total = out_shape.elem_count();
        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut x_ptr = dev_ptr(x_s)?;
        let mut w_ptr = dev_ptr(w_s)?;
        let mut b_ptr = match bias {
            Some(b) => dev_ptr(as_rocm(b)?)?,
            None => 0u64,
        };
        let mut st_ptr = dev_ptr(st_s)?;

        let dims = out_shape.dims();
        let mut batch = dims[0] as i32;
        let mut channels = dims[2] as i32;
        let mut k_size = (w_s.bytes / (channels as usize * 4)) as i32;

        let (grid, block) = linear_launch(total);
        self.launch_compute_kernel(
            "grim_short_conv1d_causal_step",
            grid,
            block,
            &mut [
                arg(&mut x_ptr),
                arg(&mut w_ptr),
                arg(&mut b_ptr),
                arg(&mut st_ptr),
                arg(&mut out_ptr),
                arg(&mut batch),
                arg(&mut channels),
                arg(&mut k_size),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }


    fn kda_gated_delta_rule_step(
        &self,
        q: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        beta: &dyn BackendStorage,
        a_gate: &dyn BackendStorage,
        recurrent_state: &dyn BackendStorage,
        d_k: usize,
        d_v: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let q_s = as_rocm(q)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let beta_s = as_rocm(beta)?;
        let gate_s = as_rocm(a_gate)?;
        let s_s = as_rocm(recurrent_state)?;

        let storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        let mut out_ptr = dev_ptr(&storage)?;
        let mut q_ptr = dev_ptr(q_s)?;
        let mut k_ptr = dev_ptr(k_s)?;
        let mut v_ptr = dev_ptr(v_s)?;
        let mut beta_ptr = dev_ptr(beta_s)?;
        let mut gate_ptr = dev_ptr(gate_s)?;
        let mut s_ptr = dev_ptr(s_s)?;
        let mut dk_i = d_k as i32;
        let mut dv_i = d_v as i32;

        let (grid, block) = linear_launch(d_v);
        self.launch_compute_kernel(
            "grim_kda_gated_delta_rule_step",
            grid,
            block,
            &mut [
                arg(&mut q_ptr),
                arg(&mut k_ptr),
                arg(&mut v_ptr),
                arg(&mut beta_ptr),
                arg(&mut gate_ptr),
                arg(&mut s_ptr),
                arg(&mut out_ptr),
                arg(&mut dk_i),
                arg(&mut dv_i),
            ],
        )?;
        Ok((
            Box::new(storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }


    fn selective_scan(
        &self,
        x: &dyn BackendStorage,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        c: &dyn BackendStorage,
        d: &dyn BackendStorage,
        state: &dyn BackendStorage,
        batch: usize,
        dim_dstate: usize,
        dim_dinner: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        let c_s = as_rocm(c)?;
        let d_s = as_rocm(d)?;
        let state_s = as_rocm(state)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_selective_scan(
            x_s,
            a_s,
            b_s,
            c_s,
            d_s,
            state_s,
            &out_storage,
            batch,
            dim_dstate,
            dim_dinner,
            seq_len,
        )?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }

    /// Headed single-step scan (Falcon-H1 / Mamba-2 decode). See the trait
    /// doc for the recurrence; state updates in place.
    fn selective_scan_headed(
        &self,
        x: &dyn BackendStorage,
        dt: &dyn BackendStorage,
        a: &dyn BackendStorage,
        b: &dyn BackendStorage,
        c: &dyn BackendStorage,
        d: &dyn BackendStorage,
        state: &dyn BackendStorage,
        n_heads: usize,
        d_state: usize,
        head_dim_ssm: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let dt_s = as_rocm(dt)?;
        let a_s = as_rocm(a)?;
        let b_s = as_rocm(b)?;
        let c_s = as_rocm(c)?;
        let d_s = as_rocm(d)?;
        let state_s = as_rocm(state)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_selective_scan_headed(
            x_s,
            dt_s,
            a_s,
            b_s,
            c_s,
            d_s,
            state_s,
            &out_storage,
            n_heads,
            d_state,
            head_dim_ssm,
        )?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }


    fn rwkv_time_mix(
        &self,
        x: &dyn BackendStorage,
        w: &dyn BackendStorage,
        k: &dyn BackendStorage,
        v: &dyn BackendStorage,
        g: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        seq_len: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let w_s = as_rocm(w)?;
        let k_s = as_rocm(k)?;
        let v_s = as_rocm(v)?;
        let g_s = as_rocm(g)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_rwkv_time_mix(x_s, w_s, k_s, v_s, g_s, &out_storage, batch, dim, seq_len)?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }


    fn rwkv_channel_mix(
        &self,
        x: &dyn BackendStorage,
        k: &dyn BackendStorage,
        r: &dyn BackendStorage,
        v: &dyn BackendStorage,
        batch: usize,
        dim: usize,
        out_shape: &Shape,
    ) -> Result<(Box<dyn BackendStorage>, Box<dyn ComputeHandle>)> {
        let x_s = as_rocm(x)?;
        let k_s = as_rocm(k)?;
        let r_s = as_rocm(r)?;
        let v_s = as_rocm(v)?;
        let out_storage =
            RocmStorage::alloc_gpu(out_shape, dtype_f32(), &self.allocator, self.ordinal)?;
        self.launch_rwkv_channel_mix(x_s, k_s, r_s, v_s, &out_storage, batch, dim)?;
        Ok((
            Box::new(out_storage),
            Box::new(RocmHandle::new(Some(self.active_stream()))),
        ))
    }
}




impl RocmDevice {
    // ─── Phase 2: Selective Scan ──────────────────────────────────

    /// Launch the JIT compiled Mamba selective scan kernel (Wave64,
    pub(crate) fn launch_selective_scan(
        &self,
        x_storage: &RocmStorage,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        c_storage: &RocmStorage,
        d_storage: &RocmStorage,
        state_storage: &RocmStorage,
        out_storage: &RocmStorage,
        batch: usize,
        dim_dstate: usize,
        dim_dinner: usize,
        _seq_len: usize,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: x has no device ptr".into()))?;
        let a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: a has no device ptr".into()))?;
        let b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: b has no device ptr".into()))?;
        let c_ptr = c_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: c has no device ptr".into()))?;
        let d_ptr = d_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: d has no device ptr".into()))?;
        let state_ptr = state_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: state has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (batch as u64)
            .checked_mul(dim_dinner as u64)
            .ok_or_else(|| Error::Backend("selective_scan: batch*dim_dinner overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("selective_scan: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        // Kernel signature: (a_log, b_tensor, c_tensor, d_tensor, dt_tensor,
        //                    h_in_out, x_tensor, y_data, batch_index, d_inner, d_state)
        let mut a_log_ptr = a_ptr;
        let mut b_tensor_ptr = b_ptr;
        let mut c_tensor_ptr = c_ptr;
        let mut d_tensor_ptr = d_ptr;
        let mut dt_tensor_ptr = d_ptr; // dt_bias passed as d_storage
        let mut h_in_out_ptr = state_ptr; // state buffer (read prev, write new)
        let mut x_tensor_ptr = x_ptr;
        let mut y_data_ptr = out_ptr; // output buffer
        let mut batch_index = batch as i32;
        let mut d_inner = dim_dinner as i32;
        let mut d_state = dim_dstate as i32;

        let shared_mem_bytes = dim_dstate * BLOCK_SIZE * std::mem::size_of::<f32>();

        self.launch_compute_kernel_with_solution(
            "grim_selective_scan",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut a_log_ptr),
                arg(&mut b_tensor_ptr),
                arg(&mut c_tensor_ptr),
                arg(&mut d_tensor_ptr),
                arg(&mut dt_tensor_ptr),
                arg(&mut h_in_out_ptr),
                arg(&mut x_tensor_ptr),
                arg(&mut y_data_ptr),
                arg(&mut batch_index),
                arg(&mut d_inner),
                arg(&mut d_state),
            ],
            None,
            shared_mem_bytes,
        )
    }

    pub(crate) fn launch_selective_scan_headed(
        &self,
        x_storage: &RocmStorage,
        dt_storage: &RocmStorage,
        a_storage: &RocmStorage,
        b_storage: &RocmStorage,
        c_storage: &RocmStorage,
        d_storage: &RocmStorage,
        state_storage: &RocmStorage,
        out_storage: &RocmStorage,
        n_heads: usize,
        d_state: usize,
        head_dim_ssm: usize,
    ) -> Result<*mut c_void> {
        let mut x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan_headed: x has no device ptr".into()))?;
        let mut dt_ptr = dt_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan_headed: dt has no device ptr".into()))?;
        let mut a_ptr = a_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan_headed: a has no device ptr".into()))?;
        let mut b_ptr = b_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan_headed: b has no device ptr".into()))?;
        let mut c_ptr = c_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan_headed: c has no device ptr".into()))?;
        let mut d_ptr = d_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan_headed: d has no device ptr".into()))?;
        let mut h_ptr = state_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan_headed: state has no device ptr".into()))?;
        let mut y_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("selective_scan_headed: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let d_inner = n_heads * head_dim_ssm;
        let grid_x: u32 = (d_inner as u64).div_ceil(BLOCK_SIZE as u64) as u32;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut nh = n_heads as i32;
        let mut ds = d_state as i32;
        let mut hds = head_dim_ssm as i32;

        let shared_mem_bytes = d_state * BLOCK_SIZE * std::mem::size_of::<f32>();
        self.launch_compute_kernel_with_solution(
            "grim_selective_scan_headed",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut x_ptr),
                arg(&mut dt_ptr),
                arg(&mut a_ptr),
                arg(&mut b_ptr),
                arg(&mut c_ptr),
                arg(&mut d_ptr),
                arg(&mut h_ptr),
                arg(&mut y_ptr),
                arg(&mut nh),
                arg(&mut ds),
                arg(&mut hds),
            ],
            None,
            shared_mem_bytes,
        )
    }

    // ─── Phase 2: Cross-Attention ─────────────────────────────────
    // ─── Phase 2: Cross-Attention ─────────────────────────────────

    /// Launch the JIT compiled Whisper cross-attention kernel
    pub(crate) fn launch_cross_attention(
        &self,
        q_storage: &RocmStorage,
        k_storage: &RocmStorage,
        v_storage: &RocmStorage,
        out_storage: &RocmStorage,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> Result<*mut c_void> {
        let q_ptr = q_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("cross_attention: q has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("cross_attention: k has no device ptr".into()))?;
        let v_ptr = v_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("cross_attention: v has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("cross_attention: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 128;
        // One block per (query position, head) row.
        let total_rows = num_heads
            .checked_mul(seq_len)
            .ok_or_else(|| Error::Backend("cross_attention: rows overflow".into()))?;
        let grid_x: u32 = total_rows
            .try_into()
            .map_err(|_| Error::Backend("cross_attention: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);
        // Shared memory: scores[seq_len_k] + red_max[block_dim] + red_sum[block_dim].
        let shared_mem_bytes = (kv_seq_len + 2 * BLOCK_SIZE)
            .checked_mul(4)
            .ok_or_else(|| Error::Backend("cross_attention: shared mem overflow".into()))?;

        let mut qptr = q_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut sq = seq_len as i32;
        let mut sk = kv_seq_len as i32;
        let mut nh = num_heads as i32;
        let mut nkh = num_heads as i32; // cross-attention uses full GQA sharing
        let mut hd = head_dim as i32;
        let mut scale = 1.0f32 / (head_dim as f32).sqrt();

        self.launch_compute_kernel_with_solution(
            "grim_cross_attention",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut qptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut sq),
                arg(&mut sk),
                arg(&mut nh),
                arg(&mut nkh),
                arg(&mut hd),
                arg(&mut scale),
            ],
            None,
            shared_mem_bytes,
        )
    }

    // ─── Phase 2: RWKV Time-Mix ───────────────────────────────────

    /// Launch the JIT compiled RWKV time-mix kernel (recurrent
    pub(crate) fn launch_rwkv_time_mix(
        &self,
        x_storage: &RocmStorage,
        w_storage: &RocmStorage,
        k_storage: &RocmStorage,
        v_storage: &RocmStorage,
        g_storage: &RocmStorage,
        out_storage: &RocmStorage,
        batch: usize,
        dim: usize,
        seq_len: usize,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: x has no device ptr".into()))?;
        let w_ptr = w_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: w has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: k has no device ptr".into()))?;
        let v_ptr = v_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: v has no device ptr".into()))?;
        let g_ptr = g_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: g has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_time_mix: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (batch as u64)
            .checked_mul(dim as u64)
            .ok_or_else(|| Error::Backend("rwkv_time_mix: batch*dim overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("rwkv_time_mix: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut xptr = x_ptr;
        let mut wptr = w_ptr;
        let mut kptr = k_ptr;
        let mut vptr = v_ptr;
        let mut gptr = g_ptr;
        let mut optr = out_ptr;
        let mut b_val = batch as i32;
        let mut d_val = dim as i32;
        let mut s_val = seq_len as i32;

        self.launch_compute_kernel(
            "grim_rwkv_time_mix",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut wptr),
                arg(&mut kptr),
                arg(&mut vptr),
                arg(&mut gptr),
                arg(&mut optr),
                arg(&mut b_val),
                arg(&mut d_val),
                arg(&mut s_val),
            ],
        )
    }

    /// Launch the JIT compiled RWKV channel-mix kernel (RWKV-5/6
    pub(crate) fn launch_rwkv_channel_mix(
        &self,
        x_storage: &RocmStorage,
        k_storage: &RocmStorage,
        r_storage: &RocmStorage,
        v_storage: &RocmStorage,
        out_storage: &RocmStorage,
        batch: usize,
        dim: usize,
    ) -> Result<*mut c_void> {
        let x_ptr = x_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: x has no device ptr".into()))?;
        let k_ptr = k_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: k has no device ptr".into()))?;
        let r_ptr = r_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: r has no device ptr".into()))?;
        let v_ptr = v_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: v has no device ptr".into()))?;
        let out_ptr = out_storage
            .device_ptr
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: out has no device ptr".into()))?;

        const BLOCK_SIZE: usize = 256;
        let total_elems: u64 = (batch as u64)
            .checked_mul(dim as u64)
            .ok_or_else(|| Error::Backend("rwkv_channel_mix: batch*dim overflow".into()))?;
        let grid_x: u32 = (total_elems.div_ceil(BLOCK_SIZE as u64))
            .try_into()
            .map_err(|_| Error::Backend("rwkv_channel_mix: grid overflow".into()))?;
        let grid_dim = HipDim3::new(grid_x, 1, 1);
        let block_dim = HipDim3::new(BLOCK_SIZE as u32, 1, 1);

        let mut xptr = x_ptr;
        let mut kptr = k_ptr;
        let mut rptr = r_ptr;
        let mut vptr = v_ptr;
        let mut optr = out_ptr;
        let mut b_val = batch as i32;
        let mut d_val = dim as i32;

        self.launch_compute_kernel(
            "grim_rwkv_channel_mix",
            grid_dim,
            block_dim,
            &mut [
                arg(&mut xptr),
                arg(&mut kptr),
                arg(&mut rptr),
                arg(&mut vptr),
                arg(&mut optr),
                arg(&mut b_val),
                arg(&mut d_val),
            ],
        )
    }

}
