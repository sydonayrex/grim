//! Xing4.0 MHC graph-step launchers (decode-graph capture-safe): every op
//! writes into caller-provided storages and reads its gate vectors from
//! device memory, so capture records pure kernel launches.

use crate::device::util::{arg, dev_ptr};
use crate::device::util::{as_rocm, DeviceGuard};
use crate::memory::storage::RocmStorage;
use grim_tensor::backend::BackendStorage;
use grim_tensor::error::Result;

macro_rules! launch_xing_step {
    ($self:ident, $kernel:literal, $grid:expr, $args:expr) => {
        $self.launch_compute_kernel(
            $kernel,
            crate::HipDim3::new($grid, 1, 1),
            crate::HipDim3::new(256, 1, 1),
            $args,
        )?;
    };
}

impl crate::RocmDevice {
    /// `out[d] = Σ_h pre[h] * streams[h*hidden + d]` at seq = 1.
    pub fn launch_hc_collapse_step(
        &self,
        streams: &dyn BackendStorage,
        pre: &dyn BackendStorage,
        out: &RocmStorage,
        hc: usize,
        hidden: usize,
    ) -> Result<()> {
        let _g = DeviceGuard::set(self.ordinal as i32);
        let s = as_rocm(streams)?;
        let p = as_rocm(pre)?;
        let mut sp = dev_ptr(s)?;
        let mut pp = dev_ptr(p)?;
        let mut op = dev_ptr(out)?;
        let mut h = hc as i32;
        let mut n = hidden as i32;
        launch_xing_step!(
            self,
            "grim_hc_collapse_step",
            hidden.div_ceil(256) as u32,
            &mut [arg(&mut sp), arg(&mut pp), arg(&mut op), arg(&mut h), arg(&mut n)]
        );
        Ok(())
    }

    /// `out[d] = (1/hc) Σ_h streams[h*hidden + d]` — the model-head mean.
    pub fn launch_hc_mean_step(
        &self,
        streams: &dyn BackendStorage,
        out: &RocmStorage,
        hc: usize,
        hidden: usize,
    ) -> Result<()> {
        let _g = DeviceGuard::set(self.ordinal as i32);
        let s = as_rocm(streams)?;
        let mut sp = dev_ptr(s)?;
        let mut op = dev_ptr(out)?;
        let mut h = hc as i32;
        let mut n = hidden as i32;
        launch_xing_step!(
            self,
            "grim_hc_mean_step",
            hidden.div_ceil(256) as u32,
            &mut [arg(&mut sp), arg(&mut op), arg(&mut h), arg(&mut n)]
        );
        Ok(())
    }

    /// `out[h*hidden+d] = post[h]*y[d] + Σ_i comb[h*hc+i]*streams[i*hidden+d]`.
    pub fn launch_hc_write_back_step(
        &self,
        y: &dyn BackendStorage,
        post: &dyn BackendStorage,
        comb: &dyn BackendStorage,
        streams: &dyn BackendStorage,
        out: &RocmStorage,
        hc: usize,
        hidden: usize,
    ) -> Result<()> {
        let _g = DeviceGuard::set(self.ordinal as i32);
        let y_s = as_rocm(y)?;
        let p = as_rocm(post)?;
        let c = as_rocm(comb)?;
        let s = as_rocm(streams)?;
        let mut yp = dev_ptr(y_s)?;
        let mut pp = dev_ptr(p)?;
        let mut cp = dev_ptr(c)?;
        let mut sp = dev_ptr(s)?;
        let mut op = dev_ptr(out)?;
        let mut h = hc as i32;
        let mut n = hidden as i32;
        launch_xing_step!(
            self,
            "grim_hc_write_back_step",
            (hc * hidden).div_ceil(256) as u32,
            &mut [
                arg(&mut yp),
                arg(&mut pp),
                arg(&mut cp),
                arg(&mut sp),
                arg(&mut op),
                arg(&mut h),
                arg(&mut n)
            ]
        );
        Ok(())
    }

    /// Split q_full `[nh*(nope+rope_d)]` into q_nope `[nh*nope]` and q_pe `[nh*rope_d]`.
    pub fn launch_xing_q_split(
        &self,
        q_full: &dyn BackendStorage,
        q_nope: &RocmStorage,
        q_pe: &RocmStorage,
        nh: usize,
        nope: usize,
        rope_d: usize,
    ) -> Result<()> {
        let _g = DeviceGuard::set(self.ordinal as i32);
        let q = as_rocm(q_full)?;
        let mut qp = dev_ptr(q)?;
        let mut np = dev_ptr(q_nope)?;
        let mut pp = dev_ptr(q_pe)?;
        let mut h = nh as i32;
        let mut n = nope as i32;
        let mut r = rope_d as i32;
        launch_xing_step!(
            self,
            "grim_xing_q_split",
            (nh * (nope + rope_d)).div_ceil(256) as u32,
            &mut [arg(&mut qp), arg(&mut np), arg(&mut pp), arg(&mut h), arg(&mut n), arg(&mut r)]
        );
        Ok(())
    }

    /// Per-head latent absorb with the F32 W_UK cache
    /// (`q_abs[h*rank + r] = Σ_d q_nope[h*nope + d] * w_uk[(h*rank+r)*nope + d]`).
    pub fn launch_xing_q_absorb(
        &self,
        q_nope: &dyn BackendStorage,
        w_uk: &dyn BackendStorage,
        q_abs: &RocmStorage,
        nh: usize,
        rank: usize,
        nope: usize,
    ) -> Result<()> {
        let _g = DeviceGuard::set(self.ordinal as i32);
        let q = as_rocm(q_nope)?;
        let w = as_rocm(w_uk)?;
        let mut qp = dev_ptr(q)?;
        let mut wp = dev_ptr(w)?;
        let mut op = dev_ptr(q_abs)?;
        let mut h = nh as i32;
        let mut r = rank as i32;
        let mut n = nope as i32;
        launch_xing_step!(
            self,
            "grim_xing_q_absorb",
            (nh * rank).div_ceil(256) as u32,
            &mut [arg(&mut qp), arg(&mut wp), arg(&mut op), arg(&mut h), arg(&mut r), arg(&mut n)]
        );
        Ok(())
    }
}

impl crate::RocmDevice {
    /// Pack `[c_kv || k_pe]` into one contiguous latent row.
    pub fn launch_xing_pack_latent(
        &self,
        c_kv: &dyn BackendStorage,
        k_pe: &dyn BackendStorage,
        out: &RocmStorage,
        rank: usize,
        rope_d: usize,
    ) -> Result<()> {
        let _g = DeviceGuard::set(self.ordinal as i32);
        let c = as_rocm(c_kv)?;
        let k = as_rocm(k_pe)?;
        let mut cp = dev_ptr(c)?;
        let mut kp = dev_ptr(k)?;
        let mut op = dev_ptr(out)?;
        let mut r = rank as i32;
        let mut d = rope_d as i32;
        launch_xing_step!(
            self,
            "grim_xing_pack_latent",
            (rank + rope_d).div_ceil(256) as u32,
            &mut [arg(&mut cp), arg(&mut kp), arg(&mut op), arg(&mut r), arg(&mut d)]
        );
        Ok(())
    }
}
