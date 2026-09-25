//! GRAVE Phase 5: Persistent wavefront megakernel launcher.
//!
//! Executes the entire decode step across persistent CU worker blocks in
//! ONE persistent HIP launch.

use crate::device::roc_device::RocmDevice;
use crate::memory::storage::RocmStorage;
use crate::{arg, dev_ptr, HipDim3};
use grim_tensor::error::{Error, Result};
use grim_tensor::MemoryOps;
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlaMegakernelTask {
    pub op_type: i32,
    pub layer_idx: i32,
    pub in_offset: i32,
    pub weight_offset: i32,
    pub out_offset: i32,
    pub aux_offset: i32,
    pub dim_m: i32,
    pub dim_n: i32,
    pub dim_k: i32,
    pub dep_counter_idx: i32,
    pub dep_counter_val: i32,
    pub post_counter_idx: i32,
}

pub struct GlaMegaLaunchArgs<'a> {
    pub tasks: &'a [GlaMegakernelTask],
    pub staging_pool: &'a RocmStorage,
    pub weight_pool: &'a RocmStorage,
    pub recurrent_state: &'a RocmStorage,
    pub stage_counters: &'a RocmStorage,
    pub task_cursor: &'a RocmStorage,
    pub total_layers: usize,
    pub hidden_dim: usize,
    pub num_cus: usize,
}

impl RocmDevice {
    /// Launches the persistent megakernel for whole-step decode (M5 gate).
    pub fn launch_gla_persistent_megakernel(
        &self,
        args: &GlaMegaLaunchArgs,
    ) -> Result<*mut c_void> {
        if args.tasks.is_empty() {
            return Err(Error::Backend(
                "launch_gla_megakernel: empty task list".into(),
            ));
        }

        // Upload tasks array
        let task_bytes = std::mem::size_of_val(args.tasks);
        let tasks_dev = self.from_cpu_bytes(
            unsafe { std::slice::from_raw_parts(args.tasks.as_ptr() as *const u8, task_bytes) },
            &grim_tensor::Shape::new(vec![args.tasks.len()]),
            grim_tensor::DType {
                arith: grim_tensor::ArithType::U8,
                storage: grim_tensor::Storage::Native,
            },
        )?;

        let mut tasks_ptr = dev_ptr(crate::as_rocm(&*tasks_dev)?)?;
        let mut num_tasks_i = args.tasks.len() as i32;
        let mut staging_ptr = dev_ptr(args.staging_pool)?;
        let mut weight_ptr = dev_ptr(args.weight_pool)?;
        let mut recurrent_ptr = dev_ptr(args.recurrent_state)?;
        let mut counters_ptr = dev_ptr(args.stage_counters)?;
        let mut cursor_ptr = dev_ptr(args.task_cursor)?;
        let mut layers_i = args.total_layers as i32;
        let mut hidden_i = args.hidden_dim as i32;

        let grid = HipDim3 {
            x: args.num_cus.clamp(16, 64) as u32,
            y: 1,
            z: 1,
        };
        let block = HipDim3 { x: 256, y: 1, z: 1 };

        self.launch_compute_kernel(
            "grim_gla_persistent_megakernel",
            grid,
            block,
            &mut [
                arg(&mut tasks_ptr),
                arg(&mut num_tasks_i),
                arg(&mut staging_ptr),
                arg(&mut weight_ptr),
                arg(&mut recurrent_ptr),
                arg(&mut counters_ptr),
                arg(&mut cursor_ptr),
                arg(&mut layers_i),
                arg(&mut hidden_i),
            ],
        )
    }
}
