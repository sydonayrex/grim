use grim_tensor::dtype::{DType, Storage};
use grim_tensor::{Shape, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavefrontPadSpec {
    pub padded_rows: usize,
    pub padded_cols: usize,
    pub wavefront_size: usize,
}

impl WavefrontPadSpec {
    pub fn new(padded_rows: usize, padded_cols: usize, wavefront_size: usize) -> Self {
        Self {
            padded_rows,
            padded_cols,
            wavefront_size,
        }
    }

    /// Compute the padded dimensions for wavefront-aligned matrix tiling.
    ///
    /// Pads the row dimension (`padded_rows`) to a multiple of the wavefront size
    /// to ensure alignment along the GPU warp reduction axis. The column dimension
    /// (`padded_cols`) is left unpadded to avoid wasting work and memory.
    pub fn compute(rows: usize, cols: usize, wavefront_size: u32) -> Self {
        let wf = wavefront_size as usize;
        let padded_rows = (rows + wf - 1) & !(wf - 1);
        let padded_cols = cols;
        Self::new(padded_rows, padded_cols, wf)
    }
}

fn wave_id(spec: WavefrontPadSpec) -> usize { spec.wavefront_size }

pub fn tile_f32(weights: &[f32], rows: usize, cols: usize, wavefront_size: u32) -> Vec<f32> {
    let spec = WavefrontPadSpec::compute(rows, cols, wavefront_size);
    let wf = spec.wavefront_size;
    let mut tiled = vec![0.0f32; spec.padded_rows * spec.padded_cols];

    for wave in 0..(spec.padded_rows / wf) {
        for lane in 0..wf {
            let src_row = wave * wf + lane;
            if src_row >= rows {
                break;
            }
            for col in 0..cols {
                let dst = (wave * spec.padded_cols + col) * wf + lane;
                let src = src_row * cols + col;
                tiled[dst] = weights[src];
            }
        }
    }

    tiled
}

pub fn untiled_f32(tiled: &[f32], rows: usize, cols: usize, wavefront_size: u32) -> Vec<f32> {
    let spec = WavefrontPadSpec::compute(rows, cols, wavefront_size);
    let wf = spec.wavefront_size;
    let mut out = vec![0.0f32; rows * cols];

    for wave in 0..(spec.padded_rows / wf) {
        for lane in 0..wf {
            let dst_row = wave * wf + lane;
            if dst_row >= rows {
                break;
            }
            for col in 0..cols {
                let src = (wave * spec.padded_cols + col) * wf + lane;
                out[dst_row * cols + col] = tiled[src];
            }
        }
    }

    out
}

pub fn aligned_tensor_for_rocm_gemm(
    input: &[f32],
    spec: WavefrontPadSpec,
) -> Tensor {
    let padded = tile_f32(input, spec.padded_rows, spec.padded_cols, spec.wavefront_size as u32);
    let shape = Shape::new(vec![spec.padded_rows, spec.padded_cols]);

    grim_backend_cpu::cpu_tensor(padded, shape)
}
