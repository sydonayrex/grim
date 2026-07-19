//! CubeCL-style compute shaders for Vulkan backend.
//!
//! §4: CubeCL kernels provide a shader-based alternative to hipBLAS.
//! Uses GLSL compute shaders compiled via `naga` for matmul/attention ops.

/// GLSL compute shader source for matrix multiplication.
/// Uses shared memory for tiling and cooperative loading.
pub const COMPUTE_SHADER_GEMM: &str = r#"
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

// Tile size - must match local_size
const uint TILE_SIZE = 16;

layout(std430, binding = 0) restrict readonly buffer A { float a_data[]; };
layout(std430, binding = 1) restrict readonly buffer B { float b_data[]; };
layout(std430, binding = 2) restrict writeonly buffer C { float c_data[]; };

uniform uvec3 work_group_info; // (M, N, K)

shared float a_tile[TILE_SIZE][TILE_SIZE + 1];
shared float b_tile[TILE_SIZE][TILE_SIZE + 1];

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint m = work_group_info.x;
    uint n = work_group_info.y;
    uint k = work_group_info.z;
    
    float sum = 0.0;
    
    for (uint t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles cooperatively
        for (uint i = 0; i < TILE_SIZE; ++i) {
            uint a_idx = (t * TILE_SIZE + i) * m + row;
            uint b_idx = (t * TILE_SIZE + i) * n + col;
            
            a_tile[gl_LocalInvocationID.y][i] = 
                (t * TILE_SIZE + i < k && row < m) ? a_data[a_idx] : 0.0;
            b_tile[i][gl_LocalInvocationID.x] = 
                (t * TILE_SIZE + i < k && col < n) ? b_data[b_idx] : 0.0;
        }
        
        barrier();
        
        // Compute partial result
        for (uint i = 0; i < TILE_SIZE; ++i) {
            sum += a_tile[gl_LocalInvocationID.y][i] * b_tile[i][gl_LocalInvocationID.x];
        }
        
        barrier();
    }
    
    // Write result
    if (row < m && col < n) {
        c_data[row * n + col] = sum;
    }
}
"#;

/// Compute shader for attention score calculation.
pub const COMPUTE_SHADER_ATTENTION: &str = r#"
#version 450

layout(local_size_x = 128) in;

layout(std430, binding = 0) restrict readonly buffer Q { float q_data[]; };
layout(std430, binding = 1) restrict readonly buffer K { float k_data[]; };
layout(std430, binding = 2) restrict writeonly buffer O { float o_data[]; };

uniform uint seq_len;
uniform uint head_dim;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    // Attention computation would go here
    // For now, placeholder
}
"#;

/// Metadata for a compute kernel.
#[derive(Debug, Clone)]
pub struct ComputeKernel {
    pub name: &'static str,
    pub glsl_source: &'static str,
    pub workgroup_size: (u32, u32, u32),
}

/// Available compute kernels.
pub const AVAILABLE_KERNELS: &[ComputeKernel] = &[
    ComputeKernel {
        name: "gemm",
        glsl_source: COMPUTE_SHADER_GEMM,
        workgroup_size: (16, 16, 1),
    },
    ComputeKernel {
        name: "attention",
        glsl_source: COMPUTE_SHADER_ATTENTION,
        workgroup_size: (128, 1, 1),
    },
];

/// Kernel builder for CubeCL-style operations.
///
/// Kernels are precompiled to real SPIR-V at build time (see `build.rs` +
/// `glslangValidator`). `build_kernel` selects the matching precompiled blob
/// from the generated [`crate::precompiled_spv`] module.
pub struct KernelBuilder;

impl KernelBuilder {
    /// Build a kernel module from GLSL source, returning genuine SPIR-V words.
    ///
    /// The source must be one of the canonical generated shaders
    /// (`generate_*_glsl` / `generate_matmul_glsl`); the matching precompiled
    /// blob is looked up and returned. This replaces the previous naga
    /// placeholder so callers get real Vulkan SPIR-V without a runtime
    /// compiler dependency.
    pub fn build_kernel(source: &str, _entry: &str) -> Result<Vec<u32>, grim_tensor::error::Error> {
        let bytes = crate::compile_glsl_to_spirv(source)?;
        if bytes.len() % 4 != 0 {
            return Err(grim_tensor::error::Error::Backend(
                "CubeCL build_kernel: SPIR-V byte length not a multiple of 4".into(),
            ));
        }
        let words = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(words)
    }

    /// Select the precompiled SPIR-V blob for a known operation directly.
    pub fn spirv_for(kernel: crate::VulkanKernel) -> &'static [u8] {
        crate::spirv_for(kernel)
    }

    /// Compute dispatches needed for a GEMM operation.
    pub fn gemm_dispatches(m: u32, n: u32, _k: u32) -> u32 {
        ((m + 15) / 16) * ((n + 15) / 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_metadata_sizes() {
        let gemm = &AVAILABLE_KERNELS[0];
        assert_eq!(gemm.name, "gemm");
        assert!(gemm.glsl_source.contains("local_size_x = 16"));
    }

    #[test]
    fn test_gemm_dispatch_calculation() {
        // M=128, N=256 should need (128/16)*(256/16) = 8*16 = 128 dispatches
        assert_eq!(KernelBuilder::gemm_dispatches(128, 256, 64), 128);
    }

    #[test]
    fn test_build_kernel_returns_real_spirv() {
        // build_kernel now returns genuine SPIR-V words from the precompiled blob.
        let words = KernelBuilder::build_kernel(&crate::generate_add_glsl(), "main")
            .expect("add kernel should precompile to SPIR-V");
        assert!(!words.is_empty());
        // SPIR-V magic number: 0x07230203 little-endian.
        assert_eq!(words[0], 0x07230203);
    }
}