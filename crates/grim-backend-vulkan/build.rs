//! Build script for the Vulkan backend.
//!
//! At compile time this script compiles every GLSL compute kernel to real SPIR-V
//! using `glslangValidator -V` (the same tool documented in `radv_repro.rs`).
//! The resulting `.spv` blobs are embedded into the crate at compile time via
//! `include_bytes!`, so `compile_glsl_to_spirv` returns genuine SPIR-V with no
//! runtime dependency on an external compiler.
//!
//! All kernels share a single `push_constant` `Params` block so the pipeline
//! layout stays uniform across every entry point. Dynamic per-dispatch values
//! (element count, inner dimension, matmul K, etc.) are supplied via push
//! constants instead of being baked into the shader source.

use std::path::PathBuf;
use std::process::Command;

/// Shared push-constant parameter block. Every kernel declares this exact
/// layout so the pipeline push-constant range is identical everywhere.
const PARAMS_GLSL: &str = r#"
layout(push_constant) uniform Params {
    uint size; // total element count
    uint dim;  // inner / feature dimension (rms, softmax, embedding)
    uint k;    // matmul K
    uint n;    // matmul N
    uint m;    // matmul M
    float eps; // rms norm epsilon
};
"#;

/// (kernel name, glsl source). The name is used for the `.comp`/`.spv` file
/// and the generated `include_bytes!` constant.
fn kernels() -> Vec<(&'static str, String)> {
    let mut out = Vec::new();

    out.push((
        "add",
        format!(
            r#"#version 450
layout(local_size_x = 256) in;
{params}
layout(std430, binding = 0) readonly buffer A {{ float a[]; }};
layout(std430, binding = 1) readonly buffer B {{ float b[]; }};
layout(std430, binding = 2) writeonly buffer C {{ float c[]; }};
void main() {{
    uint id = gl_GlobalInvocationID.x;
    if (id >= size) return;
    c[id] = a[id] + b[id];
}}
"#,
            params = PARAMS_GLSL
        ),
    ));

    out.push((
        "mul",
        format!(
            r#"#version 450
layout(local_size_x = 256) in;
{params}
layout(std430, binding = 0) readonly buffer A {{ float a[]; }};
layout(std430, binding = 1) readonly buffer B {{ float b[]; }};
layout(std430, binding = 2) writeonly buffer C {{ float c[]; }};
void main() {{
    uint id = gl_GlobalInvocationID.x;
    if (id >= size) return;
    c[id] = a[id] * b[id];
}}
"#,
            params = PARAMS_GLSL
        ),
    ));

    out.push((
        "silu_mul",
        format!(
            r#"#version 450
layout(local_size_x = 256) in;
{params}
layout(std430, binding = 0) readonly buffer A {{ float a[]; }};
layout(std430, binding = 1) readonly buffer B {{ float b[]; }};
layout(std430, binding = 2) writeonly buffer C {{ float c[]; }};
void main() {{
    uint id = gl_GlobalInvocationID.x;
    if (id >= size) return;
    float gate = a[id];
    float silu = gate / (1.0 + exp(-gate));
    c[id] = silu * b[id];
}}
"#,
            params = PARAMS_GLSL
        ),
    ));

    out.push((
        "rms_norm",
        format!(
            r#"#version 450
layout(local_size_x = 256) in;
{params}
layout(std430, binding = 0) readonly buffer X {{ float x[]; }};
layout(std430, binding = 1) readonly buffer W {{ float w[]; }};
layout(std430, binding = 2) writeonly buffer Y {{ float y[]; }};
void main() {{
    uint id = gl_GlobalInvocationID.x;
    if (id >= size) return;
    uint row = id / dim;
    uint col = id % dim;
    float sum_sq = 0.0;
    for (uint i = 0u; i < dim; ++i) {{
        float val = x[row * dim + i];
        sum_sq += val * val;
    }}
    float rms = sqrt(sum_sq / float(dim) + eps);
    y[id] = (x[id] / rms) * w[col];
}}
"#,
            params = PARAMS_GLSL
        ),
    ));

    out.push((
        "softmax",
        format!(
            r#"#version 450
layout(local_size_x = 256) in;
{params}
layout(std430, binding = 0) readonly buffer X {{ float x[]; }};
layout(std430, binding = 1) writeonly buffer Y {{ float y[]; }};
void main() {{
    uint id = gl_GlobalInvocationID.x;
    if (id >= size) return;
    uint row = id / dim;
    float max_val = -1e30;
    for (uint i = 0u; i < dim; ++i) {{
        max_val = max(max_val, x[row * dim + i]);
    }}
    float sum = 0.0;
    for (uint i = 0u; i < dim; ++i) {{
        sum += exp(x[row * dim + i] - max_val);
    }}
    y[id] = exp(x[id] - max_val) / sum;
}}
"#,
            params = PARAMS_GLSL
        ),
    ));

    out.push((
        "embedding",
        format!(
            r#"#version 450
layout(local_size_x = 256) in;
{params}
layout(std430, binding = 0) readonly buffer W {{ float w[]; }};
layout(std430, binding = 1) readonly buffer I {{ uint indices[]; }};
layout(std430, binding = 2) writeonly buffer Y {{ float y[]; }};
void main() {{
    uint id = gl_GlobalInvocationID.x;
    uint total = (size / dim) * dim;
    if (id >= total) return;
    uint idx_pos = id / dim;
    uint col = id % dim;
    uint weight_row = indices[idx_pos];
    y[id] = w[weight_row * dim + col];
}}
"#,
            params = PARAMS_GLSL
        ),
    ));

    // Matmul kernels: one per autotuner tile config. The K/N/M bounds and the
    // inner loop limit come from push constants, so a single precompiled blob
    // serves every (m, n, k) triple for that block shape.
    for (block, suffix) in [(64u32, "64"), (32u32, "32")] {
        let name = format!("matmul_{suffix}");
        let source = format!(
            r#"#version 450
layout(local_size_x = {b}, local_size_y = {b}, local_size_z = 1) in;
{params}
layout(std430, binding = 0) readonly buffer BufA {{ float a[]; }};
layout(std430, binding = 1) readonly buffer BufB {{ float b[]; }};
layout(std430, binding = 2) writeonly buffer BufC {{ float c[]; }};
void main() {{
    uint gid_x = gl_GlobalInvocationID.x;
    uint gid_y = gl_GlobalInvocationID.y;
    if (gid_x >= n || gid_y >= m) return;
    float sum = 0.0;
    for (uint p = 0u; p < k; ++p) {{
        sum += a[gid_y * k + p] * b[p * n + gid_x];
    }}
    c[gid_y * n + gid_x] = sum;
}}
"#,
            b = block,
            params = PARAMS_GLSL
        );
        out.push((Box::leak(name.into_boxed_str()), source));
    }

    out
}

fn main() {
    println!("cargo:rustc-link-lib=dylib=vulkan");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));

    let validator = std::env::var("GLSLANG_VALIDATOR")
        .unwrap_or_else(|_| "glslangValidator".to_string());

    let mut gen_code = String::new();
    gen_code.push_str("// @generated by build.rs — do not edit.\n");
    gen_code.push_str("// Real SPIR-V blobs compiled from GLSL via glslangValidator.\n\n");

    let mut any_failed = false;
    for (name, glsl) in kernels() {
        let comp_path = out_dir.join(format!("{name}.comp"));
        let spv_path = out_dir.join(format!("{name}.spv"));
        std::fs::write(&comp_path, glsl.as_bytes()).expect("write .comp");

        let status = Command::new(&validator)
            .arg("-V")
            .arg("--target-env")
            .arg("vulkan1.1")
            .arg(&comp_path)
            .arg("-o")
            .arg(&spv_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                gen_code.push_str(&format!(
                    "pub const SPIRV_{}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}.spv\"));\n",
                    sanitize(name),
                    name
                ));
            }
            Ok(s) => {
                eprintln!(
                    "build.rs: glslangValidator failed for kernel `{name}` (status {s}); omitting precompiled blob"
                );
                any_failed = true;
            }
            Err(e) => {
                eprintln!(
                    "build.rs: could not invoke glslangValidator for kernel `{name}`: {e}; omitting precompiled blob"
                );
                any_failed = true;
            }
        }
    }

    let gen_path = out_dir.join("spirv_spv.rs");
    std::fs::write(&gen_path, gen_code).expect("write generated spirv module");

    if any_failed {
        // Surface a clear error so the failure is not silently swallowed.
        panic!(
            "build.rs: one or more Vulkan kernels failed to compile to SPIR-V. \
             Ensure `glslangValidator` is installed and on PATH (or set GLSLANG_VALIDATOR)."
        );
    }
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else if c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
